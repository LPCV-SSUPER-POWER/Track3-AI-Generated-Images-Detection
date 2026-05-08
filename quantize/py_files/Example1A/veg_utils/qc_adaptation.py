# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2024 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
import copy
import math
import functools
import torch
import torch.nn as nn
import warnings
from typing import Tuple, Union
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention, Qwen2VLVisionBlock


class ConvInplaceLinear(torch.nn.Module):
    """ Convolution module that replaces a Linear layer inplace"""

    def __init__(self, linear):
        super(ConvInplaceLinear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.conv2d = torch.nn.Conv2d(linear.in_features, linear.out_features, 1, bias=True if linear.bias is not None else False)
        self.conv2d.weight.data.copy_(linear.weight.data[:, :, None, None])
        if linear.bias is not None:
            self.conv2d.bias.data.copy_(linear.bias.data)
        self.conv2d.to(linear.weight.data.device)

    def __getattr__(self, attr):
        conv2d = self._modules['conv2d']
        if attr == 'conv2d':
            return conv2d
        return getattr(conv2d, attr)

    def forward(self, x: torch.Tensor, scale: float = 1.0):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(-1).permute(0, 2, 3, 1)  # (emb_dim, C) -> (1, C, 1, emb_dim)
        elif ndim == 3:
            x = x.unsqueeze(-1).permute(0, 2, 3, 1)  # (B, emb_dim, C) -> (B, C, 1, emb_dim)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            warnings.warn("ConvInplaceLinear received an unexpected 4d input, assuming channels-last and proceeding.")
        else:
            raise NotImplementedError(f"ConvInplaceLinear could not handle input with shape {x.shape}")

        x = self.conv2d(x)

        if ndim == 2:
            return x.permute(0, 3, 1, 2).squeeze(-1).squeeze(0)  # (1, C, 1, emb_dim) -> # (emb_dim, C)
        elif ndim == 3:
            return x.permute(0, 3, 1, 2).squeeze(-1)  # (1, C, 1, emb_dim) -> # (B, emb_dim, C)
        elif ndim == 4:
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        return x


class Conv2dInplaceConv3d(torch.nn.Module):

    def __init__(self, conv3d):
        '''Only stride=1 and bias=False are supported. '''
        super().__init__()
        inc, outc, ksize = conv3d.in_channels * 2, conv3d.out_channels, conv3d.kernel_size
        ksize = (ksize[1], ksize[2])
        self.conv2d = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksize, bias=False)

        # This is the context-manager which disables the gradient calculation while inferencing reducing the consumption of the memory for faster computations.
        with torch.no_grad():
            _3Dconv_weight_data = torch.cat((conv3d.weight.data[:, :, 0, :, :], conv3d.weight.data[:, :, 1, :, :]), axis=1)
            _2Dconv_weight_data = torch.cat((_3Dconv_weight_data[:, 0:1, :, :], _3Dconv_weight_data[:, 3:4, :, :], _3Dconv_weight_data[:, 1:2, :, :], _3Dconv_weight_data[:, 4:5, :, :],
                                             _3Dconv_weight_data[:, 2:3, :, :], _3Dconv_weight_data[:, 5:6, :, :]),
                                            axis=1)

            self.conv2d.weight.data.copy_(_2Dconv_weight_data)
        self.conv2d.to(conv3d.weight.data.device)

    def __getattr__(self, attr):
        conv2d = self._modules['conv2d']
        if attr == 'conv2d':
            return conv2d
        return getattr(conv2d, attr)

    def forward(self, x: torch.Tensor):
        x = torch.reshape(x, (-1, 6, 14, 14))
        output = self.conv2d(x).unsqueeze(2)
        return output


def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    rope_real = rope_vals[0]  # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1]  # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:, :, :, :x.shape[-1] // 2]  # extract first half elements
    x_im = x[:, :, :, x.shape[-1] // 2:]  # extract second half elements
    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real, x_prod_im), dim=3).view(*x.shape)

    return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    cos = cos.transpose(1, 2)
    sin = sin.transpose(1, 2)
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class Qwen2VLVisionBlockAdaptation(nn.Module):

    def __init__(self, block: Qwen2VLVisionBlock) -> None:
        super().__init__()
        self.norm1 = block.norm1  #LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = block.norm2 #LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = block.attn
        self.mlp = block.mlp

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, attention_mask=None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionAttentionInplaceAdaptation(nn.Module):

    def __init__(self, visual: VisionAttention) -> None:
        super().__init__()
        self.num_heads = visual.num_heads
        self.head_dim = visual.head_dim
        self.dim = visual.qkv.in_features
        self.bias = visual.qkv.bias is not None

        # for proj
        o = nn.Conv2d(visual.proj.in_features, visual.proj.out_features, 1, bias=visual.proj.bias is not None)
        o.weight.data.copy_(visual.proj.weight.data[:, :, None, None])
        if visual.proj.bias is not None:
            o.bias.data.copy_(visual.proj.bias.data)
        o.to(visual.proj.weight.device)
        self.proj = o
        self.q, self.k, self.v = self.prepare_mha(visual)
        del visual

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.forward_mha(hidden_states, cu_seqlens, rotary_pos_emb, attention_mask)

    def prepare_mha(self, visual):
        q = nn.Conv2d(self.dim, self.dim, 1, bias=self.bias)
        k = nn.Conv2d(self.dim, self.dim, 1, bias=self.bias)
        v = nn.Conv2d(self.dim, self.dim, 1, bias=self.bias)

        qkv_weights = visual.qkv.weight.data
        q.weight.data.copy_(qkv_weights[:self.dim, :, None, None])
        k.weight.data.copy_(qkv_weights[self.dim:self.dim * 2, :, None, None])
        v.weight.data.copy_(qkv_weights[self.dim * 2:, :, None, None])

        if self.bias:
            qkv_bias = visual.qkv.bias.data
            q.bias.data.copy_(qkv_bias[:self.dim])
            k.bias.data.copy_(qkv_bias[self.dim:self.dim * 2])
            v.bias.data.copy_(qkv_bias[self.dim * 2:])

        q.to(qkv_weights.device)
        k.to(qkv_weights.device)
        v.to(qkv_weights.device)
        return q, k, v

    def forward_mha(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:

        seq_length = hidden_states.shape[0]

        hidden_states = torch.reshape(hidden_states, (-1, seq_length, 1, self.num_heads * self.head_dim)).transpose(1, 3)
        # q1, k1, v1 = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = self.q(hidden_states).reshape(-1, self.num_heads, self.head_dim, seq_length).transpose(2, 3)
        k = self.k(hidden_states).reshape(-1, self.num_heads, self.head_dim, seq_length).transpose(2, 3)
        v = self.v(hidden_states).reshape(-1, self.num_heads, self.head_dim, seq_length).transpose(2, 3)

        if isinstance(rotary_pos_emb, (tuple, list)):  # QC
            q = _apply_rope_single(q, rotary_pos_emb)
            k = _apply_rope_single(k, rotary_pos_emb)
        else:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if attention_mask is None:
            attention_mask = torch.full([1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype)
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = 0

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(-1, seq_length, 1, self.num_heads * self.head_dim)

        attn_output = attn_output.transpose(1, 3)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.transpose(1, 3)

        attn_output = attn_output.reshape(seq_length, self.num_heads * self.head_dim)
        return attn_output

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def replace_linears_with_convs(model: torch.nn.Module, linear_types: Union[torch.nn.Module, Tuple[torch.nn.Module]] = torch.nn.Linear) -> torch.nn.Module:

    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            conv_layer = ConvInplaceLinear(module)
            rsetattr(model, name, conv_layer)

    return model


def replace_linears_conv3d_with_convs(model: torch.nn.Module, linear_types: Union[torch.nn.Module, Tuple[torch.nn.Module]] = torch.nn.Linear) -> torch.nn.Module:

    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            conv_layer = ConvInplaceLinear(module)
            rsetattr(model, name, conv_layer)

        elif isinstance(module, nn.Conv3d):
            conv_layer = Conv2dInplaceConv3d(module)
            rsetattr(model, name, conv_layer)

    return model


def replace_visual_attention_with_adaptation(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, Qwen2VLVisionBlock):
            layer = Qwen2VLVisionBlockAdaptation(module)
            rsetattr(model, name, layer)

    for name, module in model.named_modules():
        if isinstance(module, VisionAttention):
            layer = VisionAttentionInplaceAdaptation(module)
            rsetattr(model, name, layer)

    return model

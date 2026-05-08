# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =============================================================================
import sys
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers import cache_utils
from huggingface.baseline_models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    repeat_kv,
    Cache,
    DynamicCache,
    Qwen2Config,
    apply_rotary_pos_emb,
)
from aimet_torch.nn.modules.custom import Add

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
    x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements

    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    return x

def _apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    # cos = cos[0,0,:,:]  # [seq_len, dim]
    # sin = sin[0,0,:,:]  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def _apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QcAttention(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_add = Add()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) / math.sqrt(self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if isinstance(position_ids, (tuple, list)): # QC
            rope_embedding = position_ids
            cos, sin = rope_embedding
            query_states = _apply_rope_single(query_states, rope_embedding)
            key_states = _apply_rope_single(key_states, rope_embedding)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = self.attn_add(attn_weights, attention_mask)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def prepare_conv(self):

        self.rm_div = self.config.rm_div if hasattr(self.config, 'rm_div') else False

        if not hasattr(self, 'forward_no_conv'):
            self.q_proj_conv = nn.Conv2d(self.hidden_size, self.num_heads * self.head_dim, 1, bias=True)
            self.k_proj_conv = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=True)
            self.v_proj_conv = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=True)
            self.o_proj_conv = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, 1, bias=False)

            self.forward_no_conv = self.forward
            self.forward = self.forward_conv

            self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
            if self.rm_div:
                self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None] / math.sqrt(self.head_dim))
            else:
                self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
            self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
            self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

            self.q_proj_conv.bias.data.copy_(self.q_proj.bias)
            if self.rm_div:
                self.k_proj_conv.bias.data.copy_(self.k_proj.bias / math.sqrt(self.head_dim))
            else:
                self.k_proj_conv.bias.data.copy_(self.k_proj.bias)
            self.v_proj_conv.bias.data.copy_(self.v_proj.bias)

            del self.q_proj
            del self.k_proj
            del self.v_proj
            del self.o_proj

    def forward_conv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = torch.reshape(hidden_states, (bsz, q_len, 1, self.hidden_size)).transpose(1, 3)

        query_states = self.q_proj_conv(hidden_states)
        key_states = self.k_proj_conv(hidden_states)
        value_states = self.v_proj_conv(hidden_states)

        query_states = query_states.reshape(bsz, self.num_heads, self.head_dim, q_len).transpose(2, 3)
        key_states = key_states.reshape(bsz, self.num_key_value_heads, self.head_dim, q_len).transpose(2, 3)
        value_states = value_states.reshape(bsz, self.num_key_value_heads, self.head_dim, q_len).transpose(2, 3)

        if isinstance(position_ids, (tuple, list)): # QC
            rope_embedding = position_ids
            cos, sin = rope_embedding
            query_states = _apply_rope_single(query_states, rope_embedding)
            key_states = _apply_rope_single(key_states, rope_embedding)
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = self.attn_add(attn_weights, attention_mask)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, 1, self.hidden_size)
        attn_output = attn_output.transpose(1, 3)
        attn_output = self.o_proj_conv(attn_output)
        attn_output = attn_output.transpose(1, 3)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def bypass_update_causal_mask(self, attention_mask, *args, **kwargs):
    # attention_mask is Causal mask and given as model input
    return attention_mask


def MLP_prepare_conv(self):
    if not hasattr(self, 'forward_linear'):
        self.gate_proj_conv = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)
        self.down_proj_conv = nn.Conv2d(self.intermediate_size, self.hidden_size, 1, bias=False)
        self.up_proj_conv = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)
        self.forward_linear = self.forward
        self.forward = self.forward_conv

    self.gate_proj_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None])
    self.down_proj_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None])
    self.up_proj_conv.weight.data.copy_(self.up_proj.weight[:, :, None, None])

    # del self.gate_proj
    # del self.down_proj
    # del self.up_proj

def MLP_forward_conv(self, x):
    bsz, _, _ = x.size()
    x = torch.reshape(x, (bsz, -1, 1, self.hidden_size))
    x = x.transpose(1,3) # Transpose right before and after Conv
    x = self.down_proj_conv(self.act_fn(self.gate_proj_conv(x)) * self.up_proj_conv(x))
    x = x.transpose(1,3)
    x = torch.reshape(x, (bsz, -1, self.hidden_size))
    return x


def ForCausalLM_prepare_conv(self):
    if not hasattr(self, 'lm_head_conv'):

        def lm_head_conv_forward(x):
            bsz, _, _ = x.size()
            x = torch.reshape(x, (bsz, -1, 1, self.config.hidden_size))
            x = x.transpose(1,3) # Transpose right before and after Conv
            x = self.lm_head_conv(x)
            x = x.transpose(1,3)
            x = torch.reshape(x, (bsz, -1, self.config.vocab_size))
            return x

        self.lm_head_conv = nn.Conv2d(self.config.hidden_size, self.config.vocab_size, 1, bias=False)
        self.lm_head_conv.weight.data.copy_(self.lm_head.weight[:, :, None, None])

        del self.lm_head
        self.lm_head = lm_head_conv_forward


_Use_KV_Mode = True
def set_trace_mode(use_kv_mode):
    global _Use_KV_Mode
    _Use_KV_Mode = use_kv_mode

def DynamicCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += value_states.shape[-2]
    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    else:
        return_new_key_value_only = cache_kwargs.get('return_new_key_value_only', False)
        transposed_key_cache = cache_kwargs.get('transposed_key_cache', False)
        key_cat_dim = -1 if transposed_key_cache else -2

        key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=key_cat_dim)
        value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        global _Use_KV_Mode
        # if return_new_key_value_only:
        if _Use_KV_Mode:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = key_cache
            self.value_cache[layer_idx] = value_cache
        return key_cache, value_cache


def DynamicCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.value_cache) <= layer_idx:
        return 0
    return self.value_cache[layer_idx].shape[-2]


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False

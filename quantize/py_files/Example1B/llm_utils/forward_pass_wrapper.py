#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
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

"""  utility method to adapt original model, prepared model and bert/KV model forward pass invocation """
import inspect

import contextlib

import json
import torch
from aimet_torch.utils import get_device


def flatten_tensors(tup):
    if not isinstance(tup, (tuple, list)):
        yield tup
        return
    for x in tup:
        yield from flatten_tensors(x)

def get_padded_kv_values(past_size, num_layers,
                         hidden_size, num_attention_heads, batch_size=1, separate_kv_head=False,
                         concat_head_in_batch_dimension=False, num_kv_heads=32, transposed_key_cache=True, device='cuda'):

    def _cache(shape):
        return torch.zeros(shape).to(device=device)

    if concat_head_in_batch_dimension:
        assert batch_size == 1, "batch size should be 1 if concat_head_in_batch_dimension is True"
        value = (num_kv_heads, 1, past_size, hidden_size // num_attention_heads)
    else:
        head_dim = 1 if separate_kv_head else num_kv_heads
        value = (batch_size, head_dim, past_size, hidden_size // num_attention_heads)
    key = (value[0], value[1], value[3], value[2]) if transposed_key_cache else tuple(value)

    if separate_kv_head: #(SHA)
        past_key_values = tuple((
                                    tuple(_cache(key) for _ in range(num_kv_heads)),
                                    tuple(_cache(value) for _ in range(num_kv_heads)),
                                ) for _ in range(num_layers))
    else:
        past_key_values = tuple((_cache(key), _cache(value)) for _ in range(num_layers))
    return past_key_values



class RopeEmbedding:

    def __init__(self, device, head_dim=128, max_length=2048, theta=10000.0):
        self.max_length = max_length
        self.device= device
        self.cos, self.sin = self.precompute_freqs_cis(head_dim, max_length * 2, theta=theta, device=device)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0, device=None):
        freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        freqs_cis = freqs_cis[0:self.max_length]
        freqs_real = torch.view_as_real(freqs_cis)
        freqs_real = freqs_real.unsqueeze(0).unsqueeze(0)

        freqs_cos = freqs_real[:, :, :, :, 0]  # extract even elements
        freqs_sin = freqs_real[:, :, :, :, 1]  # extract odd elements
        return freqs_cos.to(device), freqs_sin.to(device)

    def to(self, device):
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)
        self.device=device

    def get_embedding(self, position_ids):
        '''
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        '''
        position_ids = position_ids.to(self.device)
        cos = self.cos[0, 0, :, :]  # [seq_len, dim]
        sin = self.sin[0, 0, :, :]  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        return cos, sin

def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, mask_neg=-100.0):
    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
            input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0,
            mask_neg: float = -100.0
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape[0], input_ids_shape[1]
        # mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask = torch.full((tgt_len, tgt_len), torch.tensor(mask_neg, device=device), device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, mask_neg: float = -100.0, tgt_len: int = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), mask_neg)


    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            mask_neg=mask_neg,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[1], mask_neg=mask_neg).to(
            inputs_embeds.device
        )

        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

def get_position_embeddings_from_position_ids(position_ids, head_dim, max_length, rope_theta, device):
    return RopeEmbedding(device=device, head_dim=head_dim, max_length=max_length, theta=rope_theta).get_embedding(position_ids)

def prepare_combined_attention_mask(attention_mask, input_shape, past_key_values_length, device, mask_neg=-100.0,
                                     dtype=torch.float32):
    dummy_embedding = torch.tensor((1.0,)).to(dtype).to(device)
    new_mask = prepare_decoder_attention_mask(attention_mask, input_shape, dummy_embedding, past_key_values_length, mask_neg)
    return new_mask.clamp_min(mask_neg)


class LLMForwardPassManager:
    def __init__(self, cfg, model, tokenizer, model_mode, num_logits_to_return, separate_tuple_input_output, num_tokens=None):
        self.tokenizer = tokenizer
        self.model = model

        self.device = get_device(model)

        self.num_heads = getattr(cfg, 'num_attention_heads', 1)
        self.num_kv_heads = getattr(cfg, 'num_key_value_heads')
        self.num_layers = getattr(cfg, 'num_hidden_layers', 32)
        self.embed_dim = getattr(cfg, 'hidden_size', 1024)
        self.rope_theta = getattr(cfg, "rope_theta", 10000.0)

        self.max_tokens = tokenizer.model_max_length
        self.num_logits_to_return = num_logits_to_return
        self.model_mode = model_mode
        if model_mode == 'bertcache':
            self.num_tokens = self.max_tokens
        else:
            self.num_tokens = num_tokens
        self.use_position_embedding_input = getattr(cfg, 'use_position_embedding_input', False)
        self.use_combined_mask_input = getattr(cfg, 'use_combined_mask_input', False)
        self.concat_head_in_batch_dimension = getattr(cfg, 'concat_head_in_batch_dimension', False)
        self.transposed_key_cache = getattr(cfg, 'transposed_key_cache', False)
        self.mask_neg = getattr(cfg, 'mask_neg', -100)
        self.separate_kv_head = getattr(cfg, 'separate_kv_head', False)
        self.use_input_embeddings = getattr(cfg, 'use_input_embeddings', False)
        self.return_new_key_value_only = getattr(cfg, 'return_new_key_value_only', True)
        self.separate_tuple_input_output = separate_tuple_input_output
        self.record_test_vectors = False  # users of this block wil enable/disable this as necessary with provided functions
        self.dummy_kvcache_generator = None  # DummyKvcacheGenerator(cfg)
        self.input_id_to_embedding_converter = None
        # self.rotary_emb = LlamaRotaryEmbedding(
        #     self.embed_dim // self.num_heads,
        #     max_position_embeddings=self.num_tokens,
        #     base=self.rope_theta,
        # )
        self.rotary_emb = RopeEmbedding(self.device, self.embed_dim // self.num_heads, max_length=self.max_tokens, theta=self.rope_theta)

    def replace_model(self, new_model):
        self.model = new_model
        self.model.to(self.device)

    @contextlib.contextmanager
    def place_on_device(self, device):
        original_device = self.device
        try:
            self.to(device)
            yield
        finally:
            self.to(original_device)

    def to(self, device):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.rotary_emb.to(self.device)


    def get_rope_embedding(self, position_ids, device):
        cos,sin= self.rotary_emb.get_embedding(position_ids)
        return (cos.to(device),sin.to(device))

    def parameters(self):
        return self.model.parameters()

    def enable_test_vector_recording(self, file_prefix="fp", test_vector_output_dir='/tmp/test_vectors/',
                                     test_vector_hooks=None):
        self.record_test_vectors = True

        if test_vector_hooks:
            with open(test_vector_hooks) as json_data:
                test_vector_hooks = json.load(json_data)

        #self.test_vector_recorder = LLMLayerOutputUtil(model=self.model, dir_path=test_vector_output_dir,
        #                                              file_prefix=file_prefix, regex_patterns=test_vector_hooks)

    def disable_test_vector_recording(self):
        self.record_test_vectors = False
        del self.test_vector_recorder

    def _tokenize_text(self, text, max_length):
        if self.tokenizer == None:
            print(
                "No tokenizer was registered with forward pass manager. Attempt to forward text inputs has failed.")
            assert False

        encoded_tensor = self.tokenizer(text, add_special_tokens=False, max_length=max_length, truncation=True)
        return encoded_tensor

    def _update_kv_cache(self, prev_key_value, new_key_value, max_cache_size, is_concatenated=False):
        # past_key_value: [num_layers][2][key_value], where key_value can be a tensor or tuple of heads
        def _concat(a, b, dim):
            if isinstance(a, tuple):
                assert len(a) == len(b), 'Unexpected key/value pair'
                return tuple(_concat(ai, bi, dim) for ai, bi in zip(a, b))
            return torch.cat((a, b), dim=dim)

        def _do_concat(a, b, key_dim, value_dim):
            return tuple((_concat(ak, bk, key_dim), _concat(av, bv, value_dim)) for (ak, av), (bk, bv) in zip(a, b))

        def _shift(a, dim, shift_size):
            if isinstance(a, tuple):
                return tuple(_shift(ai, dim) for ai in a)
            assert dim in (2, 3), 'Unexpected shift axis'
            return a[:, :, shift_size:, :] if dim == 2 else a[:, :, :, shift_size:]

        def _do_shift(a, key_dim, value_dim, shift_size):
            return tuple((_shift(k, key_dim, shift_size), _shift(v, value_dim, shift_size)) for k, v in a)

        value_dim = 2
        key_dim = 3 if self.transposed_key_cache else 2

        if prev_key_value is None or is_concatenated:
            # some models concat new key values and old key values internally
            # `is_concatenated` indicates whether new_key_value is already concatenated
            next_key_value = new_key_value
        elif new_key_value is None:
            # when dummy_kv + None
            next_key_value = prev_key_value
        else:
            # if concat is NOT done, then concat
            next_key_value = _do_concat(prev_key_value, new_key_value, key_dim, value_dim)

        shift_size = next_key_value[0][1].shape[-2] - max_cache_size
        if shift_size > 0:
            next_key_value = _do_shift(next_key_value, key_dim, value_dim, shift_size)

        return next_key_value

    def validate_inputs(self, input_text=None, input_ids=None, input_embeddings=None, past_key_values=None):
        # make sure only one of input_text, input_ids, input_embeddings is passed in
        input_count = 0
        for input in (input_text, input_ids, input_embeddings):
            if input is not None:
                input_count = input_count + 1
        if input_count != 1:
            print("Incorrect number of arguments: one of (input_text, input_ids, input_embeddings) expected.")
            return False

        # make sure that input embedding function has been selected if input embeddings are to be used
        if self.use_input_embeddings and self.input_id_to_embedding_converter is None and input_embeddings is None:
            print(
                "use_input_embeddings is set to true, but no input_embeddings were provided, and input_id_to_embedding_converter is None.")
            return False

        # if past_key_values provided make sure forward pass manager is in kvcache mode
        if past_key_values is not None and self.model_mode != 'kvcache':
            print("past_key_values not expected in BERT mode.")
            return False

        if past_key_values is not None and past_key_values[0][1].shape[-2] > self.max_tokens - self.num_tokens:
            print(
                "Provided past_key_values are too long. past_key_values length cannot exceed max_tokens - num_tokens.")
            return False

        return True

    def validate_input_lengths(self, input_length, mask_length, attn_length):
        if 1 > input_length or input_length > self.num_tokens:
            print(
                f"Incorrect sequence length provided: input_length({input_length}) must be less than or equal to num_tokens (in BERT mode: max token length{self.num_tokens}, in KV mode: num_logits_to_return).")
            return False

        if attn_length < mask_length or mask_length < input_length:
            print(
                f"Incorrect attention length provided: mask_length({mask_length}) must be greator than or equal to input_lenght({input_length}) and less than or equal to the sum({attn_length}) of input_length and kv_length.")
            return False

        return True

    def validate_processed_inputs(self, input=None, attention_mask=None, past_key_values=None):
        # if input make sure that only correct length sequence is provided
        if input.shape[1] != self.num_tokens:
            print(
                f"Incorrect prcessing for sequence length: dim 1({input.shape[1]}) of input must be of length max_token_length in BERT mode, or of length num_logits_to_return in KV cache mode.")
            return False

        if attention_mask.shape[1] != self.max_tokens:
            print(
                f"Incorrect prcessing for attention length: dim 1({attention_mask.shape[1]}) of input must be of length max_token_length.")
            return False

        if past_key_values is not None and past_key_values[0][1].shape[-2] != self.max_tokens - self.num_tokens:
            print(
                f"Incorrect  prcessing for past_kv length: dim 1({past_key_values[0][1].shape[-2]}) of input must be of length max_token_length - num_token_length.")
            return False

        return True

    def get_position_embeddings_from_position_ids(self, position_ids):
        return get_position_embeddings_from_position_ids(position_ids,
                                                          head_dim=self.embed_dim // self.num_heads,
                                                          max_length=self.max_tokens,
                                                          rope_theta=self.rope_theta,
                                                          device=self.device)

    def prepare_combined_attention_mask(self, attention_mask, input_shape, past_kv_length):
        return prepare_combined_attention_mask(attention_mask, input_shape=input_shape,
                                                past_key_values_length=past_kv_length, device=self.device,
                                                mask_neg=self.mask_neg)

    def prepare_inputs(self, input_text=None, input_ids=None, input_embeddings=None, attention_mask=None,
                       past_key_values=None, **kwargs):
        assert self.validate_inputs(input_text, input_ids, input_embeddings, past_key_values)
        

        kvcache_info_bundle = {}  # dict to hold values needed for KV cache post-processing
        if input_text is not None:
            max_length = self.num_tokens if self.model_mode == 'kvcache' else self.max_tokens
            encoded = self._tokenize_text(input_text, max_length=max_length)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask

        if self.use_input_embeddings:
            if input_embeddings is None:
                input_embeddings = self.input_id_to_embedding_converter(input_ids)
            input = input_embeddings
            # if we cast this input to long, all floats become zero in the input which we do not want
            # input = torch.tensor(input.clone().detach(), dtype=torch.float32, device=self.device)
            input = input.clone().detach().to(torch.float32).to(self.device)
        else:
            input = input_ids
            # input = torch.tensor(input.clone().detach(), dtype=torch.long, device=self.device)
            input = input.clone().detach().to(torch.long).to(self.device)
        batch_size = input.shape[0]
        input_length = input.shape[1]

        kvcache_info_bundle["input_length"] = input_length

        if self.model_mode == 'kvcache' and past_key_values is not None:
            # get kv_length from past values because values are not transposed.
            kv_length = past_key_values[0][1].shape[-2]
        else:
            kv_length = 0
        attn_length = min(input_length + kv_length, self.max_tokens)

        # Checking attention_mask first, otherwise we will create attention_mask from input_extensions.
        # If model_mode=='kvcache', input_extensions will be empty tensors and so as attention_mask.
        # print("self.model_mode == ", self.model_mode)
        # print("attention_mask == ", attention_mask)
        # print("attention mask size == ", attention_mask.shape if attention_mask is not None else None)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, input_length + kv_length), dtype=torch.long, device=self.device)
        # print("after attention_mask == ", attention_mask)
        # print("after attention mask size == ", attention_mask.shape if attention_mask is not None else None)

        # cast type and move device
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(dtype=torch.long, device=self.device)
        else:
            # if attention_mask is not a tensor, get tensor
            attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        mask_length = attention_mask.shape[1]

        assert self.validate_input_lengths(input_length, mask_length, attn_length)
        # print("input_length == ", input_length)
        # print("self.num_tokens == ", self.num_tokens)
        # Pad inputs
        if input_length < self.num_tokens:
            shape = (batch_size, self.num_tokens - input_length)
            # expand shape if input is input_embeddings
            if self.use_input_embeddings:
                shape += (input.shape[-1],)
                print("self.tokenizer.eos_token_id == ", self.tokenizer.eos_token_id)
            input_extensions = torch.full(
                shape,
                fill_value=self.tokenizer.eos_token_id,
                dtype=input.dtype,
                device=self.device
            )
            input = torch.cat((input_extensions, input), dim=1)

        # Pad attention_mask
        attention_mask_extension_for_padded_kvcache = torch.zeros((batch_size, attn_length - mask_length),
                                                                  dtype=torch.long, device=self.device)
        attn_mask_extensions_for_padded_input = torch.zeros((batch_size, self.num_tokens - input_length), \
                                                            dtype=torch.long, device=self.device)
        attention_mask = torch.cat((
            attention_mask_extension_for_padded_kvcache,
            attention_mask[:, :-input_length],
            attn_mask_extensions_for_padded_input,
            attention_mask[:, -input_length:]
        ), dim=1
        )

        if self.model_mode == 'kvcache':
            desired_kv_length = self.max_tokens - self.num_tokens
            kv_padding_length = max(desired_kv_length - kv_length, 0)
            kvcache_info_bundle['kv_padding_length'] = kv_padding_length

            past_key_values_extension = get_padded_kv_values(past_size=kv_padding_length,
                                                             num_layers=self.num_layers,
                                                             hidden_size=self.embed_dim,
                                                             num_attention_heads=self.num_heads,
                                                             num_kv_heads=self.num_kv_heads,
                                                             device=self.device)
            past_key_values = self._update_kv_cache(past_key_values_extension, past_key_values, desired_kv_length)

            attention_mask_extension = torch.zeros((batch_size, kv_padding_length), dtype=torch.long,
                                                   device=self.device)
            attention_mask = torch.cat((attention_mask_extension, attention_mask), dim=1)
        else:
            kvcache_info_bundle['kv_padding_length'] = self.max_tokens - input_length

        assert self.validate_processed_inputs(input, attention_mask, past_key_values)

        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids = position_ids.clip(0, self.max_tokens - 1)
        position_ids = position_ids[..., -self.num_tokens:]

        if self.use_position_embedding_input:
            position_ids = self.get_rope_embedding(position_ids, self.device)
        
        # print("before use_combined_mask_input attention_mask == ", attention_mask)
        # print("before use_combined_mask_input attention mask size == ", attention_mask.shape if attention_mask is not None else None)

        if self.use_combined_mask_input:
            past_kv_length = self.max_tokens - self.num_tokens if self.model_mode == 'kvcache' else 0
            attention_mask = self.prepare_combined_attention_mask(attention_mask, input.shape, past_kv_length)
        # print("after use_combined_mask_input attention_mask == ", attention_mask)
        # print("after use_combined_mask_input attention mask size == ", attention_mask.shape if attention_mask is not None else None)

        inputs = {
            'attention_mask': attention_mask,
        }

        if self.separate_tuple_input_output:
            inputs['position_ids_cos'] = position_ids[0]
            inputs['position_ids_sin'] = position_ids[1]
        else:
            inputs['position_ids'] = position_ids

        if self.use_input_embeddings:
            inputs['inputs_embeds'] = input
        else:
            inputs['input_ids'] = input

        if self.model_mode == 'kvcache':
            if self.separate_tuple_input_output:
                if "input_names" in kwargs:
                    input_names = kwargs['input_names']
                else:
                    signature = inspect.signature(self.model.forward)
                    input_names = tuple(signature.parameters.keys())
                flattened_key_values = flatten_tensors(past_key_values)
                # input_ids, attention_mask, position_ids_cos, position_ids_sin, (past_key_values)
                # this order is different when we use the input_embeddings -> attention_mask, position_ids_cos, position_ids_sin, (past_key_values), inputs_embeds
                if not self.use_input_embeddings:
                    for key, value in zip(input_names[4:], flattened_key_values):
                        inputs[key] = value
                else:
                    # Because the order of inputs is different from llava
                    # inputs_embeds, attention_mask, position_ids_cos, position_ids_sin, (past_key_values) -> qwen2
                    # attention_mask, position_ids_cos, position_ids_sin, (past_key_values), inputs_embeds -> llava
                    for key, value in zip(input_names[4:], flattened_key_values):
                        inputs[key] = value
            else:
                inputs['past_key_values'] = past_key_values
        return inputs, kvcache_info_bundle

    def prepare_outputs(self, outputs, prepared_inputs, kvcache_info_bundle):
        """
        Args:
            outputs (tuple): Tuple of model outputs.
                outputs[0]: logits (batch, num_tokens, vocab_size)
                outputs[-1]: kv caches with max_tokens length
            prepared_inputs (dict): Dictionary of prepared inputs.
            kvcache_info_bundle (dict): Dictionary containing information about key-value cache.

        Returns:
            dict: A dictionary containing 'lm_logits' and 'past_key_values'.
                lm_logits: (batch, num_tokens, vocab_size)
                past_key_values: having length as the number of non-dummy inputs
        """
        lm_logits = outputs[0]
        lm_logits = lm_logits[:, -kvcache_info_bundle["input_length"]:, :]

        if self.num_logits_to_return != 0:
            num_logits_returned = lm_logits.shape[1]
            if num_logits_returned != self.num_logits_to_return:
                # print("WARN: Provided model is returning more logits than is specified in config.")
                lm_logits = lm_logits[:, -self.num_logits_to_return:, :]

        def _get_past_kv_from_outputs(outputs):
            if self.separate_tuple_input_output:
                return tuple((outputs[(2 * i) + 1], outputs[(2 * i) + 2]) for i in range(self.num_layers))
            else:
                return outputs[-1]

        def _get_past_kv_from_prepared_inputs(prepared_inputs):
            if self.separate_tuple_input_output and self.model_mode == "kvcache":
                return tuple((prepared_inputs[f"past_key_{i}_in"], prepared_inputs[f"past_value_{i}_in"]) for i in range(self.num_layers))
            else:
                return prepared_inputs['past_key_values'] if 'past_key_values' in prepared_inputs else None

        new_past_key_values = _get_past_kv_from_outputs(outputs)
        new_past_key_values = self._update_kv_cache(
            None,
            new_past_key_values,
            kvcache_info_bundle["input_length"]
        )
        old_past_key_values = _get_past_kv_from_prepared_inputs(prepared_inputs)

        current_kv_length_with_padding_removed = self.max_tokens - self.num_tokens - kvcache_info_bundle[
            'kv_padding_length'] + kvcache_info_bundle['input_length']  # number of non-dummy inputs

        past_key_values = self._update_kv_cache(
            old_past_key_values,
            new_past_key_values,
            current_kv_length_with_padding_removed,
            is_concatenated=False if self.model_mode == 'kvcache' else not self.return_new_key_value_only,
        )

        return {'lm_logits': lm_logits, 'past_key_values': past_key_values}

    def __call__(self, *args, **kwargs):
        # print("**kwargs == ", kwargs)
        # print("args == ", args)
        prepared_inputs, kvcache_info_bundle = self.prepare_inputs(*args, **kwargs)
        if self.record_test_vectors:
            assert "batch_idx" in kwargs, "Please provide batch_idx to write test vectors"
            outputs = self.test_vector_recorder.generate_layer_outputs(prepared_inputs, kwargs['batch_idx'])
        else:
            outputs = self.model(**prepared_inputs)
        prepared_outputs = self.prepare_outputs(outputs, prepared_inputs, kvcache_info_bundle)
        return prepared_outputs


def slice_inputs_and_run_successive_kvcache_inference(fpm, input_ids=None, input_embeds=None, **kwargs):
    if input_ids is not None:
        input_length = input_ids.shape[1]
    else:
        input_length = input_embeds.shape[1]

    outputs = {}

    attention_mask = kwargs.pop('attention_mask', None)

    for idx in range(0, input_length, fpm.num_tokens)[::-1]:
        idx = input_length - idx

        if attention_mask is not None:
            cache_offset = attention_mask.shape[1] - input_length
            kwargs["attention_mask"] = attention_mask[:, max(0, cache_offset + idx - fpm.max_tokens):cache_offset + idx]

        if input_ids is not None:
            cur_outputs = fpm(input_ids=input_ids[:, max(0, idx - fpm.num_tokens):idx], **kwargs)
        elif input_embeds is not None:
            cur_outputs = fpm(input_ids=None, input_embeddings=input_embeds[:, max(0, idx - fpm.num_tokens):idx, :],
                              **kwargs)
        else:
            print("No input_ids or inputs_embeds provided to inference generator!")
            assert False

        # get valid outputs
        bsz, length, dim = cur_outputs['lm_logits'].shape

        outputs['lm_logits'] = torch.cat(
            (outputs.get('lm_logits', torch.zeros((bsz, 0, dim), device=fpm.device)), cur_outputs['lm_logits']),
            dim=1)
        kwargs['past_key_values'] = outputs['past_key_values'] = cur_outputs['past_key_values']

    return outputs

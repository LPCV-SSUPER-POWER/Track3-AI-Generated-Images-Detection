#!/usr/bin/env python3
"""Qwen2-VL + LoRA + overall binary head for P1-lite."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration


QWEN2VL_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv",
    "attn.proj",
    "fc1",
    "fc2",
]


def build_peft_model(
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    dtype: torch.dtype = torch.bfloat16,
):
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    )
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=QWEN2VL_LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base, lora_cfg)


class Qwen2VLP1Lite(nn.Module):
    def __init__(self, peft_model, aux_hidden: int = 256, vision_dim: int = 1536):
        super().__init__()
        self.base = peft_model
        self.overall_head = nn.Sequential(
            nn.Linear(vision_dim, aux_hidden),
            nn.GELU(),
            nn.Linear(aux_hidden, 1),
        )
        self._vision_embeds_cache = None
        self._register_vision_hook()

    def _register_vision_hook(self):
        def hook(module, inputs, output):
            self._vision_embeds_cache = output

        base_inner = self.base.get_base_model() if hasattr(self.base, "get_base_model") else self.base
        base_inner.visual.register_forward_hook(hook)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        overall_label,
    ):
        self._vision_embeds_cache = None
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        schema_loss = outputs.loss

        if self._vision_embeds_cache is None:
            raise RuntimeError("vision hook cache is empty")

        pooled = self._vision_embeds_cache.mean(dim=0, keepdim=True)
        pooled = pooled.to(self.overall_head[0].weight.dtype)
        logits = self.overall_head(pooled).squeeze(-1)

        target = torch.tensor(
            [overall_label] if isinstance(overall_label, (int, float)) else overall_label,
            device=logits.device,
            dtype=logits.dtype,
        )
        cls_loss = F.binary_cross_entropy_with_logits(logits, target)
        return schema_loss, cls_loss

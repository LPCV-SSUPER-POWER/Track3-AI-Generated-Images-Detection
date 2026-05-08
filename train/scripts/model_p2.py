#!/usr/bin/env python3
"""Qwen2-VL + LoRA + auxiliary overall / criterion heads for P2."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration


def find_all_linear_names(model: nn.Module, *, exclude_visual: bool = False) -> list[str]:
    linear_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name.endswith("lm_head"):
                continue
            if exclude_visual and "visual" in name:
                continue
            linear_names.add(name)
    return sorted(linear_names)


def build_peft_model(
    base_model_id: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    freeze_vision_tower: bool = True,
):
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    )
    target_modules = find_all_linear_names(base, exclude_visual=freeze_vision_tower)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_cfg)
    if freeze_vision_tower:
        for name, param in peft_model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
    trainable_visual = [name for name, param in peft_model.named_parameters() if "visual" in name and param.requires_grad]
    print(
        f"[*] LoRA target modules: {len(target_modules)} total "
        f"(freeze_vision_tower={freeze_vision_tower}, trainable_visual={len(trainable_visual)})"
    )
    if trainable_visual:
        preview = ", ".join(trainable_visual[:8])
        print(f"[!] trainable visual parameters remain: {preview}")
    return peft_model


class Qwen2VLP2Aux(nn.Module):
    def __init__(
        self,
        peft_model,
        *,
        hidden_dim: int = 1536,
        aux_hidden: int = 256,
        use_overall_loss: bool = True,
        use_criterion_loss: bool = False,
        criterion_pos_weight: list[float] | None = None,
    ):
        super().__init__()
        self.base = peft_model
        self.use_overall_loss = bool(use_overall_loss)
        self.use_criterion_loss = bool(use_criterion_loss)
        pos_weight_tensor = None
        if criterion_pos_weight is not None:
            if len(criterion_pos_weight) != 8:
                raise ValueError("criterion_pos_weight must contain exactly 8 values")
            pos_weight_tensor = torch.tensor(criterion_pos_weight, dtype=torch.float32)
        self.register_buffer("criterion_pos_weight", pos_weight_tensor)

        self.overall_head = nn.Sequential(
            nn.Linear(hidden_dim, aux_hidden),
            nn.GELU(),
            nn.Linear(aux_hidden, 1),
        )
        self.criterion_head = nn.Sequential(
            nn.Linear(hidden_dim, aux_hidden),
            nn.GELU(),
            nn.Linear(aux_hidden, 8),
        )

    def _prompt_representation(
        self,
        *,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_pixel_values: torch.Tensor | None = None,
        prompt_image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.base(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            pixel_values=prompt_pixel_values,
            image_grid_thw=prompt_image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = prompt_attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(self, **batch: Any) -> dict[str, torch.Tensor]:
        outputs = self.base(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            use_cache=False,
            return_dict=True,
        )
        tok_loss = outputs.loss

        pooled = self._prompt_representation(
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            prompt_pixel_values=batch.get("prompt_pixel_values"),
            prompt_image_grid_thw=batch.get("prompt_image_grid_thw"),
        )
        pooled = pooled.to(self.overall_head[0].weight.dtype)

        device = pooled.device
        zero = pooled.new_zeros(())
        overall_loss = zero
        criterion_loss = zero

        if self.use_overall_loss:
            overall_logits = self.overall_head(pooled).squeeze(-1)
            overall_targets = batch["overall_labels"].to(device=device, dtype=overall_logits.dtype)
            overall_mask = batch["overall_mask"].to(device=device, dtype=overall_logits.dtype)
            raw_overall = F.binary_cross_entropy_with_logits(overall_logits, overall_targets, reduction="none")
            denom = overall_mask.sum().clamp_min(1.0)
            overall_loss = (raw_overall * overall_mask).sum() / denom

        if self.use_criterion_loss:
            criterion_logits = self.criterion_head(pooled)
            criterion_targets = batch["criterion_labels"].to(device=device, dtype=criterion_logits.dtype)
            criterion_mask = batch["criterion_mask"].to(device=device, dtype=criterion_logits.dtype)
            pos_weight = None
            if self.criterion_pos_weight is not None:
                pos_weight = self.criterion_pos_weight.to(device=device, dtype=criterion_logits.dtype)
            raw_criterion = F.binary_cross_entropy_with_logits(
                criterion_logits, criterion_targets, reduction="none", pos_weight=pos_weight
            )
            denom = criterion_mask.sum().clamp_min(1.0)
            criterion_loss = (raw_criterion * criterion_mask).sum() / denom

        return {
            "tok_loss": tok_loss,
            "overall_loss": overall_loss,
            "criterion_loss": criterion_loss,
        }

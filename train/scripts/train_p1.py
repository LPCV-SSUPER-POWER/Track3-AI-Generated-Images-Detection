#!/usr/bin/env python3
"""Train P1-lite with custom overall classification + short-schema CE."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for train_p1.py") from exc

from dataset_p1 import P1LiteDataset
from model_p1 import Qwen2VLP1Lite, build_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_single(batch):
    assert len(batch) == 1
    return batch[0]


def move_to_device(sample: dict, device: torch.device) -> dict:
    out = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(cfg.get("device", "cuda"))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = output_dir / "lora_p1"
    head_path = output_dir / "overall_cls_head.pt"
    log_path = output_dir / "train_log_p1.json"
    saved_cfg_path = output_dir / "train_p1_resolved_config.json"

    base_model = cfg["base_model"]
    processor_kwargs = {}
    if cfg.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = int(cfg["min_pixels"])
    if cfg.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = int(cfg["max_pixels"])
    processor = AutoProcessor.from_pretrained(base_model, **processor_kwargs)
    peft_model = build_peft_model(
        base_model_id=base_model,
        lora_r=int(cfg.get("lora_rank", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
    )
    peft_model.print_trainable_parameters()
    model = Qwen2VLP1Lite(peft_model).to(device)
    model.overall_head = model.overall_head.to(torch.bfloat16)

    train_ds = P1LiteDataset(
        jsonl_path=cfg["train_jsonl"],
        processor=processor,
        limit=cfg.get("limit"),
        min_image_edge=int(cfg.get("min_image_edge", 28)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_single,
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.get("learning_rate", 1e-4)))

    num_epochs = float(cfg.get("num_train_epochs", 1.0))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 16))
    total_optim_steps = int(math.ceil(len(train_loader) * num_epochs / grad_accum))
    warmup_steps = max(1, int(total_optim_steps * float(cfg.get("warmup_ratio", 0.05))))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )

    lambda_overall = float(cfg.get("lambda_overall", 2.0))
    lambda_schema = float(cfg.get("lambda_schema", 1.0))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 500))

    print(f"[*] train rows: {len(train_ds)}")
    if getattr(train_ds, "skipped_rows", 0):
        print(
            f"[*] filtered invalid rows: {train_ds.skipped_rows} "
            f"(min_image_edge={train_ds.min_image_edge})"
        )
        for example in train_ds.skipped_examples[:5]:
            print(f"    - {example['image_path']}: {example['reason']}")
    print(f"[*] total optim steps: {total_optim_steps}, warmup: {warmup_steps}")
    print(f"[*] lambda_overall={lambda_overall}, lambda_schema={lambda_schema}")

    logs = []
    step = 0
    accum = 0
    seen = 0
    skipped = 0
    running_schema = 0.0
    running_cls = 0.0
    t0 = time.time()
    samples_target = int(math.ceil(len(train_loader) * num_epochs))

    for epoch in range(int(math.ceil(num_epochs))):
        for sample in train_loader:
            if seen >= samples_target or step >= total_optim_steps:
                break
            seen += 1
            sample = move_to_device(sample, device)

            try:
                schema_loss, cls_loss = model(
                    input_ids=sample["input_ids"].unsqueeze(0),
                    attention_mask=sample["attention_mask"].unsqueeze(0),
                    labels=sample["labels"].unsqueeze(0),
                    pixel_values=sample["pixel_values"],
                    image_grid_thw=sample["image_grid_thw"],
                    overall_label=sample["overall_label"],
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1
                continue

            total_loss = lambda_schema * schema_loss + lambda_overall * cls_loss
            (total_loss / grad_accum).backward()

            running_schema += schema_loss.item()
            running_cls += cls_loss.item()
            accum += 1

            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                avg_schema = running_schema / grad_accum
                avg_cls = running_cls / grad_accum
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0

                if step % log_every == 0 or step == 1:
                    print(
                        f"  step {step:4d}/{total_optim_steps} | "
                        f"schema={avg_schema:.4f} cls={avg_cls:.4f} "
                        f"lr={lr_now:.2e} elapsed={elapsed:.0f}s skipped={skipped}"
                    )

                logs.append(
                    {
                        "step": step,
                        "schema_loss": avg_schema,
                        "cls_loss": avg_cls,
                        "lr": lr_now,
                    }
                )
                running_schema = 0.0
                running_cls = 0.0
                accum = 0

                if step % save_every == 0:
                    print(f"  [checkpoint] saving at step {step}")
                    model.base.save_pretrained(str(lora_dir))
                    torch.save(model.overall_head.state_dict(), str(head_path))

        if seen >= samples_target or step >= total_optim_steps:
            break

    model.base.save_pretrained(str(lora_dir))
    torch.save(model.overall_head.state_dict(), str(head_path))
    saved_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(
        json.dumps(
            {
                "config": cfg,
                "total_steps": step,
                "total_samples_seen": seen,
                "skipped": skipped,
                "logs": logs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[✓] P1-lite training complete: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train custom P2 variants with auxiliary overall / criterion losses."""

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
    raise RuntimeError("PyYAML is required for train_p2.py") from exc

from dataset_p2 import P2AuxCollator, P2AuxDataset
from model_p2 import Qwen2VLP2Aux, build_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(sample: dict, device: torch.device) -> dict:
    out = {}
    for key, value in sample.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def interleave_schedule(n_image: int, n_text: int) -> list[str]:
    schedule: list[str] = []
    i = 0
    j = 0
    while i < n_image or j < n_text:
        choose_image = False
        if i < n_image and j < n_text:
            choose_image = (i + 1) / max(n_image, 1) <= (j + 1) / max(n_text, 1)
        elif i < n_image:
            choose_image = True

        if choose_image:
            schedule.append("image")
            i += 1
        else:
            schedule.append("text")
            j += 1
    return schedule


def cycle_next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def save_checkpoint(model, output_dir: Path, overall_path: Path, criterion_path: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(str(output_dir))
    torch.save(model.overall_head.state_dict(), str(overall_path))
    if criterion_path is not None:
        torch.save(model.criterion_head.state_dict(), str(criterion_path))


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(cfg.get("device", "cuda"))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = output_dir / Path(cfg.get("lora_subdir", "lora_p2"))
    overall_head_path = output_dir / cfg.get("overall_head_name", "overall_head.pt")
    criterion_head_path = (
        output_dir / cfg.get("criterion_head_name", "criterion_head.pt")
        if bool(cfg.get("use_criterion_loss", False))
        else None
    )
    log_path = output_dir / cfg.get("log_name", "train_log_p2.json")
    saved_cfg_path = output_dir / cfg.get("resolved_config_name", "train_p2_resolved_config.json")

    processor_kwargs = {}
    if cfg.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = int(cfg["min_pixels"])
    if cfg.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = int(cfg["max_pixels"])
    processor = AutoProcessor.from_pretrained(cfg["base_model"], **processor_kwargs)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    peft_model = build_peft_model(
        base_model_id=cfg["base_model"],
        lora_r=int(cfg.get("lora_rank", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        freeze_vision_tower=bool(cfg.get("freeze_vision_tower", True)),
    )
    peft_model.print_trainable_parameters()
    model = Qwen2VLP2Aux(
        peft_model,
        hidden_dim=int(cfg.get("hidden_dim", 1536)),
        aux_hidden=int(cfg.get("aux_hidden", 256)),
        use_overall_loss=bool(cfg.get("use_overall_loss", True)),
        use_criterion_loss=bool(cfg.get("use_criterion_loss", False)),
    ).to(device)

    train_jsonl = cfg["train_jsonl"]
    image_ds = P2AuxDataset(
        train_jsonl,
        processor,
        image_rows=True,
        limit=cfg.get("limit"),
        min_image_edge=int(cfg.get("min_image_edge", 28)),
    )
    text_ds = P2AuxDataset(
        train_jsonl,
        processor,
        image_rows=False,
        limit=cfg.get("limit"),
        min_image_edge=int(cfg.get("min_image_edge", 28)),
    )

    batch_size = int(cfg.get("per_device_train_batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))
    image_loader = DataLoader(
        image_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=P2AuxCollator(processor, image_rows=True),
    )
    text_loader = DataLoader(
        text_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=P2AuxCollator(processor, image_rows=False),
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.get("learning_rate", 1e-4)))

    num_epochs = float(cfg.get("num_train_epochs", 1.0))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    per_epoch_microbatches = len(image_loader) + len(text_loader)
    total_optim_steps = int(math.ceil(per_epoch_microbatches * num_epochs / grad_accum))
    warmup_steps = max(1, int(total_optim_steps * float(cfg.get("warmup_ratio", 0.05))))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )

    lambda_tok = float(cfg.get("lambda_tok", 1.0))
    lambda_overall = float(cfg.get("lambda_overall", 0.5))
    lambda_criterion = float(cfg.get("lambda_criterion", 0.25))
    log_every = int(cfg.get("log_every", 20))
    save_every = int(cfg.get("save_every", 200))

    schedule = interleave_schedule(len(image_loader), len(text_loader))
    total_microbatches_target = int(math.ceil(len(schedule) * num_epochs))

    print(f"[*] image rows: {len(image_ds)} ({len(image_loader)} batches)")
    print(f"[*] text rows:  {len(text_ds)} ({len(text_loader)} batches)")
    print(f"[*] total optim steps: {total_optim_steps}, warmup: {warmup_steps}")
    print(
        f"[*] lambda_tok={lambda_tok}, lambda_overall={lambda_overall}, "
        f"lambda_criterion={lambda_criterion}, use_criterion_loss={cfg.get('use_criterion_loss', False)}"
    )

    image_iter = iter(image_loader)
    text_iter = iter(text_loader)
    logs = []
    step = 0
    accum = 0
    seen_microbatches = 0
    running_tok = 0.0
    running_overall = 0.0
    running_criterion = 0.0
    t0 = time.time()

    optimizer.zero_grad()

    while seen_microbatches < total_microbatches_target and step < total_optim_steps:
        for row_type in schedule:
            if seen_microbatches >= total_microbatches_target or step >= total_optim_steps:
                break

            if row_type == "image":
                batch, image_iter = cycle_next(image_iter, image_loader)
            else:
                batch, text_iter = cycle_next(text_iter, text_loader)

            seen_microbatches += 1
            batch = move_to_device(batch, device)

            losses = model(**batch)
            total_loss = lambda_tok * losses["tok_loss"]
            if bool(cfg.get("use_overall_loss", True)):
                total_loss = total_loss + lambda_overall * losses["overall_loss"]
            if bool(cfg.get("use_criterion_loss", False)):
                total_loss = total_loss + lambda_criterion * losses["criterion_loss"]

            (total_loss / grad_accum).backward()
            running_tok += float(losses["tok_loss"].item())
            running_overall += float(losses["overall_loss"].item())
            running_criterion += float(losses["criterion_loss"].item())
            accum += 1

            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                avg_tok = running_tok / grad_accum
                avg_overall = running_overall / grad_accum
                avg_criterion = running_criterion / grad_accum
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0

                if step % log_every == 0 or step == 1:
                    print(
                        f"  step {step:4d}/{total_optim_steps} | "
                        f"tok={avg_tok:.4f} overall={avg_overall:.4f} criterion={avg_criterion:.4f} "
                        f"lr={lr_now:.2e} elapsed={elapsed:.0f}s"
                    )

                logs.append(
                    {
                        "step": step,
                        "tok_loss": avg_tok,
                        "overall_loss": avg_overall,
                        "criterion_loss": avg_criterion,
                        "lr": lr_now,
                    }
                )
                running_tok = 0.0
                running_overall = 0.0
                running_criterion = 0.0
                accum = 0

                if step % save_every == 0:
                    ckpt_dir = lora_dir / f"checkpoint-{step}"
                    print(f"  [checkpoint] saving at step {step}")
                    save_checkpoint(model, ckpt_dir, overall_head_path, criterion_head_path)

    save_checkpoint(model, lora_dir, overall_head_path, criterion_head_path)
    saved_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(
        json.dumps(
            {
                "config": cfg,
                "total_steps": step,
                "total_microbatches_seen": seen_microbatches,
                "logs": logs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[✓] P2 aux training complete: {output_dir}")


if __name__ == "__main__":
    main()

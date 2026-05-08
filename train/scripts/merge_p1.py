#!/usr/bin/env python3
"""Merge P1-lite LoRA into the base Qwen2-VL model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for merge_p1.py") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_model = cfg["model_name_or_path"]
    adapter_path = cfg["adapter_name_or_path"]
    export_dir = Path(cfg["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] loading base: {base_model}")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(base_model)

    print(f"[*] loading adapter: {adapter_path}")
    peft = PeftModel.from_pretrained(base, adapter_path)

    print("[*] merging adapter")
    merged = peft.merge_and_unload()
    merged.save_pretrained(str(export_dir), safe_serialization=True)
    processor.save_pretrained(str(export_dir))
    print(f"[✓] merged model written to: {export_dir}")


if __name__ == "__main__":
    main()

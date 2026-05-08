#!/usr/bin/env python3
"""Dataset for P1-lite.

Reads the short structured JSON targets generated in
`datasets/data_p1/{train,val}.jsonl` and exposes:

- standard SFT labels for the short JSON target
- an explicit overall Real/AI-Generated label for auxiliary classification
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


LABEL_MAP = {
    "Real": 0.0,
    "AI-Generated": 1.0,
}


def parse_overall_label(assistant_text: str) -> float:
    payload = json.loads(assistant_text)
    label = payload.get("overall_label")
    if label not in LABEL_MAP:
        raise ValueError(f"unexpected overall_label: {label!r}")
    return LABEL_MAP[label]


class P1LiteDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        limit: int | None = None,
        min_image_edge: int = 28,
    ):
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if limit is not None:
            rows = rows[:limit]

        self.rows = []
        self.processor = processor
        self.min_image_edge = int(min_image_edge)
        self.skipped_rows = 0
        self.skipped_examples = []

        for row in rows:
            images = row.get("images") or []
            if len(images) != 1:
                self.skipped_rows += 1
                if len(self.skipped_examples) < 10:
                    self.skipped_examples.append(
                        {
                            "image_path": images[0] if images else None,
                            "reason": f"expected exactly one image, got {len(images)}",
                        }
                    )
                continue

            image_path = images[0]
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as exc:
                self.skipped_rows += 1
                if len(self.skipped_examples) < 10:
                    self.skipped_examples.append(
                        {
                            "image_path": image_path,
                            "reason": f"image_open_failed: {exc}",
                        }
                    )
                continue

            if width < self.min_image_edge or height < self.min_image_edge:
                self.skipped_rows += 1
                if len(self.skipped_examples) < 10:
                    self.skipped_examples.append(
                        {
                            "image_path": image_path,
                            "reason": f"image_too_small: {width}x{height} < {self.min_image_edge}",
                        }
                    )
                continue

            self.rows.append(row)

        if not self.rows:
            raise RuntimeError(
                f"no valid rows remain in {jsonl_path} after filtering with min_image_edge={self.min_image_edge}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        images = row.get("images") or []
        if len(images) != 1:
            raise ValueError(f"P1 row must contain exactly one image: {row}")

        convs = row["conversations"]
        user_text = convs[0]["value"].replace("<image>\n", "").replace("<image>", "").strip()
        assistant_text = convs[1]["value"].strip()
        overall_label = parse_overall_label(assistant_text)

        image_path = images[0]
        image = Image.open(image_path).convert("RGB")

        user_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_text},
        ]
        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        input_ids = full_inputs["input_ids"][0]
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full_inputs["attention_mask"][0],
            "labels": labels,
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"],
            "overall_label": overall_label,
            "_image_path": image_path,
        }

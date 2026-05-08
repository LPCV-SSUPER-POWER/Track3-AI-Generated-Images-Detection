#!/usr/bin/env python3
"""Dataset utilities for custom P2 auxiliary-loss training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


ALL_CRITERIA = [
    "Lighting & Shadows Consistency",
    "Edges & Boundaries",
    "Texture & Resolution",
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
]

CRITERION_TO_INDEX = {name: idx for idx, name in enumerate(ALL_CRITERIA)}
OVERALL_MAP = {
    "Real": 0.0,
    "AI-Generated": 1.0,
}

P3_LINE_RE = re.compile(
    r"^\-\s*(?P<criterion>.+?):\s*(?P<prefix>Observation|Artifact):\s*(?P<evidence>.*)$"
)
CONCLUSION_RE = re.compile(r"Conclusion:\s*The image is most likely\s*(?P<label>Real|AI-Generated|Uncertain)\.?", re.I)


@dataclass
class ParsedTargets:
    overall_label: float
    overall_mask: float
    criterion_labels: list[float]
    criterion_mask: list[float]


def _empty_targets() -> ParsedTargets:
    return ParsedTargets(
        overall_label=0.0,
        overall_mask=0.0,
        criterion_labels=[0.0] * len(ALL_CRITERIA),
        criterion_mask=[0.0] * len(ALL_CRITERIA),
    )


def parse_p3_targets(assistant_text: str) -> ParsedTargets:
    targets = _empty_targets()
    for raw_line in assistant_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = P3_LINE_RE.match(line)
        if match:
            criterion = match.group("criterion").strip()
            prefix = match.group("prefix").strip()
            if criterion not in CRITERION_TO_INDEX:
                continue
            idx = CRITERION_TO_INDEX[criterion]
            targets.criterion_labels[idx] = 1.0 if prefix == "Artifact" else 0.0
            targets.criterion_mask[idx] = 1.0
            continue

        conclusion = CONCLUSION_RE.search(line)
        if conclusion:
            label = conclusion.group("label").strip()
            if label in OVERALL_MAP:
                targets.overall_label = OVERALL_MAP[label]
                targets.overall_mask = 1.0

    return targets


def parse_p4_targets(assistant_text: str) -> ParsedTargets:
    payload = json.loads(assistant_text)
    targets = _empty_targets()

    overall = payload.get("overall_likelihood")
    if overall in OVERALL_MAP:
        targets.overall_label = OVERALL_MAP[overall]
        targets.overall_mask = 1.0

    for item in payload.get("per_criterion", []):
        criterion = item.get("criterion", "").strip()
        if criterion not in CRITERION_TO_INDEX:
            continue
        idx = CRITERION_TO_INDEX[criterion]
        score = item.get("aigc score", item.get("aigc_score", item.get("score", 0)))
        targets.criterion_labels[idx] = 1.0 if int(score) > 0 else 0.0
        targets.criterion_mask[idx] = 1.0

    return targets


def parse_targets(assistant_text: str) -> ParsedTargets:
    text = assistant_text.lstrip()
    if text.startswith("{"):
        return parse_p4_targets(text)
    return parse_p3_targets(text)


def _build_prompt_messages(human_text: str, image_path: str | None) -> list[dict[str, Any]]:
    if image_path is None:
        return [{"role": "user", "content": human_text}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": human_text},
            ],
        }
    ]


class P2AuxDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        *,
        image_rows: bool,
        limit: int | None = None,
        min_image_edge: int = 28,
    ):
        self.processor = processor
        self.image_rows = bool(image_rows)
        self.min_image_edge = int(min_image_edge)
        self.rows: list[dict[str, Any]] = []
        self.skipped_rows = 0

        path = Path(jsonl_path)
        raw_rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    raw_rows.append(json.loads(line))
        if limit is not None:
            raw_rows = raw_rows[:limit]

        for row in raw_rows:
            images = row.get("images") or []
            is_image_row = len(images) == 1
            if is_image_row != self.image_rows:
                continue

            image_path = images[0] if is_image_row else None
            if image_path is not None:
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                except Exception:
                    self.skipped_rows += 1
                    continue
                if width < self.min_image_edge or height < self.min_image_edge:
                    self.skipped_rows += 1
                    continue

            convs = row["conversations"]
            if len(convs) != 2:
                self.skipped_rows += 1
                continue

            human_text = convs[0]["value"].replace("<image>\n", "").replace("<image>", "").strip()
            assistant_text = convs[1]["value"].strip()
            targets = parse_targets(assistant_text)

            self.rows.append(
                {
                    "human_text": human_text,
                    "assistant_text": assistant_text,
                    "image_path": image_path,
                    "overall_label": targets.overall_label,
                    "overall_mask": targets.overall_mask,
                    "criterion_labels": targets.criterion_labels,
                    "criterion_mask": targets.criterion_mask,
                }
            )

        if not self.rows:
            raise RuntimeError(f"no valid rows remain in {jsonl_path} for image_rows={self.image_rows}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt_messages = _build_prompt_messages(row["human_text"], row["image_path"])
        full_messages = prompt_messages + [{"role": "assistant", "content": row["assistant_text"]}]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        return {
            "prompt_text": prompt_text,
            "full_text": full_text,
            "image_path": row["image_path"],
            "assistant_text": row["assistant_text"],
            "overall_label": row["overall_label"],
            "overall_mask": row["overall_mask"],
            "criterion_labels": row["criterion_labels"],
            "criterion_mask": row["criterion_mask"],
        }


class P2AuxCollator:
    def __init__(self, processor, *, image_rows: bool):
        self.processor = processor
        self.image_rows = bool(image_rows)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts = [item["prompt_text"] for item in batch]
        full_texts = [item["full_text"] for item in batch]

        if self.image_rows:
            images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
            full_inputs = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True)
            prompt_inputs = self.processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
            for image in images:
                image.close()
        else:
            full_inputs = self.processor(text=full_texts, return_tensors="pt", padding=True)
            prompt_inputs = self.processor(text=prompt_texts, return_tensors="pt", padding=True)

        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lengths):
            labels[idx, : int(prompt_len)] = -100

        output = {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "labels": labels,
            "prompt_input_ids": prompt_inputs["input_ids"],
            "prompt_attention_mask": prompt_inputs["attention_mask"],
            "overall_labels": full_inputs["input_ids"].new_tensor(
                [item["overall_label"] for item in batch], dtype=full_inputs["input_ids"].dtype
            ).float(),
            "overall_mask": full_inputs["input_ids"].new_tensor(
                [item["overall_mask"] for item in batch], dtype=full_inputs["input_ids"].dtype
            ).float(),
            "criterion_labels": full_inputs["input_ids"].new_tensor(
                [item["criterion_labels"] for item in batch], dtype=full_inputs["input_ids"].dtype
            ).float(),
            "criterion_mask": full_inputs["input_ids"].new_tensor(
                [item["criterion_mask"] for item in batch], dtype=full_inputs["input_ids"].dtype
            ).float(),
            "row_type": "image" if self.image_rows else "text",
        }

        if self.image_rows:
            output["pixel_values"] = full_inputs["pixel_values"]
            output["image_grid_thw"] = full_inputs["image_grid_thw"]
            output["prompt_pixel_values"] = prompt_inputs["pixel_values"]
            output["prompt_image_grid_thw"] = prompt_inputs["image_grid_thw"]

        return output

#!/usr/bin/env python3
"""
Build P1 training entries (Step 3 of annotation pipeline).

Reads split folder:
- annotations/data_p1   (raw annotation JSON, hardlink from master pool)

Writes ShareGPT-style JSONL:
- datasets/data_p1/{train,val}.jsonl

P2/P3/P4 are handled by build_p234_jsonl.py. This script only processes P1.
Standalone (0 library imports).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# Self-contained ROOT detection: this file lives at annotation/jsonl_build/
_THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_ROOT = _THIS_DIR.parent.parent  # jsonl_build/ -> annotation/ -> bundle root
PROMPTS_DIR = _THIS_DIR.parent / "prompts"  # annotation/prompts/
ROOT = BUNDLE_ROOT  # default; user can override with --annotations_root / --datasets_root
ANNOTATIONS_ROOT = ROOT / "annotations"
DATASETS_ROOT = ROOT / "datasets"

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

P1_PROMPT = """Classify this image per criterion and overall. Return JSON only in the exact schema below.
{
  "lighting": "Real" | "AI-Generated",
  "edge": "Real" | "AI-Generated",
  "texture": "Real" | "AI-Generated",
  "perspective": "Real" | "AI-Generated",
  "commonsense": "Real" | "AI-Generated",
  "text_symbol": "Real" | "AI-Generated",
  "human": "Real" | "AI-Generated",
  "material": "Real" | "AI-Generated",
  "overall_label": "Real" | "AI-Generated"
}
Do not add explanation. Output JSON only."""

P3_CRITERIA = [
    "Edges & Boundaries",
    "Texture & Resolution",
    "Material & Object Details",
    "Physical & Common Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Lighting & Shadows Consistency",
    "Perspective & Spatial Relationships",
]

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

CRITERION_ALIASES = {
    "Lighting & Shadows Consistency": "Lighting & Shadows Consistency",
    "Lighting & Shadow Consistency": "Lighting & Shadows Consistency",
    "Edges & Boundaries": "Edges & Boundaries",
    "Texture & Resolution": "Texture & Resolution",
    "Texture & Resolution Coherence": "Texture & Resolution",
    "Perspective & Spatial Relationships": "Perspective & Spatial Relationships",
    "Perspective & Spatial Accuracy": "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic": "Physical & Common Sense Logic",
    "Physical & Common-Sense Logic": "Physical & Common Sense Logic",
    "Text & Symbols": "Text & Symbols",
    "Text & Symbol Authenticity": "Text & Symbols",
    "Human & Biological Structure Integrity": "Human & Biological Structure Integrity",
    "Material & Object Details": "Material & Object Details",
    "Material & Object Detail Fidelity": "Material & Object Details",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage-specific SFT JSONL from split manifests")
    parser.add_argument("--annotations_root", default=str(ANNOTATIONS_ROOT))
    parser.add_argument("--datasets_root", default=str(DATASETS_ROOT))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p2_max_tokens", type=int, default=1600)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def approx_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def clean_text(text: str) -> str:
    text = str(text or "").replace("[BEGIN]:", "").replace("[END]", "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_text_prompt(path: Path) -> str:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().strip('"').strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def load_manifest_rows(split_dir: Path) -> list[dict[str, Any]]:
    path = split_dir / "_split_manifest.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_p1_targets(split_dir: Path) -> dict[str, dict[str, str]]:
    path = split_dir / "_p1_structured_targets.jsonl"
    mapping = {}
    if not path.exists():
        return mapping
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["annotation_path"]] = row["target"]
    return mapping


def load_annotation_by_path(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def partition_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["source_family"], row["label"])].append(row)

    train_rows = []
    val_rows = []
    for key, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        if len(bucket) == 1:
            n_val = 0
        else:
            n_val = int(len(bucket) * val_ratio)
            n_val = max(1, min(len(bucket) - 1, n_val))
        val_rows.extend(bucket[:n_val])
        train_rows.extend(bucket[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def partition_p4_from_p3(p4_rows: list[dict[str, Any]], p3_train: list[dict[str, Any]], p3_val: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_paths = {row["annotation_path"] for row in p3_train}
    val_paths = {row["annotation_path"] for row in p3_val}
    train_rows = [row for row in p4_rows if row["annotation_path"] in train_paths]
    val_rows = [row for row in p4_rows if row["annotation_path"] in val_paths]
    return train_rows, val_rows


def compress_sentence(text: str, max_words: int) -> str:
    words = clean_text(text).replace("\n", " ").split()
    if len(words) <= max_words:
        return " ".join(words).strip()
    clipped = " ".join(words[:max_words]).strip(" ,;:")
    for sep in [". ", "; ", ", "]:
        idx = clipped.rfind(sep)
        if idx > len(clipped) * 0.55:
            clipped = clipped[: idx + 1].strip()
            break
    if clipped and clipped[-1] not in ".!?":
        clipped += "."
    return clipped


def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    matches = list(TOKEN_RE.finditer(text))
    if len(matches) <= max_tokens:
        return text.strip()
    end_idx = matches[max_tokens - 1].end()
    clipped = text[:end_idx].strip()
    for sep in ["\n\n", ". ", "; "]:
        idx = clipped.rfind(sep)
        if idx > len(clipped) * 0.55:
            clipped = clipped[: idx + (0 if sep == "\n\n" else 1)].strip()
            break
    if clipped and clipped[-1] not in ".!?}]":
        clipped += "."
    return clipped


def normalize_score(value: Any) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        text = clean_text(value).lower().replace("_", "-")
        if text in {"fake", "ai-generated", "ai generated"}:
            parsed = 1
        else:
            parsed = 0
    return 0 if parsed <= 0 else 1


def normalize_overall(value: Any, fake_votes: int) -> str:
    text = clean_text(value)
    if text in {"Real", "AI-Generated", "Uncertain"}:
        return text
    return "AI-Generated" if fake_votes > 0 else "Real"


def build_lookup(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for item in annotation.get("per_criterion", []):
        key = CRITERION_ALIASES.get(item.get("criterion", ""), item.get("criterion", ""))
        lookup[key] = item
    return lookup


def build_structured_analysis(annotation: dict[str, Any], evidence_words: int) -> str:
    lookup = build_lookup(annotation)
    lines = ["Structured analysis:"]
    fake_votes = 0
    for criterion in P3_CRITERIA:
        item = lookup.get(criterion, {})
        evidence = item.get("evidence", "No decisive artifact is visible for this criterion.")
        score = normalize_score(item.get("aigc score", item.get("aigc_score", item.get("score", 0))))
        fake_votes += score
        prefix = "Artifact:" if score else "Observation:"
        lines.append(f"- {criterion}: {prefix} {compress_sentence(evidence, evidence_words)}")
    overall = normalize_overall(annotation.get("overall_likelihood"), fake_votes)
    if overall == "AI-Generated":
        lines.append("Conclusion: The image is most likely AI-Generated because multiple criteria show artifact evidence.")
    elif overall == "Real":
        lines.append("Conclusion: The image is most likely Real because the criteria remain visually coherent without decisive generation artifacts.")
    else:
        lines.append("Conclusion: The image remains Uncertain because the evidence is mixed and not all criteria agree.")
    return "\n".join(lines)


def build_full_analysis_text(annotation: dict[str, Any], max_tokens: int) -> str:
    responses = [clean_text(r) for r in (annotation.get("_meta", {}).get("a_step1_responses") or []) if clean_text(r)]
    if responses:
        body = "\n\n".join(f"[Analysis {idx}]\n{text}" for idx, text in enumerate(responses, start=1))
    else:
        body = build_structured_analysis(annotation, evidence_words=40)

    overall = normalize_overall(annotation.get("overall_likelihood"), 0)
    if overall == "AI-Generated":
        tail = "Final conclusion: The image is most likely AI-Generated."
    elif overall == "Real":
        tail = "Final conclusion: The image is most likely Real."
    else:
        tail = "Final conclusion: The image remains Uncertain."

    text = f"{body}\n\n{tail}".strip()
    return truncate_text_by_tokens(text, max_tokens)


def build_under500_analysis(annotation: dict[str, Any]) -> str:
    responses = [clean_text(r) for r in (annotation.get("_meta", {}).get("a_step1_responses") or []) if clean_text(r)]
    best_text = ""
    best_tokens = 10**9

    if responses:
        for budget in range(120, 69, -10):
            compact = []
            for idx, response in enumerate(responses[:3], start=1):
                clipped = truncate_text_by_tokens(response, budget)
                compact.append(f"[Analysis {idx}]\n{clipped}")
            overall = normalize_overall(annotation.get("overall_likelihood"), 0)
            if overall == "AI-Generated":
                compact.append("Final conclusion: The image is most likely AI-Generated.")
            elif overall == "Real":
                compact.append("Final conclusion: The image is most likely Real.")
            else:
                compact.append("Final conclusion: The image remains Uncertain.")
            text = "\n\n".join(compact)
            token_count = approx_tokens(text)
            if token_count <= 500:
                best_text = text
                best_tokens = token_count
                if token_count >= 330:
                    return text
            elif token_count < best_tokens:
                best_text = text
                best_tokens = token_count

    for evidence_words in range(20, 7, -2):
        text = build_structured_analysis(annotation, evidence_words=evidence_words)
        token_count = approx_tokens(text)
        if token_count <= 500:
            if not best_text or abs(token_count - 360) < abs(best_tokens - 360):
                best_text = text
                best_tokens = token_count
            if token_count >= 280:
                return text

    if not best_text:
        best_text = build_structured_analysis(annotation, evidence_words=8)
    return truncate_text_by_tokens(best_text, 500)


def build_p4_json(annotation: dict[str, Any]) -> str:
    lookup = build_lookup(annotation)
    best_text = None
    best_tokens = 10**9

    for evidence_words in range(16, 3, -1):
        out = []
        fake_votes = 0
        for criterion in ALL_CRITERIA:
            item = lookup.get(criterion, {})
            evidence = item.get("evidence", "No decisive artifact is visible for this criterion.")
            score = normalize_score(item.get("aigc score", item.get("aigc_score", item.get("score", 0))))
            fake_votes += score
            out.append(
                {
                    "criterion": criterion,
                    "evidence": compress_sentence(evidence, evidence_words),
                    "aigc score": score,
                }
            )
        overall = normalize_overall(annotation.get("overall_likelihood"), fake_votes)
        text = json.dumps({"per_criterion": out, "overall_likelihood": overall}, ensure_ascii=False)
        token_count = approx_tokens(text)
        if token_count <= 500:
            best_text = text
            best_tokens = token_count
            if token_count >= 240:
                return text
        elif token_count < best_tokens:
            best_text = text
            best_tokens = token_count

    if best_text is None:
        raise RuntimeError("Failed to build P4 JSON")
    return best_text


def ensure_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; rerun with --force")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_meta(meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not meta_rows:
        return {"count": 0}
    token_counts = [row["assistant_tokens_approx"] for row in meta_rows]
    label_counts = Counter(row["label"] for row in meta_rows)
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"real": 0, "fake": 0})
    for row in meta_rows:
        family_counts[row["source_family"]][row["label"]] += 1
    return {
        "count": len(meta_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "assistant_tokens_approx": {
            "min": min(token_counts),
            "max": max(token_counts),
            "avg": round(sum(token_counts) / len(token_counts), 2),
        },
        "per_family": {key: family_counts[key] for key in sorted(family_counts)},
    }


def build_stage_entries(
    *,
    stage: str,
    rows: list[dict[str, Any]],
    p1_targets: dict[str, dict[str, str]],
    annotation_cache: dict[str, dict[str, Any]],
    a_step1_prompt: str,
    a_step2_prompt: str,
    p2_max_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = []
    meta_rows = []
    for row in rows:
        annotation = annotation_cache[row["annotation_path"]]
        image_path = row["image_path"]

        if stage == "p1":
            target = p1_targets[row["annotation_path"]]
            assistant = json.dumps(target, ensure_ascii=False)
            entry = {
                "conversations": [
                    {"from": "human", "value": f"<image>\n{P1_PROMPT}"},
                    {"from": "gpt", "value": assistant},
                ],
                "images": [image_path],
            }
        elif stage == "p2":
            assistant = build_full_analysis_text(annotation, max_tokens=p2_max_tokens)
            entry = {
                "conversations": [
                    {"from": "human", "value": f"<image>\n{a_step1_prompt}"},
                    {"from": "gpt", "value": assistant},
                ],
                "images": [image_path],
            }
        elif stage == "p3":
            assistant = build_under500_analysis(annotation)
            entry = {
                "conversations": [
                    {"from": "human", "value": f"<image>\n{a_step1_prompt}"},
                    {"from": "gpt", "value": assistant},
                ],
                "images": [image_path],
            }
        elif stage == "p4":
            analysis = build_under500_analysis(annotation)
            user_text = f"*** INSTRUCTIONS ***\n{a_step2_prompt}\n\n*** ANALYSIS DATA TO PROCESS ***\n{analysis}\n"
            assistant = build_p4_json(annotation)
            entry = {
                "conversations": [
                    {"from": "human", "value": user_text},
                    {"from": "gpt", "value": assistant},
                ]
            }
        else:
            raise ValueError(stage)

        entries.append(entry)
        meta_rows.append(
            {
                "annotation_path": row["annotation_path"],
                "image_path": row["image_path"],
                "label": row["label"],
                "source_family": row["source_family"],
                "assistant_tokens_approx": approx_tokens(assistant),
            }
        )
    return entries, meta_rows


def main() -> None:
    args = parse_args()
    annotations_root = Path(args.annotations_root).resolve()
    datasets_root = Path(args.datasets_root).resolve()

    a_step1_prompt = load_text_prompt(PROMPTS_DIR / "a_step1.txt")
    a_step2_prompt = (PROMPTS_DIR / "a_step2.txt").read_text(encoding="utf-8").strip()

    split_dirs = {
        "p1": annotations_root / "data_p1",
        "p2": annotations_root / "data_p2",
        "p3": annotations_root / "data_p3",
        "p4": annotations_root / "data_p4",
    }
    # Only P1 is generated here; P2/P3/P4 are produced by build_p234_jsonl.py.
    output_dirs = {
        "p1": datasets_root / "data_p1",
    }

    manifests = {stage: load_manifest_rows(path) for stage, path in split_dirs.items()}
    p1_targets = load_p1_targets(split_dirs["p1"])

    partitions: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    partitions["p1"] = partition_rows(manifests["p1"], args.val_ratio, args.seed + 1)
    partitions["p2"] = partition_rows(manifests["p2"], args.val_ratio, args.seed + 2)
    partitions["p3"] = partition_rows(manifests["p3"], args.val_ratio, args.seed + 3)
    partitions["p4"] = partition_p4_from_p3(manifests["p4"], *partitions["p3"])

    all_paths = {
        row["annotation_path"]
        for stage_rows in manifests.values()
        for row in stage_rows
    }
    annotation_cache = {path: load_annotation_by_path(Path(path)) for path in sorted(all_paths)}

    final_summary = {}
    for stage in ["p1"]:  # P1 only; P2/P3/P4 are handled by build_p234_jsonl.py
        out_dir = output_dirs[stage]
        ensure_output_dir(out_dir, args.force)

        train_rows, val_rows = partitions[stage]
        train_entries, train_meta = build_stage_entries(
            stage=stage,
            rows=train_rows,
            p1_targets=p1_targets,
            annotation_cache=annotation_cache,
            a_step1_prompt=a_step1_prompt,
            a_step2_prompt=a_step2_prompt,
            p2_max_tokens=args.p2_max_tokens,
        )
        val_entries, val_meta = build_stage_entries(
            stage=stage,
            rows=val_rows,
            p1_targets=p1_targets,
            annotation_cache=annotation_cache,
            a_step1_prompt=a_step1_prompt,
            a_step2_prompt=a_step2_prompt,
            p2_max_tokens=args.p2_max_tokens,
        )

        write_jsonl(out_dir / "train.jsonl", train_entries)
        write_jsonl(out_dir / "val.jsonl", val_entries)
        write_json(out_dir / "train.meta.json", train_meta)
        write_json(out_dir / "val.meta.json", val_meta)

        summary = {
            "name": out_dir.name,
            "stage": stage,
            "source_split_dir": str(split_dirs[stage]),
            "train": summarize_meta(train_meta),
            "val": summarize_meta(val_meta),
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "p2_max_tokens_approx": args.p2_max_tokens if stage == "p2" else None,
            "raw_annotation_json_modified": False,
        }
        write_json(out_dir / "summary.json", summary)
        final_summary[stage] = {
            "train": summary["train"]["count"],
            "val": summary["val"]["count"],
            "total": summary["train"]["count"] + summary["val"]["count"],
        }

    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

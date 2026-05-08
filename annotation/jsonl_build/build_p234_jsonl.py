#!/usr/bin/env python3
"""
Build exact-format datasets for the staged pipeline.

Mapping:
- P2 = multi-prompt SFT format (3 sub-prompts in one entry)
- P3 = single-prompt image+evidence format
- P4 = text-only synthesis (JSON output) format

Design:
- P2: regenerate exactly from clean P2 split using existing local functions
- P3/P4: reuse cached exact-format datasets when possible, generate only missing rows

Run with:
  python build_p234_jsonl.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lib_p3 import build_p3_full, build_p3_slice
from lib_p3_text import fit_p3_text, get_tokenizer, parse_a_step1_prompt
from lib_p4_text import fit_p3_text as fit_p3_text_for_p4
from lib_p4_text import fit_p4_json
from lib_p234_entry import build_p4_json as build_p4_json_step0
from lib_p234_entry import make_p3p4_entries, make_sft_entries


# Self-contained ROOT detection: this file lives at annotation/jsonl_build/
_THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_ROOT = _THIS_DIR.parent.parent  # jsonl_build/ -> annotation/ -> bundle root
PROMPTS_DIR = _THIS_DIR.parent / "prompts"  # annotation/prompts/
ROOT = BUNDLE_ROOT  # default; user can override with --annotations_root / --datasets_root
ANNOTATIONS_ROOT = ROOT / "annotations"
DATASETS_ROOT = ROOT / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P2/P3/P4 SFT jsonl from split manifests")
    parser.add_argument("--annotations_root", default=str(ANNOTATIONS_ROOT))
    parser.add_argument("--datasets_root", default=str(DATASETS_ROOT))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=0)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; rerun with --force")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def partition_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["source_family"], row["label"])].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
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


def partition_like_reference(rows: list[dict[str, Any]], reference_train: list[dict[str, Any]], reference_val: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_paths = {row["annotation_path"] for row in reference_train}
    val_paths = {row["annotation_path"] for row in reference_val}
    train_rows = [row for row in rows if row["annotation_path"] in train_paths]
    val_rows = [row for row in rows if row["annotation_path"] in val_paths]
    return train_rows, val_rows


def limit_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["source_family"], row["label"])].append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def summarize_token_meta(meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"count": len(meta_rows)}
    if not meta_rows:
        return summary
    label_counts = Counter(row.get("label") for row in meta_rows)
    summary["label_counts"] = dict(sorted(label_counts.items()))
    token_counts = [row["token_count"] for row in meta_rows if "token_count" in row]
    if token_counts:
        summary["max_token_count"] = max(token_counts)
        summary["min_token_count"] = min(token_counts)
        summary["avg_token_count"] = round(sum(token_counts) / len(token_counts), 2)
    return summary


def build_cache_from_meta_and_entries(meta_path: Path, entries_path: Path) -> dict[str, dict[str, Any]]:
    meta_rows = read_json(meta_path)
    entries = read_jsonl(entries_path)
    if len(meta_rows) != len(entries):
        raise RuntimeError(f"meta/entries length mismatch: {meta_path} vs {entries_path}")
    cache = {}
    for meta, entry in zip(meta_rows, entries):
        cache[meta["annotation_file"]] = {
            "entry": entry,
            "meta": meta,
        }
    return cache


def load_annotation(path: str, annotation_dir: Path | None = None) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.exists():
        return read_json(candidate)
    if annotation_dir is not None:
        fallback = annotation_dir / candidate.name
        if fallback.exists():
            return read_json(fallback)
    raise FileNotFoundError(path)


def build_p2_exact(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    annotation_dir: Path,
) -> dict[str, Any]:
    train_p2_path = out_dir / "train.jsonl"
    val_p2_path = out_dir / "val.jsonl"

    train_meta = []
    val_meta = []
    train_entries_total = 0
    val_entries_total = 0

    def process(rows: list[dict[str, Any]], p2_path: Path, meta_store: list[dict[str, Any]], split_name: str) -> int:
        total_entries = 0
        with p2_path.open("w", encoding="utf-8") as f_all:
            for idx, row in enumerate(rows, start=1):
                ann = load_annotation(row["annotation_path"], annotation_dir=annotation_dir)
                image_path = ann.get("_meta", {}).get("image_path")
                if not image_path:
                    continue
                entries = make_p3p4_entries(Path(image_path), ann)
                for entry in entries:
                    entry.setdefault("images", [])
                for entry in entries:
                    f_all.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_entries += len(entries)
                meta_store.append(
                    {
                        "annotation_file": row["annotation_path"],
                        "image_path": image_path,
                        "label": row["label"],
                        "source_family": row["source_family"],
                        "num_entries": len(entries),
                    }
                )
                if idx % 1000 == 0:
                    print(f"[P2 {split_name}] {idx}/{len(rows)} images processed")
        return total_entries

    train_entries_total = process(train_rows, train_p2_path, train_meta, "train")
    val_entries_total = process(val_rows, val_p2_path, val_meta, "val")

    write_json(out_dir / "train.meta.json", train_meta)
    write_json(out_dir / "val.meta.json", val_meta)
    summary = {
        "name": "data_p2",
        "images_total": len(train_rows) + len(val_rows),
        "images_train": len(train_rows),
        "images_val": len(val_rows),
        "train_total": train_entries_total,
        "val_total": val_entries_total,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def build_p3_or_p4_exact(
    *,
    stage: str,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    out_dir: Path,
    cache: dict[str, dict[str, Any]],
    tokenizer,
    a_step1_prompt: str,
    a_step2_prompt: str,
    annotation_dir: Path,
) -> dict[str, Any]:
    if stage == "p3":
        train_jsonl_name = "train.jsonl"
        val_jsonl_name = "val.jsonl"
        train_meta_name = "train.meta.json"
        val_meta_name = "val.meta.json"
        summary_name = "_summary.json"
    elif stage == "p4":
        train_jsonl_name = "train.jsonl"
        val_jsonl_name = "val.jsonl"
        train_meta_name = "train.meta.json"
        val_meta_name = "val.meta.json"
        summary_name = "_summary.json"
    else:
        raise ValueError(stage)

    def build_missing(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        ann = load_annotation(row["annotation_path"], annotation_dir=annotation_dir)
        meta = ann.get("_meta", {})
        if stage == "p3":
            assistant_text = build_p3_full(ann, evidence_words=16)
            token_count = len(tokenizer.encode(assistant_text))
            entry = {
                "conversations": [
                    {"from": "human", "value": "<image>\n" + a_step1_prompt},
                    {"from": "gpt", "value": assistant_text},
                ],
                "images": [meta.get("image_path")],
            }
        else:
            p3_text = build_p3_full(ann, evidence_words=16)
            assistant_text = build_p4_json_step0(ann)
            token_count = len(tokenizer.encode(assistant_text))
            entry = {
                "conversations": [
                    {"from": "human", "value": f"*** INSTRUCTIONS ***\n{a_step2_prompt}\n\n*** ANALYSIS DATA TO PROCESS ***\n{p3_text}\n"},
                    {"from": "gpt", "value": assistant_text},
                ]
            }
        meta_row = {
            "annotation_file": row["annotation_path"],
            "image_id": meta.get("image_id"),
            "image_path": meta.get("image_path"),
            "label": meta.get("label"),
            "source": meta.get("source"),
            "generator": meta.get("generator"),
            "token_count": token_count,
        }
        return entry, meta_row

    def process(rows: list[dict[str, Any]], jsonl_name: str, meta_name: str, split_name: str) -> list[dict[str, Any]]:
        meta_rows = []
        out_jsonl = out_dir / jsonl_name
        with out_jsonl.open("w", encoding="utf-8") as handle:
            reused = 0
            generated = 0
            for idx, row in enumerate(rows, start=1):
                cached = cache.get(row["annotation_path"])
                if cached is not None:
                    entry = cached["entry"]
                    meta_row = cached["meta"]
                    reused += 1
                else:
                    entry, meta_row = build_missing(row)
                    generated += 1
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                meta_rows.append(meta_row)
                if idx % 1000 == 0:
                    print(f"[{stage.upper()} {split_name}] {idx}/{len(rows)} rows processed (reused={reused}, generated={generated})")
        write_json(out_dir / meta_name, meta_rows)
        print(f"[{stage.upper()} {split_name}] reused={reused}, generated={generated}")
        return meta_rows

    train_meta = process(train_rows, train_jsonl_name, train_meta_name, "train")
    val_meta = process(val_rows, val_jsonl_name, val_meta_name, "val")
    summary = {
        "name": out_dir.name,
        "total_selected": len(train_rows) + len(val_rows),
        "train": len(train_rows),
        "val": len(val_rows),
        **{k: v for k, v in summarize_token_meta(train_meta).items() if k in {"max_token_count", "min_token_count", "avg_token_count"}},
    }
    write_json(out_dir / summary_name, summary)
    return summary


def write_preview(
    rows: list[dict[str, Any]],
    out_path: Path,
    sample_count: int,
    seed: int,
    annotation_dir: Path,
) -> None:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    selected: list[dict[str, Any]] = []
    labels = sorted(by_label)
    if len(labels) >= 2:
        per_label = max(1, sample_count // len(labels))
        for label in labels:
            bucket = by_label[label][:]
            rng.shuffle(bucket)
            selected.extend(bucket[:per_label])
        if len(selected) < sample_count:
            remaining = [row for row in rows if row not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: sample_count - len(selected)])
    else:
        selected = rows[:]
        rng.shuffle(selected)
        selected = selected[:sample_count]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected[:sample_count]:
            ann = load_annotation(row["annotation_path"], annotation_dir=annotation_dir)
            payload = {
                "annotation_path": row["annotation_path"],
                "image_path": row["image_path"],
                "label": row["label"],
                "source_family": row["source_family"],
                "overall_likelihood": ann.get("overall_likelihood"),
                "p3_slices": {
                    "stage1_1": build_p3_slice(ann, 0, evidence_words=16),
                    "stage1_2": build_p3_slice(ann, 1, evidence_words=16),
                    "stage1_3": build_p3_slice(ann, 2, evidence_words=16),
                },
                "p3_full": build_p3_full(ann, evidence_words=16),
                "p4_target": json.loads(build_p4_json_step0(ann)),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    annotations_root = Path(args.annotations_root).resolve()
    datasets_root = Path(args.datasets_root).resolve()

    p2_rows = read_jsonl(annotations_root / "data_p2" / "_split_manifest.jsonl")
    p3_rows = read_jsonl(annotations_root / "data_p3" / "_split_manifest.jsonl")
    p4_rows = read_jsonl(annotations_root / "data_p4" / "_split_manifest.jsonl")

    if args.preview_count > 0:
        preview_path = datasets_root / "preview" / f"data_p2_preview_{args.preview_count}.jsonl"
        write_preview(
            p2_rows,
            preview_path,
            args.preview_count,
            args.seed + 500,
            annotation_dir=annotations_root / "data_p2",
        )
        print(json.dumps({"preview": str(preview_path)}, ensure_ascii=False, indent=2))
        if args.preview_only:
            return

    p2_train_rows, p2_val_rows = partition_rows(p2_rows, args.val_ratio, args.seed + 20)
    p3_train_rows, p3_val_rows = partition_rows(p3_rows, args.val_ratio, args.seed + 30)
    p2_train_rows = limit_rows(p2_train_rows, args.max_train_images, args.seed + 120)
    p2_val_rows = limit_rows(p2_val_rows, args.max_val_images, args.seed + 121)
    p3_train_rows = limit_rows(p3_train_rows, args.max_train_images, args.seed + 130)
    p3_val_rows = limit_rows(p3_val_rows, args.max_val_images, args.seed + 131)
    p4_train_rows, p4_val_rows = partition_like_reference(p4_rows, p3_train_rows, p3_val_rows)

    tokenizer = get_tokenizer()
    a_step1_prompt = parse_a_step1_prompt(PROMPTS_DIR / "a_step1.txt")
    a_step2_prompt = (PROMPTS_DIR / "a_step2.txt").read_text(encoding="utf-8").strip()

    suffix = f"_{args.output_tag.strip()}" if args.output_tag.strip() else ""
    p2_out = datasets_root / f"data_p2{suffix}"
    p3_out = datasets_root / f"data_p3{suffix}"
    p4_out = datasets_root / f"data_p4{suffix}"
    for out_dir in [p2_out, p3_out, p4_out]:
        ensure_clean_dir(out_dir, args.force)

    p2_summary = build_p2_exact(
        p2_train_rows,
        p2_val_rows,
        p2_out,
        annotation_dir=annotations_root / "data_p2",
    )
    p3_summary = build_p3_or_p4_exact(
        stage="p3",
        train_rows=p3_train_rows,
        val_rows=p3_val_rows,
        out_dir=p3_out,
        cache={},
        tokenizer=tokenizer,
        a_step1_prompt=a_step1_prompt,
        a_step2_prompt=a_step2_prompt,
        annotation_dir=annotations_root / "data_p3",
    )
    p4_summary = build_p3_or_p4_exact(
        stage="p4",
        train_rows=p4_train_rows,
        val_rows=p4_val_rows,
        out_dir=p4_out,
        cache={},
        tokenizer=tokenizer,
        a_step1_prompt=a_step1_prompt,
        a_step2_prompt=a_step2_prompt,
        annotation_dir=annotations_root / "data_p4",
    )

    print(json.dumps({"p2": p2_summary, "p3": p3_summary, "p4": p4_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

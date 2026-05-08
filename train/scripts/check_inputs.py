#!/usr/bin/env python3
"""Check whether the datasets needed for G2 P1-P4 reproduction exist."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROCESSED_JSONL = [
    "datasets/data_p1/train.jsonl",
    "datasets/data_p1/val.jsonl",
    "datasets/data_p2/train.jsonl",
    "datasets/data_p2/val.jsonl",
    "datasets/data_p3/train.jsonl",
    "datasets/data_p3/val.jsonl",
    "datasets/data_p4/train.jsonl",
    "datasets/data_p4/val.jsonl",
]

RAW_ROOTS = [
    "data/images/adm/fake",
    "data/images/biggan/fake",
    "data/images/sid_set/fake",
    "data/images/sid_set/real",
    "datasets/raw/ARForensics/ARForensics",
    "datasets/raw/SynthScars/SynthScars",
]

SHARED_ROOTS = [
    "coco/train2017",
    "ImageNet/train",
]


def resolve_under_root(root: Path, rel: str) -> Path:
    direct = root / rel
    if direct.exists():
        return direct
    if rel.startswith("datasets/"):
        stripped = root / rel[len("datasets/") :]
        if stripped.exists():
            return stripped
    return direct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None, help="Backward-compatible root for both processed JSONL and project images.")
    parser.add_argument("--processed-data-root", default=None, help="Root containing processed JSONL under datasets/.")
    parser.add_argument("--project-image-root", default=None, help="Root containing data/images/ and datasets/raw/.")
    parser.add_argument("--shared-datasets-root", required=True)
    parser.add_argument("--sample-images", type=int, default=20)
    return parser.parse_args()


def count_jsonl(path: Path) -> tuple[int, Counter[str]]:
    counter: Counter[str] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counter["json_decode_error"] += 1
                continue
            images = row.get("images") or []
            if images:
                counter["with_images"] += 1
            else:
                counter["text_only"] += 1
    return rows, counter


def main() -> int:
    args = parse_args()
    default_root = None  # caller must pass --processed-data-root + --project-image-root
    processed_root = Path(args.processed_data_root or args.data_root or default_root)
    project_image_root = Path(args.project_image_root or args.data_root or default_root)
    shared_root = Path(args.shared_datasets_root)

    ok = True
    print("[processed jsonl]")
    for rel in PROCESSED_JSONL:
        path = resolve_under_root(processed_root, rel)
        if not path.exists():
            print(f"MISSING {path}")
            ok = False
            continue
        rows, counter = count_jsonl(path)
        print(f"OK {rows:6d} rows {dict(counter)} {path}")

    print("\n[raw image roots]")
    for rel in RAW_ROOTS:
        path = resolve_under_root(project_image_root, rel)
        exists = path.exists()
        print(("OK " if exists else "MISSING ") + str(path))
        ok = ok and exists

    print("\n[shared dataset roots]")
    for rel in SHARED_ROOTS:
        path = shared_root / rel
        exists = path.exists()
        print(("OK " if exists else "MISSING ") + str(path))
        ok = ok and exists

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Build SFT data from sample-50 + VLM-1K + template-5K annotations.

Rules:
- Sample 50 annotations are always included.
- If the same image appears in multiple annotation directories, priority order wins.
- Split is done by image, stratified on source x label.
- Writes combined train/val JSONL plus P3/P4 splits.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from lib_p234_entry import make_sft_entries


# NOTE: SAMPLE_DIR is used only by the standalone main() for sample image lookup; not used on import.
# Override via DATA_ROOT env var when running standalone.
import os as _os
ROOT = Path(_os.environ.get("DATA_ROOT", _os.path.dirname(_os.path.abspath(__file__))))
SAMPLE_DIR = ROOT / "datasets/raw/sample dataset"


def find_sample_image(stem):
    for subfolder in ["Real", "Fake"]:
        candidate = SAMPLE_DIR / subfolder / f"{stem}.png"
        if candidate.exists():
            return candidate
    return None


def normalize_meta(data, fallback_path):
    meta = data.setdefault("_meta", {})
    image_path = meta.get("image_path")

    if not image_path and "poc/annotations" in str(fallback_path):
        sample_image = find_sample_image(fallback_path.stem)
        if sample_image:
            image_path = str(sample_image)

    if not image_path:
        image_path = str(fallback_path)

    meta["image_path"] = image_path
    lower_path = image_path.lower()

    if not meta.get("source"):
        if "sample dataset" in image_path:
            meta["source"] = "sample"
        else:
            meta["source"] = "unknown"

    if not meta.get("label"):
        if "/fake/" in lower_path:
            meta["label"] = "ai-generated"
        elif "/real/" in lower_path:
            meta["label"] = "real"
        else:
            meta["label"] = "unknown"

    meta.setdefault("image_id", fallback_path.stem)
    return data


def dedup_key(data, fallback_path):
    meta = data.get("_meta", {})
    image_path = meta.get("image_path") or str(fallback_path)
    return str(Path(image_path))


def load_annotations(annotation_dirs):
    annotations = {}
    duplicates = []
    for annotation_dir in annotation_dirs:
        for path in sorted(Path(annotation_dir).glob("*.json")):
            if path.name.startswith("_"):
                continue
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("parse_error"):
                continue
            data = normalize_meta(data, path)
            key = dedup_key(data, path)
            if key in annotations:
                duplicates.append({
                    "kept_from": annotations[key]["_source_file"],
                    "skipped_from": str(path),
                    "image_path": key,
                })
                continue
            data["_source_file"] = str(path)
            annotations[key] = data
    return annotations, duplicates


def stratified_split(items, train_ratio, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for item in items:
        meta = item["annotation"].get("_meta", {})
        key = (meta.get("source", "unknown"), meta.get("label", "unknown"))
        buckets[key].append(item)

    train_items = []
    val_items = []
    split_counts = {}
    for key, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        n_train = int(len(bucket) * train_ratio)
        if len(bucket) == 1:
            n_train = 1
        elif len(bucket) >= 2:
            n_train = max(1, min(len(bucket) - 1, n_train))
        train_bucket = bucket[:n_train]
        val_bucket = bucket[n_train:]
        train_items.extend(train_bucket)
        val_items.extend(val_bucket)
        split_counts[f"{key[0]}|{key[1]}"] = {
            "total": len(bucket),
            "train": len(train_bucket),
            "val": len(val_bucket),
        }

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items, split_counts


def split_p3p4_entries(entries):
    p3 = [e for e in entries if "images" in e]
    p4 = [e for e in entries if "images" not in e]
    return p3, p4


def write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    # NOTE: This module is imported by build_p234_jsonl.py for its helper functions.
    # The standalone main() args below only apply when running this module directly.
    parser = argparse.ArgumentParser(description="Build SFT data from sample-50 + 1K VLM + 5K template annotations")
    parser.add_argument("--annotation_dir", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations, duplicates = load_annotations(args.annotation_dir)
    print(f"Loaded {len(annotations)} unique annotations")
    if duplicates:
        print(f"Skipped {len(duplicates)} duplicate entries by image path")

    items = []
    source_counter = Counter()
    label_counter = Counter()
    sample_counter = 0
    for _, annotation in annotations.items():
        meta = annotation.get("_meta", {})
        image_path = meta.get("image_path")
        if not image_path:
            continue
        entries = make_sft_entries(Path(image_path), annotation)
        item = {
            "image_id": meta.get("image_id", Path(image_path).stem),
            "annotation": annotation,
            "entries": entries,
        }
        items.append(item)
        source_counter[meta.get("source", "unknown")] += 1
        label_counter[meta.get("label", "unknown")] += 1
        if meta.get("source") == "sample":
            sample_counter += 1

    train_items, val_items, split_counts = stratified_split(items, args.train_ratio, args.seed)

    train_entries = []
    val_entries = []
    for item in train_items:
        train_entries.extend(item["entries"])
    for item in val_items:
        val_entries.extend(item["entries"])

    random.Random(args.seed).shuffle(train_entries)
    random.Random(args.seed + 1).shuffle(val_entries)

    train_p3, train_p4 = split_p3p4_entries(train_entries)
    val_p3, val_p4 = split_p3p4_entries(val_entries)

    write_jsonl(output_dir / "train.jsonl", train_entries)
    write_jsonl(output_dir / "val.jsonl", val_entries)

    summary = {
        "num_images_total": len(items),
        "num_images_train": len(train_items),
        "num_images_val": len(val_items),
        "num_sample_images_total": sample_counter,
        "train_ratio": args.train_ratio,
        "sources": dict(source_counter),
        "labels": dict(label_counter),
        "split_by_source_label": split_counts,
        "entries": {
            "train_total": len(train_entries),
            "val_total": len(val_entries),
            "train_p3": len(train_p3),
            "train_p4": len(train_p4),
            "val_p3": len(val_p3),
            "val_p4": len(val_p4),
        },
        "annotation_dirs": args.annotation_dir,
        "num_duplicate_entries_skipped": len(duplicates),
    }
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    with open(output_dir / "duplicates_skipped.json", "w", encoding="utf-8") as fh:
        json.dump(duplicates, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

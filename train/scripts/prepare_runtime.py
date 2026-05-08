#!/usr/bin/env python3
"""Prepare runtime files for reproducing merged_p4 from P1 to P4.

The script copies only the processed JSONL files required by the G2 lineage,
rewrites image paths for the target machine, registers LLaMA-Factory dataset
aliases, and renders YAML configs from placeholder templates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OLD_PROJECT_ROOT = "<TRACK3_ROOT>"
OLD_SHARED_DATASETS_ROOT = "<SHARED_DATASETS_ROOT>"
RUN_NAME = "train"
MODEL_NAME = "qwen2_train"

TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
}

DATASET_SPECS = {
    "p1_train": {
        "source_rel": "datasets/data_p1/train.jsonl",
        "target_rel": "runtime_data/datasets/data_p1/train.jsonl",
        "has_images": True,
    },
    "p1_val": {
        "source_rel": "datasets/data_p1/val.jsonl",
        "target_rel": "runtime_data/datasets/data_p1/val.jsonl",
        "has_images": True,
    },
    "p2_train": {
        "source_rel": "datasets/data_p2/train.jsonl",
        "target_rel": "runtime_data/datasets/data_p2/train.jsonl",
        "has_images": True,
    },
    "p2_val": {
        "source_rel": "datasets/data_p2/val.jsonl",
        "target_rel": "runtime_data/datasets/data_p2/val.jsonl",
        "has_images": True,
    },
    "p3_train": {
        "source_rel": "datasets/data_p3/train.jsonl",
        "target_rel": "runtime_data/datasets/data_p3/train.jsonl",
        "has_images": True,
    },
    "p3_val": {
        "source_rel": "datasets/data_p3/val.jsonl",
        "target_rel": "runtime_data/datasets/data_p3/val.jsonl",
        "has_images": True,
    },
    "p4_train": {
        "source_rel": "datasets/data_p4/train.jsonl",
        "target_rel": "runtime_data/datasets/data_p4/train.jsonl",
        "has_images": False,
    },
    "p4_val": {
        "source_rel": "datasets/data_p4/val.jsonl",
        "target_rel": "runtime_data/datasets/data_p4/val.jsonl",
        "has_images": False,
    },
}

REQUIRED_DATA_ROOT_PATHS = [
    "data/images/adm/fake",
    "data/images/biggan/fake",
    "data/images/sid_set/fake",
    "data/images/sid_set/real",
    "datasets/raw/ARForensics/ARForensics",
    "datasets/raw/SynthScars/SynthScars",
    "datasets/data_p1",
    "datasets/data_p2",
    "datasets/data_p3",
    "datasets/data_p4",
]

REQUIRED_SHARED_DATASET_PATHS = [
    "coco/train2017",
    "ImageNet/train",
]


def resolve_under_root(root: Path, rel: str) -> Path:
    """Resolve both project-root and datasets-root style inputs.

    If rel is `datasets/foo` and root already points at the datasets directory,
    use `root/foo`. Otherwise use `root/datasets/foo`.
    """
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
    parser.add_argument("--project-root", required=True, help="Root containing LLaMA-Factory-qwen25vl.")
    parser.add_argument("--data-root", help="Backward-compatible alias for --processed-data-root and --project-image-root.")
    parser.add_argument("--processed-data-root", help="Root containing processed training JSONL under datasets/.")
    parser.add_argument("--project-image-root", help="Root containing data/images/ and datasets/raw/.")
    parser.add_argument("--shared-datasets-root", required=True, help="Root containing coco/ and ImageNet/.")
    parser.add_argument("--run-root", required=True, help="Output runtime/checkpoint root.")
    parser.add_argument("--config-template-root", required=True)
    parser.add_argument("--config-output-root", required=True)
    return parser.parse_args()


def ensure_required_paths(processed_root: Path, project_image_root: Path, shared_root: Path) -> None:
    processed_required = [
        "datasets/data_p1",
        "datasets/data_p2",
        "datasets/data_p3",
        "datasets/data_p4",
    ]
    raw_required = [
        rel for rel in REQUIRED_DATA_ROOT_PATHS if rel not in processed_required
    ]

    missing = [str(resolve_under_root(processed_root, rel)) for rel in processed_required if not resolve_under_root(processed_root, rel).exists()]
    missing.extend(str(resolve_under_root(project_image_root, rel)) for rel in raw_required if not resolve_under_root(project_image_root, rel).exists())
    missing.extend(str(shared_root / rel) for rel in REQUIRED_SHARED_DATASET_PATHS if not (shared_root / rel).exists())
    missing.extend(
        str(resolve_under_root(processed_root, spec["source_rel"]))
        for spec in DATASET_SPECS.values()
        if not resolve_under_root(processed_root, spec["source_rel"]).exists()
    )
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing G2 reproduction inputs:\n{joined}")


def rewrite_image_path(path: str, project_image_root: Path, shared_root: Path) -> str:
    if path.startswith(OLD_PROJECT_ROOT):
        return path.replace(OLD_PROJECT_ROOT, str(project_image_root), 1)
    if path.startswith(OLD_SHARED_DATASETS_ROOT):
        return path.replace(OLD_SHARED_DATASETS_ROOT, str(shared_root), 1)
    return path


def rewrite_jsonl(src: Path, dst: Path, project_image_root: Path, shared_root: Path, has_images: bool) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with src.open("r", encoding="utf-8") as in_handle, dst.open("w", encoding="utf-8") as out_handle:
        for line in in_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if has_images and "images" in row:
                row["images"] = [rewrite_image_path(path, project_image_root, shared_root) for path in row["images"]]
            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    return rows


def upsert_dataset_info(dataset_info_path: Path, prepared_root: Path) -> None:
    if dataset_info_path.exists():
        payload = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    else:
        payload = {}

    payload.update(
        {
            "data_p2_train": {
                "file_name": str(prepared_root / "datasets/data_p2/train.jsonl"),
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "images": "images"},
                "tags": TAGS,
            },
            "data_p2_val": {
                "file_name": str(prepared_root / "datasets/data_p2/val.jsonl"),
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "images": "images"},
                "tags": TAGS,
            },
            "data_p3_train": {
                "file_name": str(prepared_root / "datasets/data_p3/train.jsonl"),
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "images": "images"},
                "tags": TAGS,
            },
            "data_p3_val": {
                "file_name": str(prepared_root / "datasets/data_p3/val.jsonl"),
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "images": "images"},
                "tags": TAGS,
            },
            "data_p4_train": {
                "file_name": str(prepared_root / "datasets/data_p4/train.jsonl"),
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
                "tags": TAGS,
            },
            "data_p4_val": {
                "file_name": str(prepared_root / "datasets/data_p4/val.jsonl"),
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
                "tags": TAGS,
            },
        }
    )

    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_configs(template_root: Path, output_root: Path, project_root: Path, run_root: Path) -> None:
    replacements = {
        "__PROJECT_ROOT__": str(project_root),
        "__RUN_ROOT__": str(run_root),
        "__MODEL_ROOT__": str(run_root / f"models/{MODEL_NAME}"),
        "__LLAMA_DATA_DIR__": str(project_root / "LLaMA-Factory-qwen25vl/data"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for path in sorted(template_root.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        (output_root / path.name).write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    processed_arg = args.processed_data_root or args.data_root
    project_image_arg = args.project_image_root or args.data_root
    if not processed_arg or not project_image_arg:
        raise SystemExit("provide --data-root or both --processed-data-root and --project-image-root")

    processed_root = Path(processed_arg).resolve()
    project_image_root = Path(project_image_arg).resolve()
    shared_root = Path(args.shared_datasets_root).resolve()
    run_root = Path(args.run_root).resolve()
    template_root = Path(args.config_template_root).resolve()
    config_output_root = Path(args.config_output_root).resolve()
    prepared_root = run_root / "runtime_data"
    dataset_info_path = project_root / "LLaMA-Factory-qwen25vl/data/dataset_info.json"

    ensure_required_paths(processed_root, project_image_root, shared_root)
    (run_root / "models").mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)

    rewritten: dict[str, dict[str, str | int]] = {}
    for name, spec in DATASET_SPECS.items():
        src = resolve_under_root(processed_root, spec["source_rel"])
        dst = run_root / spec["target_rel"]
        rewritten[name] = {
            "source": str(src),
            "target": str(dst),
            "rows": rewrite_jsonl(src, dst, project_image_root, shared_root, bool(spec["has_images"])),
        }

    upsert_dataset_info(dataset_info_path, prepared_root)
    render_configs(template_root, config_output_root, project_root, run_root)

    summary = {
        "run_name": RUN_NAME,
        "model_name": MODEL_NAME,
        "project_root": str(project_root),
        "processed_data_root": str(processed_root),
        "project_image_root": str(project_image_root),
        "shared_datasets_root": str(shared_root),
        "run_root": str(run_root),
        "prepared_root": str(prepared_root),
        "config_output_root": str(config_output_root),
        "dataset_info_path": str(dataset_info_path),
        "rewritten": rewritten,
    }
    (run_root / "runtime_prepare_summary_train.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build a normalized master manifest for the data annotation pool.

The source directory is expected to contain flat JSON annotation files such as:
  imagenet__real__imagenet_xxx.json
  adm__fake__adm_xxx.json

Usage:
  python build_manifest.py \
    --input_dir <path to raw annotation pool, e.g. annotations/data>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANONICAL_CRITERIA = [
    "lighting",
    "edge",
    "texture",
    "perspective",
    "commonsense",
    "text_symbol",
    "human",
    "material",
]

CRITERION_NAME_TO_KEY = {
    "Lighting & Shadows Consistency": "lighting",
    "Lighting & Shadow Consistency": "lighting",
    "Edges & Boundaries": "edge",
    "Texture & Resolution": "texture",
    "Texture & Resolution Coherence": "texture",
    "Perspective & Spatial Relationships": "perspective",
    "Perspective & Spatial Accuracy": "perspective",
    "Physical & Common Sense Logic": "commonsense",
    "Physical & Common-Sense Logic": "commonsense",
    "Text & Symbols": "text_symbol",
    "Text & Symbol Authenticity": "text_symbol",
    "Human & Biological Structure Integrity": "human",
    "Material & Object Details": "material",
    "Material & Object Detail Fidelity": "material",
}

REAL_FAMILY_MAP = {
    "imagenet": "imagenet",
    "coco": "coco",
    "sid_set": "sid_real",
    "sid-real": "sid_real",
    "sid_real": "sid_real",
    "arforensics_infinity": "arforensics_infinity",
    "arforensics_janus_pro": "arforensics_janus_pro",
    "arforensics_llamagen": "arforensics_llamagen",
    "arforensics_open_magvit2": "arforensics_open_magvit2",
    "arforensics_rar": "arforensics_rar",
}

FAKE_FAMILY_PATTERNS = [
    ("adm", "adm"),
    ("biggan", "biggan"),
    ("sid_set", "sid_fake"),
    ("sid_fake", "sid_fake"),
    ("infinity", "infinity"),
    ("janus_pro", "janus_pro"),
    ("llamagen", "llamagen"),
    ("open_magvit2", "open_magvit2"),
    ("rar", "rar"),
    ("synthscars", "synthscars"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized manifest for data")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Flat directory containing annotation JSON files (raw annotation pool).",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory to write manifest outputs. Defaults to <input_dir>/manifests.",
    )
    parser.add_argument(
        "--manifest_stem",
        default="master_manifest",
        help="Base filename stem for outputs.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_task_label(value: Any) -> str | None:
    text = normalize_text(value).lower().replace("_", "-")
    mapping = {
        "real": "real",
        "fake": "fake",
        "ai-generated": "fake",
        "ai generated": "fake",
        "aigenerated": "fake",
        "generated": "fake",
    }
    return mapping.get(text)


def normalize_overall_label(value: Any) -> str | None:
    text = normalize_text(value).lower().replace("_", "-")
    mapping = {
        "real": "Real",
        "fake": "AI-Generated",
        "ai-generated": "AI-Generated",
        "ai generated": "AI-Generated",
        "aigenerated": "AI-Generated",
    }
    return mapping.get(text)


def parse_score(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        parsed = int(float(value))
    except Exception:
        text = normalize_text(value).lower().replace("_", "-")
        if text in {"real"}:
            parsed = 0
        elif text in {"fake", "ai-generated", "ai generated", "aigenerated"}:
            parsed = 1
        else:
            return None
    return 0 if parsed <= 0 else 1


def score_to_label(score: int | None) -> str | None:
    if score is None:
        return None
    return "AI-Generated" if score == 1 else "Real"


def infer_filename_label(path: Path) -> str | None:
    stem = path.stem.lower()
    if "__real__" in stem:
        return "real"
    if "__fake__" in stem:
        return "fake"
    return None


def is_annotation_payload(payload: dict[str, Any]) -> bool:
    return bool(payload.get("per_criterion")) or bool(payload.get("overall_likelihood")) or bool(payload.get("_meta"))


def normalize_source_family(meta: dict[str, Any], path: Path, label: str | None) -> str:
    source = normalize_text(meta.get("source")).lower()
    generator = normalize_text(meta.get("generator")).lower()
    prefix = path.stem.split("__", 1)[0].lower()

    if label == "real":
        candidates = [source, prefix]
        for candidate in candidates:
            family = REAL_FAMILY_MAP.get(candidate)
            if family:
                return family
        return source or prefix or "unknown_real"

    if label == "fake":
        haystacks = [generator, source, prefix, path.stem.lower()]
        for pattern, family in FAKE_FAMILY_PATTERNS:
            if any(pattern in hay for hay in haystacks if hay):
                return family
        return generator or source or prefix or "unknown_fake"

    return source or generator or prefix or "unknown"


def build_criterion_labels(per_criterion: list[Any]) -> tuple[dict[str, str | None], int, int, int]:
    labels: dict[str, str | None] = {key: None for key in CANONICAL_CRITERIA}
    present = 0
    invalid_scores = 0

    for item in per_criterion:
        if not isinstance(item, dict):
            continue
        key = CRITERION_NAME_TO_KEY.get(normalize_text(item.get("criterion")))
        if not key:
            continue
        score = parse_score(item.get("aigc score", item.get("aigc_score", item.get("score"))))
        if score is None:
            invalid_scores += 1
            continue
        if labels[key] is None:
            present += 1
        labels[key] = score_to_label(score)

    fake_count = sum(1 for value in labels.values() if value == "AI-Generated")
    return labels, present, fake_count, invalid_scores


def derive_reason_excluded(
    *,
    label: str | None,
    image_path: str,
    criteria_present: int,
    overall_label: str | None,
    has_nonempty_a_step1: bool,
) -> str:
    reasons = []
    if label is None:
        reasons.append("missing_label")
    if not image_path:
        reasons.append("missing_image_path")
    if criteria_present != len(CANONICAL_CRITERIA):
        reasons.append("incomplete_criteria")
    if overall_label is None:
        reasons.append("missing_overall")
    if not has_nonempty_a_step1:
        reasons.append("empty_a_step1_responses")
    return "|".join(reasons)


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"{args.manifest_stem}.jsonl"
    summary_path = output_dir / f"{args.manifest_stem}_summary.json"
    family_csv_path = output_dir / "family_counts.csv"

    files = sorted(input_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    parse_error_files: list[str] = []
    skipped_non_annotation_files: list[str] = []

    raw_label_counts = Counter()
    label_counts = Counter()
    overall_counts = Counter()
    family_counts = Counter()
    criterion_coverage = Counter()
    source_counts = Counter()
    generator_counts = Counter()
    exclusion_reasons = Counter()
    family_stage_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    image_paths = Counter()
    overall_criterion_conflicts = 0
    label_overall_conflicts = 0
    label_filename_conflicts = 0
    rows_with_nonempty_a_step1 = 0

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            parse_error_files.append(str(path))
            continue

        if not isinstance(payload, dict) or not is_annotation_payload(payload):
            skipped_non_annotation_files.append(str(path))
            continue

        meta = payload.get("_meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        raw_label = normalize_text(meta.get("label"))
        label = normalize_task_label(raw_label)
        filename_label = infer_filename_label(path)
        if label is None:
            label = filename_label
            label_source = "filename"
        else:
            label_source = "meta"

        if label and filename_label and label != filename_label:
            label_filename_conflicts += 1

        per_criterion = payload.get("per_criterion") or []
        if not isinstance(per_criterion, list):
            per_criterion = []

        criterion_labels, criteria_present, criterion_fake_count, invalid_scores = build_criterion_labels(per_criterion)
        for key, value in criterion_labels.items():
            if value is not None:
                criterion_coverage[key] += 1

        explicit_overall = normalize_overall_label(payload.get("overall_likelihood"))
        fallback_overall = "AI-Generated" if criteria_present == len(CANONICAL_CRITERIA) and criterion_fake_count > 0 else None
        if fallback_overall is None and criteria_present == len(CANONICAL_CRITERIA):
            fallback_overall = "Real"
        overall_label = explicit_overall or fallback_overall
        overall_label_source = "annotation" if explicit_overall else ("criteria_fallback" if fallback_overall else "")

        if explicit_overall and fallback_overall and explicit_overall != fallback_overall:
            overall_criterion_conflicts += 1

        if label == "real" and overall_label == "AI-Generated":
            label_overall_conflicts += 1
        if label == "fake" and overall_label == "Real":
            label_overall_conflicts += 1

        a_step1_responses = meta.get("a_step1_responses") or []
        if not isinstance(a_step1_responses, list):
            a_step1_responses = []
        nonempty_a_step1 = [normalize_text(item) for item in a_step1_responses if normalize_text(item)]
        has_nonempty_a_step1 = bool(nonempty_a_step1)
        if has_nonempty_a_step1:
            rows_with_nonempty_a_step1 += 1

        image_path = normalize_text(meta.get("image_path") or payload.get("image_path"))
        family = normalize_source_family(meta, path, label)
        reason_excluded = derive_reason_excluded(
            label=label,
            image_path=image_path,
            criteria_present=criteria_present,
            overall_label=overall_label,
            has_nonempty_a_step1=has_nonempty_a_step1,
        )

        p1_eligible = bool(label and image_path and criteria_present == len(CANONICAL_CRITERIA) and overall_label)
        p2_eligible = bool(p1_eligible and has_nonempty_a_step1)
        p3_eligible = p2_eligible
        p4_eligible = p3_eligible

        row = {
            "annotation_path": str(path),
            "image_path": image_path,
            "label": label,
            "raw_label": raw_label,
            "label_source": label_source,
            "filename_label": filename_label,
            "overall_label": overall_label,
            "overall_label_source": overall_label_source,
            "source_family": family,
            "source_raw": normalize_text(meta.get("source")),
            "generator_raw": normalize_text(meta.get("generator")),
            "annotation_model": normalize_text(meta.get("annotation_model")),
            "criterion_labels": criterion_labels,
            "criterion_fake_count": criterion_fake_count,
            "raw_criterion_count": len(per_criterion),
            "criteria_present_count": criteria_present,
            "criterion_invalid_score_count": invalid_scores,
            "has_per_criterion8": criteria_present == len(CANONICAL_CRITERIA),
            "has_overall": explicit_overall is not None,
            "has_nonempty_a_step1_responses": has_nonempty_a_step1,
            "nonempty_a_step1_response_count": len(nonempty_a_step1),
            "parse_error": False,
            "overall_criterion_conflict": bool(explicit_overall and fallback_overall and explicit_overall != fallback_overall),
            "label_overall_conflict": (
                (label == "real" and overall_label == "AI-Generated")
                or (label == "fake" and overall_label == "Real")
            ),
            "p1_eligible": p1_eligible,
            "p2_eligible": p2_eligible,
            "p3_eligible": p3_eligible,
            "p4_eligible": p4_eligible,
            "reason_excluded": reason_excluded,
        }
        rows.append(row)

        raw_label_counts[raw_label or "<empty>"] += 1
        label_counts[label or "<missing>"] += 1
        overall_counts[overall_label or "<missing>"] += 1
        family_counts[(family, label or "<missing>")] += 1
        source_counts[normalize_text(meta.get("source")) or "<empty>"] += 1
        generator_counts[normalize_text(meta.get("generator")) or "<empty>"] += 1
        image_paths[image_path or "<empty>"] += 1
        if reason_excluded:
            for reason in reason_excluded.split("|"):
                exclusion_reasons[reason] += 1

        stage_key = (family, label or "<missing>")
        family_stage_counts[stage_key]["total"] += 1
        family_stage_counts[stage_key]["p1_eligible"] += int(p1_eligible)
        family_stage_counts[stage_key]["p2_eligible"] += int(p2_eligible)
        family_stage_counts[stage_key]["p3_eligible"] += int(p3_eligible)
        family_stage_counts[stage_key]["p4_eligible"] += int(p4_eligible)
        family_stage_counts[stage_key]["with_nonempty_a_step1"] += int(has_nonempty_a_step1)
        family_stage_counts[stage_key]["overall_criterion_conflicts"] += int(
            bool(explicit_overall and fallback_overall and explicit_overall != fallback_overall)
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps_json(row) + "\n")

    with family_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "label",
                "total",
                "p1_eligible",
                "p2_eligible",
                "p3_eligible",
                "p4_eligible",
                "with_nonempty_a_step1",
                "overall_criterion_conflicts",
            ],
        )
        writer.writeheader()
        for (family, label), counts in sorted(family_stage_counts.items()):
            writer.writerow(
                {
                    "family": family,
                    "label": label,
                    "total": counts["total"],
                    "p1_eligible": counts["p1_eligible"],
                    "p2_eligible": counts["p2_eligible"],
                    "p3_eligible": counts["p3_eligible"],
                    "p4_eligible": counts["p4_eligible"],
                    "with_nonempty_a_step1": counts["with_nonempty_a_step1"],
                    "overall_criterion_conflicts": counts["overall_criterion_conflicts"],
                }
            )

    duplicate_image_path_count = sum(1 for _, count in image_paths.items() if count > 1)
    duplicate_image_path_examples = [
        {"image_path": image_path, "count": count}
        for image_path, count in image_paths.most_common()
        if image_path not in {"", "<empty>"} and count > 1
    ][:20]

    stage_counts = {
        "p1_eligible": sum(1 for row in rows if row["p1_eligible"]),
        "p2_eligible": sum(1 for row in rows if row["p2_eligible"]),
        "p3_eligible": sum(1 for row in rows if row["p3_eligible"]),
        "p4_eligible": sum(1 for row in rows if row["p4_eligible"]),
    }

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "family_csv_path": str(family_csv_path),
        "total_json_files": len(files),
        "valid_annotation_rows": len(rows),
        "skipped_non_annotation_files": {
            "count": len(skipped_non_annotation_files),
            "examples": skipped_non_annotation_files[:20],
        },
        "parse_error_files": {
            "count": len(parse_error_files),
            "examples": parse_error_files[:20],
        },
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "overall_label_counts": dict(sorted(overall_counts.items())),
        "criterion_coverage": {key: criterion_coverage.get(key, 0) for key in CANONICAL_CRITERIA},
        "stage_counts": stage_counts,
        "rows_with_nonempty_a_step1": rows_with_nonempty_a_step1,
        "overall_criterion_conflicts": overall_criterion_conflicts,
        "label_overall_conflicts": label_overall_conflicts,
        "label_filename_conflicts": label_filename_conflicts,
        "duplicate_image_paths": {
            "count": duplicate_image_path_count,
            "examples": duplicate_image_path_examples,
        },
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "top_sources": source_counts.most_common(20),
        "top_generators": generator_counts.most_common(20),
        "family_label_counts": [
            {"family": family, "label": label, "count": count}
            for (family, label), count in sorted(family_counts.items())
        ],
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

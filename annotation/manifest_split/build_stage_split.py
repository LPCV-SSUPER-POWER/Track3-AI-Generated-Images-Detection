#!/usr/bin/env python3
"""
Build clean stage-specific splits from master manifest.

Default policy:
- Exclude all overall/criterion conflict rows
- Keep raw annotation JSON unchanged
- Materialize split directories via hardlinks when possible
- Emit per-split summary + split manifest

Outputs:
- /annotations/data_p1
- /annotations/data_p2
- /annotations/data_p3
- /annotations/data_p4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


P1_KEYS = [
    "lighting",
    "edge",
    "texture",
    "perspective",
    "commonsense",
    "text_symbol",
    "human",
    "material",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean P1/P2/P3/P4 splits from master manifest")
    parser.add_argument(
        "--manifest_path",
        required=True,
        help="Master manifest jsonl from build_manifest.py",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Where to materialize data_p{1,2,3,4}/ split folders",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def shuffle_by_family(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[row["source_family"]].append(row)
    for family, items in family_rows.items():
        rng = random.Random(f"{seed}:{family}:{items[0]['label']}")
        rng.shuffle(items)
    return dict(family_rows)


def proportional_quota(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target > total:
        raise ValueError(f"target {target} exceeds total {total}")
    if total == 0 or target == 0:
        return {key: 0 for key in counts}

    quotas = {}
    remainders = []
    assigned = 0
    for key, count in sorted(counts.items()):
        raw = target * count / total
        base = min(count, math.floor(raw))
        quotas[key] = base
        assigned += base
        remainders.append((raw - base, count - base, key))

    remaining = target - assigned
    for _, _, key in sorted(remainders, key=lambda x: (-x[0], -x[1], x[2])):
        if remaining == 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1

    if remaining != 0:
        for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            while remaining and quotas[key] < count:
                quotas[key] += 1
                remaining -= 1
            if remaining == 0:
                break

    if sum(quotas.values()) != target:
        raise RuntimeError("quota assignment failed")
    return quotas


def pick_rows(rows: list[dict[str, Any]], target: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if target > len(rows):
        raise ValueError(f"target {target} exceeds pool size {len(rows)}")
    counts = Counter(row["source_family"] for row in rows)
    quotas = proportional_quota(dict(counts), target)
    family_rows = shuffle_by_family(rows, seed)
    selected = []
    for family in sorted(quotas):
        selected.extend(family_rows[family][: quotas[family]])
    return selected, quotas


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; rerun with --force")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def p1_target_from_row(row: dict[str, Any]) -> dict[str, str]:
    target = {key: row["criterion_labels"][key] for key in P1_KEYS}
    target["overall_label"] = row["overall_label"]
    return target


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["label"] for row in rows)
    per_family: dict[str, dict[str, int]] = defaultdict(lambda: {"real": 0, "fake": 0})
    for row in rows:
        per_family[row["source_family"]][row["label"]] += 1
    return {
        "total": len(rows),
        "counts": dict(sorted(counts.items())),
        "per_family": {key: per_family[key] for key in sorted(per_family)},
    }


def materialize_split(
    *,
    split_name: str,
    out_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    force: bool,
    emit_p1_targets: bool = False,
) -> None:
    ensure_clean_dir(out_dir, force)

    manifest_rows = []
    p1_target_rows = []
    for row in rows:
        src = Path(row["annotation_path"])
        dst = out_dir / src.name
        link_or_copy(src, dst)

        manifest_rows.append(row)
        if emit_p1_targets:
            p1_target_rows.append(
                {
                    "annotation_path": row["annotation_path"],
                    "image_path": row["image_path"],
                    "target": p1_target_from_row(row),
                }
            )

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "_split_manifest.jsonl", manifest_rows)
    if emit_p1_targets:
        write_jsonl(out_dir / "_p1_structured_targets.jsonl", p1_target_rows)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    output_root = Path(args.output_root).resolve()

    rows = read_manifest(manifest_path)
    clean_rows = [row for row in rows if not row["overall_criterion_conflict"]]

    p1_clean = [row for row in clean_rows if row["p1_eligible"]]
    p2_clean = [row for row in clean_rows if row["p2_eligible"]]
    p3_clean = [row for row in clean_rows if row["p3_eligible"]]

    # P2 first: this is the tightest clean pool.
    p2_real_pool = [row for row in p2_clean if row["label"] == "real"]
    p2_fake_pool = [row for row in p2_clean if row["label"] == "fake"]
    p2_each_target = min(30000, len(p2_real_pool), len(p2_fake_pool))

    p2_real, p2_real_quota = pick_rows(p2_real_pool, p2_each_target, args.seed)
    p2_fake, p2_fake_quota = pick_rows(p2_fake_pool, p2_each_target, args.seed)
    p2_rows = sorted(p2_real + p2_fake, key=lambda row: row["annotation_path"])
    p2_selected_paths = {row["annotation_path"] for row in p2_rows}

    # P1: extend clean P2 pool with extra clean P1-only rows up to 30k/30k.
    p1_real_pool = [row for row in p1_clean if row["label"] == "real"]
    p1_fake_pool = [row for row in p1_clean if row["label"] == "fake"]

    p1_real_remaining = [row for row in p1_real_pool if row["annotation_path"] not in p2_selected_paths]
    p1_fake_remaining = [row for row in p1_fake_pool if row["annotation_path"] not in p2_selected_paths]

    p1_real_extra_target = max(0, 30000 - len(p2_real))
    p1_fake_extra_target = max(0, 30000 - len(p2_fake))
    p1_real_extra, p1_real_extra_quota = pick_rows(p1_real_remaining, p1_real_extra_target, args.seed + 11)
    p1_fake_extra, p1_fake_extra_quota = pick_rows(p1_fake_remaining, p1_fake_extra_target, args.seed + 17)
    p1_rows = sorted(p2_rows + p1_real_extra + p1_fake_extra, key=lambda row: row["annotation_path"])
    p1_selected_paths = {row["annotation_path"] for row in p1_rows}

    # P3/P4: exact 1:2 ratio under clean no-conflict constraint.
    p3_real_pool = [row for row in p3_clean if row["label"] == "real"]
    p3_fake_pool = [row for row in p3_clean if row["label"] == "fake"]
    p3_real_requested = 20000
    p3_fake_requested = 40000
    p3_real_target = min(p3_real_requested, len(p3_real_pool), len(p3_fake_pool) // 2)
    p3_fake_target = 2 * p3_real_target

    p3_unused_real = [row for row in p3_real_pool if row["annotation_path"] not in p1_selected_paths]
    p3_unused_fake = [row for row in p3_fake_pool if row["annotation_path"] not in p1_selected_paths]
    p3_seen_real = [row for row in p3_real_pool if row["annotation_path"] in p1_selected_paths]
    p3_seen_fake = [row for row in p3_fake_pool if row["annotation_path"] in p1_selected_paths]

    p3_real_unused_take = min(len(p3_unused_real), p3_real_target)
    p3_fake_unused_take = min(len(p3_unused_fake), p3_fake_target)

    p3_real_unused, p3_real_unused_quota = pick_rows(p3_unused_real, p3_real_unused_take, args.seed + 101)
    p3_fake_unused, p3_fake_unused_quota = pick_rows(p3_unused_fake, p3_fake_unused_take, args.seed + 103)

    p3_real_remaining_target = p3_real_target - len(p3_real_unused)
    p3_fake_remaining_target = p3_fake_target - len(p3_fake_unused)

    p3_real_seen, p3_real_seen_quota = pick_rows(p3_seen_real, p3_real_remaining_target, args.seed + 107)
    p3_fake_seen, p3_fake_seen_quota = pick_rows(p3_seen_fake, p3_fake_remaining_target, args.seed + 109)

    p3_rows = sorted(
        p3_real_unused + p3_fake_unused + p3_real_seen + p3_fake_seen,
        key=lambda row: row["annotation_path"],
    )

    p1_summary = {
        **summarize_rows(p1_rows),
        "name": "data_p1",
        "seed": args.seed,
        "selection_mode": "clean no-conflict, P2 subset + P1-only top-up",
        "requested_target": {"real": 30000, "fake": 30000},
        "realized_target": summarize_rows(p1_rows)["counts"],
        "constraints": {
            "exclude_overall_criterion_conflict": True,
            "raw_annotation_json_modified": False,
            "p1_target_materialized_separately": True,
        },
        "base_from_p2": {"real": len(p2_real), "fake": len(p2_fake)},
        "extra_from_p1_only": {"real": len(p1_real_extra), "fake": len(p1_fake_extra)},
        "base_quota_from_p2": {"real": p2_real_quota, "fake": p2_fake_quota},
        "extra_quota_from_p1_only": {"real": p1_real_extra_quota, "fake": p1_fake_extra_quota},
    }

    p2_summary = {
        **summarize_rows(p2_rows),
        "name": "data_p2",
        "seed": args.seed,
        "selection_mode": "clean no-conflict, balanced, label-wise proportional by family",
        "requested_target": {"real": 30000, "fake": 30000},
        "realized_target": summarize_rows(p2_rows)["counts"],
        "constraints": {
            "exclude_overall_criterion_conflict": True,
            "raw_annotation_json_modified": False,
            "limited_by_clean_p2_fake_pool": True,
        },
        "family_quota": {"real": p2_real_quota, "fake": p2_fake_quota},
    }

    p3_summary = {
        **summarize_rows(p3_rows),
        "name": "data_p3",
        "seed": args.seed,
        "selection_mode": "clean no-conflict, unused-first + seen top-up, exact 1:2 within feasible capacity",
        "requested_target": {"real": p3_real_requested, "fake": p3_fake_requested},
        "realized_target": summarize_rows(p3_rows)["counts"],
        "constraints": {
            "exclude_overall_criterion_conflict": True,
            "raw_annotation_json_modified": False,
            "limited_by_clean_p3_fake_pool": True,
        },
        "unused_pool": {
            "real": len(p3_unused_real),
            "fake": len(p3_unused_fake),
            "total": len(p3_unused_real) + len(p3_unused_fake),
        },
        "used_carryover": {
            "real": len(p3_real_seen),
            "fake": len(p3_fake_seen),
            "total": len(p3_real_seen) + len(p3_fake_seen),
        },
        "unused_selected": {
            "real": len(p3_real_unused),
            "fake": len(p3_fake_unused),
        },
        "used_selected": {
            "real": len(p3_real_seen),
            "fake": len(p3_fake_seen),
        },
        "unused_quota": {"real": p3_real_unused_quota, "fake": p3_fake_unused_quota},
        "used_quota": {"real": p3_real_seen_quota, "fake": p3_fake_seen_quota},
    }

    p4_summary = {
        **p3_summary,
        "name": "data_p4",
    }

    materialize_split(
        split_name="p1",
        out_dir=output_root / "data_p1",
        rows=p1_rows,
        summary=p1_summary,
        force=args.force,
        emit_p1_targets=True,
    )
    materialize_split(
        split_name="p2",
        out_dir=output_root / "data_p2",
        rows=p2_rows,
        summary=p2_summary,
        force=args.force,
    )
    materialize_split(
        split_name="p3",
        out_dir=output_root / "data_p3",
        rows=p3_rows,
        summary=p3_summary,
        force=args.force,
    )
    materialize_split(
        split_name="p4",
        out_dir=output_root / "data_p4",
        rows=p3_rows,
        summary=p4_summary,
        force=args.force,
    )

    print(
        json.dumps(
            {
                "p1_total": len(p1_rows),
                "p2_total": len(p2_rows),
                "p3_total": len(p3_rows),
                "p4_total": len(p3_rows),
                "p1_counts": summarize_rows(p1_rows)["counts"],
                "p2_counts": summarize_rows(p2_rows)["counts"],
                "p3_counts": summarize_rows(p3_rows)["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

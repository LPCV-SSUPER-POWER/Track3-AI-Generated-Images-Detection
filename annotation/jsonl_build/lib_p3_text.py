import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
TARGET_TOKENS = 440
TOKEN_LIMIT = 500
INITIAL_EVIDENCE_WORDS = 20
MIN_EVIDENCE_WORDS = 8
INITIAL_RESPONSE_BUDGET = 145
MIN_RESPONSE_BUDGET = 95

ALL_CRITERIA = [
    "Edges & Boundaries",
    "Texture & Resolution",
    "Material & Object Details",
    "Physical & Common Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Lighting & Shadows Consistency",
    "Perspective & Spatial Relationships",
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


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def normalize_score(value):
    try:
        score = int(value)
    except Exception:
        score = 0
    return 0 if score <= 0 else 1


def parse_a_step1_prompt(path: Path):
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().strip('"').strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def load_annotations(annotation_dirs):
    entries = []
    for directory in annotation_dirs:
        for path in sorted(directory.glob("*.json")):
            if path.name.startswith("_"):
                continue
            try:
                entries.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
    return entries


def stratified_pick(entries, total, seed):
    buckets = {}
    for path, ann in entries:
        meta = ann.get("_meta", {})
        key = (meta.get("label", "unknown"), meta.get("generator", "unknown"))
        buckets.setdefault(key, []).append((path, ann))

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected = []
    keys = sorted(buckets)
    while len(selected) < min(total, len(entries)) and keys:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < total:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    rng.shuffle(selected)
    return selected


def build_lookup(annotation):
    lookup = {}
    for item in annotation.get("per_criterion", []):
        name = CRITERION_ALIASES.get(item.get("criterion", ""), item.get("criterion", ""))
        lookup[name] = item
    return lookup


def compress_sentence(text, max_words):
    words = text.replace("\n", " ").split()
    if len(words) <= max_words:
        return " ".join(words).strip()

    clipped = " ".join(words[:max_words]).strip(" ,;:")
    for sep in [". ", "; ", ", "]:
        idx = clipped.rfind(sep)
        if idx > len(clipped) * 0.55:
            clipped = clipped[: idx + 1].strip()
            break
    if not clipped.endswith((".", "!", "?")):
        clipped += "."
    return clipped


def compress_to_token_budget(text, tokenizer, budget):
    tokens = tokenizer.encode(text)
    if len(tokens) <= budget:
        return text.strip()

    clipped = tokenizer.decode(tokens[:budget], skip_special_tokens=True).strip()
    for sep in ["\n\n", ". ", "; "]:
        idx = clipped.rfind(sep)
        if idx > len(clipped) * 0.55:
            clipped = clipped[: idx + (0 if sep == "\n\n" else 1)].strip()
            break
    if not clipped.endswith((".", "!", "?")):
        clipped += "."
    return clipped


def build_p3_text(annotation, evidence_words):
    lookup = build_lookup(annotation)
    lines = ["Structured analysis:"]
    fake_votes = 0

    for criterion in ALL_CRITERIA:
        item = lookup.get(criterion, {})
        evidence = item.get("evidence", "No decisive artifact is visible for this criterion.")
        score = normalize_score(item.get("aigc score", item.get("aigc_score", item.get("score", 0))))
        fake_votes += score
        prefix = "Artifact:" if score else "Observation:"
        compressed = compress_sentence(evidence, evidence_words)
        lines.append(f"- {criterion}: {prefix} {compressed}")

    overall = annotation.get("overall_likelihood", "Uncertain")
    if overall not in {"Real", "AI-Generated", "Uncertain"}:
        overall = "AI-Generated" if fake_votes >= 2 else "Real"

    if overall == "AI-Generated":
        conclusion = "Conclusion: The image is most likely AI-Generated because multiple criteria show artifact evidence."
    elif overall == "Real":
        conclusion = "Conclusion: The image is most likely Real because the criteria remain visually coherent without decisive generation artifacts."
    else:
        conclusion = "Conclusion: The image remains Uncertain because the evidence is mixed and not all criteria agree."

    lines.append(conclusion)
    return "\n".join(lines)


def build_p3_from_a_step1_responses(annotation, tokenizer, per_response_budget):
    responses = annotation.get("_meta", {}).get("a_step1_responses") or []
    cleaned = []
    for idx, response in enumerate(responses[:3], start=1):
        compact = compress_to_token_budget(response.replace("[BEGIN]:", "").replace("[END]", "").strip(), tokenizer, per_response_budget)
        cleaned.append(f"[Analysis {idx}]\n{compact}")

    if not cleaned:
        return None

    overall = annotation.get("overall_likelihood", "Uncertain")
    if overall == "AI-Generated":
        conclusion = "Final conclusion: The image is most likely AI-Generated."
    elif overall == "Real":
        conclusion = "Final conclusion: The image is most likely Real."
    else:
        conclusion = "Final conclusion: The image remains Uncertain."
    return "\n\n".join(cleaned + [conclusion])


def fit_p3_text(annotation, tokenizer):
    response_budget = INITIAL_RESPONSE_BUDGET
    best_text = None
    best_tokens = 10**9

    while response_budget >= MIN_RESPONSE_BUDGET:
        text = build_p3_from_a_step1_responses(annotation, tokenizer, response_budget)
        if text:
            token_count = len(tokenizer.encode(text))
            if token_count <= TOKEN_LIMIT:
                best_text = text
                best_tokens = token_count
                if abs(token_count - TARGET_TOKENS) <= 35:
                    return text, token_count
            elif token_count < best_tokens:
                best_text = text
                best_tokens = token_count
        response_budget -= 10

    evidence_words = INITIAL_EVIDENCE_WORDS
    fallback_text = build_p3_text(annotation, evidence_words)
    fallback_tokens = len(tokenizer.encode(fallback_text))

    while evidence_words >= MIN_EVIDENCE_WORDS:
        text = build_p3_text(annotation, evidence_words)
        token_count = len(tokenizer.encode(text))
        if token_count <= TOKEN_LIMIT:
            fallback_text = text
            fallback_tokens = token_count
            if token_count <= TARGET_TOKENS:
                if best_text is None or abs(token_count - TARGET_TOKENS) < abs(best_tokens - TARGET_TOKENS):
                    return text, token_count
        evidence_words -= 2

    if best_text is None:
        best_text, best_tokens = fallback_text, fallback_tokens
    elif fallback_tokens <= TOKEN_LIMIT and abs(fallback_tokens - TARGET_TOKENS) < abs(best_tokens - TARGET_TOKENS):
        best_text, best_tokens = fallback_text, fallback_tokens

    if best_tokens > TOKEN_LIMIT:
        clipped = tokenizer.decode(tokenizer.encode(best_text)[:TOKEN_LIMIT], skip_special_tokens=True).strip()
        if not clipped.endswith((".", "!", "?")):
            clipped += "."
        return clipped, len(tokenizer.encode(clipped))

    return best_text, best_tokens


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build P3 single-prompt image+evidence dataset")
    # NOTE: This module is imported by build_p234_jsonl.py for its helper functions.
    # The standalone main() args below only apply when running this module directly.
    parser.add_argument("--annotation_dirs", nargs="+", required=True)
    parser.add_argument("--a_step1_prompt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    prompt_text = parse_a_step1_prompt(Path(args.a_step1_prompt))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_dirs = [Path(p) for p in args.annotation_dirs]
    entries = load_annotations(ann_dirs)
    selected = stratified_pick(entries, args.total, args.seed)

    rows = []
    meta_rows = []
    for path, ann in selected:
        meta = ann.get("_meta", {})
        image_path = meta.get("image_path")
        if not image_path:
            continue

        assistant_text, token_count = fit_p3_text(ann, tokenizer)
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": "<image>\n" + prompt_text},
                    {"from": "gpt", "value": assistant_text},
                ],
                "images": [image_path],
            }
        )
        meta_rows.append(
            {
                "annotation_file": str(path),
                "image_id": meta.get("image_id"),
                "image_path": image_path,
                "label": meta.get("label"),
                "source": meta.get("source"),
                "generator": meta.get("generator"),
                "token_count": token_count,
            }
        )

    split_idx = int(len(rows) * (1 - args.val_ratio))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]
    train_meta = meta_rows[:split_idx]
    val_meta = meta_rows[split_idx:]

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    (output_dir / "train.meta.json").write_text(
        json.dumps(train_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "val.meta.json").write_text(
        json.dumps(val_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "total_selected": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "max_token_count": max((m["token_count"] for m in meta_rows), default=0),
        "min_token_count": min((m["token_count"] for m in meta_rows), default=0),
        "avg_token_count": round(sum((m["token_count"] for m in meta_rows), 0) / max(len(meta_rows), 1), 2),
        "annotation_dirs": [str(p) for p in ann_dirs],
        "a_step1_prompt": str(args.a_step1_prompt),
        "target_tokens": TARGET_TOKENS,
        "token_limit": TOKEN_LIMIT,
    }
    (output_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

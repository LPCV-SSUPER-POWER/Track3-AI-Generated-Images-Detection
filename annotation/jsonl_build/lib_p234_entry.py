"""
PoC Step C: Generate SFT training data from VLM annotations.
Converts annotations into LLaMA-Factory sharegpt format.

Per image, generates 4 conversation entries:
  - Stage 1-1: image + sub-prompt 1 → evidence text (Edge, Texture, Material)
  - Stage 1-2: image + sub-prompt 2 → evidence text (Physical, Text, Human)
  - Stage 1-3: image + sub-prompt 3 → evidence text (Lighting, Perspective)
  - P4: text-only (no image) + a_step2 prompt → JSON output

Score mapping: annotation score 0~2 → aigc_score 0 or 1
  score 0 → aigc_score 0
  score 1, 2 → aigc_score 1
"""
import argparse
import json
import os
import random
from pathlib import Path
from transformers import AutoTokenizer

from lib_p3 import build_p3_full, build_p3_slice

# Load tokenizer for token counting (same as finetune target)
_TOKENIZER = None
TOKEN_LIMIT = 500

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return _TOKENIZER

def truncate_to_tokens(text, max_tokens=TOKEN_LIMIT):
    """Truncate text to max_tokens at sentence boundary."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    # Decode to max_tokens, then cut at last sentence boundary
    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    last_period = truncated.rfind(".")
    if last_period > len(truncated) // 2:
        truncated = truncated[:last_period + 1]
    return truncated


def is_garbage(text):
    """Check if text is garbage (repeated chars, too short, meaningless)."""
    if not text or len(text.strip()) < 20:
        return True
    # Check for repeated character patterns
    stripped = text.strip()
    unique_chars = set(stripped.replace(" ", "").replace("\n", ""))
    if len(unique_chars) <= 3:
        return True
    # Check for excessive repetition of single char
    for char in "!?.*#":
        if stripped.count(char) > len(stripped) * 0.3:
            return True
    return False


# Stage 1 sub-prompts
A_STEP1_PROMPTS = [
    "Is this image real or fake? Think step-by-step before giving a conclusion. Please analyze based on the following three aspects: Edge & Boundary Integrity, Texture & Resolution Coherence, Material & Object Detail Fidelity.",
    "Is this image real or fake? Think step-by-step before giving a conclusion. Please analyze based on the following three aspects: Physical & Common Sense Logic, Text & Symbol Authenticity, Human & Biological Structure Integrity.",
    "Is this image real or fake? Think step-by-step before giving a conclusion. Please analyze based on the following two aspects: Lighting & Shadow Consistency, Perspective & Spatial Accuracy.",
]

# Criterion groups matching each sub-prompt (a_step2.txt names)
CRITERION_GROUPS = [
    ["Edges & Boundaries", "Texture & Resolution", "Material & Object Details"],
    ["Physical & Common Sense Logic", "Text & Symbols", "Human & Biological Structure Integrity"],
    ["Lighting & Shadows Consistency", "Perspective & Spatial Relationships"],
]

# Load a_step2 prompt once — self-contained path (annotation/prompts/a_step2.txt)
_A_STEP2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "a_step2.txt")
A_STEP2_PROMPT = open(_A_STEP2_PATH).read().strip()

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

# Criterion name mapping (annotation.json names → a_step2.txt names)
CRITERION_MAP = {
    "Lighting & Shadows Consistency": "Lighting & Shadows Consistency",
    "Edges & Boundaries": "Edges & Boundaries",
    "Texture & Resolution": "Texture & Resolution",
    "Perspective & Spatial Relationships": "Perspective & Spatial Relationships",
    "Physical & Common-Sense Logic": "Physical & Common Sense Logic",
    "Physical & Common Sense Logic": "Physical & Common Sense Logic",
    "Text & Symbols": "Text & Symbols",
    "Human & Biological Structure Integrity": "Human & Biological Structure Integrity",
    "Material & Object Details": "Material & Object Details",
}


def score_to_aigc(score):
    """Map internal score (0~2) to aigc_score (0 or 1)."""
    try:
        s = int(score)
        return 0 if s == 0 else 1
    except (ValueError, TypeError):
        return 0


def build_p3_answer(annotation, criterion_group):
    """Build Stage 1 answer text from annotation for a specific criterion group."""
    per_criterion = annotation.get("per_criterion", [])
    # Build lookup by normalized criterion name
    criteria_lookup = {}
    for c in per_criterion:
        name = CRITERION_MAP.get(c.get("criterion", ""), c.get("criterion", ""))
        criteria_lookup[name] = c

    parts = []
    for criterion_name in criterion_group:
        c = criteria_lookup.get(criterion_name, {})
        evidence = c.get("evidence", "No specific artifacts detected.")
        parts.append(f"**{criterion_name}**: {evidence}")

    # Add conclusion
    likelihood = annotation.get("overall_likelihood", "Uncertain")
    parts.append(f"\nConclusion: This image is most likely **{likelihood}**.")
    return "\n\n".join(parts)


def build_p4_json(annotation, max_tokens=TOKEN_LIMIT):
    """Build Stage 2 JSON output from annotation with aigc_score mapping.
    Shrinks evidence text to fit within max_tokens."""
    tokenizer = get_tokenizer()
    per_criterion = annotation.get("per_criterion", [])
    criteria_lookup = {}
    for c in per_criterion:
        name = CRITERION_MAP.get(c.get("criterion", ""), c.get("criterion", ""))
        criteria_lookup[name] = c

    # Collect evidence texts
    criteria_data = []
    for criterion_name in ALL_CRITERIA:
        c = criteria_lookup.get(criterion_name, {})
        evidence = c.get("evidence", "No specific artifacts detected.")
        score = c.get("score", c.get("aigc score", c.get("aigc_score", 0)))
        aigc = score_to_aigc(score)
        criteria_data.append({
            "criterion": criterion_name,
            "evidence": evidence,
            "aigc score": aigc,
        })

    likelihood = annotation.get("overall_likelihood", "Uncertain")

    # Try full JSON first
    result = json.dumps({"per_criterion": criteria_data, "overall_likelihood": likelihood}, ensure_ascii=False)
    n_tokens = len(tokenizer.encode(result))

    if n_tokens <= max_tokens:
        return result

    # Shrink evidence texts progressively until it fits
    max_evidence_chars = 200
    while n_tokens > max_tokens and max_evidence_chars >= 30:
        for cd in criteria_data:
            if len(cd["evidence"]) > max_evidence_chars:
                # Truncate at sentence boundary if possible
                text = cd["evidence"][:max_evidence_chars]
                last_period = text.rfind(".")
                if last_period > max_evidence_chars // 2:
                    text = text[:last_period + 1]
                cd["evidence"] = text
        result = json.dumps({"per_criterion": criteria_data, "overall_likelihood": likelihood}, ensure_ascii=False)
        n_tokens = len(tokenizer.encode(result))
        max_evidence_chars -= 20

    return result


def make_sft_entries(image_path, annotation):
    """Generate 4 SFT entries for one image."""
    entries = []
    a_step1_responses = annotation.get("_meta", {}).get("a_step1_responses", [])

    # Stage 1: 3 sub-prompts (with image)
    for i, (prompt, criterion_group) in enumerate(zip(A_STEP1_PROMPTS, CRITERION_GROUPS)):
        # Use VLM-generated response if available, otherwise build from annotation
        if a_step1_responses and i < len(a_step1_responses) and not is_garbage(a_step1_responses[i]):
            answer = a_step1_responses[i]
        else:
            # Fallback to annotation-based answer if VLM response is garbage
            answer = build_p3_answer(annotation, criterion_group)

        entries.append({
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": truncate_to_tokens(answer)},
            ],
            "images": [str(image_path)],
        })

    # P4: text-only (no image)
    # Input = combined a_step1 responses + a_step2 prompt
    if a_step1_responses:
        combined = "\n\n".join([f"[Analysis {j+1}]\n{r}" for j, r in enumerate(a_step1_responses)])
    else:
        combined = "\n\n".join([
            f"[Analysis {j+1}]\n{build_p3_answer(annotation, cg)}"
            for j, cg in enumerate(CRITERION_GROUPS)
        ])

    a_step2_prompt = A_STEP2_PROMPT
    p4_input = f"{combined}\n\n{a_step2_prompt}"
    p4_answer = build_p4_json(annotation)

    entries.append({
        "conversations": [
            {"from": "human", "value": p4_input},
            {"from": "gpt", "value": p4_answer},  # already fitted to 500 tokens in build_p4_json
        ],
        # No "images" field for Stage 2 (text-only)
    })

    return entries


def make_p3p4_entries(image_path, annotation, a_step2_prompt: str | None = None):
    """Generate 4 SFT entries using criterion-based P3 text (image+evidence format)."""
    entries = []
    a_step2_prompt = a_step2_prompt or A_STEP2_PROMPT

    for i, prompt in enumerate(A_STEP1_PROMPTS):
        answer = build_p3_slice(annotation, i, evidence_words=16)
        entries.append(
            {
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt}"},
                    {"from": "gpt", "value": answer},
                ],
                "images": [str(image_path)],
            }
        )

    p4_input = f"{build_p3_full(annotation, evidence_words=16)}\n\n{a_step2_prompt}"
    p4_answer = build_p4_json(annotation)
    entries.append(
        {
            "conversations": [
                {"from": "human", "value": p4_input},
                {"from": "gpt", "value": p4_answer},
            ],
        }
    )
    return entries


def main():
    # NOTE: This module is imported by build_p234_jsonl.py for its helper functions.
    # The standalone main() args below only apply when running this module directly.
    parser = argparse.ArgumentParser(description="PoC: Generate SFT data from VLM annotations")
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load human annotation for Fake images (ground truth)
    human_annot_path = Path(args.sample_dir) / "annotation.json"
    human_annotations = {}
    if human_annot_path.exists():
        with open(human_annot_path, encoding="utf-8") as fh:
            human_annotations = json.load(fh)
        print(f"Loaded human annotations for {len(human_annotations)} Fake images")

    # Collect VLM-generated annotation files
    annotation_dir = Path(args.annotation_dir)
    annotations = {}
    for f in sorted(annotation_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("parse_error"):
            print(f"  Skipping {f.name} (parse error)")
            continue
        fname = f.stem
        label = data.get("_meta", {}).get("label", "")

        if label == "ai-generated":
            # Fake: use human annotation if available
            # Match by filename (remove "Copy of " prefix)
            clean_name = fname.replace("Copy of ", "")
            if clean_name in human_annotations:
                # Use human annotation, but keep VLM a_step1 responses for evidence style
                human_data = human_annotations[clean_name]
                data["per_criterion"] = human_data["per_criterion"]
                data["overall_likelihood"] = "AI-Generated"
                print(f"  Using human annotation for: {fname}")
            else:
                # No human annotation, force overall to AI-Generated
                data["overall_likelihood"] = "AI-Generated"
        else:
            # Real: force score=0, overall=Real
            data["overall_likelihood"] = "Real"
            if "per_criterion" in data:
                for c in data["per_criterion"]:
                    c["aigc score"] = 0
                    c["score"] = 0

        annotations[fname] = data

    print(f"Loaded {len(annotations)} annotations")

    # Match with images
    entries_by_image = {}
    for subfolder in ["Real", "Fake"]:
        img_dir = Path(args.sample_dir) / subfolder
        for img_path in sorted(img_dir.glob("*.png")):
            fname = img_path.stem
            if fname in annotations:
                entries = make_sft_entries(img_path, annotations[fname])
                entries_by_image[fname] = entries

    print(f"Generated entries for {len(entries_by_image)} images ({len(entries_by_image)*4} total)")

    # Split train/val by image (not by entry)
    image_names = list(entries_by_image.keys())
    random.seed(args.seed)
    random.shuffle(image_names)
    n_train = int(len(image_names) * args.train_ratio)
    train_names = set(image_names[:n_train])
    val_names = set(image_names[n_train:])

    train_entries = []
    val_entries = []
    for name, entries in entries_by_image.items():
        if name in train_names:
            train_entries.extend(entries)
        else:
            val_entries.extend(entries)

    # Shuffle entries
    random.shuffle(train_entries)
    random.shuffle(val_entries)

    # Write JSONL
    train_path = Path(args.output_dir) / "train.jsonl"
    val_path = Path(args.output_dir) / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nTrain: {len(train_entries)} entries ({n_train} images) → {train_path}")
    print(f"Val:   {len(val_entries)} entries ({len(val_names)} images) → {val_path}")

    # Stats
    train_with_img = sum(1 for e in train_entries if "images" in e)
    train_text_only = sum(1 for e in train_entries if "images" not in e)
    print(f"\nTrain breakdown: {train_with_img} with image (P3), {train_text_only} text-only (P4)")


if __name__ == "__main__":
    main()

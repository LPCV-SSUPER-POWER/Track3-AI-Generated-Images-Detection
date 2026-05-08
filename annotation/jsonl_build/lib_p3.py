#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any


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

P2_CRITERION_GROUPS = [
    ["Edges & Boundaries", "Texture & Resolution", "Material & Object Details"],
    ["Physical & Common Sense Logic", "Text & Symbols", "Human & Biological Structure Integrity"],
    ["Lighting & Shadows Consistency", "Perspective & Spatial Relationships"],
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

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
TRAILING_STOPWORDS = {
    "that",
    "which",
    "with",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "from",
    "by",
    "is",
    "are",
    "be",
    "a",
    "an",
    "the",
    "might",
    "could",
    "may",
    "how",
}

REALISH_HINTS = (
    "realistic",
    "natural",
    "consistent",
    "coherent",
    "well-defined",
    "sharp",
    "clear",
    "legible",
    "plausible",
    "align well",
    "looks real",
    "appears real",
)

ARTIFACT_HINTS = (
    "artifact",
    "artificial",
    "ai-generated",
    "generated",
    "fake",
    "warped",
    "distorted",
    "unnatural",
    "inconsistent",
    "misaligned",
    "gibberish",
    "extra",
    "smoothed",
    "repetitive",
    "implausible",
    "exaggerated",
)

OBSERVATION_FALLBACKS = {
    "Lighting & Shadows Consistency": "Lighting remains coherent across the scene with physically plausible shadow behavior.",
    "Edges & Boundaries": "Boundaries remain clean and preserve natural edge variation without synthetic blending.",
    "Texture & Resolution": "Surface texture stays locally consistent and preserves realistic fine detail.",
    "Perspective & Spatial Relationships": "Perspective and depth cues remain physically consistent across the scene.",
    "Physical & Common Sense Logic": "Objects and scene composition remain plausible under normal physical expectations.",
    "Text & Symbols": "Visible text or symbolic details do not show broken characters or synthetic corruption.",
    "Human & Biological Structure Integrity": "Biological structure remains anatomically consistent without visible deformation.",
    "Material & Object Details": "Material appearance and object detail remain consistent with real photographic capture.",
}

ARTIFACT_FALLBACKS = {
    "Lighting & Shadows Consistency": "Lighting may look broadly coherent, but shadow behavior remains weak for a natural photograph.",
    "Edges & Boundaries": "Boundaries look overly smooth and lack the irregular edge variation expected in a real capture.",
    "Texture & Resolution": "Texture appears unnaturally uniform and does not preserve realistic fine-grained detail.",
    "Perspective & Spatial Relationships": "Scene geometry and depth cues read as implausible for a real camera capture.",
    "Physical & Common Sense Logic": "Object structure or scene composition shows implausible physical relationships.",
    "Text & Symbols": "Text or symbolic fidelity does not provide trustworthy evidence of a natural real-image capture.",
    "Human & Biological Structure Integrity": "Biological structure and proportions are inconsistent with natural anatomy.",
    "Material & Object Details": "Surface detail looks polished or synthetic despite superficially realistic cues.",
}

NOISY_PREFIX_RE = re.compile(
    r"^(the image|this image|in the image|overall|there (?:are|is)|based on .*?|to determine .*?|after analyzing .*?)\s+",
    re.IGNORECASE,
)
NO_VISIBLE_TEXT_RE = re.compile(r"no visible text|no visible texts|no text|no symbols", re.IGNORECASE)
NO_HUMAN_RE = re.compile(r"does not contain any human|no human|no biological structure|not applicable", re.IGNORECASE)


def clean_text(text: Any) -> str:
    text = str(text or "").replace("[BEGIN]:", "").replace("[END]", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def approx_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


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
    lookup: dict[str, dict[str, Any]] = {}
    for item in annotation.get("per_criterion", []):
        key = CRITERION_ALIASES.get(item.get("criterion", ""), item.get("criterion", ""))
        lookup[key] = item
    return lookup


def compress_sentence(text: str, max_words: int) -> str:
    words = clean_text(text).split()
    truncated = len(words) > max_words
    if len(words) <= max_words:
        clipped_words = words[:]
    else:
        clipped_words = words[:max_words]
    while clipped_words and clipped_words[-1].strip(" ,;:").lower() in TRAILING_STOPWORDS:
        clipped_words.pop()
    if not clipped_words:
        clipped_words = words[: max(1, min(len(words), max_words))]
    out = " ".join(clipped_words).strip(" ,;:")
    if truncated and "." in out:
        sentence_parts = [part.strip() for part in out.split(".") if part.strip()]
        if len(sentence_parts) >= 2 and len(sentence_parts[-1].split()) <= 4:
            keep = ".".join(sentence_parts[:-1]).strip()
            out = keep + "." if keep else sentence_parts[0] + "."
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    if out and out[-1] not in ".!?":
        out += "."
    return out


def score_text(text: str, hints: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(1 for hint in hints if hint in lower)


def split_mixed_clause(text: str, keep_after: bool) -> str:
    lower = text.lower()
    for connector in (" however ", " but ", " although ", " though ", " while "):
        idx = lower.find(connector)
        if idx != -1:
            return text[idx + len(connector):] if keep_after else text[:idx]
    return text


def rewrite_evidence(evidence: Any, criterion: str, score: int, evidence_words: int = 16) -> str:
    raw = clean_text(evidence)
    if not raw:
        base = ARTIFACT_FALLBACKS[criterion] if score else OBSERVATION_FALLBACKS[criterion]
        return compress_sentence(base, evidence_words)

    if criterion == "Text & Symbols" and NO_VISIBLE_TEXT_RE.search(raw):
        base = (
            "No visible text or symbols are present, and this criterion shows no synthetic corruption."
            if not score
            else "No trustworthy text or symbol detail is available to support real-image authenticity."
        )
        return compress_sentence(base, evidence_words)

    if criterion == "Text & Symbols" and not score and "." in raw:
        first_sentence = sentences = [segment.strip() for segment in SENTENCE_RE.split(raw) if segment.strip()]
        if first_sentence:
            return compress_sentence(first_sentence[0], evidence_words)

    if criterion == "Human & Biological Structure Integrity" and NO_HUMAN_RE.search(raw):
        base = (
            "No human or biological subject is present, and this criterion shows no anatomical inconsistency."
            if not score
            else "Biological authenticity cannot be confirmed from the visible structure in this scene."
        )
        return compress_sentence(base, evidence_words)

    sentences = [segment.strip() for segment in SENTENCE_RE.split(raw) if segment.strip()]
    if not sentences:
        sentences = [raw]

    best = None
    best_gap = -10**9
    for sentence in sentences:
        art = score_text(sentence, ARTIFACT_HINTS)
        real = score_text(sentence, REALISH_HINTS)
        gap = art - real if score else real - art
        if gap > best_gap:
            best = sentence
            best_gap = gap

    assert best is not None
    best = split_mixed_clause(best, keep_after=bool(score))
    best = NOISY_PREFIX_RE.sub("", best).strip(" ,;:-")
    best = clean_text(best)

    art = score_text(best, ARTIFACT_HINTS)
    real = score_text(best, REALISH_HINTS)
    aligned = art >= real if score else real >= art
    if not aligned:
        base = ARTIFACT_FALLBACKS[criterion] if score else OBSERVATION_FALLBACKS[criterion]
        return compress_sentence(base, evidence_words)

    return compress_sentence(best, evidence_words)


def build_p3_text_internal(
    annotation: dict[str, Any],
    criterion_order: list[str],
    evidence_words: int = 16,
) -> str:
    lookup = build_lookup(annotation)
    lines = ["Structured analysis:"]
    fake_votes = 0
    for criterion in criterion_order:
        item = lookup.get(criterion, {})
        score = normalize_score(item.get("aigc score", item.get("aigc_score", item.get("score", 0))))
        fake_votes += score
        prefix = "Artifact:" if score else "Observation:"
        evidence = rewrite_evidence(item.get("evidence", ""), criterion, score, evidence_words=evidence_words)
        lines.append(f"- {criterion}: {prefix} {evidence}")

    overall = normalize_overall(annotation.get("overall_likelihood"), fake_votes)
    if overall == "AI-Generated":
        conclusion = "Conclusion: The image is most likely AI-Generated."
    elif overall == "Real":
        conclusion = "Conclusion: The image is most likely Real."
    else:
        conclusion = "Conclusion: The image remains Uncertain."
    lines.append(conclusion)
    return "\n".join(lines)


def build_p3_slice(
    annotation: dict[str, Any],
    group_index: int,
    evidence_words: int = 16,
) -> str:
    return build_p3_text_internal(annotation, P2_CRITERION_GROUPS[group_index], evidence_words=evidence_words)


def build_p3_full(annotation: dict[str, Any], evidence_words: int = 16) -> str:
    return build_p3_text_internal(annotation, ALL_CRITERIA, evidence_words=evidence_words)

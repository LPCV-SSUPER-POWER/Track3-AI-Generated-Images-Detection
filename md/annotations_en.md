# Annotations — Training Data Generation Guide

How the 4 training jsonl files (P1/P2/P3/P4) bundled in `datasets/` were created + reproduce guide.

The bundle already includes the result (199 MB, 12 files) under `datasets/`, ready to use as input for the P1→P4 chain in `train_en.md`. Follow the steps below only if you want to regenerate annotations from your own image set.

---

## `annotation/` Directory Structure

```
26LPCV_SSUPERPOWER_2nd/annotation/
│
├── prompts/                            (a_step2 sub-stage prompt for Step 1)
│   └── a_step2.txt                              a_step2 synthesis prompt   (2110 B)
│       (※ a_step1 prompt + FAKE_HINT are variables inside annotate.py)
│
├── inference/                          (Step 1 — annotation generation code)
│   └── annotate.py                              Qwen2.5-VL-7B inference   (10 KB)
│
├── manifest_split/                     (Step 2 — manifest + split code)
│   ├── build_manifest.py                        master manifest builder    (18 KB)
│   └── build_stage_split.py                     4-split builder (hardlink) (14 KB)
│
├── jsonl_build/                        (Step 3 — training JSONL builder code)
│   ├── build_p1_jsonl.py               P1 training entry main (standalone)        (21 KB)
│   ├── build_p234_jsonl.py             P2/P3/P4 training entry main               (21 KB)
│   ├── lib_p2.py                       P2 training entry builder (library)        (8 KB)
│   ├── lib_p3.py                       P3 training entry builder (library)        (11 KB)
│   ├── lib_p3_text.py                  P3 entry text token-fit helper             (12 KB)
│   ├── lib_p4_text.py                  P4 entry text/JSON token-fit helper        (14 KB)
│   └── lib_p234_entry.py               P234 common — human/gpt conversation wrap  (14 KB)
│
└── samples/                            (20 raw annotation JSON examples, by generator)
    ├── adm__fake__*.json                          3 files
    ├── biggan__fake__*.json                       3 files
    ├── sid_set__fake__*.json                      2 files
    ├── sid_set__real__*.json                      1 file
    ├── imagenet__real__*.json                     2 files
    ├── coco__real__*.json                         2 files
    ├── arforensics_infinity__fake__*.json         2 files
    ├── arforensics_janus_pro__fake__*.json        2 files
    ├── arforensics_llamagen__fake__*.json         1 file
    ├── arforensics_rar__fake__*.json              1 file
    └── synthscars__fake__*.json                   1 file
```

---

## Overall Flow

When annotation finishes → train consumes the result as training input.

### Raw images

ADM, BigGAN, SID (`data/images/`) + COCO train2017, ImageNet train (`<SHARED_DATASETS_ROOT>/`) + ARForensics, SynthScars (`datasets/raw/`)


### Annotation domain (this document)

**Step 1 — annotate** (Qwen2.5-VL-7B inference)

  - 2 inference sub-stages:
    - **a_step1** — image + prompt × 3 (`A_STEP1_PROMPTS` + `FAKE_HINT` are inside annotate.py)
    - **a_step2** — text-only × 1, per_criterion JSON synthesis (`prompts/a_step2.txt`)
  - 📦 Output → master pool: **89,263 raw annotation JSON** (fake 44,631 + real 44,631 + 1 parse error, ~50:50)
    - By generator: ARForensics (5 variants total) 54,652 (61%) + ImageNet 8,889 + SID 6,547 + BigGAN 5,214 + ADM 5,131 + SynthScars 5,000 + COCO 3,829

**Step 2 — manifest + 4-split partition**

  - `manifest_split/build_manifest.py` — read 89,264 raw annotations → **master manifest** (each row: image_path, label, source, criterion_labels normalized meta)
  - `manifest_split/build_stage_split.py` — receive master manifest → split into **4 splits (P1/P2/P3/P4)**. The same file is **shared via hardlink across 5 locations** → 0 extra disk usage
  - 📦 Output → 4 split folders (p1/p2/p3/p4) + each split's `_split_manifest.jsonl`

**Step 3 — Training JSONL conversion**

  - `jsonl_build/build_p1_jsonl.py` — split's raw annotations → **P1 training entries** (single-prompt sft format). Standalone (0 library imports)
  - `jsonl_build/build_p234_jsonl.py` — split's raw annotations → **P2/P3/P4 training entries** (P2 multi-prompt, P3 image+evidence, P4 text-only synthesis). Imports 5 libraries to build entries + token fit + human/gpt conversation wrap
  - 📦 Output → `datasets/data_p1`, `data_p2`, `data_p3`, `data_p4` (4 folders, ShareGPT-format jsonl) — see [Step 3 Results](#step-3-results--jsonl-statistics-of-bundled-datasets) for stats


---

**Terminology Guide**

| Notation | Meaning |
|---|---|
| **Step 1 / Step 2 / Step 3** | Major stages of the annotation pipeline (overall flow of this doc) |
| **a_step1 / a_step2** | Inference sub-stages within Step 1 (4 questions annotate.py asks Qwen2.5-VL) |
| **P1 / P2 / P3 / P4** | Training chain phases (covered in train_en.md) |

---

## Step 1 — Annotation Generation

### Files used

| File | Location |
|---|---|
| inference code | `annotation/inference/annotate.py` |
| a_step2 synthesis prompt | `annotation/prompts/a_step2.txt` |

> **a_step1 prompt and FAKE_HINT are variables inside `annotate.py`, not external files**:
> - `A_STEP1_PROMPTS` (lines 18–22): 3 prompts splitting 8 criteria into 3 groups
> - `FAKE_HINT` (line 24): one-line hint prepended to a_step1 prompt for fake images

### Model

- **`Qwen/Qwen2.5-VL-7B-Instruct`** (HuggingFace hub)
- **Inference only**, no training

### Multi-stage operation (Qwen2.5-VL is asked 4 times per image)

```
a_step1 (image + prompt, 3 questions):
  Q1: "Analyze only Edge·Texture·Material 3 criteria"        → response 1 (long text)
  Q2: "Analyze only Physical·Text·Human 3 criteria"          → response 2
  Q3: "Analyze only Lighting·Perspective 2 criteria"         → response 3

  → All prompts come from A_STEP1_PROMPTS in annotate.py:
     · Fake image → FAKE_HINT prepended to A_STEP1_PROMPTS[i]
     · Real image → A_STEP1_PROMPTS[i] as-is

a_step2 (text-only, no image):
  Q4: "Combine the 3 responses above into 8-criteria JSON"
  → prompt: prompts/a_step2.txt (external file)
  → output: per_criterion JSON (final annotation)
```

### Run

```bash
python annotation/inference/annotate.py \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --list_json /path/to/image_list.json \
  --output_dir annotations/0421_data \
  --batch_size 32
```

`--list_json` entry format:
```json
{"image_path": "/abs/path/to/image.png",
 "label": "fake",
 "source": "adm",
 "generator": "ADM"}
```

### Output — raw annotation JSON (1 image = 1 JSON)

See 20 examples in `annotation/samples/`. Schema:
```json
{
  "per_criterion": [
    {"criterion": "Lighting & Shadows Consistency", "evidence": "...", "aigc score": 0},
    {"criterion": "Edges & Boundaries", "evidence": "...", "aigc score": 1},
    ... (8 criteria total)
  ],
  "overall_likelihood": "Real" or "AI-Generated",
  "_meta": {
    "image_path": "...",
    "label": "fake",
    "source": "adm",
    "generator": "ADM",
    "a_step1_responses": ["response1", "response2", "response3"],
    "elapsed_sec": 31.8
  }
}
```

---

## Step 2 — Master Manifest + 4-Split Partition

### Files used

| File | Location |
|---|---|
| master manifest builder | `annotation/manifest_split/build_manifest.py` |
| 4-split builder | `annotation/manifest_split/build_stage_split.py` |

### 2-1. Build Master Manifest

```bash
python annotation/manifest_split/build_manifest.py \
  --input_dir annotations/0421_data \
  --output_dir annotations/0421_data/manifests
```

- Input: Step 1 output pool (`annotations/0421_data/`, 89,264 JSON)
- 3 outputs:
  - `manifests/master_manifest.jsonl` — normalized meta (each row: image_path, label, source, criterion_labels, ...)
  - `manifests/master_manifest_summary.json` — statistics
  - `manifests/family_counts.csv` — per-generator counts

### 2-2. Partition into 4 Splits

```bash
python annotation/manifest_split/build_stage_split.py \
  --manifest_path annotations/0421_data/manifests/master_manifest.jsonl \
  --annotations_dir annotations/0421_data \
  --output_root annotations
```

- Policy:
  - Exclude rows with overall ↔ criterion conflicts (clean only)
  - **Build splits via hardlink** → 0 extra disk usage (one file shared at 5 locations)
- Output — 4 split folders + fake/real distribution:

| split | total | fake | real | ratio |
|---|---|---|---|---|
| `data_p1/` | 60,000 | 30,000 | 30,000 | **50 : 50** |
| `data_p2/` | 59,910 | 29,955 | 29,955 | **50 : 50** |
| `data_p3/` | 44,931 | 29,954 | 14,977 | **66 : 33** |
| `data_p4/` | 44,931 | 29,954 | 14,977 | **66 : 33** |

Extra metadata in each split: `_split_manifest.jsonl` (all splits), `_p1_structured_targets.jsonl` (P1 only)

---

## Step 3 — Convert to Training JSONL

### Files used

| File | Location | Role |
|---|---|---|
| **P1 main** | `annotation/jsonl_build/build_p1_jsonl.py` | Builds P1 training entries (standalone, 0 library imports) |
| **P234 main** | `annotation/jsonl_build/build_p234_jsonl.py` | Builds P2/P3/P4 training entries (uses 5 libraries) |
| Library 1 | `annotation/jsonl_build/lib_p2.py` | P2 training entry builder (function for one entry) |
| Library 2 | `annotation/jsonl_build/lib_p3.py` | P3 training entry builder (function for one entry) |
| Library 3 | `annotation/jsonl_build/lib_p3_text.py` | Truncates text in P3 entries when over Qwen2-VL tokenizer limit |
| Library 4 | `annotation/jsonl_build/lib_p4_text.py` | Truncates text/JSON in P4 entries when over token limit |
| Library 5 | `annotation/jsonl_build/lib_p234_entry.py` | P234 common — final wrap of entries into human/gpt conversation format |

### 3-1. Build P1 training entries

```bash
python annotation/jsonl_build/build_p1_jsonl.py \
  --annotations_root annotations \
  --datasets_root datasets \
  --force
```

- Input: 4 split folders (especially `_split_manifest.jsonl`, `_p1_structured_targets.jsonl`) + raw JSON
- Output: `datasets/data_p1/{train,val}.jsonl` ← **P1**
- Format: ShareGPT single-prompt
  ```
  human: <image>\nClassify this image per criterion and overall. Return JSON only.
  gpt:   {"lighting": "Real", "edge": "Real", ..., "overall_label": "Real"}
  ```

### 3-2. Build P2/P3/P4 training entries

```bash
python annotation/jsonl_build/build_p234_jsonl.py \
  --annotations_root annotations \
  --datasets_root datasets \
  --force
```

- Input: 4 split folders + raw JSON
- Auto-imports 5 libraries (other 5 files in jsonl_build/)
- Output:
  - `datasets/data_p2/` ← **P2** (multi-prompt format entries)
  - `datasets/data_p3/` ← **P3** (image + a_step1 prompt → 8-criteria evidence)
  - `datasets/data_p4/` ← **P4** (text-only a_step2 synthesis)

Each split has a different entry format:
- P2: multi-prompt entry combining all 3 sub-prompts
- P3: image + 3 sub-prompts → 8 criteria evidence
- P4: text-only a_step2 synthesis (extract JSON from a_step1 responses, no image)

### Step 3 Results — JSONL Statistics of Bundled `datasets/`

Entry counts + fake/real ratios (per-image, based on meta.json) for the 4 jsonl files bundled in this repo:

| Folder | Format | jsonl files | train (entry) | val (entry) | train fake : real | val fake : real |
|---|---|---|---|---|---|---|
| `data_p1/` | sft (single prompt) | `train.jsonl`, `val.jsonl` | **54,007** | **5,993** | 27,004 : 27,003 (50:50) | 2,996 : 2,997 (50:50) |
| `data_p2/` | multi-prompt (4 entry/img) | `train.jsonl`, `val.jsonl` | **36,000** (= 9,000 img × 4) | **4,000** (= 1,000 × 4) | 4,505 : 4,495 (50:50) | 499 : 501 (50:50) |
| `data_p3/` | P3 image+evidence | `train.jsonl`, `val.jsonl` | **9,000** | **1,000** | 4,505 : 4,495 (50:50) | 499 : 501 (50:501) |
| `data_p4/` | P4 text-only synthesis | `train.jsonl`, `val.jsonl` | **8,104** | **896** | 6,080 : 2,024 **(75:25)** | 672 : 224 (75:25) |

> P4 only has 75% fake ratio — intentional imbalance (so P4 LoRA gets more exposure to fake annotation refinement).

---

## Dependencies

### Models (HuggingFace hub)
- `Qwen/Qwen2.5-VL-7B-Instruct` — for Step 1 annotation generation
- `Qwen/Qwen2-VL-2B-Instruct` — used by `lib_p3_text` / `lib_p4_text` in Step 3 as tokenizer (same as the training target model)

### Python packages (see `environment_en.md` for details)
- `transformers` (version supporting Qwen2.5-VL)
- `torch`, `accelerate`, `PIL`
- `qwen_vl_utils` (official Qwen2.5-VL image preprocessor)

### Raw image datasets
- ADM/BigGAN/SID — `data/images/`
- COCO train2017, ImageNet train — `<SHARED_DATASETS_ROOT>/`
- ARForensics — HuggingFace `Yanran21/ARForensics`
- SynthScars — (source TBD)

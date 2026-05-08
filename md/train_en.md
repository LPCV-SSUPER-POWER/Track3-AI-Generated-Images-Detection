# Train — `FINAL_RESULTS` Training Guide

Procedure for running the **P1→P4 training chain** using the 4 jsonl files (P1/P2/P3/P4) bundled in `datasets/`.

Training is reproducible. Since `datasets/` is already bundled, you can start training as soon as the environment + LLaMA-Factory are ready.

---

## `train/` Directory Structure

```
26LPCV_SSUPERPOWER_2nd/train/
│
├── configs/        (10 yaml — train + merge config for each P)
│   ├── train_p1.yaml        / merge_p1.yaml         P1 (custom SFT)
│   ├── train_p2.yaml        / merge_p2.yaml         P2 (custom + aux BCE)
│   ├── train_p3.yaml        / merge_p3.yaml         P3 (LLaMA-Factory)
│   ├── train_p4_warmup.yaml / merge_p4_warmup.yaml  P4 warmup — stage 1 adaptation (LLaMA-Factory)
│   └── train_p4.yaml        / merge_p4.yaml         P4 — stage 2 fine-tuning, final (LLaMA-Factory)
│
├── scripts/                     (11 py)
│   │── P1 training (custom SFT) ───────
│   ├── train_p1.py              main launcher (training entry point)
│   ├── dataset_p1.py            Dataset class (reads data_p1 jsonl)
│   ├── model_p1.py              model definition (Qwen2-VL + LoRA + overall binary head)
│   ├── merge_p1.py              LoRA → base merge
│   │── P2 training (custom + aux BCE) ───────
│   ├── train_p2.py              main launcher (token CE + overall BCE + criterion BCE)
│   ├── dataset_p2.py            Dataset class (reads data_p2 jsonl)
│   ├── model_p2.py              model definition (Qwen2-VL + LoRA + auxiliary heads)
│   ├── merge_p2.py              LoRA → base merge
│   │── Common helpers ───────
│   ├── prepare_runtime.py    runtime preparation (yaml substitution + dataset_info.json patch)
│   ├── check_inputs.py       preflight (dataset/image existence check)
│   └── fix_preprocessor.py   preprocessor config fix-up after merge
│
└── run.sh          End-to-end P1→P4 launcher
```

---

## Overall Flow

`annotations_en.md` output (`datasets/data_p1~p4`) → train chain → `FINAL_RESULTS`.

### Input — bundled `datasets/`

| split | format | jsonl |
|---|---|---|
| `data_p1/` | sft (single prompt) | `train.jsonl` (54,007), `val.jsonl` (5,993) |
| `data_p2/` | multi-prompt (4 entry/img) | `train.jsonl` (36,000), `val.jsonl` (4,000) |
| `data_p3/` | P3 image+evidence | `train.jsonl` (9,000), `val.jsonl` (1,000) |
| `data_p4/` | P4 text-only synthesis | `train.jsonl` (8,104), `val.jsonl` (896) |


### Training chain (5 stages, 10 steps)

```
Qwen/Qwen2-VL-2B-Instruct  (base)
   ↓ P1 train (data_p1, custom SFT)
   lora_p1
   ↓ P1 merge
   merged_p1
   ↓ P2 train (data_p2, custom + token CE + overall BCE + criterion BCE)
   lora_p2
   ↓ P2 merge
   merged_p2
   ↓ P3 train (data_p3, LLaMA-Factory)
   lora_p3
   ↓ P3 merge
   merged_p3
   ↓ P4 warmup train (data_p4, LLaMA-Factory, lr 5e-5, epoch 0.5)
   lora_p4_warmup
   ↓ P4 warmup merge
   merged_p4_warmup
   ↓ P4 final train (data_p4, LLaMA-Factory, lr 1e-5, epoch 0.25)
   lora_p4
   ↓ P4 final merge
   merged_p4  ⭐  (= FINAL_RESULTS, final output)
```

### Terminology Guide

| Notation | Meaning |
|---|---|
| **P1 / P2 / P3 / P4** | Phases of the training chain (4 stages, P4 has 2 sub-steps: warmup + final) |
| **train / merge** | Two sub-steps per phase: LoRA training + merging into base model |
| **lora_p\*** | LoRA adapter output (intermediate, before merge) |
| **merged_p\*** | base + LoRA merged full model (next phase's base) |

---

## Prerequisites

### 1. Python environment + LLaMA-Factory

See `environment_en.md`. Key:
- `26lpcv` conda env (Python 3.10, torch 1.13.1, transformers 4.46.1, peft 0.12.0, llamafactory 0.9.1)
- `llamafactory-cli` (`<conda env 26lpcv>/bin/llamafactory-cli`) — installed location of LLaMA-Factory v0.9.1 editable install

### 2. Preflight check — `check_inputs.py`

Verify all dataset/image dependencies exist before starting training.

```bash
cd train/
./scripts/check_inputs.py \
  --processed-data-root /path/to/26LPCV_SSUPERPOWER_2nd/datasets \
  --project-image-root <TRACK3_ROOT> \
  --shared-datasets-root /path/to/Datasets
```

Verification items:
- 8 processed JSONL files (train/val × 4 P)
- raw image directories (data/images/, ARForensics, SynthScars, COCO, ImageNet)

### 3. Runtime preparation — `prepare_runtime.py`

Auto-called by `run.sh`. Can also be invoked directly.

4 actions performed:
- yaml placeholder substitution (`__PROJECT_ROOT__`, `__RUN_ROOT__`, `__MODEL_ROOT__`, `__LLAMA_DATA_DIR__`)
- Copy processed JSONL to `${RUN_ROOT}/runtime_data/datasets/` while rewriting image_path (to user machine path)
- Register P3/P4 dataset aliases in LLaMA-Factory `dataset_info.json`
- Verify required data/image paths exist

---

## Run — `run.sh`

Run the 10-step P1→P4 chain in one line:

```bash
cd train/
./run.sh
```

For different machines/paths, override defaults via environment variables:

```bash
PROJECT_ROOT=<TRACK3_ROOT> \
PROCESSED_DATA_ROOT=/path/to/26LPCV_SSUPERPOWER_2nd/datasets \
PROJECT_IMAGE_ROOT=<TRACK3_ROOT> \
SHARED_DATASETS_ROOT=/path/to/Datasets \
RUN_ROOT=/path/to/runs/train \
P1_GPU=0 P2_GPU=0 P3_GPUS=0 P4_GPUS=0 MERGE_GPU=0 \
PYTHON=/path/to/python \
LLAMA=/path/to/llamafactory-cli \
./run.sh
```

**EXACT_REPRODUCTION policy** (default `=1`):
- `EXACT_REPRODUCTION=1` enforces single-GPU for P3/P4 (preserves effective batch size). Comma-separated GPUs (`P3_GPUS=0,1`) are rejected.
- `EXACT_REPRODUCTION=0` allows multi-GPU. But effective batch size changes → resulting weights differ.

---

## Per-Stage Details

### P1 — base SFT (custom trainer)

| Item | Value |
|---|---|
| Input | `data_p1/{train,val}.jsonl` |
| Trainer | `scripts/train_p1.py` (custom) |
| Base | `Qwen/Qwen2-VL-2B-Instruct` |
| Output | `lora_p1` → merge → `merged_p1` |
| Loss | token CE (sft single-prompt classification) |
| Data | 60k images, fake/real 50:50 |

### P2 — multi-prompt base (custom + aux BCE)

| Item | Value |
|---|---|
| Input | `data_p2/{train,val}.jsonl` |
| Trainer | `scripts/train_p2.py` (custom + auxiliary BCE) |
| Base | `merged_p1` |
| Output | `lora_p2` → merge → `merged_p2` |
| Loss | **token CE + overall BCE + criterion BCE** (3 losses combined) |
| Data | 10k images × 4 entry = 36k entries (multi-prompt) |

### P3 — image+evidence LoRA (LLaMA-Factory)

| Item | Value |
|---|---|
| Input | `data_p3/{train,val}.jsonl` (via LLaMA-Factory dataset_info.json alias) |
| Trainer | `llamafactory-cli train --config train_p3.yaml` |
| Base | `merged_p2` |
| Output | `lora_p3` → merge → `merged_p3` |
| Hyperparam | batch 4, grad_accum 1, template `qwen2_vl` |
| Data | 9k entries (50:50) |

### P4 warmup — text-only synthesis stage 1 (LLaMA-Factory)

| Item | Value |
|---|---|
| Input | `data_p4/{train,val}.jsonl` (text-only, 75:25 fake) |
| Trainer | `llamafactory-cli train --config train_p4_warmup.yaml` |
| Base | `merged_p3` |
| Output | `lora_p4_warmup` → merge → `merged_p4_warmup` |
| Hyperparam | batch 4, grad_accum 8, **effective batch 32, lr 5e-5, epoch 0.5** |
| Data | 8k entries (75:25 fake) |

### P4 final — text-only synthesis stage 2 (after warmup) (LLaMA-Factory)

| Item | Value |
|---|---|
| Input | same `data_p4/...` (same data as warmup) |
| Trainer | `llamafactory-cli train --config train_p4.yaml` |
| Base | `merged_p4_warmup` |
| Output | `lora_p4` → merge → **`merged_p4`** ⭐ (= **FINAL_RESULTS**) |
| Hyperparam | batch 4, grad_accum 8, **effective batch 32, lr 1e-5, epoch 0.25** (smaller lr/epoch than warmup for fine adjustment) |

→ **`merged_p4`** is the final output. Used as the input model for quantization (`quantize_en.md`).

---

## Dependencies

### Models (HuggingFace hub)
- `Qwen/Qwen2-VL-2B-Instruct` — base model (snapshot revision `895c3a49bc3fa70a340399125c650a463535e71c` recommended for reproducibility)

### Python packages (see `environment_en.md` for details)
- `transformers`, `peft`, `accelerate`, `torch`
- `llamafactory-cli` (LLaMA-Factory v0.9.1, editable install)

### LLaMA-Factory
- editable install location: `${PROJECT_ROOT}/LLaMA-Factory/` or `LLaMA-Factory-qwen25vl/`
- dataset registry: `${PROJECT_ROOT}/LLaMA-Factory-qwen25vl/data/dataset_info.json`
- `prepare_runtime.py` auto-patches P3/P4 dataset aliases

### Input data
- bundled `datasets/data_p1~p4` — output of `annotations_en.md`
- raw image dependencies (needed only when reproducing annotation): `data/images/`, COCO, ImageNet, ARForensics, SynthScars

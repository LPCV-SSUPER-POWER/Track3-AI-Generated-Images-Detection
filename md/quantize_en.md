# Quantize — `FINAL_RESULTS` Quantization Guide

Procedure for receiving the output of `train_en.md` (`FINAL_RESULTS` = `merged_p4`) and performing **AIMET W4A16 quantization + QNN binary export + submission zip packaging**. Output: `submit/{exp_name}.zip` (~2.62 GB).

Requires A100 + AIMET v2 (qairt) environment. Annotation/train can run on a generic ML environment, but quantize requires a machine with the **Qualcomm AI Stack** (`/opt/qcom/aistack/qairt/`) installed.

---

## `quantize/` Directory Structure

```
26LPCV_SSUPERPOWER_2nd/quantize/
│
├── run_parallel_quantize_pyfiles.sh       quantize wrapper — 2 GPUs (VEG+LLM concurrent)  ~62 min
├── run_sequential_quantize_pyfiles.sh     quantize wrapper — 1 GPU (VEG → LLM sequential) ~86 min
├── run_cosine.sh                          (optional) FP vs INT8 cosine similarity
├── verify_zip.py                          5-step submit zip integrity check
│
├── scripts/
│   └── package_submission.py              zip packaging (auto-called by wrapper)
│
├── inference/                             cosine measurement code (called by run_cosine.sh)
│   ├── llm_inout.py                       Step 1 — generate FP32 reference output
│   ├── inference_multi.py                 Step 2 — AIHub cloud (Snapdragon QRD) inference
│   └── contestant_uploads/inputs.json     submission metadata (bundled in zip)
│
└── py_files/                              vanilla AIMET quantization code (985 MB)
    ├── Example1A/                         VEG (Vision Encoder Graph) AIMET quantize
    │   ├── run_veg.py                     (19 KB)
    │   └── config/veg_config.json
    ├── Example1B/                         LLM (Language Model) AIMET quantize
    │   ├── run_llm.py                     (24 KB)
    │   └── config/
    │       ├── nb_config_tang.yml
    │       └── mixed_precision_config/qwen2_w4a16_gqa.json
    ├── Example2A/                         QNN VEG export (Qualcomm tools)
    │   └── host_linux/run_qnn_veg.py
    ├── Example2B/                         QNN LLM export (Qualcomm tools)
    │   └── host_linux/run_qnn_llm.py
    └── local_data/
        └── llava_v1_5_mix665k.json        LLM calibration JSON (external download, 1 GB)
```

`models/`, `results/`, `submit/` directories are auto-created by the wrapper at runtime (relative to `BUNDLE_ROOT` = `26LPCV_SSUPERPOWER_2nd/` parent).

---

## Overall Flow

```
output of train_en.md
   merged_p4 (= FINAL_RESULTS)
        ↓
   $BUNDLE_ROOT/models/{exp_name}_merged_stage2/   ← Step 1 placement
        ↓
   4 quantize sub-stages
   ├── Example1A — VEG AIMET     ~6 min   (GPU0)
   ├── Example2A — QNN VEG       ~15 min  (GPU0, uses Example1A output)
   ├── Example1B — LLM AIMET     ~28 min  (GPU1, when parallel)
   └── Example2B — QNN LLM       ~12 min  (GPU1, uses Example1B output)
        ↓
   Packaging (package_submission.py)
        ↓
   Verification (verify_zip.py — 5 steps)
        ↓
   $BUNDLE_ROOT/submit/{exp_name}.zip ⭐  (~2.62 GB, submission file)
```

Parallel wrapper (2 GPUs) ~62 min, sequential (1 GPU) ~86 min.

---

**Terminology Guide**

| Notation | Meaning |
|---|---|
| **Step 1 / Step 2 / Step 3** | Major stages of the quantize pipeline (overall flow of this doc) |
| **Example1A / 1B / 2A / 2B** | 4 sub-stage codes of the AIMET vanilla py_files |
| **VEG** | Vision Encoder Graph (image input → embedding) |
| **LLM** | Language Model (LM part of Qwen2-VL-2B) |
| **`{exp_name}`** | Quantization experiment name (free, e.g. `qwen2_FINAL_RESULTS`) |
| **`BUNDLE_ROOT`** | Bundle's root path (`26LPCV_SSUPERPOWER_2nd/`). Auto-detected by wrapper |

---

## Step 1 — Prepare Quantization Input Model

### 1-1. Place FP model

Place `merged_p4` (output of P1→P4 chain in `train_en.md`) under `quantize/../models/{exp_name}_merged_stage2/`.

```bash
EXP=qwen2_FINAL_RESULTS
BUNDLE_ROOT=/path/to/26LPCV_SSUPERPOWER_2nd
mkdir -p $BUNDLE_ROOT/models

# Copy (or symlink)
cp -r {train_runtime}/models/merged_p4 $BUNDLE_ROOT/models/${EXP}_merged_stage2
# ln -s {train_runtime}/models/merged_p4 $BUNDLE_ROOT/models/${EXP}_merged_stage2
```

> `{exp_name}` naming is free. Output directory must follow the pattern `{exp_name}_merged_stage2` for the wrapper to recognize it.

### 1-2. 11 files + md5 verification

```
${EXP}_merged_stage2/
├── added_tokens.json
├── chat_template.json
├── config.json
├── generation_config.json
├── merges.txt
├── model.safetensors        (4.4 GB, Qwen2-VL-2B bf16)
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer.json           (11.4 MB; 8.8 MB means broken)
├── tokenizer_config.json
└── vocab.json
```

```bash
cd $BUNDLE_ROOT/models/${EXP}_merged_stage2 && md5sum *
```

All 11 files must match the original md5 to proceed with quantization.

---

## Step 2 — AIMET W4A16 Quantization (4 sub-stages)

| Sub-stage | Code | Role | GPU memory | Time |
|---|---|---|---|---|
| **Example1A** | `py_files/Example1A/run_veg.py` | VEG AIMET quantize (FP → INT8 sim) | ~25 GB | ~6 min |
| **Example2A** | `py_files/Example2A/host_linux/run_qnn_veg.py` | QNN VEG export (Qualcomm tools) | (CPU) | ~15 min |
| **Example1B** | `py_files/Example1B/run_llm.py` | LLM AIMET quantize (W4A16) | ~20 GB | ~28 min |
| **Example2B** | `py_files/Example2B/host_linux/run_qnn_llm.py` | QNN LLM export (binary serialize) | (CPU) | ~12 min |

### 2-A. Parallel (recommended) — GPU0=VEG / GPU1=LLM concurrent — ~62 min

```bash
cd $BUNDLE_ROOT/quantize
EXP=qwen2_FINAL_RESULTS
LOG=$BUNDLE_ROOT/results/${EXP}_parallel.log
mkdir -p $BUNDLE_ROOT/results

nohup bash run_parallel_quantize_pyfiles.sh $EXP 0 1 > $LOG 2>&1 &
```

VEG chain (1A → 2A) on GPU0 + LLM chain (1B → 2B) on GPU1 run concurrently. After both chains complete: packaging + verification.

### 2-B. Sequential — 1 GPU — ~86 min

```bash
nohup bash run_sequential_quantize_pyfiles.sh $EXP 0 > $LOG 2>&1 &   # GPU0 only
```

VEG chain → LLM chain → packaging in sequence. Allows quantizing other models concurrently on different GPUs.

### Check progress

```bash
grep -E '===' $LOG
```
```
=== Example1A (VEG) ===     ~6 min
=== Example2A (QNN VEG) === ~15 min
=== Example1B (LLM) ===     ~28 min
=== Example2B (QNN LLM) === ~12 min
=== Packaging ===           ~5 sec
=== Verification ===        ~2 sec
=== ALL DONE ===
```

`ALL CHECKS PASSED` + `submit/${EXP}.zip` generated → quantization succeeded.

---

## Step 3 — Zip Packaging + Verification

Wrapper auto:
1. `scripts/package_submission.py` → `submit/${EXP}.zip` with 8 files + `SSUPER POWER/` prefix
2. `verify_zip.py` 5-step check

### 5-step check (verify_zip.py)
1. zip file listing (8 files)
2. required file existence
3. original vs zip size match
4. inputs.json contents
5. zip CRC full check

### Manual verification (optional)

```bash
ZIP=$BUNDLE_ROOT/submit/${EXP}.zip
unzip -l $ZIP                    # 8 file listing (~2.62 GB)
unzip -t $ZIP                    # CRC full check
md5sum $ZIP
```

### 8 submission files

| Path (in zip) | Size | Description |
|---|---|---|
| `SSUPER POWER/ar128-ar1-cl2048/weight_sharing_model_1_of_1.serialized.bin` | 894 MB | LLM weight |
| `SSUPER POWER/embedding_weights_151936x1536.raw` | 890 MB | Embedding |
| `SSUPER POWER/serialized_binaries/veg.serialized.bin` | ~695 MB | VEG |
| `SSUPER POWER/tokenizer.json` | 11 MB | Tokenizer |
| `SSUPER POWER/mask.raw` | 3 MB | Attention mask |
| `SSUPER POWER/position_ids_cos.raw` | 138 KB | RoPE cos |
| `SSUPER POWER/position_ids_sin.raw` | 138 KB | RoPE sin |
| `SSUPER POWER/inputs.json` | 1.7 KB | Submission metadata |

---

## (Optional) Cosine Similarity Measurement

After quantization, measure FP32 vs INT8 output similarity (~19 min). Quantitative quantization-loss evaluation.

### Run
```bash
cd $BUNDLE_ROOT/quantize
EXP=qwen2_FINAL_RESULTS
LOG=$BUNDLE_ROOT/results/${EXP}_cosine.log
CUDA_VISIBLE_DEVICES=0 nohup bash run_cosine.sh $EXP > $LOG 2>&1 &
```

### Stages
| Step | Code | Time | Description |
|---|---|---|---|
| 1 | `inference/llm_inout.py` | ~4 min | Generate FP32 reference output (10 batch × 10 token), uses GPU |
| 2 | `inference/inference_multi.py` | ~15 min | Snapdragon 8 Elite QRD (AIHub cloud) inference |
| 3 | cosine calculation | ~2 sec | FP vs QNN cosine similarity |

### Result interpretation

```
cos_avg=0.938304  per_token=[0.961, 0.925, 0.966, ...]
```

| `cos_avg` range | Verdict |
|---|---|
| ≥ 0.78 | ✅ safe |
| 0.75 ~ 0.78 | ⚠️ risky (single-outlier possible) |
| < 0.75 | ❌ risky (but vanilla `py_files` + LLaVA generic combo can still pass contest) |

> **cos↑ ≠ contest↑**. Even if `best_calib100` recovers cos, the contest score stays the same or drops. **Recommend submitting with vanilla py_files (LLaVA generic).**

---

## (Optional) py_files Variants (block_size, mixed precision)

rsync from vanilla `py_files` excluding caches:

```bash
cd $BUNDLE_ROOT/quantize
rsync -a \
  --exclude='Example2A/host_linux/exports' \
  --exclude='Example2B/host_linux/assets' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pkl_memoize_py3' \
  py_files/ py_files_a/
```

### Common modifications
- **block_size**: edit `py_files_a/Example1B/run_llm.py` → `BLOCK_QUANT_SIZE = 64` (16/32/64)
- **mixed precision**: edit `py_files_a/Example1B/config/mixed_precision_config/qwen2_w4a16_gqa.json` (e.g. add k/q/v/o proj W8)

### Variant wrapper
```bash
cp run_parallel_quantize_pyfiles.sh run_parallel_quantize_pyfiles_a.sh
sed -i "s|/py_files/|/py_files_a/|g" run_parallel_quantize_pyfiles_a.sh
chmod +x run_parallel_quantize_pyfiles_a.sh
```

---

## Dependencies

### Environment (see `environment_en.md` for details)
| Component | Location / Note |
|---|---|
| AIMET Python env | conda env `26lpcv` (A100, torch 1.13.1, AIMET 1.34) — `requirements/26lpcv_aimet_requirements.txt` |
| QNN Python env | conda env `26lpcv_qnn` (A100) — `requirements/26lpcv_qnn_requirements.txt` |
| AIMET v2 (qairt) | `/opt/qcom/aistack/qairt/2.31.0.250130/` — Qualcomm AI Stack (separate download) |
| AIMET pro 1.34 wheel | Aimet/AimetCommon/AimetTorch — separate wheel install (see `environment_en.md`) |

Scripts auto-set PATH/LD_LIBRARY_PATH — no extra activation needed.

### Input model
- `merged_p4` (= `FINAL_RESULTS`), output of `train_en.md`
- 11 files + md5 verification (Step 1)

### Calibration data
- **LLM**: `quantize/py_files/local_data/llava_v1_5_mix665k.json` — **external download** (HuggingFace `liuhaotian/LLaVA-Instruct-150K`, 1 GB)
- **VEG**: COCO train2017 100 images (external, e.g. `<COCO_CALIB_ROOT>/` or download)

---

## Quick Start (Summary)

```bash
# 0. Variables
EXP=qwen2_FINAL_RESULTS
BUNDLE_ROOT=/path/to/26LPCV_SSUPERPOWER_2nd

# 1. Place input + md5
cp -r {train_runtime}/models/merged_p4 $BUNDLE_ROOT/models/${EXP}_merged_stage2
cd $BUNDLE_ROOT/models/${EXP}_merged_stage2 && md5sum *

# 2. Parallel quantize (~62 min)
cd $BUNDLE_ROOT/quantize
nohup bash run_parallel_quantize_pyfiles.sh $EXP 0 1 \
  > $BUNDLE_ROOT/results/${EXP}_parallel.log 2>&1 &

# 3. Check progress
grep -E '===' $BUNDLE_ROOT/results/${EXP}_parallel.log

# 4. After completion, verify zip
ls -la $BUNDLE_ROOT/submit/${EXP}.zip
unzip -t $BUNDLE_ROOT/submit/${EXP}.zip

# 5. (Optional) cosine measurement
CUDA_VISIBLE_DEVICES=0 nohup bash run_cosine.sh $EXP \
  > $BUNDLE_ROOT/results/${EXP}_cosine.log 2>&1 &
```

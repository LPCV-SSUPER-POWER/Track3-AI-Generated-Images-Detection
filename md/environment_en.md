# Environment — Environments / External Models / Data / Variables Guide

Summary of **4 conda environments + external models + external data + environment variables** used by the bundle's 3 domains (`annotations_en.md`, `train_en.md`, `quantize_en.md`).

---

## 4 conda environments

| Env | Domain | Machine | Python | requirements |
|---|---|---|---|---|
| **`26lpcv_annotate`** | annotation | training machine | 3.10 | [`requirements/26lpcv_annotate_requirements.txt`](../requirements/26lpcv_annotate_requirements.txt) |
| **`26lpcv`** (training) | train | training machine | 3.10 | [`requirements/26lpcv_requirements.txt`](../requirements/26lpcv_requirements.txt) |
| **`26lpcv`** (quantize machine) | quantize (AIMET) | quantize machine (GPU 80GB) | 3.10 | [`requirements/26lpcv_aimet_requirements.txt`](../requirements/26lpcv_aimet_requirements.txt) |
| **`26lpcv_qnn`** | quantize (QNN) | quantize machine (GPU 80GB) | 3.10 | [`requirements/26lpcv_qnn_requirements.txt`](../requirements/26lpcv_qnn_requirements.txt) |

> training-side `26lpcv` and quantize-side `26lpcv` share the **same name but have different packages** (e.g. torch version). Quantization uses torch 1.13.1 (quantize machine, AIMET pro 1.34 compatible), train uses newer torch 2.10 (training).

---

## Domain-wise Flow

```
[annotation domain]                          [train domain]                         [quantize domain, quantize machine]
  Qwen2.5-VL-7B inference                      Qwen2-VL-2B + LLaMA-Factory             AIMET W4A16 + QNN export
  └── 26lpcv_annotate                          └── 26lpcv (training)                     ├── 26lpcv (quantize machine)   — Example1A/1B + cosine
                                                                                       └── 26lpcv_qnn       — Example2A/2B
       ↓ output                                     ↓ output                               ↓ output
  datasets/data_p1~4/{train,val}.jsonl       merged_p4 (= FINAL_RESULTS)             submit/{exp_name}.zip
```

---

## Env 1 — `26lpcv_annotate` (annotation domain)

**Key packages**:
- `transformers==5.5.3` (Qwen2.5-VL support)
- `torch==2.11.0`
- `qwen-vl-utils==0.0.14`
- `accelerate==1.13.0`
- `pillow==12.2.0`, `tokenizers==0.22.2`

**Install**:
```bash
conda create -n 26lpcv_annotate python=3.10 -y
conda activate 26lpcv_annotate
pip install -r requirements/26lpcv_annotate_requirements.txt
```

**Usage locations**:
- `annotation/inference/annotate.py` (Step 1 — Qwen2.5-VL-7B inference)
- `annotation/manifest_split/build_manifest.py`, `build_stage_split.py` (Step 2)
- `annotation/jsonl_build/build_p1_jsonl.py`, `build_p234_jsonl.py`, `lib_*.py` (Step 3)

---

## Env 2 — `26lpcv` (train domain, training machine)

**Key packages**:
- `torch==2.10.0+cu128`
- `transformers==4.46.1`
- `peft==0.12.0`
- `accelerate==1.0.1`
- `llamafactory==0.9.1` (editable install from github, P3/P4 training)
- `qai-hub==0.46.0`, `qwen-vl-utils==0.0.14`

**Install**:
```bash
conda create -n 26lpcv python=3.10 -y
conda activate 26lpcv
pip install -r requirements/26lpcv_requirements.txt
```

**Usage locations**:
- `train/scripts/train_p1.py`, `train_p2.py` (custom trainer)
- `train/scripts/merge_p1.py`, `merge_p2.py` (custom merge)
- `train/scripts/prepare_runtime.py`, `check_inputs.py`, `fix_preprocessor.py`
- LLaMA-Factory CLI (P3/P4 train + export — `<conda env 26lpcv>/bin/llamafactory-cli`)

---

## Env 3 — `26lpcv` (quantize AIMET, quantize machine)

**Key packages**:
- `torch==1.13.1` (1.13 for AIMET 1.34 compatibility)
- `transformers==4.46.1`
- `Aimet==1.34.0.0.207.0.44+torch.gpu.pt113` ⭐ **separate wheel download required**
- `AimetCommon`, `AimetTorch` (same version)
- `onnx==1.14.1`, `onnxruntime==1.15.1`, `onnxsim==0.6.2`
- `qai-hub==0.47.0` (cosine measurement)
- `llamafactory==0.9.1`, `peft==0.12.0`

**Install**:
```bash
# On quantize machine
conda create -n 26lpcv python=3.10 -y
conda activate 26lpcv
pip install -r requirements/26lpcv_aimet_requirements.txt

# Install AIMET wheel separately — download from Qualcomm AI Stack:
#   https://github.com/quic/aimet → release 1.34.0
# After download:
pip install /path/to/Aimet-1.34.0.0.207.0.44+torch.gpu.pt113-cp310-cp310-linux_x86_64.whl
pip install /path/to/AimetCommon-1.34.0.0.207.0.44+torch.gpu.pt113-cp310-cp310-linux_x86_64.whl
pip install /path/to/AimetTorch-1.34.0.0.207.0.44+torch.gpu.pt113-cp310-cp310-linux_x86_64.whl
```

**Usage locations** (quantize machine only):
- `quantize/py_files/Example1A/run_veg.py` (VEG AIMET quantize)
- `quantize/py_files/Example1B/run_llm.py` (LLM AIMET quantize)
- `quantize/scripts/package_submission.py`
- `quantize/verify_zip.py`
- `quantize/inference/llm_inout.py`, `inference_multi.py` (cosine measurement)

---

## Env 4 — `26lpcv_qnn` (quantize QNN, quantize machine)

**Key packages**:
- `torch==1.13.1`
- `transformers==4.46.1`
- `Aimet`, `AimetCommon`, `AimetTorch` (same 1.34, **separate wheel download**)
- `onnx==1.14.1`, `onnxruntime==1.15.1`
- `protobuf==3.20.2`

**Install**:
```bash
# On quantize machine
conda create -n 26lpcv_qnn python=3.10 -y
conda activate 26lpcv_qnn
pip install -r requirements/26lpcv_qnn_requirements.txt

# Install AIMET wheel separately (same wheels as Env 3 above)
pip install /path/to/Aimet*.whl /path/to/AimetCommon*.whl /path/to/AimetTorch*.whl
```

**Usage locations** (quantize machine only):
- `quantize/py_files/Example2A/host_linux/run_qnn_veg.py` (QNN VEG export)
- `quantize/py_files/Example2B/host_linux/run_qnn_llm.py` (QNN LLM export)

---

## Qualcomm AI Stack (qairt) Install — quantize domain only

```bash
# quantize machine
# Download qairt 2.31.0.250130 from Qualcomm Developer Network
# Install location: /opt/qcom/aistack/qairt/2.31.0.250130/
```

`quantize/run_*.sh` automatically sets `PATH` / `PYTHONPATH` / `LD_LIBRARY_PATH`:
```bash
PYTHONPATH=/opt/qcom/aistack/qairt/2.31.0.250130/lib/python
LD_LIBRARY_PATH=/opt/qcom/aistack/qairt/2.31.0.250130/lib/x86_64-linux-clang
              :{conda env 26lpcv}/lib
              :{conda env 26lpcv_qnn}/lib
```

---

## External Models (HuggingFace)

| Model | Use | dtype | Used in |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | annotation (Step 1 inference) | fp16 | 26lpcv_annotate |
| `Qwen/Qwen2-VL-2B-Instruct` | train P1 base + tokenizer (token fit) | bf16 | 26lpcv (training), 26lpcv_annotate |

**Download**:
```bash
# Auto-download from Hugging Face hub (transformers + huggingface_hub)
# Or pre-download and specify path:
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir <path>
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir <path>
```

`annotate.py --model_path` accepts a local path (default: `Qwen/Qwen2.5-VL-7B-Instruct`).

---

## External Data (Image Sets)

Image inputs for annotation Step 1 + train P1/P2/P3. **Not bundled in this repo, download separately**.

| Dataset | Use | Download |
|---|---|---|
| **ImageNet train** | annotation + train (real images) | https://www.image-net.org/ or academic mirror |
| **COCO train2017** | annotation + train (real images) | https://cocodataset.org/#download |
| **ADM, BigGAN, SID** | annotation + train (fake/real) | Each paper/model release. Place under `data/images/{adm,biggan,sid_set}/` |
| **ARForensics** | annotation + train (fake) | https://github.com/ARForensics/ARForensics |
| **SynthScars** | annotation + train (fake) | Paper release (TBD) |

**Layout** (in your environment):
```
/{PROJECT_IMAGE_ROOT}/
├── data/images/
│   ├── adm/{fake,real}/
│   ├── biggan/{fake,real}/
│   └── sid_set/{fake,real}/
└── datasets/raw/
    ├── ARForensics/ARForensics/
    └── SynthScars/SynthScars/

/{SHARED_DATASETS_ROOT}/
├── ImageNet/train/
└── coco/train2017/

/{VEG_CALIB_ROOT}/                    (quantize calibration only, quantize machine)
└── coco/train2017/                   (100 images only, e.g. <COCO_TRAIN2017_ROOT>/)
```

---

## Environment Variable Mapping

### annotation domain (`annotate.py` argparse)
| Argument | Description |
|---|---|
| `--model_path` | Qwen2.5-VL-7B-Instruct local path or HF hub ID |
| `--list_json` | input image list JSON |
| `--output_dir` | annotation output directory |
| `--batch_size` | inference batch (default 32) |
| `--device` | `cuda` (default) |

### train domain (`train/run.sh`)
| Variable | Default | Description |
|---|---|---|
| `PYTHON` | `<conda env 26lpcv>/bin/python` | 26lpcv conda Python |
| `LLAMA` | `<conda env 26lpcv>/bin/llamafactory-cli` | LLaMA-Factory CLI |
| `PROJECT_ROOT` | (user environment) | image set root (containing `data/images/`, `datasets/raw/`) |
| `PROJECT_IMAGE_ROOT` | `${PROJECT_ROOT}` | for jsonl image path redirect (prepare_runtime) |
| `PROCESSED_DATA_ROOT` | `${DATA_ROOT}` | root of `datasets/data_p1~4/` |
| `SHARED_DATASETS_ROOT` | `<SHARED_DATASETS_ROOT>` | shared dataset root for ImageNet, COCO, etc. |
| `RUN_ROOT` | `${TRAIN_ROOT}/runs/train` | directory for training outputs (lora_p?, merged_p?) |
| `P1_GPU` ~ `MERGE_GPU` | `0` | GPU index per stage |
| `EXACT_REPRODUCTION` | `1` | enforce single-GPU P3/P4 (preserve effective batch size) |

### quantize domain (`quantize/run_*.sh`)
| Variable | Description |
|---|---|
| `BUNDLE_ROOT` | auto-detected (parent of script location). `quantize/../` |
| `EXP` | quantize experiment name (passed as arg, e.g. `qwen2_FINAL_RESULTS`) |
| `GPU_VEG`, `GPU_LLM` | GPU index for parallel wrapper |
| `PYTHON_LPCV` | 26lpcv conda Python (AIMET) |
| `PYTHON_QNN` | 26lpcv_qnn conda Python (QNN) |
| `PYTHONPATH` | qairt lib (auto-set) |
| `LD_LIBRARY_PATH` | qairt + 26lpcv + 26lpcv_qnn lib (auto-set) |
| `CUDA_VISIBLE_DEVICES` | per sub-stage GPU |

---

## Calibration Data (quantize domain only)

| Data | Use | Location |
|---|---|---|
| `llava_v1_5_mix665k.json` | LLM AIMET calibration | **external download** (GitHub 100MB limit, gitignored) — HuggingFace [`liuhaotian/LLaVA-Instruct-150K`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) → place at `quantize/py_files/local_data/llava_v1_5_mix665k.json` (1 GB) |
| COCO train2017 100 imgs | VEG AIMET calibration | external, e.g. `<COCO_TRAIN2017_ROOT>/` (your environment) |

The image paths in LLaVA JSON are relative like `coco/train2017/...`. The `image_dataset_path` config must point to that root (e.g. `<COCO_CALIB_ROOT_PARENT>`).

---

## Dependency Summary at a Glance

| Domain | Env | External model | External data | qairt SDK |
|---|---|---|---|---|
| annotation | 26lpcv_annotate | Qwen2.5-VL-7B-Instruct, Qwen2-VL-2B-Instruct | image set | — |
| train | 26lpcv (training) | Qwen2-VL-2B-Instruct | image set + datasets/data_p?/ | — |
| quantize (AIMET) | 26lpcv (quantize machine) | merged_p4 (FINAL_RESULTS) | LLaVA JSON (bundled) + COCO 100 | qairt 2.31.0.250130 |
| quantize (QNN) | 26lpcv_qnn (quantize machine) | merged_p4 | (same as above) | qairt 2.31.0.250130 |

---

## Quick Start (Full Pipeline)

```bash
# [training machine] annotation
conda activate 26lpcv_annotate
python annotation/inference/annotate.py --list_json {your_list}.json \
  --output_dir annotations/0421_data --batch_size 32

# [training machine] train
conda activate 26lpcv
PROJECT_IMAGE_ROOT=/path/to/track3_images \
SHARED_DATASETS_ROOT=/path/to/datasets \
./train/run.sh

# [quantize machine] quantize
ssh -p <PORT> <USER>@<quantize machine_HOST>
cp -r {merged_p4} ./models/qwen2_FINAL_RESULTS_merged_stage2
conda activate 26lpcv
nohup bash quantize/run_parallel_quantize_pyfiles.sh qwen2_FINAL_RESULTS 0 1 \
  > results/qwen2_FINAL_RESULTS_parallel.log 2>&1 &
```

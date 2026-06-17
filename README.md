# SSUPER-AIGID

> **S**SUPER POWER's **AI-G**enerated **I**mage **D**etector
> — LPCVC 2026 Track 3 · 🥈 **2nd Place** · Team **SSUPER POWER**
> 
> Qwen2-VL-2B fine-tuning + AIMET W4A16 quantization pipeline targeting Snapdragon 8 Elite QRD.

---

## Competition

**LPCVC** (Low Power Computer Vision Challenge) **2026 — Track 3**
- Task: AI-generated image detection on edge devices
- Hardware: Snapdragon 8 Elite QRD (evaluated via Qualcomm AI Hub cloud)
- Submission: ~2.62 GB QNN binary (8 files + `SSUPER POWER/` prefix)
- Page: [2026LPCVC/winners](https://lpcv.ai/2026LPCVC/winners/)
- Venue: ECV Workshop @ CVPR 2026 — Denver, CO · June 3, 2026

## Result

🥈 **2nd Place** — Team **SSUPER POWER** · Model **SSUPER-AIGID**

🤗 [**Model weights**](https://huggingface.co/Dayoung-space/SSUPER-AIGID)

| Metric | Value |
|---|---|
| **Model** | **SSUPER-AIGID** (Qwen2-VL-2B, W4A16) |
| **TPS** (tokens per second) | **31.21** |
| **Score** | **0.72** |

### Team

- **[Dayoung Kil](https://github.com/Dayoung-Kil)**
- **[Doeon Kim](https://github.com/kimdoeon)**
- **[Junyoon Lee](https://github.com/jungyoon-lee)**

---

## Approach — 3-Stage Pipeline

**SSUPER-AIGID** is built through a 3-stage pipeline:

```
raw images
     ↓
[1. annotation]  Qwen2.5-VL-7B  →  datasets/data_p{1,2,3,4}/{train,val}.jsonl
     ↓
[2. train]       Qwen2-VL-2B  P1 → P2 → P3 → P4_warmup → P4_final  →  merged_p4 (= FINAL_RESULTS)
     ↓
[3. quantize]    AIMET W4A16 + QNN binary export  →  submit/{exp_name}.zip ⭐
```

| Stage | Model / Tool | Output |
|---|---|---|
| **1. Annotation** | Qwen2.5-VL-7B (HF) — multi-stage inference (4 questions per image) | 89,263 raw annotation JSON → 4 SFT splits (P1/P2/P3/P4) |
| **2. Train** | Qwen2-VL-2B + custom trainer (P1/P2) + LLaMA-Factory (P3/P4) | `merged_p4` = `FINAL_RESULTS` |
| **3. Quantize** | AIMET pro 1.34 (W4A16) + QNN binary export + zip packaging | `submit/{exp_name}.zip` (~2.62 GB) |

---

## Repository Structure

```
26LPCV_SSUPERPOWER_2nd/
├── README.md                            ← you are here
├── md/
│   ├── annotations_en.md  / annotations_kor.md     Stage 1: annotation guide
│   ├── train_en.md        / train_kor.md           Stage 2: train guide
│   ├── quantize_en.md     / quantize_kor.md        Stage 3: quantize guide
│   └── environment_en.md  / environment_kor.md     Environment / dependencies guide
│
├── annotation/                          Stage 1 code (Step 1/2/3)
│   ├── inference/annotate.py            Step 1: Qwen2.5-VL-7B annotation
│   ├── manifest_split/                  Step 2: master manifest + 4-split
│   ├── jsonl_build/                     Step 3: training jsonl builders
│   ├── prompts/a_step2.txt              a_step2 synthesis prompt
│   └── samples/                         20 raw annotation JSON examples
│
├── datasets/                            (197 MB)
│   ├── data_p1/{train,val}.jsonl        P1 input — 60k images, single-prompt SFT
│   ├── data_p2/{train,val}.jsonl        P2 input — 10k images × 4 entry, multi-prompt
│   ├── data_p3/{train,val}.jsonl        P3 input — image+evidence
│   └── data_p4/{train,val}.jsonl        P4 input — text-only synthesis
│
├── train/                               Stage 2 code
│   ├── configs/                         10 yaml (5 train + 5 merge)
│   ├── scripts/                         11 .py (train_p1/p2, dataset_p1/p2, model_p1/p2, merge_p1/p2, prepare_runtime, check_inputs, fix_preprocessor)
│   └── run.sh                           End-to-end P1→P4 launcher
│
├── quantize/                            Stage 3 code (985 MB)
│   ├── run_parallel_quantize_pyfiles.sh    GPU 2-way wrapper (~62 min)
│   ├── run_sequential_quantize_pyfiles.sh  GPU 1-way wrapper (~86 min)
│   ├── run_cosine.sh                       (optional) FP vs INT8 cosine
│   ├── verify_zip.py                       5-step submit zip integrity check
│   ├── scripts/package_submission.py
│   ├── inference/                          cosine measurement code
│   └── py_files/                           AIMET vanilla code (Example1A/1B/2A/2B + local_data)
│
└── requirements/
    ├── 26lpcv_annotate_requirements.txt    (annotation, training machine, transformers 5.5.3)
    ├── 26lpcv_requirements.txt             (train, training machine, torch 2.10)
    ├── 26lpcv_aimet_requirements.txt       (quantize AIMET, quantize machine, torch 1.13.1)
    └── 26lpcv_qnn_requirements.txt         (quantize QNN, quantize machine)
```

---

## Quick Start

### 1) Install environments (4 conda envs — see [environment_en.md](md/environment_en.md))

```bash
# Annotation env (training)
conda create -n 26lpcv_annotate python=3.10 -y
conda activate 26lpcv_annotate
pip install -r requirements/26lpcv_annotate_requirements.txt

# Train env (training)
conda create -n 26lpcv python=3.10 -y
conda activate 26lpcv
pip install -r requirements/26lpcv_requirements.txt

# Quantize envs (quantize machine, requires Qualcomm AI Stack qairt 2.31)
# ... (separate AIMET wheel install — see environment_en.md)
```

### 2) Download external models / data

- **Models**: HuggingFace auto-download for `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2-VL-2B-Instruct`
- **Data**: ImageNet, COCO train2017, ARForensics, SynthScars, ADM, BigGAN, SID — see [environment_en.md](md/environment_en.md)

### 3) Run pipeline

```bash
# === Stage 1: Annotation (training) ===
conda activate 26lpcv_annotate
python annotation/inference/annotate.py \
    --list_json {your_image_list}.json \
    --output_dir annotations/0421_data \
    --batch_size 32
# (See md/annotations_en.md for Step 2 manifest + Step 3 jsonl builders)

# === Stage 2: Train (training) ===
conda activate 26lpcv
PROJECT_ROOT=/path/to/track3_images \
SHARED_DATASETS_ROOT=/path/to/datasets \
./train/run.sh
# Output: merged_p4 (= FINAL_RESULTS)

# === Stage 3: Quantize (quantize machine) ===
conda activate 26lpcv
EXP=qwen2_FINAL_RESULTS
PYTHON_LPCV=$(which python) \
PYTHON_QNN=/path/to/conda/envs/26lpcv_qnn/bin/python \
CONDA_26LPCV_LIB=$(conda info --base)/envs/26lpcv/lib \
CONDA_26LPCV_QNN_LIB=$(conda info --base)/envs/26lpcv_qnn/lib \
./quantize/run_parallel_quantize_pyfiles.sh $EXP 0 1
# Output: submit/{exp_name}.zip ⭐
```

---

## Documentation

Detailed guides (English / Korean):

| English | Korean | Topic |
|---|---|---|
| [`md/annotations_en.md`](md/annotations_en.md) | [`md/annotations_kor.md`](md/annotations_kor.md) | Stage 1: Annotation pipeline (Step 1/2/3) |
| [`md/train_en.md`](md/train_en.md) | [`md/train_kor.md`](md/train_kor.md) | Stage 2: P1→P4 training chain |
| [`md/quantize_en.md`](md/quantize_en.md) | [`md/quantize_kor.md`](md/quantize_kor.md) | Stage 3: AIMET W4A16 + QNN binary export |
| [`md/environment_en.md`](md/environment_en.md) | [`md/environment_kor.md`](md/environment_kor.md) | 4 conda envs + external models/data + env vars |

---

## License

This repository is licensed under the **MIT License** (see [LICENSE](LICENSE)) — applies to first-party code authored by the team (`annotation/`, `train/`, `datasets/`, scripts, configs).

### Third-party Components

| Component | License |
|---|---|
| `quantize/py_files/Example*/G2G/MHA2SHA/` | Qualcomm Innovation Center (sample SDK code, see file headers) |
| `quantize/py_files/Example*/` (run_veg, run_llm, run_qnn_*) | Adapted from Qualcomm AIMET sample code |
| Qwen2.5-VL-7B-Instruct, Qwen2-VL-2B-Instruct | [Tongyi Qianwen LICENSE](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/LICENSE) (Alibaba) |
| LLaMA-Factory v0.9.1 | [Apache 2.0](https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE) |
| AIMET pro 1.34 | Qualcomm AI Hub (separate wheel download required) |

---

## Citation

If you use this code or build upon our approach, please cite:

```bibtex
@misc{ssuper_aigid_2026,
  title        = {SSUPER-AIGID: AI-Generated Image Detection on Edge Devices},
  author       = {Kil, Dayoung and Kim, Doeon and Lee, Junyoon},
  year         = {2026},
  howpublished = {ECV Workshop at CVPR 2026, Denver, CO},
  note         = {LPCVC 2026 Track 3, 2nd Place (Team SSUPER POWER) --- presented June 3, 2026},
  url          = {https://github.com/LPCV-SSUPER-POWER/Track3-AI-Generated-Images-Detection}
}
```

- **LPCVC 2026 — Track 3 (AI-Generated Image Detection on Edge Devices)** — Competition page: https://lpcv.ai/2026LPCVC/winners/

---

## Acknowledgments

- LPCVC 2026 organizers
- Qualcomm AI Engine Direct SDK (qairt 2.31), AIMET pro 1.34
- Alibaba Qwen team — Qwen2.5-VL-7B-Instruct, Qwen2-VL-2B-Instruct
- LLaMA-Factory v0.9.1
- All dataset providers — ImageNet, COCO, ARForensics, SynthScars, and others

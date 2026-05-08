# Environment — 환경 / 외부 모델 / 데이터 / 환경변수 가이드

bundle 의 3 영역 (`annotations_kor.md`, `train_kor.md`, `quantize_kor.md`) 이 사용하는 **conda 환경 4개 + 외부 모델 + 외부 데이터 + 환경변수** 정리.

---

## conda 환경 4개

| 환경 | 영역 | 머신 | Python | requirements |
|---|---|---|---|---|
| **`26lpcv_annotate`** | annotation | Leader (RTX 5090) | 3.10 | [`requirements/26lpcv_annotate_requirements.txt`](../requirements/26lpcv_annotate_requirements.txt) |
| **`26lpcv`** (Leader) | train | Leader (RTX 5090) | 3.10 | [`requirements/26lpcv_requirements.txt`](../requirements/26lpcv_requirements.txt) |
| **`26lpcv`** (A100) | quantize (AIMET) | A100 (80GB) | 3.10 | [`requirements/26lpcv_aimet_requirements.txt`](../requirements/26lpcv_aimet_requirements.txt) |
| **`26lpcv_qnn`** | quantize (QNN) | A100 (80GB) | 3.10 | [`requirements/26lpcv_qnn_requirements.txt`](../requirements/26lpcv_qnn_requirements.txt) |

> Leader 와 A100 의 `26lpcv` 는 **이름 동일 / 패키지 다름** (torch 버전 등). 양자화는 AIMET pro 1.34 와 호환되는 torch 1.13.1 사용 (A100), train 은 새 torch 2.10 (Leader).

---

## 영역별 사용 흐름

```
[annotation 영역]                            [train 영역]                           [quantize 영역, A100]
  Qwen2.5-VL-7B inference                      Qwen2-VL-2B + LLaMA-Factory             AIMET W4A16 + QNN export
  └── 26lpcv_annotate                          └── 26lpcv (Leader)                     ├── 26lpcv (A100)   — Example1A/1B + cosine
                                                                                       └── 26lpcv_qnn       — Example2A/2B
       ↓ 출력                                       ↓ 출력                                  ↓ 출력
  datasets/data_p1~4/{train,val}.jsonl       merged_p4 (= FINAL_RESULTS)             submit/{exp_name}.zip
```

---

## 환경 1 — `26lpcv_annotate` (annotation 영역)

**핵심 패키지**:
- `transformers==5.5.3` (Qwen2.5-VL 지원)
- `torch==2.11.0`
- `qwen-vl-utils==0.0.14`
- `accelerate==1.13.0`
- `pillow==12.2.0`, `tokenizers==0.22.2`

**설치**:
```bash
conda create -n 26lpcv_annotate python=3.10 -y
conda activate 26lpcv_annotate
pip install -r requirements/26lpcv_annotate_requirements.txt
```

**사용 위치**:
- `annotation/inference/annotate.py` (Step 1 — Qwen2.5-VL-7B inference)
- `annotation/manifest_split/build_manifest.py`, `build_stage_split.py` (Step 2)
- `annotation/jsonl_build/build_p1_jsonl.py`, `build_p234_jsonl.py`, `lib_*.py` (Step 3)

---

## 환경 2 — `26lpcv` (train 영역, Leader)

**핵심 패키지**:
- `torch==2.10.0+cu128`
- `transformers==4.46.1`
- `peft==0.12.0`
- `accelerate==1.0.1`
- `llamafactory==0.9.1` (editable install from github, P3/P4 학습)
- `qai-hub==0.46.0`, `qwen-vl-utils==0.0.14`

**설치**:
```bash
conda create -n 26lpcv python=3.10 -y
conda activate 26lpcv
pip install -r requirements/26lpcv_requirements.txt
```

**사용 위치**:
- `train/scripts/train_p1.py`, `train_p2.py` (custom trainer)
- `train/scripts/merge_p1.py`, `merge_p2.py` (custom merge)
- `train/scripts/prepare_runtime.py`, `check_inputs.py`, `fix_preprocessor.py`
- LLaMA-Factory CLI (P3/P4 train + export — `<conda env 26lpcv>/bin/llamafactory-cli`)

---

## 환경 3 — `26lpcv` (quantize AIMET, A100)

**핵심 패키지**:
- `torch==1.13.1` (AIMET 1.34 호환 위해 1.13)
- `transformers==4.46.1`
- `Aimet==1.34.0.0.207.0.44+torch.gpu.pt113` ⭐ **별도 wheel 다운로드 필요**
- `AimetCommon`, `AimetTorch` (동일 버전)
- `onnx==1.14.1`, `onnxruntime==1.15.1`, `onnxsim==0.6.2`
- `qai-hub==0.47.0` (cosine 측정)
- `llamafactory==0.9.1`, `peft==0.12.0`

**설치**:
```bash
# A100 머신에서
conda create -n 26lpcv python=3.10 -y
conda activate 26lpcv
pip install -r requirements/26lpcv_aimet_requirements.txt

# AIMET wheel 별도 설치 — Qualcomm AI Stack 에서 다운로드:
#   https://github.com/quic/aimet → release 1.34.0
# 다운로드 후:
pip install /path/to/Aimet-1.34.0.0.207.0.44+torch.gpu.pt113-cp310-cp310-linux_x86_64.whl
pip install /path/to/AimetCommon-1.34.0.0.207.0.44+torch.gpu.pt113-cp310-cp310-linux_x86_64.whl
pip install /path/to/AimetTorch-1.34.0.0.207.0.44+torch.gpu.pt113-cp310-cp310-linux_x86_64.whl
```

**사용 위치** (A100 만):
- `quantize/py_files/Example1A/run_veg.py` (VEG AIMET quantize)
- `quantize/py_files/Example1B/run_llm.py` (LLM AIMET quantize)
- `quantize/scripts/package_submission.py`
- `quantize/verify_zip.py`
- `quantize/inference/llm_inout.py`, `inference_multi.py` (cosine 측정)

---

## 환경 4 — `26lpcv_qnn` (quantize QNN, A100)

**핵심 패키지**:
- `torch==1.13.1`
- `transformers==4.46.1`
- `Aimet`, `AimetCommon`, `AimetTorch` (동일 1.34, **별도 wheel 다운로드**)
- `onnx==1.14.1`, `onnxruntime==1.15.1`
- `protobuf==3.20.2`

**설치**:
```bash
# A100 머신에서
conda create -n 26lpcv_qnn python=3.10 -y
conda activate 26lpcv_qnn
pip install -r requirements/26lpcv_qnn_requirements.txt

# AIMET wheel 별도 설치 (위 환경 3 과 동일 wheel)
pip install /path/to/Aimet*.whl /path/to/AimetCommon*.whl /path/to/AimetTorch*.whl
```

**사용 위치** (A100 만):
- `quantize/py_files/Example2A/host_linux/run_qnn_veg.py` (QNN VEG export)
- `quantize/py_files/Example2B/host_linux/run_qnn_llm.py` (QNN LLM export)

---

## Qualcomm AI Stack (qairt) 설치 — quantize 영역만

```bash
# A100 머신
# Qualcomm Developer Network 에서 qairt 2.31.0.250130 다운로드
# 설치 위치: /opt/qcom/aistack/qairt/2.31.0.250130/
```

`quantize/run_*.sh` 가 자동으로 `PATH` / `PYTHONPATH` / `LD_LIBRARY_PATH` 설정:
```bash
PYTHONPATH=/opt/qcom/aistack/qairt/2.31.0.250130/lib/python
LD_LIBRARY_PATH=/opt/qcom/aistack/qairt/2.31.0.250130/lib/x86_64-linux-clang
              :{conda env 26lpcv}/lib
              :{conda env 26lpcv_qnn}/lib
```

---

## 외부 모델 (HuggingFace)

| 모델 | 용도 | dtype | 사용 환경 |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | annotation (Step 1 inference) | fp16 | 26lpcv_annotate |
| `Qwen/Qwen2-VL-2B-Instruct` | train P1 base + tokenizer (token fit) | bf16 | 26lpcv (Leader), 26lpcv_annotate |

**다운로드**:
```bash
# Hugging Face hub 에서 자동 다운로드 (transformers + huggingface_hub)
# 또는 미리 받아놓고 path 지정:
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir <path>
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir <path>
```

`annotate.py --model_path` 인자로 local path 지정 가능 (default: `Qwen/Qwen2.5-VL-7B-Instruct`).

---

## 외부 데이터 (이미지 셋)

annotation Step 1 + train P1/P2/P3 의 이미지 입력. **bundle 에 미동봉, 다운로드 별도**.

| 데이터셋 | 용도 | 다운로드 |
|---|---|---|
| **ImageNet train** | annotation + train (real images) | https://www.image-net.org/ 또는 academic mirror |
| **COCO train2017** | annotation + train (real images) | https://cocodataset.org/#download |
| **ADM, BigGAN, SID** | annotation + train (fake/real) | 각 논문/모델 release. 사용자가 `data/images/{adm,biggan,sid_set}/` 에 배치 |
| **ARForensics** | annotation + train (fake) | https://github.com/ARForensics/ARForensics |
| **SynthScars** | annotation + train (fake) | 해당 논문 release |

**배치 구조** (사용자 환경에서):
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

/{VEG_CALIB_ROOT}/                    (양자화 calibration 만, A100)
└── coco/train2017/                   (100장만, e.g. <COCO_TRAIN2017_ROOT>/)
```

---

## 환경변수 매핑

### annotation 영역 (`annotate.py` argparse)
| 인자 | 설명 |
|---|---|
| `--model_path` | Qwen2.5-VL-7B-Instruct local path 또는 HF hub ID |
| `--list_json` | 입력 image list JSON |
| `--output_dir` | annotation 출력 디렉토리 |
| `--batch_size` | inference batch (default 32) |
| `--device` | `cuda` (default) |

### train 영역 (`train/run.sh`)
| 환경변수 | default | 설명 |
|---|---|---|
| `PYTHON` | `<conda env 26lpcv>/bin/python` | 26lpcv conda Python |
| `LLAMA` | `<conda env 26lpcv>/bin/llamafactory-cli` | LLaMA-Factory CLI |
| `PROJECT_ROOT` | (사용자 환경) | image set 의 root (`data/images/`, `datasets/raw/` 포함) |
| `PROJECT_IMAGE_ROOT` | `${PROJECT_ROOT}` | jsonl image path redirect 용 (prepare_runtime) |
| `PROCESSED_DATA_ROOT` | `${DATA_ROOT}` | `datasets/data_p1~4/` 의 root |
| `SHARED_DATASETS_ROOT` | `<SHARED_DATASETS_ROOT>` | ImageNet, COCO 등 공유 dataset root |
| `RUN_ROOT` | `${TRAIN_ROOT}/runs/train` | 학습 산출물 (lora_p?, merged_p?) 디렉토리 |
| `P1_GPU` ~ `MERGE_GPU` | `0` | 각 단계 GPU 인덱스 |
| `EXACT_REPRODUCTION` | `1` | single-GPU P3/P4 강제 (effective batch size 보존) |

### quantize 영역 (`quantize/run_*.sh`)
| 환경변수 | 설명 |
|---|---|
| `BUNDLE_ROOT` | 자동 감지 (script 위치 기준 parent). `quantize/../` |
| `EXP` | 양자화 실험 이름 (인자 전달, e.g. `qwen2_FINAL_RESULTS`) |
| `GPU_VEG`, `GPU_LLM` | 병렬 wrapper 의 GPU 인덱스 |
| `PYTHON_LPCV` | 26lpcv conda Python (AIMET) |
| `PYTHON_QNN` | 26lpcv_qnn conda Python (QNN) |
| `PYTHONPATH` | qairt lib (자동 설정) |
| `LD_LIBRARY_PATH` | qairt + 26lpcv + 26lpcv_qnn lib (자동 설정) |
| `CUDA_VISIBLE_DEVICES` | 각 sub-stage 의 GPU |

---

## Calibration data (quantize 영역만)

| 데이터 | 용도 | 위치 |
|---|---|---|
| `llava_v1_5_mix665k.json` | LLM AIMET calibration | **외부 다운로드** (GitHub 100MB 제한, .gitignore 됨) — HuggingFace [`liuhaotian/LLaVA-Instruct-150K`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) → `quantize/py_files/local_data/llava_v1_5_mix665k.json` (1 GB) |
| COCO train2017 100장 | VEG AIMET calibration | 외부, e.g. `<COCO_TRAIN2017_ROOT>/` (사용자 환경) |

LLaVA JSON 안 image path 는 `coco/train2017/...` 같이 상대경로. 학습 시 `image_dataset_path` 가 그 root (e.g. `<COCO_CALIB_ROOT_PARENT>`) 를 가리켜야.

---

## 의존성 한 눈에 보기

| 영역 | 환경 | 외부 모델 | 외부 데이터 | qairt SDK |
|---|---|---|---|---|
| annotation | 26lpcv_annotate | Qwen2.5-VL-7B-Instruct, Qwen2-VL-2B-Instruct | image set | — |
| train | 26lpcv (Leader) | Qwen2-VL-2B-Instruct | image set + datasets/data_p?/ | — |
| quantize (AIMET) | 26lpcv (A100) | merged_p4 (FINAL_RESULTS) | LLaVA JSON (동봉) + COCO 100장 | qairt 2.31.0.250130 |
| quantize (QNN) | 26lpcv_qnn (A100) | merged_p4 | (위와 동일) | qairt 2.31.0.250130 |

---

## 빠른 시작 (전체 파이프라인)

```bash
# [Leader] annotation
conda activate 26lpcv_annotate
python annotation/inference/annotate.py --list_json {your_list}.json \
  --output_dir annotations/0421_data --batch_size 32

# [Leader] train
conda activate 26lpcv
PROJECT_IMAGE_ROOT=/path/to/track3_images \
SHARED_DATASETS_ROOT=/path/to/datasets \
./train/run.sh

# [A100] quantize
ssh -p <PORT> <USER>@<A100_HOST>
cp -r {merged_p4} ./models/qwen2_FINAL_RESULTS_merged_stage2
conda activate 26lpcv
nohup bash quantize/run_parallel_quantize_pyfiles.sh qwen2_FINAL_RESULTS 0 1 \
  > results/qwen2_FINAL_RESULTS_parallel.log 2>&1 &
```

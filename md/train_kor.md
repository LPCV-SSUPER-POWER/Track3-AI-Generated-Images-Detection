# Train — `FINAL_RESULTS` 학습 가이드

`datasets/` 에 동봉된 4개 jsonl (P1/P2/P3/P4) 을 입력으로 받아 **P1→P4 학습 chain** 을 돌리는 절차. 

학습은 reproduce 가능. `datasets/` 가 이미 정리본에 동봉되어 있으니 사용자는 환경 + LLaMA-Factory 만 준비하면 즉시 학습 시작 가능.

---

## `train/` 디렉토리 구조

```
26LPCV_SSUPERPOWER_2nd/train/
│
├── configs/        (10 yaml — 각 P 의 train + merge config)
│   ├── train_p1.yaml        / merge_p1.yaml         P1 (custom SFT)
│   ├── train_p2.yaml        / merge_p2.yaml         P2 (custom + aux BCE)
│   ├── train_p3.yaml        / merge_p3.yaml         P3 (LLaMA-Factory)
│   ├── train_p4_warmup.yaml / merge_p4_warmup.yaml  P4 warmup — 1단계 적응 (LLaMA-Factory)
│   └── train_p4.yaml        / merge_p4.yaml         P4 — 2단계 정밀, final (LLaMA-Factory)
│
├── scripts/                     (11 py)
│   │── P1 학습 (custom SFT) ───────
│   ├── train_p1.py              메인 launcher (학습 entry point)
│   ├── dataset_p1.py            Dataset 클래스 (data_p1 jsonl 읽기)
│   ├── model_p1.py              모델 정의 (Qwen2-VL + LoRA + overall binary head)
│   ├── merge_p1.py              LoRA → base merge
│   │── P2 학습 (custom + aux BCE) ───────
│   ├── train_p2.py              메인 launcher (token CE + overall BCE + criterion BCE)
│   ├── dataset_p2.py            Dataset 클래스 (data_p2 jsonl 읽기)
│   ├── model_p2.py              모델 정의 (Qwen2-VL + LoRA + auxiliary heads)
│   ├── merge_p2.py              LoRA → base merge
│   │── 공통 도우미 ───────
│   ├── prepare_runtime.py    runtime 준비 (yaml 치환 + dataset_info.json patch)
│   ├── check_inputs.py       preflight (dataset/image 존재 검증)
│   └── fix_preprocessor.py   merge 후 preprocessor config 보정
│
└── run.sh          End-to-end P1→P4 launcher
```

---

## 전체 흐름

`annotations_kor.md` 의 결과물 (`datasets/data_p1~p4`) → train chain → `FINAL_RESULTS`.

### 입력 — 정리본 `datasets/`

| split | 형식 | jsonl |
|---|---|---|
| `data_p1/` | sft (단일 prompt) | `train.jsonl` (54,007), `val.jsonl` (5,993) |
| `data_p2/` | multi-prompt (4 entry/img) | `train.jsonl` (36,000), `val.jsonl` (4,000) |
| `data_p3/` | P3 image+evidence | `train.jsonl` (9,000), `val.jsonl` (1,000) |
| `data_p4/` | P4 text-only synthesis | `train.jsonl` (8,104), `val.jsonl` (896) |


### 학습 chain (5 stage, 10 step)

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
   merged_p4  ⭐  (= FINAL_RESULTS, 최종 산출물)
```

### 용어 표기 가이드

| 표기 | 의미 |
|---|---|
| **P1 / P2 / P3 / P4** | 학습 chain 의 phase (4 단계, P4 만 warmup + final 두 sub-step) |
| **train / merge** | 각 phase 의 LoRA 학습 + base 모델로 merge 하는 두 sub-step |
| **lora_p\*** | LoRA adapter 산출물 (merge 전 중간) |
| **merged_p\*** | base + LoRA merge 된 full model (다음 phase 의 base) |

---

## 사전 준비

### 1. Python 환경 + LLaMA-Factory

`environment_kor.md` 참조. 핵심:
- `26lpcv` conda env (Python 3.10, torch 1.13.1, transformers 4.46.1, peft 0.12.0, llamafactory 0.9.1)
- `llamafactory-cli` (`<conda env 26lpcv>/bin/llamafactory-cli`) — LLaMA-Factory v0.9.1 editable install 한 위치

### 2. Preflight check — `check_inputs.py`

학습 시작 전 dataset/image 모두 존재하는지 검증.

```bash
cd train/
./scripts/check_inputs.py \
  --processed-data-root /path/to/26LPCV_SSUPERPOWER_2nd/datasets \
  --project-image-root <TRACK3_ROOT> \
  --shared-datasets-root /path/to/Datasets
```

검증 항목:
- 처리된 JSONL 8개 (train/val × 4 P)
- raw image 디렉토리 (data/images/, ARForensics, SynthScars, COCO, ImageNet)

### 3. Runtime 준비 — `prepare_runtime.py`

`run.sh` 가 자동 호출. 직접 호출도 가능.

수행 작업 4가지:
- yaml placeholder 치환 (`__PROJECT_ROOT__`, `__RUN_ROOT__`, `__MODEL_ROOT__`, `__LLAMA_DATA_DIR__`)
- 처리된 JSONL 을 `${RUN_ROOT}/runtime_data/datasets/` 로 복사하면서 image_path 재작성 (사용자 머신 path 로)
- LLaMA-Factory `dataset_info.json` 에 P3/P4 dataset alias 등록
- 필수 데이터/이미지 경로 존재 검증

---

## 실행 — `run.sh`

P1→P4 10 step 을 한 줄로 실행:

```bash
cd train/
./run.sh
```

기본값 override (다른 머신/경로) 시 환경변수 사용:

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

**EXACT_REPRODUCTION 정책** (default `=1`):
- `EXACT_REPRODUCTION=1` 이면 P3/P4 단일 GPU 강제 (effective batch size 보존). 콤마 분리 GPU (`P3_GPUS=0,1`) 거부.
- `EXACT_REPRODUCTION=0` 이면 multi-GPU 가능. 단 effective batch size 가 달라져 결과 weight 다름.

---

## 각 단계 상세

### P1 — base SFT (custom trainer)

| 항목 | 값 |
|---|---|
| 입력 | `data_p1/{train,val}.jsonl` |
| Trainer | `scripts/train_p1.py` (custom) |
| Base | `Qwen/Qwen2-VL-2B-Instruct` |
| 출력 | `lora_p1` → merge → `merged_p1` |
| 손실 | token CE (sft 단일 prompt classify) |
| 데이터 | 60k images, fake/real 50:50 |

### P2 — multi-prompt base (custom + aux BCE)

| 항목 | 값 |
|---|---|
| 입력 | `data_p2/{train,val}.jsonl` |
| Trainer | `scripts/train_p2.py` (custom + auxiliary BCE) |
| Base | `merged_p1` |
| 출력 | `lora_p2` → merge → `merged_p2` |
| 손실 | **token CE + overall BCE + criterion BCE** (3 loss 합산) |
| 데이터 | 10k images × 4 entry = 36k entries (multi-prompt) |

### P3 — image+evidence LoRA (LLaMA-Factory)

| 항목 | 값 |
|---|---|
| 입력 | `data_p3/{train,val}.jsonl` (LLaMA-Factory dataset_info.json alias 통해) |
| Trainer | `llamafactory-cli train --config train_p3.yaml` |
| Base | `merged_p2` |
| 출력 | `lora_p3` → merge → `merged_p3` |
| Hyperparam | batch 4, grad_accum 1, template `qwen2_vl` |
| 데이터 | 9k entries (50:50) |

### P4 warmup — text-only synthesis 1단계 (LLaMA-Factory)

| 항목 | 값 |
|---|---|
| 입력 | `data_p4/{train,val}.jsonl` (text-only, 75:25 fake) |
| Trainer | `llamafactory-cli train --config train_p4_warmup.yaml` |
| Base | `merged_p3` |
| 출력 | `lora_p4_warmup` → merge → `merged_p4_warmup` |
| Hyperparam | batch 4, grad_accum 8, **effective batch 32, lr 5e-5, epoch 0.5** |
| 데이터 | 8k entries (75:25 fake) |

### P4 final — text-only synthesis 2단계 (warmup 이후) (LLaMA-Factory)

| 항목 | 값 |
|---|---|
| 입력 | 같은 `data_p4/...` (warmup 과 동일 데이터) |
| Trainer | `llamafactory-cli train --config train_p4.yaml` |
| Base | `merged_p4_warmup` |
| 출력 | `lora_p4` → merge → **`merged_p4`** ⭐ (= **FINAL_RESULTS**) |
| Hyperparam | batch 4, grad_accum 8, **effective batch 32, lr 1e-5, epoch 0.25** (warmup 보다 작은 lr/epoch 로 fine adjust) |

→ **`merged_p4`** 가 최종 산출물. 양자화 (`quantize_kor.md`) 의 입력 모델로 사용됨.

---

## 의존성

### 모델 (HuggingFace hub)
- `Qwen/Qwen2-VL-2B-Instruct` — base 모델 (snapshot revision `895c3a49bc3fa70a340399125c650a463535e71c` 권장, 재현성)

### Python 패키지 (자세한 건 `environment_kor.md`)
- `transformers`, `peft`, `accelerate`, `torch`
- `llamafactory-cli` (LLaMA-Factory v0.9.1, editable install)

### LLaMA-Factory
- editable install 위치: `${PROJECT_ROOT}/LLaMA-Factory/` 또는 `LLaMA-Factory-qwen25vl/`
- dataset registry: `${PROJECT_ROOT}/LLaMA-Factory-qwen25vl/data/dataset_info.json`
- `prepare_runtime.py` 가 P3/P4 dataset alias 자동 patch

### 입력 데이터
- 정리본 `datasets/data_p1~p4` — `annotations_kor.md` 의 결과물
- raw image 의존성 (annotation reproduce 시 필요): `data/images/`, COCO, ImageNet, ARForensics, SynthScars

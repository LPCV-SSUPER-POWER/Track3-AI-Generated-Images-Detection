# Annotations — 학습 데이터 생성 가이드

`datasets/` 안에 동봉된 4개 학습용 jsonl (P1/P2/P3/P4) 이 어떻게 만들어졌는지 기록 + reproduce 가이드.

`datasets/` 에 결과물(199 MB, 12 파일) 이미 동봉되어 있고, 그대로 `train_kor.md` 의 P1→P4 chain 입력으로 사용 가능. 새 이미지 셋으로 직접 만들고 싶을 때만 아래 절차 따라가면 됩니다.

---

## `annotation/` 디렉토리 구조

```
26LPCV_SSUPERPOWER_2nd/annotation/
│
├── prompts/                            (Step 1 의 a_step2 sub-stage prompt)
│   └── a_step2.txt                              a_step2 합성 prompt        (2110 B)
│       (※ a_step1 prompt + FAKE_HINT 는 annotate.py 코드 안 변수)
│
├── inference/                          (Step 1 — annotation 생성 코드)
│   └── annotate.py                              Qwen2.5-VL-7B inference   (10 KB)
│
├── manifest_split/                     (Step 2 — manifest + split 코드)
│   ├── build_manifest.py                        master manifest 생성       (18 KB)
│   └── build_stage_split.py                     4 split 분할 (hardlink)    (14 KB)
│
├── jsonl_build/                        (Step 3 — 학습용 JSONL 만드는 코드)
│   ├── build_p1_jsonl.py               P1 학습 entry 만드는 메인 (단독)         (21 KB)
│   ├── build_p234_jsonl.py             P2/P3/P4 학습 entry 만드는 메인         (21 KB)
│   ├── lib_p2.py                       P2 학습 entry 빌더 (라이브러리)          (8 KB)
│   ├── lib_p3.py                       P3 학습 entry 빌더 (라이브러리)          (11 KB)
│   ├── lib_p3_text.py                  P3 entry 안 텍스트 token fit 도우미       (12 KB)
│   ├── lib_p4_text.py                  P4 entry 안 텍스트/JSON token fit 도우미  (14 KB)
│   └── lib_p234_entry.py               P234 공통 — human/gpt 대화 형식 포장      (14 KB)
│
└── samples/                            (raw annotation JSON 20개, 생성기별)
    ├── adm__fake__*.json                          3개
    ├── biggan__fake__*.json                       3개
    ├── sid_set__fake__*.json                      2개
    ├── sid_set__real__*.json                      1개
    ├── imagenet__real__*.json                     2개
    ├── coco__real__*.json                         2개
    ├── arforensics_infinity__fake__*.json         2개
    ├── arforensics_janus_pro__fake__*.json        2개
    ├── arforensics_llamagen__fake__*.json         1개
    ├── arforensics_rar__fake__*.json              1개
    └── synthscars__fake__*.json                   1개
```

---

## 전체 흐름

annotation 작업이 끝나면 → train 작업이 그 결과물을 학습 입력으로 받음.

### Raw image

ADM, BigGAN, SID (`data/images/`) + COCO train2017, ImageNet train (`<SHARED_DATASETS_ROOT>/`) + ARForensics, SynthScars (`datasets/raw/`)


### annotation 영역 (이 문서)

**Step 1 — annotate** (Qwen2.5-VL-7B inference)

  - inference 안 2 sub-stage:
    - **a_step1** — 이미지 + prompt 3번 (`A_STEP1_PROMPTS` + `FAKE_HINT` 가 annotate.py 코드 안)
    - **a_step2** — 텍스트만 1번, per_criterion JSON 합성 (`prompts/a_step2.txt`)
  - 📦 출력 → master pool: **89,263 raw annotation JSON** (fake 44,631 + real 44,631 + 1 parse error, 약 50:50)
    - 생성기별: ARForensics 5종 합 54,652 (61%) + ImageNet 8,889 + SID 6,547 + BigGAN 5,214 + ADM 5,131 + SynthScars 5,000 + COCO 3,829

**Step 2 — manifest + 4 split 분할**

  - `manifest_split/build_manifest.py` — raw annotation 89,264개 풀을 읽어 **master manifest** (각 row 의 image_path, label, source, criterion_labels 정규화 메타) 생성
  - `manifest_split/build_stage_split.py` — master manifest 를 받아 **4 split (P1/P2/P3/P4)** 으로 분할. 같은 파일을 **hardlink 로 5 곳에 공유** → 디스크 추가 사용량 0
  - 📦 출력 → 4 split 폴더 (p1/p2/p3/p4) + 각 split 의 `_split_manifest.jsonl`

**Step 3 — 학습용 JSONL 변환**

  - `jsonl_build/build_p1_jsonl.py` — split 의 raw annotation → **P1 학습 entry** (단일 prompt sft 형식) 변환. 단독 작동 (라이브러리 import 0개)
  - `jsonl_build/build_p234_jsonl.py` — split 의 raw annotation → **P2/P3/P4 학습 entry** (P2 multi-prompt, P3 image+evidence, P4 text-only 합성) 변환. 라이브러리 5개 import 해서 entry 빌드 + token fit + human/gpt 대화 형식 포장
  - 📦 출력 → `datasets/data_p1`, `data_p2`, `data_p3`, `data_p4` (4 폴더, ShareGPT 형식 jsonl) — 통계는 [Step 3 결과](#step-3-결과--정리본-datasets-의-jsonl-통계) 참조


---

**용어 표기 가이드**

| 표기 | 의미 |
|---|---|
| **Step 1 / Step 2 / Step 3** | annotation pipeline 의 큰 단계 (이 문서의 큰 흐름) |
| **a_step1 / a_step2** | Step 1 안 inference sub-stage (annotate.py 가 Qwen2.5-VL 에 묻는 4번 질문) |
| **P1 / P2 / P3 / P4** | train 학습 chain 의 phase (train_kor.md 에서 다룸) |

---

## Step 1 — Annotation 생성

### 사용 파일

| 파일 | 위치 |
|---|---|
| inference 코드 | `annotation/inference/annotate.py` |
| a_step2 합성 prompt | `annotation/prompts/a_step2.txt` |

> **a_step1 prompt 와 FAKE_HINT 는 외부 파일이 아니라 `annotate.py` 코드 안 변수**:
> - `A_STEP1_PROMPTS` (라인 18~22): 8 criteria 를 3 그룹으로 나눈 prompt 3개
> - `FAKE_HINT` (라인 24): Fake 이미지일 때 a_step1 prompt 앞에 붙이는 한 줄 hint

### 모델

- **`Qwen/Qwen2.5-VL-7B-Instruct`** (HuggingFace hub)
- **순수 inference**, 학습 안 함

### Multi-stage 동작 (이미지 1장당 Qwen2.5-VL 에 4번 질문)

```
a_step1 (이미지 + prompt, 3번 질문):
  질문 1: "Edge·Texture·Material 3 criteria 만 분석" → 응답 1 (긴 텍스트)
  질문 2: "Physical·Text·Human 3 criteria 만 분석"   → 응답 2
  질문 3: "Lighting·Perspective 2 criteria 만 분석"  → 응답 3

  → prompt 는 모두 annotate.py 코드 안 A_STEP1_PROMPTS 에서 선택:
     · Fake 이미지 → A_STEP1_PROMPTS[i] 앞에 FAKE_HINT 부착
     · Real 이미지 → A_STEP1_PROMPTS[i] 그대로

a_step2 (텍스트만, 이미지 없음):
  질문 4: "위 응답 3개 합쳐서 8 criteria JSON 으로 정리"
  → prompt: prompts/a_step2.txt (외부 파일)
  → 출력: per_criterion JSON (최종 annotation)
```

### 실행

```bash
python annotation/inference/annotate.py \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --list_json /path/to/image_list.json \
  --output_dir annotations/0421_data \
  --batch_size 32
```

`--list_json` 한 entry 형식:
```json
{"image_path": "/abs/path/to/image.png",
 "label": "fake",
 "source": "adm",
 "generator": "ADM"}
```

### 출력 — raw annotation JSON (이미지 1장 = 1 JSON)

`annotation/samples/` 의 20개 예시 참고. Schema:
```json
{
  "per_criterion": [
    {"criterion": "Lighting & Shadows Consistency", "evidence": "...", "aigc score": 0},
    {"criterion": "Edges & Boundaries", "evidence": "...", "aigc score": 1},
    ... (총 8 criteria)
  ],
  "overall_likelihood": "Real" or "AI-Generated",
  "_meta": {
    "image_path": "...",
    "label": "fake",
    "source": "adm",
    "generator": "ADM",
    "a_step1_responses": ["응답1", "응답2", "응답3"],
    "elapsed_sec": 31.8
  }
}
```

---

## Step 2 — Master Manifest + 4 Split 분할

### 사용 파일

| 파일 | 위치 |
|---|---|
| master manifest 빌더 | `annotation/manifest_split/build_manifest.py` |
| 4 split 빌더 | `annotation/manifest_split/build_stage_split.py` |

### 2-1. Master Manifest 생성

```bash
python annotation/manifest_split/build_manifest.py \
  --input_dir annotations/0421_data \
  --output_dir annotations/0421_data/manifests
```

- 입력: Step 1 출력 풀 (`annotations/0421_data/`, 89,264 JSON)
- 출력 3개:
  - `manifests/master_manifest.jsonl` — 정규화된 메타 (각 row: image_path, label, source, criterion_labels, ...)
  - `manifests/master_manifest_summary.json` — 통계
  - `manifests/family_counts.csv` — generator 별 카운트

### 2-2. 4 Split 으로 분할

```bash
python annotation/manifest_split/build_stage_split.py \
  --manifest_path annotations/0421_data/manifests/master_manifest.jsonl \
  --annotations_dir annotations/0421_data \
  --output_root annotations
```

- 정책:
  - overall ↔ criterion 충돌 row 제외 (clean only)
  - **hardlink 로 split 만듦** → 디스크 추가 사용량 0 (같은 파일을 5개 위치에서 공유)
- 출력 — 4 split 폴더 + fake/real 분포:

| split | 총 | fake | real | 비율 |
|---|---|---|---|---|
| `data_p1/` | 60,000 | 30,000 | 30,000 | **50 : 50** |
| `data_p2/` | 59,910 | 29,955 | 29,955 | **50 : 50** |
| `data_p3/` | 44,931 | 29,954 | 14,977 | **66 : 33** |
| `data_p4/` | 44,931 | 29,954 | 14,977 | **66 : 33** |

각 split 안 추가 메타: `_split_manifest.jsonl` (전 split), `_p1_structured_targets.jsonl` (P1 만)

---

## Step 3 — 학습용 JSONL 로 변환

### 사용 파일

| 파일 | 위치 | 역할 |
|---|---|---|
| **P1 메인** | `annotation/jsonl_build/build_p1_jsonl.py` | P1 학습 entry 만드는 메인 (단독, 라이브러리 import 0개) |
| **P234 메인** | `annotation/jsonl_build/build_p234_jsonl.py` | P2/P3/P4 학습 entry 만드는 메인 (라이브러리 5개 사용) |
| 라이브러리 1 | `annotation/jsonl_build/lib_p2.py` | P2 학습 entry 빌더 (entry 1개 만드는 함수) |
| 라이브러리 2 | `annotation/jsonl_build/lib_p3.py` | P3 학습 entry 빌더 (entry 1개 만드는 함수) |
| 라이브러리 3 | `annotation/jsonl_build/lib_p3_text.py` | P3 entry 안 텍스트가 token 한도 (Qwen2-VL tokenizer 기준) 넘으면 자르는 도우미 |
| 라이브러리 4 | `annotation/jsonl_build/lib_p4_text.py` | P4 entry 안 텍스트/JSON 이 token 한도 넘으면 자르는 도우미 |
| 라이브러리 5 | `annotation/jsonl_build/lib_p234_entry.py` | P234 공통 — entry 를 human/gpt 대화 형식으로 최종 포장 |

### 3-1. P1 학습 entry 변환

```bash
python annotation/jsonl_build/build_p1_jsonl.py \
  --annotations_root annotations \
  --datasets_root datasets \
  --force
```

- 입력: 4 split 폴더 (특히 `_split_manifest.jsonl`, `_p1_structured_targets.jsonl`) + raw JSON
- 출력: `datasets/data_p1/{train,val}.jsonl` ← **P1**
- 형식: ShareGPT 단일 prompt
  ```
  human: <image>\nClassify this image per criterion and overall. Return JSON only.
  gpt:   {"lighting": "Real", "edge": "Real", ..., "overall_label": "Real"}
  ```

### 3-2. P2/P3/P4 학습 entry 변환

```bash
python annotation/jsonl_build/build_p234_jsonl.py \
  --annotations_root annotations \
  --datasets_root datasets \
  --force
```

- 입력: 4 split 폴더 + raw JSON
- 라이브러리 5개를 자동 import (jsonl_build/ 안 다른 5개 파일)
- 출력:
  - `datasets/data_p2/` ← **P2** (multi-prompt 형식 entry)
  - `datasets/data_p3/` ← **P3** (이미지 + a_step1 prompt → 8 criteria evidence)
  - `datasets/data_p4/` ← **P4** (text-only a_step2 합성)

각 split 의 entry 형식이 다름:
- P2: 3 sub-prompt 모두 합친 multi-prompt entry
- P3: image + 3 sub-prompt → 8 criteria evidence
- P4: text-only a_step2 합성 (image 없이 a_step1 응답들에서 JSON 추출)

### Step 3 결과 — 정리본 `datasets/` 의 jsonl 통계

이미 정리본에 동봉된 4 jsonl 의 entry 수 + fake/real 비율 (이미지 단위, meta.json 기반):

| 폴더 | 형식 | jsonl 파일 | train (entry) | val (entry) | train fake : real | val fake : real |
|---|---|---|---|---|---|---|
| `data_p1/` | sft (단일 prompt) | `train.jsonl`, `val.jsonl` | **54,007** | **5,993** | 27,004 : 27,003 (50:50) | 2,996 : 2,997 (50:50) |
| `data_p2/` | multi-prompt (4 entry/img) | `train.jsonl`, `val.jsonl` | **36,000** (= 9,000 img × 4) | **4,000** (= 1,000 × 4) | 4,505 : 4,495 (50:50) | 499 : 501 (50:50) |
| `data_p3/` | P3 image+evidence | `train.jsonl`, `val.jsonl` | **9,000** | **1,000** | 4,505 : 4,495 (50:50) | 499 : 501 (50:50) |
| `data_p4/` | P4 text-only synthesis | `train.jsonl`, `val.jsonl` | **8,104** | **896** | 6,080 : 2,024 **(75:25)** | 672 : 224 (75:25) |

> P4 만 fake 비율 75% 로 의도된 imbalance (P4 LoRA 가 fake annotation 정교화에 더 노출되도록).

---

## 의존성

### 모델 (HuggingFace hub)
- `Qwen/Qwen2.5-VL-7B-Instruct` — Step 1 annotation 생성용
- `Qwen/Qwen2-VL-2B-Instruct` — Step 3 의 `lib_p3_text` / `lib_p4_text` 가 tokenizer 로 사용 (학습 target 모델과 같음)

### Python 패키지 (자세한 건 `environment_kor.md`)
- `transformers` (Qwen2.5-VL 지원 버전)
- `torch`, `accelerate`, `PIL`
- `qwen_vl_utils` (Qwen2.5-VL 공식 image preprocessor)

### Raw image 데이터셋 (자세한 건 `datasets.md`)
- ADM/BigGAN/SID — `data/images/`
- COCO train2017, ImageNet train — `<SHARED_DATASETS_ROOT>/`
- ARForensics — HuggingFace `Yanran21/ARForensics`
- SynthScars — (출처 보충 필요)


# Quantize — `FINAL_RESULTS` 양자화 가이드

`train_kor.md` 의 결과물 (`FINAL_RESULTS` = `merged_p4`) 를 받아 **AIMET W4A16 양자화 + QNN binary export + 제출용 zip 패키징** 하는 절차. 산출물은 `submit/{exp_name}.zip` (~2.62 GB).

quantize machine + AIMET v2 (qairt) 환경 필요. annotation/train 은 일반 ML 환경에서 가능하지만, quantize 는 **Qualcomm AI Stack** (`/opt/qcom/aistack/qairt/`) 가 깔린 머신에서만 실행 가능.

---

## `quantize/` 디렉토리 구조

```
26LPCV_SSUPERPOWER_2nd/quantize/
│
├── run_parallel_quantize_pyfiles.sh       양자화 wrapper — GPU 2개 (VEG+LLM 동시)  ~62분
├── run_sequential_quantize_pyfiles.sh     양자화 wrapper — GPU 1개 (VEG → LLM 순차) ~86분
├── run_cosine.sh                          (선택) FP vs INT8 cosine 유사도 측정
├── verify_zip.py                          제출 zip 5단계 무결성 검증
│
├── scripts/
│   └── package_submission.py              zip 패키징 (wrapper 가 자동 호출)
│
├── inference/                             cosine 측정용 코드 (run_cosine.sh 가 호출)
│   ├── llm_inout.py                       Step 1 — FP32 reference output 생성
│   ├── inference_multi.py                 Step 2 — AIHub cloud (Snapdragon QRD) 추론
│   └── contestant_uploads/inputs.json     제출 metadata (zip 안에 동봉)
│
└── py_files/                              vanilla AIMET 양자화 코드 (985 MB)
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
        └── llava_v1_5_mix665k.json        LLM calibration JSON (외부 다운로드, 1 GB)
```

`models/`, `results/`, `submit/` 디렉토리는 wrapper 가 실행 시 자동 생성 (`BUNDLE_ROOT` 기준 = `26LPCV_SSUPERPOWER_2nd/` parent).

---

## 전체 흐름

```
train_kor.md 의 결과물
   merged_p4 (= FINAL_RESULTS)
        ↓
   $BUNDLE_ROOT/models/{exp_name}_merged_stage2/   ← Step 1 배치
        ↓
   양자화 4 sub-stage
   ├── Example1A — VEG AIMET     ~6분    (GPU0)
   ├── Example2A — QNN VEG       ~15분   (GPU0, Example1A 결과 사용)
   ├── Example1B — LLM AIMET     ~28분   (GPU1, 병렬 시)
   └── Example2B — QNN LLM       ~12분   (GPU1, Example1B 결과 사용)
        ↓
   Packaging (package_submission.py)
        ↓
   Verification (verify_zip.py — 5단계)
        ↓
   $BUNDLE_ROOT/submit/{exp_name}.zip ⭐  (~2.62 GB, 제출 파일)
```

병렬 wrapper (GPU 2개) ~62분, 순차 wrapper (GPU 1개) ~86분.

---

**용어 표기 가이드**

| 표기 | 의미 |
|---|---|
| **Step 1 / Step 2 / Step 3** | quantize pipeline 의 큰 단계 (이 문서 흐름) |
| **Example1A / 1B / 2A / 2B** | AIMET vanilla py_files 의 4 sub-stage 코드 |
| **VEG** | Vision Encoder Graph (이미지 입력 → embedding) |
| **LLM** | Language Model (Qwen2-VL-2B 의 LM 부분) |
| **`{exp_name}`** | 양자화 실험 이름 (자유, e.g. `qwen2_FINAL_RESULTS`) |
| **`BUNDLE_ROOT`** | bundle 의 root path (`26LPCV_SSUPERPOWER_2nd/`). wrapper 가 자동 감지 |

---

## Step 1 — 양자화 input 모델 준비

### 1-1. FP 모델 배치

`train_kor.md` 의 P1→P4 chain 결과물 `merged_p4` 를 `quantize/../models/{exp_name}_merged_stage2/` 에 배치.

```bash
EXP=qwen2_FINAL_RESULTS
BUNDLE_ROOT=/path/to/26LPCV_SSUPERPOWER_2nd
mkdir -p $BUNDLE_ROOT/models

# 복사 (또는 symlink)
cp -r {train_runtime}/models/merged_p4 $BUNDLE_ROOT/models/${EXP}_merged_stage2
# ln -s {train_runtime}/models/merged_p4 $BUNDLE_ROOT/models/${EXP}_merged_stage2
```

> `{exp_name}` 명명 자유. 산출물 디렉토리는 `{exp_name}_merged_stage2` 패턴이어야 wrapper 가 인식.

### 1-2. 11 파일 + md5 검증

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
├── tokenizer.json           (11.4 MB; 8.8 MB 면 깨진 거)
├── tokenizer_config.json
└── vocab.json
```

```bash
cd $BUNDLE_ROOT/models/${EXP}_merged_stage2 && md5sum *
```

원본과 비교 11 파일 모두 일치해야 양자화 진입.

---

## Step 2 — AIMET W4A16 양자화 (4 sub-stage)

| Sub-stage | 코드 | 역할 | GPU 메모리 | 시간 |
|---|---|---|---|---|
| **Example1A** | `py_files/Example1A/run_veg.py` | VEG AIMET quantize (FP → INT8 sim) | ~25 GB | ~6분 |
| **Example2A** | `py_files/Example2A/host_linux/run_qnn_veg.py` | QNN VEG export (Qualcomm tools) | (CPU) | ~15분 |
| **Example1B** | `py_files/Example1B/run_llm.py` | LLM AIMET quantize (W4A16) | ~20 GB | ~28분 |
| **Example2B** | `py_files/Example2B/host_linux/run_qnn_llm.py` | QNN LLM export (binary serialize) | (CPU) | ~12분 |

### 2-A. 병렬 (권장) — GPU0=VEG / GPU1=LLM 동시 — ~62분

```bash
cd $BUNDLE_ROOT/quantize
EXP=qwen2_FINAL_RESULTS
LOG=$BUNDLE_ROOT/results/${EXP}_parallel.log
mkdir -p $BUNDLE_ROOT/results

nohup bash run_parallel_quantize_pyfiles.sh $EXP 0 1 > $LOG 2>&1 &
```

VEG chain (1A → 2A) GPU0 + LLM chain (1B → 2B) GPU1 동시 실행. 두 chain 모두 끝나면 packaging + verification.

### 2-B. 순차 — GPU 1개 — ~86분

```bash
nohup bash run_sequential_quantize_pyfiles.sh $EXP 0 > $LOG 2>&1 &   # GPU0 만
```

VEG chain → LLM chain → packaging 순차. 다른 GPU 에 다른 모델 양자화 동시 가능.

### 진행 단계 확인

```bash
grep -E '===' $LOG
```
```
=== Example1A (VEG) ===     ~6분
=== Example2A (QNN VEG) === ~15분
=== Example1B (LLM) ===     ~28분
=== Example2B (QNN LLM) === ~12분
=== Packaging ===           ~5초
=== Verification ===        ~2초
=== ALL DONE ===
```

`ALL CHECKS PASSED` + `submit/${EXP}.zip` 생성되면 양자화 성공.

---

## Step 3 — zip 패키징 + 검증

wrapper 가 자동:
1. `scripts/package_submission.py` → 8개 파일 + `SSUPER POWER/` prefix → `submit/${EXP}.zip` 생성
2. `verify_zip.py` 5단계 검사

### 5단계 검사 (verify_zip.py)
1. zip 파일 목록 (8개)
2. 필수 파일 존재 여부
3. 원본 vs zip 사이즈 비교
4. inputs.json 내용 확인
5. zip CRC 전수 검증

### 직접 검증 (선택)

```bash
ZIP=$BUNDLE_ROOT/submit/${EXP}.zip
unzip -l $ZIP                    # 8개 파일 목록 (~2.62 GB)
unzip -t $ZIP                    # CRC 전수 검사
md5sum $ZIP
```

### 8개 제출 파일

| 경로 (zip 안) | 사이즈 | 설명 |
|---|---|---|
| `SSUPER POWER/ar128-ar1-cl2048/weight_sharing_model_1_of_1.serialized.bin` | 894 MB | LLM weight |
| `SSUPER POWER/embedding_weights_151936x1536.raw` | 890 MB | Embedding |
| `SSUPER POWER/serialized_binaries/veg.serialized.bin` | ~695 MB | VEG |
| `SSUPER POWER/tokenizer.json` | 11 MB | Tokenizer |
| `SSUPER POWER/mask.raw` | 3 MB | Attention mask |
| `SSUPER POWER/position_ids_cos.raw` | 138 KB | RoPE cos |
| `SSUPER POWER/position_ids_sin.raw` | 138 KB | RoPE sin |
| `SSUPER POWER/inputs.json` | 1.7 KB | 제출 metadata |

---

## (선택) Cosine Similarity 측정

양자화 후 FP32 vs INT8 출력 유사도 측정 (~19분). 양자화 손실 정량 평가.

### 실행
```bash
cd $BUNDLE_ROOT/quantize
EXP=qwen2_FINAL_RESULTS
LOG=$BUNDLE_ROOT/results/${EXP}_cosine.log
CUDA_VISIBLE_DEVICES=0 nohup bash run_cosine.sh $EXP > $LOG 2>&1 &
```

### 단계
| Step | 코드 | 시간 | 설명 |
|---|---|---|---|
| 1 | `inference/llm_inout.py` | ~4분 | FP32 reference output 생성 (10 batch × 10 token), GPU 사용 |
| 2 | `inference/inference_multi.py` | ~15분 | Snapdragon 8 Elite QRD (AIHub cloud) 추론 |
| 3 | cosine 계산 | ~2초 | FP vs QNN 코사인 유사도 |

### 결과 해석

```
cos_avg=0.938304  per_token=[0.961, 0.925, 0.966, ...]
```

| `cos_avg` 구간 | 판정 |
|---|---|
| ≥ 0.78 | ✅ 안전 |
| 0.75 ~ 0.78 | ⚠️ 위험 (단일 outlier 가능) |
| < 0.75 | ❌ 위험 (단, vanilla `py_files` + LLaVA generic 조합은 contest 통과 가능) |

> **cos↑ ≠ contest↑**. `best_calib100` 으로 cos 회복해도 contest 점수는 같거나 떨어짐. **제출은 vanilla py_files (LLaVA generic) 권장**.

---

## (선택) py_files 변종 (block_size, mixed precision)

vanilla `py_files` 에서 cache 제외 rsync:

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

### 주요 변경 가능 부분
- **block_size**: `py_files_a/Example1B/run_llm.py` → `BLOCK_QUANT_SIZE = 64` 수정 (16/32/64)
- **mixed precision**: `py_files_a/Example1B/config/mixed_precision_config/qwen2_w4a16_gqa.json` 수정 (k/q/v/o proj W8 추가 등)

### 변종용 wrapper
```bash
cp run_parallel_quantize_pyfiles.sh run_parallel_quantize_pyfiles_a.sh
sed -i "s|/py_files/|/py_files_a/|g" run_parallel_quantize_pyfiles_a.sh
chmod +x run_parallel_quantize_pyfiles_a.sh
```

---

## 의존성

### 환경 (자세한 건 `environment_kor.md`)
| 구성 | 위치 / 비고 |
|---|---|
| AIMET Python env | conda env `26lpcv` (quantize machine, torch 1.13.1, AIMET 1.34) — `requirements/26lpcv_aimet_requirements.txt` |
| QNN Python env | conda env `26lpcv_qnn` (quantize machine) — `requirements/26lpcv_qnn_requirements.txt` |
| AIMET v2 (qairt) | `/opt/qcom/aistack/qairt/2.31.0.250130/` — Qualcomm AI Stack (별도 다운로드) |
| AIMET pro 1.34 wheel | Aimet/AimetCommon/AimetTorch — 별도 wheel 설치 (`environment_kor.md` 참조) |

스크립트가 자동 PATH/LD_LIBRARY_PATH 설정 — 사용자 추가 활성화 불필요.

### 입력 모델
- `train_kor.md` 의 결과물 `merged_p4` (= `FINAL_RESULTS`)
- 11 파일 + md5 일치 검증 (Step 1)

### Calibration data
- **LLM**: `quantize/py_files/local_data/llava_v1_5_mix665k.json` — **외부 다운로드** (HuggingFace `liuhaotian/LLaVA-Instruct-150K`, 1 GB)
- **VEG**: COCO train2017 100장 (외부, e.g. `<COCO_CALIB_ROOT>/` 또는 다운로드)

---

## 빠른 시작 (요약)

```bash
# 0. 변수
EXP=qwen2_FINAL_RESULTS
BUNDLE_ROOT=/path/to/26LPCV_SSUPERPOWER_2nd

# 1. input 배치 + md5
cp -r {train_runtime}/models/merged_p4 $BUNDLE_ROOT/models/${EXP}_merged_stage2
cd $BUNDLE_ROOT/models/${EXP}_merged_stage2 && md5sum *

# 2. 병렬 양자화 (~62분)
cd $BUNDLE_ROOT/quantize
nohup bash run_parallel_quantize_pyfiles.sh $EXP 0 1 \
  > $BUNDLE_ROOT/results/${EXP}_parallel.log 2>&1 &

# 3. 진행 확인
grep -E '===' $BUNDLE_ROOT/results/${EXP}_parallel.log

# 4. 완료 후 zip 확인
ls -la $BUNDLE_ROOT/submit/${EXP}.zip
unzip -t $BUNDLE_ROOT/submit/${EXP}.zip

# 5. (선택) cosine 측정
CUDA_VISIBLE_DEVICES=0 nohup bash run_cosine.sh $EXP \
  > $BUNDLE_ROOT/results/${EXP}_cosine.log 2>&1 &
```

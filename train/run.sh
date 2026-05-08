#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${TRAIN_ROOT}/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set — root containing data/images/, datasets/raw/}"
DATA_ROOT="${DATA_ROOT:-${BUNDLE_ROOT}}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${DATA_ROOT}}"
PROJECT_IMAGE_ROOT="${PROJECT_IMAGE_ROOT:-${PROJECT_ROOT}}"
SHARED_DATASETS_ROOT="${SHARED_DATASETS_ROOT:?SHARED_DATASETS_ROOT must be set — root containing ImageNet/, coco/}"
RUN_ROOT="${RUN_ROOT:-${TRAIN_ROOT}/runs/train}"
CONFIG_TEMPLATE_ROOT="${CONFIG_TEMPLATE_ROOT:-${TRAIN_ROOT}/configs}"
CONFIG_ROOT="${CONFIG_ROOT:-${RUN_ROOT}/resolved_configs}"

PYTHON="${PYTHON:-python}"  # default uses PATH python (after conda activate 26lpcv)
LLAMA="${LLAMA:-llamafactory-cli}"  # default uses PATH llamafactory-cli

P1_GPU="${P1_GPU:-0}"
P2_GPU="${P2_GPU:-0}"
P3_GPUS="${P3_GPUS:-0}"
P4_GPUS="${P4_GPUS:-0}"
MERGE_GPU="${MERGE_GPU:-0}"
EXACT_REPRODUCTION="${EXACT_REPRODUCTION:-1}"
ALLOW_EXISTING_RUN_ROOT="${ALLOW_EXISTING_RUN_ROOT:-0}"

MODEL_ROOT="${RUN_ROOT}/models/qwen2_train"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

timestamp() { date '+%F %T'; }

run_llama_train() {
  local gpus="$1"
  local config="$2"
  if [[ "${gpus}" == *,* ]]; then
    CUDA_VISIBLE_DEVICES="${gpus}" FORCE_TORCHRUN=1 "${LLAMA}" train "${config}"
  else
    CUDA_VISIBLE_DEVICES="${gpus}" "${LLAMA}" train "${config}"
  fi
}

mkdir -p "${LOG_DIR}" "${MODEL_ROOT}"

if [[ "${EXACT_REPRODUCTION}" == "1" ]]; then
  if [[ "${P3_GPUS}" == *,* || "${P4_GPUS}" == *,* ]]; then
    echo "[error] EXACT_REPRODUCTION=1 requires single-GPU P3/P4 to preserve effective batch size." >&2
    echo "        Use P3_GPUS=0 P4_GPUS=0, or set EXACT_REPRODUCTION=0 knowingly." >&2
    exit 2
  fi
fi

if [[ "${ALLOW_EXISTING_RUN_ROOT}" != "1" ]]; then
  if [[ -d "${MODEL_ROOT}" && -n "$(find "${MODEL_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[error] MODEL_ROOT is not empty: ${MODEL_ROOT}" >&2
    echo "        Use a new RUN_ROOT or set ALLOW_EXISTING_RUN_ROOT=1 knowingly." >&2
    exit 2
  fi
fi

{
  echo "[$(timestamp)] G2 reproduction start"
  echo "TRAIN_ROOT=${TRAIN_ROOT}"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "PROCESSED_DATA_ROOT=${PROCESSED_DATA_ROOT}"
  echo "PROJECT_IMAGE_ROOT=${PROJECT_IMAGE_ROOT}"
  echo "SHARED_DATASETS_ROOT=${SHARED_DATASETS_ROOT}"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "PYTHON=${PYTHON}"
  echo "LLAMA=${LLAMA}"
  echo "P1_GPU=${P1_GPU} P2_GPU=${P2_GPU} P3_GPUS=${P3_GPUS} P4_GPUS=${P4_GPUS} MERGE_GPU=${MERGE_GPU}"
  echo "EXACT_REPRODUCTION=${EXACT_REPRODUCTION}"

  echo "[$(timestamp)] prepare runtime JSONL/configs"
  "${PYTHON}" "${TRAIN_ROOT}/scripts/prepare_runtime.py" \
    --project-root "${PROJECT_ROOT}" \
    --processed-data-root "${PROCESSED_DATA_ROOT}" \
    --project-image-root "${PROJECT_IMAGE_ROOT}" \
    --shared-datasets-root "${SHARED_DATASETS_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --config-template-root "${CONFIG_TEMPLATE_ROOT}" \
    --config-output-root "${CONFIG_ROOT}"

  echo "[$(timestamp)] P1 train: p1_sft -> lora_p1"
  CUDA_VISIBLE_DEVICES="${P1_GPU}" PYTHONPATH="${TRAIN_ROOT}/scripts:${PROJECT_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" "${TRAIN_ROOT}/scripts/train_p1.py" \
    --config "${CONFIG_ROOT}/train_p1.yaml"

  echo "[$(timestamp)] P1 merge: lora_p1 -> merged_p1"
  CUDA_VISIBLE_DEVICES="${MERGE_GPU}" PYTHONPATH="${TRAIN_ROOT}/scripts:${PROJECT_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" "${TRAIN_ROOT}/scripts/merge_p1.py" \
    --config "${CONFIG_ROOT}/merge_p1.yaml"
  "${PYTHON}" "${TRAIN_ROOT}/scripts/fix_preprocessor.py" "${MODEL_ROOT}/merged_p1"

  echo "[$(timestamp)] P2 train: data_p2 + token CE + overall BCE + criterion BCE"
  CUDA_VISIBLE_DEVICES="${P2_GPU}" PYTHONPATH="${TRAIN_ROOT}/scripts:${PROJECT_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" "${TRAIN_ROOT}/scripts/train_p2.py" \
    --config "${CONFIG_ROOT}/train_p2.yaml"

  echo "[$(timestamp)] P2 merge: lora_p2 -> merged_p2"
  CUDA_VISIBLE_DEVICES="${MERGE_GPU}" PYTHONPATH="${TRAIN_ROOT}/scripts:${PROJECT_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" "${TRAIN_ROOT}/scripts/merge_p2.py" \
    --config "${CONFIG_ROOT}/merge_p2.yaml"
  "${PYTHON}" "${TRAIN_ROOT}/scripts/fix_preprocessor.py" "${MODEL_ROOT}/merged_p2"

  echo "[$(timestamp)] P3 train: data_p3 image -> evidence text"
  run_llama_train "${P3_GPUS}" "${CONFIG_ROOT}/train_p3.yaml"

  echo "[$(timestamp)] P3 merge: lora_p3 -> merged_p3"
  CUDA_VISIBLE_DEVICES="${MERGE_GPU}" "${LLAMA}" export "${CONFIG_ROOT}/merge_p3.yaml"
  "${PYTHON}" "${TRAIN_ROOT}/scripts/fix_preprocessor.py" "${MODEL_ROOT}/merged_p3"

  echo "[$(timestamp)] P4 warmup train: P4 text-only synthesis (75:25 fake)"
  run_llama_train "${P4_GPUS}" "${CONFIG_ROOT}/train_p4_warmup.yaml"

  echo "[$(timestamp)] P4 warmup merge -> merged_p4_warmup"
  CUDA_VISIBLE_DEVICES="${MERGE_GPU}" "${LLAMA}" export "${CONFIG_ROOT}/merge_p4_warmup.yaml"
  "${PYTHON}" "${TRAIN_ROOT}/scripts/fix_preprocessor.py" "${MODEL_ROOT}/merged_p4_warmup"

  echo "[$(timestamp)] P4 final train: short continuation from warmup"
  run_llama_train "${P4_GPUS}" "${CONFIG_ROOT}/train_p4.yaml"

  echo "[$(timestamp)] P4 final merge -> merged_p4"
  CUDA_VISIBLE_DEVICES="${MERGE_GPU}" "${LLAMA}" export "${CONFIG_ROOT}/merge_p4.yaml"
  "${PYTHON}" "${TRAIN_ROOT}/scripts/fix_preprocessor.py" "${MODEL_ROOT}/merged_p4"

  echo "[$(timestamp)] G2 reproduction finished"
  echo "FINAL_MODEL=${MODEL_ROOT}/merged_p4"
} 2>&1 | tee -a "${LOG_FILE}"

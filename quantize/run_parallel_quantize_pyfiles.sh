#!/usr/bin/env bash
set -euo pipefail

EXP=$1
GPU_VEG=${2:-0}
GPU_LLM=${3:-1}

# Self-contained ROOT detection: derive QUANT_ROOT from script location,
# BUNDLE_ROOT from QUANT_ROOT's parent. Sub-processes (py_files) read
# BUNDLE_ROOT from the environment.
QUANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(dirname "$QUANT_ROOT")"
export BUNDLE_ROOT

PYTHON_LPCV="${PYTHON_LPCV:?PYTHON_LPCV must point to 26lpcv conda env python (e.g. /opt/conda/envs/26lpcv/bin/python)}"
PYTHON_QNN="${PYTHON_QNN:?PYTHON_QNN must point to 26lpcv_qnn conda env python}"

export PYTHONPATH=/opt/qcom/aistack/qairt/2.31.0.250130/lib/python
export LD_LIBRARY_PATH=/opt/qcom/aistack/qairt/2.31.0.250130/lib/x86_64-linux-clang:${CONDA_26LPCV_LIB:?CONDA_26LPCV_LIB must be set}:${CONDA_26LPCV_QNN_LIB:?CONDA_26LPCV_QNN_LIB must be set}

LOG_DIR=$BUNDLE_ROOT/results
mkdir -p $LOG_DIR
VEG_LOG=$LOG_DIR/${EXP}_veg.log
LLM_LOG=$LOG_DIR/${EXP}_llm.log

echo "[$(date)] Starting PARALLEL quantization for: $EXP (VEG=GPU$GPU_VEG, LLM=GPU$GPU_LLM)"
echo "[$(date)] BUNDLE_ROOT=$BUNDLE_ROOT"
echo "[$(date)] VEG log: $VEG_LOG"
echo "[$(date)] LLM log: $LLM_LOG"

(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=$GPU_VEG
  echo "[$(date)] === Example1A (VEG) GPU$GPU_VEG ==="
  cd $QUANT_ROOT/py_files/Example1A
  $PYTHON_LPCV run_veg.py --exp-name $EXP
  echo "[$(date)] === Example2A (QNN VEG) GPU$GPU_VEG ==="
  cd $QUANT_ROOT/py_files/Example2A/host_linux
  $PYTHON_QNN run_qnn_veg.py --exp-name $EXP
  echo "[$(date)] === VEG chain DONE ==="
) > $VEG_LOG 2>&1 &
VEG_PID=$!

(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=$GPU_LLM
  echo "[$(date)] === Example1B (LLM) GPU$GPU_LLM ==="
  cd $QUANT_ROOT/py_files/Example1B
  $PYTHON_LPCV run_llm.py --exp-name $EXP
  echo "[$(date)] === Example2B (QNN LLM) GPU$GPU_LLM ==="
  cd $QUANT_ROOT/py_files/Example2B/host_linux
  $PYTHON_QNN run_qnn_llm.py --exp-name $EXP
  echo "[$(date)] === LLM chain DONE ==="
) > $LLM_LOG 2>&1 &
LLM_PID=$!

echo "[$(date)] VEG PID=$VEG_PID, LLM PID=$LLM_PID"
echo "[$(date)] Waiting for both chains..."

FAILED=0
if ! wait $VEG_PID; then
  echo "[$(date)] VEG chain FAILED"
  FAILED=1
fi
if ! wait $LLM_PID; then
  echo "[$(date)] LLM chain FAILED"
  FAILED=1
fi

if [ $FAILED -ne 0 ]; then
  echo "[$(date)] One or more chains failed. Aborting packaging."
  echo "[$(date)] tail $VEG_LOG and $LLM_LOG to debug."
  exit 1
fi

echo "[$(date)] Both chains DONE. === Packaging ==="
$PYTHON_LPCV $QUANT_ROOT/scripts/package_submission.py --exp-name $EXP

echo "[$(date)] === Verification ==="
sed "s|py_files|$EXP|g" $QUANT_ROOT/verify_zip.py > /tmp/verify_zip_$EXP.py
$PYTHON_LPCV /tmp/verify_zip_$EXP.py

echo "[$(date)] === ALL DONE for $EXP ==="

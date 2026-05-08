#!/usr/bin/env bash
set -euo pipefail

EXP=$1
GPU=${2:-0}

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
export CUDA_VISIBLE_DEVICES=$GPU

LOG_DIR=$BUNDLE_ROOT/results
mkdir -p $LOG_DIR

echo "[$(date)] Sequential quantization (vanilla py_files) for $EXP on GPU$GPU"
echo "[$(date)] BUNDLE_ROOT=$BUNDLE_ROOT"

echo "[$(date)] === Example1A (VEG) ==="
cd $QUANT_ROOT/py_files/Example1A
$PYTHON_LPCV run_veg.py --exp-name $EXP

echo "[$(date)] === Example2A (QNN VEG) ==="
cd $QUANT_ROOT/py_files/Example2A/host_linux
$PYTHON_QNN run_qnn_veg.py --exp-name $EXP

echo "[$(date)] === Example1B (LLM) ==="
cd $QUANT_ROOT/py_files/Example1B
$PYTHON_LPCV run_llm.py --exp-name $EXP

echo "[$(date)] === Example2B (QNN LLM) ==="
cd $QUANT_ROOT/py_files/Example2B/host_linux
$PYTHON_QNN run_qnn_llm.py --exp-name $EXP

echo "[$(date)] === Packaging ==="
$PYTHON_LPCV $QUANT_ROOT/scripts/package_submission.py --exp-name $EXP

echo "[$(date)] === Verification ==="
sed "s|py_files|$EXP|g" $QUANT_ROOT/verify_zip.py > /tmp/verify_zip_$EXP.py
$PYTHON_LPCV /tmp/verify_zip_$EXP.py

echo "[$(date)] === ALL DONE for $EXP ==="

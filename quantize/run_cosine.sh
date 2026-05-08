#!/bin/bash
set -euo pipefail

EXP=${1:-FINAL_RESULTS}

# Self-contained ROOT detection
QUANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(dirname "$QUANT_ROOT")"
export BUNDLE_ROOT

PYTHON="${PYTHON:-python}"  # default uses PATH (after conda activate 26lpcv)
EXAMPLE1B=$QUANT_ROOT/py_files/Example1B

INOUT_DIR=$BUNDLE_ROOT/results/$EXP/inout_data
AIHUB_DIR=$BUNDLE_ROOT/results/$EXP/aihub_out
mkdir -p $INOUT_DIR $AIHUB_DIR

echo "[$(date)] Step 1: llm_inout.py for $EXP"
echo "[$(date)] BUNDLE_ROOT=$BUNDLE_ROOT"
CFG=$EXAMPLE1B/config/nb_config_tang.yml
sed -i "s|model_id:.*|model_id: $BUNDLE_ROOT/models/${EXP}_merged_stage2|" $CFG
sed -i "s|ARN:.*|ARN: 1|" $CFG
rm -f $INOUT_DIR/*.pt
cd $EXAMPLE1B
PYTHONPATH=$EXAMPLE1B $PYTHON $QUANT_ROOT/inference/llm_inout.py --eval_batch 10 --eval_token 10 --save_path $INOUT_DIR/
sed -i "s|ARN:.*|ARN: 1073|" $CFG
echo "[$(date)] Step 1 DONE"

echo "[$(date)] Step 2: inference_multi.py for $EXP"
cd $BUNDLE_ROOT/results/$EXP/Example2B
$PYTHON $QUANT_ROOT/inference/inference_multi.py --device_model "Snapdragon 8 Elite QRD" --load_path "$INOUT_DIR/inputs_b0_t*.pt" --out_path $AIHUB_DIR/ --batch 0
echo "[$(date)] Step 2 DONE"

echo "[$(date)] Step 3: cosine sim"
export INOUT_DIR_PY=$INOUT_DIR
export AIHUB_DIR_PY=$AIHUB_DIR
$PYTHON - << 'PYEOF'
import numpy as np, torch, os
from sklearn.metrics.pairwise import cosine_similarity
INOUT_DIR = os.environ['INOUT_DIR_PY']
AIHUB_DIR = os.environ['AIHUB_DIR_PY']
dir_device = f'{AIHUB_DIR}/submission_0'
cos_list = []
for run in range(10):
    fp32 = torch.load(f'{INOUT_DIR}/outputs_b0_t{run}.pt', map_location='cpu')['logits'].cpu().numpy().reshape(1,-1)
    qnn  = np.load(f'{dir_device}/output_logits.npy')[run].reshape(1,-1)
    cos_list.append(float(cosine_similarity(fp32, qnn)[0][0]))
print(f"cos_avg={sum(cos_list)/len(cos_list):.6f}  per_token={[round(c,4) for c in cos_list]}")
PYEOF
echo "[$(date)] ALL DONE cosine sim for $EXP"

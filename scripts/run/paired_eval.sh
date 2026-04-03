#!/bin/bash

# Paired evaluation: run two models on identical DR parameters
# Produces scatter plot, improvement chart, rank correlation, buckets, 2x2 table
#
# Usage:
#   ./scripts/run/paired_eval.sh <run1> <ckpt1> <label1> <run2> <ckpt2> <label2> [num_envs]
#
# Example:
#   ./scripts/run/paired_eval.sh best_v26 best_model.pt V26 \
#       2026-04-03_02-05-58_finetune_v26_robust best_model.pt V30 1000

set -e

RUN1=${1:?"Usage: $0 <run1> <ckpt1> <label1> <run2> <ckpt2> <label2> [num_envs]"}
CKPT1=${2:-best_model.pt}
LABEL1=${3:-Model1}
RUN2=${4:?"Missing run2"}
CKPT2=${5:-best_model.pt}
LABEL2=${6:-Model2}
NUM_ENVS=${7:-1000}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "========================================"
echo "  Paired Evaluation"
echo "  Model 1: $LABEL1 ($RUN1 / $CKPT1)"
echo "  Model 2: $LABEL2 ($RUN2 / $CKPT2)"
echo "  Envs:    $NUM_ENVS"
echo "========================================"

python scripts/rsl_rl/paired_eval.py \
    --task Isaac-Quadcopter-Race-v0 \
    --model1_run "$RUN1" --model1_ckpt "$CKPT1" --model1_label "$LABEL1" \
    --model2_run "$RUN2" --model2_ckpt "$CKPT2" --model2_label "$LABEL2" \
    --headless --num_envs "$NUM_ENVS" --max_steps 3000

#!/bin/bash

# Evaluate a circle track policy.
# Can run visual (with video) or headless batch evaluation.
#
# Usage:
#   ./scripts/run/eval_circle.sh <run_dir> [checkpoint] [num_envs]
#
# Examples:
#   ./scripts/run/eval_circle.sh 2026-04-07_12-00-00                    # play with video
#   ./scripts/run/eval_circle.sh 2026-04-07_12-00-00 best_model.pt 1000 # batch eval

set -e

RUN_DIR=${1:?"ERROR: Provide run directory as first argument"}
CHECKPOINT=${2:-best_model.pt}
NUM_ENVS=${3:-1}

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

# Change to the project root directory automatically
cd "$(dirname "$0")/../.."

# Export PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

if [ "$NUM_ENVS" -eq 1 ]; then
    echo "========================================"
    echo "  Circle Track Play (with video)"
    echo "  Run dir:    $RUN_DIR"
    echo "  Checkpoint: $CHECKPOINT"
    echo "========================================"

    python scripts/rsl_rl/play_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --num_envs 1 \
        --load_run "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --headless \
        --video \
        --video_length 800
else
    echo "========================================"
    echo "  Circle Track Batch Evaluation"
    echo "  Run dir:    $RUN_DIR"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Envs:       $NUM_ENVS"
    echo "========================================"

    python scripts/rsl_rl/batch_eval_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --load_run "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --headless \
        --num_trials 1 \
        --num_envs "$NUM_ENVS" \
        --max_steps 3000
fi

#!/bin/bash

# Train the fixed sim2real baseline on the powerloop track.
# This keeps the post-fix baseline reward/reset/DR settings and only switches
# the track to powerloop for the next experiment stage.
#
# Usage:
#   ./scripts/run/train_powerloop_baseline.sh [max_iterations] [num_envs]

set -e

MAX_ITERS=${1:-5000}
NUM_ENVS=${2:-8192}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "========================================"
echo "  Powerloop Fixed Baseline Training"
echo "  Iterations: $MAX_ITERS"
echo "  Envs:       $NUM_ENVS"
echo "========================================"

ENV_OVERRIDES='{"track_name":"powerloop"}' python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    --logger wandb \
    --run_name "powerloop-fixed-baseline"

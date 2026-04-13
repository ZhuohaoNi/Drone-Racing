#!/bin/bash

# Train a safe circle track policy for sim2real deployment.
# Uses the standard reward structure but on the circle track.
# This is the baseline -- use train_circle_safe.sh for the sim2real-tuned version.
#
# Usage:
#   ./scripts/run/train_circle.sh [max_iterations] [num_envs]
#
# Examples:
#   ./scripts/run/train_circle.sh              # defaults: 2000 iters, 8192 envs
#   ./scripts/run/train_circle.sh 3000 16384

set -e

MAX_ITERS=${1:-3000}
NUM_ENVS=${2:-8192}

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

# Change to the project root directory automatically
cd "$(dirname "$0")/../.."

# Export PYTHONPATH to include the project root
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "========================================"
echo "  Circle Track Training"
echo "  Iterations: $MAX_ITERS"
echo "  Envs:       $NUM_ENVS"
echo "========================================"

python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    --logger wandb

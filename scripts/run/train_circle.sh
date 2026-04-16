#!/bin/bash

# Train the fixed circle-track sim2real baseline.
# Default config now uses:
#   - sparse V3-style reward core
#   - gate_side=0.7
#   - light delta-action penalty (-0.1)
#   - no spline reset / no extra mass or motor-tau DR
#   - 1-step action latency
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

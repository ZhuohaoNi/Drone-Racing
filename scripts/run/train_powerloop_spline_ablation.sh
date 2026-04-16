#!/bin/bash

# Train the powerloop baseline with a configurable spline-reset mix.
# The ratio is applied only to the geometric reset bucket:
# replay and ground resets keep their baseline ratios.
#
# Usage:
#   ./scripts/run/train_powerloop_spline_ablation.sh [spline_ratio] [max_iterations] [num_envs]
#
# Examples:
#   ./scripts/run/train_powerloop_spline_ablation.sh 0.50
#   ./scripts/run/train_powerloop_spline_ablation.sh 1.00 5000 8192

set -e

SPLINE_RATIO=${1:-0.50}
MAX_ITERS=${2:-5000}
NUM_ENVS=${3:-8192}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

RUN_TAG=$(python - <<PY
ratio = float("${SPLINE_RATIO}")
print(f"{ratio:.2f}".replace(".", "p"))
PY
)

echo "========================================"
echo "  Powerloop Spline Ablation Training"
echo "  Spline ratio: $SPLINE_RATIO"
echo "  Iterations:   $MAX_ITERS"
echo "  Envs:         $NUM_ENVS"
echo "========================================"

ENV_OVERRIDES="{\"track_name\":\"powerloop\",\"use_spline_reset\":true,\"spline_reset_ratio\":${SPLINE_RATIO}}" \
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    --logger wandb \
    --run_name "powerloop-spline-r${RUN_TAG}"

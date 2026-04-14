#!/bin/bash

# Circle-V4: ultra-conservative policy for sim2real fallback.
# Stronger cmd_reg to limit body rates, lower lap pressure.
# Use this as the safe option at Pennovation if V3 is too aggressive.
#
# Usage:
#   ./scripts/run/train_circle_v4.sh [max_iterations] [num_envs]

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

# V4 reward overrides: stronger smoothness, less time pressure
export REW_CMD_REG_RP=-2.0        # 2x stronger roll/pitch rate penalty
export REW_CMD_REG_YAW=-1.0       # 2x stronger yaw rate penalty
export REW_LAP_INCOMPLETE=-0.02   # less time pressure, fly slow and safe

echo "========================================"
echo "  Circle-V4 (Conservative) Training"
echo "  Iterations: $MAX_ITERS"
echo "  Envs:       $NUM_ENVS"
echo "  cmd_reg_rp:       $REW_CMD_REG_RP"
echo "  cmd_reg_yaw:      $REW_CMD_REG_YAW"
echo "  lap_incomplete:   $REW_LAP_INCOMPLETE"
echo "========================================"

python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    --logger wandb

#!/bin/bash

# Train the latest powerloop R1+D1 candidate directly.
# This uses the current gate detector fixes:
#   - forward-pass hysteresis
#   - shallow wrong-way detection on the active gate
#   - gate dwell timeout near the plane
#   - previous-gate backtrack termination
#
# Usage:
#   ./scripts/run/train_powerloop_r1d1.sh [max_iterations] [num_envs]

set -e

MAX_ITERS=${1:-5000}
NUM_ENVS=${2:-8192}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

REWARD_OVERRIDES='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5}'
ENV_OVERRIDES='{"track_name":"powerloop","replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"twr_randomization_pct":0.10,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'

echo "========================================"
echo "  Powerloop R1+D1 Training"
echo "  Iterations: $MAX_ITERS"
echo "  Envs:       $NUM_ENVS"
echo "  Reward:     $REWARD_OVERRIDES"
echo "  Env:        $ENV_OVERRIDES"
echo "========================================"

REWARD_OVERRIDES="$REWARD_OVERRIDES" \
ENV_OVERRIDES="$ENV_OVERRIDES" \
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    --logger wandb \
    --run_name "powerloop-r1d1-gateprogress-stagedreset"

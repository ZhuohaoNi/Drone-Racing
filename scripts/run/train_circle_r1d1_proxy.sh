#!/bin/bash

# Train a circle-track proxy using the same generic training stack as
# `2026-04-18_14-55-19_powerloop-r1d1-gate3mask`, but without the
# powerloop-specific Gate-3 progress mask.
#
# Intent:
#   - keep the current anti-exploit gate semantics in CircleQuadcopterStrategy
#   - keep R1 reward terms: dense gate progress + lap-complete bonus
#   - keep D1 env recipe: staged replay reset + moderate DR
#   - switch track to circle for the next real-world proxy test
#
# Important:
#   - DO NOT copy `progress_skip_gate_indices=[3]` from powerloop.
#     On circle, gate index 3 is just the last ordinary gate, not the
#     powerloop loop-gate. Masking dense progress there would be arbitrary.
#
# Usage:
#   ./scripts/run/train_circle_r1d1_proxy.sh [max_iterations] [num_envs]

set -e

MAX_ITERS=${1:-3000}
NUM_ENVS=${2:-8192}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

REWARD_OVERRIDES='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5}'
ENV_OVERRIDES='{"track_name":"circle","replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"twr_randomization_pct":0.10,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'

echo "========================================"
echo "  Circle Proxy Training (R1+D1)"
echo "  Source logic: powerloop-r1d1-gate3mask"
echo "  Iterations:   $MAX_ITERS"
echo "  Envs:         $NUM_ENVS"
echo "  Reward:       $REWARD_OVERRIDES"
echo "  Env:          $ENV_OVERRIDES"
echo "========================================"

REWARD_OVERRIDES="$REWARD_OVERRIDES" \
ENV_OVERRIDES="$ENV_OVERRIDES" \
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    --logger wandb \
    --run_name "circle-r1d1-proxy"

#!/bin/bash

# Powerloop fullswitch speed-v1 candidate on real-effective TWR.
#
# Status: not recommended. User-reported batch eval degraded to 86.7% SR /
# 20.39s mean versus the TWR-1.87 base at 98.0% SR / 18.34s mean. Keep this
# script only as an ablation record; use train_powerloop_fullswitch_real_twr_speed_v2.sh
# for the next speed experiment.
#
# Starts from train_powerloop_fullswitch_real_twr.sh and changes only reward
# pressure:
#   - stronger dense progress and lap-complete pressure
#   - stronger per-step time pressure
#   - weaker action magnitude / smoothness penalties
#
# It deliberately keeps gate_side, Gate-3 progress mask, fullswitch semantics,
# and TWR at the proven real-effective values. This makes the first speed
# ablation easy to compare against the 98% SR twr1p87 base.
#
# Usage:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_speed.sh [max_iterations] [num_envs]

set -euo pipefail

MAX_ITERS=${1:-3000}
NUM_ENVS=${2:-8192}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REWARD_OVERRIDES='{"progress_goal_reward_scale":30.0,"lap_complete_reward_scale":0.75,"progress_skip_gate_indices":[3]}'
ENV_OVERRIDES='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'

echo "========================================"
echo "  Powerloop Training (fullswitch + real-effective TWR + speed)"
echo "  Iterations: $MAX_ITERS"
echo "  Envs:       $NUM_ENVS"
echo "  Reward:     $REWARD_OVERRIDES"
echo "  Env:        $ENV_OVERRIDES"
echo "  Extra:      REW_LAP_INCOMPLETE=-0.08 REW_CMD_REG_RP=-0.7 REW_CMD_REG_YAW=-0.35 REW_CMD_SMOOTHNESS=-0.05"
echo "========================================"

REW_LAP_INCOMPLETE=-0.08 \
REW_CMD_REG_RP=-0.7 \
REW_CMD_REG_YAW=-0.35 \
REW_CMD_SMOOTHNESS=-0.05 \
REWARD_OVERRIDES="$REWARD_OVERRIDES" \
ENV_OVERRIDES="$ENV_OVERRIDES" \
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    --logger wandb \
    --run_name "powerloop-r1d1-gate3mask-fullswitch-twr1p87-speed"

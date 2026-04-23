#!/bin/bash

# Powerloop fullswitch real-TWR reset ablation:
# keep the validated TWR-1.87 reward/dynamics, but make the non-spline linear
# fallback reset start with a small velocity toward the target gate.
#
# This is a clean reset-distribution ablation. It does NOT change reward,
# replay ratio, ground ratio, spline reset, or gate-switch logic.
#
# Usage:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_resetvel.sh [max_iterations] [num_envs] [seed]

set -euo pipefail

MAX_ITERS=${1:-3000}
NUM_ENVS=${2:-8192}
SEED=${3:-77}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REWARD_OVERRIDES='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5,"progress_skip_gate_indices":[3]}'
ENV_OVERRIDES='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30,"use_spline_reset":false,"linear_reset_vel_min":0.3,"linear_reset_vel_max":1.0}'

echo "========================================"
echo "  Powerloop Training (fullswitch + real-effective TWR + linear reset velocity)"
echo "  Iterations: $MAX_ITERS"
echo "  Envs:       $NUM_ENVS"
echo "  Seed:       $SEED"
echo "  Reward:     $REWARD_OVERRIDES"
echo "  Env:        $ENV_OVERRIDES"
echo "========================================"

REWARD_OVERRIDES="$REWARD_OVERRIDES" \
ENV_OVERRIDES="$ENV_OVERRIDES" \
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --seed "$SEED" \
    --headless \
    --logger wandb \
    --run_name "powerloop-r1d1-gate3mask-fullswitch-twr1p87-resetvel-seed${SEED}"

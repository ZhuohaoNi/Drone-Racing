#!/bin/bash

# Gate 3 switch-ablation training on top of the restored real-effective TWR base.
#
# Motivation:
#   The current powerloop fullswitch training still uses a sim-only Gate 3
#   approach gate (`approach_x_threshold=0.3`). In eval this can under-count
#   visibly valid Gate 3 passes and forces a conservative loop line.
#
# This script runs two clean ablations:
#   1. reduce the threshold to 0.1
#   2. disable the threshold entirely (0.0)
#
# Usage:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_gate3switch_ablation.sh [max_iterations] [num_envs] [eval_envs] [seed]

set -euo pipefail

MAX_ITERS=${1:-3000}
NUM_ENVS=${2:-8192}
EVAL_ENVS=${3:-1000}
SEED=${4:-42}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REWARD_BASE='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5,"progress_skip_gate_indices":[3]}'
ENV_BASE='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30,"backtrack_check_enabled":true}'
ENV_G3APP_0P1='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30,"approach_x_threshold":0.1,"backtrack_check_enabled":true}'
ENV_G3APP_0P0='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30,"approach_x_threshold":0.0,"backtrack_check_enabled":true}'

latest_run_for_name() {
    local run_name=$1
    local latest
    latest=$(ls -td logs/rsl_rl/quadcopter_direct/*_"$run_name" 2>/dev/null | head -n 1 || true)
    if [[ -z "$latest" ]]; then
        echo ""
    else
        basename "$latest"
    fi
}

run_experiment() {
    local run_name=$1
    local env_overrides=$2
    local description=$3

    echo ""
    echo "========================================"
    echo "  Starting: $run_name"
    echo "  Description: $description"
    echo "  Iterations: $MAX_ITERS"
    echo "  Envs:       $NUM_ENVS"
    echo "  Eval envs:  $EVAL_ENVS"
    echo "  Seed:       $SEED"
    echo "  Time:       $(date)"
    echo "  Reward:     $REWARD_BASE"
    echo "  Env:        $env_overrides"
    echo "========================================"

    unset REWARD_OVERRIDES
    unset ENV_OVERRIDES
    unset PPO_OVERRIDES

    REWARD_OVERRIDES="$REWARD_BASE" \
    ENV_OVERRIDES="$env_overrides" \
    python scripts/rsl_rl/train_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITERS" \
        --seed "$SEED" \
        --headless \
        --logger wandb \
        --run_name "$run_name"

    echo "  Finished training: $run_name at $(date)"

    if [[ "$EVAL_ENVS" -gt 0 ]]; then
        local latest_run
        latest_run=$(latest_run_for_name "$run_name")
        if [[ -z "$latest_run" ]]; then
            echo "  WARNING: Could not find latest run for $run_name; skipping eval."
            return 0
        fi

        echo ""
        echo "========================================"
        echo "  Evaluating: $latest_run"
        echo "========================================"
        ./scripts/run/eval_powerloop_real_twr.sh "$latest_run" best_model.pt "$EVAL_ENVS"
    fi
}

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-g3approach0p1-5000-seed${SEED}" \
    "$ENV_G3APP_0P1" \
    "Same TWR base, but relax Gate 3 approach gating from 0.3 to 0.1."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-g3approach0p0-5000-seed${SEED}" \
    "$ENV_G3APP_0P0" \
    "Same TWR base, but disable the Gate 3 approach gate entirely."

echo ""
echo "========================================"
echo "  All Gate 3 switch-ablation runs done"
echo "  Time: $(date)"
echo "========================================"

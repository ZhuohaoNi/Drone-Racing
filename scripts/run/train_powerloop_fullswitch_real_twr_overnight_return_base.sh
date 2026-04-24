#!/bin/bash

# Overnight return-to-base powerloop training.
#
# Current real result says the plain fullswitch real-effective TWR policy is
# still the best base. This script trains three clean 5000-iteration candidates:
#   1. base-5000: same TWR base, longer training
#   2. ground20:  same base, but ground_reset_ratio 0.20 instead of 0.10
#   3. gate0p8:   same base, but reward/obs gate_side 0.80 instead of 0.70
#
# The ground20 and gate0p8 variants are separate ablations, not cumulative.
# Here "ground20" is the cleanest first launch/takeoff robustness test:
# it increases the fraction of ground-start resets without changing rewards.
#
# Usage:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_overnight_return_base.sh [max_iterations] [num_envs] [eval_envs] [seed]
#
# Examples:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_overnight_return_base.sh
#   ./scripts/run/train_powerloop_fullswitch_real_twr_overnight_return_base.sh 5000 8192 1000 42
#   ./scripts/run/train_powerloop_fullswitch_real_twr_overnight_return_base.sh 5000 8192 0 42

set -euo pipefail

MAX_ITERS=${1:-5000}
NUM_ENVS=${2:-8192}
EVAL_ENVS=${3:-1000}
SEED=${4:-42}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REWARD_BASE='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5,"progress_skip_gate_indices":[3]}'

ENV_BASE='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'
ENV_GROUND20='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.20,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'
ENV_GATE0P8='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"gate_side":0.8,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'

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
    local eval_extra_env=${4:-}

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
        if [[ -n "$eval_extra_env" ]]; then
            EXTRA_ENV_OVERRIDES="$eval_extra_env" \
                ./scripts/run/eval_powerloop_real_twr.sh "$latest_run" best_model.pt "$EVAL_ENVS"
        else
            ./scripts/run/eval_powerloop_real_twr.sh "$latest_run" best_model.pt "$EVAL_ENVS"
        fi
    fi
}

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-5000-seed${SEED}" \
    "$ENV_BASE" \
    "Return to the current best real policy logic; only extend training to 5000 iterations."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-ground20-5000-seed${SEED}" \
    "$ENV_GROUND20" \
    "Same base, but increase ground-start reset coverage from 0.10 to 0.20."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-gate0p8-5000-seed${SEED}" \
    "$ENV_GATE0P8" \
    "Same base, but train with gate_side 0.80 instead of 0.70." \
    '{"gate_side":0.8}'

echo ""
echo "========================================"
echo "  All return-to-base overnight runs done"
echo "  Time: $(date)"
echo "========================================"

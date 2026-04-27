#!/bin/bash

# Final pre-real-flight powerloop candidates on the fullswitch real-TWR base.
#
# This script is intentionally conservative:
#   - no Gate-3 detector changes
#   - no dense velocity reward
#   - no spline moving-state resets
#   - no real-start-only narrowing
#
# It combines only the ablations with useful evidence so far, using tighter
# dynamics DR than the older training scripts:
#   1. ground20-linearvel0p6:
#        no controller change; smaller version of the linear reset velocity
#        idea that was fast but had a tail regression at 1.2 m/s.
#   2. ground20-twr1p99:
#        no controller change; center the nominal TWR on the Apr22 b5 estimate
#        with tighter DR.
#   3. gate0p8-tightdr:
#        previous pure gate0p8 already worked well; this reruns that direction
#        with tighter DR. Requires real observation gate_side = 0.8.
#   4. gate0p8-linearvel0p6-tightdr:
#        high-upside 0.8 version plus smaller linear reset velocity. Requires
#        real observation gate_side = 0.8.
#
# Usage:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_final_attempt.sh [max_iterations] [num_envs] [eval_envs] [seed]
#   Set eval_envs=0 to skip automatic eval after each training run.
#   Optional env:
#     MAX_EVAL_CANDIDATES=8  number of saved checkpoints to batch-eval per run
#     MIN_EVAL_ITER=2000     ignore early model_*.pt candidates except best_model.pt
#     REAL_GROUND_ONLY=1     final selection eval starts all envs from ground
#
# Selection rule before real flight:
#   Prefer candidates with eval_powerloop_real_twr.sh SR >= 98.5%, mean <= 18.65s,
#   std <= 0.5s, and no ugly worst-case tail. A faster best-case alone is not a
#   reason to fly the policy.

set -euo pipefail

MAX_ITERS=${1:-5000}
NUM_ENVS=${2:-8192}
EVAL_ENVS=${3:-1000}
SEED=${4:-42}
MAX_EVAL_CANDIDATES=${MAX_EVAL_CANDIDATES:-8}
MIN_EVAL_ITER=${MIN_EVAL_ITER:-2000}
REAL_GROUND_ONLY=${REAL_GROUND_ONLY:-1}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REWARD_BASE='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5,"progress_skip_gate_indices":[3]}'

# Tight DR for a fixed real vehicle: narrower than the historical base, but not
# zero. Keep action delay randomized because batch backtests showed fixed
# nominal profiles are too easy.
ENV_G20_LINEARVEL0P6='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.04,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.20,"use_spline_reset":false,"linear_reset_vel_min":0.2,"linear_reset_vel_max":0.6,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.8,"aero_randomization_scale_max":1.2,"pid_kpki_randomization_pct":0.08,"pid_kd_randomization_pct":0.15}'
ENV_G20_TWR1P99='{"track_name":"powerloop","thrust_to_weight":1.99,"twr_randomization_pct":0.04,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.20,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.8,"aero_randomization_scale_max":1.2,"pid_kpki_randomization_pct":0.08,"pid_kd_randomization_pct":0.15}'
ENV_GATE0P8_TIGHTDR='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.04,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"gate_side":0.8,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.8,"aero_randomization_scale_max":1.2,"pid_kpki_randomization_pct":0.08,"pid_kd_randomization_pct":0.15}'
ENV_GATE0P8_LINEARVEL0P6_TIGHTDR='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.04,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"gate_side":0.8,"use_spline_reset":false,"linear_reset_vel_min":0.2,"linear_reset_vel_max":0.6,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.8,"aero_randomization_scale_max":1.2,"pid_kpki_randomization_pct":0.08,"pid_kd_randomization_pct":0.15}'

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
        echo "  Checkpoint selection: batch-eval candidates, not training best_model only"
        echo "========================================"
        if [[ -n "$eval_extra_env" ]]; then
            REAL_GROUND_ONLY="$REAL_GROUND_ONLY" \
            EXTRA_ENV_OVERRIDES="$eval_extra_env" \
                ./scripts/run/eval_powerloop_checkpoint_candidates.sh \
                    "$latest_run" "$EVAL_ENVS" "$MAX_EVAL_CANDIDATES" "$MIN_EVAL_ITER"
        else
            REAL_GROUND_ONLY="$REAL_GROUND_ONLY" \
                ./scripts/run/eval_powerloop_checkpoint_candidates.sh \
                "$latest_run" "$EVAL_ENVS" "$MAX_EVAL_CANDIDATES" "$MIN_EVAL_ITER"
        fi
    fi
}

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-final-tightdr-g20-linearvel0p6-${MAX_ITERS}-seed${SEED}" \
    "$ENV_G20_LINEARVEL0P6" \
    "No controller change. Tight DR, ground20, and smaller 0.2-0.6 m/s target-directed linear-reset velocity."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p99-final-tightdr-g20-${MAX_ITERS}-seed${SEED}" \
    "$ENV_G20_TWR1P99" \
    "No controller change. Tight DR and ground20 with nominal TWR centered on Apr22 b5 estimate 1.99."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-final-tightdr-gate0p8-${MAX_ITERS}-seed${SEED}" \
    "$ENV_GATE0P8_TIGHTDR" \
    "Requires real observation gate_side 0.8. Pure gate0p8 rerun with tighter DR, matching the previous best 0.8 direction." \
    '{"gate_side":0.8}'

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-final-tightdr-gate0p8-linearvel0p6-${MAX_ITERS}-seed${SEED}" \
    "$ENV_GATE0P8_LINEARVEL0P6_TIGHTDR" \
    "Requires real observation gate_side 0.8. High-upside pure 0.8 candidate plus smaller linear reset velocity." \
    '{"gate_side":0.8}'

echo ""
echo "========================================"
echo "  All final-attempt candidates done"
echo "  Time: $(date)"
echo "========================================"

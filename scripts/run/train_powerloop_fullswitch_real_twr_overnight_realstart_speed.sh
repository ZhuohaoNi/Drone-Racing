#!/bin/bash

# Overnight reset-distribution speed ablations for the powerloop fullswitch real-TWR base.
#
# Gate3 approach-gating ablations did not beat the restored TWR base. This script
# keeps the same reward and controller-facing gate semantics, then tests reset
# distributions that better match real deployment or the slow real segments.
#
# Four independent candidates:
#   1. ground20-realstart60: more real launch-pose coverage inside ground resets
#   2. ground20-postloopfocus: oversample linear resets targeting gates 5,6,0
#   3. ground20-linearvel: add modest target-directed velocity to linear resets
#   4. ground20-splinevel: use moving spline resets with real-speed tangent velocity
#
# Usage:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_overnight_realstart_speed.sh [max_iterations] [num_envs] [eval_envs] [seed]
#   Set eval_envs=0 to skip automatic eval after each training run.

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

ENV_REALSTART='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.20,"real_start_reset_ratio":0.60,"real_start_x_local_min":-2.2,"real_start_x_local_max":-1.2,"real_start_y_local_min":-0.25,"real_start_y_local_max":0.25,"real_start_yaw_noise":0.08,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'
ENV_POSTLOOPFOCUS='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.20,"segment_focus_reset_ratio":0.50,"segment_focus_gate_indices":[5,6,0],"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'
ENV_LINEARVEL='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.20,"use_spline_reset":false,"linear_reset_vel_min":0.3,"linear_reset_vel_max":1.2,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'
ENV_SPLINEVEL='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.20,"use_spline_reset":true,"spline_vel_min":1.0,"spline_vel_max":3.0,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'

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
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-ground20-realstart60-${MAX_ITERS}-seed${SEED}" \
    "$ENV_REALSTART" \
    "Ground20 plus real-start launch-pose coverage; targets deployment takeoff and official start-to-Gate0 time."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-ground20-postloopfocus-${MAX_ITERS}-seed${SEED}" \
    "$ENV_POSTLOOPFOCUS" \
    "Ground20 plus more fallback resets on real slow/fragile target gates 5,6,0."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-ground20-linearvel-${MAX_ITERS}-seed${SEED}" \
    "$ENV_LINEARVEL" \
    "Ground20 plus modest target-directed velocity on linear fallback resets."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-ground20-splinevel-${MAX_ITERS}-seed${SEED}" \
    "$ENV_SPLINEVEL" \
    "Ground20 plus moving spline resets with 1.0-3.0 m/s tangent velocity."

echo ""
echo "========================================"
echo "  All real-start/reset-speed overnight runs done"
echo "  Time: $(date)"
echo "========================================"

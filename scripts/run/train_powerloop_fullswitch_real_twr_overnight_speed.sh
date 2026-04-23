#!/bin/bash

# Overnight sequential speed ablations for the powerloop fullswitch real-TWR base.
#
# The current best deployment base is:
#   powerloop-r1d1-gate3mask-fullswitch-twr1p87
#
# These three candidates isolate one hypothesis at a time:
#   v3: weaker/narrower current+next velocity reward on target gates 5,6 only
#   v4: pure current-gate velocity reward on target gates 5,6,0
#   v5: no velocity reward; only mild time pressure
#
# By default this trains from scratch for apples-to-apples comparison with the
# existing base/speed-v1/speed-v2 runs. To fine-tune from a checkpoint instead:
#
#   RESUME_RUN=2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87 \
#   RESUME_CHECKPOINT=best_model.pt \
#   ./scripts/run/train_powerloop_fullswitch_real_twr_overnight_speed.sh 1000 8192 1000
#
# Usage:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_overnight_speed.sh [max_iterations] [num_envs] [eval_envs]
#   Set eval_envs=0 to skip automatic batch eval after each training run.

set -euo pipefail

MAX_ITERS=${1:-3000}
NUM_ENVS=${2:-8192}
EVAL_ENVS=${3:-1000}
RESUME_RUN=${RESUME_RUN:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-best_model.pt}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

COMMON_ENV='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'

REWARD_V3='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5,"progress_skip_gate_indices":[3],"vel_toward_gate_reward_scale":0.5,"vel_reward_gate_indices":[5,6],"vel_reward_min_gates_passed":1,"vel_reward_next_weight":0.375,"vel_reward_clamp_min":-2.0,"vel_reward_clamp_max":8.0}'
REWARD_V4='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5,"progress_skip_gate_indices":[3],"vel_toward_gate_reward_scale":0.5,"vel_reward_gate_indices":[5,6,0],"vel_reward_min_gates_passed":1,"vel_reward_next_weight":0.0,"vel_reward_clamp_min":-2.0,"vel_reward_clamp_max":8.0}'
REWARD_V5='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5,"progress_skip_gate_indices":[3],"lap_incomplete_penalty_scale":-0.06}'

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
    local reward_overrides=$2
    local description=$3
    local resume_args=()

    echo ""
    echo "========================================"
    echo "  Starting: $run_name"
    echo "  Description: $description"
    echo "  Iterations: $MAX_ITERS"
    echo "  Envs:       $NUM_ENVS"
    echo "  Eval envs:  $EVAL_ENVS"
    echo "  Time:       $(date)"
    echo "  Reward:     $reward_overrides"
    echo "  Env:        $COMMON_ENV"
    echo "========================================"

    unset REWARD_OVERRIDES
    unset ENV_OVERRIDES
    unset PPO_OVERRIDES

    if [[ -n "$RESUME_RUN" ]]; then
        resume_args=(--resume True --load_run "$RESUME_RUN" --checkpoint "$RESUME_CHECKPOINT")
        echo "  Resume:     $RESUME_RUN / $RESUME_CHECKPOINT"
    fi

    REWARD_OVERRIDES="$reward_overrides" \
    ENV_OVERRIDES="$COMMON_ENV" \
    python scripts/rsl_rl/train_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITERS" \
        --headless \
        --logger wandb \
        --run_name "$run_name" \
        "${resume_args[@]}"

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
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv3-vel05-g56" \
    "$REWARD_V3" \
    "Weaker/narrower v2: target only 4->5 and 5->6 via idx_wp 5,6."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv4-currentvel-g560" \
    "$REWARD_V4" \
    "Post-loop speed without next-gate corner cutting; includes 6->0 as pure current target."

run_experiment \
    "powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv5-time06" \
    "$REWARD_V5" \
    "No velocity reward; isolate mild time pressure while keeping control penalties unchanged."

echo ""
echo "========================================"
echo "  All overnight speed ablations done"
echo "  Time: $(date)"
echo "========================================"

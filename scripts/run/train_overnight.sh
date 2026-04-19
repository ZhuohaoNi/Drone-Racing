#!/bin/bash
# Overnight sequential powerloop runs.
# Keeps the fixed baseline as a separate script and sequences the next three
# paper-inspired experiments:
#   R1   reward-only: gate-progress dense term + small lap bonus
#   D1   reset/DR-only: staged replay + moderate dynamics randomization
#   R1D1 combined
#
# Usage:
#   ./scripts/run/train_overnight.sh [max_iterations] [num_envs]

set -e

MAX_ITERS=${1:-5000}
NUM_ENVS=${2:-8192}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

POWERLOOP_ENV='{"track_name":"powerloop"}'
R1_REWARD='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5}'
D1_ENV='{"track_name":"powerloop","replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"twr_randomization_pct":0.10,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'

run_experiment() {
    local run_name=$1
    local reward_overrides=$2
    local env_overrides=$3
    echo ""
    echo "========================================"
    echo "  Starting: $run_name"
    echo "  Iterations: $MAX_ITERS | Envs: $NUM_ENVS"
    echo "  $(date)"
    echo "========================================"

    unset REWARD_OVERRIDES
    unset ENV_OVERRIDES
    if [[ -n "$reward_overrides" ]]; then
        export REWARD_OVERRIDES="$reward_overrides"
        echo "  REWARD_OVERRIDES=$REWARD_OVERRIDES"
    fi
    if [[ -n "$env_overrides" ]]; then
        export ENV_OVERRIDES="$env_overrides"
        echo "  ENV_OVERRIDES=$ENV_OVERRIDES"
    fi

    python scripts/rsl_rl/train_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITERS" \
        --headless \
        --logger wandb \
        --run_name "$run_name"
    echo "  Finished: $run_name at $(date)"
}

# ── Experiment 1: R1 reward-only ───────────────────────────────────────────────
run_experiment "powerloop-r1-gateprogress" "$R1_REWARD" "$POWERLOOP_ENV"

# ── Experiment 2: D1 reset+DR-only ────────────────────────────────────────────
run_experiment "powerloop-d1-stagedreset-dr" "" "$D1_ENV"

# ── Experiment 3: R1 + D1 ─────────────────────────────────────────────────────
run_experiment "powerloop-r1d1-gateprogress-stagedreset" "$R1_REWARD" "$D1_ENV"

echo ""
echo "========================================"
echo "  All overnight powerloop experiments done!"
echo "  $(date)"
echo "========================================"

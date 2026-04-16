#!/bin/bash
# Overnight sequential ablation runs.
# Runs the fixed baseline first, then a small set of post-fix reward ablations.
#
# Usage:
#   ./scripts/run/train_overnight.sh [max_iterations] [num_envs]
#
# Experiments:
#   1. circle-fixed-baseline      post-fix baseline with light smoothness (-0.1)
#   2. circle-fixed-nosmooth      remove smoothness to isolate its benefit
#   3. circle-fixed-smooth2       stronger smoothness penalty (-0.2)
#   4. circle-fixed-v7-rebalance  Pasumarti-aligned cmd_reg ratio: rp=-2.0, yaw=-0.05

set -e

MAX_ITERS=${1:-3000}
NUM_ENVS=${2:-8192}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

run_experiment() {
    local run_name=$1
    local extra_env=$2
    echo ""
    echo "========================================"
    echo "  Starting: $run_name"
    echo "  Iterations: $MAX_ITERS | Envs: $NUM_ENVS"
    echo "  $(date)"
    echo "========================================"
    env $extra_env python scripts/rsl_rl/train_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITERS" \
        --headless \
        --logger wandb \
        --run_name "$run_name"
    echo "  Finished: $run_name at $(date)"
}

# ── Experiment 1: Fixed baseline (default config) ──────────────────────────────
run_experiment "circle-fixed-baseline" ""

# ── Experiment 2: Baseline without smoothness penalty ─────────────────────────
run_experiment "circle-fixed-nosmooth" "REW_CMD_SMOOTHNESS=0.0"

# ── Experiment 3: Baseline with stronger smoothness penalty ───────────────────
run_experiment "circle-fixed-smooth2" "REW_CMD_SMOOTHNESS=-0.2"

# ── Experiment 4: Pasumarti-aligned cmd_reg rebalance ─────────────────────────
# Keeps the fixed baseline and changes the roll/pitch vs yaw penalty ratio to match
# Pasumarti 2026 Eq. 5: rp=-2.0 (2x stronger), yaw=-0.05 (10x weaker).
# Rationale: V2/V3/V4 real body rates (120-160 rad/s) are dominated by roll/pitch
# during gate passage; yaw rate stays low. Current yaw=-0.5 is over-penalizing a
# non-issue and may be crowding out rp exploration.
run_experiment "circle-fixed-v7-rebalance" "REW_CMD_REG_RP=-2.0 REW_CMD_REG_YAW=-0.05"

echo ""
echo "========================================"
echo "  All overnight experiments done!"
echo "  $(date)"
echo "========================================"

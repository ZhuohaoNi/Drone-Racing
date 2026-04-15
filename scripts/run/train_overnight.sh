#!/bin/bash
# Overnight sequential ablation runs.
# Runs multiple Circle-V6 variants back-to-back, each with different hyperparameters.
#
# Usage:
#   ./scripts/run/train_overnight.sh [max_iterations] [num_envs]
#
# Experiments:
#   1. circle-v6          gate_side=0.7, no smoothness (baseline)
#   2. circle-v6-smooth   gate_side=0.7, delta action penalty -0.1
#   3. circle-v6-smooth2  gate_side=0.7, delta action penalty -0.2
#   4. circle-v7-rebalance  Pasumarti-aligned cmd_reg ratio: rp=-2.0, yaw=-0.05

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

# ── Experiment 1: V6 baseline (gate_side=0.7, no smoothness penalty) ──────────
run_experiment "circle-v6" ""

# ── Experiment 2: V6 + light smoothness penalty ───────────────────────────────
run_experiment "circle-v6-smooth" "REW_CMD_SMOOTHNESS=-0.1"

# ── Experiment 3: V6 + stronger smoothness penalty ────────────────────────────
run_experiment "circle-v6-smooth2" "REW_CMD_SMOOTHNESS=-0.2"

# ── Experiment 4: V7 Pasumarti-aligned cmd_reg rebalance ──────────────────────
# Keeps V6's gate_side=0.7. Changes the roll/pitch vs yaw penalty ratio to match
# Pasumarti 2026 Eq. 5: rp=-2.0 (2x stronger), yaw=-0.05 (10x weaker).
# Rationale: V2/V3/V4 real body rates (120-160 rad/s) are dominated by roll/pitch
# during gate passage; yaw rate stays low. Current yaw=-0.5 is over-penalizing a
# non-issue and may be crowding out rp exploration.
run_experiment "circle-v7-rebalance" "REW_CMD_REG_RP=-2.0 REW_CMD_REG_YAW=-0.05"

echo ""
echo "========================================"
echo "  All overnight experiments done!"
echo "  $(date)"
echo "========================================"

#!/bin/bash

# Robustness sweep: per-axis DR sensitivity for a fixed policy.
#
# Sweeps TWR, aero, PID gains, motor tau, and action latency one axis at a
# time, each with envs partitioned across the value grid. Outputs CSV + PNG
# per axis under logs/.../robustness_sweep/<ckpt>/.
#
# Usage:
#   ./scripts/run/robustness_sweep.sh <load_run> [checkpoint] [num_envs] [max_steps]
#
# Example:
#   ./scripts/run/robustness_sweep.sh 2026-04-19_14-30-00_powerloop-r1d1-gate3mask

set -e

LOAD_RUN=${1:?"Usage: $0 <load_run> [checkpoint] [num_envs] [max_steps]"}
CHECKPOINT=${2:-"best_model.pt"}
NUM_ENVS=${3:-600}
MAX_STEPS=${4:-3000}

AXES=${AXES:-"twr,k_aero_xy,k_aero_z,kp_omega_rp,kd_omega_rp,motor_tau,action_latency,action_noise_std,gate_pos_jitter"}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Default to powerloop track for the sim2real rig; override via ENV_OVERRIDES.
: "${ENV_OVERRIDES:={\"track_name\":\"powerloop\"}}"
export ENV_OVERRIDES

echo "========================================"
echo "  Robustness sweep"
echo "  Run:        $LOAD_RUN"
echo "  Checkpoint: $CHECKPOINT"
echo "  Envs:       $NUM_ENVS  |  Max steps: $MAX_STEPS"
echo "  Axes:       $AXES"
echo "  Env:        $ENV_OVERRIDES"
echo "========================================"

EXTRA_ARGS=()
if [[ -n "${GATE_SIDE:-}" ]]; then
    EXTRA_ARGS+=(--gate_side "$GATE_SIDE")
fi
if [[ "${TRAIN_MODE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--train_mode)
fi
if [[ "${ENABLE_BACKTRACK:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--enable_backtrack_check)
fi

python scripts/rsl_rl/robustness_sweep.py \
    --task Isaac-Quadcopter-Race-v0 \
    --load_run "$LOAD_RUN" \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --max_steps "$MAX_STEPS" \
    --axes "$AXES" \
    --headless \
    "${EXTRA_ARGS[@]}"

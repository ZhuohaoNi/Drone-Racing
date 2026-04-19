#!/bin/bash

# Deterministic robustness sweep for deployment mismatch evaluation.
#
# This version runs explicit mismatch tests instead of mixed "stress profiles":
# - control mismatch: thrust scale, fixed action latency, body-rate gain scale
# - observation mismatch: velocity noise/bias, yaw bias, fixed obs delay
#
# Usage:
#   ./scripts/run/robustness_sweep.sh <load_run> [checkpoint] [num_envs] [num_trials] [max_steps]
#
# Example:
#   ./scripts/run/robustness_sweep.sh 2026-04-19_14-30-00_powerloop-r1d1-gate3mask
#
# Optional env vars:
#   ENV_OVERRIDES      Base JSON overrides merged into every scenario.
#                      Default: {"track_name":"circle"}
#   SCENARIOS          Comma-separated scenario list.
#                      Default: the built-in deterministic control/observation scenarios
#   NUM_PARAMS_PER_ENV Forwarded to batch_eval_race.py. Default: 0
#   SEED               Default: 42
#   DEVICE             Optional, e.g. cuda:0
#   OUTPUT_DIR         Optional output root override

set -euo pipefail

LOAD_RUN=${1:?"Usage: $0 <load_run> [checkpoint] [num_envs] [num_trials] [max_steps]"}
CHECKPOINT=${2:-best_model.pt}
NUM_ENVS=${3:-256}
NUM_TRIALS=${4:-3}
MAX_STEPS=${5:-3000}

SCENARIOS=${SCENARIOS:-"control_nominal,control_thrust_0p90,control_thrust_1p10,control_latency_20ms,control_latency_40ms,control_rate_gain_0p85,control_rate_gain_1p15,obs_vel_noise_0p05,obs_vel_bias_0p20,obs_yaw_bias_5deg,obs_gate_bias_5cm,obs_delay_20ms,obs_delay_40ms"}
NUM_PARAMS_PER_ENV=${NUM_PARAMS_PER_ENV:-0}
SEED=${SEED:-42}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${ENV_OVERRIDES:-}" ]]; then
    ENV_OVERRIDES='{"track_name":"powerloop"}'
fi
export ENV_OVERRIDES

EXTRA_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
    EXTRA_ARGS+=(--device "$DEVICE")
fi
if [[ "${DISABLE_FABRIC:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--disable_fabric)
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    EXTRA_ARGS+=(--output_dir "$OUTPUT_DIR")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry_run)
fi

echo "========================================"
echo "  Deterministic Robustness Sweep"
echo "  Run:               $LOAD_RUN"
echo "  Checkpoint:        $CHECKPOINT"
echo "  Envs:              $NUM_ENVS"
echo "  Trials/profile:    $NUM_TRIALS"
echo "  Max steps:         $MAX_STEPS"
echo "  Scenarios:         $SCENARIOS"
echo "  num_params_per_env $NUM_PARAMS_PER_ENV"
echo "  Base env:          $ENV_OVERRIDES"
echo "========================================"

python scripts/rsl_rl/robustness_sweep.py \
    --load_run "$LOAD_RUN" \
    --checkpoint "$CHECKPOINT" \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --num_trials "$NUM_TRIALS" \
    --max_steps "$MAX_STEPS" \
    --seed "$SEED" \
    --scenarios "$SCENARIOS" \
    --num_params_per_env "$NUM_PARAMS_PER_ENV" \
    "${EXTRA_ARGS[@]}"

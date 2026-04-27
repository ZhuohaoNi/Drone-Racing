#!/bin/bash

# Evaluate a powerloop policy under the real-effective TWR candidate dynamics.
#
# This matches scripts/run/train_powerloop_fullswitch_real_twr.sh:
#   - track_name = powerloop
#   - thrust_to_weight = 1.87
#   - no extra motor-tau/drag changes
#
# Eval-specific controller-match notes:
#   - target switching should behave like the deployed controller, so disable
#     the sim-only Gate 3 approach gate and previous-gate backtrack kill
#   - keep the rest of the real-effective dynamics unchanged
#
# Usage:
#   ./scripts/run/eval_powerloop_real_twr.sh <run_dir> [checkpoint] [num_envs] [follow_robot]
#
# Optional env:
#   BATCH_EVAL_OUTPUT_DIR=<path>  override batch_eval_race.py output directory
#   REAL_GROUND_ONLY=0            use the old mixed-reset robustness eval

set -euo pipefail

RUN_DIR=${1:?"ERROR: Provide run directory as first argument"}
CHECKPOINT=${2:-best_model.pt}
NUM_ENVS=${3:-1}
FOLLOW_ROBOT=${4:--1}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# make usre backtrack_check_enabled is False
ENV_OVERRIDES='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.0,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"approach_x_threshold":0.0,"backtrack_check_enabled":false}'
REAL_GROUND_ONLY=${REAL_GROUND_ONLY:-1}
EVAL_PROFILE="mixed"
if [[ "$REAL_GROUND_ONLY" == "1" ]]; then
    EVAL_PROFILE="ground_only"
    ENV_OVERRIDES=$(
        ENV_OVERRIDES="$ENV_OVERRIDES" python - <<'PY'
import json
import os

base = json.loads(os.environ["ENV_OVERRIDES"])
base.update({
    "ground_reset_ratio": 1.0,
    "replay_reset_ratio": 0.0,
    "real_start_reset_ratio": 1.0,
})
print(json.dumps(base, separators=(",", ":")))
PY
    )
fi
if [[ -n "${EXTRA_ENV_OVERRIDES:-}" ]]; then
    ENV_OVERRIDES=$(
        ENV_OVERRIDES="$ENV_OVERRIDES" EXTRA_ENV_OVERRIDES="$EXTRA_ENV_OVERRIDES" python - <<'PY'
import json
import os

base = json.loads(os.environ["ENV_OVERRIDES"])
extra = json.loads(os.environ["EXTRA_ENV_OVERRIDES"])
base.update(extra)
print(json.dumps(base, separators=(",", ":")))
PY
    )
fi

if [ "$NUM_ENVS" -eq 1 ]; then
    echo "========================================"
    echo "  Powerloop Real-TWR Play (with video)"
    echo "  Run dir:    $RUN_DIR"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Profile:    $EVAL_PROFILE"
    echo "  Env:        $ENV_OVERRIDES"
    echo "========================================"

    ENV_OVERRIDES="$ENV_OVERRIDES" python scripts/rsl_rl/play_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --num_envs 1 \
        --load_run "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --follow_robot "$FOLLOW_ROBOT" \
        --headless \
        --video \
        --video_length 1500
else
    if [[ -z "${BATCH_EVAL_OUTPUT_DIR:-}" && "$EVAL_PROFILE" == "ground_only" ]]; then
        checkpoint_name=$(basename "$CHECKPOINT")
        checkpoint_name=${checkpoint_name%.pt}
        if [[ "$RUN_DIR" == /* || "$RUN_DIR" == */* ]]; then
            run_log_dir="$RUN_DIR"
        else
            run_log_dir="logs/rsl_rl/quadcopter_direct/$RUN_DIR"
        fi
        BATCH_EVAL_OUTPUT_DIR="$run_log_dir/batch_eval/${checkpoint_name}_ground_only"
    fi

    echo "========================================"
    echo "  Powerloop Real-TWR Batch Evaluation"
    echo "  Run dir:    $RUN_DIR"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Envs:       $NUM_ENVS"
    echo "  Profile:    $EVAL_PROFILE"
    echo "  Env:        $ENV_OVERRIDES"
    if [[ -n "${BATCH_EVAL_OUTPUT_DIR:-}" ]]; then
        echo "  Output dir: $BATCH_EVAL_OUTPUT_DIR"
    fi
    echo "========================================"

    extra_args=()
    if [[ -n "${BATCH_EVAL_OUTPUT_DIR:-}" ]]; then
        extra_args+=(--output_dir "$BATCH_EVAL_OUTPUT_DIR")
    fi

    ENV_OVERRIDES="$ENV_OVERRIDES" python scripts/rsl_rl/batch_eval_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --load_run "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --headless \
        --num_trials 1 \
        --num_envs "$NUM_ENVS" \
        --num_params_per_env 0 \
        --max_steps 3000 \
        "${extra_args[@]}"
fi

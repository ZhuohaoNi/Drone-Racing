#!/bin/bash

# Batch self-evaluation script: TA-style parameter randomization
#
# Runs N trials, each with 100 parallel envs.
# Each trial randomly picks 3 parameters from the TA pool, samples them
# within the official bounds (per project_description.md Section 3.1),
# and evaluates the policy.
# Reports mean 3-lap time and success rate per trial, then saves summary
# charts (per-trial bar charts, parameter analysis, lap distribution).
#
# Usage:
#   ./scripts/run/batch_eval.sh <run_dir> [num_trials] [num_envs] [max_steps]
#
# Examples:
#   ./scripts/run/batch_eval.sh 2026-03-31_12-00-00
#   ./scripts/run/batch_eval.sh 2026-03-31_12-00-00 10 100 3000
#
# Charts are saved to:
#   logs/rsl_rl/quadcopter_direct/<run_dir>/batch_eval/

set -e

RUN_DIR=${1:-"[YYYY-MM-DD_XX-XX-XX]"}
NUM_TRIALS=${2:-10}
NUM_ENVS=${3:-100}
MAX_STEPS=${4:-3000}

if [ "$RUN_DIR" == "[YYYY-MM-DD_XX-XX-XX]" ]; then
    echo "ERROR: No run directory provided!"
    echo "Usage: ./scripts/run/batch_eval.sh <run_dir> [num_trials] [num_envs] [max_steps]"
    echo "Example: ./scripts/run/batch_eval.sh 2026-03-31_12-00-00 10 100 3000"
    exit 1
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

# Change to project root automatically
cd "$(dirname "$0")/../.."

# Export PYTHONPATH so Python can find the 'src' module
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "========================================"
echo "  Batch Evaluation"
echo "  Run dir:    $RUN_DIR"
echo "  Trials:     $NUM_TRIALS"
echo "  Envs/trial: $NUM_ENVS"
echo "  Max steps:  $MAX_STEPS"
echo "========================================"

python scripts/rsl_rl/batch_eval_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --load_run "$RUN_DIR" \
    --checkpoint best_model.pt \
    --headless \
    --num_trials "$NUM_TRIALS" \
    --num_envs "$NUM_ENVS" \
    --max_steps "$MAX_STEPS"

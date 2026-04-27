#!/bin/bash

# Evaluate a powerloop policy with a rosbag/sysid-calibrated diagnostic profile.
#
# IMPORTANT:
#   Apr22 backtest showed this is NOT reliable as a deployment-ranking metric.
#   The default b5_apr22 profile correctly tests ground-start / TWR / fixed-lag
#   sensitivity, but it rated the real-failed speedv2 policy as fast and 100%
#   successful. Use this as a diagnostic slice, not as a replacement for real
#   bag validation or the reset-diverse robustness eval.
#
# This is intentionally different from eval_powerloop_real_twr.sh:
#   - no broad per-env dynamics DR
#   - ground-start only, with the narrow real launch-pose distribution
#   - no training-scale observation noise
#   - fixed action delay based on the clean Apr22 rate-lag estimate
#
# Profiles:
#   b5_apr22  : default; Apr22 drone estimate, TWR ~= 1.99, fixed 3-step delay
#   b3_apr20  : Apr20 drone estimate, TWR ~= 1.87, fixed 3-step delay
#   latency4  : Apr22 TWR with fixed 4-step delay stress test
#   legacy    : old real-TWR batch profile shape, for comparison
#
# Usage:
#   ./scripts/run/eval_powerloop_real_calibrated.sh <run_dir> [checkpoint] [num_envs] [profile] [follow_robot]
#
# Examples:
#   ./scripts/run/eval_powerloop_real_calibrated.sh 2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87 best_model.pt 1000
#   ./scripts/run/eval_powerloop_real_calibrated.sh <run> best_model.pt 1000 b3_apr20

set -euo pipefail

RUN_DIR=${1:?"ERROR: Provide run directory as first argument"}
CHECKPOINT=${2:-best_model.pt}
NUM_ENVS=${3:-1000}
PROFILE=${4:-b5_apr22}
FOLLOW_ROBOT=${5:--1}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

COMMON_REAL_START='"replay_reset_ratio":0.0,"ground_reset_ratio":1.0,"real_start_reset_ratio":1.0,"real_start_x_local_min":-2.2,"real_start_x_local_max":-1.2,"real_start_y_local_min":-0.25,"real_start_y_local_max":0.25,"real_start_yaw_noise":0.08,"staged_replay_reset":false,"use_spline_reset":false'
COMMON_NO_DR='"twr_randomization_pct":0.0,"mass_variation":0.0,"aero_randomization_scale_min":1.0,"aero_randomization_scale_max":1.0,"pid_kpki_randomization_pct":0.0,"pid_kd_randomization_pct":0.0,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0'
COMMON_OBS_SYSID='"obs_noise_std_scale":0.0,"obs_lin_vel_noise_std":0.005,"obs_lin_vel_bias":[0.001,-0.0006,0.0],"obs_gate_corner_bias":[0.0,0.0,0.0],"obs_latency_prob":0.0,"fixed_obs_delay_steps":0'
COMMON_GATE='"track_name":"powerloop","approach_x_threshold":0.0,"backtrack_check_enabled":false'

case "$PROFILE" in
    b5_apr22)
        # Apr22 b5 bags identify mass ~=60.2g and max thrust 1.174N:
        # TWR = 1.174 / (0.0602 * 9.81) ~= 1.99.
        # Clean rate-lag estimate is 60-70ms; at 50Hz policy rate, 3 steps ~=60ms.
        ENV_OVERRIDES="{${COMMON_GATE},\"thrust_to_weight\":1.99,${COMMON_NO_DR},\"action_latency_max\":3,\"fixed_action_delay_steps\":3,${COMMON_OBS_SYSID},${COMMON_REAL_START}}"
        ;;
    b3_apr20)
        # Apr20 b3 bags motivated the original TWR=1.87 profile.
        ENV_OVERRIDES="{${COMMON_GATE},\"thrust_to_weight\":1.87,${COMMON_NO_DR},\"action_latency_max\":3,\"fixed_action_delay_steps\":3,${COMMON_OBS_SYSID},${COMMON_REAL_START}}"
        ;;
    latency4)
        # Stress the upper side of the 60-70ms lag estimate.
        ENV_OVERRIDES="{${COMMON_GATE},\"thrust_to_weight\":1.99,${COMMON_NO_DR},\"action_latency_max\":4,\"fixed_action_delay_steps\":4,${COMMON_OBS_SYSID},${COMMON_REAL_START}}"
        ;;
    legacy)
        # Approximate the previous real-TWR eval shape. Kept for A/B only.
        ENV_OVERRIDES='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.0,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"approach_x_threshold":0.0,"backtrack_check_enabled":false}'
        ;;
    *)
        echo "ERROR: unknown profile '$PROFILE'. Use one of: b5_apr22, b3_apr20, latency4, legacy" >&2
        exit 2
        ;;
esac

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

echo "========================================"
echo "  Powerloop Real-Calibrated Eval"
echo "  Run dir:    $RUN_DIR"
echo "  Checkpoint: $CHECKPOINT"
echo "  Envs:       $NUM_ENVS"
echo "  Profile:    $PROFILE"
echo "  Env:        $ENV_OVERRIDES"
echo "========================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ENV_OVERRIDES="$ENV_OVERRIDES" python - <<'PY'
import json
import os

parsed = json.loads(os.environ["ENV_OVERRIDES"])
print(json.dumps(parsed, indent=2, sort_keys=True))
PY
    exit 0
fi

if [ "$NUM_ENVS" -eq 1 ]; then
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
    ENV_OVERRIDES="$ENV_OVERRIDES" python scripts/rsl_rl/batch_eval_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --load_run "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --headless \
        --num_trials 1 \
        --num_envs "$NUM_ENVS" \
        --num_params_per_env 0 \
        --max_steps 3000 \
        --output_dir "logs/rsl_rl/quadcopter_direct/${RUN_DIR}/batch_eval_real_calibrated/${PROFILE}/${CHECKPOINT%.pt}"
fi

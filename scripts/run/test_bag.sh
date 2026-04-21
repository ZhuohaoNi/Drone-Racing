#!/usr/bin/env bash
# Run bag analysis + lap-time computation on a recorded rosbag.
#
# Usage:
#   scripts/run/test_bag.sh <bag_name_or_path> [namespace] [num_laps]
#
# Defaults to real powerloop bag coordinates. Use TRACK=circle for old circle
# bags, TRACK=powerloop_sim for sim-frame powerloop bags, or TRACK=config to
# use the current sim2real config.yaml waypoints as-is.
#
# Examples:
#   scripts/run/test_bag.sh group30_cyclev2
#   scripts/run/test_bag.sh group30_cyclev3 crazy_jirl_b3 3
#   scripts/run/test_bag.sh /abs/path/to/some_bag crazy_jirl_b2
#   TRACK=circle scripts/run/test_bag.sh group30_cyclev2
set -eo pipefail

BAG_ARG="${1:?Usage: $0 <bag_name_or_path> [namespace] [num_laps]}"
NAMESPACE="${2:-crazy_jirl_b3}"
NUM_LAPS="${3:-3}"
TRACK="${TRACK:-powerloop}"

REPO="$HOME/Documents/ese6510/Drone-Racing-sim2real"
DEFAULT_BAG_ROOT="$HOME/Documents/ese6510/ese651_project/rosbags"
POWERLOOP_BAG_ROOT="$HOME/Documents/ese6510/ese651_project/rosbags_powerloop_baseline_controller_04_20"
if [[ "$TRACK" == powerloop* && -d "$POWERLOOP_BAG_ROOT" ]]; then
  DEFAULT_BAG_ROOT="$POWERLOOP_BAG_ROOT"
fi
BAG_ROOT="${BAG_ROOT:-$DEFAULT_BAG_ROOT}"

# Resolve bag path: absolute path as-is, otherwise look under BAG_ROOT.
if [[ "$BAG_ARG" = /* ]]; then
  BAG_PATH="$BAG_ARG"
else
  BAG_PATH="$BAG_ROOT/$BAG_ARG"
fi

if [[ ! -d "$BAG_PATH" && ! -f "$BAG_PATH" ]]; then
  echo "Bag path not found: $BAG_PATH" >&2
  exit 1
fi

case "$TRACK" in
  powerloop|powerloop_sim|circle|config) ;;
  *)
    echo "Unknown TRACK='$TRACK'. Use TRACK=powerloop, TRACK=powerloop_sim, TRACK=circle, or TRACK=config." >&2
    exit 1
    ;;
esac

# Drop conda so ROS2 (system python3.12) is used.
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  while [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; do conda deactivate; done
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.sh
# shellcheck disable=SC1091
source "$REPO/install/setup.sh"

cd "$REPO"

echo "=== Bag:       $BAG_PATH"
echo "=== Namespace: $NAMESPACE"
echo "=== Laps:      $NUM_LAPS"
echo "=== Track:     $TRACK"
echo

echo "--- Lap times ---"
python3 bin/lap_time.py "$BAG_PATH" "$NAMESPACE" "$NUM_LAPS" --track "$TRACK"
echo

echo "--- Full analysis (plots) ---"
python3 bin/process_bag_with_br_pos_export.py \
  "$BAG_PATH" "$NAMESPACE" \
  --track "$TRACK" \
  --skip-desired-trajectory \
  --no-animation \
  --no-show

#!/usr/bin/env bash
# Batch-run race timing/summary on every rosbag under a folder.
#
# This intentionally does not generate plots. It runs lap_time.py only and
# prints/writes the metrics that matter for post-real-test comparison.
#
# Usage:
#   scripts/run/batch_test_bags.sh <bag_root> [namespace] [num_laps]
#
# Examples:
#   scripts/run/batch_test_bags.sh rosbags_powerloop_baseline_controller_04_20
#   scripts/run/batch_test_bags.sh /abs/path/to/rosbags crazy_jirl_b3 3
#   TRACK=circle scripts/run/batch_test_bags.sh rosbags crazy_jirl_b2 3
set -eo pipefail

BAG_ROOT_ARG="${1:?Usage: $0 <bag_root> [namespace] [num_laps]}"
NAMESPACE="${2:-crazy_jirl_b3}"
NUM_LAPS="${3:-3}"
TRACK="${TRACK:-powerloop}"

PROJECT="$HOME/Documents/ese6510/ese651_project"
REPO="$HOME/Documents/ese6510/Drone-Racing-sim2real"

case "$TRACK" in
  powerloop|powerloop_sim|circle|config) ;;
  *)
    echo "Unknown TRACK='$TRACK'. Use TRACK=powerloop, TRACK=powerloop_sim, TRACK=circle, or TRACK=config." >&2
    exit 1
    ;;
esac

# Resolve bag root: absolute path as-is, otherwise relative to project.
if [[ "$BAG_ROOT_ARG" = /* ]]; then
  BAG_ROOT="$BAG_ROOT_ARG"
else
  BAG_ROOT="$PROJECT/$BAG_ROOT_ARG"
fi

if [[ ! -d "$BAG_ROOT" ]]; then
  echo "Directory not found: $BAG_ROOT" >&2
  exit 1
fi

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

CSV_OUT="$BAG_ROOT/summary.csv"
TMPDIR_BATCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BATCH"' EXIT

cat > "$CSV_OUT" <<'CSV'
bag,status,eval_first_n_laps_time,race_start_cmd_t,eval_first_n_laps_end_t,eval_first_n_laps_end_gate,complete_laps_expected_order,ordered_passes,total_valid_passes,total_near_misses,sequence_breaks,valid_per_gate,near_miss_per_gate,best_lap,mean_lap,median_lap,std_lap,fastest_n_lap_window_time,takeoff_to_n_laps_gate0_close,first_n_lap_sum_gate0_to_gate0,path_length_race,mean_speed,max_speed,mean_tilt,max_tilt,mean_body_rate_cmd,max_body_rate_cmd,mean_thrust,max_thrust,bag_path
CSV

# Discover bag directories directly under BAG_ROOT. Skip plots and directories
# without bag data. ROS2 bags are usually directories containing .mcap/.db3.
BAGS=()
for dir in "$BAG_ROOT"/*/; do
  [[ -d "$dir" ]] || continue
  dirname=$(basename "$dir")
  [[ "$dirname" == "plots" ]] && continue
  if compgen -G "$dir/*.mcap" >/dev/null || compgen -G "$dir/*.db3" >/dev/null || [[ -f "$dir/metadata.yaml" ]]; then
    BAGS+=("${dir%/}")
  fi
done

# Also allow a root folder containing .mcap/.db3 files directly.
if compgen -G "$BAG_ROOT/*.mcap" >/dev/null || compgen -G "$BAG_ROOT/*.db3" >/dev/null || [[ -f "$BAG_ROOT/metadata.yaml" ]]; then
  BAGS+=("$BAG_ROOT")
fi

if [[ ${#BAGS[@]} -eq 0 ]]; then
  echo "No ROS2 bag directories/files found under $BAG_ROOT" >&2
  exit 1
fi

IFS=$'\n' BAGS=($(sort -u <<<"${BAGS[*]}")); unset IFS

echo "=== Batch bag test ==="
echo "Root:      $BAG_ROOT"
echo "Bags:      ${#BAGS[@]}"
echo "Namespace: $NAMESPACE"
echo "Track:     $TRACK"
echo "Eval:      race-start command → lap $NUM_LAPS last gate"
echo

FAIL_COUNT=0

for bag_path in "${BAGS[@]}"; do
  bag_name=$(basename "$bag_path")
  safe_name=${bag_name//[^A-Za-z0-9_.-]/_}
  txt_out="$TMPDIR_BATCH/${safe_name}.txt"
  json_out="$TMPDIR_BATCH/${safe_name}.json"

  echo "--- Processing: $bag_name ---"
  if ! python3 bin/lap_time.py "$bag_path" "$NAMESPACE" "$NUM_LAPS" \
      --track "$TRACK" \
      --summary-json "$json_out" \
      > "$txt_out" 2>&1; then
    echo "  FAILED"
    python3 - "$CSV_OUT" "$bag_name" "$bag_path" <<'PY'
import csv
import sys

csv_path, bag_name, bag_path = sys.argv[1:4]
header = open(csv_path, newline="").readline().strip().split(",")
row = {key: "" for key in header}
row.update({"bag": bag_name, "status": "FAILED", "bag_path": bag_path})
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writerow(row)
PY
    ((FAIL_COUNT++)) || true
    continue
  fi

  python3 - "$CSV_OUT" "$json_out" "$bag_name" "$bag_path" <<'PY'
import csv
import json
import sys

csv_path, json_path, bag_name, bag_path = sys.argv[1:5]
data = json.load(open(json_path))
header = open(csv_path, newline="").readline().strip().split(",")

def val(key):
    value = data.get(key)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return value

row = {key: "" for key in header}
row.update({
    "bag": bag_name,
    "status": "OK",
    "bag_path": bag_path,
})
for key in header:
    if key in row and row[key] != "":
        continue
    row[key] = val(key)

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writerow(row)
PY

  eval_time=$(python3 - "$json_out" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
v = data.get("eval_first_n_laps_time")
laps = data.get("complete_laps_expected_order")
valid = data.get("total_valid_passes")
miss = data.get("total_near_misses")
print(("n/a" if v is None else f"{v:.3f}") + f" | laps={laps} valid={valid} miss={miss}")
PY
)
  echo "  Eval: $eval_time"
done

echo
echo "=== Summary CSV written to: $CSV_OUT ==="
echo

python3 - "$CSV_OUT" "$NUM_LAPS" <<'PY'
import csv
import math
import sys

csv_path, num_laps = sys.argv[1:3]
rows = list(csv.DictReader(open(csv_path, newline="")))

def as_float(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return math.inf

rows.sort(key=lambda r: (as_float(r, "eval_first_n_laps_time"), r["bag"]))

print(f"=== Summary Table (sorted by eval first {num_laps} laps) ===")
print()
header = (
    f"{'Bag':36s} {'Eval':>8s} {'Laps':>4s} {'Ord':>4s} {'Valid':>5s} "
    f"{'Miss':>5s} {'Brk':>4s} {'Best':>7s} {'Mean':>7s} {'Std':>6s} "
    f"{'vMax':>6s} {'tiltMax':>7s} {'brMax':>7s} {'valid/gate':>18s}"
)
print(header)
print("-" * len(header))
for row in rows:
    if row["status"] != "OK":
        print(f"{row['bag']:36s} {'FAILED':>8s}")
        continue
    print(
        f"{row['bag'][:36]:36s} "
        f"{(row['eval_first_n_laps_time'] or 'n/a'):>8s} "
        f"{row['complete_laps_expected_order']:>4s} "
        f"{row['ordered_passes']:>4s} "
        f"{row['total_valid_passes']:>5s} "
        f"{row['total_near_misses']:>5s} "
        f"{row['sequence_breaks']:>4s} "
        f"{(row['best_lap'] or 'n/a'):>7s} "
        f"{(row['mean_lap'] or 'n/a'):>7s} "
        f"{(row['std_lap'] or 'n/a'):>6s} "
        f"{(row['max_speed'] or 'n/a'):>6s} "
        f"{(row['max_tilt'] or 'n/a'):>7s} "
        f"{(row['max_body_rate_cmd'] or 'n/a'):>7s} "
        f"{row['valid_per_gate'][:18]:>18s}"
    )

print()
print("Eval = first N laps measured from first race command to the N-th lap's last gate.")
PY

if [[ $FAIL_COUNT -gt 0 ]]; then
  echo
  echo "$FAIL_COUNT bag(s) failed. Raw outputs kept in: $TMPDIR_BATCH"
  trap - EXIT
fi

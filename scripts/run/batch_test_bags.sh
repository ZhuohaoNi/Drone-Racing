#!/usr/bin/env bash
# Batch-run lap_time.py on all rosbag directories under a given root,
# skipping 'plots' folders. Produces a CSV summary + printed table.
#
# Usage:
#   scripts/run/batch_test_bags.sh <bag_root> [namespace] [num_laps]
#
# Examples:
#   scripts/run/batch_test_bags.sh rosbags_04_15
#   scripts/run/batch_test_bags.sh /abs/path/to/rosbags crazy_jirl_b3 3
#   scripts/run/batch_test_bags.sh rosbags crazy_jirl_b2
set -eo pipefail

BAG_ROOT_ARG="${1:?Usage: $0 <bag_root> [namespace] [num_laps]}"
NAMESPACE="${2:-crazy_jirl_b3}"
NUM_LAPS="${3:-3}"

PROJECT="$HOME/Documents/ese6510/ese651_project"
REPO="$HOME/Documents/ese6510/Drone-Racing-sim2real"

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

# Output CSV
CSV_OUT="$BAG_ROOT/summary.csv"
echo "bag,laps_strict,seq_breaks,best_lap,mean_lap,median_lap,std_lap,takeoff_to_3,fastest_3lap,path_per_lap,mean_speed,max_speed,mean_tilt,max_tilt,mean_br,max_br,mean_thrust,max_thrust" > "$CSV_OUT"

# Temp dir for per-bag output
TMPDIR_BATCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BATCH"' EXIT

# Discover bags: directories under BAG_ROOT that contain .mcap files, skip 'plots'
BAGS=()
for dir in "$BAG_ROOT"/*/; do
  dirname=$(basename "$dir")
  [[ "$dirname" == "plots" ]] && continue
  # Check that directory contains at least one .mcap file
  if ls "$dir"/*.mcap &>/dev/null; then
    BAGS+=("$dir")
  fi
done

if [[ ${#BAGS[@]} -eq 0 ]]; then
  echo "No rosbag directories found under $BAG_ROOT" >&2
  exit 1
fi

# Sort alphabetically
IFS=$'\n' BAGS=($(sort <<<"${BAGS[*]}")); unset IFS

echo "=== Batch analysis: ${#BAGS[@]} bags under $BAG_ROOT ==="
echo "=== Namespace: $NAMESPACE | Laps: $NUM_LAPS ==="
echo

FAIL_COUNT=0

for bag_path in "${BAGS[@]}"; do
  bag_name=$(basename "$bag_path")
  outfile="$TMPDIR_BATCH/${bag_name}.txt"

  echo "--- Processing: $bag_name ---"
  if ! python3 bin/lap_time.py "$bag_path" "$NAMESPACE" "$NUM_LAPS" > "$outfile" 2>&1; then
    echo "  FAILED (see $outfile)"
    echo "$bag_name,FAILED,,,,,,,,,,,,,,,,," >> "$CSV_OUT"
    ((FAIL_COUNT++)) || true
    continue
  fi

  # Parse key metrics from lap_time.py output
  output=$(cat "$outfile")

  laps_strict=$(echo "$output" | grep -oP 'Complete laps \(strict order\):\s*\K\d+' || echo "0")
  seq_breaks=$(echo "$output" | grep -oP 'Sequence breaks ignored:\s*\K\d+' || echo "0")
  best_lap=$(echo "$output" | grep -oP 'Best lap:\s*\K[\d.]+' || echo "")
  mean_lap=$(echo "$output" | grep -oP 'Mean lap:\s*\K[\d.]+' || echo "")
  median_lap=$(echo "$output" | grep -oP 'Median lap:\s*\K[\d.]+' || echo "")
  std_lap=$(echo "$output" | grep -oP 'Lap std \(consistency\):\s*\K[\d.]+' || echo "")
  takeoff_3=$(echo "$output" | grep -oP "Takeoff → $NUM_LAPS laps done:\s*\K[\d.]+" || echo "")
  fastest_3=$(echo "$output" | grep -oP "Laps \d+–\d+:\s*\K[\d.]+" || echo "")
  path_len=$(echo "$output" | grep -oP 'Path length \(race\):\s*\K[\d.]+' || echo "")
  mean_speed=$(echo "$output" | grep -oP 'Mean / max speed:\s*\K[\d.]+' || echo "")
  max_speed=$(echo "$output" | grep -oP 'Mean / max speed:\s*[\d.]+ / \K[\d.]+' || echo "")
  mean_tilt=$(echo "$output" | grep -oP 'Mean / max tilt:\s*\K[\d.]+' || echo "")
  max_tilt=$(echo "$output" | grep -oP 'Mean / max tilt:\s*[\d.]+ / \K[\d.]+' || echo "")
  mean_br=$(echo "$output" | grep -oP 'Mean / max body rate:\s*\K[\d.]+' || echo "nan")
  max_br=$(echo "$output" | grep -oP 'Mean / max body rate:\s*[\d.]+ / \K[\d.]+' || echo "nan")
  mean_thrust=$(echo "$output" | grep -oP 'Mean / max thrust:\s*\K[\d.]+' || echo "nan")
  max_thrust=$(echo "$output" | grep -oP 'Mean / max thrust:\s*[\d.]+ / \K[\d.]+' || echo "nan")

  # Compute path per lap
  path_per_lap=""
  if [[ -n "$path_len" && -n "$laps_strict" && "$laps_strict" -gt 0 ]]; then
    path_per_lap=$(python3 -c "print(f'{${path_len}/${laps_strict}:.2f}')")
  fi

  echo "$bag_name,$laps_strict,$seq_breaks,$best_lap,$mean_lap,$median_lap,$std_lap,$takeoff_3,$fastest_3,$path_per_lap,$mean_speed,$max_speed,$mean_tilt,$max_tilt,$mean_br,$max_br,$mean_thrust,$max_thrust" >> "$CSV_OUT"

  echo "  Laps: $laps_strict (breaks: $seq_breaks) | Best: ${best_lap}s | Mean: ${mean_lap}s | Std: ${std_lap}s | Takeoff→3: ${takeoff_3:-n/a}s"
done

echo
echo "=== Summary CSV written to: $CSV_OUT ==="
echo

# Print summary table sorted by takeoff→3 laps
echo "=== Summary Table (sorted by Takeoff → $NUM_LAPS laps) ==="
echo
printf "%-35s %5s %3s %7s %7s %7s %7s %10s %10s %8s %7s %7s\n" \
  "Bag" "Laps" "Brk" "Best" "Mean" "Median" "Std" "TO→3lap" "Fast3lap" "Path/l" "v_max" "tilt_m"
printf '%.0s-' {1..130}; echo

# Sort by takeoff→3 (column 8), put empties last
tail -n +2 "$CSV_OUT" | sort -t',' -k8 -n -s | while IFS=',' read -r bag laps brk best mean median std to3 f3 ppl ms mxs mt mxt mbr mxbr mthr mxthr; do
  [[ "$laps" == "FAILED" ]] && { printf "%-35s  FAILED\n" "$bag"; continue; }
  printf "%-35s %5s %3s %7s %7s %7s %7s %10s %10s %8s %7s %7s\n" \
    "$bag" "$laps" "$brk" "$best" "$mean" "$median" "$std" "${to3:-n/a}" "${f3:-n/a}" "${ppl:-n/a}" "${mxs:-n/a}" "${mxt:-n/a}"
done

echo
if [[ $FAIL_COUNT -gt 0 ]]; then
  echo "⚠  $FAIL_COUNT bag(s) failed. Check individual output in $TMPDIR_BATCH/"
  # Don't clean up tmpdir on failure
  trap - EXIT
fi

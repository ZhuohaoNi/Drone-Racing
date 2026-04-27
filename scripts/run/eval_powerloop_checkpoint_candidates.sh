#!/bin/bash

# Evaluate multiple saved checkpoints from one powerloop run and rank them by
# deployment-style batch eval metrics. This is intentionally separate from
# RSL-RL's best_model.pt, which is selected by training reward only.
#
# Usage:
#   ./scripts/run/eval_powerloop_checkpoint_candidates.sh <run_dir> [num_envs] [max_candidates] [min_iter]
#
# Default candidate strategy is hybrid: mostly highest-training-reward
# checkpoints, plus a few latest/late/window anchors so one reward spike window
# does not monopolize the eval budget. Set CANDIDATE_STRATEGY=global_top to
# evaluate only best_model.pt + top reward checkpoints.
#
# Examples:
#   EXTRA_ENV_OVERRIDES='{"gate_side":0.8}' \
#     ./scripts/run/eval_powerloop_checkpoint_candidates.sh \
#       2026-04-27_01-43-18_powerloop-r1d1-gate3mask-fullswitch-twr1p87-final-tightdr-gate0p8-5000-seed42 \
#       1000 20 3000
#
# Optional env:
#   FORCE_EVAL=1                 re-run evals even if result JSON already exists
#   DRY_RUN=1                    print candidate list without running eval
#   REAL_GROUND_ONLY=0           use mixed reset robustness eval instead of default ground-only eval
#   CANDIDATE_STRATEGY=global_top use only best_model.pt + global top reward checkpoints
#   CANDIDATE_STRATEGY=stratified legacy anchor-heavy strategy
#   EXTRA_CANDIDATES="a.pt b.pt" append explicit checkpoints

set -euo pipefail

RUN_DIR=${1:?"ERROR: Provide run directory as first argument"}
NUM_ENVS=${2:-1000}
MAX_CANDIDATES=${3:-20}
MIN_ITER=${4:-3000}

cd "$(dirname "$0")/../.."

LOG_DIR="logs/rsl_rl/quadcopter_direct/$RUN_DIR"
if [[ ! -d "$LOG_DIR" ]]; then
    echo "ERROR: Run directory not found: $LOG_DIR" >&2
    exit 1
fi

REAL_GROUND_ONLY=${REAL_GROUND_ONLY:-1}
EVAL_PROFILE="mixed"
if [[ "$REAL_GROUND_ONLY" == "1" ]]; then
    EVAL_PROFILE="ground_only"
    EXTRA_ENV_OVERRIDES=$(
        EXTRA_ENV_OVERRIDES="${EXTRA_ENV_OVERRIDES:-}" python - <<'PY'
import json
import os

raw = os.environ.get("EXTRA_ENV_OVERRIDES", "")
base = json.loads(raw) if raw else {}
base.update({
    "ground_reset_ratio": 1.0,
    "replay_reset_ratio": 0.0,
    "real_start_reset_ratio": 1.0,
})
print(json.dumps(base, separators=(",", ":")))
PY
    )
    export EXTRA_ENV_OVERRIDES
fi

mapfile -t CANDIDATES < <(
    LOG_DIR="$LOG_DIR" \
    MAX_CANDIDATES="$MAX_CANDIDATES" \
    MIN_ITER="$MIN_ITER" \
    CANDIDATE_STRATEGY="${CANDIDATE_STRATEGY:-hybrid}" \
    EXTRA_CANDIDATES="${EXTRA_CANDIDATES:-}" \
    python - <<'PY'
import os
import re
from pathlib import Path

log_dir = Path(os.environ["LOG_DIR"])
max_candidates = max(1, int(os.environ["MAX_CANDIDATES"]))
min_iter = int(os.environ["MIN_ITER"])
candidate_strategy = os.environ.get("CANDIDATE_STRATEGY", "hybrid")
extra_candidates = os.environ.get("EXTRA_CANDIDATES", "").split()

model_re = re.compile(r"model_(\d+)_(-?\d+)\.pt$")
models = []
for path in log_dir.glob("model_*.pt"):
    match = model_re.fullmatch(path.name)
    if not match:
        continue
    models.append({
        "name": path.name,
        "iter": int(match.group(1)),
        "reward": int(match.group(2)),
    })

models_by_iter = sorted(models, key=lambda item: item["iter"])
late_models = [item for item in models_by_iter if item["iter"] >= min_iter]
early_models = [item for item in models_by_iter if item["iter"] < min_iter]

candidates = []
seen = set()

def add(name: str) -> None:
    if not name or name in seen:
        return
    if (log_dir / name).exists():
        candidates.append(name)
        seen.add(name)

def add_many(items) -> None:
    for item in items:
        if len(candidates) >= max_candidates:
            return
        add(item["name"])

top_all = sorted(models_by_iter, key=lambda item: (item["reward"], item["iter"]), reverse=True)

add("best_model.pt")
for name in extra_candidates:
    add(name)

if candidate_strategy == "global_top":
    add_many(top_all)
elif candidate_strategy == "hybrid":
    # Most slots still come from global reward rank, but reserve coverage for:
    # - latest policy state,
    # - best late-training policy,
    # - best policy in each training-time window.
    n_reward = max(1, int(max_candidates * 0.6))
    n_latest = max(1, int(max_candidates * 0.15))
    n_windows = max(1, max_candidates - 1 - n_reward - n_latest)

    if models_by_iter:
        add(max(models_by_iter, key=lambda item: item["iter"])["name"])
    if late_models:
        add(max(late_models, key=lambda item: (item["reward"], item["iter"]))["name"])

    add_many(top_all[:n_reward])

    if models_by_iter:
        n = len(models_by_iter)
        for window_idx in range(n_windows):
            start = window_idx * n // n_windows
            end = (window_idx + 1) * n // n_windows
            bucket = models_by_iter[start:end]
            if bucket:
                add(max(bucket, key=lambda item: (item["reward"], item["iter"]))["name"])

    latest_all = sorted(models_by_iter, key=lambda item: item["iter"], reverse=True)
    add_many(latest_all[:n_latest])
    add_many(top_all)
elif candidate_strategy == "stratified":
    # Anchors: latest overall plus the best reward checkpoint in each broad phase.
    if models_by_iter:
        add(max(models_by_iter, key=lambda item: item["iter"])["name"])
        add(max(models_by_iter, key=lambda item: (item["reward"], item["iter"]))["name"])
    if late_models:
        add(max(late_models, key=lambda item: item["iter"])["name"])
        add(max(late_models, key=lambda item: (item["reward"], item["iter"]))["name"])
    if early_models:
        add(max(early_models, key=lambda item: (item["reward"], item["iter"]))["name"])

    # Fill from multiple views instead of hard-filtering by MIN_ITER. This catches
    # common PPO behavior where a good real-flight policy appears mid-training and
    # later checkpoints over-specialize to the training reset distribution.
    top_late = sorted(late_models, key=lambda item: (item["reward"], item["iter"]), reverse=True)
    latest_all = sorted(models_by_iter, key=lambda item: item["iter"], reverse=True)
    latest_late = sorted(late_models, key=lambda item: item["iter"], reverse=True)
    add_many(top_all)
    add_many(top_late)
    add_many(latest_all)
    add_many(latest_late)
else:
    raise SystemExit(f"Unknown CANDIDATE_STRATEGY={candidate_strategy!r}; use hybrid, global_top, or stratified")

print("\n".join(candidates))
PY
)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
    echo "ERROR: No checkpoint candidates found in $LOG_DIR" >&2
    exit 1
fi

echo "========================================"
echo "  Powerloop Checkpoint Candidate Eval"
echo "  Run dir:        $RUN_DIR"
echo "  Envs:           $NUM_ENVS"
echo "  Max candidates: $MAX_CANDIDATES"
echo "  Min iter:       $MIN_ITER"
echo "  Strategy:       ${CANDIDATE_STRATEGY:-hybrid}"
echo "  Eval profile:   $EVAL_PROFILE"
echo "  Extra env:      ${EXTRA_ENV_OVERRIDES:-<none>}"
echo "  Candidates:"
for checkpoint in "${CANDIDATES[@]}"; do
    echo "    - $checkpoint"
done
echo "========================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: not running batch eval."
    exit 0
fi

for checkpoint in "${CANDIDATES[@]}"; do
    stem=${checkpoint%.pt}
    result_stem="$stem"
    if [[ "$EVAL_PROFILE" != "mixed" ]]; then
        result_stem="${stem}_${EVAL_PROFILE}"
    fi
    result_dir="$LOG_DIR/batch_eval/$result_stem"
    result_json="$result_dir/batch_eval_results.json"

    if [[ -f "$result_json" && "${FORCE_EVAL:-0}" != "1" ]]; then
        echo ""
        echo "Skipping existing eval: $checkpoint"
        echo "  $result_json"
        continue
    fi

    echo ""
    echo "========================================"
    echo "  Evaluating checkpoint: $checkpoint"
    echo "========================================"
    BATCH_EVAL_OUTPUT_DIR="$result_dir" \
        ./scripts/run/eval_powerloop_real_twr.sh "$RUN_DIR" "$checkpoint" "$NUM_ENVS"
done

CANDIDATE_LIST=$(printf '%s\n' "${CANDIDATES[@]}")
CANDIDATE_LIST="$CANDIDATE_LIST" LOG_DIR="$LOG_DIR" EVAL_PROFILE="$EVAL_PROFILE" python - <<'PY'
import csv
import json
import os
from pathlib import Path

log_dir = Path(os.environ["LOG_DIR"])
eval_profile = os.environ["EVAL_PROFILE"]
candidates = [line.strip() for line in os.environ["CANDIDATE_LIST"].splitlines() if line.strip()]
rows = []
missing = []

for checkpoint in candidates:
    stem = Path(checkpoint).stem
    result_stem = stem if eval_profile == "mixed" else f"{stem}_{eval_profile}"
    result_path = log_dir / "batch_eval" / result_stem / "batch_eval_results.json"
    if not result_path.exists():
        missing.append(checkpoint)
        continue
    with result_path.open() as f:
        data = json.load(f)
    overall = data["overall"]
    rows.append({
        "checkpoint": checkpoint,
        "success_rate_pct": float(overall["success_rate_pct"]),
        "takeoff_success_pct": float(overall.get("takeoff_success_pct", 0.0)),
        "ground_success_rate_pct": (
            float(overall["ground_success_rate_pct"])
            if overall.get("ground_success_rate_pct") is not None else None
        ),
        "mean_3lap_time": float(overall["mean_3lap_time"]),
        "std_3lap_time": float(overall["std_3lap_time"]),
        "best_3lap_time": float(overall["best_3lap_time"]),
        "worst_3lap_time": float(overall["worst_3lap_time"]),
        "n_3lap_success": int(overall["n_3lap_success"]),
    })

rows.sort(key=lambda row: (
    -row["success_rate_pct"],
    row["mean_3lap_time"],
    row["std_3lap_time"],
    row["worst_3lap_time"],
))

out_dir = log_dir / "batch_eval"
out_dir.mkdir(parents=True, exist_ok=True)
suffix = "" if eval_profile == "mixed" else f"_{eval_profile}"
csv_path = out_dir / f"checkpoint_selection_summary{suffix}.csv"
json_path = out_dir / f"checkpoint_selection_summary{suffix}.json"

fieldnames = [
    "checkpoint",
    "success_rate_pct",
    "takeoff_success_pct",
    "ground_success_rate_pct",
    "mean_3lap_time",
    "std_3lap_time",
    "best_3lap_time",
    "worst_3lap_time",
    "n_3lap_success",
    "meets_final_rule",
]

for row in rows:
    row["meets_final_rule"] = (
        row["success_rate_pct"] >= 98.5
        and row["mean_3lap_time"] <= 18.65
        and row["std_3lap_time"] <= 0.5
    )

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with json_path.open("w") as f:
    json.dump({"ranked": rows, "missing": missing}, f, indent=2)

print("")
print("========================================")
print("  Checkpoint Selection Summary")
print("========================================")
if not rows:
    print("No completed candidate eval results found.")
else:
    print(f"{'rank':>4}  {'checkpoint':<34} {'SR':>6} {'gSR':>6} {'takeoff':>8} {'mean':>7} {'std':>6} {'best':>7} {'worst':>7} {'rule':>6}")
    for idx, row in enumerate(rows, start=1):
        ground_sr = row["ground_success_rate_pct"]
        ground_sr_text = f"{ground_sr:.1f}%" if ground_sr is not None else "n/a"
        print(
            f"{idx:>4}  "
            f"{row['checkpoint']:<34.34} "
            f"{row['success_rate_pct']:>5.1f}% "
            f"{ground_sr_text:>6} "
            f"{row['takeoff_success_pct']:>7.1f}% "
            f"{row['mean_3lap_time']:>6.2f}s "
            f"{row['std_3lap_time']:>5.2f}s "
            f"{row['best_3lap_time']:>6.2f}s "
            f"{row['worst_3lap_time']:>6.2f}s "
            f"{'yes' if row['meets_final_rule'] else 'no':>6}"
        )
    best = rows[0]
    print("")
    print(f"Selected by eval ranking: {best['checkpoint']}")
    print(f"Summary CSV:  {csv_path}")
    print(f"Summary JSON: {json_path}")

if missing:
    print("")
    print("Missing eval results:")
    for checkpoint in missing:
        print(f"  - {checkpoint}")
PY

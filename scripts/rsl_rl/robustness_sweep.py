#!/usr/bin/env python3

"""Deterministic robustness sweep for deployment mismatch tests.

This sweep is intentionally axis-oriented:
- Control mismatch: thrust scale, fixed action latency, body-rate gain scale.
- Observation mismatch: velocity noise/bias, yaw bias, fixed observation delay.

Each scenario runs the existing `batch_eval_race.py` pipeline under one shared
environment profile, then the script aggregates takeoff / first-gate / 3-lap
success into one summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMMON_SWEEP_OVERRIDES = {
    "replay_reset_ratio": 0.0,
    "ground_reset_ratio": 1.0,
    "staged_replay_reset": False,
    "use_spline_reset": False,
    "twr_randomization_pct": 0.0,
    "aero_randomization_scale_min": 1.0,
    "aero_randomization_scale_max": 1.0,
    "pid_kpki_randomization_pct": 0.0,
    "pid_kd_randomization_pct": 0.0,
    "mass_variation": 0.0,
    "motor_tau_scale_min": 1.0,
    "motor_tau_scale_max": 1.0,
    "thrust_to_weight_scale": 1.0,
    "body_rate_gain_scale": 1.0,
    "action_latency_max": 0,
    "fixed_action_delay_steps": 0,
    "obs_noise_std_scale": 0.0,
    "obs_lin_vel_noise_std": 0.0,
    "obs_latency_prob": 0.0,
    "fixed_obs_delay_steps": 0,
    "obs_yaw_bias_deg": 0.0,
    "obs_lin_vel_bias": [0.0, 0.0, 0.0],
    "obs_gate_corner_bias": [0.0, 0.0, 0.0],
}

SCENARIOS = {
    "control_nominal": {
        "group": "control",
        "description": "Ground-start baseline with deterministic nominal dynamics.",
        "env_overrides": {},
    },
    "control_thrust_0p90": {
        "group": "control",
        "description": "Thrust scale 0.90x nominal.",
        "env_overrides": {"thrust_to_weight_scale": 0.90},
    },
    "control_thrust_1p10": {
        "group": "control",
        "description": "Thrust scale 1.10x nominal.",
        "env_overrides": {"thrust_to_weight_scale": 1.10},
    },
    "control_latency_20ms": {
        "group": "control",
        "description": "Fixed 1-step action delay at 50 Hz policy rate (20 ms).",
        "env_overrides": {"fixed_action_delay_steps": 1},
    },
    "control_latency_40ms": {
        "group": "control",
        "description": "Fixed 2-step action delay at 50 Hz policy rate (40 ms).",
        "env_overrides": {"fixed_action_delay_steps": 2},
    },
    "control_rate_gain_0p85": {
        "group": "control",
        "description": "Body-rate PID gains scaled to 0.85x nominal.",
        "env_overrides": {"body_rate_gain_scale": 0.85},
    },
    "control_rate_gain_1p15": {
        "group": "control",
        "description": "Body-rate PID gains scaled to 1.15x nominal.",
        "env_overrides": {"body_rate_gain_scale": 1.15},
    },
    "obs_vel_noise_0p05": {
        "group": "observation",
        "description": "Extra 0.05 m/s body-frame velocity noise.",
        "env_overrides": {"obs_lin_vel_noise_std": 0.05},
    },
    "obs_vel_bias_0p20": {
        "group": "observation",
        "description": "Constant +0.20 m/s x-velocity bias in body frame.",
        "env_overrides": {"obs_lin_vel_bias": [0.20, 0.0, 0.0]},
    },
    "obs_yaw_bias_5deg": {
        "group": "observation",
        "description": "Constant +5 deg yaw bias in the observation frame.",
        "env_overrides": {"obs_yaw_bias_deg": 5.0},
    },
    "obs_gate_bias_5cm": {
        "group": "observation",
        "description": "Constant +5 cm gate-corner position bias in body frame.",
        "env_overrides": {"obs_gate_corner_bias": [0.05, 0.0, 0.0]},
    },
    "obs_delay_20ms": {
        "group": "observation",
        "description": "Fixed 1-step observation delay (20 ms).",
        "env_overrides": {"fixed_obs_delay_steps": 1},
    },
    "obs_delay_40ms": {
        "group": "observation",
        "description": "Fixed 2-step observation delay (40 ms).",
        "env_overrides": {"fixed_obs_delay_steps": 2},
    },
}

DEFAULT_SCENARIOS = list(SCENARIOS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic robustness sweep.")
    parser.add_argument("--load_run", type=str, required=True, help="Run directory name under logs/rsl_rl/*/.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Race-v0")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_params_per_env", type=int, default=0,
                        help="Forwarded to batch_eval_race.py. Default 0 keeps all envs on the same scenario config.")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS),
                        help=f"Comma-separated scenario names. Available: {', '.join(SCENARIOS)}")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--list_scenarios", action="store_true", default=False)
    return parser.parse_args()


def parse_base_env_overrides() -> dict:
    raw = os.environ.get("ENV_OVERRIDES", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ENV_OVERRIDES is not valid JSON: {raw}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("ENV_OVERRIDES must decode to a JSON object.")
    return parsed


def resolve_run_dir(load_run: str) -> Path:
    matches = list((PROJECT_ROOT / "logs" / "rsl_rl").glob(f"*/{load_run}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find run directory '{load_run}' under {PROJECT_ROOT / 'logs' / 'rsl_rl'}"
        )
    raise RuntimeError(f"Multiple run directories matched '{load_run}': {matches}")


def get_output_root(args: argparse.Namespace, run_dir: Path) -> Path:
    ckpt_stem = Path(args.checkpoint).stem
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return run_dir / "robustness_sweep" / ckpt_stem


def parse_scenarios(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("No scenarios selected.")
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Available: {sorted(SCENARIOS)}")
    return names


def build_batch_eval_command(args: argparse.Namespace, scenario_out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/rsl_rl/batch_eval_race.py",
        "--task", args.task,
        "--load_run", args.load_run,
        "--checkpoint", args.checkpoint,
        "--headless",
        "--num_trials", str(args.num_trials),
        "--num_envs", str(args.num_envs),
        "--max_steps", str(args.max_steps),
        "--seed", str(args.seed),
        "--num_params_per_env", str(args.num_params_per_env),
        "--output_dir", str(scenario_out_dir),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.disable_fabric:
        cmd.append("--disable_fabric")
    return cmd


def load_batch_eval_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary_files(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    csv_path = output_root / "robustness_sweep_summary.csv"
    json_path = output_root / "robustness_sweep_summary.json"

    fieldnames = [
        "group",
        "scenario",
        "description",
        "takeoff_success_pct",
        "first_gate_success_pct",
        "three_lap_success_pct",
        "mean_3lap_time",
        "ground_takeoff_success_pct",
        "ground_first_gate_success_pct",
        "n_ground_starts",
        "best_3lap_time",
        "worst_3lap_time",
        "std_3lap_time",
        "n_3lap_success",
        "env_overrides",
        "output_dir",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return csv_path, json_path


def _fmt_pct(value) -> str:
    return "N/A" if value is None else f"{float(value):.1f}"


def _fmt_time(value) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def write_markdown_summary(rows: list[dict], output_root: Path, run_dir: Path, checkpoint: str) -> Path:
    md_path = output_root / "robustness_sweep_summary.md"

    nominal_row = next((row for row in rows if row["scenario"] == "control_nominal"), None)
    nominal_sr = nominal_row.get("three_lap_success_pct") if nominal_row else None
    nominal_time = nominal_row.get("mean_3lap_time") if nominal_row else None
    use_delta_cols = nominal_sr not in (None, 0, 0.0)

    lines = [
        "# Robustness Sweep Summary",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Output dir: `{output_root}`",
        f"- Scenarios: `{len(rows)}`",
        "",
    ]

    if nominal_row is None:
        lines += [
            "No `control_nominal` row found, so nominal-reference deltas were omitted.",
            "",
        ]
    elif not use_delta_cols:
        lines += [
            f"Nominal 3-lap success is `{_fmt_pct(nominal_sr)}%`, so delta-vs-nominal columns are omitted.",
            "",
        ]
    else:
        lines += [
            f"Nominal reference: `control_nominal`, 3-lap success `{_fmt_pct(nominal_sr)}%`, "
            f"mean successful 3-lap time `{_fmt_time(nominal_time)}s`.",
            "",
        ]

    if use_delta_cols:
        lines.append("| Scenario | Group | Takeoff (%) | First Gate (%) | 3-lap SR (%) | Mean 3-lap (s) | Delta SR vs nominal | Delta time vs nominal (s) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| Scenario | Group | Takeoff (%) | First Gate (%) | 3-lap SR (%) | Mean 3-lap (s) | Best 3-lap (s) | Worst 3-lap (s) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        takeoff = _fmt_pct(row.get("ground_takeoff_success_pct"))
        first_gate = _fmt_pct(row.get("ground_first_gate_success_pct"))
        three_lap = _fmt_pct(row.get("three_lap_success_pct"))
        mean_time = _fmt_time(row.get("mean_3lap_time"))
        if use_delta_cols:
            sr = row.get("three_lap_success_pct")
            mt = row.get("mean_3lap_time")
            delta_sr = "N/A" if sr is None else f"{float(sr) - float(nominal_sr):+.1f}"
            delta_time = "N/A" if (mt is None or nominal_time is None) else f"{float(mt) - float(nominal_time):+.2f}"
            lines.append(
                f"| `{row['scenario']}` | {row['group']} | {takeoff} | {first_gate} | {three_lap} | {mean_time} | {delta_sr} | {delta_time} |"
            )
        else:
            best_time = _fmt_time(row.get("best_3lap_time"))
            worst_time = _fmt_time(row.get("worst_3lap_time"))
            lines.append(
                f"| `{row['scenario']}` | {row['group']} | {takeoff} | {first_gate} | {three_lap} | {mean_time} | {best_time} | {worst_time} |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def print_scenarios() -> None:
    print("Common overrides:")
    print(json.dumps(COMMON_SWEEP_OVERRIDES, indent=2, sort_keys=True))
    print("\nAvailable scenarios:")
    for name, cfg in SCENARIOS.items():
        print(f"- {name} [{cfg['group']}]: {cfg['description']}")
        print(f"  env_overrides={json.dumps(cfg['env_overrides'], sort_keys=True)}")


def main() -> None:
    args = parse_args()
    if args.list_scenarios:
        print_scenarios()
        return

    run_dir = resolve_run_dir(args.load_run)
    output_root = get_output_root(args, run_dir)
    scenario_names = parse_scenarios(args.scenarios)
    base_env_overrides = parse_base_env_overrides()

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Deterministic robustness sweep")
    print(f"Run dir:               {run_dir}")
    print(f"Checkpoint:            {args.checkpoint}")
    print(f"Output root:           {output_root}")
    print(f"Scenarios:             {', '.join(scenario_names)}")
    print(f"Base ENV_OVERRIDES:    {json.dumps(base_env_overrides, sort_keys=True)}")
    print(f"Common sweep overrides:{json.dumps(COMMON_SWEEP_OVERRIDES, sort_keys=True)}")
    print(f"num_params_per_env:    {args.num_params_per_env}")
    print("=" * 72)

    rows: list[dict] = []
    for scenario_name in scenario_names:
        scenario_cfg = SCENARIOS[scenario_name]
        scenario_env = dict(base_env_overrides)
        scenario_env.update(COMMON_SWEEP_OVERRIDES)
        scenario_env.update(scenario_cfg["env_overrides"])

        scenario_out_dir = output_root / scenario_name
        cmd = build_batch_eval_command(args, scenario_out_dir)
        child_env = os.environ.copy()
        child_env["ENV_OVERRIDES"] = json.dumps(scenario_env, separators=(",", ":"))

        print()
        print("-" * 72)
        print(f"Scenario:      {scenario_name}")
        print(f"Group:         {scenario_cfg['group']}")
        print(f"Description:   {scenario_cfg['description']}")
        print(f"ENV_OVERRIDES: {child_env['ENV_OVERRIDES']}")
        print(f"Output dir:    {scenario_out_dir}")
        print(f"Command:       {' '.join(cmd)}")
        print("-" * 72)

        if args.dry_run:
            continue

        subprocess.run(cmd, cwd=PROJECT_ROOT, env=child_env, check=True)

        summary_path = scenario_out_dir / "batch_eval_results.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected summary file not found: {summary_path}")

        summary = load_batch_eval_summary(summary_path)
        overall = summary.get("overall", {})
        rows.append({
            "group": scenario_cfg["group"],
            "scenario": scenario_name,
            "description": scenario_cfg["description"],
            "takeoff_success_pct": overall.get("takeoff_success_pct"),
            "first_gate_success_pct": overall.get("first_gate_success_pct"),
            "three_lap_success_pct": overall.get("success_rate_pct"),
            "mean_3lap_time": overall.get("mean_3lap_time"),
            "ground_takeoff_success_pct": overall.get("ground_takeoff_success_pct"),
            "ground_first_gate_success_pct": overall.get("ground_first_gate_success_pct"),
            "n_ground_starts": overall.get("n_ground_starts"),
            "best_3lap_time": overall.get("best_3lap_time"),
            "worst_3lap_time": overall.get("worst_3lap_time"),
            "std_3lap_time": overall.get("std_3lap_time"),
            "n_3lap_success": overall.get("n_3lap_success"),
            "env_overrides": json.dumps(scenario_env, sort_keys=True),
            "output_dir": str(scenario_out_dir),
        })

    if args.dry_run:
        print("\nDry run only. No jobs launched.")
        return

    csv_path, json_path = write_summary_files(rows, output_root)
    md_path = write_markdown_summary(rows, output_root, run_dir, args.checkpoint)

    print()
    print("=" * 72)
    print("Summary")
    for row in rows:
        takeoff = row["ground_takeoff_success_pct"]
        first_gate = row["ground_first_gate_success_pct"]
        three_lap = row["three_lap_success_pct"]
        lap_time = row["mean_3lap_time"]
        takeoff_text = "N/A" if takeoff is None else f"{takeoff:.1f}%"
        first_gate_text = "N/A" if first_gate is None else f"{first_gate:.1f}%"
        three_lap_text = "N/A" if three_lap is None else f"{three_lap:.1f}%"
        lap_time_text = "N/A" if lap_time is None else f"{lap_time:.2f}s"
        print(
            f"{row['group']:<12} {row['scenario']:<24} "
            f"takeoff={takeoff_text:<8} first_gate={first_gate_text:<8} "
            f"3lap={three_lap_text:<8} mean_3lap={lap_time_text}"
        )
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

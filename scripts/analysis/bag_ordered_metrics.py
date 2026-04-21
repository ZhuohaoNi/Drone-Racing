#!/usr/bin/env python3
"""Compute ordered gate metrics for real/sim ROS2 race bags.

This script reuses the gate-plane detector from the sim2real repo's
``bin/lap_time.py`` and adds metrics that are useful after real tests:

- official first-N-lap time: race-start command -> N-th lap last gate
- ordered-pass clearance: physical 1 m gate margin at accepted gate crossings
- segment times: race start -> gate 0 and ordered gate-to-gate durations
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PHYSICAL_GATE_HALF_SIDE_M = 0.5


def load_lap_time_module(sim2real_repo: Path):
    module_path = sim2real_repo / "bin" / "lap_time.py"
    spec = importlib.util.spec_from_file_location("sim2real_lap_time", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import lap_time.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_or_none(value: Any):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return float(value) if isinstance(value, (float, np.floating)) else value


def interval_metrics(lt, odom, cmd, t_s: float | None, t_e: float | None):
    if t_s is None or t_e is None or t_e <= t_s:
        return {}
    metrics = lt.interval_metrics(odom, cmd, t_s, t_e)
    return {k: finite_or_none(v) for k, v in metrics.items()}


def pass_clearance(event) -> float:
    _, _, lateral, vertical = event
    return PHYSICAL_GATE_HALF_SIDE_M - max(abs(float(lateral)), abs(float(vertical)))


def per_gate_stats(values_by_gate: list[list[float]], reducer):
    out = []
    for values in values_by_gate:
        if not values:
            out.append(None)
        else:
            out.append(float(reducer(np.asarray(values, dtype=float))))
    return out


def compute(args):
    sim2real_repo = Path(args.sim2real_repo).expanduser().resolve()
    lt = load_lap_time_module(sim2real_repo)

    waypoints = lt.load_waypoints(args.track)
    n_gates = len(waypoints)
    odom, cmd = lt.read_bag(args.bag_path, args.namespace)
    if len(odom["t"]) == 0:
        raise RuntimeError("No odometry messages found.")

    t_race_start_cmd = float(cmd["t"][0]) if len(cmd["t"]) else None
    t_takeoff = lt.find_takeoff_time(odom)
    t_tracking_start = (t_takeoff + lt.POST_TAKEOFF_DELAY) if t_takeoff is not None else 0.0
    if t_race_start_cmd is not None:
        t_tracking_start = max(t_tracking_start, t_race_start_cmd)

    valid, misses = lt.detect_passes(odom, waypoints, t_tracking_start)
    ordered, sequence_breaks = lt.extract_ordered_passes(valid, n_gates)
    lap_intervals = lt.lap_intervals_from_ordered_passes(ordered, n_gates)

    pass_wp = np.array([e[1] for e in valid], dtype=int) if valid else np.array([], dtype=int)
    valid_per_gate = np.bincount(pass_wp, minlength=n_gates) if len(pass_wp) else np.zeros(n_gates, dtype=int)
    near_miss_per_gate = np.zeros(n_gates, dtype=int)
    for _, wi, *_ in misses:
        near_miss_per_gate[wi] += 1

    required_passes = args.num_laps * n_gates
    eval_events = ordered[:required_passes]
    eval_end_t = None
    eval_end_gate = None
    eval_time = None
    if t_race_start_cmd is not None and len(eval_events) >= required_passes:
        eval_end_t = float(eval_events[-1][0])
        eval_end_gate = int(eval_events[-1][1])
        eval_time = eval_end_t - t_race_start_cmd

    clearances = [pass_clearance(ev) for ev in ordered]
    clearances_by_gate = [[] for _ in range(n_gates)]
    for ev, clearance in zip(ordered, clearances):
        clearances_by_gate[int(ev[1])].append(clearance)
    ordered_pass_details = [
        {
            "seq": int(i),
            "gate": int(ev[1]),
            "t": float(ev[0]),
            "lateral": float(ev[2]),
            "vertical": float(ev[3]),
            "clearance": float(clearance),
        }
        for i, (ev, clearance) in enumerate(zip(ordered, clearances))
    ]

    min_clearance = min(clearances) if clearances else None
    mean_clearance = float(np.mean(clearances)) if clearances else None
    min_clearance_gate = None
    if clearances:
        min_idx = int(np.argmin(np.asarray(clearances, dtype=float)))
        min_clearance_gate = int(ordered[min_idx][1])

    segment_durations: dict[str, list[float]] = {f"{i}->{(i + 1) % n_gates}": [] for i in range(n_gates)}
    for prev, cur in zip(ordered, ordered[1:]):
        prev_gate = int(prev[1])
        cur_gate = int(cur[1])
        label = f"{prev_gate}->{cur_gate}"
        segment_durations.setdefault(label, []).append(float(cur[0] - prev[0]))

    mean_segment_times = {
        label: (float(np.mean(values)) if values else None)
        for label, values in segment_durations.items()
    }

    eval_segment_times = []
    eval_start_to_first_gate = None
    if t_race_start_cmd is not None and eval_events:
        eval_start_to_first_gate = float(eval_events[0][0] - t_race_start_cmd)
        eval_segment_times.append({
            "segment": f"start->{int(eval_events[0][1])}",
            "dt": eval_start_to_first_gate,
            "end_t": float(eval_events[0][0]),
        })
        for prev, cur in zip(eval_events, eval_events[1:]):
            eval_segment_times.append({
                "segment": f"{int(prev[1])}->{int(cur[1])}",
                "dt": float(cur[0] - prev[0]),
                "end_t": float(cur[0]),
            })

    lap_starts = np.array([s for s, _ in lap_intervals], dtype=float) if lap_intervals else np.array([])
    lap_ends = np.array([e for _, e in lap_intervals], dtype=float) if lap_intervals else np.array([])
    lap_times = lap_ends - lap_starts if len(lap_intervals) else np.array([])

    summary = {
        "bag_path": str(args.bag_path),
        "namespace": args.namespace,
        "track": args.track,
        "num_laps_requested": args.num_laps,
        "n_gates": n_gates,
        "bag_duration_odom": float(odom["t"][-1]),
        "race_start_cmd_t": t_race_start_cmd,
        "takeoff_t": t_takeoff,
        "tracking_start_t": float(t_tracking_start),
        "eval_first_n_laps_time": eval_time,
        "eval_first_n_laps_end_t": eval_end_t,
        "eval_first_n_laps_end_gate": eval_end_gate,
        "complete_laps_expected_order": int(len(lap_intervals)),
        "ordered_passes": int(len(ordered)),
        "total_valid_passes": int(len(valid)),
        "total_near_misses": int(len(misses)),
        "sequence_breaks": int(len(sequence_breaks)),
        "valid_per_gate": [int(x) for x in valid_per_gate.tolist()],
        "near_miss_per_gate": [int(x) for x in near_miss_per_gate.tolist()],
        "min_ordered_clearance_m": finite_or_none(min_clearance),
        "mean_ordered_clearance_m": finite_or_none(mean_clearance),
        "min_ordered_clearance_gate": min_clearance_gate,
        "min_ordered_clearance_per_gate_m": per_gate_stats(clearances_by_gate, np.min),
        "mean_ordered_clearance_per_gate_m": per_gate_stats(clearances_by_gate, np.mean),
        "ordered_pass_details": ordered_pass_details,
        "ordered_clearance_by_gate_m": clearances_by_gate,
        "eval_start_to_first_gate_time": finite_or_none(eval_start_to_first_gate),
        "mean_segment_times": mean_segment_times,
        "eval_segment_times": eval_segment_times,
    }

    if len(lap_times):
        overall = interval_metrics(lt, odom, cmd, float(lap_starts[0]), float(lap_ends[-1]))
        summary.update({
            "best_lap": float(lap_times.min()),
            "mean_lap": float(lap_times.mean()),
            "median_lap": float(np.median(lap_times)),
            "std_lap": float(lap_times.std()),
            "race_time_gate0_to_last_gate0": float(lap_ends[-1] - lap_starts[0]),
            "takeoff_to_last_gate0": float(lap_ends[-1] - t_takeoff) if t_takeoff is not None else None,
            "path_length_race": overall.get("path_len"),
            "mean_speed": overall.get("mean_speed"),
            "max_speed": overall.get("max_speed"),
            "mean_tilt": overall.get("mean_tilt"),
            "max_tilt": overall.get("max_tilt"),
            "mean_body_rate_cmd": overall.get("mean_br"),
            "max_body_rate_cmd": overall.get("max_br"),
            "mean_thrust": overall.get("mean_thrust"),
            "max_thrust": overall.get("max_thrust"),
        })
    else:
        for key in (
            "best_lap", "mean_lap", "median_lap", "std_lap",
            "race_time_gate0_to_last_gate0", "takeoff_to_last_gate0",
            "path_length_race", "mean_speed", "max_speed", "mean_tilt",
            "max_tilt", "mean_body_rate_cmd", "max_body_rate_cmd",
            "mean_thrust", "max_thrust",
        ):
            summary[key] = None

    if len(lap_times) >= args.num_laps:
        window_sums = np.array([
            lap_times[i:i + args.num_laps].sum()
            for i in range(len(lap_times) - args.num_laps + 1)
        ])
        best_i = int(np.argmin(window_sums))
        summary["fastest_n_lap_window_time"] = float(window_sums[best_i])
        summary["fastest_n_lap_window_start_lap"] = best_i + 1
        if t_takeoff is not None:
            summary["takeoff_to_n_laps_gate0_close"] = float(lap_ends[args.num_laps - 1] - t_takeoff)
        else:
            summary["takeoff_to_n_laps_gate0_close"] = None
        summary["first_n_lap_sum_gate0_to_gate0"] = float(lap_times[:args.num_laps].sum())
    else:
        summary["fastest_n_lap_window_time"] = None
        summary["fastest_n_lap_window_start_lap"] = None
        summary["takeoff_to_n_laps_gate0_close"] = None
        summary["first_n_lap_sum_gate0_to_gate0"] = None

    return summary


def fmt(value, precision=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def print_summary(summary):
    print("\n=== Ordered Race Metrics ===")
    print(f"Track:                         {summary['track']}")
    print(f"Race-start command:            {fmt(summary['race_start_cmd_t'])} s")
    print(f"Eval first {summary['num_laps_requested']} laps:             {fmt(summary['eval_first_n_laps_time'])} s")
    print(f"Complete ordered laps:         {summary['complete_laps_expected_order']}")
    print(f"Ordered passes:                {summary['ordered_passes']}")
    print(f"Valid passes:                  {summary['total_valid_passes']}")
    print(f"Min ordered clearance:         {fmt(summary['min_ordered_clearance_m'])} m"
          f" (gate {fmt(summary['min_ordered_clearance_gate'], 0)})")
    print(f"Mean ordered clearance:        {fmt(summary['mean_ordered_clearance_m'])} m")
    print(f"Clearance per gate min:        {' '.join(fmt(v) for v in summary['min_ordered_clearance_per_gate_m'])}")
    print(f"Start -> first gate:           {fmt(summary['eval_start_to_first_gate_time'])} s")

    print("\nOrdered pass clearance:")
    print("  seq gate      t     lat     vert   clearance")
    for ev in summary["ordered_pass_details"]:
        print(
            f"  {ev['seq']:>3d} {ev['gate']:>4d} "
            f"{ev['t']:>7.3f} {ev['lateral']:>7.3f} "
            f"{ev['vertical']:>7.3f} {ev['clearance']:>10.3f}"
        )

    print("\nOrdered clearance by gate:")
    for gate in range(summary["n_gates"]):
        values = summary["ordered_clearance_by_gate_m"][gate]
        formatted = " ".join(fmt(v) for v in values) if values else "n/a"
        print(f"  Gate {gate}: {formatted}")
    print("\nMean segment times:")
    print("  " + " | ".join(f"{k}: {fmt(v)}" for k, v in summary["mean_segment_times"].items()))
    print("\nEval segment times:")
    if not summary["eval_segment_times"]:
        print("  n/a")
    else:
        print("  " + " | ".join(f"{x['segment']}: {fmt(x['dt'])}" for x in summary["eval_segment_times"]))


def main():
    parser = argparse.ArgumentParser(description="Compute ordered pass clearance and segment times for ROS2 race bags.")
    parser.add_argument("bag_path")
    parser.add_argument("namespace")
    parser.add_argument("num_laps", nargs="?", type=int, default=3)
    parser.add_argument("--track", choices=["powerloop", "powerloop_sim", "circle", "config"], default="powerloop")
    parser.add_argument("--sim2real-repo", default="~/Documents/ese6510/Drone-Racing-sim2real")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    summary = compute(args)
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print_summary(summary)


if __name__ == "__main__":
    main()

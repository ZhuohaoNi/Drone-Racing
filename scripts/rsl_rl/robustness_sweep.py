#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robustness sweep: single-axis DR parameter sensitivity for a fixed policy.

For each axis (TWR, aero, PID, motor_tau, action_latency), partitions num_envs
across a set of values and runs one trial. Reports SR / mean-lap / P90-lap /
crash-rate per value per axis. Produces a CSV and PNG per axis plus a combined
summary.

Design
------
The existing batch_eval_race.py samples 3 *random* params per env. This script
is different: it fixes all axes at nominal except one, and sweeps the chosen
axis over a deterministic value grid. Each trial covers one axis end-to-end —
group-assigned envs carry one sweep value each, the other axes stay at
nominal, so one trial yields a full row of the sensitivity plot.

Usage
-----
    python scripts/rsl_rl/robustness_sweep.py \\
        --task Isaac-Quadcopter-Race-v0 \\
        --load_run 2026-04-19_XX-XX-XX_powerloop-r1d1-gate3mask \\
        --checkpoint best_model.pt \\
        --num_envs 600 \\
        --max_steps 3000 \\
        --axes twr,k_aero_xy,kp_omega_rp,motor_tau,action_latency \\
        --headless

ENV_OVERRIDES still works (e.g. to switch track_name).
"""

import sys
import os

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")

import argparse
import csv
import json
import time

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Robustness sweep — per-axis DR sensitivity.")
parser.add_argument("--num_envs", type=int, default=600,
                    help="Parallel envs; partitioned across sweep values per axis.")
parser.add_argument("--max_steps", type=int, default=3000,
                    help="Max sim steps per trial.")
parser.add_argument("--target_laps", type=int, default=3,
                    help="Laps to count as success.")
parser.add_argument("--axes", type=str, default="twr,k_aero_xy,k_aero_z,kp_omega_rp,kd_omega_rp,motor_tau,action_latency,action_noise_std,gate_pos_jitter",
                    help="Comma-separated list of axes to sweep. Any subset of the pool in SWEEP_AXES.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--gate_side", type=float, default=None,
                    help="Override env_cfg.gate_model.gate_side. IMPORTANT: gate_side also "
                         "determines the 4 gate-corner positions in the observation "
                         "(quadcopter_env.py:_local_square), so changing it makes the obs "
                         "OOD from training. Leave unset (uses cfg default, matches training) "
                         "unless you really want to probe an obs-OOD scenario.")
parser.add_argument("--train_mode", action="store_true", default=False,
                    help="Run with is_train=True (obs noise + random action latency + DR on reset). "
                         "Default is is_train=False: clean deterministic conditions so sweep isolates "
                         "the effect of one axis. Use --train_mode to stress-test on top of training noise.")
parser.add_argument("--enable_backtrack_check", action="store_true", default=False,
                    help="Re-enable the Fix B prev-gate backtrack detection. Default OFF for sweeps: "
                         "the check is a training-time termination, has no real-world analog, and "
                         "falsely kills legit loop trajectories when gate_side differs from training.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
import src.isaac_quad_sim2real.tasks  # noqa: F401


# ── Sweep axis definitions ───────────────────────────────────────────────────
# Each axis describes how to write per-env values into the sim tensors. Modes:
#   "multiplier" : sweep value is a scale factor applied to cfg nominal
#   "absolute"   : sweep value is written verbatim
# apply(env_unwrapped, values_per_env_tensor) writes into the live physics state.

def _apply_twr(env, vals):
    env._thrust_to_weight.copy_(vals)

def _apply_k_aero_xy(env, vals):
    env._K_aero[:, 0].copy_(vals)
    env._K_aero[:, 1].copy_(vals)

def _apply_k_aero_z(env, vals):
    env._K_aero[:, 2].copy_(vals)

def _apply_kp_omega_rp(env, vals):
    env._kp_omega[:, 0].copy_(vals)
    env._kp_omega[:, 1].copy_(vals)

def _apply_ki_omega_rp(env, vals):
    env._ki_omega[:, 0].copy_(vals)
    env._ki_omega[:, 1].copy_(vals)

def _apply_kd_omega_rp(env, vals):
    env._kd_omega[:, 0].copy_(vals)
    env._kd_omega[:, 1].copy_(vals)

def _apply_kp_omega_y(env, vals):
    env._kp_omega[:, 2].copy_(vals)

def _apply_ki_omega_y(env, vals):
    env._ki_omega[:, 2].copy_(vals)

def _apply_kd_omega_y(env, vals):
    env._kd_omega[:, 2].copy_(vals)

def _apply_motor_tau(env, vals):
    # _tau_m is (num_envs, 4) — replicate across motors
    env._tau_m.copy_(vals.unsqueeze(1).expand(-1, 4))

def _apply_action_latency(env, vals):
    # _action_delay is a long tensor
    env._action_delay.copy_(vals.long())


# Module-level state for axes not expressible as a write into an env tensor.
# Reset in run_sweep_trial() at the start of each trial.
_ACTION_NOISE_STD = None  # (num_envs,) tensor when active, else None

def _apply_action_noise_std(env, vals):
    # Noise is injected in run_sweep_trial after policy() — see `_ACTION_NOISE_STD`.
    global _ACTION_NOISE_STD
    _ACTION_NOISE_STD = vals.clone().to(env.device)

def _apply_gate_pos_jitter(env, vals):
    # Handled serially in run_gate_jitter_axis (one mini-trial per value).
    pass


SWEEP_AXES = {
    "twr": dict(
        label="Thrust-to-weight (× nominal)",
        mode="multiplier", nominal_attr="thrust_to_weight",
        values=[0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20],
        apply=_apply_twr,
    ),
    "k_aero_xy": dict(
        label="Aero drag xy (× nominal)",
        mode="multiplier", nominal_attr="k_aero_xy",
        values=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        apply=_apply_k_aero_xy,
    ),
    "k_aero_z": dict(
        label="Aero drag z (× nominal)",
        mode="multiplier", nominal_attr="k_aero_z",
        values=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        apply=_apply_k_aero_z,
    ),
    "kp_omega_rp": dict(
        label="PID kp roll/pitch (× nominal)",
        mode="multiplier", nominal_attr="kp_omega_rp",
        values=[0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30],
        apply=_apply_kp_omega_rp,
    ),
    "ki_omega_rp": dict(
        label="PID ki roll/pitch (× nominal)",
        mode="multiplier", nominal_attr="ki_omega_rp",
        values=[0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30],
        apply=_apply_ki_omega_rp,
    ),
    "kd_omega_rp": dict(
        label="PID kd roll/pitch (× nominal)",
        mode="multiplier", nominal_attr="kd_omega_rp",
        values=[0.60, 0.80, 0.95, 1.00, 1.05, 1.20, 1.40],
        apply=_apply_kd_omega_rp,
    ),
    "kp_omega_y": dict(
        label="PID kp yaw (× nominal)",
        mode="multiplier", nominal_attr="kp_omega_y",
        values=[0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30],
        apply=_apply_kp_omega_y,
    ),
    "motor_tau": dict(
        label="Motor time constant τ (× nominal)",
        mode="multiplier", nominal_attr="tau_m",
        values=[0.60, 0.80, 0.95, 1.00, 1.05, 1.20, 1.50],
        apply=_apply_motor_tau,
    ),
    "action_latency": dict(
        label="Action latency (policy steps)",
        mode="absolute", nominal_attr=None,
        values=[0, 1, 2, 3, 4, 5, 7],
        apply=_apply_action_latency,
    ),
    "action_noise_std": dict(
        label="Action command noise std (action units, [-1,1])",
        mode="absolute", nominal_attr=None,
        values=[0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        apply=_apply_action_noise_std,
    ),
    "gate_pos_jitter": dict(
        label="Gate position jitter (m, per-gate random xyz offset magnitude)",
        mode="absolute", nominal_attr=None,
        values=[0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        apply=_apply_gate_pos_jitter,
        serial=True,  # runs one mini-trial per value (waypoints are shared across envs)
    ),
}


def _get_nominal(cfg, axis_cfg):
    attr = axis_cfg.get("nominal_attr")
    if attr is None:
        return None
    val = getattr(cfg, attr, None)
    if val is None:
        raise AttributeError(f"cfg has no attribute '{attr}'")
    return float(val)


def _set_all_nominal(env_unwrapped, cfg):
    """Reset every sweep-controlled tensor to the cfg-nominal value."""
    dev = env_unwrapped.device
    n = env_unwrapped.num_envs

    def ones_nom(attr):
        return torch.full((n,), float(getattr(cfg, attr)), device=dev)

    env_unwrapped._thrust_to_weight.copy_(ones_nom("thrust_to_weight"))
    env_unwrapped._K_aero[:, 0].copy_(ones_nom("k_aero_xy"))
    env_unwrapped._K_aero[:, 1].copy_(ones_nom("k_aero_xy"))
    env_unwrapped._K_aero[:, 2].copy_(ones_nom("k_aero_z"))
    env_unwrapped._kp_omega[:, 0].copy_(ones_nom("kp_omega_rp"))
    env_unwrapped._kp_omega[:, 1].copy_(ones_nom("kp_omega_rp"))
    env_unwrapped._kp_omega[:, 2].copy_(ones_nom("kp_omega_y"))
    env_unwrapped._ki_omega[:, 0].copy_(ones_nom("ki_omega_rp"))
    env_unwrapped._ki_omega[:, 1].copy_(ones_nom("ki_omega_rp"))
    env_unwrapped._ki_omega[:, 2].copy_(ones_nom("ki_omega_y"))
    env_unwrapped._kd_omega[:, 0].copy_(ones_nom("kd_omega_rp"))
    env_unwrapped._kd_omega[:, 1].copy_(ones_nom("kd_omega_rp"))
    env_unwrapped._kd_omega[:, 2].copy_(ones_nom("kd_omega_y"))
    env_unwrapped._tau_m[:] = float(cfg.tau_m)
    # Leave _action_delay alone: reset_idx() randomizes it per training
    # distribution, which is what the policy was trained against. The
    # "action_latency" axis explicitly overrides this after _set_all_nominal().


def _partition_envs(num_envs, num_values, rng):
    """Assign each env to a value group as evenly as possible, shuffled."""
    base = num_envs // num_values
    rem  = num_envs - base * num_values
    group_sizes = [base + (1 if i < rem else 0) for i in range(num_values)]
    assignments = np.concatenate([
        np.full(sz, i, dtype=np.int64) for i, sz in enumerate(group_sizes)
    ])
    rng.shuffle(assignments)
    return assignments, group_sizes


def _build_per_env_values(axis_cfg, cfg, assignments, num_envs, device):
    """Turn per-env group assignments into a per-env tensor of axis values."""
    values = axis_cfg["values"]
    mode   = axis_cfg["mode"]
    nom    = _get_nominal(cfg, axis_cfg)
    actual = np.array([v * nom if mode == "multiplier" else v for v in values], dtype=np.float32)
    per_env_values = actual[assignments]  # (num_envs,)
    return torch.from_numpy(per_env_values).to(device), actual


def run_sweep_trial(env, policy, axis_name, axis_cfg, num_envs, max_steps,
                    target_laps, rng, device):
    """Run one sweep trial for a single axis.

    Returns a list of per-value dicts with success metrics.
    """
    unwrapped   = env.unwrapped
    cfg         = unwrapped.cfg
    num_gates   = unwrapped._waypoints.shape[0]
    target_g    = target_laps * num_gates
    dt          = cfg.sim.dt * cfg.decimation

    values = axis_cfg["values"]
    assignments, group_sizes = _partition_envs(num_envs, len(values), rng)
    per_env_vals, actual_values = _build_per_env_values(axis_cfg, cfg, assignments, num_envs, device)

    # Clear any sweep-scoped module state from a prior axis.
    global _ACTION_NOISE_STD
    _ACTION_NOISE_STD = None

    # Reset envs + baseline all params to nominal, then write sweep values.
    obs, _ = env.reset()
    if hasattr(obs, "get"):
        obs = obs["policy"]
    _set_all_nominal(unwrapped, cfg)
    axis_cfg["apply"](unwrapped, per_env_vals)

    finished    = torch.zeros(num_envs, dtype=torch.bool,  device=device)
    final_time  = torch.zeros(num_envs, dtype=torch.float, device=device)
    final_laps  = torch.zeros(num_envs, dtype=torch.int,   device=device)
    final_gates = torch.zeros(num_envs, dtype=torch.int,   device=device)
    crashed     = torch.zeros(num_envs, dtype=torch.bool,  device=device)

    t0 = time.time()
    for step in range(max_steps):
        with torch.no_grad():
            pre_gates = unwrapped._n_gates_passed.clone()
            pre_time  = unwrapped.episode_length_buf.float() * dt

            actions = policy(obs)
            if _ACTION_NOISE_STD is not None:
                actions = actions + torch.randn_like(actions) * _ACTION_NOISE_STD.unsqueeze(1)
            obs, _, dones, _ = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]

            # reset_idx() just fired DR for any resetting env — overwrite.
            _set_all_nominal(unwrapped, cfg)
            axis_cfg["apply"](unwrapped, per_env_vals)

            newly_3lap = (~finished) & (pre_gates >= target_g)
            if newly_3lap.any():
                ids = torch.where(newly_3lap)[0]
                final_time[ids]  = pre_time[ids]
                final_laps[ids]  = pre_gates[ids] // num_gates
                final_gates[ids] = pre_gates[ids]
                finished[ids]    = True

            newly_done = dones.bool() & ~finished
            if newly_done.any():
                ids = torch.where(newly_done)[0]
                final_time[ids]  = pre_time[ids]
                final_laps[ids]  = pre_gates[ids] // num_gates
                final_gates[ids] = pre_gates[ids]
                finished[ids]    = True
                # Anything that terminated without 3 laps = crash/timeout.
                crashed[ids]     = pre_gates[ids] < target_g

        if step % 400 == 0:
            done_n = int(finished.sum().item())
            succ_n = int((final_laps >= target_laps).sum().item())
            print(f"  [{axis_name}] step {step}/{max_steps} | "
                  f"done {done_n}/{num_envs} (succ {succ_n}) | "
                  f"{time.time()-t0:.0f}s")
        if finished.all():
            break

    # Anything still running at max_steps: treat as non-success (timeout).
    nf = ~finished
    if nf.any():
        ids = torch.where(nf)[0]
        final_time[ids]  = unwrapped.episode_length_buf[ids].float() * dt
        final_gates[ids] = unwrapped._n_gates_passed[ids]
        final_laps[ids]  = final_gates[ids] // num_gates
        crashed[ids]     = final_laps[ids] < target_laps

    # Aggregate per value.
    laps_np = final_laps.cpu().numpy()
    time_np = final_time.cpu().numpy()
    crash_np = crashed.cpu().numpy()
    success_np = laps_np >= target_laps

    rows = []
    for i, v in enumerate(values):
        mask = assignments == i
        n = int(mask.sum())
        if n == 0:
            continue
        succ = success_np[mask]
        sr = float(succ.mean())
        if succ.any():
            lap_t = time_np[mask][succ]
            mean_lap = float(lap_t.mean())
            p90_lap = float(np.percentile(lap_t, 90))
            worst_lap = float(lap_t.max())
        else:
            mean_lap = p90_lap = worst_lap = float("nan")
        crash_rate = float(crash_np[mask].mean())
        rows.append({
            "axis": axis_name,
            "sweep_value": float(v),
            "actual_value": float(actual_values[i]),
            "n": n,
            "success_rate": sr,
            "crash_rate": crash_rate,
            "mean_lap_s": mean_lap,
            "p90_lap_s": p90_lap,
            "worst_lap_s": worst_lap,
        })
    return rows


def run_gate_jitter_axis(env, policy, axis_name, axis_cfg, num_envs, max_steps,
                          target_laps, rng, device):
    """Gate-pos jitter axis: one mini-trial per value.

    Waypoints are a shared (num_gates, 7) tensor — we can't give each env its own
    gate layout without refactoring the env. So we serialize: for each sweep
    value v, apply a per-gate random xyz offset with magnitude ≈ v, run a full
    mini-trial with all num_envs, record aggregate, then restore.
    """
    unwrapped = env.unwrapped
    cfg       = unwrapped.cfg
    num_gates = unwrapped._waypoints.shape[0]
    target_g  = target_laps * num_gates
    dt        = cfg.sim.dt * cfg.decimation
    values    = axis_cfg["values"]

    original_wp = unwrapped._waypoints[:, :3].clone()

    global _ACTION_NOISE_STD
    _ACTION_NOISE_STD = None

    rows = []
    try:
        for i, v in enumerate(values):
            # Per-gate independent offset. Each axis ~ N(0, v/sqrt(3)) so the
            # expected radius (3D norm) is ~v.
            if float(v) > 0.0:
                sigma = float(v) / np.sqrt(3)
                offsets = torch.from_numpy(
                    rng.normal(0.0, sigma, size=(num_gates, 3)).astype(np.float32)
                ).to(device)
                unwrapped._waypoints[:, :3] = original_wp + offsets
            else:
                unwrapped._waypoints[:, :3] = original_wp

            obs, _ = env.reset()
            if hasattr(obs, "get"):
                obs = obs["policy"]
            _set_all_nominal(unwrapped, cfg)

            finished   = torch.zeros(num_envs, dtype=torch.bool,  device=device)
            final_time = torch.zeros(num_envs, dtype=torch.float, device=device)
            final_laps = torch.zeros(num_envs, dtype=torch.int,   device=device)
            crashed    = torch.zeros(num_envs, dtype=torch.bool,  device=device)

            t0 = time.time()
            for step in range(max_steps):
                with torch.no_grad():
                    pre_gates = unwrapped._n_gates_passed.clone()
                    pre_time  = unwrapped.episode_length_buf.float() * dt
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if hasattr(obs, "get"):
                        obs = obs["policy"]
                    _set_all_nominal(unwrapped, cfg)

                    newly_3lap = (~finished) & (pre_gates >= target_g)
                    if newly_3lap.any():
                        ids = torch.where(newly_3lap)[0]
                        final_time[ids]  = pre_time[ids]
                        final_laps[ids]  = pre_gates[ids] // num_gates
                        finished[ids]    = True
                    newly_done = dones.bool() & ~finished
                    if newly_done.any():
                        ids = torch.where(newly_done)[0]
                        final_time[ids]  = pre_time[ids]
                        final_laps[ids]  = pre_gates[ids] // num_gates
                        finished[ids]    = True
                        crashed[ids]     = pre_gates[ids] < target_g

                if finished.all():
                    break
                if step % 400 == 0:
                    print(f"  [{axis_name}={v:.3f}] step {step}/{max_steps} | "
                          f"done {int(finished.sum())}/{num_envs} | "
                          f"{time.time()-t0:.0f}s")

            nf = ~finished
            if nf.any():
                ids = torch.where(nf)[0]
                final_time[ids]  = unwrapped.episode_length_buf[ids].float() * dt
                final_laps[ids]  = unwrapped._n_gates_passed[ids] // num_gates
                crashed[ids]     = final_laps[ids] < target_laps

            laps_np  = final_laps.cpu().numpy()
            time_np  = final_time.cpu().numpy()
            crash_np = crashed.cpu().numpy()
            success_np = laps_np >= target_laps

            sr = float(success_np.mean())
            if success_np.any():
                lap_t   = time_np[success_np]
                mean_lap = float(lap_t.mean())
                p90_lap  = float(np.percentile(lap_t, 90))
                worst_lap = float(lap_t.max())
            else:
                mean_lap = p90_lap = worst_lap = float("nan")

            rows.append({
                "axis": axis_name,
                "sweep_value": float(v),
                "actual_value": float(v),
                "n": int(num_envs),
                "success_rate": sr,
                "crash_rate": float(crash_np.mean()),
                "mean_lap_s": mean_lap,
                "p90_lap_s": p90_lap,
                "worst_lap_s": worst_lap,
            })
    finally:
        unwrapped._waypoints[:, :3] = original_wp

    return rows


def plot_axis(axis_name, axis_cfg, rows, out_path):
    xs = [r["sweep_value"] for r in rows]
    sr = [r["success_rate"] * 100 for r in rows]
    cr = [r["crash_rate"]   * 100 for r in rows]
    mean_lap = [r["mean_lap_s"] for r in rows]
    p90_lap  = [r["p90_lap_s"]  for r in rows]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.suptitle(f"Robustness sweep — {axis_cfg['label']}", fontsize=12)

    axA.plot(xs, sr, "o-", color="#2e7d32", lw=2, label="Success rate")
    axA.plot(xs, cr, "s--", color="#c62828", lw=1.5, label="Crash/fail rate")
    if axis_cfg["mode"] == "multiplier":
        axA.axvline(1.0, color="grey", ls=":", lw=1, alpha=0.7, label="Nominal")
    axA.set_xlabel(axis_cfg["label"])
    axA.set_ylabel("Rate (%)")
    axA.set_ylim(-5, 108)
    axA.grid(alpha=0.3)
    axA.legend(loc="best", fontsize=9)

    has_any = any(not np.isnan(v) for v in mean_lap)
    if has_any:
        axB.plot(xs, mean_lap, "o-", color="#1565c0", lw=2, label="Mean lap (succ only)")
        axB.plot(xs, p90_lap,  "s--", color="#f9a825", lw=1.5, label="P90 lap")
        if axis_cfg["mode"] == "multiplier":
            axB.axvline(1.0, color="grey", ls=":", lw=1, alpha=0.7)
    else:
        axB.text(0.5, 0.5, "No successful completions", transform=axB.transAxes,
                 ha="center", va="center", color="red")
    axB.set_xlabel(axis_cfg["label"])
    axB.set_ylabel("Lap time (s)")
    axB.grid(alpha=0.3)
    axB.legend(loc="best", fontsize=9)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]
    out_dir = args_cli.output_dir or os.path.join(log_dir, "robustness_sweep", ckpt_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Output:     {out_dir}")

    # Parse & validate axes list.
    axes_requested = [a.strip() for a in args_cli.axes.split(",") if a.strip()]
    unknown = [a for a in axes_requested if a not in SWEEP_AXES]
    if unknown:
        raise ValueError(f"Unknown axes {unknown}. Available: {list(SWEEP_AXES.keys())}")

    # ── Init env ─────────────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    if "ENV_OVERRIDES" in os.environ:
        try:
            env_overrides = json.loads(os.environ["ENV_OVERRIDES"])
            for k, v in env_overrides.items():
                setattr(env_cfg, k, v)
            print(f"[INFO] Applied ENV_OVERRIDES: {env_overrides}")
        except Exception as e:
            print(f"[Warning] Failed to parse ENV_OVERRIDES: {e}")
    env_cfg.is_train = bool(args_cli.train_mode)
    env_cfg.seed = args_cli.seed
    if args_cli.gate_side is not None:
        env_cfg.gate_model.gate_side = float(args_cli.gate_side)
        print(f"[INFO] OVERRIDING gate_model.gate_side = {args_cli.gate_side} "
              f"(cfg default = {env_cfg.gate_model.gate_side}). "
              f"WARNING: this also shifts obs gate-corner positions; expect OOD obs.")
    else:
        print(f"[INFO] gate_model.gate_side = {env_cfg.gate_model.gate_side} (cfg default, matches training).")
    # Note: --disable_backtrack_check is applied AFTER env creation directly on
    # the strategy object (env_cfg attr might not survive configclass validation).
    print(f"[INFO] is_train={env_cfg.is_train}  "
          f"({'training-noise mode' if env_cfg.is_train else 'eval mode — no obs noise, no random DR'})")
    env_cfg.rewards = {
        'gate_pass_reward_scale':       0.0,
        'progress_goal_reward_scale':   0.0,
        'lap_complete_reward_scale':    0.0,
        'death_cost':                   0.0,
        'lap_incomplete_penalty_scale': 0.0,
        'cmd_reg_rp_scale':             0.0,
        'cmd_reg_yaw_scale':            0.0,
        'crash_reward_scale':           0.0,
        'crash_contact_scale':          0.0,
        'cmd_smoothness_scale':         0.0,
    }
    # Keep env_cfg.action_latency_max at training default so that reset_idx's
    # randomization (in train_mode) stays in the distribution the policy was
    # trained against. Buffer capacity comes from a separate sized init below.
    training_action_latency_max = int(env_cfg.action_latency_max)
    if "action_latency" in axes_requested:
        max_lat = int(max(SWEEP_AXES["action_latency"]["values"]))
        # Bump only the *buffer size* (via cfg, read in env __init__) up to the
        # largest sweep value so per-env overrides can index that far back.
        env_cfg.action_latency_max = max(env_cfg.action_latency_max, max_lat)
        print(f"[INFO] Buffer sized for action_latency_max={env_cfg.action_latency_max}; "
              f"random-delay distribution stays at training default={training_action_latency_max}.")

    env = gym.make(args_cli.task, cfg=env_cfg, rewards=env_cfg.rewards)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Restore reset_idx's random-delay upper bound to the training default so
    # non-latency sweeps run in-distribution. The buffer capacity (set via cfg
    # during env __init__) is unchanged.
    env.unwrapped._action_latency_max = training_action_latency_max

    # Backtrack check defaults OFF for sweeps: it's a training-time termination
    # with no real-world analog, and stale-policy / gate_side mismatches make
    # it fire on legitimate loop trajectories. Opt back in with --enable_backtrack_check.
    strategy = env.unwrapped.strategy
    if not hasattr(strategy, "_backtrack_check_enabled"):
        raise RuntimeError(
            "Strategy has no _backtrack_check_enabled attribute — "
            "strategy file may not have been reloaded. Restart the process."
        )
    strategy._backtrack_check_enabled = bool(args_cli.enable_backtrack_check)
    print(f"[INFO] Fix B backtrack check: "
          f"strategy._backtrack_check_enabled={strategy._backtrack_check_enabled} "
          f"({'enabled by flag' if args_cli.enable_backtrack_check else 'DISABLED (default for sweeps)'}).")

    rng = np.random.default_rng(args_cli.seed)

    all_rows = []
    for axis_name in axes_requested:
        axis_cfg = SWEEP_AXES[axis_name]
        print(f"\n{'#'*60}\n  SWEEP: {axis_name}  —  {axis_cfg['label']}\n{'#'*60}")
        print(f"  Values: {axis_cfg['values']}")
        # For the latency axis, raise the reset-time randomization ceiling so
        # sweep-assigned delays don't get clobbered before the per-env
        # override. (In eval mode reset_idx doesn't randomize delay anyway.)
        if axis_name == "action_latency":
            env.unwrapped._action_latency_max = int(max(axis_cfg["values"]))
        else:
            env.unwrapped._action_latency_max = training_action_latency_max
        if axis_cfg.get("serial", False):
            rows = run_gate_jitter_axis(
                env, policy, axis_name, axis_cfg,
                num_envs=args_cli.num_envs,
                max_steps=args_cli.max_steps,
                target_laps=args_cli.target_laps,
                rng=rng,
                device=env.unwrapped.device,
            )
        else:
            rows = run_sweep_trial(
                env, policy, axis_name, axis_cfg,
                num_envs=args_cli.num_envs,
                max_steps=args_cli.max_steps,
                target_laps=args_cli.target_laps,
                rng=rng,
                device=env.unwrapped.device,
            )
        for r in rows:
            print(f"  {axis_name}={r['sweep_value']:>6.3f}  "
                  f"(actual={r['actual_value']:.4f})  "
                  f"n={r['n']:3d}  SR={r['success_rate']*100:5.1f}%  "
                  f"crash={r['crash_rate']*100:5.1f}%  "
                  f"mean_lap={r['mean_lap_s']:.2f}s  P90={r['p90_lap_s']:.2f}s")
        all_rows.extend(rows)

        plot_path = os.path.join(out_dir, f"sweep_{axis_name}.png")
        plot_axis(axis_name, axis_cfg, rows, plot_path)
        print(f"  [Plot] {plot_path}")

    # Combined CSV.
    csv_path = os.path.join(out_dir, "robustness_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n[CSV] {csv_path}")

    # JSON summary.
    json_path = os.path.join(out_dir, "robustness_sweep.json")
    with open(json_path, "w") as f:
        json.dump({
            "checkpoint": resume_path,
            "num_envs": args_cli.num_envs,
            "max_steps": args_cli.max_steps,
            "target_laps": args_cli.target_laps,
            "rows": all_rows,
        }, f, indent=2)
    print(f"[JSON] {json_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

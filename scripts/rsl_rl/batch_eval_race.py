#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch self-evaluation script: each env can get its own sampled TA params.

Design rationale
----------------
The TA evaluation randomly picks 3 parameters from the pool and holds them
constant for *all* students.  To self-evaluate robustness we want to cover as
many such 3-parameter combinations as possible.

Strategy: run N_ENVS parallel environments simultaneously.  For each env we
independently sample *which* parameters to perturb and sample values within the
official bounds.  This gives N_ENVS distinct parameter configurations in a
single simulation run — or, when `--num_params_per_env 0`, one shared global
configuration across all envs.

We repeat this for N_TRIALS rounds (re-sampling all per-env parameters each
round) to get variance estimates.

Metrics reported per trial
--------------------------
  • 3-lap success rate  — fraction of envs that completed 3 laps
  • mean 3-lap time     — average time of envs that did complete 3 laps

Outputs
-------
  • Terminal summary (per trial + overall)
  • batch_eval_results.json
  • batch_eval_per_trial.png   — success rate & mean lap time per trial
  • batch_eval_param_usage.png — how often each param was sampled & its win-rate
  • batch_eval_lap_dist.png    — pooled lap-count distribution

Usage
-----
    python scripts/rsl_rl/batch_eval_race.py \\
        --task Isaac-Quadcopter-Race-v0 \\
        --load_run 2026-03-31_12-00-00 \\
        --checkpoint best_model.pt \\
        --headless \\
        --num_trials 5 \\
        --num_envs 100 \\
        --max_steps 3000
"""

"""Launch Isaac Sim Simulator first."""

import sys, os

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")

import argparse, time, json, random

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch eval — optional per-env TA-style DR.")
parser.add_argument("--num_trials",  type=int, default=5,    help="Number of evaluation rounds.")
parser.add_argument("--num_envs",    type=int, default=100,  help="Parallel envs (= param combos per trial).")
parser.add_argument("--max_steps",   type=int, default=3000, help="Max sim steps per trial.")
parser.add_argument("--num_params_per_env", type=int, default=3,
                    help="How many TA params to perturb per env. Use 0 to keep all envs on the shared global config.")
parser.add_argument("--task",        type=str, default=None)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--output_dir",  type=str, default=None, help="Where to save charts/JSON.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--video",       action="store_true", default=False)
parser.add_argument("--video_length",type=int, default=1600)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks          # noqa: F401
import src.isaac_quad_sim2real.tasks  # noqa: F401


def apply_env_overrides(env_cfg, env_overrides):
    """Apply flat or dotted env overrides, with a gate_side convenience alias."""
    for key, value in env_overrides.items():
        parts = ["gate_model", "gate_side"] if key == "gate_side" else key.split(".")
        target = env_cfg
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)


# ── TA parameter pool ─────────────────────────────────────────────────────────
# Each entry: (name, cfg_attr_for_nominal, scale_lo, scale_hi)
# Actual tensor writes are done manually in apply_per_env_params() because the
# env uses combined tensors (_K_aero, _kp_omega, etc.) with column indices.
TA_PARAM_POOL = [
    ("twr",         "thrust_to_weight", 0.95, 1.05),
    ("k_aero_xy",   "k_aero_xy",        0.50, 2.00),
    ("k_aero_z",    "k_aero_z",         0.50, 2.00),
    ("kp_omega_rp", "kp_omega_rp",      0.85, 1.15),
    ("ki_omega_rp", "ki_omega_rp",      0.85, 1.15),
    ("kd_omega_rp", "kd_omega_rp",      0.70, 1.30),
    ("kp_omega_y",  "kp_omega_y",       0.85, 1.15),
    ("ki_omega_y",  "ki_omega_y",       0.85, 1.15),
    ("kd_omega_y",  "kd_omega_y",       0.70, 1.30),
]

PARAM_LABELS = {
    "twr":         "TWR",
    "k_aero_xy":   "k_aero_xy",
    "k_aero_z":    "k_aero_z",
    "kp_omega_rp": "kp_ω_rp",
    "ki_omega_rp": "ki_ω_rp",
    "kd_omega_rp": "kd_ω_rp",
    "kp_omega_y":  "kp_ω_y",
    "ki_omega_y":  "ki_ω_y",
    "kd_omega_y":  "kd_ω_y",
}

CFG_ATTR_MAP = {
    "twr":         "thrust_to_weight",
    "k_aero_xy":   "k_aero_xy",
    "k_aero_z":    "k_aero_z",
    "kp_omega_rp": "kp_omega_rp",
    "ki_omega_rp": "ki_omega_rp",
    "kd_omega_rp": "kd_omega_rp",
    "kp_omega_y":  "kp_omega_y",
    "ki_omega_y":  "ki_omega_y",
    "kd_omega_y":  "kd_omega_y",
}


def get_nominal(cfg, param_name: str) -> float:
    attr = CFG_ATTR_MAP[param_name]
    val = getattr(cfg, attr, None)
    if val is None:
        raise AttributeError(f"cfg has no attribute '{attr}' for param '{param_name}'")
    return float(val)


def sample_per_env_params(cfg, num_envs: int, num_params: int = 3, rng=None):
    """For each env, independently sample which 3 params to perturb and their values.

    Returns
    -------
    env_configs : list[list[dict]]
        env_configs[env_id] = [{"name":, "value":, "scale":, ...}, ...]
    param_values : dict[param_name -> np.array shape (num_envs,)]
        Keyed by TA param name (e.g. "twr", "k_aero_xy"), NOT tensor attr name.
        Pass directly to apply_per_env_params().
    """
    if rng is None:
        rng = random.Random()

    nominals = {name: get_nominal(cfg, name) for name, *_ in TA_PARAM_POOL}
    # Start all envs at nominal value for every param
    param_values = {name: np.full(num_envs, nominals[name], dtype=np.float32)
                    for name, *_ in TA_PARAM_POOL}

    env_configs = []
    for env_id in range(num_envs):
        chosen = rng.sample(TA_PARAM_POOL, num_params)
        env_cfg = []
        for name, cfg_attr, lo, hi in chosen:
            scale = rng.uniform(lo, hi)
            value = nominals[name] * scale
            param_values[name][env_id] = value
            env_cfg.append({"name": name, "value": value, "scale": scale, "lo": lo, "hi": hi})
        env_configs.append(env_cfg)

    return env_configs, param_values



def apply_per_env_params(env_unwrapped, env_param_values: dict):
    """Write per-env sampled physics values into the correct tensor slices.

    env_param_values maps param name -> np.array of shape (num_envs,) with the
    value for each env.  This must be called after every reset so that
    reset_idx()'s _randomize_dynamics() / _set_default_dynamics() calls don't
    overwrite our intentional per-env values.

    Tensor layout (from quadcopter_strategies.py):
      _thrust_to_weight : (num_envs,)
      _K_aero           : (num_envs, 3)  cols: [0]=xy, [1]=xy, [2]=z
      _kp_omega         : (num_envs, 3)  cols: [0]=rp, [1]=rp, [2]=y
      _ki_omega         : (num_envs, 3)  cols: same
      _kd_omega         : (num_envs, 3)  cols: same
    """
    dev = env_unwrapped.device

    def _t(arr):  # numpy -> cuda tensor
        return torch.from_numpy(arr).to(dev)

    if "twr" in env_param_values:
        env_unwrapped._thrust_to_weight.copy_(_t(env_param_values["twr"]))

    if "k_aero_xy" in env_param_values:
        v = _t(env_param_values["k_aero_xy"])
        env_unwrapped._K_aero[:, 0].copy_(v)
        env_unwrapped._K_aero[:, 1].copy_(v)

    if "k_aero_z" in env_param_values:
        env_unwrapped._K_aero[:, 2].copy_(_t(env_param_values["k_aero_z"]))

    if "kp_omega_rp" in env_param_values:
        v = _t(env_param_values["kp_omega_rp"])
        env_unwrapped._kp_omega[:, 0].copy_(v)
        env_unwrapped._kp_omega[:, 1].copy_(v)

    if "kp_omega_y" in env_param_values:
        env_unwrapped._kp_omega[:, 2].copy_(_t(env_param_values["kp_omega_y"]))

    if "ki_omega_rp" in env_param_values:
        v = _t(env_param_values["ki_omega_rp"])
        env_unwrapped._ki_omega[:, 0].copy_(v)
        env_unwrapped._ki_omega[:, 1].copy_(v)

    if "ki_omega_y" in env_param_values:
        env_unwrapped._ki_omega[:, 2].copy_(_t(env_param_values["ki_omega_y"]))

    if "kd_omega_rp" in env_param_values:
        v = _t(env_param_values["kd_omega_rp"])
        env_unwrapped._kd_omega[:, 0].copy_(v)
        env_unwrapped._kd_omega[:, 1].copy_(v)

    if "kd_omega_y" in env_param_values:
        env_unwrapped._kd_omega[:, 2].copy_(_t(env_param_values["kd_omega_y"]))


def run_trial(env, policy, env_param_values: dict, max_steps: int, trial_label: str):
    """Run one trial with per-env DR params; track 3-lap completion pre-step.

    Why pre-step?
    - is_train=True: the env does NOT terminate on 3 laps.  _n_gates_passed is
      reset to 0 inside reset_idx() before env.step() returns.  So checking
      after step gives 0 for just-reset envs.
    - We save _n_gates_passed BEFORE each step and detect when it crosses
      3*num_gates.  Once crossed, we record the elapsed time and mark done.
    - After each step(), we also reapply env_param_values because reset_idx()
      (called for crashed/timed-out envs) calls _randomize_dynamics() which
      overwrites our per-env params.

    IMPORTANT: env.reset() is called at the start of each trial to clear
    _n_gates_passed from the previous trial. Without this, T2+ would see
    n_gates_passed >= target from T1 and instantly mark all envs as done.
    After reset, we reapply per-env DR params because reset_idx overwrites them.
    """
    # Force-reset all envs to clear _n_gates_passed (and episode_length_buf) from previous trial
    obs, _ = env.reset()
    if hasattr(obs, "get"):
        obs = obs["policy"]
    # Reapply per-env DR params (reset_idx's _randomize_dynamics() just overwrote them)
    apply_per_env_params(env.unwrapped, env_param_values)

    unwrapped  = env.unwrapped
    num_envs   = unwrapped.num_envs
    num_gates  = unwrapped._waypoints.shape[0]
    target_gates = 3 * num_gates   # 3 laps
    dt = unwrapped.cfg.sim.dt * unwrapped.cfg.decimation

    # Per-env tracking
    finished    = torch.zeros(num_envs, dtype=torch.bool,  device=unwrapped.device)
    final_time  = torch.zeros(num_envs, dtype=torch.float, device=unwrapped.device)
    final_laps  = torch.zeros(num_envs, dtype=torch.int,   device=unwrapped.device)
    final_gates = torch.zeros(num_envs, dtype=torch.int,   device=unwrapped.device)
    start_z     = unwrapped._robot.data.root_link_pos_w[:, 2].clone()
    max_z       = start_z.clone()
    ground_start = start_z <= 0.12
    first_gate_success = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)

    # Sim2real metrics: track control signals and flight quality across all envs
    # Running accumulators (no per-step storage to keep memory bounded)
    sum_thrust      = torch.zeros(num_envs, device=unwrapped.device)
    sum_thrust_sq   = torch.zeros(num_envs, device=unwrapped.device)
    sum_body_rate   = torch.zeros(num_envs, 3, device=unwrapped.device)  # roll, pitch, yaw
    sum_body_rate_sq = torch.zeros(num_envs, 3, device=unwrapped.device)
    sum_action_delta = torch.zeros(num_envs, device=unwrapped.device)  # action smoothness
    max_body_rate   = torch.zeros(num_envs, 3, device=unwrapped.device)
    max_tilt_angle  = torch.zeros(num_envs, device=unwrapped.device)
    step_counts     = torch.zeros(num_envs, dtype=torch.long, device=unwrapped.device)
    prev_actions    = torch.zeros(num_envs, unwrapped.cfg.action_space, device=unwrapped.device)

    t0 = time.time()
    for step in range(max_steps):
        with torch.no_grad():  # no_grad (not inference_mode) avoids marking tensors as inference tensors
            # ── Pre-step: snapshot gates before step resets done envs ────────
            pre_gates = unwrapped._n_gates_passed.clone()
            pre_time  = unwrapped.episode_length_buf.float() * dt

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]

            # ── Reapply per-env params (reset_idx overwrites them) ───────────
            apply_per_env_params(unwrapped, env_param_values)
            max_z = torch.maximum(max_z, unwrapped._robot.data.root_link_pos_w[:, 2])
            first_gate_success |= (pre_gates >= 1) | (unwrapped._n_gates_passed >= 1)

            # ── Accumulate sim2real metrics for active (not finished) envs ───
            active = ~finished
            if active.any():
                a_ids = torch.where(active)[0]
                # Thrust: actions[:, 0] is normalized [-1, 1], mapped to [0, 1]
                thrust_cmd = (actions[a_ids, 0] + 1.0) / 2.0
                sum_thrust[a_ids] += thrust_cmd
                sum_thrust_sq[a_ids] += thrust_cmd ** 2
                # Body rates: actions[:, 1:4] are normalized [-1, 1]
                br = actions[a_ids, 1:4]  # roll, pitch, yaw rate commands
                sum_body_rate[a_ids] += br.abs()
                sum_body_rate_sq[a_ids] += br ** 2
                max_body_rate[a_ids] = torch.max(max_body_rate[a_ids], br.abs())
                # Action smoothness: L2 norm of action change
                sum_action_delta[a_ids] += (actions[a_ids] - prev_actions[a_ids]).norm(dim=-1)
                # Tilt angle from gravity vector (how far from level)
                quat = unwrapped._robot.data.root_quat_w[a_ids]  # (n, 4) wxyz
                # z-component of body up vector in world = 2*(qw*qz + qx*qy) ... simplified:
                # cos(tilt) = 1 - 2*(qx^2 + qy^2) for small-angle
                qx, qy = quat[:, 1], quat[:, 2]
                cos_tilt = 1.0 - 2.0 * (qx**2 + qy**2)
                tilt_rad = torch.acos(cos_tilt.clamp(-1, 1))
                max_tilt_angle[a_ids] = torch.max(max_tilt_angle[a_ids], tilt_rad)

                step_counts[a_ids] += 1
                prev_actions[a_ids] = actions[a_ids].clone()

            # ── Detect 3-lap completion from pre-step values ─────────────────
            newly_3lap = (~finished) & (pre_gates >= target_gates)
            if newly_3lap.any():
                ids = torch.where(newly_3lap)[0]
                final_time[ids]  = pre_time[ids]
                final_laps[ids]  = pre_gates[ids] // num_gates
                final_gates[ids] = pre_gates[ids]
                finished[ids]    = True

            # ── Also capture envs that crashed/timed-out without 3 laps ──────
            newly_done = dones.bool() & ~finished
            if newly_done.any():
                ids = torch.where(newly_done)[0]
                # Use pre-step values (post-step _n_gates_passed already zeroed)
                final_time[ids]  = pre_time[ids]
                final_laps[ids]  = pre_gates[ids] // num_gates
                final_gates[ids] = pre_gates[ids]
                finished[ids]    = True

        if step % 200 == 0:
            n_3lap = (final_laps >= 3).sum().item()
            print(f"  [{trial_label}] step {step}/{max_steps} | "
                  f"{finished.sum().item()}/{num_envs} done "
                  f"(3-lap: {n_3lap}) | {time.time()-t0:.0f}s")
        if finished.all():
            print(f"  [{trial_label}] All {num_envs} envs done at step {step}!")
            break

    # Anything still running at max_steps — record current state
    nf = ~finished
    if nf.any():
        ids = torch.where(nf)[0]
        final_time[ids]  = unwrapped.episode_length_buf[ids].float() * dt
        final_gates[ids] = unwrapped._n_gates_passed[ids]
        final_laps[ids]  = final_gates[ids] // num_gates
    first_gate_success |= final_gates >= 1
    takeoff_threshold = torch.maximum(start_z + 0.15, torch.full_like(start_z, 0.20))
    takeoff_success = max_z >= takeoff_threshold

    # Compute per-env summary statistics
    safe_counts = step_counts.float().clamp(min=1)
    mean_thrust   = (sum_thrust / safe_counts).cpu().numpy()
    std_thrust    = (torch.sqrt(sum_thrust_sq / safe_counts - (sum_thrust / safe_counts)**2).clamp(min=0)).cpu().numpy()
    mean_br       = (sum_body_rate / safe_counts.unsqueeze(1)).cpu().numpy()       # (num_envs, 3)
    rms_br        = (torch.sqrt(sum_body_rate_sq / safe_counts.unsqueeze(1))).cpu().numpy()
    mean_smoothness = (sum_action_delta / safe_counts).cpu().numpy()
    max_br        = max_body_rate.cpu().numpy()
    max_tilt_deg  = (max_tilt_angle * 180.0 / 3.14159).cpu().numpy()

    return {
        "gates":    final_gates.cpu().numpy(),
        "time":     final_time.cpu().numpy(),
        "laps":     final_laps.cpu().numpy(),
        "finished": finished.cpu().numpy(),
        "ground_start": ground_start.cpu().numpy(),
        "takeoff_success": takeoff_success.cpu().numpy(),
        "first_gate_success": first_gate_success.cpu().numpy(),
        # Sim2real metrics
        "mean_thrust":      mean_thrust,
        "std_thrust":       std_thrust,
        "mean_body_rate":   mean_br,
        "rms_body_rate":    rms_br,
        "max_body_rate":    max_br,
        "mean_smoothness":  mean_smoothness,
        "max_tilt_deg":     max_tilt_deg,
    }


# ── Charts ────────────────────────────────────────────────────────────────────
PALETTE = dict(
    bg="#0f1117", card="#1a1d27", accent="#6c63ff", accent2="#ff6584",
    success="#43e97b", warn="#f7971e", fail="#ff4757",
    text="#e8eaf6", subtext="#9e9ebf", grid="#2a2d3e",
)


def _style(fig, axes):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PALETTE["card"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["grid"])
        ax.grid(color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)


def build_charts(trial_results, out_dir, num_envs):
    os.makedirs(out_dir, exist_ok=True)
    n = len(trial_results)
    total_eps = n * num_envs

    # Pool all per-env data across all trials
    all_laps  = np.concatenate([np.array(r["raw_laps"]) for r in trial_results])
    all_times = np.concatenate([np.array(r["raw_time"]) for r in trial_results])
    success_mask  = all_laps >= 3
    success_times = all_times[success_mask]
    success_count = int(success_mask.sum())
    fail_count    = int((~success_mask).sum())
    overall_sr    = 100.0 * success_count / total_eps

    # Chart 1: 3-lap time distribution histogram
    fig1, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    _style(fig1, ax)
    if len(success_times) > 0:
        mean_t   = float(success_times.mean())
        median_t = float(np.median(success_times))
        std_t    = float(success_times.std())
        n_bins = min(40, max(10, len(success_times) // 5))
        counts, edges, patches = ax.hist(
            success_times, bins=n_bins,
            color=PALETTE["accent"], edgecolor=PALETTE["bg"], linewidth=0.5,
            alpha=0.85, zorder=3,
        )
        for patch, left in zip(patches, edges[:-1]):
            if left < mean_t - std_t:
                patch.set_facecolor(PALETTE["success"])
            elif left > mean_t + std_t:
                patch.set_facecolor(PALETTE["warn"])
        ax.axvline(mean_t,   color=PALETTE["accent"],  lw=2.0, ls="--",
                   label=f"Mean   {mean_t:.2f}s", zorder=5)
        ax.axvline(median_t, color=PALETTE["accent2"], lw=2.0, ls=":",
                   label=f"Median {median_t:.2f}s", zorder=5)
        ax.axvline(mean_t - std_t, color=PALETTE["success"], lw=1.2, ls="-.", alpha=0.7,
                   label=f"Mean +/- 1 std ({std_t:.2f}s)", zorder=4)
        ax.axvline(mean_t + std_t, color=PALETTE["success"], lw=1.2, ls="-.", alpha=0.7, zorder=4)
        ax.set_xlabel("3-Lap Completion Time (s)", color=PALETTE["text"])
        ax.set_ylabel("# Environments", color=PALETTE["text"])
        ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"],
                  labelcolor=PALETTE["text"], fontsize=10)
        info = (f"n = {success_count} successful / {total_eps} total   SR = {overall_sr:.1f}%\n"
                f"mean = {mean_t:.2f}s   std = {std_t:.2f}s\n"
                f"best = {success_times.min():.2f}s   worst = {success_times.max():.2f}s")
        ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=9,
                va="top", ha="right", color=PALETTE["text"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["card"],
                          edgecolor=PALETTE["grid"], alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No successful completions", transform=ax.transAxes,
                ha="center", va="center", color=PALETTE["fail"], fontsize=14)
    fig1.suptitle(
        f"3-Lap Time Distribution  |  {n} trials x {num_envs} envs  |  SR = {overall_sr:.1f}%",
        color=PALETTE["text"], fontsize=13, fontweight="bold",
    )
    p1 = os.path.join(out_dir, "batch_eval_time_dist.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig1)
    print(f"[Chart] {p1}")

    # Chart 2: success rate overview (donut pie + per-trial bars)
    srs       = [r["success_rate_pct"] for r in trial_results]
    trial_ids = list(range(1, n + 1))
    fig2, (axP, axB) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    _style(fig2, [axP, axB])
    fig2.suptitle("Success Rate Overview", color=PALETTE["text"], fontsize=13, fontweight="bold")
    pie_vals   = [success_count, fail_count]
    pie_labels = [f"Success\n{success_count} ({overall_sr:.1f}%)",
                  f"Failed\n{fail_count} ({100-overall_sr:.1f}%)"]
    axP.pie(pie_vals, labels=pie_labels,
            colors=[PALETTE["success"], PALETTE["fail"]],
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"]),
            textprops=dict(color=PALETTE["text"], fontsize=10))
    axP.set_title(f"Overall -- {total_eps} env-episodes", color=PALETTE["text"])
    bar_colors = [PALETTE["success"] if s >= 80 else PALETTE["warn"] if s >= 50
                  else PALETTE["fail"] for s in srs]
    bars = axB.bar(trial_ids, srs, color=bar_colors, edgecolor=PALETTE["bg"],
                   linewidth=0.6, zorder=3)
    axB.axhline(overall_sr, color=PALETTE["accent"], lw=1.8, ls="--",
                label=f"Overall {overall_sr:.1f}%", zorder=4)
    axB.axhline(80, color=PALETTE["success"], lw=1.0, ls=":", alpha=0.6,
                label="Robust threshold (80%)", zorder=4)
    axB.set_ylim(0, 108)
    axB.set_ylabel("3-Lap Success Rate (%)", color=PALETTE["text"])
    axB.set_xlabel("Trial", color=PALETTE["text"])
    axB.set_xticks(trial_ids)
    axB.set_xticklabels([f"T{i}" for i in trial_ids], fontsize=9)
    axB.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=9)
    for bar, s in zip(bars, srs):
        axB.text(bar.get_x() + bar.get_width()/2, min(s + 1.5, 104),
                 f"{s:.0f}%", ha="center", va="bottom", fontsize=9, color=PALETTE["text"])
    axB.set_title("Per-Trial Success Rate", color=PALETTE["text"])
    p2 = os.path.join(out_dir, "batch_eval_success_rate.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig2)
    print(f"[Chart] {p2}")

    # Chart 3: per-param success rate + mean lap time
    param_used  = {name: 0   for name, *_ in TA_PARAM_POOL}
    param_succ  = {name: 0.0 for name, *_ in TA_PARAM_POOL}
    param_times = {name: []  for name, *_ in TA_PARAM_POOL}
    for r in trial_results:
        laps_arr  = np.array(r["raw_laps"])
        times_arr = np.array(r["raw_time"])
        for env_id, ecfg in enumerate(r["env_configs"]):
            ok = int(laps_arr[env_id] >= 3)
            for pcfg in ecfg:
                nm = pcfg["name"]
                param_used[nm] += 1
                param_succ[nm] += ok
                if ok:
                    param_times[nm].append(times_arr[env_id])
    total_param_uses = sum(param_used.values())
    if total_param_uses == 0:
        fig3, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        _style(fig3, ax)
        fig3.suptitle("Per-Parameter Analysis", color=PALETTE["text"], fontsize=12, fontweight="bold")
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No per-env TA parameter perturbations were enabled in this run.\n"
            "All envs shared the same global environment profile.",
            ha="center",
            va="center",
            color=PALETTE["text"],
            fontsize=12,
        )
    else:
        sorted_p = sorted(param_used.keys(),
                          key=lambda k: param_succ[k] / max(param_used[k], 1))
        p_labels = [PARAM_LABELS.get(k, k) for k in sorted_p]
        p_wr     = [100.0 * param_succ[k] / param_used[k] if param_used[k] > 0 else 0.0
                    for k in sorted_p]
        p_mean_t = [float(np.mean(param_times[k])) if param_times[k] else float("nan")
                    for k in sorted_p]
        fig3, (axA, axC) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        _style(fig3, [axA, axC])
        fig3.suptitle("Per-Parameter Analysis (pooled across all envs & trials)",
                      color=PALETTE["text"], fontsize=12, fontweight="bold")
        wr_colors = [PALETTE["success"] if w >= 80 else PALETTE["warn"] if w >= 50
                     else PALETTE["fail"] for w in p_wr]
        axA.barh(p_labels, p_wr, color=wr_colors, edgecolor=PALETTE["bg"], zorder=3)
        axA.axvline(80, color=PALETTE["success"], ls=":", lw=1.2, alpha=0.7)
        axA.set_xlim(0, 108)
        axA.set_xlabel("Success Rate when param perturbed (%)", color=PALETTE["text"])
        axA.set_title("Success Rate by Perturbed Param", color=PALETTE["text"])
        for i, wr in enumerate(p_wr):
            axA.text(min(wr + 1, 105), i, f"{wr:.0f}%", va="center", fontsize=8, color=PALETTE["text"])
        valid_t  = [t if not np.isnan(t) else 0 for t in p_mean_t]
        t_colors = [PALETTE["accent"] if not np.isnan(t) else PALETTE["fail"] for t in p_mean_t]
        axC.barh(p_labels, valid_t, color=t_colors, edgecolor=PALETTE["bg"], zorder=3)
        if len(success_times) > 0:
            axC.axvline(float(success_times.mean()), color=PALETTE["accent2"],
                        ls="--", lw=1.5, label=f"Overall mean {success_times.mean():.2f}s")
            axC.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"],
                       labelcolor=PALETTE["text"], fontsize=8)
        axC.set_xlabel("Mean 3-Lap Time when param perturbed (s)", color=PALETTE["text"])
        axC.set_title("Mean Lap Time by Perturbed Param", color=PALETTE["text"])
        for i, t in enumerate(p_mean_t):
            if not np.isnan(t):
                axC.text(t + 0.1, i, f"{t:.1f}s", va="center", fontsize=8, color=PALETTE["text"])
    p3 = os.path.join(out_dir, "batch_eval_param_analysis.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig3)
    print(f"[Chart] {p3}")

    return p1, p2, p3


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if args_cli.num_params_per_env < 0 or args_cli.num_params_per_env > len(TA_PARAM_POOL):
        raise ValueError(
            f"--num_params_per_env must be in [0, {len(TA_PARAM_POOL)}], got {args_cli.num_params_per_env}"
        )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir  = os.path.dirname(resume_path)
    # Use checkpoint name (without .pt) as subfolder for per-checkpoint comparison
    ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]
    out_dir  = args_cli.output_dir or os.path.join(log_dir, "batch_eval", ckpt_name)

    print(f"[INFO] Model: {resume_path}")
    print(f"[INFO] Checkpoint: {ckpt_name}")
    print(f"[INFO] Output: {out_dir}")

    # ── Init environment ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Initialising {args_cli.num_envs} parallel envs...")
    print(f"  (≈ 1-2 min)")
    print(f"{'='*60}")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    if "ENV_OVERRIDES" in os.environ:
        try:
            env_overrides = json.loads(os.environ["ENV_OVERRIDES"])
            apply_env_overrides(env_cfg, env_overrides)
            print(f"[INFO] Applied ENV_OVERRIDES: {env_overrides}")
        except Exception as e:
            print(f"[Warning] Failed to parse ENV_OVERRIDES: {e}")
    # is_train=True keeps DR hooks alive so we can write into the physics tensors
    env_cfg.is_train = True
    env_cfg.seed = args_cli.seed
    # env requires rewards in cfg when is_train=True — set directly on cfg (same as eval_race.py)
    env_cfg.rewards = {
        'gate_pass_reward_scale':        0.0,
        'progress_goal_reward_scale':    0.0,
        'lap_complete_reward_scale':     0.0,
        'death_cost':                    0.0,
        'lap_incomplete_penalty_scale':  0.0,
        'cmd_reg_rp_scale':              0.0,
        'cmd_reg_yaw_scale':             0.0,
        'crash_reward_scale':            0.0,
        'crash_contact_scale':           0.0,
        'cmd_smoothness_scale':          0.0,
    }

    env = gym.make(args_cli.task, cfg=env_cfg, rewards=env_cfg.rewards)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy    = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    unwrapped = env.unwrapped
    cfg       = unwrapped.cfg

    rng = random.Random(args_cli.seed)

    print(f"\n{'#'*60}")
    print(f"  BATCH EVAL: {args_cli.num_trials} trials × {args_cli.num_envs} envs")
    if args_cli.num_params_per_env > 0:
        print(f"  Each env: independently sample {args_cli.num_params_per_env} TA params")
    else:
        print(f"  Each env: shared global config only (no per-env TA param perturbation)")
    print(f"{'#'*60}")

    trial_results = []

    for t_idx in range(args_cli.num_trials):
        print(f"\n{'='*60}")
        print(f"  TRIAL {t_idx+1}/{args_cli.num_trials}")
        if args_cli.num_params_per_env > 0:
            print(
                f"  Sampling independent {args_cli.num_params_per_env}-param configs "
                f"for each of the {args_cli.num_envs} envs..."
            )
        else:
            print(f"  Reusing one shared global config across all {args_cli.num_envs} envs...")

        # Sample independent configs per env
        env_configs, param_values = sample_per_env_params(
            cfg, num_envs=args_cli.num_envs, num_params=args_cli.num_params_per_env, rng=rng
        )

        # Print a quick summary of what was sampled
        param_freq = {}
        for ecfg in env_configs:
            for pc in ecfg:
                param_freq[pc["name"]] = param_freq.get(pc["name"], 0) + 1
        if param_freq:
            print("  Param sampling frequency across envs:")
            for name, cnt in sorted(param_freq.items(), key=lambda x: -x[1]):
                print(f"    {PARAM_LABELS.get(name, name):<14} sampled in {cnt}/{args_cli.num_envs} envs")
        else:
            print("  Param sampling frequency across envs: none (shared global config)")

        # Apply to the live tensors (also called each step inside run_trial)
        apply_per_env_params(unwrapped, param_values)

        # Run
        raw = run_trial(env, policy, param_values, args_cli.max_steps, f"T{t_idx+1}")

        laps_arr  = raw["laps"]
        time_arr  = raw["time"]
        gates_arr = raw["gates"]
        takeoff_arr = raw["takeoff_success"].astype(bool)
        first_gate_arr = raw["first_gate_success"].astype(bool)
        ground_arr = raw["ground_start"].astype(bool)
        success_mask = laps_arr >= 3
        success_rate = float(success_mask.mean() * 100)
        mean_t = float(time_arr[success_mask].mean()) if success_mask.any() else float("nan")
        takeoff_rate = float(takeoff_arr.mean() * 100)
        first_gate_rate = float(first_gate_arr.mean() * 100)
        n_ground = int(ground_arr.sum())
        ground_takeoff_rate = float(takeoff_arr[ground_arr].mean() * 100) if n_ground > 0 else float("nan")
        ground_first_gate_rate = float(first_gate_arr[ground_arr].mean() * 100) if n_ground > 0 else float("nan")
        ground_success_rate = float(success_mask[ground_arr].mean() * 100) if n_ground > 0 else float("nan")

        # Extra-gate detection: clean 3-lap = exactly 3*num_gates passes.
        # If drone goes backward through a gate, gates_passed > target.
        num_gates_track = env.unwrapped._waypoints.shape[0]
        target_gates    = 3 * num_gates_track
        extra_gates_arr = gates_arr[success_mask].astype(int) - target_gates
        mean_extra = float(extra_gates_arr.mean()) if success_mask.any() else float("nan")
        pct_detour = float((extra_gates_arr > 0).mean() * 100) if success_mask.any() else float("nan")

        print(f"\n  TRIAL {t_idx+1} RESULT:")
        if n_ground > 0:
            print(f"    Ground starts       : {n_ground}/{args_cli.num_envs}")
            print(f"    Takeoff Success     : {ground_takeoff_rate:.1f}% on ground starts ({takeoff_rate:.1f}% overall)")
            print(f"    First-Gate Success  : {ground_first_gate_rate:.1f}% on ground starts ({first_gate_rate:.1f}% overall)")
            print(f"    Ground 3-Lap SR     : {ground_success_rate:.1f}% on ground starts")
        else:
            print(f"    Takeoff Success     : {takeoff_rate:.1f}%")
            print(f"    First-Gate Success  : {first_gate_rate:.1f}%")
        print(f"    3-Lap Success Rate : {success_rate:.1f}% ({success_mask.sum()}/{args_cli.num_envs})")
        if not np.isnan(mean_t):
            print(f"    Mean 3-Lap Time    : {mean_t:.2f}s")
            print(f"    Best 3-Lap Time    : {time_arr[success_mask].min():.2f}s")
            print(f"    Mean Extra Gates   : {mean_extra:.2f}  "
                  f"({'✅ clean' if mean_extra < 0.5 else f'⚠️ {pct_detour:.1f}% envs took detours'})")
        else:
            print(f"    Mean 3-Lap Time    : N/A (no completions)")

        # Control profile metrics (over successful envs)
        s2r = {}
        if success_mask.any():
            sm = success_mask
            tilt_arr = raw["max_tilt_deg"][sm]
            s2r["tilt_mean_deg"]  = float(tilt_arr.mean())
            s2r["tilt_p95_deg"]   = float(np.percentile(tilt_arr, 95))
            s2r["tilt_worst_deg"] = float(tilt_arr.max())
            # Peak body rate across axes, per successful env — then fraction near limit
            peak_br_per_env = np.max(raw["max_body_rate"][sm], axis=1)  # max of |roll|,|pitch|,|yaw|
            s2r["peak_rate_saturated_pct"] = float((peak_br_per_env > 0.9).mean() * 100)
            s2r["peak_rate_mean"]          = float(peak_br_per_env.mean())

            print(f"    ── Control profile ──")
            print(f"    Max tilt angle     : mean={s2r['tilt_mean_deg']:.0f}°  "
                  f"p95={s2r['tilt_p95_deg']:.0f}°  worst={s2r['tilt_worst_deg']:.0f}°")
            print(f"    Peak body-rate     : mean={s2r['peak_rate_mean']:.2f} of max  "
                  f"({s2r['peak_rate_saturated_pct']:.0f}% of envs pegged >0.9)")

        trial_results.append({
            "trial": t_idx + 1,
            "success_rate_pct": success_rate,
            "takeoff_success_pct": takeoff_rate,
            "first_gate_success_pct": first_gate_rate,
            "ground_takeoff_success_pct": ground_takeoff_rate,
            "ground_first_gate_success_pct": ground_first_gate_rate,
            "ground_success_rate_pct": ground_success_rate,
            "n_ground_starts": n_ground,
            "mean_3lap_time": mean_t,
            "mean_extra_gates": mean_extra,
            "pct_detour_envs": pct_detour,
            "sim2real": s2r,
            "env_configs": env_configs,
            "raw_laps":  laps_arr.tolist(),
            "raw_time":  time_arr.tolist(),
            "raw_gates": gates_arr.tolist(),
            "raw_finished": raw["finished"].tolist(),
            "raw_takeoff_success": takeoff_arr.tolist(),
            "raw_first_gate_success": first_gate_arr.tolist(),
            "raw_ground_start": ground_arr.tolist(),
        })

    # ── Overall summary ───────────────────────────────────────────────────────
    overall_sr   = float(np.mean([r["success_rate_pct"] for r in trial_results]))
    valid_times  = [r["mean_3lap_time"] for r in trial_results if not np.isnan(r["mean_3lap_time"])]
    overall_time = float(np.mean(valid_times)) if valid_times else float("nan")

    # Pool all successful 3-lap times for overall stats
    all_success_times = np.concatenate([
        np.array(r["raw_time"])[np.array(r["raw_laps"]) >= 3]
        for r in trial_results
    ]) if trial_results else np.array([])
    all_takeoff = np.concatenate([np.array(r["raw_takeoff_success"], dtype=bool) for r in trial_results]) if trial_results else np.array([], dtype=bool)
    all_first_gate = np.concatenate([np.array(r["raw_first_gate_success"], dtype=bool) for r in trial_results]) if trial_results else np.array([], dtype=bool)
    all_ground = np.concatenate([np.array(r["raw_ground_start"], dtype=bool) for r in trial_results]) if trial_results else np.array([], dtype=bool)
    all_laps = np.concatenate([np.array(r["raw_laps"]) for r in trial_results]) if trial_results else np.array([])
    all_success = all_laps >= 3
    overall_takeoff = float(all_takeoff.mean() * 100) if len(all_takeoff) > 0 else float("nan")
    overall_first_gate = float(all_first_gate.mean() * 100) if len(all_first_gate) > 0 else float("nan")
    overall_ground_takeoff = float(all_takeoff[all_ground].mean() * 100) if all_ground.any() else float("nan")
    overall_ground_first_gate = float(all_first_gate[all_ground].mean() * 100) if all_ground.any() else float("nan")
    overall_ground_success = float(all_success[all_ground].mean() * 100) if all_ground.any() else float("nan")

    print(f"\n{'#'*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'#'*60}")
    print(f"  Trials:        {args_cli.num_trials}")
    if args_cli.num_params_per_env > 0:
        print(f"  Envs/trial:    {args_cli.num_envs}  (each with independent {args_cli.num_params_per_env}-param DR)")
    else:
        print(f"  Envs/trial:    {args_cli.num_envs}  (shared global config)")
    print(f"  Total episodes:{args_cli.num_trials * args_cli.num_envs}")
    if all_ground.any():
        print(f"  Takeoff SR:    {overall_ground_takeoff:.1f}% on ground starts ({overall_takeoff:.1f}% overall)")
        print(f"  First-gate SR: {overall_ground_first_gate:.1f}% on ground starts ({overall_first_gate:.1f}% overall)")
    else:
        print(f"  Takeoff SR:    {overall_takeoff:.1f}%")
        print(f"  First-gate SR: {overall_first_gate:.1f}%")
    print(f"  Overall SR:    {overall_sr:.1f}%  ({int(len(all_success_times))}/{args_cli.num_trials * args_cli.num_envs} envs hit 3 laps)")
    if len(all_success_times) > 0:
        print(f"  3-lap time:    mean={all_success_times.mean():.2f}s  "
              f"std={all_success_times.std():.2f}s  "
              f"best={all_success_times.min():.2f}s  "
              f"worst={all_success_times.max():.2f}s")
    else:
        print(f"  3-lap time:    N/A (no completions)")
    print()
    for r in trial_results:
        ts = f"{r['mean_3lap_time']:.2f}s" if not np.isnan(r["mean_3lap_time"]) else "  N/A"
        print(f"  T{r['trial']:02d}  SR={r['success_rate_pct']:5.1f}%  time={ts}")

    # Aggregate control-profile metrics across trials
    s2r_trials = [r["sim2real"] for r in trial_results if r.get("sim2real")]
    if s2r_trials:
        print(f"\n  ── Control profile (pooled) ──")
        avg_tilt_mean  = float(np.mean([s["tilt_mean_deg"]  for s in s2r_trials]))
        avg_tilt_p95   = float(np.mean([s["tilt_p95_deg"]   for s in s2r_trials]))
        avg_tilt_worst = float(np.mean([s["tilt_worst_deg"] for s in s2r_trials]))
        avg_peak_rate  = float(np.mean([s["peak_rate_mean"] for s in s2r_trials]))
        avg_sat_pct    = float(np.mean([s["peak_rate_saturated_pct"] for s in s2r_trials]))
        print(f"  Max tilt:       mean={avg_tilt_mean:.0f}°  p95={avg_tilt_p95:.0f}°  worst={avg_tilt_worst:.0f}°")
        print(f"  Peak body-rate: mean={avg_peak_rate:.2f} of max  "
              f"({avg_sat_pct:.0f}% of envs pegged >0.9)")

    print(f"{'#'*60}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    json_out = []
    for r in trial_results:
        entry = {
            "trial": r["trial"],
            "success_rate_pct": r["success_rate_pct"],
            "takeoff_success_pct": r["takeoff_success_pct"],
            "first_gate_success_pct": r["first_gate_success_pct"],
            "ground_takeoff_success_pct": r["ground_takeoff_success_pct"] if not np.isnan(r["ground_takeoff_success_pct"]) else None,
            "ground_first_gate_success_pct": r["ground_first_gate_success_pct"] if not np.isnan(r["ground_first_gate_success_pct"]) else None,
            "ground_success_rate_pct": r["ground_success_rate_pct"] if not np.isnan(r["ground_success_rate_pct"]) else None,
            "n_ground_starts": r["n_ground_starts"],
            "mean_3lap_time": r["mean_3lap_time"] if not np.isnan(r["mean_3lap_time"]) else None,
            "laps_mean": float(np.mean(r["raw_laps"])),
            "laps_max": int(np.max(r["raw_laps"])),
            "n_finished": int(np.sum(r["raw_finished"])),
        }
        json_out.append(entry)

    json_path = os.path.join(out_dir, "batch_eval_results.json")
    with open(json_path, "w") as f:
        overall_entry = {
            "success_rate_pct": overall_sr,
            "takeoff_success_pct": overall_takeoff,
            "first_gate_success_pct": overall_first_gate,
            "ground_takeoff_success_pct": overall_ground_takeoff if not np.isnan(overall_ground_takeoff) else None,
            "ground_first_gate_success_pct": overall_ground_first_gate if not np.isnan(overall_ground_first_gate) else None,
            "ground_success_rate_pct": overall_ground_success if not np.isnan(overall_ground_success) else None,
            "n_ground_starts": int(all_ground.sum()),
            "mean_3lap_time": overall_time if not np.isnan(overall_time) else None,
        }
        if len(all_success_times) > 0:
            overall_entry.update({
                "best_3lap_time":  float(all_success_times.min()),
                "worst_3lap_time": float(all_success_times.max()),
                "std_3lap_time":   float(all_success_times.std()),
                "n_3lap_success":  int(len(all_success_times)),
            })
        json.dump({"overall": overall_entry, "trials": json_out}, f, indent=2)
    print(f"[INFO] JSON → {json_path}")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("[INFO] Generating charts...")
    build_charts(trial_results, out_dir, args_cli.num_envs)

    env.close()
    print(f"\n[INFO] Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

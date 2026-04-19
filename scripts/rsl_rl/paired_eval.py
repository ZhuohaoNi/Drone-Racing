#!/usr/bin/env python3
"""Paired evaluation: run two models on identical DR parameters.

Produces 5 diagnostic outputs:
  1. Scatter plot: T_model1 vs T_model2
  2. Improvement vs old performance: T_model1 vs Δ
  3. Rank correlation (Pearson + Spearman)
  4. Bucketed statistics by model1 performance
  5. Threshold crossing 2×2 table

Usage:
    python scripts/rsl_rl/paired_eval.py \\
        --task Isaac-Quadcopter-Race-v0 \\
        --model1_run best_v26 --model1_ckpt best_model.pt --model1_label V26 \\
        --model2_run 2026-04-03_02-05-58_finetune_v26_robust --model2_ckpt best_model.pt --model2_label V30 \\
        --headless --num_envs 1000 --max_steps 3000 --target_time 16.06
"""

"""Launch Isaac Sim Simulator first."""

import sys, os

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)

import argparse, time, json, random
from scipy import stats as scipy_stats

from isaaclab.app import AppLauncher
import cli_args

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Paired eval — same DR params, two models.")
parser.add_argument("--model1_run",   type=str, required=True, help="Run dir for model 1")
parser.add_argument("--model1_ckpt",  type=str, default="best_model.pt")
parser.add_argument("--model1_label", type=str, default="Model1")
parser.add_argument("--model2_run",   type=str, required=True, help="Run dir for model 2")
parser.add_argument("--model2_ckpt",  type=str, default="best_model.pt")
parser.add_argument("--model2_label", type=str, default="Model2")
parser.add_argument("--num_envs",     type=int, default=1000)
parser.add_argument("--max_steps",    type=int, default=3000)
parser.add_argument("--target_time",  type=float, default=16.06)
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--task",         type=str, default=None)
parser.add_argument("--output_dir",   type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--video",        action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1600)
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

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks
import src.isaac_quad_sim2real.tasks

# ── Reuse TA param pool from batch_eval ───────────────────────────────────────
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

CFG_ATTR_MAP = {name: attr for name, attr, *_ in TA_PARAM_POOL}


def get_nominal(cfg, param_name):
    return float(getattr(cfg, CFG_ATTR_MAP[param_name]))


def sample_fixed_params(cfg, num_envs, num_params=3, rng=None):
    """Generate a fixed set of DR params (same as batch_eval)."""
    if rng is None:
        rng = random.Random()
    nominals = {name: get_nominal(cfg, name) for name, *_ in TA_PARAM_POOL}
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
            env_cfg.append({"name": name, "value": value, "scale": scale})
        env_configs.append(env_cfg)
    return env_configs, param_values


def apply_per_env_params(env_unwrapped, pv):
    """Write per-env DR values into simulation tensors."""
    dev = env_unwrapped.device
    def _t(arr):
        return torch.from_numpy(arr).to(dev)
    if "twr" in pv:
        env_unwrapped._thrust_to_weight.copy_(_t(pv["twr"]))
    if "k_aero_xy" in pv:
        v = _t(pv["k_aero_xy"])
        env_unwrapped._K_aero[:, 0].copy_(v)
        env_unwrapped._K_aero[:, 1].copy_(v)
    if "k_aero_z" in pv:
        env_unwrapped._K_aero[:, 2].copy_(_t(pv["k_aero_z"]))
    if "kp_omega_rp" in pv:
        v = _t(pv["kp_omega_rp"])
        env_unwrapped._kp_omega[:, 0].copy_(v)
        env_unwrapped._kp_omega[:, 1].copy_(v)
    if "kp_omega_y" in pv:
        env_unwrapped._kp_omega[:, 2].copy_(_t(pv["kp_omega_y"]))
    if "ki_omega_rp" in pv:
        v = _t(pv["ki_omega_rp"])
        env_unwrapped._ki_omega[:, 0].copy_(v)
        env_unwrapped._ki_omega[:, 1].copy_(v)
    if "ki_omega_y" in pv:
        env_unwrapped._ki_omega[:, 2].copy_(_t(pv["ki_omega_y"]))
    if "kd_omega_rp" in pv:
        v = _t(pv["kd_omega_rp"])
        env_unwrapped._kd_omega[:, 0].copy_(v)
        env_unwrapped._kd_omega[:, 1].copy_(v)
    if "kd_omega_y" in pv:
        env_unwrapped._kd_omega[:, 2].copy_(_t(pv["kd_omega_y"]))


def run_eval(env, policy, param_values, max_steps, label):
    """Run one full eval pass. Returns per-env lap times."""
    obs, _ = env.reset()
    if hasattr(obs, "get"):
        obs = obs["policy"]
    apply_per_env_params(env.unwrapped, param_values)

    unwrapped = env.unwrapped
    num_envs  = unwrapped.num_envs
    num_gates = unwrapped._waypoints.shape[0]
    target_gates = 3 * num_gates
    dt = unwrapped.cfg.sim.dt * unwrapped.cfg.decimation

    finished   = torch.zeros(num_envs, dtype=torch.bool,  device=unwrapped.device)
    final_time = torch.full((num_envs,), float("nan"), dtype=torch.float, device=unwrapped.device)

    t0 = time.time()
    for step in range(max_steps):
        with torch.no_grad():
            pre_gates = unwrapped._n_gates_passed.clone()
            pre_time  = unwrapped.episode_length_buf.float() * dt

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]

            apply_per_env_params(unwrapped, param_values)

            newly_3lap = (~finished) & (pre_gates >= target_gates)
            if newly_3lap.any():
                ids = torch.where(newly_3lap)[0]
                final_time[ids] = pre_time[ids]
                finished[ids] = True

            newly_done = dones.bool() & ~finished
            if newly_done.any():
                ids = torch.where(newly_done)[0]
                final_time[ids] = pre_time[ids]
                finished[ids] = True

        if step % 500 == 0:
            n_ok = finished.sum().item()
            n_3 = (~final_time.isnan()).sum().item()
            print(f"  [{label}] step {step}/{max_steps} | {n_ok}/{num_envs} done | {time.time()-t0:.0f}s")
        if finished.all():
            print(f"  [{label}] All done at step {step}")
            break

    # Mark unfinished as NaN (failed)
    return final_time.cpu().numpy()


# ── Plotting ──────────────────────────────────────────────────────────────────
P = dict(bg="#0f1117", card="#1a1d27", accent="#6c63ff", accent2="#ff6584",
         success="#43e97b", warn="#f7971e", fail="#ff4757",
         text="#e8eaf6", grid="#2a2d3e")


def _style(fig, axes):
    fig.patch.set_facecolor(P["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(P["card"])
        ax.tick_params(colors=P["text"])
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["grid"])
        ax.grid(color=P["grid"], linewidth=0.5, linestyle="--", alpha=0.6)


def build_analysis(t1, t2, label1, label2, target, out_dir):
    """Generate all 5 diagnostic outputs. t1/t2 are per-env times (NaN=failed)."""
    os.makedirs(out_dir, exist_ok=True)

    # Only use envs where BOTH models completed 3 laps
    valid = ~(np.isnan(t1) | np.isnan(t2))
    t1v, t2v = t1[valid], t2[valid]
    n_valid = len(t1v)
    delta = t1v - t2v  # positive = model2 faster

    print(f"\n{'='*60}")
    print(f"  PAIRED ANALYSIS: {label1} vs {label2}")
    print(f"  {n_valid} envs where both completed 3 laps")
    print(f"{'='*60}")

    # ── 1. Scatter plot ───────────────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _style(fig1, ax)
    ax.scatter(t1v, t2v, s=6, alpha=0.4, color=P["accent"], zorder=3)
    lims = [min(t1v.min(), t2v.min()) - 0.2, max(t1v.max(), t2v.max()) + 0.2]
    ax.plot(lims, lims, "--", color=P["text"], alpha=0.5, lw=1, label="y = x")
    mean_delta = delta.mean()
    ax.plot(lims, [l - mean_delta for l in lims], "-.", color=P["success"], alpha=0.7, lw=1.5,
            label=f"y = x - {mean_delta:.2f}s (mean Δ)")
    ax.axvline(target, color=P["fail"], ls=":", alpha=0.6, lw=1)
    ax.axhline(target, color=P["fail"], ls=":", alpha=0.6, lw=1)
    ax.set_xlabel(f"{label1} time (s)")
    ax.set_ylabel(f"{label2} time (s)")
    ax.set_title(f"Paired Scatter: {label1} vs {label2} (n={n_valid})")
    ax.legend(facecolor=P["card"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    fig1.savefig(os.path.join(out_dir, "1_scatter.png"), dpi=150, facecolor=P["bg"])
    plt.close(fig1)

    # ── 2. Improvement vs old performance ─────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    _style(fig2, ax)
    colors = [P["success"] if d > 0 else P["fail"] for d in delta]
    ax.scatter(t1v, delta, s=8, alpha=0.4, c=colors, zorder=3)
    ax.axhline(0, color=P["text"], ls="--", alpha=0.4, lw=1)
    ax.axvline(target, color=P["fail"], ls=":", alpha=0.6, lw=1, label=f"Target {target}s")
    # Trend line
    z = np.polyfit(t1v, delta, 1)
    xfit = np.linspace(t1v.min(), t1v.max(), 100)
    ax.plot(xfit, np.polyval(z, xfit), "-", color=P["accent2"], lw=2,
            label=f"Trend: slope={z[0]:.3f}")
    ax.set_xlabel(f"{label1} time (s)")
    ax.set_ylabel(f"Δ = {label1} - {label2} (s)  [positive = {label2} faster]")
    ax.set_title(f"Improvement vs {label1} Performance")
    ax.legend(facecolor=P["card"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=9)
    fig2.savefig(os.path.join(out_dir, "2_improvement.png"), dpi=150, facecolor=P["bg"])
    plt.close(fig2)

    # ── 3. Rank correlation ───────────────────────────────────────────────────
    pearson_r, pearson_p = scipy_stats.pearsonr(t1v, t2v)
    spearman_r, spearman_p = scipy_stats.spearmanr(t1v, t2v)
    print(f"\n  ── Rank Correlation ──")
    print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.2e})")
    if spearman_r > 0.9:
        print(f"  → Rankings highly preserved. Fine-tune is mostly a shift.")
    elif spearman_r > 0.7:
        print(f"  → Rankings moderately preserved. Some reshuffling in sub-regions.")
    else:
        print(f"  → Rankings substantially changed. Different blind spots.")

    # ── 4. Bucketed statistics ────────────────────────────────────────────────
    buckets = [
        ("A: <15.6s",      t1v < 15.6),
        ("B: 15.6–15.9s",  (t1v >= 15.6) & (t1v < 15.9)),
        ("C: 15.9–16.1s",  (t1v >= 15.9) & (t1v < 16.1)),
        ("D: 16.1–16.4s",  (t1v >= 16.1) & (t1v < 16.4)),
        ("E: ≥16.4s",      t1v >= 16.4),
    ]
    print(f"\n  ── Bucketed Statistics (by {label1} performance) ──")
    print(f"  {'Bucket':<16} {'N':>5} {'Mean '+label1:>12} {'Mean '+label2:>12} {'Mean Δ':>10} {'Pass%'+label1:>10} {'Pass%'+label2:>10}")
    bucket_data = []
    for bname, mask in buckets:
        n = mask.sum()
        if n == 0:
            print(f"  {bname:<16} {0:>5}   {'—':>10} {'—':>10} {'—':>8} {'—':>8} {'—':>8}")
            continue
        m1 = t1v[mask].mean()
        m2 = t2v[mask].mean()
        md = delta[mask].mean()
        p1 = (t1v[mask] < target).mean() * 100
        p2 = (t2v[mask] < target).mean() * 100
        print(f"  {bname:<16} {n:>5} {m1:>10.2f}s {m2:>10.2f}s {md:>+9.2f}s {p1:>9.1f}% {p2:>9.1f}%")
        bucket_data.append({"bucket": bname, "n": int(n), f"mean_{label1}": m1,
                            f"mean_{label2}": m2, "mean_delta": md,
                            f"pass_rate_{label1}": p1, f"pass_rate_{label2}": p2})

    # Bucket bar chart
    fig4, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    _style(fig4, [axA, axB])
    bnames = [b["bucket"] for b in bucket_data]
    deltas_b = [b["mean_delta"] for b in bucket_data]
    pr1 = [b[f"pass_rate_{label1}"] for b in bucket_data]
    pr2 = [b[f"pass_rate_{label2}"] for b in bucket_data]
    x = np.arange(len(bnames))
    bar_colors = [P["success"] if d > 0 else P["fail"] for d in deltas_b]
    axA.bar(x, deltas_b, color=bar_colors, edgecolor=P["bg"], zorder=3)
    axA.axhline(0, color=P["text"], ls="--", alpha=0.4)
    axA.set_xticks(x)
    axA.set_xticklabels(bnames, fontsize=8, rotation=15)
    axA.set_ylabel(f"Mean Δ (s)  [positive = {label2} faster]")
    axA.set_title(f"Mean Improvement by {label1} Bucket")
    w = 0.35
    axB.bar(x - w/2, pr1, w, label=label1, color=P["accent"], edgecolor=P["bg"], zorder=3)
    axB.bar(x + w/2, pr2, w, label=label2, color=P["success"], edgecolor=P["bg"], zorder=3)
    axB.axhline(100, color=P["text"], ls=":", alpha=0.3)
    axB.set_xticks(x)
    axB.set_xticklabels(bnames, fontsize=8, rotation=15)
    axB.set_ylabel(f"Pass Rate < {target}s (%)")
    axB.set_title("Pass Rate by Bucket")
    axB.legend(facecolor=P["card"], edgecolor=P["grid"], labelcolor=P["text"])
    fig4.suptitle(f"Bucketed Analysis: {label1} vs {label2}", color=P["text"], fontweight="bold")
    fig4.savefig(os.path.join(out_dir, "4_buckets.png"), dpi=150, facecolor=P["bg"])
    plt.close(fig4)

    # ── 5. Threshold crossing 2×2 table ───────────────────────────────────────
    pass1 = t1v < target
    pass2 = t2v < target
    pp = int((pass1 & pass2).sum())
    pf = int((pass1 & ~pass2).sum())
    fp = int((~pass1 & pass2).sum())
    ff = int((~pass1 & ~pass2).sum())

    print(f"\n  ── Threshold Crossing Table (target = {target}s) ──")
    print(f"  {'':>20} {label2+' PASS':>14} {label2+' FAIL':>14} {'Total':>10}")
    print(f"  {label1+' PASS':<20} {pp:>14} {pf:>14} {pp+pf:>10}")
    print(f"  {label1+' FAIL':<20} {fp:>14} {ff:>14} {fp+ff:>10}")
    print(f"  {'Total':<20} {pp+fp:>14} {pf+ff:>14} {n_valid:>10}")
    print()
    print(f"  🔑 Key metric: {label1} FAIL → {label2} PASS = {fp} envs ({100*fp/max(fp+ff,1):.1f}% of {label1} failures rescued)")
    print(f"  ⚠️  Regression: {label1} PASS → {label2} FAIL = {pf} envs")

    # Save 2x2 as simple chart
    fig5, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    _style(fig5, ax)
    table_data = [[pp, pf], [fp, ff]]
    row_labels = [f"{label1} PASS", f"{label1} FAIL"]
    col_labels = [f"{label2} PASS", f"{label2} FAIL"]
    ax.axis("off")
    tbl = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1.4, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(P["grid"])
        if r == 0:
            cell.set_facecolor(P["accent"])
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor(P["card"])
            cell.set_text_props(color=P["text"])
        elif (r == 2 and c == 0):  # fail→pass
            cell.set_facecolor(P["success"])
            cell.set_text_props(color="black", fontweight="bold")
        elif (r == 1 and c == 1):  # pass→fail
            cell.set_facecolor(P["fail"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(P["card"])
            cell.set_text_props(color=P["text"])
    ax.set_title(f"Threshold Crossing Table (target = {target}s)", color=P["text"], fontsize=13, pad=20)
    fig5.savefig(os.path.join(out_dir, "5_threshold_table.png"), dpi=150, facecolor=P["bg"])
    plt.close(fig5)

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n  ── Summary ──")
    print(f"  {label1}: mean={t1v.mean():.2f}s  std={t1v.std():.2f}s  P(<{target}s)={100*(t1v<target).mean():.1f}%")
    print(f"  {label2}: mean={t2v.mean():.2f}s  std={t2v.std():.2f}s  P(<{target}s)={100*(t2v<target).mean():.1f}%")
    print(f"  Mean Δ: {mean_delta:+.3f}s  ({'Model2 faster' if mean_delta > 0 else 'Model1 faster'})")
    print(f"  Trend slope: {z[0]:.4f}  ({'hard cases improved more' if z[0] > 0.01 else 'uniform shift' if abs(z[0]) < 0.01 else 'easy cases improved more'})")

    # Save JSON
    results = {
        "n_valid": n_valid,
        "model1_label": label1, "model2_label": label2,
        "target_time": target,
        "correlation": {"pearson_r": pearson_r, "spearman_rho": spearman_r},
        "model1": {"mean": float(t1v.mean()), "std": float(t1v.std()), "median": float(np.median(t1v)),
                   "pass_rate": float((t1v < target).mean())},
        "model2": {"mean": float(t2v.mean()), "std": float(t2v.std()), "median": float(np.median(t2v)),
                   "pass_rate": float((t2v < target).mean())},
        "delta": {"mean": float(mean_delta), "std": float(delta.std()), "trend_slope": float(z[0])},
        "threshold_table": {"pp": pp, "pf": pf, "fp": fp, "ff": ff},
        "buckets": bucket_data,
    }
    json_path = os.path.join(out_dir, "paired_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Charts saved to: {out_dir}")
    print(f"  JSON saved to: {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    model1_path = get_checkpoint_path(log_root, args_cli.model1_run, args_cli.model1_ckpt)
    model2_path = get_checkpoint_path(log_root, args_cli.model2_run, args_cli.model2_ckpt)
    out_dir = args_cli.output_dir or os.path.join(
        os.path.dirname(model2_path), "paired_eval",
        f"{args_cli.model1_label}_vs_{args_cli.model2_label}")

    print(f"[INFO] Model 1 ({args_cli.model1_label}): {model1_path}")
    print(f"[INFO] Model 2 ({args_cli.model2_label}): {model2_path}")
    print(f"[INFO] Output: {out_dir}")

    # ── Init environment ──────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.is_train = True
    env_cfg.seed = args_cli.seed
    env_cfg.rewards = {k: 0.0 for k in [
        'gate_pass_reward_scale',
        'progress_goal_reward_scale',
        'lap_complete_reward_scale',
        'death_cost',
        'lap_incomplete_penalty_scale',
        'cmd_reg_rp_scale',
        'cmd_reg_yaw_scale',
        'crash_reward_scale',
        'crash_contact_scale',
        'cmd_smoothness_scale',
    ]}

    env = gym.make(args_cli.task, cfg=env_cfg, rewards=env_cfg.rewards)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # ── Generate fixed DR params ──────────────────────────────────────────────
    rng = random.Random(args_cli.seed)
    env_configs, param_values = sample_fixed_params(
        env.unwrapped.cfg, args_cli.num_envs, num_params=3, rng=rng)
    print(f"\n[INFO] Generated {args_cli.num_envs} fixed DR param configs (seed={args_cli.seed})")

    # ── Eval Model 1 ─────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  EVALUATING: {args_cli.model1_label}")
    print(f"{'#'*60}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(model1_path)
    policy1 = runner.get_inference_policy(device=env.unwrapped.device)
    t1 = run_eval(env, policy1, param_values, args_cli.max_steps, args_cli.model1_label)

    # ── Eval Model 2 ─────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  EVALUATING: {args_cli.model2_label}")
    print(f"{'#'*60}")
    runner.load(model2_path)
    policy2 = runner.get_inference_policy(device=env.unwrapped.device)
    t2 = run_eval(env, policy2, param_values, args_cli.max_steps, args_cli.model2_label)

    # ── Analysis ──────────────────────────────────────────────────────────────
    build_analysis(t1, t2, args_cli.model1_label, args_cli.model2_label,
                   args_cli.target_time, out_dir)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

#!/usr/bin/env python3
"""Hyperparameter sweep script for drone racing reward tuning.

Usage:
    python scripts/run/sweep.py                    # run all configs sequentially
    python scripts/run/sweep.py --config 0 2 4     # run specific configs by index
    python scripts/run/sweep.py --dry-run           # print configs without running
    python scripts/run/sweep.py --max-iterations 2000  # override iteration count

Configs are defined in SWEEP_CONFIGS below. Each entry has:
  - name: descriptive suffix for wandb run name
  - reward_overrides: dict of reward/architecture param overrides
  - ppo_overrides: dict of PPO config overrides (num_steps_per_env, gamma, etc.)
"""

import subprocess
import sys
import os
import argparse
import json
from itertools import product

# ---------- Default reward scales (Circle-V6 Sparse Baseline) ----------
REWARD_DEFAULTS = {
    "gate_pass_reward_scale": 200.0,
    "death_cost": -100.0,
    "lap_incomplete_penalty_scale": -0.05,
    "cmd_reg_rp_scale": -1.0,
    "cmd_reg_yaw_scale": -0.5,
    "crash_reward_scale": -2.0,
    "crash_contact_scale": -0.1,
}

# ---------- Default PPO config (Circle-V6 baseline) ----------
PPO_DEFAULTS = {
    "num_steps_per_env": 64,
    "gamma": 0.99,
    "lam": 0.95,
    "num_learning_epochs": 8,
    "num_mini_batches": 4,
    "entropy_coef": 0.01,
    "learning_rate": 1e-4,
}

# ---------- Environment config defaults (Circle-V6) ----------
ENV_DEFAULTS = {
    "action_latency_max": 0,               # V4: disabled (V2 had none)
    "mass_variation": 0.05,                # ±5% mass randomization (moderate)
    "motor_tau_scale_min": 0.7,            # motor time constant DR lower bound (moderate)
    "motor_tau_scale_max": 1.3,            # motor time constant DR upper bound (moderate)
    "obs_latency_prob": 0.0,              # V4: disabled
    "use_spline_reset": True,              # spline-based reset with velocity init
    "spline_vel_min": 1.0,                # V5: raised from 0.5 (real avg ~2.9 m/s)
    "spline_vel_max": 3.0,                # V5: raised from 1.5
}

# ---------- Sweep configurations ----------
# Each entry: (name, reward_overrides, ppo_overrides, env_overrides)
# env_overrides is optional (defaults to {}) for backward compat
#
# V6 sweep: 4 configs to run in parallel.
# Strategy: V6 full (all rosbag-informed changes) vs ablating the 3 change groups.
# The V6 changes live in quadcopter_strategies.py directly (TWR ±20%, yaw PID ±50%/±70%,
# obs noise 0.10/0.05, angular velocity resets). Sweep env_overrides can only control
# parameters exposed via ENV_DEFAULTS. The obs noise and DR range changes are hard-coded
# in the strategy, so ablations revert them via comments in the config name for tracking.
#
# NOTE: configs 1-3 ablate by reverting to V5 values. Since TWR/yaw PID/obs noise are
# hard-coded in CircleQuadcopterStrategy, these ablations require temporarily editing
# the strategy file. Use --dry-run to see what each config tests, then manually revert
# the relevant constants for ablation runs if needed.
#
# For the default 4-config parallel run: all 4 use the V6 strategy code as-is.
# The ablation is done via env_overrides where possible.
SWEEP_CONFIGS = [
    # ---- 2x2 factorial: (V2 base vs V4/V5 base) x (default vs tight cmd) ----
    # All 4 use V6 DR code (TWR ±20%, yaw PID ±50%/±70%, obs noise 0.10/0.05m, ang vel resets)
    #
    # V2 base: no spline reset, no mass/tau DR — proven best in real (lowest body rate,
    #          best consistency std=0.011s, best gate clearance 0.443m)
    # V4/V5 base: spline reset + mass ±5% + motor tau 0.7-1.3x + vel 1.0-3.0 m/s

    # 0: V2 base + V6 DR — minimal env changes, just better DR coverage
    ("s2r_v6_v2base", {}, {}, {
        "use_spline_reset": False,
        "mass_variation": 0.0,
        "motor_tau_scale_min": 1.0,
        "motor_tau_scale_max": 1.0,
    }),

    # 1: V2 base + V6 DR + tight cmd — V2 was smoothest; does noisier obs need more reg?
    ("s2r_v6_v2base_tight_cmd", {
        "cmd_reg_rp_scale": -1.5,
        "cmd_reg_yaw_scale": -0.8,
    }, {}, {
        "use_spline_reset": False,
        "mass_variation": 0.0,
        "motor_tau_scale_min": 1.0,
        "motor_tau_scale_max": 1.0,
    }),

    # 2: V4/V5 base + V6 DR — spline reset + mass/tau DR + rosbag DR
    ("s2r_v6_v4base", {}, {}, {}),

    # 3: V4/V5 base + V6 DR + tight cmd
    ("s2r_v6_v4base_tight_cmd", {
        "cmd_reg_rp_scale": -1.5,
        "cmd_reg_yaw_scale": -0.8,
    }, {}, {}),
]


def build_train_command(config_name, reward_overrides, ppo_overrides, env_overrides, max_iterations, num_envs):
    """Build the training command with overrides injected via env vars."""
    merged_rewards = {**REWARD_DEFAULTS, **reward_overrides}
    merged_ppo = {**PPO_DEFAULTS, **ppo_overrides}
    merged_env = {**ENV_DEFAULTS, **env_overrides}

    cmd = [
        sys.executable, "scripts/rsl_rl/train_race.py",
        "--task", "Isaac-Quadcopter-Race-v0",
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--headless",
        "--logger", "wandb",
    ]

    return cmd, merged_rewards, merged_ppo, merged_env


def generate_train_script(config_name, merged_rewards, merged_ppo, merged_env, max_iterations, num_envs):
    """Generate a standalone bash snippet that can be copy-pasted."""
    lines = [f"# Config: {config_name}"]
    lines.append("# --- Reward params ---")
    for k, v in sorted(merged_rewards.items()):
        lines.append(f"#   {k} = {v}")
    lines.append("# --- PPO params ---")
    for k, v in sorted(merged_ppo.items()):
        lines.append(f"#   {k} = {v}")
    lines.append("# --- Env params ---")
    for k, v in sorted(merged_env.items()):
        lines.append(f"#   {k} = {v}")
    lines.append(f"REWARD_OVERRIDES='{json.dumps(merged_rewards)}' \\")
    lines.append(f"PPO_OVERRIDES='{json.dumps(merged_ppo)}' \\")
    lines.append(f"ENV_OVERRIDES='{json.dumps(merged_env)}' \\")
    lines.append(f"  python scripts/rsl_rl/train_race.py \\")
    lines.append(f"    --task Isaac-Quadcopter-Race-v0 \\")
    lines.append(f"    --num_envs {num_envs} \\")
    lines.append(f"    --max_iterations {max_iterations} \\")
    lines.append(f"    --headless --logger wandb")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for drone racing")
    parser.add_argument("--config", type=int, nargs="*", default=None,
                        help="Config indices to run (default: all)")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--grid", action="store_true",
                        help="Generate grid sweep from predefined axes instead of manual configs")
    args = parser.parse_args()

    if args.grid:
        # Grid search over PPO + reward params
        grid_axes = {
            "cmd_reg_rp_scale": [-0.5, -1.0, -1.5],
            "lap_incomplete_penalty_scale": [-0.01, -0.05, -0.1],
            "gate_pass_reward_scale": [150.0, 200.0, 300.0],
        }
        # Separate PPO vs reward keys
        ppo_keys = {"num_steps_per_env", "gamma", "lam", "num_learning_epochs",
                     "num_mini_batches", "entropy_coef"}
        env_keys = set(ENV_DEFAULTS.keys())
        configs = []
        keys = list(grid_axes.keys())
        for vals in product(*grid_axes.values()):
            all_overrides = dict(zip(keys, vals))
            reward_ov = {k: v for k, v in all_overrides.items()
                         if k not in ppo_keys and k not in env_keys and v != REWARD_DEFAULTS.get(k)}
            ppo_ov = {k: v for k, v in all_overrides.items()
                      if k in ppo_keys and v != PPO_DEFAULTS.get(k)}
            env_ov = {k: v for k, v in all_overrides.items()
                      if k in env_keys and v != ENV_DEFAULTS.get(k)}
            name = "_".join(f"{k.split('_')[0]}{v}" for k, v in {**reward_ov, **ppo_ov, **env_ov}.items()) if (reward_ov or ppo_ov or env_ov) else "baseline"
            configs.append((name, reward_ov, ppo_ov, env_ov))
    else:
        configs = SWEEP_CONFIGS

    # Normalize configs to 4-tuples (backward compat with old 3-tuple format)
    normalized = []
    for c in configs:
        if len(c) == 3:
            normalized.append((c[0], c[1], c[2], {}))
        else:
            normalized.append(c)
    configs = normalized

    indices = args.config if args.config is not None else list(range(len(configs)))

    print(f"{'='*60}")
    print(f"  SWEEP: {len(indices)} configuration(s)")
    print(f"  Iterations: {args.max_iterations} | Envs: {args.num_envs}")
    print(f"{'='*60}\n")

    for i in indices:
        if i >= len(configs):
            print(f"[WARN] Config index {i} out of range (max {len(configs)-1}), skipping")
            continue

        name, reward_overrides, ppo_overrides, env_overrides = configs[i]
        merged_rewards = {**REWARD_DEFAULTS, **reward_overrides}
        merged_ppo = {**PPO_DEFAULTS, **ppo_overrides}
        merged_env = {**ENV_DEFAULTS, **env_overrides}
        config_name = f"sweep_{i}_{name}"

        print(f"--- Config {i}: {config_name} ---")
        print("  Reward params:")
        for k, v in sorted(merged_rewards.items()):
            marker = " *" if k in reward_overrides else ""
            print(f"    {k}: {v}{marker}")
        print("  PPO params:")
        for k, v in sorted(merged_ppo.items()):
            marker = " *" if k in ppo_overrides else ""
            print(f"    {k}: {v}{marker}")
        print("  Env params:")
        for k, v in sorted(merged_env.items()):
            marker = " *" if k in env_overrides else ""
            print(f"    {k}: {v}{marker}")
        print()

        if args.dry_run:
            print(generate_train_script(config_name, merged_rewards, merged_ppo, merged_env, args.max_iterations, args.num_envs))
            print()
            continue

        run_env = os.environ.copy()
        run_env["REWARD_OVERRIDES"] = json.dumps(merged_rewards)
        run_env["PPO_OVERRIDES"] = json.dumps(merged_ppo)
        run_env["ENV_OVERRIDES"] = json.dumps(merged_env)
        run_env["PYTHONPATH"] = os.path.abspath(".") + ":" + run_env.get("PYTHONPATH", "")

        cmd, _, _, _ = build_train_command(config_name, reward_overrides, ppo_overrides,
                                           env_overrides, args.max_iterations, args.num_envs)
        print(f"[RUN] {' '.join(cmd)}")
        print(f"[RUN] REWARD/PPO/ENV_OVERRIDES set in environment\n")

        result = subprocess.run(cmd, env=run_env)
        if result.returncode != 0:
            print(f"[WARN] Config {config_name} exited with code {result.returncode}")
        print(f"\n{'='*60}\n")

    print("Sweep complete.")


if __name__ == "__main__":
    main()
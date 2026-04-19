# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-evaluation script: tests policy with default params (1 env) and DR (100 envs).

Usage:
    python scripts/rsl_rl/eval_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --load_run [YYYY-MM-DD_XX-XX-XX] \
        --checkpoint best_model.pt \
        --headless

    # With video recording:
    python scripts/rsl_rl/eval_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --load_run [YYYY-MM-DD_XX-XX-XX] \
        --checkpoint best_model.pt \
        --headless \
        --video --video_length 1600
"""

"""Launch Isaac Sim Simulator first."""

import sys
import os
import time
import json
local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RL agent with default and DR params.")
parser.add_argument("--num_dr_envs", type=int, default=100, help="Number of DR envs for stability testing.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max timesteps per evaluation.")
parser.add_argument("--num_envs", type=int, default=None, help="Unused, kept for compatibility.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record video for Phase 1.")
parser.add_argument("--video_length", type=int, default=1600, help="Video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

# Import extensions to set up environment tasks
import src.isaac_quad_sim2real.tasks   # noqa: F401


def run_eval(env, policy, max_steps, label="Eval"):
    """Run evaluation and collect lap/gate statistics."""
    obs = env.get_observations()
    if hasattr(obs, "get"):
        obs = obs["policy"]

    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    num_gates = unwrapped._waypoints.shape[0]
    max_laps = unwrapped.cfg.max_n_laps

    # Track per-env stats
    finished = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
    final_gates = torch.zeros(num_envs, dtype=torch.int, device=unwrapped.device)
    final_time = torch.zeros(num_envs, dtype=torch.float, device=unwrapped.device)
    completed_laps = torch.zeros(num_envs, dtype=torch.int, device=unwrapped.device)

    t_start = time.time()
    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]

            # Track completed envs
            newly_done = dones.bool() & ~finished
            if newly_done.any():
                done_ids = torch.where(newly_done)[0]
                elapsed = unwrapped.episode_length_buf[done_ids].float() * unwrapped.cfg.sim.dt * unwrapped.cfg.decimation
                gates = unwrapped._n_gates_passed[done_ids]
                final_gates[done_ids] = gates
                final_time[done_ids] = elapsed
                completed_laps[done_ids] = gates // num_gates
                finished[done_ids] = True

            # Print progress every 100 steps
            if step % 100 == 0:
                n_finished = finished.sum().item()
                wall_time = time.time() - t_start
                print(f"  [{label}] Step {step}/{max_steps} | {n_finished}/{num_envs} envs done | {wall_time:.1f}s elapsed")

            # Stop if all envs done
            if finished.all():
                print(f"  [{label}] All envs finished at step {step}!")
                break

    # For envs that didn't finish, record their current state
    not_finished = ~finished
    if not_finished.any():
        nf_ids = torch.where(not_finished)[0]
        elapsed = unwrapped.episode_length_buf[nf_ids].float() * unwrapped.cfg.sim.dt * unwrapped.cfg.decimation
        final_gates[nf_ids] = unwrapped._n_gates_passed[nf_ids]
        final_time[nf_ids] = elapsed
        completed_laps[nf_ids] = final_gates[nf_ids] // num_gates

    wall_time = time.time() - t_start
    print(f"  [{label}] Evaluation took {wall_time:.1f}s wall time")

    return {
        "gates_passed": final_gates.cpu(),
        "time": final_time.cpu(),
        "laps": completed_laps.cpu(),
        "finished": finished.cpu(),
    }


def print_results(results, label):
    """Print evaluation results."""
    gates = results["gates_passed"].float()
    times = results["time"]
    laps = results["laps"].float()
    finished = results["finished"]
    n_envs = len(gates)

    print(f"\n{'='*60}")
    print(f"  {label} RESULTS ({n_envs} envs)")
    print(f"{'='*60}")
    print(f"  Gates passed:  mean={gates.mean():.1f}  min={gates.min():.0f}  max={gates.max():.0f}")
    print(f"  Laps completed: mean={laps.mean():.2f}  min={laps.min():.0f}  max={laps.max():.0f}")
    print(f"  Time:           mean={times.mean():.2f}s  min={times.min():.2f}s  max={times.max():.2f}s")

    # Completion rate (3+ laps)
    completed_3 = (laps >= 3).float().mean() * 100
    print(f"  3-lap completion rate: {completed_3:.1f}%")

    # For completed runs, show lap time
    completed_mask = laps >= 3
    if completed_mask.any():
        completed_times = times[completed_mask]
        print(f"  Completed 3-lap time: mean={completed_times.mean():.2f}s  best={completed_times.min():.2f}s")

    print(f"  Episodes finished: {finished.sum()}/{n_envs}")
    print(f"{'='*60}\n")


def main():
    """Run two-phase evaluation: default params + DR."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading model from: {resume_path}")

    # ===============================================
    #  PHASE 1: Default parameters, 1 env (+ video)
    # ===============================================
    print("\n" + "="*60)
    print("  PHASE 1: Default Parameters (1 env)")
    print("  Initializing environment... (this takes 1-2 min)")
    print("="*60)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    if "ENV_OVERRIDES" in os.environ:
        try:
            env_overrides = json.loads(os.environ["ENV_OVERRIDES"])
            for k, v in env_overrides.items():
                setattr(env_cfg, k, v)
            print(f"[INFO] Applied ENV_OVERRIDES (Phase 1): {env_overrides}")
        except Exception as e:
            print(f"[Warning] Failed to parse ENV_OVERRIDES for Phase 1: {e}")
    env_cfg.is_train = False
    env_cfg.max_motor_noise_std = 0.0
    env_cfg.seed = args_cli.seed
    env_cfg.rewards = {
        'gate_pass_reward_scale': 0.0,
        'progress_goal_reward_scale': 0.0,
        'lap_complete_reward_scale': 0.0,
        'death_cost': 0.0,
        'lap_incomplete_penalty_scale': 0.0,
        'cmd_reg_rp_scale': 0.0,
        'cmd_reg_yaw_scale': 0.0,
        'crash_reward_scale': 0.0,
        'crash_contact_scale': 0.0,
        'cmd_smoothness_scale': 0.0,
    }

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None, rewards=env_cfg.rewards)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for video recording if requested
    if args_cli.video:
        log_dir = os.path.dirname(resume_path)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    print("[INFO] Environment ready! Loading model...")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    max_steps_phase1 = args_cli.video_length if args_cli.video else args_cli.max_steps
    results_default = run_eval(env, policy, max_steps_phase1, "DEFAULT")
    print_results(results_default, "DEFAULT PARAMS")
    env.close()

    # ===============================================
    #  PHASE 2: DR with multiple envs
    # ===============================================
    if args_cli.num_dr_envs <= 0:
        print("\n[INFO] Skipping Phase 2 DR evaluation because --num_dr_envs <= 0")
        simulation_app.close()
        return

    print("\n" + "="*60)
    print(f"  PHASE 2: Domain Randomization ({args_cli.num_dr_envs} envs)")
    print("  Initializing environment... (this takes 1-2 min)")
    print("="*60)

    env_cfg2 = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_dr_envs,
        use_fabric=not args_cli.disable_fabric
    )
    if "ENV_OVERRIDES" in os.environ:
        try:
            env_overrides = json.loads(os.environ["ENV_OVERRIDES"])
            for k, v in env_overrides.items():
                setattr(env_cfg2, k, v)
            print(f"[INFO] Applied ENV_OVERRIDES (Phase 2): {env_overrides}")
        except Exception as e:
            print(f"[Warning] Failed to parse ENV_OVERRIDES for Phase 2: {e}")
    env_cfg2.is_train = True  # Enable DR by using train mode
    env_cfg2.seed = args_cli.seed
    # Provide dummy rewards since is_train=True requires them
    env_cfg2.rewards = {
        'gate_pass_reward_scale': 0.0,
        'progress_goal_reward_scale': 0.0,
        'lap_complete_reward_scale': 0.0,
        'death_cost': 0.0,
        'lap_incomplete_penalty_scale': 0.0,
        'cmd_reg_rp_scale': 0.0,
        'cmd_reg_yaw_scale': 0.0,
        'crash_reward_scale': 0.0,
        'crash_contact_scale': 0.0,
        'cmd_smoothness_scale': 0.0,
    }

    env2 = gym.make(args_cli.task, cfg=env_cfg2, rewards=env_cfg2.rewards)
    if isinstance(env2.unwrapped, DirectMARLEnv):
        env2 = multi_agent_to_single_agent(env2)
    env2 = RslRlVecEnvWrapper(env2)

    print("[INFO] Environment ready! Loading model...")
    ppo_runner2 = OnPolicyRunner(env2, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner2.load(resume_path)
    policy2 = ppo_runner2.get_inference_policy(device=env2.unwrapped.device)

    results_dr = run_eval(env2, policy2, args_cli.max_steps, "DR")
    print_results(results_dr, "DOMAIN RANDOMIZATION")
    env2.close()

    # ===============================================
    #  SUMMARY
    # ===============================================
    print("\n" + "#"*60)
    print("  EVALUATION SUMMARY")
    print("#"*60)
    def_laps = results_default["laps"].float().mean()
    dr_laps = results_dr["laps"].float().mean()
    dr_completion = (results_dr["laps"] >= 3).float().mean() * 100
    print(f"  Default:  {def_laps:.1f} laps avg")
    print(f"  DR:       {dr_laps:.1f} laps avg ({dr_completion:.0f}% completed 3 laps)")
    if dr_completion >= 80:
        print("  ✅ Policy looks robust for TA evaluation!")
    elif dr_completion >= 50:
        print("  ⚠️  Policy is somewhat robust, consider more training.")
    else:
        print("  ❌ Policy is fragile under DR. Needs more training/tuning.")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

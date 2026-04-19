# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import sys
import os
import json

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")
else:
    print(f"[WARNING] Local rsl_rl not found at: {local_rsl_path}")

from rsl_rl.utils import wandb_fix
import argparse
from isaaclab.app import AppLauncher
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import src.isaac_quad_sim2real.tasks   # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # TODO ----- START ----- Define rewards scales
    # Fixed sim2real baseline: V3 sparse reward core + light action smoothness.
    # Optional R1 terms (progress + lap_complete) default to zero so the baseline
    # remains unchanged unless explicitly enabled.
    # gate_pass must dominate continuous negative terms to prevent policy collapse.
    # Values can be overridden via environment variables (e.g. REW_CMD_REG_RP=-2.0).
    gate_pass_reward_scale = float(os.environ.get('REW_GATE_PASS', 200.0))
    progress_goal_reward_scale = float(os.environ.get('REW_PROGRESS_GOAL', 0.0))
    lap_complete_reward_scale = float(os.environ.get('REW_LAP_COMPLETE', 0.0))
    death_cost = float(os.environ.get('REW_DEATH_COST', -100.0))
    lap_incomplete_penalty_scale = float(os.environ.get('REW_LAP_INCOMPLETE', -0.05))
    cmd_reg_rp_scale = float(os.environ.get('REW_CMD_REG_RP', -1.0))
    cmd_reg_yaw_scale = float(os.environ.get('REW_CMD_REG_YAW', -0.5))
    crash_reward_scale = float(os.environ.get('REW_CRASH', -2.0))
    crash_contact_scale = float(os.environ.get('REW_CRASH_CONTACT', -0.1))
    cmd_smoothness_scale = float(os.environ.get('REW_CMD_SMOOTHNESS', -0.1))

    rewards = {
        'gate_pass_reward_scale': gate_pass_reward_scale,
        'progress_goal_reward_scale': progress_goal_reward_scale,
        'lap_complete_reward_scale': lap_complete_reward_scale,
        'death_cost': death_cost,
        'lap_incomplete_penalty_scale': lap_incomplete_penalty_scale,
        'cmd_reg_rp_scale': cmd_reg_rp_scale,
        'cmd_reg_yaw_scale': cmd_reg_yaw_scale,
        'crash_reward_scale': crash_reward_scale,
        'crash_contact_scale': crash_contact_scale,
        'cmd_smoothness_scale': cmd_smoothness_scale,
    }
    # TODO ----- END -----

    if "REWARD_OVERRIDES" in os.environ:
        try:
            reward_overrides = json.loads(os.environ["REWARD_OVERRIDES"])
            rewards.update(reward_overrides)
            print(f"[INFO] Applied REWARD_OVERRIDES: {reward_overrides}")
        except Exception as e:
            print(f"[Warning] Failed to parse REWARD_OVERRIDES: {e}")

    env_cfg.is_train = True
    env_cfg.rewards = rewards

    # Training-only performance tweaks (no effect on physics/observations):
    #   - debug_vis off: skip per-step goal-marker update callback
    #   - render_interval x20: we're headless w/o video, no reason to render at 50 Hz
    #   - contact history=1: strategies only read current-step net_forces_w
    # play_race.py / eval_race.py don't go through this block, so their UX is
    # unchanged (markers + normal render rate still available there).
    if not args_cli.video:
        env_cfg.debug_vis = False
        env_cfg.sim.render_interval = env_cfg.decimation * 20
    env_cfg.contact_sensor.history_length = 1

    if "ENV_OVERRIDES" in os.environ:
        try:
            env_overrides = json.loads(os.environ["ENV_OVERRIDES"])
            for k, v in env_overrides.items():
                setattr(env_cfg, k, v)
            print(f"[INFO] Applied ENV_OVERRIDES: {env_overrides}")
        except Exception as e:
            print(f"[Warning] Failed to parse ENV_OVERRIDES: {e}")

    if "PPO_OVERRIDES" in os.environ:
        try:
            ppo_overrides = json.loads(os.environ["PPO_OVERRIDES"])
            for k, v in ppo_overrides.items():
                if hasattr(agent_cfg.algorithm, k):
                    setattr(agent_cfg.algorithm, k, v)
                elif hasattr(agent_cfg.policy, k):
                    setattr(agent_cfg.policy, k, v)
                elif hasattr(agent_cfg, k):
                    setattr(agent_cfg, k, v)
                else:
                    print(f"[Warning] PPO override key '{k}' not found in agent_cfg")
            print(f"[INFO] Applied PPO_OVERRIDES: {ppo_overrides}")
        except Exception as e:
            print(f"[Warning] Failed to parse PPO_OVERRIDES: {e}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None, rewards=rewards)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

import sys
import os
local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)

import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Mimic TA Evaluation.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos.")
parser.add_argument("--video_length", type=int, default=2000, help="Max steps to record (auto-stops at 3 laps).")
parser.add_argument("--video_name", type=str, default="ta_eval", help="Folder name.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Race-v0")
parser.add_argument("--seed", type=int, default=42, help="Seed to pick the 3 random parameters.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from src.isaac_quad_sim2real.tasks.race.config.crazyflie.quadcopter_strategies import DefaultQuadcopterStrategy

def main():
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", "quadcopter_direct", args_cli.experiment_name))
    resume_path = os.path.join(log_root_path, getattr(args_cli, "checkpoint", "best_model.pt"))
    log_dir = log_root_path

    # ── Pre-compute randomized parameters ONCE (mimics TA: "sampled only once") ──
    param_names = ['TWR', 'k_aero_xy', 'k_aero_z', 'kp_omega_rp', 'kp_omega_y',
                   'kd_omega_rp', 'kd_omega_y', 'ki_omega_rp', 'ki_omega_y']

    # Use a temporary manual seed to select params, then restore RNG state
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(args_cli.seed)
    selected_indices = torch.randperm(len(param_names))[:3].tolist()
    # Sample the actual values while still under the manual seed
    sampled_rand_vals = [torch.rand(1).item() for _ in range(3)]
    torch.random.set_rng_state(rng_state)  # restore RNG so env seed works normally

    # Monkey-patch _set_default_dynamics to apply the pre-computed DR
    original_set_default = DefaultQuadcopterStrategy._set_default_dynamics

    def _randomized_eval_dynamics(self, env_ids):
        original_set_default(self, env_ids)
        c = self.cfg
        param_ranges = {
            'TWR': (c.thrust_to_weight * 0.95, c.thrust_to_weight * 1.05),
            'k_aero_xy': (c.k_aero_xy * 0.5, c.k_aero_xy * 2.0),
            'k_aero_z': (c.k_aero_z * 0.5, c.k_aero_z * 2.0),
            'kp_omega_rp': (c.kp_omega_rp * 0.85, c.kp_omega_rp * 1.15),
            'kp_omega_y': (c.kp_omega_y * 0.85, c.kp_omega_y * 1.15),
            'kd_omega_rp': (c.kd_omega_rp * 0.7, c.kd_omega_rp * 1.3),
            'kd_omega_y': (c.kd_omega_y * 0.7, c.kd_omega_y * 1.3),
            'ki_omega_rp': (c.ki_omega_rp * 0.85, c.ki_omega_rp * 1.15),
            'ki_omega_y': (c.ki_omega_y * 0.85, c.ki_omega_y * 1.15),
        }

        # Apply the SAME pre-computed parameters on every call (reset-safe)
        for i, idx in enumerate(selected_indices):
            p = param_names[idx]
            min_val, max_val = param_ranges[p]
            val = min_val + sampled_rand_vals[i] * (max_val - min_val)

            if p == 'TWR':
                self.env._thrust_to_weight[env_ids] = val
            elif p == 'k_aero_xy':
                self.env._K_aero[env_ids, 0] = val
                self.env._K_aero[env_ids, 1] = val
            elif p == 'k_aero_z':
                self.env._K_aero[env_ids, 2] = val
            elif p == 'kp_omega_rp':
                self.env._kp_omega[env_ids, 0] = val
                self.env._kp_omega[env_ids, 1] = val
            elif p == 'kp_omega_y':
                self.env._kp_omega[env_ids, 2] = val
            elif p == 'ki_omega_rp':
                self.env._ki_omega[env_ids, 0] = val
                self.env._ki_omega[env_ids, 1] = val
            elif p == 'ki_omega_y':
                self.env._ki_omega[env_ids, 2] = val
            elif p == 'kd_omega_rp':
                self.env._kd_omega[env_ids, 0] = val
                self.env._kd_omega[env_ids, 1] = val
            elif p == 'kd_omega_y':
                self.env._kd_omega[env_ids, 2] = val

    DefaultQuadcopterStrategy._set_default_dynamics = _randomized_eval_dynamics

    env_cfg.is_train = False
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array", rewards=env_cfg.rewards)
    if isinstance(env.unwrapped, DirectMARLEnv): env = multi_agent_to_single_agent(env)
    
    vid_name = f"{args_cli.video_name}_seed_{args_cli.seed}"
    video_kwargs = {"video_folder": os.path.join(log_dir, "videos", vid_name), "step_trigger": lambda step: step == 0, "video_length": args_cli.video_length, "disable_logger": True}
    print(f"[INFO] Recording video to {video_kwargs['video_folder']}")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.reset()
    if hasattr(obs, "get"): obs = obs["policy"]

    unwrapped = env.unwrapped

    # Print the selected parameters AFTER env creation (thrust_to_weight is set in __init__)
    c = unwrapped.cfg
    param_ranges_display = {
        'TWR': (c.thrust_to_weight * 0.95, c.thrust_to_weight * 1.05),
        'k_aero_xy': (c.k_aero_xy * 0.5, c.k_aero_xy * 2.0),
        'k_aero_z': (c.k_aero_z * 0.5, c.k_aero_z * 2.0),
        'kp_omega_rp': (c.kp_omega_rp * 0.85, c.kp_omega_rp * 1.15),
        'kp_omega_y': (c.kp_omega_y * 0.85, c.kp_omega_y * 1.15),
        'kd_omega_rp': (c.kd_omega_rp * 0.7, c.kd_omega_rp * 1.3),
        'kd_omega_y': (c.kd_omega_y * 0.7, c.kd_omega_y * 1.3),
        'ki_omega_rp': (c.ki_omega_rp * 0.85, c.ki_omega_rp * 1.15),
        'ki_omega_y': (c.ki_omega_y * 0.85, c.ki_omega_y * 1.15),
    }
    print("\n" + "=" * 60)
    print(f" TA EVALUATION MIMIC (Seed: {args_cli.seed})")
    print(" The following 3 parameters are randomized (held constant across resets):")
    for i, idx in enumerate(selected_indices):
        p = param_names[idx]
        min_val, max_val = param_ranges_display[p]
        val = min_val + sampled_rand_vals[i] * (max_val - min_val)
        print(f"  - {p:<15}: {val:.6f}  (range: [{min_val:.6f}, {max_val:.6f}])")
    print("=" * 60 + "\n")
    num_gates = unwrapped._waypoints.shape[0]
    three_lap_time = None
    timestep = 0

    while simulation_app.is_running():
        with torch.no_grad():
            # Pre-step: capture race state before env.step potentially resets
            pre_gates = unwrapped._n_gates_passed[0].item()
            pre_laps = pre_gates // num_gates
            pre_time = unwrapped.episode_length_buf[0].item() * unwrapped.cfg.sim.dt * unwrapped.cfg.decimation

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(obs, "get"): obs = obs["policy"]

            # Post-step race stats
            post_gates = unwrapped._n_gates_passed[0].item()
            post_laps = post_gates // num_gates
            elapsed = unwrapped.episode_length_buf[0].item() * unwrapped.cfg.sim.dt * unwrapped.cfg.decimation

            # Periodic status
            if timestep % 50 == 0:
                print(f"[t={elapsed:6.2f}s] Gates: {post_gates:3d} | Laps: {post_laps}/{unwrapped.cfg.max_n_laps}")

            # Detect 3-lap completion (use pre-step values since env may have reset)
            if pre_laps >= 3 and three_lap_time is None:
                three_lap_time = pre_time
                print(f"\n{'='*60}")
                print(f"  ✅ 3-LAP RACE COMPLETED!")
                print(f"  Time:  {three_lap_time:.2f}s")
                print(f"  Gates: {pre_gates}")
                print(f"{'='*60}\n")

            # Also catch via dones (env auto-terminates after max_n_laps)
            if dones[0] and three_lap_time is None:
                print(f"\n{'='*60}")
                print(f"  RACE ENDED (done signal)")
                print(f"  Time:  {pre_time:.2f}s")
                print(f"  Gates: {pre_gates}")
                print(f"  Laps:  {pre_laps}/{unwrapped.cfg.max_n_laps}")
                print(f"{'='*60}\n")

        timestep += 1
        # Auto-stop after 3-lap completion
        if three_lap_time is not None:
            break
        if timestep >= args_cli.video_length:
            print(f"\n[WARN] Reached max video length ({args_cli.video_length} steps) without completing 3 laps.")
            break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

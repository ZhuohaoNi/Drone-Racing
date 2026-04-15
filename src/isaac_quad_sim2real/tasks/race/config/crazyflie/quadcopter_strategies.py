# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional, Tuple
from scipy.interpolate import CubicSpline

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Lap time tracking
        self._lap_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lap_times = []  # buffer of completed lap times (seconds)
        self._best_lap_time = float('inf')

        # Success rate tracking: rolling window of episode outcomes
        self._episode_successes = deque(maxlen=100)  # 1 = completed 3 laps, 0 = did not

        # Gate 3 powerloop: 2-phase guide (apex → pre-entry)
        # Apex above double gate, slightly toward Gate 2 (x=-0.625)
        self._powerloop_apex = torch.tensor([0.0, -0.3, 1.6], device=self.device)
        # Pre-entry: on +y entry side, slightly offset toward Gate 4 direction
        # Not at gate center — gives drone room to set up exit angle
        self._gate3_pre_entry = torch.tensor([0.0, 1.0, 1.2], device=self.device)
        # Phase 2 offset target: shifted from gate center toward Gate 4
        self._gate3_offset_center = torch.tensor([0.425, 0.0, 0.75], device=self.device)
        self._powerloop_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Domain randomization: only during training so TA's fixed params are preserved during eval
        if self.cfg.is_train:
            self._randomize_dynamics(torch.arange(self.num_envs, device=self.device))
        else:
            # Set default (nominal) parameter values for evaluation
            self._set_default_dynamics(torch.arange(self.num_envs, device=self.device))

    def _randomize_dynamics(self, env_ids: torch.Tensor):
        """Randomize physical parameters for domain randomization.
        Ranges match the TA evaluation ranges from the project description."""
        n = len(env_ids)
        cfg = self.cfg

        # Thrust-to-weight ratio: ±5%
        twr_min = cfg.thrust_to_weight * 0.95
        twr_max = cfg.thrust_to_weight * 1.05
        self.env._thrust_to_weight[env_ids] = torch.empty(n, device=self.device).uniform_(twr_min, twr_max)

        # Aerodynamic drag: 0.5x - 2.0x
        k_aero_xy_min = cfg.k_aero_xy * 0.5
        k_aero_xy_max = cfg.k_aero_xy * 2.0
        k_aero_z_min = cfg.k_aero_z * 0.5
        k_aero_z_max = cfg.k_aero_z * 2.0
        self.env._K_aero[env_ids, 0] = torch.empty(n, device=self.device).uniform_(k_aero_xy_min, k_aero_xy_max)
        self.env._K_aero[env_ids, 1] = torch.empty(n, device=self.device).uniform_(k_aero_xy_min, k_aero_xy_max)
        self.env._K_aero[env_ids, 2] = torch.empty(n, device=self.device).uniform_(k_aero_z_min, k_aero_z_max)

        # PID gains - Roll/Pitch: kp/ki ±15%, kd ±30%
        self.env._kp_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(cfg.kp_omega_rp * 0.85, cfg.kp_omega_rp * 1.15)
        self.env._kp_omega[env_ids, 1] = self.env._kp_omega[env_ids, 0]  # same for roll and pitch
        self.env._ki_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(cfg.ki_omega_rp * 0.85, cfg.ki_omega_rp * 1.15)
        self.env._ki_omega[env_ids, 1] = self.env._ki_omega[env_ids, 0]
        self.env._kd_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(cfg.kd_omega_rp * 0.7, cfg.kd_omega_rp * 1.3)
        self.env._kd_omega[env_ids, 1] = self.env._kd_omega[env_ids, 0]

        # PID gains - Yaw: kp/ki ±15%, kd ±30%
        self.env._kp_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(cfg.kp_omega_y * 0.85, cfg.kp_omega_y * 1.15)
        self.env._ki_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(cfg.ki_omega_y * 0.85, cfg.ki_omega_y * 1.15)
        self.env._kd_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(cfg.kd_omega_y * 0.7, cfg.kd_omega_y * 1.3)

        # Motor time constants (fixed)
        self.env._tau_m[env_ids] = cfg.tau_m

    def _set_default_dynamics(self, env_ids: torch.Tensor):
        """Set physical parameters to nominal default values (for evaluation)."""
        n = len(env_ids)
        cfg = self.cfg

        self.env._thrust_to_weight[env_ids] = cfg.thrust_to_weight
        self.env._K_aero[env_ids, 0] = cfg.k_aero_xy
        self.env._K_aero[env_ids, 1] = cfg.k_aero_xy
        self.env._K_aero[env_ids, 2] = cfg.k_aero_z
        self.env._kp_omega[env_ids, 0] = cfg.kp_omega_rp
        self.env._kp_omega[env_ids, 1] = cfg.kp_omega_rp
        self.env._kp_omega[env_ids, 2] = cfg.kp_omega_y
        self.env._ki_omega[env_ids, 0] = cfg.ki_omega_rp
        self.env._ki_omega[env_ids, 1] = cfg.ki_omega_rp
        self.env._ki_omega[env_ids, 2] = cfg.ki_omega_y
        self.env._kd_omega[env_ids, 0] = cfg.kd_omega_rp
        self.env._kd_omega[env_ids, 1] = cfg.kd_omega_rp
        self.env._kd_omega[env_ids, 2] = cfg.kd_omega_y
        self.env._tau_m[env_ids] = cfg.tau_m
        self.env._robot_weight[env_ids] = self.env._nominal_robot_weight

    def get_rewards(self) -> torch.Tensor:
        """Compute rewards: progress, velocity toward gate, gate-pass bonus, crash, orientation, smoothness."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        num_gates = self.env._waypoints.shape[0]
        gate_side = self.cfg.gate_model.gate_side

        # --- Gate passing detection ---
        curr_x = self.env._pose_drone_wrt_gate[:, 0]
        prev_x = self.env._prev_x_drone_wrt_gate
        crossed_plane = (prev_x > 0) & (curr_x <= 0)
        gate_y = self.env._pose_drone_wrt_gate[:, 1]
        gate_z = self.env._pose_drone_wrt_gate[:, 2]
        within_bounds = (gate_y.abs() < gate_side / 2.0) & (gate_z.abs() < gate_side / 2.0)

        gate_passed = crossed_plane & within_bounds
        ids_gate_passed = torch.where(gate_passed)[0]

        # --- Wrong-side crossing detection with cooldown ---
        wrong_side_crossed = (prev_x < 0) & (curr_x >= 0) & within_bounds
        # Cooldown: skip first 5 steps after gate pass (0.1s at 50Hz — enough to clear gate)
        if not hasattr(self, '_steps_since_gate_pass'):
            self._steps_since_gate_pass = torch.full((self.num_envs,), 999, dtype=torch.long, device=self.device)
        self._steps_since_gate_pass += 1
        cooldown_ok = (self._steps_since_gate_pass > 5)
        wrong_side_valid = wrong_side_crossed & cooldown_ok
        wrong_side_ids = torch.where(wrong_side_valid)[0]
        if len(wrong_side_ids) > 0:
            self.env._crashed[wrong_side_ids] = 200
        # Track wrong-direction count for logging
        if not hasattr(self, '_wrong_side_count'):
            self._wrong_side_count = 0
        self._wrong_side_count += len(wrong_side_ids)

        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % num_gates
        self.env._n_gates_passed[ids_gate_passed] += 1
        self.env._desired_pos_w[ids_gate_passed] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
        self.env._prev_x_drone_wrt_gate = curr_x.clone()
        # Fix: recompute prev_x for gate-passing envs in NEW target gate frame
        if len(ids_gate_passed) > 0:
            new_pose, _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_pos_w[ids_gate_passed]
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pose[:, 0]
        gate_pass = gate_passed.float()
        # Reset cooldown timer for envs that just passed a gate
        if len(ids_gate_passed) > 0:
            self._steps_since_gate_pass[ids_gate_passed] = 0

        # --- Gate 2 pre-powerloop guide ---
        # When targeting Gate 2, point toward the powerloop apex instead of Gate 2 center.
        # The drone will naturally clip through Gate 2's opening (gate_pass checks y/z in gate frame,
        # not desired_pos_w) while already climbing toward the apex. This prevents the
        # "pass Gate 2 horizontally then circle back to Gate 3" pattern.
        targeting_gate2 = (self.env._idx_wp == 2)
        if targeting_gate2.any():
            g2_ids = torch.where(targeting_gate2)[0]
            self.env._desired_pos_w[g2_ids] = self._powerloop_apex.unsqueeze(0)

        # --- Gate 3 powerloop: 2-phase guide ---
        # Phase 0: apex [0, -0.3, 1.6] — climb & loop
        # Phase 1: offset_center [0.425, 0, 0.75] — descend directly into Gate 3
        # pre_entry removed: drone flies around side of gate, still valid pass, 17.34s vs V23 17.50s
        targeting_gate3 = (self.env._idx_wp == 3)
        if targeting_gate3.any():
            g3_ids = torch.where(targeting_gate3)[0]
            drone_z = self.env._robot.data.root_link_pos_w[g3_ids, 2]
            drone_pos = self.env._robot.data.root_link_pos_w[g3_ids]

            # Phase 0 → 1: apex reached when z > 1.3m OR dist to apex < 0.8m
            phase0 = (self._powerloop_phase[g3_ids] == 0)
            dist_to_apex = torch.linalg.norm(self._powerloop_apex - drone_pos[phase0], dim=1) if phase0.any() else torch.tensor([], device=self.device)
            promote_0to1 = phase0.clone()
            if phase0.any():
                promote_0to1[phase0] = (drone_z[phase0] > 1.3) | (dist_to_apex < 0.8)
            if promote_0to1.any():
                self._powerloop_phase[g3_ids[promote_0to1]] = 1

            # Apply targets: phase 0 → apex, phase 1 → offset_center
            still_phase0 = (self._powerloop_phase[g3_ids] == 0)
            if still_phase0.any():
                self.env._desired_pos_w[g3_ids[still_phase0]] = self._powerloop_apex.unsqueeze(0)
            still_phase1 = (self._powerloop_phase[g3_ids] == 1)
            if still_phase1.any():
                self.env._desired_pos_w[g3_ids[still_phase1]] = self._gate3_offset_center.unsqueeze(0)

        # Reset powerloop phase on gate pass
        if len(ids_gate_passed) > 0:
            self._powerloop_phase[ids_gate_passed] = 0




        # --- Lap time tracking ---
        if len(ids_gate_passed) > 0:
            dt = self.cfg.sim.dt * self.cfg.decimation
            # Check which of the gate-passing envs just completed a full lap
            gates_passed_count = self.env._n_gates_passed[ids_gate_passed]
            lap_complete_mask = (gates_passed_count > 0) & (gates_passed_count % num_gates == 0)
            lap_complete_ids = ids_gate_passed[lap_complete_mask]
            if len(lap_complete_ids) > 0:
                current_step = self.env.episode_length_buf[lap_complete_ids]
                lap_steps = current_step - self._lap_start_step[lap_complete_ids]
                lap_seconds = lap_steps.float() * dt
                for t in lap_seconds.cpu().tolist():
                    if t > 0:  # filter out invalid negative times from initial randomized episodes
                        self._lap_times.append(t)
                        if t < self._best_lap_time:
                            self._best_lap_time = t
                # Reset lap start for next lap
                self._lap_start_step[lap_complete_ids] = current_step

        # --- Progress reward ---
        current_distance = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        )
        progress = (self.env._last_distance_to_goal - current_distance).clamp(-1.0, 1.0)
        self.env._last_distance_to_goal = current_distance.clone()

        # --- Velocity toward gate reward ---
        # Gate 3 (powerloop): keep pointing at powerloop virtual target (apex/pre-entry/offset center)
        # All other gates: blend vel_toward_current (scale weight 5) + vel_toward_next (scale weight 3)
        # to encourage racing-line corner clipping. The reward scales in train_race.py apply to the
        # final combined value, so we pre-normalize the blend to keep the effective scale comparable.
        drone_vel_w = self.env._robot.data.root_com_lin_vel_w

        # Vel toward current target (desired_pos_w, which for Gate 3 is the powerloop virtual target)
        direction_to_gate = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        direction_to_gate = direction_to_gate / (torch.linalg.norm(direction_to_gate, dim=1, keepdim=True) + 1e-8)
        vel_toward_current = torch.sum(drone_vel_w * direction_to_gate, dim=1).clamp(-2.0, 8.0)

        # Vel toward next gate center (for racing-line bias)
        next_gate_idx_rew = (self.env._idx_wp + 1) % num_gates
        next_gate_pos_w_rew = self.env._waypoints[next_gate_idx_rew, :3]  # (num_envs, 3)
        direction_to_next = next_gate_pos_w_rew - self.env._robot.data.root_link_pos_w
        direction_to_next = direction_to_next / (torch.linalg.norm(direction_to_next, dim=1, keepdim=True) + 1e-8)
        vel_toward_next = torch.sum(drone_vel_w * direction_to_next, dim=1).clamp(-2.0, 8.0)

        # Gate 3 only: preserve powerloop virtual targets (apex/pre-entry/offset center).
        # All other gates: 5/8 current + 3/8 next blend for corner clipping (V26 baseline restored)
        is_powerloop_segment = (self.env._idx_wp == 3)
        blend_current = torch.where(is_powerloop_segment, vel_toward_current, (5.0 / 8.0) * vel_toward_current)
        blend_next    = torch.where(is_powerloop_segment, torch.zeros_like(vel_toward_next), (3.0 / 8.0) * vel_toward_next)
        vel_toward_gate = blend_current + blend_next

        # --- Orientation penalty ---
        # Penalize excessive tilt (large roll/pitch)
        roll, pitch, _ = euler_xyz_from_quat(self.env._robot.data.root_quat_w)
        tilt_penalty = (roll.abs() + pitch.abs())  # sum of absolute roll + pitch
        tilt_penalty = torch.where(tilt_penalty > 0.5, tilt_penalty, torch.zeros_like(tilt_penalty))  # only penalize > ~30°

        # --- Smoothness penalty ---
        action_diff = self.env._actions - self.env._previous_actions
        smoothness_penalty = torch.linalg.norm(action_diff, dim=1)

        # --- Crash detection ---
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "gate_pass": gate_pass * self.env.rew['gate_pass_reward_scale'],
                "vel_toward_gate": vel_toward_gate * self.env.rew['vel_toward_gate_reward_scale'],
                "orientation": tilt_penalty * self.env.rew['orientation_reward_scale'],
                "smoothness": smoothness_penalty * self.env.rew['smoothness_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
            # Extra diagnostics: track raw vel_current and vel_next separately
            # (logged as Episode_Reward/vel_current_mean and vel_next_mean in wandb)
            if "vel_current_mean" not in self._episode_sums:
                self._episode_sums["vel_current_mean"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                self._episode_sums["vel_next_mean"]    = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums["vel_current_mean"] += vel_toward_current
            self._episode_sums["vel_next_mean"]    += vel_toward_next
        else:
            reward = torch.zeros(self.num_envs, device=self.device)
            # Override _desired_pos_w to point to the actual gate center during evaluation
            # so that the TA visualizer markers (red dots) don't reveal the custom apex targeting.
            self.env._desired_pos_w = self.env._waypoints[self.env._idx_wp, :3].clone()
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """31-dim observation: V11 base + gravity_b, next_next_gate, next_gate_normal_b."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        num_gates = self.env._waypoints.shape[0]

        # Body-frame velocities (6 dims)
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b       # (num_envs, 3)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b           # (num_envs, 3)

        # Gravity vector in body frame (3 dims) — replaces quat_w (4 dims)
        drone_quat_w = self.env._robot.data.root_quat_w                 # (num_envs, 4)
        rot_matrix = matrix_from_quat(drone_quat_w)                     # (num_envs, 3, 3)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        gravity_b = torch.matmul(rot_matrix.transpose(1, 2), gravity_w) # (num_envs, 3)

        # Gate indices
        current_gate_idx = self.env._idx_wp
        next_gate_idx = (current_gate_idx + 1) % num_gates
        next_next_gate_idx = (current_gate_idx + 2) % num_gates

        # Current gate position in body frame (3 dims)
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]
        gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            drone_quat_w,
            current_gate_pos_w
        )

        # Next gate position in body frame (3 dims)
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            drone_quat_w,
            next_gate_pos_w
        )

        # Next-next gate position in body frame (3 dims) — NEW
        nn_gate_pos_w = self.env._waypoints[next_next_gate_idx, :3]
        nn_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            drone_quat_w,
            nn_gate_pos_w
        )

        # Current gate normal in body frame (3 dims)
        curr_normal_w = self.env._normal_vectors[current_gate_idx]       # (num_envs, 3)
        curr_normal_b = torch.matmul(rot_matrix.transpose(1, 2), curr_normal_w.unsqueeze(2)).squeeze(2)

        # Next gate normal in body frame (3 dims)
        next_normal_w = self.env._normal_vectors[next_gate_idx]          # (num_envs, 3)
        next_normal_b = torch.matmul(rot_matrix.transpose(1, 2), next_normal_w.unsqueeze(2)).squeeze(2)

        # Sin/cos yaw error relative to gate direction (2 dims)
        _, _, drone_yaw = euler_xyz_from_quat(drone_quat_w)
        gate_yaw = self.env._waypoints[current_gate_idx, -1]
        yaw_error = wrap_to_pi(gate_yaw - drone_yaw)
        sin_yaw_error = torch.sin(yaw_error).unsqueeze(1)               # (num_envs, 1)
        cos_yaw_error = torch.cos(yaw_error).unsqueeze(1)               # (num_envs, 1)

        # Gate index, normalized (1 dim)
        gate_index_norm = (current_gate_idx.float() / num_gates).unsqueeze(1)

        # Previous actions (4 dims)
        prev_actions = self.env._previous_actions
        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                drone_lin_vel_b,    # body linear velocity              (3)
                drone_ang_vel_b,    # body angular velocity             (3)
                gravity_b,          # gravity in body frame             (3)  — was quat_w(4)
                gate_pos_b,         # current gate in body frame        (3)
                next_gate_pos_b,    # next gate in body frame           (3)
                nn_gate_pos_b,      # next-next gate in body frame      (3)  — NEW
                curr_normal_b,      # current gate normal (body frame)  (3)
                next_normal_b,      # next gate normal (body frame)     (3)  — NEW
                sin_yaw_error,      # sin of yaw error to gate          (1)
                cos_yaw_error,      # cos of yaw error to gate          (1)
                gate_index_norm,    # normalized gate index             (1)
                prev_actions,       # previous actions                  (4)
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            if hasattr(self, '_wrong_side_count'):
                extras["Episode_Termination/wrong_side"] = self._wrong_side_count
                self._wrong_side_count = 0
            self.env.extras["log"].update(extras)

            # Track 3-lap success rate
            num_gates = self.env._waypoints.shape[0]
            for eid in env_ids:
                completed_3 = (self.env._n_gates_passed[eid].item() >= 3 * num_gates)
                self._episode_successes.append(1.0 if completed_3 else 0.0)
            if len(self._episode_successes) > 0:
                success_rate = sum(self._episode_successes) / len(self._episode_successes) * 100.0
                self.env.extras["log"]["Lap/success_rate_3lap"] = success_rate

            # Log lap time statistics
            if len(self._lap_times) > 0:
                lap_t = torch.tensor(self._lap_times)
                extras["Lap/mean_lap_time"] = lap_t.mean().item()
                extras["Lap/min_lap_time"] = lap_t.min().item()
                extras["Lap/best_lap_time"] = self._best_lap_time
                extras["Lap/laps_completed"] = len(self._lap_times)
                self.env.extras["log"].update(extras)
                self._lap_times.clear()  # clear buffer after logging

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        num_gates = self.env._waypoints.shape[0]

        # Re-randomize dynamics for resetting environments (domain randomization)
        # Only during training so TA's fixed parameters are preserved during eval
        if self.cfg.is_train:
            self._randomize_dynamics(env_ids)

        # Random starting gate for each resetting environment
        waypoint_indices = torch.randint(0, num_gates, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)

        # Get gate positions and orientations
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]
        theta = self.env._waypoints[waypoint_indices][:, -1]  # gate yaw

        # --- Curriculum-based spawn distance ---
        # Early training: spawn close [1.0, 2.0]m; later: [1.5, 4.0]m
        progress = min(self.env.iteration / 800.0, 1.0)  # ramp over 800 iterations
        dist_min = 1.0 + 0.5 * progress   # 1.0 → 1.5
        dist_max = 2.0 + 2.0 * progress   # 2.0 → 4.0

        # Spawn behind the gate with wider lateral/vertical noise for robustness
        x_local = -torch.empty(n_reset, device=self.device).uniform_(dist_min, dist_max)  # behind gate
        y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)  # wider lateral noise
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)  # wider vertical noise

        # --- 40% mid-track spawns: between consecutive gates (V26) ---
        mid_track_mask = torch.rand(n_reset, device=self.device) < 0.40
        if mid_track_mask.any():
            mid_ids = torch.where(mid_track_mask)[0]
            next_wp = (waypoint_indices[mid_ids] + 1) % num_gates
            lerp_t = torch.rand(len(mid_ids), device=self.device) * 0.6 + 0.2  # [0.2, 0.8]
            mid_pos = (1 - lerp_t).unsqueeze(1) * self.env._waypoints[waypoint_indices[mid_ids], :3] + \
                      lerp_t.unsqueeze(1) * self.env._waypoints[next_wp, :3]
            # For mid-track spawns, override x/y/z with interpolated position
            x0_wp[mid_ids] = self.env._waypoints[waypoint_indices[mid_ids], 0]  # target gate stays the same
            y0_wp[mid_ids] = self.env._waypoints[waypoint_indices[mid_ids], 1]
            # Set spawn position directly for mid-track envs
            x_local[mid_ids] = 0.0
            y_local[mid_ids] = 0.0
            z_local[mid_ids] = 0.0

        # Rotate local offset to world frame using gate yaw
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = (z_wp + z_local).clamp(min=0.15)  # ensure above ground default
        
        # --- 10% ground-level spawns (z=0.05) to mimic TA evaluation ---
        ground_spawn_mask = torch.rand(n_reset, device=self.device) < 0.1
        if ground_spawn_mask.any():
            ground_ids = torch.where(ground_spawn_mask)[0]
            initial_z[ground_ids] = 0.05

        # Override mid-track spawn positions with interpolated positions
        if mid_track_mask.any():
            mid_ids = torch.where(mid_track_mask)[0]
            next_wp = (waypoint_indices[mid_ids] + 1) % num_gates
            lerp_t = torch.rand(len(mid_ids), device=self.device) * 0.6 + 0.2
            mid_pos = (1 - lerp_t).unsqueeze(1) * self.env._waypoints[waypoint_indices[mid_ids], :3] + \
                      lerp_t.unsqueeze(1) * self.env._waypoints[next_wp, :3]
            initial_x[mid_ids] = mid_pos[:, 0]
            initial_y[mid_ids] = mid_pos[:, 1]
            initial_z[mid_ids] = mid_pos[:, 2].clamp(min=0.15)

        # --- 10% Ground takeoff spawns to mimic exact TA evaluation conditions --
        ground_takeoff_mask = torch.rand(n_reset, device=self.device) < 0.1
        ground_takeoff_mask = ground_takeoff_mask & (~mid_track_mask)
        if ground_takeoff_mask.any():
            g_ids = torch.where(ground_takeoff_mask)[0]
            initial_z[g_ids] = 0.05

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # Point drone towards the target gate with wider yaw noise
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)  # wider yaw noise
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise
        )
        default_root_state[:, 3:7] = quat

        # --- 50% velocity-initialized spawns: initial velocity along forward direction ---
        vel_init_mask = torch.rand(n_reset, device=self.device) < 0.5
        # Ensure ground takeoff spawns have 0 initial velocity
        vel_init_mask = vel_init_mask & (~ground_takeoff_mask)
        if vel_init_mask.any():
            vel_ids = torch.where(vel_init_mask)[0]
            speed = torch.empty(len(vel_ids), device=self.device).uniform_(0.5, 3.0)
            # Use initial_yaw forward direction (world-frame vx, vy)
            fwd_yaw = initial_yaw[vel_ids] + yaw_noise[vel_ids]
            vx = speed * torch.cos(fwd_yaw)
            vy = speed * torch.sin(fwd_yaw)
            default_root_state[vel_ids, 7] = vx
            default_root_state[vel_ids, 8] = vy
            default_root_state[vel_ids, 9] = 0.0  # no initial vertical velocity
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._n_gates_passed[env_ids] = 0
        # Use 0 since super()._reset_idx() will reset episode_length_buf to 0
        self._lap_start_step[env_ids] = 0
        # Reset wrong-side cooldown
        if hasattr(self, '_steps_since_gate_pass'):
            self._steps_since_gate_pass[env_ids] = 999
        # Reset powerloop phase
        self._powerloop_phase[env_ids] = 0


        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        # Initialize _last_distance_to_goal AFTER new pose is written, using 3D distance
        # For gate 3 targets, align desired_pos with powerloop phase 0 (apex)
        gate3_mask = (waypoint_indices == 3) if not isinstance(waypoint_indices, int) else False
        if isinstance(gate3_mask, torch.Tensor) and gate3_mask.any():
            g3_reset_ids = env_ids[gate3_mask]
            self.env._desired_pos_w[g3_reset_ids] = self._powerloop_apex.unsqueeze(0)
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids], dim=1
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0


class CircleQuadcopterStrategy:
    """Sim2real strategy for the Circle Track.

    Observation follows the real controller layout plus previous action:
        lin_vel_b (3) | rot_matrix_flat (9) | curr_gate_4_corners_b (12) |
        next_gate_4_corners_b (12) | prev_action (4)

    Rewards follow a sparse sim2real formulation. Resets mix two distributions:
    a gate-state replay buffer populated from successful gate passes and a
    gate-biased geometric sampler as fallback coverage.
    """

    def __init__(self, env: "QuadcopterEnv"):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        if self.cfg.is_train and hasattr(env, 'rew'):
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in ["gate_pass", "lap_incomplete", "cmd_reg", "crash"]
            }

        self._lap_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lap_times = []
        self._best_lap_time = float('inf')
        self._episode_successes = deque(maxlen=100)
        self._steps_since_gate_pass = torch.full((self.num_envs,), 999, dtype=torch.long, device=self.device)
        self._wrong_side_count = 0

        # Swift-style reset support: cache states observed near successful gate passes
        # and replay them with small perturbations during future resets.
        self._replay_capacity = 256
        self._replay_ratio = 0.3
        self._ground_reset_ratio = 0.2
        self._num_gates = int(self.env._waypoints.shape[0])
        self._gate_replay_root_state = torch.zeros(
            self._num_gates, self._replay_capacity, 13, dtype=torch.float, device=self.device
        )
        self._gate_replay_prev_action = torch.zeros(
            self._num_gates, self._replay_capacity, self.cfg.action_space, dtype=torch.float, device=self.device
        )
        self._gate_replay_counts = torch.zeros(self._num_gates, dtype=torch.long, device=self.device)
        self._gate_replay_ptr = torch.zeros(self._num_gates, dtype=torch.long, device=self.device)

        # Pre-compute reference spline through gate positions (closed loop).
        # Used for reset position/velocity sampling when use_spline_reset=True.
        self._build_reference_spline()

        # Delta action smoothness tracking (for cmd_smoothness_scale reward)
        self._prev_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device
        )

        if self.cfg.is_train:
            self._randomize_dynamics(torch.arange(self.num_envs, device=self.device))
        else:
            self._set_default_dynamics(torch.arange(self.num_envs, device=self.device))

    def _randomize_dynamics(self, env_ids: torch.Tensor):
        """V4 DR: V2 ranges for existing params (TWR ±15%, Aero 0.2-3.0x, PID kp/ki ±35%, kd ±50%),
        plus moderate mass ±5% and motor tau 0.7-1.3x."""
        n = len(env_ids)
        cfg = self.cfg

        # TWR ±15% (unchanged from V2)
        self.env._thrust_to_weight[env_ids] = torch.empty(n, device=self.device).uniform_(
            cfg.thrust_to_weight * 0.85, cfg.thrust_to_weight * 1.15)

        # Aero drag 0.2-3.0x (V2 range — wider than V3/V4)
        k_xy_min, k_xy_max = cfg.k_aero_xy * 0.2, cfg.k_aero_xy * 3.0
        k_z_min,  k_z_max  = cfg.k_aero_z  * 0.2, cfg.k_aero_z  * 3.0
        self.env._K_aero[env_ids, 0] = torch.empty(n, device=self.device).uniform_(k_xy_min, k_xy_max)
        self.env._K_aero[env_ids, 1] = torch.empty(n, device=self.device).uniform_(k_xy_min, k_xy_max)
        self.env._K_aero[env_ids, 2] = torch.empty(n, device=self.device).uniform_(k_z_min,  k_z_max)

        # PID kp/ki ±35%, kd ±50% (V2 range — wider than V3/V4)
        self.env._kp_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(cfg.kp_omega_rp * 0.65, cfg.kp_omega_rp * 1.35)
        self.env._kp_omega[env_ids, 1] = self.env._kp_omega[env_ids, 0]
        self.env._ki_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(cfg.ki_omega_rp * 0.65, cfg.ki_omega_rp * 1.35)
        self.env._ki_omega[env_ids, 1] = self.env._ki_omega[env_ids, 0]
        self.env._kd_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(cfg.kd_omega_rp * 0.5, cfg.kd_omega_rp * 1.5)
        self.env._kd_omega[env_ids, 1] = self.env._kd_omega[env_ids, 0]

        self.env._kp_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(cfg.kp_omega_y * 0.65, cfg.kp_omega_y * 1.35)
        self.env._ki_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(cfg.ki_omega_y * 0.65, cfg.ki_omega_y * 1.35)
        self.env._kd_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(cfg.kd_omega_y * 0.5, cfg.kd_omega_y * 1.5)

        # Mass randomization: ±5% (new, moderate — V4 used ±10%)
        mass_var = getattr(cfg, 'mass_variation', 0.05)
        if mass_var > 0:
            mass_scale = torch.empty(n, device=self.device).uniform_(1.0 - mass_var, 1.0 + mass_var)
            self.env._robot_weight[env_ids] = self.env._nominal_robot_weight * mass_scale
        else:
            self.env._robot_weight[env_ids] = self.env._nominal_robot_weight

        # Motor time constant randomization: 0.7-1.3x (new, moderate — V4 used 0.5-2.0x)
        tau_min = getattr(cfg, 'motor_tau_scale_min', 0.7)
        tau_max = getattr(cfg, 'motor_tau_scale_max', 1.3)
        if tau_min < tau_max:
            tau_scale = torch.empty(n, device=self.device).uniform_(tau_min, tau_max)
            self.env._tau_m[env_ids] = cfg.tau_m * tau_scale.unsqueeze(1)
        else:
            self.env._tau_m[env_ids] = cfg.tau_m

    def _set_default_dynamics(self, env_ids: torch.Tensor):
        """Nominal parameter values for evaluation."""
        cfg = self.cfg
        self.env._thrust_to_weight[env_ids] = cfg.thrust_to_weight
        self.env._K_aero[env_ids, 0] = cfg.k_aero_xy
        self.env._K_aero[env_ids, 1] = cfg.k_aero_xy
        self.env._K_aero[env_ids, 2] = cfg.k_aero_z
        self.env._kp_omega[env_ids, 0] = cfg.kp_omega_rp
        self.env._kp_omega[env_ids, 1] = cfg.kp_omega_rp
        self.env._kp_omega[env_ids, 2] = cfg.kp_omega_y
        self.env._ki_omega[env_ids, 0] = cfg.ki_omega_rp
        self.env._ki_omega[env_ids, 1] = cfg.ki_omega_rp
        self.env._ki_omega[env_ids, 2] = cfg.ki_omega_y
        self.env._kd_omega[env_ids, 0] = cfg.kd_omega_rp
        self.env._kd_omega[env_ids, 1] = cfg.kd_omega_rp
        self.env._kd_omega[env_ids, 2] = cfg.kd_omega_y
        self.env._tau_m[env_ids] = cfg.tau_m
        self.env._robot_weight[env_ids] = self.env._nominal_robot_weight

    def _build_reference_spline(self):
        """Fit a periodic cubic spline through gate positions.

        Pre-samples N points along the spline and stores positions, tangent
        vectors, and the target gate index for each sample as torch tensors.
        This is called once at init — zero runtime cost during training.
        """
        num_gates = self._num_gates
        gate_pos_np = self.env._waypoints[:, :3].cpu().numpy()  # (num_gates, 3)

        # Periodic spline: append first gate at end so the curve closes
        t_gates = np.arange(num_gates + 1, dtype=np.float64)
        pts = np.vstack([gate_pos_np, gate_pos_np[0:1]])  # (num_gates+1, 3)

        cs = CubicSpline(t_gates, pts, bc_type='periodic')

        # Sample densely along spline
        n_samples = 1024
        t_dense = np.linspace(0, num_gates, n_samples, endpoint=False)
        pos_dense = cs(t_dense)                # (n_samples, 3)
        tangent_dense = cs(t_dense, 1)         # first derivative → tangent

        # Normalize tangent vectors
        tangent_norms = np.linalg.norm(tangent_dense, axis=1, keepdims=True)
        tangent_norms = np.clip(tangent_norms, 1e-6, None)
        tangent_dense = tangent_dense / tangent_norms

        # Map each sample to its target gate (the next gate in the loop).
        # A sample at parameter t is between gate floor(t) and gate floor(t)+1,
        # so the target is gate (floor(t)+1) % num_gates.
        target_gate = (np.floor(t_dense).astype(int) + 1) % num_gates

        # Store as torch tensors
        self._spline_positions = torch.tensor(pos_dense, dtype=torch.float, device=self.device)
        self._spline_tangents = torch.tensor(tangent_dense, dtype=torch.float, device=self.device)
        self._spline_target_gate = torch.tensor(target_gate, dtype=self.env._idx_wp.dtype, device=self.device)

        # Also store per-segment gate-biased sample weights (Beta distribution on
        # within-segment fraction, more mass near gates).
        # frac_in_segment is the fractional part of t_dense
        frac = t_dense - np.floor(t_dense)
        from scipy.stats import beta as beta_dist_scipy
        # Beta(0.5, 0.5) PDF — U-shaped, peaks at 0 and 1 (near gates)
        weights = beta_dist_scipy.pdf(np.clip(frac, 0.01, 0.99), 0.5, 0.5)
        weights = weights / weights.sum()
        self._spline_sample_weights = torch.tensor(weights, dtype=torch.float, device=self.device)

    def _record_gate_pass_replay(self, env_ids: torch.Tensor):
        """Store states observed immediately after successful gate passes.

        The cache is keyed by the next target gate, so replayed resets start from
        states the policy actually encountered while flying toward that gate.
        """
        if (not self.cfg.is_train) or len(env_ids) == 0:
            return

        target_idx = self.env._idx_wp[env_ids]
        quat_w = self.env._robot.data.root_quat_w[env_ids]
        rot_body_to_world = matrix_from_quat(quat_w)
        ang_vel_w = torch.bmm(
            rot_body_to_world, self.env._robot.data.root_ang_vel_b[env_ids].unsqueeze(-1)
        ).squeeze(-1)
        root_state = torch.cat(
            [
                self.env._robot.data.root_link_pos_w[env_ids],
                quat_w,
                self.env._robot.data.root_com_lin_vel_w[env_ids],
                ang_vel_w,
            ],
            dim=-1,
        )
        prev_action = self.env._previous_actions[env_ids]

        for gate_idx in target_idx.unique(sorted=True).tolist():
            gate_mask = target_idx == gate_idx
            gate_count = int(gate_mask.sum().item())
            if gate_count == 0:
                continue

            gate_slot = int(gate_idx)
            write_start = int(self._gate_replay_ptr[gate_slot].item())
            write_ids = (torch.arange(gate_count, device=self.device) + write_start) % self._replay_capacity
            self._gate_replay_root_state[gate_slot, write_ids] = root_state[gate_mask]
            self._gate_replay_prev_action[gate_slot, write_ids] = prev_action[gate_mask]
            self._gate_replay_ptr[gate_slot] = (write_start + gate_count) % self._replay_capacity
            self._gate_replay_counts[gate_slot] = min(
                self._replay_capacity, int(self._gate_replay_counts[gate_slot].item()) + gate_count
            )

    def _sample_gate_replay_resets(
        self,
        env_ids: torch.Tensor,
        target_gate_idx: torch.Tensor,
        default_root_state: torch.Tensor,
    ) -> torch.Tensor:
        """Replay buffered gate states with small perturbations."""
        if (not self.cfg.is_train) or len(env_ids) == 0:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        available = self._gate_replay_counts[target_gate_idx] > 0
        use_replay = available & (torch.rand(len(env_ids), device=self.device) < self._replay_ratio)
        replay_rows = torch.where(use_replay)[0]
        if len(replay_rows) == 0:
            return use_replay

        replay_gate_idx = target_gate_idx[replay_rows]
        max_slots = self._gate_replay_counts[replay_gate_idx].float()
        sampled_slots = torch.floor(torch.rand(len(replay_rows), device=self.device) * max_slots).long()

        replay_root_state = self._gate_replay_root_state[replay_gate_idx, sampled_slots].clone()
        replay_prev_action = self._gate_replay_prev_action[replay_gate_idx, sampled_slots].clone()

        # Small perturbations keep resets local while preventing rote memorization.
        replay_root_state[:, 0:2] += torch.empty(len(replay_rows), 2, device=self.device).uniform_(-0.15, 0.15)
        replay_root_state[:, 2] = (replay_root_state[:, 2] + torch.empty(len(replay_rows), device=self.device).uniform_(-0.10, 0.10)).clamp(min=0.15)
        replay_root_state[:, 7:10] += torch.empty(len(replay_rows), 3, device=self.device).uniform_(-0.20, 0.20)
        replay_root_state[:, 10:13] += torch.empty(len(replay_rows), 3, device=self.device).uniform_(-0.50, 0.50)

        roll, pitch, yaw = euler_xyz_from_quat(replay_root_state[:, 3:7])
        yaw = yaw + torch.empty(len(replay_rows), device=self.device).uniform_(-0.20, 0.20)
        replay_root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        default_root_state[replay_rows] = replay_root_state
        self.env._previous_actions[env_ids[replay_rows]] = replay_prev_action
        self.env._actions[env_ids[replay_rows]] = replay_prev_action

        # Align the first-step actuator state with the replayed previous action.
        replay_wrench = torch.zeros(len(replay_rows), 4, device=self.device)
        replay_wrench[:, 0] = ((replay_prev_action[:, 0] + 1.0) / 2.0) * self.env._robot_weight[env_ids[replay_rows]] * self.env._thrust_to_weight[env_ids[replay_rows]]
        full_actions = torch.zeros_like(self.env._actions)
        full_actions[env_ids[replay_rows]] = replay_prev_action
        replay_wrench[:, 1:] = self.env._get_moment_from_ctbr(full_actions)[env_ids[replay_rows]]
        replay_motor_speeds = self.env._compute_motor_speeds(replay_wrench)
        self.env._wrench_des[env_ids[replay_rows]] = replay_wrench
        self.env._motor_speeds_des[env_ids[replay_rows]] = replay_motor_speeds
        self.env._motor_speeds[env_ids[replay_rows]] = replay_motor_speeds

        return use_replay

    def _sample_ground_resets(
        self,
        default_root_state: torch.Tensor,
        target_gate_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Sample near-ground starts behind the initial gate for takeoff coverage."""
        if not self.cfg.is_train:
            return torch.zeros(len(target_gate_idx), dtype=torch.bool, device=self.device)

        use_ground = torch.rand(len(target_gate_idx), device=self.device) < self._ground_reset_ratio
        ground_rows = torch.where(use_ground)[0]
        if len(ground_rows) == 0:
            return use_ground

        x_local = torch.empty(len(ground_rows), device=self.device).uniform_(-3.0, -0.5)
        y_local = torch.empty(len(ground_rows), device=self.device).uniform_(-1.0, 1.0)
        gate_idx = int(self.env._initial_wp)
        x0_wp = self.env._waypoints[gate_idx, 0]
        y0_wp = self.env._waypoints[gate_idx, 1]
        theta = self.env._waypoints[gate_idx, -1]
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local

        x0 = x0_wp - x_rot
        y0 = y0_wp - y_rot
        z0 = torch.empty(len(ground_rows), device=self.device).uniform_(0.03, 0.06)
        yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0) + torch.empty(len(ground_rows), device=self.device).uniform_(-0.15, 0.15)

        default_root_state[ground_rows, 0] = x0
        default_root_state[ground_rows, 1] = y0
        default_root_state[ground_rows, 2] = z0
        default_root_state[ground_rows, 7:] = 0.0
        default_root_state[ground_rows, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(ground_rows), device=self.device),
            torch.zeros(len(ground_rows), device=self.device),
            yaw0,
        )
        target_gate_idx[ground_rows] = gate_idx

        return use_ground

    def get_rewards(self) -> torch.Tensor:
        """Sparse rewards: gate_pass + lap_complete + light cmd_reg + crash.

        Inspired by Pasumarti et al. (2026) Eq. 2-6: no dense progress shaping,
        no velocity-toward-gate, no orientation penalty. Let RL discover flight
        behavior from sparse gate-pass signal alone.
        """
        num_gates = self.env._waypoints.shape[0]
        gate_side = self.cfg.gate_model.gate_side

        # --- Gate passing detection (sign-change in gate-frame x) ---
        curr_x = self.env._pose_drone_wrt_gate[:, 0]
        prev_x = self.env._prev_x_drone_wrt_gate
        crossed_plane = (prev_x > 0) & (curr_x <= 0)
        gate_y = self.env._pose_drone_wrt_gate[:, 1]
        gate_z = self.env._pose_drone_wrt_gate[:, 2]
        within_bounds = (gate_y.abs() < gate_side / 2.0) & (gate_z.abs() < gate_side / 2.0)

        gate_passed = crossed_plane & within_bounds
        ids_gate_passed = torch.where(gate_passed)[0]

        # --- Wrong-side crossing with cooldown (prevents reverse-through exploits) ---
        wrong_side_crossed = (prev_x < 0) & (curr_x >= 0) & within_bounds
        self._steps_since_gate_pass += 1
        cooldown_ok = (self._steps_since_gate_pass > 5)
        wrong_side_valid = wrong_side_crossed & cooldown_ok
        wrong_side_ids = torch.where(wrong_side_valid)[0]
        if len(wrong_side_ids) > 0:
            self.env._crashed[wrong_side_ids] = 200
        self._wrong_side_count += len(wrong_side_ids)

        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % num_gates
        self.env._n_gates_passed[ids_gate_passed] += 1
        self.env._desired_pos_w[ids_gate_passed] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
        self.env._prev_x_drone_wrt_gate = curr_x.clone()
        if len(ids_gate_passed) > 0:
            new_pose, _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_pos_w[ids_gate_passed]
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pose[:, 0]
        gate_pass = gate_passed.float()
        if len(ids_gate_passed) > 0:
            self._steps_since_gate_pass[ids_gate_passed] = 0
            self._record_gate_pass_replay(ids_gate_passed)

        # --- Lap time tracking ---
        if len(ids_gate_passed) > 0:
            dt = self.cfg.sim.dt * self.cfg.decimation
            gates_passed_count = self.env._n_gates_passed[ids_gate_passed]
            lap_complete_mask = (gates_passed_count > 0) & (gates_passed_count % num_gates == 0)
            lap_complete_ids = ids_gate_passed[lap_complete_mask]
            if len(lap_complete_ids) > 0:
                current_step = self.env.episode_length_buf[lap_complete_ids]
                lap_steps = current_step - self._lap_start_step[lap_complete_ids]
                lap_seconds = lap_steps.float() * dt
                for t in lap_seconds.cpu().tolist():
                    if t > 0:
                        self._lap_times.append(t)
                        if t < self._best_lap_time:
                            self._best_lap_time = t
                self._lap_start_step[lap_complete_ids] = current_step

        # --- Crash detection ---
        contact_forces = self.env._contact_sensor.data.net_forces_w
        in_contact = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1)
        mask = (self.env.episode_length_buf > 100)
        self.env._crashed = self.env._crashed + (in_contact & mask).int()
        contact_penalty = in_contact.float() * mask.float()

        # Update _last_distance_to_goal (still needed for reset logic)
        current_distance = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        )
        self.env._last_distance_to_goal = current_distance.clone()

        if self.cfg.is_train:
            # Command regularization: body-rate² penalty for real-world smoothness
            cmd_reg = (self.env._actions[:, 1] ** 2 + self.env._actions[:, 2] ** 2) * self.env.rew['cmd_reg_rp_scale'] \
                    + (self.env._actions[:, 3] ** 2) * self.env.rew['cmd_reg_yaw_scale']

            # Delta action smoothness: penalize abrupt changes between consecutive actions
            delta_action = ((self.env._actions - self._prev_actions) ** 2).sum(dim=1)
            cmd_smoothness = delta_action * self.env.rew['cmd_smoothness_scale']
            self._prev_actions = self.env._actions.clone()

            r_gate = gate_pass * self.env.rew['gate_pass_reward_scale']
            r_lap_incomplete = self.env.rew['lap_incomplete_penalty_scale']  # constant per-step cost
            r_crash = contact_penalty * self.env.rew['crash_contact_scale']

            reward = r_gate + r_lap_incomplete + cmd_reg + cmd_smoothness + r_crash

            # Death cost override on terminal episodes
            reward = torch.where(self.env.reset_terminated,
                                 torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            self._episode_sums["gate_pass"] += r_gate
            self._episode_sums["lap_incomplete"] += r_lap_incomplete
            self._episode_sums["cmd_reg"] += cmd_reg
            self._episode_sums["crash"] += r_crash
        else:
            reward = torch.zeros(self.num_envs, device=self.device)
            self.env._desired_pos_w = self.env._waypoints[self.env._idx_wp, :3].clone()

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """40-dim observation: extends controller_simple_policy.py with previous action.

        Layout: lin_vel_b(3) | rot_matrix_flat(9) | curr_gate_corners_b(12) |
                next_gate_corners_b(12) | prev_action(4)

        Previous action added per Swift (Kaufmann et al. 2023): helps policy infer
        current dynamics state (e.g., motor spin-up) for better sim2real transfer.

        Gate corners in body frame use the same computation as controller_simple_policy.py:
            corners_w = local_square @ rot_gate.T + gate_pos
            corners_b  = (corners_w - drone_pos) @ rot_body
        where rot_body is the body→world rotation matrix.
        """
        num_gates = self.env._waypoints.shape[0]

        # (3) Body-frame linear velocity
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b

        # (9) Body-to-world rotation matrix, row-major flattened
        drone_quat_w = self.env._robot.data.root_quat_w
        rot_body = matrix_from_quat(drone_quat_w)           # (num_envs, 3, 3), body→world
        rot_flat = rot_body.reshape(self.num_envs, 9)

        # Gate indices
        current_gate_idx = self.env._idx_wp
        next_gate_idx = (current_gate_idx + 1) % num_gates

        # Gate rotation matrices
        curr_gate_quat = self.env._waypoints_quat[current_gate_idx]   # (num_envs, 4)
        next_gate_quat = self.env._waypoints_quat[next_gate_idx]
        rot_gate_curr = matrix_from_quat(curr_gate_quat)               # (num_envs, 3, 3)
        rot_gate_next = matrix_from_quat(next_gate_quat)

        # Gate center positions in world
        gate_pos_curr_w = self.env._waypoints[current_gate_idx, :3]   # (num_envs, 3)
        gate_pos_next_w = self.env._waypoints[next_gate_idx, :3]

        # Gate corners in world frame: corners = local_square @ rot_gate.T + gate_pos
        corners_curr_w = (torch.bmm(self.env._local_square, rot_gate_curr.permute(0, 2, 1))
                          + gate_pos_curr_w.unsqueeze(1))              # (num_envs, 4, 3)
        corners_next_w = (torch.bmm(self.env._local_square, rot_gate_next.permute(0, 2, 1))
                          + gate_pos_next_w.unsqueeze(1))

        # Transform to body frame: for row vectors, v_b = v_w @ R_body
        drone_pos = self.env._robot.data.root_link_pos_w              # (num_envs, 3)
        corners_curr_b = torch.bmm(corners_curr_w - drone_pos.unsqueeze(1), rot_body)  # (num_envs, 4, 3)
        corners_next_b = torch.bmm(corners_next_w - drone_pos.unsqueeze(1), rot_body)

        obs = torch.cat([
            drone_lin_vel_b,                                    # (num_envs,  3)
            rot_flat,                                           # (num_envs,  9)
            corners_curr_b.reshape(self.num_envs, 12),          # (num_envs, 12)
            corners_next_b.reshape(self.num_envs, 12),          # (num_envs, 12)
            self.env._previous_actions,                         # (num_envs,  4)
        ], dim=-1)                                              # total: 40

        # Observation noise for sim2real robustness (Swift, Kaufmann et al. 2023)
        # Simulates Vicon measurement noise, velocity estimation error, and gate calibration error.
        if self.cfg.is_train:
            noise_std = torch.tensor(
                [0.05] * 3          # lin_vel_b: ±0.05 m/s (Vicon velocity from numerical diff)
                + [0.01] * 9        # rot_matrix: ±0.01 (small attitude noise)
                + [0.02] * 12       # curr_gate_corners: ±0.02m (gate calibration)
                + [0.02] * 12       # next_gate_corners: ±0.02m
                + [0.0] * 4,        # prev_action: no noise (known exactly)
                device=self.device,
            )
            obs = obs + torch.randn_like(obs) * noise_std

        return {"policy": obs}

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset from a mix of replayed successful states and gate-biased samples."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # --- Logging ---
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            extras["Episode_Termination/wrong_side"] = self._wrong_side_count
            self._wrong_side_count = 0
            self.env.extras["log"].update(extras)

            num_gates = self.env._waypoints.shape[0]
            for eid in env_ids:
                completed_3 = (self.env._n_gates_passed[eid].item() >= 3 * num_gates)
                self._episode_successes.append(1.0 if completed_3 else 0.0)
            if len(self._episode_successes) > 0:
                success_rate = sum(self._episode_successes) / len(self._episode_successes) * 100.0
                self.env.extras["log"]["Lap/success_rate_3lap"] = success_rate

            if len(self._lap_times) > 0:
                lap_t = torch.tensor(self._lap_times)
                extras["Lap/mean_lap_time"] = lap_t.mean().item()
                extras["Lap/min_lap_time"] = lap_t.min().item()
                extras["Lap/best_lap_time"] = self._best_lap_time
                extras["Lap/laps_completed"] = len(self._lap_times)
                self.env.extras["log"].update(extras)
                self._lap_times.clear()

        # --- Robot reset ---
        self.env._robot.reset(env_ids)

        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]
            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)
            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                              high=int(self.env.max_episode_length))

        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._motor_speeds_des[env_ids] = 0.0
        self.env._wrench_des[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]
        num_gates = self.env._waypoints.shape[0]

        if self.cfg.is_train:
            self._randomize_dynamics(env_ids)

        # --- Spawn position (training) ---
        use_spline = getattr(self.cfg, 'use_spline_reset', True)

        if use_spline:
            # Spline reset: sample from precomputed reference spline with velocity
            # along tangent. Gate-biased weights concentrate samples near gates.
            sample_idx = torch.multinomial(
                self._spline_sample_weights, n_reset, replacement=True
            )
            spawn_pos = self._spline_positions[sample_idx]          # (n_reset, 3)
            spawn_tangent = self._spline_tangents[sample_idx]       # (n_reset, 3)
            target_gate_idx = self._spline_target_gate[sample_idx]  # (n_reset,)

            # Add noise perpendicular to tangent for diversity
            lateral_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            vertical_noise = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)

            # Perpendicular direction in xy plane
            tangent_xy = spawn_tangent[:, :2]
            tangent_xy_len = torch.linalg.norm(tangent_xy, dim=1, keepdim=True).clamp(min=1e-4)
            tangent_xy_norm = tangent_xy / tangent_xy_len
            perp_dir = torch.stack([-tangent_xy_norm[:, 1], tangent_xy_norm[:, 0]], dim=1)

            initial_x = spawn_pos[:, 0] + perp_dir[:, 0] * lateral_noise
            initial_y = spawn_pos[:, 1] + perp_dir[:, 1] * lateral_noise
            initial_z = (spawn_pos[:, 2] + vertical_noise).clamp(min=0.15)

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z

            # Velocity along spline tangent: match real flight speed (~2.9 m/s avg)
            vel_min = getattr(self.cfg, 'spline_vel_min', 1.0)
            vel_max = getattr(self.cfg, 'spline_vel_max', 3.0)
            speed = torch.empty(n_reset, device=self.device).uniform_(vel_min, vel_max)
            default_root_state[:, 7:10] = spawn_tangent * speed.unsqueeze(1)
            default_root_state[:, 10:13] = 0.0  # zero angular velocity

            # Yaw aligned to spline tangent + noise
            initial_yaw = torch.atan2(spawn_tangent[:, 1], spawn_tangent[:, 0])
            yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
        else:
            # Fallback: V3-style linear interpolation between gates (zero velocity)
            waypoint_indices = torch.randint(0, num_gates, (n_reset,), device=self.device,
                                             dtype=self.env._idx_wp.dtype)
            next_wp_indices = (waypoint_indices + 1) % num_gates

            beta_dist = torch.distributions.Beta(0.5, 0.5)
            lerp_t = beta_dist.sample((n_reset,)).to(self.device)

            curr_gate_pos = self.env._waypoints[waypoint_indices, :3]
            next_gate_pos = self.env._waypoints[next_wp_indices, :3]
            spawn_pos = (1 - lerp_t).unsqueeze(1) * curr_gate_pos + lerp_t.unsqueeze(1) * next_gate_pos

            lateral_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            vertical_noise = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)

            segment_dir = next_gate_pos - curr_gate_pos
            segment_dir_xy = segment_dir[:, :2]
            segment_len = torch.linalg.norm(segment_dir_xy, dim=1, keepdim=True).clamp(min=1e-4)
            segment_dir_xy = segment_dir_xy / segment_len
            perp_dir = torch.stack([-segment_dir_xy[:, 1], segment_dir_xy[:, 0]], dim=1)

            initial_x = spawn_pos[:, 0] + perp_dir[:, 0] * lateral_noise
            initial_y = spawn_pos[:, 1] + perp_dir[:, 1] * lateral_noise
            initial_z = (spawn_pos[:, 2] + vertical_noise).clamp(min=0.15)

            x0_wp = self.env._waypoints[next_wp_indices, 0]
            y0_wp = self.env._waypoints[next_wp_indices, 1]

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z
            default_root_state[:, 7:] = 0.0

            initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
            yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            target_gate_idx = next_wp_indices.clone()

        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise
        )
        default_root_state[:, 3:7] = quat

        # Replay from previously observed successful states when available.
        replay_mask = self._sample_gate_replay_resets(env_ids, target_gate_idx, default_root_state)

        # A subset of resets starts on the ground to match real deployment.
        ground_mask = self._sample_ground_resets(default_root_state, target_gate_idx)
        if ground_mask.any():
            ground_env_ids = env_ids[ground_mask]
            self.env._actions[ground_env_ids] = 0.0
            self.env._previous_actions[ground_env_ids] = 0.0
            self.env._motor_speeds[ground_env_ids] = 0.0
            self.env._motor_speeds_des[ground_env_ids] = 0.0
            self.env._wrench_des[ground_env_ids] = 0.0
        # For non-ground, non-spline resets, restore target from next_wp_indices
        # (ground resets override target_gate_idx internally)
        if not use_spline:
            non_ground_rows = torch.where(~ground_mask)[0]
            if len(non_ground_rows) > 0:
                target_gate_idx[non_ground_rows] = next_wp_indices[non_ground_rows]

        # --- Play mode initial position ---
        if not self.cfg.is_train:
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)
            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)
            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0
            default_root_state[:, 7:] = 0.0
            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # --- Commit state ---
        # Target the next gate from the spawn position
        self.env._idx_wp[env_ids] = target_gate_idx if self.cfg.is_train else waypoint_indices
        target_idx = self.env._idx_wp[env_ids]
        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[target_idx, :2].clone()
        self.env._desired_pos_w[env_ids, 2]  = self.env._waypoints[target_idx, 2].clone()
        self.env._n_gates_passed[env_ids] = 0
        self._lap_start_step[env_ids] = 0
        self._steps_since_gate_pass[env_ids] = 999
        self._prev_actions[env_ids] = 0.0
        if self.cfg.is_train:
            effective_replay_mask = replay_mask & (~ground_mask)
            spline_mask = (~replay_mask) & (~ground_mask)
            self.env.extras.setdefault("log", {})
            self.env.extras["log"]["Reset/replay_ratio"] = effective_replay_mask.float().mean().item()
            self.env.extras["log"]["Reset/ground_ratio"] = ground_mask.float().mean().item()
            self.env.extras["log"]["Reset/spline_ratio"] = spline_mask.float().mean().item()
            self.env.extras["log"]["Reset/use_spline"] = float(use_spline)

        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids], dim=1
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0
        self.env._crashed[env_ids] = 0

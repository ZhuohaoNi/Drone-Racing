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

        # --- Gate 3 powerloop: 3-phase guide ---
        targeting_gate3 = (self.env._idx_wp == 3)
        if targeting_gate3.any():
            g3_ids = torch.where(targeting_gate3)[0]
            drone_z = self.env._robot.data.root_link_pos_w[g3_ids, 2]
            drone_pos = self.env._robot.data.root_link_pos_w[g3_ids]

            # Phase 0 → 1:
            phase0 = (self._powerloop_phase[g3_ids] == 0)
            dist_to_apex = torch.linalg.norm(self._powerloop_apex - drone_pos[phase0], dim=1) if phase0.any() else torch.tensor([], device=self.device)
            promote_0to1 = phase0.clone()
            if phase0.any():
                promote_0to1[phase0] = (drone_z[phase0] > 1.3) | (dist_to_apex < 0.8)
            if promote_0to1.any():
                self._powerloop_phase[g3_ids[promote_0to1]] = 1

            # Phase 1 → 2: close to pre-entry (<1.0m)
            phase1 = (self._powerloop_phase[g3_ids] == 1)
            if phase1.any():
                dist_to_pre = torch.linalg.norm(
                    self._gate3_pre_entry - drone_pos[phase1], dim=1
                )
                promote_1to2 = dist_to_pre < 1.0
                if promote_1to2.any():
                    p1_ids = g3_ids[phase1]
                    self._powerloop_phase[p1_ids[promote_1to2]] = 2

            # Apply targets based on current phase
            still_phase0 = (self._powerloop_phase[g3_ids] == 0)
            if still_phase0.any():
                self.env._desired_pos_w[g3_ids[still_phase0]] = self._powerloop_apex.unsqueeze(0)
            still_phase1 = (self._powerloop_phase[g3_ids] == 1)
            if still_phase1.any():
                self.env._desired_pos_w[g3_ids[still_phase1]] = self._gate3_pre_entry.unsqueeze(0)
            still_phase2 = (self._powerloop_phase[g3_ids] == 2)
            if still_phase2.any():
                self.env._desired_pos_w[g3_ids[still_phase2]] = self._gate3_offset_center.unsqueeze(0)

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

        # Gates 2 & 3: preserve existing powerloop behavior (only vel_toward_current, weight=1.0)
        # - idx_wp==3 is the active powerloop phase, idx_wp==2 is the approach segment before it
        # Other gates: blend 6/8 current + 2/8 next to encourage corner-clipping racing lines
        is_powerloop_segment = (self.env._idx_wp == 2) | (self.env._idx_wp == 3)
        blend_current = torch.where(is_powerloop_segment, vel_toward_current, (6.0 / 8.0) * vel_toward_current)
        blend_next    = torch.where(is_powerloop_segment, torch.zeros_like(vel_toward_next), (2.0 / 8.0) * vel_toward_next)
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

        # --- 10% mid-track spawns: between consecutive gates ---
        mid_track_mask = torch.rand(n_reset, device=self.device) < 0.1
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
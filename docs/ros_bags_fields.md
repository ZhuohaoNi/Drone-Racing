# Rosbag Data Fields Reference

Complete reference for all data recorded in ROS2 bag files during real-world Crazyflie flights at Pennovation. Use this to understand what's available for sim-real gap analysis and domain randomization tuning.

## Bag Format

- **Storage:** MCAP (`.mcap` files)
- **ROS distro:** Jazzy
- **Namespace:** Typically `crazy_jirl_b3` (backup: `crazy_jirl_b2`)

---

## Topics Overview

| Topic | Message Type | Typical Rate | Description |
|-------|-------------|-------------|-------------|
| `/<ns>/odom` | `nav_msgs/msg/Odometry` | ~100 Hz | Ground truth state from Vicon mocap |
| `/<ns>/observations` | `jirl_interfaces/msg/Observations` | ~88 Hz | Policy observation vector |
| `/ctbr_cmd` | `jirl_interfaces/msg/CommandCTBR` | ~88 Hz | Thrust + body rate commands to drone |
| `/<ns>/pose` | `geometry_msgs/msg/PoseStamped` | ~120 Hz | Raw Vicon pose (position + orientation) |
| `/<ns>/trajectory` | `jirl_interfaces/msg/Trajectory` | varies | Desired trajectory setpoints (SE3 controller) |
| `/tf` | `tf2_msgs/msg/TFMessage` | ~120 Hz | TF transform tree |
| `/multi_odometry` | `jirl_interfaces/msg/OdometryArray` | ~100 Hz | Multi-drone odometry array |

---

## Topic Details

### 1. `/<ns>/odom` — Odometry (Ground Truth)

**Message type:** `nav_msgs/msg/Odometry`

This is the primary state estimation from the Vicon motion capture system. It provides full 6-DOF pose and twist (velocity) in the world frame.

| Field Path | Type | Units | Description |
|-----------|------|-------|-------------|
| `pose.pose.position.x` | float64 | m | World-frame X position |
| `pose.pose.position.y` | float64 | m | World-frame Y position |
| `pose.pose.position.z` | float64 | m | World-frame Z position (altitude) |
| `pose.pose.orientation.x` | float64 | - | Quaternion X |
| `pose.pose.orientation.y` | float64 | - | Quaternion Y |
| `pose.pose.orientation.z` | float64 | - | Quaternion Z |
| `pose.pose.orientation.w` | float64 | - | Quaternion W |
| `twist.twist.linear.x` | float64 | m/s | World-frame linear velocity X |
| `twist.twist.linear.y` | float64 | m/s | World-frame linear velocity Y |
| `twist.twist.linear.z` | float64 | m/s | World-frame linear velocity Z |
| `twist.twist.angular.x` | float64 | rad/s | World-frame angular velocity X |
| `twist.twist.angular.y` | float64 | rad/s | World-frame angular velocity Y |
| `twist.twist.angular.z` | float64 | rad/s | World-frame angular velocity Z |

**Derived quantities (computed in pipeline):**
- **Body-frame linear velocity:** `R^T @ world_lin_vel` — used as policy observation input
- **Body-frame angular velocity:** `R^T @ world_ang_vel * 180/pi` — in deg/s for comparison with commands
- **Euler angles (XYZ):** Roll, Pitch, Yaw in degrees — derived from quaternion

**DR relevance:**
- Position bounds tell you the flight envelope — widen if sim track is too constrained
- Velocity statistics directly inform velocity DR ranges
- Angular velocity ranges tell you how aggressively the drone maneuvers in practice
- Euler angle ranges inform tilt limits in simulation

---

### 2. `/<ns>/observations` — Policy Observation Vector

**Message type:** `jirl_interfaces/msg/Observations`

The observation vector fed to the neural network policy at each step. This is the bridge between sim and real — if these differ, the policy sees a different world.

| Field | Type | Shape | Units | Description |
|-------|------|-------|-------|-------------|
| `lin_vel` | float64[3] | (3,) | m/s | Body-frame linear velocity [vx, vy, vz] |
| `rot` | float64[9] | (9,) | - | Flattened rotation matrix (3x3, row-major) |
| `corners_pos_b_curr` | float64[12] | (4,3) | m | Current gate 4 corner positions in body frame |
| `corners_pos_b_next` | float64[12] | (4,3) | m | Next gate 4 corner positions in body frame |
| `cond` | float64[2] | (2,) | - | Condition flags (usage depends on policy version) |

**Observation vector construction (36-dim or 40-dim):**
```
[lin_vel(3)] + [rot(9)] + [corners_curr(12)] + [corners_next(12)] = 36-dim
                                                    + [prev_action(4)] = 40-dim (if used)
```

**DR relevance:**
- Compare `lin_vel` here vs body-frame velocity derived from odom — any discrepancy indicates a Vicon/estimation issue
- Compare `rot` with quaternion-derived rotation — check for numerical drift
- Gate corner positions in body frame reveal how the drone "sees" the gates — compare with sim to verify gate geometry matches
- The `cond` field may encode gate-switching logic

---

### 3. `/ctbr_cmd` — Collective Thrust + Body Rate Commands

**Message type:** `jirl_interfaces/msg/CommandCTBR`

Commands sent from the policy (or SE3 controller) to the Crazyflie drone via Crazyradio.

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `crazyflie_name` | string | - | Target drone namespace (e.g. `crazy_jirl_b3`) |
| `thrust_pwm` | uint16 | PWM ticks | Raw PWM thrust command (0-65535) |
| `thrust_n` | float64 | N | Thrust in Newtons (converted from PWM) |
| `roll_rate` | float64 | deg/s | Commanded roll rate |
| `pitch_rate` | float64 | deg/s | Commanded pitch rate |
| `yaw_rate` | float64 | deg/s | Commanded yaw rate |

**DR relevance:**
- **Thrust distribution** is critical for sim-real gap. Compare `thrust_n` range with your sim's thrust model. The real Crazyflie has a non-linear thrust curve (PWM → force).
- **Command saturation:** If rates hit ±200 deg/s limits or thrust hits PWM extremes, the policy is too aggressive. Widen action space penalties in sim.
- **Thrust-to-weight ratio:** Crazyflie weighs ~30g. Hover thrust ≈ 0.3N. Compare mean thrust with hover to gauge how much margin the policy uses.
- **Rate tracking error** (commanded vs actual angular velocity) reveals how well the onboard controller tracks — large errors mean your sim's rate response model is too optimistic.

---

### 4. `/<ns>/pose` — PoseStamped (Raw Mocap)

**Message type:** `geometry_msgs/msg/PoseStamped`

Raw pose from the Vicon system before velocity estimation. Published at a slightly higher rate than odom.

| Field Path | Type | Units | Description |
|-----------|------|-------|-------------|
| `header.stamp` | Time | - | Vicon timestamp |
| `header.frame_id` | string | - | Reference frame |
| `pose.position.x/y/z` | float64 | m | World-frame position |
| `pose.orientation.x/y/z/w` | float64 | - | Quaternion orientation |

**DR relevance:**
- Higher rate than odom — use to check Vicon frequency and jitter
- Compare with odom to verify the velocity estimation pipeline doesn't introduce lag

---

### 5. `/<ns>/trajectory` — Desired Trajectory

**Message type:** `jirl_interfaces/msg/Trajectory`

Desired trajectory setpoints used by the SE3 controller during non-racing phases (takeoff, hover, landing). During racing (policy active), this topic typically has 0 messages.

| Field | Type | Shape | Units | Description |
|-------|------|-------|-------|-------------|
| `x` | float64[3] | (3,) | m | Desired position [x, y, z] |
| `x_dot` | float64[3] | (3,) | m/s | Desired velocity |
| `x_ddot` | float64[3] | (3,) | m/s² | Desired acceleration |
| `x_dddot` | float64[3] | (3,) | m/s³ | Desired jerk |
| `x_ddddot` | float64[3] | (3,) | m/s⁴ | Desired snap |
| `yaw` | float64 | - | rad | Desired yaw angle |
| `yaw_dot` | float64 | - | rad/s | Desired yaw rate |
| `yaw_ddot` | float64 | - | rad/s² | Desired yaw acceleration |

**Note:** In the circle track bags (v2, v3, v4), this topic has 0 messages — the policy runs directly without trajectory tracking.

---

### 6. `/tf` — Transform Tree

**Message type:** `tf2_msgs/msg/TFMessage`

TF transform broadcasts, typically from the mocap system publishing the drone's pose in the transform tree.

| Field Path | Type | Description |
|-----------|------|-------------|
| `transforms[].header.frame_id` | string | Parent frame |
| `transforms[].child_frame_id` | string | Child frame |
| `transforms[].transform.translation.x/y/z` | float64 | Translation (m) |
| `transforms[].transform.rotation.x/y/z/w` | float64 | Rotation quaternion |

**DR relevance:**
- Contains the same position data as `/pose` but in TF format
- Useful for verifying frame conventions match between sim and real

---

### 7. `/multi_odometry` — Multi-Drone Odometry

**Message type:** `jirl_interfaces/msg/OdometryArray`

Array of odometry messages for all tracked objects (drones + possibly obstacles).

| Field | Type | Description |
|-------|------|-------------|
| `odom_array` | Odometry[] | Array of `nav_msgs/Odometry` messages |

**DR relevance:**
- If multiple drones are flying, this shows all their states
- Can be used to check if gate obstacles or other objects are tracked

---

## Existing JSON Output vs Full Pipeline

The original `process_bag_with_br_pos_export.py` exports 2 JSON files. Our pipeline (`analyze_rosbag.py`) exports these same files for backward compatibility, plus additional comprehensive exports.

### Original output (2 files)

| File | Contents | Limitations |
|------|----------|-------------|
| `*_angular_velocity_data.json` | Odom timestamps + actual angular vel (body, deg/s) + cmd timestamps + commanded rates | Only angular velocity, no thrust data |
| `*_position_data.json` | Odom timestamps + position + velocity (body) + quaternion + euler | No world-frame velocity, no angular velocity in world frame |

### Full pipeline additional output (6+ files)

| File | Contents | Why it matters for DR |
|------|----------|---------------------|
| `*_statistics.json` | Summary stats: duration, rates, velocity bounds, thrust stats, euler ranges, gate passes, lap estimates | Quick overview for comparing flights |
| `*_odom_full.json` | All odom data including both world and body frame velocities | World-frame velocity needed for some DR comparisons |
| `*_ctbr_full.json` | Full command data: thrust PWM, thrust N, all rates | Thrust distribution is critical for thrust curve DR |
| `*_observations_full.json` | Complete observation vector: lin_vel, rotation matrix, gate corners (curr+next), condition flags | Direct sim-real obs comparison |
| `*_pose_full.json` | Raw Vicon pose data at higher rate | Frequency analysis, latency estimation |
| `*_gate_passes.json` | Gate pass timestamps and lap count | Per-lap analysis, consistency check |

### Additional plots (not in original)

| Plot | Purpose |
|------|---------|
| `*_observations.png` | Gate distance, gate center in body frame, condition flags over time |
| `*_policy_input_obs.png` | Full 36-dim observation vector visualization |
| `*_velocity_world.png` | World-frame velocity components and speed magnitude |
| `*_trajectory_topdown.png` | 2D top-down view of trajectory with gate positions |
| `*_frequency_analysis.png` | Message timing jitter for odom and commands |
| `*_thrust_analysis.png` | Thrust vs altitude and thrust vs speed scatter |
| `*_pose_vs_odom.png` | Comparison of /pose and /odom topics |

---

## Circle Track Gate Configuration

```python
'circle': [
    # [x, y, z, roll, pitch, yaw]
    [ 0.0, 3.0, 0.75, 0.0, 0.0,  0.00],   # Gate 0 — south, facing north
    [-1.5, 4.5, 0.75, 0.0, 0.0, -1.57],   # Gate 1 — west, facing east
    [ 0.0, 6.0, 1.75, 0.0, 0.0,  3.14],   # Gate 2 — north, facing south, ELEVATED
    [ 1.5, 4.5, 0.75, 0.0, 0.0,  1.57],   # Gate 3 — east, facing west
]
```

Gate side length: 1.0 m (half-side = 0.5 m). Each gate is a 1m x 1m square centered at the waypoint position, oriented by the Euler angles.

---

## Key Insights for Domain Randomization

When analyzing bag data, focus on these quantities for DR tuning:

1. **Velocity envelope**: What range of speeds does the drone actually reach? Set DR velocity noise to cover observed variance.
2. **Thrust curve**: Plot thrust_n vs altitude and vs speed. The real thrust-to-force mapping is non-linear — compare with your sim model.
3. **Rate tracking delay**: Compare commanded vs actual angular velocity. The gap reveals the real drone's actuator response time — add matching delay in sim.
4. **Position bounds**: How much does the drone deviate from the ideal gate-to-gate path? This informs initial position randomization.
5. **Euler angle ranges**: Real flights may have more/less tilt than sim allows. Adjust tilt limits accordingly.
6. **Gate observation accuracy**: Compare gate corners in body frame with what sim produces at the same relative position — any systematic offset indicates a calibration error.
7. **Message frequency & jitter**: If real data arrives at inconsistent rates, consider adding observation delay/noise in sim.
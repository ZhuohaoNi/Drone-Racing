# How the Drone Racing System Works: A Deep Dive

This document explains the fundamental mechanisms that make a drone fly through gates in the correct sequence and direction, how the drone "knows" where it is, and other important concepts for Part 2 of the project.

---

## Table of Contents

1. [How the Drone Knows Where It Is](#1-how-the-drone-knows-where-it-is)
2. [How the Gate Sequence and Direction Are Enforced](#2-how-the-gate-sequence-and-direction-are-enforced)
3. [How Observations Guide the Policy](#3-how-observations-guide-the-policy)
4. [How Rewards Shape Behavior](#4-how-rewards-shape-behavior)
5. [How Resets Shape What the Policy Learns](#5-how-resets-shape-what-the-policy-learns)
6. [How the Drone Actually Flies](#6-how-the-drone-actually-flies)
7. [The Powerloop and Chicane: Special Challenges](#7-the-powerloop-and-chicane-special-challenges)
8. [Frame Transformations: Why They Matter](#8-frame-transformations-why-they-matter)
9. [Domain Randomization: Surviving the Eval Gap](#9-domain-randomization-surviving-the-eval-gap)
10. [Common Pitfalls and Design Tradeoffs](#10-common-pitfalls-and-design-tradeoffs)

---

## 1. How the Drone Knows Where It Is

### No SLAM, No Map, No Sensors in the Traditional Sense

This is a **simulation-based reinforcement learning** setup, not a real robotics perception pipeline. The drone does **not** use SLAM, cameras, LiDAR, or any perception algorithm. It does not "see" the gates visually.

Instead, the simulator (NVIDIA Isaac Lab) acts as a **perfect oracle**. At every timestep, the simulator knows exactly:
- Where the drone is (position, orientation, velocity — from the physics engine)
- Where every gate is (hardcoded in the track definition)
- Whether the drone has collided with anything (contact sensor)

The strategy code in `quadcopter_strategies.py` reaches into the simulator state and **selects** which of these ground-truth values to expose to the neural network as observations. The policy network only sees what you put in the observation vector — it is "blind" to everything else.

### What the drone "knows" (current implementation)

The 25-dimensional observation vector gives the policy:

| What | How it knows | Frame |
|---|---|---|
| Its own velocity | `root_com_lin_vel_b` — physics engine | Body |
| Its own rotation rate | `root_ang_vel_b` — physics engine | Body |
| Its own orientation | `root_quat_w` — physics engine | World |
| Where the current gate is | `subtract_frame_transforms(drone_pos, drone_quat, gate_pos)` — computed | Body |
| Where the next gate is | Same transform for gate `(idx+1) % N` — computed | Body |
| Which direction to fly through the gate | `_normal_vectors[current_gate_idx]` — precomputed from gate yaw | World |
| How far off its heading is from the gate | `sin/cos(gate_yaw - drone_yaw)` — computed | N/A |
| What it did last timestep | `_previous_actions` — stored | N/A |

### What the drone does NOT know

- Its absolute world position (intentionally excluded — ego-centric design)
- How many gates it has passed
- Its lap count
- The full track layout (it only sees the current and next gate)
- Any visual information about the environment

### The "map" is implicit

The drone doesn't have a map. Instead, the track topology is implicitly encoded in the **gate sequencing system** (Section 2). The observation always points toward the *current* target gate, and when that gate is passed, the pointer advances to the next one. From the policy's perspective, it's always chasing a carrot (the current gate position in body frame), and the carrot moves to the next gate when it arrives. The policy doesn't need to know the whole track — it just needs to fly toward the carrot and through it correctly.

This is similar to how a GPS navigator works: you don't need to memorize the entire route, you just follow the next turn instruction.

---

## 2. How the Gate Sequence and Direction Are Enforced

This is the most critical mechanism in the system. The drone must pass gates **in order** (0, 1, 2, 3, 4, 5, 6, 0, 1, ...) and in the **correct direction** (flying through the front of the gate, not the back). Here's how it works:

### 2.1 The Track Definition

The track is defined as an ordered list of waypoints in `quadcopter_env.py` (line 416-424 for powerloop):

```python
'powerloop': [
    [2.0, 3.5, 0.75, 0.0, 0.0, -1.5708],    # Gate 0: x, y, z, roll, pitch, yaw
    [-1.5, 3.5, 2.00, 0.0, 0.0, 0.7854],     # Gate 1
    [-0.625, 0.0, 0.75, 0.0, 0.0, 1.5708],   # Gate 2
    [0.625, 0.0, 0.75, 0.0, 0.0, 1.5708],    # Gate 3
    [-1.5, -3.5, 2.00, 0.0, 0.0, 2.356],     # Gate 4
    [2.0, -3.5, 0.75, 0.0, 0.0, -1.5708],    # Gate 5
    [0.625, 0.0, 0.75, 0.0, 0.0, -1.5708],   # Gate 6
],
```

Each waypoint has 6 values: `[x, y, z, roll, pitch, yaw]`. The `yaw` angle defines the **direction the gate faces** — specifically, the gate's local x-axis (its normal vector) points in the direction the drone should fly through.

**Key insight:** Gate 3 and Gate 6 are at the same physical location `(0.625, 0.0, 0.75)` but with opposite yaw values (`1.5708` vs `-1.5708`). This means the drone must fly through the same physical gate in **opposite directions** at different points in the lap. This is the powerloop maneuver.

### 2.2 The Gate Normal Vector

For each gate, the environment precomputes a **normal vector** (`_normal_vectors`) from the gate's yaw angle (line 453):

```python
rotmat_np_gate = rot_from_euler.as_matrix()
gate_normal_np = rotmat_np_gate[:, 0]   # first column of rotation matrix = local x-axis
```

The normal vector points in the direction the drone should be flying when it passes through the gate. For a gate with `yaw = 1.5708` (90 degrees), the normal points in the +y direction.

### 2.3 The Waypoint Index Tracker

Each environment maintains a per-drone integer `_idx_wp` that tracks which gate the drone should pass through next. This is initialized during reset and incremented on gate passage.

Think of it like a checklist: the drone must check off gate 0 before it can work on gate 1. It cannot skip gates or go backward.

### 2.4 Gate Passage Detection (The Core Mechanism)

The gate passage detection in `get_rewards()` (lines 92-107) works by transforming the drone's position into the **gate's local coordinate frame**:

```python
# _pose_drone_wrt_gate is computed in _get_dones() via subtract_frame_transforms
curr_x = self.env._pose_drone_wrt_gate[:, 0]   # drone's x in gate frame
prev_x = self.env._prev_x_drone_wrt_gate        # same, but from previous timestep
```

In the gate's local frame:
- **x-axis** = the gate's normal direction (the direction you should fly through)
- **y-axis** = left-right across the gate opening
- **z-axis** = up-down across the gate opening

A gate passage is detected when **three conditions** are simultaneously true:

```python
crossed_plane = (prev_x > 0) & (curr_x <= 0)           # 1. Crossed from front to back
within_y = gate_y.abs() < gate_side / 2.0               # 2. Within lateral bounds
within_z = gate_z.abs() < gate_side / 2.0               # 3. Within vertical bounds
gate_passed = crossed_plane & within_bounds
```

**Condition 1: Direction enforcement.** The drone must cross from positive x to negative x in the gate frame. This means it flew through the gate in the direction the normal vector points. If it flies through the back of the gate (negative to positive), `crossed_plane` is False. This is how **direction** is enforced.

**Condition 2 & 3: Spatial bounds.** The drone must actually pass through the gate opening, not around it.

**Sequence enforcement.** The detection only checks the drone's position relative to `_waypoints[_idx_wp]` — the **current** target gate. Even if the drone happens to fly through gate 5 while its `_idx_wp` is 2, no passage is recorded because `_pose_drone_wrt_gate` is computed relative to gate 2, not gate 5. The drone **must** pass through the gates in order.

### 2.5 What Happens on Gate Passage

When `gate_passed` is True for a drone:

```python
self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % num_gates
self.env._n_gates_passed[ids_gate_passed] += 1
self.env._desired_pos_w[ids_gate_passed] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
```

1. **Waypoint index advances** — now the drone targets the next gate
2. **Gate counter increments** — for tracking progress and lap counting
3. **Desired position updates** — the "carrot" moves to the next gate

The modulo `% num_gates` wraps around: after gate 6, the next target is gate 0 again (new lap).

### 2.6 How the Policy Learns to Follow the Sequence

The policy doesn't explicitly "know" about the sequence. Instead, it learns implicitly through the **reward structure**:

1. **Progress reward** gives positive reward for reducing distance to `_desired_pos_w` (the current target gate). The policy learns: "fly toward the position I see in my observation."
2. **Gate pass reward** gives a large bonus (+200) when a gate is passed. The policy learns: "flying through that position is very good."
3. **Velocity toward gate reward** gives positive reward for flying fast toward the target gate.

After a gate is passed, `_desired_pos_w` snaps to the next gate, so the progress reward immediately starts pointing the policy toward the next target. The transition is seamless from the policy's perspective — the "carrot" just moved.

### 2.7 Alternative / Better Approaches to Consider

**Current limitation:** The gate detection relies on a sign change in x within a single timestep. If the drone is very fast, it could potentially skip across the gate plane between two physics steps without the sign change being captured. This is unlikely at 50Hz policy rate for typical drone speeds, but could become an issue at extreme speeds.

**Better gate detection approach:** Instead of relying on a single-step sign change, you could use a **distance-based** approach: detect passage when the drone is within some small radius of the gate center AND the dot product of the drone's velocity with the gate normal is positive (flying in the right direction). This is more robust to high speeds but requires careful tuning of the detection radius.

**Better sequence guidance:** The current system only shows the policy the current and next gate. For track sections that require planning ahead (powerloop, chicane), showing a **third look-ahead gate** would help the policy anticipate upcoming turns. The policy could learn to start turning before it even reaches the current gate.

---

## 3. How Observations Guide the Policy

### 3.1 The Fundamental Question: What Does the Policy Need to Know?

A good observation space answers these questions for the policy:
1. **Where am I going?** (gate positions in body frame)
2. **How am I oriented?** (quaternion or gravity vector)
3. **How am I moving?** (linear and angular velocity)
4. **Which direction should I fly through the gate?** (gate normal, yaw error)
5. **What did I do last?** (previous actions — for smooth control)

### 3.2 Why Body Frame for Gate Positions?

Gate positions are provided in the **body frame** of the drone (via `subtract_frame_transforms`). This means the observation tells the policy "the gate is 2 meters ahead and 1 meter to my left" rather than "the gate is at world coordinate (3.5, 2.0, 0.75)."

**Why this is better:**
- **Generalization:** The same body-frame vector means the same thing regardless of where the drone is on the track. "Gate is 2m ahead" always means "fly forward." A world-frame position like (3.5, 2.0) means nothing without knowing the drone's own world position.
- **Simpler learning:** The policy doesn't need to learn to subtract its own position from the gate position — the observation already provides the relative vector.
- **Rotation invariance:** The body-frame transformation handles the drone's orientation. If the drone is rotated 90 degrees, the body-frame vector automatically rotates to still point "toward the gate" from the drone's perspective.

### 3.3 Why the Quaternion (and Potential Improvements)

The world-frame quaternion (`root_quat_w`, 4 dims) tells the policy the drone's absolute orientation. This is important because:
- The drone needs to know "which way is up" to avoid flipping
- The thrust vector is along the drone's body z-axis, so orientation determines which direction thrust pushes

**Problem:** Quaternions are not intuitive for a neural network. They have a double-cover problem (q and -q represent the same rotation), and the network must learn to decode the quaternion into useful geometric information.

**Better alternative: Gravity vector in body frame** (3 dims). Transform `[0, 0, -1]` (gravity direction) into the drone's body frame. This directly tells the policy "down is in this direction relative to me." It's 1 fewer dimension and provides the critical "which way is up" information in a form that's trivial for the network to use.

### 3.4 Why Sin/Cos Yaw Error Instead of Raw Yaw

The observation includes `sin(yaw_error)` and `cos(yaw_error)` rather than `yaw_error` directly. This is because:

- Raw yaw has a **discontinuity** at +/-pi (180 degrees). A yaw error of +179 degrees and -179 degrees are almost the same orientation, but numerically they're 358 degrees apart.
- Sin/cos encoding is **continuous**: sin(179deg) = 0.017, sin(-179deg) = -0.017. The network sees these as nearby values.
- The pair `(sin, cos)` uniquely identifies any angle without ambiguity.

### 3.5 Why Previous Actions Matter

Including `_previous_actions` (4 dims) in the observation allows the policy to:
- **Smooth its control:** It can see what it did last step and choose a similar action, avoiding jitter.
- **Compensate for dynamics:** The motor response depends on the current motor state, which is related to the previous action.
- **Implement derivative-like control:** By knowing its previous action and current state, it can implicitly compute how the state changed in response to its action.

### 3.6 What's Missing from Current Observations

**Gate normal frame inconsistency:** Gate normals are in world frame while gate positions are in body frame. The policy must implicitly learn to transform between frames. Converting the gate normal to body frame would make the observation self-consistent.

**No gravity vector:** The quaternion provides orientation but is harder for the network to interpret. A body-frame gravity vector is a more direct signal.

**Only 2 gates visible:** The policy can only see the current and next gate. For the powerloop (gates 2->3, where both require passing through the same physical area in opposite directions), seeing gate idx+2 would help the policy plan the loop trajectory.

**No speed/height scalar:** The policy can compute its speed from the velocity vector, but an explicit altitude observation (height above ground) could help with ground avoidance. Currently, the policy must infer height from the gate's body-frame z-coordinate and its own orientation.

---

## 4. How Rewards Shape Behavior

### 4.1 The Reward Design Philosophy

The reward structure has a hierarchy:
1. **Gate passage** (sparse, scale=200) — The ultimate goal. Passing a gate is worth 200 points.
2. **Progress** (dense, scale=50) — The guide. Every timestep that reduces distance to the gate earns a small reward.
3. **Velocity toward gate** (dense, scale=5) — The accelerator. Rewards flying fast in the right direction.
4. **Orientation penalty** (dense, scale=-2) — The stabilizer. Penalizes excessive tilt.
5. **Smoothness penalty** (dense, scale=-0.5) — The smoother. Penalizes jerky control.
6. **Crash penalty** (dense, scale=-1) — Contact avoidance.
7. **Death cost** (sparse, scale=-10) — Episode termination penalty.

### 4.2 Dense vs Sparse Rewards

**Sparse rewards** (gate_pass, death_cost) only fire on specific events. If you ONLY had sparse rewards, the drone would need to **randomly** discover that flying through a gate is good. With 8192 parallel drones, some will stumble through gates by accident, but learning would be extremely slow.

**Dense rewards** (progress, velocity, orientation, smoothness) fire every timestep and provide continuous gradient information. The progress reward is the most important: it gives a **smooth gradient** from anywhere on the track toward the nearest gate. The drone doesn't need to accidentally find a gate — it's immediately rewarded for flying in the right direction.

**How they work together:**
1. Early training: Progress reward teaches the drone to approach gates. Most reward comes from distance reduction.
2. Mid training: The drone starts passing gates, getting the large +200 bonus. This reinforces the approach behavior.
3. Late training: The velocity reward pushes the drone to fly faster through the gates it already knows how to reach.

### 4.3 Why Reward Magnitudes Matter

The relative magnitudes of rewards determine what the policy prioritizes. From V3-V8 experiments:

- **V4:** vel_toward_gate (scale 5) contributed ~750 total reward vs gate_pass's ~200. The drone learned to fly fast but crashed on tight turns — speed dominated.
- **V6:** death_cost raised to -50 made the drone overly cautious — 26s laps vs 17s.
- **V3:** Orientation penalty reduced to -0.5 caused flips — the drone learned it was worth tilting wildly to gain speed.

**The lesson:** You cannot tune rewards independently. Every reward is relative to every other reward. The policy optimizes the **sum**, so it will sacrifice one component to maximize total reward. A reward that's too large will dominate; too small will be ignored.

### 4.4 The Death Cost Mechanism

When `reset_terminated` is True (crash, out of bounds, etc.), the entire timestep reward is **replaced** with `death_cost`:

```python
reward = torch.where(self.env.reset_terminated,
                     torch.ones_like(reward) * self.env.rew['death_cost'], reward)
```

This means the policy doesn't get the normal rewards for the terminal timestep — it only gets -10. This teaches the policy that dying is bad, but the magnitude (-10) is carefully chosen to not make the policy overly risk-averse. The V6 experiment showed that -50 made the drone too cautious.

### 4.5 Why Progress Uses Distance Reduction, Not Raw Distance

The progress reward is `last_distance - current_distance`, not `-current_distance`. Why?

- **Raw distance** (e.g., `-distance_to_gate`) would give a constant negative reward that decreases as the drone approaches. The problem: the drone gets the same total reward whether it takes 1 second or 10 seconds to reach the gate, as long as it gets there. There's no urgency.
- **Distance reduction** rewards each step where the drone got closer. If it's standing still, reward is 0. If it moves toward the gate, reward is positive. If it moves away, reward is negative. This creates an immediate signal: "you should be moving toward the gate right now."

The clamping `clamp(-1.0, 1.0)` prevents outliers (e.g., teleporting due to a physics glitch) from dominating the reward.

### 4.6 Alternative Reward Designs to Consider

**Time penalty:** A constant per-timestep negative reward (e.g., -0.1 per step) would directly incentivize completing laps faster. However, it also incentivizes dying quickly (death is one way to end the episode). Must be paired with sufficient death penalty.

**Gate proximity bonus:** Instead of binary gate passage, give a bonus that increases as the drone gets close to the gate center. This creates a smoother gradient through the gate and could help with gate alignment.

**Lap time reward:** Instead of per-gate rewards, give a large reward inversely proportional to total lap time. This directly optimizes the evaluation metric but is very sparse — only fires once per lap.

**Curriculum learning:** Start with easy rewards (just reach gate 0), then progressively add gates. This avoids the early exploration problem but requires careful scheduling.

---

## 5. How Resets Shape What the Policy Learns

### 5.1 Why Resets Matter

Every time an episode ends (crash, timeout, or lap completion), the drone resets to a new starting state. The **distribution of starting states** determines what situations the policy practices.

### 5.2 Current Reset Strategy

**Random gate selection:** The drone spawns near a random gate (uniform over all 7 gates). This means it practices all track sections equally.

**Position randomization:** 1.5-3m behind the gate, +/-0.5m lateral, +/-0.3m vertical. This teaches the policy to approach gates from slightly different angles and distances.

**Orientation:** Facing toward the target gate with +/-0.15 rad yaw noise. This is very conservative — the drone always starts roughly pointed at the gate.

**Velocity:** Always zero. The drone starts from a hover.

### 5.3 Why the Reset Distribution Matters

If you only reset near gate 0, the policy would become very good at reaching gate 0 from various positions but might never learn the gate 3->4 transition (the powerloop exit). Random gate selection ensures all transitions get practiced.

However, **uniform** sampling might not be optimal. Some gate transitions are harder than others (the powerloop is harder than a straight segment). Spending more training time on hard transitions could improve overall performance.

### 5.4 The Zero-Velocity Problem

Starting from zero velocity means the policy never practices **approaching a gate at speed**. In a real race, the drone is always moving when it reaches a gate. The policy trained with zero-velocity starts might learn to:
1. Accelerate from hover
2. Fly toward the gate
3. Pass through the gate

But it never practices step 2 at high speed — because every reset starts from zero. Adding initial velocity in the direction of the gate would create more realistic training conditions.

### 5.5 Play Mode Reset

During evaluation, the reset is different: the drone always starts near gate 0 at z=0.05m (near ground level) with random lateral/longitudinal offsets. This means:
- The policy must be able to take off from near-ground
- It must handle various starting positions relative to gate 0
- The TA will hard-code specific offsets within the same bounds

---

## 6. How the Drone Actually Flies

### 6.1 The Control Pipeline

The drone does NOT directly control its position. The control pipeline has multiple layers:

```
Neural Network (50 Hz)
  |
  v  [4 actions: thrust, roll_rate, pitch_rate, yaw_rate]
  |
PID Controller (500 Hz)  <-- runs inside _apply_action()
  |
  v  [4 motor speed commands]
  |
Motor Dynamics (500 Hz)  <-- first-order lag with time constant tau_m
  |
  v  [4 actual motor speeds]
  |
Physics Engine (500 Hz)  <-- forces and torques applied to rigid body
  |
  v  [drone position, orientation, velocity]
```

### 6.2 What the Policy Controls

The policy outputs 4 continuous values in [-1, 1]:

| Action | Mapping | Physical Effect |
|---|---|---|
| `actions[:, 0]` | Thrust | Mapped to `(action + 1) / 2 * weight * TWR`. At action=0, thrust equals weight (hover). At action=1, thrust is maximum. |
| `actions[:, 1]` | Roll rate | Mapped to `body_rate_scale_xy * action` (100 deg/s at action=1). Rolls the drone left/right. |
| `actions[:, 2]` | Pitch rate | Same scale. Pitches the drone forward/backward. |
| `actions[:, 3]` | Yaw rate | Mapped to `body_rate_scale_z * action` (200 deg/s at action=1). Rotates the drone around its vertical axis. |

This is called **CTBR (Collective Thrust + Body Rate)** control. The policy commands desired body rotation rates, and an internal PID controller tracks those rates by adjusting individual motor speeds.

### 6.3 How the Drone Moves Laterally

A quadcopter can only produce thrust along its body z-axis (upward relative to the drone body). To move forward, the drone must **tilt forward** (pitch), which angles the thrust vector. The horizontal component of thrust accelerates the drone forward, while the reduced vertical component means it descends slightly.

This is why the orientation penalty matters: too much tilt means the drone can't maintain altitude, but some tilt is necessary for any lateral movement. The V2 threshold of 0.5 rad (~30 degrees) allows normal racing tilt while penalizing dangerous attitudes.

### 6.4 The Decimation Factor

The physics simulation runs at 500 Hz, but the policy only runs at 50 Hz (decimation = 10). Between policy steps, the same action is applied for 10 physics steps. The PID controller runs at every physics step (500 Hz), continuously tracking the commanded body rates.

This means the policy doesn't need to worry about low-level motor control — the PID handles that. The policy operates at a higher level: "I want to thrust at 80% and roll right at 50 deg/s."

---

## 7. The Powerloop and Chicane: Special Challenges

### 7.1 The Powerloop (Gates 2-3)

The powerloop involves two gates that require the drone to fly through the same area in opposite directions:

- Gate 2: position (-0.625, 0.0, 0.75), yaw = +90 deg (fly in +y direction)
- Gate 3: position (0.625, 0.0, 0.75), yaw = +90 deg (fly in +y direction... wait, same direction?)

Actually, gates 2 and 3 are at slightly different x positions but similar y. The real powerloop is between gate 3 (yaw = +90 deg) and gate 6 (yaw = -90 deg) — same physical position, opposite pass direction. The drone must:

1. Fly through gate 3 in one direction
2. Loop back around
3. Fly through the same physical gate (now gate 6) in the opposite direction

**Why this is hard:** The policy must learn a non-trivial trajectory (a vertical or horizontal loop) to reverse direction. With only 2 look-ahead gates visible, the policy might not anticipate the need to loop.

### 7.2 The Chicane (Gates 5-6-0)

A chicane is a rapid sequence of offset gates requiring alternating turns:
- Gate 5: (2.0, -3.5, 0.75, yaw=-90 deg)
- Gate 6: (0.625, 0.0, 0.75, yaw=-90 deg)
- Gate 0: (2.0, 3.5, 0.75, yaw=-90 deg)

The drone must execute quick lateral adjustments while maintaining forward speed. This is where smooth control and good velocity management are critical.

### 7.3 Why Look-Ahead Matters

For the powerloop, by the time the drone is approaching gate 3, it needs to already be planning the loop trajectory for gate 6 (which is 3 gates ahead). With only 1 look-ahead gate visible (next gate), the policy can't see gate 6 until it passes gate 5. Adding a second or third look-ahead gate would give the policy more planning horizon.

---

## 8. Frame Transformations: Why They Matter

### 8.1 The Three Frames

| Frame | Origin | Axes | Use |
|---|---|---|---|
| **World frame** | Fixed in space | x=forward, y=left, z=up (convention) | Absolute positions, gate definitions |
| **Body frame** | Drone center of mass | x=forward, y=left, z=up (relative to drone) | Velocities, control inputs |
| **Gate frame** | Gate center | x=normal (pass direction), y=left, z=up | Gate passage detection |

### 8.2 The Frame Inconsistency Bug

Current observation:
- Gate positions: **body frame** (via `subtract_frame_transforms`)
- Gate normal: **world frame** (directly from `_normal_vectors`)

This means the policy receives gate position as "the gate is 2m ahead and 1m to my left" (body frame), but the gate normal as "the gate faces in the world +y direction" (world frame). The policy must implicitly learn to rotate the world-frame normal by its own orientation to interpret it relative to the body-frame gate position.

Fixing this is simple: rotate the gate normal into body frame using the drone's rotation matrix.

### 8.3 Why `subtract_frame_transforms` Works

This function computes: given the drone's world-frame pose `(pos_drone, quat_drone)` and a target world-frame position `target_pos`, return the target position in the drone's body frame.

Mathematically: `target_body = R_drone^T * (target_world - pos_drone)`

where `R_drone` is the rotation matrix from the drone's quaternion.

---

## 9. Domain Randomization: Surviving the Eval Gap

### 9.1 Why It's Needed

The TAs will alter 3 random physical parameters during evaluation. If your policy was trained with fixed physics, it may fail when the dynamics change. Domain randomization trains the policy to handle a **range** of dynamics, making it robust to the specific values the TAs choose.

### 9.2 How It Works

On every episode reset, `_randomize_dynamics()` samples new values for TWR, aero drag, and PID gains within the TA-specified ranges. Each of the 8192 parallel environments has **different** physics parameters. The policy must learn a **single behavior** that works across all these variations.

This is analogous to training a driver on both wet and dry roads — they learn a style that works on both, rather than being perfect on one and failing on the other.

### 9.3 What It Can't Fix

Domain randomization helps with **parametric** uncertainty (different values of known parameters). It doesn't help with:
- **Structural** differences (the TA's environment has features you didn't model)
- **Observation noise** (if the TA adds sensor noise)
- **Catastrophic** parameter values (if TWR drops so low the drone can't hover)

The TA ranges are moderate enough that domain randomization should suffice, but always test your policy with edge-case parameter combinations.

---

## 10. Common Pitfalls and Design Tradeoffs

### 10.1 Reward Hacking

The policy optimizes the reward function, not your intention. Common reward hacking behaviors:
- **Orbiting the gate:** If progress reward is strong but gate_pass detection has edge cases, the drone may circle near the gate collecting progress reward without actually passing through.
- **Speed runs into walls:** If velocity reward is too strong, the drone flies fast toward gates but can't brake for tight turns.
- **Playing dead:** If penalties are too strong, the policy learns that doing nothing (hovering) avoids all penalties and collects zero reward, which is better than the large negative reward from crashing.

### 10.2 The Exploration-Exploitation Tradeoff

With `entropy_coef = 0.0`, there's no explicit exploration incentive. The policy relies on:
- `init_noise_std = 1.0` (initial high randomness)
- 8192 parallel environments (diversity through parallelism)
- Random resets (exposure to all track sections)

If the policy converges too early (entropy drops to near zero before discovering the full track), consider adding a small entropy bonus. But V2 succeeded without it.

### 10.3 Observation Scale Sensitivity

Neural networks learn best when inputs are roughly in [-1, 1]. The current observations have different scales:
- Quaternion: always in [-1, 1]
- Linear velocity: could be 0-5 m/s
- Gate position: could be 0-7 m away
- Angular velocity: could be 0-5 rad/s

`empirical_normalization = True` handles this by maintaining running mean/std statistics and normalizing observations before feeding them to the network. **This is critical** — V7 showed that disabling it collapses performance, and V8 showed that manual normalization with fixed constants cannot replicate the benefit.

**Risk:** The normalizer state is part of the model but stored separately. The TA's evaluation runner must load it from the checkpoint. If it doesn't, your policy will receive unnormalized observations and likely fail.

### 10.4 The Credit Assignment Problem

When the drone receives a +200 gate_pass reward, which of the hundreds of preceding actions should get credit? The drone made many decisions (accelerate, turn, maintain altitude) over several seconds to reach the gate. GAE with `gamma=0.99` and `lam=0.95` spreads credit back over ~20-50 timesteps, but the full approach to a gate might take 50-150 timesteps (1-3 seconds at 50 Hz).

This is why dense rewards (progress, velocity) are so important — they provide immediate credit for actions that reduce distance or increase approach speed, rather than waiting for the sparse gate_pass reward.

### 10.5 Why Not Use Absolute Position?

You might think: "Why not tell the policy the drone's world position? It would know exactly where it is on the track."

Problems with absolute position:
- **Overfitting to track coordinates:** The policy memorizes "at position (2.0, 3.5, 0.75), turn right." This doesn't generalize.
- **Redundant information:** Body-frame gate position already encodes relative position. Adding world position adds dimensions without new information.
- **Harder learning:** The network must learn to subtract its position from gate positions — work that `subtract_frame_transforms` already does.

The ego-centric design forces the policy to learn **relational** behavior ("fly toward the thing in front of me") rather than **memorized** behavior ("turn right at coordinates x=2.0").

### 10.6 Summary: The Full Loop

Here is how everything connects in a single timestep:

1. **Observation:** The strategy computes a 25-dim vector: "I'm moving at 3 m/s forward, the gate is 2m ahead and 1m left, the gate faces +y, my heading is 15 degrees off..."
2. **Policy:** The actor network maps observation -> 4 actions: [thrust=0.6, roll_rate=0.3, pitch_rate=-0.1, yaw_rate=0.2]
3. **Control:** The PID controller tracks the commanded body rates by adjusting motor speeds
4. **Physics:** The motors generate forces and torques, the drone moves through space
5. **Detection:** The environment checks if the drone crossed through the current target gate
6. **Reward:** If it did, +200 bonus and target advances. Otherwise, small progress/velocity rewards.
7. **Learning:** PPO uses rewards and advantages to update the policy weights so the actor produces better actions next time
8. **Repeat:** 24 timesteps are collected, then 20 gradient steps update the networks

Over thousands of iterations, the policy converges from random flailing to gate-racing behavior — not because it was programmed to race, but because the observation + reward structure made racing the optimal strategy.

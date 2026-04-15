# Rosbag Comparative Analysis — Group 30 Circle Track

Analysis of three consecutive flights on the circle track at Pennovation. **cyclev2** was the successful run; **cyclev3** and **cyclev4** exhibited different failure modes. All flights used namespace `crazy_jirl_b3`.

Full pipeline output is in `plots_full/<bag_name>/`.

---

## Flight Summary

| Metric | cyclev2 (success) | cyclev3 (fail) | cyclev4 (fail) |
|--------|-------------------|----------------|----------------|
| Duration | 49.1 s | 47.4 s | 27.6 s (cut short) |
| Gate passes | 63 | 66 | 28 |
| Estimated laps | 15.75 | 16.5 | 7.0 |
| Mean lap time | 2.70 s | 2.44 s | 2.93 s |
| Lap time std | 0.31 s | 0.02 s | 0.26 s |
| Mean speed (body) | 2.43 m/s | 2.50 m/s | 2.17 m/s |
| Max speed | 3.73 m/s | 4.01 m/s | 4.05 m/s |

**Key observation:** v3 actually completed *more* laps than v2, faster and more consistently (2.44 s/lap vs 2.70 s/lap, with 0.02s std). v4 was cut short at 27.6s, roughly half the duration of v2/v3.

---

## 1. Odometry & Trajectory Shape

### Position Envelope

| Axis | cyclev2 | cyclev3 | cyclev4 |
|------|---------|---------|---------|
| X range | [-1.49, 1.44] | [-1.17, 1.16] | [-1.21, 1.70] |
| Y range | [2.63, 5.63] | [2.32, 5.60] | [2.90, 6.01] |
| Z range | [0.17, 1.54] | [0.33, 1.41] | [0.17, 2.03] |
| Z std | 0.326 | 0.303 | **0.535** |

**v2** flies a symmetric circle centered at (0.0, 4.45) with radius ~1.17m. The Z range [0.17, 1.54] reflects the altitude change from Gate 0/1/3 (z=0.75) to Gate 2 (z=1.75).

**v3** flies a slightly tighter circle (radius 1.09m) but with a *narrower* Z range [0.33, 1.41]. It never reaches gate 2's height of 1.75m — the max altitude is 1.41m, which is 34cm below the gate center. This suggests the drone may have been flying through the bottom portion of the elevated gate, or possibly missing it.

**v4** has the widest trajectory with max Y=6.01 (overshooting gate 2 at y=6.0) and **max Z=2.03m** — well above gate 2's height. The Z std of 0.535 (vs 0.326/0.303 for v2/v3) shows unstable altitude control.

### Trajectory Circularity

All three fly roughly circular paths centered near (0.0, 4.45) — close to the geometric center of the 4 gates. v2 and v3 are clean circles; v4 is distorted and asymmetric.

---

## 2. Orientation (Euler Angles)

| Angle | cyclev2 | cyclev3 | cyclev4 |
|-------|---------|---------|---------|
| Roll range | [-43.4, 33.3] | **[-39.8, 2.6]** | [-34.8, 46.0] |
| Pitch range | [-42.9, 43.7] | **[-1.2, 35.4]** | [-34.6, 46.4] |
| Yaw range | [-180, 180] | [-180, 180] | **[-68.6, 90.4]** |

### v3: One-sided tilt (critical finding)

v3's roll is always negative (mean -28.5 deg during flight) and pitch is always positive (mean +25.7 deg). The drone was constantly banked/pitched in one direction throughout the entire flight. In v2, roll and pitch oscillate symmetrically around zero as expected for a drone flying a circle.

This means v3 was flying a "coordinated turn" locked to one tilt direction, never transitioning between left/right bank. This is consistent with a policy that learned to fly the circle in only one rotational direction with a fixed attitude bias.

**DR implication:** If training only reinforces one turning direction, the policy can converge to a degenerate solution that works in sim but fails in real when any perturbation pushes it off the locked-in attitude.

### v4: Limited yaw range

v4 yaw only spans [-68.6, 90.4] deg — it never completes a full 360-degree rotation. v2 and v3 both traverse the full yaw range. This suggests v4's policy was not properly commanding yaw turns, or the drone was stuck in a partial orbit and never recovered.

---

## 3. CTBR Commands & Thrust

### Thrust Statistics

| Metric | cyclev2 | cyclev3 | cyclev4 |
|--------|---------|---------|---------|
| Mean thrust (N) | 0.767 | 0.698 | 0.646 |
| Std thrust (N) | 0.484 | 0.468 | **0.535** |
| PWM at max (%) | 10.3% | 0.2% | **28.3%** |
| PWM at min (%) | 5.8% | 1.7% | 9.5% |

**v4 has severe thrust saturation**: 28.3% of commands hit the max PWM (60000). The high thrust demand + high altitude (2.03m) suggests the policy was commanding aggressive vertical maneuvers, overshooting gates vertically, and then saturating thrust trying to recover.

**v3 has the lowest saturation** — only 0.2% at max PWM. Combined with its lower mean thrust (0.698 N vs 0.767 N for v2), v3 was flying more conservatively on thrust but relying heavily on attitude for maneuvering.

### Commanded Rate Saturation

| Rate | cyclev2 | cyclev3 | cyclev4 |
|------|---------|---------|---------|
| Roll at ±100 deg/s | 0.0% | 1.9% | **7.9%** |
| Pitch at ±100 deg/s | 2.1% | **7.7%** | 5.6% |
| Yaw at ±200 deg/s | 0.4% | 1.6% | 0.0% |

**v3 saturates pitch commands** at 7.7% — consistent with the one-sided pitch bias requiring persistent large pitch commands. **v4 saturates roll** at 7.9% — suggesting aggressive roll corrections, likely trying to maintain the circular path when altitude oscillations threw it off course.

---

## 4. Rate Tracking (Commanded vs Actual Angular Velocity)

| Axis | cyclev2 mean err | cyclev3 mean err | cyclev4 mean err |
|------|------------------|------------------|------------------|
| Roll | 24.5 deg/s | **18.1 deg/s** | 28.1 deg/s |
| Pitch | 30.3 deg/s | **19.9 deg/s** | 31.5 deg/s |
| Yaw | **58.1 deg/s** | 26.3 deg/s | 43.1 deg/s |

**v2 has the worst yaw tracking** (58.1 deg/s mean error, 166.6 deg/s at p95). The yaw axis is consistently the hardest to track across all flights — the onboard yaw controller has the most lag.

**v3 has the best tracking** across all axes. This is because v3 flies with constant attitude (no rapid roll/pitch reversals), which the Crazyflie's rate controller handles well. The locked-in tilt means it's essentially asking for steady-state body rates, which are easier to track.

**DR implication:** Yaw tracking error is 2-3x larger than roll/pitch. The sim should model this asymmetry — either with axis-specific rate response delays or by applying heavier noise to yaw observations.

---

## 5. Observations & Perception

### Gate Center in Body Frame

| Metric | cyclev2 | cyclev3 | cyclev4 |
|--------|---------|---------|---------|
| X (forward) range | [-2.03, 1.90] | **[-0.89, 0.38]** | [-1.87, 2.31] |
| Y (left) range | [-1.55, 1.95] | **[-0.30, 2.21]** | [-1.74, 1.48] |
| Z (up) range | [-0.49, 1.85] | [-0.51, 1.48] | [-0.83, 1.53] |

**v3's narrow forward range** [-0.89, 0.38] means the gate center is never more than 0.89m behind the drone in the body x-axis. In v2, it ranges from -2.03 to +1.90 — the drone passes through gates and swings around to approach them from various angles. v3's locked attitude means it always sees gates from a similar relative position.

**v3's asymmetric Y range** [-0.30, 2.21] means the gate is almost always to the left (positive Y in body frame). This confirms the one-directional circle flying.

### Observation Velocity vs Odom Body Velocity

| Axis | cyclev2 max err | cyclev3 max err | cyclev4 max err |
|------|-----------------|-----------------|-----------------|
| vx | 0.021 m/s | 0.022 m/s | **0.219 m/s** |
| vy | 0.018 m/s | 0.023 m/s | **0.078 m/s** |
| vz | 0.017 m/s | 0.017 m/s | **0.046 m/s** |

**v4 has notably higher observation-odom discrepancy** — the max vx error is 0.219 m/s (10x worse than v2/v3). This occurs during aggressive maneuvers where the velocity estimation pipeline can't keep up with the rapid state changes. This is a real perception gap.

### Condition Flags

The `cond` field contains near-zero float values (e.g., 8.52e-319) across all flights, suggesting they are either uninitialized memory or not used by the current policy version. These are not meaningful condition signals.

---

## 6. Environment / Mocap System

### Vicon (Pose Topic) Timing

| Metric | cyclev2 | cyclev3 | cyclev4 |
|--------|---------|---------|---------|
| Rate | 120 Hz | 120 Hz | 120 Hz |
| Median dt | 8.3 ms | 8.3 ms | 8.5 ms |
| Jitter (std) | 0.31 ms | 0.32 ms | **0.55 ms** |
| Gaps > 15ms | 0 | 0 | **3** |

The Vicon system runs at a consistent 120 Hz across all flights. **v4 has slightly higher jitter** (0.55ms std vs 0.31ms) and 3 gaps >15ms — these brief mocap dropouts could contribute to the velocity estimation errors seen in v4's observations.

### Odom Timing

Odom runs at ~100 Hz (median dt 8.4ms) but with high variance (std 3.33ms) and many gaps >15ms (~20% of intervals). This is consistent across all flights and reflects the processing pipeline adding jitter. The policy receives observations at the odom rate, not the raw Vicon rate.

**DR implication:** Sim should model observation delay/jitter on the order of 3-5ms std. The 120Hz Vicon data is downsampled to ~100Hz odom with significant jitter.

---

## 7. Failure Mode Analysis

### cyclev3: "Locked Attitude" Flying

**Symptoms:**
- Roll always negative (-39.8 to +2.6 deg), pitch always positive (-1.2 to +35.4 deg)
- Gate center always to the drone's left (Y body range [-0.30, 2.21])
- Very consistent lap times (2.44s ± 0.02s) — suspiciously consistent
- Fastest of the three, lowest thrust usage
- Excellent rate tracking (lowest errors)

**Diagnosis:** The policy converged to a degenerate "banked turn" that flies the circle with a constant roll/pitch bias. This works in sim where conditions are deterministic, but in real:
- Cannot recover from perturbations that push it to the opposite tilt
- Misses the elevated Gate 2 (max z=1.41 vs gate center z=1.75)
- Any wind gust or tracking error that breaks the banked turn → unrecoverable

**Root cause:** Likely insufficient domain randomization on initial orientation and/or rewards that don't penalize attitude bias. The policy found a local optimum.

### cyclev4: "Altitude Oscillation" Failure

**Symptoms:**
- Max altitude 2.03m (28cm above Gate 2 at 1.75m)
- 28.3% thrust saturation at max PWM
- High Z std (0.535 vs 0.326)
- Inconsistent lap times (2.93s ± 0.26s)
- Limited yaw range (-68.6 to 90.4 deg) — never completes full rotation
- Flight cut short at 27.6s (vs ~49s for v2/v3)
- Higher observation-odom velocity discrepancy (0.219 m/s max vx error)

**Diagnosis:** The policy commands thrust too aggressively, causing altitude overshoots. When above gate height, it saturates thrust at max trying to descend/recover, which causes further oscillation. The limited yaw range suggests it got stuck in a partial loop, possibly entering a spiral climb. Flight was likely terminated by TAs.

**Root cause:** Thrust curve mismatch between sim and real. The policy's thrust commands don't produce the expected force on the real drone, leading to altitude overshoot and saturation. The thrust-vs-altitude scatter plot confirms this: thrust is bimodal, clustering at either 0 N or 1.17 N (max), with almost nothing in between — the policy is "bang-bang" on thrust rather than proportional.

### Visual Confirmation from Plots

**Top-down trajectories:**
- v2: Tight, repeatable circular laps with consistent gate passes right at gate positions
- v3: Equally tight circle but notably, the gate-pass markers cluster asymmetrically — passes near Gate 2 (top of circle) are slightly offset, consistent with the altitude shortfall
- v4: Wider, messier circle with progressive degradation. Later laps swing further out, especially near the top. Gate pass positions scatter more

**Observation plots (gate distance over time):**
- v3: Gate distance shows a very uniform sawtooth pattern (approach→pass→approach), but the amplitude is small (ranging 0.5–2.0m). The gate center body-frame Y component is always positive — confirming the drone only ever sees gates to its left
- v4: Gate distance pattern degrades after ~18s. The sawtooth becomes irregular, with occasional longer approach intervals. After ~22s, the distance peaks get larger, suggesting the drone is drifting further from the gates before each approach
- v2 (reference): Clean sawtooth with full amplitude (0.1–2.4m), and body-frame gate position oscillates symmetrically in X and Y — the drone approaches gates from all directions

**Thrust analysis (v4):**
- The thrust-vs-altitude plot shows a clear "bang-bang" pattern with thrust concentrated at 0 N and ~1.17 N
- At altitudes above 1.5m, thrust is almost exclusively at max (1.17 N) — the policy is trying to climb/hold altitude aggressively
- At lower altitudes, thrust drops to near zero — indicating the policy overcorrects downward too

---

## 8. Key Takeaways for Domain Randomization

Based on comparing the successful v2 flight with the v3/v4 failures:

1. **Thrust curve calibration is critical.** v4's 28% thrust saturation shows the sim-real thrust mapping is off. The real Crazyflie has a non-linear PWM→force curve. Thrust ranges observed: mean 0.65-0.77 N, max 1.17 N, hover ~0.3 N.

2. **Yaw tracking has 2-3x more error** than roll/pitch. Add axis-specific rate response modeling or higher yaw noise in sim (mean tracking error: 26-58 deg/s for yaw vs 18-31 for roll/pitch).

3. **Randomize initial attitude and approach direction** to prevent one-sided policies like v3. The policy must handle the circle in both orientations.

4. **Observation timing jitter:** Odom arrives at ~100 Hz with 3.3ms std jitter. Vicon is clean at 120 Hz. Model the downsampled observation pipeline, not the raw mocap rate.

5. **Altitude control margins:** Gate 2 is at z=1.75m, and the drone reaches z=[1.41, 2.03] across flights. The sim needs altitude randomization in the ±0.3m range around gate heights.

6. **Velocity estimation lag:** v4 shows up to 0.22 m/s error between the observation's velocity and the odom velocity during aggressive maneuvers. Add velocity measurement noise scaled with angular velocity magnitude.

7. **Body rate limits from hardware:**
   - Roll/pitch: capped at ±100 deg/s, actual max observed ~140-158 deg/s (can exceed command due to inertia)
   - Yaw: capped at ±200 deg/s, actual max ~173 deg/s
   - Sim action space should match these limits exactly

8. **Flight envelope for DR ranges:**
   - Position: X ∈ [-1.5, 1.8], Y ∈ [2.3, 6.0], Z ∈ [0.1, 2.0]
   - Speed: 0-4.05 m/s body frame
   - Roll: ±46 deg, Pitch: ±47 deg
   - Thrust: 0-1.17 N (PWM 10001-60000)

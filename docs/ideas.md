### Part 2: Detailed Sim-to-Real Strategy

Based on the papers and the champion’s advice, here is your 5-step roadmap.

#### Step 1: Observation Space (The "Swift" Configuration)
Do not use raw images for the RL policy. Use a **State Vector** (approx. 30-40 dimensions):
*   **Kinematics:** Linear velocity (body frame), angular velocity.
*   **Orientation:** Flattened 3x3 Rotation Matrix (Avoid Euler/Quaternions).
*   **Gate Info:** Relative 3D coordinates of the **4 corners** of the current gate and the **4 corners** of the next gate.
*   **History:** The previous action (thrust/body rates) taken by the network. This helps the network "feel" the latency and momentum.

#### Step 2: Environment & Reset Design (The "Spline Hallway")
1.  **Generate the Spline:** Use a trajectory optimizer to create a smooth path through the gates.
2.  **Weighted Reset:** 
    *   60% of resets: Sample uniformly along the spline.
    *   30% of resets: Sample specifically in the "difficult zones" (sharp turns like the Split-S).
    *   10% of resets: Sample near gate entries (as the champion suggested) to force precision.
3.  **Initialization Noise:** Always spawn with a randomized "kick" in velocity so the drone doesn't start from a perfect, unrealistic state.

#### Step 3: Reward Design (The "Sparse-Simple" Strategy)
Follow the champion’s advice: **Less is More.**
*   **Primary Reward (Sparse):** $+10$ for passing a gate. $+50$ for finishing a lap.
*   **Progress Reward (Dense but simple):** A small reward for moving closer to the next point on your **Spline** (not a straight line).
*   **The "Safety" Penalty (Command Penalty):** Penalize high angular accelerations. This is crucial for sim-to-real. If the actions are too "noisy" in sim, the real motors will overheat or the battery voltage will sag.
*   **Termination:** Reset on crash or if the drone wanders too far (e.g., > 2 meters) from the spline.

#### Step 4: Closing the Reality Gap (System ID)
The "Swift" paper showed that the "Gap" is usually in the **Motors** and **Aerodynamics**, not the AI code.
*   **Thrust Curve:** Measure your real drone's thrust-to-weight ratio. If the real drone has a 4:1 ratio, and your sim has 5:1, the RL agent will be too aggressive and crash in real life.
*   **The Residual Trick:** If you have flight logs (Blackbox) from a real flight:
    1.  Feed the same inputs into your sim.
    2.  Calculate the difference (Residual) between where the real drone went and where the sim drone went.
    3.  Add that difference as a "constant noise/force" in your sim training.

#### Step 5: The Training & Deployment Loop
1.  **Train in Sim:** Use PPO (Proximal Policy Optimization). Train until the success rate is >98% on the spline resets.
2.  **Policy Frequency:** Run your policy at **50Hz or 100Hz**.
3.  **Low-Level Control:** The RL policy should output **Body Rates** ($\text{rad/s}$) and **Thrust**. Feed these into a standard PID controller (Betaflight/PX4). **Do not try to control motor PWM directly with RL.**
4.  **Zero-Shot or Fine-Tune:** If it works in sim but "drifts" in real life, check your **Observation Space** for offsets. (e.g., is your Vicon/MoCap centered exactly where the sim's (0,0,0) is?)

### Summary of the "Champion's Edge":
The secret isn't a complex reward function; it's a **robust reset strategy**. By using the **Spline Reset**, you are essentially "hand-feeding" the agent the best path, but then making it "fight" to stay on that path by adding noise. This creates an agent that is extremely stable on the racing line, which is exactly what you need for a successful real-world run.


### Some other thoughs (if not implemented yet, we can try this): Domain Randomization: Wider and More Aggressive

This is the single most important lever for sim2real transfer. The real world has dynamics that sim doesn't model perfectly. Wider DR forces the policy to be robust.

### Recommended DR ranges (beyond V30)

| Parameter | V30 Range | Sim2Real Range | Rationale |
|-----------|-----------|----------------|-----------|
| Thrust-to-weight | +/-8% | **+/-15%** | Real motor degradation, battery voltage drop |
| Aero drag XY | 0.3-2.5x | **0.2-3.0x** | Ground effect, prop wash in confined space |
| Aero drag Z | 0.3-2.5x | **0.2-3.0x** | Same |
| PID kp, ki (roll/pitch) | +/-25% | **+/-35%** | Real PID may differ significantly |
| PID kd (roll/pitch) | +/-40% | **+/-50%** | Damping varies with wear |
| PID (yaw) | same | **+/-35%/+/-50%** | Yaw authority varies |
| Motor time constant | fixed | **0.5-2.0x nominal** | Real motor response varies |

### New randomizations to add

1. **Observation noise:** Add Gaussian noise to observations during training to simulate Vicon measurement noise and velocity estimation error.
   ```python
   obs += torch.randn_like(obs) * 0.02  # 2% noise on all observations
   ```
   Velocity observations should get more noise (derived quantity):
   ```python
   vel_noise = torch.randn(n, 3) * 0.1  # 0.1 m/s noise on velocity
   ```

2. **Action delay:** Real system has communication latency (radio + processing). Simulate 1-3 step random delay in action execution.

3. **Observation latency:** Vicon has ~10ms latency. Simulate by using observations from 1-2 steps ago with some probability.

4. **Mass variation:** +/-10% of nominal mass.

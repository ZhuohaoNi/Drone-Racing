# Testing Guide

Commands and WandB metrics for verifying each project step.

---

## Step 2: PPO Verification

### Goal

Confirm that the PPO `update()` and GAE `compute_returns()` implementations are correct. With the default starter strategy (hover-to-gate-0), the drone should learn to hover near gate 0 and the WandB curves should match the reference figures from the project description.

### Train (short run, 1000 iterations)

```bash
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \
    --max_iterations 1000 \
    --headless \
    --logger wandb
```

Or use the existing script:

```bash
./scripts/run/train.sh
```

### Evaluate

After training completes, find the run directory in `logs/rsl_rl/quadcopter_direct/`. It will be named like `2026-03-27_14-30-00`. Then:

```bash
python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --load_run <YYYY-MM-DD_HH-MM-SS> \
    --checkpoint best_model.pt \
    --headless \
    --video \
    --video_length 800
```

Or use the existing script:

```bash
./scripts/run/play.sh <YYYY-MM-DD_HH-MM-SS>
```

### What to check in the video

- The drone should hover near gate 0 without drifting away or crashing.
- It should NOT pass through any gates (the starter strategy doesn't incentivize racing).

### What to track on WandB

#### Must match reference behavior

| Metric | Expected Pattern |
|---|---|
| **Episode reward (total)** | Rises and stabilizes. Should match reference Figure 3 from the project description. |
| `Episode_Reward/progress_goal` | Increases as the drone learns to approach gate 0, then plateaus. |

#### PPO health diagnostics

| Metric | What to Look For | Problem If... |
|---|---|---|
| **Surrogate loss** (policy loss) | Small and stable, roughly in [-0.05, 0.05] | Oscillates wildly -> noisy advantages or LR too high |
| **Value loss** | Decreases over training | Stays high or increases -> critic can't fit the value function |
| **Entropy** | Gradually decreases from initial value | Collapses to ~0 early -> premature convergence, increase `entropy_coef` |
| **KL divergence** | Hovers near 0.01 (`desired_kl`) | Repeated spikes -> updates too aggressive |
| **Learning rate** | Adjusts within [1e-5, 1e-2] | Stuck at 1e-5 -> policy unstable. Stuck at 1e-2 -> could use higher initial LR |
| **Mean episode length** | Stabilizes near max (drone hovers the full 30s) | Very short -> drone crashing (check video) |

#### Pass criteria for Step 2

- Total episode reward curve rises and stabilizes (matches reference)
- Value loss decreases over training
- KL divergence stays near 0.01 without sustained spikes
- Video shows stable hover near gate 0

---

## Step 3: Strategy Verification

### Goal

Verify that the racing strategy (rewards, observations, resets) produces a policy that passes gates and completes laps. This is an iterative process — train, check WandB, adjust, repeat.

### Train (longer run, 3000-5000 iterations)

The racing task is much harder than hovering. Start with 3000 iterations and increase if the policy is still improving:

```bash
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \
    --max_iterations 3000 \
    --headless \
    --logger wandb
```

For a full training run (recommended for final submission):

```bash
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \
    --max_iterations 5000 \
    --headless \
    --logger wandb
```

### Evaluate

```bash
python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --load_run <YYYY-MM-DD_HH-MM-SS> \
    --checkpoint best_model.pt \
    --headless \
    --video \
    --video_length 800
```

### What to check in the video

- Does the drone pass through gates in the correct sequence (0 -> 1 -> 2 -> ... -> 6 -> 0)?
- Does it complete at least 1 full lap? Ideally 3 laps.
- Does it handle the powerloop (gates 2-3) and chicane (gates 5-6-0)?
- Does it crash into gate frames or the ground?

### What to track on WandB

#### Reward components (strategy-specific)

| Metric | What to Look For | Problem If... |
|---|---|---|
| `Episode_Reward/progress_goal` | Positive and growing over training | Flat or negative -> drone not approaching gates |
| `Episode_Reward/gate_pass` | Increases in discrete steps over training | Flat -> drone not passing gates. Each visible step = learned to pass one more gate in the sequence |
| `Episode_Reward/crash` | Decreases toward zero | Stays high -> drone colliding frequently, increase crash penalty or add orientation reward |
| **Total episode reward** | Steady upward trend | Plateaus early -> check entropy (PPO issue) or reward design (strategy issue) |

#### Termination stats

| Metric | What to Look For | Problem If... |
|---|---|---|
| `Episode_Termination/died` | Decreasing over training | Stays high -> drone too aggressive or unstable |
| `Episode_Termination/time_out` | Decreasing as drone gets faster | All time-outs, no deaths -> drone is safe but slow, add speed reward |

#### PPO diagnostics (same as Step 2, but watch for new patterns)

| Metric | New Concern for Strategy Training |
|---|---|
| **Entropy** | If it collapses to 0, the policy locked into a suboptimal trajectory (e.g., passes 3 gates but can't learn the rest). Increase `entropy_coef`. |
| **Value loss** | May spike when the policy discovers new gates (reward landscape changes). Should recover. |
| **KL divergence** | Larger spikes expected when the policy makes breakthroughs (new gate passage). Should stay manageable. |

#### Common WandB patterns and fixes

| Pattern | Diagnosis | Fix |
|---|---|---|
| Reward up but `gate_pass` flat | Reward hacking: oscillating near gate without passing | Increase `gate_pass_reward_scale`, reduce `progress_goal_reward_scale` |
| `gate_pass` rises then suddenly drops | Catastrophic forgetting | Reduce learning rate, increase `clip_param` |
| High `died` count throughout | Drone too aggressive | Increase crash penalty, add orientation alignment reward |
| All episodes time out, no deaths | Drone too cautious/slow | Add speed reward, reduce crash penalty |
| `gate_pass` plateaus at 3-4 gates | Can't learn powerloop or chicane | Add look-ahead gates to observations, try weighted gate sampling in resets |
| Entropy near 0, reward plateaued | Premature convergence | Set `entropy_coef = 0.005`, increase `init_noise_std` |

### Iterative tuning workflow

1. **Run 1000 iterations** — Quick check that rewards are shaped correctly. `gate_pass` should start rising by iteration 500.
2. **Run 3000 iterations** — Check if the policy passes all 7 gates in sequence. If `gate_pass` plateaus, adjust reward scales or observations.
3. **Run 5000 iterations** — Full training for submission-quality policy. Check video for 3-lap completion.
4. **Compare runs on WandB** — Use WandB's run comparison to overlay reward curves from different hyperparameter settings side-by-side.

### Pass criteria for Step 3

- `Episode_Reward/gate_pass` visibly increases over training (not flat)
- `Episode_Termination/died` decreases over training
- Video shows the drone passing through multiple gates in sequence
- Ideally: drone completes 3 laps on the powerloop track

---

## Quick Reference: Log Locations

| Artifact | Path |
|---|---|
| Training logs | `logs/rsl_rl/quadcopter_direct/<timestamp>/` |
| Best model | `logs/rsl_rl/quadcopter_direct/<timestamp>/best_model.pt` |
| Evaluation videos | `logs/rsl_rl/quadcopter_direct/<timestamp>/videos/` |
| WandB dashboard | Project `ese651_quadcopter` on wandb.ai |

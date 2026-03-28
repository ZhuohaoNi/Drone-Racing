# Improvement Plan: Metric-Driven Tuning

This document is a decision tree for improving training performance. After each training run, observe the WandB metrics, find the matching symptom below, and apply the prescribed changes. Each section specifies the exact parameter or code location to modify.

---

## How to Use This Document

1. Run a training session (see `docs/testing_guide.md` for commands).
2. Open WandB and examine the metrics listed below.
3. Find the symptom that best matches your curves.
4. Apply the recommended change.
5. Retrain and compare the new run against the old one on WandB.

Changes are organized into three categories:
- **PPO tuning** — Hyperparameters in `rsl_rl_ppo_cfg.py` or logic in `ppo.py`
- **Reward tuning** — Reward scales in `train_race.py` or reward logic in `quadcopter_strategies.py`
- **Strategy tuning** — Observations, resets, or domain randomization in `quadcopter_strategies.py`

---

## Symptom 1: Reward Plateaus Early (iterations < 1000)

### What you see on WandB
- Total episode reward rises initially, then flattens well before `max_iterations`.
- `Episode_Reward/gate_pass` is flat or barely increasing.

### Diagnose further: Check entropy
- **Entropy near 0** -> Symptom 1A (premature convergence)
- **Entropy still healthy (> 0.5)** -> Symptom 1B (strategy ceiling)

### 1A: Premature Convergence

The policy collapsed to a narrow behavior before exploring enough.

| Change | File | Current | Recommended |
|---|---|---|---|
| Enable entropy bonus | `rsl_rl_ppo_cfg.py:30` | `entropy_coef=0.0` | `entropy_coef=0.005` |
| Increase initial noise | `rsl_rl_ppo_cfg.py:20` | `init_noise_std=1.0` | `init_noise_std=1.5` |
| Set minimum std floor | `rsl_rl_ppo_cfg.py:24` | `min_std=0.0` | `min_std=0.01` |

Try `entropy_coef` first. If that alone doesn't help, combine with `init_noise_std`.

### 1B: Strategy Ceiling

PPO is learning fine but the reward/observation design limits what's achievable.

| Change | File | What to Do |
|---|---|---|
| Add speed reward | `train_race.py` + `quadcopter_strategies.py` | Add velocity-toward-gate reward component (see `strategy_design.md` Section 2.4.2) with `speed_reward_scale=10.0` |
| Fix gate normal frame | `quadcopter_strategies.py:173` | Transform gate normal to body frame (see `strategy_design.md` Section 2.4.3) |
| Add gravity vector | `quadcopter_strategies.py` obs | Replace quaternion (4D) with gravity-in-body (3D) (see `strategy_design.md` Section 2.4.4) |
| Add look-ahead gate | `quadcopter_strategies.py` obs | Add third gate in body frame (+3 dims) for better planning |

---

## Symptom 2: Reward Increases but `gate_pass` Stays Flat

### What you see on WandB
- Total reward climbing.
- `Episode_Reward/progress_goal` is the main contributor.
- `Episode_Reward/gate_pass` near zero throughout.

### Diagnosis
Reward hacking. The policy is exploiting the progress reward by oscillating near a gate (getting closer then drifting, collecting progress reward repeatedly) without ever passing through.

### Fixes

| Change | File | Current | Recommended |
|---|---|---|---|
| Increase gate pass bonus | `train_race.py:111` | `gate_pass_reward_scale=200.0` | `gate_pass_reward_scale=500.0` |
| Decrease progress scale | `train_race.py:110` | `progress_goal_reward_scale=50.0` | `progress_goal_reward_scale=20.0` |

The ratio `gate_pass / progress_goal` should be at least 10:1. Currently 4:1. Increasing it ensures passing a gate is always more valuable than hovering near one.

If the drone still doesn't pass gates after scale adjustment, the observation space may not provide enough information. Check that gate positions in body frame are correct (verify `subtract_frame_transforms` is called with the right argument order).

---

## Symptom 3: High Death Rate Throughout Training

### What you see on WandB
- `Episode_Termination/died` stays high (not decreasing after the first ~500 iterations).
- `Episode_Reward/crash` is persistently negative.
- Episodes are short.

### Diagnosis
The drone is crashing frequently — either into gates, the ground, or flying out of bounds.

### Fixes (apply in order)

**Step 1 — Increase crash penalty:**

| Change | File | Current | Recommended |
|---|---|---|---|
| Crash penalty | `train_race.py:112` | `crash_reward=-1.0` | `crash_reward=-5.0` |
| Death cost | `train_race.py:113` | `death_cost=-10.0` | `death_cost=-50.0` |

**Step 2 — If still crashing after penalty increase, add orientation reward:**

Add an alignment reward in `quadcopter_strategies.py` `get_rewards()` to penalize extreme tilt angles (see `strategy_design.md` Section 2.4.6). This makes the drone fly more conservatively.

**Step 3 — If crashing only at specific gates (check video):**

Use weighted gate sampling to give more practice on those gates (see `strategy_design.md` Section 2.4.8).

---

## Symptom 4: `gate_pass` Increases Then Suddenly Drops

### What you see on WandB
- `Episode_Reward/gate_pass` rises for 1000+ iterations, then drops sharply.
- Total reward drops correspondingly.
- May recover, may not.

### Diagnosis
Catastrophic forgetting. A large policy update overwrote previously learned gate-passing behavior.

### Fixes

| Change | File | Current | Recommended |
|---|---|---|---|
| Reduce clip param | `rsl_rl_ppo_cfg.py:29` | `clip_param=0.2` | `clip_param=0.1` |
| Reduce num epochs | `rsl_rl_ppo_cfg.py:31` | `num_learning_epochs=5` | `num_learning_epochs=3` |
| Reduce desired KL | `rsl_rl_ppo_cfg.py:37` | `desired_kl=0.01` | `desired_kl=0.005` |

Apply `clip_param` change first — it's the most direct control on update magnitude. If still happening, reduce epochs.

---

## Symptom 5: KL Divergence Spikes / Learning Rate Stuck at Minimum

### What you see on WandB
- KL divergence frequently exceeds 0.02-0.05.
- Learning rate keeps dropping, eventually stuck at 1e-5.
- Training becomes very slow (reward barely changes).

### Diagnosis
Policy updates are too aggressive. The adaptive LR keeps cutting LR until it bottoms out.

### Fixes

| Change | File | Current | Recommended |
|---|---|---|---|
| Reduce clip param | `rsl_rl_ppo_cfg.py:29` | `clip_param=0.2` | `clip_param=0.15` |
| Reduce initial LR | `rsl_rl_ppo_cfg.py:33` | `learning_rate=5e-4` | `learning_rate=3e-4` |
| Increase mini-batches | `rsl_rl_ppo_cfg.py:32` | `num_mini_batches=4` | `num_mini_batches=8` |
| Widen LR adaptive band | `ppo.py:205-208` | `2.0x / 0.5x desired_kl` | `3.0x / 0.33x desired_kl` |

More mini-batches means smaller batch size per gradient step, which naturally limits the per-step policy change.

---

## Symptom 6: Value Loss Stays High or Increases

### What you see on WandB
- Value loss doesn't decrease, or actively increases over training.
- Reward may still be climbing (the actor is learning despite the critic struggling).

### Diagnosis
The critic cannot fit the value function. Usually caused by reward scale being too large or too variable, or the critic network being undersized.

### Fixes

| Change | File | Current | Recommended |
|---|---|---|---|
| Scale down all rewards | `train_race.py:110-113` | Various | Divide all scales by 5 (progress: 10, gate_pass: 40, crash: -0.2, death: -2.0) |
| Increase value loss coef | `rsl_rl_ppo_cfg.py:27` | `value_loss_coef=1.0` | `value_loss_coef=2.0` |
| Increase critic size | `rsl_rl_ppo_cfg.py:22` | `[512, 256, 128, 128]` | `[512, 512, 256, 128]` |

Try scaling rewards first — it's the easiest change and doesn't affect relative reward ratios.

---

## Symptom 7: Surrogate Loss Oscillates Wildly

### What you see on WandB
- Policy loss swings between large positive and negative values.
- Reward may be unstable.

### Diagnosis
Advantage estimates are noisy, causing erratic gradient signals.

### Fixes

| Change | File | Current | Recommended |
|---|---|---|---|
| Increase rollout length | `rsl_rl_ppo_cfg.py:13` | `num_steps_per_env=24` | `num_steps_per_env=48` |
| Reduce GAE lambda | `rsl_rl_ppo_cfg.py:35` | `lam=0.95` | `lam=0.90` |
| Enable per-mini-batch normalization | `ppo.py` constructor | `normalize_advantage_per_mini_batch=False` | `True` |

Longer rollouts give GAE more data. Lower lambda reduces variance (at the cost of more bias). Per-mini-batch normalization ensures each gradient step sees zero-mean advantages.

---

## Symptom 8: Drone Completes Laps but Too Slowly

### What you see on WandB
- `Episode_Reward/gate_pass` is healthy and increasing.
- `Episode_Termination/time_out` is high (drone uses most of the 30s episode).
- Video shows the drone passing gates but at low speed.

### Diagnosis
No speed incentive. The policy optimizes for safely passing gates, not for doing it quickly.

### Fixes

| Change | File | What to Do |
|---|---|---|
| Add speed reward | `train_race.py` + `quadcopter_strategies.py` | Add velocity-toward-gate reward, `speed_reward_scale=10.0` (see `strategy_design.md` Section 2.4.2) |
| Add initial velocity to resets | `quadcopter_strategies.py` `reset_idx()` | Spawn with 0-3 m/s forward velocity (see `strategy_design.md` Section 2.4.5) |
| Reduce episode length | `quadcopter_env.py:134` | `episode_length_s=30.0` | `episode_length_s=20.0` |

Reducing episode length creates time pressure — the policy must complete laps faster to earn more gate_pass rewards before timeout.

---

## Symptom 9: Policy Works in Training but Fails Under Domain Randomization

### What you see
- Good WandB metrics during training.
- Video from `play_race.py` (which applies domain randomization) shows the drone crashing or drifting.

### Diagnosis
The policy overfit to nominal physics parameters. It can't handle the +/-5% TWR, 0.5-2x drag, or +/-15-30% PID gain changes.

### Fix

Add domain randomization to `quadcopter_strategies.py` `reset_idx()`. See `strategy_design.md` Section 2.4.1 for the exact code. This is the single highest-impact change for evaluation performance.

After adding domain randomization, you may need to:

| Adjustment | Reason |
|---|---|
| Increase `max_iterations` to 5000+ | Policy needs more training to generalize across parameter ranges |
| Slightly increase `progress_goal_reward_scale` | Drone may be more conservative under randomized dynamics |
| Widen spawn distance range | Help policy learn recovery from diverse initial conditions |

---

## Symptom 10: `gate_pass` Plateaus at 3-5 Gates (Can't Learn Full Loop)

### What you see on WandB
- `Episode_Reward/gate_pass` increases to a level corresponding to 3-5 gates, then stops.
- The drone passes some gates but fails at the powerloop (gates 2-3) or chicane (gates 5-6-0).

### Diagnosis
The policy can't learn the harder maneuvers with the current observation/reset design.

### Fixes

| Change | File | What to Do |
|---|---|---|
| Add third look-ahead gate | `quadcopter_strategies.py` obs | +3 dims: body-frame position of gate `idx+2` (see `strategy_design.md` Section 2.4.7) |
| Weighted gate sampling | `quadcopter_strategies.py` `reset_idx()` | 2x weight on gates [2, 3, 5, 6, 0] (see `strategy_design.md` Section 2.4.8) |
| Non-zero initial velocity | `quadcopter_strategies.py` `reset_idx()` | 0-3 m/s forward (see `strategy_design.md` Section 2.4.5) |
| Increase training iterations | CLI flag | `--max_iterations 5000` or higher |

The powerloop requires a vertical loop — the policy needs enough look-ahead to plan the trajectory. The chicane requires rapid alternation — spawning with velocity helps practice these in-flight transitions.

---

## Recommended Improvement Order

For a fresh training run, apply changes in this sequence. After each step, run a training and verify improvement on WandB before proceeding to the next.

| Step | Changes | Files Modified | Verify By |
|---|---|---|---|
| 1 | `entropy_coef=0.005` | `rsl_rl_ppo_cfg.py` | Entropy stays above 0 throughout training |
| 2 | Domain randomization in `reset_idx()` | `quadcopter_strategies.py` | Policy survives `play_race.py` with randomized dynamics |
| 3 | Gate normal to body frame + gravity vector | `quadcopter_strategies.py` | Faster reward growth in early iterations vs. previous run |
| 4 | Speed reward (`scale=10.0`) | `train_race.py` + `quadcopter_strategies.py` | `Episode_Termination/time_out` decreases; shorter episodes |
| 5 | Non-zero initial velocity in resets | `quadcopter_strategies.py` | Fewer crashes at gate passage (drone practiced in-flight) |
| 6 | Third look-ahead gate | `quadcopter_strategies.py` | `gate_pass` plateau breaks past 5+ gates |
| 7 | Reward scale tuning based on WandB | `train_race.py` | Ratio adjustments per Symptom 2 or 3 |
| 8 | PPO hyperparameter tuning | `rsl_rl_ppo_cfg.py` | Per symptoms 4-7 as they arise |

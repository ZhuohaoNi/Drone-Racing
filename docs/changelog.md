# Changelog

All notable changes to this project will be documented in this file.

## [V1] - 2026-03-27

### Added
- **Adaptive LR schedule** in `ppo.py`: Computes approximate KL divergence per update. If KL > 2×desired_kl, LR is halved; if KL < desired_kl/2, LR is doubled (clamped to [1e-5, 1e-2]). Fixes training instability in later iterations.
- **23-dim ego-centric observation space** in `quadcopter_strategies.py`:
  - Body-frame linear velocity (3), angular velocity (3), quaternion (4)
  - Current gate position in body frame (3), next gate position in body frame (3)
  - Current gate normal vector (3), previous actions (4)
  - World-frame position intentionally excluded for ego-centric generalization
- **Gate-pass detection** using x-component sign change in gate frame (positive→negative = correct passage) with y/z bounds checking to ensure passage through the gate opening.
- **Reward structure**: progress toward gate (distance reduction), gate-pass bonus (200.0 scale), crash penalty (-1.0 scale), death cost (-10.0).
- **Randomized reset strategy**: Random starting gate, spawn 1.5-3m behind gate with ±0.5m lateral, ±0.3m vertical, and ±0.15 rad yaw noise. Altitude clamped above 0.15m.
- Added `gate_pass_reward_scale` to `train_race.py` reward scales.

## [Unreleased]

### Added
- Created this changelog file to track project modifications.
- Implemented the PPO algorithm update step in `ppo.py`, including surrogate loss, clipped value loss, entropy bonus, and model optimization.
- Created convenient bash scripts (`scripts/run/train.sh` and `scripts/run/play.sh`) to easily run training and evaluation.

### Changed
- Refactored `compute_returns()` in `rollout_storage.py` to use a GPU-optimized vectorized calculation for generalized advantage estimation (GAE). By computing the TD errors (deltas) in parallel across the sequence, PyTorch kernel launch overhead is minimized during loop iteration.

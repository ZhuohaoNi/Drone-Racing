"""
controller_simple_policy.py — drop-in replacement that runs a
system-identification (sysid) reference player instead of a neural-network
racing policy.

This single file is all that needs to be swapped in. The surrounding
ControllerNode, FSM, services, and topics are UNCHANGED. To run sysid:

    preferred: take off -> hover -> call `/race` service (same as before)
    supported: call `/race` from the ground; the policy ramps to hover first
    call `/stop` to land

Interface preserved exactly:
- Class name: `SimpleRacingPolicy`
- Constructor: `SimpleRacingPolicy(vehicle, model_path, params, device="cpu", **kw)`
- `.update(state) -> (control, obs)` where
      control = {'cmd_thrust': float in [0,1], 'cmd_w': np.array([rx,ry,rz]) deg/s}
      obs     = 40-D vector sliceable as [lin_vel(3) | rot(9) | curr(12) | next(12) | prev(4)]
- `.idx_wp` attribute (int) so the node's lap-counter attribute access stays valid

Data we produce for the rosbag (no new topic, no new msg required):
- `/ctbr_cmd`: thrust + body-rate commands that actually drove the drone.
- `/observations`:
    lin_vel[3]            = actual body lin vel (mocap-derived)
    rot[9]                = actual rotation matrix
    corners_pos_b_curr[12]= commanded flat output (x, x_dot, x_ddot, yaw[1..3])
    corners_pos_b_next[12]= packed phase info:
        [0] dim_code  (1=mass 2=vert 3=roll 4=pitch 5=yaw 6=drag
                       7=latency 8=gate_bias 9=aux 0=idle)
        [1] detail_level
        [2] t_in_phase_s
        [3] phase_duration_s
        [4] t_since_sysid_start_s
        [5] round_index
        [6] phases_emitted
        [7] aborted_flag (0/1)
        [8..11] reserved
    cond[2]               = [sysid_active_flag, phase_id_counter]
  This is *enough* to segment phases offline and to correlate every
  command with its ground-truth reference state.

Design (multi-pass round-robin):
Round 1 (~50-60s): minimum-useful probe on EVERY dimension — so flights cut
    short still yield signal on mass, vertical, roll, pitch, yaw, drag,
    latency, gate-corner bias.
Round 2 (~180s): refined probes (longer chirps, more straight-line segments,
    all gates).
Round 3 (~300s): fine-detail probes.
After round 3: indefinite drift-tracking loop (hover + vertical sin).

All phases produce SE3 flat outputs and go through the same SE3ControlCTBR
attitude stabilization the TA's code uses for hover/flying — we never bypass
the attitude loop, even for rate-excitation phases (those use lateral
position chirps that force the rate loop through the target frequencies).

Safety:
- Commanded positions clamped to a box around the sysid center.
- Phase step functions wrapped in try/except with NaN sanitization.
- Extreme tilt / NaN mocap / missing pose keys -> hold last good flat output.
- Any exception in the policy path -> hover fallback, preferring the sysid
  center if available.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import torch  # type: ignore
except ImportError:  # bag-processing env may not have torch; sysid doesn't need it
    torch = None  # type: ignore
from scipy.spatial.transform import Rotation as R  # noqa: F401  (kept for parity)


# =============================================================================
# constants
# =============================================================================

_G = 9.80665
_DEG = math.pi / 180.0
_EPS = 1e-9

# Must match the scaling used in controller_utils.single_update:
#   thrust_des_newtons = thrust_des_perc * (0.038 * 9.81 * 3.15)
_THRUST_PERC_SCALE_N = 0.038 * 9.81 * 3.15  # ≈ 1.1744 N

# Dimension codes, stored in obs[24] for offline phase segmentation.
_DIM_CODE = {
    "": 0, "idle": 0, "aux": 9,
    "mass": 1, "vert": 2, "roll": 3, "pitch": 4, "yaw": 5,
    "drag": 6, "latency": 7, "gate_bias": 8,
}


# =============================================================================
# helpers
# =============================================================================

def _zeros3() -> np.ndarray:
    return np.zeros(3, dtype=np.float64)


def _flat_hover(pos: np.ndarray, yaw: float) -> Dict[str, Any]:
    """Flat output for a stationary hover at (pos, yaw)."""
    return {
        "x": np.asarray(pos, dtype=np.float64).copy(),
        "x_dot": _zeros3(),
        "x_ddot": _zeros3(),
        "x_dddot": _zeros3(),
        "x_ddddot": _zeros3(),
        "yaw": float(yaw),
        "yaw_dot": 0.0,
        "yaw_ddot": 0.0,
    }


def _sanitize_flat(flat: Dict[str, Any], fallback_pos: np.ndarray,
                   fallback_yaw: float) -> Dict[str, Any]:
    """Replace any NaN/Inf in a flat output with the safe hover value."""
    safe = _flat_hover(fallback_pos, fallback_yaw)
    for k, v in flat.items():
        if isinstance(v, np.ndarray):
            if not np.all(np.isfinite(v)):
                safe[k] = safe.get(k, _zeros3())
            else:
                safe[k] = v
        else:
            try:
                fv = float(v)
            except Exception:
                safe[k] = safe.get(k, 0.0)
                continue
            if not math.isfinite(fv):
                safe[k] = safe.get(k, 0.0)
            else:
                safe[k] = fv
    return safe


def _clamp_position(flat: Dict[str, Any], center: np.ndarray,
                    box_half: np.ndarray) -> Dict[str, Any]:
    """Clamp flat['x'] to center ± box_half. If clamped on an axis, zero
    that axis's higher-order derivatives so we don't claim to be moving."""
    x = flat["x"]
    lower = center - box_half
    upper = center + box_half
    clamped = np.clip(x, lower, upper)
    if np.any(clamped != x):
        flat["x"] = clamped
        for key in ("x_dot", "x_ddot", "x_dddot", "x_ddddot"):
            d = flat[key].copy()
            for ax in range(3):
                if clamped[ax] != x[ax]:
                    d[ax] = 0.0
            flat[key] = d
    return flat


# =============================================================================
# phases
# =============================================================================

class Phase:
    """Base class. Subclasses implement `_step(t_rel)`. `step()` wraps it
    with NaN sanitization so a bug in any phase cannot emit a bad command."""

    def __init__(self, name: str, duration: float, dim: str = "", detail: int = 0,
                 meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.duration = float(max(0.1, duration))
        self.dim = dim
        self.detail = int(detail)
        self.meta = dict(meta) if meta else {}
        self.center_pos: np.ndarray = _zeros3()
        self.center_yaw: float = 0.0

    def bind(self, center_pos: np.ndarray, center_yaw: float) -> None:
        self.center_pos = np.asarray(center_pos, dtype=np.float64).copy()
        self.center_yaw = float(center_yaw)

    def step(self, t_rel: float) -> Dict[str, Any]:
        t = float(max(0.0, min(self.duration, t_rel)))
        try:
            flat = self._step(t)
        except Exception:
            flat = _flat_hover(self.center_pos, self.center_yaw)
        return _sanitize_flat(flat, self.center_pos, self.center_yaw)

    def _step(self, t: float) -> Dict[str, Any]:
        return _flat_hover(self.center_pos, self.center_yaw)

    def info(self, t_rel: float) -> Dict[str, Any]:
        return {
            "phase": self.name, "dim": self.dim, "detail": self.detail,
            "t": float(t_rel), "dur": self.duration, **self.meta,
        }

    def desired_start_pose(self, center_pos: np.ndarray,
                           center_yaw: float) -> Tuple[np.ndarray, float]:
        return np.asarray(center_pos, dtype=np.float64).copy(), float(center_yaw)


class HoverPhase(Phase):
    pass


class TakeoffRampPhase(Phase):
    """Z-only min-jerk (quintic) ramp from a ground pose up to a hover pose.

    Used once at the start of a sysid run so the operator can call /race from
    the ground (SOP) without the scheduler pinning `center_pos` to the floor.

    `bind()` is NOT used to set start_pos — call `bind_from_ground()` instead
    so we keep the ground z separately from the hover center. The per-step
    output keeps x/y at the ground pose and linearly ramps z via quintic s.
    """

    def __init__(self, duration: float = 3.0):
        super().__init__("takeoff_ramp", duration=float(max(0.5, duration)),
                         dim="aux", detail=0)
        self._start_pos: np.ndarray = _zeros3()
        self._start_yaw: float = 0.0

    def bind_from_ground(self, ground_pos: np.ndarray, ground_yaw: float,
                         hover_pos: np.ndarray, hover_yaw: float) -> None:
        self._start_pos = np.asarray(ground_pos, dtype=np.float64).copy()
        self._start_yaw = float(ground_yaw)
        self.center_pos = np.asarray(hover_pos, dtype=np.float64).copy()
        dy = (float(hover_yaw) - self._start_yaw + math.pi) % (2 * math.pi) - math.pi
        self.center_yaw = self._start_yaw + dy

    def _step(self, t: float) -> Dict[str, Any]:
        T = self.duration
        s = min(1.0, max(0.0, t / T))
        p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        pd = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
        pdd = (60 * s - 180 * s ** 2 + 120 * s ** 3) / (T ** 2)
        pddd = (60 - 360 * s + 360 * s ** 2) / (T ** 3)
        pdddd = (-360 + 720 * s) / (T ** 4)
        delta = self.center_pos - self._start_pos
        x = self._start_pos + delta * p
        dyaw = self.center_yaw - self._start_yaw
        return {
            "x": x, "x_dot": delta * pd, "x_ddot": delta * pdd,
            "x_dddot": delta * pddd, "x_ddddot": delta * pdddd,
            "yaw": self._start_yaw + dyaw * p,
            "yaw_dot": dyaw * pd, "yaw_ddot": dyaw * pdd,
        }


class ReturnToCenterPhase(Phase):
    """Minimum-jerk (quintic) homing motion to a target pose. Duration is
    adaptive -- long enough to cover the distance at the configured max
    speed, with a floor and ceiling."""

    def __init__(self, min_duration: float = 1.2, max_speed: float = 0.7):
        super().__init__("return_to_center", duration=min_duration, dim="aux")
        self.min_duration = float(min_duration)
        self.max_speed = float(max_speed)
        self._start_pos: np.ndarray = _zeros3()
        self._start_yaw: float = 0.0

    def bind_from_state(self, start_pos: np.ndarray, start_yaw: float,
                        center_pos: np.ndarray, center_yaw: float) -> None:
        self._start_pos = np.asarray(start_pos, dtype=np.float64).copy()
        self._start_yaw = float(start_yaw)
        self.center_pos = np.asarray(center_pos, dtype=np.float64).copy()
        dy = (float(center_yaw) - self._start_yaw + math.pi) % (2 * math.pi) - math.pi
        self.center_yaw = self._start_yaw + dy
        dist = float(np.linalg.norm(self.center_pos - self._start_pos))
        t_dist = dist / max(self.max_speed, 0.05)
        self.duration = float(max(self.min_duration, min(6.0, t_dist * 1.5)))

    def _step(self, t: float) -> Dict[str, Any]:
        T = self.duration
        s = min(1.0, max(0.0, t / T))
        p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        pd = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
        pdd = (60 * s - 180 * s ** 2 + 120 * s ** 3) / (T ** 2)
        pddd = (60 - 360 * s + 360 * s ** 2) / (T ** 3)
        pdddd = (-360 + 720 * s) / (T ** 4)
        delta = self.center_pos - self._start_pos
        x = self._start_pos + delta * p
        dyaw = self.center_yaw - self._start_yaw
        return {
            "x": x, "x_dot": delta * pd, "x_ddot": delta * pdd,
            "x_dddot": delta * pddd, "x_ddddot": delta * pdddd,
            "yaw": self._start_yaw + dyaw * p,
            "yaw_dot": dyaw * pd, "yaw_ddot": dyaw * pdd,
        }


class VerticalSinePhase(Phase):
    def __init__(self, amp: float, freq: float, cycles: int, dim: str = "vert",
                 detail: int = 0):
        freq = float(max(0.05, min(3.0, freq)))
        amp = float(max(0.02, min(0.4, amp)))
        duration = cycles / freq
        super().__init__(f"vert_sin_{freq:.2f}Hz_{amp:.2f}m", duration,
                         dim=dim, detail=detail,
                         meta={"amp_m": amp, "freq_hz": freq, "cycles": cycles})
        self.amp = amp
        self.w = 2 * math.pi * freq

    def _step(self, t: float) -> Dict[str, Any]:
        w = self.w; A = self.amp
        s = math.sin(w * t); c = math.cos(w * t)
        x = self.center_pos.copy(); xd = _zeros3(); xdd = _zeros3()
        xddd = _zeros3(); xdddd = _zeros3()
        x[2] += A * s
        xd[2] = A * w * c
        xdd[2] = -A * w * w * s
        xddd[2] = -A * w ** 3 * c
        xdddd[2] = A * w ** 4 * s
        return {"x": x, "x_dot": xd, "x_ddot": xdd, "x_dddot": xddd,
                "x_ddddot": xdddd, "yaw": self.center_yaw,
                "yaw_dot": 0.0, "yaw_ddot": 0.0}


class VerticalChirpPhase(Phase):
    """Linear frequency chirp in z."""

    def __init__(self, amp: float, f0: float, f1: float, duration: float,
                 dim: str = "vert", detail: int = 0):
        amp = float(max(0.02, min(0.3, amp)))
        f0 = float(max(0.05, f0))
        f1 = float(max(f0 + 0.05, f1))
        duration = float(max(2.0, duration))
        super().__init__(f"vert_chirp_{f0:.1f}_{f1:.1f}_{amp:.2f}m", duration,
                         dim=dim, detail=detail,
                         meta={"amp_m": amp, "f0_hz": f0, "f1_hz": f1})
        self.amp = amp; self.f0 = f0; self.f1 = f1
        self.k = (f1 - f0) / duration

    def _step(self, t: float) -> Dict[str, Any]:
        A = self.amp
        phi = 2 * math.pi * (self.f0 * t + 0.5 * self.k * t * t)
        wf = 2 * math.pi * (self.f0 + self.k * t)
        wdot = 2 * math.pi * self.k
        s = math.sin(phi); c = math.cos(phi)
        x = self.center_pos.copy(); xd = _zeros3(); xdd = _zeros3()
        xddd = _zeros3(); xdddd = _zeros3()
        x[2] += A * s
        xd[2] = A * wf * c
        xdd[2] = A * (wdot * c - wf * wf * s)
        xddd[2] = A * (-3 * wf * wdot * s - wf ** 3 * c)
        xdddd[2] = A * (-6 * wf * wf * wdot * c - 3 * wdot * wdot * s + wf ** 4 * s)
        return {"x": x, "x_dot": xd, "x_ddot": xdd, "x_dddot": xddd,
                "x_ddddot": xdddd, "yaw": self.center_yaw,
                "yaw_dot": 0.0, "yaw_ddot": 0.0}


class LateralOscPhase(Phase):
    """Lateral chirp/osc in x or y to excite pitch/roll dynamics."""

    def __init__(self, axis: str, amp: float, f0: float, f1: float, duration: float,
                 dim: str = "", detail: int = 0):
        assert axis in ("x", "y")
        amp = float(max(0.02, min(0.25, amp)))
        f0 = float(max(0.1, f0))
        f1 = float(max(f0, f1))
        duration = float(max(1.5, duration))
        super().__init__(f"lat_{axis}_{f0:.1f}_{f1:.1f}_{amp:.2f}m", duration,
                         dim=dim or ("roll" if axis == "y" else "pitch"),
                         detail=detail,
                         meta={"axis": axis, "amp_m": amp, "f0_hz": f0, "f1_hz": f1})
        self.axis_idx = 0 if axis == "x" else 1
        self.amp = amp; self.f0 = f0; self.f1 = f1
        self.k = (f1 - f0) / duration if f1 > f0 else 0.0

    def _step(self, t: float) -> Dict[str, Any]:
        A = self.amp
        phi = 2 * math.pi * (self.f0 * t + 0.5 * self.k * t * t)
        wf = 2 * math.pi * (self.f0 + self.k * t)
        wdot = 2 * math.pi * self.k
        s = math.sin(phi); c = math.cos(phi)
        x = self.center_pos.copy(); xd = _zeros3(); xdd = _zeros3()
        xddd = _zeros3(); xdddd = _zeros3()
        ax = self.axis_idx
        x[ax] += A * s
        xd[ax] = A * wf * c
        xdd[ax] = A * (wdot * c - wf * wf * s)
        xddd[ax] = A * (-3 * wf * wdot * s - wf ** 3 * c)
        xdddd[ax] = A * (-6 * wf * wf * wdot * c - 3 * wdot * wdot * s + wf ** 4 * s)
        return {"x": x, "x_dot": xd, "x_ddot": xdd, "x_dddot": xddd,
                "x_ddddot": xdddd, "yaw": self.center_yaw,
                "yaw_dot": 0.0, "yaw_ddot": 0.0}


class YawChirpPhase(Phase):
    def __init__(self, amp_deg: float, f0: float, f1: float, duration: float,
                 detail: int = 0):
        amp_deg = float(max(2.0, min(45.0, amp_deg)))
        f0 = float(max(0.2, f0))
        f1 = float(max(f0, f1))
        duration = float(max(1.5, duration))
        super().__init__(f"yaw_chirp_{f0:.1f}_{f1:.1f}_{amp_deg:.0f}deg", duration,
                         dim="yaw", detail=detail,
                         meta={"amp_deg": amp_deg, "f0_hz": f0, "f1_hz": f1})
        self.A = amp_deg * _DEG
        self.f0 = f0; self.f1 = f1
        self.k = (f1 - f0) / duration if f1 > f0 else 0.0

    def _step(self, t: float) -> Dict[str, Any]:
        phi = 2 * math.pi * (self.f0 * t + 0.5 * self.k * t * t)
        wf = 2 * math.pi * (self.f0 + self.k * t)
        wdot = 2 * math.pi * self.k
        s = math.sin(phi); c = math.cos(phi)
        yaw = self.center_yaw + self.A * s
        return {"x": self.center_pos.copy(), "x_dot": _zeros3(),
                "x_ddot": _zeros3(), "x_dddot": _zeros3(),
                "x_ddddot": _zeros3(), "yaw": yaw,
                "yaw_dot": self.A * wf * c,
                "yaw_ddot": self.A * (wdot * c - wf * wf * s)}


class RateDoubletPhase(Phase):
    """Single-cycle sinusoidal 'doublet' that excites a targeted peak body-rate
    on one axis. A full sine cycle of period T leaves position, velocity,
    attitude and rate all at zero at the end, so it is safe to chain.

    For lateral axes the body-rate from SE3 feedforward is approximately
    jerk/g, so amplitude is sized to produce the requested peak rate:
        axis='y' -> peak roll-rate  ≈ target_dps  (lateral y motion)
        axis='x' -> peak pitch-rate ≈ target_dps  (lateral x motion)
        axis='yaw' -> peak yaw-rate = target_dps (direct yaw command)

    A short hover settle is appended so the SE3 tracker can recover before the
    next phase begins.
    """

    def __init__(self, axis: str, target_dps: float, period: float = 1.2,
                 settle: float = 0.6, detail: int = 0):
        assert axis in ("x", "y", "yaw")
        target_dps = float(max(5.0, min(120.0, abs(target_dps))))
        period = float(max(0.4, min(3.0, period)))
        settle = float(max(0.2, settle))
        dim = {"y": "roll", "x": "pitch", "yaw": "yaw"}[axis]
        name = f"rate_doublet_{axis}_{int(target_dps):d}dps_T{period:.2f}"
        super().__init__(name, period + settle, dim=dim, detail=detail,
                         meta={"axis": axis, "target_dps": target_dps,
                               "period_s": period})
        self.axis = axis
        self.T = period
        self.w = 2.0 * math.pi / period
        target_rate = target_dps * _DEG
        if axis == "yaw":
            self.A = target_rate / self.w
        else:
            self.A = target_rate * _G / (self.w ** 3)

    def _step(self, t: float) -> Dict[str, Any]:
        x = self.center_pos.copy(); xd = _zeros3(); xdd = _zeros3()
        xddd = _zeros3(); xdddd = _zeros3()
        yaw = self.center_yaw; yaw_dot = 0.0; yaw_ddot = 0.0
        if t < self.T:
            w = self.w; A = self.A
            s = math.sin(w * t); c = math.cos(w * t)
            if self.axis == "yaw":
                yaw = self.center_yaw + A * s
                yaw_dot = A * w * c
                yaw_ddot = -A * w * w * s
            else:
                ax = 0 if self.axis == "x" else 1
                x[ax] += A * s
                xd[ax] = A * w * c
                xdd[ax] = -A * w * w * s
                xddd[ax] = -A * w ** 3 * c
                xdddd[ax] = A * w ** 4 * s
        return {"x": x, "x_dot": xd, "x_ddot": xdd, "x_dddot": xddd,
                "x_ddddot": xdddd, "yaw": yaw,
                "yaw_dot": yaw_dot, "yaw_ddot": yaw_ddot}


class StraightLinePhase(Phase):
    """Constant-velocity straight-line segment with raised-cosine accel/decel."""

    def __init__(self, direction: np.ndarray, v_max: float,
                 cruise_duration: float, t_ramp: float = 0.6, detail: int = 0):
        d = np.asarray(direction, dtype=np.float64)
        n = float(np.linalg.norm(d))
        d = d / n if n >= _EPS else np.array([1.0, 0.0, 0.0])
        v_max = float(max(0.1, min(3.0, v_max)))
        t_ramp = float(max(0.2, min(2.0, t_ramp)))
        cruise_duration = float(max(0.5, cruise_duration))
        duration = 2 * t_ramp + cruise_duration
        super().__init__(f"straight_v{v_max:.1f}", duration, dim="drag",
                         detail=detail,
                         meta={"direction": d.tolist(), "v_max": v_max, "t_ramp": t_ramp})
        self.d = d; self.v_max = v_max; self.t_ramp = t_ramp
        self.t_cruise = cruise_duration

    def _profile(self, t: float) -> Tuple[float, float, float, float, float]:
        tr = self.t_ramp; tc = self.t_cruise; v_max = self.v_max
        if t <= tr:
            tau = t / tr
            v = 0.5 * v_max * (1 - math.cos(math.pi * tau))
            a = 0.5 * v_max * (math.pi / tr) * math.sin(math.pi * tau)
            j = 0.5 * v_max * (math.pi / tr) ** 2 * math.cos(math.pi * tau)
            snap = -0.5 * v_max * (math.pi / tr) ** 3 * math.sin(math.pi * tau)
            s = 0.5 * v_max * t - 0.5 * v_max * (tr / math.pi) * math.sin(math.pi * tau)
        elif t <= tr + tc:
            dt = t - tr
            v = v_max; a = 0.0; j = 0.0; snap = 0.0
            s = 0.5 * v_max * tr + v_max * dt
        elif t <= 2 * tr + tc:
            dt = t - tr - tc; tau = dt / tr
            v = 0.5 * v_max * (1 + math.cos(math.pi * tau))
            a = -0.5 * v_max * (math.pi / tr) * math.sin(math.pi * tau)
            j = -0.5 * v_max * (math.pi / tr) ** 2 * math.cos(math.pi * tau)
            snap = 0.5 * v_max * (math.pi / tr) ** 3 * math.sin(math.pi * tau)
            s = 0.5 * v_max * tr + v_max * tc + 0.5 * v_max * dt \
                + 0.5 * v_max * (tr / math.pi) * math.sin(math.pi * tau)
        else:
            s = 2 * (0.5 * v_max * tr) + v_max * tc
            v = 0.0; a = 0.0; j = 0.0; snap = 0.0
        return s, v, a, j, snap

    def _step(self, t: float) -> Dict[str, Any]:
        s, v, a, j, snap = self._profile(t)
        return {"x": self.center_pos + self.d * s,
                "x_dot": self.d * v, "x_ddot": self.d * a,
                "x_dddot": self.d * j, "x_ddddot": self.d * snap,
                "yaw": self.center_yaw, "yaw_dot": 0.0, "yaw_ddot": 0.0}


class ThrustStepPhase(Phase):
    """Smooth vertical step reference, N reps. Used for command-latency probe."""

    def __init__(self, amp: float = 0.12, n_reps: int = 4, pulse_width: float = 0.35,
                 rest: float = 0.35, rise: float = 0.05, detail: int = 0):
        amp = float(max(0.05, min(0.25, amp)))
        n_reps = int(max(1, n_reps))
        pulse_width = float(max(0.1, pulse_width))
        rest = float(max(0.2, rest))
        rise = float(max(0.02, min(0.15, rise)))
        duration = n_reps * (2 * rise + pulse_width + rest)
        super().__init__(f"thrust_step_{amp:.2f}m_x{n_reps}", duration,
                         dim="latency", detail=detail,
                         meta={"amp_m": amp, "reps": n_reps,
                               "pulse_s": pulse_width, "rise_s": rise})
        self.amp = amp; self.n_reps = n_reps
        self.pulse = pulse_width; self.rest = rest; self.rise = rise

    def _one_rep_dz(self, t: float) -> Tuple[float, float, float]:
        A = self.amp; r = self.rise; p = self.pulse
        if t < 0:
            return 0.0, 0.0, 0.0
        if t <= r:
            s = 0.5 * (1 - math.cos(math.pi * t / r))
            sd = 0.5 * (math.pi / r) * math.sin(math.pi * t / r)
            sdd = 0.5 * (math.pi / r) ** 2 * math.cos(math.pi * t / r)
            return A * s, A * sd, A * sdd
        if t <= r + p:
            return A, 0.0, 0.0
        if t <= 2 * r + p:
            tau = (t - r - p) / r
            s = 0.5 * (1 + math.cos(math.pi * tau))
            sd = -0.5 * (math.pi / r) * math.sin(math.pi * tau)
            sdd = -0.5 * (math.pi / r) ** 2 * math.cos(math.pi * tau)
            return A * s, A * sd, A * sdd
        return 0.0, 0.0, 0.0

    def _step(self, t: float) -> Dict[str, Any]:
        period = 2 * self.rise + self.pulse + self.rest
        rep_i = int(t // period)
        if rep_i >= self.n_reps:
            z, zd, zdd = 0.0, 0.0, 0.0
        else:
            z, zd, zdd = self._one_rep_dz(t - rep_i * period)
        x = self.center_pos.copy(); x[2] += z
        xd = _zeros3(); xd[2] = zd
        xdd = _zeros3(); xdd[2] = zdd
        return {"x": x, "x_dot": xd, "x_ddot": xdd,
                "x_dddot": _zeros3(), "x_ddddot": _zeros3(),
                "yaw": self.center_yaw, "yaw_dot": 0.0, "yaw_ddot": 0.0}


class GateFlybyPhase(Phase):
    """Straight-line flyby through a gate center with yaw aligned to the
    gate's normal. The pre-gate approach point is used as the phase start
    pose so the scheduler routes the drone there before the phase begins."""

    def __init__(self, gate_pos: np.ndarray, gate_yaw: float, approach: float = 1.0,
                 v_fly: float = 1.2, detail: int = 0, gate_idx: int = 0):
        gate_pos = np.asarray(gate_pos, dtype=np.float64)
        approach = float(max(0.5, approach))
        v_fly = float(max(0.3, min(2.0, v_fly)))
        c = math.cos(gate_yaw); s = math.sin(gate_yaw)
        n = np.array([c, s, 0.0], dtype=np.float64)
        self._pre = gate_pos - n * approach
        self._post = gate_pos + n * approach
        t_ramp = 0.6
        t_cruise = max(0.4, (2 * approach) / v_fly - 2 * t_ramp)
        duration = 2 * t_ramp + t_cruise
        super().__init__(f"gate_flyby_g{gate_idx}_v{v_fly:.1f}", duration,
                         dim="gate_bias", detail=detail,
                         meta={"gate_idx": gate_idx,
                               "gate_pos": gate_pos.tolist(),
                               "gate_yaw": gate_yaw, "v_fly": v_fly})
        self._gate_yaw = float(gate_yaw)
        self._line = StraightLinePhase(direction=n, v_max=v_fly,
                                       cruise_duration=t_cruise, t_ramp=t_ramp)

    def bind(self, center_pos: np.ndarray, center_yaw: float) -> None:  # type: ignore[override]
        super().bind(center_pos, center_yaw)
        self._line.bind(self._pre, self._gate_yaw)

    def _step(self, t: float) -> Dict[str, Any]:
        flat = self._line._step(t)
        flat["yaw"] = self._gate_yaw
        flat["yaw_dot"] = 0.0
        flat["yaw_ddot"] = 0.0
        return flat

    def desired_start_pose(self, center_pos: np.ndarray,
                           center_yaw: float) -> Tuple[np.ndarray, float]:
        return self._pre.copy(), self._gate_yaw


# =============================================================================
# scheduler
# =============================================================================

@dataclass
class SysIDConfig:
    safety_box_half: Tuple[float, float, float] = (1.2, 1.2, 0.5)
    max_tilt_deg: float = 30.0
    pose_err_start_thresh_m: float = 0.18
    mocap_stale_s: float = 0.2
    return_min_duration: float = 1.2
    return_max_speed: float = 0.7
    hover_settle_s: float = 0.6
    # Autonomous takeoff: /race is called from the ground (SOP), so before the
    # first probe the scheduler ramps the drone from its current pose up to a
    # nominal hover altitude. The hover pose then becomes the sysid center.
    hover_altitude_m: float = 1.0
    takeoff_duration_s: float = 3.0
    takeoff_ready_z_tol_m: float = 0.20
    gates: List[Tuple[Tuple[float, float, float], float]] = field(default_factory=list)
    loop_drift: bool = True


class SysIDController:
    """Multi-pass, round-robin system-identification scheduler.

    Public API (used by SimpleRacingPolicy):
        start(center_pos, center_yaw, t_wall)
        update(mocap_pose, t_wall) -> dict with flat_output / info / aborted
        stop()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 gates: Optional[List[Tuple[Tuple[float, float, float], float]]] = None):
        self.cfg = SysIDConfig()
        if config:
            for k, v in config.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
        if gates:
            self.cfg.gates = list(gates)
        self.box_half = np.asarray(self.cfg.safety_box_half, dtype=np.float64)
        self.max_tilt_rad = float(self.cfg.max_tilt_deg) * _DEG

        self._started = False
        self._aborted = False
        self._abort_reason = ""
        self._done = False
        self.center_pos: np.ndarray = _zeros3()
        self.center_yaw: float = 0.0
        self._t_start_wall = 0.0
        self._phase: Optional[Phase] = None
        self._phase_t_start = 0.0
        self._last_good_flat: Optional[Dict[str, Any]] = None
        self._probe_queue: List[Phase] = []
        self._rounds_completed = 0
        self._phases_emitted = 0
        self._pending_probe: Optional[Phase] = None
        self._pending_probe_start_pos: np.ndarray = _zeros3()
        self._pending_probe_start_yaw = 0.0
        self._pending_probe_after_settle = False
        self._ground_pos: np.ndarray = _zeros3()
        self._ground_yaw: float = 0.0
        self._takeoff_done = False

    def start(self, center_pos: np.ndarray, center_yaw: float,
              t_wall: float = 0.0) -> None:
        # The caller passes in the *current* (ground) pose because SOP is to
        # call /race from the ground. We save it as the takeoff start and lift
        # the sysid hover center to the configured hover altitude.
        ground = np.asarray(center_pos, dtype=np.float64).copy()
        self._ground_pos = ground.copy()
        self._ground_yaw = float(center_yaw)
        hover = ground.copy()
        hover[2] = float(self.cfg.hover_altitude_m)
        self.center_pos = hover
        self.center_yaw = float(center_yaw)
        self._t_start_wall = float(t_wall)
        self._started = True
        self._aborted = False
        self._abort_reason = ""
        self._done = False
        self._phase = None
        self._phase_t_start = 0.0
        self._phases_emitted = 0
        self._rounds_completed = 0
        self._takeoff_done = False
        self._probe_queue = self._build_probe_queue()
        self._last_good_flat = _flat_hover(self._ground_pos, self.center_yaw)
        self._pending_probe = None
        self._pending_probe_after_settle = False

    def stop(self) -> None:
        self._done = True

    def abort(self, reason: str) -> None:
        self._aborted = True
        self._abort_reason = str(reason)[:200]

    def update(self, mocap_pose: Dict[str, Any], t_wall: float) -> Dict[str, Any]:
        try:
            return self._update(mocap_pose, t_wall)
        except Exception as exc:
            self.abort(f"internal:{type(exc).__name__}:{exc}")
            flat = _flat_hover(self.center_pos, self.center_yaw)
            return self._wrap(flat, {"phase": "abort_hover",
                                     "reason": self._abort_reason, "dim": "aux"})

    # --- internals ---

    def _safe_hover(self) -> Dict[str, Any]:
        """Hover at the post-takeoff center, or at the ground pose if we
        haven't finished the takeoff ramp yet — prevents aggressive altitude
        commands before the drone has actually left the ground."""
        pos = self.center_pos if self._takeoff_done else self._ground_pos
        return _flat_hover(pos, self.center_yaw)

    def _update(self, mocap_pose: Dict[str, Any], t_wall: float) -> Dict[str, Any]:
        if not self._started:
            return self._wrap(self._safe_hover(), {"phase": "idle", "dim": ""})
        if self._done or self._aborted:
            return self._wrap(self._safe_hover(),
                              {"phase": "done" if self._done else "aborted",
                               "dim": "aux", "reason": self._abort_reason})

        if not self._mocap_valid(mocap_pose):
            flat = self._last_good_flat or self._safe_hover()
            return self._wrap(flat, {"phase": "mocap_stale_hold", "dim": "aux"})

        tilt = self._tilt_angle(mocap_pose)
        if tilt is not None and tilt > self.max_tilt_rad * 1.5:
            # Tilt exceeded the nominal envelope. For sysid data collection we
            # intentionally do NOT abort — aggressive probes can momentarily
            # tip past the hover threshold and we still want the rest of the
            # schedule to run. Log a rate-limited warning and keep tracking.
            now_deg = math.degrees(tilt)
            last = getattr(self, "_last_tilt_warn_t", 0.0)
            if (t_wall - last) > 1.0:
                print(f"[SYSID] tilt warning: {now_deg:.1f}deg (continuing, "
                      f"threshold {math.degrees(self.max_tilt_rad)*1.5:.0f}deg)")
                self._last_tilt_warn_t = t_wall

        if self._phase is None:
            self._next_phase(mocap_pose, t_wall)
        if self._phase is None:
            self._done = True
            return self._wrap(self._safe_hover(),
                              {"phase": "schedule_exhausted", "dim": "aux"})

        t_in_phase = t_wall - self._phase_t_start
        if (isinstance(self._phase, TakeoffRampPhase)
                and t_in_phase >= self._phase.duration
                and not self._takeoff_altitude_ready(mocap_pose)):
            flat = _flat_hover(self.center_pos, self.center_yaw)
            info = self._phase.info(t_in_phase)
            info.update({
                "phase": "takeoff_wait_altitude",
                "reason": "below_hover_altitude",
                "t_since_start": t_wall - self._t_start_wall,
                "round": self._rounds_completed,
                "phases_emitted": self._phases_emitted,
            })
            self._last_good_flat = flat
            return self._wrap(flat, info)

        if t_in_phase >= self._phase.duration:
            self._next_phase(mocap_pose, t_wall)
            if self._phase is None:
                return self._wrap(self._safe_hover(),
                                  {"phase": "schedule_exhausted", "dim": "aux"})
            t_in_phase = 0.0

        flat = self._phase.step(t_in_phase)
        # Do not clamp the z-only takeoff ramp around the hover center; otherwise
        # a ground-start command would jump immediately to center_z - box_half_z.
        if not isinstance(self._phase, TakeoffRampPhase):
            flat = _clamp_position(flat, self.center_pos, self.box_half)
        flat = _sanitize_flat(flat, self.center_pos, self.center_yaw)
        info = self._phase.info(t_in_phase)
        info["t_since_start"] = t_wall - self._t_start_wall
        info["round"] = self._rounds_completed
        info["phases_emitted"] = self._phases_emitted
        self._last_good_flat = flat
        return self._wrap(flat, info)

    def _wrap(self, flat: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        return {"flat_output": flat, "info": info,
                "done": self._done, "aborted": self._aborted,
                "abort_reason": self._abort_reason}

    def _mocap_valid(self, mocap_pose: Dict[str, Any]) -> bool:
        x = mocap_pose.get("x")
        if x is None:
            return False
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        if x_arr.size != 3 or not np.all(np.isfinite(x_arr)):
            return False
        try:
            if not math.isfinite(float(mocap_pose.get("yaw", 0.0))):
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _tilt_angle(mocap_pose: Dict[str, Any]) -> Optional[float]:
        Rm = mocap_pose.get("R")
        if Rm is None:
            return None
        Rm = np.asarray(Rm, dtype=np.float64)
        if Rm.shape != (3, 3) or not np.all(np.isfinite(Rm)):
            return None
        return math.acos(float(np.clip(Rm[2, 2], -1.0, 1.0)))

    def _takeoff_altitude_ready(self, mocap_pose: Dict[str, Any]) -> bool:
        x = mocap_pose.get("x")
        if x is None:
            return False
        pos = np.asarray(x, dtype=np.float64).ravel()
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return False
        tol = max(0.02, float(getattr(self.cfg, "takeoff_ready_z_tol_m", 0.20)))
        return float(pos[2]) >= float(self.center_pos[2]) - tol

    def _next_phase(self, mocap_pose: Dict[str, Any], t_wall: float) -> None:
        """Advance scheduler: probe -> aux(return) -> aux(settle) -> probe -> ..."""
        # First emission (or anytime before takeoff completes): command the
        # z-only ramp from ground to the sysid hover center. This honors the
        # SOP of calling /race from the ground. The ramp is bound from the
        # drone's live mocap pose so it starts wherever the drone currently
        # is (not where it was when /race was first called).
        if not self._takeoff_done:
            start_pos = np.asarray(mocap_pose.get("x", self._ground_pos),
                                   dtype=np.float64)
            start_yaw = float(mocap_pose.get("yaw", self._ground_yaw))
            ramp = TakeoffRampPhase(duration=self.cfg.takeoff_duration_s)
            ramp.bind_from_ground(start_pos, start_yaw,
                                  self.center_pos, self.center_yaw)
            self._phase = ramp
            self._phase_t_start = t_wall
            self._takeoff_done = True  # flipped here; ramp runs to completion
            print(f"[SYSID] takeoff ramp: z {start_pos[2]:.2f} -> "
                  f"{self.center_pos[2]:.2f} m over "
                  f"{self.cfg.takeoff_duration_s:.1f}s")
            return

        if self._phase is None or self._phase.dim != "aux":
            next_probe = self._pop_next_probe()
            if next_probe is None:
                self._phase = None
                return
            target_pos, target_yaw = next_probe.desired_start_pose(
                self.center_pos, self.center_yaw)
            target_pos = np.clip(target_pos,
                                 self.center_pos - self.box_half,
                                 self.center_pos + self.box_half)
            start_pos = np.asarray(mocap_pose.get("x", self.center_pos),
                                   dtype=np.float64)
            start_yaw = float(mocap_pose.get("yaw", self.center_yaw))
            rtc = ReturnToCenterPhase(min_duration=self.cfg.return_min_duration,
                                      max_speed=self.cfg.return_max_speed)
            rtc.bind_from_state(start_pos, start_yaw, target_pos, target_yaw)
            self._phase = rtc
            self._phase_t_start = t_wall
            self._pending_probe = next_probe
            self._pending_probe_start_pos = target_pos
            self._pending_probe_start_yaw = target_yaw
            self._pending_probe_after_settle = True
            return

        if self._pending_probe_after_settle and self.cfg.hover_settle_s > 0:
            hover = HoverPhase("hover_settle", duration=self.cfg.hover_settle_s,
                               dim="aux")
            hover.bind(self._pending_probe_start_pos, self._pending_probe_start_yaw)
            self._phase = hover
            self._phase_t_start = t_wall
            self._pending_probe_after_settle = False
            return

        if self._pending_probe is not None:
            probe = self._pending_probe
            self._pending_probe = None
            probe.bind(self._pending_probe_start_pos, self._pending_probe_start_yaw)
            self._phase = probe
            self._phase_t_start = t_wall
            self._phases_emitted += 1
            return

        # Shouldn't happen; fallback
        np_ = self._pop_next_probe()
        if np_ is None:
            self._phase = None
            return
        tp, ty = np_.desired_start_pose(self.center_pos, self.center_yaw)
        np_.bind(tp, ty)
        self._phase = np_
        self._phase_t_start = t_wall
        self._phases_emitted += 1

    def _pop_next_probe(self) -> Optional[Phase]:
        if not self._probe_queue:
            if self.cfg.loop_drift:
                self._probe_queue = self._build_drift_queue()
                self._rounds_completed += 1
            else:
                return None
        return self._probe_queue.pop(0) if self._probe_queue else None

    def _build_probe_queue(self) -> List[Phase]:
        q: List[Phase] = []
        q.extend(self._round_1_survey())
        q.extend(self._round_2_medium())
        q.extend(self._round_3_fine())
        return q

    def _round_1_survey(self) -> List[Phase]:
        out: List[Phase] = []
        out.append(HoverPhase("hover_mass_survey", duration=7.0, dim="mass", detail=1))
        out.append(VerticalSinePhase(amp=0.15, freq=0.5, cycles=3, detail=1))
        out.append(LateralOscPhase("y", amp=0.10, f0=0.8, f1=0.8, duration=5.0, detail=1))
        out.append(LateralOscPhase("x", amp=0.10, f0=0.8, f1=0.8, duration=5.0, detail=1))
        out.append(YawChirpPhase(amp_deg=15.0, f0=0.5, f1=0.5, duration=5.0, detail=1))
        out.append(ThrustStepPhase(amp=0.10, n_reps=3, detail=1))
        out.append(StraightLinePhase(direction=np.array([1.0, 0, 0]),
                                     v_max=1.0, cruise_duration=1.0, t_ramp=0.4, detail=1))
        if self.cfg.gates:
            gp, gy = self.cfg.gates[0]
            out.append(GateFlybyPhase(np.asarray(gp), gy, approach=0.9,
                                      v_fly=1.2, detail=1, gate_idx=0))
        return out

    def _round_2_medium(self) -> List[Phase]:
        out: List[Phase] = []
        out.append(HoverPhase("hover_mass_medium", duration=25.0, dim="mass", detail=2))
        out.append(VerticalChirpPhase(amp=0.15, f0=0.2, f1=2.0, duration=20.0, detail=2))
        out.append(LateralOscPhase("y", amp=0.08, f0=0.5, f1=4.0, duration=12.0, detail=2))
        out.append(LateralOscPhase("x", amp=0.08, f0=0.5, f1=4.0, duration=12.0, detail=2))
        out.append(YawChirpPhase(amp_deg=20.0, f0=0.3, f1=3.0, duration=12.0, detail=2))
        for dps in (20.0, 40.0):
            out.append(RateDoubletPhase("y", target_dps=dps, period=1.2, detail=2))
            out.append(RateDoubletPhase("x", target_dps=dps, period=1.2, detail=2))
            out.append(RateDoubletPhase("yaw", target_dps=dps, period=1.2, detail=2))
        out.append(ThrustStepPhase(amp=0.12, n_reps=10, detail=2))
        for d in [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]]:
            out.append(StraightLinePhase(direction=np.asarray(d), v_max=1.5,
                                         cruise_duration=2.0, t_ramp=0.6, detail=2))
        for i in range(1, len(self.cfg.gates)):
            gp, gy = self.cfg.gates[i]
            out.append(GateFlybyPhase(np.asarray(gp), gy, approach=1.0,
                                      v_fly=1.2, detail=2, gate_idx=i))
        return out

    def _round_3_fine(self) -> List[Phase]:
        out: List[Phase] = []
        out.append(HoverPhase("hover_mass_fine", duration=40.0, dim="mass", detail=3))
        out.append(VerticalChirpPhase(amp=0.18, f0=0.2, f1=3.0, duration=30.0, detail=3))
        out.append(LateralOscPhase("y", amp=0.10, f0=0.5, f1=6.0, duration=18.0, detail=3))
        out.append(LateralOscPhase("x", amp=0.10, f0=0.5, f1=6.0, duration=18.0, detail=3))
        out.append(YawChirpPhase(amp_deg=25.0, f0=0.3, f1=4.0, duration=18.0, detail=3))
        for dps in (60.0, 80.0):
            out.append(RateDoubletPhase("y", target_dps=dps, period=1.0, detail=3))
            out.append(RateDoubletPhase("x", target_dps=dps, period=1.0, detail=3))
            out.append(RateDoubletPhase("yaw", target_dps=dps, period=1.0, detail=3))
        out.append(ThrustStepPhase(amp=0.14, n_reps=15, detail=3))
        for d in [[1.0, 0, 0], [0, 1.0, 0]]:
            out.append(StraightLinePhase(direction=np.asarray(d), v_max=2.0,
                                         cruise_duration=2.0, t_ramp=0.7, detail=3))
        for i in range(len(self.cfg.gates)):
            gp, gy = self.cfg.gates[i]
            out.append(GateFlybyPhase(np.asarray(gp), gy, approach=1.0,
                                      v_fly=1.5, detail=3, gate_idx=i))
        return out

    def _build_drift_queue(self) -> List[Phase]:
        return [
            HoverPhase("hover_drift", duration=30.0, dim="mass", detail=99),
            VerticalSinePhase(amp=0.12, freq=0.5, cycles=10, detail=99),
        ]


# =============================================================================
# Legacy Actor placeholder (kept so existing imports don't break)
# =============================================================================

if torch is not None:
    class Actor(torch.nn.Module):
        """Retained only for backwards-compatible import. Not used by sysid."""

        def __init__(self, *_, **__):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)
else:
    class Actor:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            raise RuntimeError("Actor requires torch")


# =============================================================================
# Drop-in class: SimpleRacingPolicy
# =============================================================================

class SimpleRacingPolicy:
    """Drop-in replacement for the racing-policy class. Runs the sysid
    scheduler inside an SE3ControlCTBR attitude loop, returning
    (control, obs) in exactly the format the surrounding node expects."""

    def __init__(self, vehicle, model_path, params, device: str = "cpu",
                 **kwargs: Any) -> None:
        self.quadrotor = vehicle
        self.device = torch.device(device) if torch is not None else device
        self.params = params or {}

        # Lazy import so this module still loads on machines without rotorpy
        # (e.g. laptop bag-processing env). Real deploy always has rotorpy.
        from rotorpy.controllers.quadrotor_control import SE3ControlCTBR
        self._se3 = SE3ControlCTBR(vehicle)

        # Pull gate layout out of params. params['waypoints'] is an (N,6)
        # numpy-ish array with columns [x, y, z, roll, pitch, yaw]. Yaw at
        # col 5 is used as the gate normal.
        gates: List[Tuple[Tuple[float, float, float], float]] = []
        wpts = self.params.get("waypoints")
        if wpts is not None:
            try:
                w = np.asarray(wpts, dtype=np.float64)
                if w.ndim == 2 and w.shape[1] >= 6:
                    for row in w:
                        gates.append(((float(row[0]), float(row[1]), float(row[2])),
                                      float(row[5])))
            except Exception:
                gates = []

        cfg = {
            "safety_box_half": (1.2, 1.2, 0.5),
            "max_tilt_deg": 30.0,
            "return_min_duration": 1.2,
            "return_max_speed": 0.7,
            "hover_settle_s": 0.6,
            "loop_drift": True,
        }
        # Allow the caller to override via params['sysid']
        sysid_overrides = self.params.get("sysid", {}) if isinstance(self.params, dict) else {}
        if isinstance(sysid_overrides, dict):
            cfg.update(sysid_overrides)
        self._sysid = SysIDController(config=cfg, gates=gates)

        # Bookkeeping that the surrounding node's racing path reads/writes.
        # The node sets prev_idx_wp from self.idx_wp when /race is called;
        # we keep it at 0 so the lap-counter never spuriously fires.
        self.idx_wp = 0

        # Auto-start sysid on first update, using whatever pose we see.
        self._sysid_started = False
        self._last_update_wall = 0.0
        self._phase_counter = 0
        self._prev_phase_name: Optional[str] = None

        # Keep the 40-dim obs format exact (lin_vel 3 | rot 9 | curr 12 | next 12 | prev 4)
        self.obs_dim = 3 + 9 + 12 + 12 + 4
        self.action_dim = 4
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

        # Hover thrust scaling used when computing cmd_thrust_perc below.
        self._thrust_scale_N = _THRUST_PERC_SCALE_N
        mass_kg = self._vehicle_mass_kg(vehicle)
        self._hover_thrust_perc = float(np.clip(
            mass_kg * _G / self._thrust_scale_N, 0.05, 1.0))

        print("[SimpleRacingPolicy/SYSID] initialized. "
              f"gates={len(gates)} safety_box={cfg['safety_box_half']} "
              f"max_tilt={cfg['max_tilt_deg']}deg "
              f"hover_thrust={self._hover_thrust_perc:.3f}")

    # ------------------------------------------------------------------
    def update(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
        """One control step. `state` is the mocap_pose dict; we return
        (control, obs) to match the TA's policy interface."""
        try:
            return self._update_impl(state)
        except Exception as exc:
            print(f"[SYSID] EXCEPTION in update(): {type(exc).__name__}: {exc}")
            return self._hover_fallback(
                state,
                {"phase": "exception", "dim": "aux", "aborted": True,
                 "abort_reason": f"{type(exc).__name__}: {exc}"},
            )

    @staticmethod
    def _vehicle_mass_kg(vehicle: Any) -> float:
        if isinstance(vehicle, dict):
            for key in ("mass", "m"):
                if key in vehicle:
                    try:
                        mass = float(vehicle[key])
                        if math.isfinite(mass) and mass > 0.0:
                            return mass
                    except Exception:
                        pass
        for key in ("mass", "m"):
            if hasattr(vehicle, key):
                try:
                    mass = float(getattr(vehicle, key))
                    if math.isfinite(mass) and mass > 0.0:
                        return mass
                except Exception:
                    pass
        return 0.038

    def _sanitize_control(self, ctrl: Dict[str, Any]) -> Dict[str, Any]:
        thrust_N = float(ctrl.get("cmd_thrust", 0.0))
        cmd_w = np.asarray(ctrl.get("cmd_w", np.zeros(3)), dtype=np.float64).ravel()
        if cmd_w.size != 3 or not np.all(np.isfinite(cmd_w)):
            cmd_w = np.zeros(3, dtype=np.float64)
        if not math.isfinite(thrust_N) or thrust_N <= 0.0:
            thrust_perc = self._hover_thrust_perc
        else:
            thrust_perc = float(np.clip(thrust_N / self._thrust_scale_N, 0.0, 1.0))

        cmd_w[0] = float(np.clip(cmd_w[0], -200.0, 200.0))
        cmd_w[1] = float(np.clip(cmd_w[1], -200.0, 200.0))
        cmd_w[2] = float(np.clip(cmd_w[2], -150.0, 150.0))

        self.prev_action[0] = 2.0 * thrust_perc - 1.0
        self.prev_action[1] = float(cmd_w[0]) / 200.0
        self.prev_action[2] = float(cmd_w[1]) / 200.0
        self.prev_action[3] = float(cmd_w[2]) / 150.0
        return {"cmd_thrust": thrust_perc, "cmd_w": cmd_w}

    def _hover_fallback(self, state: Dict[str, Any], info: Dict[str, Any],
                        flat: Optional[Dict[str, Any]] = None
                        ) -> Tuple[Dict[str, Any], np.ndarray]:
        try:
            pos = np.asarray(state.get("x", _zeros3()), dtype=np.float64).ravel()
            if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
                pos = _zeros3()
            yaw = float(state.get("yaw", 0.0))
            if not math.isfinite(yaw):
                yaw = 0.0
            target_pos = pos[:3]
            target_yaw = yaw
            if getattr(self, "_sysid_started", False):
                center = np.asarray(self._sysid.center_pos, dtype=np.float64).ravel()
                if center.size >= 3 and np.all(np.isfinite(center[:3])):
                    target_pos = center[:3]
                    target_yaw = float(self._sysid.center_yaw)
            hover_flat = _flat_hover(target_pos, target_yaw)
            try:
                control = self._sanitize_control(self._se3.update(0, state, hover_flat))
            except Exception as exc:
                print(f"[SYSID] hover fallback using open-loop thrust: {exc}")
                control = {"cmd_thrust": self._hover_thrust_perc,
                           "cmd_w": np.zeros(3, dtype=np.float64)}
                self.prev_action[0] = 2.0 * self._hover_thrust_perc - 1.0
                self.prev_action[1:] = 0.0
            obs = self._build_obs(state, flat=flat or hover_flat, info=info)
            return control, obs
        except Exception as exc:
            print(f"[SYSID] hard fallback: {type(exc).__name__}: {exc}")
            return ({"cmd_thrust": self._hover_thrust_perc,
                     "cmd_w": np.zeros(3, dtype=np.float64)},
                    np.zeros(self.obs_dim, dtype=np.float64))

    # ------------------------------------------------------------------
    def _update_impl(self, state: Dict[str, Any]
                     ) -> Tuple[Dict[str, Any], np.ndarray]:
        now = time.time()

        # First update, or /race re-triggered after >1s gap: (re-)initialize.
        if (not self._sysid_started) or (now - self._last_update_wall > 1.0):
            self._sysid.start(
                center_pos=np.asarray(state.get("x", _zeros3()), dtype=np.float64).copy(),
                center_yaw=float(state.get("yaw", 0.0)),
                t_wall=now,
            )
            self._sysid_started = True
            self._phase_counter = 0
            self._prev_phase_name = None
            print(f"[SYSID] started at pos={state.get('x')} yaw={state.get('yaw')}")
        self._last_update_wall = now

        out = self._sysid.update(state, now)
        flat = out["flat_output"]
        info = dict(out.get("info", {}))
        # Surface the outer scheduler-level abort into info so obs[31]
        # (aborted_flag) actually flips in the bagged observation.
        if out.get("aborted"):
            info["aborted"] = True
            info["abort_reason"] = out.get("abort_reason", "")

        # Phase-change logging (single stdout line per phase entry)
        pname = info.get("phase", "?")
        if pname != self._prev_phase_name:
            self._phase_counter += 1
            dim = info.get("dim", "")
            dur = info.get("dur", 0.0)
            det = info.get("detail", 0)
            print(f"[SYSID #{self._phase_counter:03d}] "
                  f"dim={dim:10s} detail={det} phase={pname:36s} dur={dur:.1f}s "
                  f"round={info.get('round',0)} t_total={info.get('t_since_start',0):.1f}s")
            self._prev_phase_name = pname

        if out.get("aborted"):
            # Keep flying (SE3 hover) but signal it in stdout so operator can /stop.
            print(f"[SYSID] ABORTED: {out.get('abort_reason','?')} — "
                  "commanding hover at center; please /stop.")

        # Run SE3 on the flat output to get thrust (N) + body rates (deg/s).
        try:
            ctrl = self._se3.update(0, state, flat)
        except Exception as exc:
            print(f"[SYSID] SE3 update failed: {exc}")
            info["aborted"] = True
            info["abort_reason"] = f"se3_failed:{exc}"
            return self._hover_fallback(state, info, flat=flat)

        control = self._sanitize_control(ctrl)

        obs = self._build_obs(state, flat=flat, info=info)
        return control, obs

    # ------------------------------------------------------------------
    def _build_obs(self, state: Dict[str, Any],
                   flat: Optional[Dict[str, Any]],
                   info: Dict[str, Any]) -> np.ndarray:
        """Produce the 40-dim obs vector. lin_vel and rot carry real state;
        corners_pos_b_curr carries the commanded reference; corners_pos_b_next
        carries phase metadata for offline segmentation."""
        obs = np.zeros(self.obs_dim, dtype=np.float64)

        # lin_vel (body frame)
        v_b = state.get("v_b")
        if v_b is not None:
            v = np.asarray(v_b, dtype=np.float64).ravel()
            if v.size == 3 and np.all(np.isfinite(v)):
                obs[0:3] = v

        # rotation matrix, flattened row-major
        Rm = state.get("R")
        if Rm is not None:
            Rm = np.asarray(Rm, dtype=np.float64)
            if Rm.shape == (3, 3) and np.all(np.isfinite(Rm)):
                obs[3:12] = Rm.flatten()

        # corners_pos_b_curr (12) := reference snapshot (commanded state)
        if flat is not None:
            obs[12:15] = np.asarray(flat.get("x", _zeros3()), dtype=np.float64)
            obs[15:18] = np.asarray(flat.get("x_dot", _zeros3()), dtype=np.float64)
            obs[18:21] = np.asarray(flat.get("x_ddot", _zeros3()), dtype=np.float64)
            obs[21] = float(flat.get("yaw", 0.0))
            obs[22] = float(flat.get("yaw_dot", 0.0))
            obs[23] = float(flat.get("yaw_ddot", 0.0))

        # corners_pos_b_next (12) := phase metadata
        dim_code = _DIM_CODE.get(info.get("dim", ""), 0)
        obs[24] = float(dim_code)
        obs[25] = float(info.get("detail", 0))
        obs[26] = float(info.get("t", 0.0))
        obs[27] = float(info.get("dur", 0.0))
        obs[28] = float(info.get("t_since_start", 0.0))
        obs[29] = float(info.get("round", 0))
        obs[30] = float(info.get("phases_emitted", 0))
        obs[31] = 1.0 if info.get("aborted") else 0.0
        # obs[32:36] reserved for future use (zeros)

        # prev_action (kept for parity with V2 obs shape)
        obs[36:40] = self.prev_action.astype(np.float64)

        # Final finiteness guard
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs


__all__ = [
    "SimpleRacingPolicy",
    "Actor",
    "SysIDController",
    "SysIDConfig",
    "Phase",
]

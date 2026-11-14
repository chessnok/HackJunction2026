"""Simplest possible SO-101 follower motion: straight line in joint space.

delta = target - current; every joint moves linearly at a constant speed so that all joints start
and arrive together: q(t) = current + delta * t / T, with T = max|delta| / speed.

Units are Studio's normalized units: joints -100..100 over their calibrated range, gripper 0..100.
Safety kept: the target must be inside the calibrated range (minus a margin); the move stops and
holds the measured pose when the arm cannot follow (blocked) or on Ctrl+C.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LO = np.array([-100.0, -100.0, -100.0, -100.0, -100.0, 0.0])
HI = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])


def vec(pose: dict[str, float] | list[float] | np.ndarray) -> np.ndarray:
    if isinstance(pose, dict):
        return np.array([float(pose[j]) for j in JOINTS])
    arr = np.asarray(pose, dtype=float)
    if arr.shape != (6,):
        raise ValueError(f"a pose needs 6 values ({', '.join(JOINTS)}), got {arr.shape}")
    return arr


def as_dict(q: np.ndarray) -> dict[str, float]:
    return {j: round(float(v), 2) for j, v in zip(JOINTS, q, strict=True)}


def out_of_range(q: np.ndarray, margin: float = 3.0) -> list[str]:
    """Joints keep ``margin`` away from their calibrated ends; the gripper may use its full 0..100
    (fully closed on an object is a normal target)."""
    margins = np.array([margin] * 5 + [0.0])
    lo, hi = LO + margins, HI - margins
    return [f"{j}={q[i]:.1f} outside [{lo[i]:.0f}, {hi[i]:.0f}]"
            for i, j in enumerate(JOINTS) if not lo[i] <= q[i] <= hi[i]]


def duration(q0: np.ndarray, q1: np.ndarray, speed: float) -> float:
    """Time for the joint with the largest change to cover it at ``speed`` units/s."""
    return float(np.abs(q1 - q0).max()) / speed


@dataclass
class RunResult:
    ok: bool
    reason: str
    final: np.ndarray
    worst_error: float


def move_linear(
    send: Callable[[np.ndarray], None],
    read: Callable[[], np.ndarray],
    q0: np.ndarray,
    q1: np.ndarray,
    speed: float,
    *,
    hz: float = 50.0,
    track_error: float = 12.0,
    track_time: float = 0.3,
    settle_error: float = 3.0,
) -> RunResult:
    """q(t) = q0 + (q1 - q0) * t / T. Stops and holds if the arm lags more than ``track_error``
    units for ``track_time`` s (excluding the gripper, which may be holding an object) or on Ctrl+C."""
    delta = q1 - q0
    total = duration(q0, q1, speed)
    steps = max(1, round(total * hz))
    worst, bad_since = 0.0, None

    def hold(reason: str) -> RunResult:
        q = read()
        send(q)
        return RunResult(False, reason, q, worst)

    try:
        t0 = time.monotonic()
        for k in range(1, steps + 1):
            q_cmd = q0 + delta * (k / steps)
            send(q_cmd)
            err = float(np.abs(read() - q_cmd)[:5].max())
            worst = max(worst, err)
            if err > track_error:
                bad_since = bad_since or time.monotonic()
                if time.monotonic() - bad_since > track_time:
                    return hold(f"arm not following (error {err:.1f} > {track_error})")
            else:
                bad_since = None
            time.sleep(max(0.0, t0 + k / hz - time.monotonic()))
        deadline = time.monotonic() + 1.5
        q = read()
        while time.monotonic() < deadline and float(np.abs(q - q1)[:5].max()) > settle_error:  # settle
            time.sleep(1 / hz)
            q = read()
        final_err = float(np.abs(q - q1)[:5].max())
        if final_err > settle_error:  # e.g. blocked during a move shorter than track_time
            return hold(f"target not reached (off by {final_err:.1f} > {settle_error})")
        return RunResult(True, "reached", q, worst)
    except KeyboardInterrupt:
        return hold("Ctrl+C")

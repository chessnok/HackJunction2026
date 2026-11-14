"""Joint-space motion planning for the SO-101 follower, without a kinematic model.

Safety comes from three things we can measure on the real arm:

1. Joint limits: hard limits from the calibration (normalized range with a margin) plus optional
   soft limits measured by sweeping the arm through the safe workspace (``measure-limits``).
2. Floor poses: poses where the gripper tip touches the table (``measure-floor``). A target close
   to one of them is treated as "near floor": slower approach, tighter stall detection.
3. Route shape: "up, over, down". First raise the arm to the operator-recorded ``transit`` pose
   (shoulder_lift / elbow_flex / wrist_flex only), then rotate pan / wrist roll / gripper at that
   height, descend to a point above the target, and approach the target slowly.

Units are Studio's normalized units: joints -100..100 over their calibrated range, gripper 0..100.
The same numbers appear in Studio datasets and in the physicalai SO101 driver (unit="normalized").
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PAN, ROLL, GRIP = 0, 4, 5
LIFT = [1, 2, 3]  # joints that move the arm in its vertical plane
HARD_LO = np.array([-100.0, -100.0, -100.0, -100.0, -100.0, 0.0])
HARD_HI = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])


def vec(pose: dict[str, float] | list[float] | np.ndarray) -> np.ndarray:
    if isinstance(pose, dict):
        return np.array([float(pose[j]) for j in JOINTS])
    arr = np.asarray(pose, dtype=float)
    if arr.shape != (6,):
        raise ValueError(f"a pose needs 6 values ({', '.join(JOINTS)}), got {arr.shape}")
    return arr


def as_dict(q: np.ndarray) -> dict[str, float]:
    return {j: round(float(v), 2) for j, v in zip(JOINTS, q, strict=True)}


# --------------------------------------------------------------------------- limits
@dataclass
class Limits:
    lo: np.ndarray
    hi: np.ndarray
    source: str

    @classmethod
    def build(cls, margin: float = 3.0, soft: dict | None = None) -> Limits:
        lo, hi = HARD_LO + margin, HARD_HI - margin
        source = f"calibrated range minus {margin} margin"
        if soft:
            lo = np.maximum(lo, vec(soft["min"]))
            hi = np.minimum(hi, vec(soft["max"]))
            source += " ∩ measured soft limits"
        return cls(lo, hi, source)

    def violations(self, q: np.ndarray, label: str) -> list[str]:
        out = []
        for i, j in enumerate(JOINTS):
            if not self.lo[i] <= q[i] <= self.hi[i]:
                out.append(f"{label}: {j}={q[i]:.1f} outside [{self.lo[i]:.1f}, {self.hi[i]:.1f}]")
        return out


# --------------------------------------------------------------------------- floor
def floor_distance(q: np.ndarray, floor_poses: list[np.ndarray], pan_window: float = 15.0) -> float | None:
    """Joint-space distance (LIFT joints) to the nearest measured floor pose with a similar pan."""
    near = [f for f in floor_poses if abs(f[PAN] - q[PAN]) <= pan_window]
    if not near:
        return None
    return float(min(np.linalg.norm(q[LIFT] - f[LIFT]) for f in near))


# --------------------------------------------------------------------------- trajectory
def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


@dataclass
class Segment:
    name: str
    q0: np.ndarray
    q1: np.ndarray
    duration: float

    def at(self, t: float) -> np.ndarray:
        return self.q0 + (self.q1 - self.q0) * min_jerk(t / self.duration if self.duration else 1.0)


def make_segment(name: str, q0: np.ndarray, q1: np.ndarray, vmax: float, min_duration: float = 0.5) -> Segment | None:
    """All joints start and stop together; the slowest joint peaks at ``vmax`` units/s (min-jerk)."""
    delta = float(np.abs(q1 - q0).max())
    if delta < 0.2:
        return None
    return Segment(name, q0.copy(), q1.copy(), max(min_duration, 1.875 * delta / vmax))


@dataclass
class Plan:
    segments: list[Segment]
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    near_floor: bool = False

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)


def plan(
    q0: np.ndarray,
    target: np.ndarray,
    limits: Limits,
    *,
    transit: np.ndarray | None,
    floor_poses: list[np.ndarray] | None = None,
    vmax: float = 25.0,
    approach_vmax: float = 8.0,
    approach_lift: float = 0.3,
    direct: bool = False,
    near_floor_dist: float = 10.0,
) -> Plan:
    """Plan current pose -> target. ``direct`` skips the up-over-down route (small moves only)."""
    floor_poses = floor_poses or []
    p = Plan([])
    p.problems += limits.violations(target, "target")
    dist = floor_distance(target, floor_poses)
    if dist is not None and dist < near_floor_dist:
        p.near_floor = True
        approach_vmax = min(approach_vmax, 4.0)
        p.notes.append(f"target is near the measured floor (distance {dist:.1f}): slow approach")

    if direct or transit is None:
        if transit is None and not direct:
            p.problems.append("no 'transit' pose saved: record one (arm raised above the table) or use --direct")
        seg = make_segment("direct", q0, target, approach_vmax if p.near_floor else vmax)
        p.segments = [s for s in [seg] if s]
        return p

    p.problems += limits.violations(transit, "transit")
    up = q0.copy()
    up[LIFT] = transit[LIFT]  # raise in place
    over = up.copy()
    over[[PAN, ROLL, GRIP]] = target[[PAN, ROLL, GRIP]]  # rotate / open at transit height
    above = target.copy()
    above[LIFT] = target[LIFT] + (transit[LIFT] - target[LIFT]) * approach_lift  # a bit above the target
    for name, q in (("up", up), ("over", over), ("above", above)):
        p.problems += limits.violations(q, name)
        d = floor_distance(q, floor_poses)
        if d is not None and d < near_floor_dist:
            p.problems.append(f"waypoint '{name}' is near the measured floor (distance {d:.1f})")
    chain = [
        make_segment("up", q0, up, vmax),
        make_segment("over", up, over, vmax),
        make_segment("down", over, above, vmax),
        make_segment("approach", above, target, approach_vmax),
    ]
    p.segments = [s for s in chain if s]
    return p


# --------------------------------------------------------------------------- execution
@dataclass
class RunResult:
    ok: bool
    reason: str
    final: np.ndarray | None = None
    worst_error: float = 0.0


def execute(
    send: Callable[[np.ndarray], None],
    read: Callable[[], np.ndarray],
    segments: list[Segment],
    *,
    hz: float = 50.0,
    track_error: float = 12.0,
    track_time: float = 0.3,
    watch_gripper: bool = False,
    log: Callable[[str], None] = print,
) -> RunResult:
    """Stream the trajectory. Stops and holds the measured pose when the arm cannot follow
    (tracking error above ``track_error`` units for ``track_time`` s: likely blocked) or on Ctrl+C."""
    period = 1.0 / hz
    watched = slice(0, 6) if watch_gripper else slice(0, 5)
    worst, bad_since = 0.0, None
    q_meas = read()

    def hold(reason: str) -> RunResult:
        q_hold = read()
        send(q_hold)
        log(f"STOP: {reason}; holding {as_dict(q_hold)}")
        return RunResult(False, reason, q_hold, worst)

    try:
        for seg in segments:
            steps = max(1, round(seg.duration * hz))
            log(f"  {seg.name}: {seg.duration:.1f}s")
            t0 = time.monotonic()
            for k in range(1, steps + 1):
                q_cmd = seg.at(k * seg.duration / steps)
                send(q_cmd)
                q_meas = read()
                err = float(np.abs(q_meas - q_cmd)[watched].max())
                worst = max(worst, err)
                now = time.monotonic()
                if err > track_error:
                    bad_since = bad_since or now
                    if now - bad_since > track_time:
                        return hold(f"arm not following in '{seg.name}' (error {err:.1f} > {track_error})")
                else:
                    bad_since = None
                time.sleep(max(0.0, t0 + k * period - time.monotonic()))
        target = segments[-1].q1 if segments else q_meas
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:  # settle
            q_meas = read()
            if float(np.abs(q_meas - target)[watched].max()) < 2.0:
                break
            time.sleep(period)
        return RunResult(True, "reached", q_meas, worst)
    except KeyboardInterrupt:
        return hold("Ctrl+C")

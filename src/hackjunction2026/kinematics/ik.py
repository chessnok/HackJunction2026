"""Numeric inverse kinematics of the SO-101 follower for a "gripper straight down" pose (numpy only).

Task: put a tool point of the gripper at a table position and height, with the gripper axis (wrist
towards the fingertips) pointing straight down and the jaws closing along a given horizontal
direction (the jaws close across a brick's short side). Five arm joints, six conditions; the
structure makes them consistent: the shoulder_lift, elbow_flex and wrist_flex axes are horizontal and
parallel and the wrist_roll axis is perpendicular to them, so the pitch sum makes the gripper vertical
and wrist_roll then sets the jaw direction about the vertical axis.

Solver: Levenberg-Marquardt with forward-difference Jacobians, vectorized over many targets x seeds
at once, then the elbow-up solution inside the joint ranges that is closest to a start pose is kept.

Tool points (``TOOL_POINTS``), all in ``gripper_link`` coordinates:

* ``jaw_centre`` (default): centre of the gap between the jaws, at fingertip height. This is where the
  middle of a grasped brick sits. It depends on the gripper opening; at the observe opening
  (normalized 93, jaw gap 53 mm) it is 19 mm from the roll axis towards the moving jaw.
* ``gripper_tip``: midpoint of the two fingertip points (agent B's keypoint). With wide-open jaws it
  moves far out and up (38 mm off the roll axis, 15 mm above the fixed fingertip at normalized 93).
* ``tcp``: URDF ``gripper_frame_link`` (LeRobot's TCP). It sits on the fixed jaw, 8 mm off the roll
  axis: over a brick it would put the fixed jaw on top of the brick.

Heights are measured at the tool point's level; for ``jaw_centre`` that is the fixed fingertip, the
lowest part of the open gripper when it points down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hackjunction2026.kinematics.mapping import Calibration, normalized_to_urdf, urdf_to_ticks
from hackjunction2026.kinematics.model import (
    FIXED_JAW_TIP_MM,
    JOINT_ORIGINS_MM,
    MOVING_JAW_TIP_MM,
    _rot_z,
    jaw_gap_mm,
    link_frames,
)

TOOL_POINTS = ("jaw_centre", "gripper_tip", "tcp")
JAW_HALF_THICKNESS_MM = 1.9
"""Half the summed fingertip thickness: when the jaws touch, the two fingertip points are 3.8 mm
apart (moving tip at x = -6.12 vs fixed tip at x = -9.93 in gripper_link, q_gripper = -11.6 deg)."""

ROLL_MARGIN = 3.0  # normalized units kept from every calibrated range end, as so101_motion.out_of_range
_W_ANGLE = 100.0   # residual weight: 1 rad of axis error counts like 100 mm (1 deg ~ 1.75 mm)


def tool_point_mm(point: str, q_gripper: np.ndarray | float) -> np.ndarray:
    """Tool point in gripper_link coordinates (mm), shape (..., 3), for URDF gripper angles (rad)."""
    q_gripper = np.asarray(q_gripper, dtype=np.float64)
    if point == "tcp":
        return np.broadcast_to(JOINT_ORIGINS_MM["gripper_frame_joint"][:3, 3], q_gripper.shape + (3,)).copy()
    if point == "gripper_tip":
        t = JOINT_ORIGINS_MM["gripper"] @ _rot_z(q_gripper)
        moving = t[..., :3, :3] @ MOVING_JAW_TIP_MM + t[..., :3, 3]
        return (moving + FIXED_JAW_TIP_MM) / 2.0
    if point == "jaw_centre":
        out = np.empty(q_gripper.shape + (3,))
        out[..., 0] = FIXED_JAW_TIP_MM[0] + JAW_HALF_THICKNESS_MM + jaw_gap_mm(q_gripper) / 2.0
        out[..., 1] = FIXED_JAW_TIP_MM[1]
        out[..., 2] = FIXED_JAW_TIP_MM[2]
        return out
    raise ValueError(f"unknown tool point {point!r}, expected one of {TOOL_POINTS}")


def tool_pose(q: np.ndarray, point: str = "jaw_centre") -> dict[str, np.ndarray]:
    """For URDF angles q (..., 6): tool position (..., 3) mm, gripper axis (unit, wrist -> fingertips)
    and closing direction (unit, gripper_link +x: the way the moving jaw opens), all in the base frame."""
    q = np.asarray(q, dtype=np.float64)
    g = link_frames(q)["gripper_link"]
    p = tool_point_mm(point, q[..., 5])
    return {
        "position": np.einsum("...ij,...j->...i", g[..., :3, :3], p) + g[..., :3, 3],
        "axis": -g[..., :3, 2],
        "closing": g[..., :3, 0],
    }


def tilt_deg(axis: np.ndarray) -> np.ndarray:
    """Angle between a gripper axis and straight down, degrees."""
    return np.degrees(np.arccos(np.clip(-np.asarray(axis)[..., 2], -1.0, 1.0)))


def closing_yaw_deg(closing: np.ndarray) -> np.ndarray:
    """Direction of the closing axis in the table plane, degrees from +x towards +y, in [-90, 90)."""
    c = np.asarray(closing)
    ang = np.degrees(np.arctan2(c[..., 1], c[..., 0]))
    return (ang + 90.0) % 180.0 - 90.0


def axis_angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest difference between two undirected horizontal directions (degrees, 0..90)."""
    d = (np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)


def elbow_up(q: np.ndarray) -> np.ndarray:
    """True where the elbow lies above the shoulder -> wrist line (in the arm's vertical plane)."""
    f = link_frames(np.asarray(q, dtype=np.float64))
    s = f["upper_arm_link"][..., :3, 3]
    e = f["lower_arm_link"][..., :3, 3]
    w = f["wrist_link"][..., :3, 3]
    sw = w - s
    horiz = np.linalg.norm(sw[..., :2], axis=-1, keepdims=True)
    # Horizontal unit vector of the arm plane; when the wrist is right above the shoulder use the
    # elbow's horizontal direction instead.
    u = np.where(horiz > 1e-6, sw[..., :2] / np.maximum(horiz, 1e-9), np.array([1.0, 0.0]))
    r_e = np.einsum("...i,...i->...", e[..., :2] - s[..., :2], u)
    r_w = horiz[..., 0]
    z_e, z_w = e[..., 2] - s[..., 2], sw[..., 2]
    return r_w * z_e - z_w * r_e > 0


def joint_limits_rad(cal: Calibration, margin: float = ROLL_MARGIN) -> tuple[np.ndarray, np.ndarray]:
    """URDF angle limits (5,) of the arm joints for normalized values in [-100 + margin, 100 - margin]."""
    lo = np.r_[np.full(5, -100.0 + margin), 0.0]
    hi = np.r_[np.full(5, 100.0 - margin), 100.0]
    a, b = normalized_to_urdf(lo, cal)[:5], normalized_to_urdf(hi, cal)[:5]
    return np.minimum(a, b), np.maximum(a, b)


def urdf_to_normalized_unclipped(q: np.ndarray, cal: Calibration) -> np.ndarray:
    """Like mapping.urdf_to_normalized but without the driver's clipping (to detect out-of-range)."""
    ticks = urdf_to_ticks(q)
    span = cal.range_max - cal.range_min
    norm = np.empty_like(ticks)
    norm[..., :5] = (ticks[..., :5] - cal.range_min[:5]) / span[:5] * 200.0 - 100.0
    norm[..., 5] = (ticks[..., 5] - cal.range_min[5]) / span[5] * 100.0
    return norm


def _residuals(q: np.ndarray, target: np.ndarray, closing: np.ndarray, point: str) -> np.ndarray:
    """(..., 7): position error (mm), axis - down (weighted), closing x desired (weighted)."""
    tp = tool_pose(q, point)
    r_pos = tp["position"] - target
    r_ax = _W_ANGLE * (tp["axis"] - np.array([0.0, 0.0, -1.0]))
    c = tp["closing"]
    r_yaw = _W_ANGLE * (c[..., 0] * closing[..., 1] - c[..., 1] * closing[..., 0])
    return np.concatenate([r_pos, r_ax, r_yaw[..., None]], axis=-1)


def _lm(q0: np.ndarray, target: np.ndarray, closing: np.ndarray, point: str, iters: int = 80,
        tol: float = 1e-7) -> np.ndarray:
    """Batched Levenberg-Marquardt over q[..., :5]; q[..., 5] (gripper) stays fixed. Shapes (N, 6)."""
    q = q0.copy()
    n = q.shape[0]
    lam = np.full(n, 1e-2)
    eps = 1e-6
    r = _residuals(q, target, closing, point)
    cost = (r ** 2).sum(-1)
    eye = np.eye(5)
    active = np.ones(n, bool)
    for _ in range(iters):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        qa, ra, ta, ca = q[idx], r[idx], target[idx], closing[idx]
        # Jacobian by forward differences: evaluate the 5 perturbed configurations in one batch.
        qp = np.repeat(qa[:, None, :], 5, axis=1)
        qp[:, np.arange(5), np.arange(5)] += eps
        rp = _residuals(qp, ta[:, None, :], ca[:, None, :], point)          # (A, 5, 7)
        jac = np.swapaxes((rp - ra[:, None, :]) / eps, 1, 2)                 # (A, 7, 5)
        jtj = jac.transpose(0, 2, 1) @ jac
        jtr = np.einsum("nij,ni->nj", jac, ra)
        la = lam[idx]
        a = jtj + la[:, None, None] * (np.diagonal(jtj, axis1=1, axis2=2)[:, :, None] * eye + 1e-6 * eye)
        step = -np.linalg.solve(a, jtr[..., None])[..., 0]
        q_new = qa.copy()
        q_new[:, :5] += step
        r_new = _residuals(q_new, ta, ca, point)
        cost_new = (r_new ** 2).sum(-1)
        better = cost_new < cost[idx]
        gain = np.where(better, cost[idx] - cost_new, 0.0)
        bi = idx[better]
        q[bi], r[bi], cost[bi] = q_new[better], r_new[better], cost_new[better]
        lam[idx] = np.where(better, la * 0.3, la * 10.0).clip(1e-9, 1e9)
        # done: converged, or stuck (damping maxed out / no relative progress: an unreachable target)
        stuck = (lam[idx] >= 1e8) | (better & (gain < 1e-9 * (1.0 + cost[idx])) & (np.abs(step).max(-1) < 1e-9))
        active[idx[(cost[idx] < tol) | stuck]] = False
    return q


@dataclass
class IKSolution:
    ok: bool
    reason: str
    norm: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))  # Studio normalized (6,)
    q: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))     # URDF rad (6,)
    position_mm: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    pos_err_mm: float = np.nan
    tilt_deg: float = np.nan
    yaw_err_deg: float = np.nan
    elbow_up: bool = False
    candidates: int = 0  # converged solutions found (any branch, any range)


def _seeds(target: np.ndarray, closing_yaw: np.ndarray, q_gripper: float) -> np.ndarray:
    """Seeds (N, S, 6) around a geometric guess: pan towards the target, (lift, elbow) on a grid,
    wrist_flex closing the pitch to vertical (lift + elbow + wrist_flex = 90 deg), both roll branches."""
    n = target.shape[0]
    pan = -np.arctan2(target[:, 1], target[:, 0] - 38.8)  # positive pan turns the arm towards -y
    lifts = np.radians([-30.0, 10.0, 50.0])
    elbows = np.radians([-20.0, 30.0, 80.0])
    seeds = []
    for lift in lifts:
        for elbow in elbows:
            wflex = np.pi / 2 - lift - elbow
            base = np.stack([pan, np.full(n, lift), np.full(n, elbow), np.full(n, wflex), np.zeros(n),
                             np.full(n, q_gripper)], axis=-1)
            # roll that gives the wanted closing yaw for this seed (yaw changes 1:1 with roll when vertical)
            yaw0 = np.radians(closing_yaw_deg(tool_pose(base)["closing"]))
            sign = _roll_yaw_sign()
            for branch in (0.0, np.pi):
                s = base.copy()
                s[:, 4] = _wrap(sign * (closing_yaw - yaw0) + branch)
                seeds.append(s)
    return np.stack(seeds, axis=1)


_ROLL_SIGN: float | None = None


def _roll_yaw_sign() -> float:
    """+1 if increasing wrist_roll turns the closing direction counter-clockwise (seen from above) when
    the gripper points down, else -1 (computed once from the model)."""
    global _ROLL_SIGN
    if _ROLL_SIGN is None:
        q = np.array([0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0])
        c0 = tool_pose(q)["closing"]
        q[4] = 0.1
        c1 = tool_pose(q)["closing"]
        _ROLL_SIGN = 1.0 if np.cross(c0, c1)[2] > 0 else -1.0
    return _ROLL_SIGN


def _wrap(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def solve_down_batch(target_mm: np.ndarray, closing_yaw_deg_: np.ndarray, cal: Calibration,
                     gripper_norm: float, start_norm: np.ndarray | None = None, point: str = "jaw_centre",
                     margin: float = ROLL_MARGIN, pos_tol_mm: float = 0.5, ang_tol_deg: float = 0.5,
                     ) -> dict[str, np.ndarray]:
    """Vectorized IK for N targets. Returns arrays: ok (N,), norm (N, 6), q (N, 6), pos_err_mm, tilt_deg,
    yaw_err_deg (N,), elbow_up (N,), n_converged (N,), n_elbow_up (N,), in_range_any (N,).

    ``closing_yaw_deg_`` is the direction the jaws close along (undirected), degrees from +x to +y.
    Among converged elbow-up solutions inside the ranges, the one with the smallest largest joint change
    from ``start_norm`` (default: all zeros) is returned."""
    target = np.atleast_2d(np.asarray(target_mm, dtype=np.float64))
    n = target.shape[0]
    yaw = np.radians(np.broadcast_to(np.asarray(closing_yaw_deg_, dtype=np.float64), (n,)))
    closing = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(n)], axis=-1)
    q_grip = float(normalized_to_urdf(np.r_[np.zeros(5), gripper_norm], cal)[5])
    seeds = _seeds(target, yaw, q_grip)                                   # (N, S, 6)
    s = seeds.shape[1]
    q = _lm(seeds.reshape(n * s, 6), np.repeat(target, s, axis=0), np.repeat(closing, s, axis=0), point)
    q[:, :5] = _wrap(q[:, :5])
    q = q.reshape(n, s, 6)
    tp = tool_pose(q, point)
    pos_err = np.linalg.norm(tp["position"] - target[:, None, :], axis=-1)
    tilt = tilt_deg(tp["axis"])
    yaw_err = axis_angle_diff_deg(closing_yaw_deg(tp["closing"]), np.degrees(yaw)[:, None])
    conv = (pos_err < pos_tol_mm) & (tilt < ang_tol_deg) & (yaw_err < ang_tol_deg)
    norm = urdf_to_normalized_unclipped(q, cal)
    lo, hi = -100.0 + margin, 100.0 - margin
    in_range = np.all((norm[..., :5] >= lo) & (norm[..., :5] <= hi), axis=-1)
    up = elbow_up(q)
    good = conv & in_range & up
    start = np.zeros(6) if start_norm is None else np.asarray(start_norm, dtype=np.float64)
    dist = np.abs(norm[..., :5] - start[:5]).max(-1)
    dist = np.where(good, dist, np.inf)
    best = dist.argmin(axis=1)
    idx = np.arange(n)
    ok = np.isfinite(dist[idx, best])
    out_norm = norm[idx, best].copy()
    out_norm[:, 5] = gripper_norm
    return {
        "ok": ok,
        "norm": out_norm,
        "q": q[idx, best],
        "position_mm": tp["position"][idx, best],
        "pos_err_mm": pos_err[idx, best],
        "tilt_deg": tilt[idx, best],
        "yaw_err_deg": yaw_err[idx, best],
        "elbow_up": up[idx, best],
        "n_converged": conv.sum(1),
        "n_elbow_up": (conv & up).sum(1),
        "in_range_any": (conv & in_range).any(1),
    }


def solve_down(x_mm: float, y_mm: float, z_mm: float, closing_yaw_deg_: float, cal: Calibration,
               gripper_norm: float, start_norm: np.ndarray | None = None, point: str = "jaw_centre",
               margin: float = ROLL_MARGIN) -> IKSolution:
    """IK for one target: tool point at (x, y, z) mm in the base frame, gripper straight down, jaws
    closing along ``closing_yaw_deg_``. Refuses (ok=False) when no elbow-up solution lies inside the
    joint ranges minus ``margin`` normalized units."""
    r = solve_down_batch(np.array([[x_mm, y_mm, z_mm]]), np.array([closing_yaw_deg_]), cal, gripper_norm,
                         start_norm, point, margin)
    if r["ok"][0]:
        reason = "ok"
    elif r["n_converged"][0] == 0:
        reason = "unreachable: no joint solution puts the gripper there pointing straight down"
    elif not r["in_range_any"][0]:
        reason = f"every solution leaves the joint ranges (margin {margin:g} normalized units)"
    elif r["n_elbow_up"][0] == 0:
        reason = "only elbow-down solutions"
    else:
        reason = "no elbow-up solution inside the joint ranges"
    return IKSolution(
        ok=bool(r["ok"][0]), reason=reason, norm=r["norm"][0], q=r["q"][0], position_mm=r["position_mm"][0],
        pos_err_mm=float(r["pos_err_mm"][0]), tilt_deg=float(r["tilt_deg"][0]),
        yaw_err_deg=float(r["yaw_err_deg"][0]), elbow_up=bool(r["elbow_up"][0]),
        candidates=int(r["n_converged"][0]),
    )


def roll_yaw_sign() -> float:
    """Public: +1 when a larger wrist_roll value turns the jaws' closing direction counter-clockwise
    seen from above (gripper pointing down), -1 when clockwise."""
    return _roll_yaw_sign()

"""Gripper-down IK: FK -> IK -> FK round trip, refusals, conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hackjunction2026.kinematics import TABLE_Z_MM, Calibration, ik, normalized_to_urdf

CAL = Calibration.from_file(Path(__file__).resolve().parents[1] / "configs/calibration/follower.json")
OBSERVE = np.array([-12.76, -97.0, 26.4, 97.0, -59.58, 93.18])


def _random_vertical_poses(n: int, seed: int, gripper_norm: float):
    """Random elbow-up poses with the gripper exactly vertical, inside the ranges (margin 3)."""
    rng = np.random.default_rng(seed)
    lo, hi = ik.joint_limits_rad(CAL)
    q = np.zeros((20 * n, 6))
    q[:, :3] = rng.uniform(lo[:3], hi[:3], size=(len(q), 3))
    q[:, 3] = np.pi / 2 - q[:, 1] - q[:, 2]
    q[:, 4] = rng.uniform(lo[4], hi[4], size=len(q))
    q[:, 5] = normalized_to_urdf(np.r_[np.zeros(5), gripper_norm], CAL)[5]
    tp = ik.tool_pose(q)
    norm = ik.urdf_to_normalized_unclipped(q, CAL)
    keep = (np.all(np.abs(norm[:, :5]) <= 96.5, axis=1) & ik.elbow_up(q)
            & (tp["position"][:, 2] > TABLE_Z_MM + 10) & (ik.tilt_deg(tp["axis"]) < 1e-3))
    assert keep.sum() >= n
    return q[keep][:n]


@pytest.mark.parametrize("point,gripper", [("jaw_centre", 93.18), ("jaw_centre", 20.0), ("tcp", 50.0),
                                           ("gripper_tip", 40.0)])
def test_fk_ik_fk_round_trip(point, gripper):
    q = _random_vertical_poses(120, seed=ik.TOOL_POINTS.index(point) * 1000 + int(gripper), gripper_norm=gripper)
    tp = ik.tool_pose(q, point)
    yaw = ik.closing_yaw_deg(tp["closing"])
    r = ik.solve_down_batch(tp["position"], yaw, CAL, gripper, start_norm=OBSERVE, point=point)
    assert r["ok"].all()
    # FK of the returned normalized joints (the values that go to the robot)
    tp2 = ik.tool_pose(normalized_to_urdf(r["norm"], CAL), point)
    assert np.linalg.norm(tp2["position"] - tp["position"], axis=1).max() < 1.0
    assert ik.tilt_deg(tp2["axis"]).max() < 1.0
    assert ik.axis_angle_diff_deg(ik.closing_yaw_deg(tp2["closing"]), yaw).max() < 1.0
    assert r["elbow_up"].all()
    assert np.abs(r["norm"][:, :5]).max() <= 97.0


def test_single_solve_and_conventions():
    s = ik.solve_down(220.0, 40.0, TABLE_Z_MM + 60.0, 90.0, CAL, OBSERVE[5], start_norm=OBSERVE)
    assert s.ok and s.elbow_up
    assert s.pos_err_mm < 1e-3 and s.tilt_deg < 0.01 and s.yaw_err_deg < 0.01
    assert s.norm[5] == OBSERVE[5]
    # a larger wrist_roll turns the closing direction counter-clockwise seen from above (towards +y)
    q = normalized_to_urdf(s.norm, CAL)
    q2 = q.copy()
    q2[4] += np.radians(10.0)
    d = ik.closing_yaw_deg(ik.tool_pose(q2)["closing"]) - ik.closing_yaw_deg(ik.tool_pose(q)["closing"])
    assert ik.roll_yaw_sign() == 1.0
    assert abs(((d + 90) % 180 - 90) - 10.0) < 1e-6


@pytest.mark.parametrize("xyz", [(600.0, 0.0, 60.0), (40.0, 0.0, 60.0), (200.0, 0.0, 400.0)])
def test_unreachable_targets_are_refused(xyz):
    s = ik.solve_down(xyz[0], xyz[1], TABLE_Z_MM + xyz[2], 0.0, CAL, OBSERVE[5])
    assert not s.ok
    assert "unreachable" in s.reason or "range" in s.reason or "elbow" in s.reason


def test_tool_points():
    q_open = normalized_to_urdf(OBSERVE, CAL)[5]
    jc = ik.tool_point_mm("jaw_centre", q_open)
    assert abs(jc[2] - (-103.95)) < 1e-9 and 15 < jc[0] < 25  # 53 mm gap: centre ~19 mm off the roll axis
    tcp = ik.tool_point_mm("tcp", q_open)
    assert np.allclose(tcp, [-7.9, -0.218121, -98.1274])
    with pytest.raises(ValueError):
        ik.tool_point_mm("nose", q_open)

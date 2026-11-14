"""SO-101 follower geometry and vectorized forward kinematics (numpy only).

Geometry is copied from Studio's bundled follower URDF
(``site-packages/urdf/so101-follower/so101_follower.urdf`` of the physicalai-so101-plugin,
the "new calibration" SO-101 model from TheRobotStudio, generated with onshape-to-robot).
``tests/test_kinematics.py`` re-parses that file, when present, and checks these numbers.

Frames and units
----------------
* All positions are in **millimetres** in the URDF ``base_link`` frame ("robot base frame"):
  +x points forward (away from the base, towards the workspace when shoulder_pan = 0),
  +y to the robot's left, +z up. The origin lies 2.4 mm above the bottom of the base mesh
  (z = -2.4 mm, the table) and 38.8 mm behind the vertical shoulder_pan axis (x = 38.8, y = 0).
* Joint angles are URDF radians (``q[..., i]`` for :data:`JOINTS`), see ``mapping.py`` for
  the conversion from Studio's normalized values.
"""

from __future__ import annotations

from typing import Final

import numpy as np

JOINTS: Final = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# Joint origins (xyz in metres, rpy in radians) exactly as in so101_follower.urdf. Every joint
# rotates about its own +z axis.
_URDF_JOINTS: Final = {
    # name: (parent link, child link, xyz, rpy)
    "shoulder_pan": ("base_link", "shoulder_link", (0.0388353, -8.97657e-09, 0.0624),
                     (3.14159, 4.18253e-17, -3.14159)),
    "shoulder_lift": ("shoulder_link", "upper_arm_link", (-0.0303992, -0.0182778, -0.0542),
                      (-1.5708, -1.5708, 0.0)),
    "elbow_flex": ("upper_arm_link", "lower_arm_link", (-0.11257, -0.028, 1.73763e-16),
                   (-3.63608e-16, 8.74301e-16, 1.5708)),
    "wrist_flex": ("lower_arm_link", "wrist_link", (-0.1349, 0.0052, 3.62355e-17),
                   (4.02456e-15, 8.67362e-16, -1.5708)),
    "wrist_roll": ("wrist_link", "gripper_link", (5.55112e-17, -0.0611, 0.0181),
                   (1.5708, 0.0486795, 3.14159)),
    "gripper": ("gripper_link", "moving_jaw_so101_v1_link", (0.0202, 0.0188, -0.0234),
                (1.5708, -5.24284e-08, -1.41553e-15)),
    # fixed joint: LeRobot's tool centre point
    "gripper_frame_joint": ("gripper_link", "gripper_frame_link", (-0.0079, -0.000218121, -0.0981274),
                            (0.0, 3.14159, 0.0)),
}

# Points derived from the URDF meshes (mm, in the given link frame):
#  - fixed jaw tip: centroid of the lowest 1 mm of wrist_roll_follower_so101_v1.stl (gripper_link)
#  - moving jaw tip: centroid of the outermost 1 mm of moving_jaw_so101_v1.stl (its own link)
FIXED_JAW_TIP_MM: Final = np.array([-9.93, -0.23, -103.95])
MOVING_JAW_TIP_MM: Final = np.array([-10.13, -81.53, 18.90])
BASE_BOTTOM_Z_MM: Final = -2.4
"""Lowest point of the base meshes in base_link (the base stands on the table there)."""

# Jaw gap (minimum distance between the two jaws within their lowest 35 mm, from the meshes)
# as a function of the URDF gripper angle. Monotone up to 45 deg, held constant above.
_GAP_Q_DEG: Final = np.arange(-14.0, 46.0, 1.0)
_GAP_MM: Final = np.array([
    0.06, 0.06, 0.07, 0.61, 1.93, 3.28, 4.66, 6.03, 7.4, 8.8, 10.19, 11.59, 12.99, 14.4, 15.8, 17.06,
    18.32, 19.59, 20.85, 22.12, 23.4, 24.67, 25.94, 27.22, 28.49, 29.77, 31.04, 32.31, 33.59, 34.86,
    36.12, 37.39, 38.18, 38.95, 39.69, 40.43, 41.16, 41.88, 42.59, 43.28, 43.96, 44.62, 45.28, 45.92,
    46.54, 47.14, 47.74, 48.3, 48.85, 49.38, 49.9, 50.39, 50.86, 51.31, 51.72, 52.12, 52.49, 52.82,
    53.14, 53.42,
])

KEYPOINTS: Final = (
    "base", "shoulder", "elbow", "wrist", "gripper_base", "gripper_tip",
    "fixed_jaw_tip", "moving_jaw_tip", "tcp",
)
"""Keypoint names returned by :func:`keypoints` (definitions in its docstring)."""


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _origin(name: str) -> np.ndarray:
    _, _, xyz, rpy = _URDF_JOINTS[name]
    t = np.eye(4)
    t[:3, :3] = _rpy_matrix(*rpy)
    t[:3, 3] = np.asarray(xyz) * 1000.0  # mm
    return t


JOINT_ORIGINS_MM: Final = {name: _origin(name) for name in _URDF_JOINTS}


def _rot_z(q: np.ndarray) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    t = np.zeros(q.shape + (4, 4))
    t[..., 0, 0] = c
    t[..., 0, 1] = -s
    t[..., 1, 0] = s
    t[..., 1, 1] = c
    t[..., 2, 2] = 1.0
    t[..., 3, 3] = 1.0
    return t


def link_frames(q: np.ndarray) -> dict[str, np.ndarray]:
    """Homogeneous transforms (..., 4, 4) of every URDF link in base_link, translations in mm.

    Args:
        q: URDF joint angles in radians, shape (..., 6) in :data:`JOINTS` order.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 6:
        raise ValueError(f"expected (..., 6) joint angles, got {q.shape}")
    batch = q.shape[:-1]
    t = np.broadcast_to(np.eye(4), batch + (4, 4))
    out = {"base_link": t}
    for i, name in enumerate(JOINTS[:5]):
        _, child, _, _ = _URDF_JOINTS[name]
        t = t @ JOINT_ORIGINS_MM[name] @ _rot_z(q[..., i])
        out[child] = t
    out["gripper_frame_link"] = out["gripper_link"] @ JOINT_ORIGINS_MM["gripper_frame_joint"]
    out["moving_jaw_so101_v1_link"] = out["gripper_link"] @ JOINT_ORIGINS_MM["gripper"] @ _rot_z(q[..., 5])
    return out


def _apply(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return t[..., :3, :3] @ p + t[..., :3, 3]


def keypoints(q: np.ndarray) -> dict[str, np.ndarray]:
    """Keypoint positions (..., 3) in mm in the robot base frame.

    * ``base``: base_link origin (constant, 0 0 0).
    * ``shoulder``, ``elbow``, ``wrist``: origins of the shoulder_lift, elbow_flex and wrist_flex
      joints (points on the joint axes, as placed by the URDF).
    * ``gripper_base``: wrist_roll joint origin (on the roll axis, where the gripper starts).
    * ``fixed_jaw_tip`` / ``moving_jaw_tip``: fingertips from the meshes.
    * ``gripper_tip``: midpoint of the two fingertips (what touches / surrounds a grasped object).
    * ``tcp``: the URDF ``gripper_frame_link`` (LeRobot's tool centre point), 5.8 mm back from
      the fixed fingertip along the gripper axis.
    """
    f = link_frames(q)
    fixed = _apply(f["gripper_link"], FIXED_JAW_TIP_MM)
    moving = _apply(f["moving_jaw_so101_v1_link"], MOVING_JAW_TIP_MM)
    return {
        "base": np.broadcast_to(f["base_link"][..., :3, 3], fixed.shape).copy(),
        "shoulder": f["upper_arm_link"][..., :3, 3].copy(),
        "elbow": f["lower_arm_link"][..., :3, 3].copy(),
        "wrist": f["wrist_link"][..., :3, 3].copy(),
        "gripper_base": f["gripper_link"][..., :3, 3].copy(),
        "gripper_tip": (fixed + moving) / 2.0,
        "fixed_jaw_tip": fixed,
        "moving_jaw_tip": moving,
        "tcp": f["gripper_frame_link"][..., :3, 3].copy(),
    }


def jaw_gap_mm(q_gripper: np.ndarray) -> np.ndarray:
    """Gap between the jaws near the fingertips (mm) for URDF gripper angles in radians.

    0 when the jaws touch (about -11.6 deg); valid up to 45 deg (about 53 mm), constant above.
    """
    return np.interp(np.degrees(np.asarray(q_gripper, dtype=np.float64)), _GAP_Q_DEG, _GAP_MM)

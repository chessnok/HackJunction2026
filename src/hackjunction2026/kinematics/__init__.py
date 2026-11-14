"""SO-101 follower forward kinematics from Studio's recorded joint values.

Quick use::

    from hackjunction2026.kinematics import Calibration, arm_keypoints
    cal = Calibration.from_file("configs/calibration/follower.json")
    kp = arm_keypoints(state, cal)   # state: (N, 6) normalized values from observation.state
    kp["gripper_tip"]                # (N, 3) mm in the robot base frame

Conventions (base frame, joint mapping, table plane, accuracy) are documented in
``docs/perception/B-arm-kinematics.md``. Runtime dependency: numpy only.
"""

from __future__ import annotations

import numpy as np

from hackjunction2026.kinematics.mapping import (
    JOINT_MAP,
    Calibration,
    normalized_to_ticks,
    normalized_to_urdf,
    ticks_to_normalized,
    ticks_to_urdf,
    urdf_to_normalized,
    urdf_to_ticks,
)
from hackjunction2026.kinematics.model import (
    BASE_BOTTOM_Z_MM,
    JOINTS,
    KEYPOINTS,
    jaw_gap_mm,
    keypoints,
    link_frames,
)

TABLE_Z_MM = -2.4
"""Table surface height in the robot base frame (mm): the bottom of the base, which stands on the
table. Fingertips at 55 grasps are 5.0 +- 4.1 mm above it and a plane through them is not tilted
measurably (see the doc). Treat the table as the plane z = TABLE_Z_MM."""


def arm_keypoints(norm: np.ndarray, cal: Calibration) -> dict[str, np.ndarray]:
    """Keypoints (..., 3) in mm plus ``jaw_gap_mm`` and ``jaw_tip_dist_mm`` (...,) from normalized values."""
    q = normalized_to_urdf(norm, cal)
    kp = keypoints(q)
    kp["jaw_gap_mm"] = jaw_gap_mm(q[..., 5])
    kp["jaw_tip_dist_mm"] = np.linalg.norm(kp["moving_jaw_tip"] - kp["fixed_jaw_tip"], axis=-1)
    return kp


__all__ = [
    "BASE_BOTTOM_Z_MM",
    "JOINTS",
    "JOINT_MAP",
    "KEYPOINTS",
    "TABLE_Z_MM",
    "Calibration",
    "arm_keypoints",
    "jaw_gap_mm",
    "keypoints",
    "link_frames",
    "normalized_to_ticks",
    "normalized_to_urdf",
    "ticks_to_normalized",
    "ticks_to_urdf",
    "urdf_to_normalized",
    "urdf_to_ticks",
]

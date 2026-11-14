"""Table clearance of a straight joint-space move (what ``scripts/so101_goto.py`` executes).

so101_goto moves every joint linearly in Studio's normalized units from the current pose to the
target, all joints arriving together: q(s) = q0 + s * (q1 - q0), s in [0, 1]. Normalized units map
linearly to URDF angles, so sampling s and running FK gives the swept arm exactly (up to tracking lag).

Checked points: the joint origins from the shoulder on (the base and shoulder_pan housing stand on the
table and do not move down), points along the links between them, both fingertips and the moving
jaw. The arm's link bodies are 20-30 mm thick around these lines; the default clearance of 30 mm
above the table covers that, it does not cover objects standing on the table (bricks, plates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hackjunction2026.kinematics.mapping import Calibration, normalized_to_urdf
from hackjunction2026.kinematics.model import (
    BASE_BOTTOM_Z_MM,
    FIXED_JAW_TIP_MM,
    MOVING_JAW_TIP_MM,
    link_frames,
)

TABLE_Z = BASE_BOTTOM_Z_MM
"""Table plane z in the base frame (same value as ``hackjunction2026.kinematics.TABLE_Z_MM``)."""
_MOVING_PIVOT_MM = np.zeros(3)  # origin of the moving jaw link


def _apply(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,j->...i", t[..., :3, :3], p) + t[..., :3, 3]


def link_points(q: np.ndarray, per_segment: int = 6) -> tuple[np.ndarray, list[str]]:
    """Points (..., P, 3) mm along the moving links, and a name per point."""
    f = link_frames(np.asarray(q, dtype=np.float64))
    named = {
        "shoulder": f["upper_arm_link"][..., :3, 3],
        "elbow": f["lower_arm_link"][..., :3, 3],
        "wrist": f["wrist_link"][..., :3, 3],
        "gripper_base": f["gripper_link"][..., :3, 3],
        "fixed_jaw_tip": _apply(f["gripper_link"], FIXED_JAW_TIP_MM),
        "moving_jaw_pivot": _apply(f["moving_jaw_so101_v1_link"], _MOVING_PIVOT_MM),
        "moving_jaw_tip": _apply(f["moving_jaw_so101_v1_link"], MOVING_JAW_TIP_MM),
        "tcp": f["gripper_frame_link"][..., :3, 3],
    }
    segments = [("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "gripper_base"),
                ("gripper_base", "fixed_jaw_tip"), ("moving_jaw_pivot", "moving_jaw_tip")]
    pts, names = [], []
    for a, b in segments:
        for k in range(per_segment):
            s = k / (per_segment - 1)
            pts.append(named[a] * (1 - s) + named[b] * s)
            names.append(a if s == 0 else b if s == 1 else f"{a}->{b} {s:.1f}")
    pts.append(named["tcp"])
    names.append("tcp")
    return np.stack(pts, axis=-2), names


@dataclass
class PathCheck:
    ok: bool
    min_clearance_mm: float  # lowest checked point above the table, over the whole path
    at_fraction: float       # where along the move (0 = start, 1 = target)
    at_point: str            # which arm point
    end_clearance_mm: float  # lowest point at the target pose
    samples: int
    required_mm: float

    def describe(self) -> str:
        verdict = "OK" if self.ok else "REFUSED"
        return (f"{verdict}: lowest arm point {self.min_clearance_mm:.1f} mm above the table "
                f"({self.at_point}, at {self.at_fraction:.0%} of the move; required {self.required_mm:.0f} mm); "
                f"at the target {self.end_clearance_mm:.1f} mm ({self.samples} samples)")


def check_straight_move(start_norm: np.ndarray, target_norm: np.ndarray, cal: Calibration, samples: int = 50,
                        min_clearance_mm: float = 30.0, table_z_mm: float = TABLE_Z) -> PathCheck:
    """FK along q(s) = start + s * (target - start) (normalized units, all 6 joints), ``samples`` points
    including both ends; ok if every checked arm point stays ``min_clearance_mm`` above the table."""
    q0, q1 = np.asarray(start_norm, float), np.asarray(target_norm, float)
    s = np.linspace(0.0, 1.0, samples)
    norm = q0 + s[:, None] * (q1 - q0)
    pts, names = link_points(normalized_to_urdf(norm, cal))  # (S, P, 3)
    clear = pts[..., 2] - table_z_mm
    i, j = np.unravel_index(int(np.argmin(clear)), clear.shape)
    worst = float(clear[i, j])
    return PathCheck(ok=worst >= min_clearance_mm, min_clearance_mm=worst, at_fraction=float(s[i]),
                     at_point=names[j], end_clearance_mm=float(clear[-1].min()), samples=samples,
                     required_mm=min_clearance_mm)

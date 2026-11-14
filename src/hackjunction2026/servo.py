"""Wrist-camera centring at the hover pose: move the arm until the brick sits where a brick held between the closed
jaws appears in the gripper-down wrist image.

The camera is rigid on the gripper, so at a fixed hover height a brick centred under the closed-jaw point always
shows at the same pixel (``target_px``). How the brick's pixel moves when the arm moves by 1 mm in table x / y is
the 2x2 image Jacobian ``J`` (px per mm). Both are measured once by ``scripts/servo_calib.py`` and saved to
``configs/servo.yaml``; ``centre()`` then corrects the hover XY:  p + J @ d = target  ->  d = J^-1 (target - p).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
SERVO_YAML = REPO / "configs/servo.yaml"


@dataclass
class ServoModel:
    target_px: np.ndarray            # (2,) pixel of a centred brick at the hover height
    J: np.ndarray                    # (2, 2) d(pixel) / d(arm xy mm)
    hover_mm: float                  # hover height (jaw centre above the table) the model was measured at
    yaw0_deg: float = 0.0            # directed jaw yaw (fixed -> moving jaw tip) when J was measured
    samples: list = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = SERVO_YAML) -> ServoModel | None:
        if not Path(path).exists():
            return None
        d = yaml.safe_load(Path(path).read_text())
        return cls(np.array(d["target_px"], float), np.array(d["J"], float), float(d["hover_mm"]),
                   float(d.get("yaw0_deg", 0.0)), d.get("samples", []))

    def save(self, path: Path = SERVO_YAML, note: str = "") -> None:
        Path(path).write_text(yaml.safe_dump({
            "description": "Wrist-camera centring at the hover (src/hackjunction2026/servo.py): target_px = pixel of a "
                           "brick centred under the closed jaws, J = d pixel / d arm xy (px/mm).",
            "hover_mm": self.hover_mm, "yaw0_deg": self.yaw0_deg, "target_px": [float(v) for v in self.target_px],
            "J": [[float(v) for v in row] for row in self.J], "note": note, "samples": self.samples},
            sort_keys=False))

    def J_at(self, yaw_deg: float) -> np.ndarray:
        """The camera turns with the gripper: J(yaw) = J0 R(yaw0 - yaw)."""
        a = np.radians(self.yaw0_deg - yaw_deg)
        return self.J @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def correction_mm(self, px: np.ndarray, yaw_deg: float | None = None) -> np.ndarray:
        """Arm XY step (mm) that moves the brick from ``px`` to ``target_px``."""
        J = self.J if yaw_deg is None else self.J_at(yaw_deg)
        return np.linalg.solve(J, self.target_px - np.asarray(px, float))


def directed_yaw_deg(q_norm: np.ndarray, cal) -> float:
    """Direction from the fixed to the moving jaw tip in the table plane (deg from +x): the wrist camera's yaw."""
    from hackjunction2026.kinematics import arm_keypoints

    kp = arm_keypoints(np.asarray(q_norm, float)[None], cal)
    d = kp["moving_jaw_tip"][0][:2] - kp["fixed_jaw_tip"][0][:2]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def brick_px(frame_bgr: np.ndarray, near: np.ndarray | None = None, area=(1500, 60000)) -> np.ndarray | None:
    """Pixel centroid of the brick in a gripper-down wrist frame: the coloured blob (not the purple jaws) nearest
    ``near`` (default: the image centre), within a size range."""
    from hackjunction2026 import table_map as tm

    det = tm.detect(frame_bgr)
    cands = [b for b in det.bricks if area[0] <= b.area_px <= area[1]]
    if not cands:
        return None
    ref = np.array([frame_bgr.shape[1] / 2, frame_bgr.shape[0] / 2]) if near is None else np.asarray(near, float)
    b = min(cands, key=lambda b: float(np.hypot(*(np.asarray(b.centroid) - ref))))
    return np.asarray(b.centroid, float)


def fit_jacobian(moves_mm: np.ndarray, shifts_px: np.ndarray) -> np.ndarray:
    """Least squares J with shifts = moves @ J.T (rows: one test move each)."""
    X, *_ = np.linalg.lstsq(np.asarray(moves_mm, float), np.asarray(shifts_px, float), rcond=None)
    return X.T

"""Table coordinate frame from the top D415: metric rectification of the table plane.

No camera sees all four sheet corners, so the plane is recovered from the camera's factory
intrinsics plus straight sheet edges (see `sheet.py`):

1. The plane normal n (camera coordinates) is the one for which, in all three positions the
   lower sheet had during recording, its far edge is perpendicular to its left edge and parallel
   to the seam of the upper sheet (least squares over the 6 angle residuals).
2. A pixel p maps to the 3D point X = h * K^-1 p / (n . K^-1 p) on the plane n . X = h, where h is
   the camera height above the table.
3. Table axes: x along the sheet's far edge (left -> right in the top image), y perpendicular to
   it in the plane, pointing toward the top camera (down in the top image). Origin: the far-left
   sheet corner (far edge x left edge) in the reference sheet position (episodes 31-61).

Everything here is in units of h; `scale_mm` (= h in millimetres) turns them into millimetres.
The pixel -> table map is the 3x3 homography `H_px_to_table`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .geometry import apply_h, intersect


def K_matrix(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def _ndir(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def line_dir_on_plane(K: np.ndarray, n: np.ndarray, line: np.ndarray) -> np.ndarray:
    """3D direction (camera coords) of an image line lying on the plane with normal n."""
    d = np.cross(n, K.T @ line)
    return d / np.linalg.norm(d)


def line_angle(K, n, l1, l2) -> float:
    c = abs(line_dir_on_plane(K, n, l1) @ line_dir_on_plane(K, n, l2))
    return float(np.degrees(np.arccos(np.clip(c, 0, 1))))


def fit_plane_normal(K: np.ndarray, setups: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Unit normal (pointing away from the camera) from per-setup lines {far, left, seam}."""

    def resid(p):
        n = _ndir(*p)
        out = []
        for s in setups:
            out.append(line_angle(K, n, s["far"], s["left"]) - 90.0)
            out.append(line_angle(K, n, s["far"], s["seam"]))
        return out

    best = None
    for th in np.linspace(0.2, 1.5, 12):
        for ph in np.linspace(0.3, 2.8, 12):
            r = least_squares(resid, [th, ph])
            if best is None or r.cost < best.cost:
                best = r
    n = _ndir(*best.x)
    if n[2] < 0:
        n = -n
    return n, np.asarray(best.fun)


@dataclass
class TableFrame:
    K: np.ndarray
    n: np.ndarray  # plane normal, camera coords, pointing away from the camera
    e1: np.ndarray  # table x axis (camera coords)
    e2: np.ndarray  # table y axis (camera coords)
    origin: np.ndarray  # origin on the plane for h = 1 (camera coords)
    scale_mm: float = 1.0  # camera height above the table in mm

    @classmethod
    def from_lines(cls, K, n, far, left, scale_mm=1.0) -> TableFrame:
        e1 = line_dir_on_plane(K, n, far)
        # +x = left -> right in the image: project a step along e1 and check
        if e1[0] < 0:
            e1 = -e1
        e2 = np.cross(n, e1)
        e2 /= np.linalg.norm(e2)
        if e2[2] > 0:  # +y toward the camera (smaller depth)
            e2 = -e2
        corner = intersect(far, left)
        r = np.linalg.solve(K, [corner[0], corner[1], 1.0])
        origin = r / (n @ r)
        return cls(K, n, e1, e2, origin, scale_mm)

    @property
    def H_px_to_table(self) -> np.ndarray:
        """Homography pixel -> table (x, y) in mm (or in units of h if scale_mm == 1)."""
        h = self.scale_mm
        n, e1, e2, o = self.n, self.e1, self.e2, self.origin * h
        M = np.vstack([h * e1 - (o @ e1) * n, h * e2 - (o @ e2) * n, n])
        H = M @ np.linalg.inv(self.K)
        return H / H[2, 2]

    @property
    def H_table_to_px(self) -> np.ndarray:
        H = np.linalg.inv(self.H_px_to_table)
        return H / H[2, 2]

    def px_to_table(self, pts) -> np.ndarray:
        return apply_h(self.H_px_to_table, pts)

    def table_to_px(self, pts) -> np.ndarray:
        return apply_h(self.H_table_to_px, pts)

    @property
    def R_cam_from_table(self) -> np.ndarray:
        """Columns: table x, y, z axes in camera coords; z = x cross y (points into the table)."""
        return np.c_[self.e1, self.e2, np.cross(self.e1, self.e2)]

    @property
    def camera_in_table(self) -> np.ndarray:
        """Camera centre in table coordinates (mm): x, y, z (z < 0 = above the table)."""
        R = self.R_cam_from_table
        return R.T @ (-self.origin * self.scale_mm)

    def to_json(self) -> dict:
        return {
            "K": self.K.tolist(), "n_cam": self.n.tolist(), "x_axis_cam": self.e1.tolist(),
            "y_axis_cam": self.e2.tolist(), "origin_cam_unit_height": self.origin.tolist(),
            "camera_height_mm": self.scale_mm,
        }


def load_frame_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())

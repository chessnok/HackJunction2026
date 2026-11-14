"""Detect the white paper sheet's straight edges in the two D415 views.

Scene (dataset 431a71f7, see docs/perception/A-object-localization.md): two large white sheets
overlap on the table. No camera sees all four sheet corners, so the table frame is built from
straight lines plus the factory colour intrinsics:

- top camera: far edge (sheet edge on the robot's right side), left edge (sheet edge
  opposite the robot), and the seam where the two sheets overlap (parallel to the far edge);
- floor camera: far edge (the sheet edge on the robot side), plus the short visible pieces of
  the left and right sheet edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import fit_line_ransac


def white_mask(rgb: np.ndarray, s_max: int = 50, v_min: int = 75) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] < s_max) & (hsv[..., 2] > v_min)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def _first_true(a: np.ndarray, run: int = 8) -> int:
    """Index of the first element starting a run of `run` True values, or -1."""
    a = a.astype(np.int32)
    c = np.convolve(a, np.ones(run, np.int32), "valid")
    idx = np.flatnonzero(c == run)
    return int(idx[0]) if len(idx) else -1


def top_boundary_points(mask: np.ndarray, y_max: int, step: int = 2) -> np.ndarray:
    pts = []
    for x in range(0, mask.shape[1], step):
        y = _first_true(mask[:y_max + 10, x])
        if 2 < y < y_max:
            pts.append((x, y - 0.5))
    return np.array(pts, float)


def left_boundary_points(mask: np.ndarray, y_range: tuple[int, int], x_max: int, step: int = 2) -> np.ndarray:
    pts = []
    for y in range(y_range[0], y_range[1], step):
        x = _first_true(mask[y, :x_max + 10])
        if 2 < x < x_max:
            pts.append((x - 0.5, y))
    return np.array(pts, float)


def right_boundary_points(mask: np.ndarray, y_range: tuple[int, int], x_min: int, step: int = 2) -> np.ndarray:
    pts = []
    w = mask.shape[1]
    for y in range(y_range[0], y_range[1], step):
        x = _first_true(mask[y, x_min:][::-1])
        if x > 2:
            pts.append((w - x - 0.5, y))
    return np.array(pts, float)


def seam_points(rgb: np.ndarray, mask: np.ndarray, y_lo: np.ndarray, y_hi: int, step: int = 4) -> np.ndarray:
    """Thin dark nearly-horizontal line inside the sheet: max of a vertical ridge filter per column."""
    v = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    v = cv2.GaussianBlur(v, (9, 1), 0)
    ridge = 0.5 * (np.roll(v, 3, 0) + np.roll(v, -3, 0)) - v
    ridge[~mask] = 0
    pts = []
    for x in range(0, rgb.shape[1], step):
        lo = int(y_lo[x])
        if lo >= y_hi:
            continue
        col = ridge[lo:y_hi, x]
        y = int(np.argmax(col))
        if col[y] > 2.0:
            pts.append((x, lo + y))
    return np.array(pts, float)


def detect_top_lines(rgb: np.ndarray) -> dict:
    """Lines (a, b, c) in top-camera pixels: far edge, left edge, seam. Plus inlier counts."""
    h, w = rgb.shape[:2]
    m = white_mask(rgb)
    far, far_in = fit_line_ransac(top_boundary_points(m, y_max=h // 3), thresh=1.5)
    # left edge: rows below the far edge, boundary inside the image
    left, left_in = fit_line_ransac(left_boundary_points(m, (20, h - 20), x_max=w // 3), thresh=1.5)
    # seam: search from 60 px below the far edge to the bottom quarter
    xs = np.arange(w)
    y_far = -(far[0] * xs + far[2]) / far[1]
    seam, seam_in = fit_line_ransac(seam_points(rgb, m, y_far + 80, int(h * 0.8)), thresh=2.0)
    return {
        "far": far, "left": left, "seam": seam,
        "n_inliers": {"far": int(far_in.sum()), "left": int(left_in.sum()), "seam": int(seam_in.sum())},
        "mask": m,
    }


def detect_floor_lines(rgb: np.ndarray) -> dict:
    """Lines in floor-camera pixels: far edge, right edge (short, top right), left edge (bottom left)."""
    h, w = rgb.shape[:2]
    m = white_mask(rgb)
    far, far_in = fit_line_ransac(top_boundary_points(m, y_max=h // 3), thresh=1.5)
    right, right_in = fit_line_ransac(right_boundary_points(m, (20, h // 3), x_min=w * 2 // 3), thresh=1.5)
    left, left_in = fit_line_ransac(left_boundary_points(m, (h * 2 // 3, h - 2), x_max=w // 3), thresh=1.5)
    return {
        "far": far, "right": right, "left": left,
        "n_inliers": {
            "far": int(far_in.sum()),
            "right": 0 if right_in is None else int(right_in.sum()),
            "left": 0 if left_in is None else int(left_in.sum()),
        },
        "mask": m,
    }

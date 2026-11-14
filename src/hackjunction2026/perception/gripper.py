"""Gripper (dark indigo SO-101 jaws) in the top / floor D415 images.

Both D415s stand low (about 8-12 cm above the table), so the arm enters the image from the top
and the jaw tips are the lowest arm pixels. The robot body seen in the episode's first frame
(arm at home) is masked out. Hue 108-135 / S >= 60 / V <= 100 separates the jaws from blue
bricks (hue 96-104).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GripperDet:
    bbox: tuple[float, float, float, float]
    tip: tuple[float, float]  # lowest arm point in the image (jaw tips)
    area: int


def arm_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(np.int32) for i in range(3))
    m = ((h >= 108) & (h <= 135) & (s >= 60) & (v <= 100)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


def static_robot_mask(first_rgb: np.ndarray, dilate: int = 6) -> np.ndarray:
    """Robot parts visible with the arm at home: arm-coloured components touching the image border."""
    m = arm_mask(first_rgb).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    H, W = m.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= 300 and (x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2):
            keep[lab == i] = 1
    return cv2.dilate(keep, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)).astype(bool)


def detect_gripper(rgb: np.ndarray, static: np.ndarray | None, roi_rows: int | None = None,
                   min_area: int = 250) -> GripperDet | None:
    m = arm_mask(rgb)
    if static is not None:
        m &= ~static
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return None
    # the arm: largest component plus components close to it (the two jaws can separate)
    main = max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h = stats[main, :4]
    sel = [main]
    for i in comps:
        if i == main:
            continue
        xi, yi, wi, hi = stats[i, :4]
        gap_x = max(0, max(x, xi) - min(x + w, xi + wi))
        gap_y = max(0, max(y, yi) - min(y + h, yi + hi))
        if gap_x < 40 and gap_y < 40:
            sel.append(i)
    arm = np.isin(lab, sel)
    ys, xs = np.nonzero(arm)
    ymax = ys.max()
    low = ys >= ymax - 8
    tip = (float(xs[low].mean()), float(ymax))
    return GripperDet((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ymax + 1)), tip, int(arm.sum()))

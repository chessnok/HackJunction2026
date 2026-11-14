"""Classic CV baseline: saturated-colour blobs on the white sheet (HSV thresholds + connected components).

Colour classes (OpenCV hue 0..179) measured on the dataset's first frames:
red 0-6 / 170-179, yellow (orange-yellow Duplo) 8-25, green (lime Duplo) 26-48, blue 88-110.
The green bowl (hue ~60-80), robot body and gripper (purple, hue ~115-150) and wood (hue 14-20,
only outside the sheet) are excluded by hue or by the sheet ROI.

A brick made of two colours (e.g. red on blue) gives touching blobs of different colours; they
are merged into one object whose `colors` lists both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

HUE_RANGES = {
    "red": [(0, 7), (170, 180)],
    "yellow": [(8, 25)],
    "green": [(26, 48)],
    "blue": [(88, 107)],
}
COLOR_ORDER = ("red", "yellow", "green", "blue")
# v2 (2026-11-14), from first frames of the 51 episodes outside the evaluation set: blue bricks
# have S >= 110 (5th percentile) and H 89-106; the gripper's dark indigo is H 101-119 at V < 70;
# shadowed green faces go down to V ~ 20 with high S.
S_MIN = {"red": 85, "yellow": 85, "green": 85, "blue": 100}
DARK_S_MIN, DARK_V_MIN = 120, 18  # dark but strongly saturated pixels still count


@dataclass
class Blob:
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 (pixels, x1/y1 exclusive)
    area: int
    colors: list[str]
    color_areas: dict[str, int] = field(default_factory=dict)
    score: float = 1.0

    @property
    def color(self) -> str:
        """Dominant colour first, e.g. 'red' or 'red+blue' (sorted by area)."""
        return "+".join(self.colors)

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return 0.5 * (x0 + x1), y1


def color_masks(rgb: np.ndarray, roi: np.ndarray | None = None, v_min: int = 40) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(np.int32) for i in range(3))
    bright = (v >= v_min) | ((v >= DARK_V_MIN) & (s >= DARK_S_MIN))
    gripper = (h >= 102) & (h <= 135) & (s >= 85) & (v < 62)
    if roi is not None:
        bright &= roi
    out = {}
    k = np.ones((3, 3), np.uint8)
    for name, ranges in HUE_RANGES.items():
        m = np.zeros(h.shape, bool)
        for lo, hi in ranges:
            m |= (h >= lo) & (h < hi)
        m &= bright & (s >= S_MIN[name])
        if name == "blue":
            m &= ~gripper
        m = m.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        out[name] = m.astype(bool)
    return out


def detect_blobs(rgb: np.ndarray, roi: np.ndarray | None = None, min_area: int = 80,
                 merge_gap: int = 4) -> list[Blob]:
    """Coloured blobs on the sheet; touching blobs of different colours form one object."""
    masks = color_masks(rgb, roi)
    any_m = np.zeros(rgb.shape[:2], np.uint8)
    for m in masks.values():
        any_m |= m.astype(np.uint8)
    # merge parts that are a few pixels apart (two-colour bricks, studs)
    joined = cv2.dilate(any_m, np.ones((2 * merge_gap + 1, 2 * merge_gap + 1), np.uint8))
    n, lab = cv2.connectedComponents(joined, connectivity=8)
    blobs = []
    for i in range(1, n):
        comp = (lab == i) & any_m.astype(bool)
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(comp)
        areas = {c: int((masks[c] & comp).sum()) for c in COLOR_ORDER}
        # a colour counts if it is a real part of the object, not a fringe
        colors = [c for c in sorted(areas, key=areas.get, reverse=True) if areas[c] >= max(40, 0.12 * area)]
        if not colors:
            continue
        blobs.append(Blob((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
                          area, colors, areas, score=min(1.0, area / 400)))
    blobs.sort(key=lambda b: -b.area)
    return blobs

"""Per-episode scene context: sheet ROI per camera, camera setup, pixel -> table homographies.

The table frame JSON (written by scripts/perception_table_frame.py) holds one homography for the
top camera (fixed for the whole dataset) and one per floor-camera setup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .color_cv import Blob, detect_blobs
from .geometry import apply_h
from .sheet import detect_floor_lines, detect_top_lines


def half_plane(shape, line, positive_side_point) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    a, b, c = line
    s = np.sign(a * positive_side_point[0] + b * positive_side_point[1] + c)
    return s * (a * xx + b * yy + c) > 2.0


def sheet_roi(cam: str, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Pixels on the sheet: below its far edge (and right of the left edge for the top camera)."""
    h, w = rgb.shape[:2]
    if cam == "top":
        lines = detect_top_lines(rgb)
        roi = half_plane(rgb.shape, lines["far"], (w / 2, h - 1)) & half_plane(rgb.shape, lines["left"], (w - 1, h / 2))
    else:
        lines = detect_floor_lines(rgb)
        roi = half_plane(rgb.shape, lines["far"], (w / 2, h - 1))
    # and inside the convex hull of the white sheet (drops wood beyond the side edges)
    white = lines.pop("mask").astype(np.uint8)
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(white)
    if cnts:
        cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
        hull = cv2.erode(hull, np.ones((5, 5), np.uint8))
    return roi & hull.astype(bool), lines


def _components_hull(mask: np.ndarray, min_area: int, dilate: int, hull: bool,
                     border_only: bool = False, border_min_area: int | None = None) -> np.ndarray:
    """Union of components with area >= min_area (or >= border_min_area if they touch the image
    border); with border_only, only components touching the border count."""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros(mask.shape, np.uint8)
    H, W = mask.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        border = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        if border_only and not border:
            continue
        if area < min_area and not (border and border_min_area is not None and area >= border_min_area):
            continue
        comp = (lab == i).astype(np.uint8)
        if hull:
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.fillPoly(out, [cv2.convexHull(np.vstack(cnts))], 1)
        else:
            out |= comp
    if dilate:
        out = cv2.dilate(out, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return out.astype(bool)


def exclusion_mask(rgb: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
    """Static distractors in an episode's first frame: robot body (dark purple), lavender bowl,
    green bowl. Detections whose bottom-centre falls inside are ignored for the whole episode
    (bricks already in a bowl, the robot's orange cable clip, ...). With `roi` (sheet ROI), robot
    candidates are limited to 20 px around it, so dark clutter behind the table is not merged in."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(np.int32) for i in range(3))
    robot = (h >= 106) & (h <= 130) & (s >= 70) & (v <= 75)
    if roi is not None:
        robot &= cv2.dilate(roi.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    lavender = (h >= 106) & (h <= 130) & (s >= 40) & (s < 110) & (v >= 75)
    green_bowl = (h >= 60) & (h <= 90) & (s >= 70)
    # green bowl cut by the left/right image edge: its rim looks blue-green (hue up to ~94);
    # turquoise bricks start at hue ~95, and only the outer 60 px columns are searched
    bowl_rim = (h >= 60) & (h <= 94) & (s >= 70)
    bowl_rim[:, 60:-60] = False
    return (_components_hull(robot, 800, 25, False, border_only=True)
            | _components_hull(lavender, 1500, 6, True)
            | _components_hull(green_bowl, 1500, 10, True)
            | _components_hull(bowl_rim, 200, 10, True, border_only=True))


def detect_bricks_cv(rgb: np.ndarray, roi: np.ndarray, exclude: np.ndarray | None = None,
                     min_area: int = 250) -> list[Blob]:
    out = []
    h, w = rgb.shape[:2]
    for b in detect_blobs(rgb, roi):
        x0, y0, x1, y1 = b.bbox
        if b.area < min_area or (x1 - x0) < 8 or (y1 - y0) < 8:
            continue
        # a large blob cut by the side or bottom image border is a hand or arm, not a brick
        if b.area > 5000 and (x0 <= 1 or x1 >= w - 1 or y1 >= h - 1):
            continue
        if exclude is not None:
            bx, by = b.bottom_center
            if exclude[min(h - 1, int(by)), min(w - 1, int(bx))]:
                continue
        out.append(b)
    return out


@dataclass
class Frame:
    """Loaded table-frame JSON."""

    data: dict

    @classmethod
    def load(cls, path: str | Path) -> "Frame":
        return cls(json.loads(Path(path).read_text()))

    def setup_of(self, cam: str, episode: int) -> str:
        for name, c in self.data["cameras"].items():
            if c["camera"] == cam and c["episodes"][0] <= episode <= c["episodes"][1]:
                return name
        raise KeyError(f"no {cam} camera setup for episode {episode}; rerun perception_table_frame.py")

    def H(self, cam: str, episode: int) -> np.ndarray:
        return np.array(self.data["cameras"][self.setup_of(cam, episode)]["H_px_to_table_mm"])

    def to_table(self, cam: str, episode: int, pts) -> np.ndarray:
        return apply_h(self.H(cam, episode), pts)

    def to_uv(self, xy_mm) -> np.ndarray:
        c = self.data["canvas"]
        xy = np.atleast_2d(np.asarray(xy_mm, float))
        return np.c_[(xy[:, 0] - c["x_min_mm"]) / (c["x_max_mm"] - c["x_min_mm"]),
                     (xy[:, 1] - c["y_min_mm"]) / (c["y_max_mm"] - c["y_min_mm"])]

    def footprint_center(self, cam: str, episode: int, bottom_center_px, depth_mm: float = 16.0) -> np.ndarray:
        """Table point of an object's footprint centre from its image bottom-centre: the bottom-centre
        is the footprint edge nearest the camera, so step `depth_mm` (half a 2x2 Duplo) away from
        the camera along the ground direction of the viewing ray."""
        p = self.to_table(cam, episode, [bottom_center_px])[0]
        cam_xy = np.array(self.data["cameras"][self.setup_of(cam, episode)]["camera_xy_mm"])
        d = p - cam_xy
        return p + depth_mm * d / np.linalg.norm(d)

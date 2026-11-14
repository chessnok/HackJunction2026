"""Fixed crop boxes and the OpenCV quality check.

The same crop function is used when preparing the training data and at runtime in the robot
loop, so training and runtime images are framed identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle in pixels: x, y = top-left corner."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_list(cls, v: list[int] | tuple[int, ...]) -> Box:
        x, y, w, h = (int(a) for a in v)
        return cls(x, y, w, h)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def expand(self, margin: float, frame_w: int, frame_h: int) -> Box:
        """Grow by `margin` × size on every side, clipped to the frame."""
        dx, dy = int(self.w * margin), int(self.h * margin)
        x0, y0 = max(self.x - dx, 0), max(self.y - dy, 0)
        x1, y1 = min(self.x + self.w + dx, frame_w), min(self.y + self.h + dy, frame_h)
        return Box(x0, y0, x1 - x0, y1 - y0)

    def relative_to(self, outer: Box) -> Box:
        return Box(self.x - outer.x, self.y - outer.y, self.w, self.h)


def cut(frame: np.ndarray, box: Box) -> np.ndarray:
    return frame[box.y : box.y + box.h, box.x : box.x + box.w]


def crop(frame: np.ndarray, box: Box, size: int = 256) -> np.ndarray:
    """Fixed-box crop resized to size×size: the model input, at training and at runtime."""
    return cv2.resize(cut(frame, box), (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class CheckResult:
    ok: bool
    reason: str
    brick_fraction: float
    sharpness: float


def check_view(frame_bgr: np.ndarray, box: Box, cfg: dict) -> CheckResult:
    """Is exactly one brick fully inside the box, and is the image sharp?"""
    roi = cut(frame_bgr, box)
    if roi.size == 0:
        return CheckResult(False, "crop box outside the frame", 0.0, 0.0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = (sat >= cfg["min_saturation"]).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fraction = float(mask.mean())

    if fraction < cfg["min_brick_fraction"]:
        return CheckResult(False, "no brick in the box", fraction, sharpness)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return CheckResult(False, "no brick in the box", fraction, sharpness)
    big = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h = (int(stats[big, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                               cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    b = cfg["border_px"]
    if x <= b or y <= b or x + w >= box.w - b or y + h >= box.h - b:
        return CheckResult(False, "something touches the box border (hand? off-centre?)",
                           fraction, sharpness)
    if sharpness < cfg["min_sharpness"]:
        return CheckResult(False, f"blurred (sharpness {sharpness:.0f})", fraction, sharpness)
    return CheckResult(True, "OK", fraction, sharpness)

"""Shared test helpers: a synthetic wrist-camera scene (no real camera frames in the repo)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import cv2
except ImportError:  # IK / path tests run without OpenCV; image tests skip themselves
    cv2 = None

SAMPLE_IMAGE = ("/home/intel/.local/share/physicalai/state_snapshots/0fd76ff2-a6b1-4472-b687-486befe329be/"
                "20261114-055500-789/arm.color.png")

# OpenCV hue per colour name (S = V = 200 in the synthetic scene)
HUES = {"red": 3, "orange": 12, "yellow": 22, "lime": 38, "green": 65, "blue": 105}
SHEET = [(290, 270), (790, 255), (1023, 640), (100, 560)]


def hsv_bgr(h: int, s: int = 200, v: int = 200) -> tuple[int, int, int]:
    px = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in px)


def make_scene(bricks: dict[str, tuple[float, float]], angle: float = 20.0, side: int = 40,
               off_sheet: bool = True) -> np.ndarray:
    """1024x768 BGR: wooden table, white sheet, purple gripper jaws + white gripper body at the bottom,
    a green plate in the bottom-right corner, one red brick on the wood, and the given bricks
    (colour -> pixel centre) as rotated squares on the sheet."""
    img = np.zeros((768, 1024, 3), np.uint8)
    img[:] = hsv_bgr(16, 230, 110)  # wood
    cv2.fillPoly(img, [np.array(SHEET, np.int32)], (222, 218, 219))
    purple = hsv_bgr(134, 150, 130)
    cv2.fillPoly(img, [np.array([(0, 767), (0, 640), (380, 430), (405, 440), (420, 767)], np.int32)], purple)
    cv2.fillPoly(img, [np.array([(560, 767), (560, 640), (870, 540), (930, 560), (930, 767)], np.int32)], purple)
    cv2.rectangle(img, (420, 655), (560, 767), (200, 200, 200), -1)  # white gripper body, touches the sheet
    cv2.circle(img, (990, 730), 130, hsv_bgr(67, 180, 90), -1)   # green plate
    if off_sheet:
        _square(img, (985, 380), 40, 10.0, hsv_bgr(HUES["red"]))
    for colour, c in bricks.items():
        _square(img, c, side, angle, hsv_bgr(HUES[colour]))
    return img


def _square(img, centre, side, angle, colour):
    box = cv2.boxPoints(((float(centre[0]), float(centre[1])), (side, side), angle))
    cv2.fillPoly(img, [np.round(box).astype(np.int32)], colour)


@pytest.fixture
def scene():
    if cv2 is None:
        pytest.skip("OpenCV: uv run --extra table pytest")
    return make_scene

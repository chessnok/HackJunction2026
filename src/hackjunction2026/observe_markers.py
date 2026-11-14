"""Fixed black marks on the sheet, seen from the ``observe`` pose, correct the wrist camera for small pose changes.

The table homography is valid for the camera pose it was fitted in. The arm never returns to ``observe`` exactly
(the shoulder sags into its stop), and a 1 deg camera turn moves a table point by ~7 mm. Four black dots near the
corners of the view are recorded once (``record``) right after the calibration; at run time the dots are found
again and the homography current -> reference image is applied to the brick pixel before the table mapping.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MARKERS_YAML = REPO / "configs/observe_markers.yaml"
MATCH_PX = 70.0


def find_dots(frame_bgr: np.ndarray, area=(150, 6000)) -> np.ndarray:
    """Centroids (N, 2) of dark, compact blobs: the black marks. Bricks are coloured, the sheet is light."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    dark = ((lab[..., 0] < 70) & (chroma < 25)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(dark)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if area[0] <= a <= area[1] and max(w, h) <= 2.5 * max(min(w, h), 1) and a >= 0.4 * w * h:
            out.append(cen[k])
    return np.array(out, float).reshape(-1, 2)


def record(frame_bgr: np.ndarray, expected: list[tuple[float, float]], path: Path = MARKERS_YAML) -> np.ndarray:
    """Store the dot nearest each expected position (rough pixel spots) as the reference."""
    dots = find_dots(frame_bgr)
    ref = []
    for e in expected:
        if not len(dots):
            raise ValueError("no dark dots found")
        d = np.hypot(*(dots - np.asarray(e, float)).T)
        i = int(np.argmin(d))
        if d[i] > 150:
            raise ValueError(f"no dot near {e} (nearest {d[i]:.0f} px)")
        ref.append(dots[i])
    ref = np.array(ref)
    path.write_text(yaml.safe_dump({"description": "Black reference dots seen from observe (src/hackjunction2026/"
                                    "observe_markers.py); recorded with the table homography's pose.",
                                    "reference_px": ref.round(2).tolist()}, sort_keys=False))
    return ref


def load(path: Path = MARKERS_YAML) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    return np.array(yaml.safe_load(Path(path).read_text())["reference_px"], float)


def correction(frame_bgr: np.ndarray, ref: np.ndarray | None = None) -> tuple[np.ndarray | None, int]:
    """3x3 map current pixel -> reference pixel from the matched dots (None if fewer than 3 match)."""
    ref = load() if ref is None else ref
    if ref is None:
        return None, 0
    dots = find_dots(frame_bgr)
    cur, rr = [], []
    for r in ref:
        if not len(dots):
            break
        d = np.hypot(*(dots - r).T)
        i = int(np.argmin(d))
        if d[i] <= MATCH_PX:
            cur.append(dots[i])
            rr.append(r)
    if len(cur) == 4:
        return cv2.getPerspectiveTransform(np.float32(cur), np.float32(rr)), 4
    if len(cur) == 3:
        A = cv2.getAffineTransform(np.float32(cur), np.float32(rr))
        return np.vstack([A, [0, 0, 1]]), 3
    return None, len(cur)


def apply(M: np.ndarray | None, px) -> np.ndarray:
    p = np.asarray(px, float).reshape(-1, 2)
    if M is None:
        return p
    q = np.hstack([p, np.ones((len(p), 1))]) @ M.T
    return q[:, :2] / q[:, 2:3]

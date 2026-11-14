"""Find coloured bricks in a camera frame and cut a fixed-size, brick-centred crop.

Shared by dataset building (scripts/brick_crops.py) and runtime inference
(scripts/brick_anomaly.py), so training and runtime crops are made by the same code.

Scene assumptions (dataset 431a71f7, state snapshots of 2026-11-14): saturated bricks on a white
sheet; purple gripper and robot base; a dark-green plate and a lavender plate at the sheet border;
wooden table, cables and people outside the sheet. A brick is a saturated, non-purple blob that
does not touch the image border and is mostly surrounded by white sheet (or by the purple gripper,
when the brick is held).

The brick mask is the convex hull of the blob, so dark marker strokes inside the brick (which are
not saturated) stay inside the mask and survive background whitening.

Dependencies: numpy, opencv only (runs in anomalib-env and in Studio's backend env).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

# OpenCV hue is 0..179. Bins used to name the brick colour.
COLOUR_BINS = (
    ("red", 0, 7),
    ("orange", 7, 14),
    ("yellow", 14, 30),
    ("green", 30, 90),
    ("blue", 90, 120),
    ("red", 165, 180),
)


@dataclass
class SegConfig:
    sat_min: int = 80  # brick pixels: saturation >= this (lime brick is about 105, sheet < 20)
    val_min: int = 45  # ... and value >= this (wood in shade and dark base are below or near)
    purple_lo: int = 120  # gripper / robot base hue range (excluded from bricks)
    purple_hi: int = 165
    white_sat_max: int = 45  # sheet pixels: low saturation, not dark
    white_val_min: int = 80
    min_side_px: int = 18  # bricks smaller than this (bbox max side) are too small to inspect
    max_area_frac: float = 0.20  # larger blobs are plates / arm parts, not bricks
    min_fill: float = 0.45  # blob area / hull area: rejects thin curved things (plate rims, cables)
    ring_px_frac: float = 0.25  # ring width around the blob, fraction of sqrt(blob area)
    min_ring_white: float = 0.30  # ring must be at least this much white sheet
    min_ring_ok: float = 0.65  # ... and this much white + purple gripper + other brick colour
    border_px: int = 2  # blobs touching the image border are rejected (cut-off bricks)
    close_px: int = 5  # morphological closing to merge two-tone bricks and stud gaps


@dataclass
class Brick:
    bbox: tuple[int, int, int, int]  # x, y, w, h of the hull, full-frame pixels
    area: int  # hull area in pixels
    colour: str  # dominant colour name, e.g. "yellow" or "red+blue"
    ring_white: float
    ring_ok: float
    fill: float
    hull: np.ndarray = field(repr=False)  # (N, 1, 2) int32 contour, full-frame pixels

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("hull")
        d["bbox"] = list(self.bbox)
        return d


def _colour_name(hue: np.ndarray) -> str:
    if hue.size == 0:
        return "unknown"
    counts: dict[str, int] = {}
    for name, lo, hi in COLOUR_BINS:
        counts[name] = counts.get(name, 0) + int(((hue >= lo) & (hue < hi)).sum())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    total = hue.size
    first, n1 = ranked[0]
    second, n2 = ranked[1]
    if n2 >= 0.25 * total and n2 >= 0.4 * n1:
        return "+".join(sorted([first, second]))
    return first


def find_bricks(frame_bgr: np.ndarray, cfg: SegConfig | None = None) -> list[Brick]:
    """All brick candidates in one frame, largest first."""
    cfg = cfg or SegConfig()
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    purple = (hue >= cfg.purple_lo) & (hue < cfg.purple_hi) & (sat >= 60)
    colour = (sat >= cfg.sat_min) & (val >= cfg.val_min) & ~purple
    white = (sat <= cfg.white_sat_max) & (val >= cfg.white_val_min)

    mask = colour.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_px, cfg.close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bricks: list[Brick] = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if max(bw, bh) < cfg.min_side_px or area > cfg.max_area_frac * h * w:
            continue
        b = cfg.border_px
        if x <= b or y <= b or x + bw >= w - b or y + bh >= h - b:
            continue
        comp = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8)
        pts = cv2.findNonZero(comp)
        hull = cv2.convexHull(pts) + np.array([[x, y]], dtype=np.int32)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        fill = area / hull_area
        if fill < cfg.min_fill:
            continue

        # Ring of pixels around the hull: what does the brick lie on?
        r = max(3, int(cfg.ring_px_frac * np.sqrt(area)))
        x0, y0 = max(x - 2 * r, 0), max(y - 2 * r, 0)
        x1, y1 = min(x + bw + 2 * r, w), min(y + bh + 2 * r, h)
        hull_local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(hull_local, hull - np.array([[x0, y0]], dtype=np.int32), 1)
        near = cv2.dilate(hull_local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        far = cv2.dilate(hull_local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        ring = (far > 0) & (near == 0)
        n_ring = int(ring.sum())
        if n_ring == 0:
            continue
        ring_white = float(white[y0:y1, x0:x1][ring].mean())
        ok = white[y0:y1, x0:x1] | purple[y0:y1, x0:x1] | colour[y0:y1, x0:x1]
        ring_ok = float(ok[ring].mean())
        if ring_white < cfg.min_ring_white or ring_ok < cfg.min_ring_ok:
            continue

        name = _colour_name(hue[y : y + bh, x : x + bw][comp > 0])
        hx, hy, hw_, hh = cv2.boundingRect(hull)
        bricks.append(
            Brick(
                bbox=(hx, hy, hw_, hh),
                area=int(hull_area),
                colour=name,
                ring_white=round(ring_white, 3),
                ring_ok=round(ring_ok, 3),
                fill=round(fill, 3),
                hull=hull,
            )
        )
    bricks.sort(key=lambda b: -b.area)
    return bricks


@dataclass
class CropConfig:
    size: int = 256  # model input side
    margin: float = 0.15  # extra space around the brick, fraction of its larger side, per side
    whiten: bool = True  # replace everything outside the brick mask with `fill`
    fill: tuple[int, int, int] = (255, 255, 255)
    feather_px: int = 2  # soft mask edge (in crop pixels) when whitening


def crop_brick(
    frame_bgr: np.ndarray, brick: Brick, cfg: CropConfig | None = None
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Square crop centred on the brick, resized to cfg.size.

    Returns (crop_bgr, mask, square) where mask is the uint8 (0/255) brick hull in crop pixels and
    square is the x, y, side of the square in frame pixels (may extend past the frame; padded).
    """
    cfg = cfg or CropConfig()
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = brick.bbox
    side = int(round(max(bw, bh) * (1 + 2 * cfg.margin)))
    cx, cy = x + bw / 2, y + bh / 2
    sx, sy = int(round(cx - side / 2)), int(round(cy - side / 2))

    pad_val = cfg.fill if cfg.whiten else (0, 0, 0)
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = pad_val
    fx0, fy0, fx1, fy1 = max(sx, 0), max(sy, 0), min(sx + side, w), min(sy + side, h)
    canvas[fy0 - sy : fy1 - sy, fx0 - sx : fx1 - sx] = frame_bgr[fy0:fy1, fx0:fx1]
    if not cfg.whiten:
        # Edge-replicate instead of black bars when the square leaves the frame.
        top, left = fy0 - sy, fx0 - sx
        bottom, right = side - (fy1 - sy), side - (fx1 - sx)
        if top or left or bottom or right:
            inner = frame_bgr[fy0:fy1, fx0:fx1]
            canvas = cv2.copyMakeBorder(inner, top, bottom, left, right, cv2.BORDER_REPLICATE)

    mask = np.zeros((side, side), np.uint8)
    cv2.fillConvexPoly(mask, brick.hull - np.array([[sx, sy]], dtype=np.int32), 255)

    interp = cv2.INTER_AREA if side > cfg.size else cv2.INTER_CUBIC
    crop = cv2.resize(canvas, (cfg.size, cfg.size), interpolation=interp)
    mask = cv2.resize(mask, (cfg.size, cfg.size), interpolation=cv2.INTER_NEAREST)

    if cfg.whiten:
        soft = mask.astype(np.float32) / 255.0
        if cfg.feather_px > 0:
            k = 2 * cfg.feather_px + 1
            soft = cv2.GaussianBlur(soft, (k, k), 0)
        bg = np.empty_like(crop)
        bg[:] = cfg.fill
        crop = (crop.astype(np.float32) * soft[..., None] + bg.astype(np.float32) * (1 - soft[..., None]))
        crop = np.clip(crop + 0.5, 0, 255).astype(np.uint8)
    return crop, mask, (sx, sy, side)


def sharpness(crop_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian (inside the mask if given): low = blurred."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if mask is not None:
        inner = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
        if inner.sum() > 50:
            return float(lap[inner].var())
    return float(lap.var())


def draw_bricks(frame_bgr: np.ndarray, bricks: list[Brick]) -> np.ndarray:
    """Debug overlay: hull and colour label of each brick."""
    out = frame_bgr.copy()
    for b in bricks:
        cv2.polylines(out, [b.hull], True, (0, 0, 255), 2)
        x, y, _, _ = b.bbox
        cv2.putText(out, b.colour, (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out

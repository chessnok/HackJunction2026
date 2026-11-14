"""Find coloured bricks on the white sheet and cut a zoomed, brick-centred square crop.

Shared by dataset building (cropping.brick_dataset / scripts/brick_crops.py) and runtime inference
(scripts/brick_anomaly.py), so training and runtime crops are made by the same code.

Scene (dataset 431a71f7, state snapshots of 2026-11-14): coloured bricks on a white sheet; purple
gripper and robot base; a dark-green and a lavender plate; a wooden table, cables and people
outside the sheet. Rules:

- Foreground = Lab chroma >= chroma_min (sheet 0-3, pale lime brick 17-22, dark olive 10-23).
  HSV saturation missed the lime and olive bricks in the top view.
- Purple hues (gripper, base, lavender plate) are never brick.
- Only inside the sheet: the white region (gripper counts as sheet support, so a held brick
  stays inside) with its holes filled. The wooden table matches yellow/orange bricks.
- Per-camera exclusion zones by blob centroid: the gripper screws in the wrist view.
- The brick mask is the convex hull of the blob, so a dark marker stroke inside the brick (not
  coloured) stays inside the mask and survives background whitening.

Crop (ADR 0008): square of `zoom` x the brick's larger side, centred on the brick, cut from the
full-resolution frame and resized to `size`.

Dependencies: numpy, opencv only (runs in the project env and in Studio's backend env).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

# OpenCV hue is 0..179. Bins used to name the brick colour.
COLOUR_BINS = (
    ("red", 0, 7),
    ("orange", 7, 14),
    ("yellow", 14, 27),
    ("green", 27, 90),
    ("blue", 90, 112),
    ("red", 165, 180),
)

# Normalized (x0, y0, x1, y1) boxes per camera: blobs whose centroid falls inside are ignored.
# arm: the gripper base and its screws (fixed in the wrist view). A held brick's centroid is higher.
EXCLUDE = {"arm": [(0.25, 0.80, 0.75, 1.0)]}


@dataclass
class SegConfig:
    chroma_min: float = 9.0  # brick pixels: Lab chroma >= this
    light_min: int = 15  # ... and Lab L >= this (not black noise)
    purple_lo: int = 112  # purple hue range (gripper 131, base 115, lavender plate 117)
    purple_hi: int = 165
    purple_sat: int = 45
    white_chroma_max: float = 7.0  # sheet pixels: low chroma, not dark
    white_light_min: int = 60
    sheet_close_px: int = 15  # bridge the sheet crease and thin shadows
    sheet_min_frac: float = 0.2  # keep sheet parts >= this x the largest part
    min_in_sheet: float = 0.9  # hull share inside the (hole-filled) sheet region
    min_side_px: int = 30  # bricks smaller than this (bbox max side) are too small to judge a mark
    max_aspect: float = 3.5  # bbox long / short side (finger-edge slivers are long and thin)
    min_median_chroma: float = 13.0  # glare on the glossy sheet is yellowish but weak (bricks >= 16)
    max_area_frac: float = 0.20  # larger blobs are plates / arm parts, not bricks
    min_fill: float = 0.45  # blob area / hull area: rejects thin curved things (plate rims, cables)
    ring_px_frac: float = 0.25  # ring width around the blob, fraction of sqrt(blob area)
    min_ring_white: float = 0.5  # ring must be at least this much white sheet
    min_ring_ok: float = 0.65  # ... and this much white + purple gripper + other brick colour
    border_px: int = 2  # blobs touching the image border are rejected (cut-off bricks)
    close_px: int = 5  # morphological closing to merge two-tone bricks and stud gaps


@dataclass
class Brick:
    bbox: tuple[int, int, int, int]  # x, y, w, h of the hull, full-frame pixels
    area: int  # hull area in pixels
    colour: str  # dominant colour name, e.g. "yellow" or "blue+red"
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
    (first, n1), (second, n2) = ranked[0], ranked[1]
    if n2 >= 0.25 * hue.size and n2 >= 0.4 * n1:
        return "+".join(sorted([first, second]))
    return first


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """uint8 0/1 mask with every hole (region not connected to the image border) filled."""
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    flood = pad.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return (mask.astype(bool) | holes).astype(np.uint8)


def sheet_region(white: np.ndarray, support: np.ndarray, cfg: SegConfig) -> np.ndarray:
    """uint8 0/1: the white sheet (+ gripper) with holes filled, i.e. where bricks may lie."""
    m = (white | support).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.sheet_close_px, cfg.sheet_close_px))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= cfg.sheet_min_frac * areas.max())[0] + 1
    return _fill_holes(np.isin(labels, keep).astype(np.uint8))


def find_bricks(frame_bgr: np.ndarray, cfg: SegConfig | None = None, camera: str | None = None,
                debug: dict | None = None) -> list[Brick]:
    """All bricks on the sheet in one frame, largest first. `camera` enables its exclusion zones."""
    cfg = cfg or SegConfig()
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    hue, sat = hsv[..., 0], hsv[..., 1]
    light = lab[..., 0]
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)

    purple = (hue >= cfg.purple_lo) & (hue < cfg.purple_hi) & (sat >= cfg.purple_sat)
    colour = (chroma >= cfg.chroma_min) & (light >= cfg.light_min) & ~purple
    white = (chroma <= cfg.white_chroma_max) & (light >= cfg.white_light_min)
    sheet = sheet_region(white, purple, cfg)
    if debug is not None:
        debug.update(colour=colour, white=white, purple=purple, sheet=sheet, rejected=[])

    mask = (colour & (sheet > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_px, cfg.close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    zones = [(int(a * w), int(b * h), int(c * w), int(d * h)) for a, b, c, d in EXCLUDE.get(camera or "", [])]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bricks: list[Brick] = []

    def reject(i: int, why: str) -> None:
        if debug is not None:
            debug["rejected"].append((tuple(int(v) for v in stats[i][:4]), why))

    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if max(bw, bh) < cfg.min_side_px:
            continue
        if max(bw, bh) > cfg.max_aspect * max(min(bw, bh), 1):
            reject(i, "thin")
            continue
        if area > cfg.max_area_frac * h * w:
            reject(i, "too large")
            continue
        b = cfg.border_px
        if x <= b or y <= b or x + bw >= w - b or y + bh >= h - b:
            reject(i, "touches border")
            continue
        cx, cy = cents[i]
        if any(x0 <= cx < x1 and y0 <= cy < y1 for x0, y0, x1, y1 in zones):
            reject(i, "exclusion zone")
            continue
        comp = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8)
        hull = cv2.convexHull(cv2.findNonZero(comp)) + np.array([[x, y]], dtype=np.int32)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        fill = area / hull_area
        if fill < cfg.min_fill:
            reject(i, f"fill {fill:.2f}")
            continue
        med_chroma = float(np.median(chroma[y : y + bh, x : x + bw][comp > 0]))
        if med_chroma < cfg.min_median_chroma:
            reject(i, f"pale {med_chroma:.0f}")
            continue

        # Ring of pixels around the hull: what does the brick lie on?
        r = max(3, int(cfg.ring_px_frac * np.sqrt(area)))
        x0, y0 = max(x - 2 * r, 0), max(y - 2 * r, 0)
        x1, y1 = min(x + bw + 2 * r, w), min(y + bh + 2 * r, h)
        hull_local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(hull_local, hull - np.array([[x0, y0]], dtype=np.int32), 1)
        in_sheet = float(sheet[y0:y1, x0:x1][hull_local > 0].mean())
        if in_sheet < cfg.min_in_sheet:
            reject(i, f"off sheet {in_sheet:.2f}")
            continue
        near = cv2.dilate(hull_local, np.ones((3, 3), np.uint8))
        far = cv2.dilate(hull_local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        ring = (far > 0) & (near == 0)
        if not ring.any():
            continue
        ring_white = float(white[y0:y1, x0:x1][ring].mean())
        ok = white[y0:y1, x0:x1] | purple[y0:y1, x0:x1] | colour[y0:y1, x0:x1]
        ring_ok = float(ok[ring].mean())
        if ring_white < cfg.min_ring_white or ring_ok < cfg.min_ring_ok:
            reject(i, f"ring white {ring_white:.2f} ok {ring_ok:.2f}")
            continue

        name = _colour_name(hue[y : y + bh, x : x + bw][comp > 0])
        hx, hy, hw_, hh = cv2.boundingRect(hull)
        bricks.append(Brick(bbox=(hx, hy, hw_, hh), area=int(hull_area), colour=name,
                            ring_white=round(ring_white, 3), ring_ok=round(ring_ok, 3), fill=round(fill, 3),
                            hull=hull))
    bricks.sort(key=lambda b: -b.area)
    return bricks


@dataclass
class CropConfig:
    size: int = 256  # output side in pixels (model input)
    zoom: float = 2.4  # square side = zoom x the brick's larger bbox side (ADR 0008)
    whiten: bool = True  # replace everything outside the (dilated) brick hull with `fill`
    fill: tuple[int, int, int] = (255, 255, 255)
    hull_grow: float = 0.04  # dilate the hull by this x the brick side before whitening
    feather_px: int = 2  # soft mask edge (in output pixels) when whitening
    clahe: bool = False  # CLAHE on the Lab luminance of the final crop (contrast normalisation)


def clahe_luminance(bgr: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE on the Lab L channel: lifts faint dark marks on dark bricks, keeps the colours."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_square(brick: Brick, zoom: float) -> tuple[int, int, int]:
    """(x, y, side) of the crop square in frame pixels; may extend past the frame."""
    x, y, bw, bh = brick.bbox
    side = max(int(round(max(bw, bh) * zoom)), 8)
    return int(round(x + bw / 2 - side / 2)), int(round(y + bh / 2 - side / 2)), side


def crop_brick(frame_bgr: np.ndarray, brick: Brick, cfg: CropConfig | None = None
               ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Square, brick-centred crop resized to cfg.size.

    Returns (crop_bgr, mask, (x, y, side)): mask is the uint8 0/255 brick hull in output pixels;
    (x, y, side) is the square in frame pixels. Parts outside the frame are padded (white when
    whitening, edge-replicated otherwise).
    """
    cfg = cfg or CropConfig()
    h, w = frame_bgr.shape[:2]
    sx, sy, side = crop_square(brick, cfg.zoom)
    fx0, fy0, fx1, fy1 = max(sx, 0), max(sy, 0), min(sx + side, w), min(sy + side, h)
    inner = frame_bgr[fy0:fy1, fx0:fx1]
    top, left = fy0 - sy, fx0 - sx
    bottom, right = side - top - inner.shape[0], side - left - inner.shape[1]
    border = cv2.BORDER_CONSTANT if cfg.whiten else cv2.BORDER_REPLICATE
    canvas = cv2.copyMakeBorder(inner, top, bottom, left, right, border, value=cfg.fill)

    mask = np.zeros((side, side), np.uint8)
    cv2.fillConvexPoly(mask, brick.hull - np.array([[sx, sy]], dtype=np.int32), 255)
    grow = int(round(cfg.hull_grow * max(brick.bbox[2], brick.bbox[3])))
    if grow > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))

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
        a = soft[..., None]
        crop = np.clip(crop.astype(np.float32) * a + bg.astype(np.float32) * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    if cfg.clahe:
        crop = clahe_luminance(crop)
    return crop, mask, (sx, sy, side)


def sharpness(crop_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian (inside the eroded mask if given): low = blurred."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if mask is not None:
        inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
        if inner.sum() > 50:
            return float(lap[inner].var())
    return float(lap.var())


def draw_bricks(frame_bgr: np.ndarray, bricks: list[Brick], camera: str | None = None,
                zoom: float | None = None) -> np.ndarray:
    """Debug overlay: hull, colour label, crop square and exclusion zones."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for a, b, c, d in EXCLUDE.get(camera or "", []):
        cv2.rectangle(out, (int(a * w), int(b * h)), (int(c * w) - 1, int(d * h) - 1), (128, 128, 128), 1)
    for br in bricks:
        cv2.polylines(out, [br.hull], True, (0, 0, 255), 2)
        if zoom:
            sx, sy, side = crop_square(br, zoom)
            cv2.rectangle(out, (sx, sy), (sx + side, sy + side), (0, 200, 0), 1)
        x, y, _, _ = br.bbox
        cv2.putText(out, br.colour, (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out

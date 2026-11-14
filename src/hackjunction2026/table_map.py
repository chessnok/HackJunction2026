"""Bricks on the white sheet in the wrist camera image, and image pixel -> table XY (mm).

Valid only for one fixed arm pose (``observe`` in ``configs/poses.yaml``): the wrist camera rides on
the gripper, so a pixel maps to one table point only while the arm stands exactly in that pose. The
mapping is a plane-to-plane homography fitted from bricks whose table position is known from forward
kinematics (the operator held the gripper over each brick, ``scripts/pick_above.py calib-points``).
Why a homography and not camera extrinsics: ``docs/decisions/0007-observe-pose-homography-ik.md``.

Image coordinates: pixels, u to the right, v down, origin at the top-left pixel centre (OpenCV).
Table coordinates: robot base frame of ``hackjunction2026.kinematics`` (mm, +x forward, +y left).

Detection (OpenCV, tuned on the sample image of 2026-11-14, 1024x768, Sonix USB2.0_CAM1):

1. Gripper zone: large purple components that touch the bottom border (the gripper jaws, always in
   the same place because the camera is mounted on the gripper); every pixel below the topmost
   purple pixel of a column belongs to the zone, and narrow horizontal gaps in it are closed (this
   covers the white gripper body between the jaws).
2. Sheet: largest connected bright, low-saturation region outside the gripper zone; its convex hull
   (simplified) is the sheet polygon. Its bottom edge is hidden by the gripper.
3. Bricks: saturated blobs inside the sheet polygon and outside the gripper zone, of brick size,
   compact, not touching the image border and not purple (the gripper colour). A blob whose hues
   form two clusters (two touching bricks of different colours) is split. Large things (the green
   plate) and things beyond the sheet (bricks on the wooden table) drop out. White, grey and black
   bricks are not found (no saturation).
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------------------------

# OpenCV hue (0..180) ranges -> colour name, for this warm camera. Measured on 107 Studio snapshots
# (2026-11-14): red 1-4, yellow 17-21, lime 37-45, blue 98-105; gripper purple 134, green plate 67,
# wooden table 14-17. Purple is listed so that it can be excluded.
HUE_NAMES: list[tuple[float, float, str]] = [
    (0, 7, "red"),
    (7, 14, "orange"),
    (14, 30, "yellow"),
    (30, 52, "lime"),
    (52, 85, "green"),
    (85, 125, "blue"),
    (125, 160, "purple"),
    (160, 168, "pink"),
    (168, 181, "red"),
]
EXCLUDED_COLOURS = frozenset({"purple"})


@dataclass(frozen=True)
class DetectParams:
    sheet_s_max: int = 40          # sheet: saturation below this ...
    sheet_v_min: int = 120         # ... and value above this
    brick_s_min: int = 90          # brick pixel: saturation at least ...
    brick_v_min: int = 60          # ... and value at least, or (pale bricks) ...
    pale_s_min: int = 55           # ... saturation at least (pale lime brick: 56-89, sheet ~15) ...
    pale_v_min: int = 130          # ... and bright (bluish shadow at a plate rim: S 59, V 76)
    gripper_s_min: int = 50        # purple pixel of the gripper
    gripper_min_area: int = 5000   # purple component must be this large and touch the bottom border
    gripper_pad: int = 8           # px, gripper zone grown by this much
    gripper_gap_fill: int = 151    # px, horizontal gaps narrower than this inside the zone are filled
    brick_area_min: int = 300      # px^2 (brick at the far edge of the sheet is ~25x25 px)
    brick_area_max: int = 15000    # px^2 (the green plate is far larger)
    brick_aspect_max: float = 3.5  # long / short side of the min-area rectangle
    brick_solidity_min: float = 0.6  # blob area / convex hull area (glare arcs on a plate rim are thin)
    hue_split: float = 25.0        # split a blob whose hues form two clusters this far apart (touching
                                   # bricks of different colours)
    sheet_margin: int = 4          # px, sheet polygon shrunk by this much for brick candidates


@dataclass
class Brick:
    colour: str
    centroid: tuple[float, float]  # (u, v) px
    angle_deg: float               # image direction of the long side, degrees in [-90, 90), v down
    rect_angle_deg: float          # raw cv2.minAreaRect angle
    size_px: tuple[float, float]   # (long, short) side of the min-area rectangle
    area_px: int                   # blob pixel count
    hue: float                     # median OpenCV hue (0..180)
    saturation: float
    value: float
    box: list[list[float]] = field(default_factory=list)  # 4 corners of the min-area rectangle
    table_xy: tuple[float, float] | None = None            # mm, filled by locate()
    long_axis_yaw_deg: float | None = None                  # table frame, filled by locate()

    @property
    def aspect(self) -> float:
        return self.size_px[0] / max(self.size_px[1], 1e-6)


@dataclass
class Detection:
    image_size: tuple[int, int]            # (width, height)
    sheet_polygon: np.ndarray | None       # (k, 2) int px, convex
    sheet_mask: np.ndarray | None          # uint8 0/255, filled polygon
    gripper_zone: np.ndarray               # uint8 0/255
    bricks: list[Brick]
    rejected: list[tuple[Brick, str]]      # saturated blobs that are not bricks on the sheet, with reason


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - environment problem
        raise SystemExit("OpenCV missing: run with `uv run --extra table ...`") from e
    return cv2


def colour_name(hue: float) -> str:
    for lo, hi, name in HUE_NAMES:
        if lo <= hue < hi:
            return name
    return "red"


def _median_hue(h: np.ndarray) -> float:
    """Median hue on the 0..180 circle (red wraps around)."""
    h = h.astype(np.float64)
    shifted = np.where(h >= 90, h - 180, h)  # -90..90: red around 0
    if np.median(np.abs(shifted)) < 45:      # mostly reds/oranges/yellows: use the shifted scale
        m = float(np.median(shifted))
        return m + 180 if m < 0 else m
    return float(np.median(h))


def gripper_zone(hsv: np.ndarray, p: DetectParams = DetectParams()) -> np.ndarray:
    cv2 = _cv2()
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    purple = ((h >= 120) & (h < 165) & (s >= p.gripper_s_min) & (v >= 30)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(purple, connectivity=8)
    height = purple.shape[0]
    zone = np.zeros_like(purple)
    for i in range(1, n):
        _, y, _, hh, area = stats[i]
        if area >= p.gripper_min_area and y + hh >= height - 1:
            zone[lab == i] = 1
    # everything below the topmost gripper pixel of each column
    has = zone.any(axis=0)
    top = np.where(has, zone.argmax(axis=0), height)
    rows = np.arange(height)[:, None]
    zone = (rows >= top[None, :]).astype(np.uint8) * 255
    if p.gripper_gap_fill > 1:  # the white gripper body between the jaws has no purple above it
        zone = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, np.ones((1, p.gripper_gap_fill), np.uint8))
    if p.gripper_pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.gripper_pad + 1, 2 * p.gripper_pad + 1))
        zone = cv2.dilate(zone, k)
    return zone


def find_sheet(hsv: np.ndarray, zone: np.ndarray, p: DetectParams = DetectParams()):
    """(polygon (k, 2) int px or None, filled mask uint8) of the largest bright low-saturation region."""
    cv2 = _cv2()
    s, v = hsv[..., 1], hsv[..., 2]
    m = ((s < p.sheet_s_max) & (v > p.sheet_v_min) & (zone == 0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None, np.zeros_like(m)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < 0.05 * m.size:
        return None, np.zeros_like(m)
    comp = (lab == big).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    poly = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    mask = np.zeros_like(m)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
    return poly.astype(np.int32), mask


def _long_side_angle(box: np.ndarray) -> tuple[float, float, float]:
    """(angle of the long side in degrees in [-90, 90), long, short) from 4 box corners."""
    e1, e2 = box[1] - box[0], box[2] - box[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    e = e1 if l1 >= l2 else e2
    ang = float(np.degrees(np.arctan2(e[1], e[0])))
    ang = (ang + 90.0) % 180.0 - 90.0
    return ang, max(l1, l2), min(l1, l2)


def _split_by_hue(blob: np.ndarray, hue: np.ndarray, p: DetectParams) -> list[np.ndarray] | None:
    """Two blobs if the blob's hues form two clusters at least ``p.hue_split`` apart (two touching bricks
    of different colours), else None. 2-means on the hue circle."""
    cv2 = _cv2()
    ys, xs = np.nonzero(blob)
    z = np.exp(1j * hue[ys, xs].astype(np.float64) * (2 * np.pi / 180.0))
    c0 = z.mean()
    c0 = c0 / abs(c0) if abs(c0) > 1e-9 else 1.0 + 0j
    centres = np.array([c0, z[np.argmin((z * np.conj(c0)).real)]])
    for _ in range(10):
        lbl = np.argmax((z[:, None] * np.conj(centres[None, :])).real, axis=1)
        for k in range(2):
            if (lbl == k).any():
                m = z[lbl == k].mean()
                if abs(m) > 1e-9:
                    centres[k] = m / abs(m)
    sep = abs(np.angle(centres[0] * np.conj(centres[1]))) * 180.0 / (2 * np.pi)
    if sep < p.hue_split or min((lbl == 0).mean(), (lbl == 1).mean()) < 0.2:
        return None
    parts = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for k in range(2):
        sub = np.zeros(blob.shape, np.uint8)
        sub[ys[lbl == k], xs[lbl == k]] = 255
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        parts += [lab == j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= p.brick_area_min]
    return parts if len(parts) >= 2 else None


def _blobs(mask: np.ndarray, hue: np.ndarray, p: DetectParams, split: bool):
    """Boolean masks of the connected components of ``mask`` (>= brick_area_min), touching bricks of
    different colours split apart when ``split``."""
    cv2 = _cv2()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p.brick_area_min:
            continue  # noise, not worth reporting
        blob = lab == i
        if split and area <= 2 * p.brick_area_max:
            parts = _split_by_hue(blob, hue, p)
            if parts:
                yield from parts
                continue
        yield blob


def detect(image_bgr: np.ndarray, p: DetectParams = DetectParams()) -> Detection:
    """Sheet polygon and the bricks on it, in a wrist camera image taken in the observe pose."""
    cv2 = _cv2()
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    zone = gripper_zone(hsv, p)
    poly, sheet = find_sheet(hsv, zone, p)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colourful = ((s >= p.brick_s_min) & (v >= p.brick_v_min)) | ((s >= p.pale_s_min) & (v >= p.pale_v_min))
    sat = (colourful & (zone == 0)).astype(np.uint8) * 255
    sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Candidates are labelled inside the sheet only, so that a brick near the sheet edge does not merge
    # with the wooden table or a plate next to it; blobs outside are labelled separately for the report.
    inner = sheet
    if p.sheet_margin > 0:
        inner = cv2.erode(sheet, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * p.sheet_margin + 1,) * 2))
    on = cv2.bitwise_and(sat, inner)
    off = cv2.bitwise_and(sat, cv2.bitwise_not(inner))
    bricks: list[Brick] = []
    rejected: list[tuple[Brick, str]] = []
    for mask, on_sheet in ((on, True), (off, False)):
        for blob in _blobs(mask, h, p, split=on_sheet):
            ys, xs = np.nonzero(blob)
            area = int(xs.size)
            pts = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            ang, long_, short = _long_side_angle(box)
            hue = _median_hue(h[blob])
            hull_area = float(cv2.contourArea(cv2.convexHull(pts)))
            solidity = area / max(hull_area, 1.0)
            b = Brick(
                colour=colour_name(hue), centroid=(float(xs.mean()), float(ys.mean())), angle_deg=ang,
                rect_angle_deg=float(rect[2]), size_px=(long_, short), area_px=area, hue=hue,
                saturation=float(np.median(s[blob])), value=float(np.median(v[blob])),
                box=box.round(1).tolist(),
            )
            if not on_sheet:
                reason = "off the sheet"
            elif area > p.brick_area_max:
                reason = "too large"
            elif xs.min() <= 0 or ys.min() <= 0 or xs.max() >= width - 1 or ys.max() >= height - 1:
                reason = "touches the image border"
            elif b.colour in EXCLUDED_COLOURS:
                reason = f"excluded colour {b.colour}"
            elif b.aspect > p.brick_aspect_max:
                reason = f"too elongated ({b.aspect:.1f})"
            elif solidity < p.brick_solidity_min:
                reason = f"not compact (solidity {solidity:.2f})"
            else:
                bricks.append(b)
                continue
            rejected.append((b, reason))
    bricks.sort(key=lambda b: (b.centroid[1], b.centroid[0]))
    return Detection((width, height), poly, sheet, zone, bricks, rejected)


def annotate(image_bgr: np.ndarray, det: Detection, dim_outside_sheet: bool = True) -> np.ndarray:
    """Debug image: sheet polygon, gripper zone, bricks (green box) and rejected blobs (red text).

    Everything outside the sheet is darkened (people may be in the background of the wrist camera)."""
    cv2 = _cv2()
    vis = image_bgr.copy()
    if dim_outside_sheet and det.sheet_mask is not None:
        vis[det.sheet_mask == 0] = (vis[det.sheet_mask == 0] * 0.2).astype(np.uint8)
    if det.sheet_polygon is not None:
        cv2.polylines(vis, [det.sheet_polygon.reshape(-1, 1, 2)], True, (255, 200, 0), 2)
    edge = cv2.morphologyEx(det.gripper_zone, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    vis[edge > 0] = (255, 0, 255)
    def label(text: str, org: tuple[int, int], colour: tuple[int, int, int]) -> None:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = org
        cv2.rectangle(vis, (x - 2, y - th - 3), (x + tw + 2, y + base), (255, 255, 255), -1)
        cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    for b in det.bricks:
        cv2.polylines(vis, [np.array(b.box, np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        u, v = (round(c) for c in b.centroid)
        cv2.drawMarker(vis, (u, v), (0, 160, 0), cv2.MARKER_CROSS, 14, 2)
        text = b.colour
        if b.table_xy is not None:
            text += f" x {b.table_xy[0]:.0f} y {b.table_xy[1]:.0f} mm"
        label(text, (u + 14, v - 14), (0, 120, 0))
    for b, reason in det.rejected:
        if reason == "off the sheet" and b.area_px > DetectParams().brick_area_max:
            continue  # the table, people, clutter: not worth a label
        u, v = (round(c) for c in b.centroid)
        label(f"not a brick: {b.colour}, {reason}", (u - 40, v), (0, 0, 200))
    return vis


# ----------------------------------------------------------------------------------------------
# Homography pixel -> table XY
# ----------------------------------------------------------------------------------------------

def apply_homography(hmat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (..., 2) points with a 3x3 homography."""
    pts = np.asarray(pts, dtype=np.float64)
    ph = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1) @ np.asarray(hmat).T
    return ph[..., :2] / ph[..., 2:3]


def _normalizer(pts: np.ndarray) -> np.ndarray:
    """Hartley normalization: centroid to 0, mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    ts, td = _normalizer(src), _normalizer(dst)
    s = apply_homography(ts, src)
    d = apply_homography(td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    hn = vt[-1].reshape(3, 3)
    hmat = np.linalg.inv(td) @ hn @ ts
    return hmat / hmat[2, 2]


def _refine(hmat: np.ndarray, src: np.ndarray, dst: np.ndarray, iters: int = 20) -> np.ndarray:
    """Gauss-Newton on the transfer error in table mm (8 parameters, h33 = 1)."""
    if len(src) <= 4:
        return hmat
    p = (hmat / hmat[2, 2]).ravel()[:8].copy()

    def res(pp):
        return (apply_homography(np.append(pp, 1.0).reshape(3, 3), src) - dst).ravel()

    for _ in range(iters):
        r = res(p)
        jac = np.empty((r.size, 8))
        for k in range(8):
            dp = np.zeros(8)
            dp[k] = 1e-6 * max(1.0, abs(p[k]))
            jac[:, k] = (res(p + dp) - r) / dp[k]
        step, *_ = np.linalg.lstsq(jac, -r, rcond=None)
        p_new = p + step
        if (res(p_new) ** 2).sum() > (r ** 2).sum():
            break
        p = p_new
        if np.abs(step).max() < 1e-12:
            break
    return np.append(p, 1.0).reshape(3, 3)


def fit_homography(px: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Least-squares homography pixel -> table mm from >= 4 correspondences (DLT + refinement)."""
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    if len(px) < 4 or len(px) != len(xy):
        raise ValueError(f"need >= 4 pixel/table pairs, got {len(px)}/{len(xy)}")
    return _refine(_dlt(px, xy), px, xy)


def fit_homography_ransac(px: np.ndarray, xy: np.ndarray, threshold_mm: float = 15.0, iters: int = 500,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC over minimal 4-point sets (only with >= 6 points), then least squares on the inliers.

    Returns (H, inlier mask)."""
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    n = len(px)
    if n < 6:
        return fit_homography(px, xy), np.ones(n, bool)
    rng = np.random.default_rng(seed)
    best = np.ones(n, bool)
    best_count, best_err = -1, np.inf
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            hmat = _dlt(px[idx], xy[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm(apply_homography(hmat, px) - xy, axis=1)
        inl = err < threshold_mm
        cnt, tot = int(inl.sum()), float(np.minimum(err, threshold_mm).sum())
        if cnt > best_count or (cnt == best_count and tot < best_err):
            best, best_count, best_err = inl, cnt, tot
    if best.sum() < 4:
        best = np.ones(n, bool)
    return fit_homography(px[best], xy[best]), best


def leave_one_out(px: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Per point: distance (mm) between its table position and the prediction of a homography fitted
    without it. Needs >= 5 points (4 remain for each fit); NaN otherwise."""
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    n = len(px)
    out = np.full(n, np.nan)
    if n < 5:
        return out
    for i in range(n):
        keep = np.arange(n) != i
        hmat = fit_homography(px[keep], xy[keep])
        out[i] = float(np.linalg.norm(apply_homography(hmat, px[i]) - xy[i]))
    return out


@dataclass
class TableHomography:
    """Pixel -> table XY (mm) for the wrist camera in one fixed arm pose."""

    H: np.ndarray                      # 3x3, pixel (u, v, 1) -> table (x, y, 1)
    image_size: tuple[int, int]        # (width, height) the fit was made on
    pose: str                          # saved pose name, e.g. "observe"
    pose_joints: dict[str, float]      # normalized joint values of that pose when fitted
    point: str                         # FK keypoint the calibration put over each brick
    date: str
    points: list[dict] = field(default_factory=list)  # correspondences + residuals
    stats: dict = field(default_factory=dict)
    note: str = ""

    def to_table(self, px) -> np.ndarray:
        return apply_homography(self.H, px)

    def to_pixel(self, xy) -> np.ndarray:
        return apply_homography(np.linalg.inv(self.H), xy)

    def yaw_deg(self, centroid, image_angle_deg: float, step_px: float = 15.0) -> float:
        """Table-frame direction (degrees, from +x towards +y) of an image direction at a pixel."""
        c = np.asarray(centroid, float)
        d = np.array([np.cos(np.radians(image_angle_deg)), np.sin(np.radians(image_angle_deg))])
        a, b = self.to_table(c - step_px * d), self.to_table(c + step_px * d)
        return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))

    def save(self, path: str | Path) -> None:
        import yaml

        data = {
            "description": ("Wrist camera pixel (u, v) -> table XY in mm, robot base frame "
                            "(hackjunction2026.kinematics). Valid ONLY with the arm in the pose below and "
                            "the camera at this image size. Written by scripts/pick_above.py calib-fit."),
            "pose": self.pose,
            "pose_joints": {k: round(float(v), 2) for k, v in self.pose_joints.items()},
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "point": self.point,
            "date": self.date,
            "H": [[float(f"{v:.10g}") for v in row] for row in np.asarray(self.H)],
            "stats": self.stats,
            "points": self.points,
            "note": self.note,
        }
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False, width=110), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TableHomography:
        import yaml

        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(H=np.array(d["H"], float), image_size=tuple(d["image_size"]), pose=d["pose"],
                   pose_joints=d.get("pose_joints", {}), point=d.get("point", "gripper_tip"),
                   date=str(d.get("date", "")), points=d.get("points", []), stats=d.get("stats", {}),
                   note=d.get("note", ""))


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def locate(det: Detection, th: TableHomography) -> list[Brick]:
    """Fill table_xy and long_axis_yaw_deg of every detected brick (in place); returns det.bricks."""
    if tuple(det.image_size) != tuple(th.image_size):
        raise ValueError(f"image is {det.image_size[0]}x{det.image_size[1]}, the homography was fitted on "
                         f"{th.image_size[0]}x{th.image_size[1]}")
    for b in det.bricks:
        x, y = th.to_table(b.centroid)
        b.table_xy = (float(x), float(y))
        b.long_axis_yaw_deg = th.yaw_deg(b.centroid, b.angle_deg)
    return det.bricks


def brick_dict(b: Brick) -> dict:
    d = asdict(b)
    d.pop("box")
    return d

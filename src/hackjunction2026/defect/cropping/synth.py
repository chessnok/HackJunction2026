"""Synthetic black-marker strokes on good brick crops. EVALUATION ONLY, never training data.

The task's defect is "the brick is clearly marked with a black marker". Until real marked bricks
are recorded, these strokes measure recall and help choose a threshold. Real marked bricks override
any number measured on them.

Calibration (2026-11-14, snapshot 20261114-051059-914, floor camera): a real marker arc on a
~70 px wide brick is ~1.5 px wide (~2 % of the brick side), dark brown-black, softened by the
camera. Strokes here are 2-6 % of the brick side wide, 60-95 % opaque, and blurred by the crop's
upscale factor so they are not sharper than the camera could see.
"""

from __future__ import annotations

import cv2
import numpy as np

KINDS = ("line", "arc", "scribble", "dot", "cross")


def _inside_points(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    i = rng.integers(len(xs))
    return np.array([xs[i], ys[i]], np.float32)


def _polyline(kind: str, start: np.ndarray, length: float, rng: np.random.Generator) -> list[np.ndarray]:
    ang = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(ang), np.sin(ang)], np.float32)
    if kind == "line":
        return [np.stack([start, start + d * length])]
    if kind == "arc":
        r = length / rng.uniform(1.0, 2.5)
        a0 = rng.uniform(0, 2 * np.pi)
        sweep = length / r
        t = np.linspace(a0, a0 + sweep, 24)
        centre = start - r * np.array([np.cos(a0), np.sin(a0)], np.float32)
        return [centre + r * np.stack([np.cos(t), np.sin(t)], axis=1)]
    if kind == "scribble":
        pts = [start]
        step = length / 6
        heading = ang
        for _ in range(6):
            heading += rng.uniform(-2.2, 2.2)
            pts.append(pts[-1] + step * np.array([np.cos(heading), np.sin(heading)], np.float32))
        return [np.stack(pts)]
    if kind == "cross":
        p = np.array([-d[1], d[0]], np.float32)
        h = length / 2
        return [np.stack([start - d * h, start + d * h]), np.stack([start - p * h, start + p * h])]
    if kind == "dot":
        return [start[None, :]]
    raise ValueError(kind)


def add_marker(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    upscale: float = 1.0,
    kind: str | None = None,
    opacity_range: tuple[float, float] = (0.6, 0.95),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Draw one marker mark inside `mask` (uint8 0/255, brick hull in crop pixels).

    upscale: crop pixels per camera pixel (crop size / source square side). Blur and the minimum
    width scale with it, so a stroke is never finer than one camera pixel.
    Returns (marked crop, stroke alpha mask uint8 0..255, params).
    """
    h, w = mask.shape
    side = float(np.sqrt(max((mask > 0).sum(), 1)))
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                      iterations=max(1, int(side * 0.04)))
    if inner.sum() == 0:
        inner = mask
    kind = kind or str(rng.choice(KINDS, p=[0.25, 0.25, 0.25, 0.1, 0.15]))
    width = max(rng.uniform(0.02, 0.06) * side, 1.2 * upscale, 2.0)
    if kind == "dot":
        width = max(rng.uniform(0.06, 0.12) * side, 2.0 * upscale, 3.0)
    length = rng.uniform(0.25, 0.6) * side
    opacity = rng.uniform(*opacity_range)
    ink = np.array([rng.uniform(15, 45), rng.uniform(15, 40), rng.uniform(15, 40)], np.float32)  # BGR

    best = None
    for _ in range(12):
        start = _inside_points(inner > 0, rng)
        lines = _polyline(kind, start, length, rng)
        canvas = np.zeros((h, w), np.uint8)
        for pts in lines:
            p = np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)
            if len(pts) == 1:
                cv2.circle(canvas, tuple(p[0, 0]), int(round(width / 2 * 16)), 255, -1, cv2.LINE_AA, shift=4)
            else:
                cv2.polylines(canvas, [p], False, 255, int(round(width)), cv2.LINE_AA, shift=4)
        drawn = int((canvas > 0).sum())
        kept = int(((canvas > 0) & (inner > 0)).sum())
        if drawn and (best is None or kept / drawn > best[0]):
            best = (kept / drawn, canvas)
        if drawn and kept / drawn >= 0.8:
            break
    canvas = best[1]
    canvas = np.where(inner > 0, canvas, 0).astype(np.float32) / 255.0
    sigma = max(0.5, 0.45 * upscale)
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigma)
    alpha = (canvas * opacity)[..., None]
    out = crop_bgr.astype(np.float32) * (1 - alpha) + ink[None, None, :] * alpha
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    params = {
        "synth_kind": kind,
        "synth_width_px": round(float(width), 1),
        "synth_width_frac": round(float(width / side), 3),
        "synth_opacity": round(float(opacity), 2),
        "synth_area_frac": round(float((canvas > 0.25).sum() / max((mask > 0).sum(), 1)), 4),
    }
    return out, np.clip(canvas * 255, 0, 255).astype(np.uint8), params

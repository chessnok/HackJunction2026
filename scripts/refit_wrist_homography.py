#!/usr/bin/env python3
"""Refit configs/table_homography.yaml from a calibration run's pairs, with the lens distortion (k1) fitted too.

Drops outlier pairs (wrong blob, brick moved), picks k1 by leave-one-out error, fits H on the undistorted pixels.

  PYTHONPATH=src $PY scripts/refit_wrist_homography.py [captures/calib_wrist/<run>]   # default: the newest run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from hackjunction2026 import table_map as tm

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "configs/table_homography.yaml"


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((REPO / "captures/calib_wrist").glob("*/summary.json"))[-1].parent
    pairs = [p for p in json.loads((run / "summary.json").read_text())["pairs"] if "pixel" in p and not p.get("dropped")]
    px = np.array([p["pixel"] for p in pairs], float)
    xy = np.array([p["place_xy"] for p in pairs], float)
    th = tm.TableHomography.load(OUT)
    size = tuple(th.image_size)
    # outliers: iterate k1 / RANSAC once each way
    keep = np.ones(len(px), bool)
    for _ in range(2):
        k1, _ = tm.fit_k1(px[keep], xy[keep], size)
        u = tm.undistort_px(px, k1, size)
        H, inl = tm.fit_homography_ransac(u, xy, threshold_mm=12.0)
        keep = np.asarray(inl, bool)
    k1, loo = tm.fit_k1(px[keep], xy[keep], size)
    u = tm.undistort_px(px, k1, size)
    H = tm.fit_homography(u[keep], xy[keep])
    res = np.hypot(*(tm.apply_homography(H, u) - xy).T)
    _, loo0 = tm.fit_k1(px[keep], xy[keep], size, ks=np.array([0.0]))
    print(f"{run.name}: {len(px)} pairs, {int(keep.sum())} inliers; k1 = {k1:+.3f}")
    print(f"  leave-one-out RMS: {loo0:.1f} mm without the lens fix -> {loo:.1f} mm with it")
    for p, r, k in zip(pairs, res, keep, strict=True):
        print(f"  pixel ({p['pixel'][0]:4.0f}, {p['pixel'][1]:4.0f}) placed ({p['place_xy'][0]:5.0f}, {p['place_xy'][1]:5.0f})"
              f"  residual {r:5.1f} mm{'' if k else '  OUTLIER'}")
    th.H, th.k1 = H, float(k1)
    th.stats = {"n": len(px), "inliers": int(keep.sum()), "rms_mm": round(float(np.sqrt(np.mean(res[keep] ** 2))), 1),
                "max_mm": round(float(res[keep].max()), 1), "loo_rms_mm": round(float(loo), 1),
                "loo_rms_mm_without_k1": round(float(loo0), 1), "k1": round(float(k1), 4), "run": run.name}
    th.points = [{"pixel": p["pixel"], "place_xy": p["place_xy"], "residual_mm": round(float(r), 1), "inlier": bool(k)}
                 for p, r, k in zip(pairs, res, keep, strict=True)]
    th.date = tm.now_iso()
    th.note = f"refit by scripts/refit_wrist_homography.py from {run.name}, radial k1 fitted"
    th.save(OUT)
    print(f"saved {OUT.relative_to(REPO)}: {th.stats}")


if __name__ == "__main__":
    main()

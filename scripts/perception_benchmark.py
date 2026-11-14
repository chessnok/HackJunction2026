#!/usr/bin/env python3
"""Per-frame latency of the live CV path (top + floor bricks, fusion, gripper) on decoded frames.

Decoding is excluded (live frames come from the cameras). Single thread (cv2.setNumThreads(1)).

  PYTHONPATH=src nice -n 19 python scripts/perception_benchmark.py 431a71f7 --episodes 20,41,63
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from hackjunction2026.perception.annotate import merge_close, projection, triangulate
from hackjunction2026.perception.dataset import Dataset
from hackjunction2026.perception.gripper import detect_gripper, static_robot_mask
from hackjunction2026.perception.scene import Frame, detect_bricks_cv, exclusion_mask, sheet_roi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--episodes", default="20,41,63")
    ap.add_argument("--frame", default=None)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)
    ds = Dataset(args.dataset)
    fr = Frame.load(args.frame or f"/home/intel/dev/captures/perception/A/table_frame_{ds.id}.json")
    t_det, t_grip, t_all = [], [], []
    for ep in map(int, args.episodes.split(",")):
        ctx = {}
        t0 = time.perf_counter()
        for cam in ("top", "floor"):
            f0 = ds.frame(ep, cam, 0)
            roi, _ = sheet_roi(cam, f0)
            ctx[cam] = (roi, exclusion_mask(f0, roi), static_robot_mask(f0))
        print(f"ep {ep}: per-episode setup (sheet ROI, masks) {1000 * (time.perf_counter() - t0):.0f} ms")
        P = {cam: projection(fr.data["cameras"][fr.setup_of(cam, ep)]) for cam in ("top", "floor")}
        frames = {cam: dict(ds.iter_frames(ep, cam)) for cam in ("top", "floor")}
        for i in sorted(frames["top"]):
            a = time.perf_counter()
            dets = {}
            for cam in ("top", "floor"):
                roi, ex, _ = ctx[cam]
                blobs = detect_bricks_cv(frames[cam][i], roi, ex)
                dets[cam] = merge_close([(b, fr.footprint_center(cam, ep, b.bottom_center)) for b in blobs])
            if dets["top"] and dets["floor"]:
                triangulate(P["top"], P["floor"], dets["top"][0][0].center, dets["floor"][0][0].center)
            b = time.perf_counter()
            for cam in ("top", "floor"):
                detect_gripper(frames[cam][i], ctx[cam][2])
            c = time.perf_counter()
            t_det.append(b - a); t_grip.append(c - b); t_all.append(c - a)
    for name, t in (("bricks top+floor+fusion", t_det), ("gripper top+floor", t_grip), ("total", t_all)):
        t = 1000 * np.array(t)
        print(f"{name:26s} median {np.median(t):5.1f} ms  p90 {np.percentile(t, 90):5.1f} ms  max {t.max():5.1f} ms"
              f"  -> {1000 / np.median(t):.0f} Hz")
    print(f"{len(t_all)} frames, {args.threads} thread(s)")


if __name__ == "__main__":
    main()

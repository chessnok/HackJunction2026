#!/usr/bin/env python3
"""Check the local VLM server and pick its box coordinate convention on frames OUTSIDE the GT set.

For first frames of a few non-evaluation episodes, asks the VLM for brick boxes with the prompt for
relative 0-1000 coordinates and with the prompt for pixel coordinates, and compares both with the
CV boxes (IoU). One request at a time.

  PYTHONPATH=src python scripts/perception_vlm_probe.py 431a71f7 --episodes 0,20,50
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from hackjunction2026.perception.dataset import Dataset
from hackjunction2026.perception.scene import detect_bricks_cv, exclusion_mask, sheet_roi
from hackjunction2026.perception.vlm import VLMClient, iou


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--episodes", default="0,20,50")
    ap.add_argument("--url", default="http://localhost:8000/v3")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    ds = Dataset(args.dataset)
    res = {}
    for mode in ("rel1000", "pixel"):
        cl = VLMClient(args.url, args.model, coord_mode=mode)
        ious, lat = [], []
        for ep in map(int, args.episodes.split(",")):
            for cam in ("top", "floor"):
                img = ds.frame(ep, cam, 0)
                roi, _ = sheet_roi(cam, img)
                ref = detect_bricks_cv(img, roi, exclusion_mask(img, roi))
                dets = cl.detect(img, cam)
                lat.append(cl.last_latency)
                best = max((iou(d.bbox, ref[0].bbox) for d in dets), default=0.0) if ref else float("nan")
                ious.append(best)
                print(mode, ep, cam, "cv", [round(v) for v in ref[0].bbox] if ref else None, "vlm",
                      [([round(v) for v in d.bbox], d.color) for d in dets], f"iou {best:.2f} {cl.last_latency:.1f} s",
                      "raw", cl.last_raw[:160], flush=True)
        res[mode] = {"iou_median": float(np.nanmedian(ious)), "latency_s_median": float(np.median(lat)), "model": cl.model}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

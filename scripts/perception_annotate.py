#!/usr/bin/env python3
"""Annotate every frame of a Studio dataset with brick and gripper positions (CV v2, top + floor D415).

Outputs in <out> (default /home/intel/dev/captures/perception/A):
  objects_<dataset>.parquet   one row per (frame, tracked brick)
  gripper_<dataset>.parquet   one row per frame
Needs table_frame_<dataset>.json from scripts/perception_table_frame.py. Schemas:
docs/perception/A-object-localization.md. Read-only on the dataset.

  PYTHONPATH=src nice -n 19 python scripts/perception_annotate.py 431a71f7 --workers 4
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ARGS = {}


def work(ep: int):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2

    cv2.setNumThreads(1)
    from hackjunction2026.perception.annotate import annotate_episode
    from hackjunction2026.perception.dataset import Dataset
    from hackjunction2026.perception.scene import Frame

    ds = Dataset(ARGS["dataset"])
    fr = Frame.load(ARGS["frame"])
    t = time.time()
    rows, grows = annotate_episode(ds, fr, ep, step=ARGS["step"])
    return ep, rows, grows, time.time() - t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default="/home/intel/dev/captures/perception/A")
    ap.add_argument("--frame", default=None, help="table-frame JSON (default: <out>/table_frame_<dataset>.json)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step", type=int, default=1, help="annotate every Nth frame (default: all)")
    ap.add_argument("--episodes", default=None, help="comma list (default: all)")
    args = ap.parse_args()
    from hackjunction2026.perception.dataset import Dataset

    ds = Dataset(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ARGS.update(dataset=str(ds.root), frame=args.frame or str(out / f"table_frame_{ds.id}.json"), step=args.step)
    eps = [int(e) for e in args.episodes.split(",")] if args.episodes else [e.index for e in ds.episodes]
    objs, grips = [], []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for ep, rows, grows, dt in pool.imap_unordered(work, eps):
            objs += rows
            grips += grows
            n = ds.episodes[ep].length
            print(f"ep {ep:3d} {ds.episodes[ep].task:18s} {n:4d} frames {dt:6.1f} s  objects {len({r['object_id'] for r in rows})}",
                  flush=True)
    od = pd.DataFrame(objs).sort_values(["episode_index", "frame_index", "object_id"]).reset_index(drop=True)
    gd = pd.DataFrame(grips).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    od.insert(0, "dataset", ds.id)
    gd.insert(0, "dataset", ds.id)
    od.to_parquet(out / f"objects_{ds.id}.parquet", index=False)
    gd.to_parquet(out / f"gripper_{ds.id}.parquet", index=False)
    print(f"{len(od)} object rows, {len(gd)} gripper rows, {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()

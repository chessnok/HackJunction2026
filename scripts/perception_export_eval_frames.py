#!/usr/bin/env python3
"""Export the evaluation frames (all cameras) used for the perception ground truth.

Selection: 16 episodes spread over every task / colour / camera setup, 3 phases each:
start (frame 0), mid (55 % of the episode: arm approaching) and late (90 %: gripper at or around the
brick, the usual occlusion case). Output: <out>/ep<NNN>_f<FFFF>_<cam>.png + frames.csv.

  PYTHONPATH=src python scripts/perception_export_eval_frames.py 431a71f7
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from hackjunction2026.perception.dataset import Dataset

EPISODES = [4, 8, 13, 19, 22, 25, 29, 33, 36, 41, 44, 46, 49, 53, 58, 63]
PHASES = {"start": 0.0, "mid": 0.55, "late": 0.90}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default="/home/intel/dev/captures/perception/A/eval_frames")
    args = ap.parse_args()
    ds = Dataset(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in EPISODES:
        L = ds.episodes[ep].length
        frames = {ph: int(round(f * (L - 1))) for ph, f in PHASES.items()}
        for cam in ("top", "floor", "arm"):
            for i, img in ds.iter_frames(ep, cam, set(frames.values())):
                cv2.imwrite(str(out / f"ep{ep:03d}_f{i:04d}_{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for ph, i in frames.items():
            rows.append({"episode": ep, "frame": i, "phase": ph, "task": ds.episodes[ep].task})
    with (out / "frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["episode", "frame", "phase", "task"])
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "frames ->", out)


if __name__ == "__main__":
    main()

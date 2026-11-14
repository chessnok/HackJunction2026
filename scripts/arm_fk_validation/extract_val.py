"""Extract validation frames (approach + grasp) for each grasp event, top and floor cameras."""
from pathlib import Path

import cv2
import numpy as np

from data import load, grasp_events
from frames import get_frames

OFFS = [-75, -60, -45, -30, -20, -10, 0, 15]
out = Path("img/val")
out.mkdir(parents=True, exist_ok=True)
df, st, ac = load()
ev = grasp_events(df, st, ac)
seen = set()
items = []
for e in ev:
    if e[0] in seen:
        continue
    seen.add(e[0])
    ep, fr_k, row_k, i0, j0 = e
    d = df[df.episode_index == ep]
    f0 = int(d.frame_index.values[i0])
    fids = [f0 + o for o in OFFS if 0 <= f0 + o < len(d)]
    for cam in ("top", "floor"):
        todo = [f for f in fids if not (out / f"ep{ep:03d}_{cam}_f{f:04d}.png").exists()]
        if todo:
            fr = get_frames(ep, cam, todo)
            for f, img in fr.items():
                cv2.imwrite(str(out / f"ep{ep:03d}_{cam}_f{f:04d}.png"), img[:, :, ::-1])
    print(ep, fids, flush=True)

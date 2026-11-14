"""Detect the brick in the first frame of each episode by chroma difference to a per-camera-group
median background. Exploration / validation only."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from frames import get_frames, episodes

cam = sys.argv[1] if len(sys.argv) > 1 else "top"
groups = [list(range(0, 6)), list(range(6, len(episodes())))] if cam == "top" else [list(range(0, 47)), list(range(47, len(episodes())))]
cache = Path(f"img/first_{cam}")
cache.mkdir(parents=True, exist_ok=True)
ims = {}
for ep in range(len(episodes())):
    p = cache / f"{ep:03d}.png"
    if not p.exists():
        cv2.imwrite(str(p), get_frames(ep, cam, [0])[0][:, :, ::-1])
    ims[ep] = cv2.imread(str(p))
res = {}
vis = []
for grp in groups:
    labs = {k: cv2.cvtColor(ims[k], cv2.COLOR_BGR2LAB).astype(np.float32) for k in grp}
    bg = np.median(np.stack(list(labs.values())), axis=0)
    for ep in grp:
        lab = labs[ep]
        d = np.linalg.norm(lab[:, :, 1:] - bg[:, :, 1:], axis=2)
        sat = cv2.cvtColor(ims[ep], cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
        raw = (d > 12) & (sat > 60)
        m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.dilate(m, np.ones((7, 7), np.uint8))
        n, lbl, stats, cents = cv2.connectedComponentsWithStats(m)
        best = None
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 80 or a > 20000 or y < 8:
                continue
            if cam == "top" and (x + w > 815 or (y < 70 and x + w > 590 and x > 560)):
                continue  # robot at the right edge, green bowl
            if cam == "floor" and (x < 160 and y < 150 or x < 120 and y < 260):
                continue
            if cam == "top" and ep < 6 and x < 60:
                continue
            if best is None or a > stats[best][4]:
                best = i
        out = ims[ep].copy()
        if best is not None:
            x, y, w, h, a = stats[best]
            core = (lbl == best) & raw
            ys, xs = np.nonzero(core)
            cx, cy = xs.mean(), ys.mean()
            res[ep] = dict(cx=float(cx), cy=float(cy), bottom=float(np.percentile(ys, 97)), area=int(core.sum()),
                           bbox=[int(x), int(y), int(w), int(h)])
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(out, f"{ep:03d}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        vis.append(cv2.resize(out, (424, 240)))
while len(vis) % 6:
    vis.append(np.zeros_like(vis[0]))
sheet = np.vstack([np.hstack(vis[i:i + 6]) for i in range(0, len(vis), 6)])
h = sheet.shape[0] // 240
cv2.imwrite(f"img/bricks_{cam}_a.png", sheet[: (h // 2) * 240])
cv2.imwrite(f"img/bricks_{cam}_b.png", sheet[(h // 2) * 240:])
json.dump(res, open(f"bricks_{cam}.json", "w"), indent=1)
print(len(res), "detected of", len(ims))

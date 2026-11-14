"""Contact sheet: observed (red) vs model-predicted (green) gripper extreme point."""
import json
import sys

import cv2
import numpy as np

js = json.load(open(sys.argv[1]))
out = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
obs = js["obs"]
rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
sel = rng.choice(len(obs), min(n, len(obs)), replace=False)
tiles = []
for i in sorted(sel):
    o = obs[i]
    im = cv2.imread(o["path"])
    cv2.circle(im, (int(o["u"]), int(o["v"])), 6, (0, 0, 255), 2)
    cv2.putText(im, f"ep{o['ep']} f{o['f']} r={o['res']:.0f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    tiles.append(cv2.resize(im[:300], (424, 150)))
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))

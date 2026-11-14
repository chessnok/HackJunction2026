"""Bowl positions (static per episode) from an episode's first top and floor frames.

The green bowl is fully visible in the top camera (at the far sheet edge), the lavender bowl in the
floor camera. A bowl's image blob ends at its rim nearest to the camera; its centre is estimated as
the table point of that bottom-centre moved by the radius (half the bottom-edge width on the table)
away from the camera. Accuracy about +-2 cm (rim height and perspective are ignored).
"""

from __future__ import annotations

import cv2
import numpy as np

from .scene import Frame

BOWLS = {
    # colour, camera that sees the whole bowl, HSV test
    "green": ("top", lambda h, s, v: (h >= 60) & (h <= 90) & (s >= 70)),
    "lavender": ("floor", lambda h, s, v: (h >= 106) & (h <= 130) & (s >= 40) & (s < 110) & (v >= 75)),
}


def detect_bowls(first: dict[str, np.ndarray], frame: Frame, ep: int) -> list[dict]:
    out = []
    for color, (cam, test) in BOWLS.items():
        hsv = cv2.cvtColor(first[cam], cv2.COLOR_RGB2HSV)
        h, s, v = (hsv[..., i].astype(np.int32) for i in range(3))
        m = cv2.morphologyEx(test(h, s, v).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            continue
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] < 1500:
            continue
        ys, xs = np.nonzero(lab == k)
        y1 = ys.max()
        low = ys >= y1 - 3
        xl, xr = xs[low].min(), xs[low].max()
        x0, x1 = xs.min(), xs.max()
        pl, pr = frame.to_table(cam, ep, [(x0, y1), (x1, y1)])
        radius = 0.5 * float(np.linalg.norm(pl - pr))
        bc = frame.to_table(cam, ep, [(0.5 * (xl + xr), y1)])[0]
        cam_xy = np.array(frame.data["cameras"][frame.setup_of(cam, ep)]["camera_xy_mm"])
        d = (bc - cam_xy) / np.linalg.norm(bc - cam_xy)
        c = bc + radius * d
        touches = x0 <= 1 or x1 >= first[cam].shape[1] - 2 or ys.min() <= 1
        out.append({"episode_index": ep, "class": "bowl", "color": color, "camera": cam,
                    "x_mm": float(c[0]), "y_mm": float(c[1]), "radius_mm": radius,
                    "bbox_px": [float(x0), float(ys.min()), float(x1 + 1), float(y1 + 1)],
                    "cut_by_image_border": bool(touches)})
    return out

"""Overlay the FK arm (mesh points + skeleton) on camera frames using a fitted camera."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from hackjunction2026.kinematics import Calibration, normalized_to_urdf, link_frames, keypoints  # noqa: E402
from data import load  # noqa: E402
from frames import get_frames  # noqa: E402
from meshes import link_meshes  # noqa: E402

W, H = 848, 480
CAL = Calibration.from_file("/home/intel/dev/configs/calibration/follower.json")
LM = link_meshes()
rng = np.random.default_rng(0)
PTS = {}
for ln, parts in LM.items():
    if not parts:
        continue
    v = np.vstack([m.vertices for _, m in parts]) * 1000
    PTS[ln] = v[rng.choice(len(v), min(len(v), 6000), replace=False)]
COL = {"base_link": (200, 200, 200), "shoulder_link": (255, 200, 0), "upper_arm_link": (0, 200, 255),
       "lower_arm_link": (0, 255, 0), "wrist_link": (255, 0, 255), "gripper_link": (0, 128, 255),
       "moving_jaw_so101_v1_link": (0, 0, 255)}


def project(cp, P):
    R, _ = cv2.Rodrigues(np.asarray(cp[:3]))
    X = P @ R.T + np.asarray(cp[3:6])
    f = cp[6]
    cx = cp[7] if len(cp) > 7 else W / 2
    cy = cp[8] if len(cp) > 8 else H / 2
    ok = X[:, 2] > 1
    return np.c_[f * X[:, 0] / X[:, 2] + cx, f * X[:, 1] / X[:, 2] + cy], ok


def draw(img, cp, norm, label=""):
    q = normalized_to_urdf(norm[None], CAL)
    F = link_frames(q)
    out = img.copy()
    over = img.copy()
    for ln, p in PTS.items():
        T = F[ln][0]
        P = p @ T[:3, :3].T + T[:3, 3]
        uv, ok = project(cp, P)
        for (u, v) in uv[ok].astype(int):
            if 0 <= u < W and 0 <= v < H:
                over[v, u] = COL.get(ln, (255, 255, 255))
    out = cv2.addWeighted(over, 0.55, out, 0.45, 0)
    kp = keypoints(q)
    names = ["base", "shoulder", "elbow", "wrist", "gripper_base", "gripper_tip"]
    P = np.array([kp[k][0] for k in names])
    uv, ok = project(cp, P)
    for a, b in zip(uv[:-1], uv[1:]):
        cv2.line(out, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
    for (u, v) in uv:
        cv2.circle(out, (int(round(u)), int(round(v))), 3, (255, 255, 255), -1, cv2.LINE_AA)
    u, v = uv[-1]
    cv2.drawMarker(out, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    if label:
        cv2.putText(out, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


if __name__ == "__main__":
    cam = sys.argv[1]
    cp = json.load(open(sys.argv[2]))["cp"]
    spec = sys.argv[3]  # "ep:frame,ep:frame,..."
    outp = sys.argv[4]
    df, st, ac = load()
    key = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(df.episode_index.values, df.frame_index.values))}
    tiles = []
    for item in spec.split(","):
        ep, f = map(int, item.split(":"))
        img = get_frames(ep, cam, [f])[f][:, :, ::-1].copy()
        tiles.append(draw(img, cp, st[key[(ep, f)]], f"ep{ep} f{f}"))
    ncol = 2 if len(tiles) > 1 else 1
    while len(tiles) % ncol:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)])
    cv2.imwrite(outp, sheet)
    print(outp, sheet.shape)

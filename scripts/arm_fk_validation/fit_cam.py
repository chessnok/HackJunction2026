"""Fit a pinhole camera to (FK grasp point, brick centroid in image) pairs; compare joint mappings."""
import json
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

import vfk
from data import load, grasp_events

cam = sys.argv[1] if len(sys.argv) > 1 else "top"
EPMIN = int(sys.argv[2]) if len(sys.argv) > 2 else 6
EPMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 999
df, st, ac = load()
ev = grasp_events(df, st, ac)
BAD = {"top": {2, 17, 39, 49, 51, 52}, "floor": {4, 10, 59, 64, 65}}[cam]
B = {int(k): v for k, v in json.load(open(f"bricks_{cam}.json")).items() if int(k) not in BAD}
seen = set(); E = []
for e in ev:
    if e[0] not in seen and e[0] in B and EPMIN <= e[0] <= EPMAX:
        seen.add(e[0]); E.append(e)
rows = np.array([e[2] for e in E])
uv = np.array([[B[e[0]]["cx"], B[e[0]]["cy"]] for e in E])
t = vfk.norm_to_ticks(st[rows])
fixed = np.load("fixed_pts.npy"); mov = np.load("mov_pts.npy")
FIX_TIP = np.array([-0.0089, -0.0002, -0.1044])  # fixed jaw tip in gripper_link (from mesh)
MOV_TIP = np.array([-0.0114, -0.0820, 0.0189])  # moving jaw tip in moving jaw link (from mesh)
W, H = 848, 480


def grasp_point(d, centers0, zup=10.0):
    centers = np.array(centers0, float).copy()
    centers[:5] -= np.asarray(d)[:5] / 360 * 4096
    q = vfk.q_from_ticks(t, centers)
    L = vfk.fk_all(q)
    pf = vfk.apply(L["gripper_link"], FIX_TIP[None])[:, 0]
    pm = vfk.apply(L["moving_jaw_so101_v1_link"], MOV_TIP[None])[:, 0]
    p = (pf + pm) / 2 * 1000
    p[:, 2] += zup
    return p


def project(cp, P):
    rvec, tvec, f, cx, cy = cp[:3], cp[3:6], cp[6], cp[7], cp[8]
    R, _ = cv2.Rodrigues(rvec)
    X = P @ R.T + tvec
    return np.c_[f * X[:, 0] / X[:, 2] + cx, f * X[:, 1] / X[:, 2] + cy], X[:, 2]


def init_cam(P, uv, f=610.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    best = None
    for flags in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        ok, rv, tv = cv2.solvePnP(P, uv, K, None, flags=flags)
        if ok:
            pr, _ = cv2.projectPoints(P, rv, tv, K, None)
            err = np.sqrt(((pr[:, 0] - uv) ** 2).sum(1).mean())
            if best is None or err < best[0]:
                best = (err, rv.ravel(), tv.ravel())
    return np.r_[best[1], best[2], f, W / 2, H / 2]


def fit_cam(P, uv, free_c=True):
    cp0 = init_cam(P, uv)

    def r(cp):
        full = cp if free_c else np.r_[cp, W / 2, H / 2]
        pr, z = project(full, P)
        return (pr - uv).ravel()
    x0 = cp0 if free_c else cp0[:7]
    s = least_squares(r, x0, loss="soft_l1", f_scale=5.0)
    full = s.x if free_c else np.r_[s.x, W / 2, H / 2]
    pr, z = project(full, P)
    res = np.linalg.norm(pr - uv, axis=1)
    return full, res, z


def cam_center(cp):
    R, _ = cv2.Rodrigues(cp[:3])
    return -R.T @ cp[3:6]


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=1, linewidth=180)
    print(len(E), "pairs, episodes", EPMIN, "..", EPMAX)
    MID = vfk.MID; C2047 = np.full(6, 2047.0)
    for name, c in [("2047", C2047), ("mid", MID)]:
        P = grasp_point(np.zeros(5), c)
        for free_c in (False, True):
            cp, res, z = fit_cam(P, uv, free_c)
            C = cam_center(cp)
            print(f"{name:5s} free_c={free_c}: rms {np.sqrt((res**2).mean()):.1f}px median {np.median(res):.1f}px"
                  f" max {res.max():.1f}  f={cp[6]:.0f} c=({cp[7]:.0f},{cp[8]:.0f}) cam centre {C.round(0)} mm"
                  f" mm/px@median depth {np.median(z)/cp[6]:.2f}")
    # pan sign flip test
    for i in range(5):
        P = grasp_point(np.zeros(5), C2047)
        signs = [1] * 6; signs[i] = -1
        q = vfk.q_from_ticks(t, C2047, signs=signs)
        L = vfk.fk_all(q)
        pf = vfk.apply(L["gripper_link"], FIX_TIP[None])[:, 0]
        pm = vfk.apply(L["moving_jaw_so101_v1_link"], MOV_TIP[None])[:, 0]
        P = (pf + pm) / 2 * 1000
        cp, res, z = fit_cam(P, uv, False)
        C = cam_center(cp)
        print(f"sign flip {vfk.NAMES[i]}: rms {np.sqrt((res**2).mean()):.1f}px f={cp[6]:.0f} cam centre {C.round(0)}")

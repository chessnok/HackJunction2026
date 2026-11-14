"""Camera + joint-mapping fit from the gripper's lowest image point (silhouette extreme point).

For each validation frame: observed = lowest (max v) arm-mask pixel inside a region of interest;
model = lowest projected vertex of the gripper meshes (fixed jaw body + moving jaw).
"""
import glob
import json
import re
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

import vfk
from armmask import MASKS
from data import load
from meshes import link_meshes

cam = sys.argv[1]
EPMIN, EPMAX = int(sys.argv[2]), int(sys.argv[3])
W, H = 848, 480
df, st, ac = load()
key = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(df.episode_index.values, df.frame_index.values))}

LM = link_meshes()
rng = np.random.default_rng(1)
G = np.vstack([m.vertices for fn, m in LM["gripper_link"]])
G = G[rng.choice(len(G), 4000, replace=False)]
MJ = np.vstack([m.vertices for fn, m in LM["moving_jaw_so101_v1_link"]])
MJ = MJ[rng.choice(len(MJ), 3000, replace=False)]


def roi(cam):
    r = np.zeros((H, W), np.uint8)
    if cam == "top":
        poly = np.array([[0, 300], [150, 22], [790, 18], [790, 480], [0, 480]])
    else:
        poly = np.array([[0, 60], [848, 60], [848, 480], [0, 480]])
    cv2.fillPoly(r, [poly.astype(np.int32)], 1)
    if cam == "floor":
        r[:170, :175] = 0  # robot base
    return r


R0 = roi(cam)
obs = []
for p in sorted(glob.glob(f"img/val/ep*_{cam}_f*.png")):
    ep, f = map(int, re.findall(r"ep(\d+)_\w+_f(\d+)", p)[0])
    if not (EPMIN <= ep <= EPMAX):
        continue
    im = cv2.imread(p)
    m = MASKS[cam](im) & R0
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
    # keep components touching the top of the ROI (arm enters from above) and big enough
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= 150 and y <= (40 if cam == "top" else 75):
            keep[i] = True
    mm = keep[lbl]
    if not mm.any():
        continue
    ys, xs = np.nonzero(mm)
    vmax = ys.max()
    if vmax < (60 if cam == "top" else 100):
        continue  # gripper not really in view
    sel = ys >= vmax - 2
    obs.append(dict(ep=ep, f=f, row=key[(ep, f)], u=float(np.median(xs[sel])), v=float(vmax), path=p))
print(len(obs), "observations")
rows = np.array([o["row"] for o in obs])
UV = np.array([[o["u"], o["v"]] for o in obs])
T = vfk.norm_to_ticks(st[rows])


def model_points(d, centers0=np.full(6, 2047.0)):
    centers = np.array(centers0, float).copy()
    centers[:5] -= np.asarray(d)[:5] / 360 * 4096
    q = vfk.q_from_ticks(T, centers)
    L = vfk.fk_all(q)
    pg = vfk.apply(L["gripper_link"], G)
    pm = vfk.apply(L["moving_jaw_so101_v1_link"], MJ)
    return np.concatenate([pg, pm], axis=1) * 1000  # (N, M, 3) mm


def project(cp, P):
    R, _ = cv2.Rodrigues(cp[:3])
    X = P @ R.T + cp[3:6]
    f = cp[6]
    return np.stack([f * X[..., 0] / X[..., 2] + W / 2, f * X[..., 1] / X[..., 2] + H / 2], -1), X[..., 2]


def extreme(cp, P):
    uv, z = project(cp, P)
    i = np.argmax(uv[..., 1], axis=1)
    return uv[np.arange(len(uv)), i]


def fit(centers0, d_free=(), cp0=None, d0=None):
    nd = len(d_free)

    def r(x):
        cp = x[:7]
        d = np.zeros(5)
        d[list(d_free)] = x[7:]
        P = model_points(d, centers0)
        return (extreme(cp, P) - UV).ravel()
    x0 = np.r_[cp0, np.zeros(nd) if d0 is None else d0]
    s = least_squares(r, x0, loss="soft_l1", f_scale=6.0, x_scale=np.r_[0.05, 0.05, 0.05, 20, 20, 20, 20, np.ones(nd)])
    res = np.linalg.norm(s.fun.reshape(-1, 2), axis=1)
    return s, res


def init_cp(centers0):
    # coarse init: PnP of the lowest model vertex vs observations, f=610
    P = model_points(np.zeros(5), centers0)
    lowest = P[np.arange(len(P)), np.argmin(P[..., 2], axis=1)]
    K = np.array([[610, 0, W / 2], [0, 610, H / 2], [0, 0, 1.0]])
    best = None
    for flags in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        ok, rv, tv = cv2.solvePnPRansac(lowest, UV, K, None, flags=flags, reprojectionError=15, iterationsCount=500)[:3]
        if ok:
            pr, _ = cv2.projectPoints(lowest, rv, tv, K, None)
            e = np.median(np.linalg.norm(pr[:, 0] - UV, axis=1))
            if best is None or e < best[0]:
                best = (e, np.r_[rv.ravel(), tv.ravel(), 610.0])
    return best[1]


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2, linewidth=180)
    results = {}
    for name, c in [("2047", np.full(6, 2047.0)), ("mid", vfk.MID)]:
        cp0 = init_cp(c)
        s, res = fit(c, (), cp0)
        Rm, _ = cv2.Rodrigues(s.x[:3])
        C = -Rm.T @ s.x[3:6]
        inl = res < 20
        print(f"{name}: n={len(res)} rms {np.sqrt((res**2).mean()):.1f}px median {np.median(res):.1f}px "
              f"rms(inliers<20px) {np.sqrt((res[inl]**2).mean()):.1f} ({inl.sum()}) f={s.x[6]:.0f} cam centre {C.round(0)}")
        results[name] = (s, res)
        for free in ([1, 2, 3],):
            s2, res2 = fit(c, free, s.x[:7])
            inl2 = res2 < 20
            Jm = s2.jac
            try:
                cov = np.linalg.inv(Jm.T @ Jm) * np.median(res2) ** 2
                sd = np.sqrt(np.diag(cov))[7:]
            except np.linalg.LinAlgError:
                sd = np.full(len(free), np.nan)
            print(f"   + free offsets {[vfk.NAMES[i] for i in free]}: {s2.x[7:].round(2)} deg (+-{sd.round(2)}) "
                  f"median {np.median(res2):.1f}px rms(inl) {np.sqrt((res2[inl2]**2).mean()):.1f} f={s2.x[6]:.0f}")
    s, res = results["2047"]
    json.dump(dict(cp=s.x.tolist(), obs=[dict(o, res=float(r)) for o, r in zip(obs, res)]),
              open(f"extreme_{cam}_{EPMIN}_{EPMAX}.json", "w"), indent=1)

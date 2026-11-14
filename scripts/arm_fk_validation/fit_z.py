import numpy as np
from scipy.optimize import least_squares
from data import load, grasp_events
import vfk

df, st, ac = load()
ev = grasp_events(df, st, ac)
seen = set(); E = []
for e in ev:
    if e[0] not in seen:
        seen.add(e[0]); E.append(e)
rows = np.array([e[2] for e in E])
fixed = np.load("fixed_pts.npy"); mov = np.load("mov_pts.npy")
t = vfk.norm_to_ticks(st[rows])
D2R = np.pi / 180


def lowest(d, t=t):
    centers = np.full(6, 2047.0)
    centers[:5] -= d[:5] / 360 * 4096  # d in degrees added to q
    q = vfk.q_from_ticks(t, centers)
    L = vfk.fk_all(q)
    pf = vfk.apply(L["gripper_link"], fixed)
    pm = vfk.apply(L["moving_jaw_so101_v1_link"], mov)
    return np.minimum(pf[:, :, 2].min(1), pm[:, :, 2].min(1)) * 1000, L


def resid(p, which):
    d = np.zeros(5)
    d[which] = p[:-1]
    z, _ = lowest(d)
    return z - p[-1]


for which in ([], [1], [2], [3], [1, 2], [1, 2, 3]):
    p0 = np.zeros(len(which) + 1)
    r = least_squares(resid, p0, args=(which,))
    Jm = r.jac
    cov = np.linalg.inv(Jm.T @ Jm) * (r.fun ** 2).sum() / max(1, len(r.fun) - len(p0))
    sd = np.sqrt(np.diag(cov))
    print("free", [vfk.NAMES[w] for w in which], "params(deg..., h mm)", np.round(r.x, 2), "+-", np.round(sd, 2),
          "resid std %.2f" % r.fun.std(), "cond %.0f" % np.linalg.cond(Jm))

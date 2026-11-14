"""Joint fit: both cameras + shared joint-zero corrections (lift, elbow, wrist_flex) on the final
mapping. Reports corrections with standard errors."""
import json

import numpy as np
from scipy.optimize import least_squares

import camera_final as cf  # runs the per-camera fits on import; reuse its helpers and data

cams = json.load(open("cameras_fit.json"))
sets = []
for cam, fn in [("floor", "extreme_floor_0_46.json"), ("top", "extreme_top_6_66.json")]:
    js = json.load(open(fn))
    obs = [o for o in js["obs"] if o["res"] < 25]
    rows = np.array([cf.key[(o["ep"], o["f"])] for o in obs])
    UV = np.array([[o["u"], o["v"]] for o in obs])
    T = cf.normalized_to_ticks(cf.st[rows], cf.CAL)
    c = cams[cam]
    import cv2
    rv, _ = cv2.Rodrigues(np.array(c["R_cam_from_base"]))
    cp = np.r_[rv.ravel(), c["t_cam_from_base_mm"], c["K"][0][0]]
    sets.append((cam, T, UV, cp))

FREE = [1, 2, 3]


def resid(x):
    d = np.zeros(6)
    d[FREE] = x[14:]
    out = []
    for k, (cam, T, UV, _) in enumerate(sets):
        Tm = T + d / 360 * 4096  # q increases by d degrees
        P = cf.model_pts(Tm, np.ones(6))
        pr, _ = cf.extreme(x[7 * k:7 * k + 7], P)
        out.append((pr - UV).ravel())
    return np.concatenate(out)


x0 = np.r_[sets[0][3], sets[1][3], np.zeros(len(FREE))]
s = least_squares(resid, x0, loss="soft_l1", f_scale=6.0,
                  x_scale=np.r_[[0.05] * 3, [20] * 4, [0.05] * 3, [20] * 4, [1.0] * len(FREE)], diff_step=1e-4)
r = np.linalg.norm(s.fun.reshape(-1, 2), axis=1)
J = s.jac
cov = np.linalg.pinv(J.T @ J) * (1.4826 * np.median(np.abs(s.fun))) ** 2
sd = np.sqrt(np.diag(cov))[14:]
r0 = np.linalg.norm(resid(np.r_[x0[:14], np.zeros(len(FREE))]).reshape(-1, 2), axis=1)
print("no correction: median %.2f px" % np.median(r0))
print("corrections (deg) lift/elbow/wrist:", s.x[14:].round(2), "+-", sd.round(2), "median %.2f px" % np.median(r))
json.dump(dict(corr_deg=s.x[14:].tolist(), sd=sd.tolist(), median_px=float(np.median(r)),
               median_px_nocorr=float(np.median(r0))), open("offsets_joint.json", "w"))

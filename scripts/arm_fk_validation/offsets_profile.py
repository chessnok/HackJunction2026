"""Profile the camera-fit cost over a fixed zero correction of one joint (others 0); each point
refits both cameras. Robust cost = sum of soft_l1 over both cameras."""
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

import camera_final as cf

cams = json.load(open("cameras_fit.json"))
sets = []
for cam, fn in [("floor", "extreme_floor_0_46.json"), ("top", "extreme_top_6_66.json")]:
    js = json.load(open(fn))
    obs = [o for o in js["obs"] if o["res"] < 25]
    rows = np.array([cf.key[(o["ep"], o["f"])] for o in obs])
    UV = np.array([[o["u"], o["v"]] for o in obs])
    T = cf.normalized_to_ticks(cf.st[rows], cf.CAL)
    c = cams[cam]
    rv, _ = cv2.Rodrigues(np.array(c["R_cam_from_base"]))
    cp = np.r_[rv.ravel(), c["t_cam_from_base_mm"], c["K"][0][0]]
    sets.append((cam, T, UV, cp))

out = {}
for j in (1, 2, 3):
    for dval in (-6, -3, 0, 3, 6):
        d = np.zeros(6)
        d[j] = dval
        cost, meds = 0.0, []
        for cam, T, UV, cp in sets:
            P = cf.model_pts(T + d / 360 * 4096, np.ones(6))
            s = least_squares(lambda x: (cf.extreme(x, P)[0] - UV).ravel(), cp, loss="soft_l1", f_scale=6.0,
                              x_scale=np.r_[0.05, 0.05, 0.05, 20, 20, 20, 20])
            cost += s.cost
            meds.append(float(np.median(np.linalg.norm(s.fun.reshape(-1, 2), axis=1))))
        out[f"{cf.model.JOINTS[j]} {dval:+d}"] = dict(cost=cost, median_floor=meds[0], median_top=meds[1])
        print(cf.model.JOINTS[j], dval, "cost %.0f  median floor %.2f top %.2f px" % (cost, *meds), flush=True)
json.dump(out, open("offsets_profile.json", "w"), indent=1)

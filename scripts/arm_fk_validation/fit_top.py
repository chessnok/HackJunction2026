"""Top camera: init from the brick-centroid fit, then gripper extreme-point fit with outlier rejection."""
import json
import sys

import cv2
import numpy as np

sys.argv = ["fit_cam", "top", "6", "66"]
import fit_cam  # noqa: E402

P = fit_cam.grasp_point(np.zeros(5), np.full(6, 2047.0))
cp_b, res_b, _ = fit_cam.fit_cam(P, fit_cam.uv, False)
print("brick fit: rms %.1f px f=%.0f" % (np.sqrt((res_b ** 2).mean()), cp_b[6]))

sys.argv = ["fit_extreme", "top", "6", "66"]
import fit_extreme as fe  # noqa: E402

c2047 = np.full(6, 2047.0)
P = fe.model_points(np.zeros(5), c2047)
pred = fe.extreme(cp_b[:7], P)
err = np.linalg.norm(pred - fe.UV, axis=1)
print("init residuals: median %.1f, <40px: %d of %d" % (np.median(err), (err < 40).sum(), len(err)))
keep = err < 40
for it in range(3):
    UV_all, T_all = fe.UV, fe.T
    fe.UV, fe.T = UV_all[keep], T_all[keep]
    s, res = fe.fit(c2047, (), cp_b[:7] if it == 0 else s.x[:7])
    fe.UV, fe.T = UV_all, T_all
    P = fe.model_points(np.zeros(5), c2047)
    err = np.linalg.norm(fe.extreme(s.x[:7], P) - fe.UV, axis=1)
    keep = err < 25
    print(f"iter {it}: fit on {len(res)}: median {np.median(res):.1f}px rms {np.sqrt((res**2).mean()):.1f} f={s.x[6]:.0f};"
          f" all obs median {np.median(err):.1f}, inliers(<25) {keep.sum()}")
R, _ = cv2.Rodrigues(s.x[:3])
print("cam centre", (-R.T @ s.x[3:6]).round(0))
cp_2047 = s.x[:7].copy()
# compare hypotheses and free offsets on the inlier set
UV_all, T_all = fe.UV, fe.T
fe.UV, fe.T = UV_all[keep], T_all[keep]
for name, c in [("2047", c2047), ("mid", fe.vfk.MID)]:
    s0, r0 = fe.fit(c, (), cp_2047)
    s1, r1 = fe.fit(c, [1, 2, 3], s0.x[:7])
    Jm = s1.jac
    cov = np.linalg.pinv(Jm.T @ Jm) * np.median(r1) ** 2
    sd = np.sqrt(np.abs(np.diag(cov)))[7:]
    print(f"{name}: n={len(r0)} median {np.median(r0):.1f}px rms {np.sqrt((r0**2).mean()):.1f} f={s0.x[6]:.0f} | free lift/elbow/wrist "
          f"{s1.x[7:].round(2)} +- {sd.round(2)} median {np.median(r1):.1f} rms {np.sqrt((r1**2).mean()):.1f} f={s1.x[6]:.0f}")
fe.UV, fe.T = UV_all, T_all
json.dump(dict(cp=cp_2047.tolist(), obs=[dict(o, res=float(r)) for o, r in zip(fe.obs, err)]),
          open("extreme_top_6_66.json", "w"), indent=1)

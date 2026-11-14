"""Final camera fits on cached gripper extreme-point observations; sign checks for pan / roll;
export camera models (robot base frame -> camera) for agent C."""
import json
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from hackjunction2026.kinematics import Calibration, normalized_to_ticks  # noqa: E402
from hackjunction2026.kinematics import model  # noqa: E402
from data import load  # noqa: E402
from meshes import link_meshes  # noqa: E402

W, H = 848, 480
CAL = Calibration.from_file("/home/intel/dev/configs/calibration/follower.json")
df, st, ac = load()
key = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(df.episode_index.values, df.frame_index.values))}
LM = link_meshes()
rng = np.random.default_rng(1)
G = np.vstack([m.vertices for fn, m in LM["gripper_link"]]) * 1000
G = G[rng.choice(len(G), 4000, replace=False)]
MJ = np.vstack([m.vertices for fn, m in LM["moving_jaw_so101_v1_link"]]) * 1000
MJ = MJ[rng.choice(len(MJ), 3000, replace=False)]
ZERO = np.array([2047, 2047, 2047, 2047, 2047, 2201.0])


def model_pts(T, sign):
    q = sign * (T - ZERO) * 2 * np.pi / 4096
    F = model.link_frames(q)
    pg = np.einsum("nij,mj->nmi", F["gripper_link"][:, :3, :3], G) + F["gripper_link"][:, None, :3, 3]
    pm = np.einsum("nij,mj->nmi", F["moving_jaw_so101_v1_link"][:, :3, :3], MJ) + F["moving_jaw_so101_v1_link"][:, None, :3, 3]
    return np.concatenate([pg, pm], 1)


def project(cp, P):
    R, _ = cv2.Rodrigues(cp[:3])
    X = P @ R.T + cp[3:6]
    return np.stack([cp[6] * X[..., 0] / X[..., 2] + W / 2, cp[6] * X[..., 1] / X[..., 2] + H / 2], -1), X[..., 2]


def extreme(cp, P):
    uv, z = project(cp, P)
    i = np.argmax(uv[..., 1], 1)
    n = np.arange(len(uv))
    return uv[n, i], z[n, i]


def fit(cp0, P, UV):
    s = least_squares(lambda x: (extreme(x, P)[0] - UV).ravel(), cp0, loss="soft_l1", f_scale=6.0,
                      x_scale=np.r_[0.05, 0.05, 0.05, 20, 20, 20, 20])
    pr, depth = extreme(s.x, P)
    return s.x, np.linalg.norm(pr - UV, axis=1), depth


def fit_multi_start(cp0, P, UV):
    best = None
    for dr in ([0, 0, 0], [0, 0, 0.3], [0, 0, -0.3], [0.2, 0, 0], [-0.2, 0, 0]):
        x0 = cp0.copy()
        x0[:3] = x0[:3] + np.array(dr)
        x, r, d = fit(x0, P, UV)
        if best is None or np.median(r) < np.median(best[1]):
            best = (x, r, d)
    return best



if __name__ == "__main__":
    out = {}
    for cam, fn, groups in [("floor", "extreme_floor_0_46.json", "0-46"), ("top", "extreme_top_6_66.json", "6-66")]:
        js = json.load(open(fn))
        obs = [o for o in js["obs"] if o["res"] < 25]
        rows = np.array([key[(o["ep"], o["f"])] for o in obs])
        UV = np.array([[o["u"], o["v"]] for o in obs])
        T = normalized_to_ticks(st[rows], CAL)
        cp0 = np.array(js["cp"][:7])
        report = {}
        for name, sign in [("chosen", np.ones(6)), ("pan_flipped", np.array([-1, 1, 1, 1, 1, 1.0])),
                           ("roll_flipped", np.array([1, 1, 1, 1, -1, 1.0])),
                           ("gripper_flipped", np.array([1, 1, 1, 1, 1, -1.0]))]:
            P = model_pts(T, sign)
            x, r, d = fit_multi_start(cp0, P, UV) if name != "chosen" else fit(cp0, P, UV)
            R, _ = cv2.Rodrigues(x[:3])
            C = -R.T @ x[3:6]
            mm = r * d / x[6]
            report[name] = dict(n=len(r), median_px=float(np.median(r)), rms_px=float(np.sqrt((r ** 2).mean())),
                                p90_px=float(np.percentile(r, 90)), median_mm=float(np.median(mm)),
                                p90_mm=float(np.percentile(mm, 90)), f=float(x[6]), cam_centre_mm=C.round(1).tolist())
            print(cam, name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in report[name].items()})
            if name == "chosen":
                cp_final = x
        R, _ = cv2.Rodrigues(cp_final[:3])
        out[cam] = dict(
            episodes=groups,
            image_size=[W, H],
            K=[[cp_final[6], 0, W / 2], [0, cp_final[6], H / 2], [0, 0, 1]],
            R_cam_from_base=R.tolist(), t_cam_from_base_mm=cp_final[3:6].tolist(),
            camera_centre_in_base_mm=(-R.T @ cp_final[3:6]).tolist(),
            model="pinhole, no distortion, principal point fixed at image centre; x_cam = R @ x_base + t",
            fit=report,
        )
    json.dump(out, open("cameras_fit.json", "w"), indent=1)

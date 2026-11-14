"""Lowest point of the moving arm (all link meshes) per frame; rest pose check; grasp heights."""
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from hackjunction2026.kinematics import Calibration, normalized_to_urdf, link_frames, arm_keypoints
from data import load, grasp_events
from meshes import link_meshes

df, st, ac = load()
cal = Calibration.from_file("/home/intel/dev/configs/calibration/follower.json")
q = normalized_to_urdf(st, cal)
LM = link_meshes()
rng = np.random.default_rng(0)
pts = {}
for ln in ["shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "moving_jaw_so101_v1_link"]:
    v = np.vstack([m.vertices for _, m in LM[ln]]) * 1000
    pts[ln] = v[rng.choice(len(v), min(len(v), 2500), replace=False)]
zmin = np.full(len(q), np.inf)
zlink = np.empty(len(q), dtype=object)
for i0 in range(0, len(q), 2000):
    F = link_frames(q[i0:i0 + 2000])
    best = np.full(len(F["base_link"]), np.inf)
    bl = np.array(["" for _ in range(len(best))], dtype=object)
    for ln, p in pts.items():
        T = F[ln]
        z = (np.einsum("nj,mj->nm", T[:, 2, :3], p) + T[:, 2, 3][:, None]).min(1)
        better = z < best
        best[better] = z[better]
        bl[better] = ln
    zmin[i0:i0 + 2000] = best
    zlink[i0:i0 + 2000] = bl
print("lowest arm point over all %d frames: min %.1f mm, 0.1%% %.1f, 1%% %.1f, median %.1f" %
      (len(zmin), zmin.min(), np.percentile(zmin, 0.1), np.percentile(zmin, 1), np.median(zmin)))
below = zmin < -2.4 - 5
print("frames with arm > 5 mm below the base bottom (z < -7.4):", below.sum(),
      "episodes:", sorted(set(df.episode_index.values[below].tolist())))
i = np.argmin(zmin)
print("global lowest: ep", df.episode_index.values[i], "frame", df.frame_index.values[i], "link", zlink[i],
      "state", st[i].round(1))
# rest pose at episode start
first = df.groupby("episode_index").head(1).index.values
print("episode-start lowest arm point: min %.1f median %.1f max %.1f mm" % (zmin[first].min(), np.median(zmin[first]), zmin[first].max()))
kp = arm_keypoints(st[first], cal)
for k in ["shoulder", "elbow", "wrist", "gripper_base", "gripper_tip"]:
    print("  rest", k, "median", np.median(kp[k], 0).round(0), "z range", kp[k][:, 2].min().round(0), kp[k][:, 2].max().round(0))
np.save("zmin_all.npy", zmin)

"""Derive keypoint constants and the jaw-gap table from the URDF meshes (printed for the library)."""
import numpy as np
from scipy.spatial import cKDTree

import vfk
from meshes import link_meshes
from urdf_fk import rot_axis

LM = link_meshes()
fixed = [m for fn, m in LM["gripper_link"] if "wrist_roll_follower" in fn][0]
mov = LM["moving_jaw_so101_v1_link"][0][1]
fv = fixed.vertices * 1000
mv = mov.vertices * 1000
ft = fv[fv[:, 2] < fv[:, 2].min() + 1.0].mean(0)
mt = mv[mv[:, 1] < mv[:, 1].min() + 1.0].mean(0)
print("FIXED_JAW_TIP (gripper_link, mm)", ft.round(2))
print("MOVING_JAW_TIP (moving jaw link, mm)", mt.round(2))
base = np.vstack([m.vertices for _, m in LM["base_link"]]) * 1000
print("base bottom z", base[:, 2].min().round(2))

Tj = vfk.J["gripper"]["T"]
fp = fixed.sample(80000) * 1000
mp_local = mov.sample(80000)
tip_z = fp[:, 2].min()
fz = fp[fp[:, 2] < tip_z + 35]
tree = cKDTree(fz)
qs = np.arange(-14.0, 46.0, 1.0)
gaps = []
for deg in qs:
    T = Tj @ rot_axis(vfk.J["gripper"]["axis"], np.radians(deg))
    mp = (mp_local @ T[:3, :3].T + T[:3, 3]) * 1000
    mz = mp[mp[:, 2] < mp[:, 2].min() + 35]
    d, _ = tree.query(mz)
    gaps.append(d.min())
gaps = np.array(gaps)
# touch angle: first q where gap > 0.3 mm, interpolate
for a, b, ga, gb in zip(qs[:-1], qs[1:], gaps[:-1], gaps[1:]):
    print(f"{a:5.1f} {ga:5.2f}")
print("GAP_TABLE_Q_DEG =", qs.tolist())
print("GAP_TABLE_MM =", np.round(np.maximum.accumulate(gaps), 2).tolist())

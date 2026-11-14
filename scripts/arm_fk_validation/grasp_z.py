import numpy as np
from data import load, grasp_events
from meshes import link_meshes
import vfk

df, st, ac = load()
ev = grasp_events(df, st, ac)
# dedupe: one event per episode (first)
seen = set(); E = []
for e in ev:
    if e[0] not in seen:
        seen.add(e[0]); E.append(e)
rows = np.array([e[2] for e in E])
LM = link_meshes()
rng = np.random.default_rng(0)
fixed = np.vstack([m.vertices for fn, m in LM["gripper_link"] if "wrist_roll_follower" in fn])
mov = np.vstack([m.vertices for fn, m in LM["moving_jaw_so101_v1_link"]])
fixed = fixed[rng.choice(len(fixed), 3000, replace=False)]
mov = mov[rng.choice(len(mov), 3000, replace=False)]
np.save("fixed_pts.npy", fixed); np.save("mov_pts.npy", mov)

t = vfk.norm_to_ticks(st[rows])
for name, centers in [("mid", vfk.MID), ("2047", np.full(6, 2047.0))]:
    q = vfk.q_from_ticks(t, centers)
    L = vfk.fk_all(q)
    pf = vfk.apply(L["gripper_link"], fixed)
    pm = vfk.apply(L["moving_jaw_so101_v1_link"], mov)
    zlow = np.minimum(pf[:, :, 2].min(1), pm[:, :, 2].min(1)) * 1000
    gf = L["gripper_frame_link"][:, :3, 3] * 1000
    r = np.linalg.norm(gf[:, :2], axis=1)
    print(name, "lowest jaw z: mean %.1f std %.1f min %.1f max %.1f" % (zlow.mean(), zlow.std(), zlow.min(), zlow.max()))
    A = np.c_[np.ones_like(r), gf[:, 0], gf[:, 1]]
    coef, *_ = np.linalg.lstsq(A, zlow, rcond=None)
    print("   plane fit z = %.1f + %.4f x + %.4f y ; resid std %.1f" % (*coef, (zlow - A @ coef).std()))
    print("   corr(z, reach r) %.2f" % np.corrcoef(zlow, r)[0, 1])
    for e, z, p in zip(E, zlow, gf):
        pass
    print("   z per event:", np.round(zlow, 0).astype(int).tolist())
    print("   reach r:", np.round(r, 0).astype(int).tolist())

import numpy as np
from data import load
df, st, ac = load()
g, a = st[:, 5], ac[:, 5]
sel = a < 3
print("frames with action gripper < 3:", sel.sum())
h, e = np.histogram(g[sel], bins=np.arange(0, 40, 1))
for hh, ee in zip(h, e):
    if hh:
        print(f"state {ee:4.0f}-{ee+1:<4.0f} {hh}")
# per-episode min gripper state
mins = df.assign(g=g).groupby("episode_index").g.min()
print("per-episode min gripper state:", np.round(mins.values, 1).tolist())

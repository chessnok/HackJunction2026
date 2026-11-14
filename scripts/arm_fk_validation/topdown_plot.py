#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyarrow", "matplotlib"]
# ///
"""Top-down (x-y) and side (reach-z) view of the annotated arm keypoints, as a sanity picture of
the robot base frame. Reads captures/perception/B/arm_<dataset>.parquet and grasps_<dataset>.csv.

  uv run scripts/arm_fk_validation/topdown_plot.py [captures/perception/B/arm_<dataset>.parquet]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/intel/dev/captures/perception/B")
src = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(OUT.glob("arm_*.parquet"))[0]
t = pq.read_table(src).to_pydict()
ds = src.stem.removeprefix("arm_")
ep = np.array(t["episode_index"])
P = {k: np.c_[t[f"{k}_x_mm"], t[f"{k}_y_mm"], t[f"{k}_z_mm"]] for k in
     ("base", "shoulder", "elbow", "wrist", "gripper_base", "gripper_tip")}
grasps = list(csv.DictReader(open(OUT / f"grasps_{ds}.csv")))
gx = np.array([float(g["gripper_tip_x_mm"]) for g in grasps])
gy = np.array([float(g["gripper_tip_y_mm"]) for g in grasps])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e"})
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), dpi=130, gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor("#fcfcfb")
ax = axs[0]
ax.set_facecolor("#fcfcfb")
ax.plot(P["gripper_tip"][:, 0], P["gripper_tip"][:, 1], ",", color="#86b6ef", alpha=0.5)
ax.scatter(gx, gy, s=22, color="#eb6834", edgecolors="#fcfcfb", linewidths=0.8, zorder=3,
           label=f"gripper tip at {len(grasps)} grasps")
ax.scatter([0], [0], marker="s", s=40, color="#0b0b0b", zorder=4, label="base_link origin")
ax.scatter([38.8], [0], marker="+", s=60, color="#0b0b0b", zorder=4, label="shoulder_pan axis")
ax.annotate("", xy=(80, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#0b0b0b"))
ax.annotate("", xy=(0, 80), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#0b0b0b"))
ax.text(84, -4, "+x", fontsize=9)
ax.text(-6, 84, "+y", fontsize=9)
ax.set_aspect("equal")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("Top view, robot base frame (light: gripper tip, all frames)", fontsize=10, loc="left")
ax.grid(color="#e4e3df", lw=0.6)
ax.legend(frameon=False, fontsize=8, loc="lower left")

ax = axs[1]
ax.set_facecolor("#fcfcfb")
e0 = 20
sel = np.nonzero(ep == e0)[0]
for i in sel[::12]:
    pts = np.array([P[k][i] for k in ("base", "shoulder", "elbow", "wrist", "gripper_base", "gripper_tip")])
    r = np.hypot(pts[:, 0], pts[:, 1])
    ax.plot(r, pts[:, 2], "-o", ms=2.5, lw=1, color="#2a78d6", alpha=0.55)
ax.axhline(-2.4, color="#52514e", lw=1, ls="--")
ax.text(5, 2, "table (z = -2.4)", color="#52514e", fontsize=8)
ax.set_aspect("equal")
ax.set_xlabel("horizontal distance from base_link origin (mm)")
ax.set_ylabel("z (mm)")
ax.set_title(f"Side view, episode {e0}, every 12th frame", fontsize=10, loc="left")
ax.grid(color="#e4e3df", lw=0.6)
for a in axs:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fk_topdown_side.png")
print(OUT / "fk_topdown_side.png")

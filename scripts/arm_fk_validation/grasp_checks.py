#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "matplotlib"]
# ///
"""Data checks of the SO-101 joint mapping (no images): grasp heights, rest pose, lowest point.

Run from the repo root:
  uv run scripts/arm_fk_validation/grasp_checks.py [dataset-id] [--out captures/perception/B]

Prints a table and writes validation_grasps.json, grasp_height_vs_reach.png and
lowest_point_hist.png to --out. Read-only on the dataset.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hackjunction2026.kinematics import (
    JOINTS,
    TABLE_Z_MM,
    Calibration,
    keypoints,
    normalized_to_ticks,
    normalized_to_urdf,
)
from hackjunction2026.kinematics.mapping import JOINT_MAP

DATASETS = Path.home() / ".local/share/physicalai/datasets"


def load(ds: Path):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ds / "data/chunk-*/*.parquet")))])
    df = df.sort_values("index").reset_index(drop=True)
    st = np.stack(df["observation.state"].values).astype(np.float64)
    ac = np.stack(df["action"].values).astype(np.float64)
    return df, st, ac


def grasp_events(df, st, ac, min_stall=10, gap=6.0):
    """First frame + 5 of each run where the gripper is commanded > 6 units more closed than it is,
    is not moving and is not empty-closed (> 5): the jaws hold an object. One event per episode."""
    ev = {}
    for ep, d in df.groupby("episode_index"):
        rows = d.index.values
        g, a = st[rows, 5], ac[rows, 5]
        stall = (a < g - gap) & (g > 5) & (np.abs(np.gradient(g)) < 0.5)
        i = 0
        while i < len(rows):
            if stall[i]:
                j = i
                while j < len(rows) and stall[j]:
                    j += 1
                if j - i >= min_stall and ep not in ev:
                    ev[ep] = int(rows[min(i + 5, j - 1)])
                i = j
            else:
                i += 1
    return ev


def q_with(ticks, zero, sign):
    return sign * (ticks - zero) * 2 * np.pi / 4096


def jaw_low_z(q):
    kp = keypoints(q)
    return np.minimum(kp["fixed_jaw_tip"][:, 2], kp["moving_jaw_tip"][:, 2]), kp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", nargs="?", default="431a71f7")
    p.add_argument("--calibration", default=str(REPO / "configs/calibration/follower.json"))
    p.add_argument("--out", default="/home/intel/dev/captures/perception/B")
    args = p.parse_args()
    ds = sorted(DATASETS.glob(f"{args.dataset}*"))[0] if not Path(args.dataset).exists() else Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = Calibration.from_file(args.calibration)
    df, st, ac = load(ds)
    res: dict = {"dataset": ds.name, "frames": len(df), "episodes": int(df.episode_index.nunique())}

    # 1. the calibration file is the one used for recording: every state maps to an integer tick
    t_all = normalized_to_ticks(st, cal)
    res["max_tick_fraction"] = float(np.abs(t_all - np.round(t_all)).max())
    print(f"{res['frames']} frames / {res['episodes']} episodes; state -> ticks max |t - round(t)| = "
          f"{res['max_tick_fraction']:.4f} (0 = recorded with this calibration)")

    # 2. grasp heights under candidate mappings
    ev = grasp_events(df, st, ac)
    rows = np.array(list(ev.values()))
    res["grasp_events"] = len(rows)
    t = normalized_to_ticks(st[rows], cal)
    zero = np.array([JOINT_MAP[j].zero_tick for j in JOINTS])
    sign = np.array([JOINT_MAP[j].sign for j in JOINTS], float)
    mid = (cal.range_min + cal.range_max) / 2
    cands = {"chosen (zero = homing tick 2047)": q_with(t, zero, sign)}
    z_mid = zero.copy()
    z_mid[:5] = mid[:5]
    cands["zero = range midpoint (Studio 'degrees')"] = q_with(t, z_mid, sign)
    qui = np.radians(st[rows].copy())  # Studio UI viewer: normalized value used as degrees
    cands["Studio UI viewer (normalized as degrees)"] = qui
    for i, j in enumerate(JOINTS[:5]):
        s2 = sign.copy()
        s2[i] = -s2[i]
        cands[f"sign of {j} flipped"] = q_with(t, zero, s2)
    print(f"\n{len(rows)} grasp events. Lowest fingertip z (mm, base frame); table/base bottom at {TABLE_Z_MM}")
    print(f"{'mapping':45s} {'mean':>7s} {'std':>6s} {'min':>7s} {'max':>7s} {'slope z/reach':>14s}")
    res["candidates"] = {}
    for name, q in cands.items():
        z, kp = jaw_low_z(q)
        r = np.linalg.norm(kp["gripper_tip"][:, :2], axis=1)
        slope = np.polyfit(r, z, 1)[0]
        res["candidates"][name] = dict(mean=float(z.mean()), std=float(z.std()), min=float(z.min()),
                                       max=float(z.max()), slope_per_100mm=float(slope * 100))
        print(f"{name:45s} {z.mean():7.1f} {z.std():6.1f} {z.min():7.1f} {z.max():7.1f} {slope * 100:10.1f}/100mm")

    # plane fit through the grasp fingertips (chosen mapping)
    z, kp = jaw_low_z(cands["chosen (zero = homing tick 2047)"])
    xy = kp["gripper_tip"][:, :2]
    A = np.c_[np.ones(len(z)), xy]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = z - A @ coef
    cov = np.linalg.inv(A.T @ A) * resid.var(ddof=3)
    se = np.sqrt(np.diag(cov))
    tilt = np.degrees(np.arctan(np.hypot(coef[1], coef[2])))
    res["grasp_plane"] = dict(z0=float(coef[0]), dz_dx=float(coef[1]), dz_dy=float(coef[2]), se=se.tolist(),
                              tilt_deg=float(tilt), resid_std=float(resid.std()))
    print(f"\nplane through grasp fingertips: z = {coef[0]:.1f} + {coef[1]:.4f} x + {coef[2]:.4f} y "
          f"(se {se.round(4).tolist()}), tilt {tilt:.2f} deg, residual std {resid.std():.1f} mm")
    print(f"grasp fingertips above the base-bottom plane: {np.mean(z - TABLE_Z_MM):.1f} +- {np.std(z):.1f} mm "
          f"(standard error of the mean {np.std(z) / np.sqrt(len(z)):.1f})")

    # gripper opening at the grasps
    qg = cands["chosen (zero = homing tick 2047)"][:, 5]
    from hackjunction2026.kinematics import jaw_gap_mm
    gaps = jaw_gap_mm(qg)
    res["grasp_gripper_norm"] = st[rows, 5].round(1).tolist()
    res["grasp_jaw_gap_mm"] = gaps.round(1).tolist()
    tip = kp["gripper_tip"]
    pd.DataFrame({
        "episode_index": df.episode_index.values[rows], "frame_index": df.frame_index.values[rows],
        "gripper_norm": st[rows, 5].round(2), "jaw_gap_mm": gaps.round(1),
        "gripper_tip_x_mm": tip[:, 0].round(1), "gripper_tip_y_mm": tip[:, 1].round(1),
        "gripper_tip_z_mm": tip[:, 2].round(1), "fingertip_low_z_mm": z.round(1),
    }).to_csv(out / f"grasps_{ds.name}.csv", index=False)

    # 3. rest pose (first frame of every episode) and lowest fingertip over all frames
    q_all = normalized_to_urdf(st, cal)
    kp_all = keypoints(q_all)
    low_all = np.minimum(kp_all["fixed_jaw_tip"][:, 2], kp_all["moving_jaw_tip"][:, 2])
    first = df.groupby("episode_index").head(1).index.values
    res["rest"] = {k: np.median(kp_all[k][first], 0).round(1).tolist()
                   for k in ("shoulder", "elbow", "wrist", "gripper_base", "gripper_tip")}
    res["rest_min_z_fingertip"] = float(low_all[first].min())
    pct = np.percentile(low_all, [0, 0.1, 1, 5, 50])
    res["all_frames_fingertip_low_z_percentiles"] = dict(zip(["min", "p0.1", "p1", "p5", "median"], pct.tolist()))
    print("\nrest pose (episode start) median keypoints:", res["rest"])
    print("lowest fingertip over all frames: min %.1f, 0.1%% %.1f, 1%% %.1f, 5%% %.1f, median %.1f mm" % tuple(pct))
    elbow_low = kp_all["elbow"][:, 2].min()
    res["min_elbow_z"] = float(elbow_low)

    json.dump(res, open(out / "validation_grasps.json", "w"), indent=1)

    # plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for name, col, mk in [("chosen (zero = homing tick 2047)", "#2a78d6", "o"),
                          ("zero = range midpoint (Studio 'degrees')", "#eb6834", "s")]:
        zz, kk = jaw_low_z(cands[name])
        rr = np.linalg.norm(kk["gripper_tip"][:, :2], axis=1)
        ax.scatter(rr, zz, s=22, color=col, marker=mk, label=f"{name}: {zz.mean():.1f} ± {zz.std():.1f} mm",
                   edgecolors="#fcfcfb", linewidths=0.8)
    ax.axhline(TABLE_Z_MM, color="#52514e", lw=1, ls="--")
    ax.text(ax.get_xlim()[0] + 3, TABLE_Z_MM + 1, "base bottom / table", color="#52514e", fontsize=9)
    ax.set_xlabel("horizontal reach of the gripper tip (mm)")
    ax.set_ylabel("lowest fingertip z at grasp (mm)")
    ax.set_title(f"Fingertip height at {len(rows)} grasps, two zero conventions", fontsize=11, loc="left")
    ax.grid(color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "grasp_height_vs_reach.png")

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.hist(low_all, bins=np.arange(-20, 260, 4), color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.6)
    ax.axvline(TABLE_Z_MM, color="#52514e", lw=1, ls="--")
    ax.set_xlabel("lowest fingertip z (mm), every frame")
    ax.set_ylabel("frames")
    ax.set_title(f"Lowest fingertip height over {len(low_all)} frames (dashed: table)", fontsize=11, loc="left")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "lowest_point_hist.png")
    print("wrote", out / "validation_grasps.json", out / "grasp_height_vs_reach.png", out / "lowest_point_hist.png")


if __name__ == "__main__":
    main()

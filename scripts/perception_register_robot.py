#!/usr/bin/env python3
"""Register agent B's robot base frame (FK, metric mm) to agent A's table frame with grasps.

At each grasp (B: grasps_<dataset>.csv, frame where the jaws hold the brick) the FK gripper tip
(midpoint of the fingertips) is at the brick. A's brick position at that frame (objects parquet,
table frame) is the same point seen by the cameras. A 2D similarity (rotation, translation, scale)
robot xy -> table xy is fitted with RANSAC + least squares; its scale measures A's mm scale error.
B's frame has z up, A's table frame z down (into the table), so the 2D map contains a mirror:
the fit is done on (x_robot, -y_robot).

Writes <out>/robot_to_table_<dataset>.json. Needs B's files in /home/intel/dev/captures/perception/B/.

  PYTHONPATH=src python scripts/perception_register_robot.py 431a71f7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hackjunction2026.perception.geometry import apply_h, fit_similarity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default="/home/intel/dev/captures/perception/A")
    ap.add_argument("--b-dir", default="/home/intel/dev/captures/perception/B")
    ap.add_argument("--window", type=int, default=5, help="+- frames of brick positions averaged around the grasp")
    args = ap.parse_args()
    out = Path(args.out)
    ds_id = next(out.glob(f"objects_{args.dataset}*.parquet")).name[len("objects_"):-len(".parquet")]
    g = pd.read_csv(Path(args.b_dir) / f"grasps_{ds_id}.csv")
    o = pd.read_parquet(out / f"objects_{ds_id}.parquet")
    src, dst, eps = [], [], []
    for _, r in g.iterrows():
        e = o[(o.episode_index == r.episode_index) & (abs(o.frame_index - r.frame_index) <= args.window)
              & (o.source != "hold")]
        if e.empty:
            continue
        # the brick position just before the jaws close (it does not move while they close)
        before = o[(o.episode_index == r.episode_index) & (o.frame_index <= r.frame_index) & (o.source == "top+floor")
                   & ~o.lifted].tail(2 * args.window + 1)
        e = before if len(before) else e
        src.append((r.gripper_tip_x_mm, -r.gripper_tip_y_mm))  # mirror: z up (B) vs z down (A)
        dst.append((e.x_mm.median(), e.y_mm.median()))
        eps.append(int(r.episode_index))
    src, dst = np.array(src), np.array(dst)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(2000):
        idx = rng.choice(len(src), 3, replace=False)
        M = fit_similarity(src[idx], dst[idx])
        err = np.linalg.norm(apply_h(M, src) - dst, axis=1)
        inl = err < 15.0
        if best is None or inl.sum() > best.sum():
            best = inl
    M = fit_similarity(src[best], dst[best])
    err = np.linalg.norm(apply_h(M, src) - dst, axis=1)
    s = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    Mr = fit_similarity(src[best], dst[best], scale=False)
    err_r = np.linalg.norm(apply_h(Mr, src) - dst, axis=1)
    res = {
        "dataset": ds_id,
        "model": "table_xy = s * R(theta) @ (x_robot, -y_robot) + t  (robot = SO-101 URDF base_link, mm, z up; "
                 "table = A's frame, mm, z into the table); table_z = -(z_robot + 2.4) * s",
        "M_robot_to_table_3x3": M.tolist(),
        "scale_s": s, "theta_deg": rot, "t_mm": M[:2, 2].tolist(),
        "robot_base_origin_in_table_mm": M[:2, 2].tolist(),
        "n_grasps": len(src), "n_inliers": int(best.sum()),
        "outlier_episodes": [eps[i] for i in np.flatnonzero(~best)],
        "residual_mm": {"median": float(np.median(err[best])), "p90": float(np.percentile(err[best], 90)),
                        "max_inlier": float(err[best].max())},
        "rigid_fit_residual_mm": {"median": float(np.median(err_r[best])), "p90": float(np.percentile(err_r[best], 90))},
        "M_robot_to_table_rigid_3x3": Mr.tolist(),
        "note": "scale_s != 1 means A's table mm differ from metric mm by that factor; to get metric table "
                "coordinates multiply A's mm by 1/scale_s about the origin, or use the rigid fit after rescaling",
    }
    (out / f"robot_to_table_{ds_id}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("M_")}, indent=1))


if __name__ == "__main__":
    main()

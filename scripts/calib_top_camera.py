#!/usr/bin/env python3
"""Calibrate the top RealSense against the robot, with the arm itself as the marker (no human needed).

The follower visits a grid of points over the sheet with the gripper straight down and the jaws
closed. At each point the top camera grabs a frame and the lowest purple pixel near the predicted spot is
taken as the jaw tip. Its 3D position comes from forward kinematics of the measured pose. solvePnP with
the camera's factory intrinsics then gives the camera pose in the robot base frame, saved to
configs/top_camera.yaml.

Moves go through scripts/so101_goto.py (straight joint-space moves, lag stop, final check). Every move is
checked first: no arm point below the clearance. Ctrl+C stops between moves; the arm keeps holding.

Run with Studio's backend Python (SO-101 plugin, pyrealsense2, OpenCV), from the repo root, with Studio's
environment closed (the follower port and the camera must be free):
  PYTHONPATH=src third_party/physical-ai-studio/application/backend/.venv/bin/python scripts/calib_top_camera.py --dry-run
  PYTHONPATH=src third_party/physical-ai-studio/application/backend/.venv/bin/python scripts/calib_top_camera.py --go
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from hackjunction2026.kinematics import Calibration, arm_keypoints
from hackjunction2026.kinematics.ik import solve_down
from hackjunction2026.kinematics.path import check_straight_move

REPO = Path(__file__).resolve().parents[1]
PY = REPO / "third_party/physical-ai-studio/application/backend/.venv/bin/python"
TOP_SERIAL = "243622062352"          # Studio camera "top" (state snapshot fingerprint)
SIZE = (848, 480)
B_MODEL = REPO / "captures/perception/B/cameras_431a71f7-50c6-4e38-9e38-d85984dde78b.json"
OUT = REPO / "configs/top_camera.yaml"
CAPT = REPO / "captures/top_calib"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_CLOSED = 5.0                 # jaws almost touching: both tips ~ one point


def prior_model():
    m = json.loads(B_MODEL.read_text())["top"]
    return np.array(m["K"]), np.array(m["R_cam_from_base"]), np.array(m["t_cam_from_base_mm"])


def project(K, R, t, X):
    c = (R @ np.atleast_2d(X).T).T + t
    return (K @ c.T).T[:, :2] / c[:, 2:3]


def tip_point(norm: np.ndarray, cal: Calibration) -> np.ndarray:
    kp = arm_keypoints(norm[None], cal)
    return 0.5 * (kp["fixed_jaw_tip"][0] + kp["moving_jaw_tip"][0])


def candidate_targets(cal, K, R, t, start, heights=(40.0, 90.0)):
    pts = []
    for z in heights:
        for x in np.arange(120, 291, 40):
            for y in np.arange(-160, 161, 40):
                sol = solve_down(float(x), float(y), z, 90.0, cal, GRIPPER_CLOSED, start_norm=start)
                if not sol.ok:
                    continue
                uv = project(K, R, t, tip_point(sol.norm, cal))[0]
                if 50 < uv[0] < SIZE[0] - 50 and 40 < uv[1] < SIZE[1] - 30:
                    pts.append((np.array([x, y, z]), sol.norm, uv))
    return pts


def order_nearest(pts, start_norm):
    out, cur, rest = [], start_norm, list(pts)
    while rest:
        i = int(np.argmin([np.abs(p[1][:5] - cur[:5]).max() for p in rest]))
        out.append(rest.pop(i))
        cur = out[-1][1]
    return out


def goto(vec: np.ndarray, speed: float) -> np.ndarray | None:
    v = ",".join(f"{x:.2f}" for x in vec)
    r = subprocess.run([str(PY), str(REPO / "scripts/so101_goto.py"), "goto", "--speed", str(speed), "--go",
                        "--yes", "--", v], capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    m = re.search(r"^final\s+(.*)$", out, re.M)
    if r.returncode != 0 or "REACHED" not in out or not m:
        print(out[-1500:])
        return None
    vals = dict(re.findall(r"(\w+)=\s*(-?[\d.]+)", m.group(1)))
    return np.array([float(vals[j]) for j in JOINTS])


def lowest_purple(img, near_uv, radius=90):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= 120) & (h <= 160) & (s > 60) & (v > 40)).astype(np.uint8)
    win = np.zeros_like(mask)
    u0, v0 = int(near_uv[0]), int(near_uv[1])
    win[max(v0 - radius, 0):v0 + radius, max(u0 - radius, 0):u0 + radius] = 1
    mask = cv2.morphologyEx(mask * win, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 40:
        return None
    ys, xs = np.nonzero(lab == k)
    low = ys >= ys.max() - 3
    return np.array([xs[low].mean(), ys[low].mean()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="actually move the arm")
    ap.add_argument("--dry-run", action="store_true", help="plan the points only")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--clearance", type=float, default=25.0)
    ap.add_argument("--max-points", type=int, default=14)
    args = ap.parse_args()

    cal = Calibration.from_file(REPO / "configs/calibration/follower.json")
    poses = yaml.safe_load((REPO / "configs/poses.yaml").read_text())
    observe = np.array([poses["observe"]["joints"][j] for j in JOINTS])
    K0, R0, t0 = prior_model()
    pts = candidate_targets(cal, K0, R0, t0, observe)
    if len(pts) > args.max_points:          # spread: keep every k-th
        pts = pts[:: int(np.ceil(len(pts) / args.max_points))]
    pts = order_nearest(pts, observe)
    plan, prev = [], observe
    for target, norm, uv in pts:
        chk = check_straight_move(prev, norm, cal, min_clearance_mm=args.clearance)
        if not chk.ok:
            print(f"skip {target}: {chk.describe()}")
            continue
        plan.append((target, norm, uv))
        prev = norm
    back = check_straight_move(prev, observe, cal, min_clearance_mm=args.clearance)
    print(f"{len(plan)} points (jaw tip mm, predicted top px):")
    for target, norm, uv in plan:
        print(f"  {target.round(0)}  px {uv.round(0)}  joints {np.round(norm, 1)}")
    print(f"return to observe: {'ok' if back.ok else back.describe()}")
    if args.dry_run or not args.go:
        print("dry run: add --go to move")
        return
    if not back.ok:
        sys.exit("the return move fails the clearance check; refusing")

    import pyrealsense2 as rs
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(TOP_SERIAL)
    cfg.enable_stream(rs.stream.color, SIZE[0], SIZE[1], rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist = np.array(intr.coeffs[:5], dtype=float)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = CAPT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    obs3d, obs2d, rows = [], [], []
    try:
        for _ in range(30):
            pipe.wait_for_frames()
        for i, (target, norm, uv_pred) in enumerate(plan):
            reached = goto(norm, args.speed)
            if reached is None:
                print(f"point {i}: move failed, stopping")
                break
            time.sleep(0.6)
            for _ in range(5):
                frames = pipe.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data()).copy()
            X = tip_point(reached, cal)
            uv = lowest_purple(img, project(K0, R0, t0, X)[0])
            cv2.imwrite(str(out_dir / f"p{i:02d}.png"), img)
            rows.append({"i": i, "target_mm": target.tolist(), "reached": reached.tolist(), "tip_mm": X.tolist(),
                         "uv": None if uv is None else uv.tolist()})
            print(f"point {i}: tip {X.round(1)} mm -> px {None if uv is None else uv.round(1)}")
            if uv is not None:
                obs3d.append(X)
                obs2d.append(uv)
    except KeyboardInterrupt:
        print("interrupted: the arm holds where it is")
    finally:
        pipe.stop()
        (out_dir / "points.json").write_text(json.dumps(rows, indent=1))
        print("back to observe:", "ok" if goto(observe, args.speed) is not None else "FAILED")

    if len(obs3d) < 6:
        sys.exit(f"only {len(obs3d)} usable points; not fitting")
    P3, P2 = np.array(obs3d, float), np.array(obs2d, float)
    rv0, _ = cv2.Rodrigues(R0)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(P3, P2, K, dist, rvec=rv0, tvec=t0.reshape(3, 1).copy(),
                                             useExtrinsicGuess=True, reprojectionError=8.0,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        sys.exit("solvePnP failed")
    inl = inl.ravel() if inl is not None else np.arange(len(P3))
    rvec, tvec = cv2.solvePnPRefineLM(P3[inl], P2[inl], K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(P3, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - P2, axis=1)
    R, _ = cv2.Rodrigues(rvec)
    data = {
        "description": "Top RealSense (serial %s) pose in the robot base frame (mm): x_cam = R @ x_base + t. "
                       "Fitted by scripts/calib_top_camera.py with the jaw tip as marker." % TOP_SERIAL,
        "date": stamp, "image_size": list(SIZE), "K": K.round(4).tolist(), "dist": dist.round(6).tolist(),
        "R_cam_from_base": R.round(8).tolist(), "t_cam_from_base_mm": tvec.ravel().round(3).tolist(),
        "camera_centre_in_base_mm": (-R.T @ tvec.ravel()).round(1).tolist(),
        "fit": {"points": len(P3), "inliers": int(len(inl)), "median_px": float(np.median(err[inl])),
                "max_px": float(err[inl].max()), "all_px": err.round(2).tolist()},
        "captures": str(out_dir.relative_to(REPO)),
    }
    OUT.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"saved {OUT.relative_to(REPO)}: {len(inl)}/{len(P3)} inliers, median {np.median(err[inl]):.1f} px, "
          f"max {err[inl].max():.1f} px; camera at {data['camera_centre_in_base_mm']} mm")


if __name__ == "__main__":
    main()

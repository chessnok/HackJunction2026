#!/usr/bin/env python3
"""One hover cycle: observe pose -> top-camera photo -> find the brick -> hover above it, gripper down.

Uses configs/top_camera.yaml (scripts/calib_top_camera.py) to turn the brick's bottom edge in the top
RealSense frame into table XY, IK from hackjunction2026.kinematics.ik, a clearance check of every straight
joint move, and scripts/so101_goto.py for the moves. Saves the frames under captures/pipeline/<time>/.

Run with Studio's backend Python from the repo root, Studio's environment closed:
  PYTHONPATH=src third_party/physical-ai-studio/application/backend/.venv/bin/python scripts/pipeline_hover.py [--go]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from hackjunction2026 import table_map as tm
from hackjunction2026.kinematics import TABLE_Z_MM, Calibration
from hackjunction2026.kinematics.ik import solve_down
from hackjunction2026.kinematics.path import check_straight_move

REPO = Path(__file__).resolve().parents[1]
PY = REPO / "third_party/physical-ai-studio/application/backend/.venv/bin/python"
TOP_SERIAL = "243622062352"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HALF_DEPTH_MM = 12.0       # bottom front edge -> brick centre, along the camera's horizontal view direction


def run_goto(args: list[str]) -> str:
    r = subprocess.run([str(PY), str(REPO / "scripts/so101_goto.py"), *args], capture_output=True, text=True,
                       timeout=180)
    return r.stdout + r.stderr


def parse(line_name: str, text: str) -> np.ndarray | None:
    m = re.search(rf"^{line_name}\s+(.*)$", text, re.M)
    if not m:
        return None
    vals = dict(re.findall(r"(\w+)=\s*(-?[\d.]+)", m.group(1)))
    return np.array([float(vals[j]) for j in JOINTS])


def move(target: np.ndarray, cal, speed: float, clearance: float, what: str) -> np.ndarray:
    cur = parse("follower", run_goto(["where"]))
    if cur is None:
        sys.exit("cannot read the follower pose")
    chk = check_straight_move(cur, target, cal, min_clearance_mm=clearance)
    if not chk.ok:
        sys.exit(f"{what}: refused by the clearance check: {chk.describe()}")
    out = run_goto(["goto", "--speed", str(speed), "--go", "--yes", "--", ",".join(f"{x:.2f}" for x in target)])
    final = parse("final", out)
    if "REACHED" not in out or final is None:
        sys.exit(f"{what}: move failed\n{out[-1500:]}")
    print(f"{what}: reached {np.round(final, 1)}")
    return final


def top_frame() -> np.ndarray:
    import pyrealsense2 as rs
    p, c = rs.pipeline(), rs.config()
    c.enable_device(TOP_SERIAL)
    c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    p.start(c)
    try:
        for _ in range(30):
            f = p.wait_for_frames()
        return np.asanyarray(f.get_color_frame().get_data()).copy()
    finally:
        p.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="move (else stop after planning)")
    ap.add_argument("--height", type=float, default=60.0, help="jaw tips above the table, mm")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--clearance", type=float, default=30.0)
    args = ap.parse_args()

    cal = Calibration.from_file(REPO / "configs/calibration/follower.json")
    poses = yaml.safe_load((REPO / "configs/poses.yaml").read_text())
    observe = np.array([poses["observe"]["joints"][j] for j in JOINTS])
    cam = yaml.safe_load((REPO / "configs/top_camera.yaml").read_text())
    K, dist = np.array(cam["K"]), np.array(cam["dist"])
    R, t = np.array(cam["R_cam_from_base"]), np.array(cam["t_cam_from_base_mm"])
    C = -R.T @ t
    out_dir = REPO / "captures/pipeline" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.go:
        move(observe, cal, args.speed, args.clearance, "observe")
    img = top_frame()
    cv2.imwrite(str(out_dir / "top_observe.png"), img)
    det = tm.detect(img)
    bricks = sorted(det.bricks, key=lambda b: -b.area_px)
    if not bricks:
        sys.exit("no brick found in the top camera frame")
    b = bricks[0]
    box = np.array(b.box)
    bottom = box[np.argsort(box[:, 1])[-2:]].mean(axis=0)          # front bottom edge, nearest the camera
    u = cv2.undistortPoints(bottom.reshape(1, 1, 2).astype(float), K, dist, P=K).ravel()
    d = R.T @ np.linalg.solve(K, [u[0], u[1], 1.0])
    front = C + (TABLE_Z_MM - C[2]) / d[2] * d
    away = front[:2] - C[:2]
    away /= np.linalg.norm(away)
    xy = front[:2] + HALF_DEPTH_MM * away
    reach = float(np.hypot(xy[0] - 38.8, xy[1]))
    print(f"brick {b.colour} at px {np.round(b.centroid, 1)} -> table ({xy[0]:.1f}, {xy[1]:.1f}) mm, "
          f"{reach:.0f} mm from the pan axis ({len(bricks)} found)")
    cv2.imwrite(str(out_dir / "top_detect.png"), tm.annotate(img, det))

    sol = solve_down(float(xy[0]), float(xy[1]), TABLE_Z_MM + args.height, 90.0, cal, float(observe[5]),
                     start_norm=observe)
    if not sol.ok:
        sys.exit(f"IK refused: {sol.reason}")
    print(f"hover target {np.round(sol.norm, 2)}")
    if not args.go:
        print("dry run: add --go")
        return
    move(sol.norm, cal, args.speed, args.clearance, "hover")
    time.sleep(0.5)
    cv2.imwrite(str(out_dir / "top_hover.png"), top_frame())
    try:
        from hackjunction2026 import pick_above as pa
        fr = pa.capture_frame((1024, 768), check_pose=False)
        cv2.imwrite(str(out_dir / "wrist_hover.png"), fr.image if hasattr(fr, "image") else fr[0])
    except Exception as e:  # the wrist frame is only a record
        print(f"wrist frame not saved: {e}")
    print(f"frames: {out_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()

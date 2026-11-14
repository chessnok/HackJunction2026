#!/usr/bin/env python3
"""Measure the wrist-camera centring model (configs/servo.yaml) with one brick, no human help.

The arm takes the brick from the photo spot, puts it down at 2 central points; at each it releases, rises to the
hover height (gripper straight down, jaws open, exactly above the brick it just placed) and grabs a wrist frame:
the brick pixel there is ``target_px``. It then shifts the hover by +-15 mm in x and y and records how the brick
pixel moves (image Jacobian J), picks the brick up again and finally returns it to the photo spot.

  PYTHONPATH=src $PY scripts/servo_calib.py            # dry run
  PYTHONPATH=src $PY scripts/servo_calib.py --go --yes  # real arm (Studio's environment closed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hackjunction2026 import calib_wrist as cw
from hackjunction2026 import servo
from hackjunction2026 import sort_cycle as sc
from hackjunction2026.kinematics import Calibration

REPO = Path(__file__).resolve().parents[1]
POINTS = [(225.0, 0.0), (240.0, -60.0)]
YAW = 90.0           # jaws close along table y while measuring (theta0 in servo.ServoModel)
STEP = 15.0          # mm test moves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    cfg = sc.Config.load()
    poses, missing = sc.load_poses(cfg)
    if missing:
        raise SystemExit("missing poses: " + ", ".join(missing))
    cal = Calibration.from_file(sc.FOLLOWER_CAL)
    if a.go:
        if not a.yes:
            raise SystemExit("pass --yes")
        arm, cams = sc.PluginArm(), sc.RealCameras(cfg)
    else:
        arm, cams = sc.FakeArm(poses["observe"]), None
    cyc = sc.Cycle(arm, cams, None, sc.ScriptedGrasp(), cfg, cal, poses, dry_run=not a.go, step=False,
                   out_root=REPO / "captures/servo_calib")
    calib, mv = cw.Calib(cyc), cyc.mv
    hover_rel = cfg.hover_mm
    targets, moves, shifts, samples = [], [], [], []
    yaw0 = 0.0
    print(f"log: {cyc.out}")
    try:
        mv.move(poses["observe"], "observe")
        calib.take_from_photo()
        for i, xy in enumerate(POINTS):
            held = mv.arm.read()[sc.GRIPPER]
            mv.aim(xy, cfg.hover_mm, YAW, held, f"p{i}: hover (holding)")
            mv.aim(xy, cfg.grasp_mm, YAW, held, f"p{i}: down", near_table=True, tries=1)
            mv.gripper(cfg.gripper_open, f"p{i}: release")
            mv.aim(xy, hover_rel, YAW, cfg.gripper_open, f"p{i}: up to the hover above the brick", near_table=True)
            if mv.dry_run:
                continue
            f0 = cams.wrist()
            cyc.save(f"p{i}_centred", f0)
            p0 = servo.brick_px(f0)
            if p0 is None:
                raise sc.StopCycle(f"p{i}: brick not seen from the hover")
            targets.append(p0)
            yaw0 = servo.directed_yaw_deg(mv.arm.read(), cal)
            print(f"  p{i}: centred brick at pixel ({p0[0]:.0f}, {p0[1]:.0f})")
            for d in ((STEP, 0.0), (-STEP, 0.0), (0.0, STEP), (0.0, -STEP)):
                mv.aim((xy[0] + d[0], xy[1] + d[1]), hover_rel, YAW, cfg.gripper_open, f"p{i}: test move {d}")
                f = cams.wrist()
                p = servo.brick_px(f, near=p0)
                cyc.save(f"p{i}_move_{int(d[0])}_{int(d[1])}", f)
                if p is not None:
                    moves.append(d)
                    shifts.append(p - p0)
                    samples.append({"point": i, "move_mm": list(d), "shift_px": [float(v) for v in p - p0]})
                    print(f"    move {d} mm -> pixel shift ({p[0]-p0[0]:.0f}, {p[1]-p0[1]:.0f})")
            # pick it up again
            mv.aim(xy, hover_rel, YAW, cfg.gripper_open, f"p{i}: back above")
            mv.aim(xy, cfg.grasp_mm, YAW, cfg.gripper_open, f"p{i}: down to grasp", near_table=True, tries=1)
            calib.grasp_here(f"p{i}: grasp")
            mv.move(mv.lifted(mv.arm.read(), cfg.hover_mm - cfg.grasp_mm, keep_roll=False), f"p{i}: lift",
                    near_table=True)
            calib.check_still_holding(f"p{i}: lift")
        calib.return_to_photo()
        mv.move(poses["observe"], "observe")
    except sc.StopCycle as e:
        print(f"\nSTOPPED: {e}\nThe arm holds its pose (torque on).")
    finally:
        if a.go:
            arm.gripper_torque_limit(1000)
            arm.close()
    if a.go and len(targets) >= 1 and len(moves) >= 2:
        J = servo.fit_jacobian(np.array(moves), np.array(shifts))
        model = servo.ServoModel(np.mean(targets, axis=0), J, hover_rel, yaw0,
                                 [{"targets_px": [[float(v) for v in t] for t in targets], "yaw0_deg": YAW}] + samples)
        model.save(note=f"measured {cyc.out.name}, directed jaw yaw {yaw0:.1f} deg")
        print(f"saved configs/servo.yaml: target_px {np.round(model.target_px, 1)}, J (px/mm) {np.round(J, 2).tolist()}")
    elif a.go:
        sys.exit("not enough measurements for a model")


if __name__ == "__main__":
    main()

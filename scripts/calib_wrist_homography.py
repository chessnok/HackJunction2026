#!/usr/bin/env python3
"""Calibrate the wrist-camera homography of the `observe` pose by moving ONE brick around (no human help).

Put ONE brick at the photo spot (where the jaws are in the taught `photo` pose) and nothing else on the sheet.
The robot grasps it there, then for each of ~6 reachable points: puts it down, raises, looks from `observe`,
pairs the brick pixel with the FK jaw-centre XY of the measured place pose, and picks it up again. Then it
fits configs/table_homography.yaml (RANSAC + leave-one-out), returns the brick to the photo spot and goes to
`observe`. Servo 6 is logged closed empty and closed on the brick (holding thresholds). ADR 0007 / 0013.
UNTESTED ON THE ROBOT.

  PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
  PYTHONPATH=src $PY scripts/calib_wrist_homography.py                 # dry run: plan + point list, no motion
  PYTHONPATH=src $PY scripts/calib_wrist_homography.py --go --step     # real, asks before every move
Options: --n 6, --points "200,40;240,-60;..." (table mm), --colour red, --out configs/table_homography.yaml.
Real runs need Studio's environment closed. Stop: Ctrl+C or q at a --step prompt (the arm holds).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hackjunction2026 import calib_wrist as cw  # noqa: E402
from hackjunction2026 import sort_cycle as sc  # noqa: E402
from hackjunction2026.kinematics import Calibration  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from sort_cycle import check_devices_free, snapshot_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="move the real arm (default: dry run)")
    ap.add_argument("--step", action="store_true", help="ask before every move")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--n", type=int, default=6, help="number of points")
    ap.add_argument("--points", help="explicit table points 'x,y;x,y;...' (mm, robot base frame)")
    ap.add_argument("--colour", help="only count bricks of this colour in the observe frames")
    ap.add_argument("--out", default=str(REPO / "configs/table_homography.yaml"))
    ap.add_argument("--snapshot", help="dry run: wrist frame from this Studio snapshot")
    a = ap.parse_args()

    cfg = sc.Config.load()
    poses, missing = sc.load_poses(cfg)
    if missing:
        raise SystemExit("REFUSED, missing poses: " + ", ".join(missing))
    cal = Calibration.from_file(sc.FOLLOWER_CAL)
    given = [tuple(map(float, p.split(","))) for p in a.points.split(";")] if a.points else None
    points, rejected = cw.plan_points(poses["observe"], cal, cfg, n=a.n, given=given)
    print(f"observe looks towards {cw.observe_direction_deg(poses['observe'], cal):.0f} deg from +x")
    print(f"{len(points)} points (jaw centre, mm; r = from the pan axis):")
    for i, p in enumerate(points):
        r = ((p.xy[0] - cw.PAN_AXIS_XY[0]) ** 2 + p.xy[1] ** 2) ** 0.5
        print(f"  {i}: ({p.xy[0]:6.1f}, {p.xy[1]:6.1f})  r {r:5.0f}")
    for w in rejected:
        print(f"  rejected {w}")
    if len(points) < 4:
        raise SystemExit("fewer than 4 usable points: give --points")

    if a.go:
        check_devices_free()
        if not a.yes:
            if not sys.stdin.isatty():
                raise SystemExit("not a terminal: pass --yes")
            if input("ONE brick at the photo spot, sheet otherwise empty, hands clear? Start? [y/N] ").strip().lower() not in ("y", "yes"):
                raise SystemExit("cancelled")
        arm, cams = sc.PluginArm(), sc.RealCameras(cfg)
    else:
        arm, cams = sc.FakeArm(poses["observe"]), sc.FileCameras(snapshot_dir(a.snapshot))
        print("DRY RUN: nothing moves; every 'observe' frame is the same snapshot, so no fit is made")

    cycle = sc.Cycle(arm, cams, None, sc.ScriptedGrasp(), cfg, cal, poses, dry_run=not a.go, step=a.step,
                     out_root=REPO / "captures/calib_wrist")
    calib = cw.Calib(cycle, colour=a.colour)
    print(f"log: {cycle.out}/cycle.jsonl")
    original_limit = arm.gripper_torque_limit() if a.go else None
    stats = None
    try:
        cycle.enter(sc.S.OBSERVE)
        cycle.mv.move(poses["observe"], "observe")
        calib.closed_empty_sample()
        cycle.enter(sc.S.GRASP)
        calib.take_from_photo()
        for i, p in enumerate(points):
            print(f"--- point {i}")
            calib.place_and_look(i, p)
        calib.return_to_photo()
        if a.go:
            stats = calib.fit(Path(a.out))
            print(f"saved {a.out}: {stats}")
    except (sc.StopCycle, KeyboardInterrupt) as e:
        cycle.log("stopped", reason=str(e) or "Ctrl+C")
        print(f"\nSTOPPED: {e or 'Ctrl+C'}\nThe arm holds its pose (torque on). {len(calib.pairs)} pairs so far.")
        if a.go and len(calib.pairs) >= 4:
            try:
                stats = calib.fit(Path(a.out))
                print(f"fitted from the pairs so far -> {a.out}: {stats}")
            except sc.StopCycle as e2:
                print(f"no fit: {e2}")
    finally:
        cw.write_summary(calib, stats, cycle.out / "summary.json")
        print(f"gripper samples: {calib.gripper_summary()}")
        if a.go and original_limit is not None:
            arm.gripper_torque_limit(original_limit)
        arm.close()
        cycle.close()


if __name__ == "__main__":
    main()

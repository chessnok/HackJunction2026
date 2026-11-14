#!/usr/bin/env python3
"""Sorting cycle: observe -> find the brick -> hover -> grasp (checked on servo 6, 3 tries) -> photo spot ->
release + raise -> photo + defect detection (PatchCore + mark classifier, NPU) -> re-grasp -> good / defect
bowl -> release + raise -> observe. ADR 0013. UNTESTED ON THE ROBOT.

Run with Studio's backend Python (SO-101 plugin, OpenVINO, pyrealsense2) from the repo root:
  PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
  PYTHONPATH=src $PY scripts/sort_cycle.py                         # dry run: plans and prints, moves nothing
  PYTHONPATH=src $PY scripts/sort_cycle.py --snapshot 20261114-061847-779   # dry run on a snapshot's frames
  PYTHONPATH=src $PY scripts/sort_cycle.py --go --step              # real, asks before every move
  PYTHONPATH=src $PY scripts/sort_cycle.py --go --cycles 5          # real, 5 bricks

Real runs need Studio's environment closed (the cycle owns the follower port and the cameras).
Stop: Ctrl+C (the move in progress stops and the arm holds), or q at a --step prompt.
Settings: configs/sort_cycle.yaml (heights, speeds, holding thresholds, detector models).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hackjunction2026 import sort_cycle as sc  # noqa: E402
from hackjunction2026.kinematics import Calibration  # noqa: E402

SNAPSHOTS = Path.home() / ".local/share/physicalai/state_snapshots/0fd76ff2-a6b1-4472-b687-486befe329be"

SAVE_HELP = """Save the missing poses first (configs/poses.yaml). With Studio's environment closed, put the LEADER
arm where the follower should go (or use Studio teleop and a snapshot, as for 'observe'), then:
  PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
  $PY scripts/so101_goto.py save bowl_good   --from leader   # jaws open above the good bowl
  $PY scripts/so101_goto.py save bowl_defect --from leader   # jaws open above the defect bowl
  $PY scripts/so101_goto.py save photo       --from leader   # brick resting on the photo spot, in the jaws"""


def snapshot_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser()
        return p if p.is_dir() else SNAPSHOTS / arg
    dirs = sorted(d for d in SNAPSHOTS.glob("2026*") if all((d / f"{c}.color.png").exists() for c in ("top", "floor", "arm")))
    if not dirs:
        raise SystemExit(f"no snapshot with top/floor/arm frames in {SNAPSHOTS}: pass --snapshot")
    return dirs[-1]


def check_devices_free() -> None:
    from hackjunction2026.pick_above import resolve_wrist_camera, wrist_camera_name
    from hackjunction2026.ports import by_serial, port_users

    goto = sc._so101_goto()
    port = by_serial(goto.HW_SERIALS["follower"])
    if port is None:
        raise SystemExit("the follower board is not connected")
    if users := port_users(port):
        raise SystemExit(f"{port} is in use by {', '.join(users)}: close the environment in Studio first")
    cam = str(resolve_wrist_camera(wrist_camera_name()).resolve())
    if users := port_users(cam):
        raise SystemExit(f"the wrist camera is open in {', '.join(users)}: close the environment in Studio first")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="move the real arm (default: dry run)")
    ap.add_argument("--step", action="store_true", help="ask before every move")
    ap.add_argument("--yes", action="store_true", help="skip the one start confirmation of --go")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--grasp", choices=("scripted", "act"), default="scripted", help="first grasp (re-grasp is always scripted)")
    ap.add_argument("--locate", choices=("auto", "wrist", "top"), help="override configs/sort_cycle.yaml")
    ap.add_argument("--snapshot", help="dry run: frames from this Studio snapshot (id or folder); default the newest")
    ap.add_argument("--target", help="skip brick finding and use this table point, 'X,Y' mm (tests, calibration checks)")
    ap.add_argument("--no-detect", action="store_true", help="skip the defect models (everything goes to the good bowl)")
    ap.add_argument("--placeholder-bowls", action="store_true",
                    help="dry run only: use 'observe' for missing bowl poses so the rest can be planned")
    a = ap.parse_args()

    cfg = sc.Config.load()
    if a.locate:
        cfg.locate = a.locate
    poses, missing = sc.load_poses(cfg)
    if missing:
        if a.go or not a.placeholder_bowls or any(not x.startswith("bowl") for x in missing):
            raise SystemExit("REFUSED, missing poses: " + ", ".join(missing) + "\n" + SAVE_HELP)
        for role in ("bowl_good", "bowl_defect"):
            poses.setdefault(role, poses["observe"].copy())
        print(f"(dry run) placeholder bowl poses = observe for: {', '.join(missing)}")
    cal = Calibration.from_file(sc.FOLLOWER_CAL)

    if a.go:
        check_devices_free()
        if not a.yes:
            if not sys.stdin.isatty():
                raise SystemExit("not a terminal: pass --yes to confirm")
            if input("Hands clear of the table? Start the sorting cycle on the real arm? [y/N] ").strip().lower() not in ("y", "yes"):
                raise SystemExit("cancelled")
        arm, cams = sc.PluginArm(), sc.RealCameras(cfg)
    else:
        arm = sc.FakeArm(poses["observe"])
        cams = sc.FileCameras(snapshot_dir(a.snapshot))
        print(f"DRY RUN: nothing moves; frames from {cams.frames and snapshot_dir(a.snapshot)}")

    detector = None if a.no_detect else sc.PatchCoreAndClassifier(cfg)
    grasp = sc.ACTGrasp(cfg) if a.grasp == "act" else sc.ScriptedGrasp()
    cycle = sc.Cycle(arm, cams, detector, grasp, cfg, cal, poses, dry_run=not a.go, step=a.step)
    if a.target:
        x, y = (float(v) for v in a.target.split(","))
        fixed = sc.Target((x, y), 90.0, "given", "--target")
        cycle.locate = lambda q: fixed
    print(f"log: {cycle.out}/cycle.jsonl")
    original_limit = None if not a.go else arm.gripper_torque_limit()
    try:
        for i in range(a.cycles):
            print(f"=== brick {i + 1} / {a.cycles}")
            res = cycle.run_one()
            cycle.log("result", **res)
            print(f"=== {res}")
            if res.get("result") == "no brick on the sheet":
                break
    except (sc.StopCycle, KeyboardInterrupt) as e:
        reason = str(e) or "Ctrl+C"
        cycle.log("stopped", reason=reason)
        print(f"\nSTOPPED: {reason}\nThe arm holds its pose (torque on).")
        if a.go and sys.stdin.isatty():
            try:
                if input("Turn the GRIPPER torque off (drops whatever it holds)? [y/N] ").strip().lower() in ("y", "yes"):
                    arm.gripper_torque_off()
                    print("gripper torque off")
            except (EOFError, KeyboardInterrupt):
                pass
        sys.exitcode = 1
    finally:
        if a.go and original_limit is not None:
            arm.gripper_torque_limit(original_limit)
        arm.close()
        cycle.close()


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    main()

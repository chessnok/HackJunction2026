#!/usr/bin/env python3
"""Move the SO-101 follower to a target pose: straight line in joint space, all joints together.

delta = target - current; each joint moves linearly so that all arrive at the same time.
Run with Studio's backend Python (it has the SO-101 driver), from the repo root:
  PY=third_party/physical-ai-studio/application/backend/.venv/bin/python

  $PY scripts/so101_goto.py where                         # both arms' current poses (read-only)
  $PY scripts/so101_goto.py save pick_1 --from leader      # save a pose (put the leader there by hand)
  $PY scripts/so101_goto.py list                          # saved poses
  $PY scripts/so101_goto.py goto pick_1                   # dry run: show the move
  $PY scripts/so101_goto.py goto pick_1 --go              # move (asks for confirmation)
  $PY scripts/so101_goto.py goto "0,-20,30,40,0,10" --go  # raw target in Studio normalized units

Units: Studio normalized units (joints -100..100 over the calibrated range, gripper 0..100).
The follower's serial port must be free: close the environment in Studio (or stop Studio) first.
Safety (AGENTS.md): motion only on an explicit human request; keep hands clear; Ctrl+C stops and
holds the pose. At the end the follower keeps holding (torque on); relax with scripts/so101_relax.py.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hackjunction2026 import so101_motion as m  # noqa: E402
from hackjunction2026.feetech import Bus  # noqa: E402
from hackjunction2026.ports import by_serial, port_users  # noqa: E402

STUDIO = "http://localhost:7860"
CFG = REPO / "configs"
POSES = CFG / "poses.yaml"
ROLE_TYPE = {"follower": "SO101_Follower", "leader": "SO101_Leader"}


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_poses() -> dict:
    return yaml.safe_load(POSES.read_text()) or {} if POSES.exists() else {}


def arm(role: str) -> tuple[str, dict]:
    """Serial number and calibration of an arm, from Studio's API; cached in configs/calibration/."""
    cache = CFG / "calibration" / f"{role}.json"
    try:
        with urllib.request.urlopen(f"{STUDIO}/api/projects", timeout=3) as r:
            projects = json.load(r)
        for p in projects:
            with urllib.request.urlopen(f"{STUDIO}/api/projects/{p['id']}/robots", timeout=3) as r:
                for robot in json.load(r):
                    if robot.get("type") == ROLE_TYPE[role]:
                        info = {"serial": robot["payload"]["serial_number"],
                                "calibration": robot["payload"]["calibration"], "fetched": now()}
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps(info, indent=2))
                        return info["serial"], info["calibration"]
    except OSError:
        pass
    if cache.exists():
        info = json.loads(cache.read_text())
        return info["serial"], info["calibration"]
    serial = HW_SERIALS[role]
    print(f"(no Studio and no cache: reading the {role} calibration from the servo EEPROM)")
    return serial, calibration_from_board(port_for(serial))


# Fallback when Studio is down: serials from configs/hardware.yaml roles
HW_SERIALS = {"follower": "5B3D041025", "leader": "5B91030613"}


def calibration_from_board(port: str) -> dict:
    """Calibration as Studio stores it, rebuilt from what the calibration wrote into each servo:
    homing offset (addr 31, sign bit 11) and position limits (addr 9 / 11). drive_mode is 0 on SO-101."""
    from hackjunction2026.feetech import sign_mag

    calib = {}
    with Bus(port) as bus:
        for sid, j in enumerate(m.JOINTS, start=1):
            vals = {k: bus.read(sid, k) for k in ("homing_offset", "min_pos", "max_pos")}
            if None in vals.values():
                sys.exit(f"cannot read the calibration of {j} (ID {sid}) from {port}")
            calib[j] = {"id": sid, "drive_mode": 0, "homing_offset": sign_mag(vals["homing_offset"], 11),
                        "range_min": vals["min_pos"], "range_max": vals["max_pos"]}
    return calib


def port_for(serial: str) -> str:
    port = by_serial(serial)
    if port is None:
        sys.exit(f"board {serial} is not connected")
    if users := port_users(port):
        sys.exit(f"{port} is in use by {', '.join(users)}: close the environment in Studio (or stop Studio)")
    return port


def read_pose(role: str) -> np.ndarray:
    """Current pose in normalized units, read from the servos without touching torque."""
    from physicalai_so101_plugin._conversion import ticks_to_normalized
    from physicalai_so101_plugin.calibration import SO101Calibration

    serial, calib = arm(role)
    with Bus(port_for(serial)) as bus:
        ticks = []
        for j in m.JOINTS:
            pos = bus.read(calib[j]["id"], "pos")
            if pos is None:
                sys.exit(f"{role}: servo {j} (ID {calib[j]['id']}) does not answer")
            ticks.append(pos)
    return ticks_to_normalized(np.array(ticks), SO101Calibration.from_dict(calib)).astype(float)


def fmt(q: np.ndarray) -> str:
    return "  ".join(f"{j}={v:7.2f}" for j, v in zip(m.JOINTS, q, strict=True))


def cmd_where(_args) -> None:
    for role in ("follower", "leader"):
        print(f"{role:8} {fmt(read_pose(role))}")


def cmd_save(args) -> None:
    q = read_pose(args.source)
    poses = load_poses()
    poses[args.name] = {"joints": m.as_dict(q), "from": args.source, "saved": now()}
    POSES.parent.mkdir(parents=True, exist_ok=True)
    POSES.write_text(yaml.safe_dump(poses, sort_keys=False))
    print(f"saved {args.name}: {fmt(q)}")


def cmd_list(_args) -> None:
    for name, p in load_poses().items():
        print(f"{name:12} {fmt(m.vec(p['joints']))}")


def resolve_target(text: str) -> np.ndarray:
    poses = load_poses()
    if text in poses:
        return m.vec(poses[text]["joints"])
    try:
        return m.vec([float(x) for x in text.split(",")])
    except ValueError:
        sys.exit(f"unknown pose '{text}' (saved: {', '.join(poses) or 'none'}) and not 6 numbers")


DIRECTIONS = CFG / "so101_directions.yaml"


def cmd_measure_up(args) -> None:
    """Two-step measurement on the leader: 'start' records its pose; after the operator raises the
    arm a little by hand, 'end' stores which sign of the joint means 'up'."""
    state = CFG / ".measure_up_start.json"
    q = read_pose("leader")
    if args.step == "start":
        state.write_text(json.dumps(m.as_dict(q)))
        print(f"start recorded ({args.joint}={q[m.JOINTS.index(args.joint)]:.2f}); now raise the leader arm "
              "a little by hand, then run: measure-up end")
        return
    if not state.exists():
        sys.exit("run 'measure-up start' first")
    start = m.vec(json.loads(state.read_text()))
    i = m.JOINTS.index(args.joint)
    change = q[i] - start[i]
    if abs(change) < 3:
        sys.exit(f"{args.joint} changed only {change:+.2f}: raise the arm more clearly and repeat")
    dirs = yaml.safe_load(DIRECTIONS.read_text()) if DIRECTIONS.exists() else {}
    dirs[args.joint] = {"up": 1 if change > 0 else -1, "measured": now(),
                        "evidence": f"leader {args.joint} {start[i]:.2f} -> {q[i]:.2f} when raised by hand"}
    DIRECTIONS.write_text(yaml.safe_dump(dirs, sort_keys=False))
    state.unlink()
    print(f"{args.joint}: up = {'+' if change > 0 else '-'} (changed {change:+.2f}); saved {DIRECTIONS.relative_to(REPO)}")


def cmd_raise(args) -> None:
    """Relative move of one joint 'up' by a fraction of its full range (or of its current value)."""
    dirs = yaml.safe_load(DIRECTIONS.read_text()) if DIRECTIONS.exists() else {}
    if args.joint not in dirs:
        sys.exit(f"'up' for {args.joint} is not measured yet: measure-up start / measure-up end")
    i = m.JOINTS.index(args.joint)
    q0 = read_pose("follower")
    amount = args.frac * ((m.HI[i] - m.LO[i]) if args.of == "range" else abs(q0[i]))
    target = q0.copy()
    target[i] += dirs[args.joint]["up"] * amount
    print(f"{args.joint} up by {args.frac:.3f} of its {args.of} = {amount:.1f} units: "
          f"{q0[i]:.2f} -> {target[i]:.2f}")
    args.target = ",".join(f"{v:.3f}" for v in target)
    cmd_goto(args)


def cmd_goto(args) -> None:
    target = resolve_target(args.target)
    if problems := m.out_of_range(target):
        sys.exit("REFUSED, target out of range: " + "; ".join(problems))
    q0 = read_pose("follower")
    print(f"from  {fmt(q0)}\nto    {fmt(target)}\ndelta {fmt(target - q0)}")
    print(f"time  {m.duration(q0, target, args.speed):.1f}s at {args.speed} units/s")
    if not args.go:
        print("dry run: add --go to move")
        return
    if not args.yes:
        if not sys.stdin.isatty():
            sys.exit("not a terminal: pass --yes to confirm the move")
        if input("Hands clear? Move the follower now? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("cancelled")

    from physicalai_so101_plugin.so101 import SO101

    serial, calib = arm("follower")
    robot = SO101(port=port_for(serial), calibration=calib, role="follower", unit="normalized")
    robot.connect()  # follower: torque on, holds the current pose
    try:
        read = lambda: robot.get_observation().joint_positions.astype(float)  # noqa: E731
        start = read()  # re-read: the arm may have moved since the dry-run numbers
        r = m.move_linear(lambda q: robot.send_action(q.astype(np.float64)), read, start, target,
                          args.speed, track_error=args.track_error)
        print(("REACHED" if r.ok else f"STOPPED ({r.reason})") + f", worst lag {r.worst_error:.1f}")
        print(f"final {fmt(r.final)}")
    finally:
        robot.disconnect()  # follower keeps torque on: the arm holds its pose
    if not r.ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("where", help="current poses of both arms (read-only)").set_defaults(fn=cmd_where)
    s = sub.add_parser("save", help="save the current pose of an arm under a name")
    s.add_argument("name")
    s.add_argument("--from", dest="source", choices=["leader", "follower"], default="leader")
    s.set_defaults(fn=cmd_save)
    sub.add_parser("list", help="saved poses").set_defaults(fn=cmd_list)
    def motion_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--speed", type=float, default=20.0, help="speed of the joint that moves most, units/s")
        p.add_argument("--track-error", type=float, default=12.0, help="stop if the arm lags more than this")
        p.add_argument("--go", action="store_true", help="actually move")
        p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    s = sub.add_parser("goto", help="show (dry run) or execute (--go) a straight joint-space move")
    s.add_argument("target", help="saved pose name or 6 comma-separated normalized values")
    motion_args(s)
    s.set_defaults(fn=cmd_goto)
    s = sub.add_parser("raise", help="move one joint 'up' by a fraction of its range (default: motor 2, 1/3)")
    s.add_argument("--joint", default="shoulder_lift", choices=m.JOINTS[:5])
    s.add_argument("--frac", type=float, default=1 / 3)
    s.add_argument("--of", choices=["range", "value"], default="range",
                   help="fraction of the joint's full range (200 units) or of its current value")
    motion_args(s)
    s.set_defaults(fn=cmd_raise)
    s = sub.add_parser("measure-up", help="measure which sign is 'up' for a joint, using the leader")
    s.add_argument("step", choices=["start", "end"])
    s.add_argument("--joint", default="shoulder_lift", choices=m.JOINTS[:5])
    s.set_defaults(fn=cmd_measure_up)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Move the SO-101 follower to a target pose along a safe joint-space route, no kinematic model.

Run with Studio's backend Python (it has the SO-101 driver), from the repo root:
  PY=third_party/physical-ai-studio/application/backend/.venv/bin/python

  $PY scripts/so101_goto.py where                         # both arms' current poses (read-only)
  $PY scripts/so101_goto.py save transit --from leader     # save a pose (put the leader there by hand)
  $PY scripts/so101_goto.py list                          # saved poses, limits, floor points
  $PY scripts/so101_goto.py measure-limits --seconds 30   # sweep the leader through the SAFE workspace
  $PY scripts/so101_goto.py measure-floor                 # touch the table with the follower tip, Enter each time
  $PY scripts/so101_goto.py goto pick_1                   # DRY RUN: print the plan and checks
  $PY scripts/so101_goto.py goto pick_1 --go              # move (asks for confirmation)
  $PY scripts/so101_goto.py goto "0,-20,30,40,0,10" --go  # raw target, Studio normalized units

Units: Studio normalized units (joints -100..100 over the calibrated range, gripper 0..100).
The arm's serial port must be free: close the environment in Studio (or stop Studio) first.
Safety (AGENTS.md): motion only on an explicit human request; keep hands clear; Ctrl+C stops and
holds the pose. At the end the follower keeps holding (torque on); relax with scripts/so101_relax.py.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
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
POSES, LIMITS, FLOOR = CFG / "poses.yaml", CFG / "so101_limits.yaml", CFG / "so101_floor.yaml"
ROLE_TYPE = {"follower": "SO101_Follower", "leader": "SO101_Leader"}


# --------------------------------------------------------------------------- config files
def load_yaml(path: Path, default):
    return yaml.safe_load(path.read_text()) if path.exists() else default


def save_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"saved {path.relative_to(REPO)}")


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- arms + calibration
def arm(role: str) -> tuple[str, dict]:
    """Serial number and calibration for an arm: from Studio's API, cached in configs/calibration/."""
    cache = CFG / "calibration" / f"{role}.json"
    try:
        with urllib.request.urlopen(f"{STUDIO}/api/projects", timeout=3) as r:
            projects = json.load(r)
        for p in projects:
            with urllib.request.urlopen(f"{STUDIO}/api/projects/{p['id']}/robots", timeout=3) as r:
                for robot in json.load(r):
                    if robot.get("type") == ROLE_TYPE[role]:
                        info = {"serial": robot["payload"]["serial_number"],
                                "calibration": robot["payload"]["calibration"],
                                "studio_robot_id": robot["id"], "fetched": now()}
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps(info, indent=2))
                        return info["serial"], info["calibration"]
    except OSError:
        pass
    if cache.exists():
        info = json.loads(cache.read_text())
        print(f"(Studio not reachable: using cached calibration from {info['fetched']})")
        return info["serial"], info["calibration"]
    sys.exit(f"no calibration for the {role}: start Studio once (it holds the calibration)")


def port_for(serial: str) -> str:
    port = by_serial(serial)
    if port is None:
        sys.exit(f"board {serial} is not connected")
    if users := port_users(port):
        sys.exit(f"{port} is in use by {', '.join(users)}: close the environment in Studio (or stop Studio)")
    return port


def read_pose(role: str) -> tuple[np.ndarray, bool]:
    """Current pose in normalized units, read from the servos without changing torque."""
    from physicalai_so101_plugin._conversion import ticks_to_normalized
    from physicalai_so101_plugin.calibration import SO101Calibration

    serial, calib = arm(role)
    cal = SO101Calibration.from_dict(calib)
    with Bus(port_for(serial)) as bus:
        ticks, torque = [], []
        for j in m.JOINTS:
            sid = calib[j]["id"]
            pos = bus.read(sid, "pos")
            if pos is None:
                sys.exit(f"{role}: servo {j} (ID {sid}) does not answer")
            ticks.append(pos)
            torque.append(bus.read(sid, "torque_enable") == 1)
    return ticks_to_normalized(np.array(ticks), cal).astype(float), any(torque)


def fmt(q: np.ndarray) -> str:
    return "  ".join(f"{j}={v:7.2f}" for j, v in zip(m.JOINTS, q, strict=True))


# --------------------------------------------------------------------------- commands
def cmd_where(_args) -> None:
    for role in ("follower", "leader"):
        q, torque = read_pose(role)
        print(f"{role:8} torque {'ON ' if torque else 'off'}  {fmt(q)}")


def cmd_save(args) -> None:
    q, _ = read_pose(args.source)
    poses = load_yaml(POSES, {})
    poses[args.name] = {"unit": "normalized", "joints": m.as_dict(q), "from": args.source, "saved": now()}
    save_yaml(POSES, poses)
    print(f"{args.name}: {fmt(q)}")


def cmd_list(_args) -> None:
    limits = m.Limits.build(soft=load_yaml(LIMITS, None))
    print(f"limits ({limits.source}):")
    for i, j in enumerate(m.JOINTS):
        print(f"  {j:14} [{limits.lo[i]:7.2f}, {limits.hi[i]:7.2f}]")
    floor = load_yaml(FLOOR, {"poses": []})["poses"]
    print(f"floor points: {len(floor)}")
    for name, p in load_yaml(POSES, {}).items():
        print(f"pose {name:12} {fmt(m.vec(p['joints']))}  (from {p.get('from')}, {p.get('saved')})")


def cmd_measure_limits(args) -> None:
    print(f"Move the {args.source} slowly through the SAFE workspace (above the table, every joint to its"
          f" safe extremes) for {args.seconds:.0f} s...")
    lo = hi = None
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        q, _ = read_pose(args.source)
        lo = q if lo is None else np.minimum(lo, q)
        hi = q if hi is None else np.maximum(hi, q)
        time.sleep(0.05)
    data = {"unit": "normalized", "measured": now(), "from": args.source, "seconds": args.seconds,
            "min": m.as_dict(lo), "max": m.as_dict(hi)}
    print("min:", fmt(lo))
    print("max:", fmt(hi))
    save_yaml(LIMITS, data)


def cmd_measure_floor(args) -> None:
    data = load_yaml(FLOOR, {"unit": "normalized", "poses": []})
    while True:
        if not args.once:
            answer = input(f"[{len(data['poses'])} points] touch the table with the follower gripper tip, "
                           "Enter = capture, q = done: ").strip().lower()
            if answer == "q":
                break
        q, torque = read_pose("follower")
        if torque:
            print("  follower torque is ON (it may be pushing): relax it first (scripts/so101_relax.py)")
        data["poses"].append({"joints": m.as_dict(q), "measured": now()})
        print(f"  captured: {fmt(q)}")
        if args.once:
            break
    save_yaml(FLOOR, data)


def resolve_target(text: str) -> np.ndarray:
    poses = load_yaml(POSES, {})
    if text in poses:
        return m.vec(poses[text]["joints"])
    try:
        return m.vec([float(x) for x in text.split(",")])
    except ValueError:
        sys.exit(f"unknown pose '{text}' (saved: {', '.join(poses) or 'none'}) and not 6 numbers")


def cmd_goto(args) -> None:
    target = resolve_target(args.target)
    poses = load_yaml(POSES, {})
    transit = m.vec(poses[args.via]["joints"]) if args.via in poses else None
    floor = [m.vec(p["joints"]) for p in load_yaml(FLOOR, {"poses": []})["poses"]]
    limits = m.Limits.build(margin=args.margin, soft=load_yaml(LIMITS, None))

    q0, _ = read_pose("follower")
    p = m.plan(q0, target, limits, transit=transit, floor_poses=floor, vmax=args.speed,
               approach_vmax=args.approach_speed, direct=args.direct)
    if args.direct and float(np.abs(target - q0).max()) > args.direct_max:
        p.problems.append(f"--direct only for moves up to {args.direct_max} units (this one is "
                          f"{float(np.abs(target - q0).max()):.1f}); use the transit route")
    print(f"from    {fmt(q0)}\nto      {fmt(target)}")
    for s in p.segments:
        print(f"  {s.name:9} {s.duration:4.1f}s  -> {fmt(s.q1)}")
    print(f"total {p.duration:.1f}s, limits: {limits.source}, floor points: {len(floor)}")
    for n in p.notes:
        print("note:", n)
    if p.problems:
        print("REFUSED:\n  " + "\n  ".join(p.problems))
        sys.exit(2)
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
        q_start = robot.get_observation().joint_positions.astype(float)
        p = m.plan(q_start, target, limits, transit=transit, floor_poses=floor, vmax=args.speed,
                   approach_vmax=args.approach_speed, direct=args.direct)
        if p.problems:
            sys.exit("REFUSED after reconnect:\n  " + "\n  ".join(p.problems))
        track = args.track_error / (2 if p.near_floor else 1)
        result = m.execute(lambda q: robot.send_action(q.astype(np.float64)),
                           lambda: robot.get_observation().joint_positions.astype(float),
                           p.segments, track_error=track)
        print(("REACHED" if result.ok else "STOPPED") + f": {result.reason}; worst tracking error "
              f"{result.worst_error:.1f}\nfinal   {fmt(result.final)}")
    finally:
        robot.disconnect()  # follower keeps torque on: the arm holds its pose
    if not result.ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("where", help="current poses of both arms (read-only)").set_defaults(fn=cmd_where)
    s = sub.add_parser("save", help="save the current pose under a name")
    s.add_argument("name")
    s.add_argument("--from", dest="source", choices=["leader", "follower"], default="leader")
    s.set_defaults(fn=cmd_save)
    sub.add_parser("list", help="saved poses, limits, floor points").set_defaults(fn=cmd_list)
    s = sub.add_parser("measure-limits", help="record soft joint limits while sweeping an arm")
    s.add_argument("--from", dest="source", choices=["leader", "follower"], default="leader")
    s.add_argument("--seconds", type=float, default=30.0)
    s.set_defaults(fn=cmd_measure_limits)
    s = sub.add_parser("measure-floor", help="record poses where the follower tip touches the table")
    s.add_argument("--once", action="store_true", help="capture one point without prompting")
    s.set_defaults(fn=cmd_measure_floor)
    s = sub.add_parser("goto", help="plan (dry run) or execute (--go) a move")
    s.add_argument("target", help="saved pose name or 6 comma-separated normalized values")
    s.add_argument("--via", default="transit", help="transit pose name (default: transit)")
    s.add_argument("--direct", action="store_true", help="straight joint-space move (small moves only)")
    s.add_argument("--direct-max", type=float, default=15.0)
    s.add_argument("--speed", type=float, default=25.0, help="peak joint speed, units/s")
    s.add_argument("--approach-speed", type=float, default=8.0, help="final approach speed, units/s")
    s.add_argument("--margin", type=float, default=3.0, help="margin inside the calibrated range")
    s.add_argument("--track-error", type=float, default=12.0, help="stop if the arm lags more than this")
    s.add_argument("--go", action="store_true", help="actually move")
    s.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    s.set_defaults(fn=cmd_goto)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

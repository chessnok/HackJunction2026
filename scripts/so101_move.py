#!/usr/bin/env python3
"""Gently move ONE servo of an SO-101 arm, then switch its torque off.

Safety (AGENTS.md): run only on an explicit human request, announce the move, keep hands clear.
Guards: reduced torque limit, slow speed and acceleration, target clamped to the calibrated
limits with a margin, RAM registers only, torque always disabled in `finally`.

usage:
  uv run scripts/so101_move.py PORT ID --to max|min|<pos> [--back]
  uv run scripts/so101_move.py PORT ID --delta 200 --back
"""

import argparse
import sys
import time

from hackjunction2026.feetech import SO101_JOINTS, Bus, decode_status


def wait_pos(bus: Bus, sid: int, target: int, timeout: float) -> list[int | None]:
    t0, trace = time.monotonic(), []
    while time.monotonic() - t0 < timeout:
        pos = bus.read(sid, "pos")
        trace.append(pos)
        if pos is not None and abs(pos - target) <= 8:
            break
        time.sleep(0.05)
    return trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("id", type=int)
    goal = ap.add_mutually_exclusive_group(required=True)
    goal.add_argument("--to", help="absolute position, or 'min' / 'max' (calibrated limit -/+ margin)")
    goal.add_argument("--delta", type=int, help="relative move in steps (4096 steps = 360 deg)")
    ap.add_argument("--back", action="store_true", help="return to the start position afterwards")
    ap.add_argument("--speed", type=int, default=400, help="steps/s, 1..1000 (0 would mean max speed)")
    ap.add_argument("--torque-limit", type=int, default=500, help="0..1000, default 500 = 50%%")
    ap.add_argument("--margin", type=int, default=20, help="steps kept away from the limits")
    args = ap.parse_args()
    if not 1 <= args.speed <= 1000 or not 0 < args.torque_limit <= 1000:
        sys.exit("speed must be 1..1000 and torque limit 1..1000")

    sid = args.id
    with Bus(args.port) as bus:
        p0, lo, hi = bus.read(sid, "pos"), bus.read(sid, "min_pos"), bus.read(sid, "max_pos")
        status, tl0 = bus.read(sid, "status"), bus.read(sid, "torque_limit")
        name = SO101_JOINTS.get(sid, f"id{sid}")
        print(f"{name} (id {sid}): pos={p0} limits=[{lo},{hi}] status={decode_status(status) or 'OK'}")
        if None in (p0, lo, hi) or status:
            sys.exit("servo not answering or reporting an error; aborting")
        if args.to is not None:
            target = {"max": hi - args.margin, "min": lo + args.margin}.get(args.to)
            target = int(args.to) if target is None else target
        else:
            target = p0 + args.delta
        target = max(lo + args.margin, min(hi - args.margin, target))
        timeout = abs(target - p0) / args.speed * 2 + 1.5

        try:
            bus.write(sid, "torque_limit", args.torque_limit)
            bus.write(sid, "acc", 20)
            bus.write(sid, "goal_vel", args.speed)
            bus.write(sid, "goal_pos", p0)  # hold the current position first: no jump
            bus.write(sid, "torque_enable", 1)
            time.sleep(0.2)
            print(f"-> {p0} to {target}")
            bus.write(sid, "goal_pos", target)
            trace = wait_pos(bus, sid, target, timeout)
            print(f"   trace {trace[::5]} final {trace[-1]}")
            if args.back:
                time.sleep(0.5)
                print(f"-> back to {p0}")
                bus.write(sid, "goal_pos", p0)
                trace = wait_pos(bus, sid, p0, timeout)
                print(f"   trace {trace[::5]} final {trace[-1]}")
        finally:
            bus.write(sid, "torque_enable", 0)
            bus.write(sid, "torque_limit", tl0 if tl0 is not None else 1000)
            time.sleep(0.3)
            print(
                f"torque off: {bus.read(sid, 'torque_enable') == 0}, pos {bus.read(sid, 'pos')},"
                f" status {decode_status(bus.read(sid, 'status')) or 'OK'}"
            )


if __name__ == "__main__":
    main()

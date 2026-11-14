#!/usr/bin/env python3
"""Compliant hold for SO-101 joints: the arm carries its own weight but can be moved by hand,
and stays where you leave it.

How: the torque limit of the selected servos is lowered (default 10-20 % of max; holding the arm
against gravity needs only a few %), and a 50 Hz loop moves each servo's goal to wherever the
joint has been pushed beyond a small deadband. RAM registers only (torque limit, goal position).

On exit (Ctrl+C or SIGTERM) full torque is restored and the arm holds its last pose.
The port must be free (no Studio environment, no other script).

usage: uv run scripts/so101_compliant.py [--port P] [--ids 1,2,3,4,5] [--deadband 25]
"""

import argparse
import signal
import sys
import time

from hackjunction2026.feetech import SO101_JOINTS, Bus
from hackjunction2026.ports import by_serial, port_users

FOLLOWER_SERIAL = "5B3D041025"
# Torque limits (0..1000): joints that lift the arm against gravity get more.
DEFAULT_LIMIT = {1: 120, 2: 200, 3: 200, 4: 150, 5: 100, 6: 150}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="default: the follower board")
    ap.add_argument("--ids", default="1,2,3,4,5")
    ap.add_argument("--deadband", type=int, default=25, help="ticks (4096 per turn) before the goal follows")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the default torque limits")
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()

    port = args.port or by_serial(FOLLOWER_SERIAL)
    if not port:
        sys.exit("follower board not connected")
    if users := port_users(port):
        sys.exit(f"{port} is in use by {', '.join(users)}")
    ids = [int(x) for x in args.ids.split(",")]

    def ignore_signals() -> None:
        # A second SIGTERM (e.g. uv forwarding the one sent to its process group) must not
        # interrupt the restore in `finally`.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def stop(*_):
        ignore_signals()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)

    with Bus(port) as bus:
        # Restore to the servo's power-on torque limit (EEPROM max_torque), not to whatever the RAM
        # limit was at start: a previous interrupted run may have left it lowered.
        original = {i: bus.read(i, "max_torque") or 1000 for i in ids}
        goal = {}
        for i in ids:
            pos = bus.read(i, "pos")
            if pos is None:
                sys.exit(f"servo {i} does not answer")
            goal[i] = pos
            bus.write(i, "goal_pos", pos)  # hold exactly here, no jump
            bus.write(i, "torque_limit", min(1000, int(DEFAULT_LIMIT[i] * args.scale)))
            bus.write(i, "torque_enable", 1)
        names = ", ".join(f"{SO101_JOINTS[i]}={min(1000, int(DEFAULT_LIMIT[i] * args.scale)) / 10:.0f}%" for i in ids)
        print(f"compliant hold ON ({names}); move the arm by hand. Stop: Ctrl+C / SIGTERM", flush=True)
        moved = 0
        try:
            while True:
                t0 = time.monotonic()
                for i in ids:
                    pos = bus.read(i, "pos")
                    if pos is not None and abs(pos - goal[i]) > args.deadband:
                        goal[i] = pos
                        bus.write(i, "goal_pos", pos)
                        moved += 1
                time.sleep(max(0.0, 1 / args.hz - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            ignore_signals()
            for i in ids:  # restore full torque and hold the current pose
                pos = bus.read(i, "pos")
                if pos is not None:
                    bus.write(i, "goal_pos", pos)
                bus.write(i, "torque_limit", original[i])
            print(f"compliant hold OFF: full torque restored, holding the last pose ({moved} goal updates)",
                  flush=True)


if __name__ == "__main__":
    main()

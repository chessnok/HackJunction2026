#!/usr/bin/env python3
"""Relax SO-101 arms: switch torque off gently so they do not drop.

Ramps the RAM torque limit down to 0 over --seconds (the arm sinks slowly under gravity instead of
falling), then disables torque and restores the torque limit register for the next session.
RAM registers only. Stop Studio first (scripts/studio_stop.sh): it holds the ports.

usage: uv run scripts/so101_relax.py [PORT ...] [--seconds 4]
       (default: every /dev/serial/by-id/usb-1a86_* board)
"""

import argparse
import glob
import sys
import time

from hackjunction2026.feetech import SO101_JOINTS, Bus
from hackjunction2026.ports import port_users

IDS = list(SO101_JOINTS)


def relax(port: str, seconds: float) -> None:
    name = port.split("Serial_")[-1][:10]
    if users := port_users(port):
        print(f"{name}: port open by {', '.join(users)}; stop Studio first (scripts/studio_stop.sh)")
        return
    with Bus(port) as bus:
        present = [i for i in IDS if bus.ping(i)]
        on = [i for i in present if bus.read(i, "torque_enable") == 1]
        print(f"{name}: servos {present}, torque on: {on or 'none'}")
        if on:
            limits = {i: bus.read(i, "torque_limit") or 1000 for i in on}
            steps = max(1, int(seconds / 0.1))
            for k in range(steps, -1, -1):
                for i in on:
                    bus.write(i, "torque_limit", int(limits[i] * k / steps))
                time.sleep(seconds / steps)
            for i in on:
                bus.write(i, "torque_enable", 0)
            time.sleep(0.1)
            for i in on:  # restore the limit; torque is off, so this does not move anything
                bus.write(i, "torque_limit", limits[i])
        still = [i for i in present if bus.read(i, "torque_enable") == 1]
        print(f"{name}: {'RELAXED (torque off on all servos)' if not still else f'torque still on: {still}'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ports", nargs="*")
    ap.add_argument("--seconds", type=float, default=4.0, help="ramp duration")
    args = ap.parse_args()
    ports = args.ports or sorted(glob.glob("/dev/serial/by-id/usb-1a86_*"))
    if not ports:
        sys.exit("no servo boards found")
    for port in ports:
        relax(port, args.seconds)


if __name__ == "__main__":
    main()

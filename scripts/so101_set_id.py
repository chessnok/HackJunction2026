#!/usr/bin/env python3
"""Change the ID of ONE Feetech STS3215 servo (EEPROM write).

Only on explicit human request (AGENTS.md rule 5); log every change in docs/log and docs/hardware.md.

New servos all ship as ID 1, so connect exactly one servo with the old ID to the bus. The script
refuses to write when:
  - the old ID does not answer reliably (two servos sharing an ID collide on the bus),
  - the new ID is already taken,
  - the servo reports an error status.

usage: uv run scripts/so101_set_id.py PORT --from 1 --to 2
"""

import argparse
import sys
import time

from hackjunction2026.feetech import Bus, decode_status

PING_TRIES = 20


def reliability(bus: Bus, sid: int) -> int:
    return sum(bus.ping(sid) for _ in range(PING_TRIES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("--from", dest="old", type=int, required=True)
    ap.add_argument("--to", dest="new", type=int, required=True)
    args = ap.parse_args()
    if not (0 <= args.old <= 253 and 1 <= args.new <= 253) or args.old == args.new:
        sys.exit("IDs must be 0..253, new ID 1..253, and different")

    with Bus(args.port) as bus:
        present = bus.scan(range(0, 254))
        print(f"IDs on the bus before: {present}")
        if args.old not in present:
            sys.exit(f"ID {args.old} does not answer; nothing to change")
        if args.new in present:
            sys.exit(f"ID {args.new} is already taken on this bus; refusing")
        ok = reliability(bus, args.old)
        print(f"ID {args.old} answered {ok}/{PING_TRIES} pings")
        if ok < PING_TRIES:
            sys.exit(
                f"ID {args.old} answers unreliably: probably several servos share ID {args.old}. "
                "Disconnect all but the one servo you want to renumber, then retry."
            )
        status = bus.read(args.old, "status")
        if status:
            sys.exit(f"servo reports {decode_status(status)}; fix that first")
        model, volt = bus.read(args.old, "model"), bus.read(args.old, "volt_x10")
        print(f"servo: model {model}, {volt / 10 if volt else '?'} V")

        print(f"writing ID {args.old} -> {args.new} (EEPROM, human-approved)")
        bus.write_eeprom(args.old, "id", args.new, human_approved=True)
        time.sleep(0.1)

        new_ok, old_ok = reliability(bus, args.new), reliability(bus, args.old)
        lock = bus.read(args.new, "lock")
        print(f"after: ID {args.new} answered {new_ok}/{PING_TRIES}, ID {args.old} answered {old_ok}/{PING_TRIES},"
              f" EEPROM lock={lock}")
        print(f"IDs on the bus after: {bus.scan(range(0, 254))}")
        if new_ok != PING_TRIES or old_ok:
            sys.exit("verification FAILED")
        print("OK")


if __name__ == "__main__":
    main()

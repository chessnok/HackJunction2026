#!/usr/bin/env python3
"""Assign SO-101 servo IDs one motor at a time, with checks (like `lerobot-setup-motors`).

EEPROM writes: run only on an explicit human request (AGENTS.md rule 5). Every write is appended
to docs/log/<today>.md automatically.

Interactive flow, by default for IDs 6..1 (gripper first, as LeRobot does):
  1. "Connect ONLY the <joint> motor, press Enter"  (s = skip, q = quit)
  2. Scan the bus. Exactly one servo must answer, 20/20 pings (fewer means several servos share
     an ID), with no error status and, with --expect, a supply voltage that fits the arm's role.
  3. Already the right ID: nothing is written. Otherwise confirm, write the ID, relock the EEPROM,
     verify (new ID 20/20, old ID 0/20, lock=1) and log it.
  4. At the end: connect the whole chain and verify IDs, stability, voltage and firmware versions.

usage:
  uv run scripts/so101_setup_ids.py PORT --expect follower           # wizard, IDs 6..1
  uv run scripts/so101_setup_ids.py PORT --ids 6 --no-wait --yes     # one motor, no prompts
  uv run scripts/so101_setup_ids.py PORT --verify 1,2,3,4,5,6        # check the assembled chain
"""

import argparse
import datetime
import glob
import os
import sys
import time
from pathlib import Path

from hackjunction2026.feetech import DEFAULT_BAUD, MODELS, SO101_JOINTS, Bus, decode_status

REPO = Path(__file__).resolve().parents[1]
PINGS = 20
ALL_IDS = range(0, 254)
OTHER_BAUDS = [500_000, 115_200, 57_600, 38_400]
# Studio's SO-101 setup wizard: follower on >= 7.0 V (12 V PSU), leader below 7.0 V (5 V PSU)
VOLT_SPLIT_X10 = 70


# --------------------------------------------------------------------------- helpers
def port_users(port: str) -> list[str]:
    """Other processes that have the serial device open (e.g. the Studio backend)."""
    real, me, users = os.path.realpath(port), str(os.getpid()), set()
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.path.realpath(fd) == real:
                pid = fd.split("/")[2]
                if pid != me:
                    users.add(f"{pid} ({Path(f'/proc/{pid}/comm').read_text().strip()})")
        except OSError:
            continue
    return sorted(users)


def log_line(text: str) -> None:
    today = datetime.date.today()
    path = REPO / "docs" / "log" / f"{today:%Y-%m-%d}.md"
    header = "" if path.exists() else f"# {today:%Y-%m-%d}\n\nTimes are the laptop clock.\n\n"
    with path.open("a") as f:
        f.write(f"{header}- {datetime.datetime.now():%H:%M:%S} {text}\n")
    print(f"  logged to {path.relative_to(REPO)}")


def reliability(bus: Bus, sid: int) -> int:
    return sum(bus.ping(sid) for _ in range(PINGS))


def describe(bus: Bus, sid: int) -> dict:
    r = {k: bus.read(sid, k) for k in ("model", "fw_major", "fw_minor", "volt_x10", "temp_c", "status", "pos")}
    r["fw"] = f"{r['fw_major']}.{r['fw_minor']}"
    r["volt"] = r["volt_x10"] / 10 if r["volt_x10"] is not None else None
    return r


def voltage_problem(volt_x10: int | None, expect: str | None) -> str | None:
    if expect is None or volt_x10 is None:
        return None
    if expect == "follower" and volt_x10 < VOLT_SPLIT_X10:
        return f"{volt_x10 / 10} V looks like a leader (5 V) supply, but --expect follower"
    if expect == "leader" and volt_x10 >= VOLT_SPLIT_X10:
        return f"{volt_x10 / 10} V looks like a follower (12 V) supply, but --expect leader"
    return None


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{question} yes (--yes)")
        return True
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def scan_other_bauds(port: str) -> tuple[int | None, list[int]]:
    for baud in OTHER_BAUDS:
        try:
            with Bus(port, baud) as bus:
                ids = bus.scan(ALL_IDS)
        except Exception:  # noqa: BLE001  (some bauds are not supported by the board)
            continue
        if ids:
            return baud, ids
    return None, []


# --------------------------------------------------------------------------- one motor
def assign(port: str, target: int, args: argparse.Namespace) -> str:
    """Return 'ok', 'retry' or 'fail'."""
    joint = SO101_JOINTS.get(target, f"id{target}")
    with Bus(port) as bus:
        found = bus.scan(ALL_IDS)
        if len(found) > 1:
            print(f"  several servos answer: IDs {found}. Connect ONLY the {joint} motor.")
            return "retry"
        if len(found) == 1:
            sid = found[0]
            ok = reliability(bus, sid)
            if ok < PINGS:
                print(f"  ID {sid} answered {ok}/{PINGS} pings: several servos probably share ID {sid}."
                      f" Connect ONLY the {joint} motor.")
                return "retry"
            info = describe(bus, sid)
            print(f"  found ID {sid}: {MODELS.get(info['model'], info['model'])} fw {info['fw']},"
                  f" {info['volt']} V, {info['temp_c']} C, pos {info['pos']}")
            if info["status"]:
                print(f"  servo reports {decode_status(info['status'])}; fix that first (power-cycle the arm).")
                return "retry"
            if info["model"] not in MODELS and not confirm(f"  unknown model {info['model']}, continue?", False):
                return "retry"
            problem = voltage_problem(info["volt_x10"], args.expect)
            if problem:
                print(f"  WARNING: {problem}")
                if not confirm("  continue anyway?", False):
                    return "retry"
            if sid == target:
                print(f"  already ID {target}: nothing to write.")
                return "ok"
            if not confirm(f"  write ID {sid} -> {target} ({joint}) to EEPROM?", args.yes):
                return "retry"
            bus.write_eeprom(sid, "id", target, human_approved=True)
            time.sleep(0.1)
            new_ok, old_ok, lock = reliability(bus, target), reliability(bus, sid), bus.read(target, "lock")
            name = Path(port).name
            if new_ok == PINGS and old_ok == 0 and lock == 1:
                print(f"  OK: ID {target} {new_ok}/{PINGS}, old ID {sid} {old_ok}/{PINGS}, lock=1")
                log_line(f"**EEPROM write** (`scripts/so101_setup_ids.py`, human-requested): `{name}` {joint}"
                         f" servo ID {sid} → {target}; verified {new_ok}/{PINGS}, old ID silent, lock=1.")
                return "ok"
            print(f"  verification FAILED: ID {target} {new_ok}/{PINGS}, ID {sid} {old_ok}/{PINGS}, lock={lock}")
            log_line(f"**EEPROM write FAILED verification** (`scripts/so101_setup_ids.py`): `{name}` ID {sid} → "
                     f"{target}: new {new_ok}/{PINGS}, old {old_ok}/{PINGS}, lock={lock}. See troubleshooting.")
            return "fail"

    baud, ids = scan_other_bauds(port)
    if ids:
        print(f"  servo(s) {ids} answer at {baud} baud, not {DEFAULT_BAUD}. Changing the baud rate is not"
              " automated here; set it to 1M first (e.g. Studio's SO-101 setup wizard or lerobot-setup-motors).")
    else:
        print("  no servo answers: check servo power, the 3-pin bus cable and the board jumper (B = USB).")
    return "retry"


# --------------------------------------------------------------------------- whole chain
def verify(port: str, expected: list[int], expect_role: str | None) -> bool:
    with Bus(port) as bus:
        found = bus.scan(ALL_IDS)
        rows = []
        for sid in found:
            info = describe(bus, sid)
            info["pings"] = reliability(bus, sid)
            rows.append((sid, info))
    print(f"\nChain on {port}: IDs {found}")
    print(f"  {'ID':>3} {'joint':13} {'model':8} {'fw':5} {'volt':>5} {'temp':>4} {'pings':>6} status")
    for sid, i in rows:
        print(f"  {sid:>3} {SO101_JOINTS.get(sid, '?'):13} {MODELS.get(i['model'], i['model'])!s:8} {i['fw']:5}"
              f" {i['volt']!s:>5} {i['temp_c']!s:>4} {i['pings']:>3}/{PINGS} {decode_status(i['status']) or 'OK'}")
    problems = []
    if missing := sorted(set(expected) - set(found)):
        problems.append(f"missing IDs {missing}")
    if extra := sorted(set(found) - set(expected)):
        problems.append(f"unexpected IDs {extra}")
    if shaky := [sid for sid, i in rows if i["pings"] < PINGS]:
        problems.append(f"unreliable IDs {shaky} (duplicate IDs or loose cable)")
    if errors := [sid for sid, i in rows if i["status"]]:
        problems.append(f"error status on IDs {errors}")
    if len({i["fw"] for _, i in rows}) > 1:
        problems.append("mixed firmware versions (known to break calibration/teleop, lerobot issue #1010)")
    for sid, i in rows:
        if p := voltage_problem(i["volt_x10"], expect_role):
            problems.append(f"ID {sid}: {p}")
            break
    print("RESULT: " + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)))
    return not problems


# --------------------------------------------------------------------------- main
def parse_ids(text: str) -> list[int]:
    ids = [int(x) for x in text.split(",") if x.strip()]
    if not ids or any(not 1 <= i <= 253 for i in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("IDs must be unique integers in 1..253, e.g. 6,5,4,3,2,1")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", help="servo board, preferably /dev/serial/by-id/...")
    ap.add_argument("--ids", type=parse_ids, default=[6, 5, 4, 3, 2, 1], help="target IDs in order (default 6..1)")
    ap.add_argument("--expect", choices=["leader", "follower"], help="check the supply voltage fits this role")
    ap.add_argument("--yes", action="store_true", help="do not ask before each EEPROM write")
    ap.add_argument("--no-wait", action="store_true", help="no 'connect the motor' prompts: one attempt per ID")
    ap.add_argument("--verify", type=parse_ids, metavar="IDS", help="only verify the assembled chain")
    ap.add_argument("--force", action="store_true", help="run even if another process has the port open")
    args = ap.parse_args()

    if not Path(args.port).exists():
        sys.exit(f"{args.port} does not exist (board unplugged?)")
    if (users := port_users(args.port)) and not args.force:
        sys.exit(f"{args.port} is open by {', '.join(users)}. Stop it (e.g. disconnect the robot in Studio)"
                 " or pass --force.")

    if args.verify:
        sys.exit(0 if verify(args.port, args.verify, args.expect) else 1)

    done: list[int] = []
    for target in args.ids:
        joint = SO101_JOINTS.get(target, f"id{target}")
        while True:
            if args.no_wait:
                print(f"\n[{joint}, ID {target}]")
            else:
                answer = input(f"\n[{joint}, ID {target}] connect ONLY this motor to the bus, then Enter"
                               " (s = skip, q = quit): ").strip().lower()
                if answer == "q":
                    sys.exit("quit")
                if answer == "s":
                    break
            result = assign(args.port, target, args)
            if result == "ok":
                done.append(target)
                break
            if result == "fail" or args.no_wait:
                sys.exit(1)

    if not args.no_wait and len(args.ids) > 1:
        input("\nNow connect the whole chain (all motors), then Enter to verify: ")
        sys.exit(0 if verify(args.port, args.ids, args.expect) else 1)
    print(f"\ndone: IDs {done}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only probe of SO-101 servo buses: which servos answer, their state and limits.

Never writes anything; safe to run at any time.

usage: uv run scripts/so101_probe.py [PORT ...]
       (default: every /dev/serial/by-id/usb-1a86_* port)
"""

import glob
import sys

from hackjunction2026.feetech import MODELS, SO101_JOINTS, Bus, decode_status, sign_mag

BAUDS = [1_000_000, 500_000, 250_000, 115_200, 57_600, 38_400]
FIELDS = ("model", "fw_major", "fw_minor", "pos", "min_pos", "max_pos", "homing_offset",
          "torque_enable", "volt_x10", "temp_c", "status")


def probe(port: str) -> None:
    print(f"\n=== {port}")
    for baud in BAUDS:
        try:
            bus = Bus(port, baud)
        except Exception as exc:  # noqa: BLE001
            # e.g. the CH343 board under cdc_acm rejects 250000 baud with EINVAL (T-005)
            print(f"  baud {baud}: cannot open ({exc}), skipped")
            continue
        with bus:
            ids = bus.scan()
            if not ids:
                continue
            print(f"  baud {baud}: servo IDs {ids}")
            for sid in ids:
                r = {k: bus.read(sid, k) for k in FIELDS}
                volt = f"{r['volt_x10'] / 10:.1f}V" if r["volt_x10"] is not None else "?"
                print(
                    f"   id={sid} {SO101_JOINTS.get(sid, '?'):13s} {MODELS.get(r['model'], r['model'])}"
                    f" fw={r['fw_major']}.{r['fw_minor']} pos={r['pos']!s:<5}"
                    f" limits=[{r['min_pos']},{r['max_pos']}] homing={sign_mag(r['homing_offset'], 11)}"
                    f" torque={'ON' if r['torque_enable'] else 'off'} {volt} {r['temp_c']}C"
                    f" status={decode_status(r['status']) or 'OK'}"
                )
            return
    print("  no servo answered at any baud rate (servo power? bus cable? board jumper?)")


def main() -> None:
    ports = sys.argv[1:] or sorted(glob.glob("/dev/serial/by-id/usb-1a86_*"))
    if not ports:
        sys.exit("no servo boards found under /dev/serial/by-id/")
    for port in ports:
        probe(port)


if __name__ == "__main__":
    main()

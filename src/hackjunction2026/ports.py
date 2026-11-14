"""Serial-port helpers: find boards by USB serial number, see who has a port open."""

from __future__ import annotations

import glob
import os
from pathlib import Path


def by_serial(serial_number: str) -> str | None:
    """Stable /dev/serial/by-id path of the board with this USB serial number."""
    matches = sorted(glob.glob(f"/dev/serial/by-id/*{serial_number}*"))
    return matches[0] if matches else None


def port_users(port: str) -> list[str]:
    """Other processes that have the serial device open (e.g. a Studio robot worker)."""
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

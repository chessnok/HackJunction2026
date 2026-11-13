"""Minimal Feetech STS bus-servo protocol (STS3215 on SO-101 arms), pure pyserial.

Only what our tools need: PING, READ and RAM-only WRITE. EEPROM writes are refused on purpose
(AGENTS.md, robot safety rule 5).
"""

from __future__ import annotations

import time

import serial

PING, READ, WRITE = 0x01, 0x02, 0x03
DEFAULT_BAUD = 1_000_000
EEPROM_END = 40  # addresses below this are EEPROM

# STS3215 control table: name -> (address, size in bytes)
REG: dict[str, tuple[int, int]] = {
    "fw_major": (0, 1),
    "fw_minor": (1, 1),
    "model": (3, 2),
    "id": (5, 1),
    "baud_idx": (6, 1),
    "min_pos": (9, 2),
    "max_pos": (11, 2),
    "max_temp": (13, 1),
    "max_volt": (14, 1),
    "min_volt": (15, 1),
    "max_torque": (16, 2),
    "led_alarm": (19, 1),
    "unload_cond": (20, 1),
    "p_gain": (21, 1),
    "d_gain": (22, 1),
    "i_gain": (23, 1),
    "prot_current": (28, 2),
    "homing_offset": (31, 2),
    "mode": (33, 1),
    "overload_torque": (36, 1),
    "torque_enable": (40, 1),
    "acc": (41, 1),
    "goal_pos": (42, 2),
    "goal_time": (44, 2),
    "goal_vel": (46, 2),
    "torque_limit": (48, 2),
    "lock": (55, 1),
    "pos": (56, 2),
    "speed": (58, 2),
    "load": (60, 2),
    "volt_x10": (62, 1),
    "temp_c": (63, 1),
    "status": (65, 1),
    "moving": (66, 1),
    "current": (69, 2),
}
MODELS = {777: "STS3215"}
STATUS_BITS = ["VOLTAGE", "SENSOR", "OVERHEAT", "OVERCURRENT", "ANGLE", "OVERLOAD"]
SO101_JOINTS = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


def packet(sid: int, instr: int, params: tuple[int, ...] = ()) -> bytes:
    body = [sid, len(params) + 2, instr, *params]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def sign_mag(value: int | None, sign_bit: int) -> int | None:
    """Feetech encodes negative values as sign bit + magnitude."""
    if value is None:
        return None
    return -(value & ~(1 << sign_bit)) if value & (1 << sign_bit) else value


def decode_status(status: int | None) -> list[str]:
    return [name for bit, name in enumerate(STATUS_BITS) if status and status >> bit & 1]


class Bus:
    """One serial servo bus. Use as a context manager."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD, reply_timeout_s: float = 0.03):
        self.ser = serial.Serial(port, baud, timeout=0.005)
        self.reply_timeout_s = reply_timeout_s

    def close(self) -> None:
        self.ser.close()

    def __enter__(self) -> Bus:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def transact(self, sid: int, instr: int, params: tuple[int, ...] = (), expect: int = 0):
        """Send one instruction; return (error_byte, payload) or None if there is no valid reply."""
        s = self.ser
        s.reset_input_buffer()
        s.write(packet(sid, instr, params))
        s.flush()
        want = 6 + expect
        buf = b""
        deadline = time.monotonic() + self.reply_timeout_s
        while time.monotonic() < deadline:
            chunk = s.read(s.in_waiting or 1)
            if chunk:
                buf += chunk
                i = buf.find(b"\xff\xff")
                if i >= 0 and len(buf) - i >= want:
                    break
        i = buf.find(b"\xff\xff")
        if i < 0 or len(buf) - i < 6:
            return None
        rid, length, err = buf[i + 2], buf[i + 3], buf[i + 4]
        if rid != sid or len(buf) - i < 4 + length:
            return None
        payload = buf[i + 5 : i + 3 + length]
        if (~(rid + length + err + sum(payload))) & 0xFF != buf[i + 3 + length]:
            return None
        return err, payload

    def ping(self, sid: int) -> bool:
        return self.transact(sid, PING) is not None

    def scan(self, ids: range = range(0, 21)) -> list[int]:
        return [i for i in ids if self.ping(i)]

    def read(self, sid: int, name: str) -> int | None:
        addr, size = REG[name]
        reply = self.transact(sid, READ, (addr, size), size)
        if not reply or len(reply[1]) != size:
            return None
        data = reply[1]
        return data[0] | (data[1] << 8 if size == 2 else 0)

    def write(self, sid: int, name: str, value: int) -> bool:
        """Write a RAM register. EEPROM registers raise PermissionError."""
        addr, size = REG[name]
        if addr < EEPROM_END:
            raise PermissionError(f"{name} (addr {addr}) is EEPROM; refusing, see AGENTS.md safety rule 5")
        data = [value & 0xFF] + ([(value >> 8) & 0xFF] if size == 2 else [])
        return self.transact(sid, WRITE, (addr, *data)) is not None

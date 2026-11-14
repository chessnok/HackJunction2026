"""Studio normalized joint values -> servo ticks -> URDF joint angles (numpy only).

Studio's SO-101 driver (physicalai_so101_plugin ``_conversion.ticks_to_normalized``) reports
body joints as ``(tick - range_min) / (range_max - range_min) * 200 - 100`` (clipped to
[-100, 100]) and the gripper as ``(tick - range_min) / span * 100`` (clipped to [0, 100]), where
``tick`` is the servo's Present_Position, which already includes the EEPROM homing offset.
:func:`normalized_to_ticks` is the exact inverse of that (checked against the plugin by
``scripts/arm_fk.py check-plugin``).

Ticks -> URDF angle: ``q = sign * (tick - zero_tick) * 2*pi / 4096`` with the values in
:data:`JOINT_MAP`. Evidence for every number is in ``docs/perception/B-arm-kinematics.md``.
In short: Studio's calibration wizard (lerobot ``set_half_turn_homings``) makes the pose in
which the arm was held during homing read 2047 on every servo; that pose is the SO-101
"middle" pose that the URDF uses as its zero (upper arm vertical, forearm and gripper
horizontal, pointing forward). The midpoint of the recorded range, which Studio's "degrees"
unit and LeRobot use as zero, is *not* the URDF zero for this arm (up to 8 deg off).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from hackjunction2026.kinematics.model import JOINTS

TICKS_PER_TURN: Final = 4096
HOMING_TICK: Final = 2047
"""Present_Position of every servo in the pose held during Studio's homing step."""


@dataclass(frozen=True)
class JointMap:
    sign: int
    zero_tick: float
    """Servo tick at which the URDF joint angle is 0."""


JOINT_MAP: Final = {
    "shoulder_pan": JointMap(+1, HOMING_TICK),
    "shoulder_lift": JointMap(+1, HOMING_TICK),
    "elbow_flex": JointMap(+1, HOMING_TICK),
    "wrist_flex": JointMap(+1, HOMING_TICK),
    "wrist_roll": JointMap(+1, HOMING_TICK),
    # The jaws touch at tick 2069 (normalized 2.5, measured: empty gripper stalls there when
    # commanded closed) and at URDF angle -11.6 deg (meshes) -> zero = 2069 + 11.6/360*4096.
    "gripper": JointMap(+1, 2201.0),
}


@dataclass(frozen=True)
class Calibration:
    """Per-joint calibrated tick range (Studio / LeRobot calibration file)."""

    range_min: np.ndarray  # (6,) ticks, JOINTS order
    range_max: np.ndarray

    @classmethod
    def from_dict(cls, data: dict) -> Calibration:
        cal = data.get("calibration", data)
        return cls(
            range_min=np.array([float(cal[j]["range_min"]) for j in JOINTS]),
            range_max=np.array([float(cal[j]["range_max"]) for j in JOINTS]),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Calibration:
        """Load Studio's ``configs/calibration/follower.json`` or a plain LeRobot calibration file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def normalized_to_ticks(norm: np.ndarray, cal: Calibration) -> np.ndarray:
    """Inverse of the plugin's ``ticks_to_normalized`` (float ticks, not rounded), shape (..., 6)."""
    norm = np.asarray(norm, dtype=np.float64)
    span = cal.range_max - cal.range_min
    ticks = np.empty_like(norm)
    ticks[..., :5] = cal.range_min[:5] + (norm[..., :5] + 100.0) / 200.0 * span[:5]
    ticks[..., 5] = cal.range_min[5] + norm[..., 5] / 100.0 * span[5]
    return ticks


def ticks_to_normalized(ticks: np.ndarray, cal: Calibration) -> np.ndarray:
    """Same formula as the plugin (with its clipping), vectorized."""
    ticks = np.clip(np.asarray(ticks, dtype=np.float64), cal.range_min, cal.range_max)
    span = cal.range_max - cal.range_min
    norm = np.empty_like(ticks)
    norm[..., :5] = (ticks[..., :5] - cal.range_min[:5]) / span[:5] * 200.0 - 100.0
    norm[..., 5] = (ticks[..., 5] - cal.range_min[5]) / span[5] * 100.0
    return norm


_SIGN = np.array([JOINT_MAP[j].sign for j in JOINTS], dtype=np.float64)
_ZERO = np.array([JOINT_MAP[j].zero_tick for j in JOINTS], dtype=np.float64)


def ticks_to_urdf(ticks: np.ndarray) -> np.ndarray:
    """Servo ticks (..., 6) -> URDF joint angles in radians (..., 6)."""
    return _SIGN * (np.asarray(ticks, dtype=np.float64) - _ZERO) * (2.0 * np.pi / TICKS_PER_TURN)


def urdf_to_ticks(q: np.ndarray) -> np.ndarray:
    """URDF joint angles in radians (..., 6) -> servo ticks (float)."""
    return np.asarray(q, dtype=np.float64) * _SIGN * (TICKS_PER_TURN / (2.0 * np.pi)) + _ZERO


def normalized_to_urdf(norm: np.ndarray, cal: Calibration) -> np.ndarray:
    """Studio normalized joint values (..., 6) -> URDF joint angles in radians (..., 6)."""
    return ticks_to_urdf(normalized_to_ticks(norm, cal))


def urdf_to_normalized(q: np.ndarray, cal: Calibration) -> np.ndarray:
    """URDF joint angles (..., 6) -> Studio normalized values (clipped like the driver)."""
    return ticks_to_normalized(urdf_to_ticks(q), cal)

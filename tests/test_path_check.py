"""Straight joint-space path check against the table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hackjunction2026.kinematics import TABLE_Z_MM, Calibration, ik
from hackjunction2026.kinematics.path import check_straight_move

CAL = Calibration.from_file(Path(__file__).resolve().parents[1] / "configs/calibration/follower.json")
OBSERVE = np.array([-12.76, -97.0, 26.4, 97.0, -59.58, 93.18])


def test_observe_to_hover_is_clear():
    s = ik.solve_down(230.0, 0.0, TABLE_Z_MM + 60.0, 90.0, CAL, OBSERVE[5], start_norm=OBSERVE)
    pc = check_straight_move(OBSERVE, s.norm, CAL)
    assert pc.ok and pc.samples == 50
    assert abs(pc.end_clearance_mm - 60.0) < 0.5  # the fixed fingertip is the lowest point at the hover


def test_refuses_a_target_below_the_clearance():
    s = ik.solve_down(230.0, 0.0, TABLE_Z_MM + 10.0, 90.0, CAL, OBSERVE[5], start_norm=OBSERVE)
    assert s.ok
    pc = check_straight_move(OBSERVE, s.norm, CAL)
    assert not pc.ok and pc.at_fraction == 1.0 and pc.min_clearance_mm < 30.0


def test_refuses_a_move_that_dips_below_the_table_halfway():
    # both ends are more than 50 mm above the table, but the straight joint path swings the gripper
    # through the table (found by a random search over pose pairs)
    start = np.array([-10.7, 14.0, 28.0, -12.9, -48.9, 30.0])
    target = np.array([-52.2, 22.8, 88.5, 73.1, -37.7, 30.0])
    assert check_straight_move(start, start, CAL).min_clearance_mm > 50
    assert check_straight_move(target, target, CAL).min_clearance_mm > 50
    pc = check_straight_move(start, target, CAL)
    assert not pc.ok
    assert pc.min_clearance_mm < 0 and 0.3 < pc.at_fraction < 0.7
    assert "REFUSED" in pc.describe()


@pytest.mark.parametrize("samples", [10, 50, 200])
def test_samples_include_both_ends(samples):
    pc = check_straight_move(OBSERVE, OBSERVE, CAL, samples=samples)
    assert pc.ok and pc.samples == samples and pc.min_clearance_mm > 100

"""Wrist homography calibration against a fake arm and a fake camera that "sees" the brick through a known
homography: the fit must recover it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hackjunction2026 import calib_wrist as cw
from hackjunction2026 import sort_cycle as sc
from hackjunction2026 import table_map as tm
from hackjunction2026.kinematics import Calibration

REPO = Path(__file__).resolve().parents[1]
OBSERVE = np.array([-12.76, -97.0, 26.4, 97.0, -59.58, 93.18])
PHOTO = np.array([-8.55, 51.63, -40.66, 80.44, -27.73, 3.02])
H_TRUE = np.array([[0.9, 0.05, 40.0], [-0.04, -0.8, 520.0], [0.0001, 0.0002, 1.0]])  # table -> pixel


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(sc.m.time, "sleep", lambda s: None)
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    monkeypatch.setattr(cw.time, "sleep", lambda s: None)


class Cams:
    def wrist(self):
        return np.zeros((768, 1024, 3), np.uint8)




class FakeBrick:
    def __init__(self, c):
        self.centroid, self.area_px, self.colour = c, 1600, "red"


def test_calibration_recovers_the_homography(tmp_path, monkeypatch):
    cal = Calibration.from_file(REPO / "configs/calibration/follower.json")
    cfg = sc.Config()
    poses = {"observe": OBSERVE, "photo": PHOTO, "bowl_good": OBSERVE, "bowl_defect": OBSERVE}
    points, _ = cw.plan_points(OBSERVE, cal, cfg, n=6)
    assert len(points) == 6
    cycle = sc.Cycle(sc.FakeArm(OBSERVE), Cams(), None, sc.ScriptedGrasp(), cfg, cal, poses, dry_run=True, step=False,
                     out_root=tmp_path)
    calib = cw.Calib(cycle)

    def fake_detect(frame):
        px = tm.apply_homography(H_TRUE, np.array(calib.last_place_xy))
        return type("D", (), {"bricks": [FakeBrick(tuple(px))]})()

    monkeypatch.setattr(cw.tm, "detect", fake_detect)
    monkeypatch.setattr(cw.tm, "annotate", lambda frame, det: frame)
    cycle.mv.move(OBSERVE, "observe")
    calib.closed_empty_sample()
    calib.take_from_photo()
    for i, p in enumerate(points):
        calib.place_and_look(i, p)
    calib.return_to_photo()
    out = tmp_path / "h.yaml"
    stats = calib.fit(out)
    assert stats["n"] == 6 and stats["rms_mm"] < 0.5
    th = tm.TableHomography.load(out)
    for pr in calib.pairs:
        assert np.allclose(th.to_table(np.array(pr["pixel"])), pr["place_xy"], atol=0.5)
    g = calib.gripper_summary()
    assert g["closed_empty"]["pos"][0] < 3 and min(g["closed_on_brick"]["pos"]) > 8


def test_points_are_reachable_and_spread():
    cal = Calibration.from_file(REPO / "configs/calibration/follower.json")
    points, _ = cw.plan_points(OBSERVE, cal, sc.Config(), n=6)
    xy = np.array([p.xy for p in points])
    r = np.linalg.norm(xy - cw.PAN_AXIS_XY, axis=1)
    assert r.min() >= 185 and r.max() <= 260
    assert np.ptp(xy[:, 1]) > 150

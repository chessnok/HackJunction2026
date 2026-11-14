"""Sorting cycle state machine against a fake arm, fake cameras and a fake detector (no hardware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hackjunction2026 import sort_cycle as sc
from hackjunction2026.kinematics import Calibration

REPO = Path(__file__).resolve().parents[1]
OBSERVE = np.array([-12.76, -97.0, 26.4, 97.0, -59.58, 93.18])
PHOTO = np.array([-8.55, 51.63, -40.66, 80.44, -27.73, 3.02])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(sc.m.time, "sleep", lambda s: None)
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)


@pytest.fixture
def cal():
    return Calibration.from_file(REPO / "configs/calibration/follower.json")


class Cams:
    def wrist(self):
        return np.zeros((768, 1024, 3), np.uint8)

    def all(self):
        return {"top": np.zeros((480, 848, 3), np.uint8)}


class Detector:
    def __init__(self, bad: bool):
        self.bad = bad

    def judge(self, frames, scene, out):
        return {"is_defective": self.bad, "n_crops": len(frames)}


def make_cycle(tmp_path, cal, monkeypatch, *, bad=False, brick_at=32.0, step=False, answers=None):
    cfg = sc.Config()
    poses = {"observe": OBSERVE, "photo": PHOTO, "bowl_good": OBSERVE.copy(), "bowl_defect": OBSERVE.copy()}
    target = sc.Target((200.0, 40.0), 90.0, "red", "test")
    monkeypatch.setattr(sc.Cycle, "locate", lambda self, q: target)
    arm = sc.FakeArm(OBSERVE, brick_at=brick_at)
    asked = iter(answers or [])
    cyc = sc.Cycle(arm, Cams(), Detector(bad), sc.ScriptedGrasp(), cfg, cal, poses, dry_run=True, step=step,
                   out_root=tmp_path, ask=lambda prompt: next(asked))
    return cyc, arm


def events(cyc):
    import json

    return [json.loads(line) for line in (cyc.out / "cycle.jsonl").read_text().splitlines()]


def test_full_cycle_good_brick(tmp_path, cal, monkeypatch):
    cyc, arm = make_cycle(tmp_path, cal, monkeypatch, bad=False)
    res = cyc.run_one()
    cyc.close()
    assert res == {"result": "sorted", "defective": False, "colour": "red"}
    states = [e["state"] for e in events(cyc) if e["event"] == "enter"]
    assert states == ["observe", "locate", "hover", "grasp", "to_photo", "release_photo", "inspect", "regrasp",
                      "to_bowl", "release_bowl", "done"]


def test_photo_pose_reached_exactly_without_changing_the_roll(tmp_path, cal, monkeypatch):
    cyc, arm = make_cycle(tmp_path, cal, monkeypatch)
    cyc.run_one()
    photo_moves = [e for e in events(cyc) if e["event"] == "move" and e["what"].startswith(("photo spot", "back to the photo spot"))]
    assert photo_moves
    for e in photo_moves:
        assert np.allclose(e["target"][:5], PHOTO[:5], atol=1e-3)
    pre = [e for e in events(cyc) if e["event"] == "move" and e["what"] == "above photo spot"]
    assert pre and abs(pre[0]["target"][0] - PHOTO[0]) < 1e-6 and abs(pre[0]["target"][4] - PHOTO[4]) < 1e-6


def test_defective_goes_to_the_defect_bowl(tmp_path, cal, monkeypatch):
    cyc, arm = make_cycle(tmp_path, cal, monkeypatch, bad=True)
    assert cyc.run_one()["defective"] is True
    assert any(e["event"] == "move" and "defect bowl" in e["what"] for e in events(cyc))


def test_empty_grasp_retries_three_times_then_starts_over_then_stops(tmp_path, cal, monkeypatch):
    cyc, arm = make_cycle(tmp_path, cal, monkeypatch, brick_at=None)
    with pytest.raises(sc.StopCycle, match="could not grasp"):
        cyc.run_one()
    grasps = [e for e in events(cyc) if e["event"] == "grasp"]
    assert len(grasps) == 6 and not any(g["ok"] for g in grasps)          # 3 tries, start over, 3 tries
    assert sum(e["event"] == "retries_exhausted" for e in events(cyc)) == 2


def test_holding_rule():
    cfg = sc.Config()
    assert sc.GripperState(32.0, 300, []).holding(cfg)
    assert not sc.GripperState(2.6, 30, []).holding(cfg)
    assert sc.GripperState(6.0, 10, ["OVERLOAD"]).holding(cfg)


def test_step_prompt_q_stops(tmp_path, cal, monkeypatch):
    cyc, arm = make_cycle(tmp_path, cal, monkeypatch, step=True, answers=["q"])
    cyc.mv.dry_run = False                           # the prompt only appears for real moves
    with pytest.raises(sc.StopCycle, match="step prompt"):
        cyc.mv.move(OBSERVE + np.array([2, 0, 0, 0, 0, 0]), "test")


def test_refuses_a_move_through_the_table(tmp_path, cal, monkeypatch):
    cyc, arm = make_cycle(tmp_path, cal, monkeypatch)
    low = PHOTO.copy()
    low[1] += 30                                     # push the shoulder further down: jaws into the table
    with pytest.raises(sc.StopCycle):
        cyc.mv.move(low, "into the table", near_table=True)


def test_missing_poses_are_reported(monkeypatch, tmp_path):
    cfg = sc.Config()
    cfg.poses = {"observe": "observe", "bowl_good": "no_such_pose_xyz"}
    poses, missing = sc.load_poses(cfg)
    assert "observe" in poses and missing == ["bowl_good (pose 'no_such_pose_xyz')"]

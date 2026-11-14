"""End to end on synthetic data: calib snapshots + observe image -> homography -> plan -> joints.

A synthetic 'true' homography places the bricks on the table; fake Studio snapshots hold joint
values (from IK) that put the gripper over each brick, as an operator would in teleop."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV: uv run --extra table pytest")

from hackjunction2026 import pick_above as pa
from hackjunction2026 import table_map as tm
from hackjunction2026.kinematics import (
    TABLE_Z_MM,
    Calibration,
    ik,
    normalized_to_urdf,
)

CAL = Calibration.from_file(pa.FOLLOWER_CAL)
BRICKS = {"red": (420, 330), "blue": (640, 330), "lime": (280, 400), "orange": (760, 470),
          "yellow": (520, 560), "green": (560, 420)}


def _h_true():
    # pixel -> table: the near part of the sheet lies inside the arm's reach
    src = np.array([[290, 270], [790, 255], [1000, 620], [120, 560]], float)
    dst = np.array([[290, 150], [290, -150], [120, -140], [120, 140]], float)
    return tm.fit_homography(src, dst)


def _write_snapshot(root, sid, comment, joints, cal_json):
    d = root / "proj" / sid
    d.mkdir(parents=True)
    state = {"snapshot_id": sid, "comment": comment,
             "robots": {"follower": {"calibration": cal_json,
                                     "joint_positions": {f"{j}.pos": float(v) for j, v in zip(pa.JOINTS, joints)}}}}
    (d / "state.json").write_text(json.dumps(state))


@pytest.fixture
def setup(tmp_path, scene, monkeypatch):
    monkeypatch.setattr(pa, "CAPTURES", tmp_path / "captures")
    h = _h_true()
    rng = np.random.default_rng(5)
    cal_json = json.loads(pa.FOLLOWER_CAL.read_text())["calibration"]
    snaps = tmp_path / "snapshots"
    truth = {}
    for k, (colour, px) in enumerate(BRICKS.items()):
        x, y = tm.apply_homography(h, np.array(px, float)) + rng.normal(0, 1.0, 2)  # operator error ~1 mm
        truth[colour] = (x, y)
        # operator holds the jaws (gripper 25) around the brick, fingertips 15 mm above the table
        s = ik.solve_down(x, y, TABLE_Z_MM + 15, 30.0 * k, CAL, 25.0)
        assert s.ok, (colour, x, y, s.reason)
        _write_snapshot(snaps, f"20261114-0700{k:02d}-000", f"calib {colour}", s.norm, cal_json)
    _write_snapshot(snaps, "20261114-065900-000", "calib red", np.zeros(6), cal_json)  # older, ignored
    _write_snapshot(snaps, "20261114-070500-000", "pose: observe", np.zeros(6), cal_json)  # not calib
    img = tmp_path / "observe.png"
    cv2.imwrite(str(img), scene(BRICKS, angle=0.0))
    return {"h": h, "snaps": snaps, "img": img, "truth": truth, "out": tmp_path / "table_homography.yaml"}


def test_calib_points(setup):
    pts, notes = pa.calib_points(setup["snaps"])
    assert sorted(p.colour for p in pts) == sorted(BRICKS)
    for p in pts:
        assert np.hypot(*(p.xyz_mm[:2] - setup["truth"][p.colour])) < 1e-3
        assert abs(p.xyz_mm[2] - (TABLE_Z_MM + 15)) < 1e-3 and p.tilt_deg < 0.01
    assert any("ignoring 20261114-065900-000" in n for n in notes)


def test_calib_fit_then_plan(setup, capsys):
    pa.main(["calib-fit", "--image", str(setup["img"]), "--snapshots", str(setup["snaps"]),
             "--out", str(setup["out"])])
    out = capsys.readouterr().out
    th = tm.TableHomography.load(setup["out"])
    assert th.pose == "observe" and th.image_size == (1024, 768) and th.point == "jaw_centre"
    assert th.stats["n_points"] == 6 and th.stats["loo_rms_mm"] < 5.0
    assert "leave-one-out RMS" in out

    pa.main(["plan", "--image", str(setup["img"]), "--homography", str(setup["out"]), "--brick", "blue"])
    out = capsys.readouterr().out
    m = re.search(r'goto "([-\d.,]+)" --go', out)
    assert m, out
    target = np.array([float(v) for v in m.group(1).split(",")])
    tp = ik.tool_pose(normalized_to_urdf(target, CAL), "jaw_centre")
    x_true, y_true = tm.apply_homography(setup["h"], np.array(BRICKS["blue"], float))
    assert np.hypot(tp["position"][0] - x_true, tp["position"][1] - y_true) < 4.0  # homography + 1 mm noise
    assert abs(tp["position"][2] - (TABLE_Z_MM + 60)) < 0.5
    assert ik.tilt_deg(tp["axis"]) < 1.0
    assert target[5] == pytest.approx(pa.load_pose("observe")[5], abs=0.01)  # gripper stays as in observe
    # square brick, image rectangle aligned with the image axes -> jaws close along one of its sides
    yaw = ik.closing_yaw_deg(tp["closing"])
    th_long = tm.TableHomography.load(setup["out"])
    side = th_long.yaw_deg(BRICKS["blue"], 0.0)
    assert min(ik.axis_angle_diff_deg(yaw, side), ik.axis_angle_diff_deg(yaw, side + 90)) < 2.0


def test_plan_refuses_low_hover_and_unreachable(setup, capsys):
    pa.main(["calib-fit", "--image", str(setup["img"]), "--snapshots", str(setup["snaps"]),
             "--out", str(setup["out"])])
    with pytest.raises(SystemExit, match="path check"):
        pa.main(["plan", "--image", str(setup["img"]), "--homography", str(setup["out"]), "--brick", "red",
                 "--height", "15"])
    with pytest.raises(SystemExit, match="several|choose one|bricks on the sheet"):
        pa.main(["plan", "--image", str(setup["img"]), "--homography", str(setup["out"])])


def test_plan_refuses_a_brick_out_of_reach(setup, scene, tmp_path):
    pa.main(["calib-fit", "--image", str(setup["img"]), "--snapshots", str(setup["snaps"]),
             "--out", str(setup["out"])])
    th = tm.TableHomography.load(setup["out"])
    px = (540.0, 285.0)  # far edge of the synthetic sheet
    x, y = th.to_table(np.array(px))
    assert 250 < x < 300 and abs(y) < 20
    far = tmp_path / "far.png"
    cv2.imwrite(str(far), scene({"red": px}, off_sheet=False))
    # reachable at 60 mm (the reach ends at x = 290 on the centre line), not at 150 mm
    assert ik.solve_down(x, y, TABLE_Z_MM + 150, 0.0, CAL, 93.18).ok is False
    with pytest.raises(SystemExit, match="REFUSED: IK"):
        pa.main(["plan", "--image", str(far), "--homography", str(setup["out"]), "--height", "150"])


def test_calib_fit_needs_four_points(setup, tmp_path):
    snaps = tmp_path / "few"
    cal_json = json.loads(pa.FOLLOWER_CAL.read_text())["calibration"]
    for k, c in enumerate(["red", "blue", "lime"]):
        _write_snapshot(snaps, f"20261114-0800{k:02d}-000", f"calib {c}", np.zeros(6), cal_json)
    with pytest.raises(SystemExit, match=">= 4"):
        pa.main(["calib-fit", "--image", str(setup["img"]), "--snapshots", str(snaps),
                 "--out", str(tmp_path / "h.yaml")])


def test_wrist_camera_resolution_never_picks_the_hp_camera(tmp_path):
    by_id, sysfs = tmp_path / "by-id", tmp_path / "sys"
    by_id.mkdir()
    for node, name in [("video0", "HP 5MP Camera: HP 5MP Camera"), ("video4", "USB2.0_CAM1: USB2.0_CAM1")]:
        (tmp_path / node).touch()
        (sysfs / node).mkdir(parents=True)
        (sysfs / node / "name").write_text(name + "\n")
    (by_id / "usb-DUYWV0A5ALZ30A_HP_5MP_Camera_200901010001-video-index0").symlink_to(tmp_path / "video0")
    (by_id / "usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0").symlink_to(tmp_path / "video4")
    link = pa.resolve_wrist_camera("USB2.0_CAM1", by_id, sysfs)
    assert link.name.startswith("usb-Sonix") and link.resolve().name == "video4"
    with pytest.raises(SystemExit):
        pa.resolve_wrist_camera("HP_5MP_Camera", by_id, sysfs)
    with pytest.raises(SystemExit):
        pa.resolve_wrist_camera("NoSuchCam", by_id, sysfs)

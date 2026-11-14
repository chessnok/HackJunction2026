"""Brick detection and the pixel -> table homography."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV: uv run --extra table pytest")

from conftest import SAMPLE_IMAGE

from hackjunction2026 import table_map as tm


@pytest.mark.skipif(not Path(SAMPLE_IMAGE).exists(), reason="sample snapshot not on this machine")
def test_sample_image_one_yellow_brick_on_the_sheet():
    det = tm.detect(cv2.imread(SAMPLE_IMAGE))
    assert det.sheet_polygon is not None
    assert [b.colour for b in det.bricks] == ["yellow"]
    u, v = det.bricks[0].centroid
    assert abs(u - 510) < 10 and abs(v - 389) < 10
    # off-sheet red and green bricks on the wood, the green plate and the gripper: not bricks, and the
    # off-sheet points lie outside the sheet polygon
    for p in [(1000, 390), (1010, 480), (960, 700), (300, 520), (700, 640)]:
        assert all(np.hypot(b.centroid[0] - p[0], b.centroid[1] - p[1]) > 60 for b in det.bricks)
    poly = det.sheet_polygon.reshape(-1, 1, 2).astype(np.float32)
    for p in [(1000, 390), (1010, 480)]:
        assert cv2.pointPolygonTest(poly, p, False) < 0
    assert cv2.pointPolygonTest(poly, (510.0, 389.0), False) > 0


def test_synthetic_scene_bricks_colours_and_exclusions(scene):
    want = {"red": (400, 330), "blue": (650, 320), "lime": (280, 400), "orange": (760, 450), "yellow": (520, 560)}
    det = tm.detect(scene(want))
    got = {b.colour: b.centroid for b in det.bricks}
    assert set(got) == set(want)  # off-sheet red brick, plate and gripper excluded
    for c, (u, v) in want.items():
        assert abs(got[c][0] - u) < 2 and abs(got[c][1] - v) < 2
    assert all(abs(abs(b.angle_deg) - 45) < 30 or abs(b.angle_deg) < 30 for b in det.bricks)


def test_synthetic_angle_of_an_elongated_brick(scene):
    img = scene({}, off_sheet=False)
    box = cv2.boxPoints(((500.0, 400.0), (70.0, 35.0), 30.0))
    cv2.fillPoly(img, [np.round(box).astype(np.int32)], (30, 30, 200))
    (b,) = tm.detect(img).bricks
    assert abs(b.angle_deg - 30.0) < 3.0
    assert b.aspect > 1.7


def _h_true():
    src = np.array([[290, 270], [790, 255], [1000, 620], [120, 560]], float)
    dst = np.array([[330, 170], [330, -170], [130, -150], [130, 150]], float)
    return tm.fit_homography(src, dst)


def test_homography_exact_and_noisy_fit():
    h = _h_true()
    rng = np.random.default_rng(0)
    px = rng.uniform([150, 280], [900, 620], size=(7, 2))
    xy = tm.apply_homography(h, px)
    h_fit = tm.fit_homography(px, xy)
    grid = np.stack(np.meshgrid(np.linspace(200, 900, 8), np.linspace(280, 600, 6)), -1).reshape(-1, 2)
    assert np.abs(tm.apply_homography(h_fit, grid) - tm.apply_homography(h, grid)).max() < 1e-6
    assert np.nanmax(tm.leave_one_out(px, xy)) < 1e-6
    noisy = xy + rng.normal(0, 1.0, xy.shape)
    loo = tm.leave_one_out(px, noisy)
    assert np.all(np.isfinite(loo)) and loo.max() < 8.0
    h_noisy = tm.fit_homography(px, noisy)
    assert np.abs(tm.apply_homography(h_noisy, grid) - tm.apply_homography(h, grid)).max() < 6.0


def test_homography_ransac_rejects_an_outlier_and_loo_needs_five():
    h = _h_true()
    px = np.array([[300, 300], [700, 290], [850, 520], [200, 520], [500, 400], [450, 560], [650, 450]], float)
    xy = tm.apply_homography(h, px)
    xy[2] += [40.0, -30.0]  # a mismatched brick
    h_fit, inl = tm.fit_homography_ransac(px, xy, threshold_mm=10.0)
    assert inl.tolist() == [True, True, False, True, True, True, True]
    assert np.abs(tm.apply_homography(h_fit, px[inl]) - xy[inl]).max() < 1e-6
    assert np.isnan(tm.leave_one_out(px[:4], xy[:4])).all()


def test_homography_yaml_round_trip_and_yaw(tmp_path):
    rot = np.radians(30)
    h = np.array([[0.5 * np.cos(rot), -0.5 * np.sin(rot), 10], [0.5 * np.sin(rot), 0.5 * np.cos(rot), -20],
                  [0, 0, 1]])
    th = tm.TableHomography(H=h, image_size=(1024, 768), pose="observe", pose_joints={"shoulder_pan": 1.0},
                            point="jaw_centre", date="2026-11-14")
    th.save(tmp_path / "h.yaml")
    th2 = tm.TableHomography.load(tmp_path / "h.yaml")
    assert np.allclose(th2.H, h) and th2.image_size == (1024, 768) and th2.pose == "observe"
    assert abs(th2.yaw_deg((500, 400), 0.0) - 30.0) < 1e-6
    assert abs(th2.yaw_deg((500, 400), 45.0) - 75.0) < 1e-6


def test_touching_bricks_of_different_colours_are_split(scene):
    from conftest import HUES, hsv_bgr

    img = scene({}, off_sheet=False)
    cv2.rectangle(img, (480, 380), (520, 420), hsv_bgr(HUES["blue"]), -1)
    cv2.rectangle(img, (521, 380), (561, 420), hsv_bgr(HUES["yellow"]), -1)  # touching, one blob
    det = tm.detect(img)
    got = sorted((b.colour, round(b.centroid[0])) for b in det.bricks)
    assert [c for c, _ in got] == ["blue", "yellow"]
    assert abs(got[0][1] - 500) < 4 and abs(got[1][1] - 541) < 4


def test_pale_bright_brick_found_dark_bluish_shadow_not(scene):
    from conftest import hsv_bgr

    img = scene({}, off_sheet=False)
    cv2.rectangle(img, (480, 380), (520, 420), hsv_bgr(40, 70, 200), -1)    # pale lime brick
    cv2.rectangle(img, (650, 380), (690, 420), hsv_bgr(116, 59, 76), -1)    # bluish shadow, dark
    det = tm.detect(img)
    assert [(b.colour, round(b.centroid[0])) for b in det.bricks] == [("lime", 500)]

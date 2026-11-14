"""Grasp-merge overrides: cut and end trimming on a synthetic OLD episode (no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from hackjunction2026.grasp_merge import cuts as C  # noqa: E402

N = 120
REF = {"h_mm": {"median": 120.0, "std": 15.0, "min": 85.0, "max": 135.0},
       "d_mm": {"median": 25.0, "std": 12.0, "max": 65.0}}


def episode() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rest (closed, far), hover above the brick with open jaws at 120 mm (40-60), descent to the
    table (61-100), close on the brick (command 5 from frame 98, jaws stop at 15)."""
    f = np.arange(N)
    g = np.where(f < 40, 5.0, np.where(f < 100, 50.0, 15.0))
    act = np.where(f < 40, 5.0, np.where(f < 98, 50.0, 5.0))
    h = np.where(f < 40, 150.0, np.where(f <= 60, 120.0, np.maximum(120.0 - 3.0 * (f - 60), 5.0)))
    d = np.where(f < 30, 100.0, 10.0)
    df = pd.DataFrame({"episode_index": 0, "frame_index": f, "gripper": g, "act_gripper": act, "h_mm": h,
                       "d_mm": d, "gripper_tip_x_mm": 250.0, "gripper_tip_y_mm": 0.0, "jaw_gap_mm": 40.0,
                       "brick_x_mm": 250.0, "brick_y_mm": 0.0, "brick_source": "top+floor"})
    for j in C.JOINTS[:-1]:
        df[j] = 0.0
    eps = pd.DataFrame({"task": ["t"], "length": [N]}, index=pd.Index([0], name="episode_index"))
    return df, eps


def cut(overrides=None) -> dict:
    df, eps = episode()
    return C.compute_cuts(df, eps, REF, C.Params(), 30.0, overrides)[0]


def test_parse_overrides(tmp_path):
    f = tmp_path / "overrides.csv"
    f.write_text("# reviewer edits\nepisode,cut_frame,end_frame\n3,auto,80\n4,10,50\n5,skip\n6,12\n7,auto,\n"
                 "8,20,90  # trailing comment\n")
    assert C.read_overrides(f) == {3: {"cut": "auto", "end": 80}, 4: {"cut": 10, "end": 50},
                                   5: {"cut": "skip", "end": None}, 6: {"cut": 12, "end": None},
                                   7: {"cut": "auto", "end": None}, 8: {"cut": 20, "end": 90}}
    assert C.read_overrides(tmp_path / "missing.csv") == {}


def test_no_override_keeps_to_the_last_frame():
    r = cut()
    assert r["source"] == "auto"
    assert 40 <= r["cut_frame"] <= 63  # end of the 120 mm hover
    assert r["end_frame"] == N - 1
    assert r["kept_frames"] == N - r["cut_frame"]
    assert r["close_frame"] == 98


def test_auto_cut_with_end_trim():
    auto = cut()["cut_frame"]
    r = cut({0: {"cut": "auto", "end": 105}})
    assert (r["cut_frame"], r["end_frame"], r["source"]) == (auto, 105, "override")
    assert r["kept_frames"] == 105 - auto + 1
    assert "ends_before_close" not in r["flags"]


def test_start_and_end_override():
    r = cut({0: {"cut": 50, "end": 90}})
    assert (r["cut_frame"], r["end_frame"], r["kept_frames"]) == (50, 90, 41)
    assert "ends_before_close" in r["flags"]  # the close is at 98
    assert r["confidence"] == "low"


def test_skip_and_bad_ranges():
    assert cut({0: {"cut": "skip", "end": None}})["kept_frames"] == 0
    for bad in ({"cut": 90, "end": 50}, {"cut": 10, "end": N}, {"cut": -1, "end": None}):
        with pytest.raises(ValueError):
            cut({0: bad})

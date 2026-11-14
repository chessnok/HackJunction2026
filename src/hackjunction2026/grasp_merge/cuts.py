"""Cut point per OLD episode: the frame that looks most like a NEW episode's frame 0.

NEW episodes start with the arm hovering above the brick, gripper open, and end grasped. OLD
episodes start at rest and approach the brick first. For every OLD frame we measure, in the robot
base frame (agent B's forward kinematics, agent A's brick positions mapped with the
robot-to-table registration of the same dataset):

- ``h_mm``: height of the gripper tip (midpoint of the fingertips) above the table plane;
- ``d_mm``: horizontal distance from the gripper tip to the brick's resting position (median of
  the first ``brick_frames`` frames of the main brick track, both cameras);
- ``gripper``: Studio normalized gripper value (0 closed, 100 fully open).

Criterion (numbers in docs/datasets/grasp-merged.md):

1. Reference: frame 0 of every NEW episode with at least ``min_episode_frames`` frames gives the
   median and standard deviation of ``h_mm`` and ``d_mm``.
2. Open: ``gripper >= open_abs`` (30: jaw gap 28.5 mm). The OLD operator grasped with openings as
   small as 33; NEW's 60-98 is never reached in OLD (widest OLD openings 33-67), so the gripper
   value is a condition, not a score term.
3. Final approach: the last run of open frames (runs closer than ``run_gap`` frames merged). The
   close (``close_frame``) is the frame after the last frame, from the start of that run, at which
   the commanded gripper (``action``, the leader arm) is at least ``close_drop`` above its final
   value (median of the last 5 frames); if there is none, ``no_close``. The command is used, not
   the measured value: the jaws stop on the brick (a 2x2 brick at 13-23, larger bricks at 32-34,
   above the open threshold) while the command goes to 7-16, and some operators partly close a
   wide-open gripper during the approach.
4. Score per frame from the start of that run to the close:
   ``((h - h_ref) / sd_h)^2 + (max(0, d - d_ref) / sd_d)^2``. Being closer to the brick than NEW's
   median is not penalised.
5. Cut: the latest of those frames whose score is within ``plateau`` of their best score (the last
   hover frame before the descent rather than the middle of a long hover).
6. For review only: ``alt_cut_frame`` = the same rule over all frames before the close, gripper
   ignored (where the arm hovered if the jaws opened only during the descent).

The tail ``[cut_frame, end_frame]`` is kept (inclusive; ``end_frame`` is the episode's last frame
unless a human override trims the end). Flags mark episodes a human should look at.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hackjunction2026.kinematics import TABLE_Z_MM

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

FLAG_HELP = {
    "no_open": "the gripper never opens (>= open threshold); cut chosen on height and distance only",
    "no_close": "the gripper never closes after its last opening: the tail contains no grasp",
    "multi_open": "the gripper opened more than once (several approaches or a re-grasp)",
    "multi_descent": "after the cut the tip goes below 40 mm and back above 80 mm before the close",
    "low_start": "the best open frame is lower than every NEW frame 0 (the jaws open only during the descent)",
    "high_start": "the best open frame is more than 30 mm higher than every NEW frame 0",
    "far_start": "the best open frame is farther from the brick than every NEW frame 0",
    "no_brick": "no camera brick position; distance taken from the FK tip at the close",
    "brick_moved": "the brick's resting position and the FK tip at the close differ by more than 40 mm",
    "short_tail": "fewer than 45 frames (1.5 s) kept",
    "ends_before_close": "end_frame is before the close: the kept range contains no grasp",
}


@dataclass(frozen=True)
class Params:
    min_episode_frames: int = 15  # shorter NEW episodes are accidents: not in the reference, not merged
    brick_frames: int = 10  # resting brick = median over the first N frames (two-camera fixes)
    open_abs: float = 30.0  # open: gripper >= this (jaw gap 28.5 mm)
    run_gap: int = 15  # open runs separated by fewer frames are one run
    close_drop: float = 10.0  # the close starts after the last frame commanded this much above the final command
    plateau: float = 0.25  # score ties: frames within this of the best; the latest wins
    low_descent_mm: float = 40.0  # multi_descent: below this ...
    rise_mm: float = 80.0  # ... then above this again before the close
    high_margin_mm: float = 30.0  # high_start: above the NEW maximum by more than this
    short_tail_frames: int = 45
    brick_moved_mm: float = 40.0
    close_low_mm: float = 30.0  # a close with the tip below this is a grasp at the table


# ---------------------------------------------------------------- inputs


def episode_table(dataset_root: Path) -> pd.DataFrame:
    """episode_index, task, length from a LeRobot v3 dataset's meta/episodes."""
    files = sorted((Path(dataset_root) / "meta/episodes").glob("*/*.parquet"))
    eps = pd.concat([pd.read_parquet(f, columns=["episode_index", "tasks", "length"]) for f in files])
    eps["task"] = [next(iter(t), "") for t in eps.tasks]
    return eps[["episode_index", "task", "length"]].sort_values("episode_index").set_index("episode_index")


def table_to_robot(M_robot_to_table: np.ndarray, xy_table: np.ndarray) -> np.ndarray:
    """Inverse of agent A's registration ``table_xy = M @ (x_robot, -y_robot, 1)``."""
    p = np.c_[xy_table, np.ones(len(xy_table))] @ np.linalg.inv(M_robot_to_table).T
    p = p[:, :2] / p[:, 2:3]
    return np.c_[p[:, 0], -p[:, 1]]


def resting_bricks(objects: pd.DataFrame, robot_to_table: dict, n_frames: int) -> pd.DataFrame:
    """Per episode: the main brick track's resting position in the robot frame (mm).

    Main track = the track with the most rows. Position = median over its first ``n_frames``
    frames seen by both cameras, else over its first ``3 * n_frames`` frames from any camera.
    """
    M = np.array(robot_to_table["M_robot_to_table_3x3"])
    size = objects.groupby(["episode_index", "object_id"]).size().rename("n").reset_index()
    main = size.sort_values("n").groupby("episode_index").tail(1)[["episode_index", "object_id"]]
    o = objects.merge(main, on=["episode_index", "object_id"])
    rows = []
    for ep, e in o.groupby("episode_index"):
        sel = e[(e.frame_index < n_frames) & (e.source == "top+floor")]
        src = "top+floor"
        if sel.empty:
            sel = e[(e.frame_index < 3 * n_frames) & (e.source != "hold")]
            src = "one camera"
        if sel.empty:
            continue
        xy = table_to_robot(M, np.array([[sel.x_mm.median(), sel.y_mm.median()]]))[0]
        rows.append({"episode_index": ep, "brick_x_mm": xy[0], "brick_y_mm": xy[1], "brick_source": src,
                     "brick_color": e.color.mode().iat[0]})
    return pd.DataFrame(rows).set_index("episode_index")


def frame_features(arm: pd.DataFrame, bricks: pd.DataFrame) -> pd.DataFrame:
    """Per frame: joints, gripper tip (robot frame, mm), h_mm, d_mm (NaN without a brick)."""
    cols = ["episode_index", "frame_index", *JOINTS, "act_gripper", "gripper_tip_x_mm", "gripper_tip_y_mm",
            "gripper_tip_z_mm", "jaw_gap_mm"]
    f = arm[cols].sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    f["h_mm"] = f.gripper_tip_z_mm - TABLE_Z_MM
    f = f.join(bricks, on="episode_index")
    f["d_mm"] = np.hypot(f.gripper_tip_x_mm - f.brick_x_mm, f.gripper_tip_y_mm - f.brick_y_mm)
    return f


# ---------------------------------------------------------------- reference (NEW frame 0)


def new_reference(feat_new: pd.DataFrame, episodes_new: pd.DataFrame, p: Params) -> dict:
    keep = episodes_new.index[episodes_new.length >= p.min_episode_frames]
    f0 = feat_new[(feat_new.frame_index == 0) & feat_new.episode_index.isin(keep)]
    ref = {"n_episodes": len(f0), "episodes": [int(x) for x in f0.episode_index]}
    for c in ("h_mm", "d_mm", "jaw_gap_mm", *JOINTS):
        v = f0[c].dropna().to_numpy(float)
        ref[c] = {"median": float(np.median(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                  "min": float(v.min()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                  "max": float(v.max()), "n": len(v)}
    return ref


# ---------------------------------------------------------------- cut per episode


def _runs(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) runs of True, merging runs separated by fewer than ``gap`` frames."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    runs = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - runs[-1][1] <= gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(int(a), int(b)) for a, b in runs]


def score(h: np.ndarray, d: np.ndarray, ref: dict) -> np.ndarray:
    zh = (h - ref["h_mm"]["median"]) / ref["h_mm"]["std"]
    zd = np.maximum(0.0, d - ref["d_mm"]["median"]) / ref["d_mm"]["std"]
    return zh**2 + np.nan_to_num(zd) ** 2


def choose_cut(e: pd.DataFrame, ref: dict, p: Params) -> dict:
    """Automatic cut for one OLD episode (rows sorted by frame_index, frame_index == 0..n-1)."""
    n = len(e)
    g, h = e.gripper.to_numpy(float), e.h_mm.to_numpy(float)
    d = e.d_mm.to_numpy(float).copy()
    flags: list[str] = []
    g_peak = float(g.max())
    thr = p.open_abs
    runs = _runs(g >= thr, p.run_gap)
    if runs:
        a, b = runs[-1]
        act = e.act_gripper.to_numpy(float)
        act_end = float(np.median(act[-5:]))
        above = np.flatnonzero(act[a:] >= act_end + p.close_drop)
        close = a + int(above.max()) + 1 if len(above) else n
        if len(runs) > 1:
            flags.append("multi_open")
        if close >= n:
            flags.append("no_close")
    else:
        a, b, close = 0, n - 1, n
        flags.append("no_open")
    tip = e[["gripper_tip_x_mm", "gripper_tip_y_mm"]].to_numpy(float)
    grasp_tip = tip[close] if close < n and h[close] < p.close_low_mm else None
    if np.isnan(d).all():
        flags.append("no_brick")
        if grasp_tip is not None:
            d = np.hypot(*(tip - grasp_tip).T)
    elif grasp_tip is not None:
        brick = e[["brick_x_mm", "brick_y_mm"]].iloc[0].to_numpy(float)
        if np.hypot(*(brick - grasp_tip)) > p.brick_moved_mm:
            flags.append("brick_moved")
    s = score(h, d, ref)
    cand = np.arange(a, max(close, a + 1))
    best = float(s[cand].min())
    cut = int(cand[s[cand] <= best + p.plateau].max())
    before = np.arange(0, max(close, 1))
    alt = int(before[s[before] <= float(s[before].min()) + p.plateau].max())
    return {"cut_frame": cut, "alt_cut_frame": alt, "close_frame": int(close), "open_threshold": round(thr, 1),
            "gripper_peak": round(g_peak, 1), "n_open_runs": len(runs), "open_run": [a, b],
            "auto_flags": flags, "_d": d, "_s": s}


def _metrics(e: pd.DataFrame, cut: int, end: int, d: np.ndarray, s: np.ndarray, close: int, ref: dict,
             p: Params) -> tuple[dict, list[str]]:
    r = e.iloc[cut]
    m = {"h_mm": float(r.h_mm), "d_mm": float(d[cut]), "gripper": float(r.gripper),
         "jaw_gap_mm": float(r.jaw_gap_mm), "score": float(s[cut])}
    m["z_h"] = (m["h_mm"] - ref["h_mm"]["median"]) / ref["h_mm"]["std"]
    m["z_d"] = (m["d_mm"] - ref["d_mm"]["median"]) / ref["d_mm"]["std"]
    for j in JOINTS:
        m[f"state_{j}"] = float(r[j])
    flags = []
    if m["h_mm"] < ref["h_mm"]["min"]:
        flags.append("low_start")
    if m["h_mm"] > ref["h_mm"]["max"] + p.high_margin_mm:
        flags.append("high_start")
    if m["d_mm"] > ref["d_mm"]["max"]:
        flags.append("far_start")
    seg = e.h_mm.to_numpy(float)[cut:min(close, end + 1)]
    low = np.flatnonzero(seg < p.low_descent_mm)
    if len(low) and (seg[low[0]:] > p.rise_mm).any():
        flags.append("multi_descent")
    if end - cut + 1 < p.short_tail_frames:
        flags.append("short_tail")
    if end < close:
        flags.append("ends_before_close")
    return m, flags


# ---------------------------------------------------------------- overrides


def parse_override(line: str) -> tuple[int, dict] | None:
    """One overrides line -> ``(episode, {"cut": int | "auto" | "skip", "end": int | None})``.

    Format ``episode,cut_frame[,end_frame]`` or ``episode,skip``: ``cut_frame`` is the first frame
    kept or ``auto`` (keep the automatic cut), ``end_frame`` the last frame kept (inclusive,
    optional: default the episode's last frame). ``#`` starts a comment; a header line is ignored.
    """
    line = line.split("#", 1)[0].strip()
    if not line or line.lower().startswith("episode"):
        return None
    parts = [x.strip() for x in line.split(",")]
    ep = int(parts[0])
    val = parts[1].lower() if len(parts) > 1 else ""
    if val == "skip":
        return ep, {"cut": "skip", "end": None}
    cut: int | str = "auto" if val in ("", "auto") else int(val)
    end = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return ep, {"cut": cut, "end": end}


def read_overrides(path: Path | None) -> dict[int, dict]:
    """``{episode: {"cut": int | "auto" | "skip", "end": int | None}}`` from an overrides file
    (format: ``parse_override``); a missing file means no overrides."""
    out: dict[int, dict] = {}
    if path is None or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        parsed = parse_override(line)
        if parsed:
            out[parsed[0]] = parsed[1]
    return out


# ---------------------------------------------------------------- all episodes


def compute_cuts(feat_old: pd.DataFrame, episodes_old: pd.DataFrame, ref: dict, p: Params, fps: float,
                 overrides: dict[int, dict] | None = None) -> list[dict]:
    """One row per OLD episode: the kept range ``[cut_frame, end_frame]`` (inclusive) and its metrics."""
    overrides = overrides or {}
    rows = []
    for ep, e in feat_old.groupby("episode_index"):
        e = e.reset_index(drop=True)
        n = int(episodes_old.loc[ep, "length"])
        assert len(e) == n and (e.frame_index.to_numpy() == np.arange(n)).all(), f"episode {ep}: frames"
        auto = choose_cut(e, ref, p)
        cut, end, source = auto["cut_frame"], n - 1, "auto"
        o = overrides.get(int(ep))
        if o is not None:
            if o["cut"] == "skip":
                source = "skip"
            else:
                source = "override"
                cut = cut if o["cut"] == "auto" else int(o["cut"])
                end = end if o["end"] is None else int(o["end"])
                if not 0 <= cut <= end <= n - 1:
                    raise ValueError(f"override for episode {ep}: kept range [{cut}, {end}] not inside "
                                     f"0..{n - 1} (cut_frame <= end_frame)")
        m, cut_flags = _metrics(e, cut, end, auto["_d"], auto["_s"], auto["close_frame"], ref, p)
        flags = auto["auto_flags"] + cut_flags
        kept = 0 if source == "skip" else end - cut + 1
        row = {"episode": int(ep), "task": episodes_old.loc[ep, "task"], "length": n,
               "cut_frame": int(cut), "cut_time_s": round(cut / fps, 3), "end_frame": int(end),
               "end_time_s": round(end / fps, 3), "kept_frames": kept,
               "kept_s": round(kept / fps, 3), "auto_cut_frame": auto["cut_frame"], "source": source,
               "confidence": "high" if not flags else "low", "flags": flags,
               "alt_cut_frame": auto["alt_cut_frame"], "close_frame": auto["close_frame"],
               "frames_after_close": n - auto["close_frame"], "open_threshold": auto["open_threshold"],
               "gripper_peak": auto["gripper_peak"], "n_open_runs": auto["n_open_runs"],
               "open_run": auto["open_run"],
               "brick_x_mm": float(e.brick_x_mm.iat[0]) if "brick_x_mm" in e else None,
               "brick_y_mm": float(e.brick_y_mm.iat[0]) if "brick_y_mm" in e else None,
               "brick_source": e.brick_source.iat[0] if "brick_source" in e else None}
        row.update({k: round(v, 2) for k, v in m.items()})
        rows.append(row)
    return rows


def criterion_text(p: Params, ref: dict) -> str:
    h, d = ref["h_mm"], ref["d_mm"]
    return (
        f"Reference = frame 0 of {ref['n_episodes']} NEW episodes (>= {p.min_episode_frames} frames): "
        f"tip height h {h['median']:.1f} mm median (sd {h['std']:.1f}, {h['min']:.1f}-{h['max']:.1f}), "
        f"tip-to-brick distance d {d['median']:.1f} mm median (sd {d['std']:.1f}, max {d['max']:.1f}). "
        f"Open = gripper >= {p.open_abs}. Final approach = "
        f"last open run (gaps < {p.run_gap} frames merged). Score = ((h - {h['median']:.1f}) / {h['std']:.1f})^2 "
        f"+ (max(0, d - {d['median']:.1f}) / {d['std']:.1f})^2. Cut = latest frame of the final approach "
        f"with score <= best + {p.plateau}. Keep [cut_frame, end_frame] (end_frame = the episode's last "
        f"frame unless an override trims it)."
    )


def write_outputs(out_dir: Path, rows: list[dict], ref: dict, p: Params, extra: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"format": "hackjunction2026/grasp-merge-cuts/v1", "criterion": criterion_text(p, ref),
           "params": asdict(p), "flag_help": FLAG_HELP, "new_reference": ref, **extra, "episodes": rows}
    (out_dir / "cuts.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    cols = ["episode", "task", "length", "cut_frame", "cut_time_s", "kept_frames", "kept_s", "source",
            "confidence", "flags", "h_mm", "d_mm", "gripper", "jaw_gap_mm", "score", "z_h", "z_d",
            "close_frame", "frames_after_close", "alt_cut_frame", "open_threshold", "gripper_peak",
            "n_open_runs", "auto_cut_frame",
            *[f"state_{j}" for j in JOINTS], "brick_source", "end_frame", "end_time_s"]
    with (out_dir / "cuts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # task names contain newlines in OLD ("red-2\n"): write them escaped, one line per episode
            w.writerow({**r, "task": json.dumps(r["task"])[1:-1], "flags": ";".join(r["flags"])})

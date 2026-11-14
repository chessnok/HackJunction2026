"""Per-frame brick and gripper annotation of one episode from the top + floor D415 views.

Per frame and camera: CV detections (color_cv) -> table footprint centres (scene.Frame). The two
cameras face each other across the table, so a brick seen by both gets two independent
estimates; their bbox centres are also triangulated to a 3D centre whose height tells whether the
brick is on the table or lifted by the gripper. Detections are tracked per episode (nearest
neighbour in the table plane, same colour), short tracks are dropped, and a track that is not
seen in a frame (occluded by the arm) is carried forward with source = "hold".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .color_cv import Blob
from .dataset import Dataset
from .gripper import detect_gripper, static_robot_mask
from .scene import Frame, detect_bricks_cv, exclusion_mask, sheet_roi

BRICK_CENTRE_HEIGHT_MM = 14.0  # triangulated bbox centre of a resting brick: 13-18 mm above the table
LIFT_MM = 13.0  # centre more than 27 mm above the table -> lifted by the gripper


def projection(cam: dict) -> np.ndarray:
    K = np.array(cam["K"]); R = np.array(cam["R_cam_from_table"]); t = np.array(cam["t_cam_from_table_mm"])
    return K @ np.c_[R, t]


def triangulate(P1, P2, p1, p2):
    X = cv2.triangulatePoints(P1, P2, np.array(p1, float).reshape(2, 1), np.array(p2, float).reshape(2, 1))
    X = (X[:3] / X[3]).ravel()
    e = []
    for P, p in ((P1, p1), (P2, p2)):
        q = P @ np.r_[X, 1.0]
        e.append(np.linalg.norm(q[:2] / q[2] - np.asarray(p)))
    return X, max(e)


def colors_compatible(a: str, b: str) -> bool:
    return bool(set(a.split("+")) & set(b.split("+")))


@dataclass
class Track:
    id: int
    color: str
    xy: np.ndarray
    n_seen: int = 0
    color_votes: dict = field(default_factory=dict)


MERGE_MM = 40.0  # same-colour detections closer than this are one brick (split by the jaws)
PAIR_ANY_COLOR_MM = 30.0  # top and floor views this close are one brick even if their colours differ
START_FRAMES = 15  # a real brick is on the sheet when the episode starts (dataset assumption)
MIN_LATE_TRACK_FRAC = 0.25  # ... or it is seen for at least this share of the episode
MAX_HOLD_FRAMES = 45  # keep an unseen brick at its last position for 1.5 s, then drop it


def merge_close(dets: list[tuple[Blob, np.ndarray]]) -> list[tuple[Blob, np.ndarray]]:
    """Merge detections of compatible colour whose table positions are within MERGE_MM (one brick cut
    in two by a jaw, or a two-colour brick seen as two blobs)."""
    dets = list(dets)
    changed = True
    while changed:
        changed = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                (a, xa), (b, xb) = dets[i], dets[j]
                if np.linalg.norm(xa - xb) < MERGE_MM and colors_compatible(a.color, b.color) or \
                        np.linalg.norm(xa - xb) < 0.5 * MERGE_MM:
                    areas = dict(a.color_areas)
                    for k, v in b.color_areas.items():
                        areas[k] = areas.get(k, 0) + v
                    area = a.area + b.area
                    colors = [c for c in sorted(areas, key=areas.get, reverse=True) if areas[c] >= max(40, 0.12 * area)]
                    box = (min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]), max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3]))
                    m = Blob(box, area, colors or a.colors, areas, max(a.score, b.score))
                    big = a if a.area >= b.area else b
                    x = xa if big is a else xb
                    dets[i] = (m, x)
                    del dets[j]
                    changed = True
                    break
            if changed:
                break
    return dets


def annotate_episode(ds: Dataset, frame: Frame, ep: int, step: int = 1, min_track_frac: float = 0.03,
                     detector=None) -> tuple[list[dict], list[dict]]:
    """Returns (object rows, gripper rows). `detector(cam, rgb, roi, exclude) -> list[Blob]` defaults
    to the CV baseline."""
    detector = detector or (lambda cam, rgb, roi, ex: detect_bricks_cv(rgb, roi, ex))
    e = ds.episodes[ep]
    cams = {cam: frame.data["cameras"][frame.setup_of(cam, ep)] for cam in ("top", "floor")}
    P = {cam: projection(c) for cam, c in cams.items()}
    ctx = {}
    for cam in ("top", "floor"):
        f0 = ds.frame(ep, cam, 0)
        roi, _ = sheet_roi(cam, f0)
        ctx[cam] = {"roi": roi, "excl": exclusion_mask(f0, roi), "static": static_robot_mask(f0)}
    wanted = set(range(0, e.length, step)) | {e.length - 1}
    it = {cam: ds.iter_frames(ep, cam, wanted) for cam in ("top", "floor")}
    per_frame = []
    grip_rows = []
    while True:
        try:
            ft, img_t = next(it["top"])
            ff, img_f = next(it["floor"])
        except StopIteration:
            break
        assert ft == ff, (ft, ff)
        dets = {}
        for cam, img in (("top", img_t), ("floor", img_f)):
            blobs = detector(cam, img, ctx[cam]["roi"], ctx[cam]["excl"])
            dets[cam] = merge_close([(b, frame.footprint_center(cam, ep, b.bottom_center)) for b in blobs])
        # top <-> floor matching
        T, F = dets["top"], dets["floor"]
        pairs = []
        if T and F:
            C = np.full((len(T), len(F)), 1e6)
            for i, (bt, xt) in enumerate(T):
                for j, (bf, xf) in enumerate(F):
                    d = np.linalg.norm(xt - xf)
                    # a two-colour brick can show a different part to each camera (top: yellow,
                    # floor: blue): colour-incompatible views still pair when they are this close
                    if not colors_compatible(bt.color, bf.color) and d >= PAIR_ANY_COLOR_MM:
                        continue
                    X, err = triangulate(P["top"], P["floor"], bt.center, bf.center)
                    # either the plane estimates agree or the two rays meet (lifted brick)
                    if d < 80 or (err < 25 and -250 < X[2] < 0):
                        C[i, j] = min(d, err * 3)
            ri, ci = linear_sum_assignment(C)
            pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e6]
        objs = []
        used_t, used_f = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            (bt, xt), (bf, xf) = T[i], F[j]
            X, err = triangulate(P["top"], P["floor"], bt.center, bf.center)
            lifted = X[2] < -(BRICK_CENTRE_HEIGHT_MM + LIFT_MM) and err < 25
            xy = X[:2] if lifted else 0.5 * (xt + xf)
            big, small = (bt, bf) if bt.area >= bf.area else (bf, bt)
            color = "+".join(big.colors + [c for c in small.colors if c not in big.colors])
            objs.append({"xy": xy, "color": color, "top": (bt, xt), "floor": (bf, xf), "z": X[2], "tri_err": err,
                         "lifted": bool(lifted), "source": "top+floor"})
        for i, (bt, xt) in enumerate(T):
            if i not in used_t:
                objs.append({"xy": xt, "color": bt.color, "top": (bt, xt), "floor": None, "z": np.nan,
                             "tri_err": np.nan, "lifted": False, "source": "top"})
        for j, (bf, xf) in enumerate(F):
            if j not in used_f:
                objs.append({"xy": xf, "color": bf.color, "top": None, "floor": (bf, xf), "z": np.nan,
                             "tri_err": np.nan, "lifted": False, "source": "floor"})
        # gripper
        g = {cam: detect_gripper(img, ctx[cam]["static"]) for cam, img in (("top", img_t), ("floor", img_f))}
        grow = {"episode_index": ep, "frame_index": ft, "timestamp": ft / ds.fps}
        for cam in ("top", "floor"):
            d = g[cam]
            grow[f"{cam}_visible"] = d is not None
            grow[f"{cam}_tip_px_x"] = d.tip[0] if d else np.nan
            grow[f"{cam}_tip_px_y"] = d.tip[1] if d else np.nan
            grow[f"{cam}_bbox"] = list(d.bbox) if d else None
            grow[f"{cam}_area_px"] = d.area if d else 0
            xy = frame.to_table(cam, ep, [d.tip])[0] if d else (np.nan, np.nan)
            grow[f"{cam}_tip_plane_x_mm"], grow[f"{cam}_tip_plane_y_mm"] = float(xy[0]), float(xy[1])
        if g["top"] and g["floor"]:
            X, err = triangulate(P["top"], P["floor"], g["top"].tip, g["floor"].tip)
            grow.update({"tip_x_mm": X[0], "tip_y_mm": X[1], "tip_z_mm": X[2], "tip_tri_err_px": err})
        else:
            grow.update({"tip_x_mm": np.nan, "tip_y_mm": np.nan, "tip_z_mm": np.nan, "tip_tri_err_px": np.nan})
        grip_rows.append(grow)
        per_frame.append((ft, objs))

    # tracking
    tracks: list[Track] = []
    assign = []
    for ft, objs in per_frame:
        ids = []
        if objs and tracks:
            C = np.full((len(objs), len(tracks)), 1e6)
            for i, o in enumerate(objs):
                for k, tr in enumerate(tracks):
                    if colors_compatible(o["color"], tr.color):
                        d = np.linalg.norm(o["xy"] - tr.xy)
                        if d < 120:
                            C[i, k] = d
            ri, ki = linear_sum_assignment(C)
            m = {i: k for i, k in zip(ri, ki) if C[i, k] < 1e6}
        else:
            m = {}
        taken = set(m.values())
        for i, o in enumerate(objs):
            near = [k for k, t in enumerate(tracks) if np.linalg.norm(o["xy"] - t.xy) < MERGE_MM]
            if i in m:
                tr = tracks[m[i]]
            elif near:
                # same brick as an existing track (colour seen differently, or a duplicate view)
                free = [k for k in near if k not in taken]
                if not free:
                    ids.append(None)
                    continue
                tr = tracks[free[0]]
                taken.add(free[0])
            else:
                tr = Track(len(tracks), o["color"], o["xy"])
                tracks.append(tr)
            tr.xy = o["xy"]; tr.n_seen += 1
            tr.color_votes[o["color"]] = tr.color_votes.get(o["color"], 0) + 1
            tr.color = max(tr.color_votes, key=tr.color_votes.get)
            ids.append(tr.id)
        assign.append(ids)
    n_frames = len(per_frame)
    first_seen = {}
    for (ft, _), ids in zip(per_frame, assign):
        for tid in ids:
            first_seen.setdefault(tid, ft)
    keep = {tr.id for tr in tracks if tr.n_seen >= max(5, min_track_frac * n_frames)
            and (first_seen[tr.id] <= START_FRAMES or tr.n_seen >= MIN_LATE_TRACK_FRAC * n_frames)}
    remap = {tid: k for k, tid in enumerate(sorted(keep))}
    color_of = {tr.id: tr.color for tr in tracks}
    rows = []
    last, last_ft = {}, {}
    for (ft, objs), ids in zip(per_frame, assign):
        seen = set()
        for o, tid in zip(objs, ids):
            if tid not in keep or tid in seen:
                continue
            seen.add(tid)
            last[tid], last_ft[tid] = o, ft
            rows.append(_row(ep, ft, ds.fps, remap[tid], color_of[tid], o, frame, "observed", 0))
        for tid, o in last.items():
            if tid not in seen and ft - last_ft[tid] <= MAX_HOLD_FRAMES:
                rows.append(_row(ep, ft, ds.fps, remap[tid], color_of[tid], o, frame, "hold", ft - last_ft[tid]))
    return rows, grip_rows


def _row(ep, ft, fps, oid, color, o, frame: Frame, status, since) -> dict:
    uv = frame.to_uv([o["xy"]])[0]
    r = {
        "episode_index": ep, "frame_index": ft, "timestamp": ft / fps, "object_id": oid,
        "class": "brick", "color": color, "frame_color": o["color"],
        "x_mm": float(o["xy"][0]), "y_mm": float(o["xy"][1]), "u": float(uv[0]), "v": float(uv[1]),
        "z_centre_mm": float(o["z"]), "lifted": o["lifted"], "source": o["source"] if status == "observed" else "hold",
        "confidence": 0.0, "frames_since_seen": since,
    }
    for cam in ("top", "floor"):
        d = o[cam]
        if d is not None and status == "observed":
            b, xy = d
            r[f"{cam}_bbox"] = [float(v) for v in b.bbox]
            r[f"{cam}_point_px"] = [float(v) for v in b.bottom_center]
            r[f"{cam}_x_mm"], r[f"{cam}_y_mm"] = float(xy[0]), float(xy[1])
            r[f"{cam}_area_px"] = int(b.area)
        else:
            r[f"{cam}_bbox"] = None
            r[f"{cam}_point_px"] = None
            r[f"{cam}_x_mm"] = r[f"{cam}_y_mm"] = float("nan")
            r[f"{cam}_area_px"] = 0
    if status == "observed":
        r["confidence"] = 0.9 if o["source"] == "top+floor" else 0.6
    else:
        r["confidence"] = 0.3
    return r

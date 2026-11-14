#!/usr/bin/env python3
"""Build the table coordinate frame for a Studio dataset from the top / floor D415 views.

Writes <out>/table_frame_<dataset>.json and check images in <out>/frame_check/. Read-only on the
dataset. Method and conventions: docs/perception/A-object-localization.md.

  PYTHONPATH=src python scripts/perception_table_frame.py 431a71f7 [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026.perception import cameras
from hackjunction2026.perception.calib import Correspondence, JointCalib, SheetLines
from hackjunction2026.perception.dataset import Dataset
from hackjunction2026.perception.geometry import apply_h, intersect
from hackjunction2026.perception.scene import detect_bricks_cv, exclusion_mask, sheet_roi
from hackjunction2026.perception.table_frame import fit_plane_normal

# Height of the top D415 above the table in mm; every mm value in the table frame scales with it.
# Not measured with a ruler. Three estimates (2026-11-14):
# - nine Duplo edges of known size read by eye (2x2 footprint 31.7 mm, body 19.2 mm, 2x4 front edge
#   63.5 mm; episodes 6, 14, 36, 63): 103-143 mm, mean 126 mm;
# - agent B's camera fit from the arm (FK, metric): 129.6 mm above the table plane;
# - registration of B's FK gripper tip at 55 grasps to our brick positions with h = 126 mm gave a
#   scale of 0.958, i.e. h = 126 / 0.958 = 131.5 mm (scripts/perception_register_robot.py).
# The last one averages 50 grasps and is used. Uncertainty about +-3 %.
TOP_CAMERA_HEIGHT_MM = 131.5
HALF_BRICK_MM = 16.0
CANVAS = {"x_min_mm": -50.0, "x_max_mm": 750.0, "y_min_mm": 0.0, "y_max_mm": 600.0}
W, H = 848, 480
LINE_REF = {"far": (W / 2, H - 1), "left": (W - 1, H / 2), "seam": (W / 2, H - 1)}


def canon(line, ref_point):
    line = np.asarray(line, float)
    return line if line @ [ref_point[0], ref_point[1], 1.0] > 0 else -line


def row_at(l, x):
    return -(l[0] * x + l[2]) / l[1]


def col_at(l, y):
    return -(l[1] * y + l[2]) / l[0]


def run_name(prefix, a, b):
    return f"{prefix}_ep{a:03d}-{b:03d}"


def draw_grid(img, H_table_to_px, canvas, step=100):
    xr, yr = (canvas["x_min_mm"], canvas["x_max_mm"]), (canvas["y_min_mm"], canvas["y_max_mm"])
    for x in [xr[0]] + list(np.arange(0, xr[1] + 1, step)):
        pts = apply_h(H_table_to_px, [(x, y) for y in np.linspace(yr[0], yr[1], 60)])
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    for y in np.arange(yr[0], yr[1] + 1, step):
        pts = apply_h(H_table_to_px, [(x, y) for x in np.linspace(xr[0], xr[1], 60)])
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    o = [tuple(int(v) for v in np.round(p)) for p in apply_h(H_table_to_px, [(0, 0), (100, 0), (0, 100)])]
    cv2.arrowedLine(img, o[0], o[1], (255, 0, 0), 2)
    cv2.arrowedLine(img, o[0], o[2], (0, 200, 0), 2)


def parse_runs(text):
    return [tuple(int(v) for v in r.split("-")) for r in text.split(",")]


BG_PATCHES = [(0, 0, 200, 34)]  # top image: wood edge behind the table, top left (nothing moves there)


def background_shift(ref_rgb, rgb, patches=BG_PATCHES, pad=25):
    """(dx, dy) per patch of the gradient image of `rgb` against `ref_rgb` (template matching)."""
    def grad(img):
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    a, b = grad(ref_rgb), grad(rgb)
    out = []
    for x0, y0, x1, y1 in patches:
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        r = cv2.matchTemplate(b[Y0:y1 + pad, X0:x1 + pad], a[y0:y1, x0:x1], cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(r)
        out += [loc[0] + X0 - x0, loc[1] + Y0 - y0]
    return np.array(out, float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default="/home/intel/dev/captures/perception/A")
    ap.add_argument("--top-runs", help="force top-camera setups, e.g. 0-5,6-66")
    ap.add_argument("--floor-runs", help="force floor-camera setups, e.g. 0-46,47-61,62-66")
    args = ap.parse_args()
    ds = Dataset(args.dataset)
    out = Path(args.out)
    (out / "frame_check").mkdir(parents=True, exist_ok=True)
    Kt, Kf = cameras.K_of("top"), cameras.K_of("floor")
    h_mm = TOP_CAMERA_HEIGHT_MM

    # 1. first frames: sheet lines, distractor masks, bricks
    first, lines, bricks = {}, {}, {}
    for e in ds.episodes:
        ep = e.index
        first[ep] = {cam: ds.frame(ep, cam, 0) for cam in ("top", "floor")}
        lines[ep], bricks[ep] = {}, {}
        for cam in ("top", "floor"):
            roi, lines[ep][cam] = sheet_roi(cam, first[ep][cam])
            bricks[ep][cam] = detect_bricks_cv(first[ep][cam], roi, exclusion_mask(first[ep][cam], roi))
    eps = sorted(first)

    # 2. runs: candidate top runs = changes of the sheet's left edge in the top image (a sheet move
    # or a re-aimed camera); floor runs = changes of the floor view of the near sheet edges
    top_sig = {ep: np.array([col_at(lines[ep]["top"]["left"], y) for y in (50, 150)]) for ep in eps}
    fl = lines
    floor_sig = {ep: np.array([row_at(fl[ep]["floor"]["far"], 450), row_at(fl[ep]["floor"]["far"], 750),
                               col_at(fl[ep]["floor"]["right"], 40) if fl[ep]["floor"]["right"] is not None else 0.0])
                 for ep in eps}
    cand_runs = [r for r in cameras.group_consecutive(top_sig, 3.5)]
    floor_runs = cameras.group_consecutive(floor_sig, 6.0)
    print("candidate top/sheet runs", cand_runs, "floor runs", floor_runs)

    def runs_of(ep, runs, prefix):
        return [run_name(prefix, a, b) for a, b in runs if a <= ep <= b][0]

    def build(top_runs):
        corr = []
        for ep in eps:
            bt, bf = bricks[ep]["top"], bricks[ep]["floor"]
            if len(bt) == 1 and len(bf) == 1:
                corr.append(Correspondence(ep, runs_of(ep, top_runs, "top"), runs_of(ep, floor_runs, "floor"),
                                           np.array(bt[0].bottom_center), np.array(bf[0].bottom_center)))
        sl = []
        for a, b in cand_runs:
            if b - a + 1 < 3:
                continue
            med = {k: np.median([canon(lines[ep]["top"][k], LINE_REF[k]) for ep in range(a, b + 1)], axis=0)
                   for k in ("far", "left", "seam")}
            sl.append(SheetLines(runs_of(a, top_runs, "top"), med["far"], med["left"], med["seam"]))
        return corr, sl

    # top-camera setups: the camera counts as moved when static background above the table (wood,
    # wall) shifts by more than 8 px against the longest sheet run; sheet moves alone do not count
    ref_a, ref_b = max(cand_runs, key=lambda r: r[1] - r[0])
    ref_ep = ref_b
    bg = {ep: background_shift(first[ref_ep]["top"], first[ep]["top"]) for ep in eps}
    top_setups = parse_runs(args.top_runs) if args.top_runs else cameras.group_consecutive(bg, 8.0)
    if args.floor_runs:
        floor_runs = parse_runs(args.floor_runs)
    print("top background shift vs ep", ref_ep, {ep: bg[ep].round(0).tolist() for ep in eps[::5]})
    print("top setups", top_setups, "floor setups", floor_runs)
    ref_span = [r for r in top_setups if r[0] <= ref_ep <= r[1]][0]
    ref_run = run_name("top", *ref_span)

    def top_setup_of(ep):
        return [run_name("top", a, b) for a, b in top_setups if a <= ep <= b][0]

    corr, _ = build(top_setups)
    sl = []
    for a, b in cand_runs:
        if b - a + 1 < 3:
            continue
        med = {k: np.median([canon(lines[ep]["top"][k], LINE_REF[k]) for ep in range(a, b + 1)], axis=0)
               for k in ("far", "left", "seam")}
        sl.append(SheetLines(top_setup_of(a), med["far"], med["left"], med["seam"]))
    n0, _ = fit_plane_normal(Kt, [{"far": x.far, "left": x.left, "seam": x.seam} for x in sl if x.top_run == ref_run])
    jc = JointCalib(Kt, Kf, ref_run, [run_name("top", a, b) for a, b in top_setups],
                    [run_name("floor", a, b) for a, b in floor_runs], corr, sl, HALF_BRICK_MM / h_mm)
    sol = jc.fit(n0)
    n, R, F = jc.unpack(sol.x)
    errs = jc.corr_errors(sol.x)
    print("final top setups", top_setups, "plane normal", n.round(4), "corr px median", np.median(errs).round(1))

    # 3. table axes in the reference top camera
    # use the longest sheet run inside the reference setup for the axes and origin
    a0, b0 = max([r for r in cand_runs if top_setup_of((r[0] + r[1]) // 2) == ref_run], key=lambda r: r[1] - r[0])
    far_ref = np.median([canon(lines[ep]["top"]["far"], LINE_REF["far"]) for ep in range(a0, b0 + 1)], axis=0)
    left_ref = np.median([canon(lines[ep]["top"]["left"], LINE_REF["left"]) for ep in range(a0, b0 + 1)], axis=0)
    e1 = np.cross(n, Kt.T @ far_ref); e1 /= np.linalg.norm(e1)
    if e1[0] < 0:
        e1 = -e1
    e2 = np.cross(n, e1); e2 /= np.linalg.norm(e2)
    if e2[2] > 0:
        e2 = -e2
    c = intersect(far_ref, left_ref)
    r0 = np.linalg.solve(Kt, [c[0], c[1], 1.0])
    O = r0 / (n @ r0) * h_mm  # origin, mm, reference top camera coords
    R_ct = np.c_[e1, e2, np.cross(e1, e2)]  # columns: table axes in reference top-camera coords
    M = np.vstack([h_mm * e1 - (O @ e1) * n, h_mm * e2 - (O @ e2) * n, n])  # ref ray -> table (homog.)

    cams_json, diag_runs = {}, {}
    for a, b in top_setups:
        name = run_name("top", a, b)
        Hp2t = M @ R[name] @ np.linalg.inv(Kt)
        Hp2t /= Hp2t[2, 2]
        Rc = R[name].T @ R_ct  # this setup's camera from table
        tc = R[name].T @ O
        cams_json[name] = {
            "camera": "top", "episodes": [a, b], "K": Kt.tolist(),
            "H_px_to_table_mm": Hp2t.tolist(), "H_table_mm_to_px": (np.linalg.inv(Hp2t) / np.linalg.inv(Hp2t)[2, 2]).tolist(),
            "R_cam_from_table": Rc.tolist(), "t_cam_from_table_mm": tc.tolist(),
            "camera_xy_mm": (R_ct.T @ (-O))[:2].tolist(), "camera_height_mm": h_mm,
            "rotation_vs_reference_deg": float(np.degrees(np.arccos(np.clip((np.trace(R[name]) - 1) / 2, -1, 1)))),
        }
    for a, b in floor_runs:
        name = run_name("floor", a, b)
        Rf, tf = F[name]
        Rc = Rf @ R_ct
        tc = Rf @ O + tf * h_mm
        Ht2p = Kf @ np.c_[Rc[:, 0], Rc[:, 1], tc]
        Ht2p /= Ht2p[2, 2]
        Hp2t = np.linalg.inv(Ht2p); Hp2t /= Hp2t[2, 2]
        C = -Rc.T @ tc
        e_run = [float(e) for e, cc in zip(errs, corr) if cc.floor_run == name]
        cams_json[name] = {
            "camera": "floor", "episodes": [a, b], "K": Kf.tolist(),
            "H_px_to_table_mm": Hp2t.tolist(), "H_table_mm_to_px": Ht2p.tolist(),
            "R_cam_from_table": Rc.tolist(), "t_cam_from_table_mm": tc.tolist(),
            "camera_xy_mm": C[:2].tolist(), "camera_height_mm": float(-C[2]),
        }
        diag_runs[name] = {"n_bricks": len(e_run), "reproj_px_median": float(np.median(e_run)) if e_run else None,
                           "reproj_px_p90": float(np.percentile(e_run, 90)) if e_run else None}
    for a, b in top_setups:
        name = run_name("top", a, b)
        e_run = [float(e) for e, cc in zip(errs, corr) if cc.top_run == name]
        diag_runs[name] = {"n_bricks": len(e_run), "reproj_px_median": float(np.median(e_run)) if e_run else None}

    # correspondence consistency in table mm: brick centre from the top vs from the floor camera
    def ctr(cam_name, px, D=HALF_BRICK_MM):
        cj = cams_json[cam_name]
        p = apply_h(np.array(cj["H_px_to_table_mm"]), [px])[0]
        d = p - np.array(cj["camera_xy_mm"])
        return p + D * d / np.linalg.norm(d)
    dmm = [np.linalg.norm(ctr(c.top_run, c.top_px) - ctr(c.floor_run, c.floor_px)) for c in corr]

    sheet_info = {}
    for s in sl:
        Hp = np.array(cams_json[s.top_run]["H_px_to_table_mm"])
        a, b = jc.line_angles(n, R, s)
        sheet_info[s.top_run + f"/far{row_at(s.far, 424):.0f}"] = {
            "far_left_corner_px": intersect(s.far, s.left).tolist(),
            "far_edge_px": [(x, row_at(s.far, x)) for x in (0.0, 847.0)],
            "left_edge_px": [(col_at(s.left, y), y) for y in (0.0, 479.0)],
            "seam_px": [(x, row_at(s.seam, x)) for x in (0.0, 847.0)],
            "lines_px_abc": {"far": s.far.tolist(), "left": s.left.tolist(), "seam": s.seam.tolist()},
            "far_left_corner_mm": apply_h(Hp, [intersect(s.far, s.left)])[0].tolist(),
            "far_edge_mm": apply_h(Hp, [(x, row_at(s.far, x)) for x in (150, 800)]).tolist(),
            "angle_far_left_deg": a, "angle_far_seam_deg": b,
        }

    frame = {
        "format": "hackjunction2026/table-frame/v1",
        "dataset": ds.id,
        "episodes_seen": [eps[0], eps[-1]],
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "created_by": "scripts/perception_table_frame.py (agent A)",
        "units": "mm; absolute scale from the top-camera height 131.5 mm, checked against the arm FK (+-3 %); u, v normalised to the canvas",
        "convention": {
            "plane": "table / paper surface, z = 0",
            "origin": "far-left corner of the lower paper sheet as the top camera sees it in episodes "
                      f"{a0}-{b0} (far edge x left edge); fixed to the table and robot, not to the paper",
            "x_axis": "along that far sheet edge, left -> right in the top image (the robot stands right of "
                      "the top camera, at large x and large y)",
            "y_axis": "in the table plane, perpendicular to x, from the far edge toward the top camera "
                      "(down in the top image); the floor camera stands beyond the far edge (y < 0)",
            "z_axis": "x cross y = into the table (cameras above the table have z < 0)",
            "canvas_view": "draw x to the right and y downward: the canvas then looks like a rectified top-camera image",
            "u_v": "u = (x - x_min_mm) / (x_max_mm - x_min_mm), v = (y - y_min_mm) / (y_max_mm - y_min_mm)",
            "object_position": "footprint-centre estimate: table point of the image bbox bottom-centre moved "
                               f"{HALF_BRICK_MM} mm (half a 2x2 Duplo) away from that camera",
            "camera_extrinsics": "X_cam = R_cam_from_table @ X_table_mm + t_cam_from_table_mm",
        },
        "canvas": CANVAS,
        "cameras": cams_json,
        "sheet_lines": sheet_info,
        "sheet_note": "No camera sees all four sheet corners (two overlapping white sheets, both larger than the "
                      "views). Per lower-sheet position (sheet_runs): its far-left corner, far edge and left edge, "
                      "and the seam of the upper sheet, in top-image pixels and table mm.",
        "diagnostics": {
            "plane_normal_reference_top_cam": n.tolist(),
            "sheet_runs": cand_runs, "top_background_shift_px": {str(k): v.tolist() for k, v in bg.items()},
            "top_setups": top_setups,
            "floor_runs": floor_runs, "per_run": diag_runs,
            "n_correspondences": len(corr),
            "brick_centre_top_vs_floor_mm": {"median": float(np.median(dmm)), "p90": float(np.percentile(dmm, 90)),
                                             "max": float(np.max(dmm))},
            "optimizer_cost": float(sol.cost),
        },
    }
    path = out / f"table_frame_{ds.id}.json"
    path.write_text(json.dumps(frame, indent=1))
    print(json.dumps(frame["diagnostics"], indent=1))
    print(json.dumps(sheet_info, indent=1))
    print({k: (np.round(v["camera_xy_mm"], 1).tolist(), round(v["camera_height_mm"], 1)) for k, v in cams_json.items()})

    # 4. check images: 100 mm grid on the first frame of each setup
    for name, cj in cams_json.items():
        ep = cj["episodes"][0]
        img = first[ep][cj["camera"]].copy()
        draw_grid(img, np.array(cj["H_table_mm_to_px"]), CANVAS)
        cv2.imwrite(str(out / "frame_check" / f"grid_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # rectified views of one episode per floor setup, 1 px = 1 mm, top | floor
    S = np.array([[1, 0, -CANVAS["x_min_mm"] + 50], [0, 1, -CANVAS["y_min_mm"] + 100], [0, 0, 1.0]])
    size = (int(CANVAS["x_max_mm"] - CANVAS["x_min_mm"] + 100), int(CANVAS["y_max_mm"] - CANVAS["y_min_mm"] + 200))
    for a, b in floor_runs:
        ep = b
        views = []
        for cam in ("top", "floor"):
            nm = [k for k, v in cams_json.items() if v["camera"] == cam and v["episodes"][0] <= ep <= v["episodes"][1]][0]
            views.append(cv2.warpPerspective(first[ep][cam], S @ np.array(cams_json[nm]["H_px_to_table_mm"]), size))
        cv2.imwrite(str(out / "frame_check" / f"rectified_ep{ep:03d}_top_floor.png"),
                    cv2.cvtColor(np.hstack(views), cv2.COLOR_RGB2BGR))
    print("wrote", path)


if __name__ == "__main__":
    main()

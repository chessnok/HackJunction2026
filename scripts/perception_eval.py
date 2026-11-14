#!/usr/bin/env python3
"""Evaluate brick localizers against the hand-annotated ground truth (gt_431a71f7.py).

Methods: cv (HSV colour blobs, profile v2), cv_v1 (first thresholds), vlm (local VLM per camera),
vlm_mv (local VLM, three views in one request), hybrid (VLM objects + CV boxes), gripper. Predictions
are cached in <out>/evaluation/pred_<method>.json, so a VLM run can be resumed. Metrics go to
<out>/evaluation/metrics.json and failure images to <out>/evaluation/fail_<method>/.

  PYTHONPATH=src python scripts/perception_eval.py --methods cv
  PYTHONPATH=src python scripts/perception_eval.py --methods vlm,vlm_mv,hybrid --vlm-url http://localhost:8000/v3
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026.perception.color_cv import detect_blobs
from hackjunction2026.perception.gt_431a71f7 import GT
from hackjunction2026.perception.scene import Frame, detect_bricks_cv, exclusion_mask, sheet_roi
from hackjunction2026.perception.vlm import VLMClient, VLMDet, hybrid_merge, iou

OUT = Path("/home/intel/dev/captures/perception/A")
RESULTS = OUT / "evaluation"
EVAL = OUT / "eval_frames"
DATASET = "431a71f7-50c6-4e38-9e38-d85984dde78b"
CAMS = ("top", "floor", "arm")


def load(ep, f, cam):
    return cv2.cvtColor(cv2.imread(str(EVAL / f"ep{ep:03d}_f{f:04d}_{cam}.png")), cv2.COLOR_BGR2RGB)


def frames():
    return [(int(r["episode"]), int(r["frame"]), r["phase"]) for r in csv.DictReader(open(EVAL / "frames.csv"))]


def arm_roi(rgb):
    """Wrist camera: inside the convex hull of the white paper."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] < 50) & (hsv[..., 2] > 120)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones(m.shape, bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(m)
    cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
    return hull.astype(bool)


class CVMethod:
    def __init__(self, profile="v2"):
        self.ctx = {}
        self.profile = profile

    def context(self, ep, cam):
        if (ep, cam) not in self.ctx:
            f0 = load(ep, 0, cam)
            if cam == "arm":
                self.ctx[(ep, cam)] = (None, None)
            else:
                roi, _ = sheet_roi(cam, f0)
                self.ctx[(ep, cam)] = (roi, exclusion_mask(f0, roi))
        return self.ctx[(ep, cam)]

    def __call__(self, ep, f, cam, rgb):
        roi, ex = self.context(ep, cam)
        if cam == "arm":
            r = arm_roi(rgb)
            blobs = [b for b in detect_blobs(rgb, r, min_area=400, profile=self.profile)]
            ex = exclusion_mask(rgb)
            h, w = rgb.shape[:2]
            blobs = [b for b in blobs if not ex[min(h - 1, int(b.bottom_center[1])), min(w - 1, int(b.bottom_center[0]))]]
            return blobs
        return detect_bricks_cv(rgb, roi, ex, profile=self.profile)


def as_dict(d):
    return {"bbox": [float(v) for v in d.bbox], "colors": list(d.colors)}


def run_method(method, client, cvm, cache_path, src_path=None):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for ep, f, phase in frames():
        key = f"{ep}:{f}"
        if key in cache:
            continue
        imgs = {c: load(ep, f, c) for c in CAMS}
        rec = {"time_s": {}}
        if method in ("cv", "cv_v1"):
            for c in CAMS:
                t = time.perf_counter()
                dets = cvm(ep, f, c, imgs[c])
                rec["time_s"][c] = time.perf_counter() - t
                rec[c] = [as_dict(d) for d in dets]
        elif method in ("vlm", "hybrid"):
            src = json.loads(src_path.read_text()) if method == "hybrid" else None
            for c in CAMS:
                if method == "vlm":
                    dets = client.detect(imgs[c], c)
                    rec["time_s"][c] = client.last_latency
                    rec.setdefault("usage", {})[c] = client.last_usage
                    rec.setdefault("raw", {})[c] = client.last_raw
                else:
                    v = [VLMDet(tuple(d["bbox"]), d["colors"]) for d in src[key][c]]
                    t = time.perf_counter()
                    dets = hybrid_merge(v, cvm(ep, f, c, imgs[c]))
                    rec["time_s"][c] = src[key]["time_s"][c] + time.perf_counter() - t
                rec[c] = [as_dict(d) for d in dets]
        elif method == "vlm_mv":
            res = client.detect_multi(imgs)
            rec["time_s"]["all"] = client.last_latency
            rec["usage"] = client.last_usage
            rec["raw"] = client.last_raw
            for c in CAMS:
                rec[c] = [as_dict(d) for d in res[c]]
        cache[key] = rec
        cache_path.write_text(json.dumps(cache))
        print(method, key, phase, {c: [(np.round(d["bbox"]).astype(int).tolist(), "+".join(d["colors"])) for d in rec[c]] for c in CAMS},
              {k: round(v, 2) for k, v in rec["time_s"].items()}, flush=True)
    return cache


def colorset(c):
    return frozenset(c.split("+") if isinstance(c, str) else c)


def bottom_center(b):
    return np.array([(b[0] + b[2]) / 2, b[3]])


def match(gt_box, preds):
    best, best_s = None, 0.0
    for k, p in enumerate(preds):
        s = iou(gt_box, p["bbox"])
        cx, cy = (p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2
        inside = gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]
        if (s >= 0.3 or inside) and s + inside >= best_s:
            best, best_s = k, s + inside
    return best


def evaluate(method, cache, frame: Frame):
    per_cam = {c: {"n": 0, "det": 0, "cls_exact": 0, "cls_any": 0, "fp": 0, "px_bc": [], "px_c": [], "mm": [],
                   "iou": [], "by_occ": {}} for c in ("top", "floor")}
    arm = {"n": 0, "det": 0, "cls_exact": 0}
    fused, fails = [], []
    times = []
    for (ep, f), g in sorted(GT.items()):
        key = f"{ep}:{f}"
        rec = cache[key]
        times.append(sum(rec["time_s"].values()))
        lifted = g.get("lifted", False)
        est, gt_xy = {}, {}
        for c in ("top", "floor"):
            if c not in g:
                continue
            color, box, occ = g[c]
            S = per_cam[c]
            S["n"] += 1
            o = S["by_occ"].setdefault(occ, {"n": 0, "det": 0})
            o["n"] += 1
            preds = rec[c]
            k = match(box, preds)
            S["fp"] += len(preds) - (k is not None)
            gbc = bottom_center(box)
            gt_xy[c] = frame.footprint_center(c, ep, gbc)
            if k is None:
                fails.append((ep, f, c, "missed"))
                continue
            p = preds[k]
            S["det"] += 1; o["det"] += 1
            S["iou"].append(iou(box, p["bbox"]))
            pc, gc = colorset(p["colors"]), colorset(color)
            S["cls_exact"] += pc == gc
            S["cls_any"] += bool(pc & gc)
            if pc != gc:
                fails.append((ep, f, c, f"colour {'+'.join(sorted(pc))} vs {color}"))
            pbc = bottom_center(p["bbox"])
            S["px_bc"].append(float(np.linalg.norm(pbc - gbc)))
            S["px_c"].append(float(np.linalg.norm(np.array([(p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2])
                                                  - np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]))))
            est[c] = frame.footprint_center(c, ep, pbc)
            if not lifted:
                S["mm"].append(float(np.linalg.norm(est[c] - gt_xy[c])))
            if S["px_bc"][-1] > 15:
                fails.append((ep, f, c, f"bottom-centre off by {S['px_bc'][-1]:.0f} px"))
        if "arm" in rec:
            arm["n"] += 1
            gc = colorset(g.get("top", g.get("floor"))[0])
            hit = [p for p in rec["arm"] if colorset(p["colors"]) & gc]
            arm["det"] += bool(hit)
            arm["cls_exact"] += any(colorset(p["colors"]) == gc for p in rec["arm"])
        if not lifted and gt_xy:
            g_ref = np.mean(list(gt_xy.values()), axis=0)
            if est:
                fused.append({"ep": ep, "f": f, "err_mm": float(np.linalg.norm(np.mean(list(est.values()), axis=0) - g_ref)),
                              "n_views": len(est)})
    def stats(v):
        return {"median": float(np.median(v)) if v else None, "p90": float(np.percentile(v, 90)) if v else None,
                "n": len(v)}
    out = {"cams": {}, "arm": arm, "fused_mm": stats([x["err_mm"] for x in fused]),
           "fused_detected_frames": len(fused), "time_s_per_frame_all_cams": stats(times)}
    for c, S in per_cam.items():
        out["cams"][c] = {
            "n_gt": S["n"], "detection_rate": S["det"] / S["n"], "class_exact": S["cls_exact"] / max(1, S["det"]),
            "class_any": S["cls_any"] / max(1, S["det"]), "false_pos_per_frame": S["fp"] / S["n"],
            "px_bottom_centre": stats(S["px_bc"]), "px_centre": stats(S["px_c"]), "iou": stats(S["iou"]),
            "table_mm": stats(S["mm"]),
            "detection_by_occlusion": {str(k): v["det"] / v["n"] for k, v in sorted(S["by_occ"].items())},
        }
    out["failures"] = fails
    return out


def draw_failures(method, cache, fails, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for ep, f, c, why in fails[:40]:
        img = cv2.cvtColor(load(ep, f, c), cv2.COLOR_RGB2BGR)
        g = GT[(ep, f)][c]
        x0, y0, x1, y1 = map(int, g[1])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
        for p in cache[f"{ep}:{f}"][c]:
            a, b, cc, d = map(int, p["bbox"])
            cv2.rectangle(img, (a, b), (cc, d), (255, 0, 255), 2)
            cv2.putText(img, "+".join(p["colors"]), (a, max(12, b - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(img, f"{method} ep{ep} f{f} {c}: {why}", (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(dest / f"ep{ep:03d}_f{f:04d}_{c}.png"), img)


def eval_gripper():
    """Gripper detector check. Late frames: the jaws hold the brick, so the detected tip must lie in
    the ground-truth brick box grown by 30 px. Start frames: the arm is at home, out of both views,
    so any detection is a false positive. Mid frames are not scored (the arm may or may not be in view)."""
    from hackjunction2026.perception.gripper import detect_gripper, static_robot_mask

    res = {}
    for c in ("top", "floor"):
        late_ok = late_n = start_fp = start_n = 0
        times, misses = [], []
        for (ep, f), g in sorted(GT.items()):
            phase = {p[1]: p[2] for p in frames() if p[0] == ep}[f]
            if phase == "mid" or c not in g:
                continue
            st = static_robot_mask(load(ep, 0, c))
            img = load(ep, f, c)
            t = time.perf_counter()
            d = detect_gripper(img, st)
            times.append(time.perf_counter() - t)
            if phase == "start":
                start_n += 1
                start_fp += d is not None
            else:
                late_n += 1
                x0, y0, x1, y1 = g[c][1]
                ok = d is not None and x0 - 30 <= d.tip[0] <= x1 + 30 and y0 - 30 <= d.tip[1] <= y1 + 30
                late_ok += ok
                if not ok:
                    misses.append((ep, f))
        res[c] = {"late_tip_at_brick": late_ok / late_n, "n_late": late_n, "start_false_pos": start_fp / start_n,
                  "n_start": start_n, "late_misses": misses, "ms_per_frame_median": 1000 * float(np.median(times))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="cv")
    ap.add_argument("--vlm-url", default="http://localhost:8000/v3")
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--coord-mode", default="rel1000")
    ap.add_argument("--tag", default="", help="suffix for the VLM cache files (e.g. the model name)")
    args = ap.parse_args()
    frame = Frame.load(OUT / f"table_frame_{DATASET}.json")
    ev = RESULTS
    ev.mkdir(parents=True, exist_ok=True)
    metrics_path = ev / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    client = None
    for m in args.methods.split(","):
        if m == "gripper":
            metrics["gripper_cv"] = eval_gripper()
            print(json.dumps(metrics["gripper_cv"], indent=1))
            continue
        if not m.startswith("cv") and client is None:
            client = VLMClient(args.vlm_url, args.vlm_model, coord_mode=args.coord_mode)
            print("VLM model:", client.model)
        name = m if m.startswith("cv") else f"{m}{args.tag}"
        cache_path = ev / f"pred_{name}.json"
        src = ev / f"pred_vlm{args.tag}.json"  # hybrid reads the per-camera VLM predictions of the same tag
        if m == "hybrid" and not src.exists():
            raise SystemExit(f"run vlm{args.tag} first")
        cache = run_method(m, client, CVMethod("v1" if m == "cv_v1" else "v2"), cache_path, src)
        res = evaluate(name, cache, frame)
        if client is not None and not m.startswith("cv"):
            res["model"] = client.model
        metrics[name] = res
        draw_failures(name, cache, res["failures"], ev / f"fail_{name}")
        print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=1))
        print("failures:", res["failures"])
    metrics_path.write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()

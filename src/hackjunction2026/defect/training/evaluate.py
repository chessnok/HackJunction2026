"""Evaluate a trained run with its EXPORTED OpenVINO model (the file the robot will use).

    uv run --extra anomalib python -m hackjunction2026.defect.training.evaluate            # latest run
    uv run --extra anomalib python -m hackjunction2026.defect.training.evaluate --run <run_id> --device NPU

Evaluates EVERY exported file (FP16 and INT8): quantization can break a model (on the mock data,
PaDiM INT8 returned the same score for every image), so never trust an export without this.
Uses the dataset's test images, never seen in training:
  - per view (top_pre, side_FR, ...): AUROC and error rates at the threshold, to find weak views
  - per cycle, as the robot decides: a brick is defective if max(score over its views) > threshold
  - a threshold sweep: the lowest threshold that keeps wrongly rejected good bricks <= --max-fpr
  - latency per image on --device
  - error images: heatmaps of the worst false alarms and misses -> <run>/errors/<compression>/
  - a "broken" flag when the scores hardly vary (constant output)
Writes <run>/eval.json: {compression: result}.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026.defect.training.common import RUNS_DIR


def auroc(neg: list[float], pos: list[float]) -> float | None:
    """Probability that a defect scores higher than a good image (ties count half)."""
    if not neg or not pos:
        return None
    n, p = np.asarray(neg), np.asarray(pos)
    wins = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
    return float(wins / (len(n) * len(p)))


def rates(neg: list[float], pos: list[float], thr: float) -> dict:
    fpr = float(np.mean(np.asarray(neg) > thr)) if neg else None   # good wrongly rejected
    fnr = float(np.mean(np.asarray(pos) <= thr)) if pos else None  # defects missed
    return {"fpr": fpr, "fnr": fnr}


def overlay(img_bgr: np.ndarray, amap: np.ndarray | None, text: str) -> np.ndarray:
    out = img_bgr.copy()
    if amap is not None:
        a = cv2.resize(np.squeeze(amap).astype(np.float32), out.shape[1::-1])
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out = cv2.addWeighted(out, 0.6, cv2.applyColorMap(a, cv2.COLORMAP_JET), 0.4, 0)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def latest_run() -> Path:
    runs = sorted(p for p in RUNS_DIR.glob("*") if (p / "run.json").exists())
    if not runs:
        msg = f"no runs in {RUNS_DIR}"
        raise SystemExit(msg)
    return runs[-1]


def evaluate_run(run_dir: Path, device: str = "CPU", compression: str | None = None,
                 threshold: float = 0.5, max_fpr: float = 0.05, n_errors: int = 8) -> dict:
    """Evaluate all successful exports of a run (or only `compression`); writes eval.json."""
    run = json.loads((run_dir / "run.json").read_text())
    exports = {k: v for k, v in run["exports"].items() if "xml" in v and (compression in (None, k))}
    if not exports:
        msg = f"{run_dir}: no successful export to evaluate"
        raise SystemExit(msg)
    root = Path(run["dataset"]["path"])
    with open(root / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if not rows:
        msg = f"{root}: no test images"
        raise SystemExit(msg)
    old = json.loads((run_dir / "eval.json").read_text()) if (run_dir / "eval.json").exists() else {}
    results = {**old, **{comp: evaluate_export(run, run_dir, root, rows, comp, e["xml"], device, threshold,
                                               max_fpr, n_errors) for comp, e in exports.items()}}
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    return results


def evaluate_export(run: dict, run_dir: Path, root: Path, rows: list[dict], comp: str, xml: str, device: str,
                    threshold: float, max_fpr: float, n_errors: int) -> dict:
    from anomalib.deploy import OpenVINOInferencer

    t0 = time.time()
    inf = OpenVINOInferencer(path=xml, device=device)
    compile_s = time.time() - t0
    lat, scored = [], []
    for r in rows:
        img = cv2.imread(str(root / r["path"]))
        t = time.perf_counter()
        pred = inf.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lat.append((time.perf_counter() - t) * 1000)
        score = float(np.ravel(pred.pred_score)[0])
        amap = getattr(pred, "anomaly_map", None)
        scored.append({**r, "score": score, "_img": img, "_amap": None if amap is None else np.asarray(amap)})

    # per view: good vs defect images (defect_clean = other views of a defective brick, not used here)
    per_view = {}
    for view in sorted({r["view"] for r in scored}):
        neg = [r["score"] for r in scored if r["view"] == view and r["cls"] == "good"]
        pos = [r["score"] for r in scored if r["view"] == view and r["cls"] == "defect"]
        per_view[view] = {"n_good": len(neg), "n_defect": len(pos), "auroc": auroc(neg, pos),
                          **rates(neg, pos, threshold)}
    img_neg = [r["score"] for r in scored if r["cls"] == "good"]
    img_pos = [r["score"] for r in scored if r["cls"] == "defect"]

    # per cycle: the robot's decision = max over all views of that placement
    cycles: dict[tuple, dict] = defaultdict(lambda: {"score": 0.0, "label": "good"})
    for r in scored:
        c = cycles[(r["session"], r["label"], r["brick"], r["cycle"])]
        c["score"], c["label"] = max(c["score"], r["score"]), r["label"]
    cyc_neg = [c["score"] for c in cycles.values() if c["label"] == "good"]
    cyc_pos = [c["score"] for c in cycles.values() if c["label"] == "defect"]
    sweep = []
    for thr in np.round(np.arange(0.05, 1.0, 0.05), 2):
        sweep.append({"threshold": float(thr), **rates(cyc_neg, cyc_pos, float(thr))})
    ok = [s for s in sweep if s["fpr"] is not None and s["fpr"] <= max_fpr]
    recommended = ok[0] if ok else None

    # error images: worst false alarms (good, high score) and misses (defect, low score)
    err_dir = run_dir / "errors" / comp
    err_dir.mkdir(parents=True, exist_ok=True)
    for f in err_dir.glob("*.png"):
        f.unlink()
    goods = sorted((r for r in scored if r["cls"] == "good"), key=lambda r: -r["score"])[:n_errors]
    misses = sorted((r for r in scored if r["cls"] == "defect"), key=lambda r: r["score"])[:n_errors]
    for kind, items in (("false_alarm", goods), ("miss", misses)):
        for i, r in enumerate(items):
            name = f"{kind}_{i:02d}_{r['score']:.2f}_{Path(r['path']).stem}.png"
            cv2.imwrite(str(err_dir / name), overlay(r["_img"], r["_amap"], f"{r['score']:.2f} {r['view']}"))

    all_scores = [r["score"] for r in scored]
    broken = float(np.std(all_scores)) < 1e-4
    result = {
        "run_id": run["run_id"], "model": xml, "compression": comp, "device": device, "threshold": threshold,
        "broken": broken, "score_std": round(float(np.std(all_scores)), 6),
        "real_defects": bool(img_pos), "compile_seconds": round(compile_s, 2),
        "latency_ms": {"median": round(float(np.median(lat)), 2), "p95": round(float(np.percentile(lat, 95)), 2),
                       "n": len(lat)},
        "images": {"n_good": len(img_neg), "n_defect": len(img_pos), "auroc": auroc(img_neg, img_pos),
                   **rates(img_neg, img_pos, threshold)},
        "cycles": {"n_good": len(cyc_neg), "n_defect": len(cyc_pos), "auroc": auroc(cyc_neg, cyc_pos),
                   **rates(cyc_neg, cyc_pos, threshold)},
        "per_view": per_view, "max_fpr": max_fpr, "recommended": recommended, "sweep": sweep,
        "score_range_good": [round(min(img_neg), 3), round(max(img_neg), 3)] if img_neg else None,
    }
    c = result["cycles"]
    print(f"[eval] {run['run_id']} ({comp}, {device}): cycles good={c['n_good']} defect={c['n_defect']} "
          f"FPR={c['fpr']} FNR={c['fnr']} AUROC={c['auroc']}; median {result['latency_ms']['median']} ms; "
          f"recommended threshold {recommended['threshold'] if recommended else '-'}")
    if broken:
        print(f"[eval] WARNING {comp}: scores are constant ({all_scores[0]:.3f}) -> this export is BROKEN, "
              "do not deploy it")
    if not img_pos:
        print("[eval] no real defect images in the test set: only false alarms on good bricks are measured")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help=f"run id under {RUNS_DIR} (default: latest)")
    p.add_argument("--device", default="CPU", help="OpenVINO device: CPU, GPU, NPU")
    p.add_argument("--compression", choices=("fp16", "int8_ptq"), help="default: every successful export")
    p.add_argument("--threshold", type=float, default=0.5, help="0.5 = the threshold learned in training")
    p.add_argument("--max-fpr", type=float, default=0.05, help="target share of wrongly rejected good bricks")
    a = p.parse_args(argv)
    run_dir = RUNS_DIR / a.run if a.run else latest_run()
    evaluate_run(run_dir, a.device, a.compression, a.threshold, a.max_fpr)


if __name__ == "__main__":
    main()

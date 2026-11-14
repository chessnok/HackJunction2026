"""Evaluation table for brick-crop runs (datasets from cropping.brick_dataset), per model and device.

    .venv/bin/python -m hackjunction2026.defect.training.brick_report --runs <run_id> [<run_id> ...]

Reads the per-image scores written by training.evaluate (scores_<comp>_<device>.csv for the real
test set, scores_<name>-synth_<comp>_<device>.csv for the SYNTHETIC set) and reports:

- real defects = state snapshots flagged defective. Per crop: crops where the mark is visible
  (test/defect, labelled by Gemini). Per brick: OR over all crops of the snapshot (ADR 0008).
- false alarms on good bricks, separately for held-out dataset episodes and for good snapshots.
- synthetic strokes (secondary, clearly labelled): per crop and per brick.
Thresholds on the normalized score (0.5 = Anomalib's own, learned on half of the test set):
  anomalib      0.5
  ep_p99        99th percentile of held-out EPISODE good crops (uses no defect and no snapshot)
  snap_b5       lowest threshold with <= 5 % false alarms per good-snapshot brick. Uses good bricks
                only, so the recall is honest; its own false-alarm rate is optimistic, see cv_snap_b5
                (threshold from one half of the good snapshots, false alarms on the other half).
Writes outputs/anomalib/brick_report.md + .json and failure contact sheets in captures/anomalib/.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026.defect.training.common import DATASETS_DIR, RUNS_DIR
from hackjunction2026.defect.training.evaluate import auroc


DEVICE_ORDER = ("CPU", "NPU", "GPU")


def load(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = float(r["score"])
        r["latency_ms"] = float(r["latency_ms"])
    return rows


def bricks(rows: list[dict]) -> dict[tuple, dict]:
    """(session, label, brick, cycle) -> {"score": max, "label", "source", "visible": any crop is cls defect}."""
    out: dict[tuple, dict] = {}
    for r in rows:
        k = (r["session"], r["label"], r["brick"], r["cycle"])
        b = out.setdefault(k, {"score": -1.0, "label": r["label"], "snap": r["source"].startswith("snapshot:"),
                               "visible": False})
        b["score"] = max(b["score"], r["score"])
        b["visible"] |= r["cls"] == "defect"
    return out


def rate(xs: list[float], thr: float, above: bool = True) -> float | None:
    if not xs:
        return None
    a = np.asarray(xs)
    return float(np.mean(a > thr) if above else np.mean(a <= thr))


def metrics(real: list[dict], synth: list[dict] | None) -> dict:
    good_ep = [r["score"] for r in real if r["cls"] == "good" and r["source"].startswith("dataset:")]
    good_sn = [r["score"] for r in real if r["cls"] == "good" and r["source"].startswith("snapshot:")]
    mark = [r["score"] for r in real if r["cls"] == "defect"]
    br = bricks(real)
    b_good_ep = [b["score"] for b in br.values() if b["label"] == "good" and not b["snap"]]
    b_good_sn = [b["score"] for b in br.values() if b["label"] == "good" and b["snap"]]
    b_def = [b["score"] for b in br.values() if b["label"] == "defect"]
    b_def_vis = [b["score"] for b in br.values() if b["label"] == "defect" and b["visible"]]
    s_good = s_mark = sb_good = sb_def = []
    if synth:
        s_good = [r["score"] for r in synth if r["cls"] == "good"]
        s_mark = [r["score"] for r in synth if r["cls"] == "defect"]
        sbr = bricks(synth)
        sb_good = [b["score"] for b in sbr.values() if b["label"] == "good"]
        sb_def = [b["score"] for b in sbr.values() if b["label"] == "defect"]

    def t_for(scores: list[float], max_fpr: float) -> float:
        cand = sorted(set(scores))
        ok = [t for t in cand if rate(scores, t) <= max_fpr]
        return float(ok[0]) if ok else float(max(cand))

    thresholds = {"anomalib": 0.5}
    if good_ep:
        thresholds["ep_p99"] = float(np.percentile(good_ep, 99))
    cv = None
    if b_good_sn:
        thresholds["snap_b5"] = t_for(b_good_sn, 0.05)
        # 2-fold by time block of good snapshots: threshold from one half, false alarms on the other.
        ids = sorted(k for k, b in br.items() if b["label"] == "good" and b["snap"])
        halves = [[br[k]["score"] for i, k in enumerate(ids) if (i // 6) % 2 == f] for f in (0, 1)]
        if all(halves):
            folds = []
            for a_, b_ in ((0, 1), (1, 0)):
                t = t_for(halves[a_], 0.05)
                folds.append({"threshold": t, "fpr_brick_snapshots": rate(halves[b_], t),
                              "recall_brick_real": rate(b_def, t), "recall_crop_real": rate(mark, t)})
            cv = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    # Per-crop recall at fixed false-positive rates, threshold set on one kind of good crop at a time.
    at_fpr = {}
    for neg_name, neg in (("snapshot_good_crops", good_sn), ("episode_good_crops", good_ep)):
        if not neg:
            continue
        for f in (0.0, 0.05, 0.10):
            t = t_for(neg, f)
            at_fpr[f"{neg_name}@{int(round(100 * f))}%"] = {
                "threshold": round(t, 4), "fpr": rate(neg, t), "recall_crop_real": rate(mark, t),
                "recall_brick_real": rate(b_def, t), "recall_crop_synth": rate(s_mark, t),
                "fpr_brick_snapshots": rate(b_good_sn, t)}
    flagged = sorted(({"crop": r["path"], "score": round(r["score"], 4), "by": r.get("gemini_model", "")}
                      for r in real if r["label"] == "good" and r.get("gemini_mark_visible") == "true"),
                     key=lambda d: -d["score"])
    at = {}
    for name, t in thresholds.items():
        at[name] = {
            "threshold": round(t, 4),
            "fpr_crop_episodes": rate(good_ep, t), "fpr_crop_snapshots": rate(good_sn, t),
            "fpr_brick_episodes": rate(b_good_ep, t), "fpr_brick_snapshots": rate(b_good_sn, t),
            "recall_crop_real": rate(mark, t), "recall_brick_real": rate(b_def, t),
            "recall_brick_real_visible": rate(b_def_vis, t),
            "recall_crop_synth": rate(s_mark, t), "recall_brick_synth": rate(sb_def, t),
            "fpr_brick_synth_set": rate(sb_good, t),
        }
    lat = [r["latency_ms"] for r in real]
    return {
        "n": {"good_crops_episodes": len(good_ep), "good_crops_snapshots": len(good_sn), "mark_crops": len(mark),
              "good_bricks_episodes": len(b_good_ep), "good_bricks_snapshots": len(b_good_sn),
              "defective_bricks": len(b_def), "defective_bricks_mark_visible": len(b_def_vis),
              "synth_good_crops": len(s_good), "synth_mark_crops": len(s_mark), "synth_defective_bricks": len(sb_def)},
        "auroc": {
            "crop_real_vs_all_good": auroc(good_ep + good_sn, mark), "crop_real_vs_snapshot_good": auroc(good_sn, mark),
            "brick_real_vs_snapshot_good": auroc(b_good_sn, b_def),
            "brick_real_visible_vs_snapshot_good": auroc(b_good_sn, b_def_vis),
            "crop_synth": auroc(s_good, s_mark), "brick_synth": auroc(sb_good, sb_def),
        },
        "at": at, "cv_snap_b5": cv, "at_fpr": at_fpr, "good_crops_flagged_by_gemini": flagged,
        "latency_ms": {"median": round(float(np.median(lat)), 2), "p95": round(float(np.percentile(lat, 95)), 2)}
        if lat else None,
        "score_good_snapshots": [round(float(np.percentile(good_sn, q)), 3) for q in (50, 90, 99)] if good_sn else None,
        "score_good_episodes": [round(float(np.percentile(good_ep, q)), 3) for q in (50, 90, 99)] if good_ep else None,
    }


def contact_sheet(rows: list[dict], root: Path, thr: float, out: Path, n: int = 12, size: int = 160) -> None:
    """Top row: worst false alarms on good snapshot crops. Bottom row: worst misses on mark-visible crops."""
    fa = sorted((r for r in rows if r["cls"] == "good"), key=lambda r: -r["score"])[:n]
    miss = sorted((r for r in rows if r["cls"] == "defect"), key=lambda r: r["score"])[:n]
    lines = []
    for title, items in (("false alarms (good)", fa), ("misses (mark visible)", miss)):
        tiles = []
        for r in items:
            img = cv2.imread(str(root / r["path"]))
            img = cv2.resize(img, (size, size)) if img is not None else np.zeros((size, size, 3), np.uint8)
            colour = (0, 0, 255) if (r["score"] > thr) == (r["cls"] == "good") else (0, 160, 0)
            cv2.putText(img, f"{r['score']:.2f} {r['view']}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
            cv2.putText(img, r["brick"][-12:], (3, size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            tiles.append(img)
        while len(tiles) < n:
            tiles.append(np.full((size, size, 3), 255, np.uint8))
        head = np.full((22, size * n, 3), 255, np.uint8)
        cv2.putText(head, f"{title}, threshold {thr:.3f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        lines += [head, np.hstack(tiles)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(lines))


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def num(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--threshold-rule", default="snap_b5", help="which threshold the table and sheets use")
    p.add_argument("--out", default="outputs/anomalib/brick_report")
    a = p.parse_args(argv)

    results, lines, scored = [], [], {}
    hdr = ("| run | model | comp | device | AUROC crop real | AUROC brick real | AUROC crop synth | AUROC brick synth | "
           "thr | FA crop ep | FA crop snap | FA brick snap | FA brick snap 2-fold | recall crop real | "
           "recall brick real | recall crop synth | recall brick synth | latency ms |")
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for run_id in a.runs:
        run_dir = RUNS_DIR / run_id
        run = json.loads((run_dir / "run.json").read_text())
        name = run["dataset"]["name"]
        for f in sorted(run_dir.glob("scores_*.csv")):
            if "-synth_" in f.stem:
                continue
            parts = f.stem.split("_")  # scores_<comp>_<device>, comp may be int8_ptq
            comp, device = "_".join(parts[1:-1]), parts[-1]
            synth_f = run_dir / f"scores_{name}-synth_{comp}_{device}.csv"
            real = load(f)
            synth = load(synth_f) if synth_f.exists() else None
            m = metrics(real, synth)
            m.update(run_id=run_id, model=run["model_class"], compression=comp, device=device, dataset=name)
            results.append(m)
            t = m["at"].get(a.threshold_rule) or m["at"]["anomalib"]
            au = m["auroc"]
            lines.append("| " + " | ".join([
                run_id, m["model"], comp, device, num(au["crop_real_vs_all_good"]), num(au["brick_real_vs_snapshot_good"]),
                num(au["crop_synth"]), num(au["brick_synth"]), f"{t['threshold']:.3f}",
                pct(t["fpr_crop_episodes"]), pct(t["fpr_crop_snapshots"]), pct(t["fpr_brick_snapshots"]),
                pct(m["cv_snap_b5"]["fpr_brick_snapshots"]) if m["cv_snap_b5"] and a.threshold_rule == "snap_b5" else "-",
                pct(t["recall_crop_real"]), pct(t["recall_brick_real"]), pct(t["recall_crop_synth"]),
                pct(t["recall_brick_synth"]), str(m["latency_ms"]["median"]) if m["latency_ms"] else "-"]) + " |")
            scored[(run_id, comp, device)] = (real, name, t["threshold"])
    table = "\n".join(lines)

    # One device per run + compression for the per-crop table and the sheets: CPU, else NPU, else GPU
    # (scores agree across devices to about the third decimal).
    primary = {}
    for run_id, comp, device in scored:
        best = primary.get((run_id, comp))
        if best is None or DEVICE_ORDER.index(device) < DEVICE_ORDER.index(best):
            primary[(run_id, comp)] = device
    for (run_id, comp), device in primary.items():
        real, name, thr = scored[(run_id, comp, device)]
        contact_sheet(real, DATASETS_DIR / name, thr, Path("captures/anomalib") / f"failures_{run_id}_{comp}.png")

    # Per-crop recall at fixed false-positive rates.
    fl = ["| run | comp | device | threshold set on | FPR target | threshold | crop FPR | recall crop real | "
          "recall brick real | brick FPR snap | recall crop synth |", "|" + "---|" * 11]
    for m in results:
        if primary.get((m["run_id"], m["compression"])) != m["device"]:
            continue
        for k, v in m["at_fpr"].items():
            neg, target = k.split("@")
            fl.append("| " + " | ".join([m["run_id"], m["compression"], m["device"], neg, target, f"{v['threshold']:.3f}",
                                         pct(v["fpr"]), pct(v["recall_crop_real"]), pct(v["recall_brick_real"]),
                                         pct(v["fpr_brick_snapshots"]), pct(v["recall_crop_synth"])]) + " |")
    flags = [f"- {m['run_id']} {m['compression']}: {f['crop']} score {f['score']} (flagged by {f['by']})"
             for m in results if primary.get((m["run_id"], m["compression"])) == m["device"]
             for f in m["good_crops_flagged_by_gemini"]]
    extra = "\n".join(fl) + "\n\nGood-brick test crops flagged as marked by the label model:\n" + ("\n".join(flags) or "- none")
    print(table + "\n\n" + extra)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(f"Threshold rule: {a.threshold_rule}\n\n{table}\n\n{extra}\n")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {out}.md / .json; failure sheets in captures/anomalib/failures_*.png")


if __name__ == "__main__":
    main()

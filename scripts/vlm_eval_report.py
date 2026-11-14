#!/usr/bin/env python3
"""Report for scripts/vlm_eval.sh results (brick_judge.py JSONL files). Stdlib only (ADR 0005).

  uv run --no-project python scripts/vlm_eval_report.py captures/vlm/eval/<tag>/*.jsonl

Prints a table (FP on real good snapshots, real defective found, SYNTHETIC found, latency); for
per-image (OR) runs the per-crop scores against the crop labels (Gemini labels in the Anomalib index,
overridden by configs/defect_mark_labels.csv: true / false / exclude) and the brick-level union with
an Anomalib PatchCore run; for locate-vlm runs the IoU of the VLM boxes against the colour boxes.
Brick truth is the current `defective` flag in Studio's snapshots.csv. Scenes ending in -synth are
the synthetic marked bricks.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

SNAPCSV = Path.home() / ".local/share/physicalai/state_snapshots/0fd76ff2-a6b1-4472-b687-486befe329be/snapshots.csv"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def crop_labels(index: Path, manual: Path, truth: dict) -> dict:
    """(snapshot, view) -> (brick label, mark visible); manual overrides Gemini, 'exclude' drops the crop."""
    over = {r["crop_id"]: r["mark_visible"].strip() for r in csv.DictReader(manual.open())} if manual.exists() else {}
    labels = {}
    for r in csv.DictReader(index.open()):
        if not r["scene"].startswith("snap/"):
            continue
        snap = r["scene"].split("/", 1)[1]
        m = over.get(r["crop_id"], "")
        if m == "exclude":
            continue
        mark = m == "true" if m in ("true", "false") else {"true": True, "false": False}.get(r["gemini_mark_visible"])
        labels[(snap, r["view"])] = (truth.get(snap, r["label"]), mark)
    return labels


def per_crop(rows: list[dict], labels: dict) -> None:
    stats = collections.defaultdict(collections.Counter)
    for r in rows:
        for name, vote in r["per_image"].items():
            view = name.split(".")[0]
            lab = labels.get((r["scene"], view))
            if lab is None:
                stats[view]["unlabelled"] += 1
            elif lab[0] == "defect" and lab[1]:
                stats[view]["m_n"] += 1
                stats[view]["m_hit"] += vote
            elif lab[0] == "defect":
                stats[view]["u_n"] += 1
                stats[view]["u_fp"] += vote
            else:
                stats[view]["g_n"] += 1
                stats[view]["g_fp"] += vote
    print("   | camera | marked crops found | FP good-brick crops | FP unmarked crops of defective bricks |")
    total = collections.Counter()
    for view in ("top", "floor", "arm"):
        s = stats[view]
        total.update(s)
        print(f"   | {view} | {s['m_hit']}/{s['m_n']} | {s['g_fp']}/{s['g_n']} | {s['u_fp']}/{s['u_n']} |")
    print(f"   | all | {total['m_hit']}/{total['m_n']} | {total['g_fp']}/{total['g_n']} | {total['u_fp']}/{total['u_n']} |"
          f" ({total['unlabelled']} crops without a label)")


def union(rows: list[dict], scores: Path, truth: dict) -> None:
    brick, label = collections.defaultdict(float), {}
    for r in csv.DictReader(scores.open()):
        if r["scene"].startswith("snap/"):
            snap = r["scene"].split("/", 1)[1]
            brick[snap] = max(brick[snap], float(r["score"]))
            label[snap] = truth.get(snap, r["label"])
    good = [s for s in label if label[s] == "good"]
    thr = max(brick[s] for s in good) + 1e-6  # zero FP on the held-out good bricks
    vlm = {r["scene"]: r["is_broken"] for r in rows}
    c = collections.Counter()
    for s in (s for s in label if label[s] == "defect"):
        a, q = brick[s] >= thr, vlm.get(s, False)
        c["both" if a and q else "only Anomalib" if a else "only VLM" if q else "neither"] += 1
    n = sum(c.values())
    print(f"   union with PatchCore (threshold {thr:.4f} = 0 FP on {len(good)} held-out good bricks): "
          f"{n - c['neither']}/{n} defective: {dict(c)}")


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union_ = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union_ if union_ else 0.0


def grounding(rows: list[dict], ref_file: Path) -> None:
    ref = {r["scene"]: r["boxes"] for r in load(ref_file)}
    for cam in ("top", "floor", "arm"):
        vals = []
        for r in rows:
            for name, box in (r.get("boxes") or {}).items():
                rb = ref.get(r["scene"], {}).get(name)
                if name.split(".")[0] != cam or not rb:
                    continue
                if box is None:
                    vals.append((0.0, None))
                    continue
                err = ((box[0] + box[2] - rb[0] - rb[2]) ** 2 + (box[1] + box[3] - rb[1] - rb[3]) ** 2) ** 0.5 / 2
                vals.append((iou(box, rb), err))
        if vals:
            errs = [e for _, e in vals if e is not None]
            print(f"   {cam}: {len(vals)} boxes, IoU median {statistics.median(v for v, _ in vals):.2f}, "
                  f"IoU>=0.5 {sum(v >= 0.5 for v, _ in vals)}/{len(vals)}, centre error median "
                  f"{statistics.median(errs) if errs else float('nan'):.1f} px")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--index", type=Path, default=Path("captures/anomalib/dataset/bricks_z14/index.csv"))
    ap.add_argument("--manual", type=Path, default=Path("configs/defect_mark_labels.csv"))
    ap.add_argument("--patchcore", type=Path,
                    default=Path("outputs/anomalib/runs/20261114-070921_patchcore_bricks_z14/scores_fp16_CPU.csv"))
    ap.add_argument("--color-boxes", type=Path, default=Path("captures/vlm/color_boxes_all.jsonl"),
                    help="brick_judge.py --csv --no-vlm output (reference boxes, checked by eye)")
    args = ap.parse_args()
    truth = {r["snapshot_id"]: "defect" if r["defective"] == "true" else "good" for r in csv.DictReader(SNAPCSV.open())}
    labels = crop_labels(args.index, args.manual, truth) if args.index.exists() else {}

    print("| run | model | images | latency median s | FP good | real defective found | synthetic found |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    details = []
    for f in args.results:
        rows = [r for r in load(f) if "is_broken" in r]
        if not rows:
            continue
        for r in rows:  # current label, not the one at run time
            if not r["scene"].endswith("-synth") and r["scene"] in truth:
                r["label"] = truth[r["scene"]] == "defect"
        real = [r for r in rows if not r["scene"].endswith("-synth")]
        syn = [r for r in rows if r["scene"].endswith("-synth")]
        good = [r for r in real if r["label"] is False]
        bad = [r for r in real if r["label"] is True]
        lat = statistics.median(r["latency_s"] for r in rows)
        print(f"| {f.stem} | {rows[0]['model']} | {statistics.median(r['images'] for r in rows):g} | {lat:.1f} | "
              f"{sum(r['is_broken'] for r in good)}/{len(good)} | {sum(r['is_broken'] for r in bad)}/{len(bad)} | "
              f"{sum(r['is_broken'] for r in syn)}/{len(syn)} |")
        if "per_image" in rows[0]:
            details.append((f, real, "per_image"))
        if "locate_s" in rows[0]:
            details.append((f, real, "locate"))
    for f, real, kind in details:
        print(f"\n== {f.parent.name}/{f.name}")
        if kind == "per_image":
            req = [v for r in real for v in r["per_image_latency_s"].values()]
            print(f"   one request per image: latency median {statistics.median(req):.1f} s")
            if labels:
                per_crop(real, labels)
            if args.patchcore.exists():
                union(real, args.patchcore, truth)
        else:
            print(f"   locate (3 boxes per scene): median {statistics.median(r['locate_s'] for r in real):.1f} s")
            if args.color_boxes.exists():
                grounding(real, args.color_boxes)


if __name__ == "__main__":
    main()

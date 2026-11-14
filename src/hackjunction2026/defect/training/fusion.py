"""Brick-level OR fusion of PatchCore, the supervised mark classifier and the VLM (ADR 0006, ADR 0008).

    anomalib-env/bin/python -m hackjunction2026.defect.training.fusion \
        --patchcore outputs/anomalib/runs/<run>/scores_fp16_NPU.csv --patchcore-threshold 0.52 \
        --classifier outputs/mark_classifier/<run>/oof_scores.csv --classifier-threshold <t> \
        --vlm captures/vlm/eval/qwen8-int8/crops-or-v3.jsonl

Bricks = state snapshots in PatchCore's TEST split (good test snapshots + every defective snapshot).
Ground truth = the snapshot's current `defective` flag (dataset index). A detector votes "defective" if
any crop of the brick is above its threshold (VLM: its scene verdict `is_broken`). Classifier scores
are out-of-fold (5-fold by snapshot), so every brick is scored by a model that never saw it.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


def brick_max(path: Path) -> dict[str, tuple[float, str]]:
    out: dict[str, tuple[float, str]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r["source"].startswith("snapshot:") or r["split"] != "test":
                continue
            sid = r["scene"].split("/")[-1]
            s = float(r["score"])
            if sid not in out or s > out[sid][0]:
                out[sid] = (s, r["label"])
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patchcore", required=True)
    p.add_argument("--patchcore-threshold", type=float, required=True)
    p.add_argument("--classifier", required=True)
    p.add_argument("--classifier-threshold", type=float, required=True)
    p.add_argument("--vlm", required=True)
    p.add_argument("--out", default="outputs/fusion_bricks.md")
    a = p.parse_args(argv)

    pc = brick_max(Path(a.patchcore))
    cl = brick_max(Path(a.classifier))
    vlm: dict[str, bool] = {}
    for line in Path(a.vlm).read_text().splitlines():
        if line.strip().startswith("{"):
            j = json.loads(line)
            if "scene" in j and "is_broken" in j:
                vlm[j["scene"]] = bool(j["is_broken"])
    bricks = sorted(set(pc) & set(cl) & set(vlm))
    label = {b: pc[b][1] == "defect" for b in bricks}
    votes = {
        "PatchCore": {b: pc[b][0] > a.patchcore_threshold for b in bricks},
        "classifier": {b: cl[b][0] > a.classifier_threshold for b in bricks},
        "Qwen3-VL": {b: vlm[b] for b in bricks},
    }
    n_def = sum(label.values())
    n_good = len(bricks) - n_def
    lines = [f"Bricks: {len(bricks)} ({n_def} defective, {n_good} good test snapshots); OR over the named detectors.",
             f"Thresholds: PatchCore {a.patchcore_threshold}, classifier {a.classifier_threshold}, Qwen = scene verdict.",
             "", "| detectors (OR) | recall (defective bricks) | false alarms (good bricks) | missed bricks |",
             "| --- | --- | --- | --- |"]
    names = list(votes)
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            dec = {b: any(votes[c][b] for c in combo) for b in bricks}
            tp = sum(dec[b] and label[b] for b in bricks)
            fp = sum(dec[b] and not label[b] for b in bricks)
            missed = [b[9:] for b in bricks if label[b] and not dec[b]]
            lines.append(f"| {' + '.join(combo)} | {tp}/{n_def} ({100 * tp / n_def:.0f} %) | "
                         f"{fp}/{n_good} ({100 * fp / max(n_good, 1):.0f} %) | {', '.join(missed) if len(missed) <= 6 else len(missed)} |")
    missing = {"PatchCore": len(pc), "classifier": len(cl), "Qwen3-VL": len(vlm)}
    lines += ["", f"Bricks available per detector before intersecting: {missing}"]
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

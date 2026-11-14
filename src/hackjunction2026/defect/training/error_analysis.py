"""Error analysis of the brick detector: PatchCore + supervised classifier, OR per brick (ADR 0006).

    anomalib-env/bin/python -m hackjunction2026.defect.training.error_analysis \
        --dataset bricks_z14t --patchcore <run scores_fp16_NPU.csv> --pc-threshold 0.521 \
        --classifier <oof_scores.csv> --clf-threshold 0.998

Bricks = state snapshots in PatchCore's test split (good test snapshots + every defective one).
Per crop it measures: colour, camera, brick size in the source frame (bbox long side, px), mark-like
dark pixels inside the brick mask (L < 0.55 x the brick's median L) and dark contrast
((median L - 2nd percentile L) / median L), sharpness, face (gemini "where" text), source, session.
Writes outputs/error_analysis/<dataset>/: crops.csv, bricks.csv, breakdown.md, errors.png (one row per
missed or falsely flagged brick, its crops left to right, labelled with both scores).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATASETS = Path("captures/anomalib/dataset")
PX_BUCKETS = ((0, 60, "<60"), (60, 75, "60-74"), (75, 90, "75-89"), (90, 10_000, ">=90"))


def scores(path: Path) -> dict[str, float]:
    with open(path, newline="") as f:
        return {r["path"]: float(r["score"]) for r in csv.DictReader(f)}


def crop_stats(img: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    if inner.sum() < 50:
        inner = mask > 0
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)[inner]
    med = float(np.median(L))
    dark = float(np.mean(L < 0.55 * med))
    contrast = float((med - np.percentile(L, 2)) / max(med, 1.0))
    return med, dark, contrast


def bucket(px: float) -> str:
    return next(name for lo, hi, name in PX_BUCKETS if lo <= px < hi)


def rate(xs: list[bool]) -> str:
    return f"{sum(xs)}/{len(xs)} ({100 * sum(xs) / len(xs):.0f} %)" if xs else "-"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="bricks_z14t")
    p.add_argument("--patchcore", required=True)
    p.add_argument("--pc-threshold", type=float, default=0.521)
    p.add_argument("--classifier", required=True)
    p.add_argument("--clf-threshold", type=float, default=0.998)
    p.add_argument("--out", default="outputs/error_analysis")
    a = p.parse_args(argv)
    root = DATASETS / a.dataset
    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    pc, cl = scores(Path(a.patchcore)), scores(Path(a.classifier))

    with open(root / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source"].startswith("snapshot:") and r["split"] == "test"
                and r["path"] in pc]
    crops = []
    for r in rows:
        img = cv2.imread(str(root / r["path"]))
        mask = cv2.imread(str(root / "masks" / f"{r['crop_id']}.png"), cv2.IMREAD_GRAYSCALE)
        med, dark, contrast = crop_stats(img, mask)
        px = max(int(r["bbox_w"]), int(r["bbox_h"]))
        c = {"path": r["path"], "brick": r["brick"], "label": r["label"], "mark": r["cls"] == "defect",
             "colour": r["colour"], "camera": r["view"], "px": px, "px_bucket": bucket(px),
             "brick_L": round(med, 1), "dark_frac": round(dark, 4), "dark_contrast": round(contrast, 3),
             "sharpness": float(r["sharpness"] or 0), "face": r.get("gemini_where", ""), "source": r["source"],
             "pc": pc[r["path"]], "clf": cl.get(r["path"], float("nan"))}
        c["pc_hit"] = c["pc"] > a.pc_threshold
        c["clf_hit"] = c["clf"] > a.clf_threshold
        c["or_hit"] = c["pc_hit"] or c["clf_hit"]
        crops.append(c)
    with open(out / "crops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crops[0]))
        w.writeheader()
        w.writerows(crops)

    by_brick: dict[str, list[dict]] = defaultdict(list)
    for c in crops:
        by_brick[c["brick"]].append(c)
    bricks = []
    for b, cs in sorted(by_brick.items()):
        colours = [c["colour"] for c in cs]
        bricks.append({"brick": b, "label": cs[0]["label"], "colour": max(set(colours), key=colours.count),
                       "px": int(np.median([c["px"] for c in cs])), "n_crops": len(cs),
                       "n_mark_crops": sum(c["mark"] for c in cs),
                       "pc": max(c["pc"] for c in cs), "clf": max(c["clf"] for c in cs),
                       "pc_hit": any(c["pc_hit"] for c in cs), "clf_hit": any(c["clf_hit"] for c in cs),
                       "or_hit": any(c["or_hit"] for c in cs), "cameras": ",".join(sorted(c["camera"] for c in cs))})
    with open(out / "bricks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bricks[0]))
        w.writeheader()
        w.writerows(bricks)

    # ---- breakdown tables
    L = [f"Dataset `{a.dataset}`: {len(bricks)} snapshot bricks ({sum(b['label'] == 'defect' for b in bricks)} defective), "
         f"{len(crops)} crops. PatchCore > {a.pc_threshold}, classifier > {a.clf_threshold} (out-of-fold), OR per brick.", ""]

    def table(title, keyf, items, kind):
        L.extend([f"### {title}", "", "| group | recall PatchCore | recall classifier | recall OR | false alarms PatchCore | "
                  "false alarms classifier | false alarms OR |", "| --- | --- | --- | --- | --- | --- | --- |"])
        groups = sorted({keyf(i) for i in items})
        for g in groups:
            sel = [i for i in items if keyf(i) == g]
            pos = [i for i in sel if (i["mark"] if kind == "crop" else i["label"] == "defect")]
            neg = [i for i in sel if i["label"] == "good"]
            L.append(f"| {g} | {rate([i['pc_hit'] for i in pos])} | {rate([i['clf_hit'] for i in pos])} | "
                     f"{rate([i['or_hit'] for i in pos])} | {rate([i['pc_hit'] for i in neg])} | "
                     f"{rate([i['clf_hit'] for i in neg])} | {rate([i['or_hit'] for i in neg])} |")
        L.append("")

    table("Bricks: all", lambda b: "all", bricks, "brick")
    table("Bricks by colour", lambda b: b["colour"], bricks, "brick")
    table("Bricks by size (median bbox long side, px)", lambda b: bucket(b["px"]), bricks, "brick")
    table("Crops (recall over mark-visible crops) by camera", lambda c: c["camera"], crops, "crop")
    table("Crops by colour", lambda c: c["colour"], crops, "crop")
    table("Crops by size (px)", lambda c: c["px_bucket"], crops, "crop")
    marks = [c for c in crops if c["mark"]]
    for name in ("pc_hit", "clf_hit"):
        hit = [c for c in marks if c[name]]
        miss = [c for c in marks if not c[name]]
        L.append(f"- {name[:-4]}: mark crops caught vs missed: median dark_frac {np.median([c['dark_frac'] for c in hit]):.4f} vs "
                 f"{np.median([c['dark_frac'] for c in miss]):.4f}; dark_contrast {np.median([c['dark_contrast'] for c in hit]):.3f} "
                 f"vs {np.median([c['dark_contrast'] for c in miss]):.3f}; px {np.median([c['px'] for c in hit]):.0f} vs "
                 f"{np.median([c['px'] for c in miss]):.0f}; sharpness {np.median([c['sharpness'] for c in hit]):.0f} vs "
                 f"{np.median([c['sharpness'] for c in miss]):.0f}")
    goods = [c for c in crops if c["label"] == "good"]
    L.append(f"- good crops: median dark_frac {np.median([c['dark_frac'] for c in goods]):.4f}, "
             f"dark_contrast {np.median([c['dark_contrast'] for c in goods]):.3f}")
    errs = [b for b in bricks if (b["label"] == "defect" and not (b["pc_hit"] and b["clf_hit"])) or
            (b["label"] == "good" and (b["pc_hit"] or b["clf_hit"]))]
    L += ["", "### Every brick that at least one model gets wrong", "",
          "| brick | label | colour | px | mark crops | PatchCore max | classifier max | PatchCore | classifier | OR |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for b in errs:
        ok = lambda hit: "ok" if hit == (b["label"] == "defect") else ("MISS" if b["label"] == "defect" else "FALSE ALARM")
        L.append(f"| {b['brick'][9:]} | {b['label']} | {b['colour']} | {b['px']} | {b['n_mark_crops']}/{b['n_crops']} | "
                 f"{b['pc']:.3f} | {b['clf']:.3f} | {ok(b['pc_hit'])} | {ok(b['clf_hit'])} | {ok(b['or_hit'])} |")
    (out / "breakdown.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- contact sheet: one row per wrong brick, its crops, labelled
    size, rows_img = 170, []
    for b in errs:
        tiles = []
        for c in sorted(by_brick[b["brick"]], key=lambda c: c["camera"]):
            img = cv2.resize(cv2.imread(str(root / c["path"])), (size, size))
            col = (0, 140, 0) if c["mark"] else (0, 0, 0)
            cv2.putText(img, f"{c['camera']} {c['px']}px {'MARK' if c['mark'] else ''}", (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            cv2.putText(img, f"P {c['pc']:.2f}{'*' if c['pc_hit'] else ''}  C {c['clf']:.2f}{'*' if c['clf_hit'] else ''}",
                        (3, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
            tiles.append(img)
        while len(tiles) < 3:
            tiles.append(np.full((size, size, 3), 255, np.uint8))
        head = np.full((size, 150, 3), 255, np.uint8)
        verdict = "MISSED" if b["label"] == "defect" and not b["or_hit"] else (
            "FALSE ALARM" if b["label"] == "good" and b["or_hit"] else "one model wrong")
        for i, t in enumerate([b["brick"][9:], b["label"], b["colour"], verdict]):
            cv2.putText(head, t, (4, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        rows_img.append(np.hstack([head, *tiles[:3]]))
    if rows_img:
        cv2.imwrite(str(out / "errors.png"), np.vstack(rows_img))
    print(f"\nwritten {out}/crops.csv, bricks.csv, breakdown.md, errors.png ({len(errs)} bricks)")


if __name__ == "__main__":
    main()

"""Build the Anomalib training dataset: raw capture sessions -> fixed-box 256x256 crops + split.

    uv run --extra capture python -m hackjunction2026.defect.cropping.make_dataset --name v1

Reads every captures/anomalib/raw/<session>/manifest.jsonl (or only --sessions), keeps complete
cycles (PRE and POST saved), crops each image with the same `crops.crop()` the robot uses, and
writes:

    captures/anomalib/dataset/<name>/
        train/good/      ~80% of the good cycles          (the ONLY training data)
        test/good/       the other good cycles             (never seen in training)
        test/defect/     views that show a defect          (defective bricks are never trained on)
        test/defect_clean/  other views of defective bricks (not a "defect" image; used per cycle)
        index.csv        one row per image: split, class, session, brick, cycle, view, note
        dataset.json     summary: counts, sources, settings, git commit

The split is by cycle (all 4 views of one placement stay together) and per brick, with a fixed seed,
so the same raw data always gives the same dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2

from hackjunction2026.defect.cropping.crops import Box, crop

FIELDS = ["path", "split", "cls", "session", "brick", "label", "cycle", "view", "defect_note", "source"]


def load_records(raw: Path, sessions: list[str] | None) -> list[dict]:
    dirs = [raw / s for s in sessions] if sessions else sorted(p for p in raw.iterdir() if p.is_dir())
    records = []
    for d in dirs:
        manifest = d / "manifest.jsonl"
        if not manifest.exists():
            print(f"skip {d}: no manifest.jsonl")
            continue
        for line in manifest.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["session"] = d.name
                records.append(rec)
    return records


def complete_cycles(records: list[dict]) -> dict[tuple, list[dict]]:
    """(session, label, brick, cycle) -> [pre record, post record]; drops unfinished cycles."""
    cycles: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in records:
        cycles[(r["session"], r["label"], r["brick"], r["cycle"])][r["position"]] = r
    out, dropped = {}, 0
    for key, by_pos in cycles.items():
        if set(by_pos) == {"pre", "post"}:
            out[key] = [by_pos["pre"], by_pos["post"]]
        else:
            dropped += 1
    if dropped:
        print(f"dropped {dropped} unfinished cycle(s) (PRE without POST)")
    return out


def split_good(keys: list[tuple], test_fraction: float, seed: int) -> set[tuple]:
    """Pick test cycles per brick: ~test_fraction of each brick's cycles, at least 1 if it has >= 2."""
    by_brick: dict[str, list[tuple]] = defaultdict(list)
    for k in keys:
        by_brick[k[2]].append(k)
    rng = random.Random(seed)
    test = set()
    for brick in sorted(by_brick):
        ks = sorted(by_brick[brick])
        rng.shuffle(ks)
        n = math.ceil(len(ks) * test_fraction) if len(ks) >= 2 else 0
        test.update(ks[:n])
    return test


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(raw: Path, out_root: Path, name: str, sessions: list[str] | None, test_fraction: float,
          seed: int, size: int, overwrite: bool) -> Path:
    out = out_root / name
    if out.exists():
        if not overwrite:
            msg = f"{out} exists; use --overwrite or another --name"
            raise SystemExit(msg)
        shutil.rmtree(out)

    cycles = complete_cycles(load_records(raw, sessions))
    if not cycles:
        msg = f"no complete cycles found in {raw}"
        raise SystemExit(msg)
    good_test = split_good([k for k in cycles if k[1] == "good"], test_fraction, seed)

    rows = []
    for key in sorted(cycles):
        session, label, brick, cycle = key
        for rec in cycles[key]:
            for view, rel in rec["files"].items():
                if label == "good":
                    split, cls = ("test" if key in good_test else "train"), "good"
                else:
                    shows = rec.get("defect_views") or list(rec["files"])  # None/empty = all views
                    split, cls = "test", ("defect" if view in shows else "defect_clean")
                src = raw / session / rel
                img = cv2.imread(str(src))
                if img is None:
                    msg = f"cannot read {src}"
                    raise SystemExit(msg)
                dst_rel = Path(split) / cls / f"{session}__{brick}_{cycle:03d}_{view}.png"
                (out / dst_rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / dst_rel), crop(img, Box.from_list(rec["box_in_region"][view]), size))
                rows.append({"path": dst_rel.as_posix(), "split": split, "cls": cls, "session": session,
                             "brick": brick, "label": label, "cycle": cycle, "view": view,
                             "defect_note": rec.get("defect_note") or "", "source": f"{session}/{rel}"})

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[f"{r['split']}/{r['cls']}"] += 1
    n_cycles = defaultdict(int)
    for k in cycles:
        n_cycles["good_test" if k in good_test else ("good_train" if k[1] == "good" else "defect")] += 1
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_sha(), "raw": str(raw),
               "sessions": sorted({k[0] for k in cycles}), "bricks": sorted({k[2] for k in cycles}),
               "image_size": size, "test_fraction": test_fraction, "seed": seed,
               "images": dict(sorted(counts.items())), "cycles": dict(n_cycles)}
    (out / "dataset.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["images"]), json.dumps(summary["cycles"]))
    if not counts.get("test/defect"):
        print("note: no defect images yet -> training validates with synthetic anomalies only")
    print(f"dataset written to {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("captures/anomalib/raw"))
    p.add_argument("--sessions", nargs="*", help="session folder names (default: all)")
    p.add_argument("--out", type=Path, default=Path("captures/anomalib/dataset"))
    p.add_argument("--name", default="v1", help="dataset name, e.g. v1 (folder under --out)")
    p.add_argument("--test-fraction", type=float, default=0.2, help="share of good cycles per brick for test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=256, help="crop size in pixels (model input)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    build(a.raw, a.out, a.name, a.sessions, a.test_fraction, a.seed, a.size, a.overwrite)


if __name__ == "__main__":
    main()

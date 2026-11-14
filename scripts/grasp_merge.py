#!/usr/bin/env python3
"""Merge the grasp-only dataset (NEW) with the grasp tails of the approach dataset (OLD).

Run from the repo root with Studio's backend Python (it has LeRobot, PyAV, OpenCV, pandas):

  PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
  export PYTHONPATH=src
  nice -n 19 $PY scripts/grasp_merge.py cuts                 # cuts.csv / cuts.json (+ overrides)
  nice -n 19 $PY scripts/grasp_merge.py clips                # H.264 review clips, top camera
  nice -n 19 $PY scripts/grasp_merge.py build --name grasp-merged-2026-11-14
  nice -n 19 $PY scripts/grasp_merge.py validate <dataset dir>

Read-only on both source datasets. Criterion, numbers and the re-run recipe:
docs/datasets/grasp-merged.md. Perception inputs (FK, bricks, registration) must exist first; the
commands that make them for NEW are in that doc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATASETS = Path.home() / ".local/share/physicalai/datasets"
OLD = "431a71f7-50c6-4e38-9e38-d85984dde78b"
NEW = "7efd5320-a08b-49d6-abed-89b38deaf1ed"
CAPTURES = Path("/home/intel/dev/captures")
OUT = CAPTURES / "merge_review"


def cmd_cuts(args: argparse.Namespace) -> None:
    from hackjunction2026.grasp_merge import cuts as C

    p = C.Params()
    old_root, new_root = DATASETS / args.old, DATASETS / args.new
    fps = json.loads((old_root / "meta/info.json").read_text())["fps"]
    eps_old, eps_new = C.episode_table(old_root), C.episode_table(new_root)

    def feats(ds: str, pdir: Path):
        rt = json.loads((pdir / f"robot_to_table_{ds}.json").read_text())
        bricks = C.resting_bricks(pd.read_parquet(pdir / f"objects_{ds}.parquet"), rt, p.brick_frames)
        return C.frame_features(pd.read_parquet(Path(args.arm_dir) / f"arm_{ds}.parquet"), bricks), rt

    f_new, rt_new = feats(args.new, Path(args.new_perception))
    f_old, rt_old = feats(args.old, Path(args.old_perception))
    ref = C.new_reference(f_new, eps_new, p)
    overrides = C.read_overrides(Path(args.overrides))
    rows = C.compute_cuts(f_old, eps_old, ref, p, fps, overrides)
    excluded = [{"episode": int(i), "task": r.task, "length": int(r.length)}
                for i, r in eps_new.iterrows() if r.length < p.min_episode_frames]
    extra = {"old_dataset": args.old, "new_dataset": args.new, "fps": fps,
             "overrides_file": str(args.overrides), "overrides": {str(k): v for k, v in overrides.items()},
             "new_excluded": excluded,
             "registration": {"old": {k: rt_old[k] for k in ("scale_s", "theta_deg", "n_inliers", "residual_mm")},
                              "new": {k: rt_new[k] for k in ("scale_s", "theta_deg", "n_inliers", "residual_mm")}}}
    C.write_outputs(Path(args.out), rows, ref, p, extra)
    df = pd.DataFrame(rows)
    print(C.criterion_text(p, ref))
    print(df[["episode", "length", "cut_frame", "kept_frames", "h_mm", "d_mm", "gripper", "score", "source",
              "flags"]].to_string(index=False))
    k = df[df.source != "skip"].kept_frames
    print(f"kept frames: median {k.median():.0f} ({k.median() / fps:.1f} s), range {k.min()}-{k.max()}, "
          f"total {k.sum()}; low confidence: {(df.confidence == 'low').sum()} of {len(df)}; "
          f"NEW excluded: {excluded}")
    print(f"wrote {args.out}/cuts.csv, cuts.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--new", default=NEW)
    ap.add_argument("--out", default=str(OUT), help="review / cuts folder (default: %(default)s)")
    ap.add_argument("--overrides", default=str(REPO / "configs/grasp_merge_overrides.csv"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("cuts", help="cut point per OLD episode -> cuts.csv, cuts.json")
    sc.add_argument("--arm-dir", default=str(CAPTURES / "perception/B"), help="arm_<dataset>.parquet (FK)")
    sc.add_argument("--old-perception", default=str(CAPTURES / "perception/A"))
    sc.add_argument("--new-perception", default=str(OUT / "perception"))
    sc.set_defaults(func=cmd_cuts)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

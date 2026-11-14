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


def features(args: argparse.Namespace, ds: str, pdir: str):
    """Per-frame features of one dataset and its robot-to-table registration."""
    from hackjunction2026.grasp_merge import cuts as C

    pdir = Path(pdir)
    rt = json.loads((pdir / f"robot_to_table_{ds}.json").read_text())
    bricks = C.resting_bricks(pd.read_parquet(pdir / f"objects_{ds}.parquet"), rt, C.Params().brick_frames)
    return C.frame_features(pd.read_parquet(Path(args.arm_dir) / f"arm_{ds}.parquet"), bricks), rt


def cmd_cuts(args: argparse.Namespace) -> None:
    from hackjunction2026.grasp_merge import cuts as C

    p = C.Params()
    old_root, new_root = DATASETS / args.old, DATASETS / args.new
    fps = json.loads((old_root / "meta/info.json").read_text())["fps"]
    eps_old, eps_new = C.episode_table(old_root), C.episode_table(new_root)
    f_new, rt_new = features(args, args.new, args.new_perception)
    f_old, rt_old = features(args, args.old, args.old_perception)
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


def cmd_clips(args: argparse.Namespace) -> None:
    from hackjunction2026.grasp_merge.clips import make_clips

    cuts_doc = json.loads((Path(args.out) / "cuts.json").read_text())
    f_old, _ = features(args, args.old, args.old_perception)
    eps = {int(x) for x in args.episodes.split(",")} if args.episodes else None
    folder = Path(args.out) / ("clips" if args.camera == "top" else f"clips_{args.camera}")
    doc = make_clips(DATASETS / args.old, cuts_doc, f_old, folder, camera=args.camera, crf=args.crf,
                     threads=args.threads, max_load=args.max_load, max_wait_s=args.max_wait, episodes=eps,
                     max_kb=args.max_kb)
    sizes = [e["clip_bytes"] for e in doc["episodes"] if e["clip_bytes"]]
    print(f"{len(sizes)} clips, {sum(sizes) / 1e6:.1f} MB, largest {max(sizes) / 1e3:.0f} kB -> {folder}")


def cmd_build(args: argparse.Namespace) -> None:
    from hackjunction2026.grasp_merge import cuts as C
    from hackjunction2026.grasp_merge.build import build
    from hackjunction2026.grasp_merge.validate import validate

    cuts_doc = json.loads((Path(args.out) / "cuts.json").read_text())
    current = {str(k): v for k, v in C.read_overrides(Path(args.overrides)).items()}
    if current != cuts_doc.get("overrides", {}):
        sys.exit(f"{args.overrides} changed since cuts.json was written: run 'grasp_merge.py cuts' first")
    dst = Path(args.dest) if args.dest else Path(args.out) / "dataset" / args.name
    prov = build(DATASETS / args.new, DATASETS / args.old, cuts_doc, Path(args.out) / "work", dst, args.name,
                 threads=args.threads, max_load=args.max_load, max_wait_s=args.max_wait, keep_work=args.keep_work,
                 cache=Path(args.cache),
                 new_exclude=tuple(int(x) for x in args.new_exclude.split(",")) if args.new_exclude else ())
    print(f"merged dataset: {dst} ({prov['episodes']} episodes, {prov['frames']} frames; "
          f"OLD tails encoded in this build: {prov['parts'][1]['encoded_this_build']})")
    if not args.no_validate:
        res = validate(dst, Path(args.out))
        if not res["ok"]:
            sys.exit(1)


def cmd_seed(args: argparse.Namespace) -> None:
    from hackjunction2026.grasp_merge.build import seed_cache

    seed_cache(Path(args.dataset), Path(args.cache))


def cmd_import(args: argparse.Namespace) -> None:
    from hackjunction2026.grasp_merge.studio_import import import_dataset, zip_dataset

    root = Path(args.dataset)
    name = args.name or root.name
    zip_path = Path(args.out) / f"{name}.zip"
    if not zip_path.exists():
        print(f"zipping {root} -> {zip_path} (stored)")
        zip_dataset(root, zip_path)
    res = import_dataset(zip_path, name, args.project, args.environment, base_url=args.url)
    print(json.dumps(res, indent=1))
    (Path(args.out) / "studio_import.json").write_text(json.dumps({"name": name, **res}, indent=1))


def cmd_validate(args: argparse.Namespace) -> None:
    from hackjunction2026.grasp_merge.validate import validate

    res = validate(Path(args.dataset), Path(args.out))
    sys.exit(0 if res["ok"] else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--new", default=NEW)
    ap.add_argument("--out", default=str(OUT), help="review / cuts folder (default: %(default)s)")
    ap.add_argument("--overrides", default=str(REPO / "configs/grasp_merge_overrides.csv"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("cuts", help="cut point per OLD episode -> cuts.csv, cuts.json")
    sc.set_defaults(func=cmd_cuts)

    sv = sub.add_parser("clips", help="H.264 review clip per OLD episode (top camera) + clips/index.json")
    sv.add_argument("--camera", default="top", choices=["top", "arm"],
                    help="top -> <out>/clips (no people); arm (wrist) -> <out>/clips_arm (may show people)")
    sv.add_argument("--max-kb", type=float, help="size budget per clip (x264 bitrate cap), e.g. 250")
    sv.add_argument("--episodes", help="comma list (default: all; index.json always lists all)")
    sv.add_argument("--crf", type=int, default=26)
    sv.add_argument("--threads", type=int, default=2, help="decoder and encoder threads")
    sv.add_argument("--max-load", type=float, default=8.0, help="pause between episodes above this load")
    sv.add_argument("--max-wait", type=float, default=180.0, help="longest pause per episode, s")
    sv.set_defaults(func=cmd_clips)
    for sp in (sc, sv):
        sp.add_argument("--arm-dir", default=str(CAPTURES / "perception/B"), help="arm_<dataset>.parquet (FK)")
        sp.add_argument("--old-perception", default=str(CAPTURES / "perception/A"))
        sp.add_argument("--new-perception", default=str(OUT / "perception"))

    sb = sub.add_parser("build", help="merged LeRobot v3 dataset from cuts.json, then validate")
    sb.add_argument("--name", default="grasp-merged-2026-11-14")
    sb.add_argument("--dest", help="output folder (default: <out>/dataset/<name>); must not exist")
    sb.add_argument("--threads", type=int, default=2, help="SVT-AV1 threads per camera encoder (3 run in parallel)")
    sb.add_argument("--max-load", type=float, default=8.0, help="pause between episodes above this load")
    sb.add_argument("--max-wait", type=float, default=20.0, help="longest pause per episode, s")
    sb.add_argument("--keep-work", action="store_true", help="keep the intermediate datasets in <out>/work")
    sb.add_argument("--no-validate", action="store_true")
    sb.add_argument("--new-exclude", help="comma list of NEW episodes to leave out (besides those < 15 frames)")
    sb.set_defaults(func=cmd_build)

    ss = sub.add_parser("seed-cache", help="split the OLD tails of a merged dataset into the tail cache (lossless)")
    ss.add_argument("dataset")
    ss.set_defaults(func=cmd_seed)
    for sp in (sb, ss):
        sp.add_argument("--cache", default=str(OUT / "work/old_tails"),
                        help="one dataset per (OLD episode, cut), reused between builds (default: %(default)s)")

    sa = sub.add_parser("validate", help="check a merged dataset against its sources")
    sa.add_argument("dataset")
    sa.set_defaults(func=cmd_validate)

    si = sub.add_parser("import", help="import a dataset folder into the running Studio (its import API)")
    si.add_argument("dataset")
    si.add_argument("--name", help="Studio dataset name (default: folder name)")
    si.add_argument("--project", default="0fd76ff2-a6b1-4472-b687-486befe329be", help="project 'junction'")
    si.add_argument("--environment", default="a5c1cc42-0a40-4dba-a961-5ea241be7fdc",
                    help="environment 'follow' (the one both sources use)")
    si.add_argument("--url", default="http://localhost:7860")
    si.set_defaults(func=cmd_import)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build an Anomalib crop set of single bricks from Studio datasets and state snapshots.

Every sampled camera frame goes through hackjunction2026.defect.bricks (HSV segmentation, the same
code the runtime uses), and each brick becomes a square, brick-centred 256x256 crop.

Output: captures/anomalib/<set>/
  train/good/                      dataset episodes of the train split
  val/good/  val/synthetic_defect/ val episodes (+ SYNTHETIC marker strokes, eval only)
  test/good/ test/synthetic_defect/ test episodes + good snapshots (+ SYNTHETIC strokes)
  test/defective/                  REAL marked bricks: every crop of a scene labelled defective
  */*_mask/                        brick hull (good) or stroke (synthetic) masks
  index.csv                        one row per crop: source, scene, episode, frame, camera, bbox, ...
  config.json                      segmentation + crop settings (read by training and runtime)
  preview/                         annotated frames for a visual check of the segmentation
Splits are by EPISODE (snapshot = its own group), never by frame.
Synthetic crops are never used for training.

Run from the repo root with the Anomalib env (needs numpy, opencv, av, pyarrow):
  nice -n 19 anomalib-env/bin/python scripts/brick_crops.py --set v0 --dataset 431a71f7 --snapshots all
  ... --background both          # also write <set>-raw (background kept) with the same crops/strokes
  ... --pose inspect --pose-tol 6 # only frames near configs/poses.yaml:inspect (inspection pose)
Re-running a set deletes and rebuilds it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hackjunction2026.defect.bricks import (  # noqa: E402
    CropConfig,
    SegConfig,
    crop_brick,
    draw_bricks,
    find_bricks,
    sharpness,
)
from hackjunction2026.defect.synth import add_marker  # noqa: E402

DATASETS = Path.home() / ".local/share/physicalai/datasets"
SNAPSHOTS = Path.home() / ".local/share/physicalai/state_snapshots"
CAMERAS = ("top", "floor", "arm")
DEFECT_TASK = r"defect|marked|broken|bad|mark"

cv2.setNumThreads(2)


# ------------------------------------------------------------------ sources


def resolve_dataset(arg: str) -> Path:
    path = Path(arg).expanduser()
    if (path / "meta/info.json").exists():
        return path
    matches = sorted(DATASETS.glob(f"{arg}*/meta/info.json"))
    if len(matches) != 1:
        sys.exit(f"dataset '{arg}': {len(matches)} matches in {DATASETS}")
    return matches[0].parent.parent


def load_episodes(ds: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for f in sorted(ds.glob("meta/episodes/*/*.parquet")):
        cols = [c for c in pq.read_schema(f).names if not c.startswith("stats/")]
        rows += pq.read_table(f, columns=cols).to_pylist()
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def load_states(ds: Path) -> dict[int, np.ndarray]:
    """episode -> (n_frames, 6) follower state (normalized joint values)."""
    import pyarrow.parquet as pq

    out: dict[int, list] = defaultdict(list)
    for f in sorted(ds.glob("data/*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for ep, fr, st in zip(t["episode_index"], t["frame_index"], t["observation.state"]):
            out[ep].append((fr, st))
    return {ep: np.array([s for _, s in sorted(v)], np.float32) for ep, v in out.items()}


def task_of(row: dict) -> str:
    tasks = row.get("tasks") or [""]
    return str(tasks[0]).strip()


def decode_frames(video: Path, timestamps: list[float]):
    """Yield (i, BGR frame) for each requested timestamp (seconds, ascending)."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        for i, ts in enumerate(timestamps):
            container.seek(int(ts / stream.time_base), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.time is not None and frame.time >= ts - 1e-3:
                    yield i, frame.to_ndarray(format="bgr24")
                    break


def pose_mask(states: np.ndarray, pose: dict, tol: float) -> np.ndarray:
    """Frames whose body joints are within `tol` (normalized units) of the pose.

    Wrist roll and gripper are ignored: the inspection pose rotates the wrist, and the gripper
    holds bricks of different sizes.
    """
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    target = np.array([pose["joints"][n] for n in names], np.float32)
    return np.abs(states[:, :4] - target[None, :]).max(axis=1) <= tol


# ------------------------------------------------------------------ splitting


def assign_splits(episodes: dict[int, str], val_frac: float, test_frac: float, seed: int) -> dict[int, str]:
    """Episode -> split. Stratified by task name so every colour lands in every split if possible."""
    by_task: dict[str, list[int]] = defaultdict(list)
    for ep, task in episodes.items():
        by_task[re.sub(r"[-_ ]?\d+$", "", task.lower())].append(ep)
    rng = np.random.default_rng(seed)
    split: dict[int, str] = {}
    for task in sorted(by_task):
        eps = sorted(by_task[task])
        rng.shuffle(eps)
        n = len(eps)
        n_test = int(round(n * test_frac)) if n >= 3 else 0
        n_val = int(round(n * val_frac)) if n >= 3 else 0
        if n >= 4:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        for i, ep in enumerate(eps):
            split[ep] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split


# ------------------------------------------------------------------ writer


class Writer:
    def __init__(self, roots: dict[str, Path], seg: SegConfig, crops: dict[str, CropConfig], synth: bool, seed: int):
        self.roots, self.seg, self.crops, self.synth, self.seed = roots, seg, crops, synth, seed
        self.rows: list[dict] = []
        self.n_frames = 0
        self.n_empty = 0

    def add_frame(self, frame: np.ndarray, base: dict, split: str, label: str, max_bricks: int,
                  preview: Path | None = None) -> int:
        self.n_frames += 1
        bricks = find_bricks(frame, self.seg)[:max_bricks]
        if not bricks:
            self.n_empty += 1
        if preview is not None:
            cv2.imwrite(str(preview), draw_bricks(frame, bricks))
        for k, b in enumerate(bricks):
            stem = f"{base['scene'].replace('/', '_')}_{base['camera']}_b{k}"
            row = {**base, "split": split, "label": label, "file": "", "parent": "",
                   "bbox_x": b.bbox[0], "bbox_y": b.bbox[1], "bbox_w": b.bbox[2], "bbox_h": b.bbox[3],
                   "colour": b.colour, "mask_area": b.area, "ring_white": b.ring_white, "fill": b.fill}
            first = True
            for variant, root in self.roots.items():
                crop, mask, (sx, sy, side) = crop_brick(frame, b, self.crops[variant])
                rel = f"{split}/{label}/{stem}.png"
                cv2.imwrite(str(root / rel), crop)
                cv2.imwrite(str(root / f"{split}/{label}_mask/{stem}.png"), mask)
                if first:
                    row.update(file=rel, square_x=sx, square_y=sy, square_side=side,
                               sharpness=round(sharpness(crop, mask), 1))
                    self.rows.append(row)
                if self.synth and label == "good" and split in ("val", "test"):
                    rng = np.random.default_rng([self.seed, zlib.crc32(stem.encode())])
                    upscale = self.crops[variant].size / side
                    marked, stroke, params = add_marker(crop, mask, rng, upscale=upscale)
                    srel = f"{split}/synthetic_defect/{stem}_synth.png"
                    cv2.imwrite(str(root / srel), marked)
                    cv2.imwrite(str(root / f"{split}/synthetic_defect_mask/{stem}_synth.png"), stroke)
                    if first:
                        self.rows.append({**row, "label": "synthetic_defect", "file": srel, "parent": rel, **params})
                first = False
        return len(bricks)


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="output name under captures/anomalib/")
    ap.add_argument("--dataset", action="append", default=[], help="Studio dataset id (prefix ok) or path; repeatable")
    ap.add_argument("--snapshots", action="append", default=[],
                    help="state-snapshot project id (prefix ok) or 'all'; repeatable")
    ap.add_argument("--snapshot-role", choices=["test", "split"], default="test",
                    help="test: good snapshots only in test/good (default); split: treat each snapshot as a group "
                         "and split them like episodes (use for inspection-pose snapshots)")
    ap.add_argument("--every", type=int, default=10, help="use every Nth dataset frame (default 10)")
    ap.add_argument("--cameras", default=",".join(CAMERAS))
    ap.add_argument("--episodes", help="only these episodes, e.g. 0-9,12")
    ap.add_argument("--pose", help="only frames near this pose from configs/poses.yaml (e.g. inspect)")
    ap.add_argument("--pose-tol", type=float, default=6.0, help="max abs joint difference, normalized units")
    ap.add_argument("--defect-task", default=DEFECT_TASK,
                    help=f"regex on the episode task: matching episodes are real defects (default '{DEFECT_TASK}')")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--background", choices=["white", "raw", "both"], default="white",
                    help="white: outside the brick hull = white (default); raw: keep; both: <set> white + <set>-raw")
    ap.add_argument("--max-bricks", type=int, default=3, help="max bricks per image (largest first)")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--preview", type=int, default=40, help="annotated frames to save in preview/ (default 40)")
    args = ap.parse_args()

    out_root = REPO / "captures/anomalib"
    variants = {"white": {"white": args.set}, "raw": {"raw": args.set}, "both": {"white": args.set, "raw": f"{args.set}-raw"}}[args.background]
    roots = {v: out_root / name for v, name in variants.items()}
    crops = {v: CropConfig(size=args.size, margin=args.margin, whiten=(v == "white")) for v in roots}
    seg = SegConfig()
    for root in roots.values():
        if root.exists():
            shutil.rmtree(root)
        for split, labels in {"train": ["good"], "val": ["good", "synthetic_defect"],
                              "test": ["good", "synthetic_defect", "defective"]}.items():
            for label in labels:
                (root / split / label).mkdir(parents=True, exist_ok=True)
                (root / split / f"{label}_mask").mkdir(parents=True, exist_ok=True)
    main_root = next(iter(roots.values()))
    (main_root / "preview").mkdir(exist_ok=True)

    writer = Writer(roots, seg, crops, synth=not args.no_synthetic, seed=args.seed)
    cams = [c for c in args.cameras.split(",") if c]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    split_of_group: dict[str, str] = {}

    pose = None
    if args.pose:
        import yaml

        pose = yaml.safe_load((REPO / "configs/poses.yaml").read_text())[args.pose]

    wanted_eps = None
    if args.episodes:
        wanted_eps = set()
        for part in args.episodes.split(","):
            a, _, b = part.partition("-")
            wanted_eps.update(range(int(a), int(b or a) + 1))

    # ---- datasets
    for ds_arg in args.dataset:
        ds = resolve_dataset(ds_arg)
        ds_id = ds.name[:8]
        info = json.loads((ds / "meta/info.json").read_text())
        fps = float(info["fps"])
        episodes = [r for r in load_episodes(ds) if wanted_eps is None or r["episode_index"] in wanted_eps]
        states = load_states(ds) if pose else {}
        defect_re = re.compile(args.defect_task, re.I)
        good_eps = {r["episode_index"]: task_of(r) for r in episodes if not defect_re.search(task_of(r))}
        splits = assign_splits(good_eps, args.val_frac, args.test_frac, args.seed)
        n_prev = 0
        for r in episodes:
            ep, task = r["episode_index"], task_of(r)
            is_defect = ep not in good_eps
            split = "test" if is_defect else splits[ep]
            label = "defective" if is_defect else "good"
            split_of_group[f"{ds_id}/ep{ep:03d}"] = split
            frames = list(range(0, int(r["length"]), args.every))
            if pose is not None:
                ok = pose_mask(states[ep], pose, args.pose_tol)
                frames = [f for f in frames if f < len(ok) and ok[f]]
            n_ep = 0
            for cam in cams:
                key = f"observation.images.{cam}"
                if f"videos/{key}/file_index" not in r:
                    continue
                video = ds / "videos" / key / f"chunk-{r[f'videos/{key}/chunk_index']:03d}" / f"file-{r[f'videos/{key}/file_index']:03d}.mp4"
                t_from = float(r[f"videos/{key}/from_timestamp"])
                for i, frame in decode_frames(video, [t_from + f / fps for f in frames]):
                    fr = frames[i]
                    base = {"source": f"dataset:{ds_id}", "scene": f"{ds_id}/ep{ep:03d}/f{fr:05d}", "group": f"{ds_id}/ep{ep:03d}",
                            "episode": ep, "frame": fr, "task": task, "camera": cam}
                    prev = None
                    if n_prev < args.preview and rng.random() < 0.02:
                        prev = main_root / "preview" / f"{ds_id}_ep{ep:03d}_f{fr:05d}_{cam}.jpg"
                        n_prev += 1
                    n_ep += writer.add_frame(frame, base, split, label, args.max_bricks, prev)
            print(f"{ds_id} ep {ep:3d} {task[:20]:20s} {split:5s} {label:9s} frames {len(frames):3d} crops {n_ep:4d} "
                  f"({time.time() - t0:.0f} s)", flush=True)

    # ---- state snapshots
    projects: list[Path] = []
    for s in args.snapshots:
        projects += sorted(p for p in SNAPSHOTS.iterdir() if p.is_dir()) if s == "all" else sorted(SNAPSHOTS.glob(f"{s}*"))
    snap_dirs = [d for p in projects for d in sorted(p.iterdir()) if (d / "state.json").exists()]
    snap_split = {}
    if args.snapshot_role == "split":
        snap_split = assign_splits({i: "snap" for i in range(len(snap_dirs))}, args.val_frac, args.test_frac, args.seed)
    for i, d in enumerate(snap_dirs):
        state = json.loads((d / "state.json").read_text())
        defective = bool(state.get("defective"))
        label = "defective" if defective else "good"
        split = "test" if defective or args.snapshot_role == "test" else snap_split[i]
        split_of_group[f"snap/{d.name}"] = split
        n = 0
        for cam in cams:
            img_path = d / f"{cam}.color.png"
            if not img_path.exists():
                continue
            frame = cv2.imread(str(img_path))
            base = {"source": f"snapshot:{d.parent.name[:8]}", "scene": f"snap/{d.name}", "group": f"snap/{d.name}",
                    "episode": "", "frame": "", "task": state.get("comment") or "", "camera": cam}
            prev = main_root / "preview" / f"snap_{d.name}_{cam}.jpg" if defective else None
            n += writer.add_frame(frame, base, split, label, args.max_bricks, prev)
        print(f"snapshot {d.name} {split:5s} {label:9s} crops {n}", flush=True)

    # ---- index, config, summary
    fields = ["file", "split", "label", "source", "scene", "group", "episode", "frame", "task", "camera",
              "bbox_x", "bbox_y", "bbox_w", "bbox_h", "square_x", "square_y", "square_side", "colour",
              "mask_area", "ring_white", "fill", "sharpness", "parent",
              "synth_kind", "synth_width_px", "synth_width_frac", "synth_opacity", "synth_area_frac"]
    for variant, root in roots.items():
        with (root / "index.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(writer.rows)
        cfg = {"set": root.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": vars(args),
               "seg": asdict(seg), "crop": asdict(crops[variant]), "groups": split_of_group}
        (root / "config.json").write_text(json.dumps(cfg, indent=1))

    counts = Counter((r["split"], r["label"], r["camera"], r["colour"]) for r in writer.rows)
    by_split = Counter((r["split"], r["label"]) for r in writer.rows)
    print(f"\n{writer.n_frames} images, {writer.n_empty} without a brick, {len(writer.rows)} crops "
          f"in {time.time() - t0:.0f} s -> {', '.join(str(r) for r in roots.values())}")
    for k in sorted(by_split):
        print(f"  {k[0]:5s} {k[1]:17s} {by_split[k]:5d}")
    good = [r for r in writer.rows if r["label"] == "good"]
    colours = sorted({r["colour"] for r in good})
    print("\ngood crops per colour x camera (all splits):")
    print(f"  {'colour':16s}" + "".join(f"{c:>7s}" for c in cams) + "   total")
    for col in colours:
        n = [sum(1 for r in good if r["colour"] == col and r["camera"] == c) for c in cams]
        print(f"  {col:16s}" + "".join(f"{x:7d}" for x in n) + f"   {sum(n):5d}")
    with (main_root / "counts.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "label", "camera", "colour", "n"])
        for k in sorted(counts):
            w.writerow([*k, counts[k]])


if __name__ == "__main__":
    main()

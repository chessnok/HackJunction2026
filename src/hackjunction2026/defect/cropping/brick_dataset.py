"""Build an Anomalib dataset of zoomed single-brick crops from Studio datasets and state snapshots.

Same layout as cropping.make_dataset, so training.train_* and training.evaluate work unchanged:

    captures/anomalib/dataset/<name>/
        train/good/               crops of good bricks (the ONLY training data)
        test/good/                held-out episodes + good snapshots (+ the other snapshots' half)
        test/defect/              crops of defective bricks where the mark is VISIBLE in that crop
        test/defect_clean/        other crops of defective bricks (mark not visible in that crop)
        test/defect_unlabelled/   crops of defective bricks without a per-crop label yet
        index.csv, dataset.json
    captures/anomalib/dataset/<name>-synth/      (--synthetic) SYNTHETIC marker strokes, eval only:
        test/good/ = the test/good crops of <name>; test/defect/ = one stroked crop per test scene;
        test/defect_clean/ = the other crops of that scene

Every sampled camera frame goes through cropping.bricks (the runtime code): bricks found by colour
on the white sheet, square of `--zoom` x the brick (ADR 0008), resized to `--size`.
index.csv columns used by training.evaluate: path, split, cls, session, brick, label, cycle, view.
Here: view = camera; cycle = one scene (a dataset frame or a snapshot), so the per-cycle decision
in evaluate.py (max over the scene's crops) is the brick-level OR of ADR 0008.
Splits are by EPISODE (a snapshot is its own group), never by frame. Defective bricks are never
in train. Per-crop "mark visible" labels come from scripts/label_crops_gemini.py --apply.

Needs numpy, opencv, av, pyarrow: run with Studio's backend Python (read-only use, ADR 0003):
    STUDIO_PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
    nice -n 19 $STUDIO_PY scripts/brick_crops.py --name bricks_v1 --dataset 431a71f7 --snapshots 0fd76ff2 --synthetic
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026.defect.cropping.bricks import CropConfig, SegConfig, crop_brick, draw_bricks, find_bricks, sharpness
from hackjunction2026.defect.cropping.synth import add_marker

DATASETS = Path.home() / ".local/share/physicalai/datasets"
SNAPSHOTS = Path.home() / ".local/share/physicalai/state_snapshots"
OUT = Path("captures/anomalib/dataset")
CAMERAS = ("top", "floor", "arm")
DEFECT_TASK = r"defect|marked|broken"
FIELDS = ["path", "split", "cls", "session", "brick", "label", "cycle", "view", "defect_note", "source",
          "crop_id", "scene", "episode", "frame", "task", "colour", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
          "square_x", "square_y", "square_side", "mask_area", "ring_white", "sharpness", "parent",
          "synth_kind", "synth_width_px", "synth_width_frac", "synth_opacity"]

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
    return sorted(rows, key=lambda r: r["episode_index"])


def load_states(ds: Path) -> dict[int, np.ndarray]:
    """episode -> (n_frames, 6) follower state (Studio normalized joint values)."""
    import pyarrow.parquet as pq

    out: dict[int, list] = defaultdict(list)
    for f in sorted(ds.glob("data/*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for ep, fr, st in zip(t["episode_index"], t["frame_index"], t["observation.state"], strict=True):
            out[ep].append((fr, st))
    return {ep: np.array([s for _, s in sorted(v)], np.float32) for ep, v in out.items()}


def task_of(row: dict) -> str:
    tasks = row.get("tasks") or [""]
    return str(tasks[0]).strip()


def decode_frames(video: Path, timestamps: list[float]):
    """Yield (i, BGR frame) for each requested timestamp (seconds, ascending). GOP is 2, so seeking is cheap."""
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


def pose_ok(states: np.ndarray, pose: dict, tol: float) -> np.ndarray:
    """Frames whose body joints are within `tol` (normalized units) of the pose.

    Wrist roll and gripper are ignored: the inspection pose rotates the wrist.
    """
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    target = np.array([pose["joints"][n] for n in names], np.float32)
    return np.abs(states[:, :4] - target[None, :]).max(axis=1) <= tol


def assign_splits(groups: dict[str, str], test_frac: float, seed: int) -> dict[str, str]:
    """group -> 'train' | 'test', stratified by task name (trailing digits ignored)."""
    by_task: dict[str, list[str]] = defaultdict(list)
    for g, task in groups.items():
        by_task[re.sub(r"[-_ ]?\d+$", "", task.lower().strip())].append(g)
    rng = np.random.default_rng(seed)
    split = {}
    for task in sorted(by_task):
        gs = sorted(by_task[task])
        rng.shuffle(gs)
        n_test = int(round(len(gs) * test_frac)) if len(gs) >= 2 else 0
        if len(gs) >= 3:
            n_test = max(n_test, 1)
        for i, g in enumerate(gs):
            split[g] = "test" if i < n_test else "train"
    return split


# ------------------------------------------------------------------ crops


class Builder:
    def __init__(self, root: Path, seg: SegConfig, crop: CropConfig, max_bricks: int, preview: int, seed: int):
        self.root, self.seg, self.crop, self.max_bricks = root, seg, crop, max_bricks
        self.rows: list[dict] = []
        self.images = 0
        self.empty = 0
        self.preview_left = preview
        self.rng = np.random.default_rng(seed)

    def add(self, frame: np.ndarray, *, split: str, label: str, session: str, brick: str, cycle: str,
            camera: str, scene: str, source: str, extra: dict, preview: bool = False) -> int:
        self.images += 1
        bricks = find_bricks(frame, self.seg, camera=camera)[: self.max_bricks]
        if not bricks:
            self.empty += 1
        if preview or (self.preview_left > 0 and self.rng.random() < 0.03):
            self.preview_left -= 0 if preview else 1
            (self.root / "preview").mkdir(exist_ok=True)
            cv2.imwrite(str(self.root / "preview" / f"{scene.replace('/', '_')}_{camera}.jpg"),
                        draw_bricks(frame, bricks, camera, self.crop.zoom), [cv2.IMWRITE_JPEG_QUALITY, 85])
        for k, b in enumerate(bricks):
            crop, mask, (sx, sy, side) = crop_brick(frame, b, self.crop)
            crop_id = f"{scene.replace('/', '_')}_{camera}_b{k}"
            cls = "good" if label == "good" else "defect_unlabelled"
            rel = Path(split) / cls / f"{crop_id}.png"
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.root / rel), crop)
            mrel = Path("masks") / f"{crop_id}.png"
            (self.root / mrel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.root / mrel), mask)
            self.rows.append({
                "path": rel.as_posix(), "split": split, "cls": cls, "session": session, "brick": brick,
                "label": label, "cycle": cycle, "view": camera, "defect_note": "", "source": source,
                "crop_id": crop_id, "scene": scene, "colour": b.colour,
                "bbox_x": b.bbox[0], "bbox_y": b.bbox[1], "bbox_w": b.bbox[2], "bbox_h": b.bbox[3],
                "square_x": sx, "square_y": sy, "square_side": side, "mask_area": b.area,
                "ring_white": b.ring_white, "sharpness": round(sharpness(crop, mask), 1), **extra,
            })
        return len(bricks)


def subsample(rows: list[dict], keep: set[str], cap: int | None, key: str, strata, seed: int) -> list[dict]:
    """Keep at most `cap` dataset-episode rows of split `keep`, sampling whole `key` groups evenly over
    `strata`. Snapshot rows are always kept."""
    if cap is None:
        return rows
    sel = [r for r in rows if r["split"] in keep and r["label"] == "good" and r["source"].startswith("dataset:")]
    if len(sel) <= cap:
        return rows
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in sel:
        groups[r[key]].append(r)
    by_stratum: dict[tuple, list[str]] = defaultdict(list)
    for g, rs in groups.items():
        by_stratum[strata(rs[0])].append(g)
    rng = np.random.default_rng(seed)
    for gs in by_stratum.values():
        rng.shuffle(gs)
    chosen, n = set(), 0
    while n < cap and any(by_stratum.values()):  # round-robin over strata
        for s in sorted(by_stratum):
            if by_stratum[s] and n < cap:
                g = by_stratum[s].pop()
                chosen.add(g)
                n += len(groups[g])
    drop = {id(r) for r in sel if r[key] not in chosen}
    return [r for r in rows if id(r) not in drop]


def write_index(root: Path, rows: list[dict], summary: dict) -> None:
    with open(root / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (root / "dataset.json").write_text(json.dumps(summary, indent=2))


def build_synthetic(root: Path, rows: list[dict], crop: CropConfig, seed: int) -> Path:
    """<name>-synth: test/good of <name> + one SYNTHETIC stroked crop per test scene (eval only)."""
    out = root.parent / f"{root.name}-synth"
    if out.exists():
        shutil.rmtree(out)
    goods = [r for r in rows if r["split"] == "test" and r["cls"] == "good"]
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in goods:
        by_scene[r["scene"]].append(r)
    rng = np.random.default_rng(seed)
    srows = []
    for r in goods:
        dst = out / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / r["path"], dst)
        srows.append(dict(r))
    for scene in sorted(by_scene):
        rs = by_scene[scene]
        marked = rs[int(rng.integers(len(rs)))]
        for r in rs:
            img = cv2.imread(str(root / r["path"]))
            new = {**r, "label": "defect", "parent": r["path"]}
            if r is marked:
                mask = cv2.imread(str(root / "masks" / f"{r['crop_id']}.png"), cv2.IMREAD_GRAYSCALE)
                srng = np.random.default_rng([seed, zlib.crc32(r["crop_id"].encode())])
                img, stroke, params = add_marker(img, mask, srng, upscale=crop.size / int(r["square_side"]))
                new.update(cls="defect", path=f"test/defect/{r['crop_id']}_synth.png", **params)
                (out / "stroke_masks").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "stroke_masks" / f"{r['crop_id']}_synth.png"), stroke)
            else:
                new.update(cls="defect_clean", path=f"test/defect_clean/{r['crop_id']}_synth.png")
            (out / new["path"]).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / new["path"]), img)
            srows.append(new)
    counts = Counter(f"{r['split']}/{r['cls']}" for r in srows)
    write_index(out, srows, {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "synthetic": True,
                             "note": "SYNTHETIC marker strokes on test/good crops of " + root.name +
                                     ": evaluation only, never training data",
                             "parent_dataset": root.name, "images": dict(sorted(counts.items()))})
    print(f"synthetic eval set: {dict(sorted(counts.items()))} -> {out}")
    return out


# ------------------------------------------------------------------ main


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_episodes(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    eps: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        eps.update(range(int(a), int(b or a) + 1))
    return eps


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help=f"dataset name (folder under {OUT})")
    p.add_argument("--dataset", action="append", default=[], help="Studio dataset id (prefix ok) or path; repeatable")
    p.add_argument("--snapshots", action="append", default=[], help="snapshot project id (prefix ok); repeatable")
    p.add_argument("--snapshot-train-frac", type=float, default=0.0,
                   help="share of GOOD snapshots put into train/good, in time blocks; the rest are test (default 0)")
    p.add_argument("--snapshot-block", type=int, default=6, help="consecutive good snapshots per split block")
    p.add_argument("--every", type=int, default=10, help="every Nth dataset frame (default 10)")
    p.add_argument("--episodes", help="only these dataset episodes, e.g. 0-9,12")
    p.add_argument("--cameras", default=",".join(CAMERAS))
    p.add_argument("--pose", help="only dataset frames near this pose of configs/poses.yaml (e.g. an inspection pose)")
    p.add_argument("--pose-tol", type=float, default=6.0, help="max joint difference for --pose, normalized units")
    p.add_argument("--defect-task", default=DEFECT_TASK, help="regex on the episode task: defective-brick episodes")
    p.add_argument("--test-frac", type=float, default=0.2, help="share of episodes (per task) held out for test")
    p.add_argument("--max-train", type=int, default=800, help="cap on train/good crops (whole frames, stratified)")
    p.add_argument("--max-test-episodes", type=int, default=450, help="cap on test/good crops from dataset episodes")
    p.add_argument("--size", type=int, default=256, help="crop size = model input (training --image-size)")
    p.add_argument("--zoom", type=float, default=2.4, help="crop square / brick size (ADR 0008)")
    p.add_argument("--background", choices=("white", "raw"), default="white",
                   help="white: everything outside the brick hull is white (default); raw: keep the scene")
    p.add_argument("--max-bricks", type=int, default=2, help="max bricks per image (largest first)")
    p.add_argument("--synthetic", action="store_true", help="also write <name>-synth (SYNTHETIC strokes, eval only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview", type=int, default=30, help="random annotated frames saved to preview/")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    root = OUT / a.name
    if root.exists():
        if not a.overwrite:
            sys.exit(f"{root} exists; use --overwrite or another --name")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    seg = SegConfig()
    crop = CropConfig(size=a.size, zoom=a.zoom, whiten=a.background == "white")
    b = Builder(root, seg, crop, a.max_bricks, a.preview, a.seed)
    cams = [c for c in a.cameras.split(",") if c]
    wanted = parse_episodes(a.episodes)
    pose = None
    if a.pose:
        import yaml

        pose = yaml.safe_load(Path("configs/poses.yaml").read_text())[a.pose]
    t0 = time.time()
    sources = []

    for ds_arg in a.dataset:
        ds = resolve_dataset(ds_arg)
        sid = ds.name[:8]
        fps = float(json.loads((ds / "meta/info.json").read_text())["fps"])
        eps = [r for r in load_episodes(ds) if wanted is None or r["episode_index"] in wanted]
        states = load_states(ds) if pose else {}
        defect_re = re.compile(a.defect_task, re.I)
        good = {f"ep{r['episode_index']:03d}": task_of(r) for r in eps if not defect_re.search(task_of(r))}
        splits = assign_splits(good, a.test_frac, a.seed)
        sources.append({"dataset": ds.name, "episodes": len(eps), "test_episodes": sorted(g for g, s in splits.items() if s == "test")})
        for r in eps:
            ep, task = r["episode_index"], task_of(r)
            g = f"ep{ep:03d}"
            label = "good" if g in good else "defect"
            split = splits.get(g, "test")
            frames = list(range(0, int(r["length"]), a.every))
            if pose is not None:
                ok = pose_ok(states[ep], pose, a.pose_tol)
                frames = [f for f in frames if f < len(ok) and ok[f]]
            n = 0
            for cam in cams:
                key = f"observation.images.{cam}"
                if f"videos/{key}/file_index" not in r:
                    continue
                video = (ds / "videos" / key / f"chunk-{r[f'videos/{key}/chunk_index']:03d}"
                         / f"file-{r[f'videos/{key}/file_index']:03d}.mp4")
                t_from = float(r[f"videos/{key}/from_timestamp"])
                for i, frame in decode_frames(video, [t_from + f / fps for f in frames]):
                    fr = frames[i]
                    n += b.add(frame, split=split, label=label, session=sid, brick=g, cycle=str(fr), camera=cam,
                               scene=f"{sid}/{g}/f{fr:05d}", source=f"dataset:{ds.name}",
                               extra={"episode": ep, "frame": fr, "task": task})
            print(f"{sid} {g} {task[:18]:18s} {split:5s} {label:6s} frames {len(frames):3d} crops {n:4d} "
                  f"({time.time() - t0:.0f} s)", flush=True)

    for s in a.snapshots:
        projects = sorted(SNAPSHOTS.glob(f"{s}*"))
        if len(projects) != 1:
            sys.exit(f"snapshot project '{s}': {len(projects)} matches in {SNAPSHOTS}")
        proj = projects[0]
        dirs = [d for d in sorted(proj.iterdir()) if (d / "state.json").exists()]
        states = {d.name: json.loads((d / "state.json").read_text()) for d in dirs}
        good = {d.name: "snap" for d in dirs if not states[d.name].get("defective")}
        snap_split = dict.fromkeys(good, "test")
        if a.snapshot_train_frac > 0:
            # Consecutive snapshots often show the same brick seconds apart: split whole time blocks.
            # Stratify the blocks by capture hour (snapshot id YYYYMMDD-HHMMSS-mmm), so both splits get
            # early and late good snapshots (lighting and set-up drift over the session).
            names = sorted(good)
            blocks = {f"block{i // a.snapshot_block:03d}": f"h{n[9:11]}x" for i, n in enumerate(names)}
            block_split = assign_splits(blocks, 1.0 - a.snapshot_train_frac, a.seed)
            snap_split = {n: block_split[f"block{i // a.snapshot_block:03d}"] for i, n in enumerate(names)}
        n_by = Counter()
        for d in dirs:
            st = states[d.name]
            label = "defect" if st.get("defective") else "good"
            split = "test" if label == "defect" else snap_split[d.name]
            for cam in cams:
                img = cv2.imread(str(d / f"{cam}.color.png"))
                if img is None:
                    continue
                n_by[(split, label)] += b.add(img, split=split, label=label, session=f"snap-{proj.name[:8]}",
                                              brick=d.name, cycle="0", camera=cam, scene=f"snap/{d.name}",
                                              source=f"snapshot:{proj.name}/{d.name}",
                                              extra={"task": st.get("comment") or ""}, preview=label == "defect")
        sources.append({"snapshots": proj.name, "n": len(dirs), "defective": len(dirs) - len(good),
                        "crops": {f"{k[0]}/{k[1]}": v for k, v in n_by.items()}})
        print(f"snapshots {proj.name[:8]}: {len(dirs)} ({len(dirs) - len(good)} defective), crops {dict(n_by)}")

    # caps: keep training and evaluation sizes sane (PatchCore keeps every training patch in RAM)
    rows = subsample(b.rows, {"train"}, a.max_train, "scene", lambda r: (r["view"], r["colour"]), a.seed)
    ep_test = [r for r in rows if r["split"] == "test" and r["source"].startswith("dataset:")]
    if a.max_test_episodes and len(ep_test) > a.max_test_episodes:
        keep_rows = subsample(ep_test, {"test"}, a.max_test_episodes, "scene", lambda r: (r["view"], r["colour"]), a.seed)
        keep = {id(r) for r in keep_rows}
        rows = [r for r in rows if not (r["split"] == "test" and r["source"].startswith("dataset:")) or id(r) in keep]
    kept = {r["path"] for r in rows}
    for r in b.rows:
        if r["path"] not in kept:
            (root / r["path"]).unlink(missing_ok=True)
            (root / "masks" / f"{r['crop_id']}.png").unlink(missing_ok=True)

    counts = Counter(f"{r['split']}/{r['cls']}" for r in rows)
    per_cam_colour = Counter((r["split"], r["view"], r["colour"]) for r in rows if r["label"] == "good")
    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_sha(), "builder": "cropping.brick_dataset",
        "sources": sources, "image_size": a.size, "zoom": a.zoom, "background": a.background,
        "seg": asdict(seg), "crop": asdict(crop), "args": vars(a),
        "frames_read": b.images, "frames_without_brick": b.empty, "images": dict(sorted(counts.items())),
        "good_per_split_camera_colour": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in sorted(per_cam_colour.items())},
    }
    write_index(root, rows, summary)
    print(f"\n{b.images} images read, {b.empty} without a brick, {len(rows)} crops kept in {time.time() - t0:.0f} s")
    print(json.dumps(summary["images"]))
    goods = [r for r in rows if r["label"] == "good"]
    print("\ngood crops per colour x camera (train / test):")
    print(f"  {'colour':16s}" + "".join(f"{c:>12s}" for c in cams))
    for col in sorted({r["colour"] for r in goods}):
        cells = [f"{sum(1 for r in goods if r['colour'] == col and r['view'] == c and r['split'] == 'train')}"
                 f"/{sum(1 for r in goods if r['colour'] == col and r['view'] == c and r['split'] == 'test')}" for c in cams]
        print(f"  {col:16s}" + "".join(f"{x:>12s}" for x in cells))
    if counts.get("test/defect_unlabelled"):
        print("\nnote: defective-brick crops are in test/defect_unlabelled until labelled per crop:\n"
              f"  uv run scripts/label_crops_gemini.py --dataset {a.name} --apply")
    if a.synthetic:
        build_synthetic(root, rows, crop, a.seed)
    print(f"dataset written to {root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export the first frame of every camera for every episode of a Studio (LeRobot v3) dataset.

Output: <out>/ep<NNN>_<task>/<camera>.png (one folder per episode) and <out>/index.csv.
Read-only on the dataset. Depth videos (<cam>_depth) are skipped.

Run with Studio's backend Python (it has av, pyarrow, Pillow), from the repo root:
  PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
  $PY scripts/episode_first_frames.py 431a71f7            # dataset id (prefix ok) or a path
  $PY scripts/episode_first_frames.py 431a71f7 --out captures/first_frames/my_set
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import av
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
DATASETS = Path.home() / ".local/share/physicalai/datasets"


def resolve_dataset(arg: str) -> Path:
    path = Path(arg).expanduser()
    if (path / "meta/info.json").exists():
        return path
    matches = sorted(DATASETS.glob(f"{arg}*/meta/info.json"))
    if len(matches) != 1:
        sys.exit(f"dataset '{arg}': {len(matches)} matches in {DATASETS}")
    return matches[0].parent.parent


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "task"


def first_frame(video: Path, from_ts: float):
    """Decoded frame of `video` at (or right after) `from_ts` seconds."""
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(from_ts / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time >= from_ts - 1e-3:
                return frame
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset id (prefix ok) or path")
    ap.add_argument("--out", help="output dir (default: captures/first_frames/<dataset id>)")
    args = ap.parse_args()

    ds = resolve_dataset(args.dataset)
    info = json.loads((ds / "meta/info.json").read_text())
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video" and not k.endswith("_depth")]
    out = Path(args.out) if args.out else REPO / "captures/first_frames" / ds.name
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(glob.glob(str(ds / "meta/episodes/*/*.parquet"))):
        rows += pq.read_table(f).to_pylist()
    rows.sort(key=lambda r: r["episode_index"])

    index, missing = [], 0
    for r in rows:
        ep, tasks = r["episode_index"], r.get("tasks") or []
        task = tasks[0] if tasks else ""
        folder = out / f"ep{ep:03d}_{slug(task)}"
        folder.mkdir(exist_ok=True)
        files = {}
        for key in cams:
            cam = key.split(".")[-1]
            video = ds / "videos" / key / f"chunk-{r[f'videos/{key}/chunk_index']:03d}" / f"file-{r[f'videos/{key}/file_index']:03d}.mp4"
            frame = first_frame(video, float(r[f"videos/{key}/from_timestamp"]))
            if frame is None:
                missing += 1
                print(f"  ep {ep}: no frame for {cam}")
                continue
            frame.to_image().save(folder / f"{cam}.png")
            files[cam] = f"{folder.name}/{cam}.png"
        index.append({"episode": ep, "task": task, "length_frames": r.get("length"), **files})
        print(f"ep {ep:3d} {task:24} -> {folder.name} ({len(files)} images)")

    with (out / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["episode", "task", "length_frames", *[k.split(".")[-1] for k in cams]])
        w.writeheader()
        w.writerows(index)
    print(f"\n{len(index)} episodes, {sum(len(i) - 3 for i in index)} images, {missing} missing -> {out}")


if __name__ == "__main__":
    main()

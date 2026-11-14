"""Read-only access to a Studio (LeRobot v3.0) dataset: episode metadata and decoded video frames.

Several episodes share one AV1 mp4 per camera; `meta/episodes/*/*.parquet` gives each episode's file
and its [from_timestamp, to_timestamp) window. Frame `i` of an episode is the decoded frame whose
pts is closest to `from_timestamp + i / fps`.
"""

from __future__ import annotations

import glob
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

DATASETS = Path.home() / ".local/share/physicalai/datasets"
CAMERAS = ("top", "floor", "arm")


def resolve_dataset(arg: str) -> Path:
    path = Path(arg).expanduser()
    if (path / "meta/info.json").exists():
        return path
    matches = sorted(DATASETS.glob(f"{arg}*/meta/info.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"dataset '{arg}': {len(matches)} matches in {DATASETS}")
    return matches[0].parent.parent


@dataclass
class Episode:
    index: int
    task: str
    length: int
    video: dict[str, Path]  # camera -> mp4 file
    from_ts: dict[str, float]  # camera -> start time inside that file


class Dataset:
    def __init__(self, arg: str):
        self.root = resolve_dataset(arg)
        self.id = self.root.name
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = float(self.info["fps"])
        rows = []
        for f in sorted(glob.glob(str(self.root / "meta/episodes/*/*.parquet"))):
            rows += pq.read_table(f).to_pylist()
        rows.sort(key=lambda r: r["episode_index"])
        self.episodes: list[Episode] = []
        for r in rows:
            video, from_ts = {}, {}
            for cam in CAMERAS:
                key = f"observation.images.{cam}"
                if f"videos/{key}/chunk_index" not in r:
                    continue
                video[cam] = (
                    self.root / "videos" / key / f"chunk-{r[f'videos/{key}/chunk_index']:03d}"
                    / f"file-{r[f'videos/{key}/file_index']:03d}.mp4"
                )
                from_ts[cam] = float(r[f"videos/{key}/from_timestamp"])
            tasks = r.get("tasks") or [""]
            self.episodes.append(Episode(r["episode_index"], tasks[0].strip(), int(r["length"]), video, from_ts))

    def states(self):
        """DataFrame with episode_index, frame_index, timestamp, observation.state, action."""
        import pandas as pd

        parts = [pq.read_table(f).to_pandas() for f in sorted(glob.glob(str(self.root / "data/*/*.parquet")))]
        return pd.concat(parts).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    def iter_frames(self, ep: int, cam: str, frames: set[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, RGB uint8 HxWx3) for an episode, decoding sequentially.

        `frames`: only yield these frame indices (all are still decoded; AV1 here has g=2 so
        seeking saves little). Frames are matched by pts to from_ts + i / fps.
        """
        e = self.episodes[ep]
        t0 = e.from_ts[cam]
        with av.open(str(e.video[cam])) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(int(max(t0 - 0.2, 0) / stream.time_base), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                i = int(round((frame.time - t0) * self.fps))
                if i < 0:
                    continue
                if i >= e.length:
                    break
                if frames is None or i in frames:
                    yield i, frame.to_ndarray(format="rgb24")
                    if frames is not None and i >= max(frames):
                        break

    def frame(self, ep: int, cam: str, i: int) -> np.ndarray:
        for _, img in self.iter_frames(ep, cam, {i}):
            return img
        raise IndexError(f"episode {ep} camera {cam}: no frame {i}")

"""Build the merged LeRobot v3 dataset: NEW (minus accidents) + the tail of every OLD episode.

Runs in Studio's backend venv (LeRobot 0.6, PyAV). Read-only on both sources. Steps, all into a
work folder:

1. ``new_src``: a copy of NEW (the team may record into it while we build), checked for
   consistency (info.json = episodes metadata = data rows).
2. ``new_clean``: LeRobot's ``delete_episodes`` drops the accidental short NEW episodes. Its video
   step re-encodes a file that mixes kept and dropped episodes; here it is swapped for a lossless
   stream copy (``stream_copy_ranges``): every kept episode starts on a keyframe (each episode is
   encoded on its own, g = 2), so its packets are copied unchanged with shifted timestamps.
3. ``old_tails/ep<NNN>_cut<CCCC>``: each OLD episode from its cut frame to its end, one
   single-episode dataset per (episode, cut), kept between builds: a rebuild after a changed
   override encodes only the episodes whose cut changed. Written by LeRobot's own dataset writer
   (``LeRobotDataset.create`` / ``add_frame`` / ``save_episode``, streaming encoder, the way Studio
   records), encoder rebuilt from OLD's ``info.json`` with LeRobot's ``encoder_config_from_video_info``
   (libsvtav1, yuv420p, g 2, crf 30, preset 12) and Studio's write arguments
   (``VideoEncodingSettings.to_lerobot_write_kwargs``). frame_index and timestamp restart at 0;
   per-episode stats are computed by the writer. ``seed_cache`` fills the cache from an earlier
   merged dataset without re-encoding.
4. ``aggregate_datasets`` (LeRobot) merges ``new_clean`` + the tails: task table, indices, episode
   metadata, video files (stream-copy concatenation) and ``stats.json``.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import av
import numpy as np
import pandas as pd

from hackjunction2026.grasp_merge.clips import episode_frames, wait_for_load

STUDIO_BACKEND_SRC = Path("/home/intel/dev/third_party/physical-ai-studio/application/backend/src")
DEFAULT_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


# ---------------------------------------------------------------- helpers


def read_episodes_meta(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta/episodes").glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index").set_index("episode_index")


def read_data(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("index").reset_index(drop=True)


def consistency(root: Path) -> dict:
    """Counts that must agree in a finished LeRobot v3 dataset; raises if they do not."""
    info = json.loads((root / "meta/info.json").read_text())
    eps = read_episodes_meta(root)
    data = read_data(root)
    counts = {"info_episodes": info["total_episodes"], "info_frames": info["total_frames"],
              "meta_episodes": len(eps), "meta_frames": int(eps.length.sum()), "data_rows": len(data),
              "data_episodes": int(data.episode_index.nunique())}
    if not (counts["info_episodes"] == counts["meta_episodes"] == counts["data_episodes"]
            and counts["info_frames"] == counts["meta_frames"] == counts["data_rows"]):
        raise RuntimeError(f"{root}: inconsistent dataset (being written?): {counts}")
    return counts


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------- 1. snapshot of NEW


def snapshot(src: Path, dst: Path, log=print) -> dict:
    for attempt in range(3):
        if dst.exists():
            shutil.rmtree(dst)
        before = (src / "meta/info.json").stat().st_mtime_ns
        shutil.copytree(src, dst)
        after = (src / "meta/info.json").stat().st_mtime_ns
        if before == after:
            counts = consistency(dst)
            log(f"snapshot {src.name}: {counts['info_episodes']} episodes, {counts['info_frames']} frames")
            return counts
        log(f"snapshot {src.name}: info.json changed while copying (attempt {attempt + 1}), retrying in 30 s")
        time.sleep(30)
    raise RuntimeError(f"{src} keeps changing; is someone recording into it?")


# ---------------------------------------------------------------- 2. drop short NEW episodes, lossless


def stream_copy_ranges(input_path: Path, output_path: Path, episodes_to_keep: list[tuple[int, int]],
                       fps: float, video_encoder=None) -> None:
    """Drop-in for LeRobot's ``_keep_episodes_from_video_with_av`` without re-encoding.

    Copies the packets of the half-open frame ranges ``[start, end)`` in order and shifts their
    timestamps so the output is contiguous from 0, like LeRobot's own remux in
    ``concatenate_video_files``. Every range must start on a keyframe; otherwise this raises
    (decoding would need frames that are not copied).
    """
    ranges = sorted(episodes_to_keep)
    starts = {s for s, _ in ranges}
    expected = sum(e - s for s, e in ranges)
    copied, r = 0, 0
    with av.open(str(input_path)) as inp:
        vin = inp.streams.video[0]
        tb = vin.time_base
        with av.open(str(output_path), "w", options={"movflags": "faststart"}) as out:
            vout = out.add_stream_from_template(template=vin, opaque=True)
            vout.time_base = tb
            for pkt in inp.demux(vin):
                if pkt.dts is None or pkt.pts is None:
                    continue
                i = round(float(pkt.pts * tb) * fps)
                while r < len(ranges) and i >= ranges[r][1]:
                    r += 1
                if r >= len(ranges):
                    break
                if i < ranges[r][0]:
                    continue
                if i in starts and not pkt.is_keyframe:
                    raise ValueError(f"{input_path}: kept range starting at frame {i} does not start on a keyframe")
                shift = pkt.pts - round(copied / fps / tb)
                pkt.pts -= shift
                pkt.dts -= shift
                pkt.stream = vout
                out.mux(pkt)
                copied += 1
    if copied != expected:
        raise RuntimeError(f"{input_path}: copied {copied} packets, expected {expected}")


@contextlib.contextmanager
def lossless_episode_removal():
    import lerobot.datasets.dataset_tools as dt

    original = dt._keep_episodes_from_video_with_av
    dt._keep_episodes_from_video_with_av = stream_copy_ranges
    try:
        yield
    finally:
        dt._keep_episodes_from_video_with_av = original


def drop_episodes(src: Path, drop: list[int], dst: Path, log=print) -> None:
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if dst.exists():
        shutil.rmtree(dst)
    if not drop:
        shutil.copytree(src, dst)
        return
    ds = LeRobotDataset("grasp-merge/new-src", root=src)
    with lossless_episode_removal():
        delete_episodes(ds, episode_indices=drop, output_dir=dst, repo_id="grasp-merge/new-clean")
    log(f"dropped NEW episodes {drop} (videos stream-copied): {consistency(dst)}")


# ---------------------------------------------------------------- 3. OLD tails through LeRobot's writer


def write_kwargs(info: dict, threads: int, queue: int) -> dict:
    """Studio's recording writer arguments, with the RGB encoder rebuilt from the source info.json."""
    from lerobot.configs.video import encoder_config_from_video_info

    if str(STUDIO_BACKEND_SRC) not in sys.path:
        sys.path.insert(0, str(STUDIO_BACKEND_SRC))
    from internal_datasets.lerobot.video_encoding_settings import VideoEncodingSettings

    videos = {k: f["info"] for k, f in info["features"].items() if f["dtype"] == "video"}
    encoders = {k: encoder_config_from_video_info(v) for k, v in videos.items()}
    first = next(iter(encoders.values()))
    if any(vars(e) != vars(first) for e in encoders.values()):
        raise RuntimeError(f"cameras were encoded differently: { {k: vars(e) for k, e in encoders.items()} }")
    kw = VideoEncodingSettings(streaming_encoding=True, encoder_threads=threads,
                               encoder_queue_maxsize=queue).to_lerobot_write_kwargs({})
    kw["rgb_encoder"] = first
    return kw


def tail_dir(cache: Path, episode: int, cut: int) -> Path:
    """Cache folder of one OLD tail: a single-episode LeRobot dataset, keyed by episode and cut."""
    return cache / f"ep{episode:03d}_cut{cut:04d}"


def _cached(d: Path, frames: int) -> bool:
    try:
        c = consistency(d)
    except (OSError, RuntimeError, ValueError, KeyError):
        return False
    return c["info_episodes"] == 1 and c["info_frames"] == frames


def write_tail(old_root: Path, c: dict, dst: Path, kw: dict, info: dict, meta: pd.DataFrame, data: pd.DataFrame,
               threads: int) -> None:
    """One OLD episode from its cut to its end as a single-episode dataset, through LeRobot's writer."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = float(info["fps"])
    features = {k: {kk: vv for kk, vv in f.items() if kk != "info"} for k, f in info["features"].items()
                if k not in DEFAULT_KEYS}
    video_keys = [k for k, f in features.items() if f["dtype"] == "video"]
    ep, cut, n = c["episode"], c["cut_frame"], c["length"]
    tmp = dst.with_name(dst.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    ds = LeRobotDataset.create(repo_id=f"grasp-merge/old-{ep:03d}", fps=int(fps), features=features, root=tmp,
                               robot_type=info["robot_type"], use_videos=True, **kw)
    try:
        m = meta.loc[ep]
        rows = data[data.episode_index == ep].sort_values("frame_index")
        assert len(rows) == n and (rows.frame_index.to_numpy() == np.arange(n)).all(), f"OLD episode {ep}"
        rows = rows.iloc[cut:]
        gens = {}
        for k in video_keys:
            video = old_root / f"videos/{k}/chunk-{int(m[f'videos/{k}/chunk_index']):03d}" \
                / f"file-{int(m[f'videos/{k}/file_index']):03d}.mp4"
            gens[k] = episode_frames(video, float(m[f"videos/{k}/from_timestamp"]), n, fps, None, threads, start=cut)
        for fi, st, ac in zip(rows.frame_index, rows["observation.state"], rows["action"], strict=True):
            frame = {"observation.state": np.asarray(st, np.float32), "action": np.asarray(ac, np.float32),
                     "task": c["task"]}
            for k, g in gens.items():
                i, img = next(g)
                if i != fi:
                    raise RuntimeError(f"OLD episode {ep} {k}: decoded frame {i}, expected {fi}")
                frame[k] = img
            ds.add_frame(frame)
        for k, g in gens.items():
            if next(g, None) is not None:
                raise RuntimeError(f"OLD episode {ep} {k}: more video frames than data rows")
        ds.save_episode()
        enc = ds.writer._streaming_encoder  # LeRobot 0.6 drops frames on a full queue (T-022): must be none
        dropped = dict(getattr(enc, "_dropped_frames", {}) or {})
        if any(dropped.values()):
            raise RuntimeError(f"OLD episode {ep}: encoder dropped frames {dropped}")
    finally:
        ds.finalize()
    if not _cached(tmp, n - cut):
        raise RuntimeError(f"OLD episode {ep}: tail dataset inconsistent: {consistency(tmp)}")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)


def ensure_tails(old_root: Path, cuts: list[dict], cache: Path, threads: int = 2, max_load: float = 8.0,
                 max_wait_s: float = 180.0, log=print) -> tuple[list[Path], list[int]]:
    """Cache folders of every kept OLD tail, in episode order; encodes only the missing ones."""
    keep = [c for c in cuts if c["source"] != "skip"]
    dirs = [tail_dir(cache, c["episode"], c["cut_frame"]) for c in keep]
    todo = [c for c, d in zip(keep, dirs, strict=True) if not _cached(d, c["kept_frames"])]
    log(f"OLD tails: {len(keep) - len(todo)} cached in {cache}, {len(todo)} to encode")
    if todo:
        info = json.loads((old_root / "meta/info.json").read_text())
        queue = max(c["kept_frames"] for c in todo) + 64  # an episode never waits on a full queue: no drops
        kw = write_kwargs(info, threads, queue)
        log(f"  encoder {kw['rgb_encoder']}, {threads} threads per camera encoder, queue {queue}")
        meta, data = read_episodes_meta(old_root), read_data(old_root)
        t_all = time.time()
        for c in todo:
            waited = wait_for_load(max_load, max_wait_s, log)
            t0 = time.time()
            write_tail(old_root, c, tail_dir(cache, c["episode"], c["cut_frame"]), kw, info, meta, data, threads)
            log(f"  OLD ep {c['episode']:3d}: frames {c['cut_frame']}..{c['length'] - 1} -> {c['kept_frames']} "
                f"frames, {time.time() - t0:.1f} s" + (f" (waited {waited:.0f} s for load)" if waited > 1 else ""))
        log(f"  encoded {len(todo)} tails in {time.time() - t_all:.0f} s")
    return dirs, [c["episode"] for c in todo]


def seed_cache(merged_root: Path, cache: Path, log=print) -> int:
    """Split the OLD tails of an existing merged dataset into the cache, without re-encoding.

    Each tail was encoded on its own, so it starts on a keyframe in the merged video files and
    LeRobot's ``delete_episodes`` with the stream copy keeps it bit-identical.
    """
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    prov = json.loads((merged_root / "meta/grasp_merge.json").read_text())
    part = prov["parts"][1]
    a, b = part["merged_episodes"]
    ds = LeRobotDataset("grasp-merge/seed", root=merged_root)
    meta = read_episodes_meta(merged_root)
    n_all = ds.meta.total_episodes
    added = 0
    for i, src_ep in zip(range(a, b + 1), part["source_episodes"], strict=True):
        d = tail_dir(cache, src_ep, int(part["cuts"][str(src_ep)]))
        if _cached(d, int(meta.loc[i, "length"])):
            continue
        tmp = d.with_name(d.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        with lossless_episode_removal():
            delete_episodes(ds, episode_indices=[j for j in range(n_all) if j != i], output_dir=tmp,
                            repo_id=f"grasp-merge/old-{src_ep:03d}")
        if not _cached(tmp, int(meta.loc[i, "length"])):
            raise RuntimeError(f"seed of OLD episode {src_ep} inconsistent: {consistency(tmp)}")
        tmp.rename(d)
        added += 1
    log(f"cache {cache}: {added} tails added from {merged_root.name}")
    return added


# ---------------------------------------------------------------- 4. merge


def merge(parts: list[Path], dst: Path, name: str, log=print) -> dict:
    from lerobot.datasets.aggregate import aggregate_datasets

    if dst.exists():
        raise FileExistsError(f"{dst} exists; remove it or choose another --name")
    aggregate_datasets(repo_ids=[f"grasp-merge/part{i}" for i in range(len(parts))], aggr_repo_id=name,
                       roots=parts, aggr_root=dst)
    counts = consistency(dst)
    log(f"merged -> {dst}: {counts}")
    return counts


def build(new_root: Path, old_root: Path, cuts_doc: dict, work: Path, dst: Path, name: str, threads: int = 2,
          max_load: float = 8.0, max_wait_s: float = 180.0, keep_work: bool = False, cache: Path | None = None,
          new_exclude: tuple[int, ...] = (), log=print) -> dict:
    """``new_exclude``: NEW episodes to leave out besides the short ones (e.g. not grasp demos)."""
    work.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; remove it or choose another --name")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    new_counts = snapshot(new_root, work / "new_src", log)
    new_eps = read_episodes_meta(work / "new_src")
    min_len = cuts_doc["params"]["min_episode_frames"]
    short = [int(i) for i in new_eps.index[new_eps.length < min_len]]
    unknown = sorted(set(new_exclude) - {int(i) for i in new_eps.index})
    if unknown:
        raise ValueError(f"--new-exclude: NEW has no episodes {unknown}")
    drop = sorted(set(short) | set(new_exclude))
    drop_episodes(work / "new_src", drop, work / "new_clean", log)
    cache = cache or work / "old_tails"
    tails, encoded = ensure_tails(old_root, cuts_doc["episodes"], cache, threads, max_load, max_wait_s, log)
    merged = merge([work / "new_clean", *tails], dst, name, log)
    kept = [c for c in cuts_doc["episodes"] if c["source"] != "skip"]
    old_frames = sum(c["kept_frames"] for c in kept)
    prov = {
        "format": "hackjunction2026/grasp-merge-provenance/v1", "name": name, "built_utc": started,
        "code_commit": git_commit(), "doc": "docs/datasets/grasp-merged.md",
        "episodes": merged["info_episodes"], "frames": merged["info_frames"],
        "parts": [
            {"source": str(new_root), "dataset": new_root.name, "role": "NEW, whole episodes",
             "merged_episodes": [0, new_counts["info_episodes"] - len(drop) - 1],
             "source_episodes": [int(i) for i in new_eps.index if int(i) not in drop],
             "frames": new_counts["info_frames"] - int(new_eps.loc[drop].length.sum()) if drop else
             new_counts["info_frames"],
             "excluded": [{"episode": i, "length": int(new_eps.loc[i].length),
                           "task": next(iter(new_eps.loc[i, "tasks"]), ""),
                           "reason": f"shorter than {min_len} frames" if i in short else "--new-exclude"}
                          for i in drop],
             "snapshot": new_counts, "video": "stream copy (lossless)"},
            {"source": str(old_root), "dataset": old_root.name, "role": "OLD, tail [cut_frame, end)",
             "merged_episodes": [new_counts["info_episodes"] - len(drop), merged["info_episodes"] - 1],
             "source_episodes": [c["episode"] for c in kept], "frames": old_frames,
             "encoded_this_build": encoded, "cache": str(cache),
             "skipped": [c["episode"] for c in cuts_doc["episodes"] if c["source"] == "skip"],
             "cuts": {str(c["episode"]): c["cut_frame"] for c in kept},
             "video": "re-encoded with the source encoder settings (LeRobot writer)"},
        ],
        "criterion": cuts_doc["criterion"], "overrides": cuts_doc.get("overrides", {}),
    }
    (dst / "meta/grasp_merge.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    if not keep_work:
        for d in ("new_src", "new_clean"):
            shutil.rmtree(work / d, ignore_errors=True)
    return prov

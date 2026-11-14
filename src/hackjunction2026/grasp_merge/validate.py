"""Checks of a merged dataset against its sources.

1. Loads with LeRobot's ``LeRobotDataset`` and with Studio's own wrapper ``InternalLeRobotDataset``.
2. Metadata: episode_index / index contiguous, frame_index 0..n-1 and timestamp = frame_index / fps
   per episode, info.json = episodes metadata = data rows, stats.json has every feature.
3. Video: per episode and camera, the packets inside [from_timestamp, to_timestamp) = parquet rows;
   per file, packets = sum of its episodes (no orphan frames).
4. Content: state / action / task of every merged episode equal the source rows (OLD from its cut).
5. Decode spot checks through ``LeRobotDataset.__getitem__``: NEW frames bit-identical to the
   source, OLD frames close to the source (PSNR; they were re-encoded).
6. Contact sheet: first frame (top camera) of 10 random merged episodes.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import av
import cv2
import numpy as np

from hackjunction2026.grasp_merge.build import STUDIO_BACKEND_SRC, read_data, read_episodes_meta
from hackjunction2026.grasp_merge.clips import episode_frames


def packet_frames(video: Path, fps: float) -> list[int]:
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        return [round(float(p.pts * s.time_base) * fps) for p in c.demux(s) if p.pts is not None and p.size]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def source_of(prov: dict, merged_ep: int) -> tuple[dict, int, int]:
    """(part, source episode, first source frame) of a merged episode."""
    for part in prov["parts"]:
        a, b = part["merged_episodes"]
        if a <= merged_ep <= b:
            src_ep = part["source_episodes"][merged_ep - a]
            start = int(part.get("cuts", {}).get(str(src_ep), 0))
            return part, src_ep, start
    raise KeyError(merged_ep)


def validate(root: Path, out_dir: Path, n_decode: int = 12, seed: int = 0, log=print) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = random.Random(seed)
    res: dict = {"dataset": str(root), "checks": {}, "errors": []}
    err = res["errors"].append
    info = json.loads((root / "meta/info.json").read_text())
    prov = json.loads((root / "meta/grasp_merge.json").read_text())
    fps = float(info["fps"])
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]

    # 1. loads
    ds = LeRobotDataset("grasp-merge/check", root=root)
    res["checks"]["lerobot_load"] = {"num_episodes": ds.num_episodes, "num_frames": ds.num_frames}
    try:
        if str(STUDIO_BACKEND_SRC) not in sys.path:
            sys.path.insert(0, str(STUDIO_BACKEND_SRC))
        from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset

        studio = InternalLeRobotDataset(root)
        eps = studio.get_episode_infos()
        res["checks"]["studio_load"] = {"episodes": len(eps), "frames": int(sum(e.length for e in eps)),
                                        "tasks": studio.get_tasks()}
    except Exception as exc:  # noqa: BLE001  (report any failure, do not hide it)
        err(f"Studio InternalLeRobotDataset: {type(exc).__name__}: {exc}")
    log(f"load: {res['checks']}")

    # 2. metadata
    meta = read_episodes_meta(root)
    data = read_data(root)
    n_ep = info["total_episodes"]
    if list(meta.index) != list(range(n_ep)) or sorted(data.episode_index.unique()) != list(range(n_ep)):
        err("episode_index not contiguous 0..n-1")
    if (data["index"].to_numpy() != np.arange(len(data))).any():
        err("index not contiguous")
    bad_fi = []
    for ep, d in data.groupby("episode_index"):
        fi = d.frame_index.to_numpy()
        if (fi != np.arange(len(d))).any() or not np.allclose(d.timestamp.to_numpy(), fi / fps, atol=1e-4):
            bad_fi.append(int(ep))
        if len(d) != int(meta.loc[ep, "length"]):
            err(f"episode {ep}: {len(d)} rows, metadata length {meta.loc[ep, 'length']}")
    if bad_fi:
        err(f"frame_index / timestamp not reset in episodes {bad_fi}")
    stats = json.loads((root / "meta/stats.json").read_text())
    missing = [k for k in info["features"] if k not in stats]
    if missing:
        err(f"stats.json lacks {missing}")
    res["checks"]["metadata"] = {"episodes": n_ep, "frames": len(data), "frame_index_reset": not bad_fi,
                                 "stats_features": sorted(stats)}

    # 3. video frame counts
    vc = {"episodes_checked": 0, "mismatches": [], "files": {}}
    for k in video_keys:
        files: dict[Path, list[int]] = {}
        for ep, m in meta.iterrows():
            f = root / f"videos/{k}/chunk-{int(m[f'videos/{k}/chunk_index']):03d}" \
                / f"file-{int(m[f'videos/{k}/file_index']):03d}.mp4"
            files.setdefault(f, []).append(int(ep))
        for f, eps_in in files.items():
            frames = np.array(packet_frames(f, fps))
            total = 0
            for ep in eps_in:
                m = meta.loc[ep]
                a = round(float(m[f"videos/{k}/from_timestamp"]) * fps)
                b = round(float(m[f"videos/{k}/to_timestamp"]) * fps)
                n_v = int(((frames >= a) & (frames < b)).sum())
                total += int(m["length"])
                vc["episodes_checked"] += 1
                if n_v != int(m["length"]) or b - a != int(m["length"]):
                    vc["mismatches"].append({"camera": k, "episode": ep, "video": n_v, "rows": int(m["length"])})
            vc["files"][str(f.relative_to(root))] = {"packets": len(frames), "episode_frames": total}
            if len(frames) != total:
                err(f"{f.relative_to(root)}: {len(frames)} packets, episodes cover {total}")
    if vc["mismatches"]:
        err(f"video/parquet frame count mismatches: {vc['mismatches'][:5]}")
    res["checks"]["video_frame_counts"] = vc
    log(f"video frame counts: {vc['episodes_checked']} episode-camera pairs, {len(vc['mismatches'])} mismatches")

    # 4. content vs sources
    src_cache: dict[str, tuple] = {}
    tasks = {}
    import pandas as pd

    tdf = pd.read_parquet(root / "meta/tasks.parquet")
    tasks = {int(r.task_index): t for t, r in tdf.iterrows()}
    content_bad = []
    for ep, d in data.groupby("episode_index"):
        part, src_ep, start = source_of(prov, int(ep))
        if part["source"] not in src_cache:
            sroot = Path(part["source"])
            src_cache[part["source"]] = (read_data(sroot), read_episodes_meta(sroot))
        sdata, smeta = src_cache[part["source"]]
        s = sdata[sdata.episode_index == src_ep].sort_values("frame_index").iloc[start:]
        same = (len(s) == len(d)
                and np.array_equal(np.stack(s["observation.state"]), np.stack(d["observation.state"]))
                and np.array_equal(np.stack(s["action"]), np.stack(d["action"]))
                and {tasks[int(t)] for t in d.task_index} == {next(iter(smeta.loc[src_ep, "tasks"]))})
        if not same:
            content_bad.append(int(ep))
    if content_bad:
        err(f"state/action/task differ from the source in merged episodes {content_bad}")
    res["checks"]["content_vs_source"] = {"episodes": n_ep, "differ": content_bad}
    log(f"content vs source: {n_ep - len(content_bad)}/{n_ep} episodes identical (state, action, task)")

    # 5. decode spot checks through LeRobotDataset.__getitem__
    spots = []
    first_old = prov["parts"][1]["merged_episodes"][0]
    picks = rng.sample(range(first_old), n_decode // 2) + rng.sample(range(first_old, n_ep), n_decode - n_decode // 2)
    for ep in picks:
        n = int(meta.loc[ep, "length"])
        i = rng.randrange(n)
        item = ds[int(meta.loc[ep, "dataset_from_index"]) + i]
        part, src_ep, start = source_of(prov, ep)
        sroot = Path(part["source"])
        smeta = src_cache[part["source"]][1].loc[src_ep]
        row = {"episode": ep, "frame": i, "source_episode": src_ep, "source_frame": start + i}
        for k in video_keys:
            got = (item[k].numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
            video = sroot / f"videos/{k}/chunk-{int(smeta[f'videos/{k}/chunk_index']):03d}" \
                / f"file-{int(smeta[f'videos/{k}/file_index']):03d}.mp4"
            j, ref = next(episode_frames(video, float(smeta[f"videos/{k}/from_timestamp"]), int(smeta["length"]),
                                         fps, None, 2, start=start + i))
            assert j == start + i
            row[k.rsplit(".", 1)[-1]] = round(psnr(got, ref), 1)
        spots.append(row)
    new_ok = all(all(v == float("inf") or v > 45 for kk, v in r.items() if kk in ("top", "floor", "arm"))
                 for r in spots if r["episode"] < first_old)
    old_min = min(min(r[c] for c in ("top", "floor", "arm")) for r in spots if r["episode"] >= first_old)
    res["checks"]["decode_spot"] = {"samples": spots, "new_identical_or_psnr_gt_45": new_ok,
                                    "old_min_psnr_db": old_min}
    if not new_ok:
        err("NEW frames differ from the source after the merge")
    if old_min < 30:
        err(f"OLD re-encoded frames PSNR {old_min:.1f} dB < 30")
    log(f"decode spot checks: {len(spots)} frames, NEW lossless: {new_ok}, OLD min PSNR {old_min:.1f} dB")

    # 6. contact sheet (top camera only: the other views show people)
    sheet_eps = sorted(rng.sample(range(n_ep), 10))
    tiles = []
    for ep in sheet_eps:
        img = (ds[int(meta.loc[ep, "dataset_from_index"])]["observation.images.top"].numpy().transpose(1, 2, 0) * 255)
        img = cv2.resize(img.round().clip(0, 255).astype(np.uint8), (424, 240))
        part, src_ep, start = source_of(prov, ep)
        label = f"merged {ep} <- {'NEW' if part is prov['parts'][0] else 'OLD'} {src_ep} f{start}"
        cv2.putText(img, label, (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    sheet = np.vstack([np.hstack(tiles[r:r + 2]) for r in range(0, 10, 2)])
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "contact_sheet_first_frames.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    res["checks"]["contact_sheet"] = {"file": str(out_dir / "contact_sheet_first_frames.jpg"),
                                      "merged_episodes": sheet_eps}
    res["ok"] = not res["errors"]
    (out_dir / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"validation {'OK' if res['ok'] else 'FAILED'}: {res['errors']}")
    return res

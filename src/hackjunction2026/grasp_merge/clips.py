"""Review clips: one full OLD episode per clip, small H.264, the proposed cut burned in.

Cameras: ``top`` (424x240) was checked for people (5 frames per episode, all OLD and NEW episodes,
2026-11-14): none. ``arm`` (wrist, 424x318) shows the gripper and the brick best but can show people
in the background: the user accepted that for a team-private review page (2026-11-14). Never commit
the clips (captures/ is gitignored).

Overlay: frame index / time and h, d, gripper at that frame (left), confidence and flags (right),
a red border and "CUT" from the cut frame on, and a timeline bar at the bottom: grey = dropped,
green = kept, red tick = cut, yellow tick = alternative cut (gripper ignored), cyan tick = close,
white = playhead. The text sits where the view is empty: bottom for ``top`` (the arm and the brick
are in the upper half), top for ``arm`` (the jaws fill the bottom).
"""

from __future__ import annotations

import json
import os
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

W, H = 424, 240  # top camera clip size
BAR_H = 8
CLIP = {"top": {"size": (424, 240), "text": "bottom"},
        "arm": {"size": (424, 318), "text": "top"},
        "floor": {"size": (424, 240), "text": "bottom"}}


def wait_for_load(max_load: float, max_wait_s: float, log=print) -> float:
    """Sleep while the 1-minute load average is above ``max_load`` (at most ``max_wait_s``)."""
    t0 = time.time()
    while os.getloadavg()[0] > max_load and time.time() - t0 < max_wait_s:
        log(f"  load {os.getloadavg()[0]:.1f} > {max_load}: pausing")
        time.sleep(20)
    return time.time() - t0


def episode_frames(video: Path, from_ts: float, length: int, fps: float, size: tuple[int, int] | None = None,
                   threads: int = 2, start: int = 0):
    """Yield (frame_index, RGB uint8) for frames ``start..length-1`` of one episode of a LeRobot
    video file, decoded in order (the seek lands on a keyframe at or before ``start``)."""
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = threads
        t0 = from_ts + start / fps
        c.seek(int(max(t0 - 0.2, 0) / s.time_base), stream=s, backward=True, any_frame=False)
        for fr in c.decode(s):
            if fr.time is None:
                continue
            i = round((fr.time - from_ts) * fps)
            if i < start:
                continue
            if i >= length:
                break
            if size:
                yield i, fr.reformat(width=size[0], height=size[1], format="rgb24").to_ndarray()
            else:
                yield i, fr.to_ndarray(format="rgb24")


def _x(i: int, n: int, w: int) -> int:
    return round(i / max(n - 1, 1) * (w - 1))


def draw_overlay(img: np.ndarray, i: int, ep: dict, feat: dict | None, fps: float,
                 text_at: str = "bottom") -> np.ndarray:
    H, W = img.shape[:2]
    n, cut = ep["length"], ep["cut_frame"]
    skip = ep.get("source") == "skip"
    out = img.copy()
    font, small = cv2.FONT_HERSHEY_SIMPLEX, 0.38

    def text(s, org, color=(255, 255, 255), scale=small):
        cv2.putText(out, s, org, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, s, org, font, scale, color, 1, cv2.LINE_AA)

    y1, y2 = (H - BAR_H - 22, H - BAR_H - 8) if text_at == "bottom" else (14, 28)
    text(f"ep {ep['episode']:03d}  f {i}/{n - 1}  {i / fps:5.2f} s", (5, y1))
    if feat is not None:
        text(f"h {feat['h']:4.0f} mm  d {feat['d']:3.0f} mm  g {feat['g']:3.0f}", (5, y2))
    conf = "SKIP" if skip else f"{ep['confidence']}"
    color = (255, 210, 0) if ep["confidence"] == "low" or skip else (120, 255, 120)
    label = conf + (" " + ",".join(ep["flags"]) if ep["flags"] else "")
    (tw, _), _ = cv2.getTextSize(label, font, small, 1)
    text(label, (W - tw - 5, y1), color)
    if i >= cut and not skip:
        cv2.rectangle(out, (0, 0), (W - 1, H - BAR_H - 1), (255, 40, 40), 3)
        text("CUT", (W - 52, y2 + 2), (255, 60, 60), 0.5)
        cv2.fillPoly(out, [np.array([[W - 16, y2 - 9], [W - 16, y2 + 1], [W - 8, y2 - 4]])], (255, 60, 60))
    # timeline bar
    y0 = H - BAR_H
    out[y0:, :] = (90, 90, 90)
    if not skip:
        out[y0:, _x(cut, n, W):] = (40, 170, 70)
    for f, c in ((ep.get("close_frame"), (0, 220, 255)), (ep.get("alt_cut_frame"), (255, 220, 0)),
                 (cut, (255, 30, 30))):
        if f is not None and f < n:
            x = _x(f, n, W)
            out[y0:, max(x - 1, 0):x + 1] = c
    x = _x(i, n, W)
    out[y0 - 3:, x:x + 1] = (255, 255, 255)
    return out


def write_clip(path: Path, frames, fps: float, crf: int = 26, threads: int = 2, size=(W, H),
               maxrate_bps: int | None = None) -> None:
    """``maxrate_bps``: x264 VBV cap (CRF below it), to keep a clip under a size budget."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.mp4")
    opts = {"crf": str(crf), "preset": "medium", "threads": str(threads)}
    if maxrate_bps:
        opts |= {"maxrate": str(maxrate_bps), "bufsize": str(maxrate_bps)}
    with av.open(str(tmp), "w", options={"movflags": "+faststart"}) as c:
        s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000), options=opts)
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        s.codec_context.thread_count = threads
        for k, img in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            vf.pts = k
            vf.time_base = Fraction(1, round(fps))
            for pkt in s.encode(vf):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    tmp.replace(path)


def make_clips(dataset_root: Path, cuts_doc: dict, feat, out_dir: Path, camera: str = "top", crf: int = 26,
               threads: int = 2, max_load: float = 8.0, max_wait_s: float = 180.0, episodes=None,
               max_kb: float | None = None, log=print) -> dict:
    """One clip per OLD episode + index.json in ``out_dir``. ``feat``: per-frame features (cuts module).
    ``max_kb``: size budget per clip (sets an x264 bitrate cap from the episode's duration)."""
    import pandas as pd

    size, text_at = CLIP[camera]["size"], CLIP[camera]["text"]

    fps = float(cuts_doc["fps"])
    meta = pd.concat([pd.read_parquet(f) for f in sorted((dataset_root / "meta/episodes").glob("*/*.parquet"))])
    meta = meta.set_index("episode_index")
    key = f"observation.images.{camera}"
    index = []
    for ep in cuts_doc["episodes"]:
        e = ep["episode"]
        clip = out_dir / f"ep{e:03d}.mp4"
        if episodes is None or e in episodes:
            waited = wait_for_load(max_load, max_wait_s, log)
            m = meta.loc[e]
            video = dataset_root / f"videos/{key}/chunk-{int(m[f'videos/{key}/chunk_index']):03d}" \
                / f"file-{int(m[f'videos/{key}/file_index']):03d}.mp4"
            fe = feat[feat.episode_index == e].set_index("frame_index")
            t0 = time.time()
            n_frames = 0

            def gen(video=video, m=m, fe=fe, ep=ep):
                nonlocal n_frames
                for i, img in episode_frames(video, float(m[f"videos/{key}/from_timestamp"]), int(m["length"]),
                                             fps, size, threads):
                    r = fe.loc[i] if i in fe.index else None
                    f = None if r is None else {"h": r.h_mm, "d": r.d_mm, "g": r.gripper}
                    n_frames += 1
                    yield draw_overlay(img, i, ep, f, fps, text_at)

            # 0.9: container overhead and VBV slack
            maxrate = int(0.9 * max_kb * 8000 / (ep["length"] / fps)) if max_kb else None
            write_clip(clip, gen(), fps, crf, threads, size, maxrate)
            if n_frames != ep["length"]:
                raise RuntimeError(f"episode {e}: decoded {n_frames} frames, expected {ep['length']}")
            log(f"ep {e:3d}: {n_frames} frames, {clip.stat().st_size / 1e3:.0f} kB, {time.time() - t0:.1f} s"
                + (f" (waited {waited:.0f} s for load)" if waited > 1 else ""))
        index.append({k: ep.get(k) for k in ("episode", "task", "length", "cut_frame", "auto_cut_frame",
                                             "alt_cut_frame", "close_frame", "kept_frames", "confidence",
                                             "flags", "source", "h_mm", "d_mm", "gripper")}
                     | {"fps": fps, "clip": clip.name, "clip_bytes": clip.stat().st_size if clip.exists() else None})
    doc = {"dataset": cuts_doc["old_dataset"], "camera": camera, "fps": fps, "size": list(size), "codec": "h264",
           "criterion": cuts_doc["criterion"], "flag_help": cuts_doc["flag_help"],
           "legend": {"border": "red border and CUT: frames from the cut on (kept)",
                      "bar": "grey = dropped, green = kept, red = cut, yellow = alternative cut (gripper ignored), "
                             "cyan = close, white = playhead",
                      "text": "h = gripper-tip height above the table, d = horizontal tip-to-brick distance, "
                              "g = gripper (0 closed, 100 open)"},
           "episodes": index}
    (out_dir / "index.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    return doc

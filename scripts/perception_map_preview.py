#!/usr/bin/env python3
"""Preview of the annotations: top + floor frames with detections next to the table-frame "map".

Writes <out>/preview/ep<NNN>.mp4 (or a PNG strip with --png). The map is the canvas of the table-
frame JSON (x right, y down, 1 px = 1 mm / scale), bricks as coloured squares (hollow = held from
the last observation, ringed = lifted), gripper tip as a purple dot. This is a check tool, not the
policy's map renderer (agent C owns that).

  PYTHONPATH=src python scripts/perception_map_preview.py 431a71f7 --episodes 13,63
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hackjunction2026.perception.dataset import Dataset
from hackjunction2026.perception.scene import Frame

BGR = {"red": (40, 40, 220), "yellow": (0, 200, 240), "green": (40, 200, 120), "blue": (220, 120, 30)}


def draw_map(fr: Frame, objs: pd.DataFrame, grip: pd.Series | None, scale: float = 0.6) -> np.ndarray:
    c = fr.data["canvas"]
    W = int((c["x_max_mm"] - c["x_min_mm"]) * scale)
    H = int((c["y_max_mm"] - c["y_min_mm"]) * scale)
    img = np.full((H, W, 3), 255, np.uint8)

    def px(x, y):
        return int((x - c["x_min_mm"]) * scale), int((y - c["y_min_mm"]) * scale)

    for g in range(0, int(c["x_max_mm"]) + 1, 100):
        cv2.line(img, px(g, c["y_min_mm"]), px(g, c["y_max_mm"]), (230, 230, 230), 1)
    for g in range(0, int(c["y_max_mm"]) + 1, 100):
        cv2.line(img, px(c["x_min_mm"], g), px(c["x_max_mm"], g), (230, 230, 230), 1)
    for _, o in objs.iterrows():
        p = px(o.x_mm, o.y_mm)
        col = BGR.get(o.color.split("+")[0], (90, 90, 90))
        s = int(16 * scale)
        cv2.rectangle(img, (p[0] - s, p[1] - s), (p[0] + s, p[1] + s), col, -1 if o.source != "hold" else 2)
        if "+" in o.color:
            col2 = BGR.get(o.color.split("+")[1], (90, 90, 90))
            cv2.rectangle(img, (p[0] - s // 2, p[1] - s // 2), (p[0] + s // 2, p[1] + s // 2), col2, -1)
        if o.lifted:
            cv2.circle(img, p, s + 6, (0, 0, 0), 1)
    if grip is not None:
        x, y = grip.tip_x_mm, grip.tip_y_mm
        if not np.isfinite(x) and np.isfinite(grip.top_tip_plane_x_mm):
            x, y = grip.top_tip_plane_x_mm, grip.top_tip_plane_y_mm
        if np.isfinite(x):
            cv2.circle(img, px(x, y), 7, (160, 40, 110), -1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default="/home/intel/dev/captures/perception/A")
    ap.add_argument("--episodes", default="0")
    ap.add_argument("--png", action="store_true", help="8-frame PNG strip instead of a video")
    args = ap.parse_args()
    ds = Dataset(args.dataset)
    out = Path(args.out)
    fr = Frame.load(out / f"table_frame_{ds.id}.json")
    objs = pd.read_parquet(out / f"objects_{ds.id}.parquet")
    grip = pd.read_parquet(out / f"gripper_{ds.id}.parquet")
    (out / "preview").mkdir(exist_ok=True)
    for ep in map(int, args.episodes.split(",")):
        L = ds.episodes[ep].length
        want = set(np.linspace(0, L - 1, 8).round().astype(int)) if args.png else None
        eo, eg = objs[objs.episode_index == ep], grip[grip.episode_index == ep].set_index("frame_index")
        frames_out = []
        it_f = ds.iter_frames(ep, "floor", want)
        writer = None
        for (i, top), (j, floor) in zip(ds.iter_frames(ep, "top", want), it_f):
            o = eo[eo.frame_index == i]
            vis = {}
            for cam, img in (("top", top), ("floor", floor)):
                v = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for _, r in o.iterrows():
                    b = r[f"{cam}_bbox"]
                    if b is not None:
                        b = [int(t) for t in b]
                        cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
                if i in eg.index and eg.loc[i, f"{cam}_visible"]:
                    cv2.circle(v, (int(eg.loc[i, f"{cam}_tip_px_x"]), int(eg.loc[i, f"{cam}_tip_px_y"])), 6, (0, 0, 255), -1)
                vis[cam] = cv2.resize(v, (424, 240))
            m = draw_map(fr, o, eg.loc[i] if i in eg.index else None)
            m = cv2.resize(m, (int(m.shape[1] * 480 / m.shape[0]), 480))
            canvas = np.hstack([np.vstack([vis["top"], vis["floor"]]), m])
            cv2.putText(canvas, f"ep{ep} f{i} {ds.episodes[ep].task}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if args.png:
                frames_out.append(canvas)
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(out / "preview" / f"ep{ep:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             ds.fps, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
        if writer is not None:
            writer.release()
        if args.png:
            cv2.imwrite(str(out / "preview" / f"ep{ep:03d}.png"), np.vstack(frames_out))
        print("ep", ep, "done")


if __name__ == "__main__":
    main()

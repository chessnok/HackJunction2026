"""Extract frames of an episode from the LeRobot v3 dataset videos (PyAV)."""
import glob
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

D = "/home/intel/.local/share/physicalai/datasets/431a71f7-50c6-4e38-9e38-d85984dde78b"
_EPS = None


def episodes():
    global _EPS
    if _EPS is None:
        _EPS = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(D + "/meta/episodes/chunk-*/*.parquet"))])
        _EPS = _EPS.set_index("episode_index")
    return _EPS


def get_frames(ep, cam, frame_indices):
    """Return dict frame_index -> RGB uint8 image for one episode/camera."""
    e = episodes().loc[ep]
    key = f"observation.images.{cam}"
    ch = e[f"videos/{key}/chunk_index"]
    fi = e[f"videos/{key}/file_index"]
    t0 = e[f"videos/{key}/from_timestamp"]
    path = f"{D}/videos/{key}/chunk-{ch:03d}/file-{fi:03d}.mp4"
    want = {int(f): t0 + f / 30.0 for f in frame_indices}
    out = {}
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_count = 4
        tb = s.time_base
        tmin = min(want.values())
        c.seek(int(max(0, tmin - 1.0) / tb), stream=s, backward=True, any_frame=False)
        remaining = dict(want)
        for fr in c.decode(s):
            t = float(fr.pts * tb)
            hit = [f for f, tt in remaining.items() if abs(t - tt) < 0.5 / 30]
            for f in hit:
                out[f] = fr.to_ndarray(format="rgb24")
                del remaining[f]
            if not remaining or t > max(want.values()) + 0.1:
                break
    return out


if __name__ == "__main__":
    import cv2

    ep = int(sys.argv[1])
    cam = sys.argv[2]
    fids = [int(x) for x in sys.argv[3].split(",")]
    outdir = Path(sys.argv[4])
    outdir.mkdir(parents=True, exist_ok=True)
    fr = get_frames(ep, cam, fids)
    for f, img in fr.items():
        cv2.imwrite(str(outdir / f"ep{ep:03d}_{cam}_f{f:04d}.png"), img[:, :, ::-1])
    print(sorted(fr))

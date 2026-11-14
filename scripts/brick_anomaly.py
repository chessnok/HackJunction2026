#!/usr/bin/env python3
"""Score bricks with the exported Anomalib model (OpenVINO) on NPU, GPU or CPU.

One JSON line per scene: {"scene", "detector": "anomalib", "anomaly_score" (max over crops),
"is_anomalous" (OR over crops, ADR 0008), "threshold", "latency_s", "device", "model", "crops": [...]}.
Model, threshold and crop settings come from configs/anomalib_deploy.yaml unless overridden.

Inputs:
  snapshot dirs (state.json + <camera>.color.png)  one scene per dir; same cropping as training
  --snapshot ID                                    a Studio state snapshot by id (any project)
  raw frames top.png floor.png arm.png             one scene; camera from the file name or --camera
  --crops a.png b.png                              images that are already brick crops; one scene

Run from the repo root in the project env (needs openvino, opencv, numpy, pyyaml) or Studio's env:
  ZE_ENABLE_ALT_DRIVERS=... (see scripts/anomalib.sh, T-008) is needed for the NPU.
  .venv/bin/python scripts/brick_anomaly.py --device NPU ~/.local/share/physicalai/state_snapshots/<p>/<id>
  .venv/bin/python scripts/brick_anomaly.py --device GPU --heatmaps --snapshot 20261114-051059-914
  .venv/bin/python scripts/brick_anomaly.py --benchmark CPU,GPU,NPU          # latency per crop
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hackjunction2026.defect.runtime import DEPLOY, BrickAnomalyDetector  # noqa: E402

SNAPSHOTS = Path.home() / ".local/share/physicalai/state_snapshots"
CAMERAS = ("top", "floor", "arm")


def ensure_npu_driver() -> None:
    """Level Zero must list both drivers or the NPU is invisible (T-008); set it if unset."""
    gpu = "/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1"
    npu = "/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1"
    if "ZE_ENABLE_ALT_DRIVERS" not in os.environ and Path(gpu).exists() and Path(npu).exists():
        os.environ["ZE_ENABLE_ALT_DRIVERS"] = f"{gpu},{npu}"


def scenes(args) -> list[tuple[str, dict[str, np.ndarray] | None, list[Path] | None]]:
    out = []
    for sid in args.snapshot or []:
        dirs = sorted(SNAPSHOTS.glob(f"*/{sid}*"))
        if not dirs:
            sys.exit(f"snapshot {sid} not found under {SNAPSHOTS}")
        args.inputs.extend(str(d) for d in dirs)
    loose: dict[str, np.ndarray] = {}
    for p in map(Path, args.inputs):
        if p.is_dir():
            frames = {c: cv2.imread(str(p / f"{c}.color.png")) for c in CAMERAS if (p / f"{c}.color.png").exists()}
            out.append((p.name, frames, None))
        elif args.crops:
            out.append((p.stem, None, [p]))
        else:
            cam = args.camera or next((c for c in CAMERAS if c in p.name), p.stem)
            loose[cam] = cv2.imread(str(p))
    if loose:
        out.append((args.scene or "frames", loose, None))
    return out


def benchmark(args) -> None:
    import openvino as ov

    det0 = BrickAnomalyDetector.from_deploy(args.deploy, device="CPU") if not args.model else None
    xml = Path(args.model) if args.model else det0.xml
    crops = sorted(Path(args.bench_crops).glob("*.png"))[:16] if args.bench_crops else []
    imgs = [cv2.imread(str(c)) for c in crops] or [np.full((256, 256, 3), 200, np.uint8)]
    print(f"available devices: {ov.Core().available_devices}", file=sys.stderr)
    results = []
    for dev in args.benchmark.split(","):
        try:
            det = BrickAnomalyDetector(xml, dev, crop=det0.crop if det0 else None)
        except Exception as exc:  # noqa: BLE001  report and continue with the next device
            results.append({"device": dev, "error": repr(exc)[:300]})
            continue
        for i in range(10):
            det.score_crop(imgs[i % len(imgs)])
        lat = [det.score_crop(imgs[i % len(imgs)])[2] for i in range(args.n)]
        results.append({"device": dev, "model": str(xml), "compile_s": round(det.compile_s, 2), "n": args.n,
                        "median_ms": round(statistics.median(lat), 2),
                        "p95_ms": round(float(np.percentile(lat, 95)), 2), "mean_ms": round(statistics.mean(lat), 2)})
        print(json.dumps(results[-1]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*")
    p.add_argument("--snapshot", action="append", help="state snapshot id (prefix ok); repeatable")
    p.add_argument("--crops", action="store_true", help="inputs are already brick crops")
    p.add_argument("--camera", help="camera name for a single raw frame")
    p.add_argument("--scene", help="scene name for loose raw frames")
    p.add_argument("--deploy", default=str(DEPLOY), help="deploy config (model, threshold, crop)")
    p.add_argument("--model", help="model.xml instead of the deploy config's")
    p.add_argument("--device", help="NPU, GPU or CPU (default: deploy config)")
    p.add_argument("--threshold", type=float, help="override the deploy threshold")
    p.add_argument("--max-bricks", type=int, default=1, help="bricks per camera frame (largest first)")
    p.add_argument("--heatmaps", nargs="?", const="captures/anomalib/heatmaps", help="write overlay PNGs here")
    p.add_argument("--benchmark", help="comma list of devices: latency per crop instead of scoring")
    p.add_argument("--bench-crops", help="folder of crops for --benchmark (default: a flat grey image)")
    p.add_argument("-n", type=int, default=200, help="--benchmark iterations")
    p.add_argument("--out", help="also write results (JSON lines / benchmark JSON) here")
    args = p.parse_args()
    ensure_npu_driver()
    if args.benchmark:
        benchmark(args)
        return

    if args.model:
        det = BrickAnomalyDetector(args.model, args.device or "CPU", args.threshold if args.threshold is not None else 0.5)
    else:
        det = BrickAnomalyDetector.from_deploy(args.deploy, device=args.device, threshold=args.threshold)
    print(f"model {det.xml} on {det.device}, compiled in {det.compile_s:.1f} s, threshold {det.threshold}",
          file=sys.stderr)
    heat = Path(args.heatmaps) if args.heatmaps else None
    fh = open(args.out, "a") if args.out else None  # noqa: SIM115
    for name, frames, crop_files in scenes(args):
        if frames is not None:
            res = det.judge_frames(frames, scene=name, heatmap_dir=heat, max_bricks=args.max_bricks)
        else:
            crops = [{"image": cv2.imread(str(f)), "camera": f.stem} for f in crop_files]
            res = det.judge_crops(crops, scene=name, heatmap_dir=heat)
        res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = json.dumps(res)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()

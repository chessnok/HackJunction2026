"""Runtime brick anomaly detector: exported Anomalib model (OpenVINO IR) on NPU / GPU / CPU.

Plain OpenVINO + OpenCV (no Anomalib, no torch), so it runs in the project env, in Studio's
backend env, or inside the orchestrator. Same cropping code as the training data
(cropping.bricks), with the crop settings recorded in the run's dataset.json.

    det = BrickAnomalyDetector.from_deploy()             # configs/anomalib_deploy.yaml
    result = det.judge_frames({"top": bgr, "floor": bgr, "arm": bgr}, scene="...")
    result["is_anomalous"]   # brick-level OR over crops (ADR 0008)

The score is Anomalib's normalized pred_score (0..1; 0.5 = the threshold learned in training).
The deploy threshold is chosen in training.brick_report and stored in configs/anomalib_deploy.yaml.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026.defect.cropping.bricks import CropConfig, SegConfig, crop_brick, find_bricks

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "configs/anomalib_deploy.yaml"


def ov_core():
    import openvino as ov

    return ov.Core()


class BrickAnomalyDetector:
    def __init__(self, xml: str | Path, device: str = "CPU", threshold: float = 0.5,
                 crop: CropConfig | None = None, seg: SegConfig | None = None, name: str = "",
                 cpu_threads: int = 4):
        self.xml, self.device, self.threshold = Path(xml), device, float(threshold)
        self.crop, self.seg = crop or CropConfig(), seg or SegConfig()
        self.name = name or self.xml.parent.as_posix()
        core = ov_core()
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if device == "CPU":
            config["INFERENCE_NUM_THREADS"] = cpu_threads  # all cores by default; Studio records alongside
        t0 = time.time()
        self.compiled = core.compile_model(str(self.xml), device, config)
        self.compile_s = time.time() - t0
        self.input = self.compiled.inputs[0]
        shape = list(self.input.get_partial_shape().get_min_shape())
        self.in_h, self.in_w = (int(shape[2]), int(shape[3])) if len(shape) == 4 and shape[2] > 0 else (
            self.crop.size, self.crop.size)
        self.out_names = {o.get_any_name(): o for o in self.compiled.outputs}
        self.request = self.compiled.create_infer_request()

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_deploy(cls, path: str | Path = DEPLOY, device: str | None = None, threshold: float | None = None):
        import yaml

        cfg = yaml.safe_load(Path(path).read_text())
        xml = Path(cfg["model_xml"]).expanduser()
        if not xml.is_absolute():  # relative to the repo root (exports/ is git-ignored), else to cwd
            xml = REPO / xml if (REPO / xml).exists() else Path.cwd() / xml
        crop = CropConfig(**{f.name: cfg["crop"][f.name] for f in fields(CropConfig) if f.name in cfg.get("crop", {})})
        if isinstance(crop.fill, list):
            crop.fill = tuple(crop.fill)
        return cls(xml, device or cfg.get("device", "CPU"),
                   cfg["threshold"] if threshold is None else threshold, crop=crop, name=cfg.get("run_id", ""))

    @staticmethod
    def crop_from_run(run_json: str | Path) -> CropConfig:
        run = json.loads(Path(run_json).read_text())
        c = ((run.get("dataset") or {}).get("summary") or {}).get("crop") or {}
        crop = CropConfig(**{f.name: c[f.name] for f in fields(CropConfig) if f.name in c})
        if isinstance(crop.fill, list):
            crop.fill = tuple(crop.fill)
        return crop

    # ---------------------------------------------------------------- inference

    def score_crop(self, crop_bgr: np.ndarray) -> tuple[float, np.ndarray | None, float]:
        """(normalized score, anomaly map or None, latency ms) for one crop."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.in_h, self.in_w):
            rgb = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        t = time.perf_counter()
        res = self.request.infer({0: x})
        ms = (time.perf_counter() - t) * 1000
        score, amap = None, None
        for name, out in self.out_names.items():
            v = res[out]
            if "score" in name:
                score = float(np.ravel(v)[0])
            elif "anomaly_map" in name:
                amap = np.squeeze(v)
        if score is None:  # unnamed outputs: the scalar one is the score
            for v in res.values():
                if np.size(v) == 1:
                    score = float(np.ravel(v)[0])
        return score, amap, ms

    def judge_crops(self, crops: list[dict], scene: str = "", heatmap_dir: Path | None = None) -> dict:
        """crops: [{"image": bgr, ...meta}] -> scene result (OR over crops)."""
        t0 = time.perf_counter()
        out = []
        for c in crops:
            score, amap, ms = self.score_crop(c["image"])
            item = {k: v for k, v in c.items() if k not in ("image", "mask")}
            item.update(score=round(score, 4), is_anomalous=score > self.threshold, latency_ms=round(ms, 2))
            if heatmap_dir is not None and amap is not None:
                heatmap_dir.mkdir(parents=True, exist_ok=True)
                f = heatmap_dir / f"{scene.replace('/', '_') or 'scene'}_{item.get('camera', 'crop')}_{len(out)}.png"
                cv2.imwrite(str(f), heatmap_overlay(c["image"], amap, f"{score:.2f}"))
                item["heatmap"] = str(f)
            out.append(item)
        best = max(out, key=lambda i: i["score"]) if out else None
        return {
            "scene": scene, "detector": "anomalib", "model": self.name, "device": self.device,
            "anomaly_score": best["score"] if best else None,
            "is_anomalous": any(i["is_anomalous"] for i in out),
            "threshold": self.threshold, "n_crops": len(out),
            "worst_camera": best.get("camera") if best else None,
            "latency_s": round(time.perf_counter() - t0, 4), "crops": out,
        }

    def crops_from_frames(self, frames: dict[str, np.ndarray], max_bricks: int = 1) -> list[dict]:
        crops = []
        for cam, frame in frames.items():
            t = time.perf_counter()
            bricks = find_bricks(frame, self.seg, camera=cam)[:max_bricks]
            seg_ms = (time.perf_counter() - t) * 1000
            for k, b in enumerate(bricks):
                img, _, square = crop_brick(frame, b, self.crop)
                crops.append({"image": img, "camera": cam, "brick": k, "bbox": list(b.bbox), "square": list(square),
                              "colour": b.colour, "seg_ms": round(seg_ms, 2)})
        return crops

    def judge_frames(self, frames: dict[str, np.ndarray], scene: str = "", heatmap_dir: Path | None = None,
                     max_bricks: int = 1) -> dict:
        t0 = time.perf_counter()
        res = self.judge_crops(self.crops_from_frames(frames, max_bricks), scene, heatmap_dir)
        res["latency_s"] = round(time.perf_counter() - t0, 4)
        if not res["n_crops"]:
            res["note"] = "no brick found on the sheet in any camera"
        return res


def heatmap_overlay(crop_bgr: np.ndarray, amap: np.ndarray, text: str) -> np.ndarray:
    a = cv2.resize(np.asarray(amap, np.float32), crop_bgr.shape[1::-1])
    a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = cv2.addWeighted(crop_bgr, 0.6, cv2.applyColorMap(a, cv2.COLORMAP_JET), 0.4, 0)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out

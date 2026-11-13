"""Colour streams of the RealSense D415 cameras, with locked exposure and white balance.

Uses pyrealsense2 directly: physicalai's RealSenseCamera always opens the depth stream too and
has no exposure / white-balance control. Both are available in Studio's backend environment.
Cameras are opened by serial number only, so the laptop's built-in camera is never touched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    bgr: np.ndarray
    timestamp_ms: float
    number: int


def _rs():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        msg = ("pyrealsense2 not found. Run with Studio's backend environment: "
               "uv run --project third_party/physical-ai-studio/application/backend ...")
        raise SystemExit(msg) from exc
    return rs


def list_realsense() -> list[dict]:
    rs = _rs()
    out = []
    for dev in rs.context().query_devices():
        info = {}
        for key, field in (("name", "name"), ("serial", "serial_number"), ("firmware", "firmware_version"),
                           ("usb", "usb_type_descriptor"), ("port", "physical_port")):
            attr = getattr(rs.camera_info, field)
            info[key] = dev.get_info(attr) if dev.supports(attr) else "?"
        out.append(info)
    return out


class RealSenseColor:
    """One D415 colour stream (BGR) with optional locked exposure / white balance."""

    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        if not serial:
            msg = "camera serial is empty in configs/inspection.yaml (find it with --list-cameras)"
            raise SystemExit(msg)
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self._rs = _rs()
        self._pipeline = None
        self._sensor = None
        self._last = None

    def start(self) -> None:
        rs = self._rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            profile = self._pipeline.start(cfg)
        except RuntimeError as exc:
            msg = (f"cannot start D415 {self.serial} at {self.width}x{self.height}@{self.fps}: {exc}\n"
                   "  - Is Studio using the camera (recording / Run model / environment preview)? Stop it.\n"
                   "  - On USB 2 (see --list-cameras) use width 640, height 480 in configs/inspection.yaml.")
            raise SystemExit(msg) from exc
        self._sensor = profile.get_device().first_color_sensor()

    def read(self, timeout_ms: int = 2000) -> Frame:
        frames = self._pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        self._last = color
        return Frame(np.asanyarray(color.get_data()).copy(), color.get_timestamp(), color.get_frame_number())

    def _set(self, name: str, value: float) -> float | None:
        opt = getattr(self._rs.option, name)
        if not self._sensor.supports(opt):
            return None
        rng = self._sensor.get_option_range(opt)
        value = float(min(max(value, rng.min), rng.max))
        self._sensor.set_option(opt, value)
        return value

    def _meta(self, name: str) -> float | None:
        md = getattr(self._rs.frame_metadata_value, name)
        if self._last is not None and self._last.supports_frame_metadata(md):
            return float(self._last.get_frame_metadata(md))
        return None

    def configure_exposure(self, cfg: dict) -> dict:
        """Apply the exposure mode from the config; returns the values in use (for the manifest)."""
        mode = cfg["mode"]
        if mode == "auto":
            self._set("enable_auto_exposure", 1)
            self._set("enable_auto_white_balance", 1)
            return {"mode": "auto"}

        if mode == "freeze_auto":
            self._set("enable_auto_exposure", 1)
            self._set("enable_auto_white_balance", 1)
            t_end = time.monotonic() + cfg["warmup_s"]
            while time.monotonic() < t_end:
                self.read()
            exposure = self._meta("actual_exposure")
            rng = self._sensor.get_option_range(self._rs.option.exposure)
            if exposure is not None and exposure > rng.max:
                exposure /= 100.0  # metadata in µs, option in units of 100 µs (untested assumption)
            values = {"exposure": exposure, "gain": self._meta("gain_level"),
                      "white_balance": self._meta("white_balance")}
            fallback = cfg["manual"]
            values = {k: (v if v is not None else fallback[k]) for k, v in values.items()}
        elif mode == "manual":
            values = dict(cfg["manual"])
        else:
            msg = f"unknown exposure mode {mode!r}"
            raise SystemExit(msg)

        self._set("enable_auto_exposure", 0)
        self._set("enable_auto_white_balance", 0)
        applied = {k: self._set(k, v) for k, v in values.items()}
        for _ in range(5):  # let the new settings reach the frames
            self.read()
        return {"mode": mode, **applied}

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline = None


class WebcamCamera:
    """TEST ONLY: an ordinary webcam via OpenCV, to try the tool on a machine without D415s.

    Never use this on the station laptop: AGENTS.md forbids its built-in camera (it films people).
    Both roles share one device, so "top" and "side" show the same picture.
    """

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        import cv2

        self._cv2 = cv2
        self.serial, self.index = f"webcam{index}", index
        self.width, self.height, self.fps = width, height, fps
        self._cap = None
        self._n = 0

    def start(self) -> None:
        if self._cap is not None:  # shared by both roles
            return
        cv2 = self._cv2
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened() or not self._cap.read()[0]:
            msg = f"cannot open webcam {self.index}"
            raise SystemExit(msg)

    def read(self, timeout_ms: int = 2000) -> Frame:
        ok, img = self._cap.read()
        if not ok:
            msg = "webcam read failed"
            raise RuntimeError(msg)
        self._n += 1
        return Frame(img, time.monotonic() * 1000, self._n)

    def configure_exposure(self, cfg: dict) -> dict:
        return {"mode": f"webcam, not locked (config: {cfg['mode']})"}

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FakeCamera:
    """Synthetic frames (a coloured square on grey) to test the tool without hardware."""

    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        self.serial, self.width, self.height, self.fps = serial or "fake", width, height, fps
        self._n = 0
        self._rng = np.random.default_rng(0)

    def start(self) -> None:
        pass

    def read(self, timeout_ms: int = 2000) -> Frame:
        self._n += 1
        img = np.full((self.height, self.width, 3), 180, np.uint8)
        img += self._rng.integers(0, 12, img.shape, dtype=np.uint8)
        cx, cy, s = self.width // 2, self.height // 2, min(self.width, self.height) // 8
        img[cy - s : cy + s, cx - s : cx + s] = (30, 40, 200)  # red-ish brick (BGR)
        for i in range(2):  # stud-like texture so the sharpness check passes
            for j in range(2):
                yy, xx = cy - s // 2 + i * s, cx - s // 2 + j * s
                img[yy - 6 : yy + 6, xx - 6 : xx + 6] = (60, 80, 240)
        time.sleep(1 / self.fps)
        return Frame(img, time.monotonic() * 1000, self._n)

    def configure_exposure(self, cfg: dict) -> dict:
        return {"mode": f"fake ({cfg['mode']})"}

    def stop(self) -> None:
        pass

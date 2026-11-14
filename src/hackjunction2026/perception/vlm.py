"""VLM brick localizer against a LOCAL OpenAI-compatible server (OpenVINO Model Server on this laptop).

Team rule (2026-11-14): object localization must run on a local model; no frames go to cloud APIs.
The server (another agent sets it up) serves an OpenVINO IR VLM at
http://localhost:8000/v3/chat/completions with JSON-schema `response_format`. One request at a
time: RAM is shared with Studio while it records.

Three modes:
- per_camera: one image, list of bricks with colour + bbox;
- multi_view: top + floor + wrist images in one request, each brick with a bbox per view;
- hybrid: VLM boxes decide which objects exist and their colour, the CV blobs inside give the
  precise box (see `hybrid_merge`).
Box coordinates: Qwen-VL models answer in a 0-1000 grid relative to the image (Qwen2.5-VL in
pixels); `coord_mode` selects how to convert ("rel1000" or "pixel"), checked on frames outside
the evaluation set.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from .color_cv import Blob

COLORS = ["red", "yellow", "green", "blue", "other"]

SCENE = ("The camera looks at a white paper sheet on a table. A purple SO-101 robot arm with a purple "
         "two-jaw gripper picks toy building bricks (Duplo) from the paper and drops them into bowls "
         "(a green bowl and a lavender bowl).")
TASK = ("Find every toy brick that lies on the white paper or is held by the gripper. Ignore the robot, the "
        "bowls and anything inside them, and everything off the paper (wooden table, cables, background). "
        "A brick can be made of two joined or stacked bricks of different colours: report it once, with "
        "both colours (colour = the larger part, second_color = the other part, or none). Colours: red, "
        "yellow (also orange-yellow), green (also lime / yellow-green), blue (also turquoise / teal). "
        "bbox_2d is [x1, y1, x2, y2] of the whole brick, tight. If there is no brick, return an empty list.")

DET_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "enum": COLORS},
                    "second_color": {"type": "string", "enum": ["none"] + COLORS[:-1]},
                    "bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                },
                "required": ["color", "second_color", "bbox_2d"],
            },
        }
    },
    "required": ["objects"],
}

MV_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "enum": COLORS},
                    "second_color": {"type": "string", "enum": ["none"] + COLORS[:-1]},
                    "image1_bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 0, "maxItems": 4},
                    "image2_bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 0, "maxItems": 4},
                    "image3_bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 0, "maxItems": 4},
                },
                "required": ["color", "second_color", "image1_bbox_2d", "image2_bbox_2d", "image3_bbox_2d"],
            },
        }
    },
    "required": ["objects"],
}


@dataclass
class VLMDet:
    bbox: tuple[float, float, float, float]
    colors: list[str]

    @property
    def color(self) -> str:
        return "+".join(self.colors)

    @property
    def bottom_center(self):
        x0, y0, x1, y1 = self.bbox
        return 0.5 * (x0 + x1), y1

    @property
    def center(self):
        x0, y0, x1, y1 = self.bbox
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def encode(rgb: np.ndarray, max_side: int | None = None, quality: int = 92) -> tuple[str, float]:
    """JPEG data URL of an RGB image (optionally downscaled); returns (url, scale applied)."""
    s = 1.0
    if max_side and max(rgb.shape[:2]) > max_side:
        s = max_side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(), s


class VLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v3", model: str | None = None,
                 coord_mode: str = "rel1000", max_side: int | None = None, timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self.model = model or self.first_model()
        self.coord_mode = coord_mode
        self.max_side = max_side
        self.timeout = timeout
        self.last_latency = 0.0
        self.last_raw = ""

    def first_model(self) -> str:
        with urllib.request.urlopen(self.base + "/models", timeout=10) as r:
            data = json.load(r)
        return data["data"][0]["id"]

    def chat(self, images: list[np.ndarray], text: str, schema: dict, max_tokens: int = 512) -> dict:
        content, scales = [], []
        for img in images:
            url, s = encode(img, self.max_side)
            scales.append(s)
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": text})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "bricks", "schema": schema}},
        }
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.load(r)
        self.last_latency = time.time() - t
        self.last_raw = resp["choices"][0]["message"]["content"]
        self.last_usage = resp.get("usage", {})
        try:
            return json.loads(self.last_raw)
        except json.JSONDecodeError:
            return {"objects": [], "_error": "invalid json"}

    def to_px(self, box, w: int, h: int):
        if len(box) != 4:
            return None
        x0, y0, x1, y1 = (float(v) for v in box)
        if self.coord_mode == "rel1000":
            x0, x1 = x0 * w / 1000, x1 * w / 1000
            y0, y1 = y0 * h / 1000, y1 * h / 1000
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return (float(x0), float(y0), float(x1), float(y1))

    @staticmethod
    def _colors(o) -> list[str]:
        cs = [o.get("color", "other")]
        if o.get("second_color", "none") not in ("none", cs[0]):
            cs.append(o["second_color"])
        return cs

    def detect(self, rgb: np.ndarray, cam: str) -> list[VLMDet]:
        h, w = rgb.shape[:2]
        view = {"top": "This is a low camera at the side of the table.",
                "floor": "This is a low camera at the far side of the table, facing the robot.",
                "arm": "This is the wrist camera on the robot gripper; the purple jaws are at the bottom."}[cam]
        coord = (" Coordinates are relative, 0-1000 on both axes." if self.coord_mode == "rel1000"
                 else f" Coordinates are in pixels of this {w}x{h} image.")
        out = self.chat([rgb], f"{SCENE} {view} {TASK}{coord}", DET_SCHEMA)
        dets = []
        for o in out.get("objects", []):
            b = self.to_px(o.get("bbox_2d", []), w, h)
            if b:
                dets.append(VLMDet(b, self._colors(o)))
        return dets

    def detect_multi(self, imgs: dict[str, np.ndarray]) -> dict[str, list[VLMDet]]:
        order = ["top", "floor", "arm"]
        coord = (" Coordinates are relative to each image, 0-1000 on both axes." if self.coord_mode == "rel1000"
                 else " Coordinates are in pixels of each image.")
        text = (f"{SCENE} Image 1 and image 2 come from two low cameras on opposite sides of the table, facing "
                "each other (left and right are swapped between them). Image 3 is the wrist camera on the "
                f"gripper. {TASK} Give each physical brick once, with its box in every image where it is "
                f"visible (empty list where it is not).{coord}")
        out = self.chat([imgs[c] for c in order], text, MV_SCHEMA, max_tokens=768)
        res = {c: [] for c in order}
        for o in out.get("objects", []):
            for k, c in enumerate(order, 1):
                h, w = imgs[c].shape[:2]
                b = self.to_px(o.get(f"image{k}_bbox_2d", []), w, h)
                if b:
                    res[c].append(VLMDet(b, self._colors(o)))
        return res


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def hybrid_merge(vlm: list[VLMDet], blobs: list[Blob], min_overlap: float = 0.1) -> list[VLMDet]:
    """VLM decides which bricks exist and their colour; the best-overlapping CV blob of a compatible
    colour gives the box. VLM boxes without a CV blob are kept as they are."""
    out, used = [], set()
    for v in vlm:
        best, best_s = None, min_overlap
        for k, b in enumerate(blobs):
            if k in used or not (set(b.colors) & set(v.colors)):
                continue
            s = iou(v.bbox, b.bbox)
            if s > best_s:
                best, best_s = k, s
        if best is None:
            out.append(v)
        else:
            used.add(best)
            out.append(VLMDet(blobs[best].bbox, v.colors))
    return out

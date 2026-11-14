#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow>=11", "numpy>=2", "opencv-python-headless>=4.10"]
# ///
"""Ask the local VLM (OVMS, see scripts/vlm_server.sh) whether the brick in a scene is broken.

Per scene: find the brick by colour on the white sheet in every camera frame, cut a square crop of
2.4x the brick size from the full-resolution frame, upscale it to 448 px ("digital zoom": a 1.5 px
marker line becomes visible) and send all crops plus the prompt as ONE request. The answer is forced
to the JSON schema {"is_broken": boolean} by the server (response_format json_schema = constrained
decoding), temperature 0. Nothing is parsed out of free text. A frame without a found brick is sent
whole instead of its crop.

Run from the repo root. uv reads the inline dependencies above; the project env is not used:
  uv run scripts/brick_judge.py <snapshot dir> [<snapshot dir> ...]   # one scene per dir
  uv run scripts/brick_judge.py top.png floor.png arm.png              # loose files = one scene
  uv run scripts/brick_judge.py --snapshot 20261114-051059-914         # --project optional if only one
  uv run scripts/brick_judge.py --all --eval                           # every snapshot vs its label
  uv run scripts/brick_judge.py --csv --eval --out captures/vlm/eval.jsonl
  uv run scripts/brick_judge.py --all --eval --view frames             # full frames instead of crops
  uv run scripts/brick_judge.py --all --eval --view both               # crops + full frames
  uv run scripts/brick_judge.py --all --eval --mode per-image          # one request per image, OR
  uv run scripts/brick_judge.py --all --save-crops captures/vlm/crops  # keep the crops for a look
Snapshots live in ~/.local/share/physicalai/state_snapshots/<project>/<snapshot>/ (read-only here).
Output: one JSON line per scene ({"scene", "is_broken", "latency_s", "boxes", ...}), then a summary.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)  # the laptop records with Studio at the same time: no CPU bursts on all cores

SNAPSHOTS = Path.home() / ".local/share/physicalai/state_snapshots"
DEFAULT_URL = os.environ.get("VLM_URL", "http://127.0.0.1:8000/v3")

# The user's prompt, verbatim (typos included). Change it with --prompt-file, not here.
DEFAULT_PROMPT = (
    "Consider a brick broken, if and only if on any of the photos u see it is clearly marked with "
    "black marker.\n\nIs this brick broken?"
)

SCHEMA = {
    "type": "object",
    "properties": {"is_broken": {"type": "boolean"}},
    "required": ["is_broken"],
    "additionalProperties": False,
}

# --reason: the model first writes what it sees, then decides (still one constrained JSON answer).
REASON_SCHEMA = {
    "type": "object",
    "properties": {"observation": {"type": "string", "maxLength": 300}, "is_broken": {"type": "boolean"}},
    "required": ["observation", "is_broken"],
    "additionalProperties": False,
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels of the full frame


@dataclass
class Scene:
    name: str
    images: list[Path]
    label: bool | None = None  # True = defective (broken), None = unknown
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- scene discovery


def parse_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def scene_from_dir(path: Path) -> Scene:
    """A Studio state snapshot dir (state.json + <camera>.color.png) or any dir of images."""
    state_file = path / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        images = []
        for cam in state.get("cameras") or []:
            color = (cam.get("files") or {}).get("color")
            if color and (path / color).exists():
                images.append(path / color)
        if not images:
            images = sorted(path.glob("*.color.png"))
        return Scene(path.name, images, parse_bool(state.get("defective")), {"comment": state.get("comment")})
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return Scene(path.name, images)


def resolve_project(project: str | None) -> Path:
    if project:
        path = Path(project).expanduser()
        if path.is_dir():
            return path
        matches = sorted(p for p in SNAPSHOTS.glob(f"{project}*") if p.is_dir())
    else:
        matches = sorted(p for p in SNAPSHOTS.iterdir() if p.is_dir()) if SNAPSHOTS.is_dir() else []
    if len(matches) != 1:
        names = ", ".join(p.name for p in matches) or "none"
        sys.exit(f"project '{project or ''}': {len(matches)} matches in {SNAPSHOTS} ({names}); use --project")
    return matches[0]


def scenes_from_csv(csv_path: Path) -> list[Scene]:
    """Studio's snapshots.csv: one row per snapshot, camera_N_color paths relative to the project dir."""
    root = csv_path.parent
    scenes = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            images = [root / row[f"camera_{n}_color"] for n in range(1, 10) if row.get(f"camera_{n}_color")]
            missing = [str(p) for p in images if not p.exists()]
            if missing:
                print(f"skip {row['snapshot_id']}: missing {missing}", file=sys.stderr)
                continue
            scenes.append(
                Scene(row["snapshot_id"], images, parse_bool(row.get("defective")), {"comment": row.get("comment")})
            )
    return scenes


def collect_scenes(args) -> list[Scene]:
    scenes: list[Scene] = []
    if args.csv is not None or args.all or args.snapshot:
        project = resolve_project(args.project)
        if args.csv is not None:
            scenes += scenes_from_csv(Path(args.csv).expanduser() if args.csv else project / "snapshots.csv")
        elif args.all:
            scenes += [scene_from_dir(p) for p in sorted(project.iterdir()) if (p / "state.json").exists()]
        for snap in args.snapshot or []:
            path = project / snap
            if not path.is_dir():
                sys.exit(f"snapshot {snap} not found in {project}")
            scenes.append(scene_from_dir(path))
    loose = []
    for item in args.inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            scenes.append(scene_from_dir(path))
        elif path.is_file():
            loose.append(path)
        else:
            sys.exit(f"not found: {item}")
    if loose:
        scenes.append(Scene("+".join(p.stem for p in loose), loose))
    if not scenes:
        sys.exit("no input: give snapshot dirs / image files, or --snapshot / --all / --csv")
    return scenes


# ---------------------------------------------------------------- brick by colour (digital zoom)


def find_brick(rgb: np.ndarray) -> Box | None:
    """Box of the coloured brick lying on the white sheet, or None.

    Sheet = bright, unsaturated pixels (largest such region after closing seams and thin lines).
    Brick = saturated, non-purple blob inside the sheet's convex hull, mostly surrounded by
    unsaturated pixels (paper or its shadow), preferably not cut by the frame border. This rejects
    wood, off-sheet bricks, plates (big, at the border), the purple gripper and its screw glints (tiny).
    Tuned and checked by eye on all 49 snapshots of project 0fd76ff2 (every camera: brick found).
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    k = max(3, round(min(h, w) / 100)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    sheet = ((sat < 60) & (val > 110)).astype(np.uint8)
    sheet = cv2.morphologyEx(sheet, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sheet, connectivity=8)
    if n < 2:
        return None
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[main, cv2.CC_STAT_AREA] < 0.1 * h * w:
        return None
    hull_pts = cv2.convexHull(cv2.findNonZero((labels == main).astype(np.uint8)))
    inside = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inside, hull_pts, 1)

    unsat = sat < 70
    hue = hsv[..., 0]  # OpenCV: 0..179
    gripper = (hue >= 118) & (hue <= 165)  # the SO-101 gripper is purple/violet; bricks so far are not
    colour = ((sat > 70) & (val > 60) & ~gripper).astype(np.uint8) & inside
    colour = cv2.morphologyEx(colour, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    colour = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, kernel)  # rejoin faces split by a marker line
    n, labels, stats, _ = cv2.connectedComponentsWithStats(colour, connectivity=8)
    candidates = []  # (ring score, area, box) of blobs inside the frame
    edge_candidates = []  # same for blobs cut by the frame border (wrist view: brick at the top edge)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.0008 * h * w or area > 0.08 * h * w:  # < 0.08 %: screw glints, specks
            continue
        on_edge = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        if on_edge and area > 0.025 * h * w:  # plates, table, gripper reach the border and are big
            continue
        if max(bw, bh) > 4 * min(bw, bh):  # thin streaks: paper edges, cables
            continue
        pad = max(4, round(0.35 * max(bw, bh)))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool) & ~comp.astype(bool)
        ring &= ~gripper[y0:y1, x0:x1]  # a brick held by / next to the gripper is still on paper
        if not ring.any():
            continue
        score = float(unsat[y0:y1, x0:x1][ring].mean())
        if score > 0.6:
            (edge_candidates if on_edge else candidates).append(
                (score, int(area), (int(x), int(y), int(x + bw), int(y + bh)))
            )
    candidates = candidates or edge_candidates
    if not candidates:
        return None
    top = max(c[0] for c in candidates)  # best-ringed blob; near-ties go to the larger one
    return max((c for c in candidates if c[0] >= top - 0.05), key=lambda c: c[1])[2]


def zoom_crop(img: Image.Image, box: Box, factor: float, side: int) -> Image.Image:
    """Square crop of `factor` x the box size around its centre, kept inside the frame, resized to side."""
    w, h = img.size
    x1, y1, x2, y2 = box
    size = min(max(round(factor * max(x2 - x1, y2 - y1)), 48), w, h)
    left = min(max(round((x1 + x2) / 2 - size / 2), 0), w - size)
    top = min(max(round((y1 + y2) / 2 - size / 2), 0), h - size)
    return img.crop((left, top, left + size, top + size)).resize((side, side), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------- request

ImageLike = Path | Image.Image


def encode_image(image: ImageLike, max_side: int) -> str:
    if isinstance(image, Path) and max_side <= 0 and image.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        data = image.read_bytes()
        mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    else:
        img = (Image.open(image) if isinstance(image, Path) else image).convert("RGB")
        if max_side > 0 and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data, mime = buf.getvalue(), "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def http_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as err:
        body = err.read().decode(errors="replace")[:500]
        raise RuntimeError(f"HTTP {err.code} from {url}: {body}") from None


def served_model(base_url: str) -> str:
    """The only model the server has loaded (OVMS /v3/models), so --model is optional."""
    try:
        models = [m["id"] for m in http_json(f"{base_url}/models", timeout=5).get("data", [])]
    except (OSError, RuntimeError) as err:
        sys.exit(f"cannot reach the VLM server at {base_url} ({err}); start it: scripts/vlm_server.sh start")
    if len(models) != 1:
        sys.exit(f"server has models {models}; choose one with --model")
    return models[0]


def ask(images: list[ImageLike], text: str, schema: dict, args) -> tuple[dict, float, dict]:
    """One chat request: images + text, answer constrained to `schema`. Returns (answer, seconds, usage)."""
    content = [{"type": "image_url", "image_url": {"url": encode_image(im, args.max_side)}} for im in images]
    content.append({"type": "text", "text": text})
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}},
    }
    start = time.perf_counter()
    resp = http_json(f"{args.url}/chat/completions", payload, timeout=args.timeout)
    latency = time.perf_counter() - start
    text_out = resp["choices"][0]["message"]["content"]
    try:
        answer = json.loads(text_out)
    except ValueError:
        raise RuntimeError(f"answer is not JSON: {text_out!r}") from None
    return answer, latency, resp.get("usage") or {}


def verdict(images: list[ImageLike], prompt: str, args) -> tuple[bool, float, dict]:
    answer, latency, usage = ask(images, prompt, REASON_SCHEMA if args.reason else SCHEMA, args)
    if not isinstance(answer, dict) or not isinstance(answer.get("is_broken"), bool):
        raise RuntimeError(f"answer does not match the schema: {answer!r}")  # noqa: TRY004
    if args.reason:
        usage = {**usage, "observation": answer.get("observation")}
    return answer["is_broken"], latency, usage


# --locator vlm: ask the served model for the box instead of the colour search. Each model family is
# asked in the box format it was trained on; both answer in coordinates normalized to 0..1000:
#   Qwen3-VL: "bbox_2d": [x1, y1, x2, y2]      Gemma: "box_2d": [ymin, xmin, ymax, xmax]
BOX_FORMATS = {"qwen": ("bbox_2d", "[x1, y1, x2, y2]", "xyxy"), "gemma": ("box_2d", "[ymin, xmin, ymax, xmax]", "yxyx")}


def locate_vlm(image: Path, args) -> tuple[Box | None, float]:
    key, spec, order = BOX_FORMATS["gemma" if "gemma" in args.model.lower() else "qwen"]
    prompt = (
        "Find the toy brick that lies on the white paper sheet in this photo (ignore bricks that are off "
        f"the sheet). Give its bounding box as {key} = {spec}, normalized to 0-1000. If there is no brick "
        "on the sheet, set found to false."
    )
    schema = {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            key: {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        },
        "required": ["found", key],
        "additionalProperties": False,
    }
    answer, latency, _ = ask([image], prompt, schema, args)
    box = answer.get(key) or []
    if not answer.get("found") or len(box) != 4:
        return None, latency
    w, h = Image.open(image).size
    a, b, c, d = box
    x1, y1, x2, y2 = (a, b, c, d) if order == "xyxy" else (b, a, d, c)
    x1, x2 = sorted((x1 * w / 1000, x2 * w / 1000))
    y1, y2 = sorted((y1 * h / 1000, y2 * h / 1000))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, latency
    return (round(x1), round(y1), round(x2), round(y2)), latency


def scene_images(scene: Scene, args) -> tuple[list[ImageLike], list[str], dict, float, float]:
    """Images to send for --view, their names, the brick boxes, colour-search and VLM-locate seconds."""
    if args.view == "frames":
        return list(scene.images), [p.name for p in scene.images], {}, 0.0, 0.0
    crops: list[ImageLike] = []
    names, boxes = [], {}
    find_s = locate_s = 0.0
    for path in scene.images:
        img = Image.open(path).convert("RGB")
        if args.locator == "vlm":
            box, seconds = locate_vlm(path, args)
            locate_s += seconds
        else:
            start = time.perf_counter()
            box = find_brick(np.asarray(img))
            find_s += time.perf_counter() - start
        boxes[path.name] = box
        cam = path.name.split(".")[0]
        if box is None:  # brick not found: send the whole frame rather than nothing
            crops.append(path)
            names.append(path.name)
            continue
        crop = zoom_crop(img, box, args.zoom, args.crop_side)
        crops.append(crop)
        names.append(f"{cam}.crop")
        if args.save_crops:
            out = args.save_crops / scene.name / f"{cam}.crop.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out)
    if args.view == "both":
        return list(scene.images) + crops, [p.name for p in scene.images] + names, boxes, find_s, locate_s
    return crops, names, boxes, find_s, locate_s


def judge(scene: Scene, args, prompt: str) -> dict:
    images, names, boxes, find_s, locate_s = scene_images(scene, args)
    result: dict = {"scene": scene.name, "label": scene.label, "model": args.model, "view": args.view,
                    "mode": args.mode, "images": len(images)}
    if args.mode == "multi":
        is_broken, latency, usage = verdict(images, prompt, args)
        result.update(is_broken=is_broken, latency_s=round(latency, 3), prompt_tokens=usage.get("prompt_tokens"))
        if args.reason:
            result["observation"] = usage.get("observation")
    else:  # per-image: one request per image, broken if any image says so
        start = time.perf_counter()
        if args.parallel:
            with ThreadPoolExecutor(max_workers=len(images)) as pool:
                answers = list(pool.map(lambda im: verdict([im], prompt, args), images))
        else:
            answers = [verdict([im], prompt, args) for im in images]
        result.update(
            is_broken=any(a[0] for a in answers),
            latency_s=round(time.perf_counter() - start, 3),
            per_image={n: a[0] for n, a in zip(names, answers)},
            per_image_latency_s={n: round(a[1], 3) for n, a in zip(names, answers)},
        )
    if args.view != "frames":
        result.update(boxes=boxes, found=sum(b is not None for b in boxes.values()), find_s=round(find_s, 3))
    if locate_s:
        result.update(locate_s=round(locate_s, 3))
    result["max_side"] = args.max_side
    return result


# ---------------------------------------------------------------- report


def print_eval(results: list[dict]) -> None:
    labelled = [r for r in results if r.get("label") is not None and "is_broken" in r]
    if not labelled:
        print("eval: no labelled scenes", file=sys.stderr)
        return
    tp = sum(r["label"] and r["is_broken"] for r in labelled)
    fn = sum(r["label"] and not r["is_broken"] for r in labelled)
    fp = sum(not r["label"] and r["is_broken"] for r in labelled)
    tn = sum(not r["label"] and not r["is_broken"] for r in labelled)
    n = len(labelled)
    print(f"\nconfusion matrix ({n} labelled scenes; label = 'defective'):")
    print("                     predicted broken   predicted ok")
    print(f"  label defective    {tp:>16}   {fn:>12}")
    print(f"  label ok           {fp:>16}   {tn:>12}")
    print(f"accuracy {(tp + tn) / n:.3f} ({tp + tn}/{n})")
    if fp + tn:
        print(f"false-positive rate {fp / (fp + tn):.3f} ({fp}/{fp + tn})")
    if tp + fn:
        print(f"recall (defective found) {tp / (tp + fn):.3f} ({tp}/{tp + fn})")
    wrong = [r["scene"] for r in labelled if r["label"] != r["is_broken"]]
    if wrong:
        print("wrong: " + " ".join(wrong))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="snapshot dirs (one scene each) and/or image files (one scene)")
    ap.add_argument("--project", help="Studio project id (prefix ok) or path; default: the only project")
    ap.add_argument("--snapshot", action="append", help="snapshot id in the project (repeatable)")
    ap.add_argument("--all", action="store_true", help="every snapshot dir of the project")
    ap.add_argument("--csv", nargs="?", const="", help="read scenes + labels from snapshots.csv (default: the project's)")
    ap.add_argument("--eval", action="store_true", help="compare with the 'defective' label, print a confusion matrix")
    ap.add_argument("--model", help="served model name (default: the only model on the server)")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"OVMS OpenAI API base (default {DEFAULT_URL})")
    ap.add_argument("--prompt-file", type=Path, help="prompt text file (default: the built-in verbatim prompt)")
    ap.add_argument("--view", choices=["crops", "frames", "both"], default="crops",
                    help="crops: zoomed brick crop per camera (default); frames: full frames; both: frames + crops")
    ap.add_argument("--locator", choices=["color", "vlm"], default="color",
                    help="how crops find the brick: colour search on the sheet (default) or a VLM box request")
    ap.add_argument("--zoom", type=float, default=2.4, help="crop side as a multiple of the brick size (2.4)")
    ap.add_argument("--crop-side", type=int, default=448, help="crop size sent to the model, px (448)")
    ap.add_argument("--save-crops", type=Path, help="save crops to DIR/<scene>/<camera>.crop.png")
    ap.add_argument("--mode", choices=["multi", "per-image"], default="multi",
                    help="multi: all images in one request (default); per-image: one request per image, OR")
    ap.add_argument("--parallel", action="store_true", help="per-image mode: send the requests concurrently")
    ap.add_argument("--max-side", type=int, default=0, help="downscale so the longer side is at most N px (0 = off)")
    ap.add_argument("--reason", action="store_true",
                    help='schema {"observation": string, "is_broken": boolean}: describe first, then decide')
    ap.add_argument("--max-tokens", type=int, default=None, help="default 64, or 160 with --reason")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", type=Path, help="append one JSON line per scene to this file")
    ap.add_argument("--limit", type=int, help="only the first N scenes")
    ap.add_argument("--skip", action="append", default=[], help="snapshot id to leave out (repeatable)")
    ap.add_argument("--no-vlm", action="store_true", help="only find bricks / save crops, no requests")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to wait between scenes (batch evals while Studio records: keep the GPU duty low)")
    args = ap.parse_args()

    prompt = args.prompt_file.read_text().strip() if args.prompt_file else DEFAULT_PROMPT
    args.max_tokens = args.max_tokens or (160 if args.reason else 64)
    scenes = [s for s in collect_scenes(args) if s.name not in args.skip][: args.limit]
    if args.no_vlm:
        if args.locator == "vlm" or args.view == "frames":
            sys.exit("--no-vlm needs --locator color and --view crops/both")
        for scene in scenes:
            _, _, boxes, find_s, _ = scene_images(scene, args)
            print(json.dumps({"scene": scene.name, "label": scene.label, "boxes": boxes, "find_s": round(find_s, 3)}))
        return
    args.model = args.model or served_model(args.url)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in scenes:
        try:
            result = judge(scene, args, prompt)
        except (OSError, RuntimeError) as err:
            result = {"scene": scene.name, "label": scene.label, "error": str(err)}
        results.append(result)
        print(json.dumps(result), flush=True)
        if args.pause:
            time.sleep(args.pause)
        if args.out:
            with args.out.open("a") as f:
                f.write(json.dumps(result) + "\n")

    latencies = [r["latency_s"] for r in results if "latency_s" in r]
    errors = sum("error" in r for r in results)
    if len(latencies) > 1:
        print(
            f"\n{len(latencies)} scenes, latency mean {statistics.mean(latencies):.2f} s, "
            f"median {statistics.median(latencies):.2f} s, max {max(latencies):.2f} s (first {latencies[0]:.2f} s)"
        )
    if errors:
        print(f"{errors} scene(s) failed", file=sys.stderr)
    if args.eval:
        print_eval(results)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

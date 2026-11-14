#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["google-genai>=1.40", "pillow>=11", "numpy>=2"]
# ///
"""Label every brick crop of an Anomalib dataset: "is a black marker mark visible in THIS crop?"

One request per crop to Gemini on Google AI Studio, temperature 0, JSON-schema output
{"mark_visible": bool, "confidence": "low|medium|high", "where": str}.

Privacy: only the brick is sent. Each crop is masked with its brick hull (masks/<crop_id>.png from
cropping.brick_dataset), everything else is white, then cut tight around the brick and upscaled.
Full frames are never sent (wrist and floor views show people in the background).

Key: AI_STUDIO_API_KEY from the environment or from .env in the repo root (never printed or logged).
Cache: captures/anomalib/labels/gemini_responses.jsonl (raw responses, append-only), keyed on
the crop (snapshot id or episode/frame, camera, brick index), its crop box, the model and the
prompt. A re-run only queries new crops (e.g. new snapshots) or crops whose box changed.

Run from the repo root (uv reads the inline dependencies above; the project env is not used):
  uv run scripts/label_crops_gemini.py --dataset bricks_v1                 # every snapshot crop (default)
  uv run scripts/label_crops_gemini.py --dataset bricks_v1 --apply         # + move defective-brick crops to
                                                                           #   test/defect or test/defect_clean
  uv run scripts/label_crops_gemini.py --dataset bricks_v1 --which all     # also every dataset-episode crop
  ... --model gemini-3.1-pro-preview --concurrency 3
configs/defect_mark_labels.csv holds manual spot-check labels (true/false/unsure); they are compared
with Gemini, and beat it only with --manual-wins. Good-brick crops that Gemini flags are only
reported, never moved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("captures/anomalib/labels/gemini_responses.jsonl")
OVERRIDES = REPO / "configs/defect_mark_labels.csv"  # manual spot-check labels (true/false/unsure)

PROMPT = """This image shows ONE toy building brick (LEGO/Duplo style), cut out of a camera photo.
Everything around the brick was replaced with plain white on purpose.
Some bricks were deliberately marked by hand with a BLACK MARKER PEN: a drawn line, arc, tick,
dot, letter, number or scribble on the plastic surface.

Question: is such a hand-drawn marker mark visible on the brick in THIS image?

Do NOT count as a mark: the brick's own molded shapes (studs, stud outlines, the round tube and
hollow underside, holes, edges, corners, seams between faces), shadows, dark edges, reflections or
highlights, blur or compression noise, colour variation of the plastic, or the slightly dark rim
left by the white cut-out.
Answer mark_visible=true only if you can actually see a drawn mark. If you are unsure, say false
with low confidence.
"where": where the mark is and what it looks like (e.g. "thin black arc on the front face, left");
empty string if there is no mark."""

SCHEMA = {
    "type": "object",
    "properties": {
        "mark_visible": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "where": {"type": "string"},
    },
    "required": ["mark_visible", "confidence", "where"],
    "additionalProperties": False,
}


def api_key() -> str:
    key = os.environ.get("AI_STUDIO_API_KEY")
    for env in (Path.cwd() / ".env", REPO / ".env"):
        if key or not env.exists():
            continue
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip().removeprefix("export ").strip() == "AI_STUDIO_API_KEY":
                key = v.strip().strip('"').strip("'")
    if not key:
        sys.exit("AI_STUDIO_API_KEY not set (environment or .env in the repo root)")
    return key


def brick_only_png(img_path: Path, mask_path: Path, out_side: int = 768) -> bytes:
    """White everywhere outside the (slightly grown) brick hull; tight square around it; upscaled."""
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)
    grow = max(3, (img.size[0] // 64) | 1)
    mask = mask.filter(ImageFilter.MaxFilter(grow))
    white = Image.new("RGB", img.size, (255, 255, 255))
    img = Image.composite(img, white, mask)
    m = np.asarray(mask) > 0
    if m.any():
        ys, xs = np.nonzero(m)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.65 + 4
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
        canvas.paste(img.crop((max(box[0], 0), max(box[1], 0), min(box[2], img.size[0]), min(box[3], img.size[1]))),
                     (max(-box[0], 0), max(-box[1], 0)))
        img = canvas
    img = img.resize((out_side, out_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_cache() -> dict[str, dict]:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if "response" in rec:
                    out[rec["key"]] = rec
    return out


class Labeler:
    def __init__(self, model: str, concurrency: int, api_version: str = "v1beta"):
        from google import genai
        from google.genai import types

        self.types = types
        # Preview / -latest models are served on v1beta only ("not found for API version v1").
        self.client = genai.Client(api_key=api_key(), http_options=types.HttpOptions(api_version=api_version))
        self.model = model
        self.lock = threading.Lock()
        self.prompt_sha = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]
        self.concurrency = concurrency

    def key(self, row: dict) -> str:
        """Crop identity: scene + camera + brick index (crop_id), its box, model and prompt."""
        box = ",".join(row[k] for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        return f"{row['crop_id']}|{box}|{self.model}|{self.prompt_sha}"

    def ask(self, png: bytes, crop_id: str) -> dict:
        from google.genai import errors

        t = self.types
        cfg = t.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                      response_json_schema=SCHEMA)
        delay = 5.0
        for attempt in range(8):
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=[t.Part.from_bytes(data=png, mime_type="image/png"), PROMPT], config=cfg)
                text = resp.text or ""
                return {"response": json.loads(text), "raw_text": text, "model_version": resp.model_version,
                        "latency_s": round(time.time() - t0, 2), "attempts": attempt + 1,
                        "usage": resp.usage_metadata.model_dump(exclude_none=True) if resp.usage_metadata else None}
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code in (429, 500, 502, 503, 504) and attempt < 7:
                    wait = delay * (1 + random.random())
                    print(f"  {crop_id}: HTTP {code}, retry in {wait:.0f} s", flush=True)
                    time.sleep(wait)
                    delay = min(delay * 2, 120)
                    continue
                return {"error": f"HTTP {code}: {str(exc)[:300]}"}
            except (json.JSONDecodeError, ValueError) as exc:
                return {"error": f"bad JSON: {exc}"}
        return {"error": "gave up after retries"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="name under captures/anomalib/dataset/")
    p.add_argument("--model", default="gemini-pro-latest")
    p.add_argument("--which", choices=("snapshots", "defect", "test", "all"), default="snapshots",
                   help="snapshots: every crop of every state snapshot (default); defect: defective bricks only; "
                        "test: every test crop; all: every crop")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--limit", type=int, help="label at most N new crops (for a trial)")
    p.add_argument("--apply", action="store_true", help="move defective-brick crops by their per-crop label")
    p.add_argument("--manual-wins", action="store_true",
                   help="with --apply: a true/false label in configs/defect_mark_labels.csv beats Gemini")
    a = p.parse_args()

    root = Path("captures/anomalib/dataset") / a.dataset
    with open(root / "index.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for c in ("gemini_mark_visible", "gemini_confidence", "gemini_where", "gemini_model",
              "manual_mark_visible", "mark_label_source"):
        if c not in fields:
            fields.append(c)
    todo = {"snapshots": [r for r in rows if r["source"].startswith("snapshot:")],
            "defect": [r for r in rows if r["label"] == "defect"],
            "test": [r for r in rows if r["split"] == "test"], "all": rows}[a.which]

    cache = load_cache()
    lab = Labeler(a.model, a.concurrency)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["_key"] = lab.key(r)
    jobs = [r for r in todo if r["_key"] not in cache]
    print(f"{len(todo)} crops selected, {len(todo) - len(jobs)} cached, {len(jobs)} new ({a.model})", flush=True)
    if a.limit is not None:
        jobs = jobs[: a.limit]

    def work(r):
        png = brick_only_png(root / r["path"], root / "masks" / f"{r['crop_id']}.png")
        res = lab.ask(png, r["crop_id"])
        rec = {"key": r["_key"], "crop_id": r["crop_id"], "dataset": a.dataset, "path": r["path"],
               "label": r["label"], "model_requested": a.model, "prompt_sha": lab.prompt_sha,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **res}
        with lab.lock, CACHE.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            rec = fut.result()
            done += 1
            if "response" in rec:
                cache[rec["key"]] = rec
                flag = "MARK" if rec["response"]["mark_visible"] else "-"
            else:
                flag = "ERROR " + rec["error"][:80]
            if done % 20 == 0 or flag != "-":
                print(f"[{done}/{len(jobs)} {time.time() - t0:.0f} s] {rec['crop_id']} ({rec['label']}): {flag}", flush=True)

    overrides = {}
    if OVERRIDES.exists():
        with open(OVERRIDES, newline="") as f:
            overrides = {o["crop_id"]: o["mark_visible"].strip().lower() for o in csv.DictReader(f)}

    flagged_good, moved = [], 0
    for r in rows:
        rec = cache.get(r.pop("_key", None) or "")
        if rec:
            resp = rec["response"]
            r.update(gemini_mark_visible=str(bool(resp["mark_visible"])).lower(), gemini_confidence=resp["confidence"],
                     gemini_where=resp["where"], gemini_model=rec.get("model_version") or rec["model_requested"])
            if r["label"] == "good" and resp["mark_visible"]:
                flagged_good.append(r)
        if r["crop_id"] in overrides:
            r["manual_mark_visible"] = overrides[r["crop_id"]]
        if a.apply and r["label"] == "defect":
            if a.manual_wins and overrides.get(r["crop_id"]) in ("true", "false"):
                vis, src = overrides[r["crop_id"]] == "true", "manual"
            elif r.get("gemini_mark_visible"):
                vis, src = r["gemini_mark_visible"] == "true", "gemini"
            else:
                continue
            cls = "defect" if vis else "defect_clean"
            new = f"test/{cls}/{Path(r['path']).name}"
            if new != r["path"]:
                (root / new).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(root / r["path"], root / new)
                r["path"], moved = new, moved + 1
            r["cls"], r["mark_label_source"] = cls, src

    with open(root / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    labelled = [r for r in rows if r.get("gemini_mark_visible")]
    print(f"\ncrops with a Gemini label: {len(labelled)} (models: "
          f"{sorted({r['gemini_model'] for r in labelled})})")
    defect = [r for r in rows if r["label"] == "defect"]
    vis = [r for r in defect if r.get("gemini_mark_visible") == "true"]
    print(f"defective-brick crops: {len(defect)}, Gemini says mark visible in {len(vis)}")
    for cam in sorted({r["view"] for r in defect}):
        d = [r for r in defect if r["view"] == cam and r.get("gemini_mark_visible")]
        v = sum(r["gemini_mark_visible"] == "true" for r in d)
        print(f"  {cam:6s} mark visible in {v}/{len(d)} crops ({100 * v / max(len(d), 1):.0f} %)")
    scenes = {r["scene"] for r in defect}
    seen = {r["scene"] for r in vis}
    print(f"  defective bricks with the mark visible in at least one crop: {len(seen)}/{len(scenes)}")
    both = [r for r in defect if r.get("manual_mark_visible") in ("true", "false") and r.get("gemini_mark_visible")]
    unsure = [r for r in defect if r.get("manual_mark_visible") == "unsure" and r.get("gemini_mark_visible")]
    if unsure:
        print(f"manual 'unsure' crops ({len(unsure)}): Gemini says mark visible in "
              f"{sum(r['gemini_mark_visible'] == 'true' for r in unsure)}")
    if both:
        agree = sum(r["manual_mark_visible"] == r["gemini_mark_visible"] for r in both)
        print(f"manual vs Gemini on {len(both)} defective crops: {agree} agree ({100 * agree / len(both):.0f} %)")
        for r in both:
            if r["manual_mark_visible"] != r["gemini_mark_visible"]:
                print(f"  disagree {r['crop_id']}: manual {r['manual_mark_visible']}, Gemini {r['gemini_mark_visible']} "
                      f"({r['gemini_confidence']}: {r['gemini_where']})")
    print(f"good-brick crops flagged by Gemini: {len(flagged_good)}")
    for r in flagged_good:
        print(f"  {r['path']} ({r['gemini_confidence']}): {r['gemini_where']}")
    if a.apply:
        counts = {c: sum(1 for r in rows if r["cls"] == c) for c in ("defect", "defect_clean", "defect_unlabelled")}
        print(f"applied: moved {moved} files; {counts}")
    ds = root / "dataset.json"
    if ds.exists():
        summary = json.loads(ds.read_text())
        summary["images"] = {}
        for r in rows:
            k = f"{r['split']}/{r['cls']}"
            summary["images"][k] = summary["images"].get(k, 0) + 1
        summary["mark_labels"] = {"model": a.model, "prompt_sha": lab.prompt_sha, "applied": a.apply,
                                  "flagged_good": [r["crop_id"] for r in flagged_good]}
        ds.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

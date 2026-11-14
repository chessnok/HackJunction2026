"""Copy a brick dataset with AUGMENTED good train crops (flips, rot90, colour jitter); test split unchanged.

    anomalib-env/bin/python -m hackjunction2026.defect.cropping.augment_normals --src bricks_z14t --name bricks_z14ta

Bricks look the same flipped or turned by 90 degrees, so rotated / flipped / slightly re-lit copies of
good crops widen what PatchCore's memory bank calls normal. Only train/good changes; test/ and masks
are copied as they are, so results compare 1:1 with the source dataset.
PatchCore coreset time grows with the number of train crops (T-019): keep the total near 550.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path("captures/anomalib/dataset")


def jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild brightness / contrast / saturation / hue change; white background stays white-ish."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.85, 1.15)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


GEOM = {
    "hflip": lambda im: cv2.flip(im, 1),
    "vflip": lambda im: cv2.flip(im, 0),
    "rot90": lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
    "rot180": lambda im: cv2.rotate(im, cv2.ROTATE_180),
    "rot270": lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--episode-train", type=int, default=150, help="episode train crops kept (each + 1 geometric copy)")
    p.add_argument("--snapshot-copies", default="rot90,rot180,jitter",
                   help="extra copies per good-snapshot train crop (from hflip,vflip,rot90,rot180,rot270,jitter)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    src, out = OUT / a.src, OUT / a.name
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(a.seed)
    with open(src / "index.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or []) + ["augment"]
        rows = list(reader)

    keep = [r for r in rows if r["split"] != "train"]
    for d in ("test", "masks"):
        shutil.copytree(src / d, out / d)
    if (src / "excluded").exists():
        shutil.copytree(src / "excluded", out / "excluded")
    (out / "train/good").mkdir(parents=True)

    train = [r for r in rows if r["split"] == "train" and r["cls"] == "good"]
    snap = [r for r in train if r["source"].startswith("snapshot:")]
    ep = [r for r in train if r["source"].startswith("dataset:")]
    ep = [ep[i] for i in sorted(rng.choice(len(ep), size=min(a.episode_train, len(ep)), replace=False))]
    new_rows = []

    def put(r: dict, img: np.ndarray, tag: str) -> None:
        name = Path(r["path"]).stem + (f"_{tag}" if tag else "") + ".png"
        cv2.imwrite(str(out / "train/good" / name), img)
        new_rows.append({**r, "path": f"train/good/{name}", "augment": tag})

    for r in ep:
        img = cv2.imread(str(src / r["path"]))
        put(r, img, "")
        tag = str(rng.choice(list(GEOM)))
        put(r, GEOM[tag](img), tag)
    for r in snap:
        img = cv2.imread(str(src / r["path"]))
        put(r, img, "")
        for tag in a.snapshot_copies.split(","):
            put(r, jitter(img, rng) if tag == "jitter" else GEOM[tag](img), tag)

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(new_rows + keep)
    summary = json.loads((src / "dataset.json").read_text())
    summary.update(created=time.strftime("%Y-%m-%dT%H:%M:%S"), augmented_from=a.src,
                   augment={"episode_train": len(ep), "snapshot_train": len(snap), "snapshot_copies": a.snapshot_copies,
                            "train_crops": len(new_rows)})
    images: dict[str, int] = {}
    for r in new_rows + keep:
        images[f"{r['split']}/{r['cls']}"] = images.get(f"{r['split']}/{r['cls']}", 0) + 1
    summary["images"] = images
    (out / "dataset.json").write_text(json.dumps(summary, indent=2))
    print(f"{out}: {len(new_rows)} train crops ({len(ep)} episode x2, {len(snap)} snapshot x"
          f"{1 + len(a.snapshot_copies.split(','))}); test unchanged: {images}")


if __name__ == "__main__":
    main()

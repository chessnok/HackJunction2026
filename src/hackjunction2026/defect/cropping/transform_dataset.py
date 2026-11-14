"""Copy a brick dataset (and its -synth twin) with an image transform applied to EVERY crop.

    anomalib-env/bin/python -m hackjunction2026.defect.cropping.transform_dataset --src bricks_z14t --name bricks_z14c --clahe

Same split, same index, same masks; only the pixels change, so results compare 1:1 with the source.
--clahe: CLAHE on the Lab luminance (clip 2.0, 8x8 tiles), a per-crop contrast normalisation that
lifts faint dark marks on dark bricks. The runtime must apply the same transform (CropConfig.clahe).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from hackjunction2026.defect.cropping.bricks import clahe_luminance

OUT = Path("captures/anomalib/dataset")


def copy_transformed(src: Path, dst: Path, clahe: bool) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.png"))
    n = 0
    for f in src.rglob("*.png"):
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if rel.parts[0] in ("masks", "stroke_masks") or not clahe:
            shutil.copy2(f, out)
            continue
        cv2.imwrite(str(out), clahe_luminance(cv2.imread(str(f))))
        n += 1
    meta = dst / "dataset.json"
    if meta.exists():
        d = json.loads(meta.read_text())
        d["transform"] = {"source": src.name, "clahe": clahe}
        d.setdefault("crop", {})["clahe"] = clahe
        meta.write_text(json.dumps(d, indent=2))
    return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--clahe", action="store_true")
    a = p.parse_args(argv)
    n = copy_transformed(OUT / a.src, OUT / a.name, a.clahe)
    if (OUT / f"{a.src}-synth").exists():
        n += copy_transformed(OUT / f"{a.src}-synth", OUT / f"{a.name}-synth", a.clahe)
    print(f"{OUT / a.name}: {n} crops transformed")


if __name__ == "__main__":
    main()

"""Generate a synthetic capture session (no cameras) to test the dataset + training pipeline.

    uv run --extra capture python -m hackjunction2026.defect.capture.mock_session \
        --output-dir /tmp/mock_raw --session mock --good-cycles 12 --defect-cycles 3

Writes exactly what capture_inspection writes (same Session code, manifest, file names), with drawn
bricks: 2x2 in three colours, 4x2 and a slope brick, small random shifts / rotations / lighting
changes, and defects (scratch, missing stud, paint blob, crack) that each appear in one view only.
The numbers you get from training on this are meaningless for the real task; it only proves that
the pipeline runs end to end.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from hackjunction2026.defect.capture.cameras import Frame
from hackjunction2026.defect.capture.capture_inspection import ROLES, Session, git_sha
from hackjunction2026.defect.cropping.crops import Box, check_view

W, H, BOX = 1280, 720, 240
BRICKS = {  # name: (colour BGR, studs x, studs y, slope)
    "2x2red": ((40, 40, 200), 2, 2, False),
    "2x2blue": ((190, 90, 30), 2, 2, False),
    "2x2yellow": ((40, 200, 230), 2, 2, False),
    "4x2": ((60, 160, 60), 4, 2, False),
    "slope": ((40, 110, 220), 2, 2, True),
}
DEFECTS = ("scratch", "missing_stud", "blob", "crack")


def shade(c: tuple, f: float) -> tuple:
    return tuple(int(min(255, max(0, v * f))) for v in c)


def draw_top(color: tuple, sx: int, sy: int, slope: bool, rng: np.random.Generator) -> np.ndarray:
    img = np.full((H, W, 3), 175, np.uint8)
    unit = 34
    bw, bh = sx * unit, sy * unit
    cx, cy = W // 2 + rng.integers(-4, 5), H // 2 + rng.integers(-4, 5)
    x0, y0 = cx - bw // 2, cy - bh // 2
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), color, -1)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), shade(color, 0.6), 2)
    rows = 1 if slope else sy
    if slope:  # sloped half: gradient instead of studs
        for i in range(bh // 2):
            cv2.line(img, (x0 + 1, y0 + bh // 2 + i), (x0 + bw - 1, y0 + bh // 2 + i), shade(color, 1.25 - i / bh), 1)
    for i in range(sx):
        for j in range(rows):
            p = (x0 + unit // 2 + i * unit, y0 + unit // 2 + j * unit)
            cv2.circle(img, p, 11, shade(color, 1.15), -1)
            cv2.circle(img, p, 11, shade(color, 0.7), 1)
            cv2.circle(img, (p[0] - 3, p[1] - 3), 3, shade(color, 1.5), -1)
    return img


def draw_side(color: tuple, sx: int, slope: bool, rng: np.random.Generator) -> np.ndarray:
    img = np.full((H, W, 3), 165, np.uint8)
    cx, cy = W // 2 + rng.integers(-4, 5), H // 2 + 20 + rng.integers(-4, 5)
    fw, rw, h = 22 * sx, 44, 60  # front face F, right face R (brick at 45° to the camera)
    f = np.array([[cx - fw, cy - h // 2], [cx, cy - h // 2 + 12], [cx, cy + h // 2 + 12], [cx - fw, cy + h // 2]])
    r = np.array([[cx, cy - h // 2 + 12], [cx + rw, cy - h // 2], [cx + rw, cy + h // 2], [cx, cy + h // 2 + 12]])
    cv2.fillPoly(img, [f], color)
    cv2.fillPoly(img, [r], shade(color, 0.75))
    if slope:
        cv2.fillPoly(img, [np.array([[cx - fw, cy - h // 2], [cx, cy - h // 2 + 12], [cx, cy - 4], [cx - fw, cy - 16]])],
                     shade(color, 1.2))
    for i in range(sx):  # studs on the top edge
        px = cx - fw + 11 + i * (fw // sx)
        cv2.ellipse(img, (px, cy - h // 2 - 2 + i * 12 // sx), (8, 5), 0, 180, 360, shade(color, 1.1), -1)
    return img


def add_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> None:
    cx, cy = W // 2 + rng.integers(-20, 21), H // 2 + rng.integers(-15, 16)
    if kind == "scratch":
        cv2.line(img, (cx - 25, cy - 8), (cx + 25, cy + 10), (35, 35, 35), 3)
    elif kind == "crack":
        pts = np.array([[cx - 20, cy - 15], [cx - 5, cy - 2], [cx + 2, cy + 12], [cx + 18, cy + 20]])
        cv2.polylines(img, [pts], False, (20, 20, 20), 2)
    elif kind == "blob":
        cv2.circle(img, (cx, cy), 9, (245, 245, 245), -1)
    elif kind == "missing_stud":  # flat patch where a stud should be
        color = img[cy, cx].tolist()
        cv2.circle(img, (W // 2 - 17, H // 2 - 17), 13, color, -1)


def finish(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-3, 3)
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REPLICATE)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.92, 1.08), beta=rng.uniform(-8, 8))
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/inspection.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("captures/anomalib/raw"))
    p.add_argument("--session", default="mock")
    p.add_argument("--good-cycles", type=int, default=12, help="good cycles per brick")
    p.add_argument("--defect-cycles", type=int, default=3, help="defect cycles per brick")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    cfg = yaml.safe_load(a.config.read_text())
    cfg["output_dir"] = str(a.output_dir)
    rng = np.random.default_rng(a.seed)
    box = Box(W // 2 - BOX // 2, H // 2 - BOX // 2, BOX, BOX)
    crops = dict.fromkeys(ROLES, box)
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_sha(), "mock": True,
            "cameras": {r: {"serial": f"mock-{r}", "exposure": {"mode": "mock"}, "crop_box": box.as_list()}
                        for r in ROLES}, "save_margin": cfg["save_margin"], "views": cfg["views"]}
    session = Session(a.output_dir / a.session, cfg, crops, meta)
    n = 0
    view_of = {"top": ("top_pre", "top_post"), "side": ("side_FR", "side_BL")}

    for brick, (color, sx, sy, slope) in BRICKS.items():
        for label, count in (("good", a.good_cycles), ("defect", a.defect_cycles)):
            session.set_label(label)
            session.set_brick(brick)
            for _ in range(count):
                kind = rng.choice(DEFECTS) if label == "defect" else None
                where = (rng.choice(["top", "side"]), int(rng.integers(0, 2))) if kind else None  # role, 0=pre 1=post
                if label == "defect":
                    session.note = f"mock {kind}"
                    session.defect_views = [view_of[where[0]][where[1]]]
                for pos_i in (0, 1):
                    imgs = {"top": draw_top(color, sx, sy, slope, rng), "side": draw_side(color, sx, slope, rng)}
                    if where and where[1] == pos_i:
                        add_defect(imgs[where[0]], kind, rng)
                    frames = {}
                    for role in ROLES:
                        n += 1
                        frames[role] = Frame(finish(imgs[role], rng), time.monotonic() * 1000, n)
                    checks = {r: check_view(frames[r].bgr, box, cfg["check"]) for r in ROLES}
                    session.save(frames, checks, check_enabled=False)
    c = session.counts()
    print(f"mock session {session.root}: {c['good']} good + {c['defect']} defect cycles, "
          f"{len(session.records) * 2} images")


if __name__ == "__main__":
    main()

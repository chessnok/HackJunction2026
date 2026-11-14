"""Capture the Anomalib inspection dataset with the two D415 cameras (top + side).

Status: untested on the real cameras (written 2026-11-14 off the laptop; tested with --fake only).

The window shows both cameras live; photos are saved ONLY when you press SPACE, so you can swap
bricks, turn them and pick defective ones in between. One cycle = one placement of one brick:

    PRE  brick pushed into the L-shape   -> SPACE saves top_pre  + side_FR
    POST brick turned 180° (vertical)    -> SPACE saves top_post + side_BL   (cycle += 1)

Keys:  SPACE save   b brick   g/d good/defect   f defect note + views   c check on/off   u undo   q quit
       (b, g, d only at the start of a cycle; text is typed in the terminal)

For a defective brick you also say which views show the defect (e.g. a scratch on face F is only
visible in side_FR). The dataset step uses that to label each image; the other views of that brick
are not used as "defect" images.

Files: captures/anomalib/raw/<session>/{good,defect}/<brick>_<cycle>_<view>.png + manifest.jsonl.
Each PNG is the crop box enlarged by `save_margin`, the manifest says where the crop box is inside
it. Cropping to 256x256 happens later (prepare step) with the same fixed box.

Run on the laptop, from the repo root, with Studio's backend environment (pyrealsense2, opencv):

    PYTHONPATH=src uv run --no-sync --project third_party/physical-ai-studio/application/backend \
        python -m hackjunction2026.defect.capture.capture_inspection [--list-cameras | --set-crops | ...]

(scripts/capture_inspection.sh does exactly this.) Stop recording / Run model / camera preview
in Studio first: only one program can use a camera. No robot motion happens in this tool.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from hackjunction2026.defect.capture.cameras import FakeCamera, RealSenseColor, WebcamCamera, list_realsense
from hackjunction2026.defect.cropping.crops import Box, CheckResult, check_view, cut

ROLES = ("top", "side")
NAME_RE = re.compile(r"[^a-z0-9]+")
WINDOW = "capture_inspection"


# --------------------------------------------------------------------------- config helpers
def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def centred_crops(frames: dict) -> dict[str, Box]:
    """Test modes (--fake / --webcam): a square box in the middle of each frame."""
    out = {}
    for role, frame in frames.items():
        h, w = frame.bgr.shape[:2]
        s = h // 3
        out[role] = Box(w // 2 - s // 2, h // 2 - s // 2, s, s)
    return out


def load_crops(cfg: dict) -> dict[str, Box]:
    path = Path(cfg["crops_file"])
    if not path.exists():
        sys.exit(f"{path} not found: run with --set-crops first")
    data = yaml.safe_load(path.read_text())
    size = [cfg["cameras"]["width"], cfg["cameras"]["height"]]
    if data.get("frame_size") != size:
        sys.exit(f"{path} was made for frame size {data.get('frame_size')}, config uses {size}: "
                 "run --set-crops again")
    for role in ROLES:
        if data["cameras"][role]["serial"] != str(cfg["cameras"][role]["serial"]):
            sys.exit(f"{path}: {role} serial differs from configs/inspection.yaml, run --set-crops again")
    return {r: Box.from_list(data["cameras"][r]["box"]) for r in ROLES}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def clean_name(text: str) -> str:
    return NAME_RE.sub("", text.strip().lower())


# --------------------------------------------------------------------------- session (no GUI)
@dataclass
class Session:
    """Naming, saving, manifest and undo. Kept free of GUI code so it can be tested."""

    root: Path
    cfg: dict
    crops: dict[str, Box]
    meta: dict
    brick: str = ""
    label: str = "good"
    note: str = ""
    defect_views: list[str] = field(default_factory=list)  # empty = all views
    position: str = "pre"
    cycle: int = 0
    records: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for label in ("good", "defect"):
            (self.root / label).mkdir(exist_ok=True)
        manifest = self.root / "manifest.jsonl"
        if manifest.exists():  # continue an earlier session
            self.records = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        info = self.root / "session.json"
        sessions = json.loads(info.read_text())["starts"] if info.exists() else []
        info.write_text(json.dumps({"starts": [*sessions, self.meta]}, indent=2))

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    def resume(self) -> None:
        """Set cycle/position for the current brick + label from what is already saved.

        An unfinished cycle (PRE saved, POST missing) is continued at POST; otherwise a new cycle starts.
        """
        mine = [r for r in self.records if r["brick"] == self.brick and r["label"] == self.label]
        if not mine:
            self.cycle, self.position = 0, "pre"
            return
        last = max(r["cycle"] for r in mine)
        positions = {r["position"] for r in mine if r["cycle"] == last}
        if positions == {"pre"}:
            self.cycle, self.position = last, "post"
        else:
            self.cycle, self.position = last + 1, "pre"

    def set_brick(self, brick: str) -> None:
        self.brick = brick
        self.resume()

    def set_label(self, label: str) -> None:
        self.label = label
        if label == "good":
            self.note, self.defect_views = "", []
        self.resume()

    def save(self, frames: dict, checks: dict[str, CheckResult], check_enabled: bool) -> dict:
        views = self.cfg["views"][self.position]
        rec = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "brick": self.brick, "label": self.label,
            "cycle": self.cycle, "position": self.position, "defect_note": self.note or None,
            "defect_views": (self.defect_views or None) if self.label == "defect" else None,
            "files": {}, "region": {}, "box_in_region": {}, "camera": {}, "frame": {}, "frame_size": {},
            "check_enabled": check_enabled,
            "check": {r: {"ok": c.ok, "reason": c.reason, "brick_fraction": round(c.brick_fraction, 4),
                          "sharpness": round(c.sharpness, 1)} for r, c in checks.items()},
        }
        for role in ROLES:
            frame = frames[role]
            h, w = frame.bgr.shape[:2]
            box = self.crops[role]
            region = box.expand(self.cfg["save_margin"], w, h)
            view = views[role]
            rel = f"{self.label}/{self.brick}_{self.cycle:03d}_{view}.png"
            if not cv2.imwrite(str(self.root / rel), cut(frame.bgr, region)):
                msg = f"could not write {rel}"
                raise OSError(msg)
            rec["files"][view] = rel
            rec["region"][view] = region.as_list()
            rec["frame_size"][view] = [w, h]
            rec["box_in_region"][view] = box.relative_to(region).as_list()
            rec["camera"][view] = self.meta["cameras"][role]["serial"]
            rec["frame"][view] = {"number": frame.number, "timestamp_ms": round(frame.timestamp_ms, 1)}
        stamps = [frames[r].timestamp_ms for r in ROLES]
        rec["camera_skew_ms"] = round(max(stamps) - min(stamps), 1)
        with open(self.manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.records.append(rec)
        if self.position == "pre":
            self.position = "post"
        else:
            self.position, self.cycle = "pre", self.cycle + 1
        return rec

    def undo(self) -> dict | None:
        if not self.records:
            return None
        rec = self.records.pop()
        for rel in rec["files"].values():
            (self.root / rel).unlink(missing_ok=True)
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in self.records))
        self.brick, self.label, self.cycle, self.position = rec["brick"], rec["label"], rec["cycle"], rec["position"]
        self.note = rec["defect_note"] or ""
        self.defect_views = rec.get("defect_views") or []
        return rec

    def counts(self) -> dict[str, int]:
        out = {"good": 0, "defect": 0}
        for r in self.records:
            if r["position"] == "post":
                out[r["label"]] += 1
        return out


# --------------------------------------------------------------------------- cameras
def open_cameras(cfg: dict, fake: bool, webcam: int | None = None) -> dict:
    c = cfg["cameras"]
    if webcam is not None:
        shared = WebcamCamera(webcam, c["width"], c["height"], c["fps"])
        cams = dict.fromkeys(ROLES, shared)
    else:
        cls = FakeCamera if fake else RealSenseColor
        cams = {r: cls(str(c[r]["serial"]), c["width"], c["height"], c["fps"]) for r in ROLES}
    started = []
    try:
        for cam in cams.values():
            cam.start()
            started.append(cam)
    except BaseException:
        for cam in started:
            cam.stop()
        raise
    return cams


def read_both(cams: dict) -> dict:
    return {r: cams[r].read() for r in ROLES}


# --------------------------------------------------------------------------- GUI helpers
def require_gui() -> None:
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        sys.exit(f"OpenCV has no GUI support here ({exc}). In Studio's env, opencv-python-headless may "
                 "shadow opencv-python; check: python -c 'import cv2; print(cv2.__file__)'")


def render(frames: dict, crops: dict[str, Box], checks: dict[str, CheckResult], lines: list[str],
           margin: float) -> np.ndarray:
    tiles = []
    for role in ROLES:
        img = frames[role].bgr.copy()
        h, w = img.shape[:2]
        box, ok = crops[role], checks[role].ok
        reg = box.expand(margin, w, h)
        cv2.rectangle(img, (reg.x, reg.y), (reg.x + reg.w, reg.y + reg.h), (160, 160, 160), 1)
        cv2.rectangle(img, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 200, 0) if ok else (0, 0, 255), 3)
        cv2.putText(img, f"{role}: {checks[role].reason}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 200, 0) if ok else (0, 0, 255), 2)
        scale = 640 / w
        tiles.append(cv2.resize(img, (640, int(h * scale))))
    view = np.hstack(tiles)
    bar = np.zeros((28 * len(lines) + 12, view.shape[1], 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    return np.vstack([view, bar])


def ask(prompt: str) -> str:
    print(prompt, end="", flush=True)
    return input()


def all_views(cfg: dict) -> list[str]:
    return [v for pos in ("pre", "post") for v in cfg["views"][pos].values()]


def parse_views(text: str, cfg: dict) -> list[str]:
    """'side_FR, top_pre' -> ['side_FR', 'top_pre']; empty -> [] (= all views). Unknown names: error."""
    names = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    valid = all_views(cfg)
    lower = {v.lower(): v for v in valid}
    out = []
    for n in names:
        if n.lower() not in lower:
            msg = f"unknown view {n!r}; valid: {', '.join(valid)}"
            raise ValueError(msg)
        out.append(lower[n.lower()])
    return out


def ask_defect(session: Session) -> None:
    session.note = ask("defect note (e.g. 'scratch on F'): ").strip()
    valid = ", ".join(all_views(session.cfg))
    while True:
        try:
            session.defect_views = parse_views(
                ask(f"which views show it? comma list of {valid} (Enter = all): "), session.cfg)
            return
        except ValueError as exc:
            print(exc)


# --------------------------------------------------------------------------- modes
def mode_list_cameras() -> None:
    devices = list_realsense()
    if not devices:
        print("no RealSense camera found (USB cable? Studio using it?)")
    for d in devices:
        print(f"{d['name']:<28} serial {d['serial']:<14} USB {d['usb']:<5} firmware {d['firmware']}")
    print("Put the serials into configs/inspection.yaml (cameras.top / cameras.side) and record them "
          "in docs/hardware.md + configs/hardware.yaml.")


def mode_set_crops(cfg: dict, cams: dict, path: Path) -> None:
    for _ in range(10):  # let auto exposure settle a bit for a usable picture
        frames = read_both(cams)
    out = {"frame_size": [cfg["cameras"]["width"], cfg["cameras"]["height"]], "cameras": {}}
    for role in ROLES:
        print(f"[{role}] drag a box around the L-shape with a brick in it (a bit larger than the "
              "biggest brick), then ENTER; c = cancel")
        x, y, w, h = cv2.selectROI(WINDOW, frames[role].bgr, showCrosshair=True)
        if w == 0 or h == 0:
            sys.exit("cancelled, nothing written")
        side = max(w, h)  # square box: the model input is square
        out["cameras"][role] = {"serial": cams[role].serial,
                                "box": [int(x + w / 2 - side / 2), int(y + h / 2 - side / 2), side, side]}
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Generated by capture_inspection --set-crops. Square crop boxes [x, y, w, h] per camera.\n"
    path.write_text(header + yaml.safe_dump(out, sort_keys=False))
    print(f"wrote {path}: {out['cameras']}")


def make_session(args: argparse.Namespace, cfg: dict, crops: dict, cams: dict, exposure: dict) -> Session:
    name = args.session or time.strftime("%Y-%m-%d_%H%M")
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_sha(), "fake": args.fake,
            "webcam": args.webcam,
            "cameras": {r: {"serial": cams[r].serial, "exposure": exposure[r], "crop_box": crops[r].as_list(),
                            "requested_size": [cfg["cameras"]["width"], cfg["cameras"]["height"]]}
                        for r in ROLES},
            "save_margin": cfg["save_margin"], "views": cfg["views"]}
    session = Session(Path(cfg["output_dir"]) / name, cfg, crops, meta)
    session.set_label(args.label)
    session.set_brick(clean_name(args.brick) if args.brick else clean_name(ask("brick name (e.g. 2x2red): ")))
    if not session.brick:
        sys.exit("empty brick name")
    if session.label == "defect":
        if args.note is not None or args.defect_views is not None:
            session.note = args.note or ""
            session.defect_views = parse_views(args.defect_views or "", cfg)
        else:
            ask_defect(session)
    return session


def mode_once(session: Session, cams: dict) -> None:
    """Save the position the session expects next (PRE, or POST of an unfinished cycle)."""
    print(f"--once: saving {session.brick} cycle {session.cycle:03d} {session.position.upper()}")
    frames = read_both(cams)
    checks = {r: check_view(frames[r].bgr, session.crops[r], session.cfg["check"]) for r in ROLES}
    if session.cfg["check"]["enabled"] and not all(c.ok for c in checks.values()):
        sys.exit("check failed: " + "; ".join(f"{r}: {c.reason}" for r, c in checks.items()))
    rec = session.save(frames, checks, session.cfg["check"]["enabled"])
    print("saved", rec["files"])


def mode_interactive(session: Session, cams: dict) -> None:
    cfg = session.cfg
    check_enabled = bool(cfg["check"]["enabled"])
    message = "ready"
    while True:
        frames = read_both(cams)
        checks = {r: check_view(frames[r].bgr, session.crops[r], cfg["check"]) for r in ROLES}
        views = cfg["views"][session.position]
        counts = session.counts()
        lines = [
            (f"brick {session.brick}   label {session.label.upper()}   cycle {session.cycle:03d}   "
             f"next {session.position.upper()} ({views['top']} + {views['side']})   "
             f"done: good {counts['good']} / defect {counts['defect']} cycles"),
            f"note: {session.note or '-'}"
            + (f" (views: {', '.join(session.defect_views) or 'all'})" if session.label == "defect" else "")
            + f"   check {'ON' if check_enabled else 'OFF'}   {message}",
            "SPACE save   b brick   g/d good/defect   f note   c check   u undo   q quit",
        ]
        cv2.imshow(WINDOW, render(frames, session.crops, checks, lines, cfg["save_margin"]))
        key = cv2.waitKey(1) & 0xFF
        at_start = session.position == "pre"

        if key in (ord("q"), 27):
            if session.position == "post":
                print(f"warning: cycle {session.cycle:03d} of {session.brick} has PRE but no POST; "
                      "the prepare step drops incomplete cycles")
            break
        if key == ord(" "):
            if check_enabled and not all(c.ok for c in checks.values()):
                message = "NOT saved: " + "; ".join(f"{r} {c.reason}" for r, c in checks.items() if not c.ok)
                continue
            rec = session.save(frames, checks, check_enabled)
            message = f"saved {', '.join(rec['files'])}"
            if rec["position"] == "pre":
                message += "  -> now turn the brick 180° and press SPACE"
            print(message)
        elif key == ord("u"):
            rec = session.undo()
            message = f"undone {', '.join(rec['files'])}" if rec else "nothing to undo"
        elif key == ord("c"):
            check_enabled = not check_enabled
            message = f"check {'ON' if check_enabled else 'OFF'}"
        elif key in (ord("b"), ord("g"), ord("d")) and not at_start:
            message = "finish the cycle first (POST), or undo"
        elif key == ord("b"):
            name = clean_name(ask("new brick name: "))
            if name:
                session.set_brick(name)
                message = f"brick {name}, cycle {session.cycle:03d} {session.position.upper()}"
        elif key in (ord("g"), ord("d")):
            session.set_label("good" if key == ord("g") else "defect")
            if session.label == "defect":
                ask_defect(session)
            message = f"label {session.label}"
        elif key == ord("f"):
            if session.label != "defect":
                message = "notes are for defects: press d first"
                continue
            if not at_start:
                message = "finish the cycle first (POST), or undo"
                continue
            ask_defect(session)
            message = "defect note set"
    cv2.destroyAllWindows()
    c = session.counts()
    print(f"session {session.root}: {c['good']} good + {c['defect']} defect complete cycles")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/inspection.yaml"))
    p.add_argument("--list-cameras", action="store_true", help="print connected RealSense cameras and exit")
    p.add_argument("--set-crops", action="store_true", help="draw the crop box for each camera and exit")
    p.add_argument("--fake", action="store_true", help="synthetic cameras, to test without hardware")
    p.add_argument("--webcam", type=int, metavar="INDEX",
                   help="TEST ONLY: use an ordinary webcam for both roles (never on the station laptop)")
    p.add_argument("--no-check", action="store_true", help="start with the OpenCV check off")
    p.add_argument("--output-dir", type=Path, help="override output_dir from the config (e.g. /tmp for tests)")
    p.add_argument("--session", help="session folder name (default: date_time); reuse a name to continue")
    p.add_argument("--brick", help="brick name, e.g. 2x2red (asked if missing)")
    p.add_argument("--label", choices=("good", "defect"), default="good")
    p.add_argument("--note", help="defect note, e.g. 'scratch on F'")
    p.add_argument("--defect-views", help="views that show the defect, e.g. 'side_FR' (default: all)")
    p.add_argument("--once", action="store_true",
                   help="save the next position (PRE, or POST of an unfinished cycle) without the window, exit")
    args = p.parse_args(argv)

    if args.list_cameras:
        mode_list_cameras()
        return
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = str(args.output_dir)
    if args.no_check:
        cfg["check"]["enabled"] = False
    test_mode = args.fake or args.webcam is not None
    if args.defect_views:
        try:
            parse_views(args.defect_views, cfg)
        except ValueError as exc:
            sys.exit(str(exc))
    if not args.once:
        require_gui()  # before cameras and the session folder are touched
    cams = open_cameras(cfg, args.fake, args.webcam)
    try:
        if args.set_crops:
            # test modes never overwrite the real crop file
            path = Path(cfg["output_dir"]) / "inspection_crops_test.yaml" if test_mode else Path(cfg["crops_file"])
            mode_set_crops(cfg, cams, path)
            return
        crops = centred_crops(read_both(cams)) if test_mode else load_crops(cfg)
        exposure = {r: cams[r].configure_exposure(cfg["exposure"]) for r in ROLES}
        print("exposure:", json.dumps(exposure))
        session = make_session(args, cfg, crops, cams, exposure)
        if args.once:
            mode_once(session, cams)
        else:
            mode_interactive(session, cams)
    finally:
        for cam in set(cams.values()):
            cam.stop()


if __name__ == "__main__":
    main()

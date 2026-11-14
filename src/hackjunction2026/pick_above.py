"""Brick in the wrist camera -> table XY -> follower joint target that hovers above it, gripper down.

Library behind ``scripts/pick_above.py`` (subcommands documented there and in the README section
"Hover above a brick"). Decisions: ``docs/decisions/0007-observe-pose-homography-ik.md``.

Offline by default: ``--image`` reads a file. ``--capture`` grabs a frame from the wrist camera and
reads the follower pose (read-only bus access); both need Studio's environment closed. Nothing here
moves the robot: ``plan`` prints a ``so101_goto.py goto`` command for a human to run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hackjunction2026 import table_map as tm
from hackjunction2026.kinematics import TABLE_Z_MM, Calibration, ik, jaw_gap_mm, normalized_to_urdf
from hackjunction2026.kinematics.path import check_straight_move

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
POSES = CONFIGS / "poses.yaml"
HARDWARE = CONFIGS / "hardware.yaml"
FOLLOWER_CAL = CONFIGS / "calibration/follower.json"
DEFAULT_HOMOGRAPHY = CONFIGS / "table_homography.yaml"
SNAPSHOTS = Path.home() / ".local/share/physicalai/state_snapshots"
CAPTURES = REPO / "captures/pick_above"
STUDIO_PY = "third_party/physical-ai-studio/application/backend/.venv/bin/python"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CALIB_RE = re.compile(r"^\s*calib\s+([A-Za-z][\w-]*)", re.IGNORECASE)

# Words people may type in a "calib <colour>" comment -> the detector's colour names.
COLOUR_ALIASES = {
    "lightgreen": "lime", "light-green": "lime", "light_green": "lime", "limegreen": "lime",
    "darkgreen": "green", "dark-green": "green",
    "lightblue": "blue", "light-blue": "blue", "azure": "blue", "cyan": "blue", "darkblue": "blue",
    "magenta": "pink",
}


def _yaml():
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pyyaml missing: run with `uv run --extra table ...`") from e
    return yaml


def vec(d: dict) -> np.ndarray:
    """6 joint values from a dict keyed by joint name or by Studio's '<joint>.pos'."""
    return np.array([float(d[j] if j in d else d[f"{j}.pos"]) for j in JOINTS])


def load_pose(name: str) -> np.ndarray:
    poses = _yaml().safe_load(POSES.read_text(encoding="utf-8")) or {}
    if name not in poses:
        raise SystemExit(f"pose '{name}' not in {POSES.relative_to(REPO)} (saved: {', '.join(poses)})")
    return vec(poses[name]["joints"])


def fmt_joints(q: np.ndarray) -> str:
    return "  ".join(f"{j}={v:7.2f}" for j, v in zip(JOINTS, q, strict=True))


def goto_arg(q: np.ndarray) -> str:
    return ",".join(f"{v:.2f}" for v in q)


# ----------------------------------------------------------------------------------------------
# Snapshots and calibration points
# ----------------------------------------------------------------------------------------------

@dataclass
class CalibPoint:
    colour: str
    snapshot: str            # snapshot id, e.g. 20261114-061502-123
    path: Path
    joints: np.ndarray       # normalized (6,)
    xyz_mm: np.ndarray       # tool point, base frame
    tilt_deg: float          # gripper axis vs straight down
    jaw_gap_mm: float
    calibration_differs: bool


def iter_snapshots(root: Path = SNAPSHOTS):
    """(snapshot dir, state dict) for every Studio state snapshot under root, oldest first."""
    for state in sorted(root.glob("**/state.json"), key=lambda p: p.parent.name):
        try:
            yield state.parent, json.loads(state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue


def calib_points(root: Path = SNAPSHOTS, since: str | None = None, point: str = "jaw_centre",
                 cal_file: Path = FOLLOWER_CAL) -> tuple[list[CalibPoint], list[str]]:
    """Latest 'calib <colour>' snapshot per colour -> FK tool point. Returns (points, notes)."""
    ref = Calibration.from_file(cal_file)
    found: dict[str, list[CalibPoint]] = {}
    notes: list[str] = []
    for d, st in iter_snapshots(root):
        if since and d.name < since:
            continue
        m = CALIB_RE.match(st.get("comment") or "")
        if not m:
            continue
        colour = m.group(1).lower()
        colour = COLOUR_ALIASES.get(colour, colour)
        fol = (st.get("robots") or {}).get("follower") or {}
        if "joint_positions" not in fol:
            notes.append(f"{d.name}: no follower joint positions, skipped")
            continue
        q = vec(fol["joint_positions"])
        cal = Calibration.from_dict(fol["calibration"]) if fol.get("calibration") else ref
        differs = not (np.array_equal(cal.range_min, ref.range_min) and np.array_equal(cal.range_max, ref.range_max))
        tp = ik.tool_pose(normalized_to_urdf(q, cal), point)
        gap = float(jaw_gap_mm(normalized_to_urdf(q, cal)[5]))
        found.setdefault(colour, []).append(CalibPoint(colour, d.name, d, q, tp["position"],
                                                       float(ik.tilt_deg(tp["axis"])), gap, differs))
    points = []
    for colour, lst in sorted(found.items()):
        points.append(lst[-1])
        if len(lst) > 1:
            notes.append(f"calib {colour}: using the latest snapshot {lst[-1].snapshot}, ignoring "
                         + ", ".join(p.snapshot for p in lst[:-1]))
    for p in points:
        if p.calibration_differs:
            notes.append(f"{p.snapshot}: the follower calibration in the snapshot differs from "
                         f"{cal_file.name}; FK used the snapshot's calibration")
        if p.tilt_deg > 25:
            notes.append(f"calib {p.colour}: gripper tilted {p.tilt_deg:.0f} deg from vertical; hold it "
                         "more upright over the brick for a better point")
        if p.xyz_mm[2] - TABLE_Z_MM > 50:
            notes.append(f"calib {p.colour}: tool point {p.xyz_mm[2] - TABLE_Z_MM:.0f} mm above the table; "
                         "bring the fingertips down to the brick")
    return points, notes


# ----------------------------------------------------------------------------------------------
# Images: file, snapshot or wrist camera
# ----------------------------------------------------------------------------------------------

@dataclass
class Frame:
    image: np.ndarray
    source: str
    measured_joints: np.ndarray | None  # follower pose when the frame was taken, if known


def load_frame(path: str) -> Frame:
    """An image file, or a snapshot directory (uses its arm.color.png and the follower joints)."""
    import cv2

    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "arm.color.png"
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"cannot read image {p}")
    joints = None
    state = p.parent / "state.json"
    if state.exists():
        try:
            joints = vec(json.loads(state.read_text())["robots"]["follower"]["joint_positions"])
        except (KeyError, ValueError, json.JSONDecodeError):
            joints = None
    return Frame(img, str(p), joints)


def wrist_camera_name() -> str:
    hw = _yaml().safe_load(HARDWARE.read_text(encoding="utf-8"))
    return str(hw["cameras"]["wrist"]["name"])


FORBIDDEN_CAMERA_WORDS = ("HP", "5MP", "IR Camera", "RealSense")


def resolve_wrist_camera(name: str, by_id: Path = Path("/dev/v4l/by-id"),
                         sysfs: Path = Path("/sys/class/video4linux")) -> Path:
    """/dev/v4l/by-id link of the wrist camera's capture node, found by its USB name (never a
    /dev/videoN number). Refuses anything that looks like the laptop's built-in HP camera."""
    links = sorted(p for p in by_id.glob("*-video-index0") if name in p.name)
    links = [p for p in links if not any(w in p.name for w in FORBIDDEN_CAMERA_WORDS)]
    if len(links) != 1:
        raise SystemExit(f"wrist camera '{name}': {len(links)} matching links in {by_id} "
                         f"({', '.join(p.name for p in links) or 'none'}); is it plugged in?")
    node = links[0].resolve().name  # e.g. video4, only used to read its sysfs name
    sys_name_file = sysfs / node / "name"
    if sys_name_file.exists():
        sys_name = sys_name_file.read_text().strip()
        if name not in sys_name or any(w in sys_name for w in FORBIDDEN_CAMERA_WORDS):
            raise SystemExit(f"{links[0]} -> {node} reports '{sys_name}', not the wrist camera '{name}'")
    return links[0]


def grab_wrist_frame(size: tuple[int, int], warmup: int = 15) -> np.ndarray:
    """One BGR frame from the wrist camera at ``size`` (width, height). UNTESTED on the hardware."""
    import cv2

    from hackjunction2026.ports import port_users

    link = resolve_wrist_camera(wrist_camera_name())
    dev = str(link.resolve())
    if users := port_users(dev):
        raise SystemExit(f"{link.name} is open in {', '.join(users)}: close the environment in Studio first")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise SystemExit(f"cannot open {link}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        frame = None
        for _ in range(warmup + 1):  # let auto exposure settle
            ok, f = cap.read()
            if ok:
                frame = f
    finally:
        cap.release()
    if frame is None:
        raise SystemExit(f"no frame from {link}")
    if (frame.shape[1], frame.shape[0]) != tuple(size):
        raise SystemExit(f"camera gave {frame.shape[1]}x{frame.shape[0]}, need {size[0]}x{size[1]}")
    return frame


def read_follower_pose() -> np.ndarray:
    """Follower joints (normalized) read from the servos: READ only, no torque change. UNTESTED."""
    from hackjunction2026.feetech import Bus
    from hackjunction2026.kinematics import ticks_to_normalized
    from hackjunction2026.ports import port_users

    hw = _yaml().safe_load(HARDWARE.read_text(encoding="utf-8"))
    port = next(a["port"] for a in hw["arms"].values() if str(a.get("role", "")).startswith("follower"))
    if users := port_users(port):
        raise SystemExit(f"{port} is in use by {', '.join(users)}: close the environment in Studio first")
    cal_json = json.loads(FOLLOWER_CAL.read_text())["calibration"]
    ticks = []
    with Bus(port) as bus:
        for j in JOINTS:
            pos = bus.read(cal_json[j]["id"], "pos")
            if pos is None:
                raise SystemExit(f"follower servo {j} (ID {cal_json[j]['id']}) does not answer")
            ticks.append(pos)
    return ticks_to_normalized(np.array(ticks, float), Calibration.from_file(FOLLOWER_CAL))


def capture_frame(size: tuple[int, int], check_pose: bool) -> Frame:
    """Wrist camera frame + the follower pose read before it (UNTESTED on the hardware)."""
    import cv2

    joints = read_follower_pose() if check_pose else None
    if joints is None:
        print("WARNING: follower pose not read (--no-pose-check): cannot confirm the arm is in the observe pose")
    img = grab_wrist_frame(size)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    out = CAPTURES / f"capture-{datetime.datetime.now():%Y%m%d-%H%M%S}.png"
    cv2.imwrite(str(out), img)  # captures/ is git-ignored
    return Frame(img, str(out), joints)


def get_frame(args, size: tuple[int, int] = (1024, 768)) -> Frame:
    if args.capture:
        return capture_frame(size, check_pose=not args.no_pose_check)
    if not args.image:
        raise SystemExit("give --image PATH (file or snapshot directory) or --capture")
    return load_frame(args.image)


def check_pose(frame: Frame, reference: np.ndarray, tol: float, what: str) -> list[str]:
    """Problems if the frame's arm pose differs from the reference by more than tol units (joints 0-4)."""
    if frame.measured_joints is None:
        return []
    d = np.abs(frame.measured_joints[:5] - reference[:5])
    if d.max() <= tol:
        return []
    worst = int(d.argmax())
    return [(f"the arm was not in the {what} pose when the image was taken: {JOINTS[worst]} off by "
             f"{d[worst]:.1f} units (tolerance {tol})")]


def save_annotated(img: np.ndarray, det: tm.Detection, tag: str) -> Path:
    import cv2

    CAPTURES.mkdir(parents=True, exist_ok=True)
    out = CAPTURES / f"{tag}-{datetime.datetime.now():%Y%m%d-%H%M%S}.png"
    cv2.imwrite(str(out), tm.annotate(img, det))
    return out


# ----------------------------------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------------------------------

@dataclass
class HoverPlan:
    ok: bool
    reason: str
    brick: tm.Brick | None = None
    closing_yaw_deg: float = float("nan")
    size_mm: tuple[float, float] = (float("nan"), float("nan"))
    solution: ik.IKSolution | None = None
    path: object | None = None
    target: np.ndarray | None = None


def brick_size_mm(th: tm.TableHomography, b: tm.Brick) -> tuple[float, float]:
    """(long, short) side of the brick's image rectangle mapped to the table, mm (approximate: the blob
    includes visible side faces)."""
    c = th.to_table(np.array(b.box))
    e = [float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4)]
    a, bb = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
    return max(a, bb), min(a, bb)


def plan_hover(brick: tm.Brick, th: tm.TableHomography, cal: Calibration, start: np.ndarray,
               height_mm: float = 60.0, gripper_norm: float | None = None, clearance_mm: float = 30.0,
               square_ratio: float = 1.25, samples: int = 50) -> HoverPlan:
    """Joint target (normalized, 6) that puts the homography's tool point ``height_mm`` above the table
    over the brick, gripper straight down, jaws closing across the brick's short side; then the
    straight joint-space path check from ``start``."""
    if brick.table_xy is None:
        brick.table_xy = tuple(float(c) for c in th.to_table(brick.centroid))
        brick.long_axis_yaw_deg = th.yaw_deg(brick.centroid, brick.angle_deg)
    x, y = brick.table_xy
    long_yaw = th.yaw_deg(brick.centroid, brick.angle_deg)
    size = brick_size_mm(th, brick)
    yaws = [long_yaw + 90.0]
    if size[0] / max(size[1], 1e-6) < square_ratio:  # (almost) square: either side will do
        yaws.append(long_yaw)
    grip = float(start[5] if gripper_norm is None else gripper_norm)
    z = TABLE_Z_MM + height_mm
    best: ik.IKSolution | None = None
    best_yaw = float("nan")
    reasons = []
    for yaw in yaws:
        sol = ik.solve_down(x, y, z, yaw, cal, grip, start_norm=start, point=th.point)
        if not sol.ok:
            reasons.append(sol.reason)
            continue
        if best is None or np.abs(sol.norm[:5] - start[:5]).max() < np.abs(best.norm[:5] - start[:5]).max():
            best, best_yaw = sol, ((yaw + 90.0) % 180.0) - 90.0
    if best is None:
        return HoverPlan(False, f"IK: {reasons[0]}", brick, float("nan"), size)
    from hackjunction2026.so101_motion import out_of_range

    if problems := out_of_range(best.norm):
        return HoverPlan(False, "target out of range: " + "; ".join(problems), brick, best_yaw, size, best)
    pc = check_straight_move(start, best.norm, cal, samples=samples, min_clearance_mm=clearance_mm)
    if not pc.ok:
        return HoverPlan(False, "path check: " + pc.describe(), brick, best_yaw, size, best, pc, best.norm)
    return HoverPlan(True, "ok", brick, best_yaw, size, best, pc, best.norm)


def pick_brick(bricks: list[tm.Brick], colour: str | None) -> tm.Brick:
    if colour:
        colour = COLOUR_ALIASES.get(colour.lower(), colour.lower())
        sel = [b for b in bricks if b.colour == colour]
        if len(sel) != 1:
            raise SystemExit(f"{len(sel)} {colour} bricks on the sheet (found: "
                             f"{', '.join(b.colour for b in bricks) or 'none'})")
        return sel[0]
    if len(bricks) != 1:
        raise SystemExit(f"{len(bricks)} bricks on the sheet ({', '.join(b.colour for b in bricks) or 'none'}): "
                         "choose one with --brick <colour>")
    return bricks[0]


def outside_calibrated_mm(th: tm.TableHomography, xy) -> float:
    """How far (mm) a table point lies outside the convex hull of the calibration points (0 inside)."""
    import cv2

    pts = np.array([p["table_xy"] for p in th.points if p.get("inlier", True)], float)
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(float)
    c = hull.mean(0)
    # largest outward distance to an edge line (exact next to an edge, a lower bound near a corner)
    xy = np.asarray(xy, float)
    worst = 0.0
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        n = np.array([b[1] - a[1], -(b[0] - a[0])])
        n /= np.linalg.norm(n) + 1e-12
        if np.dot(c - a, n) > 0:
            n = -n
        worst = max(worst, float(np.dot(xy - a, n)))
    return worst


# ----------------------------------------------------------------------------------------------
# Reach map
# ----------------------------------------------------------------------------------------------

def reach_map(cal: Calibration, height_mm: float, gripper_norm: float, point: str = "jaw_centre",
              x_range=(-100, 420), y_range=(-360, 360), step: float = 10.0,
              yaws=(0.0, 45.0, 90.0, 135.0), chunk: int = 600) -> dict:
    """Grid of table XY -> number of jaw directions (out of len(yaws)) with a valid IK solution."""
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([gx.ravel(), gy.ravel()], -1)
    count = np.zeros(len(pts), int)
    z = TABLE_Z_MM + height_mm
    for yaw in yaws:
        for i in range(0, len(pts), chunk):
            p = pts[i:i + chunk]
            tgt = np.column_stack([p, np.full(len(p), z)])
            r = ik.solve_down_batch(tgt, np.full(len(p), yaw), cal, gripper_norm, point=point)
            count[i:i + chunk] += r["ok"]
    return {"xs": xs, "ys": ys, "count": count.reshape(gy.shape), "n_yaws": len(yaws), "step": step}


def draw_reach(rm: dict, th: tm.TableHomography | None, title: str, sheet_px: np.ndarray | None = None,
               bricks: list[tm.Brick] | None = None) -> np.ndarray:
    """Top view (x up, y left as seen from behind the robot), 1 px per mm."""
    import cv2

    xs, ys, cnt, n = rm["xs"], rm["ys"], rm["count"], rm["n_yaws"]
    margin = 60
    w = int(ys[-1] - ys[0]) + 2 * margin
    h = int(xs[-1] - xs[0]) + 2 * margin + 30

    def to_img(x, y):  # +x up, +y left
        return round(margin + (ys[-1] - y)), round(30 + margin + (xs[-1] - x))

    img = np.full((h, w, 3), 255, np.uint8)
    half = rm["step"] / 2
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            c = cnt[iy, ix]
            col = (90, 200, 90) if c == n else (80, 210, 240) if c > 0 else (235, 235, 235)
            u0, v0 = to_img(x + half, y + half)
            u1, v1 = to_img(x - half, y - half)
            cv2.rectangle(img, (u0, v0), (u1, v1), col, -1)
    for v in range(int(np.ceil(xs[0] / 50) * 50), int(xs[-1]) + 1, 50):
        cv2.line(img, to_img(v, ys[0]), to_img(v, ys[-1]), (200, 200, 200), 1)
        cv2.putText(img, f"x={v}", (2, to_img(v, 0)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)
    for v in range(int(np.ceil(ys[0] / 50) * 50), int(ys[-1]) + 1, 50):
        cv2.line(img, to_img(xs[0], v), to_img(xs[-1], v), (200, 200, 200), 1)
        cv2.putText(img, f"{v}", (to_img(0, v)[0] - 10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)
    cv2.circle(img, to_img(0, 0), 5, (0, 0, 0), -1)
    cv2.putText(img, "base", (to_img(0, 0)[0] + 6, to_img(0, 0)[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 0), 1)
    cv2.drawMarker(img, to_img(38.8, 0), (0, 0, 0), cv2.MARKER_CROSS, 10, 1)
    if th is not None and sheet_px is not None:
        poly = th.to_table(sheet_px.astype(float))
        cv2.polylines(img, [np.array([to_img(*p) for p in poly], np.int32).reshape(-1, 1, 2)], True,
                      (200, 120, 0), 2)
    if th is not None:
        for p in th.points:
            cv2.drawMarker(img, to_img(*p["table_xy"]), (0, 0, 200), cv2.MARKER_TILTED_CROSS, 10, 2)
    for b in bricks or []:
        if b.table_xy is not None:
            cv2.circle(img, to_img(*b.table_xy), 7, (0, 0, 255), 2)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"green: all {n} jaw directions, yellow: some, grey: none. 1 px = 1 mm, +x up, +y left",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def _load_h(path: str) -> tm.TableHomography:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no homography at {p}: run calib-fit first (README: Hover above a brick)")
    return tm.TableHomography.load(p)


def cmd_calib_points(args) -> list[CalibPoint]:
    points, notes = calib_points(Path(args.snapshots), args.since, args.point)
    if not points:
        print(f"no snapshots with a 'calib <colour>' comment under {args.snapshots}"
              + (f" since {args.since}" if args.since else ""))
        return points
    print(f"tool point: {args.point}; base frame mm, table at z = {TABLE_Z_MM}")
    print(f"{'colour':8} {'snapshot':20} {'x':>7} {'y':>7} {'z>table':>8} {'tilt':>6} {'jaw gap':>8}")
    for p in points:
        print(f"{p.colour:8} {p.snapshot:20} {p.xyz_mm[0]:7.1f} {p.xyz_mm[1]:7.1f} "
              f"{p.xyz_mm[2] - TABLE_Z_MM:8.1f} {p.tilt_deg:5.1f}° {p.jaw_gap_mm:6.1f}mm")
    for n in notes:
        print("note:", n)
    return points


def cmd_calib_fit(args) -> None:
    import cv2  # noqa: F401  (fail early with a clear message if OpenCV is missing)

    points, notes = calib_points(Path(args.snapshots), args.since, args.point)
    for n in notes:
        print("note:", n)
    if len(points) < 4:
        raise SystemExit(f"need >= 4 calib snapshots of distinct colours, found {len(points)}")
    pose = load_pose(args.pose)
    frame = get_frame(args)
    problems = check_pose(frame, pose, args.pose_tol, args.pose)
    if problems and not args.force:
        raise SystemExit("REFUSED: " + "; ".join(problems) + " (--force to fit anyway)")
    det = tm.detect(frame.image)
    print(f"image {frame.source}: {len(det.bricks)} bricks on the sheet: "
          + ", ".join(f"{b.colour}@({b.centroid[0]:.0f},{b.centroid[1]:.0f})" for b in det.bricks))
    aliases = dict(a.split("=", 1) for a in args.alias)
    px, xy, used = [], [], []
    for p in points:
        want = aliases.get(p.colour, p.colour)
        sel = [b for b in det.bricks if b.colour == want]
        if len(sel) != 1:
            print(f"skip calib {p.colour}: {len(sel)} '{want}' bricks in the image")
            continue
        px.append(sel[0].centroid)
        xy.append(p.xyz_mm[:2])
        used.append((p, sel[0]))
    if len(used) < 4:
        raise SystemExit(f"only {len(used)} calib points matched a brick in the image; need >= 4. "
                         "Use --alias calibcolour=detectedcolour if names differ.")
    px, xy = np.array(px), np.array(xy)
    hmat, inl = tm.fit_homography_ransac(px, xy, threshold_mm=args.ransac_mm)
    fit_res = np.linalg.norm(tm.apply_homography(hmat, px) - xy, axis=1)
    loo = tm.leave_one_out(px[inl], xy[inl])
    loo_all = np.full(len(px), np.nan)
    loo_all[np.flatnonzero(inl)] = loo
    print(f"\n{'colour':8} {'u':>6} {'v':>6} {'x FK':>7} {'y FK':>7} {'fit res':>8} {'LOO res':>8}  inlier")
    rows = []
    for k, (p, b) in enumerate(used):
        print(f"{p.colour:8} {px[k, 0]:6.1f} {px[k, 1]:6.1f} {xy[k, 0]:7.1f} {xy[k, 1]:7.1f} "
              f"{fit_res[k]:7.1f}mm {loo_all[k]:7.1f}mm  {'yes' if inl[k] else 'NO'}")
        rows.append({"colour": p.colour, "snapshot": p.snapshot, "pixel": [round(float(c), 1) for c in px[k]],
                     "table_xy": [round(float(c), 1) for c in xy[k]], "tilt_deg": round(p.tilt_deg, 1),
                     "fit_residual_mm": round(float(fit_res[k]), 2),
                     "loo_residual_mm": None if np.isnan(loo_all[k]) else round(float(loo_all[k]), 2),
                     "inlier": bool(inl[k])})
    loo_ok = loo[np.isfinite(loo)]
    stats = {"n_points": len(px), "n_inliers": int(inl.sum()),
             "fit_rms_mm": round(float(np.sqrt(np.mean(fit_res[inl] ** 2))), 2),
             "loo_rms_mm": round(float(np.sqrt(np.mean(loo_ok ** 2))), 2) if loo_ok.size else None,
             "loo_max_mm": round(float(loo_ok.max()), 2) if loo_ok.size else None}
    print(f"\nfit RMS {stats['fit_rms_mm']} mm over {stats['n_inliers']} inliers; leave-one-out RMS "
          f"{stats['loo_rms_mm']} mm, max {stats['loo_max_mm']} mm")
    if not loo_ok.size:
        print("warning: with 4 points the fit is exact and cannot be validated; calibrate 5-6 bricks")
    bad = loo_ok.size and stats["loo_rms_mm"] > args.max_loo_mm
    th = tm.TableHomography(
        H=hmat, image_size=det.image_size, pose=args.pose,
        pose_joints=dict(zip(JOINTS, map(float, pose), strict=True)), point=args.point, date=tm.now_iso(),
        points=rows, stats=stats,
        note=(f"image {frame.source}; calib snapshots since {args.since or 'any'}"
              + ("" if frame.measured_joints is None else
                 "; measured follower joints at capture: " + goto_arg(frame.measured_joints))),
    )
    if frame.measured_joints is not None:
        th.stats["measured_joints"] = [round(float(v), 2) for v in frame.measured_joints]
    print("annotated:", save_annotated(frame.image, det, "calib-fit"))
    if args.dry_run:
        print("dry run: not saved")
        return
    if bad and not args.force:
        raise SystemExit(f"NOT SAVED: leave-one-out RMS {stats['loo_rms_mm']} mm > {args.max_loo_mm} mm "
                         "(check the matches above; --force to save anyway)")
    th.save(args.out)
    print(f"saved {args.out}")


def _locate(args) -> tuple[Frame, tm.Detection, tm.TableHomography]:
    th = _load_h(args.homography)
    frame = get_frame(args, th.image_size)
    det = tm.detect(frame.image)
    tm.locate(det, th)
    return frame, det, th


def cmd_locate(args) -> None:
    frame, det, th = _locate(args)
    ref = np.array(th.stats["measured_joints"]) if "measured_joints" in th.stats else vec(th.pose_joints)
    for p in check_pose(frame, ref, args.pose_tol, th.pose):
        print("WARNING:", p)
    print(f"image {frame.source}; homography {args.homography} ({th.date}, point {th.point})")
    if not det.bricks:
        print("no bricks on the sheet")
    for b in det.bricks:
        size = brick_size_mm(th, b)
        out = outside_calibrated_mm(th, b.table_xy)
        print(f"{b.colour:7} pixel ({b.centroid[0]:6.1f}, {b.centroid[1]:6.1f})  table x {b.table_xy[0]:6.1f} "
              f"y {b.table_xy[1]:6.1f} mm  long axis {b.long_axis_yaw_deg:6.1f}°  ~{size[0]:.0f}x{size[1]:.0f} mm  "
              f"area {b.area_px} px  hue {b.hue:.0f}" + (f"  ({out:.0f} mm outside the calibrated area)"
                                                         if out > 20 else ""))
    print("annotated:", save_annotated(frame.image, det, "locate"))


def cmd_plan(args) -> None:
    frame, det, th = _locate(args)
    cal = Calibration.from_file(FOLLOWER_CAL)
    ref = np.array(th.stats["measured_joints"]) if "measured_joints" in th.stats else vec(th.pose_joints)
    saved = load_pose(th.pose)
    if np.abs(saved[:5] - vec(th.pose_joints)[:5]).max() > 0.5:
        raise SystemExit(f"pose '{th.pose}' in poses.yaml changed since the homography was fitted: re-run calib-fit")
    problems = check_pose(frame, ref, args.pose_tol, th.pose)
    if problems and not args.force:
        raise SystemExit("REFUSED: " + "; ".join(problems) + " (run so101_goto.py goto "
                         f"{th.pose} --go again, or --force)")
    brick = pick_brick(det.bricks, args.brick)
    start = load_pose(args.start) if args.start else (frame.measured_joints if frame.measured_joints is not None
                                                       else saved)
    hp = plan_hover(brick, th, cal, start, args.height, args.gripper, args.clearance)
    out = outside_calibrated_mm(th, brick.table_xy)
    print(f"brick   {brick.colour} at pixel ({brick.centroid[0]:.0f}, {brick.centroid[1]:.0f}) -> table "
          f"x {brick.table_xy[0]:.1f} y {brick.table_xy[1]:.1f} mm, ~{hp.size_mm[0]:.0f}x{hp.size_mm[1]:.0f} mm")
    if out > 20:
        print(f"WARNING: {out:.0f} mm outside the area covered by the calibration bricks (extrapolated)")
    print(f"start   {fmt_joints(start)}")
    print("annotated:", save_annotated(frame.image, det, "plan"))
    if hp.solution is not None and hp.solution.ok:
        s = hp.solution
        print(f"target  {fmt_joints(s.norm)}")
        print(f"check   {th.point} at ({s.position_mm[0]:.1f}, {s.position_mm[1]:.1f}, "
              f"{s.position_mm[2] - TABLE_Z_MM:.1f} above table) mm, tilt {s.tilt_deg:.3f}°, jaws close along "
              f"{hp.closing_yaw_deg:.1f}° (brick long axis {brick.long_axis_yaw_deg:.1f}°)")
    if hp.path is not None:
        print(f"path    {hp.path.describe()}")
    if not hp.ok:
        raise SystemExit(f"REFUSED: {hp.reason}")
    print("\nMove (human, hands clear; Studio environment closed; arm still in the observe pose):")
    # "--" ends the options: a vector starting with a minus sign is otherwise taken for an option (T-017)
    print(f"  {STUDIO_PY} scripts/so101_goto.py goto -- \"{goto_arg(hp.target)}\"        # dry run")
    print(f"  {STUDIO_PY} scripts/so101_goto.py goto --go -- \"{goto_arg(hp.target)}\"   # move")


def cmd_reach(args) -> None:
    import cv2

    cal = Calibration.from_file(FOLLOWER_CAL)
    pose = load_pose(args.pose)
    grip = float(pose[5] if args.gripper is None else args.gripper)
    rm = reach_map(cal, args.height, grip, args.point, step=args.step)
    a = rm["step"] ** 2 / 100.0  # cm^2 per cell
    full, some = int((rm["count"] == rm["n_yaws"]).sum()), int((rm["count"] > 0).sum())
    xs, ys, cnt = rm["xs"], rm["ys"], rm["count"]
    on_axis = cnt[np.argmin(np.abs(ys)), :] == rm["n_yaws"]
    print(f"height {args.height} mm ({args.point}), gripper {grip:.1f}, grid {rm['step']:.0f} mm")
    print(f"reachable for every jaw direction: {full * a:.0f} cm^2; for some direction: {some * a:.0f} cm^2")
    if on_axis.any():
        print(f"along y = 0: x from {xs[on_axis].min():.0f} to {xs[on_axis].max():.0f} mm")
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    rad = np.hypot(xx - 38.8, yy)[cnt == rm["n_yaws"]]
    if rad.size:
        print(f"distance from the pan axis (x = 38.8): {rad.min():.0f} .. {rad.max():.0f} mm")
    th, sheet, bricks = None, None, None
    if Path(args.homography).exists():
        th = tm.TableHomography.load(args.homography)
        if args.image:
            frame = load_frame(args.image)
            det = tm.detect(frame.image)
            tm.locate(det, th)
            sheet, bricks = det.sheet_polygon, det.bricks
    img = draw_reach(rm, th, f"SO-101 follower, gripper straight down, {args.point} {args.height:.0f} mm above "
                     "the table", sheet, bricks)
    out = Path(args.out or CAPTURES / f"reach_h{args.height:.0f}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print("plot:", out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pick_above.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def img_args(p):
        p.add_argument("--image", help="image file, or a Studio snapshot directory (uses arm.color.png)")
        p.add_argument("--capture", action="store_true",
                       help="grab a frame from the wrist camera (Studio environment must be closed)")
        p.add_argument("--no-pose-check", action="store_true",
                       help="with --capture: do not read the follower pose from the servos")
        p.add_argument("--pose-tol", type=float, default=1.5,
                       help="max joint difference (normalized units) from the pose measured at calibration "
                            "(calib-fit: from the saved pose, default 5 = so101_goto's settle tolerance)")

    def calib_args(p):
        p.add_argument("--snapshots", default=str(SNAPSHOTS), help="Studio state snapshot directory")
        p.add_argument("--since", help="only snapshots with id >= this, e.g. 20261114-060000")
        p.add_argument("--point", default="jaw_centre", choices=ik.TOOL_POINTS, help="gripper point")

    p = sub.add_parser("calib-points", help="FK tool point of every 'calib <colour>' snapshot")
    calib_args(p)
    p.set_defaults(fn=cmd_calib_points)

    p = sub.add_parser("calib-fit", help="detect bricks in the observe image, fit + save the homography")
    calib_args(p)
    img_args(p)
    p.add_argument("--pose", default="observe", help="saved pose the image was taken in")
    p.add_argument("--alias", action="append", default=[], help="calibcolour=detectedcolour")
    p.add_argument("--ransac-mm", type=float, default=15.0)
    p.add_argument("--max-loo-mm", type=float, default=15.0, help="refuse to save above this LOO RMS")
    p.add_argument("--out", default=str(DEFAULT_HOMOGRAPHY))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_calib_fit, pose_tol=5.0)

    p = sub.add_parser("locate", help="bricks on the sheet with their table XY")
    img_args(p)
    p.add_argument("--homography", default=str(DEFAULT_HOMOGRAPHY))
    p.set_defaults(fn=cmd_locate)

    p = sub.add_parser("plan", help="joint target hovering above a brick + path check + goto command")
    img_args(p)
    p.add_argument("--homography", default=str(DEFAULT_HOMOGRAPHY))
    p.add_argument("--height", type=float, default=60.0, help="tool point above the table, mm")
    p.add_argument("--brick", help="colour of the brick to go to (needed when there are several)")
    p.add_argument("--gripper", type=float, help="gripper value at the target (default: keep the start value)")
    p.add_argument("--start", help="saved pose the move starts from (default: measured pose, else observe)")
    p.add_argument("--clearance", type=float, default=30.0, help="min height of every arm point, mm")
    p.add_argument("--force", action="store_true", help="ignore the pose check")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("reach", help="reachable table area, gripper straight down -> PNG under captures/")
    p.add_argument("--height", type=float, default=60.0)
    p.add_argument("--point", default="jaw_centre", choices=ik.TOOL_POINTS)
    p.add_argument("--pose", default="observe", help="its gripper value is used")
    p.add_argument("--gripper", type=float)
    p.add_argument("--step", type=float, default=10.0, help="grid step, mm")
    p.add_argument("--homography", default=str(DEFAULT_HOMOGRAPHY), help="overlay calib points if present")
    p.add_argument("--image", help="with a homography: overlay the sheet and bricks of this image")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_reach)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Sorting cycle for one brick at a time (ADR 0013), as an explicit state machine.

    OBSERVE -> LOCATE -> HOVER -> GRASP (+ servo 6 holding check; up to 3 tries per brick, then OBSERVE)
    -> TO_PHOTO -> RELEASE_PHOTO (open, then RAISE the arm) -> INSPECT (photo + PatchCore + classifier, OR)
    -> REGRASP (scripted: the brick is where we left it) -> TO_BOWL (good / defect) -> RELEASE_BOWL (open,
    raise) -> OBSERVE

Everything that moves the arm goes through :class:`Mover`: a range check, a clearance check of the whole
straight joint-space path (``kinematics.path``), a speed cap, and ``so101_motion.move_linear`` (stops and
holds when the arm cannot follow or on Ctrl+C). Nothing here writes EEPROM; the only register writes are the
gripper's RAM Torque_Limit (address 48) while it holds a brick.

UNTESTED ON THE ROBOT: the holding thresholds, the grasp heights, the ACT grasp and the whole cycle have only
run against a fake arm and fake cameras. See ADR 0013 for what must be measured first.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from hackjunction2026 import so101_motion as m
from hackjunction2026.kinematics import TABLE_Z_MM, Calibration
from hackjunction2026.kinematics import ik
from hackjunction2026.kinematics.mapping import normalized_to_urdf
from hackjunction2026.kinematics.path import check_straight_move

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/sort_cycle.yaml"
POSES = REPO / "configs/poses.yaml"
FOLLOWER_CAL = REPO / "configs/calibration/follower.json"
CAPTURES = REPO / "captures/sort_cycle"
GRIPPER = 5                      # index of the gripper in the joint vector
GRIPPER_ID = 6                   # its bus ID
REG_LOAD, REG_STATUS, REG_TORQUE_LIMIT = 60, 65, 48
MAX_JOINT_SPEED = 15.0           # units/s, hard cap for arm joints (AGENTS.md, ADR 0013)


class StopCycle(Exception):
    """A human stop, a failed move or a refused plan: the arm holds where it is."""


# ---------------------------------------------------------------------------------------------- config

@dataclass
class Config:
    poses: dict = field(default_factory=lambda: {"observe": "observe", "photo": "photo",
                                                 "bowl_good": "bowl_good", "bowl_defect": "bowl_defect"})
    hover_mm: float = 60.0           # jaw centre above the table over the brick
    grasp_mm: float = 12.0           # jaw centre above the table when closing (half a 2x2 DUPLO-size brick, UNTESTED)
    approach_mm: float = 50.0        # pre-pose above a low saved pose (photo, bowls)
    raise_mm: float = 80.0           # straight up after a release, so the cameras see the brick
    travel_speed: float = 10.0
    near_table_speed: float = 4.0
    gripper_speed: float = 30.0      # gripper-only moves
    travel_clearance_mm: float = 30.0
    near_table_clearance_mm: float = 2.0
    gripper_open: float = 90.0
    gripper_close_target: float = 0.0
    empty_closed: float = 2.5        # gripper value when it closes on nothing (agent B: 2.57)
    photo_clear_frac: float = 0.5    # photo: arm this far from the raised photo pose towards observe
    clear_speed: float = 15.0        # units/s for that move (user: faster is fine there)
    retry_raise_mm: float = 30.0
    photo_px: dict | None = None     # camera -> (u, v) where it sees the photo spot (inspection fallback)
    photo_roi_px: int = 260     # each grasp retry hovers this much higher
    ik_gripper: float = 22.6         # jaw value with a brick held: the IK tool point is the CLOSED jaw centre
    hold_min: float = 8.0            # stopped above this when commanded closed -> holding (UNTESTED)
    load_hold_min: int = 150         # |Present_Load| (0.1 % units) above this also counts (UNTESTED)
    squeeze: float = 3.0             # after a stop, command only this much past the stop position
    hold_torque_limit: int = 300     # gripper RAM Torque_Limit while holding (0..1000) (UNTESTED)
    retries_per_brick: int = 3
    locate: str = "auto"             # wrist (homography, needs calibration) | top (top camera model) | auto
    homography: str = "configs/table_homography.yaml"
    top_camera: str = "configs/top_camera.yaml"
    top_serial: str = "243622062352"
    floor_serial: str = "243622062278"
    brick_half_depth_mm: float = 12.0
    patchcore_deploy: str = "configs/anomalib_deploy.yaml"
    classifier_xml: str = "outputs/mark_classifier/20261114-085214_resnet18lr_bricks_z14t/model.xml"
    classifier_threshold: float = 0.998  # 0 % crop false positives in cross-validation (ADR 0006 follow-up)
    detector_device: str = "NPU"
    act_export: str = "~/.local/share/physicalai/models/62003713-0c7d-44f0-878c-436b9fc2ed16/exports/openvino"
    act_device: str = "GPU"
    act_max_s: float = 12.0
    act_replan_every: int = 10
    act_max_speed: float = 10.0

    @classmethod
    def load(cls, path: Path = CONFIG) -> Config:
        cfg = cls()
        if path.exists():
            import yaml

            for k, v in (yaml.safe_load(path.read_text()) or {}).items():
                if not hasattr(cfg, k):
                    raise SystemExit(f"{path}: unknown key {k!r}")
                setattr(cfg, k, v)
        return cfg


def load_poses(cfg: Config) -> tuple[dict[str, np.ndarray], list[str]]:
    """The cycle's poses from configs/poses.yaml, and the names that are missing."""
    import yaml

    saved = (yaml.safe_load(POSES.read_text()) or {}) if POSES.exists() else {}
    poses, missing = {}, []
    for role, name in cfg.poses.items():
        if name in saved:
            poses[role] = m.vec(saved[name]["joints"])
        else:
            missing.append(f"{role} (pose '{name}')")
    return poses, missing


# ---------------------------------------------------------------------------------------------- arm

@dataclass
class GripperState:
    pos: float                   # normalized 0..100
    load: int | None             # signed Present_Load, 0.1 % of max torque
    status: list[str]            # decoded Status register bits

    def holding(self, cfg: Config) -> bool:
        """Holding = it stopped well above the empty-closed value, or it pushes hard, or it tripped a
        protection bit (OVERLOAD / OVERCURRENT). Thresholds are UNTESTED on the robot."""
        return (self.pos >= cfg.hold_min
                or (self.load is not None and abs(self.load) >= cfg.load_hold_min and self.pos > cfg.empty_closed + 1)
                or any(b in self.status for b in ("OVERLOAD", "OVERCURRENT")))


class Arm(Protocol):
    def read(self) -> np.ndarray: ...
    def send(self, q: np.ndarray) -> None: ...
    def gripper_state(self) -> GripperState: ...
    def gripper_torque_limit(self, value: int | None = None) -> int | None: ...
    def gripper_torque_off(self) -> None: ...
    def close(self) -> None: ...


class PluginArm:
    """The follower through Studio's SO-101 plugin (torque on, holds on disconnect), plus raw reads of the
    gripper's Present_Load / Status and RAM writes of its Torque_Limit through the plugin's own bus."""

    def __init__(self) -> None:
        from physicalai_so101_plugin.so101 import SO101

        goto = _so101_goto()
        serial, calib = goto.arm("follower")
        self.robot = SO101(port=goto.port_for(serial), calibration=calib, role="follower", unit="normalized")
        self.robot.connect()
        self.conn = self.robot._require_connection()

    def read(self) -> np.ndarray:
        return self.robot.get_observation().joint_positions.astype(float)

    def send(self, q: np.ndarray) -> None:
        self.robot.send_action(np.asarray(q, dtype=np.float64))

    def _read(self, addr: int, width: int) -> int | None:
        ph, port = self.conn.packet_handler, self.conn.port_handler
        val, comm, err = (ph.read1ByteTxRx if width == 1 else ph.read2ByteTxRx)(port, GRIPPER_ID, addr)
        return None if comm != 0 else int(val)

    def gripper_state(self) -> GripperState:
        from hackjunction2026.feetech import decode_status, sign_mag

        load = self._read(REG_LOAD, 2)
        status = self._read(REG_STATUS, 1)
        return GripperState(float(self.read()[GRIPPER]), sign_mag(load, 10) if load is not None else None,
                            decode_status(status))

    def gripper_torque_limit(self, value: int | None = None) -> int | None:
        """Read (value None) or write the gripper's RAM Torque_Limit (address 48 >= 40: RAM, never EEPROM)."""
        assert REG_TORQUE_LIMIT >= 40
        if value is None:
            return self._read(REG_TORQUE_LIMIT, 2)
        value = int(max(0, min(1000, value)))
        self.conn.packet_handler.write2ByteTxRx(self.conn.port_handler, GRIPPER_ID, REG_TORQUE_LIMIT, value)
        return value

    def gripper_torque_off(self) -> None:
        self.conn.packet_handler.write1ByteTxRx(self.conn.port_handler, GRIPPER_ID, 40, 0)  # Torque_Enable, RAM

    def close(self) -> None:
        self.robot.disconnect()  # follower keeps torque on: holds the pose


class FakeArm:
    """For --dry-run and tests: reaches every target at once; the gripper stops on a brick when told
    (``brick_at`` = the gripper value where the jaws meet the brick, None = nothing between the jaws)."""

    def __init__(self, start: np.ndarray, brick_at: float | None = 32.0) -> None:
        self.q = np.array(start, float)
        self.brick_at = brick_at
        self.limit = 1000
        self.sent: list[np.ndarray] = []

    def read(self) -> np.ndarray:
        return self.q.copy()

    def send(self, q: np.ndarray) -> None:
        q = np.array(q, float)
        stop = self.brick_at if self.brick_at is not None else 2.5
        q[GRIPPER] = max(q[GRIPPER], stop)
        self.q = q
        self.sent.append(q)

    def gripper_state(self) -> GripperState:
        on_brick = self.brick_at is not None and self.q[GRIPPER] <= self.brick_at + 0.5
        return GripperState(float(self.q[GRIPPER]), 400 if on_brick else 20, [])

    def gripper_torque_limit(self, value: int | None = None) -> int | None:
        if value is not None:
            self.limit = value
        return self.limit

    def gripper_torque_off(self) -> None:
        pass

    def close(self) -> None:
        pass


def _so101_goto():
    """scripts/so101_goto.py as a module (arm serial + calibration from Studio or its cache, port checks)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("so101_goto", REPO / "scripts/so101_goto.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "src"))
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------- motion

class Mover:
    """Every move: range check -> clearance check of the straight joint-space path -> speed cap -> prompt
    (step mode) -> move_linear. Dry run: prints the plan and moves the fake arm instantly."""

    def __init__(self, arm: Arm, cal: Calibration, cfg: Config, log: Callable[..., None], *,
                 dry_run: bool, step: bool, ask: Callable[[str], str] = input) -> None:
        self.arm, self.cal, self.cfg, self.log = arm, cal, cfg, log
        self.dry_run, self.step, self.ask = dry_run, step, ask
        self.observe: np.ndarray | None = None  # set by the cycle: the fallback direction for "raise"

    def move(self, target: np.ndarray, what: str, *, near_table: bool = False, speed: float | None = None,
             taught: bool = False) -> np.ndarray:
        """``taught``: the move goes into or out of a pose a human put the arm in (photo spot, bowls). Its own
        lowest point may sit at or a few mm "below" the table in FK (FK is good to about 5 mm), so the path may
        go as low as that pose itself minus 3 mm, but no lower."""
        target = np.asarray(target, float)
        start = self.arm.read()
        joints_move = float(np.abs(target[:5] - start[:5]).max()) > 0.5
        problems = m.out_of_range(target)
        if not joints_move:
            # gripper only: the arm joints are the measured pose, which may sag past the range margin under
            # gravity (observe: lift -97 commanded, -99.2 measured); only the gripper target is checked then
            problems = [pr for pr in problems if pr.startswith("gripper")]
        if problems:
            raise StopCycle(f"{what}: target out of range: {'; '.join(problems)}")
        clearance = self.cfg.near_table_clearance_mm if near_table else self.cfg.travel_clearance_mm
        if not joints_move:
            # gripper only: the arm stays put and a jaw cannot pass through the table (it would stall on it),
            # so only the range check applies; e.g. opening the jaws at the taught photo spot on the table
            clearance = -1e9
        if taught:
            own = min(check_straight_move(q, q, self.cal, samples=2, min_clearance_mm=-1e9).end_clearance_mm
                      for q in (start, target))  # into or out of the taught low pose
            clearance = min(clearance, own - 3.0)
        # never refuse a move because of where the arm already is: the path may start as low as the current
        # pose (e.g. just lifted off the photo spot with a sagging, loaded shoulder) but never go below it
        here = check_straight_move(start, start, self.cal, samples=2, min_clearance_mm=-1e9).end_clearance_mm
        clearance = min(clearance, here - 3.0)
        pc = check_straight_move(start, target, self.cal, min_clearance_mm=clearance)
        if not pc.ok:
            raise StopCycle(f"{what}: {pc.describe()}")
        v = speed or (self.cfg.near_table_speed if near_table else self.cfg.travel_speed)
        v = min(v, MAX_JOINT_SPEED) if joints_move else (speed or self.cfg.gripper_speed)
        t = m.duration(start, target, v) if np.abs(target - start).max() > 0 else 0.0
        self.log("move", what=what, start=start, target=target, speed=v, seconds=round(t, 2),
                 min_clearance_mm=round(pc.min_clearance_mm, 1), dry_run=self.dry_run)
        print(f"  move {what}: {t:.1f}s at {v:g} u/s, lowest point {pc.min_clearance_mm:.0f} mm")
        if self.step and not self.dry_run:
            ans = self.ask(f"    [{what}] Enter = move, q = stop: ").strip().lower()
            if ans in ("q", "s", "stop", "quit"):
                raise StopCycle("stopped at the step prompt")
        if self.dry_run:
            self.arm.send(target)
            return self.arm.read()
        # going up (jaw >= 10 mm higher at the end): a loaded shoulder may stop a few units short under gravity,
        # which only leaves the arm a little lower than planned: 10 units there, 8 elsewhere (user: +-1 cm is fine)
        up = float(self.jaw(target)[2] - self.jaw(start)[2]) >= 10.0
        r = m.move_linear(self.arm.send, self.arm.read, start, target, v, settle_error=10.0 if up else 8.0)
        self.log("moved", what=what, ok=r.ok, reason=r.reason, final=r.final, worst_error=round(r.worst_error, 2))
        if not r.ok:
            raise StopCycle(f"{what}: {r.reason}")
        return r.final

    def gripper(self, value: float, what: str) -> np.ndarray:
        q = self.arm.read()
        q[GRIPPER] = value
        return self.move(q, what, near_table=True, speed=self.cfg.gripper_speed)

    # -- poses derived with IK (gripper straight down) --

    def tool(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        tp = ik.tool_pose(normalized_to_urdf(np.asarray(q, float), self.cal))
        return tp["position"], float(ik.closing_yaw_deg(tp["closing"]))

    def down_at(self, xy, z_above_table_mm: float, yaw_deg: float, gripper: float, start: np.ndarray) -> np.ndarray:
        sol = ik.solve_down(float(xy[0]), float(xy[1]), TABLE_Z_MM + z_above_table_mm, yaw_deg, self.cal, gripper,
                            start_norm=start)
        if not sol.ok:
            raise StopCycle(f"IK: {sol.reason} at ({xy[0]:.0f}, {xy[1]:.0f}) mm, {z_above_table_mm:.0f} mm up")
        return sol.norm

    def aim(self, xy, z_above_table_mm: float, yaw_deg: float, gripper: float, what: str, *,
            near_table: bool = False, tries: int = 2, tol_mm: float = 3.0) -> np.ndarray:
        """Solve IK for the CLOSED jaw centre (where a held brick sits; the calibration measured that point)
        over ``xy``, move there with the jaws at ``gripper``, then correct for sag: FK of the MEASURED pose
        vs the target, re-aim at target + error, up to ``tries`` corrections."""
        goal = np.asarray(xy, float)
        aim_xy = goal.copy()
        q = None
        for k in range(tries + 1):
            sol = self.down_at(aim_xy, z_above_table_mm, yaw_deg, self.cfg.ik_gripper, self.arm.read())
            sol[GRIPPER] = gripper
            q = self.move(sol, what if k == 0 else f"{what} (sag correction {k})", near_table=near_table)
            if self.dry_run:
                return q
            meas = self.arm.read()
            meas[GRIPPER] = self.cfg.ik_gripper
            err = goal - self.jaw(meas)[:2]
            self.log("aim", what=what, k=k, err_mm=np.round(err, 1))
            print(f"    aim error {np.hypot(*err):.1f} mm")
            if float(np.hypot(*err)) <= tol_mm:
                break
            aim_xy = aim_xy + err
        return q

    def jaw(self, q: np.ndarray) -> np.ndarray:
        return ik.tool_pose(normalized_to_urdf(np.asarray(q, float), self.cal))["position"]

    def lifted(self, q: np.ndarray, mm: float, keep_roll: bool = True) -> np.ndarray:
        """``q`` with the jaw centre ``mm`` higher.

        1. Only with ``keep_roll=False`` and the gripper (nearly) straight down: gripper-down IK at the same XY
           and jaw direction (it may change wrist_roll; used near the brick on the sheet).
        2. Otherwise (e.g. the stretched-out ``photo`` pose): position-only damped least squares on
           shoulder_lift, elbow_flex and wrist_flex, so the jaw centre goes straight up and the wrist may tilt.
        3. If that does not converge: the first point on the straight line towards ``observe`` whose jaw
           centre is ``mm`` higher (``observe`` is high and pulled back)."""
        q = np.asarray(q, float)
        pos, yaw = self.tool(q)
        tilt = float(ik.tilt_deg(ik.tool_pose(normalized_to_urdf(q, self.cal))["axis"]))
        if not keep_roll and tilt < 15.0:
            try:
                return self.down_at(pos[:2], pos[2] - TABLE_Z_MM + mm, yaw, float(q[GRIPPER]), q)
            except StopCycle:
                pass
        goal = pos + np.array([0.0, 0.0, mm])
        x = q.copy()
        for _ in range(40):
            err = goal - self.jaw(x)
            if np.linalg.norm(err) < 1.0:
                break
            J = np.empty((3, 3))
            for c, j in enumerate((1, 2, 3)):
                d = x.copy()
                d[j] += 0.5
                J[:, c] = (self.jaw(d) - self.jaw(x)) / 0.5
            dx = J.T @ np.linalg.solve(J @ J.T + 4.0 * np.eye(3), err)
            x[1:4] = np.clip(x[1:4] + np.clip(dx, -8.0, 8.0), -97.0, 97.0)
        if np.linalg.norm(goal - self.jaw(x)) < 5.0 and not m.out_of_range(x):
            return x
        obs = self.observe if self.observe is not None else q
        for s in np.linspace(0.05, 1.0, 20):
            y = q + s * (obs - q)
            y[GRIPPER] = q[GRIPPER]
            if self.jaw(y)[2] >= pos[2] + mm:
                return y
        raise StopCycle(f"cannot find a pose {mm:.0f} mm above the current one")

    def approach(self, target: np.ndarray, what: str) -> np.ndarray:
        """Go above a taught low pose first (only shoulder_lift / elbow / wrist_flex differ, so pan and
        wrist_roll already match), then down to EXACTLY the taught joints, slowly. No IK re-solve."""
        self.move(self.lifted(target, self.cfg.approach_mm), f"above {what}", taught=True)
        return self.move(target, what, near_table=True, taught=True)


# ---------------------------------------------------------------------------------------------- cameras

class Cameras(Protocol):
    def wrist(self) -> np.ndarray: ...
    def all(self) -> dict[str, np.ndarray]: ...


class RealCameras:
    """One frame per call: the wrist camera by its USB name (never the laptop camera, pick_above), and the
    two D415s by serial through pyrealsense2. Each call opens and closes the device (slow but simple)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def wrist(self) -> np.ndarray:
        from hackjunction2026.pick_above import grab_wrist_frame

        return grab_wrist_frame((1024, 768))

    def realsense(self, serial: str) -> np.ndarray:
        import pyrealsense2 as rs

        p, c = rs.pipeline(), rs.config()
        c.enable_device(serial)
        c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        p.start(c)
        try:
            for _ in range(20):
                f = p.wait_for_frames()
            return np.asanyarray(f.get_color_frame().get_data()).copy()
        finally:
            p.stop()

    def all(self) -> dict[str, np.ndarray]:
        return {"top": self.realsense(self.cfg.top_serial), "floor": self.realsense(self.cfg.floor_serial),
                "arm": self.wrist()}


class FileCameras:
    """Frames from a Studio state snapshot folder ({top,floor,arm}.color.png): dry runs and tests."""

    def __init__(self, folder: Path) -> None:
        import cv2

        self.frames = {c: cv2.imread(str(folder / f"{c}.color.png")) for c in ("top", "floor", "arm")}
        if any(v is None for v in self.frames.values()):
            raise SystemExit(f"{folder}: needs top/floor/arm .color.png")

    def wrist(self) -> np.ndarray:
        return self.frames["arm"].copy()

    def all(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.frames.items()}


# ---------------------------------------------------------------------------------------------- locate

@dataclass
class Target:
    xy: tuple[float, float]
    yaw_deg: float               # direction the jaws close along
    colour: str
    source: str


def locate_wrist(frame: np.ndarray, cfg: Config, measured: np.ndarray) -> Target | None:
    """Brick in the observe-pose wrist frame -> table XY through the calibrated homography (ADR 0007)."""
    from hackjunction2026 import table_map as tm

    th = tm.TableHomography.load(REPO / cfg.homography)
    ref = m.vec(th.pose_joints) if th.pose_joints else None
    if ref is not None and np.abs(measured[:5] - ref[:5]).max() > 1.5:
        raise StopCycle(f"not in the calibrated observe pose (off by {np.abs(measured[:5] - ref[:5]).max():.1f})")
    bricks = tm.locate(tm.detect(frame), th)
    keep = [b for b in bricks if brick_allowed(b, cfg)]
    for b in bricks:
        if b not in keep:
            print(f"    ignored {b.colour} blob at ({b.table_xy[0]:.0f}, {b.table_xy[1]:.0f}) mm, {b.area_px:.0f} px "
                  f"({brick_allowed(b, cfg, why=True)})")
    if not keep:
        return None
    b = max(keep, key=lambda b: b.area_px)
    return Target(tuple(b.table_xy), float(b.long_axis_yaw_deg) + 90.0, b.colour, "wrist homography")


BOWL_KEEP_OUT_MM = 120.0      # nothing within this distance of a bowl centre is a brick (bowls: r ~9 cm)
BRICK_AREA_PX = (1500, 9000)  # a brick in the observe wrist frame is ~3000-6000 px; the bowls are bigger
REACH_MM = (110.0, 265.0)     # gripper-down reach from the pan axis (x = 38.8 mm)


def bowl_centres(cfg: Config) -> list[np.ndarray]:
    """Table XY of the bowls: FK gripper tip of the taught bowl poses."""
    poses, _ = load_poses(cfg)
    cal = Calibration.from_file(REPO / "configs/calibration/follower.json")
    out = []
    for name in ("bowl_good", "bowl_defect"):
        if name in poses:
            from hackjunction2026.kinematics import arm_keypoints
            out.append(arm_keypoints(poses[name][None], cal)["gripper_tip"][0][:2])
    return out


def brick_allowed(b, cfg: Config, why: bool = False):
    """Reject the bowls (by table position and size) and anything the arm cannot reach gripper-down."""
    xy = np.asarray(b.table_xy, float)
    reasons = []
    if not BRICK_AREA_PX[0] <= b.area_px <= BRICK_AREA_PX[1]:
        reasons.append("size")
    if any(np.hypot(*(xy - c)) < BOWL_KEEP_OUT_MM for c in bowl_centres(cfg)):
        reasons.append("bowl")
    r = float(np.hypot(xy[0] - 38.8, xy[1]))
    if not REACH_MM[0] <= r <= REACH_MM[1]:
        reasons.append(f"out of reach r={r:.0f}")
    return ", ".join(reasons) if why else not reasons


def locate_top(frame: np.ndarray, cfg: Config) -> Target | None:
    """Brick in the top D415 frame -> table XY by the fitted camera model (configs/top_camera.yaml): the
    ray through the brick's front-bottom edge meets the table, plus half a brick away from the camera.
    Measured error 3-4 cm at the edge of reach (log 2026-11-14 07:12)."""
    import cv2
    import yaml

    from hackjunction2026 import table_map as tm

    cam = yaml.safe_load((REPO / cfg.top_camera).read_text())
    K, dist = np.array(cam["K"]), np.array(cam["dist"])
    R, t = np.array(cam["R_cam_from_base"]), np.array(cam["t_cam_from_base_mm"])
    c0 = -R.T @ t
    bricks = sorted(tm.detect(frame).bricks, key=lambda b: -b.area_px)
    if not bricks:
        return None
    b = bricks[0]
    box = np.array(b.box)
    bottom = box[np.argsort(box[:, 1])[-2:]].mean(axis=0)
    u = cv2.undistortPoints(bottom.reshape(1, 1, 2).astype(float), K, dist, P=K).ravel()
    d = R.T @ np.linalg.solve(K, [u[0], u[1], 1.0])
    front = c0 + (TABLE_Z_MM - c0[2]) / d[2] * d
    away = front[:2] - c0[:2]
    away /= np.linalg.norm(away)
    xy = front[:2] + cfg.brick_half_depth_mm * away
    return Target((float(xy[0]), float(xy[1])), 90.0, b.colour, "top camera model")


# ---------------------------------------------------------------------------------------------- detect

class Detector(Protocol):
    def judge(self, frames: dict[str, np.ndarray], scene: str, out: Path | None) -> dict: ...


def colour_blob_brick(roi: np.ndarray) -> list:
    """Largest strongly coloured, non-purple blob of brick size in a small window, as a ``Brick`` for
    ``crop_brick``. No sheet segmentation: it fails under the venue light in the fixed RealSense views."""
    from hackjunction2026.defect.cropping.bricks import Brick
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    hue = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 0]
    purple = (hue >= 125) & (hue <= 165)
    mask = ((chroma >= 18) & (lab[..., 0] >= 25) & ~purple).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if 600 <= a <= 0.5 * roi.shape[0] * roi.shape[1] and max(w, h) <= 3 * max(min(w, h), 1):
            best = k if best is None or a > stats[best][4] else best
    if best is None:
        return []
    cnts, _ = cv2.findContours((lab_img == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea)).astype(np.int32)
    x, y, w, h, a = (int(v) for v in stats[best])
    return [Brick(bbox=(x, y, w, h), area=a, colour="unknown", ring_white=0.0, ring_ok=0.0, fill=1.0, hull=hull)]


class PatchCoreAndClassifier:
    """ADR 0008 with the VLM dropped: defective if PatchCore OR the mark classifier flags ANY crop."""

    def __init__(self, cfg: Config) -> None:
        import openvino as ov

        from hackjunction2026.defect.runtime import BrickAnomalyDetector

        import yaml

        self.cfg = cfg
        self.patchcore = BrickAnomalyDetector.from_deploy(REPO / cfg.patchcore_deploy, device=cfg.detector_device)
        # the deployed classifier (configs/anomalib_deploy.yaml "classifier:") wins over configs/sort_cycle.yaml
        dep = (yaml.safe_load((REPO / cfg.patchcore_deploy).read_text()) or {}).get("classifier") or {}
        self.classifier_xml = dep.get("model_xml", cfg.classifier_xml)
        self.threshold = float(dep.get("threshold", cfg.classifier_threshold))
        self.clf = ov.Core().compile_model(str(REPO / self.classifier_xml), dep.get("device", cfg.detector_device))

    def photo_spot_crops(self, frames: dict[str, np.ndarray], have: set[str]) -> list[dict]:
        """The brick always lies at the photo spot during inspection, so when the whole-frame finder misses it in
        a fixed camera (its sheet segmentation fails when the arm or a bowl is in view), look in a window around
        where that camera sees the photo spot (``photo_px`` in the config)."""
        from hackjunction2026.defect.cropping.bricks import crop_brick, find_bricks
        out = []
        for cam, (u, v) in (self.cfg.photo_px or {}).items():
            if cam in have or cam not in frames:
                continue
            f, r = frames[cam], self.cfg.photo_roi_px // 2
            x0, y0 = max(0, int(u) - r), max(0, int(v) - r)
            roi = np.ascontiguousarray(f[y0:int(v) + r, x0:int(u) + r])
            found = find_bricks(roi, self.patchcore.seg, camera=None) or colour_blob_brick(roi)
            if found:
                img, _, square = crop_brick(roi, found[0], self.patchcore.crop)
                out.append({"image": img, "camera": cam, "brick": 0, "bbox": list(found[0].bbox), "square": list(square),
                            "colour": found[0].colour, "seg_ms": 0.0, "roi": [x0, y0]})
        return out

    def classify(self, crop_bgr: np.ndarray) -> float:
        from hackjunction2026.defect.training.mark_classifier import prep

        return float(np.asarray(self.clf(prep(crop_bgr))[0]).ravel()[0])

    def judge(self, frames: dict[str, np.ndarray], scene: str, out: Path | None) -> dict:
        crops = self.patchcore.crops_from_frames(frames)
        crops += self.photo_spot_crops(frames, {c["camera"] for c in crops})
        pc = self.patchcore.judge_crops(crops, scene, out)
        votes = []
        for c, item in zip(crops, pc["crops"], strict=True):
            p = self.classify(c["image"])
            item["classifier"] = round(p, 4)
            item["classifier_flags"] = p > self.threshold
            votes.append(item["is_anomalous"] or item["classifier_flags"])
        return {"is_defective": any(votes), "n_crops": len(votes), "patchcore": pc["is_anomalous"],
                "classifier": any(i["classifier_flags"] for i in pc["crops"]), "crops": pc["crops"],
                "latency_s": pc["latency_s"]}


# ---------------------------------------------------------------------------------------------- grasp

@dataclass
class GraspResult:
    ok: bool
    reason: str
    state: GripperState | None = None


def close_and_check(mv: Mover, what: str) -> GraspResult:
    """Lower the gripper torque limit, close, read servo 6; on a hold, ease the squeeze to just past the
    stop position so the overload protection does not trip and drop the brick (ADR 0013)."""
    arm, cfg = mv.arm, mv.cfg
    if not mv.dry_run:
        arm.gripper_torque_limit(cfg.hold_torque_limit)
    try:
        mv.gripper(cfg.gripper_close_target, f"close ({what})")
    except StopCycle as e:
        if "target not reached" not in str(e):  # the gripper stopping on the brick is expected
            raise
    time.sleep(0 if mv.dry_run else 0.3)
    st = arm.gripper_state()
    holding = st.holding(cfg)
    mv.log("grasp_check", what=what, pos=st.pos, load=st.load, status=st.status, holding=holding)
    print(f"  gripper {st.pos:.1f}, load {st.load}, status {st.status or 'ok'} -> {'HOLDING' if holding else 'empty'}")
    if holding:
        q = arm.read()
        q[GRIPPER] = max(cfg.gripper_close_target, st.pos - cfg.squeeze)
        if not mv.dry_run:
            arm.send(q)
        return GraspResult(True, "holding", st)
    return GraspResult(False, "nothing between the jaws", st)


class GraspStrategy(Protocol):
    def grasp(self, mv: Mover, target: Target, hover: np.ndarray) -> GraspResult: ...


class ScriptedGrasp:
    """From the hover: open, straight down to the grasp height at the same XY and jaw direction, close,
    check, lift back to the hover (UNTESTED heights)."""

    def grasp(self, mv: Mover, target: Target, hover: np.ndarray) -> GraspResult:
        cfg = mv.cfg
        mv.gripper(cfg.gripper_open, "open")
        q = mv.arm.read()
        mv.aim(target.xy, cfg.grasp_mm, target.yaw_deg, cfg.gripper_open, "down to the brick", near_table=True, tries=1)
        r = close_and_check(mv, "first grasp")
        lift = mv.arm.read()
        lift = mv.lifted(lift, cfg.hover_mm - cfg.grasp_mm, keep_roll=False)
        if not r.ok:
            mv.gripper(cfg.gripper_open, "open again")
            lift[GRIPPER] = cfg.gripper_open
        mv.move(lift, "lift", near_table=True)
        return r


def mv_here(mv: Mover, q: np.ndarray) -> float:
    """Lowest arm point of pose ``q`` above the table (mm)."""
    return check_straight_move(q, q, mv.cal, samples=2, min_clearance_mm=-1e9).end_clearance_mm


class ACTGrasp:
    """The Studio-trained ACT policy from the hover, run outside Studio (physicalai.inference.InferenceModel
    on the OpenVINO export; observation = {"state": (1, 6) normalized, "images.<cam>": (1, H, W, 3) RGB
    uint8}, the same layout Studio's model worker sends). Action chunks of 100 steps, re-planned every
    ``act_replan_every`` steps at 30 Hz, each step clamped to the speed cap and checked against the table.
    UNTESTED: never run on the robot; the camera streams below are a minimal stand-in for Studio's."""

    def __init__(self, cfg: Config) -> None:
        from physicalai.inference import InferenceModel

        self.cfg = cfg
        self.model = InferenceModel(Path(cfg.act_export).expanduser(), backend="openvino", device=cfg.act_device)
        self.streams = _Streams(cfg)

    def grasp(self, mv: Mover, target: Target, hover: np.ndarray) -> GraspResult:
        if mv.dry_run:
            print("  (dry run: ACT grasp not executed; the fake arm takes the scripted path)")
            return ScriptedGrasp().grasp(mv, target, hover)
        cfg, arm, hz = self.cfg, mv.arm, 30.0
        max_step = min(cfg.act_max_speed, MAX_JOINT_SPEED) / hz
        self.model.reset()
        self.streams.start()
        chunk, k, held_since = None, 0, None
        t_end = time.monotonic() + cfg.act_max_s
        try:
            while time.monotonic() < t_end:
                t0 = time.monotonic()
                q = arm.read()
                if chunk is None or k >= cfg.act_replan_every:
                    obs = {"state": q[None].astype(np.float32), **self.streams.observation()}
                    chunk, k = np.asarray(self.model.predict_action_chunk(obs)).reshape(-1, 6), 0
                a = np.array(chunk[min(k, len(chunk) - 1)], float)
                k += 1
                step = np.clip(a - q, -max_step, max_step)
                step[GRIPPER] = a[GRIPPER] - q[GRIPPER]              # the gripper is not speed-capped
                cmd = q + step
                if m.out_of_range(cmd) or not check_straight_move(q, cmd, mv.cal, samples=2,
                                                                  min_clearance_mm=min(cfg.near_table_clearance_mm, -5.0, mv_here(mv, q) - 3.0)).ok:  # FK +-5 mm; the table stops a real collision
                    raise StopCycle("ACT: action refused (range or table clearance)")
                arm.send(cmd)
                st = arm.gripper_state() if a[GRIPPER] < cfg.hold_min else None
                if st is not None and st.holding(cfg):
                    held_since = held_since or time.monotonic()
                    if time.monotonic() - held_since > 0.3:
                        return close_and_check(mv, "after ACT")
                else:
                    held_since = None
                time.sleep(max(0.0, 1 / hz - (time.monotonic() - t0)))
            return GraspResult(False, "ACT: timeout without a hold")
        except KeyboardInterrupt:
            arm.send(arm.read())
            raise StopCycle("Ctrl+C during ACT") from None
        finally:
            self.streams.stop()


class _Streams:
    """Latest RGB frame of each camera in a background thread each (ACT needs ~30 fps of all three)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg, self.latest, self.threads, self.run = cfg, {}, [], False

    def start(self) -> None:
        import threading

        self.run = True
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._wrist, self._rs_top, self._rs_floor)]
        for th in self.threads:
            th.start()
        t0 = time.monotonic()
        while len(self.latest) < 3 and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        if len(self.latest) < 3:
            self.stop()
            raise StopCycle(f"cameras not streaming: have {sorted(self.latest)}")

    def stop(self) -> None:
        self.run = False
        for th in self.threads:
            th.join(timeout=3)

    def observation(self) -> dict[str, np.ndarray]:
        return {f"images.{k}": v[None] for k, v in self.latest.items()}

    def _wrist(self) -> None:
        import cv2

        from hackjunction2026.pick_above import resolve_wrist_camera, wrist_camera_name

        cap = cv2.VideoCapture(str(resolve_wrist_camera(wrist_camera_name()).resolve()), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        try:
            while self.run:
                ok, f = cap.read()
                if ok:
                    self.latest["arm"] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def _rs(self, serial: str, key: str) -> None:
        import pyrealsense2 as rs

        p, c = rs.pipeline(), rs.config()
        c.enable_device(serial)
        c.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        p.start(c)
        try:
            while self.run:
                f = p.wait_for_frames()
                self.latest[key] = np.asanyarray(f.get_color_frame().get_data()).copy()
        finally:
            p.stop()

    def _rs_top(self) -> None:
        self._rs(self.cfg.top_serial, "top")

    def _rs_floor(self) -> None:
        self._rs(self.cfg.floor_serial, "floor")


# ---------------------------------------------------------------------------------------------- cycle

class S(str, Enum):
    OBSERVE = "observe"
    LOCATE = "locate"
    HOVER = "hover"
    GRASP = "grasp"
    TO_PHOTO = "to_photo"
    RELEASE_PHOTO = "release_photo"
    INSPECT = "inspect"
    REGRASP = "regrasp"
    TO_BOWL = "to_bowl"
    RELEASE_BOWL = "release_bowl"
    DONE = "done"


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return [round(float(x), 3) for x in v.ravel()]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if dataclasses.is_dataclass(v):
        return _jsonable(dataclasses.asdict(v))
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class Cycle:
    def __init__(self, arm: Arm, cams: Cameras, detector: Detector | None, grasp: GraspStrategy, cfg: Config,
                 cal: Calibration, poses: dict[str, np.ndarray], *, dry_run: bool, step: bool,
                 out_root: Path = CAPTURES, ask: Callable[[str], str] = input) -> None:
        self.arm, self.cams, self.detector, self.grasper, self.cfg = arm, cams, detector, grasp, cfg
        self.poses, self.dry_run = poses, dry_run
        self.out = out_root / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out.mkdir(parents=True, exist_ok=True)
        self.logf = (self.out / "cycle.jsonl").open("a")
        self.mv = Mover(arm, cal, cfg, self.log, dry_run=dry_run, step=step, ask=ask)
        self.mv.observe = poses.get("observe")
        self.state = S.OBSERVE

    def log(self, event: str, **data) -> None:
        rec = {"t": datetime.datetime.now().isoformat(timespec="milliseconds"), "state": self.state.value,
               "event": event, **_jsonable(data)}
        self.logf.write(json.dumps(rec) + "\n")
        self.logf.flush()

    def save(self, name: str, img: np.ndarray) -> str:
        import cv2

        p = self.out / f"{len(list(self.out.glob('*.png'))):03d}_{name}.png"
        cv2.imwrite(str(p), img)
        return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)

    def enter(self, s: S) -> None:
        self.state = s
        print(f"[{s.value}]")
        self.log("enter")

    def run_one(self) -> dict:
        """One brick from observe to a bowl. Returns the result; raises StopCycle to stop."""
        cfg, mv = self.cfg, self.mv
        for attempt_round in range(2):                       # retries exhausted once -> start from the beginning
            self.enter(S.OBSERVE)
            q = mv.move(self.poses["observe"], "observe")
            self.enter(S.LOCATE)
            target = self.locate(q)
            if target is None:
                self.log("no_brick")
                return {"result": "no brick on the sheet"}
            for attempt in range(1, cfg.retries_per_brick + 1):
                self.enter(S.HOVER)
                hover_mm = cfg.hover_mm + cfg.retry_raise_mm * (attempt - 1)   # higher each retry: wider view
                while True:                                                     # ... as high as IK allows
                    try:
                        hover = mv.aim(target.xy, hover_mm, target.yaw_deg, cfg.gripper_open,
                                       f"hover over the {target.colour} brick (try {attempt}, {hover_mm:.0f} mm)")
                        break
                    except StopCycle as e:
                        if not str(e).startswith("IK") or hover_mm <= cfg.hover_mm:
                            raise
                        hover_mm = max(cfg.hover_mm, hover_mm - 10.0)
                self.enter(S.GRASP)
                r = self.grasper.grasp(mv, target, hover)
                self.log("grasp", attempt=attempt, ok=r.ok, reason=r.reason)
                if r.ok:
                    return self.after_grasp(target)
            print(f"  {cfg.retries_per_brick} failed grasps: starting over from observe")
            self.log("retries_exhausted", round=attempt_round)
        raise StopCycle("could not grasp the brick after starting over")

    def locate(self, q: np.ndarray) -> Target | None:
        cfg = self.cfg
        source = cfg.locate
        if source == "auto":
            source = "wrist" if (REPO / cfg.homography).exists() else "top"
        if source == "wrist":
            frame = self.cams.wrist()
            self.log("frame", camera="arm", path=self.save("observe_arm", frame))
            t = locate_wrist(frame, cfg, q)
        else:
            frame = self.cams.all()["top"]
            self.log("frame", camera="top", path=self.save("observe_top", frame))
            t = locate_top(frame, cfg)
        if t is not None:
            print(f"  {t.colour} brick at ({t.xy[0]:.0f}, {t.xy[1]:.0f}) mm via {t.source}")
            self.log("located", target=t)
        return t

    def after_grasp(self, target: Target) -> dict:
        cfg, mv = self.cfg, self.mv
        self.enter(S.TO_PHOTO)
        photo = self.poses["photo"].copy()
        photo[GRIPPER] = mv.arm.read()[GRIPPER]
        mv.approach(photo, "photo spot")
        self.enter(S.RELEASE_PHOTO)
        mv.gripper(cfg.gripper_open, "release for the photo")
        if not self.dry_run:
            mv.arm.gripper_torque_limit(1000)
        up = mv.lifted(mv.arm.read(), cfg.raise_mm)
        mv.move(up, "raise for the photo", near_table=True, taught=True)   # slowly: do not drag the brick
        clear = cfg.photo_clear_frac * self.poses["observe"] + (1 - cfg.photo_clear_frac) * mv.arm.read()
        clear[GRIPPER] = cfg.gripper_open
        mv.move(clear, "out of the cameras' way for the photo", speed=cfg.clear_speed)  # user: further away, faster
        self.enter(S.INSPECT)
        frames = self.cams.all()
        for k, v in frames.items():
            self.log("frame", camera=k, path=self.save(f"inspect_{k}", v))
        verdict = self.detector.judge(frames, "cycle", self.out / "heatmaps") if self.detector else \
            {"is_defective": False, "note": "no detector (--no-detect)"}
        self.log("verdict", **verdict)
        bad = bool(verdict["is_defective"])
        print(f"  verdict: {'DEFECTIVE' if bad else 'good'} ({verdict.get('n_crops', 0)} crops)")
        self.enter(S.REGRASP)
        for attempt in range(1, cfg.retries_per_brick + 1):
            down = self.poses["photo"].copy()
            down[GRIPPER] = cfg.gripper_open
            mv.approach(down, f"back to the photo spot (try {attempt})")
            r = close_and_check(mv, "re-grasp")
            if r.ok:
                break
            mv.gripper(cfg.gripper_open, "open again")
        else:
            raise StopCycle("re-grasp at the photo spot failed 3 times: put the brick back by hand")
        self.enter(S.TO_BOWL)
        mv.move(mv.lifted(mv.arm.read(), cfg.approach_mm), "lift from the photo spot", near_table=True, taught=True)
        bowl = self.poses["bowl_defect" if bad else "bowl_good"].copy()
        bowl[GRIPPER] = mv.arm.read()[GRIPPER]
        mv.approach(bowl, f"the {'defect' if bad else 'good'} bowl")
        self.enter(S.RELEASE_BOWL)
        mv.gripper(cfg.gripper_open, "release into the bowl")
        if not self.dry_run:
            mv.arm.gripper_torque_limit(1000)
        try:
            mv.move(mv.lifted(mv.arm.read(), cfg.raise_mm), "raise after the bowl", taught=True)
        except StopCycle as e:  # a bowl pose that is not gripper-down: go straight back to observe instead
            self.log("raise_skipped", reason=str(e))
        self.enter(S.DONE)
        return {"result": "sorted", "defective": bad, "colour": target.colour}

    def close(self) -> None:
        self.logf.close()

"""Autonomous calibration of the wrist-camera homography in the ``observe`` pose (ADR 0007, ADR 0013).

The robot moves ONE brick itself: it takes the brick from the taught photo spot, puts it down at about 6
reachable table points, looks at each from ``observe`` (wrist camera, ``table_map.detect``), and pairs the brick
pixel with the FK jaw-centre XY of the MEASURED pose at the moment it put the brick down. The pairs are fitted
with ``table_map`` (RANSAC + leave-one-out mm) into ``configs/table_homography.yaml``. Along the way it logs
servo 6 (position / load / status) closed empty and closed on the brick, for the sort-cycle holding thresholds.
At the end the brick goes back to the photo spot and the arm to ``observe``.

Every move goes through ``sort_cycle.Mover`` (range + path clearance checks, speed caps, --step prompt,
move_linear with stop-and-hold). UNTESTED ON THE ROBOT.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hackjunction2026 import sort_cycle as sc
from hackjunction2026 import table_map as tm
from hackjunction2026.kinematics import TABLE_Z_MM, Calibration
from hackjunction2026.kinematics import ik
from hackjunction2026.kinematics.mapping import normalized_to_urdf
from hackjunction2026.kinematics.path import check_straight_move

PAN_AXIS_XY = np.array([38.8, 0.0])   # shoulder_pan axis in the base frame (agent B)
YAW = 90.0                            # jaws close along +y everywhere: one fixed, simple grasp direction


@dataclass
class Point:
    xy: tuple[float, float]
    hover: np.ndarray
    down: np.ndarray


def observe_direction_deg(observe: np.ndarray, cal: Calibration) -> float:
    """Direction from the pan axis to where the observe pose's jaws point (degrees from +x)."""
    p = ik.tool_pose(normalized_to_urdf(observe, cal))["position"]
    d = p[:2] - PAN_AXIS_XY
    return float(np.degrees(np.arctan2(d[1], d[0])))


def plan_points(observe: np.ndarray, cal: Calibration, cfg: sc.Config, n: int = 6,
                radii=(190.0, 222.0, 252.0), spread_deg=(-32.0, 0.0, 32.0),
                given: list[tuple[float, float]] | None = None) -> tuple[list[Point], list[str]]:
    """Candidate table points (reachable gripper-down at hover and place height, clear path from observe),
    spread by farthest-point sampling. Default grid: 3 radii x 3 directions around where observe looks, at the
    far side of the reach ring (reach ends at r ~255-265 mm): the observe view looks further out (a brick at the
    image centre was at x ~330-340 mm by the top camera) and the jaws hide the bottom of the image, so near
    points are probably not seen. Points that are not seen are skipped at run time."""
    if given:
        cands = [tuple(map(float, p)) for p in given]
    else:
        base = observe_direction_deg(observe, cal)
        cands = [tuple(PAN_AXIS_XY + r * np.array([np.cos(np.radians(base + a)), np.sin(np.radians(base + a))]))
                 for r in radii for a in spread_deg]
    ok, why = [], []
    for xy in cands:
        hov = ik.solve_down(xy[0], xy[1], TABLE_Z_MM + cfg.hover_mm, YAW, cal, cfg.gripper_open, start_norm=observe)
        dwn = ik.solve_down(xy[0], xy[1], TABLE_Z_MM + cfg.grasp_mm, YAW, cal, cfg.gripper_open,
                            start_norm=hov.norm if hov.ok else observe)
        if not (hov.ok and dwn.ok):
            why.append(f"({xy[0]:.0f}, {xy[1]:.0f}) unreachable: {(hov if not hov.ok else dwn).reason}")
            continue
        if not check_straight_move(observe, hov.norm, cal, min_clearance_mm=cfg.travel_clearance_mm).ok:
            why.append(f"({xy[0]:.0f}, {xy[1]:.0f}) refused: path from observe")
            continue
        ok.append(Point((float(xy[0]), float(xy[1])), hov.norm, dwn.norm))
    if given or len(ok) <= n:
        return ok, why
    pts = np.array([p.xy for p in ok])
    chosen = [int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))]
    while len(chosen) < n:
        d = np.min(np.linalg.norm(pts[:, None] - pts[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return [ok[i] for i in sorted(chosen)], why


class Calib:
    def __init__(self, cycle: sc.Cycle, colour: str | None = None) -> None:
        self.c, self.mv, self.cfg = cycle, cycle.mv, cycle.cfg
        self.colour = colour
        self.pairs: list[dict] = []
        self.gripper_log: list[dict] = []
        self.observe_measured: list[np.ndarray] = []
        self.last_place_xy: tuple[float, float] | None = None

    def grip_record(self, kind: str, st: sc.GripperState | None) -> None:
        if st is None:
            st = self.mv.arm.gripper_state()
        rec = {"kind": kind, "pos": st.pos, "load": st.load, "status": st.status}
        self.gripper_log.append(rec)
        self.c.log("gripper_sample", **rec)

    def closed_empty_sample(self) -> None:
        """Close on nothing in the air (at observe), record servo 6, open again."""
        mv = self.mv
        fake = isinstance(mv.arm, sc.FakeArm)
        saved = mv.arm.brick_at if fake else None
        if fake:
            mv.arm.brick_at = None                     # the fake has nothing between its jaws here
        try:
            if not mv.dry_run:
                mv.arm.gripper_torque_limit(self.cfg.hold_torque_limit)
            mv.gripper(self.cfg.gripper_close_target, "close empty (reference)")
            time.sleep(0 if mv.dry_run else 0.4)
            self.grip_record("closed_empty", None)
            mv.gripper(self.cfg.gripper_open, "open")
        finally:
            if fake:
                mv.arm.brick_at = saved

    def grasp_here(self, what: str) -> None:
        """Close at the current (low) pose and require a hold; up to 3 tries with a re-open between."""
        for attempt in range(1, self.cfg.retries_per_brick + 1):
            r = sc.close_and_check(self.mv, f"{what} (try {attempt})")
            self.grip_record("closed_on_brick" if r.ok else "closed_missed", r.state)
            if r.ok:
                return
            self.mv.gripper(self.cfg.gripper_open, "open again")
        raise sc.StopCycle(f"{what}: no hold after {self.cfg.retries_per_brick} tries: put the brick back by hand")

    def point_yaw(self, p: Point) -> float:
        """Jaw closing yaw of a planned point (from FK of its hover pose)."""
        from hackjunction2026.kinematics import arm_keypoints
        kp = arm_keypoints(np.asarray(p.hover, float)[None], self.mv.cal)
        d = kp["moving_jaw_tip"][0][:2] - kp["fixed_jaw_tip"][0][:2]
        return float(np.degrees(np.arctan2(d[1], d[0])))

    def look_down(self, i: int, when: str):
        """Wrist frame from the hover (gripper straight down): where is the brick relative to the closed jaws?
        Returns the table XY offset (mm) of the brick from the jaw centre, or None (no servo model / not seen)."""
        if self.mv.dry_run:
            return None
        from hackjunction2026 import servo
        frame = self.c.cams.wrist()
        self.c.save(f"point{i}_look_{when.replace(' ', '_')}", frame)
        model = servo.ServoModel.load()
        if model is None:
            return None
        px = servo.brick_px(frame, near=model.target_px)
        if px is None:
            print(f"  point {i} ({when}): brick NOT seen under the jaws")
            self.c.log("look_down", i=i, when=when, seen=False)
            return None
        yaw = servo.directed_yaw_deg(self.mv.arm.read(), self.mv.cal)
        d = model.correction_mm(px, yaw)
        print(f"  point {i} ({when}): brick {np.hypot(*d):.1f} mm from the jaw centre")
        self.c.log("look_down", i=i, when=when, seen=True, px=px, offset_mm=np.round(d, 1))
        return d

    def check_still_holding(self, what: str) -> None:
        """After a lift: the jaws must still stop on the brick, else it slipped out (user saw this at point 0)."""
        if self.mv.dry_run:
            return
        st = self.mv.arm.gripper_state()
        self.c.log("hold_after_lift", what=what, pos=st.pos, load=st.load, status=st.status)
        if st.pos < self.cfg.hold_min:
            raise sc.StopCycle(f"{what}: the brick slipped out after the lift (gripper {st.pos:.1f}): put it back by hand")

    def take_from_photo(self) -> None:
        mv, photo = self.mv, self.c.poses["photo"].copy()
        photo[sc.GRIPPER] = self.cfg.gripper_open
        mv.gripper(self.cfg.gripper_open, "open")
        mv.approach(photo, "photo spot")
        self.grasp_here("grasp at the photo spot")
        mv.move(mv.lifted(mv.arm.read(), self.cfg.approach_mm), "lift from the photo spot", near_table=True, taught=True)
        self.check_still_holding("lift from the photo spot")

    def place_and_look(self, i: int, p: Point) -> None:
        mv, cfg = self.mv, self.cfg
        held = mv.arm.read()[sc.GRIPPER]
        hov = p.hover.copy()
        hov[sc.GRIPPER] = held
        mv.move(hov, f"point {i}: hover ({p.xy[0]:.0f}, {p.xy[1]:.0f})")
        dwn = p.down.copy()
        dwn[sc.GRIPPER] = held
        mv.move(dwn, f"point {i}: down", near_table=True)
        q_place = mv.arm.read()                                   # MEASURED pose with the brick in the jaws
        xy = ik.tool_pose(normalized_to_urdf(q_place, mv.cal))["position"][:2]
        self.last_place_xy = (float(xy[0]), float(xy[1]))
        mv.gripper(cfg.gripper_open, f"point {i}: release")
        up = mv.lifted(mv.arm.read(), cfg.hover_mm - cfg.grasp_mm, keep_roll=False)
        mv.move(up, f"point {i}: raise", near_table=True)
        shift = self.look_down(i, "after release")                # did the brick stay under the jaws?
        if shift is not None:
            xy = np.asarray(xy, float) + shift
            self.last_place_xy = (float(xy[0]), float(xy[1]))
        q_obs = mv.move(self.c.poses["observe"], "observe")
        self.observe_measured.append(np.array(mv.arm.read() if not mv.dry_run else q_obs))
        frame = self.c.cams.wrist()
        path = self.c.save(f"point{i}_observe", frame)
        det = tm.detect(frame)
        bricks = [b for b in det.bricks if self.colour is None or b.colour == self.colour]
        rec = {"i": i, "target_xy": p.xy, "place_xy": [float(xy[0]), float(xy[1])], "place_joints": q_place,
               "frame": path, "n_bricks": len(bricks)}
        if bricks:
            b = max(bricks, key=lambda b: b.area_px)
            rec.update(pixel=[float(b.centroid[0]), float(b.centroid[1])], colour=b.colour, area_px=b.area_px)
            self.c.save(f"point{i}_detect", tm.annotate(frame, det))
            prev = self.pairs[-1] if self.pairs else None
            if not self.mv.dry_run and prev is not None and np.hypot(*(np.subtract(rec["pixel"], prev["pixel"]))) < 25 \
                    and np.hypot(*(np.subtract(rec["place_xy"], prev["place_xy"]))) > 30:
                print(f"  point {i}: the brick did not move in the image since point {prev['i']}: pair dropped")
                rec["dropped"] = "brick did not move"
                self.c.log("pair", **rec)
                raise sc.StopCycle(f"point {i}: the brick was not carried here (still at point {prev['i']}?)")
            self.pairs.append(rec)
            print(f"  point {i}: placed at ({xy[0]:.0f}, {xy[1]:.0f}) mm -> pixel ({b.centroid[0]:.0f}, {b.centroid[1]:.0f})")
        else:
            print(f"  point {i}: placed at ({xy[0]:.0f}, {xy[1]:.0f}) mm -> NOT SEEN from observe (skipped)")
        self.c.log("pair", **rec)
        # pick it up again
        hov_open = p.hover.copy()
        hov_open[sc.GRIPPER] = cfg.gripper_open
        mv.move(hov_open, f"point {i}: back above")
        shift = self.look_down(i, "before grasp")
        if shift is not None and float(np.hypot(*shift)) > 2.0:  # brick not where it was left: re-aim on it
            gxy = np.asarray(p.xy, float) + shift
            mv.aim(gxy, cfg.hover_mm, self.point_yaw(p), cfg.gripper_open, f"point {i}: centre on the brick", tries=1)
            mv.aim(gxy, cfg.grasp_mm, self.point_yaw(p), cfg.gripper_open, f"point {i}: down to grasp", near_table=True, tries=1)
        else:
            dwn_open = p.down.copy()
            dwn_open[sc.GRIPPER] = cfg.gripper_open
            mv.move(dwn_open, f"point {i}: down to grasp", near_table=True)
        self.grasp_here(f"point {i}: grasp")
        mv.move(mv.lifted(mv.arm.read(), cfg.hover_mm - cfg.grasp_mm, keep_roll=False), f"point {i}: lift", near_table=True)
        self.check_still_holding(f"point {i}: lift")

    def return_to_photo(self) -> None:
        mv, photo = self.mv, self.c.poses["photo"].copy()
        photo[sc.GRIPPER] = mv.arm.read()[sc.GRIPPER]
        mv.approach(photo, "photo spot (return the brick)")
        mv.gripper(self.cfg.gripper_open, "release at the photo spot")
        mv.move(mv.lifted(mv.arm.read(), self.cfg.raise_mm), "raise", near_table=True, taught=True)
        mv.move(self.c.poses["observe"], "observe")

    def fit(self, out: Path, image_size=(1024, 768)) -> dict:
        if len(self.pairs) < 4:
            raise sc.StopCycle(f"only {len(self.pairs)} points seen from observe: need >= 4 (5+ for leave-one-out)")
        px = np.array([p["pixel"] for p in self.pairs])
        xy = np.array([p["place_xy"] for p in self.pairs])
        H, inl = tm.fit_homography_ransac(px, xy)
        res = np.linalg.norm(tm.apply_homography(H, px) - xy, axis=1)
        loo = tm.leave_one_out(px[inl], xy[inl])
        obs = np.mean(self.observe_measured, axis=0) if self.observe_measured else self.c.poses["observe"]
        points = [{"pixel": [round(v, 1) for v in p["pixel"]], "table_xy": [round(v, 1) for v in p["place_xy"]],
                   "residual_mm": round(float(r), 2), "inlier": bool(k), "colour": p.get("colour")}
                  for p, r, k in zip(self.pairs, res, inl, strict=True)]
        stats = {"n": len(px), "inliers": int(inl.sum()), "rms_mm": round(float(np.sqrt(np.mean(res[inl] ** 2))), 2),
                 "max_mm": round(float(res[inl].max()), 2),
                 "loo_rms_mm": None if np.isnan(loo).all() else round(float(np.sqrt(np.nanmean(loo ** 2))), 2),
                 "loo_max_mm": None if np.isnan(loo).all() else round(float(np.nanmax(loo)), 2),
                 "observe_spread_units": round(float(np.ptp(np.array(self.observe_measured), axis=0)[:5].max()), 2)
                 if len(self.observe_measured) > 1 else 0.0}
        th = tm.TableHomography(H=H, image_size=image_size, pose="observe",
                                pose_joints=sc.m.as_dict(obs), point="jaw_centre", date=tm.now_iso(),
                                points=points, stats=stats,
                                note="scripts/calib_wrist_homography.py: the robot placed one brick at each point; "
                                     "table XY = FK jaw centre of the measured place pose.")
        th.save(out)
        self.c.log("fit", path=str(out), **stats)
        return stats

    def gripper_summary(self) -> dict:
        out = {}
        for kind in sorted({g["kind"] for g in self.gripper_log}):
            g = [x for x in self.gripper_log if x["kind"] == kind]
            out[kind] = {"n": len(g), "pos": [round(x["pos"], 1) for x in g],
                         "load": [x["load"] for x in g], "status": sorted({s for x in g for s in x["status"]})}
        self.c.log("gripper_summary", **out)
        return out


def write_summary(calib: Calib, stats: dict | None, path: Path) -> None:
    path.write_text(json.dumps({"fit": stats, "gripper": calib.gripper_summary(), "pairs": sc._jsonable(calib.pairs)},
                               indent=1))

"""Command line for the SO-101 kinematics (entry point: ``scripts/arm_fk.py``).

Subcommands:
  pose      6 normalized joint values -> URDF angles and keypoints (mm, robot base frame)
  annotate  per-frame keypoints for every episode of a Studio (LeRobot v3) dataset -> parquet
            (needs pyarrow: ``uv run --with pyarrow scripts/arm_fk.py annotate ...``)
  check-plugin  verify the normalized<->ticks inverse against Studio's SO-101 plugin
            (run with Studio's backend Python, which has the plugin installed)
  check-urdf    compare the built-in geometry with Studio's follower URDF file
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from hackjunction2026.kinematics import (
    JOINT_MAP,
    JOINTS,
    KEYPOINTS,
    TABLE_Z_MM,
    Calibration,
    arm_keypoints,
    normalized_to_ticks,
    normalized_to_urdf,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CAL = REPO / "configs/calibration/follower.json"
DATASETS = Path.home() / ".local/share/physicalai/datasets"
DEFAULT_OUT = Path("/home/intel/dev/captures/perception/B")


def _calibration_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if DEFAULT_CAL.exists():
        return DEFAULT_CAL
    return Path("/home/intel/dev/configs/calibration/follower.json")


def cmd_pose(args: argparse.Namespace) -> None:
    cal = Calibration.from_file(_calibration_path(args.calibration))
    norm = np.array(args.values, dtype=np.float64)
    q = normalized_to_urdf(norm, cal)
    ticks = normalized_to_ticks(norm, cal)
    kp = arm_keypoints(norm, cal)
    print("joint          normalized    ticks   URDF deg")
    for j, n, t, a in zip(JOINTS, norm, ticks, np.degrees(q)):
        print(f"{j:14s} {n:10.2f} {t:8.1f} {a:10.2f}")
    print(f"\nkeypoint            x mm      y mm      z mm   (robot base frame; table at z = {TABLE_Z_MM:.1f})")
    for k in KEYPOINTS:
        x, y, z = kp[k]
        print(f"{k:16s} {x:9.1f} {y:9.1f} {z:9.1f}")
    print(f"\njaw gap {kp['jaw_gap_mm']:.1f} mm, jaw tip distance {kp['jaw_tip_dist_mm']:.1f} mm")


def _resolve_dataset(arg: str) -> Path:
    p = Path(arg).expanduser()
    if (p / "meta/info.json").exists():
        return p
    matches = sorted(DATASETS.glob(f"{arg}*/meta/info.json"))
    if len(matches) != 1:
        sys.exit(f"dataset '{arg}': {len(matches)} matches in {DATASETS}")
    return matches[0].parent.parent


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_annotate(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = _resolve_dataset(args.dataset)
    cal_path = _calibration_path(args.calibration)
    cal = Calibration.from_file(cal_path)
    info = json.loads((ds / "meta/info.json").read_text())
    names = [n.removesuffix(".pos") for n in info["features"]["observation.state"]["names"]]
    if names != list(JOINTS):
        sys.exit(f"unexpected joint order in {ds}: {names}")
    files = sorted(glob.glob(str(ds / "data/chunk-*/file-*.parquet")))
    cols = ["episode_index", "frame_index", "index", "timestamp", "observation.state", "action"]
    tab = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
    order = np.argsort(tab["index"].to_numpy(), kind="stable")
    tab = tab.take(pa.array(order))

    def mat(col: str) -> np.ndarray:
        return np.asarray(tab[col].combine_chunks().flatten().to_numpy(zero_copy_only=False),
                          dtype=np.float64).reshape(-1, 6)

    t0 = time.perf_counter()
    out: dict[str, np.ndarray] = {
        "episode_index": tab["episode_index"].to_numpy(),
        "frame_index": tab["frame_index"].to_numpy(),
        "index": tab["index"].to_numpy(),
        "timestamp": tab["timestamp"].to_numpy(),
    }
    sources = [("", mat("observation.state"))]
    if not args.no_action:
        sources.append(("act_", mat("action")))
    for prefix, norm in sources:
        for i, j in enumerate(JOINTS):
            out[f"{prefix}{j}"] = norm[:, i].astype(np.float32)
        q = np.degrees(normalized_to_urdf(norm, cal))
        for i, j in enumerate(JOINTS):
            out[f"{prefix}q_{j}_deg"] = q[:, i].astype(np.float32)
        kp = arm_keypoints(norm, cal)
        for k in KEYPOINTS:
            for a, axis in enumerate("xyz"):
                out[f"{prefix}{k}_{axis}_mm"] = kp[k][:, a].astype(np.float32)
        out[f"{prefix}jaw_gap_mm"] = kp["jaw_gap_mm"].astype(np.float32)
        out[f"{prefix}jaw_tip_dist_mm"] = kp["jaw_tip_dist_mm"].astype(np.float32)
    dt = time.perf_counter() - t0
    meta = {
        "dataset": ds.name,
        "dataset_path": str(ds),
        "calibration": str(cal_path),
        "calibration_ranges": {j: [float(a), float(b)] for j, a, b in zip(JOINTS, cal.range_min, cal.range_max)},
        "joint_map": {j: {"sign": m.sign, "zero_tick": m.zero_tick} for j, m in JOINT_MAP.items()},
        "frame": f"SO-101 URDF base_link, mm; +x forward, +y left, +z up; table plane z = {TABLE_Z_MM:.1f}",
        "source_columns": "observation.state (no prefix) and action (act_ prefix), Studio normalized units",
        "code": f"hackjunction2026.kinematics @ {_git_commit()}",
        "doc": "docs/perception/B-arm-kinematics.md",
    }
    table = pa.table(out).replace_schema_metadata({"hackjunction2026.arm_fk": json.dumps(meta)})
    dest = Path(args.out) if args.out else DEFAULT_OUT / f"arm_{ds.name}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest, compression="zstd")
    n_ep = len(np.unique(out["episode_index"]))
    print(f"{dest}: {table.num_rows} frames, {n_ep} episodes, {table.num_columns} columns "
          f"(FK {dt * 1000:.0f} ms)")


def cmd_check_plugin(args: argparse.Namespace) -> None:
    try:
        from physicalai_so101_plugin._conversion import ticks_to_normalized as plugin_t2n
        from physicalai_so101_plugin.calibration import SO101Calibration
    except ImportError:
        sys.exit("physicalai_so101_plugin not importable: run this with Studio's backend Python")
    cal_path = _calibration_path(args.calibration)
    cal = Calibration.from_file(cal_path)
    data = json.loads(cal_path.read_text())
    pcal = SO101Calibration.from_dict(data.get("calibration", data))
    worst = 0.0
    n = 0
    for i, j in enumerate(JOINTS):
        lo, hi = int(cal.range_min[i]), int(cal.range_max[i])
        for t in range(lo, hi + 1):
            ticks = np.array([int(c) for c in (cal.range_min + cal.range_max) // 2])
            ticks[i] = t
            norm = plugin_t2n(ticks, pcal)  # float32, as recorded
            back = normalized_to_ticks(norm.astype(np.float64), cal)
            worst = max(worst, abs(back[i] - t))
            n += 1
    print(f"checked {n} ticks: max |normalized_to_ticks(plugin(t)) - t| = {worst:.4f} ticks")
    if worst >= 0.5:
        sys.exit("inverse does not round to the original ticks")


STUDIO_URDF = (REPO / "third_party/physical-ai-studio/application/backend/.venv/lib/python3.13/site-packages/"
               "urdf/so101-follower/so101_follower.urdf")


def cmd_check_urdf(args: argparse.Namespace) -> None:
    """Compare the built-in geometry with a URDF file (joint origins and FK at random poses)."""
    import xml.etree.ElementTree as ET

    from hackjunction2026.kinematics import model

    path = Path(args.urdf) if args.urdf else STUDIO_URDF
    if not path.exists():
        path = Path("/home/intel/dev") / STUDIO_URDF.relative_to(REPO)
    root = ET.parse(path).getroot()
    worst = 0.0
    for j in root.findall("joint"):
        name = j.get("name")
        if name not in model._URDF_JOINTS:
            continue
        parent, child, xyz, rpy = model._URDF_JOINTS[name]
        o = j.find("origin")
        fx = [float(v) for v in o.get("xyz").split()]
        fr = [float(v) for v in o.get("rpy").split()]
        ax = [float(v) for v in j.find("axis").get("xyz").split()]
        same = (j.find("parent").get("link") == parent and j.find("child").get("link") == child
                and np.allclose(fx, xyz, atol=1e-12) and np.allclose(fr, rpy, atol=1e-12)
                and (j.get("type") == "fixed" or ax == [0.0, 0.0, 1.0]))
        worst = max(worst, 0.0 if same else 1.0)
        print(f"{name:20s} {'OK' if same else 'DIFFERENT'}")
    print(f"{path}: {'all joints match' if worst == 0 else 'MISMATCH'}")
    if worst:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", help="calibration JSON (default: configs/calibration/follower.json)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("pose", help="keypoints for one joint vector")
    sp.add_argument("values", nargs=6, type=float, metavar="NORM",
                    help="shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper (normalized)")
    sp.set_defaults(func=cmd_pose)
    sa = sub.add_parser("annotate", help="keypoints for every frame of a dataset -> parquet")
    sa.add_argument("dataset", help="dataset id (prefix ok) under ~/.local/share/physicalai/datasets, or a path")
    sa.add_argument("--out", help=f"output parquet (default: {DEFAULT_OUT}/arm_<dataset>.parquet)")
    sa.add_argument("--no-action", action="store_true", help="skip the act_* columns computed from action")
    sa.set_defaults(func=cmd_annotate)
    sc = sub.add_parser("check-plugin", help="verify the tick conversion against Studio's plugin")
    sc.set_defaults(func=cmd_check_plugin)
    su = sub.add_parser("check-urdf", help="compare the built-in geometry with Studio's follower URDF")
    su.add_argument("--urdf", help="URDF path (default: the one bundled with Studio's SO-101 plugin)")
    su.set_defaults(func=cmd_check_urdf)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

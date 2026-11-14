"""Camera intrinsics for the brick-sorting station.

RealSense factory colour intrinsics at 848x480, as Studio stored them in the state snapshots of
the recording session (~/.local/share/physicalai/state_snapshots/0fd76ff2-.../*/state.json,
`cameras[].realsense.color_intrinsics`; distortion coefficients are all zero).
Camera setups (which episodes share a camera pose) are detected per dataset by
scripts/perception_table_frame.py and stored in the table-frame JSON.
"""

from __future__ import annotations

import numpy as np

INTRINSICS = {
    # D415 serial 243622062352
    "top": {"fx": 607.4505615234375, "fy": 606.6509399414062, "cx": 420.9905090332031, "cy": 247.98826599121094},
    # D415 serial 243622062278
    "floor": {"fx": 609.8396606445312, "fy": 608.9093627929688, "cx": 420.78106689453125, "cy": 246.39544677734375},
}


def K_of(cam: str) -> np.ndarray:
    i = INTRINSICS[cam]
    return np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])


def group_consecutive(values: dict[int, np.ndarray], tol: float) -> list[tuple[int, int]]:
    """Split sorted episodes into runs whose signature stays within `tol` of the run's first episode
    (a camera that was bumped or a sheet that was moved starts a new run)."""
    eps = sorted(values)
    runs, start = [], eps[0]
    for prev, ep in zip(eps, eps[1:]):
        if np.max(np.abs(values[ep] - values[start])) > tol:
            runs.append((start, prev))
            start = ep
    runs.append((start, eps[-1]))
    return runs

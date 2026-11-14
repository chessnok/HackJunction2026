#!/usr/bin/env python3
"""Hover the SO-101 follower above a brick seen by the wrist camera, gripper straight down.

From the observe pose (configs/poses.yaml) the wrist camera sees the white sheet; a homography maps
image pixels to table XY (robot base frame, mm); numeric IK gives the joint target; a straight
joint-space path check guards the table. This tool never moves the arm: `plan` prints the
`so101_goto.py goto "..."` command for a human. Logic: src/hackjunction2026/pick_above.py.
Procedure: README, "Hover above a brick". Why: docs/decisions/0007-observe-pose-homography-ik.md.

From the repo root (OpenCV + PyYAML come from the `table` extra):
  uv run --extra table scripts/pick_above.py calib-points                       # FK of 'calib <colour>' snapshots
  uv run --extra table scripts/pick_above.py calib-fit --capture                # fit configs/table_homography.yaml
  uv run --extra table scripts/pick_above.py locate --image <png|snapshot dir>  # bricks with table XY
  uv run --extra table scripts/pick_above.py plan --capture [--height 60] [--brick red]
  uv run --extra table scripts/pick_above.py reach [--height 60]                # reach map PNG in captures/

--capture needs Studio's environment closed (camera and follower port free); it is UNTESTED.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hackjunction2026.pick_above import main

if __name__ == "__main__":
    main()

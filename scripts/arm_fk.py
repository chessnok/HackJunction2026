#!/usr/bin/env python3
"""SO-101 follower forward kinematics CLI (see hackjunction2026/kinematics/cli.py).

From the repo root:
  uv run scripts/arm_fk.py pose 0 0 0 0 0 0                    # one joint vector (normalized)
  uv run --with pyarrow scripts/arm_fk.py annotate 431a71f7     # dataset -> captures/perception/B/
  third_party/physical-ai-studio/application/backend/.venv/bin/python scripts/arm_fk.py check-plugin
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hackjunction2026.kinematics.cli import main

if __name__ == "__main__":
    main()

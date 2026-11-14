#!/usr/bin/env python3
"""Build an Anomalib dataset of zoomed brick crops (see hackjunction2026/defect/cropping/brick_dataset.py).

Needs av + pyarrow (Studio's backend env has them; use it read-only, never install into it). From the repo root:
  STUDIO_PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
  nice -n 19 $STUDIO_PY scripts/brick_crops.py --name bricks_v1 --dataset 431a71f7 --snapshots 0fd76ff2 --synthetic
Then label the defective crops per crop and train:
  uv run scripts/label_crops_gemini.py --dataset bricks_v1 --apply
  nice -n 19 scripts/anomalib.sh training.train_patchcore --dataset bricks_v1 --accelerator cpu
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hackjunction2026.defect.cropping.brick_dataset import main  # noqa: E402

if __name__ == "__main__":
    main()

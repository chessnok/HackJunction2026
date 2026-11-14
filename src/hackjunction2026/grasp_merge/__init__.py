"""Merge the grasp-only dataset (NEW) with the grasp tails of the older approach dataset (OLD).

Modules:

- ``cuts``: data-driven cut point per OLD episode (numpy / pandas only).
- ``clips``: H.264 review clips of the top camera with the proposed cut burned in (PyAV, OpenCV).
- ``build``: the merged LeRobot v3 dataset (LeRobot from Studio's backend venv).
- ``validate``: load / frame-count / decode checks of the merged dataset.

Entry point: ``scripts/grasp_merge.py``. Criterion, numbers and how to re-run:
``docs/datasets/grasp-merged.md``.
"""

"""Defect detection with Anomalib.

    capture/    take photos of bricks in the L-shape (D415 cameras)
    cropping/   fixed-box crop to 256x256 (same code at training and at runtime) + dataset building
    training/   training scenarios (PaDiM, PatchCore, EfficientAd), evaluation, comparison

Flow: capture -> captures/anomalib/raw/<session> -> cropping.make_dataset -> captures/anomalib/dataset/<name>
      -> training.train_<model> -> exports/anomalib/<run> (OpenVINO) + outputs/anomalib/runs/<run>
Plan: docs/plan/02-data-and-training.md
"""

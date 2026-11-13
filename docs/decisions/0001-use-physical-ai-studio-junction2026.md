# 0001. Use Intel Physical AI Studio (junction2026 branch) as the main stack

- Status: Accepted
- Date: 2026-11-13
- Deciders: team

## Context

- The challenge is Intel's; "Intel tech integration" is 25 of 100 points.
- The official guide is `JUNCTION2026.md` on the `junction2026` branch of
  `open-edge-platform/physical-ai-studio`. It prescribes Studio for robot and camera setup,
  teleoperation recording, training (ACT, SmolVLA, Pi0.5; VLA training on Modal) and export
  (PyTorch, OpenVINO, ONNX, ExecuTorch), Anomalib for defect detection, and
  `openvinotoolkit/physicalai` (`InferenceModel`) to run policies outside the UI.
- Studio supports SO-101 natively (setup wizard, calibration, diagnostics;
  `physicalai-so101-plugin`).
- Studio uses LeRobot datasets internally, so plain LeRobot knowledge transfers.

## Decision

Build the pipeline on Physical AI Studio from the `junction2026` branch, Anomalib for detection,
the `physicalai` runtime and OpenVINO for the closed loop, with models placed on CPU / iGPU / NPU
and measured.

## Alternatives considered

- Plain Hugging Face LeRobot: works with SO-101 but is not Intel's stack and scores worse on
  integration.
- ROS 2 Robotics AI Suite Stationary Arm blueprint: see ADR 0002.

## Consequences

- We follow upstream's branch; pin the commit we use in the log when something works.
- Anomalib runs in its own venv to avoid PyTorch clashes.
- The rubric also mentions "Robotics AI Suite" deployment; revisit if time allows (ADR 0002).

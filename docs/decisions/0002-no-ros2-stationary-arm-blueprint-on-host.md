# 0002. Do not base the project on the ROS 2 Stationary Arm blueprint

- Status: Accepted
- Date: 2026-11-13
- Deciders: team

## Context

- Intel Robotics AI Suite (`open-edge-platform/robotics-ai-suite`) supports Ubuntu 24.04 with
  ROS 2 Jazzy only. The laptop runs Ubuntu 26.04. The suite's own `src/workshops/README.md` says the
  installer, the AMR repo and the OpenVINO apt repo have no 26.04 builds yet.
- The suite installer uninstalls host packages matching `ros-*`, `*realsense*`, `*openvino*`,
  `*level-zero*`, `libze1` and others before installing.
- The Stationary Arm blueprint (vision-guided pick and place, MoveIt 2 Servo) is validated only on a
  UR5e with a Robotiq 2F-85. The repo has no SO-101, LeRobot or Feetech code: we would have to write
  a URDF, a `ros2_control` hardware interface and a MoveIt config ourselves.
- Physical AI Studio (ADR 0001) supports SO-101 directly and runs on 26.04.

## Decision

Do not install the Robotics AI Suite on the host and do not build on the Stationary Arm blueprint.

## Alternatives considered

- Reinstall the laptop with Ubuntu 24.04: hours of work, loses the current setup.
- Canonical Workshop containers (LXD, Ubuntu 24.04 + Jazzy, GPU / NPU / camera passthrough) from
  `robotics-ai-suite/src/workshops`: viable for individual components (e.g. RealSense or OpenVINO
  object detection). Kept as an option.

## Consequences

- If we want "Robotics AI Suite" rubric points, run a suite component in a Workshop container
  (needs `sudo snap install lxd workshop`) and connect it to our pipeline. Revisit after the core
  loop works.

# Architecture decision records

One file per decision: `NNNN-kebab-title.md`, copied from [0000-template.md](0000-template.md).
Numbers are never reused. To change a decision, write a new ADR and mark the old one
`Superseded by NNNN`.

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-use-physical-ai-studio-junction2026.md) | Use Intel Physical AI Studio (`junction2026` branch) as the main stack | Accepted |
| [0002](0002-no-ros2-stationary-arm-blueprint-on-host.md) | Do not base the project on the ROS 2 Stationary Arm blueprint | Accepted |
| [0003](0003-run-studio-natively-from-upstream-lockfiles.md) | Run Studio natively from upstream lockfiles, not Docker or PyPI | Accepted |
| [0004](0004-patch-studio-for-realsense-rgbd-recording.md) | Patch Studio to record RealSense color + depth (`hj/realsense-rgbd`) | Proposed |
| [0005](0005-local-vlm-via-openvino-model-server.md) | Serve a local VLM with OpenVINO Model Server on the iGPU (Qwen3-VL-8B for now) | Accepted |
| [0006](0006-anomalib-brick-defect-detector.md) | Anomalib brick detector: PatchCore on 1.4x brick crops from all three cameras, on the NPU | Accepted |
| [0007](0007-observe-pose-homography-ik.md) | Locate bricks with a fixed-pose homography and hover above them with numeric IK | Proposed |
| [0008](0008-defect-decision-any-vote-on-zoomed-crops.md) | Defective if any camera or any detector says so, on zoomed brick crops | Accepted |
| [0009](0009-anomalib-as-uv-extra-in-project-env.md) | Install Anomalib as a uv extra of this project, not a separate venv | Accepted |
| [0010](0010-brick-localization-colour-cv-local-vlm-only.md) | Localize bricks with colour CV on the two D415s; VLMs only local and only as an option | Proposed |
| [0011](0011-model-weights-in-git-lfs.md) | Keep released model weights and annotations in git LFS under `artifacts/` | Accepted |

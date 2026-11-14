# 0004. Patch Studio to record RealSense color + depth

- Status: Proposed (implemented on branch `hj/realsense-rgbd`, under independent review)
- Date: 2026-11-13
- Deciders: team

## Context

- The team wants both D415 cameras recorded with color **and** depth in Studio datasets, at
  1280×720 @ 30 fps (T-011).
- Studio `junction2026` @ 17714488 cannot do this. It drops `output_type` / depth range in
  `camera_factory.py`, never reads depth, its publisher only encodes color, and its store only holds
  uint8 images. On top of that, changing a camera's resolution kills the shared-camera publisher and
  leaves a phantom iceoryx2 service, so every preview then times out (T-013), and the backend ignores
  SIGTERM (T-012).
- Upstream is a moving hackathon branch; we cannot wait for upstream fixes.

## Decision

Carry our own patch series on top of `junction2026`: branch `hj/realsense-rgbd` in
`third_party/physical-ai-studio` (local only), exported as `patches/physical-ai-studio/0001-…0006`.
Key choices:

- Feature names: color keeps `observation.images.<cam>`; depth is `observation.images.<cam>_depth`
  (`both` records both).
- Depth is a LeRobot 0.6 native depth feature, [H, W, 1], millimetres. It is stored as 12-bit log-
  quantized, lossless HEVC `gray12le` over 0–10 m; 0 stays "no data"; the step is about 1.3 mm at
  0.3 m and about 2.1 mm at 3 m. Values outside the camera's depth range are set to 0.
- Depth is aligned to color (`rs.align`), and both come from one frameset with one timestamp.
- Training excludes depth features for now (no Studio policy uses depth), via a new
  `exclude_depth` option.
- Studio starts and supervises its own camera publishers (one per device), replaces them on a
  settings change and recovers dead ones. The backend has a 10 s graceful-shutdown timeout.

## Alternatives considered

- 8-bit depth video: about 10 mm steps over 0.3–3 m plus lossy chroma artefacts; not LeRobot's
  native depth format.
- A separate depth recorder outside Studio, synced by timestamps: avoids patching Studio, but
  datasets would not contain depth and syncing is fragile.

## Consequences

- About 300 MB per minute per depth camera. LeRobot needs libx265 to read depth datasets.
- `git pull` of upstream now needs a rebase of our branch. Keep the patch series small and
  exported.
- To revisit when a policy should consume depth, or when upstream adds depth support.

# 0010. Localize bricks with colour CV on the two D415s; VLMs only local and only as an option

- Status: Proposed
- Date: 2026-11-14
- Deciders: agent A (perception), for the team

## Context

The policy should get a synthetic top-down "map" of the brick and the arm. The team asked for a
language/vision model to find the objects, and later (2026-11-14, about 05:40 PST) required that
localization and map generation run on a **local** model on the laptop: no frames to Google AI
Studio / Gemini or any other cloud API. A shared OpenVINO Model Server on this laptop serves the VLM
(Qwen3-VL-8B-Instruct or Gemma 4, OpenVINO IR, iGPU, OpenAI-compatible API on `localhost:8000/v3`).
RAM is shared with Studio while it records, so no second model may be loaded in parallel.

Scene facts (dataset `431a71f7`, 67 episodes): one Duplo brick (or two joined bricks) per episode
on white paper; colours red, yellow, lime green, blue/turquoise; the gripper is dark indigo; both
D415s stand low (about 9-13 cm above the table) on opposite sides of the table. Measured on 48
hand-annotated frames (`docs/perception/A-object-localization.md`): HSV colour blobs find the brick
in 100 % of top and floor frames with 100 % correct colour, median bottom-centre error 1.9-3.2 px,
fused table-position error 2.0 mm median / 6.3 mm p90, at 7-9 ms per camera on one CPU core.
The local VLM (Qwen3-VL-8B-Instruct int8 on the iGPU, per camera) on the same frames: 98 %
detection, colour 100 % (top) / 85 % (floor: lime green read as yellow), tighter boxes in the top
camera, fused 1.7 mm / 6.7 mm, 7.5 s per image; on the wrist camera 48/48 vs CV's 43/48. One
three-image request (top + floor + wrist) gets 100 % detection and colour in all views but looser
floor boxes (fused 4.3 / 8.7 mm) and takes 28 s.

## Decision

Use the classic CV pipeline (`hackjunction2026.perception`: sheet ROI + HSV blobs + top/floor
fusion + per-episode tracking) for offline annotation of the dataset and for live use. A VLM is used
only through the shared local OpenVINO Model Server, one request at a time, as an evaluated
alternative (per-camera, three-view and hybrid modes in `vlm.py`). It is not in the per-frame path.
Its sensible role is naming objects once per episode (three-view request) or when CV meets a colour
it does not know.

## Alternatives considered

- VLM for every frame: about 1000× slower (7.5 s per image on the iGPU vs 8 ms; the whole dataset
  would take ~80 h per camera pair), not more accurate in the table frame, and it misreads lime green
  in the floor camera. Measured, see the doc.
- Cloud VLMs (Gemini) or Modal-hosted open models: excluded by the team rule above. Modal was not used.
- Own OpenVINO detector (OWLv2 / Grounding DINO) loaded by this agent: not allowed while Studio
  records (shared RAM); also unnecessary for four saturated colours on white paper.

## Consequences

- Fast and deterministic, runs at camera rate, no GPU needed.
- Depends on the scene: white paper, saturated brick colours, lighting similar to the recordings.
  New colours or objects need new HSV ranges (or the VLM path). Thresholds and their sources are in
  `color_cv.py`.
- Revisit when objects are not plain coloured bricks (e.g. defect inspection), when the background
  changes, or when the VLM evaluation shows an advantage.

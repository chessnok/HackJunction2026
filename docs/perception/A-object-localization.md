# A: Table frame, brick and gripper localization, and how good it is

Agent A, 2026-11-14 (laptop clock, PST). Offline only: recorded data, no robot, no cameras, Studio
untouched. Dataset `431a71f7-50c6-4e38-9e38-d85984dde78b` ("collect-block"): **67 episodes / 19228
frames** when processed (the brief said 62 / 17724; the team kept recording).

Outputs live in `/home/intel/dev/captures/perception/A/` (gitignored). Code:
`src/hackjunction2026/perception/`, entry points `scripts/perception_*.py`. Decision record:
[ADR 0005](../decisions/0005-brick-localization-colour-cv-local-vlm-only.md).

**Requirement change (about 05:40 PST):** localization and the map must run on a **local** model on
this laptop; no frames to Google AI Studio / Gemini or any other cloud API. This agent never sent
frames to a cloud API and did not use Modal (no cloud baseline exists to compare with). The VLM path
talks only to the shared local OpenVINO Model Server (`localhost:8000/v3`), one request at a time.

## For agent C (short version)

- **Use `objects_<dataset>.parquet` (brick per frame) and `gripper_<dataset>.parquet`** (gripper
  tip per frame), both in the table frame, plus `bowls_<dataset>.parquet` (bowl centres per episode).
- **Table frame** (`table_frame_<dataset>.json`): plane z = 0 on the paper, **millimetres**,
  origin = far-left corner of the lower paper sheet in episodes 31-66, **x** along that sheet's far
  edge (left → right in the top image), **y** across the table toward the top camera, **z into the
  table** (x × y; cameras above the table have z < 0). Draw the map with x right and y down: it then
  looks like a rectified top-camera image. Canvas: x ∈ [-50, 750], y ∈ [0, 600] mm; `u, v` are
  normalised to that canvas.
- **Layout in that frame** (mm): top D415 at (143, 593), 131.5 mm high, looking toward -y; floor
  D415 at (140, -77) / (133, -78) / (166, -51), 79-109 mm high, looking toward +y; robot base origin
  at (528, 273), its +x points toward table -x (rotation 177.9°, plus a mirror because robot z is
  up); green bowl ≈ (499, 85), lavender bowl ≈ (498, 528), radius ≈ 86-96; bricks start at
  x 195-362, y 123-407.
- **Robot ↔ table**: `robot_to_table_<dataset>.json` maps agent B's FK frame to this one:
  `table_xy = s·R(θ)·(x_robot, -y_robot) + t`, s = 1.008, θ = 177.9°, t = (528.1, 273.1) mm;
  49 of 55 grasps fit with **6.1 mm median / 10.8 mm p90** residual. `table_z = -(z_robot + 2.4)`.
- **Accuracy** of a brick position (vs hand-annotated ground truth): fused top+floor estimate
  **2.0 mm median, 6.3 mm p90** relative to the same frame; the frame itself agrees with the arm FK
  to ~6 mm (above). Scale uncertainty ±3 %. Gripper tip: right on 16/16 (top) and 15/16 (floor)
  grasp frames, no false detections with the arm at home.
- **Failure modes**: lifted bricks (plane assumption breaks; `lifted` flag uses triangulation,
  0.8 % of frames), bricks cut in two by the closing jaws (merged), the gold wrist-camera PCB
  and hands look yellow (filtered by the track rules), tall/stacked/tipped bricks (each single
  camera is biased up to ~2.5 cm along its viewing direction, the two biases cancel in the fused
  mean). Task names are **not** colour labels (table below).
- **Live**: the same CV path runs at 15 ms per frame for both D415s plus the gripper on one CPU
  thread (67 Hz).

## Scene and cameras (what the data shows)

- Two RealSense D415 (848×480 colour, factory intrinsics from Studio's state snapshots, zero
  distortion) stand **low** (about 10-13 cm) on opposite sides of the table and look across it at
  each other. "top" (serial 243622062352) is not overhead; it sees the paper at a grazing angle
  (optical axis 58° from the table normal; the floor camera's 65-68°). "floor" (243622062278) faces it from beyond the
  far sheet edge. The wrist camera (1024×768) looks down past the purple jaws.
- Two overlapping white sheets cover the table; both are larger than the views, so **no camera sees
  all four sheet corners**. The top camera sees the lower sheet's far edge, its left edge and the
  seam where the upper sheet lies on it.
- **Camera / sheet changes between episodes** (all detected automatically, see the frame script):

  | Change | Episodes | How detected |
  | --- | --- | --- |
  | Top camera re-aimed | 0-5 vs 6-66 | wood edge behind the table shifts 21 px; fixed for 6-66 (shift ≤ 1 px; agrees with agent B's arm-based fit) |
  | Floor camera moved | 0-46 / 47-61 / 62-66 | sheet edges in the floor image jump 9-15 px (ep 47) and 14 px + 64 px (ep 62) |
  | Lower sheet moved | 0-5 / 6-30 / 31-66 | its left edge in the top image moves 30-100 px; does not affect the camera-fixed frame |

## Table frame

`scripts/perception_table_frame.py` → `table_frame_<dataset>.json` + check images in
`frame_check/` (100 mm grid drawn on each camera setup's first frame; rectified top|floor pairs).

Method:

1. **Sheet lines** (`sheet.py`): white mask (S < 50, V > 75), largest component; far edge = first
   white pixel per column (RANSAC line), left edge = first white pixel per row, seam = dark ridge
   inside the sheet. Per sheet position the median line over its episodes.
2. **Plane and joint calibration** (`calib.py`): with the factory intrinsics, the table normal n
   (top-camera coordinates) is the one for which the sheet's far edge is perpendicular to its left
   edge and parallel to the seam, in all three sheet positions. Jointly, a rotation per top-camera
   setup and a 6-DOF pose per floor-camera setup are fitted to 67 bricks seen alone in both cameras
   in episode first frames (top bottom-centre → table → brick footprint → far face → floor image).
   Soft-L1 least squares; brick correspondences reproject with 5.5 px median.
3. **Axes and origin**: x = direction of the far edge (episodes 31-66), y = n × x pointing toward
   the top camera, origin at the far-left corner of that sheet.
4. **Scale**: the homography fixes everything up to the top camera's height h. h = **131.5 mm**:
   Duplo edges read by eye gave 126 mm (103-143), agent B's metric camera fit from the arm gives
   129.6 mm, and registering B's FK gripper tip at 55 grasps to our brick positions (with h = 126)
   gave scale 0.958 → 131.5 mm. After the change the same registration gives s = 1.008. ±3 %.

JSON contents (`format: hackjunction2026/table-frame/v1`): `convention` (text above), `canvas`,
`cameras.<setup>` for `top_ep000-005`, `top_ep006-066`, `floor_ep000-046`, `floor_ep047-061`,
`floor_ep062-066` with `episodes`, `K`, `H_px_to_table_mm`, `H_table_mm_to_px`,
`R_cam_from_table`, `t_cam_from_table_mm` (`X_cam = R X_table + t`), `camera_xy_mm`,
`camera_height_mm`; `sheet_lines` (per sheet position: far-left corner, far / left / seam lines in
top-image pixels and mm, the corner angle 88.6-89.8° after the fit); `diagnostics` (plane normal,
runs, background shifts, per-setup reprojection errors, top-vs-floor brick agreement).

Checks: brick footprint centres from the top and from the floor camera agree to **7.1 mm median**,
17 mm p90 (67 first frames; on the hand-annotated frames 6.8 mm median, 26 mm p90 because of the
footprint model, see below);
the 2×4 Duplo front edges (63.5 mm) measure 60-64 mm after the scale fix; registration to B's FK
6.1 mm median.

Use: `hackjunction2026.perception.scene.Frame.load(path)`, then `to_table(cam, episode, pixels)`,
`footprint_center(cam, episode, bottom_centre_px)`, `to_uv(xy_mm)`, `setup_of(cam, episode)`.

## Localization methods

| Method | What it does | Code |
| --- | --- | --- |
| **CV v2** (classic baseline, recommended) | per camera: sheet ROI from the episode's first frame, minus static distractors (robot body, both bowls); HSV colour masks (red 0-7/170-180, yellow 8-25, green 26-48, blue 88-107; S ≥ 85, blue S ≥ 100; dark pixels allowed when S ≥ 120; the gripper's dark indigo is excluded); connected components, touching parts of different colours merge into one two-colour brick | `color_cv.py`, `scene.py` |
| **CV v1** | first version: blue up to hue 111, S ≥ 85, V ≥ 40 for all colours, no gripper exclusion | `color_cv.py` profile `v1` |
| **VLM per camera** | local VLM, one image, JSON-schema output `{objects: [{color, second_color, bbox_2d}]}` | `vlm.py` |
| **VLM multi-view** | top + floor + wrist images in one request, one entry per physical brick with a box per image | `vlm.py` |
| **Hybrid** | VLM decides which bricks exist and their colour, the overlapping CV blob gives the box | `vlm.py:hybrid_merge` |

Per-frame fusion and tracking (`annotate.py`): every detection becomes a table point (bottom-centre
of the box → plane → moved 16 mm, half a 2×2 Duplo, away from that camera). Top and floor
detections are paired by colour and either plane distance < 80 mm or meeting rays (triangulated
box centres); views closer than 30 mm pair whatever their colour (a stacked two-colour brick can
show only its yellow part to one camera and only its blue part to the other; the pair gets both
colours). A paired brick's position is the mean of the two plane estimates, or the triangulated
point when the brick is lifted (centre > 27 mm above the table; resting bricks triangulate at
13-18 mm). Same-colour detections within 40 mm merge. Tracks: nearest neighbour in the table plane,
same colour, gate 120 mm; a detection within 40 mm of an existing track never starts a new one; a
track is kept if it exists within the first 15 frames or covers ≥ 25 % of the episode (dataset
assumption: every brick is on the paper when recording starts); an unseen track is held at its
last position for at most 45 frames (1.5 s).

## Ground truth and evaluation

`scripts/perception_export_eval_frames.py` exports 48 frames: 16 episodes covering every colour,
two-colour bricks, the 2×4 bricks and all camera setups (4, 8, 13, 19, 22, 25, 29, 33, 36, 41, 44,
46, 49, 53, 58, 63) × 3 phases (start = frame 0, mid = 55 % with the arm approaching, late = 90 %
with the jaws at or around the brick; 32 of the 96 top/floor views are partly occluded by the
gripper, 2 frames have the brick lifted).

Annotation (`src/hackjunction2026/perception/gt_431a71f7.py`): crops around the brick, upscaled 2×
with a 10 px grid labelled in full-image pixels; the box was read off by eye (about ±2 px, the
bottom edge most carefully) with **no detector output shown**. Colours from the image, cross-checked
with the task name. Wrist camera: presence and colour only (the brick is visible in all 48).
Matching: a prediction matches if IoU ≥ 0.3 or its centre lies in the GT box. Table error: the
prediction's footprint point vs the GT box's footprint point through the same camera
(detection error in mm); fused: mean of the available camera estimates vs the mean of the GT
estimates. Lifted frames are excluded from mm errors.

`scripts/perception_eval.py` → `evaluation/metrics.json`, cached predictions `evaluation/pred_<method>.json`,
failure images `evaluation/fail_<method>/`.

### Results (48 frames; top and floor: 48 GT boxes each)

| Method | Camera | Detected | Colour exact | False pos. / frame | Bottom-centre px median / p90 | IoU median | Table mm median / p90 | Time per image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CV v2 | top | 100 % | 100 % | 0.08 | 3.2 / 11.0 | 0.87 | 3.7 / 12.0 | 9 ms |
| CV v2 | floor | 100 % | 100 % | 0.00 | 1.9 / 4.2 | 0.88 | 3.0 / 5.3 | 7 ms |
| CV v2 | **fused** | | | | | | **2.0 / 6.3** | |
| CV v2 | wrist | 43/48 colour hit, 41/48 exact | | | | | (no table position) | 48 ms |
| CV v1 | top | 100 % | 100 % | 0.04 | 3.0 / 21.0 | 0.88 | 3.4 / 22.2 | 6 ms |
| CV v1 | floor | 100 % | 100 % | 0.00 | 2.0 / 4.0 | 0.88 | 3.0 / 5.4 | 5 ms |
| CV v1 | fused | | | | | | 1.9 / 11.4 | |

VLM rows: see "Local VLM results" below.

Detection by occlusion (CV v2): 100 % for free, partly hidden and mostly hidden bricks in both
cameras.

CV v1 → v2: the v1 failures were the shadowed lower face of the dark-green 2×4 brick (V 19-34,
missed), and the blue range catching the indigo jaws and bluish shadows (top p90 21 px). The v2
thresholds were set from colour statistics of the **51 episodes outside the evaluation set** (blue
bricks: H 89-106, S ≥ 110 at the 5th percentile; jaws: H 101-119 at V < 70; green faces down to V ≈ 20),
but they were motivated by evaluation failures, so v2 numbers are slightly optimistic.

Remaining CV failures (`evaluation/fail_cv/`): ep 49 top (a bluish reflection under a turquoise brick
extends the box 18 px down), ep 19 late top (lifted brick).

Which cameras matter: the floor camera is the most precise for bricks (1.9 px, its view of the brick
bottom is less foreshortened), the top camera adds independent coverage and, together with the floor
camera, cancels the footprint-depth bias (single-camera table error p90 12 / 5 mm, fused 6 mm,
GT-vs-GT across cameras 26 mm p90 for tall or tipped bricks). The wrist camera does not give table
positions without the arm pose; CV finds the brick in 43/48 wrist frames (misses: brick far away and
small, or held tight between the jaws), so it is not used for the map.

### Gripper (`gripper.py`)

Dark indigo jaws: H 106-135, S ≥ 80, V ≤ 80 (the lavender bowl has the same hue but V 83-124), minus
robot parts touching the image border in the episode's first frame (arm at home); the arm component
must touch the top border (the arm always enters the low D415 views from above). Tip = lowest arm
pixel (mean x of the lowest 8 rows). Top + floor tips are triangulated (`tip_*_mm`; the two lowest
points are not exactly the same physical point, error up to a jaw width, `tip_tri_err_px` median
9.7 px). Evaluation on the GT frames: late frames (jaws at the brick) tip within 30 px of the brick:
**top 16/16, floor 15/16** (ep 46: only a sliver of the jaw visible behind the brick); start frames
(arm at home): **0/16 false detections** in both cameras. 3 ms per image. Over the dataset the
gripper is visible in 36 % (top) / 50 % (floor) of frames, in both in 34 %.

For registering FK to the table, prefer `robot_to_table_<dataset>.json` (grasp-based, uses B's
FK and our bricks) over the gripper tip: the tip is the lowest *visible* arm pixel, not a fixed
point on the gripper.

### Local VLM results

PENDING: the shared OpenVINO Model Server was still downloading Qwen3-VL-8B-Instruct (int8) at
06:15 PST.

## Dataset annotation (outputs)

`scripts/perception_annotate.py` ran over **every frame** of all 67 episodes (no interpolation):
19228 frames in 179 s with 4 niced worker processes (decode + CV, single-threaded OpenCV per worker).

`objects_<dataset>.parquet`: one row per frame and brick (19228 rows, one brick per episode).

| Column | Meaning |
| --- | --- |
| `dataset`, `episode_index`, `frame_index`, `timestamp` | as in the dataset (`timestamp` = frame / 30) |
| `object_id` | track id within the episode (0, 1, …) |
| `class` | `brick` |
| `color` | majority colour of the track: `red`, `yellow`, `green`, `blue` or two of them joined by `+` (larger part first) |
| `frame_color` | colour seen in this frame |
| `x_mm`, `y_mm` | table position (footprint centre) |
| `u`, `v` | normalised canvas position |
| `z_centre_mm` | triangulated height of the box centre (negative = above the table; resting ≈ -13 … -18) |
| `lifted` | brick above the table (in the gripper) |
| `source` | `top+floor`, `top`, `floor` or `hold` (not seen; last position kept) |
| `confidence` | 0.9 both cameras, 0.6 one camera, 0.3 hold |
| `frames_since_seen` | 0 when observed |
| `top_bbox`, `floor_bbox` | `[x0, y0, x1, y1]` image pixels (null when not seen by that camera) |
| `top_point_px`, `floor_point_px` | box bottom-centre used for the table point |
| `top_x_mm`, `top_y_mm`, `floor_x_mm`, `floor_y_mm` | single-camera table estimates |
| `top_area_px`, `floor_area_px` | blob area |

`gripper_<dataset>.parquet`: one row per frame: `<cam>_visible`, `<cam>_tip_px_x/y`, `<cam>_bbox`,
`<cam>_area_px`, `<cam>_tip_plane_x/y_mm` (tip pixel on the table plane: right only when the tip
touches the table), `tip_x/y/z_mm` and `tip_tri_err_px` (triangulated when both see it).

`bowls_<dataset>.parquet`: per episode and bowl: `color` (`green` from the top camera, `lavender`
from the floor camera), `x_mm`, `y_mm`, `radius_mm` (rim nearest to the camera + radius; ±2 cm),
`bbox_px`, `cut_by_image_border`.

`robot_to_table_<dataset>.json`: see the C section. `preview/ep<NNN>.mp4|png`
(`scripts/perception_map_preview.py`): camera frames with detections next to the map, a check tool.

Annotation statistics: the brick is seen by both cameras in 99.3 % of frames, one camera 0.4 %,
held 0.2 %; lifted in 0.8 % of frames (13 episodes; recordings end around the grasp; the brick is never carried to a bowl). Start position jitter over the first 30 frames: 1.4 mm median, 6.4 mm
max. Single-camera estimates of the same brick differ by 7.5 mm median / 18.4 mm p90 (the fused mean
cancels most of it).

### Task names vs actual brick colour

| Episodes | Task name | Brick |
| --- | --- | --- |
| 6-11 | yellow-right | yellow (orange-yellow) |
| 12-17 | yellow-right | **red** |
| 40-42, 46 | colored / ne-potuzno | red + blue (joined or stacked) |
| 43-45 | potuzno | blue + yellow (stacked) |
| 56, 65, 66 | red-blue | red + blue |
| 57-59 | red-blue | **blue only** |
| 60 | red-blue | **green** |

## Runtime

| What | Where | Time |
| --- | --- | --- |
| CV bricks top + floor + fusion | 1 CPU thread, nice 19 | 7.7 ms median, 9.6 ms p90 |
| Gripper top + floor | same | 7.3 ms median |
| Full live step (both) | same | 15 ms median, 17 ms p90 → **67 Hz** |
| Per-episode setup (sheet ROI, masks from the first frame) | same | 110-140 ms once |
| Whole dataset (decode + CV), 19228 frames | 4 processes, nice 19 | 179 s |

`scripts/perception_benchmark.py` reproduces the first rows (827 frames of episodes 20, 41, 63).
Modal GPU: not used (local-only rule; nothing heavy to offload).

## Recommendation

- **Offline annotation: CV (v2) with top+floor fusion and tracking**, as in the parquet files. 100 %
  detection and colour on the GT, 2 mm median / 6 mm p90 fused, the whole dataset in 3 minutes.
- **Live (≥ 10 Hz): the same CV path**, 67 Hz on one thread. Compute the sheet ROI and distractor
  masks once per episode (or when the scene changes) from a frame with the arm at home.
- The VLM is not needed for the map of this scene; it could label new object types once per episode
  (see the VLM section for its measured accuracy and latency).

## Limits, untested, blocked

- Scale comes from the arm FK registration (±3 %); no ruler measurement on the table.
- Footprint model: a fixed 16 mm from the box bottom-centre; tall, stacked or tipped bricks put a
  single camera off by up to ~2.5 cm. Use fused rows (`source == top+floor`) where possible.
- The CV thresholds are tuned to this paper, these colours and this light. New brick colours (e.g.
  white, black, purple) would be missed or confused with the gripper.
- Episodes 0-5 use a separately fitted top-camera rotation from only 6 bricks (reprojection 10.6 px):
  less accurate than the rest.
- Wrist camera: detection evaluated (43/48), no table positions (needs the arm pose).
- The policy-side map renderer is agent C's; `perception_map_preview.py` only checks the data.
- Not verified on live cameras: the live path is measured on recorded frames.

## How to run

```bash
PY=/path/to/a/venv/with/numpy,opencv-python-headless,av,pyarrow,pandas,scipy/bin/python
export PYTHONPATH=src
nice -n 19 $PY scripts/perception_table_frame.py 431a71f7          # frame JSON + check images
nice -n 19 ionice -c 3 $PY scripts/perception_annotate.py 431a71f7 --workers 4
$PY scripts/perception_bowls.py 431a71f7
$PY scripts/perception_register_robot.py 431a71f7                   # needs agent B's grasps CSV
$PY scripts/perception_export_eval_frames.py 431a71f7                # GT frames (once)
$PY scripts/perception_eval.py --methods cv,gripper                  # + vlm,vlm_mv,hybrid with the local server
$PY scripts/perception_map_preview.py 431a71f7 --episodes 20,63
$PY scripts/perception_benchmark.py 431a71f7
```

Re-run the first four after new episodes are recorded: camera setups are detected again, and a new
camera position needs at least ~4 single-brick episodes to fit.

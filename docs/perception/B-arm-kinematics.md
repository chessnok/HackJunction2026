# B: SO-101 follower forward kinematics from recorded joints

Agent B, 2026-11-14 (laptop clock, PST). Offline only: no robot, no cameras, Studio untouched.
Dataset: `431a71f7-50c6-4e38-9e38-d85984dde78b` (67 episodes / 19228 frames when processed; the brief
said 62 / 17724, the team kept recording). Calibration: `configs/calibration/follower.json`.

Images and data referenced below live in `captures/perception/B/` (gitignored; paths are relative to
the repo root of the main checkout `/home/intel/dev`).

## For agent C (short version)

- **Frame**: SO-101 URDF `base_link`, **millimetres**. +x forward (the direction the arm points at
  shoulder_pan = 0), +y to the robot's left, +z up. The shoulder_pan axis is the vertical line
  x = 38.8, y = 0. The bottom of the base is at z = -2.4.
- **Table**: the plane **z = -2.4 mm** (`hackjunction2026.kinematics.TABLE_Z_MM`), normal = +z.
  The base stands on the table; 55 grasps put the fingertips 5.0 ± 4.1 mm above that plane, and a
  plane fitted through them is tilted 0.42° (not significant, slope standard error larger than the
  slope). So robot → table registration is planar: a rotation about z, an xy translation and
  z_table = z_robot + 2.4.
- **Per-frame positions**: `captures/perception/B/arm_431a71f7-50c6-4e38-9e38-d85984dde78b.parquet`
  (schema below). For the map use `gripper_tip_{x,y,z}_mm` (midpoint of the two fingertips) and
  `jaw_gap_mm`; the arm polyline is `base → shoulder → elbow → wrist → gripper_base → gripper_tip`.
- **Registration aids** (optional):
  - `grasps_<dataset>.csv`: one grasp per episode (55), frame where the jaws hold the brick and the
    gripper tip there. Pair these with agent A's brick positions to fit robot → table.
  - `cameras_<dataset>.json`: pinhole models robot base frame → camera for the **floor** camera
    (episodes 0–46) and the **top** camera (episodes 6–66), fitted from the arm itself. With A's
    table frame in top-camera coordinates this gives robot → table directly. Median reprojection
    error 7.7 px (floor) / 9.4 px (top), about 4–5 mm.
- **Accuracy**: about ±5 mm typical, worst ~15 mm (details under Limits).
- A constant error of the shoulder_pan zero would rotate the whole base frame about z; nothing in
  our data can see it (it is absorbed by any registration that fits a rotation). Fit the rotation,
  do not assume +x is a known direction on the table.

## Library and CLI

`src/hackjunction2026/kinematics/` (numpy only at runtime):

| File | Content |
| --- | --- |
| `model.py` | URDF geometry (joint origins copied from the follower URDF), `link_frames(q)`, `keypoints(q)`, `jaw_gap_mm(q_gripper)`; vectorized over any leading shape |
| `mapping.py` | `Calibration.from_file`, normalized ↔ ticks (exact inverse of the plugin), ticks ↔ URDF radians (`JOINT_MAP`) |
| `__init__.py` | `arm_keypoints(norm, cal)`: keypoints + `jaw_gap_mm` + `jaw_tip_dist_mm` from Studio normalized values; `TABLE_Z_MM` |
| `cli.py` | entry point `scripts/arm_fk.py` |

```bash
uv run scripts/arm_fk.py pose 0 -99.2 62.7 77.6 -56.9 13.5        # one joint vector (normalized)
uv run --with pyarrow scripts/arm_fk.py annotate 431a71f7          # dataset -> captures/perception/B/arm_<id>.parquet
uv run scripts/arm_fk.py check-urdf                                # built-in geometry == Studio's URDF
third_party/physical-ai-studio/application/backend/.venv/bin/python scripts/arm_fk.py check-plugin
```

```python
from hackjunction2026.kinematics import Calibration, arm_keypoints
cal = Calibration.from_file("configs/calibration/follower.json")
kp = arm_keypoints(state, cal)        # state: (N, 6) observation.state -> dict of (N, 3) mm, (N,) mm
```

FK of 2 × 19228 frames (state and action) takes about 60 ms.

### Keypoints

All in mm in the robot base frame. Joint "origins" are points on the joint axes as placed by the
URDF (the upper-arm chain sits about 18 mm to the right of the arm's centre plane, y ≈ -18 at
pan = 0, because the URDF puts the joint frames on the servo side).

| Keypoint | Definition |
| --- | --- |
| `base` | `base_link` origin (constant 0, 0, 0) |
| `shoulder` | shoulder_lift joint origin |
| `elbow` | elbow_flex joint origin |
| `wrist` | wrist_flex joint origin |
| `gripper_base` | wrist_roll joint origin (`gripper_link`) |
| `fixed_jaw_tip` | lowest 1 mm of the fixed jaw mesh, (-9.93, -0.23, -103.95) in `gripper_link` |
| `moving_jaw_tip` | outermost 1 mm of the moving jaw mesh, (-10.13, -81.53, 18.90) in its link |
| `gripper_tip` | midpoint of the two fingertips: the point that surrounds a grasped object |
| `tcp` | URDF `gripper_frame_link` (LeRobot's tool centre point), 5.8 mm back from the fixed fingertip along the gripper axis |

`jaw_gap_mm`: smallest distance between the two jaws within their lowest 35 mm, from the meshes, as a
function of the gripper angle (table in `model.py`; 0 when the jaws touch, monotone up to 45°, held at
53.4 mm above). `jaw_tip_dist_mm`: distance between the two fingertip points.

### Parquet schema (`arm_<dataset>.parquet`, one row per frame, 86 columns)

- `episode_index`, `frame_index`, `index` (int64), `timestamp` (float32 s): copied from the dataset.
- From `observation.state` (no prefix): `shoulder_pan` … `gripper` (Studio normalized, float32),
  `q_<joint>_deg` (URDF angle, degrees), `<keypoint>_{x,y,z}_mm` for the 9 keypoints above,
  `jaw_gap_mm`, `jaw_tip_dist_mm`.
- The same from `action` with prefix `act_`. `action` is the leader arm's normalized value, which
  the follower maps through its own calibration, so `act_*` is the **commanded** follower pose. It
  differs from the measured one by the tracking lag (elbow sags 2–3 units under gravity) and when the
  gripper is blocked by a brick (`act_jaw_gap_mm` ≈ 0 while `jaw_gap_mm` is the brick width).
- Schema metadata key `hackjunction2026.arm_fk`: JSON with dataset, calibration ranges, joint map,
  frame description, code commit.

Re-run `annotate` after new episodes are recorded; it always processes the whole dataset.

## Joint mapping

Studio records `normalized = (tick - range_min) / (range_max - range_min) * 200 - 100` (body joints,
clipped to ±100) and `* 100` (gripper, 0..100), where `tick` is the servo's Present_Position, which
already includes the EEPROM homing offset (plugin `_conversion.ticks_to_normalized`). Then

```
tick  = range_min + (normalized + 100) / 200 * (range_max - range_min)     (gripper: normalized / 100 * span)
q_rad = sign * (tick - zero_tick) * 2π / 4096
```

| Joint | range_min | range_max | sign | zero_tick | q_deg = a·normalized + b: a | b | q range over the calibrated span |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shoulder_pan | 764 | 3467 | +1 | 2047 | 1.1878 | +6.02 | -112.8 .. +124.8 |
| shoulder_lift | 826 | 3220 | +1 | 2047 | 1.0521 | -2.11 | -107.3 .. +103.1 |
| elbow_flex | 853 | 3054 | +1 | 2047 | 0.9672 | -8.22 | -104.9 .. +88.5 |
| wrist_flex | 828 | 3292 | +1 | 2047 | 1.0828 | +1.14 | -107.1 .. +109.4 |
| wrist_roll | 162 | 3992 | +1 | 2047 | 1.6831 | +2.64 | -165.7 .. +170.9 |
| gripper | 2047 | 2941 | +1 | 2201 | 0.7857 | -13.54 | -13.5 .. +65.0 |

URDF zero pose: upper arm (almost) vertical, forearm and gripper horizontal pointing +x. Positive
shoulder_lift / elbow_flex / wrist_flex tilt the respective link forward/down; positive shoulder_pan
turns the arm towards -y (clockwise seen from above); positive gripper opens.

Why these numbers:

- **Body joints, zero = tick 2047.** Studio's calibration wizard calls lerobot
  `set_half_turn_homings` (`so101_setup_worker.py`), which writes the homing offset so that the pose
  the arm is held in at that moment reads `int(4095 / 2) = 2047` on every servo. The operator holds
  the arm in the "middle" pose, which is the zero pose of this ("new calibration") URDF. The **range
  midpoint** (Studio's `degrees` unit, LeRobot's `MotorNormMode.DEGREES`) is *not* the URDF zero
  here: the recorded ranges are not symmetric about the homing pose (elbow 8.2°, pan 6.0° off).
  Studio's 3D viewer is further off: it sets URDF angle = normalized value in degrees
  (`robot-models-context.tsx`, `degToRad(joint.value)`), which ignores both scale (0.97–1.68° per
  unit) and offset.
- **Signs +1.** All `drive_mode` are 0 and the URDF axes follow the servo direction (LeRobot uses
  this URDF with drive_mode 0). Verified per joint below: flipping lift, elbow or wrist_flex puts the
  fingertips 100–300 mm above the table at grasps; flipping pan, roll or gripper raises the camera
  reprojection error 2–4× (pan flipped has no valid camera at all).
- **Gripper, zero = tick 2201.** Normalized 0 = tick 2047 = the calibration minimum is not where the
  jaws touch. When the empty gripper is commanded closed (action < 3, 673 frames) the follower stops
  at normalized 2.57 median (2.24–3.13, 5–95 %), i.e. tick 2070: the jaws touch there. In the URDF
  meshes the jaws touch at q = -11.6° (gap 0.6 mm at -11°, 0.07 mm at -12°). Hence
  zero_tick = 2070 + 11.6 / 360 · 4096 ≈ 2201. The gripper servo drives the moving jaw directly
  (360° / 4096 ticks).
- **Calibration matches the recording.** Every one of the 19228 recorded states maps back to an
  integer tick with this calibration file (max deviation 0.0001 ticks); a 1-tick change of a single
  range_min breaks this (0.5). `check-plugin` confirms that `normalized_to_ticks` inverts the plugin's
  own conversion over all 14492 ticks inside the ranges (max error 0.0001 ticks).
- **Geometry.** Studio serves the same URDF it bundles (`GET /api/robots/catalog/SO101_Follower/urdf`
  is byte-identical to `site-packages/urdf/so101-follower/so101_follower.urdf`, sha256
  `3a65d2d3…022c`); `check-urdf` compares the built-in numbers with that file.

## Validation

### 1. Fingertips at the table when a brick is grasped

Grasp event = the gripper is commanded at least 6 units more closed than it is, does not move, and is
above the empty-closed value, for ≥ 10 frames (the jaws hold an object); first such run per episode,
5 frames in. 55 of 67 episodes have one. Lowest fingertip z (keypoints, mm):

| Mapping | mean | std | min | max | trend vs reach |
| --- | --- | --- | --- | --- | --- |
| **chosen (zero = 2047)** | **2.6** | **4.1** | -7.2 | 11.9 | +0.7 mm / 100 mm |
| zero = range midpoint (Studio "degrees") | -19.5 | 5.6 | -34.6 | -9.8 | -7.3 mm / 100 mm |
| Studio UI viewer (normalized as degrees) | -16.3 | 7.5 | -32.4 | -1.0 | +10.8 mm / 100 mm |
| shoulder_lift sign flipped | 231.7 | 143.2 | | | |
| elbow_flex sign flipped | 100.6 | 107.9 | | | |
| wrist_flex sign flipped | 301.9 | 36.9 | | | |

With the chosen mapping the fingertips are 5.0 ± 4.1 mm above the base bottom (standard error of the
mean 0.6 mm), independent of reach; the range-midpoint convention puts every grasp 10–35 mm *below*
the table and gets lower with reach. With the full jaw meshes instead of the two tip points the
chosen mapping gives 0.5 ± 4.7 mm (the lowest jaw point is slightly lower than the tip points when
the gripper is tilted). A plane through the grasp fingertips: z = 0.8 + 0.0073 x + 0.0005 y
(standard errors 3.2, 0.013, 0.007), tilt 0.42°, residual 4.1 mm: no measurable tilt between the
table and the base frame.

![grasp heights](../../captures/perception/B/grasp_height_vs_reach.png)

Fitting zero corrections from the grasp heights alone is ill-posed (three angles trade off against
the table height: a free lift + elbow + wrist fit runs off to +46° elbow with the table 50 mm below
the base). One joint at a time, with the table height free: lift +0.6 ± 0.9°, elbow +4.6 ± 1.6° (but
then the table is at -9 mm, below the base bottom, which is impossible), wrist_flex +4.0 ± 1.6°;
residual std drops only from 4.7 to 4.4–4.5 mm. No correction is applied.

### 2. Rest pose and nothing below the table

At the first frame of every episode (folded rest pose) the median keypoints are shoulder
(70, -18, 117), elbow (-46, -20, 111), wrist (-17, -18, 243), gripper base (34, 0, 213), gripper tip
(121, 4, 164): upper arm lying backwards, forearm folded up over it, gripper hanging forward, no moving
link below z = 16 mm. Over all 19228 frames the lowest fingertip is above -0.7 mm in 95 % of frames,
above -5.8 mm in 99 %, minimum -13.9 mm (episode 64 frame 255, gripper wide open with wrist_flex near
its limit, pressing on the paper); with the full link meshes 104 frames (0.5 %, episodes 22, 28, 35,
47, 58, 64) reach more than 5 mm below the base bottom, minimum -14.4 mm.

![lowest fingertip](../../captures/perception/B/lowest_point_hist.png)
![top and side view](../../captures/perception/B/fk_topdown_side.png)

### 3. Cameras: the FK gripper lands on the gripper, and the base on the base

For each grasp episode, 8 frames around the grasp (75 frames before to 15 after) were taken from the
top and floor videos. The arm is segmented by colour (purple; plus very dark pixels in the backlit top
view) and the **lowest arm pixel** is taken as the observation. The model prediction is the lowest
projected vertex of the gripper and moving-jaw meshes from FK. A pinhole camera (focal length,
rotation, translation; principal point at the image centre, no distortion) was fitted per static
camera group by robust least squares, initialised from brick centroids in the first frames vs FK
grasp points. Observations > 25 px off (wrong segmentation: shadows, dark red bricks, the bowl) were
dropped after the first pass.

| Camera (episodes) | obs used / detected | median | p90 | median in mm | focal | camera centre in base frame (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| floor (0–46) | 158 / 167 | 7.7 px | 14.6 px | 3.8 | 709 | (403, -409, 98) |
| top (6–66) | 176 / 312 | 9.4 px | 23.2 px | 4.9 | 589 | (389, 319, 127) |

Checks with the same observations, camera refitted each time (median px floor / top): range-midpoint
zero 9.8 / 11.3 and needs implausible focal lengths (460 / 479 vs about 600 for a D415 at 848×480);
pan sign flipped 27.3 / 34.4 (camera driven to infinity); wrist_roll flipped 21.9 / 27.7; gripper
flipped 28.4 / 16.5. A first floor fit with free zero corrections gave lift -3.2 ± 2.5°, elbow
+0.2 ± 1.8°, wrist_flex -0.4 ± 2.3°; profiling each correction at ±3° and ±6° (both cameras refitted)
changes the median by less than 1 px, so the cameras confirm the mapping to about ±5° per joint, not
better. The brick-centroid fits alone (16.9 vs 17.2 px RMS) cannot tell the conventions apart.

Independent check: the cameras were fitted on the gripper only, yet the projected base and shoulder
meshes land on the robot base visible in the corner of both views, and the rest-pose arm overlays the
folded arm (floor, episode 20 frame 0).

![floor overlay](../../captures/perception/B/overlay_floor.png)
![top overlay](../../captures/perception/B/overlay_top.png)

Overlay colours: white polyline = keypoints base → gripper tip, red cross = `gripper_tip`, coloured
dots = projected link meshes (orange gripper, red moving jaw, green forearm, cyan shoulder, grey base).

Camera groups (the cameras were moved between episodes; phase correlation of first frames): top
0–5 | 6–66; floor 0–46 | 47–61 | 62–66. Only top 6–66 and floor 0–46 were fitted. The top camera saw
the arm consistently over 6–66 (median 7.0–9.4 px in every block of ~15 episodes), so the robot base
did not move there.

### 4. Gripper opening

At the 55 grasps the stalled gripper values fall into three groups: normalized 12.4 (25 grasps) →
jaw gap 10.5 mm, 22.9 (26) → 21.4 mm, 32.9 (4, all two-brick stacks: episodes 41, 42, 56, 65) →
31.4 mm. The bricks were not measured, so the mm scale is **not validated** (if they are real Duplo,
a 2-stud side is 31.7 mm and a 1-stud side 15.7 mm; in the wrist camera, episode 0 (21 mm group)
seems to hold its brick across the 2-stud side, which would mean the gap is underestimated by up to
10 mm). Measuring two bricks with calipers would settle it. Jaw closure (touch
at normalized 2.6) and the opening direction are validated.

## Limits

- Position error: grasp heights scatter by 4.1 mm (this includes how high the operator gripped);
  camera residuals 4–5 mm median, p90 7–13 mm (includes segmentation and lens errors). Expect about
  ±5 mm, up to ~15 mm at the edges of the workspace. Not modelled: gear backlash (about ±0.5° per
  joint), link and gripper flex under load, servo tracking lag (we use the measured position, so lag
  is not an error for `observation.state`).
- Zero corrections up to about ±3° (lift, elbow, wrist_flex) cannot be excluded by the data; the
  wrist_roll zero is only checked through the jaw silhouette; the shoulder_pan zero is unobservable
  (gauge, see above).
- The 2047 convention holds because the operator held this arm in the URDF middle pose during
  homing. After a recalibration, re-run `scripts/arm_fk_validation/grasp_checks.py` on a new dataset.
- Calibration file and URDF are fixed inputs; the follower is the only arm covered.
- Camera models: approximate intrinsics (focal fitted, no distortion), only two of five camera groups.
- `jaw_gap_mm` in mm: see section 4.
- Untested: nothing here was run on the robot; all checks are offline against recorded data.

## Reproduce

Library-only checks (numbers in sections 1–2, plots, grasp CSV):

```bash
uv run scripts/arm_fk_validation/grasp_checks.py 431a71f7
uv run scripts/arm_fk_validation/topdown_plot.py
```

The image-based work (section 3, constants in `model.py`) is exploration code kept in
`scripts/arm_fk_validation/` as it was run. It needs an env with numpy, pandas, pyarrow, scipy,
matplotlib, av, opencv-python-headless and trimesh (agent B used a scratch uv venv) and writes
`img/` and `*.json` into the current directory; run it from a scratch directory with the scripts
folder on `PYTHONPATH`, in this order:

| Script | What it does |
| --- | --- |
| `urdf_fk.py`, `vfk.py`, `meshes.py` | independent URDF parser + FK (used to cross-check the library) and mesh loading |
| `consts.py` | fingertip points and jaw-gap table from the meshes (values in `model.py`) |
| `grip_closed.py` | gripper value when commanded closed (jaw touch at normalized 2.6) |
| `data.py`, `grasp_z.py`, `fit_z.py`, `lowest.py` | early grasp-height and lowest-point analysis with the mesh points |
| `camgroups.py` | camera moves between episodes |
| `frames.py`, `bricks.py`, `fit_cam.py` | brick centroids in first frames, brick-based camera fit (initialisation) |
| `extract_val.py`, `armmask.py`, `fit_extreme.py`, `fit_top.py` | validation frames, arm masks, extreme-point camera fits (early runs used gripper zero 2047) |
| `camera_final.py`, `offsets_joint.py`, `offsets_profile.py` | final camera fits with the library mapping, sign checks, correction profile |
| `overlay.py`, `obs_sheet.py` | overlay sheets and observation contact sheets |

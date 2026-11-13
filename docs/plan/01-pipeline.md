# Plan 1/3: Overall pipeline

Status: **plan, nothing here is tested on hardware yet.** Written 2026-11-13.
Other parts: [02-data-and-training.md](02-data-and-training.md) (datasets, training, Modal budget),
[03-setup.md](03-setup.md) (installing and configuring everything).

## Goal

The SO-101 takes a Duplo/Lego brick from anywhere on the table (shifted, rotated, maybe upside
down, maybe among clutter), inspects **all 6 faces**, decides "perfect" or "defective", and puts it
on the matching plate. No human chooses what the robot does.

Our bricks (practice set; the kickoff set may differ, so keep shapes configurable):

| Shape | Count | Notes |
| --- | --- | --- |
| 2×2 brick | 3 (different colours) | colour is not a defect unless kickoff says so |
| 4×2 brick | 1 | footprint ratio about 2:1 |
| 2×2 brick + slope (one stud row longer) | 1 | footprint ratio about 1.5:1; may not lie flat when upside down (test) |

## Key design choices

1. **AI only where the situation varies.** The brick starts anywhere, so the first pick uses a
   learned **ACT** policy. After that the brick sits at a **fixed inspection spot**, so every
   later motion (look, flip, place) is **scripted**: fixed joint poses recorded once with the
   leader arm and replayed. Scripted motions are repeatable, which the anomaly model needs.
2. **Inspect on the table, not in the air.** No finger occlusion, no motion blur, fixed background.
   Two placements (before and after a flip) cover all 6 faces.
3. **One generalized Anomalib model** for every view (top, bottom, sides) and every shape, trained
   on "good" images only. No routing between models. We split it into per-view or per-shape models
   only if the per-view evaluation shows a weak view (likely missing studs on tops, see
   "Known risks").
4. **REASON is plain Python rules** (`max(scores) > threshold` → plate), set in a config file so
   the kickoff rules can be applied without touching code.

## Physical layout

```text
          ┌──────────── table ────────────┐
          │  [GOOD plate]   [DEFECT plate] │     plates: fixed, taped down
          │                                │
          │   workspace: bricks start      │     overhead: RealSense D415 (sees everything)
          │   anywhere here (+ clutter)    │
          │                                │
          │        ◣ inspection corner     │     L-shaped guide, 45° to the side camera
          │          (fixed spot)          │     side camera: low, looks at the corner
          │            SO-101 base         │
          └────────────────────────────────┘
```

- **Inspection corner guide:** a small L-shaped wall (cardboard, 3D print, or Duplo plates)
  taped at the inspection spot, rotated 45° to the side camera. The brick pushed into it always
  ends at the same position and yaw, so the scripted flip and the camera crops always match.
  If the organisers do not allow fixtures, use the "saved release joints" fallback in Step 3.
- **Cameras** (see [docs/hardware.md](../hardware.md)):

| Camera | Mount | Used for |
| --- | --- | --- |
| RealSense D415 (colour, optionally depth) | overhead | ACT input; detection of top/bottom face; brick present?; shape by footprint |
| Side camera (second external, **not connected yet**) | low, facing the corner at 45° | detection of 2 side faces (F+R); ACT input |
| Wrist camera USB2.0_CAM1 | on the follower gripper | ACT input (grasping); detection of the other 2 side faces (B+L) from a fixed "look pose" |

If the side camera does not arrive: use the wrist camera from **two** look poses (one per corner).

## Pipeline: step by step

`HOW` column: **ACT** = learned policy (Physical AI Studio), **Scripted** = fixed joint poses
replayed (no AI), **Anomalib** = defect model, **OpenCV/Python** = classic code.

| # | Stage | What happens | HOW | Technology | Device |
| --- | --- | --- | --- | --- | --- |
| 0 | PERCEIVE: wait | Overhead frame: a brick-coloured blob in the workspace, inspection corner empty → start a cycle | OpenCV | OpenCV, D415 | CPU |
| 1 | EXECUTE: pick → corner | Pick the brick wherever it is and push it into the inspection corner, release, return to home | **ACT** | Physical AI Studio (record + train), Modal (GPU training), OpenVINO PhysicalAI `InferenceModel` (run) | iGPU |
| 1b | check | Brick now inside the corner ROI? If not, rerun step 1 (max 2 retries), else stop and ask for help | OpenCV | OpenCV | CPU |
| 2 | DETECT #1 | Arm goes to the **look pose**. Capture 3 crops: overhead → face *a* (top, or bottom if the brick started upside down), side camera → faces F+R (*b*), wrist camera → faces B+L (*b2*). Score each | Scripted + **Anomalib** | Anomalib model (OpenVINO IR, INT8) | NPU |
| 2b | early exit | Any score above the threshold → skip to step 5 (defective). Saves the flip (about 8 s) | Python | | CPU |
| 3 | EXECUTE: flip | Grasp the brick in the corner, lift, `wrist_roll` 180°, put it back into the corner, release, back to the look pose | **Scripted** | physicalai SO-101 driver | CPU |
| 4 | DETECT #2 | Capture again: overhead → opposite face (*c*), side camera → F+R from below (*d*, optional) | **Anomalib** | same model | NPU |
| 5 | REASON | `defective = max(a, b, b2, c, d) > threshold`; `plate = rules[verdict]` (plus `shape` from footprint if the rules need it). Log scores, verdict, timings | Python rules | `configs/` rules file | CPU |
| 6 | EXECUTE: place | Grasp in the corner, carry to the chosen plate, release, home | **Scripted** | physicalai SO-101 driver | CPU |
| 7 | OPTIMIZE (continuous) | Every cycle logs latency per stage, cycle time, power, temperature | Python | OpenVINO `benchmark_app`, physicalai `InferenceLatencyBenchmark`, RAPL / sensors | n/a |

### Step details

**Step 1: ACT pick.** ACT does not take a language task, so there is one task and one policy:
"put the brick into the inspection corner". The policy ends when the arm is back near the home pose
with the gripper open (joint distance below a tolerance) or after a timeout. Grasp failure check:
if the gripper position closes almost fully, it is holding nothing → retry.

**Step 2 and 4: crops.** Because the corner is a fixed spot, each camera uses a **fixed crop
rectangle** around the corner (set once in a config file), resized to 256×256. No segmentation
is needed for detection. OpenCV is still used for "is a brick there" and for shape by footprint
(`cv2.minAreaRect` aspect ratio: about 1.0 = 2×2, 1.5 = slope brick, 2.0 = 4×2).

**Which faces are covered.** Before the flip: top (or bottom) from overhead, F+R from the side
camera, B+L from the wrist camera. After the flip: the opposite horizontal face from overhead.
That is all 6 faces. **Verify this on day 1 with a brick that has paper labels F/R/B/L/T/D**.
The face mapping depends on the grasp axis.

**Step 3: flip, choose after a hardware test** (untested):

| Option | How | Risk |
| --- | --- | --- |
| F1 (preferred) | Gripper horizontal, jaws on two opposite side faces, lift, `wrist_roll` 180°, lower, release | fingertips must clear the table (Duplo is about 19 mm high) |
| F2 | Top-down grasp, tip the brick 90° onto its side, regrasp, tip 90° again | more moves, slower |
| F3 | Passive fixture: push the brick against a small ramp/wall so it tips over | needs a fixture; less precise |

Fallback without a corner guide ("saved release joints"): store the follower joint angles at the
moment ACT opens the gripper in step 1. The brick is wherever the gripper left it, so returning to
exactly those joints regrasps it. Requires ACT demos that end with the same grasp style the flip
needs (for F1: a horizontal side grasp).

**Step 5: REASON rules** (example, real ones come at kickoff):

```yaml
threshold: 0.5            # Anomalib exported scores are normalised, 0.5 = learned threshold
rules:
  good:   good_plate
  defect: defect_plate
# later, if kickoff rules need it: rules[shape][verdict], rules[colour][verdict], ...
```

Many views raise false alarms: with 5 views at 5% each, about 23% of good bricks get rejected
(1 − 0.95⁵). Tune the threshold on the **cycle level** (max over views), see
[02-data-and-training.md](02-data-and-training.md#evaluation).

### Safety (AGENTS.md rules apply)

- A stop key in the loop window plus a Ctrl+C handler that switches off follower torque in a
  `finally` block, tested before the first closed-loop run.
- Scripted moves are slow (≥ 1.5 s between poses) and checked first at reduced torque.
- Arm workspace clear of hands during loop runs; never leave the loop running unattended.

## Device placement (OPTIMIZE)

| Model / work | Device | Why |
| --- | --- | --- |
| Anomalib generalized model, OpenVINO INT8, static 256×256 input | **NPU** | small, steady load, frees the iGPU; static shape is NPU friendly |
| ACT policy, OpenVINO IR from Studio export | **iGPU** (Arc B390) | biggest model, runs at 30 Hz during step 1 |
| OpenCV, rules, scripted motion, logging | **CPU** | light |

Measure and show (judges want evidence):

- Anomalib latency on CPU vs GPU vs NPU, FP16 vs INT8 (`benchmark_app`).
- ACT latency per inference step on CPU vs GPU (physicalai `InferenceLatencyBenchmark`); try NPU
  as an experiment.
- Cycle time per stage (pick / inspect / flip / place) from the loop log.
- Package power (RAPL) and temperatures during a 10-cycle run, compared with all models on CPU.

## Where the code goes

All in this repo, `src/hackjunction2026/` + `scripts/` (planned, not written yet):

| Module | Purpose |
| --- | --- |
| `cameras.py` | open D415 / side / wrist by serial or name (never `/dev/videoN`, never the HP camera) |
| `crops.py` | fixed crop rectangles per camera, brick-present and footprint checks (OpenCV) |
| `waypoints.py` + `scripts/record_waypoints.py` | teach poses with the leader arm, replay with smooth interpolation |
| `detector.py` | load the Anomalib OpenVINO model with plain `openvino` (no anomalib import at runtime), NPU → GPU → CPU fallback |
| `policy.py` | run the Studio ACT export via `physicalai.inference.InferenceModel` |
| `loop.py` + `scripts/sort.py` | the state machine above, stop key, logging, metrics |
| `scripts/capture_inspection.py` | automatic Anomalib dataset capture (see part 2) |

A prototype of some of this exists outside the repo in `~/lego-sorter/` (single camera, LeRobot
API). It is outdated: reuse ideas, not code.

## Work order (milestones)

Each milestone leaves a working demo behind it.

| M | Deliverable | Depends on | Parallel owner idea |
| --- | --- | --- | --- |
| M0 | Setup done ([03-setup.md](03-setup.md)): both arms answer, 3 cameras mounted, Studio environment, HF token, Modal connected, anomalib-env | | everyone |
| M1 | **Scripted skeleton**: corner guide, waypoints recorded, flip + place work with a hand-placed brick | M0 | arm person |
| M2 | Anomalib dataset captured automatically, model v1 exported, per-view evaluation | M1 | vision person |
| M3 | ACT dataset recorded in Studio, smoke-trained locally, trained on Modal, tested with **Run model** | M0 (can start before M1) | data person |
| M4 | Closed loop: M1 + M2 + M3 in `scripts/sort.py`, stop path tested | M1–M3 | everyone |
| M5 | Measurements, latency/power report, demo screen (camera + heatmap + scores + verdict + timings) | M4 | |
| M6 (stretch) | SmolVLA with language ("pick the red brick") to choose among several bricks in clutter | M4 | |

Fallback demo at any time after M1+M2: hand-place the brick in the corner, everything else
automatic.

## Known risks and open questions

- **Logical defects** (missing or extra stud): one generalized model may miss them, because flat
  plastic is "normal" elsewhere (side walls, slope). Check in the per-view evaluation; if weak,
  split only the top/bottom views per shape (shape from footprint).
- Slope brick upside down may rock → unrepeatable view. Test; if needed inspect its bottom while
  holding it.
- Flip feasibility on the SO-101 (F1 clearance).
- One physical brick per shape: the model may learn that exact brick. Ask the organisers for more
  good samples.
- USB bandwidth: 3 cameras at 640×480@30 on one hub chain; drop the wrist camera resolution if
  frames drop.
- **Ask at kickoff:** which objects, which defects (surface vs. missing parts, colour?), how many
  plates and rules, are fixtures (corner guide) and mirrors allowed, how judges shift objects /
  add clutter, can plates move.

Decisions in this plan (corner guide, single generalized Anomalib model, ACT only for the pick,
scripted inspection) should be recorded as ADRs in `docs/decisions/` once the team confirms them.

# Plan 1/3: Overall pipeline

Status: **plan, nothing here is tested on hardware yet.** Written 2026-11-13.
Other parts: [02-data-and-training.md](02-data-and-training.md) (datasets, training, Modal budget),
[03-setup.md](03-setup.md) (installing and configuring everything).

## Goal

The SO-101 takes a Duplo/Lego brick from anywhere on the table (shifted, rotated, maybe upside
down, maybe among clutter), inspects **the top and all 4 side faces**, decides "perfect" or
"defective", and puts it into the matching bowl. No human chooses what the robot does.

**Scope v1: bottom defects are out of scope.** The bottom is not inspected. If the kickoff defects
can be on the bottom, add the "show the bottom" step (see "Later: bottom inspection").

### What the organisers' slides expect (`Intel_Physical AI Hackathon.pdf`, 15 slides)

- **End goal (slide 9):** **ACT** brings the brick in front of the camera → **Anomalib** checks it
  → **SmolVLA** puts it into the "defective" or the "perfect" bowl. Our plan follows this and adds
  a 180° spin so all 4 side faces are inspected.
- **REASON is VLA reasoning in Physical AI Studio (slides 5, 6):** "The VLA layer interprets the task
  instruction and perception results, determines the appropriate response, and produces the action
  intent". The defect output must reach the robot policy "as a structured signal, label, score, or
  localized region". So the detection result has to **drive a VLA policy**, not only a scripted move.
- **Judging (slide 7)** differs from the rubric in `JUNCTION2026.md` / `docs/challenge.md`; ask the
  organisers which one counts:

| Slides criterion | Pts | Consequence for us |
| --- | --- | --- |
| End-to-end Physical AI solution (one integrated closed loop) | 25 | one loop script, no separate component demos |
| Defect detection with Anomalib (quality, normal vs defective, **how outputs are consumed downstream**) | 20 | score + label + heatmap go into the SmolVLA instruction and onto the demo screen |
| VLA & Physical AI Studio integration (perception + task instruction → robot action) | 20 | SmolVLA does the placing, trained in Studio |
| OpenVINO & Core Ultra optimization (latency, throughput, NPU/GPU/CPU use) | 20 | device placement + measurements below |
| Robotic execution & reliability (localization, accuracy, repeatability, **recovery from incorrect or uncertain detections**) | 10 | retries, uncertainty re-inspection (step 5) |
| Innovation & technical demonstration | 5 | multi-view inspection with a scripted 180° spin |

Our bricks (practice set; the kickoff set may differ, so keep shapes configurable):

| Shape | Count | Notes |
| --- | --- | --- |
| 2×2 brick | 3 (different colours) | colour is not a defect unless kickoff says so |
| 4×2 brick | 1 | footprint ratio about 2:1 |
| 2×2 brick + slope (one stud row longer) | 1 | footprint ratio about 1.5:1 |

## Key design choices

1. **Learned policies for the moves that vary or that the judges score; scripts for
   inspection.** The brick starts anywhere, so the first pick uses a learned **ACT** policy.
   Placing into a bowl uses **SmolVLA** with a language instruction built from the detection
   result (the VLA part the slides require). In between, the brick sits at a **fixed inspection
   spot**, so the inspection motion (the 180° spin) is **scripted**: fixed joint poses recorded
   once with the leader arm and replayed. Repeatable views are what the anomaly model needs.
2. **Inspect on the table, not in the air.** No finger occlusion, no motion blur, fixed background.
   Two placements in the L-shape (before and after a 180° spin around the vertical axis) show the
   top twice and all 4 side faces once.
3. **One generalized Anomalib model** for every view (top, sides) and every shape, trained
   on "good" images only. No routing between models. We split it into per-view or per-shape models
   only if the per-view evaluation shows a weak view (likely missing studs on tops, see
   "Known risks").
4. **REASON = rules → language instruction → SmolVLA.** A small rules table (from a config file,
   so kickoff rules need no code change) turns the structured Anomalib result
   (`max(scores) > threshold`, label, worst view) into a task instruction such as
   "Put the brick into the pink bowl". SmolVLA interprets that instruction together with the camera
   images and produces the actions.

## Physical layout

```text
          ┌──────────── table ────────────┐
          │  (PERFECT bowl)  (DEFECT bowl) │     bowls: fixed position, taped marks
          │                                │
          │   workspace: bricks start      │     overhead: D415 #1 on a desk mount
          │   anywhere here (+ clutter)    │
          │                                │
          │        ◇ L-shape (fixed)       │     brick at 45° to the inspection camera,
          │                                │     pushed into a low L-shaped guide (see below)
          │            SO-101 base         │     inspection camera: D415 #2, low,
          └────────────────────────────────┘     facing the spot (as on slide 9)
```

- **L-shape guide** (cardboard strip, Duplo plate, 3D print), taped at a **constant position**:
  - **Orientation:** the corner of the L points **away** from the inspection camera, rotated so the
    brick sits at 45° to it. The walls touch the two faces the camera is *not* looking at, so they
    never hide a face that is being inspected.
  - **Height: low**, about 5 mm for Duplo (about 3 mm for Lego). A top-down grasp puts the fingers on
    two opposite faces, and one of them always touches a wall, so the fingers must grip the brick
    above the lip.
  - **Why:** the brick pushed into it always ends at the same position and yaw, so the scripted
    spin, the scripted fallback place and the camera crops always match.
  - If the organisers do not allow fixtures, use the "saved release joints" fallback (Step 3).
- **Cameras.** Slide 8 lists one RealSense D415 + one Pro QHD 2K/60 fps webcam + the wrist
  camera. **Our station actually has two D415** (see [docs/hardware.md](../hardware.md), second one
  appeared 2026-11-13 ~13:15), so the plan uses D415 #1 overhead and D415 #2 low at the inspection
  spot. If a 2K webcam turns up instead, it can take either fixed role. Proposed roles:

| Camera | Mount | Used for |
| --- | --- | --- |
| RealSense D415 #1 (serial 344223023464, colour; depth optional) | overhead, desk mount | policy input (ACT, SmolVLA); top face inspection; "brick present?"; shape by footprint |
| RealSense D415 #2 (serial TBD, colour + depth) | low, in front of the L-shape (like slide 9) | inspection of 2 side faces: F+R before the spin, B+L after it (see below); policy input; depth gives brick height/shape |
| Wrist camera USB2.0_CAM1 | on the follower gripper | policy input only (grasping) |

The two fixed cameras can swap roles if the mounts work better the other way round. What matters
is one camera **above** (top face) and one **low at the side** (side faces).

### Face names: what F+R and B+L mean

A brick has 6 faces: top, bottom and 4 vertical side faces. The 4 side faces are named relative
to the **inspection camera** while the brick sits in the L-shape, rotated 45°:

```text
   seen from above, BEFORE the spin            AFTER the 180° spin (vertical axis)

          L-shape walls                               L-shape walls
         ╲            ╱                              ╲            ╱
          ╲   ╱╲     ╱                                ╲   ╱╲     ╱
           L╱    ╲B                                    R╱    ╲F
           ╱      ╲                                    ╱      ╲
           ╲      ╱                                    ╲      ╱
           F╲    ╱R                                    B╲    ╱L
              ╲╱                                          ╲╱
              ▲                                           ▲
     inspection camera (D415 #2, low)            inspection camera (D415 #2, low)
     sees F + R                                   sees B + L
```

- **F** = Front, **R** = Right: the two faces turned towards the inspection camera. The brick is
  at 45°, so the camera looks at a vertical **edge** and sees **two faces at once**.
- **B** = Back (opposite F), **L** = Left (opposite R): the two faces against the L-shape walls.
- **A 180° spin around the vertical axis** swaps F↔B and R↔L. The camera then sees B+L, and the
  top still faces up. Together that is all 4 side faces.
- The overhead camera sees the **top** both times (two views of the same face, which is fine).
- **Not covered in v1:** the bottom.
- The letters are only labels for "the 2 faces this camera sees". They are not features of the
  brick. Check the mapping on day 1 with paper labels on a test brick.
- **Only 180° works, not 90°:** after a 90° spin the 4×2 and the slope brick no longer fit the
  L-shape the same way, and the camera would see F+B or R+L at a different angle.

## Pipeline: step by step

`HOW` column: **ACT** / **SmolVLA** = learned policies (recorded and trained in Physical AI Studio),
**Scripted** = fixed joint poses replayed (no AI), **Anomalib** = defect model,
**OpenCV/Python** = classic code. The slides call the motion stage "ACT"; here it is "EXECUTE" so it
is not confused with the ACT policy.

| # | Stage | What happens | HOW | Technology | Device |
| --- | --- | --- | --- | --- | --- |
| 0 | PERCEIVE: wait | Overhead frame: a brick-coloured blob in the workspace, L-shape empty → start a cycle | OpenCV | OpenCV, overhead D415 #1 | CPU |
| 1 | EXECUTE: pick → L-shape | Pick the brick wherever it is, push it into the L-shape corner, release, return to home | **ACT** | Physical AI Studio (record + train), Modal (GPU training), OpenVINO PhysicalAI `InferenceModel` (run) | iGPU |
| 1b | check | Brick inside the L-shape ROI? If not, rerun step 1 (max 2 retries), else stop and ask for help | OpenCV | OpenCV | CPU |
| 2 | DETECT #1 | Arm is at home (out of both views). Capture 2 crops: overhead → top (*a*), D415 #2 → faces F+R (*b*). Score each | **Anomalib** | Anomalib v2.7.0 model (OpenVINO IR, INT8) | NPU |
| 2b | early exit | Any score clearly above the threshold → skip to step 5 (defective). Saves the spin (about 6 s) | Python | | CPU |
| 3 | EXECUTE: spin 180° | Top-down grasp in the L-shape, lift about 3 cm, `wrist_roll` +180°, lower back into the L-shape (push gently into the corner), release, back to home | **Scripted** | physicalai SO-101 driver | CPU |
| 4 | DETECT #2 | Capture again: overhead → top again (*c*), D415 #2 → faces B+L (*d*) | **Anomalib** | same model | NPU |
| 5 | REASON | Structured result `{label, max_score, worst_view, heatmap}` → rules table → **task instruction**, e.g. "Put the brick into the pink bowl". Score inside the uncertainty band → re-capture the worst view once (recovery from uncertain detections) | Python rules + **SmolVLA** reads the instruction | `configs/` rules file | CPU |
| 6 | EXECUTE: place | SmolVLA executes the instruction: picks the brick from the L-shape, puts it into the chosen bowl, returns home | **SmolVLA** (fallback: scripted waypoints) | Physical AI Studio (record + train on Modal), `InferenceModel` | iGPU |
| 7 | OPTIMIZE (continuous) | Every cycle logs latency per stage, cycle time, power, temperature | Python | OpenVINO `benchmark_app`, physicalai `InferenceLatencyBenchmark`, RAPL / sensors | n/a |

### Step details

**Step 1: ACT pick.** ACT does not take a language task, so there is one task and one policy:
"put the brick into the inspection spot". The policy ends when the arm is back near the home pose
with the gripper open (joint distance below a tolerance) or after a timeout. Grasp failure check:
if the gripper position closes almost fully, it is holding nothing → retry.

**Step 2 and 4: crops.** Because the inspection spot is fixed, each camera uses a **fixed crop
rectangle** around it (set once in a config file), resized to 256×256. No segmentation is needed
for detection. OpenCV is still used for "is a brick there" and for shape by footprint
(`cv2.minAreaRect` aspect ratio: about 1.0 = 2×2, 1.5 = slope brick, 2.0 = 4×2). The D415
depth can confirm shape by height profile.

**Which faces are covered:** see "Face names" above. **Verify on day 1 with paper labels
F/R/B/L/T on a test brick.**

**Step 3: 180° spin** (untested). Why the SO-101 can do it: the spin is only a `wrist_roll`
move with the gripper pointing down, and the roll range on arm A is about 334°
(limits 189–3992 steps, `docs/hardware.md`). Check the follower's limits once arm B answers.

1. `spot_above` → `spot_grasp_open` → `spot_grasp_closed`: top-down, fingers across the brick's
   **narrow** side (32 mm on Duplo for every shape), above the L-shape lip.
2. `spot_lift`: about 3 cm up, enough to clear the lip.
3. `spot_lift_spun`: same pose with `wrist_roll` + 180° (2048 steps). Slow (≥ 1.5 s).
4. `spot_place_spun`: lower, about 2–3 mm from the corner, then a small push into the corner.
5. Open, `spot_above`, `home`.

All 180°-symmetric footprints (2×2, 4×2, slope brick) fit the L-shape again after the spin. Check:
the brick must not slip in the gripper during the roll, and the gripper must not touch the lip.

Fallback without an L-shape ("saved release joints"): store the follower joint angles at the
moment ACT opens the gripper in step 1. The brick is wherever the gripper left it, so returning to
exactly those joints regrasps it. The spin and place-back then reuse those joints instead of the
fixed `spot_*` poses.

**Step 5: REASON** (example, real rules come at kickoff):

```yaml
threshold: 0.5            # Anomalib exported scores are normalised, 0.5 = learned threshold
uncertain_band: [0.4, 0.6]  # re-capture the worst view once if max score falls in here
instructions:
  good:   "Put the brick into the blue bowl"
  defect: "Put the brick into the pink bowl"
# later, if kickoff rules need it: instructions[shape][verdict], instructions[colour][verdict], ...
```

The instruction strings must be **exactly** the task strings used when recording the SmolVLA
episodes. Show the structured result (score, label, heatmap, instruction) on the demo screen. This
is how the judges see that "defect outputs are consumed downstream".

Many views raise false alarms: with 4 views at 5% each, about 19% of good bricks get rejected
(1 − 0.95⁴). Tune the threshold on the **cycle level** (max over views), see
[02-data-and-training.md](02-data-and-training.md#d-evaluation).

**Step 6: SmolVLA place.** One language-conditioned policy for both bowls. The start is always the
L-shape (constant position), so the policy only has to learn "which bowl", which is easy for a
VLA. For the same reason a **scripted place** is trivial: fixed L-shape → fixed bowl waypoints.
Keep it as a fallback switch (`--place scripted`) in case SmolVLA is not reliable on demo day.

**Later: bottom inspection (not in v1).** If bottom defects matter: top-down grasp in the L-shape,
lift about 10 cm, tilt the wrist about 90° so the gripper points horizontally away from the robot.
The bottom then faces outward, and a camera on that line (D415 #2 placed beyond the L-shape,
looking back at the robot) sees it. Pitch only, all in the air, no upside-down placing.

### Safety (AGENTS.md rules apply)

- A stop key in the loop window plus a Ctrl+C handler that switches off follower torque in a
  `finally` block, tested before the first closed-loop run. Also use the OpenVINO Physical AI
  runtime's built-in action clamping / velocity limits / emergency stop (slide 13).
- Scripted moves are slow (≥ 1.5 s between poses) and checked first at reduced torque.
- Arm workspace clear of hands during loop runs; never leave the loop running unattended. Follow
  the printed robot safety guidelines provided at the station (slide 8).

## Device placement (OPTIMIZE)

| Model / work | Device | Why |
| --- | --- | --- |
| Anomalib generalized model, OpenVINO INT8, static 256×256 input | **NPU** | small, steady load, frees the iGPU; static shape is NPU friendly |
| ACT policy (step 1), OpenVINO IR from Studio export | **iGPU** (Arc B390) | runs at 30 Hz during the pick |
| SmolVLA policy (step 6), OpenVINO IR from Studio export | **iGPU** | largest model; never runs at the same time as ACT |
| OpenCV, rules, scripted motion, logging | **CPU** | light |

Measure and show (judges want evidence):

- Anomalib latency on CPU vs GPU vs NPU, FP16 vs INT8 (`benchmark_app`).
- ACT and SmolVLA latency per inference step on CPU vs GPU (physicalai
  `InferenceLatencyBenchmark`); try the NPU as an experiment.
- Cycle time per stage (pick / inspect / spin / place) from the loop log.
- Package power (RAPL) and temperatures during a 10-cycle run, compared with all models on CPU.

## Where the code goes

All in this repo, `src/hackjunction2026/` + `scripts/` (planned, not written yet):

| Module | Purpose |
| --- | --- |
| `cameras.py` | open both D415s (by serial) and the wrist camera (by name) (never `/dev/videoN`, never the HP camera) |
| `crops.py` | fixed crop rectangles per camera, brick-present and footprint checks (OpenCV) |
| `waypoints.py` + `scripts/record_waypoints.py` | teach poses with the leader arm, replay with smooth interpolation |
| `detector.py` | load the Anomalib OpenVINO model with plain `openvino` (no anomalib import at runtime), NPU → GPU → CPU fallback |
| `policy.py` | run the Studio ACT and SmolVLA exports via `physicalai.inference.InferenceModel` |
| `reason.py` | structured detection result → rules → instruction string |
| `loop.py` + `scripts/sort.py` | the state machine above, stop key, logging, metrics, demo screen |
| `scripts/capture_inspection.py` | automatic Anomalib dataset capture (see part 2) |

A prototype of some of this exists outside the repo in `~/lego-sorter/` (single camera, LeRobot
API). It is outdated: reuse ideas, not code.

## Work order (milestones)

Each milestone leaves a working demo behind it.

| M | Deliverable | Depends on | Parallel owner idea |
| --- | --- | --- | --- |
| M0 | Setup done ([03-setup.md](03-setup.md)): both arms answer, 3 cameras mounted, Studio environment, HF token, Modal connected, anomalib-env | | everyone |
| M1 | **Scripted skeleton**: L-shape taped, waypoints recorded, 180° spin + scripted place work with a hand-placed brick | M0 | arm person |
| M2 | Anomalib dataset captured automatically, model v1 exported, per-view evaluation | M1 | vision person |
| M3 | ACT pick dataset recorded in Studio, smoke-trained locally, trained on Modal, tested with **Run model** | M0 (can start before M1) | data person |
| M4 | Closed loop with scripted place: M1 + M2 + M3 in `scripts/sort.py`, stop path tested | M1–M3 | everyone |
| M5 | SmolVLA place dataset (both bowls) recorded, trained on Modal, replaces the scripted place | M4 | data person |
| M6 | Measurements, latency/power report, demo screen (camera + heatmap + scores + instruction + timings) | M4 | |
| M7 (stretch) | SmolVLA also chooses among several bricks in clutter ("pick the red brick") | M5 | |

Fallback demo at any time after M1+M2: hand-place the brick at the inspection spot, everything
else automatic.

## Known risks and open questions

- **Logical defects** (missing or extra stud): one generalized model may miss them, because flat
  plastic is "normal" elsewhere (side walls, slope). Check in the per-view evaluation; if weak,
  split only the top view per shape (shape from footprint).
- **Brick that starts upside down:** ACT puts it into the L-shape as it lies, so the overhead
  camera sees the bottom and the top is never inspected. v1: treat that view like any other (the
  good images must include bottoms seen from above) and accept the gap; or ask whether the judges
  ever place bricks upside down.
- Spin feasibility: brick slipping in the gripper during the roll; fingers touching the lip.
- One physical brick per shape: the model may learn that exact brick. Ask the organisers for more
  good samples.
- USB bandwidth: 3 cameras at 640×480@30 on one hub chain; drop the wrist camera resolution if
  frames drop.
- **Two different rubrics**: slides (slide 7) vs `JUNCTION2026.md`. Ask which one the judges use.
- **Ask at kickoff:** which objects, which defects (surface vs. missing parts, colour?), how many
  bowls and rules, are fixtures (inspection guide) allowed, how judges shift objects / add clutter,
  can bowls move.

Decisions in this plan (inspection guide, single generalized Anomalib model, ACT for the pick,
SmolVLA for the place, scripted inspection) should be recorded as ADRs in `docs/decisions/` once
the team confirms them.

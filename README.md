# HackJunction2026

Junction 2026, **Intel Physical AI Challenge**: *Stop Simulating, Start Moving: See It. Detect It. Act on It.*
An SO-101 arm inspects tabletop objects, detects defective ones and sorts them, all on an Intel Core Ultra AI PC.

```text
PERCEIVE ──► DETECT ──► REASON ──► EXECUTE ──► OPTIMIZE
cameras      Anomalib    VLA policy  SO-101      OpenVINO on CPU / iGPU / NPU
```

Agents and humans: read [AGENTS.md](AGENTS.md) first.

## Repository layout

| Path | Contents |
| --- | --- |
| `AGENTS.md` | Rules for agents and humans: record keeping, robot safety, environment, git |
| `docs/challenge.md` | Official guide link and our progress checklist |
| `docs/plan/` | Project plan: pipeline, datasets and training (Modal budget), setup |
| `docs/hardware.md` | Hardware inventory and facts (ports, servos, cameras, power) |
| `docs/decisions/` | Architecture decision records (ADRs) |
| `docs/troubleshooting.md` | Errors we hit, causes, fixes |
| `docs/log/` | Daily engineering log |
| `configs/hardware.yaml` | Stable device addresses used by code |
| `configs/inspection.yaml` | Inspection cameras (D415 serials), exposure, crop and capture settings |
| `src/hackjunction2026/` | Our Python package: Feetech bus helper; `defect/` = Anomalib capture, cropping, training; `kinematics/` = FK, gripper-down IK, path check; `table_map.py` = bricks in the wrist camera and pixel → table homography |
| `scripts/` | Command-line tools (`so101_probe.py`, `so101_move.py`, `so101_goto.py`, `arm_fk.py`, `pick_above.py`, `capture_inspection.sh`, `anomalib.sh`) |
| `tests/` | pytest (`uv run --extra table pytest`) |
| `configs/table_homography.yaml` | Wrist camera pixel → table XY in the `observe` pose (written by `pick_above.py calib-fit`) |
| `third_party/` | Upstream checkouts, **not committed** |

## Setup (laptop: Ubuntu 26.04, native, no Docker)

1. System packages (needs sudo):

   ```bash
   sudo apt-get update && sudo apt-get install -y ffmpeg libgl1 libglib2.0-0 libusb-1.0-0 \
     libusb-1.0-0-dev libclang-dev pkg-config build-essential g++ git \
     libze1 libze-intel-gpu1 intel-opencl-icd
   sudo usermod -aG dialout $USER   # serial ports; log out and back in
   ```

2. Tools: [uv](https://docs.astral.sh/uv/) and Node.js >= 24.2 with npm >= 11.14 (we use Node 26 via nvm).

3. Physical AI Studio. Our private copy has upstream `junction2026` plus our patches
   (RealSense RGB-D recording, camera fixes, Space-hotkey snapshots; see its `HACKJUNCTION.md`):

   ```bash
   git clone -b hj/state-snapshots git@github.com:chessnok/physical-ai-studio-hj.git \
     third_party/physical-ai-studio
   ```

   Plain upstream instead: `git clone -b junction2026 https://github.com/open-edge-platform/physical-ai-studio.git third_party/physical-ai-studio`.
   The robot laptop runs `hj/state-snapshots` (see the deployment log in `docs/log/2026-11-13.md`;
   ADR 0004 records the RGB-D patch decision).

4. Run Studio in two terminals, then open http://localhost:3000:

   ```bash
   scripts/studio_backend.sh   # Terminal 1: backend, http://localhost:7860 (iGPU + NPU visible)
   scripts/studio_ui.sh        # Terminal 2: UI, http://localhost:3000
   ```

   Stop both with `scripts/studio_stop.sh` (it also kills orphaned workers, T-012).

   The scripts run the guide's commands (`uv run --extra xpu physicalai-studio serve`,
   `npm run start`) and add `ZE_ENABLE_ALT_DRIVERS` so OpenVINO also sees the NPU (T-008).

5. Project tools:

   ```bash
   uv run scripts/so101_probe.py            # read-only: which servos answer on which bus
   ```

## Anomalib dataset capture (untested on the real cameras)

Photos of bricks in the L-shape for the defect model (plan: `docs/plan/02-data-and-training.md`,
section A). Runs with Studio's backend environment without modifying it (`uv run --no-sync`).
Stop recording / Run model / camera previews in Studio first: one program per camera.

```bash
scripts/capture_inspection.sh --list-cameras      # D415 serials + USB type → configs/inspection.yaml
scripts/capture_inspection.sh --set-crops         # draw the crop box per camera → configs/inspection_crops.yaml
scripts/capture_inspection.sh --brick 2x2red      # capture session: SPACE = save, see the keys in the window
scripts/capture_inspection.sh --fake --once --brick test   # self-test without cameras
```

Without Studio (e.g. a teammate's machine): `uv sync --extra capture`, then
`uv run --extra capture python -m hackjunction2026.defect.capture.capture_inspection ...`.
`--webcam 0 --output-dir /tmp/x --no-check` tries the window with an ordinary webcam
(**test only, never on the station laptop**: AGENTS.md forbids its built-in camera).

One cycle: brick in the L-shape → SPACE (`top_pre` + `side_FR`) → turn the brick 180° → SPACE
(`top_post` + `side_BL`). Output: `captures/anomalib/raw/<session>/{good,defect}/*.png` +
`manifest.jsonl` (git-ignored). Reuse `--session <name>` to continue a session.

## Anomalib training (tested on mock data only)

```bash
uv sync --extra capture --extra anomalib                    # once; ~8 GB (PyTorch XPU), ADR 0009
scripts/anomalib.sh cropping.make_dataset --name v1         # raw captures → cropped 256×256 dataset
scripts/anomalib.sh training.train_padim --dataset v1       # scenario 1 (baseline, ~1 min)
scripts/anomalib.sh training.train_patchcore --dataset v1   # scenario 2 (--coreset 0.05 --tag c005)
scripts/anomalib.sh training.train_efficientad --dataset v1 # scenario 3 (downloads ImageNette)
scripts/anomalib.sh training.evaluate --device NPU          # latest run, every export (FP16, INT8)
scripts/anomalib.sh training.compare                        # table of all runs → outputs/anomalib/compare.md
scripts/anomalib.sh capture.mock_session --output-dir /tmp/mock   # synthetic session to test the chain
```

Code: `src/hackjunction2026/defect/{capture,cropping,training}/`. Runs: `outputs/anomalib/runs/<run>/`
(`run.json`, `eval.json`, `errors/`); models: `exports/anomalib/<run>/{fp16,int8_ptq}/weights/openvino/model.xml`.
Training uses the Intel GPU (XPU) when available, else CPU. Exports are static batch 1.

## Anomalib on brick crops from the three cameras (ADR 0006, ADR 0008)

Bricks are found by colour on the white sheet in the `top`, `floor` and `arm` frames, cut as a square of
1.4x the brick (Anomalib; the VLM uses 2.4x, ADR 0008), background whitened, 256x256. One crop = one
model input; a brick is defective if ANY crop scores above the threshold. Real defects are the state
snapshots flagged `defective`; Gemini (`gemini-4-argon`) labels each crop "mark visible?" so only crops
that show the mark count as `test/defect`. Results and the chosen threshold: ADR 0006.

**Environment:** Anomalib lives in its own env `anomalib-env/` (anomalib 2.7.0, torch 2.15.0+xpu,
openvino 2026.4.1, ~7.9 GB) so it never locks the project `.venv`. Every anomalib command needs
`UV_PROJECT_ENVIRONMENT=$PWD/anomalib-env`; never run `uv sync` / `uv run --extra anomalib` without
it (that starts the 8 GB install into `.venv` again and blocks everyone's `uv run`).

```bash
export UV_PROJECT_ENVIRONMENT=$PWD/anomalib-env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
STUDIO_PY=third_party/physical-ai-studio/application/backend/.venv/bin/python   # has av + pyarrow
# 1. crops: dataset episodes (train/test split by episode) + snapshots (half of the good ones in train,
#    hour-stratified time blocks; defective ones in test only) + a SYNTHETIC-stroke eval set (~5 min)
nice -n 19 $STUDIO_PY scripts/brick_crops.py --name bricks_z14t --zoom 1.4 --max-train 400 --max-test-episodes 400 \
    --dataset 431a71f7 --snapshots 0fd76ff2 --snapshot-train-frac 0.5 --synthetic
# 2. per-crop labels with Gemini (AI_STUDIO_API_KEY in .env; only masked brick crops are sent; cached per crop)
uv run --no-project --script scripts/label_crops_gemini.py --dataset bricks_z14t --model gemini-4-argon --apply
# 3. train + evaluate on CPU (T-015; ~12 min, T-019), then the synthetic set and the other devices
nice -n 19 scripts/anomalib.sh training.train_patchcore --dataset bricks_z14t --accelerator cpu --num-workers 2 --coreset 0.004
nice -n 19 scripts/anomalib.sh training.evaluate --run <run_id> --dataset bricks_z14t-synth
nice -n 19 scripts/anomalib.sh training.evaluate --run <run_id> --device NPU
scripts/anomalib.sh training.brick_report --runs <run_id> [...]     # → outputs/anomalib/brick_report.md
# 4. run it: one JSON line per scene (OR over crops), heatmaps optional; model/threshold/device from
#    configs/anomalib_deploy.yaml (openvino + opencv only: anomalib-env or Studio's env)
anomalib-env/bin/python scripts/brick_anomaly.py --heatmaps --snapshot 20261114-051059-914
anomalib-env/bin/python scripts/brick_anomaly.py --run <run_id> --device NPU --snapshot <id>   # try a run
anomalib-env/bin/python scripts/brick_anomaly.py --benchmark CPU,GPU,NPU --bench-crops captures/anomalib/dataset/bricks_z14t/test/good
```

Retrain in one command (e.g. on inspection-pose recordings): `scripts/brick_retrain.sh <name> patchcore --
--dataset <id> --snapshots 0fd76ff2 [--pose inspect]`. Labels: Gemini per crop, manual spot checks in
`configs/defect_mark_labels.csv`.

## Moving the follower to saved poses (`scripts/so101_goto.py`)

Simplest possible move: `delta = target - current`, every joint moves linearly so all arrive
together. Checks: target inside the calibrated range; stop-and-hold if the arm cannot follow
(blocked) or does not reach the target, or on Ctrl+C. Dry run by default; `--go` moves.
Close the environment in Studio first (the arm's port must be free). Run with Studio's Python:

```bash
PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
$PY scripts/so101_goto.py where                      # both arms' poses, read-only
$PY scripts/so101_goto.py save pick_1 --from leader  # move the leader by hand, save its pose
$PY scripts/so101_goto.py goto pick_1                # show from / to / delta / time
$PY scripts/so101_goto.py goto pick_1 --go           # move (asks to confirm), default 20 units/s
```

## Arm positions from recorded joints (`scripts/arm_fk.py`, offline)

Forward kinematics of the follower from Studio's normalized joint values (no robot connection).
Conventions and validation: [docs/perception/B-arm-kinematics.md](docs/perception/B-arm-kinematics.md).

```bash
uv run scripts/arm_fk.py pose 0 -99.2 62.7 77.6 -56.9 13.5     # keypoints (mm) for one joint vector
uv run --with pyarrow scripts/arm_fk.py annotate 431a71f7       # every frame -> captures/perception/B/arm_<id>.parquet
```

### Hover above a brick (`scripts/pick_above.py`, hardware procedure untested)

From the saved pose `observe` the wrist camera sees the white sheet. `pick_above.py` finds the
coloured bricks on the sheet, maps image pixels to table XY with a homography that is valid only in
that pose, and computes the joint target that puts the gripper straight down above a brick, with
the centre of the open jaws 60 mm above the table. It checks that the straight move from `observe`
keeps every arm point 30 mm above the table. It never moves the arm: it prints the
`so101_goto.py goto "..."` command for you. A grasp policy takes over after that.
Why it works this way: [ADR 0007](docs/decisions/0007-observe-pose-homography-ik.md).

Tools run from the repo root with the `table` extra (headless OpenCV + PyYAML):

```bash
uv run --extra table scripts/pick_above.py --help
uv run --extra table pytest                      # offline tests (synthetic scenes + the sample image)
```

**Calibration** (once; again after changing `observe`, re-mounting the wrist camera, moving the
robot base on the table or recalibrating the arm):

1. Lay out 5-6 bricks of **distinct colours** (red, orange, yellow, lime, green, blue; not white, grey,
   black or purple) spread over the part of the sheet the arm can reach (see `reach` below), not
   touching each other.
2. In Studio teleop, bring the follower's gripper over each brick as if to grasp it: gripper about
   vertical, jaws open a little wider than the brick and centred on it, fingertips at the brick's
   mid-height. Press **Space** and enter the comment `calib <colour>` (e.g. `calib red`). Redo a
   brick if needed: the latest snapshot per colour wins. Do not move the bricks after this.
3. Check the points: `uv run --extra table scripts/pick_above.py calib-points --since <YYYYMMDD-HHMMSS>`
   (the time you started, laptop clock; it skips older `calib` snapshots). It prints each brick's
   table XY and warns about a tilted gripper or fingertips far above the table.
4. Close the environment in Studio (camera and follower port must be free).
5. Move to the observe pose (hands clear; asks for confirmation):
   `third_party/physical-ai-studio/application/backend/.venv/bin/python scripts/so101_goto.py goto observe --go`
6. Fit: `uv run --extra table scripts/pick_above.py calib-fit --capture --since <YYYYMMDD-HHMMSS>`.
   It reads the follower pose (read-only), grabs a wrist camera frame, matches bricks to calib
   points by colour, prints the residual per brick (fit and leave-one-out, mm) and writes
   `configs/table_homography.yaml`. It refuses to save above 15 mm leave-one-out RMS. Look at the
   annotated image under `captures/pick_above/`. If a colour is named differently by the detector,
   add `--alias calibname=detectedname`. Commit the YAML.

**End-to-end test** (one brick):

1. Studio environment closed, arm in `observe` (step 5 above). Put one brick on the sheet.
2. `uv run --extra table scripts/pick_above.py plan --capture` (`--brick <colour>` if there are
   several, `--height <mm>` for another hover height). It prints the brick's table XY, the target
   joints, the FK check (jaw centre position, tilt), the path check and the goto command.
   It refuses when the arm is not in the calibrated pose (more than 1.5 units off), the brick is out
   of reach, a joint would leave its range, or the move would bring an arm point within 30 mm of
   the table.
3. Run the printed `... so101_goto.py goto --go -- "<6 values>"` (hands clear; the `--` is needed
   when the first value is negative, T-017), check that the gripper
   hangs straight down above the brick with the brick between the jaws, and note the offset in mm.
4. Go back with `... so101_goto.py goto observe --go` before the next `plan`: the path check assumes
   the move starts in `observe`.

Other commands: `locate --image <png or snapshot dir>` (bricks with table XY, offline),
`plan --image <snapshot dir>` (offline dry run on a saved snapshot), `reach [--height 60]` (reachable
table area with the gripper straight down, plot in `captures/pick_above/reach_h60.png`).
`--capture` refuses when Studio holds the camera or the port; it resolves the wrist camera by name
under `/dev/v4l/by-id` and never uses the laptop's HP camera.

## Status

See [docs/challenge.md](docs/challenge.md) and the latest file in [docs/log/](docs/log/).

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
| `src/hackjunction2026/` | Our Python package: Feetech bus helper; `defect/` = Anomalib dataset capture (later: training, detector) |
| `scripts/` | Command-line tools (`so101_probe.py`, `so101_move.py`) |
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
   The robot laptop still runs plain `junction2026` until the patches are accepted (ADR 0004).

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
`uv run --extra capture python -m hackjunction2026.defect.capture_inspection ...`.
`--webcam 0 --output-dir /tmp/x --no-check` tries the window with an ordinary webcam
(**test only, never on the station laptop**: AGENTS.md forbids its built-in camera).

One cycle: brick in the L-shape → SPACE (`top_pre` + `side_FR`) → turn the brick 180° → SPACE
(`top_post` + `side_BL`). Output: `captures/anomalib/raw/<session>/{good,defect}/*.png` +
`manifest.jsonl` (git-ignored). Reuse `--session <name>` to continue a session.

### Moving the follower to saved poses (`scripts/so101_goto.py`)

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

## Perception: table frame, brick and gripper positions (offline, recorded data)

Details, conventions and accuracy: [docs/perception/A-object-localization.md](docs/perception/A-object-localization.md).
Outputs go to `captures/perception/A/` (gitignored). Needs numpy, opencv-python-headless, av,
pyarrow, pandas, scipy (any venv; Studio's backend venv has all but scipy).

```bash
export PYTHONPATH=src
nice -n 19 $PY scripts/perception_table_frame.py 431a71f7                 # table frame JSON
nice -n 19 ionice -c 3 $PY scripts/perception_annotate.py 431a71f7 --workers 4   # objects + gripper parquet
$PY scripts/perception_bowls.py 431a71f7
$PY scripts/perception_register_robot.py 431a71f7                         # robot (FK) -> table
$PY scripts/perception_eval.py --methods cv,gripper                        # accuracy vs hand-made GT
```

## Status

See [docs/challenge.md](docs/challenge.md) and the latest file in [docs/log/](docs/log/).

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
| `src/hackjunction2026/` | Our Python package: Feetech bus helper; `defect/` = Anomalib capture, cropping, training |
| `scripts/` | Command-line tools (`so101_probe.py`, `so101_move.py`, `arm_fk.py`, `capture_inspection.sh`, `anomalib.sh`) |
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

## Status

See [docs/challenge.md](docs/challenge.md) and the latest file in [docs/log/](docs/log/).

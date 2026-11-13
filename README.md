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
| `src/hackjunction2026/` | Our Python package (Feetech bus helper, later: pipeline code) |
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

3. Physical AI Studio, `junction2026` branch:

   ```bash
   git clone -b junction2026 https://github.com/open-edge-platform/physical-ai-studio.git \
     third_party/physical-ai-studio
   ```

4. Run Studio in two terminals, then open http://localhost:3000:

   ```bash
   scripts/studio_backend.sh   # Terminal 1: backend, http://localhost:7860 (iGPU + NPU visible)
   scripts/studio_ui.sh        # Terminal 2: UI, http://localhost:3000
   ```

   The scripts run the guide's commands (`uv run --extra xpu physicalai-studio serve`,
   `npm run start`) and add `ZE_ENABLE_ALT_DRIVERS` so OpenVINO also sees the NPU (T-008).

5. Project tools:

   ```bash
   uv run scripts/so101_probe.py            # read-only: which servos answer on which bus
   ```

## Status

See [docs/challenge.md](docs/challenge.md) and the latest file in [docs/log/](docs/log/).

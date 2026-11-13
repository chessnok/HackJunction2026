# Plan 3/3: Setup

Status: written 2026-11-13. "Done" items come from [docs/log/2026-11-13.md](../log/2026-11-13.md);
everything else is **untested**. Pipeline: [01-pipeline.md](01-pipeline.md), data and training:
[02-data-and-training.md](02-data-and-training.md). Base install steps are in the
[README](../../README.md#setup-laptop-ubuntu-2604-native-no-docker). Hardware facts are in
[docs/hardware.md](../hardware.md).

Per AGENTS.md: agents do not run `sudo` or `curl | sh`; a human runs those (`! <command>` in
Claude Code). Never `pip install` into Studio's environment.

## Three Python environments

| Env | Contents | Used for |
| --- | --- | --- |
| **Studio backend** (`third_party/physical-ai-studio/application/backend`, `uv sync --extra xpu`) | physicalai-studio 0.3.0, physicalai 0.4.1.dev5, physicalai-train 0.3.0, so101 plugin, torch 2.11.0+xpu, openvino 2026.4.1 | Studio UI/backend: robots, recording, training, Run model |
| **anomalib-env** | `anomalib[openvino,xpu]` | training and exporting the anomaly model only |
| **Runtime for our loop** | see section 5 | `scripts/sort.py`: cameras, ACT via `InferenceModel`, anomaly model via plain `openvino` |

Anomalib and Studio pin different PyTorch builds, which is why they are separate (guide, Step 4).

## 1. Hardware

- [ ] **Arm B (follower) answers on the bus**, see T-005 in `docs/troubleshooting.md` (power
      supply voltage, cable, board). Then confirm leader/follower roles visually and update
      `docs/hardware.md` + `configs/hardware.yaml`.
- [ ] **Side camera** (second external camera) obtained and connected; record its ID in
      `docs/hardware.md`.
- [ ] Cameras mounted **firmly** (overhead D415, side low at 45° to the corner, wrist). Mark
      positions with tape.
- [ ] Inspection **corner guide** taped down (ask the organisers if fixtures are allowed).
- [ ] GOOD / DEFECT plates taped down.
- [ ] Plain background behind the corner for the side camera; lighting as stable as possible.
- [ ] USB: if frames drop with 3 cameras, spread them over different laptop ports or lower the
      wrist camera resolution.

## 2. Physical AI Studio

Already done (2026-11-13): native backend + UI running via `scripts/studio_backend.sh` and
`scripts/studio_ui.sh`; torch sees the Arc iGPU, OpenVINO sees CPU/GPU/NPU (T-008).

Remaining (Studio docs `03-getting-started.md`, `04-environment-setup.md`):

- [ ] **Robots:** add the SO-101 follower and SO-101 leader (use the `/dev/serial/by-id/...`
      ports from `configs/hardware.yaml`). Arm A is already calibrated: use "Skip verification"
      only if the on-screen arm matches the real one; otherwise run the wizard's calibration.
- [ ] **Cameras:** add D415 (colour), side, wrist at 640×480 @ 30 FPS. Check each preview. Never
      add the built-in HP camera.
- [ ] **Environment:** follower + leader + the 3 cameras. Use the same environment for recording
      and running.
- [ ] **Hugging Face token** (read-only): Settings → General → Hugging Face. Needed for SmolVLA /
      Pi0.5 weights. For the native backend, `HF_TOKEN` in `.env` has no effect. Never commit it.
- [ ] Teleop test: the follower mirrors the leader smoothly; record and replay one test episode.

## 3. Modal (GPU training, $100 credit per participant)

Do this early: the first image build takes several minutes.

1. [ ] Create an account at https://modal.com and redeem the credit as the organisers instruct.
2. [ ] Give **the backend user** Modal credentials (browser login alone is not enough). Run as the
       user that runs `scripts/studio_backend.sh`:

   ```bash
   uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal setup
   ```

3. [ ] Studio → Settings → Training Targets → **Modal Provider** → name it → **Deploy and connect**.
       Wait for the health status to become healthy.
4. [ ] Check the Modal dashboard: the app and Volume exist, no GPU is running.

Rules: choose the GPU per job (A10G default), cancel stuck jobs, and import models into Studio
before deleting a target (deleting it deletes its Volume). Budget is in
[02-data-and-training.md](02-data-and-training.md#3-modal-budget-100-per-participant).

## 4. Anomalib environment

From the repo root (`*-env/` is already git-ignored):

```bash
uv venv anomalib-env --python 3.12
source anomalib-env/bin/activate
UV_HTTP_TIMEOUT=600 uv pip install "anomalib[openvino,xpu]"     # big download on venue Wi-Fi (T-010)
```

Verify:

```bash
python -c "import torch; print(torch.xpu.is_available())"                 # True
ZE_ENABLE_ALT_DRIVERS=/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1,/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1 \
  python -c "import openvino as ov; print(ov.Core().available_devices)"   # ['CPU', 'GPU', 'NPU']
anomalib --help
```

`ZE_ENABLE_ALT_DRIVERS` must list **both** drivers or torch XPU breaks (T-008). Training on XPU:
`Engine(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())`.

## 5. Runtime environment for our loop

The loop needs: `physicalai` (SO-101 driver + `InferenceModel` for the Studio ACT export),
`openvino` (runs the exported Anomalib model directly; the export contains pre/post-processing,
**to verify**), `opencv`, `pyyaml`.

**Option A (try first, no new installs):** run our scripts with Studio's backend environment,
read-only:

```bash
uv run --project third_party/physical-ai-studio/application/backend python scripts/sort.py
```

It already has the exact `physicalai` / so101-plugin / OpenVINO versions that produced the export.
Check that `cv2` and `yaml` import. Do **not** add packages to it (ADR 0003).

**Option B:** add `physicalai[so101-plugin]`, `openvino`, `opencv-python`, `pyyaml` to this repo's
`pyproject.toml`. Pin `physicalai` to the same version or commit as Studio's backend
(0.4.1.dev5+g8381d43) so the export loads.

Either way:

- Set the same `ZE_ENABLE_ALT_DRIVERS` as `scripts/studio_backend.sh` so the NPU is visible.
- **Only one process can own the arms and cameras.** Stop recording / Run model in Studio before
  starting `scripts/sort.py`, and the other way round.

## 6. Measurement tools (OPTIMIZE)

| Tool | For | Install |
| --- | --- | --- |
| `benchmark_app` | Anomalib and ACT model latency per device | ships with the `openvino` pip package |
| `physicalai.benchmark.performance.InferenceLatencyBenchmark` | ACT policy latency | in physicalai |
| RAPL `/sys/class/powercap/intel-rapl:0/energy_uj` | package power (read twice, divide by time) | built in (may need read permission) |
| `sensors` | temperatures | `sudo apt install lm-sensors` (human) |
| `intel_gpu_top` | iGPU load | `sudo apt install intel-gpu-tools` (human) |
| NPU busy time `/sys/class/accel/accel0/device/npu_busy_time_us` | NPU utilisation | built in (verify path) |

```bash
benchmark_app -m exports/anomalib/model.xml -d NPU -hint latency -t 15
```

## 7. Ready-to-start checklist

- [ ] Both arms answer; teleop works in Studio
- [ ] 3 cameras show stable 30 FPS previews in one Studio environment
- [ ] Corner guide and plates fixed; labelled-brick face test done (all 6 faces covered)
- [ ] HF token set; Modal target healthy
- [ ] anomalib-env sees XPU; OpenVINO sees NPU
- [ ] Runtime env decided (Option A or B), stop path implemented

Then start with milestone M1 in [01-pipeline.md](01-pipeline.md#work-order-milestones).

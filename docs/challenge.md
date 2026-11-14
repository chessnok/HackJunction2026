# Challenge: guide and progress

Official walkthrough (read it, it is the spec):
<https://github.com/open-edge-platform/physical-ai-studio/blob/junction2026/JUNCTION2026.md>

Task: build a closed-loop system on an SO-101 arm that inspects tabletop objects (e.g. Duplo
bricks), detects the defective ones and physically sorts them. Exact objects, defects and sorting
rules are announced at kickoff. Record them here when known.

## Kickoff details

- Objects: _TBD_
- Defects: _TBD_
- Sorting rule: _TBD_

## Progress checklist

Update this list when a step is done or blocked; link log / troubleshooting entries.

### Step 1: Station

- [x] Studio backend runs natively (`scripts/studio_backend.sh`, :7860), 2026-11-13
- [x] Studio UI runs (`scripts/studio_ui.sh`, :3000), 2026-11-13
- [x] Leader and follower powered and answering on the bus (2026-11-13 ~15:35)
- [ ] Cameras mounted firmly: overhead, wrist, second external. Roles decided (docs/hardware.md)
- [ ] Hugging Face read-only token set in Studio (Settings → General → Hugging Face)
- [x] PyTorch sees the Arc iGPU (`torch.xpu.is_available()`), OpenVINO sees GPU and NPU (T-008)

### Step 2: Modal (VLA training on NVIDIA GPUs, $100 credit per participant)

- [ ] Modal account and credit redeemed. Deploy failed with "spend limit exceeded" (T-014)
- [x] Modal credentials for the backend user (`~/.modal.toml`, profile chessnok; modal 1.6.3)
- [ ] Studio: Settings → Training Targets → Modal Provider → Deploy and connect

### Step 3: Studio configuration

- [ ] Project created; robots, cameras, environment added

### Step 4: Anomaly detection (Anomalib, separate venv, trains on the iGPU)

- [x] `uv venv anomalib-env && uv pip install "anomalib[openvino,xpu]"`: `anomalib-env/` (anomalib 2.7.0,
  torch 2.15.0+xpu, openvino 2026.4.1); use `UV_PROJECT_ENVIRONMENT=$PWD/anomalib-env` (README), 2026-11-14
- [x] Good-image dataset captured: brick crops from dataset `431a71f7` + good state snapshots, real
  defects from 49 defective snapshots, per-crop Gemini labels (ADR 0006), 2026-11-14
- [x] Model chosen (PaDiM / PatchCore / ...) and exported to OpenVINO: PatchCore on 1.4x crops
  (run `20261114-081831_patchcore_bricks_z14t`), FP16 IR on the NPU, 70 ms / crop, threshold 0.52:
  brick recall 84 %, 3 % (2-fold 6 %) false alarms per good brick, brick AUROC 0.94 on 50 real
  defective bricks (ADR 0006). Training runs on the CPU, not the iGPU (T-015, T-021).

### Step 5: Demonstrations and policy (ACT / SmolVLA / Pi0.5)

- [ ] Teleoperation episodes recorded and reviewed
- [ ] Policy trained and tested with **Run model**

### Step 6: Closed loop

- [ ] Detection result drives the policy without a human choosing
- [ ] Safe stop path tested

### Step 7: Optimization on Intel hardware

- [ ] Placement decided (CPU / iGPU / NPU per model)
- [ ] Measured: latency per inference step, per sorting cycle, power, temperature

## Judging rubric (100 points)

| Criterion | Points | What judges look at |
| --- | --- | --- |
| Task success rate | 25 | Accuracy, cycle time, robustness to shifted objects and clutter |
| Intel tech integration | 25 | NPU / iGPU offload, OpenVINO and Robotics AI Suite deployment |
| Dataset and pipeline quality | 20 | Clean teleoperation data, effective fine-tuning in Studio |
| System efficiency | 20 | Latency per inference step, thermal headroom, power on Core Ultra |
| Creativity | 10 | Original ideas beyond the baseline |

### Judging criteria in the organisers' slides (differ from the guide's rubric above)

From `Intel_Physical AI Hackathon.pdf`, slide 7. Ask the organisers which one counts.

| Criterion | Points | What judges look at |
| --- | --- | --- |
| End-to-end Physical AI solution | 25 | One integrated closed loop from perception and defect detection to SO-101 action |
| Defect detection with Anomalib | 20 | Detection quality, normal vs defective handling, how defect outputs are consumed downstream |
| VLA and Physical AI Studio integration | 20 | Perception results and task instructions translated into robot actions |
| OpenVINO and Core Ultra Series 3 optimization | 20 | Latency, throughput, deployment efficiency, use of NPU / GPU / CPU |
| Robotic execution and reliability | 10 | Localization, manipulation accuracy, repeatability, recovery from incorrect or uncertain detections |
| Innovation and technical demonstration | 5 | Originality, completeness, clarity of demo, architecture and optimization decisions |

Slide 9 "End goal": ACT moves the brick in front of the camera → Anomalib → SmolVLA puts it into
the "defective" or the "perfect" bowl.

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
- [ ] Leader and follower powered and answering on the bus. Arm A OK; arm B silent, see T-005
- [ ] Cameras mounted firmly: overhead, wrist, second external. Roles decided (docs/hardware.md)
- [ ] Hugging Face read-only token set in Studio (Settings → General → Hugging Face)
- [x] PyTorch sees the Arc iGPU (`torch.xpu.is_available()`), OpenVINO sees GPU and NPU (T-008)

### Step 2: Modal (VLA training on NVIDIA GPUs, $100 credit per participant)

- [ ] Modal account and credit redeemed
- [ ] `uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal setup` run as the backend user
- [ ] Studio: Settings → Training Targets → Modal Provider → Deploy and connect

### Step 3: Studio configuration

- [ ] Project created; robots, cameras, environment added

### Step 4: Anomaly detection (Anomalib, separate venv, trains on the iGPU)

- [ ] `uv venv anomalib-env && uv pip install "anomalib[openvino,xpu]"`
- [ ] Good-image dataset captured
- [ ] Model chosen (PaDiM / PatchCore / ...) and exported to OpenVINO

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

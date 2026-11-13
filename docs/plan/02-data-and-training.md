# Plan 2/3: Datasets and training (local → Modal)

Status: **plan, untested.** Written 2026-11-13. Pipeline: [01-pipeline.md](01-pipeline.md),
setup: [03-setup.md](03-setup.md).

We collect four kinds of data:

| Dataset | Used by | Tool | Where it lives (never in git) |
| --- | --- | --- | --- |
| A. Inspection images (good + defect) | Anomalib generalized model | our script `scripts/capture_inspection.py` | `captures/anomalib/` |
| B. Teleoperation episodes | ACT pick policy | Physical AI Studio recording UI | Studio storage; export LeRobot v3 copy to `datasets/` |
| C. Waypoints (teach poses) | scripted look / flip / place | `scripts/record_waypoints.py` | `configs/waypoints.yaml` (small, OK in git) |
| D. Evaluation runs | judging evidence, threshold tuning | `scripts/sort.py` logs | `outputs/eval/` |

Order: **C first** (A is captured using the scripted motions), then A and B in parallel.

---

## C. Waypoints (teach poses)

With the follower following the leader arm, save the follower joints (all 6, including gripper)
at each named pose:

| Pose | Description |
| --- | --- |
| `home` | rest, out of all camera views of the corner |
| `look` | wrist camera looks at faces B+L of a brick in the corner; arm not in the side camera's view |
| `corner_above`, `corner_grasp_open`, `corner_grasp_closed`, `corner_lift` | grasping in the corner (style depends on flip option F1/F2/F3) |
| `flip_rolled`, `corner_place_flipped` | flip sequence: after `wrist_roll` 180°, then back in the corner |
| `good_above`, `good_release`, `defect_above`, `defect_release` | the two plates |

Record **with a real brick** so gripper values are right. Replay at low speed and reduced torque
first (AGENTS.md safety rules). Re-record if the corner guide, plates or cameras move.

---

## A. Anomalib dataset: inspection images

### How to capture (semi-automatic)

The arm collects the data itself using the same scripted motions as the real loop, so training
images match runtime images exactly:

1. A person drops a brick into the corner (random allowed orientation, upside down half the time).
2. `capture_inspection.py` runs: `look` pose → capture overhead / side / wrist crops → flip →
   capture overhead / side crops → saves 5 images tagged with brick id and view.
3. Repeat.

```text
captures/anomalib/
  good/    overhead_pre/  side_pre/  wrist_pre/  overhead_post/  side_post/
  defect/  overhead_pre/  side_pre/  wrist_pre/  overhead_post/  side_post/
```

File names: `<brick>_<cycle>_<view>.png` (e.g. `2x2red_017_side_pre.png`). One model is trained
on **all good views together**; the view folders exist only for per-view evaluation.

### How much and what variation

| Item | Target |
| --- | --- |
| Good cycles | 20–30 per brick × 5 bricks = 100–150 cycles → 500–750 good images (about 20 s per cycle) |
| Defect cycles | at least 10–20 cycles over all shapes, covering every view |
| Orientations | every allowed way the brick fits the corner: 4×2 long side left/right; slope brick all 4 slope directions; upside-down starts |
| Lighting | room lights on/off, daylight changes, people walking in the background of the side camera (or put a plain board behind the corner) |
| Colours | all three 2×2 colours |

"Good" must cover all normal variation, or normal bricks get rejected (guide hint). If a good
brick gets flagged during testing, add more good images like it.

### Defect samples

Ask the organisers for defective samples first. Otherwise make them on spare bricks:
marker scribbles, tape over a stud, scratches, a chipped edge, a cut-off stud (logical defect).
Keep defect images **out of training**. They are only for the test set and threshold. Re-collect
when the kickoff defects are known.

### Splits

- train: 80% of good cycles (split **by cycle**, so the views of one cycle stay together)
- test: remaining 20% good cycles + all defect cycles

---

## B. ACT dataset: teleoperation in Physical AI Studio

Follow Studio docs `application/docs/05-recording-datasets.md`.

- **Environment:** SO-101 follower + SO-101 leader, cameras overhead D415 (colour), side, wrist,
  all 640×480 @ 30 FPS (Studio resizes internally; more pixels only cost disk and USB bandwidth).
- **Task (one):** "Put the brick into the inspection corner".
- **Amount:** 60–80 episodes to start (Studio recommends at least 50), then 20–40 targeted episodes
  for the failure cases found in testing.
- **Episode protocol:** reset the scene → start from `home` → one smooth pick → push into the
  corner → release → back to `home` → Accept. Same grasp style every time (the one the flip needs).
- **Variation matrix** (robustness to shifted objects and clutter is 25 points):

| Factor | Spread |
| --- | --- |
| Start position | whole workspace, think 3×3 grid, every cell several times |
| Start yaw | random |
| Brick | all 5 bricks, roughly equal |
| Upside-down start | about 25% |
| Clutter | about 30% of episodes with distractor objects near the brick |
| Lighting | a few episodes per lighting condition |
| Recovery | a few episodes where the first grasp is poor and you regrasp |

- **Review before training:** replay each episode, check the joint graph, Discard jerky, failed or
  hesitant ones. Bad demonstrations make a bad policy.
- **Do not move** cameras, corner guide, or plates between recording and running. Mark them with tape.
- Export a LeRobot v3 copy as backup (`datasets/`, not git).

If we try SmolVLA later (stretch), re-use the same episodes with task strings, plus new episodes
with several bricks and tasks such as "pick the red brick".

---

## D. Evaluation

Measured, logged, and shown to the judges:

| Metric | How |
| --- | --- |
| Anomalib per-view image AUROC and false-alarm rate | `scripts/eval_detector.py` on the test split |
| **Cycle-level** accuracy (max over views) | same script, grouped by cycle → choose the threshold here |
| Pick success rate | 20 trials, shifted positions, with/without clutter |
| End-to-end sorting accuracy and cycle time | 20 bricks (mix good/defect) through `scripts/sort.py` |
| Latency / power / temperature | see [01-pipeline.md](01-pipeline.md#device-placement-optimize) |

If one view is weak (e.g. missing studs on tops), split **only that view** into its own model
(per shape if needed) and compare before/after. That comparison is good judging evidence.

---

## Training

### 1. Anomalib: locally on the Arc iGPU (no Modal, costs nothing)

Run in `anomalib-env` ([03-setup.md](03-setup.md#4-anomalib-environment)).

```python
from anomalib.data import Folder
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Patchcore

datamodule = Folder(
    name="lego",
    root="captures/anomalib",
    normal_dir="good",          # all views together = one generalized model
    abnormal_dir="defect",
)
engine = Engine(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())
model = Patchcore()
engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)
engine.export(model=model, export_type="openvino", compression_type="int8_ptq",
              datamodule=datamodule, input_size=(256, 256))   # static shape for the NPU
```

(Untested: argument names follow the Anomalib 2.x docs. Check them against the installed
version. The CLI equivalent is `anomalib train ... --trainer.accelerator xpu --trainer.strategy xpu_single`.)

Model plan:

| Run | Model | Why |
| --- | --- | --- |
| 1 | PaDiM | baseline, trains in about 1 minute |
| 2 | **PatchCore** | expected best, 3–5 minutes |
| 3 | FoundAD | reportedly better with few good samples |
| (4) | EfficientAd | only if NPU latency is a problem |

Pick by cycle-level accuracy first, latency second. Export the winner as INT8 (and FP16 for
comparison) and benchmark both on CPU / GPU / NPU.

### 2. ACT: local smoke test first, then Modal

**Step 2a: local smoke test (free).** In Studio, train ACT on the **backend host** (the laptop)
with ~10 episodes and a small number of steps. The goal is not a good policy. It checks the whole
path: dataset → training → export → **Run model** on the arm → OpenVINO export loads in
`physicalai.InferenceModel`. Catch mistakes before paying for GPU time.

**Step 2b: real training on Modal.** Studio → Models → Train model → Run on: the Modal target →
NVIDIA GPU: **A10G** (default, cheapest, enough for ACT). Start with the default advanced settings,
watch the loss curve live, and stop early if it is flat. Interrupting a job saves a checkpoint and
still exports the model.

How Studio's Modal training works (from `application/modal/README.md`):

- Uploads a dataset snapshot to a Modal Volume, runs one GPU call (max 24 h), downloads the
  exported model (PyTorch, OpenVINO, ONNX, ExecuTorch) back into Studio.
- **No resume / no checkpoint continuation:** every retrain with more episodes is a new full run.
- **Billing continues while a job runs even if Studio is closed.** Cancel failed jobs from Studio
  or the Modal dashboard.
- **Deleting the Modal target deletes the whole Volume.** Import all models into Studio first.

**Step 2c: iterate.** Run the new model with **Run model** (Studio, GPU) → note failure cases →
record 20–40 targeted episodes → retrain.

### 3. Modal budget ($100 per participant)

Prices are approximate; **check https://modal.com/pricing** and measure the first run's time per
1k steps before planning the rest.

| GPU | Rough price | Use |
| --- | --- | --- |
| A10G | ~$1.1 / h | ACT (all runs) |
| A100 80 GB | ~$2.5 / h | SmolVLA fine-tune (stretch) |
| H100 | ~$4 / h | avoid; Pi0.5 only if really needed |

| Item | Estimate |
| --- | --- |
| Image build + Volume storage | a few $ |
| ACT v1 (60–80 episodes, A10G, ~1–3 h) | ~$2–4 |
| ACT v2, v3 (more episodes, retrain from scratch) | ~$5–10 |
| SmolVLA stretch run (A100, a few hours) | ~$10–25 |
| **Reserve** | keep ≥ $40 for the last hours before the demo |

Each participant has their own $100. The Studio backend uses one Modal account at a time, so a
second account means re-running `modal setup` and deploying a second target.

### 4. After training

1. Download the OpenVINO export from Studio's Models page into `exports/act/` (not git).
2. Check latency: `InferenceLatencyBenchmark` on CPU and GPU.
3. Plug into `scripts/sort.py` (step 1 of the pipeline).

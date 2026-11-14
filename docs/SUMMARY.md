# Project summary: where we are (2026-11-14, 07:50 laptop clock)

This page is a snapshot. Details, numbers and reasons are in the linked ADRs, reports and the daily
logs (`docs/log/`). Anything marked **untested** has not run on the real robot.

## Pipeline status

| Stage | What exists | Status |
| --- | --- | --- |
| PERCEIVE | Brick, gripper and bowl positions from the two D415s by colour CV, plus a metric table frame. Every dataset frame annotated (A, ADR 0010). Forward kinematics from recorded joints (B). Top-camera pose fitted with the arm as the marker (`configs/top_camera.yaml`). | Offline data done; live use only in the hover script |
| DETECT | Zoomed brick crops per camera. Anomalib PatchCore on CPU / OpenVINO. Local VLM (Qwen3-VL-8B on OVMS, iGPU). Per-crop "mark visible" labels from Gemini. | Working, being tuned |
| REASON | Defective if ANY camera or ANY detector says so (ADR 0008). | Rule decided; not wired into a loop yet |
| EXECUTE | ACT trained on Modal (30 epochs). `pipeline_hover.py`: observe → top photo → brick → gripper-down IK → hover. | The hover runs end to end but misses by 3–4 cm |
| OPTIMIZE | Latency numbers per model (below). NPU runs not done yet. | Partly |

## Detection quality (107 real snapshots: 49 defective, 58 good)

Brick-level, OR over the three camera crops:

| Detector | Recall at 0 false alarms | Recall at ~5 % false alarms | Time |
| --- | --- | --- | --- |
| Qwen3-VL-8B int8, prompt v3, one request with 3 crops | 0.78 (38/49) | n/a (0 false alarms) | 2.5 s per brick |
| Qwen3-VL-8B int8, one request per crop, OR | n/a (3/58 false alarms) | 0.86 (42/49), 3/58 false alarms | 6.5 s per brick |
| Anomalib PatchCore `bricks_z14` (30 held-out good snapshots) | 0.80 | 0.88 | 86 ms (int8) / 229 ms (fp16) per crop, CPU |
| **Both, OR** | **0.94 (46/49)** | | |

- The user's verbatim prompt reaches only 0.31 recall; prompt v3 (ADR 0005) is the one to use.
- PatchCore is weak on dataset video frames: recall 0.37 at 5 % false alarms when they count as good. Its snapshot numbers may be optimistic, because the good test set is small and the defective bricks come from one capture session.
- Per-crop labels: Gemini 3.1 Pro marked 97 of 147 crops of defective bricks as "mark visible". A re-label with `gemini-4-argon` is running.
- The 3 bricks that both detectors miss probably show the mark to no camera.

## Arm and localization

- **Brick localization** (A): colour CV, 2.0 mm median and 6.3 mm p90 error on the table, 7–9 ms per image. The VLM reaches similar accuracy but takes 7.5 s and confuses lime with yellow. Report: `docs/perception/A-object-localization.md`.
- **Kinematics** (B): about ±5 mm. Studio's own degree unit and 3D viewer use the wrong zero (up to 8°). Report: `docs/perception/B-arm-kinematics.md`.
- **Hover above a brick** (ADR 0007): two paths.
  - `pick_above.py`, the wrist camera with a fixed `observe` pose and a homography. Not calibrated: no `calib` snapshots yet.
  - `pipeline_hover.py`, the top camera. Run twice: it reached the hover pose, but 3–4 cm off. About 1.8 cm comes from gravity sag at full reach; 1–2 cm from the top camera's grazing view.
  - Planned fixes: one sag-correction step, then visual servoing with the wrist camera.
- **Reach:** with the gripper straight down and the jaws 6 cm above the table, 7.6–26 cm from the pan axis. Put bricks within about 22 cm.

## Models

| Model | Where | Notes |
| --- | --- | --- |
| ACT, Modal job 7b07e8c5 | Studio models dir `62003713…`; `artifacts/` (LFS, pending) | 30 epochs on A10G, val/loss 0.434 at epoch 29, still improving (no overfitting). Not yet tested on the robot. |
| Anomalib PatchCore `bricks_z14` | `exports/anomalib/20261114-070921_patchcore_bricks_z14`; `artifacts/` (LFS, pending) | fp16 54 MB, int8 39 MB |
| Qwen3-VL-8B-Instruct int8 | public `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov`, served by `scripts/vlm_server.sh` | Gemma 4 26B-A4B comparison pending (download) |
| Pi0.5 | not trained | OOM on A10G 24 GB (needs A100 or H100, or LoRA) |

## Studio (our fork, `chessnok/physical-ai-studio-hj`)

- **Deployed:** `hj/snapshot-viewer`, which includes:
  - RealSense RGB-D recording and camera fixes;
  - Space snapshots with a defective flag, comment and CSV;
  - Enter / Escape episode hotkeys;
  - the relaxed lag check;
  - snapshot preview and the trash button.
- **Ready, not deployed:** `hj/early-stopping`, optional early stopping (off by default). It needs a Studio restart, plus a Modal redeploy to use it on Modal.

## Open items

1. Install `git-lfs`, then push `artifacts/` (ADR 0011).
2. Hover accuracy: sag correction and wrist-camera servoing.
3. A grasp policy from the hover pose: ACT trained on "from hover, grasp" episodes, or evaluate the existing ACT.
4. Detector: finish the Argon re-label and the Gemma 4 comparison. Run PatchCore on the NPU. Capture more good and marked bricks from different sessions.
5. Closed loop: detect → decide (ADR 0008) → pick → place into the bowl. It needs a stop path (AGENTS.md rule 8).
6. Wire the defect verdict into the place step: scripted bowl poses first, SmolVLA later.

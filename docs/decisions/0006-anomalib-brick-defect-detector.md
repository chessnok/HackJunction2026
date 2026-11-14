# 0006. Anomalib brick detector: PatchCore on 1.4x brick crops from all three cameras, on the NPU

- Status: Accepted
- Date: 2026-11-14
- Deciders: team lead (chessnok), agent

## Context

- Defect rule: a brick is defective if and only if it is clearly marked with a black marker. The
  organisers' rubric gives 20 points to "Defect detection with Anomalib" alone.
- ADR 0008: zoomed brick crops from every camera, brick-level OR over crops and detectors.
  ADR 0009 + README: Anomalib 2.7.0 in its own env `anomalib-env/` (torch 2.15.0+xpu, openvino
  2026.4.1); every uv command for it needs `UV_PROJECT_ENVIRONMENT=$PWD/anomalib-env`.
  The teammate's `defect/training/` code trains, exports FP16 + INT8 OpenVINO models and evaluates.
- Data (2026-11-14, laptop clock):
  - Studio dataset `431a71f7`: 67 episodes, 19228 frames, AV1 video (CRF 30), cameras `top`, `floor`
    (RealSense D415 colour 848x480) and `arm` (wrist camera 1024x768); every brick is good.
  - State snapshots, project `0fd76ff2`: 108 snapshots (lossless PNG from the live cameras), 49
    flagged defective (06:10-06:28), 59 good (05:00-05:10 and 06:27-06:38).
  - The first positive (`20261114-051059-914`) has a black arc about 1.5 px wide on a 70 px brick,
    visible only from the floor camera.
- XPU training hung the GPU before (T-015); the VLM server (OVMS) holds the GPU. CPU load made
  Studio drop recordings twice (T-021).

## Decision

- **Model:** PatchCore (`wide_resnet50_2`, layer2 + layer3, coreset 0.004, 256x256) from
  `training.train_patchcore`, trained on CPU only.
- **Input:** one crop per brick per camera: a square of **1.4x** the brick's larger side from the
  full-resolution frame, background outside the brick hull replaced by white, 256x256
  (`defect/cropping/bricks.py`; the same code builds the training data and runs at inference). The
  VLM keeps its 2.4x crops (ADR 0008): with a white background a wider margin only makes the brick,
  and the mark, smaller in Anomalib's 256 px input.
- **Training data:** good bricks only: 400 crops from dataset episodes (every 10th frame, train/test
  split by episode, stratified by camera x colour) + half of the good snapshots (hour-stratified
  6-snapshot time blocks). Defective bricks are never in train; good crops that the label model flags
  as marked are moved out of train (`excluded/`).
- **Per-crop labels:** `scripts/label_crops_gemini.py --model gemini-4-argon`. Each snapshot crop
  (brick only, masked) gets "mark visible in this crop?". The user's corrections in
  `configs/defect_mark_labels.csv` (note `user review ...`; `true` / `false` / `exclude`) override
  it. Crops of defective bricks go to `test/defect` (visible) or `test/defect_clean` (not visible), so
  per-crop recall is measured only on crops that show the mark. Brick-level metrics use every crop (OR).
- **Decision rule:** brick is defective if max(normalized score over its crops) > **0.493**,
  the lowest threshold with <= 5 % false alarms per good-snapshot brick (uses good bricks only, so
  recall at it is not tuned on defects).
- **Device:** NPU (FP16 IR), 71 ms per crop. The GPU is faster (22 ms) but is shared with the VLM
  server; CPU FP16 is 229 ms (4 threads). `configs/anomalib_deploy.yaml` holds run, model,
  threshold, device and crop settings.

## Results

Run `20261114-070921_patchcore_bricks_z14` (dataset `bricks_z14`: 470 train crops; test = 401
held-out episode crops, 77 crops / 30 bricks of good snapshots, 147 crops / 49 bricks of defective
snapshots, of which 97 crops show the mark). Per-crop labels: gemini-3.1-pro-preview (the
gemini-4-argon relabelling was paused at 74/295 crops, see the log). Normalized scores; snapshot good
bricks are all from 05:00-05:10, defective ones from 06:10-06:28.

| Model, export | device | AUROC brick (OR) | AUROC crop (mark visible vs good snapshot) | recall brick @ 0.493 | recall crop @ 0.493 | false alarms brick, good snapshots (in-sample / 2-fold) | false alarms crop, episode frames | latency / crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PatchCore FP16 | CPU | 0.978 | 0.863 | 88 % | 55 % | 3 % / 17 % | 10 % | 229 ms (4 thr, Anomalib inferencer) |
| PatchCore FP16 | NPU | 0.977 | 0.862 | 88 % | 55 % | 3 % / 17 % | 11 % | 71 ms (73 ms in eval) |
| PatchCore FP16 | GPU | 0.978 | 0.863 | 88 % | 55 % | 3 % / 17 % | 10 % | 22 ms (VLM on the same GPU) |
| PatchCore INT8 | CPU | 0.952 | 0.850 | 76 % (thr 0.531) | 44 % | 3 % / 14 % | 7 % | 86 ms |
| PaDiM FP16 (`bricks_v1`, zoom 2.4, episodes-only train) | CPU | 0.696 | 0.582 | 10 % (thr 0.844) | 4 % | 3 % | 4 % | 26 ms |

Per-crop recall at fixed false-positive rates (PatchCore FP16; threshold set on the good crops named):

| threshold set on | FPR | threshold | recall crop (mark visible) | recall brick |
| --- | --- | --- | --- | --- |
| good snapshot crops (77) | 0 % | 0.503 | 47 % | 80 % |
| good snapshot crops | 5 % | 0.487 | 57 % | 88 % |
| good snapshot crops | 10 % | 0.442 | 66 % | 96 % |
| episode-frame crops (401) | 0 % | 0.774 | 6 % | 12 % |
| episode-frame crops | 5 % | 0.556 | 33 % | 59 % |
| episode-frame crops | 10 % | 0.494 | 55 % | 88 % |

Checks:
- Confound check: crops of the same defective bricks where the mark is NOT visible score like good
  bricks (median 0.31 vs 0.29, AUROC 0.59); mark-visible crops score 0.50 (AUROC 0.80 against the
  mark-free crops of the same bricks). The model reacts to the mark, only a little to the session.
- Per camera, crop AUROC mark-visible vs good snapshot: top 0.86, floor 0.87, arm 0.89.
- SYNTHETIC strokes (secondary; 478 good crops, 199 stroked crops / bricks): crop AUROC 0.975, brick
  AUROC 0.953, recall 93 % crop / 94 % brick at 0.493 (INT8: 0.943 / 0.921).
- Segmentation + crop on CPU per frame: top 37 ms, floor 32 ms, arm 67 ms. One brick decision (3
  cameras) on the NPU: about 0.3 s compute, 0.5 s end to end in `scripts/brick_anomaly.py`.

## Alternatives considered

- **PaDiM** (resnet18; one Gaussian per image position): brick AUROC 0.70, crop 0.58, synthetic brick
  0.63 on `bricks_v1` (zoom 2.4); 10 % brick recall at 3 % false alarms. Unsuited to freely rotated
  bricks; its INT8 export is broken (constant score, as on the teammate's mock data). A PaDiM run on
  the 1.4x crops is queued (2 threads) but not done.
- **Zoom 2.4 (ADR 0008's VLM crop) for Anomalib**: with a white background it only shrinks the brick
  to ~42 % of the 256 px input (1.4x: ~70 %). The PatchCore `bricks_z24` run finished its coreset step
  but was lost when the pipeline was paused (T-021); a rerun is queued. Not measured yet.
- **Raw background instead of white:** not measured end to end (the `bricks_v1raw` twin was built,
  its training was dropped when CPU time ran short). With raw crops at 2.4x zoom the model would
  also learn the table, plates and people behind the sheet; whitening was kept.
- **INT8 PTQ:** no speed-up on NPU (70 ms) or GPU (23 ms), 2.6x on CPU (86 ms), but lower quality
  (brick AUROC 0.95 vs 0.98, recall at 5 % false alarms 76 % vs 88 %). FP16 is deployed.
- **EfficientAd:** not trained: needs ImageNette (~1.5 GB over the venue Wi-Fi, T-010) and long
  training; PatchCore already runs at 71 ms per crop on the NPU.
- **XPU training:** not used (T-015; GPU shared with the VLM server).
- **Label model:** gemini-3.1-pro-preview (`gemini-pro-latest`) first, then `gemini-4-argon` on the
  user's request. On the 295 snapshot crops they agree on 97 % of good crops and 93 % of defective-brick
  crops (top 94-96 %, floor 98 %, arm 86 %). The user's review (9 rows: 7 true/false, 2 exclude) flips
  7 of the 7 corresponding 3.1-pro labels but only 1 Argon label (`061847-779` arm, dark-blue brick).
  Argon found the mark on `055500-789` top (the snapshot was mislabelled good in Studio) and cleared
  3.1-pro's three false flags on good crops. Its errors: marks "seen" on the 2 crops where the brick is
  not visible (user: exclude), and one good crop flagged (`050956-674` arm, medium: the "smudge" is the
  purple gripper finger; kept in `test/good` and reported). With Argon + review: 100/150 defective
  crops show the mark (arm 28, floor 37, top 35 of 50); 49/50 bricks in at least one crop.
- **Labels by eye only:** 90 crops of 30 defective snapshots were labelled by eye as a spot check
  (agent rows of `configs/defect_mark_labels.csv`). Against gemini-3.1-pro: 72/81 agreed at first; on
  re-checking the 9 disagreements at 360 px, Gemini was right on 5, wrong on 1, 3 undecidable
  (dark-blue brick). Against Argon: 74/77 agree. Gemini scales to every new snapshot and is cached
  per crop; the user's review stays the final word.

## Consequences

- One command retrains on new data, e.g. inspection-pose recordings:
  `scripts/brick_retrain.sh <name> patchcore -- --dataset <id> --snapshots 0fd76ff2 [--pose inspect]`
  (crops → Argon labels → train → evaluate → report). Training takes ~12-25 min on 2 CPU threads.
- Episode crops (AV1 video) score higher than live-camera crops; thresholds must be set on
  live-camera good bricks (snapshots), not on dataset frames.
- Known gaps: marks on dark-blue bricks are nearly invisible (the label model often disagrees with
  itself there); small marks at a face edge are missed; the wrist camera finds the brick in only 84 of
  108 snapshots. Per crop only about half of the visible marks are caught at 5 % false alarms; the OR
  over three cameras lifts that to 88 % per brick. The false-alarm rate rests on 30 good bricks from
  one early time window (2-fold estimate 17 %): treat it as optimistic until `bricks_z14t` (late good
  snapshots in test) is evaluated. Worst false alarms are blurred, finger-occluded wrist-camera
  frames from episodes, which a static inspection pose avoids.
- Revisit when inspection-pose data exists (close-up crops should raise crop-level recall), and when
  more same-session good snapshots allow a tighter false-alarm estimate.

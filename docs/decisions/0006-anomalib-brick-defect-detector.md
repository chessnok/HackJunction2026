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
  Studio drop recordings twice (T-022).

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
- **Decision rule:** brick is defective if max(normalized score over its crops) > **0.521** (exact value 0.5205; do not round to 0.52),
  the lowest threshold with <= 5 % false alarms per good-snapshot brick (uses good bricks only, so
  recall at it is not tuned on defects).
- **Device:** NPU, FP16 IR: 70 ms per crop. GPU 14-22 ms but shared with the VLM server; CPU 360 ms
  (2 threads). INT8 is not used: it gives wrong scores on the NPU (every crop saturates at 1.0) and no
  speed-up there. `configs/anomalib_deploy.yaml` holds run, model, threshold, device and crop settings.
- **Model files:** run `20261114-081831_patchcore_bricks_z14t`,
  `exports/anomalib/<run>/fp16/weights/openvino/model.xml` (56 MB) and `int8_ptq/...` (40 MB), on the
  robot laptop (not in git).

## Results

Final run `20261114-081831_patchcore_bricks_z14t`, trained on CPU with `--threads 2 --num-workers 0`
(903 s including export). Dataset `bricks_z14t` (108 snapshots, 50 defective): 461 train crops (400
episode + 61 good-snapshot crops); test = 401 held-out episode crops, 85 crops / 34 bricks of good
snapshots (29 from 05:00-05:10, 5 from 06:27-06:38), 150 crops / 50 bricks of defective snapshots:
99 crops show the mark, 49 do not, 2 excluded by the user. Labels: gemini-4-argon + the user's 9
corrections. Scores from the NPU (they match CPU and GPU to the third decimal).

| Model, export, device | AUROC brick (OR; defective vs good snapshots) | AUROC crop (mark visible vs good snapshot crops) | recall brick @ 0.52 | recall crop @ 0.52 | false alarms per good-snapshot brick @ 0.52 (in-sample / 2-fold) | false alarms per episode-frame crop | latency / crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PatchCore FP16, NPU** | **0.943** | **0.835** | **84 %** | **52 %** | **3 % / 6 %** | 13 % | **70 ms** |
| PatchCore FP16, GPU | same | same | same | same | same | same | 14 ms (VLM on the GPU) |
| PatchCore FP16, CPU 2 threads | same | same | same | same | same | same | 360 ms |
| PatchCore INT8, CPU | 0.84 per cycle (FP16: 0.85) | - | - | - | - | - | 174 ms |
| PatchCore INT8, NPU | 0.68 (scores saturate at 1.0: broken) | 0.55 | - | - | - | - | 63 ms |
| PaDiM FP16 (resnet18, same crops), NPU | 0.549 | 0.511 | - | - | - | - | 20 ms |

Per-crop recall at fixed false-positive rates (PatchCore FP16; threshold set on the good crops named,
recall on the 99 mark-visible crops; brick recall = OR over each defective brick's crops):

| threshold set on | crop FPR | threshold | recall crop | recall brick | false alarms per good-snapshot brick |
| --- | --- | --- | --- | --- | --- |
| good snapshot crops (85) | 0 % | 0.527 | 46 % | 76 % | 0 % |
| good snapshot crops | 5 % | 0.471 | 65 % | 90 % | 12 % |
| good snapshot crops | 10 % (8 %) | 0.447 | 70 % | 90 % | 21 % |
| episode-frame crops (401) | 0 % | 0.826 | 5 % | 10 % | 0 % |
| episode-frame crops | 5 % | 0.593 | 31 % | 60 % | 0 % |
| episode-frame crops | 10 % | 0.537 | 45 % | 76 % | 0 % |

Checks:
- Confound check: crops of defective bricks where the mark is NOT visible score exactly like good
  crops (median 0.353 vs 0.351, AUROC 0.51); mark-visible crops score 0.525 (AUROC 0.79 against the
  mark-free crops of the same bricks, same minutes). The model reacts to the mark, not to the session.
- Same-session check: brick AUROC 0.947 against the 29 early good snapshots, 0.916 against the 5 late
  ones (06:27-06:38, same session as the defects; all 5 score 0.36-0.47 < 0.52).
- Per camera, crop AUROC (mark visible vs good snapshot): top 0.83, floor 0.81, arm 0.88.
- SYNTHETIC strokes (secondary; 486 good test crops, 203 stroked crops / bricks): crop AUROC 0.975,
  brick AUROC 0.952; recall 93 % crop / 94 % brick at 0.52.
- The one good test crop the label model flags (`050956-674` arm, Argon, medium) scores 0.35.
- Segmentation + crop on CPU per frame: top 37 ms, floor 32 ms, arm 67 ms. One brick (3 cameras) on
  the NPU: about 0.35 s compute, 0.5 s end to end in `scripts/brick_anomaly.py`.
- Earlier run `20261114-070921_patchcore_bricks_z14` (gemini-3.1-pro labels, all 30 good test
  snapshots from 05:00-05:10, 4 threads): brick AUROC 0.978, crop 0.863, recall 88 % brick / 55 % crop
  at 0.493, 2-fold false alarms 17 %. Superseded by the z14t run (time-stratified test, Argon + user
  labels).

### Follow-up: augmentation, a supervised classifier, fusion (08:50-09:10)

- **Supervised mark classifier** (`training/mark_classifier.py`, run
  `outputs/mark_classifier/20261114-085214_resnet18lr_bricks_z14t`): frozen ImageNet resnet18 (avg + max
  pooled layer3/layer4) on the NPU + logistic regression; trained on mark-visible crops (99 real + 203
  SYNTHETIC strokes) vs good / mark-free crops, each ×6 with hflip, rot90, rot270, colour jitter, blur.
  5-fold CV grouped by snapshot, out-of-fold scores, 58 good + 50 defective snapshot bricks:
  crop AUROC 0.881 (PatchCore 0.835), brick AUROC 0.935 (PatchCore 0.943); at 0 % crop FPR: crop
  recall 43 %, brick 66 %; at 5 % crop FPR (threshold 0.774): crop 71 %, brick 88 %, 10 % false alarms
  per good brick. Exported as one IR: **1.0 ms per crop on the NPU**. Not a clear win over PatchCore
  alone; it is a cheap extra vote.
- **Fusion, brick-level OR** (`training/fusion.py`; 50 defective + 34 good test snapshots; PatchCore at
  0.521, classifier out-of-fold, Qwen3-VL-8B scene verdicts from `crops-or-v3.jsonl`):

| detectors (OR) | recall | false alarms |
| --- | --- | --- |
| PatchCore | 84 % | 3 % |
| classifier (strict, thr 0.998) / (5 % FPR, thr 0.774) | 66 % / 88 % | 3 % / 15 % |
| Qwen3-VL | 86 % | 9 % |
| PatchCore + classifier (strict) | 94 % | 6 % |
| PatchCore + Qwen3-VL | 96 % | 9 % |
| PatchCore + classifier (strict) + Qwen3-VL | 98 % | 12 % |
| PatchCore + classifier (5 %) | 98 % | 18 % |

  The one brick no combination catches is `061820-859` (dark blue; its only mark-visible crop per
  Argon is the one the user excluded).
- **Augmented normals for PatchCore** (`cropping/augment_normals.py`, `bricks_z14ta`: 150 episode
  crops + 1 flip/rot90 copy each, 60 good-snapshot crops + rot90, rot180 and colour-jitter copies;
  540 train crops; same test split; run `20261114-085036_patchcore_bricks_z14ta`): WORSE. Brick AUROC
  0.907 (plain 0.943), crop 0.810 (0.835); brick recall at 0 % / 5 % crop FPR 56 % / 72 % (plain
  76 % / 90 %); at the 5 %-brick rule 58 % recall (plain 84 %). Wider "normal" also absorbs part of
  the marks; only the false alarms on episode crops drop (13 % → 7 %). Not adopted.
  `layers=[layer1, layer2]` was not run (4x the patches per image, about 16x the coreset time on 2 CPU
  threads) and `bricks_z24t` was not trained (deadline).

### Follow-up 2: VLM dropped; detector = PatchCore OR classifier; error analysis (09:20-10:15)

- **Decision (user, 2026-11-14 ~09:15):** the VLM (Qwen3-VL on OVMS) is dropped and its server is
  stopped. The detector is PatchCore (NPU, threshold 0.521) OR the supervised mark classifier (NPU,
  strict threshold 0.9984), per crop and per brick. `configs/anomalib_deploy.yaml` has both;
  `scripts/brick_anomaly.py` / `runtime.BrickAnomalyDetector.from_deploy()` run both (PatchCore 70 ms +
  classifier 1-3 ms per crop). On the 50 defective / 34 good test bricks: **94 % recall, 6 % false
  alarms** (PatchCore alone 84 % / 3 %).
- **Error analysis** (`training/error_analysis.py` → `outputs/error_analysis/bricks_z14t/`: crops.csv,
  bricks.csv, breakdown.md, errors.png):
  1. Dark-blue bricks: 12 of the 50 defective bricks, 2 of the 3 bricks the OR misses. Crop recall on
     blue mark crops: PatchCore 5/15 (33 %), classifier 1/15 (7 %); brick recall classifier 1/12. In the
     top / floor views the dark-blue brick is nearly black (auto exposure), so a black mark has almost no
     contrast.
  2. Faint / small marks: mark crops PatchCore misses have 4x fewer dark pixels inside the brick than
     the ones it catches (median dark fraction 0.0028 vs 0.0108; contrast 0.30 vs 0.37); the classifier
     shows the same (0.0019 vs 0.0116).
  3. The top camera sees the marks worst: crop recall top 37 % (PatchCore) / 40 % (classifier), floor
     54 / 57 %, arm 67 / 30 %.
  4. Brick size is not a driver in this data (bricks are 56-91 px; recall by size bucket is flat), but
     every brick is small: a 1.5-3 px mark on a ~70 px brick.
  5. Wrist camera finds the brick in 84 of 108 snapshots: all 24 misses are 05:00-05:02 snapshots
     taken from a higher wrist pose where the brick touches the top image edge (rejected on purpose as
     cut off). In the current observe pose the brick is always found: a framing issue, not a code bug.
  6. Labels: of the 3 bricks the OR misses, 2 had their only mark-visible crop excluded by the user
     (brick not visible in that view), so 48 of 50 are catchable from the crops that exist.
- **Quick fixes tried** (same split, classifier out of fold; OR with PatchCore at the strict / 5 %
  thresholds): tighter centre crop (0.8): crop AUROC 0.873 vs 0.881, OR 92 % / 3 %; faint synthetic
  strokes on dark bricks (+150): 0.872, OR 90 % / 3 % strict but **98 % / 9 %** at the 5 % point (baseline
  98 % / 18 %); CLAHE on luminance: 0.841, OR 90 % / 6 %. None beats the baseline overall; the dark-stroke
  variant gives a better high-recall point (selected post hoc on 34 good bricks: confirm on new data).
  RESULTS_FT
- **Data is the real lever. Capture checklist** (Studio state snapshots, Space key, `defective` flag):
  - [ ] Resolution: D415 colour at **1920x1080** (Studio allows width <= 1920; colour-only output; USB 3
        port needed): the brick grows ~2.3x (70 → ~160 px), marks from ~2 px to ~4-5 px. Fallback 1280x720
        (1.5x). Wrist camera (USB2.0_CAM1, 05a3:9230): MJPEG modes up to 1920x1080, 1440x1080, 1280x960
        (USB descriptors; `v4l2-ctl` is not installed: `sudo apt install v4l-utils`); now 1024x768.
  - [ ] Exposure: lock exposure / white balance, or add light, until a dark-blue brick is clearly blue,
        not black, in the top and floor views.
  - [ ] Pose: the current observe pose (all 84/84 wrist detections there); avoid poses where the brick
        touches the image edge.
  - [ ] Good bricks: every colour (dark blue, cyan, red, yellow, lime / olive, two-tone), ~10 bricks per
        colour, 6 orientations each (top up, each side, underside), spread over the sheet: ~300 snapshots
        (~900 crops).
  - [ ] Marked bricks: ~30 more, at least 10 dark-blue, marks of 3 sizes (dot, thin line, letter) on
        different faces, 3-4 orientations each so the mark faces each camera once: ~120 snapshots. Write
        the marked face in the snapshot comment.
  - [ ] Interleave good and marked bricks in the same session (no "all good at 05:00, all marked at
        06:00" time confound).
  - [ ] Then: `scripts/brick_retrain.sh <name> patchcore -- --dataset 431a71f7 --snapshots 0fd76ff2`
        (crops ~5 min; Argon labels only for new crops, 7-35 s each at concurrency 3 → ~1 h for 900 new
        crops; PatchCore ~15 min on 2 CPU threads) + `training.mark_classifier` (~2 min, NPU) and review the
        Gemini labels on the review page; user corrections go to `configs/defect_mark_labels.csv`.

## Alternatives considered

- **PaDiM** (resnet18; one Gaussian per image position): brick AUROC 0.55, crop 0.51 on the same
  1.4x crops (`20261114-083956_padim_bricks_z14t`); 0.70 / 0.58 on the 2.4x `bricks_v1`. Unsuited to
  freely rotated bricks; its INT8 export is broken (constant score, as on the teammate's mock data).
- **Zoom 2.4 (ADR 0008's VLM crop) for Anomalib:** not measured. With a white background it only
  shrinks the brick to ~42 % of the 256 px input (1.4x: ~70 %). The first `bricks_z24` PatchCore run
  was lost in the pause (T-022); `bricks_z24t` is built and labelled but its training was skipped for
  the deadline. Run `scripts/brick_retrain.sh`-style training on it to compare.
- **Raw background instead of white:** not measured end to end (the `bricks_v1raw` twin was built,
  its training was dropped when CPU time ran short). Raw crops would also teach the model the table,
  plates and the people behind the sheet; whitening was kept.
- **INT8 PTQ:** wrong on the NPU (scores saturate at 1.0), fine on the CPU (AUROC 0.84 vs 0.85,
  174 ms vs 360 ms at 2 threads). No speed-up on NPU or GPU. FP16 is deployed.
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
- Known gaps: marks on dark-blue bricks are nearly invisible (the label models disagree there); small
  marks at a face edge are missed; the wrist camera finds the brick in only 84 of 108 snapshots. Per
  crop only about half of the visible marks are caught at the deployed threshold (52 %); the OR over
  three cameras lifts that to 84 % per brick. The false-alarm rate rests on 34 good test bricks, only
  5 of them from the defects' session (2-fold estimate 6 %). Worst false alarms are blurred,
  finger-occluded wrist-camera frames from episodes, which a static inspection pose avoids.
- Untested on the robot loop: the detector has run on saved snapshots only, not on live frames inside
  the sorting cycle.
- Revisit when inspection-pose data exists (close-up crops should raise crop-level recall), and when
  more same-session good snapshots allow a tighter false-alarm estimate.

# Merged grasp dataset: NEW grasp-only episodes + the grasp tail of every OLD episode

2026-11-14 (laptop clock, PST). Built offline on the robot laptop; read-only on both sources; Studio
not stopped or restarted. **Untested**: no policy has been trained on it yet.

Code: `src/hackjunction2026/grasp_merge/`, entry point `scripts/grasp_merge.py`. Decision record:
[ADR 0012](../decisions/0012-merge-grasp-dataset-new-plus-old-tails.md). Outputs that are not in git
(clips, cuts, perception files, the dataset itself) are under `/home/intel/dev/captures/merge_review/`.

## Result

| | |
| --- | --- |
| Name | `grasp-merged-2026-11-14` |
| Folder (build output) | `captures/merge_review/dataset/grasp-merged-2026-11-14/` |
| Studio | **not imported yet**: v0 uses the automatic cuts; the user is setting cuts by hand (`configs/grasp_merge_overrides.csv`), then rebuild and import |
| Episodes / frames | 96 / 12756 (v0, 193 MB, one video file per camera) |
| Merged episodes 0-28 | NEW episodes 0-1, 3-29 (whole), 4778 frames |
| Merged episodes 29-95 | OLD episodes 0-66, frames `[cut_frame, end)`, 7978 frames |
| Tasks | names kept as recorded (21 distinct, see below) |
| Format | LeRobot v3.0, 30 fps, same features as both sources: `observation.state` and `action` (6, Studio normalized), cameras `top` / `floor` 848x480, `arm` 1024x768, AV1 (yuv420p, g 2, crf 30, preset 12) |
| Provenance | `meta/grasp_merge.json` inside the dataset: sources, cut per OLD episode, overrides, criterion, code commit |

Sources:

| | Dataset | Studio name | Episodes / frames | Content |
| --- | --- | --- | --- | --- |
| NEW | `7efd5320-a08b-49d6-abed-89b38deaf1ed` | taking-v-rot | 30 / 4783 (snapshot 08:49) | starts hovering above the brick with the gripper open, ends grasped; tasks `blue`, `blue-2`, `red` |
| OLD | `431a71f7-50c6-4e38-9e38-d85984dde78b` | collect-block | 67 / 19228 | starts at rest, approaches, ends around the grasp; 19 task names, **not** colour labels (ADR 0010 / A's doc) |

Excluded: NEW episode 2 (`blue`, 5 frames: an accidental recording; rule: NEW episodes shorter than
15 frames). No OLD episode is skipped (no overrides yet).

**NEW keeps growing.** The team recorded NEW episodes 30 (`potuzno`, 204 frames) and 31 (`task`,
201 frames) at 08:53-08:54, after v0's snapshot (08:49); NEW then had 32 episodes / 5188 frames.
Every build copies NEW as it is at build time, so the next build includes them unless they are left
out with `build --new-exclude 30,31` (recorded in `meta/grasp_merge.json`). Whether they are
grasp-only demonstrations like 0-29 is for the team to say.

Task names: kept exactly as recorded, including OLD's trailing newlines (`colored\n`, `red-2\n`,
`red-blue\n`). `blue-2` exists in both sources and becomes one task; `red-blue` and `red-blue\n`
stay two tasks. OLD names do not describe the brick colour reliably.

## Cut criterion (OLD episodes)

Per OLD frame, in the robot base frame (mm):

- `h`: gripper-tip height above the table (agent B's FK, midpoint of the fingertips, table plane
  z = -2.4 mm);
- `d`: horizontal distance from the gripper tip to the brick's resting position (agent A's brick
  track: median of its first 10 frames with both cameras, mapped table -> robot with the dataset's
  robot-to-table registration);
- `g`: Studio normalized gripper (0 closed, 100 open); `act g`: the commanded (leader) gripper.

Reference = frame 0 of the 29 NEW episodes with at least 15 frames:

| NEW frame 0 | median | sd | min | p5 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| h (mm) | 120.3 | 15.1 | 84.4 | 90.7 | 134.5 | 134.9 |
| d (mm) | 25.1 | 12.7 | 3.3 | 9.6 | 43.1 | 65.4 |
| gripper | 89.5 | 10.1 | 60.3 | 67.7 | 96.8 | 97.7 |
| jaw gap (mm) | 53.4 | 1.0 | 48.8 | 51.5 | 53.4 | 53.4 |
| wrist_flex | 99.1 | 3.8 | 85.9 | 88.6 | 99.3 | 99.3 |

Joint state medians at NEW frame 0: pan -13.7, lift 13.6, elbow -22.9, wrist_flex 99.1,
wrist_roll -51.4, gripper 89.5. Pan, lift and elbow follow the brick position, so they are reported
but not matched.

Rule (`cuts.py`, parameters in `cuts.json`):

1. **Open**: `g >= 30` (jaw gap 28.5 mm). OLD never opens like NEW (widest OLD openings 33-67,
   NEW 60-98), and the OLD operator grasped with openings as small as 33, so the gripper is a
   condition, not a score term.
2. **Final approach**: the last run of open frames (runs less than 15 frames apart merged).
3. **Close**: the frame after the last frame, from the start of that run, where the commanded
   gripper is at least 10 above its final value (median of the last 5 frames). The command is used
   because the jaws stop on the brick: a 2x2 brick holds at g 13-23 but larger bricks at 32-34,
   above the open threshold, and some operators partly close a wide-open gripper while approaching.
4. **Score** for every frame from the start of the final approach to the close:
   `((h - 120.3) / 15.1)^2 + (max(0, d - 25.1) / 12.7)^2` (being closer to the brick than NEW's
   median is not penalised).
5. **Cut** = the latest of those frames with score <= best + 0.25 (the last hover frame before
   the descent, not the middle of a long hover). Keep `[cut_frame, end)`; frame_index and
   timestamp restart at 0.
6. For review only: `alt_cut_frame` = the same rule over all frames before the close with the
   gripper ignored (where the arm hovered when the jaws opened only during the descent).

Confidence is `low` if any flag is set: `low_start` (cut lower than every NEW frame 0, 84 mm),
`high_start` (more than 30 mm above every NEW frame 0), `far_start` (d above NEW's max 65 mm),
`no_open`, `no_close`, `multi_open` (opened more than once), `multi_descent` (after the cut below
40 mm and back above 80 mm before the close), `no_brick`, `brick_moved` (resting brick vs FK tip at
the close > 40 mm), `short_tail` (< 45 frames).

### Result of the rule

- Kept per OLD episode: **116 frames median (3.9 s)**, range 69-226 (2.3-7.5 s), 7978 frames in
  total (41 % of OLD). Cut to close: 82 frames median (2.7 s); after the close the OLD tails
  continue 31 frames median (1.0 s, 9-127) while the operator holds the brick.
- At the cut: h 113 mm median (p10 71, p90 116), d 21 mm (p10 10, p90 35), gripper 44 (33-55).
  The 57 high-confidence cuts: h 85-130 mm, d up to 45 mm, all inside NEW's frame-0 ranges.
- **Low confidence: 10 of 67**:

  | OLD ep | task | cut | kept | alt cut | h / d / g at cut | flags |
  | --- | --- | --- | --- | --- | --- | --- |
  | 0 | blue-right | 215 | 109 | 144 | 45 / 21 / 42 | low_start |
  | 2 | blue-right-rev-2 | 262 | 148 | 198 | 68 / 21 / 44 | low_start |
  | 4 | blue-left-far | 391 | 75 | 218 | 12 / 6 / 39 | multi_open, low_start (failed grasp, re-open at the table) |
  | 10 | yellow-right | 126 | 93 | 102 | 66 / 25 / 36 | low_start |
  | 14 | yellow-right | 148 | 91 | 110 | 75 / 17 / 33 | low_start |
  | 25 | green | 155 | 123 | 126 | 74 / 13 / 38 | low_start |
  | 27 | green | 169 | 116 | 138 | 71 / 25 / 32 | low_start |
  | 29 | green | 143 | 85 | 104 | 66 / 17 / 34 | low_start |
  | 30 | green | 154 | 85 | 118 | 69 / 29 / 49 | low_start |
  | 33 | yet-another-green | 140 | 91 | 111 | 71 / 20 / 42 | low_start |

  `low_start` means the operator opened the jaws only during the descent: the tail starts lower
  than any NEW episode. The alternative cut keeps the hover with a closed (or barely open) gripper,
  so that tail contains the opening, which NEW episodes never do. Decide per episode on the clip.

Differences from NEW that remain after the cut (by construction): the OLD gripper is less open at
the start (44 vs 89.5), wrist_flex is lower (92.7 vs 99.1 median), and OLD tails go on ~1 s after the
grasp.

## Perception inputs

OLD: agent A's `objects_431a71f7-....parquet` and `robot_to_table_431a71f7-....json`
(`captures/perception/A/`), agent B's `arm_431a71f7-....parquet` (`captures/perception/B/`); see
`docs/perception/`. NEW, made for this merge with the same code (outputs under
`captures/merge_review/perception/`, B's FK file next to OLD's in `captures/perception/B/`):

```bash
PY=/home/intel/dev/third_party/physical-ai-studio/application/backend/.venv/bin/python
export PYTHONPATH=src
P=/home/intel/dev/captures/merge_review/perception
nice -n 19 $PY scripts/arm_fk.py annotate 7efd5320 \
  --out /home/intel/dev/captures/perception/B/arm_7efd5320-a08b-49d6-abed-89b38deaf1ed.parquet   # 35 ms
OMP_NUM_THREADS=2 nice -n 19 $PY scripts/perception_table_frame.py 7efd5320 --out $P
OMP_NUM_THREADS=1 nice -n 19 $PY scripts/perception_annotate.py 7efd5320 --workers 2 --out $P  # 121 s
# grasps_<NEW>.csv: B's grasp_events() with min_stall=5 (NEW episodes end soon after the close;
# the default 10 finds 15 of 30), then:
nice -n 19 $PY scripts/perception_register_robot.py 7efd5320 --out $P --b-dir $P
```

NEW table frame: top camera re-aimed after NEW episode 1 (setups 0-1 / 2-29), floor camera one
setup; brick correspondences 5.0 px median. NEW registration: 22 grasps, 16 inliers, 7.9 mm median
residual, rotation 177.8 deg (OLD: 177.6 deg), scale 1.146 (NEW's top camera is not at the 131.5 mm
height A's frame assumes; the similarity fit absorbs it). Check: the camera brick and the FK tip at
the end of the episode agree to 16 mm median in NEW, 12 mm in OLD.

## Video

- NEW: not re-encoded. LeRobot's `delete_episodes` dropped episode 2; its video step (which would
  re-encode the file) was swapped for a stream copy: every episode starts on a keyframe (each
  episode is encoded separately, g = 2), so the kept packets are copied with shifted timestamps.
  Checked: decoded frames bit-identical to the source, including across the removed episode.
- OLD tails: decoded from the cut and re-encoded through LeRobot's own writer
  (`LeRobotDataset.create` / `add_frame` / `save_episode`, streaming encoder, as Studio records),
  one single-episode dataset per (episode, cut) in a cache
  (`captures/merge_review/work/old_tails/ep<NNN>_cut<CCCC>/`) that is kept between builds: a rebuild
  encodes only tails whose cut changed (2-4 s each) and reuses the rest unchanged. The cache was seeded
  from v0 with `seed-cache` (lossless split: each tail starts on a keyframe).
  Encoder from OLD's `info.json` via LeRobot's `encoder_config_from_video_info` (libsvtav1, yuv420p,
  g 2, crf 30, preset 12), writer arguments from Studio's `VideoEncodingSettings`; 2 SVT threads per
  camera encoder (3 in parallel), nice 19, a pause of up to 20 s before each episode while the
  1-minute load is above 8. The written `info.json` features are identical to NEW's.
- Note: this SVT-AV1 build logs `preset 10` for a requested 12 (presets above 10 are clamped). Studio
  recordings use the same library, so the info value 12 means the same thing for both parts.
- Pitfall: LeRobot 0.6's streaming encoder **drops** a frame (warning only) when its queue is full.
  For offline conversion the queue is set larger than the longest episode, and the build fails if
  `_dropped_frames` is non-zero.
- Merge: LeRobot's `aggregate_datasets` (task table, indices, episode metadata, `stats.json` from
  the per-episode stats, video files concatenated without re-encoding).

## Validation

`scripts/grasp_merge.py validate <dataset>` (runs after every build; result in
`captures/merge_review/validation.json`). v0, 2026-11-14 09:01:

- Loads with LeRobot's `LeRobotDataset` (96 episodes, 12756 frames) and with Studio's own
  `InternalLeRobotDataset` (96 episodes, 12756 frames, 21 tasks).
- Metadata: episode_index and index contiguous; frame_index 0..n-1 and timestamp = frame_index / 30
  in every episode; info.json = episodes metadata = data rows; `stats.json` covers all 10 features.
- Video: for all 288 episode-camera pairs the packets inside `[from_timestamp, to_timestamp)` equal the
  parquet rows; each of the 3 video files has exactly 12756 packets (no orphan frames).
- Content: `observation.state`, `action` and task of all 96 merged episodes equal the source rows
  (OLD from the cut frame).
- Decode spot checks through `LeRobotDataset.__getitem__` (12 random frames, all 3 cameras): NEW
  frames bit-identical to the source (PSNR inf), OLD frames 41.2-45.0 dB against the source frame.
- Contact sheet: `captures/merge_review/contact_sheet_first_frames.jpg` (top camera, first frame of
  10 random merged episodes: 3 NEW, 7 OLD tails; no people).

## Review clips

`captures/merge_review/clips/ep000.mp4` ... `ep066.mp4` (one per OLD episode, full episode, top
camera only: 5 frames per episode of every OLD and NEW episode were checked, no people; the floor
and wrist views show people), H.264 424x240, CRF 26, 92-174 kB each, 7.9 MB in total.
`clips/index.json`: episode, task, length, fps, cut_frame, auto/alt cut, close frame, kept frames,
confidence, flags, h / d / g at the cut. Overlay: frame index, time and h / d / g (bottom left),
confidence and flags (bottom right), red border and `CUT` from the cut on; bar at the bottom: grey
dropped, green kept, red tick cut, yellow tick alternative cut, cyan tick close, white playhead.

`scripts/grasp_merge.py clips --camera arm --max-kb 240` renders the same clips from the wrist
camera into `clips_arm/` (424x318, text at the top). The wrist view can show people: run it only with
the user's go-ahead and never commit the output.

## Re-run with corrections

1. Write corrections to `configs/grasp_merge_overrides.csv`, one line per OLD episode:
   `episode,cut_frame` (first frame kept) or `episode,skip`. (On 2026-11-14 the user's local cut tool,
   `/home/intel/dev/scripts/cut_tool.py`, writes this file of the build's worktree directly.)
   Leave NEW episodes out with `build --new-exclude <list>`.
2. Run from the repo root:

```bash
PY=/home/intel/dev/third_party/physical-ai-studio/application/backend/.venv/bin/python
export PYTHONPATH=src
nice -n 19 $PY scripts/grasp_merge.py cuts                   # cuts.csv / cuts.json with the overrides
nice -n 19 $PY scripts/grasp_merge.py clips                  # optional: clips show the new cuts
nice -n 19 $PY scripts/grasp_merge.py build --name grasp-merged-2026-11-14-v2     # build + validate
nice -n 19 $PY scripts/grasp_merge.py import captures/merge_review/dataset/grasp-merged-2026-11-14-v2
```

`build` refuses to run if the overrides file changed after `cuts` and never overwrites an existing
folder; `import` refuses a name that already exists in the project. Time: v0 encoded all 67 tails in
760 s, almost all of it pauses for the load rule (2-4 s of encoding per tail). With the cache, measured
2026-11-14 09:08: unchanged cuts 35 s for build + validation; one changed cut 43 s (that tail encoded
in 3 s); each further changed tail adds its encoding time plus a pause of up to `--max-wait` s
(default 20) while the load is above 8. If the cache is
lost: `grasp_merge.py seed-cache captures/merge_review/dataset/<previous build>` restores it without
re-encoding.

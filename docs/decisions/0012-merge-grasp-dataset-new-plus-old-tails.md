# 0012. Train grasping on NEW plus the data-driven grasp tail of every OLD episode

- Status: Proposed (cuts await a human review of the clips)
- Date: 2026-11-14
- Deciders: team lead (user), agent

## Context

- The team recorded a grasp-only dataset (NEW, `7efd5320`, 30 episodes / 4783 frames): every episode
  starts with the arm hovering above the brick, gripper open, and ends grasped.
- The older dataset (OLD, `431a71f7`, 67 episodes / 19228 frames) starts at rest and approaches the
  brick first. The user asked to add "the pieces at the end, when we are close to the brick" so the
  policy gets more grasp examples without the approach.
- Per-frame FK (agent B) and brick positions (agent A) exist for OLD; the same code runs on NEW in a
  few minutes. NEW frame 0: tip 120 mm above the table (84-135), 25 mm from the brick (max 65),
  gripper 89.5 (60-98).
- The OLD operator opened the jaws only as far as the brick needed (33-67) and in 10 episodes only
  during the descent; NEW opens to 60-98 before recording starts.
- Both datasets have identical features and video settings (AV1, yuv420p, g 2, crf 30, preset 12).
  The team may record at any moment; agent CPU load has caused two discarded recordings today.

## Decision

Build `grasp-merged-2026-11-14` = all NEW episodes with at least 15 frames + every OLD episode cut to
`[cut_frame, end)`. The cut is the latest frame of the final open-gripper approach whose tip height and
tip-to-brick distance best match NEW frame 0 (score in units of NEW's standard deviations; criterion
and numbers in `docs/datasets/grasp-merged.md`). Episodes whose cut is unlike NEW get a `low`
confidence flag and a review clip; a human can override any cut or skip an episode in
`configs/grasp_merge_overrides.csv` and rebuild with one command. Video: NEW is copied without
re-encoding (stream copy, all episodes start on keyframes); only the OLD tails are re-encoded,
through LeRobot's own writer with the encoder rebuilt from OLD's `info.json`; LeRobot's
`aggregate_datasets` merges. The dataset is registered through Studio's own import API.

## Alternatives considered

- **Fixed time before the grasp** (e.g. the last 4 s): simpler, but OLD descents take 18-156 frames
  from hover to close, so a fixed window either keeps approach motion or cuts into the descent.
- **Match the gripper value too**: impossible, OLD never opens like NEW; used as a condition instead.
- **Cut at the hover even when the jaws are still closed** (`alt_cut_frame`): keeps a NEW-like height
  but adds an "open the jaws" phase NEW never has; offered per episode for review, not the default.
- **Re-encode everything** (LeRobot `delete_episodes` unchanged, or one writer pass for both): one
  more generation loss on NEW and about 3x the CPU, during recording hours.
- **Import by copying into Studio's datasets folder or with the backend CLI** (`datasets import-dir`):
  writes Studio's database from a second process while the server runs; the HTTP import is the
  path the UI uses.

## Consequences

- About 7978 extra frames (67 tails, 3.9 s median) next to NEW's 4778; the OLD gripper starts less
  open (44 vs 89.5) and OLD tails continue about 1 s after the grasp. Whether this helps the policy is
  untested: compare a policy trained on NEW alone with one trained on the merge.
- OLD task names are kept; they are not colour labels, and `blue-2` now names episodes from both.
- Rebuild after any override; a new build needs a new name (folders and Studio names are not
  overwritten).

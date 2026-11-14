# 0008. Call a brick defective if any camera or any detector says so, on zoomed brick crops

- Status: Accepted
- Date: 2026-11-14
- Deciders: team lead (chessnok), agent

## Context

- A brick is defective if and only if it is clearly marked with a black marker.
- Two detectors will run on the laptop:
  - Anomalib, with scripts from a teammate (pending);
  - a local VLM on OpenVINO Model Server (ADR 0005), prompted with that rule and returning `{"is_broken": boolean}` under JSON-schema decoding.
- First real positive: snapshot `20261114-051059-914`. It shows a yellow brick, about 70 px wide in the 848x480 floor frame, with a black arc about 1.5 px wide.
  - In the full frame the mark is barely visible.
  - In a 130x130 crop around the brick, upscaled to 400 px, it is clearly visible.
  - Only the floor camera sees the marked face. The top camera sees an unmarked face.

## Decision

- Each camera frame is reduced to a square crop around the brick, found by colour inside the white sheet. The crop is about 2.4x the brick size, taken from the full-resolution frame and upscaled to about 448-512 px. Both detectors see these crops.
- The brick is defective if ANY detector flags ANY crop as defective: OR over cameras and over detectors.
- Every individual vote is logged with the decision.

## Alternatives considered

- **Majority vote or AND:** a mark on one face is usually seen by one camera only, as in the first positive, so these would miss real defects.
- **Full frames only:** the mark covers too few pixels.
- **A dedicated in-hand inspection pose before deciding:** still useful for faces no camera sees, but not needed for resolution, because digital zoom is enough.

## Consequences

- Recall is favoured over precision. The false-positive rate adds up over cameras and detectors, so each detector's per-crop false-positive rate has to be low. It is measured on the good snapshots, 47 so far.
- Colour detection must stay inside the sheet. In the wrist view the wooden table and off-sheet bricks match brick hues.
- A face that no camera sees cannot be judged. Revisit with more views or a wrist rotation if misses show up.

# Patches for Physical AI Studio (`junction2026` @ 17714488)

Our changes to Studio. The same commits are pushed as branches to the private repo
`chessnok/physical-ai-studio-hj` (`hj/realsense-rgbd`, `hj/state-snapshots` + notes in
`HACKJUNCTION.md`); prefer cloning that. These patch files are a backup. Apply on a fresh clone of the `junction2026` branch:

```bash
cd third_party/physical-ai-studio
git checkout -b hj/realsense-rgbd junction2026 && git am ../../patches/physical-ai-studio/00*.patch
git checkout -b hj/state-snapshots && git am ../../patches/physical-ai-studio/snapshots/00*.patch
```

| Series | What | Status |
| --- | --- | --- |
| `0001–0006` | RealSense color + aligned depth recording, Studio-run camera publishers, training excludes depth, UI (ADR 0004) | reviewed: "deploy after fixes" |
| `0007–0013` | Fixes from the review: half-resolution depth + discard on drops, seqlock store, orphan sweep by registry, same-device check, dataset warning, single-image policy, edge cases | re-review stopped before finishing: not independently re-verified |
| `snapshots/0001–0003` | Space-hotkey state snapshots (frames, depth, joints, intrinsics) | re-review stopped before finishing: not independently re-verified |
| `snapshot-labels/0001–0009` | Snapshot defective flag, comment, `snapshots.csv`, Enter/Escape episode hotkeys, relaxed lag check | reviewed, deployed 2026-11-14 (`hj/state-snapshots` @ 3fb5fdfa) |
| `snapshot-viewer/0001–0002` | Thumbnail + preview dialog and move-to-trash button in the Recent snapshots list; `GET .../files/{name}`, `DELETE` endpoints (branch `hj/snapshot-viewer`) | tests pass (backend 2302, UI 625); deployed 2026-11-14 06:19 |

Studio on the laptop runs `hj/snapshot-viewer` @ 67a1ffe5 (since 2026-11-14 06:19; before: `hj/state-snapshots` @ 3fb5fdfa).

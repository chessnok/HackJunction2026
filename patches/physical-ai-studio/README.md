# Patches for Physical AI Studio (`junction2026` @ 17714488)

Our local changes, kept as branches in `third_party/physical-ai-studio` and exported here so the
team can re-create them. Apply on a fresh clone of the `junction2026` branch:

```bash
cd third_party/physical-ai-studio
git checkout -b hj/realsense-rgbd junction2026 && git am ../../patches/physical-ai-studio/00*.patch
git checkout -b hj/state-snapshots && git am ../../patches/physical-ai-studio/snapshots/00*.patch
```

| Series | What | Status |
| --- | --- | --- |
| `0001–0006` | RealSense color + aligned depth recording, Studio-run camera publishers, training excludes depth, UI (ADR 0004) | reviewed: "deploy after fixes" |
| `0007–0013` | Fixes from the review: half-resolution depth + discard on drops, seqlock store, orphan sweep by registry, same-device check, dataset warning, single-image policy, edge cases | re-review in progress |
| `snapshots/0001–0003` | Space-hotkey state snapshots (frames, depth, joints, intrinsics) | re-review in progress |

Not deployed yet: Studio still runs `junction2026` until the re-review passes.

#!/usr/bin/env bash
# Anomalib dataset capture with the two D415s, using Studio's backend environment
# (pyrealsense2 + opencv). --no-sync: never modify that env (a plain `uv run` would re-sync it and
# could swap its XPU torch build). Run from anywhere; arguments are passed on:
#   scripts/capture_inspection.sh --list-cameras
#   scripts/capture_inspection.sh --set-crops
#   scripts/capture_inspection.sh --brick 2x2red
# Stop recording / Run model / camera previews in Studio first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-sync --project "$ROOT/third_party/physical-ai-studio/application/backend" \
  python -m hackjunction2026.defect.capture_inspection "$@"

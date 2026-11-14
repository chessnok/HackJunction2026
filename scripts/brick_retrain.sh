#!/usr/bin/env bash
# One command: build brick crops -> label them with Gemini -> train -> evaluate (real + SYNTHETIC) -> report.
#   scripts/brick_retrain.sh <dataset name> [model] -- [brick_crops.py options]
#   scripts/brick_retrain.sh bricks_v2 patchcore -- --dataset 431a71f7 --dataset <new id> --snapshots 0fd76ff2
#   scripts/brick_retrain.sh inspect_v1 padim -- --dataset <inspection dataset id> --pose inspect --snapshots 0fd76ff2
# model: patchcore (default) or padim. CPU only, niced, 2 threads, no loader workers (T-015, T-021).
# Needs: anomalib-env/ (anomalib 2.7.0; UV_PROJECT_ENVIRONMENT=$PWD/anomalib-env for any uv command),
# Studio's backend env (reads videos: av, pyarrow), AI_STUDIO_API_KEY in .env (crops only are sent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ $# -ge 1 ]] || { sed -n '2,9p' "$0"; exit 2; }
NAME="$1"; shift
MODEL="patchcore"
if [[ $# -ge 1 && "$1" != "--" ]]; then MODEL="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift

STUDIO_PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
PY=anomalib-env/bin/python   # separate Anomalib env (never uv sync it into .venv)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OV_CPU_THREADS=2 PYTHONPATH="$ROOT/src"   # Studio drops recordings when the CPU is busy
GPU_DRV=/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
NPU_DRV=/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1
if [[ -e $GPU_DRV && -e $NPU_DRV ]]; then export ZE_ENABLE_ALT_DRIVERS="$GPU_DRV,$NPU_DRV"; fi  # T-008

nice -n 19 "$STUDIO_PY" scripts/brick_crops.py --name "$NAME" --synthetic --overwrite "$@"
nice -n 19 uv run --no-project --script scripts/label_crops_gemini.py --dataset "$NAME" --model gemini-4-argon --concurrency 3 --apply
nice -n 19 "$PY" -m "hackjunction2026.defect.training.train_$MODEL" --dataset "$NAME" --accelerator cpu --threads 2 --num-workers 0 $([[ $MODEL == patchcore ]] && echo --coreset 0.004)
RUN=$(ls -1t outputs/anomalib/runs | grep "_${MODEL}_${NAME}" | head -1)
nice -n 19 "$PY" -m hackjunction2026.defect.training.evaluate --run "$RUN" --dataset "$NAME-synth"
nice -n 19 "$PY" -m hackjunction2026.defect.training.brick_report --runs "$RUN"
echo "run: $RUN  (benchmark: $PY scripts/brick_anomaly.py --model exports/anomalib/$RUN/fp16/weights/openvino/model.xml --benchmark CPU,GPU,NPU)"

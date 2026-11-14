#!/usr/bin/env bash
# Run a defect-pipeline module in the repo env (uv extras capture + anomalib), with the Intel GPU
# and NPU visible to Level Zero (T-008). The module name is relative to hackjunction2026.defect:
#   scripts/anomalib.sh cropping.make_dataset --name v1
#   scripts/anomalib.sh training.train_patchcore --dataset v1
#   scripts/anomalib.sh training.evaluate --device NPU
#   scripts/anomalib.sh training.compare
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ $# -ge 1 ]] || { sed -n '2,8p' "$0"; exit 2; }
GPU_DRV=/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
NPU_DRV=/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1
if [[ -e $GPU_DRV && -e $NPU_DRV ]]; then
  export ZE_ENABLE_ALT_DRIVERS="$GPU_DRV,$NPU_DRV"   # must list both, or torch.xpu breaks (T-008)
fi
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"     # big downloads on venue Wi-Fi (T-010)
MODULE="$1"; shift
exec uv run --extra capture --extra anomalib python -m "hackjunction2026.defect.$MODULE" "$@"

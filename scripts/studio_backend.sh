#!/usr/bin/env bash
# Start the Physical AI Studio backend (http://localhost:7860) as in JUNCTION2026.md,
# with both the Arc iGPU and the NPU visible to Level Zero (see T-008).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/third_party/physical-ai-studio/application/backend"
[ -f .env ] || cp .env.example .env

# The NPU user-space driver ships only in the intel-npu-driver snap. ZE_ENABLE_ALT_DRIVERS
# replaces the default driver list, so the GPU driver must be listed too or torch.xpu breaks.
GPU_DRV=/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
NPU_DRV=/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1
if [[ -e $GPU_DRV && -e $NPU_DRV ]]; then
  export ZE_ENABLE_ALT_DRIVERS="$GPU_DRV,$NPU_DRV"
fi

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"  # venue Wi-Fi is slow (T-010)
exec uv run --extra xpu physicalai-studio serve

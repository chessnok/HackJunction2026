#!/usr/bin/env bash
# Stop Physical AI Studio (backend + UI) and clean up orphaned worker processes (T-012).
# The backend often ignores SIGTERM and its camera publisher workers survive it, keeping
# /dev/video* busy; this sends SIGTERM, waits, then SIGKILLs whatever is left.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/third_party/physical-ai-studio/application/backend/.venv"

pids() {
  { pgrep -f "physicalai-studio serve"; pgrep -f "rsbuild dev"; pgrep -f "$VENV/bin/python"; } 2>/dev/null \
    | grep -vx "$$" | sort -u
}

targets=$(pids)
if [[ -z $targets ]]; then
  echo "Studio is not running"
  exit 0
fi
echo "stopping: $(echo $targets)"
kill $targets 2>/dev/null
for _ in $(seq 1 15); do
  [[ -z $(pids) ]] && break
  sleep 1
done
left=$(pids)
if [[ -n $left ]]; then
  echo "force-killing (ignored SIGTERM): $(echo $left)"
  kill -9 $left 2>/dev/null
  sleep 1
fi
[[ -z $(pids) ]] || { echo "still running: $(echo $(pids))"; exit 1; }
echo "Studio stopped"

# Killed camera publishers leave their iceoryx2 services behind; the next Studio then "finds" a
# publisher that does not exist and every camera preview fails with
# "CaptureTimeoutError: no publisher responded within timeout" (T-013). With no iceoryx2 user left,
# wipe its state.
if grep -qls iceoryx2 /proc/[0-9]*/maps 2>/dev/null; then
  echo "iceoryx2 still in use by another process; leaving its state alone"
else
  rm -rf /tmp/iceoryx2 && rm -f /dev/shm/iox2_* && echo "iceoryx2 state cleaned"
fi

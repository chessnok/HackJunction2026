#!/usr/bin/env bash
# Local vision-language model server: OpenVINO Model Server (OVMS), native, on the Arc iGPU (ADR 0005).
# OpenAI-compatible API on http://127.0.0.1:${VLM_PORT:-8000}/v3 (chat/completions, models).
#
#   scripts/vlm_server.sh start [MODEL]   # MODEL = dir under ~/models/ovms/OpenVINO (default below)
#   scripts/vlm_server.sh stop
#   scripts/vlm_server.sh status          # pid, served model, RAM (RSS + GPU buffers), watchdog
#   scripts/vlm_server.sh log             # follow the server log
#
# Built for a laptop that records datasets with Physical AI Studio at the same time:
#  - nice 19, pinned to VLM_CPUS (default 4 E-cores), OMP/TBB threads limited;
#  - refuses to start without enough free RAM (model size + VLM_START_HEADROOM_GB);
#  - oom_score_adj 1000, so the kernel OOM killer takes OVMS before Studio;
#  - a watchdog kills OVMS when MemAvailable drops below VLM_MIN_AVAIL_MB;
#  - compiled models are cached in VLM_CACHE_DIR, so restarts skip the GPU compile.
# OVMS install: ~/opt/ovms/current (tarball, see README "Local VLM"); libxml2 shim: T-016.
set -euo pipefail

OVMS_HOME="${OVMS_HOME:-$HOME/opt/ovms/current}"
MODELS="${VLM_MODELS:-$HOME/models/ovms/OpenVINO}"
STATE="${VLM_STATE_DIR:-$HOME/.local/state/hj-vlm}"
PORT="${VLM_PORT:-8000}"
DEVICE="${VLM_DEVICE:-GPU}"
CPUS="${VLM_CPUS:-8-11}"
KV_CACHE_GB="${VLM_KV_CACHE_GB:-2}"
MAX_SEQS="${VLM_MAX_SEQS:-4}"
CACHE_DIR="${VLM_CACHE_DIR:-$HOME/models/ovms/.ov_cache}"
MIN_AVAIL_MB="${VLM_MIN_AVAIL_MB:-4096}"
START_HEADROOM_GB="${VLM_START_HEADROOM_GB:-7}"
DEFAULT_MODEL="${VLM_MODEL:-Qwen3-VL-8B-Instruct-int8-ov}"
PIDFILE="$STATE/ovms.pid"
WDFILE="$STATE/watchdog.pid"
LOG="$STATE/ovms.log"
STATS="$STATE/memstats.log"

mem_avail_mb() { awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo; }
alive() { [[ -f $1 ]] && kill -0 "$(cat "$1")" 2>/dev/null; }
ready() { curl -sf "http://127.0.0.1:$PORT/v2/models/$1/ready" >/dev/null 2>&1; }

# GPU buffer memory of a process from the xe driver's fdinfo (drm-total-<region>, once per DRM
# client), in MB. On the iGPU this is system RAM that does not show up in RSS.
gpu_mb() {
  local pid=$1
  cat /proc/"$pid"/fdinfo/* 2>/dev/null | awk '
    /^drm-client-id:/ {id=$2}
    /^drm-total-(gtt|system|stolen|vram[0-9]*):/ {
      v=$2; u=$3; kb=(u=="MiB")?v*1024:(u=="GiB")?v*1048576:(u=="KiB")?v:v/1024; t[id SUBSEP $1]=kb}
    END {s=0; for (k in t) s+=t[k]; print int(s/1024)}'
}

status() {
  if ! alive "$PIDFILE"; then
    echo "VLM server: not running"
    return 1
  fi
  local pid; pid=$(cat "$PIDFILE")
  local rss hwm
  rss=$(awk '/^VmRSS:/ {print int($2/1024)}' /proc/"$pid"/status)
  hwm=$(awk '/^VmHWM:/ {print int($2/1024)}' /proc/"$pid"/status)
  echo "VLM server: pid $pid, model $(cat "$STATE/model"), http://127.0.0.1:$PORT/v3"
  echo "  RAM: RSS ${rss} MB (peak ${hwm} MB) + GPU buffers $(gpu_mb "$pid") MB; MemAvailable $(mem_avail_mb) MB"
  if alive "$WDFILE"; then
    echo "  watchdog: pid $(cat "$WDFILE"), kills OVMS below ${MIN_AVAIL_MB} MB available"
  else
    echo "  watchdog: NOT running"
  fi
  if ready "$(cat "$STATE/model")"; then echo "  API: ready"; else echo "  API: not ready (loading?)"; fi
}

# Runs detached next to OVMS: kills it when RAM gets low, logs RSS / GPU / MemAvailable every 2 s
# while loading and every 30 s afterwards.
watchdog() {
  local pid=$1 n=0 avail
  while kill -0 "$pid" 2>/dev/null; do
    avail=$(mem_avail_mb)
    if (( avail < MIN_AVAIL_MB )); then
      echo "$(date '+%F %T') MemAvailable ${avail} MB < ${MIN_AVAIL_MB} MB: killing OVMS $pid" | tee -a "$STATS" >>"$LOG"
      kill -9 "$pid" 2>/dev/null || true
      break
    fi
    if (( n < 300 || n % 15 == 0 )); then
      echo "$(date '+%F %T') avail_mb=$avail rss_mb=$(awk '/^VmRSS:/ {print int($2/1024)}' /proc/"$pid"/status 2>/dev/null) hwm_mb=$(awk '/^VmHWM:/ {print int($2/1024)}' /proc/"$pid"/status 2>/dev/null) gpu_mb=$(gpu_mb "$pid")" >>"$STATS"
    fi
    n=$((n + 1))
    sleep 2
  done
  rm -f "$WDFILE"
}

start() {
  local model=${1:-$DEFAULT_MODEL}
  local path="$MODELS/$model"
  mkdir -p "$STATE" "$CACHE_DIR"
  if alive "$PIDFILE"; then
    echo "VLM server already running (pid $(cat "$PIDFILE"), model $(cat "$STATE/model")); stop it first: $0 stop" >&2
    exit 1
  fi
  if pgrep -x ovms >/dev/null; then
    echo "an ovms process is already running (pid $(pgrep -x ovms | tr '\n' ' ')); not starting a second one" >&2
    exit 1
  fi
  if ss -Hltn "sport = :$PORT" | grep -q .; then
    echo "port $PORT is already in use:" >&2; ss -ltnp "sport = :$PORT" >&2; exit 1
  fi
  [[ -x $OVMS_HOME/bin/ovms ]] || { echo "OVMS not found in $OVMS_HOME (see README: Local VLM)" >&2; exit 1; }
  [[ -f $path/openvino_language_model.xml ]] || { echo "model not found or incomplete: $path" >&2; exit 1; }
  if compgen -G "$path/*.part" >/dev/null; then echo "model still downloading: $path" >&2; exit 1; fi

  local size_mb need_mb avail
  size_mb=$(du -sm --exclude='*.part' "$path" | cut -f1)
  need_mb=$(( size_mb + START_HEADROOM_GB * 1024 ))
  avail=$(mem_avail_mb)
  if (( avail < need_mb )); then
    echo "not enough free RAM: ${avail} MB available, need ${need_mb} MB (model ${size_mb} MB + ${START_HEADROOM_GB} GB headroom)" >&2
    exit 1
  fi

  # Same driver list as Studio (T-008): replacing it without the GPU driver breaks the GPU.
  local gpu_drv=/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
  local npu_drv=/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1
  if [[ -e $gpu_drv && -e $npu_drv ]]; then export ZE_ENABLE_ALT_DRIVERS="$gpu_drv,$npu_drv"; fi
  # libxml2.so.2 shim for the Ubuntu 24 build on Ubuntu 26.04 (T-016).
  export LD_LIBRARY_PATH="$OVMS_HOME/lib:$(dirname "$OVMS_HOME")/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export OMP_NUM_THREADS=4 TBB_NUM_THREADS=4

  echo "$model" >"$STATE/model"
  : >"$STATS"
  echo "$(date '+%F %T') start $model on $DEVICE, cpus $CPUS, MemAvailable ${avail} MB" | tee -a "$LOG"
  # VLM_CB = continuous-batching VLM pipeline: paged KV cache of fixed size (--cache_size GB), several
  # clients at once (brick judge + localization). Prefix caching off: every request has new frames.
  setsid nohup choom -n 1000 -- nice -n 19 taskset -c "$CPUS" "$OVMS_HOME/bin/ovms" \
    --rest_port "$PORT" --rest_bind_address 127.0.0.1 --rest_workers 4 \
    --model_name "$model" --model_path "$path" \
    --task text_generation --pipeline_type VLM_CB --target_device "$DEVICE" \
    --cache_size "$KV_CACHE_GB" --max_num_seqs "$MAX_SEQS" --enable_prefix_caching false \
    --cache_dir "$CACHE_DIR" --log_level INFO \
    >>"$LOG" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$PIDFILE"
  setsid nohup nice -n 19 "$0" _watchdog "$pid" >/dev/null 2>&1 </dev/null &
  echo $! >"$WDFILE"

  echo "OVMS pid $pid; loading (log: $LOG) ..."
  local t0=$SECONDS
  until ready "$model"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "OVMS exited during load; last log lines:" >&2; tail -20 "$LOG" >&2; rm -f "$PIDFILE"; exit 1
    fi
    sleep 2
  done
  echo "ready after $((SECONDS - t0)) s: model '$model' on http://127.0.0.1:$PORT/v3" | tee -a "$LOG"
  status
}

stop() {
  if ! alive "$PIDFILE"; then
    echo "VLM server is not running"
    rm -f "$PIDFILE"
    return 0
  fi
  local pid; pid=$(cat "$PIDFILE")
  echo "stopping OVMS $pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "force-killing $pid"; kill -9 "$pid" 2>/dev/null || true; fi
  alive "$WDFILE" && kill "$(cat "$WDFILE")" 2>/dev/null || true
  rm -f "$PIDFILE" "$WDFILE"
  echo "$(date '+%F %T') stopped" >>"$LOG"
  echo "VLM server stopped"
}

case "${1:-}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  status) status ;;
  log) tail -f "$LOG" ;;
  _watchdog) watchdog "$2" ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac

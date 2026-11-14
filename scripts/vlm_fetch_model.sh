#!/usr/bin/env bash
# Download OpenVINO IR models from Hugging Face (org OpenVINO) for scripts/vlm_server.sh (ADR 0005).
#
#   scripts/vlm_fetch_model.sh Qwen3-VL-8B-Instruct-int8-ov [more models ...]
#
# -> ~/models/ovms/OpenVINO/<model>/. Plain curl, one file at a time, niced, resumable (re-run after
# an interruption), every file checked against the HF API size and LFS sha256. Used instead of
# `hf download` because its xet backend grew to >1.2 GB RSS while writing almost nothing to disk on
# the venue Wi-Fi (laptop RAM is shared with Studio).
set -euo pipefail
ROOT="${VLM_MODELS:-$HOME/models/ovms/OpenVINO}"
[[ $# -ge 1 ]] || { sed -n '2,9p' "$0"; exit 2; }

for model in "$@"; do
  dir="$ROOT/$model"
  mkdir -p "$dir"
  echo "$(date +%T) $model -> $dir"
  curl -sSf "https://huggingface.co/api/models/OpenVINO/$model/tree/main" >"$dir/.tree.json"
  python3 -I -c '
import json, sys
for f in json.load(open(sys.argv[1])):
    if f["type"] == "file":
        print(f["path"], f["size"], (f.get("lfs") or {}).get("oid", "-"))
' "$dir/.tree.json" | sort -k2 -n | while read -r path size oid; do
    out="$dir/$path"
    if [[ -f $out && $(stat -c %s "$out") == "$size" ]]; then continue; fi
    for try in 1 2 3 4 5 6 7 8 9 10; do
      [[ -f $out.part && $(stat -c %s "$out.part") == "$size" ]] && break  # finished earlier
      nice -n 19 curl -sSL --retry 5 --retry-delay 5 -C - -o "$out.part" \
        "https://huggingface.co/OpenVINO/$model/resolve/main/$path" && break
      echo "  retry $try: $path"
      sleep 10
    done
    got=$(stat -c %s "$out.part" 2>/dev/null || echo 0)
    [[ $got == "$size" ]] || { echo "  FAILED $path: $got of $size bytes (re-run to resume)" >&2; exit 1; }
    if [[ $oid != "-" && $(nice -n 19 sha256sum "$out.part" | cut -d' ' -f1) != "$oid" ]]; then
      rm -f "$out.part"
      echo "  FAILED $path: sha256 mismatch (deleted, re-run)" >&2
      exit 1
    fi
    mv "$out.part" "$out"
    echo "  ok $path ($size bytes)"
  done
  echo "$(date +%T) done $model ($(du -sh "$dir" | cut -f1))"
done

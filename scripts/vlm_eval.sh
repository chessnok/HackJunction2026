#!/usr/bin/env bash
# Evaluate the model served by scripts/vlm_server.sh with scripts/brick_judge.py (ADR 0005, ADR 0008).
#
#   scripts/vlm_eval.sh <tag> [run ...]      # results in captures/vlm/eval/<tag>/<run>.{jsonl,txt}
#   uv run --no-project python scripts/vlm_eval_report.py captures/vlm/eval/<tag>/*.jsonl
#
# Runs (default: all): crops-multi-v3 crops-multi-verbatim-sub crops-or-v3 frames-multi-v3-sub
# both-multi-v3-sub locate-vlm. Scenes: the frozen list captures/vlm/eval/eval_set.txt (Studio snapshot
# ids) plus the SYNTHETIC marked bricks in captures/vlm/synthetic/*-synth; "-sub" runs use every 3rd
# snapshot, locate-vlm every 4th. v3 = configs/brick_judge_prompt_explicit.txt, verbatim = the default.
# Loads the iGPU for an hour: only when nobody records with Studio (ADR 0005). --pause 0.5 between scenes.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TAG=${1:?usage: scripts/vlm_eval.sh <tag> [run ...]}; shift
RUNS=${*:-crops-multi-v3 crops-multi-verbatim-sub crops-or-v3 frames-multi-v3-sub both-multi-v3-sub locate-vlm}
O=captures/vlm/eval/$TAG
mkdir -p "$O"
E=configs/brick_judge_prompt_explicit.txt
J=(nice -n 19 uv run scripts/brick_judge.py --pause 0.5 --eval)
SYN=(captures/vlm/synthetic/*-synth)
[[ -d ${SYN[0]} ]] || SYN=()
LIST=captures/vlm/eval/eval_set.txt
[[ -f $LIST ]] || { echo "missing $LIST (one snapshot id per line)" >&2; exit 1; }
mapfile -t ALL < <(sed "s/^/--snapshot\n/" "$LIST")
mapfile -t SUB < <(awk 'NR%3==1' "$LIST" | sed "s/^/--snapshot\n/")
mapfile -t LOC < <(awk 'NR%4==1' "$LIST" | sed "s/^/--snapshot\n/")
for r in $RUNS; do
  case $r in
    crops-multi-v3) args=("${ALL[@]}" "${SYN[@]}" --prompt-file "$E") ;;
    crops-multi-verbatim-sub) args=("${SUB[@]}") ;;
    crops-or-v3) args=("${ALL[@]}" "${SYN[@]}" --prompt-file "$E" --mode per-image) ;;
    frames-multi-v3-sub) args=("${SUB[@]}" --view frames --prompt-file "$E") ;;
    both-multi-v3-sub) args=("${SUB[@]}" --view both --prompt-file "$E") ;;
    locate-vlm) args=("${LOC[@]}" --locator vlm --prompt-file "$E") ;;
    *) echo "unknown run $r" >&2; exit 2 ;;
  esac
  "${J[@]}" "${args[@]}" --out "$O/$r.jsonl" >"$O/$r.txt" 2>&1
  echo "$(date +%T) $TAG $r: $(grep -E 'accuracy|false-positive|recall' "$O/$r.txt" | tr '\n' ' ')"
done

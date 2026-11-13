#!/usr/bin/env bash
# Start the Physical AI Studio UI (http://localhost:3000) as in JUNCTION2026.md.
# Needs Node >= 24.2 and npm >= 11.14 (we use Node 26 via nvm, see ADR 0003).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/third_party/physical-ai-studio/application/ui"
# shellcheck disable=SC1091
[[ -s $HOME/.nvm/nvm.sh ]] && source "$HOME/.nvm/nvm.sh"
[[ -d node_modules ]] || npm install --no-fund --no-audit
exec npm run start

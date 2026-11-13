# 0003. Run Studio natively from upstream lockfiles, not Docker or PyPI

- Status: Accepted
- Date: 2026-11-13
- Deciders: team

## Context

- The guide recommends running Studio natively (backend `uv run --extra xpu physicalai-studio serve`,
  UI `npm run start`). Docker is not installed on the laptop.
- `pip`/`uv pip install "physicalai-train[xpu]"` from PyPI with `--torch-backend=xpu` cannot
  resolve version 0.3.0: it pins `lerobot[dataset]==0.6.0`, which needs `torchcodec<0.12`, while
  the PyTorch XPU index only has `torchcodec>=0.13`. The resolver silently fell back to
  `physicalai-train 0.1.0` (T-007).
- Upstream solves this in its `pyproject.toml` / `uv.lock` with several indexes,
  `index-strategy = "unsafe-best-match"` and git sources (e.g. `physicalai` from a pinned commit).
- The backend needs Python 3.13. The UI needs Node >= 24.2.0 and npm >= 11.14.0. Node v24.2.0
  from `.nvmrc` ships an older npm, so we use Node 26.13.1 (npm 11.20.0) via nvm, which satisfies
  `engines`.

## Decision

Clone the `junction2026` branch into `third_party/physical-ai-studio` (not committed) and install
with upstream's lockfiles: `uv sync --extra xpu` in `application/backend`, `npm install` in
`application/ui`. Never `pip install` into those environments. Anomalib gets its own venv.

## Alternatives considered

- Docker (`setup-devices.sh --xpu && docker compose up -d`): needs Docker and sudo; fine as a
  fallback.
- PyPI install: does not resolve with XPU (above).

## Consequences

- `git pull` in `third_party/physical-ai-studio` updates Studio; record the commit SHA in the log
  whenever a known-good state is reached.
- System libraries must be installed by a human with sudo (README Setup step 1).

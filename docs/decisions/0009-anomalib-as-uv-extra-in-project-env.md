# 0009. Install Anomalib as a uv extra of this project, not a separate venv

- Status: Accepted
- Date: 2026-11-14
- Deciders: team (requested by Ivan)

## Context

AGENTS.md said "Anomalib gets its own venv (its PyTorch must not clash with Studio's)". The clash
that matters is with Studio's backend env (`third_party/physical-ai-studio/application/backend`,
its own uv lock, torch 2.11.0+xpu). Our training code (`hackjunction2026.defect.training`) imports
both Anomalib and our own package, and a hand-made `anomalib-env` has no lock file.
Slide 8 requires Anomalib v2.7.0.

## Decision

Anomalib is the `anomalib` extra in `pyproject.toml`: `anomalib[openvino,xpu]==2.7.0` plus torch,
torchvision and triton-xpu listed directly so `[tool.uv.sources]` can take them from the PyTorch
XPU index (uv sources only apply to direct dependencies). The lock is limited to Linux
(`[tool.uv] environments`). `pytorch-triton-xpu` (old triton for torch <= 2.9, listed by
anomalib[xpu]) is excluded with an override because torch 2.15+xpu uses `triton-xpu` and both
install a `triton` package. Install: `uv sync --extra capture --extra anomalib`. Studio's env is
untouched (scripts call it with `uv run --no-sync`).

Locked 2026-11-14: anomalib 2.7.0, torch 2.15.0+xpu, torchvision 0.29.2+xpu, triton-xpu 3.9.0,
openvino 2026.4.1 (same as Studio's), nncf 3.4.0, lightning 2.6.6, numpy 2.4.6.

## Alternatives considered

- Separate `anomalib-env` (`uv pip install --torch-backend=xpu ...`): no lock, a second interpreter
  in PyCharm, the project package must be put on PYTHONPATH by hand.
- Installing Anomalib into Studio's env: forbidden by ADR 0003 (never pip install into it), and its
  torch pin differs.

## Consequences

- One reproducible env (`uv.lock`) for capture + training; `.venv` is about 8 GB with the XPU stack.
- numpy is capped below 2.5 by nncf (`numpy>=2.5.4` was briefly added by an IDE quick-fix and
  broke the lock).
- On machines without an Intel GPU, torch.xpu is unavailable and training falls back to CPU.
- Verified on a Fedora dev machine (CPU only). **Untested on the station laptop's Arc iGPU / NPU.**

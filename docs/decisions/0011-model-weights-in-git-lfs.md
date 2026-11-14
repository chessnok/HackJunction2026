# 0011. Keep released model weights and annotations in git LFS under `artifacts/`

- Status: Accepted
- Date: 2026-11-14
- Deciders: team lead (chessnok), agent

## Context

- AGENTS.md said never to commit checkpoints or exported models. Trained weights lived only on the robot laptop: Studio's models directory, `exports/anomalib/`, and Modal volumes.
- The team works from several machines. The user asked to put the weights and everything needed in the repo through git LFS.
- Sizes on 2026-11-14:
  - ACT (Modal job 7b07e8c5): OpenVINO 133 MB, torch export 198 MB, Lightning checkpoint 591 MB.
  - Anomalib PatchCore `bricks_z14`: fp16 54 MB, int8 39 MB.
  - Perception parquet and JSON: about 6 MB.
  - VLMs (OVMS) are public on Hugging Face (`OpenVINO/...`).
- GitHub's free LFS quota is limited, so the exact current limits need checking.

## Decision

- Released weights and derived annotations go to `artifacts/`, tracked by git LFS (`.gitattributes`).
- `artifacts/MANIFEST.md` records, per file: source run, metrics, size, how to load.
- What goes in:
  - inference exports only (OpenVINO IR, plus the torch export for ACT);
  - Anomalib exports with their deploy config;
  - perception annotations (parquet, JSON).
- What stays out:
  - training checkpoints with optimizer state (the 591 MB `model.ckpt`) and datasets;
  - camera frames and crops, including any image of people;
  - public VLMs: the manifest names their Hugging Face ids instead.

## Alternatives considered

- **Hugging Face Hub private repo:** good for weights, but a second place with its own token, and the team already uses GitHub.
- **Modal volume or a shared drive:** not versioned with the code, so the link between a model and its commit gets lost.
- **Keep everything local:** this was the status quo; it blocks other machines.

## Consequences

- Clones need `git lfs install`; `git lfs pull` fetches the files.
- Keep the total small. Remove superseded weights from the manifest and from `artifacts/`; history still counts toward quota.
- Revisit if the quota is hit: move to an HF private repo and keep the manifest here.

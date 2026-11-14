# Per-crop marker labels

`gemini_crop_marks.csv`: one row per brick crop of a Studio state snapshot (project
`0fd76ff2-a6b1-4472-b687-486befe329be`), one crop per camera.

- `brick_label`: the snapshot's own `defective` flag, set by a human at capture.
- `gemini_mark_visible`: the Gemini answer to "is a black marker mark visible on the brick in THIS crop?". A mark is often visible from one camera only.
- Model: `gemini-3.1-pro-preview` (via `gemini-pro-latest`). A re-label with `gemini-4-argon` is in progress.
- Script: `scripts/label_crops_gemini.py`. Crops come from `scripts/brick_crops.py` (dataset `bricks_z14`).

Human review of these labels: the private page https://claude.ai/artifact/J3vgfu6cLV1oRrvTXXhUoD (owner shares it).
Reviews are stored in the page's database (collection `reviews`, one document per `crop_id`):
`verdict` is `agree`, `disagree` or `unsure`, plus `note`. Export them with the ArtifactData tool, then merge into
`configs/defect_mark_labels.csv` (`manual_mark_visible`), which overrides Gemini.

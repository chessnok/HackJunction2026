# 0005. Serve a local VLM with OpenVINO Model Server on the iGPU

- Status: Accepted
- Date: 2026-11-14
- Deciders: team (request), agent (measurements)

## Context

- The team wants a local vision-language model that answers, for the photos of one scene (Studio
  state snapshot: `top`, `floor`, `arm`), the fixed prompt "Consider a brick broken, if and only if
  on any of the photos u see it is clearly marked with black marker. Is this brick broken?" as JSON
  `{"is_broken": boolean}`, enforced by constrained decoding, not parsed from free text. A second
  agent uses the same server for object localization (brick box / point as JSON).
- The request was "OpenVINO + Ollama + a medium model that fits in about 32 GB of RAM".
- Laptop: Core Ultra X9 388H, Arc B390 iGPU (shares the 61 GB system RAM), NPU, Ubuntu 26.04.
  Studio records datasets at the same time: about 40 GB available, and CPU load has already made
  Studio discard episodes.
- Ollama's OpenVINO backend (`openvino_contrib/modules/ollama_openvino`) is text-only (multimodal is
  "future development") and ships stale binaries (OpenVINO GenAI 2025.3 dev, Ubuntu 22, Google Drive).
- OpenVINO Model Server (OVMS) is Intel's equivalent: native tarball, OpenAI-compatible
  `/v3/chat/completions`, VLM pipelines, `response_format` json_schema (XGrammar). Latest release
  v2026.4.1 has Ubuntu 22 / 24 builds only.
- How a brick is judged (crops, OR over cameras and detectors) is ADR 0008; this ADR is only about
  the VLM server and the model.

## Measurements (2026-11-14, laptop clock 05:20–09:00)

Setup: OVMS 2026.4.1 `ubuntu24_python_off` tarball on Ubuntu 26.04 with a `libxml2.so.2` shim (T-016),
`--pipeline_type VLM_CB --target_device GPU --cache_size 2`, nice 19, pinned to 4 E-cores. All
latencies were measured while the perception agent used the same server and other agents loaded the
CPU (load average 10–14); expect lower numbers on an idle laptop. Reproduce: `scripts/vlm_eval.sh`,
`scripts/vlm_eval_report.py`.

- Constrained decoding works: a "describe the photo" request with the schema `{"zq_flag": integer 7..9}`
  answered `{"zq_flag": 7}` (Qwen) / `{"zq_flag": 8}` (gemma). `usage.prompt_tokens` counts text only.
- Qwen3-VL-8B-Instruct-int8-ov (8.2 GB on disk): ready 36 s after start (first compile; cache in
  `~/models/ovms/.ov_cache`); peak RSS 3.1 GB; steady RSS 0.9–1.7 GB plus 8.6–12.0 GB of GPU buffers
  (system RAM on the iGPU, includes the 2 GB KV cache); MemAvailable 42.5 → about 31 GB. Stop: 2 s.
  Text decode 4.1 tokens/s (GPU at 1300 of 2500 MHz).
- gemma-4-26b-a4b-it-int4-ov (15 GB): ready 34 s (25 s from cache) after a chat-template fix (T-021);
  peak RSS 2.5 GB, steady 1.3 GB plus 16.1 GB GPU buffers; MemAvailable 42.5 → 32 GB.
- Eval set: all 108 labelled Studio snapshots of project `0fd76ff2` (50 defective, 58 good; list in
  `captures/vlm/eval/eval_set.txt`, from `snapshots.csv` at 08:05) plus 20 SYNTHETIC positives (one black
  stroke, 1.5–4.5 px, drawn into one camera of a real negative; `captures/vlm/synthetic/`). Crops: colour
  search on the sheet, 2.4× brick size, 448 px (ADR 0008). Temperature 0. Brick-level numbers:

| model | prompt | images per request | mode | latency (median) | FP, good bricks | defective found | synthetic found |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B int8 | verbatim | 3 crops | one request | 3.9 s | 0/58 | 15/50 | 2/20 |
| Qwen3-VL-8B int8 | explicit | 3 crops | one request | 2.4 s | 0/58 | 38/50 | 19/20 |
| **Qwen3-VL-8B int8** | **explicit** | **1 crop × 3** | **per crop, OR (ADR 0008)** | **2.0 s per crop, 6.4 s per brick** | **3/58** | **43/50** | **20/20** |
| Qwen3-VL-8B int8 | explicit | 3 full frames | one request | 11.9 s | 0/58 | 24/50 | 9/20 |
| Qwen3-VL-8B int8 | explicit | 3 frames + 3 crops | one request | 14.3 s | 0/19 | 12/17 (crops alone on these 36: 13/17, 2.5 s) | – |
| gemma-4-26b-a4b int4 | explicit | 1 crop × 3 | per crop, OR (3 crops in parallel) | 3.3 s per brick | 0/58 | 14/50 (per crop 17/99; + PatchCore 40/50) | not run (time) |
| gemma-4-31B int4 / int8 | | | | not evaluated (time: venue Wi-Fi 1–5 MB/s, download stopped) | | | |

- Latency vs images (Qwen, one request): 1 crop 2.0 s, 3 crops 2.4 s, 3 full frames 11.9 s, 3 frames +
  3 crops 14.3 s. Image tokens, not image count, set the cost: a 448 px crop is about 200 visual tokens,
  an 848×480 / 1024×768 frame about 400 / 770.
- The team's verbatim prompt fails with Qwen3-VL: in free text the model describes the marks ("a
  smiley face", "a question mark") and then says they are "not black marker marks". An observation
  field before `is_broken` (`--reason`) did not help (1/37 on a probe set). The explicit prompt
  (`configs/brick_judge_prompt_explicit.txt`: a mark is a line, arc, cross, dot or scribble; studs,
  holes, shadows, edges are not) fixes it.
- Misses are mostly blue bricks (black on blue, low contrast). The 3 false positives of the per-crop OR
  are all wrist crops with a black speck on the paper next to the brick (clean the sheet).
- Per crop (Qwen, per-crop OR), against the crop labels (Gemini `gemini-3.1-pro-preview` in
  `captures/anomalib/dataset/bricks_z14/index.csv`, overridden by `configs/defect_mark_labels.csv`):
  marked crops found 73/99 (top 26/34, floor 30/37, wrist 17/28); false positives on good-brick crops
  1/145 (+2 on wrist crops the Anomalib index has no label for: 3/174 overall); on unmarked crops of
  defective bricks 0/49.
- Union with Anomalib PatchCore (`20261114-070921_patchcore_bricks_z14`, max crop score per brick,
  threshold = highest of 29 held-out good bricks, so 0 FP there): per-crop OR Qwen + PatchCore finds
  47/50 defective bricks (35 both, 8 only Qwen, 4 only PatchCore, 3 neither).
- Grounding (Qwen, 81 boxes on 27 snapshots, against the colour-search boxes checked by eye): answer
  `bbox_2d` = [x1, y1, x2, y2] normalized to 0–1000; IoU median 0.91 / 0.87 / 0.91 (top / floor / wrist),
  every box centre inside the reference box, centre error median 1.8–2.8 px; about 5.5 s per box.

## Decision

Serve the VLM with OVMS 2026.4.1 natively (`scripts/vlm_server.sh`, one model, REST on
127.0.0.1:8000) on the iGPU, with **Qwen3-VL-8B-Instruct-int8-ov**. Judge each brick with
`scripts/brick_judge.py --prompt-file configs/brick_judge_prompt_explicit.txt --mode per-image`
(colour crops, one request per camera crop, OR, ADR 0008): 43/50 defective found, 3/58 false positives,
about 6.4 s per brick (2.0 s per crop; `--parallel` sends the three at once). gemma-4-26b-a4b in the
same mode found only 14/50. Qwen also gives accurate boxes for localization. The team's verbatim prompt stays the script's default only because it was asked
for; it finds 15/50.

## Alternatives considered

- Ollama with the OpenVINO backend: no image input; stale binaries. Rejected.
- Ollama (llama.cpp) without OpenVINO: not what was asked (OpenVINO on the Intel iGPU), and its
  GPU path on Xe3 + Ubuntu 26.04 is unproven here.
- `openvino-genai` `VLMPipeline` + `StructuredOutputConfig` in an own venv behind a small HTTP
  wrapper: the planned fallback; not needed because OVMS runs.
- Docker OVMS image: the team runs everything natively (ADR 0003).
- Parsing a yes/no out of free text: asked against; the schema is enforced by the server.

## Consequences

- One server serves both the brick judge and localization; requests from both share the GPU and
  slow each other down (continuous batching, `--max_num_seqs 4`).
- The server can never take Studio down: `oom_score_adj` 1000 and a watchdog that kills it below 4 GB
  available.
- Prompt wording matters more than the model for this task; re-check it when the model changes.
- Batch evals compete with Studio's recording: nice and CPU pinning do not limit GPU time or the
  memory bandwidth the iGPU shares with the CPU. At 07:41 Studio discarded an episode ("fell 654 ms
  behind") while the full-frame eval (about 1550 image tokens per request, back to back) ran next to
  PatchCore training (log 2026-11-14). Run batch evals only when nobody records, or with
  `brick_judge.py --pause`; one judgment per brick (3 crops, about 2.4 s) is a short burst.

## Open

- gemma-4-31B (int4 18.7 GB, int8 32.8 GB): not evaluated (time). `scripts/vlm_fetch_model.sh` resumes the
  download; T-021 applies to its chat template; then `scripts/vlm_eval.sh <tag>`.
- Blue bricks: most misses. Try a larger crop size or a colour-specific hint in the prompt.

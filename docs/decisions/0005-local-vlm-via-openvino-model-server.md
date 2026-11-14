# 0005. Serve a local VLM with OpenVINO Model Server on the iGPU

- Status: Accepted (model comparison unfinished: gemma-4 models still downloading, see "Open")
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

## Measurements (2026-11-14, laptop clock 05:20–08:00)

Setup: OVMS 2026.4.1 `ubuntu24_python_off` tarball on Ubuntu 26.04 with a `libxml2.so.2` shim (T-016),
`--pipeline_type VLM_CB --target_device GPU --cache_size 2`, nice 19, pinned to 4 E-cores. All
latencies were measured while the perception agent used the same server and other agents loaded the
CPU (load average 10–14); expect lower numbers on an idle laptop.

- Constrained decoding works: a "describe the photo" request with the schema `{"zq_flag": integer 7..9}`
  answered `{"zq_flag": 7}`. `usage.prompt_tokens` counts text tokens only.
- Qwen3-VL-8B-Instruct-int8-ov (8.2 GB on disk) on GPU: ready 36 s after start (first compile, cache in
  `~/models/ovms/.ov_cache`); peak RSS 3.1 GB; steady RSS 0.9–1.5 GB plus 8.6–11.9 GB of GPU buffers
  (system RAM on the iGPU; includes the 2 GB KV cache); MemAvailable 42.5 → about 32 GB. Text decode
  4.1 tokens/s (GPU at 1300 of 2500 MHz).
- Eval set (frozen at 06:42): 107 real Studio snapshots of project `0fd76ff2` (49 labelled defective,
  58 good; the observe-pose snapshot `055500-789` is left out: labelled good but shows a marked brick)
  plus 20 SYNTHETIC positives (one black stroke, 1.5–4.5 px, drawn into one camera of a real negative).
  Crops: colour search on the sheet, 2.4× brick size, 448 px (ADR 0008). Temperature 0.

| model | prompt | images per request | mode | latency median | FP (good snapshots) | real defective found | synthetic found |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B int8 | verbatim | 3 crops | one request | 3.9 s | 0/58 | 15/49 | 2/20 |
| Qwen3-VL-8B int8 | explicit | 3 crops | one request | 2.4 s | 0/58 | 38/49 | 19/20 |
| Qwen3-VL-8B int8 | explicit | 1 crop × 3 | per crop, OR | 1.9 s per crop, 6.4 s per scene | 3/58 | 42/49 | 20/20 |
| Qwen3-VL-8B int8 | explicit | 3 full frames | one request | about 13 s | (paused at 82/127, see Consequences) | | |
| Qwen3-VL-8B int8 | explicit | 3 frames + 3 crops | one request | (pending) | | | |
| gemma-4-26b-a4b int4 | | | | (downloading) | | | |
| gemma-4-31B int4 / int8 | | | | (not downloaded) | | | |

- The team's verbatim prompt fails with Qwen3-VL: in free text the model describes the marks ("a
  smiley face", "a question mark") and then says they are "not black marker marks". An observation
  field before `is_broken` (`--reason`) did not help (1/37 on a probe set). The explicit prompt
  (`configs/brick_judge_prompt_explicit.txt`: a mark is a line, arc, cross, dot or scribble; studs,
  holes, shadows, edges are not) fixes it.
- Misses with the explicit prompt are mostly blue bricks (black on blue, low contrast). The 3 false
  positives of the per-crop OR are all wrist crops with a black speck on the paper next to the brick.
- Per crop, scored against the Gemini crop labels (`captures/anomalib/dataset/bricks_z14/index.csv`,
  `gemini-3.1-pro-preview`): marked crops found 70/86 (top 24/28, floor 29/34, wrist 17/24); false
  positives on good-brick crops 1/145; on unmarked crops of defective bricks 2/61.
- Union with Anomalib PatchCore (`20261114-070921_patchcore_bricks_z14`, max crop score per brick,
  threshold = highest held-out good brick, so 0/30 FP): per-crop OR Qwen + PatchCore finds 46/49
  defective bricks (35 both, 7 only Qwen, 4 only PatchCore, 3 neither); one-request Qwen + PatchCore 44/49.

## Decision

Serve the VLM with OVMS 2026.4.1 natively (`scripts/vlm_server.sh`, one model, REST on
127.0.0.1:8000), on the iGPU. Current model: **Qwen3-VL-8B-Instruct-int8-ov** (works, fits with room
to spare, good grounding for the localization agent; final pick pending the gemma-4 comparison).
`scripts/brick_judge.py` keeps the team's verbatim prompt as default, but the explicit prompt file is
recommended: it raises real recall from 15/49 to 38/49 at 0 false positives.

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

- gemma-4-26b-a4b-it-int4-ov (15.3 GB) is downloading at 1–5 MB/s (venue Wi-Fi); gemma-4-31B int4 /
  int8 (18.7 / 32.8 GB) were not reachable in time. Same eval commands: README "Local VLM".
- Full-frame and frames + crops runs, the VLM grounding check, the newest (gemini-4-argon) crop labels.

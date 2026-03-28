# Non-GPT-OSS Qwen3.5 Pilot

> Status: `implemented_early` on 2026-03-26 for the first Psionic-owned
> `qwen35` pilot lane.

This document records the first explicit Psionic pilot for the Ollama
`qwen3.5:0.8b` GGUF.

The pilot row is the downloaded artifact at:

- default path:
  `/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf`
- model digest:
  `afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5`
- chat-template digest:
  `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`

## Current Lane

Psionic currently supports this row through a bounded native `qwen35` CUDA lane:

- the GGUF is admitted as `qwen35`, not mislabeled as `qwen2`
- the runtime is a Psionic-owned CUDA text-generation path
- the shipped pilot is GPU-first, not a subprocess proxy
- image and video inputs are accepted through prompt projection onto the real
  `qwen35` marker surface
- the row publishes truthful prompt-projection posture for multimodal inputs
- the row still refuses tools
- structured outputs are supported on the native lane through a bounded
  `top_k_candidates:128` path for greedy no-penalty requests, with exact sparse
  allowed-logit gather on candidate misses
- structured requests outside that bounded envelope still fall back to
  explicit `raw_logits`
- structured-output candidate misses no longer replay from a cloned pre-step
  qwen35 state or double-advance the live decode state
- structured-output append caches now bucket token ids by leading char so
  sparse schema recovery does not linearly rescan the whole vocabulary on
  candidate misses
- session reuse, adapter serving, and prefix caching are still refused on this
  early lane

The source artifact itself is not text-only. It carries:

- `qwen35.vision.*` metadata
- vision token ids
- a multimodal-aware chat template

Psionic now uses those facts to derive one bounded multimodal projection config
and to project image and video request parts into the real Qwen3.5 template
markers:

- image: `<|vision_start|><|image_pad|><|vision_end|>`
- video: `<|vision_start|><|video_pad|><|vision_end|>`

The pilot still does not claim a native image or video encoder.

## Canonical Runner

Run the pilot from the repo root:

```bash
scripts/release/check-psionic-qwen35-pilot.sh
```

Optional override when the artifact lives somewhere else:

```bash
PSIONIC_QWEN35_PILOT_GGUF_PATH=/abs/path/to/qwen3.5-0.8b.gguf \
  scripts/release/check-psionic-qwen35-pilot.sh
```

## What The Runner Proves

The runner executes two evidence layers:

1. `psionic-models` artifact and fixture evidence
   - real `qwen35` tokenizer facts from the downloaded GGUF
   - real `qwen35` prompt-template digest and render cases
   - real `qwen35` multimodal projection config from GGUF family facts
   - synthetic tiny `qwen35` loader and hybrid-layout coverage
2. `psionic-serve` execution evidence
   - direct qwen35 native CUDA execution on a deterministic tiny GGUF
   - generic-server publication and request execution
   - `/v1/chat/completions` image and video projection through real Qwen markers
   - `/v1/responses` image projection through the same prompt surface
   - system-message media refusal that matches the real template rules

## Pass Criteria

The pilot is green only if all of the following remain true:

- the real downloaded row still loads as `qwen35`
- the real downloaded row still exposes the expected tokenizer and template
  facts
- the tiny deterministic qwen35 GGUF still traverses the Psionic native CUDA runtime
- the generic server still publishes:
  - `backend = cuda`
  - `execution_mode = native`
  - `execution_engine = psionic`
  - `residency_mode = cuda_accelerated`
  - `fallback_policy = refuse`
- the generic server still publishes:
  - `multimodal_projection_mode = prompt_projection_only`
  - `multimodal_supported_media = ["image", "video"]`
  - the derived `qwen35` multimodal projection config
- the generic server still exposes prompt-replay response-state support on
  `/v1/responses`
- the generic server still accepts image and video request parts through prompt
  projection on `/v1/chat/completions` and `/v1/responses`
- the generic server still refuses system-message image and video parts to stay
  aligned with the real template semantics

## Current Limitations

The pilot is intentionally bounded:

- it is still an early native Psionic CUDA execution slice
- it does not claim native multimodal inference
- it does not claim a native image or video encoder
- it now claims bounded generic-server tool calling on native Psionic CUDA for
  `/v1/chat/completions` with `none`, `auto`, `required`, and named tool
  choice, ordered `tool_calls`, and request-level `parallel_tool_calls`
- it does not claim full structured-output acceleration; the bounded native
  lane is still limited to greedy no-penalty requests and the wider surface can
  still fall back to explicit dense `raw_logits`
- it does not claim adapter serving

## Current Throughput

Measured on this host on March 27, 2026 with the downloaded
`qwen3.5:0.8b-q8_0.gguf`, the same one-sentence prompt, and a `128` token cap:

- Psionic native CUDA qwen35 decode throughput: about `514.13 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `329.34 tok/s`
Measured again on the same host and prompt after the next decode-kernel pass:

- Psionic native CUDA qwen35 decode throughput: about `492.31 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `326.14 tok/s`

Measured again on the same host and prompt after fusing the full-attention key
and value pack path:

- Psionic native CUDA qwen35 decode throughput: about `494.19 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `326.14 tok/s`

Measured again on the same host and prompt after fusing the qwen35 dense and
hybrid SiLU activation tails directly into GGML `Q8_1` scratch and then doing
the same for the full-attention sigmoid gating tail:

- Psionic native CUDA qwen35 decode throughput: about `498.94 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `329.34 tok/s`

Measured again on the same host and prompt after replaying qwen35 prompt-prefix
`NoOutput` submissions through a second captured CUDA graph:

- Psionic native CUDA qwen35 decode throughput: about `507.29 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `329.34 tok/s`

Measured again on the same host and prompt after fusing qwen35 hybrid q/k
RMSNorm plus v staging into one CUDA kernel across the `18` hybrid blocks:

- Psionic native CUDA qwen35 decode throughput: about `507.82 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `329.34 tok/s`

Measured again on the same host and prompt after fusing qwen35 hybrid depthwise
conv1d and the immediate SiLU activation into one CUDA kernel across the same
`18` hybrid blocks:

- Psionic native CUDA qwen35 decode throughput: about `515.24 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `329.34 tok/s`

Later in the same March 27, 2026 session, after restoring the clean head and
rebuilding locally again, the clean restored head remeasured at about
`509.54 tok/s` decode on the same prompt. Switching the q8_1 output-head
argmax path from the shared-input launcher to the MMVQ argmax kernel then
remeasured at:

- Psionic native CUDA qwen35 decode throughput: about `514.13 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `329.34 tok/s`

Measured again on the same host and prompt after routing the dense GGML
`Q8_0` to `Q8_1` matvec fast path onto the MMVQ kernel instead of the
shared-input launcher:

- Psionic native CUDA qwen35 decode throughput: about `520.13 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `329.34 tok/s`

Measured again on the same host and prompt after retuning that q8.0 MMVQ
matvec launch from four warps per row down to two:

- Psionic native CUDA qwen35 decode throughput: about `524.61 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `329.34 tok/s`

Measured again on the same host and prompt after broadcasting q8.0 and q8.1
block scales once per four-lane MMVQ subgroup instead of rereading them on
every lane:

- Psionic native CUDA qwen35 decode throughput: about `533.45 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `328.73 tok/s`

Measured again on the same host and prompt after grouping the q8.0 MMVQ
output-head argmax path into two-row blocks before the global argmax CAS:

- Psionic native CUDA qwen35 decode throughput: about `535.18 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `328.73 tok/s`

Measured again on the same host and prompt after specializing the fully active
dense q8.0 MMVQ decode path to use a fixed four-lane subgroup shuffle mask
instead of recomputing `__activemask()` on every subgroup:

- Psionic native CUDA qwen35 decode throughput: about `523.20 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `328.72 tok/s`

The current bounded lane now depends on fourteen architectural changes inside the native
Psionic runtime:

- qwen35 derives hybrid-layer SSM `decay` and `beta` on CUDA and normalizes
  q/k regions directly into the packed attention and gated-delta input buffers
  instead of copying q and k through extra scratch regions first
- greedy argmax decode now replays the fused qwen35 CUDA submission through a
  captured CUDA graph, and the slower token-embedding mirror experiment is no
  longer on the default path
- full-attention decode now fuses the qwen35 query/gate split with per-head
  query RMSNorm into one CUDA kernel before the attention decode kernel
- full-attention decode now also fuses per-head key RMSNorm with value packing
  into the packed qkv buffer before the attention decode kernel
- qwen35 dense FFN and hybrid SSM-out tails now fuse SiLU activation with
  direct GGML `Q8_1` quantization before the down and output projections
- full-attention decode now also fuses sigmoid gating with direct GGML `Q8_1`
  quantization before the output projection matvec
- qwen35 prompt-prefix replay now reuses a dedicated captured CUDA graph for
  `NoOutput` prompt tokens instead of launching each submission separately
- qwen35 hybrid blocks now fuse q RMSNorm, k RMSNorm, and v staging into one
  CUDA kernel before the recurrent gated-delta step
- qwen35 hybrid blocks now also fuse depthwise causal conv1d with the
  immediately following SiLU activation instead of writing and rereading a
  separate pre-activation buffer
- the qwen35 output head now routes q8_1 argmax decode through the MMVQ kernel
  instead of the older shared-input argmax launcher
- dense GGML `Q8_0` to `Q8_1` matvec on the qwen35 CUDA lane now routes
  through a dedicated two-warp-per-row MMVQ launch shape instead of the older
  shared-input launcher or the earlier four-warp MMVQ launch
- that same q8.0 MMVQ dot path now loads the q8.0 and q8.1 block scales once
  per four-lane subgroup and broadcasts them across the subgroup instead of
  rereading the same scale pair on every participating lane
- that same fully active q8.0 MMVQ decode path now uses a fixed four-lane
  subgroup shuffle mask instead of recomputing `__activemask()` for every
  subgroup
- the q8.0 output-head MMVQ argmax path now reduces two rows inside each CUDA
  block before it updates the global argmax state, instead of doing one global
  CAS update per row block

This pilot therefore proves native CUDA execution correctness, honest
publication, and a wider throughput win over the local Ollama baseline on this
host.

The same native qwen35 CUDA lane now also beats local Ollama on the harvested
registry `2b`, `4b`, and `9b` rows on this host under the same one-sentence
prompt and `128` token-cap contract:

- `qwen3.5:2b`: Psionic about `244.03 tok/s`, Ollama about `205.24 tok/s`
- `qwen3.5:4b`: Psionic about `166.75 tok/s`, Ollama about `141.62 tok/s`
- `qwen3.5:9b`: Psionic about `102.68 tok/s`, Ollama about `94.62 tok/s`

The 4B row needed one extra runtime fix. Its output path includes `Q6_K`
weights, so the fused greedy decode branch now uses `Q8_1` projection plus
`argmax_f32` for that output head instead of the slower generic quantized
matvec path.

The 9B row needed one extra benchmark constraint on this 16 GB GPU host:
Ollama must be unloaded before the native Psionic measurement, because Ollama
keeps model weights resident in VRAM after the reference run.

The same native qwen35 CUDA lane now also beats local Ollama on the explicit
sampled benchmark contract from `docs/QWEN35_OLLAMA_COMPARISON.md`:

- prompt:
  `Respond with plain text only. Write 20 numbered one-sentence reasons why GPU decode throughput matters for local inference. Start at 1 and keep going until all 20 are written.`
- token cap: `128`
- sampled settings:
  - `temperature = 0.8`
  - `top_k = 40`
  - `top_p = 0.9`
  - `repeat_penalty = 1.0`
  - `presence_penalty = 0.0`
  - `frequency_penalty = 0.0`
  - `seed = 42`
  - `think = false` on Ollama
- Psionic output-mode evidence on every sampled row:
  - `qwen35_output_modes=[top_k_candidates:40]`
  - `qwen35_raw_logits=false`

Measured on the same host with `3` repeats per backend after a clean
`CARGO_INCREMENTAL=0` rebuild of `qwen35_cuda_bench` on the Psionic side:

- `qwen3.5:0.8b`: Psionic about `498.37 tok/s`, Ollama about `329.08 tok/s`
- `qwen3.5:2b`: Psionic about `243.77 tok/s`, Ollama about `201.45 tok/s`
- `qwen3.5:4b`: Psionic about `172.86 tok/s`, Ollama about `140.18 tok/s`
- `qwen3.5:9b`: Psionic about `105.38 tok/s`, Ollama about `93.46 tok/s`

The sampled lane stays bounded and honest. qwen35 only uses the fast
`TopKCandidates { top_k }` output path when the request remains exact for a
candidate-only decode surface. Repeat, presence, and frequency penalties now
stay on that bounded lane too. Structured-output masking and unsupported
sampling shapes still fall back to explicit dense raw-logit readback, and
`mirostat` still stays exact through the same fallback path.

The same clean host now also has a larger bounded-candidate follow-on contract
after replacing the older one-row radix-sort path with a partitioned
multi-block CUDA top-k path for single-row decode:

- prompt:
  `Explain what Psionic is in one sentence.`
- token cap: `128`
- sampled settings:
  - `temperature = 0.8`
  - `top_k = 100`
  - `top_p = 0.9`
  - `min_p = 0.05`
  - `seed = 42`
  - `think = false` on Ollama
- Psionic output-mode evidence on every row:
  - `qwen35_output_modes=[top_k_candidates:100]`
  - `qwen35_raw_logits=false`

Measured on the same host with `3` repeats per backend and serialized Ollama
residency between rows:

- `qwen3.5:0.8b`: Psionic about `416.06 tok/s`, Ollama about `320.24 tok/s`
- `qwen3.5:2b`: Psionic about `224.83 tok/s`, Ollama about `204.06 tok/s`
- `qwen3.5:4b`: Psionic about `163.12 tok/s`, Ollama about `124.38 tok/s`
- `qwen3.5:9b`: Psionic about `101.18 tok/s`, Ollama about `93.13 tok/s`

The same local runtime now also keeps repeat, presence, and frequency penalties
on the bounded qwen35 CUDA candidate lane instead of forcing dense-vocab
readback. On the short prompt `Explain what Psionic is in one sentence.` with
the same `128` token cap and these settings:

- `temperature = 0.8`
- `top_k = 40`
- `top_p = 0.9`
- `min_p = 0.05`
- `repeat_penalty = 1.1`
- `repeat_last_n = 64`
- `presence_penalty = 0.2`
- `frequency_penalty = 0.1`
- `seed = 42`

Psionic now publishes `qwen35_output_modes=[top_k_candidates:40]` with
`qwen35_raw_logits=false` on all four local rows and measures about:

- `qwen3.5:0.8b`: `89.29 tok/s`
- `qwen3.5:2b`: `120.71 tok/s`
- `qwen3.5:4b`: `56.47 tok/s`
- `qwen3.5:9b`: `43.27 tok/s`

The local sampler surface now also honors `min_p` and request-level
`repeat_last_n` in addition to `temperature`, `top_k`, `top_p`, `min_p`,
`typical_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`,
`seed`, `mirostat`, `mirostat_tau`, and `mirostat_eta`.
The generic OpenAI-compatible qwen35 server surface now forwards the same
controls on both `/v1/chat/completions` and `/v1/responses`, and the proxy
test harness verifies that those fields reach the qwen35 backend request body.
The local `qwen35_cuda_bench` harness now also reproduces native-versus-Ollama
JSON object and JSON schema requests through `--json-object` and
`--json-schema-file`.
`repeat_last_n` follows the Ollama-compatible local contract:

- default `64`
- `0` disables the penalty lookback window
- `-1` expands the penalty window to the full available history

`min_p` remains compatible with the bounded qwen35 sampled lane because
Psionic applies it after exact top-k candidate selection on both the dense and
bounded sampling paths.

`typical_p` remains compatible with the same bounded lane for the same reason,
and repeat, presence, and frequency penalties now stay on that same bounded
lane too. `mirostat` is implemented on the local sampler, benchmark harness,
and generic qwen35 request surface too, but it still routes through explicit
`raw_logits` fallback on qwen35 CUDA.

Those controls are not part of the canonical Psionic-versus-Ollama matrix on
this checkout, because the local Ollama `qwen3.5` runner routes through
`runner/ollamarunner` and does not wire them through the same active sampler
path. The apples-to-apples local parity matrix therefore stays limited to the
controls wired by `sample.NewSampler(temperature, topK, topP, minP, seed, grammar)`.

Structured-output throughput is also outside the canonical beat-Ollama matrix.
After the leading-char token-cache bucketing pass, the matcher memo-path
allocation cut, and a clean isolated rebuild on March 28, 2026, the local
`qwen3.5:0.8b` summary-schema spot check measured native Psionic at about
`78 tok/s` on the first bounded sparse-gather run and about `162 tok/s` mean
across a warmed three-repeat pass, versus local Ollama at about `331 tok/s`.
Psionic published
`qwen35_output_modes=[top_k_candidates:128,sparse_logits:2,sparse_logits:3,sparse_logits:10]`,
`qwen35_readback_bytes=5700`, and `qwen35_raw_logits=false` on the sparse run.
The later warmed repeats stayed on `qwen35_output_modes=[top_k_candidates:128]`
and hit the token cap without materializing `structured_output_value`, so this
remains a bounded parity note instead of a canonical throughput row.

The same March 27, 2026 benchmark also shows the current boundary clearly:

- Psionic greedy prompt replay for this prompt now spends about `35-38 ms` on
  the `22` prompt tokens
- local Ollama still leads on end-to-end prompt-plus-decode throughput on the
  same prompt

## Current Bottlenecks

The remaining optimization headroom is still inside Psionic's native qwen35
runtime:

- token embedding gather still enters the decode path through a less
  device-native route than it should
- the output-head full-logit path still does not reuse the faster MMVQ argmax
  kernel shape that now serves greedy decode
- the full-attention path still enters the attention kernel through a separate
  q/gate normalization pass instead of a more integrated decode kernel
- the host-seeded hidden vector still reaches the device outside the captured
  qwen35 prompt and decode graphs
- the lane still refuses KV-session reuse, prefix caching, and adapter serving
- the lane has not yet proven a wider batch, longer context, or concurrent
  throughput lead over Ollama-class runtimes

## Claim Rule

This pilot is sufficient to claim that Psionic can load and run the downloaded
`qwen3.5:0.8b` GGUF through a bounded native CUDA prompt-projection lane with truthful
publication for text, image, and video request parts.

It is not sufficient to claim:

- Qwen3.5 multimodal parity
- native image or video understanding
- video understanding
- tool-loop support
- structured-output support
- broad throughput leadership across prompts, lengths, or concurrency

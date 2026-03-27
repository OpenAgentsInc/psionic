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
- the row still refuses tools and structured outputs
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
- it does not claim tool calling
- it does not claim structured-output fallback
- it does not claim adapter serving

## Current Throughput

Measured on this host on March 27, 2026 with the downloaded
`qwen3.5:0.8b-q8_0.gguf`, the same one-sentence prompt, and a `128` token cap:

- Psionic native CUDA qwen35 decode throughput: about `486.78 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `326.14 tok/s`
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

This improvement now comes from nine architectural changes inside the native
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

This pilot therefore proves native CUDA execution correctness, honest
publication, and a wider throughput win over the local Ollama baseline on this
host.

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

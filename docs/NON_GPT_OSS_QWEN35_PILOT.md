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

- it is not throughput-closed against Ollama or llama.cpp-class runtimes
- it is still only a first native Psionic hybrid execution slice
- it does not claim native multimodal inference
- it does not claim a native image or video encoder
- it does not claim tool calling
- it does not claim structured-output fallback
- it does not claim adapter serving

## Current Throughput

Measured on this host on March 26, 2026 with the downloaded
`qwen3.5:0.8b-q8_0.gguf`, the same one-sentence prompt, and a `128` token cap:

- Psionic native CUDA qwen35 decode throughput: about `85.07 tok/s`
- local Ollama `qwen3.5:0.8b` decode throughput: about `321.04 tok/s`

This pilot therefore proves native CUDA execution correctness and honest
publication. It does not prove competitive throughput yet.

## Current Bottlenecks

The remaining performance gap is inside Psionic's native qwen35 runtime:

- hybrid SSM layers still run their recurrent delta-net math on the host
- full-attention decode still uses host-owned KV history and host-side softmax
- the runtime still executes one synchronized CUDA submission per projection
  instead of a deeper device-resident decode plan
- the lane still refuses KV-session reuse, prefix caching, and adapter serving

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
- throughput parity with Ollama

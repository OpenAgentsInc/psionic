# Non-GPT-OSS Gemma 4 Pilot

> Status: `partial` on 2026-04-02 after issue `#863`; the first native CUDA
> runtime plus bounded server and conformance coverage exist, but broader
> publication remains gated on later issues.

This document freezes what the first honest `Gemma 4` claim means and now
tracks the first bounded implementation state.

It still does not mean that the whole `Gemma 4` family is implemented on this
checkout.

## Frozen First Target

The first target artifact is:

- artifact = `gemma4:e4b`
- family shape = dense `Gemma 4`
- intended runtime posture = Psionic-owned CUDA-backed text generation

This is the first useful target because it is already available as a real GGUF
artifact, it avoids the current non-`GptOss` MoE gate, and it is large enough
to exercise real prompt, routing, and long-context behavior without forcing the
first claim to absorb the whole `Gemma 4` family.

## Frozen First Claim

The first bounded `Gemma 4` claim for Psionic is:

- one dense `Gemma 4` text lane
- one exact artifact target: `gemma4:e4b`
- one accelerator target: CUDA
- one server surface: the generic OpenAI-compatible server

When this lane is implemented, the publication bar is:

- `backend = cuda`
- `execution_mode = native`
- `execution_engine = psionic`
- refusal-required unsupported regions stay explicit in capability and health
  publication

## Explicit Non-Goals

The first `Gemma 4` claim does not include:

- image
- video
- audio
- `31B Dense`
- `26B A4B`
- Metal
- full parity with `llama.cpp`
- full parity with `ollama`

CPU-only bring-up may still be useful as a debug lane, but CPU alone does not
complete the first published `Gemma 4` milestone.

## Current Repo State

After issues `#861` through `#863`, the repo now has the first honest
`Gemma 4` admission, runtime, and bounded conformance work:

- `psionic-models` now classifies `general.architecture = gemma4` as its own
  decoder family instead of silently aliasing it to `llama`, `qwen`, or
  another generic family.
- the local `gemma4:e4b` GGUF tokenizer facts are now fixture-covered,
  including the Gemma-specific turn, tool-call, tool-response, and channel
  tags.
- dense `Gemma 4` artifacts classify cleanly, while expert-bearing `Gemma 4`
  artifacts still fail closed at the current non-`GptOss` MoE gate.
- the local `e4b` GGUF does not currently embed `tokenizer.chat_template`, so
  the repo now carries one checked-in bounded text template fixture at
  `crates/psionic-models/src/testdata/gemma4_chat_template.jinja`.
- `psionic-serve` now has a first bounded native `Gemma 4` CUDA runtime for
  quantized dense projection artifacts, using the Psionic-owned tokenizer,
  prompt, KV-cache, and decode loop instead of routing through `llama.cpp`.
- the generic OpenAI-compatible server now admits `Gemma 4` on CUDA with
  `backend = cuda`, `execution_mode = native`, and
  `execution_engine = psionic`, and the bounded lane now stamps both
  `x-psionic-backend` and `x-psionic-served-backend` as machine-checkable
  backend labels, but the bounded `#862` surface is still chat-only on
  `/v1/chat/completions`.
- the repo now has bounded prompt-render, health/model publication, and
  refusal coverage for the `Gemma 4` CUDA lane, including explicit fail-closed
  checks for tools, structured outputs, multimodal inputs, and
  `/v1/responses`.
- the conformance harness now accepts the real `gemma4:e4b` fixture shape and
  the repo now carries one repeatable `gemma4:e4b` CUDA conformance repeat
  test that runs when both the pilot GGUF and a CUDA host are available.
- the managed dense-GGUF lane now allocates whole-model KV-cache width instead
  of silently falling back to one-layer fixture geometry, and the repo now
  carries a multi-layer regression that keeps that boundary explicit.
- the live `gemma4:e4b` CUDA lane now respects Gemma 4's mixed per-layer
  attention geometry, including narrower sliding-window layers and wider
  full-attention layers, instead of assuming one uniform KV shape across the
  whole stack.

What still does not exist:

- no tool-calling or `/v1/responses` Gemma semantics
- no image, video, or audio Gemma lane
- no broad published support claim beyond the bounded runtime, server
  admission, and repo-owned conformance coverage

## Why The Claim Is Narrow

`Gemma 4` is a family, not one model. The first useful Psionic result is not
"supports Gemma 4" in the broad sense. The first useful result is a narrow,
truthful lane that proves Psionic can admit a dense `Gemma 4` artifact, run it
through a CUDA-backed serving path, and publish exactly what still remains
unsupported.

That bounded claim keeps later work cleanly separated:

- tool calling and `/v1/responses`
- image and video processing
- audio support for `E2B` and `E4B`
- `31B Dense`
- `26B A4B` and wider non-`GptOss` MoE admission
- Metal and other accelerator parity

## Follow-On Issues

This claim freeze is the prerequisite for the next runtime issues:

1. family and prompt classification
2. dense CUDA runtime bring-up
3. server and conformance coverage
4. mesh validation
5. bounded publication of the finished first lane

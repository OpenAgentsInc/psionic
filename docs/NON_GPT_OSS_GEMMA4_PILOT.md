# Non-GPT-OSS Gemma 4 Pilot

> Status: `planned` on 2026-04-02 for the first Psionic-owned `Gemma 4` lane.

This document freezes what the first honest `Gemma 4` claim means before the
runtime work starts.

It does not mean that `Gemma 4` is already implemented on this checkout.

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

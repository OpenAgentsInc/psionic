# Non-GPT-OSS Gemma 4 Pilot

> Status: `published_bounded_lane` on 2026-04-02 after issues `#864`,
> `#865`, and `#866`; Psionic now publishes one narrow dense `Gemma 4` CUDA
> lane and keeps the rest of the family explicitly out of scope.

This document freezes what the first honest `Gemma 4` claim means and now
records the published first lane.

It still does not mean that the whole `Gemma 4` family is implemented on this
checkout.

The pilot row is the local `e4b` artifact at:

- default path:
  `/Users/christopherdavid/models/gemma4/gemma4-e4b-ollama.gguf`
- canonical model id:
  `gemma4:e4b`

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
- one bounded Gemma-native tool-call and response-state contract

When this lane is implemented, the publication bar is:

- `backend = cuda`
- `execution_mode = native`
- `execution_engine = psionic`
- refusal-required unsupported regions stay explicit in capability and health
  publication

## Published Lane

Psionic now publishes exactly one bounded dense `Gemma 4` lane:

- artifact = `gemma4:e4b`
- family shape = dense only
- accelerator = CUDA
- runtime = native Psionic GGUF runtime
- local server surface = generic OpenAI-compatible server
- admitted request surfaces = `/v1/chat/completions` and `/v1/responses`
- admitted publication surfaces = `/health`, `/v1/models`, and response
  headers
- bounded tool contract = Gemma-native `<|tool_call>call:<tool>{...}<tool_call|>`
  blocks with JSON-schema-subset argument validation and replayable tool-result
  state on `/v1/responses`
- distributed proof surface = one bootstrap-routed remote `gemma4:e4b` lane
  with honest proxy publication

This published lane is not the claim "Psionic supports Gemma 4" in the broad
sense. It is the smaller claim that Psionic can admit one dense `Gemma 4`
artifact, execute it natively on CUDA, route to the same family through the
mesh bootstrap path, and keep unsupported regions refused instead of widening
the claim by implication.

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

After issues `#861` through `#866`, the repo now has the first honest
published `Gemma 4` admission, runtime, bounded mesh validation, and
conformance work:

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
  backend labels across both `/v1/chat/completions` and `/v1/responses`.
- the same bounded lane now admits Gemma-native tool calling on both server
  surfaces through explicit `<|tool_call>call:<tool>{...}<tool_call|>` blocks
  instead of borrowing the tagged-JSON Qwen or GPT-OSS contract.
- `/v1/responses` now stores replayable Gemma assistant tool-call turns and
  projects replayed tool results back into one explicit user-side
  `tool_responses` turn instead of failing the continuation surface closed.
- the repo now has bounded prompt-render, health/model publication, and
  refusal coverage for the `Gemma 4` CUDA lane, including explicit fail-closed
  checks for structured outputs and multimodal inputs.
- the conformance harness now accepts the real `gemma4:e4b` fixture shape and
  the repo now carries one repeatable `gemma4:e4b` CUDA conformance repeat
  test that runs when both the pilot GGUF and a CUDA host are available.
- the same `gemma4:e4b` lane now also has one first distributed bootstrap-mesh
  validation path, keeping routed remote publication explicit instead of
  silently widening the thin-client lane.
- the managed dense-GGUF lane now allocates whole-model KV-cache width instead
  of silently falling back to one-layer fixture geometry, and the repo now
  carries a multi-layer regression that keeps that boundary explicit.
- the live `gemma4:e4b` CUDA lane now respects Gemma 4's mixed per-layer
  attention geometry, including narrower sliding-window layers and wider
  full-attention layers, instead of assuming one uniform KV shape across the
  whole stack.

What still does not exist:

- no image, video, or audio Gemma lane
- no generic structured-output Gemma surface
- no multimodal Gemma request surface
- no broader published support claim beyond the bounded dense CUDA lane,
  server admission, routed mesh validation, and repo-owned conformance
  coverage

## Canonical Repeat

The published lane is repeatable through repo-owned tests.

Documented local validation:

```bash
cargo test -p psionic-serve conformance --manifest-path Cargo.toml --no-default-features
cargo test -p psionic-serve gemma4 --manifest-path Cargo.toml --no-default-features
```

Real-artifact repeat lane:

```bash
PSIONIC_GEMMA4_PILOT_GGUF_PATH=/abs/path/to/gemma4-e4b-ollama.gguf \
  cargo test -p psionic-serve \
  gemma4_e4b_cuda_conformance_repeat_is_machine_checkable_when_available \
  --manifest-path Cargo.toml --no-default-features
```

Those runs are the current publication bar. They prove:

- bounded prompt-render and fixture alignment for the real `gemma4:e4b`
  instruction-first shape
- dense `Gemma 4` family admission and fail-closed refusal for expert-bearing
  `Gemma 4` artifacts
- native CUDA load planning for quantized dense projection artifacts
- honest generic-server publication with:
  - `backend = cuda`
  - `execution_mode = native`
  - `execution_engine = psionic`
- admitted `/v1/chat/completions` and `/v1/responses` execution on the bounded
  lane
- admitted Gemma-native tool calling with JSON-schema-subset argument
  validation and replayable response-state storage
- explicit refusal for:
  - structured outputs
  - multimodal inputs
- repeatable real-artifact CUDA conformance against the local `e4b` GGUF when
  both the artifact and a usable CUDA host are present
- bootstrap-routed mesh publication that keeps remote `Gemma 4` route truth
  explicit instead of widening the thin-client claim

## Pass Criteria

The pilot stays green only if all of the following remain true:

- `gemma4:e4b` is still admitted as dense `gemma4`
- the bounded text template remains aligned to the real `e4b` tokenizer facts
- the native CUDA runtime still loads the dense GGUF lane without silently
  degrading to another family or engine
- the generic server still publishes truthful backend, execution-mode, and
  execution-engine metadata
- the generic server still admits `/v1/chat/completions` and `/v1/responses`
- Gemma-native tool calling still produces machine-checkable tool calls on the
  admitted lane
- structured outputs and multimodal inputs still fail closed on the published
  lane
- mesh bootstrap publication keeps routed remote truth explicit and does not
  borrow a local host's wider endpoint claim

## Why The Claim Is Narrow

`Gemma 4` is a family, not one model. The first useful Psionic result is not
"supports Gemma 4" in the broad sense. The first useful result is a narrow,
truthful lane that proves Psionic can admit a dense `Gemma 4` artifact, run it
through a CUDA-backed serving path, and publish exactly what still remains
unsupported.

That bounded claim keeps later work cleanly separated:

- image and video processing
- audio support for `E2B` and `E4B`
- generic structured outputs
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

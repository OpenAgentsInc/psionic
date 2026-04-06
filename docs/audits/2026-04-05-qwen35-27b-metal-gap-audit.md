# Qwen3.5 27B Metal Gap Audit

> Status: audit, 2026-04-05
>
> Scope: explain why the real Hugging Face `Qwen3.5-27B-Q4_K_M.gguf` artifact
> is still slow on Psionic local Mac execution, compare the current Psionic
> shape against `llama.cpp` and Ollama, and define the next architecture work
> required to close the gap.
>
> This is an audit doc. It is not the canonical inference spec. The canonical
> owner docs remain `docs/INFERENCE_ENGINE.md`.

## Current State

Current `main` admits the real Hugging Face `Qwen3.5-27B-Q4_K_M.gguf` artifact
on Psionic without a `llama.cpp` proxy runtime.

That closed parser and family-admission work. It did not close performance.

The current measured local receipt on this Mac from the native Psionic harness
was:

| Artifact | Backend | Load | Prompt | Decode | Total | TTFT | End-to-end throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.5-27B-Q4_K_M.gguf` | `cpu` | `30.646 s` | `303.782 s` | `220.120 s` | `523.902 s` | `0.000195 s` | `0.0305 tok/s` |

That number is not a small-kernel tuning problem. It is a runtime-shape problem.

## What The Reference Systems Are Doing That We Are Not

### `llama.cpp`

`llama.cpp` treats `qwen35` as a first-class hybrid linear-attention family.

The checked-in local reference code does all of these things explicitly:

- loads dedicated `qwen35` and `qwen35moe` model builders
- admits the gated-delta layout directly instead of pretending it is generic
  dense attention
- carries model-specific `gla` and recurrent memory handling in the runtime
- builds backend-owned compute graphs and cache structures around that shape

Relevant local reference files:

- `competition/repos/llama.cpp/src/models/qwen35.cpp`
- `competition/repos/llama.cpp/src/models/qwen35moe.cpp`
- `competition/repos/llama.cpp/ggml/src/ggml-cuda/gla.cu`

### Ollama

Ollama also treats the family as first-class.

The local checked-in Qwen3-next implementation:

- owns the full hybrid operator
- owns the recurrent cache shape
- owns the split `ssm_beta` and `ssm_alpha` path used by `qwen35`
- owns the `ssm_dt.bias` alternate tensor naming
- validates the recurrent/full-attention mixed layout before execution

Relevant local reference files:

- `competition/repos/ollama/model/models/qwen3next/model.go`
- `competition/repos/ollama/model/models/qwen3next/deltanet.go`

### MLX

The local `mlx` checkout in this workspace does not currently expose a checked
in `qwen35` implementation to compare directly. It is still useful as a
reference for Apple unified-memory execution and device-owned array semantics.

That matters because Psionic is failing before the Qwen35-specific recurrent
math even becomes the dominant problem. The Mac lane is still spending the
majority of its time in a host-owned decode loop.

## Where Psionic Is Slow

The current real Mac `Qwen3.5-27B-Q4_K_M.gguf` lane is slow for three direct
reasons.

### 1. The admitted local lane is still CPU

The benchmark harness in `crates/psionic-serve/examples/qwen35_bench.rs` only
admits `cpu` and `cuda`.

The OpenAI server in `crates/psionic-serve/src/openai_http.rs` only admits:

- `cuda` for `qwen35`
- `cpu` for everything else on the fallback path
- no native Metal runtime for `qwen35`

So Apple Silicon is not reaching a device-native `qwen35` runtime at all.

### 2. The real 27B artifact depends on `Q5_K`

The public Hugging Face `Qwen3.5-27B-Q4_K_M.gguf` artifact is not a pure
`Q4_K` artifact internally. It carries `Q5_K` tensors.

Psionic currently has:

- CPU decode and dot support for `Q5_K` in `psionic-backend-cpu`
- no `Q5_K` decode in `psionic-nn`
- no `Q5_K` Metal quantized matvec support in `psionic-backend-metal`

The current Metal backend only admits native quantized projection kernels for:

- `Q4_K`
- `Q6_K`
- `Q8_0`
- `MXFP4`

That means the public 27B artifact falls off the Metal fast path before the
family-specific recurrent operator even matters.

### 3. The current native Psionic `qwen35` implementation is model-specific on CUDA only

`crates/psionic-serve/src/qwen35.rs` already contains a large amount of
Qwen35-specific accelerated logic.

That logic currently exists for CUDA:

- native quantized projection upload
- native Qwen35 hybrid-layer step planning
- native full-attention path
- native recurrent state handling
- native decode output selection and profiling

There is no equivalent Metal service. The Mac lane therefore bypasses the only
family-specific accelerated implementation Psionic currently owns.

## Direct Comparison Against Current Psionic Shape

Today Psionic has:

- a real native CUDA `qwen35` runtime
- a real native CPU `qwen35` runtime
- no native Metal `qwen35` runtime
- no Metal `Q5_K` matvec support

That is the whole gap in one sentence.

The reference systems do not route Apple execution for this family into a
generic CPU fallback. They route it into model-aware accelerated code.

## What Has To Change

The next architecture step is not ambiguous.

### First required change: admit `Q5_K` on Metal

Without native `Q5_K` quantized matvec on Metal, the public 27B artifact cannot
become a serious Apple lane.

This requires:

- native `Q5_K` Metal quantized matvec
- native `Q5_K` Metal logits-selection path
- truthful Metal backend admission for the mode

### Second required change: add a native Metal `qwen35` runtime

This does not need to start as a full Metal recurrent-kernel stack on day one.

The first bounded honest version can:

- move all large projection work to Metal
- keep the smaller recurrent and attention glue on host where needed
- keep refusal posture explicit for unsupported execution regions

That is still materially better than the current all-CPU lane because the
largest per-token costs are the repeated large quantized projections.

### Third required change: expose the real lane through the public runtime selectors

After the service exists, Psionic needs to admit it in:

- `qwen35_bench`
- `openai_http.rs`
- runtime-support publication
- `docs/INFERENCE_ENGINE.md`

If the code exists but the selectors keep routing Apple devices to CPU, the
architecture is still wrong.

## What This Audit Does Not Claim

This audit does not claim that a first Metal `qwen35` runtime will instantly
match the best `llama.cpp` or Ollama numbers.

It claims something narrower and concrete:

- the current local Mac number is dominated by the absence of a real Metal
  runtime
- the public 27B artifact is blocked specifically by missing `Q5_K` Metal
  support
- Psionic already has enough family-specific structure to justify a dedicated
  Metal lane now instead of treating Qwen35 as CPU-only on Apple

## Immediate Follow-On Work

The required follow-on sequence is:

1. land native `Q5_K` Metal quantized matvec support
2. land `MetalGgufQwen35TextGenerationService`
3. admit `metal` in `qwen35_bench`
4. admit `qwen35` on Metal in the generic OpenAI server
5. record new receipts for the real Hugging Face 27B artifact on current `main`

That is the shortest honest path from the current `0.0305 tok/s` CPU lane to a
real Apple execution lane.

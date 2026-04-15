# Gemma 4 26B Full Gap Audit: Why Psionic Still Loses to Ollama and `llama.cpp`

Status: 2026-04-15

Scope: same-host single-node text inference on Apple Silicon for
`gemma-4-26B-A4B-it-Q4_K_M.gguf`, after the native sparse Metal closure work
and after the second same-day tuning pass.

This is the blunt answer to the current state:

- Psionic is still materially slower than both same-host `ollama` and
  `llama.cpp`
- the live gap is now mostly architectural, not a hidden host sparse fallback
- `ollama` is not beating us with a fundamentally different Apple-local core;
  on this lane it is mostly shipping `llama.cpp` with runner, cache, and
  product glue around it

## Retained Competitive Position

Same-host retained gate receipt from
`fixtures/gemma4/benchmarks/gemma4_26b_competitive_20260415_072548_christophers-macbook-pro-2-.json`:

| Engine | Prompt Eval | Decode | Decode tok/s | Total |
| --- | ---: | ---: | ---: | ---: |
| Psionic | `5.323 s` | `2.089 s` | `30.63` | `7.413 s` |
| Ollama | `0.391 s` | `0.625 s` | `102.34` | `2.580 s` |
| `llama.cpp` | `0.073 s` | `0.562 s` | `113.79` | `2.704 s` |

What that means:

- Psionic decode is `0.299x` Ollama and `0.269x` `llama.cpp`
- Ollama is `3.34x` faster than Psionic on decode
- `llama.cpp` is `3.71x` faster than Psionic on decode
- Psionic prompt eval is about `13.6x` slower than Ollama
- Psionic prompt eval is about `72.9x` slower than `llama.cpp`

The second same-day local follow-on pass did not materially change that
position. The latest retained local check stayed around `30.20 tok/s`, which
is still nowhere near the same-host competitor floor.

## Executive Verdict

Psionic is no longer losing for the same reason as the earlier audits.

The current Gemma 4 Metal lane already proves all of these:

- all `30 / 30` sparse FFN layers stay on device
- sparse FFN backend is `metal_grouped_experts`
- router backend is `metal_router_topk_softmax`
- expert dispatch backend is `metal_grouped_ids`
- host sparse fallback is not the active explanation for the retained run
- the hot local Metal step is not one command buffer per layer

The problem now is this:

1. Psionic still drives Gemma 4 through a serve-owned per-token execution loop.
2. Prompt prefill still replays decode-style orchestration token-by-token.
3. The live Gemma path does not use a full-model reserved and reused graph
   runtime.
4. Our Metal kernel surface is better than it was, but still much narrower and
   less scheduler-owned than the `ggml` path shipping inside `llama.cpp`.
5. The remaining gap is too large for one more tiny kernel tweak to close.

## What Changed Since The Earlier Audit

The older `2026-04-04` audit was directionally useful, but parts of it are now
stale.

What is no longer the main story:

- not silent host sparse FFN fallback
- not "we are still bouncing expert execution through the CPU"
- not "we issue one Metal command buffer per layer"

What is now the real story:

- we issue one full host-planned Metal submission per token
- we wait for that submission to complete before moving to the next token
- prompt processing is still a loop over token-level decode steps rather than a
  distinct prompt-processing runtime shape
- the live model path still lives mostly in `crates/psionic-serve/src/gguf.rs`
  instead of a graph-owned runtime comparable to `llama.cpp`

That is a much better place than the fallback cliff, but it is still far from
competitor-grade runtime architecture.

## Full Difference Map

## 1. Ollama Is Mostly A Product Shell Around `llama.cpp`, Not A Separate Local Inference Core

Ollama's own checked-in docs and bindings are explicit:

- `competition/repos/ollama/llama/README.md` says Ollama vendors
  `llama.cpp` / `ggml`
- `competition/repos/ollama/llama/llama.go` binds directly into vendored
  `llama.cpp`
- `competition/repos/ollama/llm/server.go` still constructs `llama` context
  params, batch size, ubatch size, KV cache type, and flash-attention policy
  for the compatibility runner
- `competition/repos/ollama/runner/llamarunner/runner.go` manages sequence
  batching, cache slots, and HTTP streaming around that core

The practical implication is important:

- if Psionic wants to beat Ollama on this lane, it mostly has to beat the
  `llama.cpp` core that Ollama ships
- Ollama's runner and cache logic matter, but they are not the main reason
  `102.34 tok/s` exists
- the real engine bar is still the `ggml` / `llama.cpp` local Apple path

This also explains why Ollama and `llama.cpp` land in the same rough class on
the retained receipt while Psionic is far below both.

## 2. `llama.cpp` Owns Prompt Processing And Decode As Different Runtime Shapes; Psionic Does Not

`llama.cpp` is explicit about this in `src/llama-context.cpp`:

- it reserves prompt-processing (`pp`) graph shape first
- it reserves token-generation (`tg`) graph shape separately
- it reserves prompt-processing again to avoid allocator churn later
- it chooses `n_batch` and `n_ubatch` as first-class runtime parameters

`llama.cpp` also splits work into ubatches through:

- `src/llama-batch.h`
- `src/llama-memory-hybrid.cpp`
- `src/llama-kv-cache.cpp`

That means the competitor runtime can process prompt tokens as prompt work, not
just as a long series of decode-shaped single-token steps.

Psionic's live Gemma 4 Metal path still does the opposite.

In `crates/psionic-serve/src/gguf.rs`, prompt processing walks:

- `for (relative_prompt_index, token) in prompt_tokens ...`
- then calls `forward_step_with_layer_caches(...)` once per prompt token

That is the clearest reason prompt eval is catastrophically behind.

The current numbers are not a subtle signal:

- Psionic prompt eval: `5.323 s`
- Ollama prompt eval: `0.391 s`
- `llama.cpp` prompt eval: `0.073 s`

That is not a q8_1 problem. That is runtime shape failure.

## 3. Psionic's Live Gemma Path Is Still Serve-Owned Host Planning; `llama.cpp` Is Graph-Owned

The hot Psionic path for this lane is still built in
`crates/psionic-serve/src/gguf.rs`, especially
`forward_local_step_with_layer_caches(...)`.

That function:

- builds a Metal submission in the serving layer
- manually encodes the layer stack op by op
- manages which buffers are written and synchronized
- decides when to produce logits, top-k, or greedy token selection
- commits the command buffer and waits for completion before returning

That is much better than the older host fallback path, but it is still a
serve-owned step engine.

By contrast, `llama.cpp`:

- builds model-owned graphs through `model.build_graph(...)`
- reserves scheduler state with `graph_reserve(...)`
- reuses prior graph state when `res->can_reuse(gparams)` holds
- couples graph reserve to backend samplers and output ownership

Psionic does have early reserve machinery in
`crates/psionic-backend-metal/src/lib.rs`:

- `reserve_attention_graph(...)`
- `MetalAttentionGraphRuntime`
- `decode_attention_f32_reserved(...)`

But that is still attention-scoped backend runtime, not the live full-model
Gemma 4 execution path. The current `gguf.rs` hot path does not use that
full-model reserve/reuse story.

This is one of the biggest remaining differences.

## 4. Psionic Still Pays One Host Roundtrip Per Token

The current live Metal step is not one command buffer per layer.

That matters because we should not keep blaming the wrong thing.

`forward_local_step_with_layer_caches(...)` starts one submission with:

- `backend.begin_submission(...)`

Then it encodes the whole layer stack into that submission, and finally:

- `submission.commit(MetalCommandWait::Completed)`

So the current hot path is:

- one host-planned submission per token
- one completed wait per token
- optional buffer synchronization and readback for logits or selected tokens

That is still expensive for prompt prefill because prompt length now directly
means:

- one host orchestration pass per prompt token
- one command-buffer commit per prompt token
- one completion wait per prompt token

`llama.cpp` does not pay this same shape cost on prompt processing because it
can run prompt-sized ubatches through reserved graph execution.

## 5. `llama.cpp` Has A Much Deeper Metal Kernel Family And Scheduler Surface

Psionic's Metal backend is no longer bare. It now has meaningful native ops,
including:

- fused `qk` / `qkv` RMS+RoPE+append-dense-KV kernels
- dense decode attention
- `top_k`
- grouped sparse gate/up SWIGLU kernels
- MoE down-aggregate kernels for `q4_k`, `q5_0`, and `q8_0`
- `q8_1` quantize plus grouped sparse expert kernels

That is real progress and the audit should say so plainly.

But `llama.cpp` / `ggml` is still broader and more mature:

- `ggml-metal-common.cpp` performs backend-owned graph reorder with memory-range
  safety checks
- `ggml-metal-ops.cpp` and `ggml-metal.metal` carry a far broader kernel and
  pipeline family
- the vendored Metal path inside Ollama includes specialized
  `kernel_mul_mv_id_*` variants for many quantization layouts, not just a small
  subset of the modes that matter to this exact Gemma path
- the graph scheduler decides split count, allocation, reuse, and backend
  placement at graph level

The important part is not that `llama.cpp` has "more files."

The important part is that its kernel family, graph scheduler, and reserve
logic were designed together. Psionic still has those concerns split between:

- backend primitives in `psionic-backend-metal`
- host step planning in `psionic-serve`

That split makes it harder to reach the same steady-state behavior.

## 6. `llama.cpp` Owns Ubatch Splitting, Sequence Packing, And KV Scheduling More Deeply

The current competitor path does not just run one prompt through one monolithic
black box.

`llama.cpp` has explicit batch-allocation and ubatch logic in:

- `src/llama-batch.h`
- `src/llama-memory-hybrid.cpp`
- `src/llama-kv-cache.cpp`

That code decides how to split sequence work through:

- `split_simple`
- `split_equal`
- `split_seq`
- sequence-aware KV preparation

Ollama then adds runner-level sequence management and cache-slot reuse on top,
for example:

- `runner/llamarunner/runner.go` batches prompt work across active sequences
- `runner/llamarunner/cache.go` picks cache slots, reuses common prefixes,
  shifts context, and replays or copies cached prefixes when helpful

For the exact one-request benchmark, this is not the whole speed story.

But it does show an important difference in ownership:

- competitor runtimes treat batch shape, resume position, and KV policy as
  central runtime concerns
- Psionic's live Gemma 4 lane is still much closer to a single-request serve
  loop with backend calls underneath it

## 7. Psionic Still Reads And Synchronizes At The Serving Layer More Than The Competitor Core

In the current Gemma 4 Metal step, `gguf.rs` still decides when to:

- `synchronize_buffer(...)`
- read logits
- read top-k buffers
- read greedy selected-token buffers
- materialize host-visible stage outputs

The optimized greedy path helps because it bounds readback to one token id.
The top-k path helps because it avoids full logits readback.
The native sparse path helps because FFN execution stays on device.

But the serving layer still owns these decisions directly.

In `llama.cpp`, output-path decisions are much more tightly coupled to graph
reserve, output buffers, and backend samplers. That tighter ownership is one of
the reasons graph reuse and scheduler reuse stay coherent.

## 8. The Remaining Gap Is Not The Sparse FFN Anymore

We already proved a few things that matter:

- native sparse execution is real on the retained lane
- `host_fallback_layer_count = 0`
- the dense shared q8_1 control-flow bug was real, but fixing it barely moved
  throughput
- real `DenseF16Mirror` was slower than dense `f32` KV on this host
- widening dense `q5_0` / `q8_0` threadgroups moved the needle only slightly

The concrete same-day evidence is:

- q8_1 enabled vs disabled stayed basically flat
- real `DenseF16Mirror` regressed the lane instead of improving it
- post-fix local checks still sat around `30 tok/s`

So the honest conclusion is:

- sparse FFN host fallback was a cliff and we fixed it
- it was not the final competitive gap
- the surviving gap is upstream of that fix, mostly in runtime shape and kernel
  ecosystem depth

## 9. Ollama's Advantage Over Psionic Is Smaller Than `llama.cpp`'s On Decode, But The Same Core Reason Applies

On the retained receipt:

- Ollama = `102.34 tok/s`
- `llama.cpp` = `113.79 tok/s`

That spread is much smaller than the gap between either of them and Psionic.

This matters because it says:

- Ollama's extra runner layer is not the main reason Psionic loses
- the dominant difference is the local inference core
- if Psionic only copies Ollama's outer process shell without matching the
  graph/runtime/kernel core, it will still lose

## What This Means In Practice

If the target is to exceed both same-host Ollama and same-host `llama.cpp` on
this lane, the work item is not "one more q5_0 tweak."

It is a proper runtime rewrite with at least these characteristics:

1. A real prompt-processing runtime for Gemma 4 instead of token-by-token
   decode replay during prefill.
2. A full-model reserved and reused graph or graph-equivalent execution shape,
   not just reserved attention primitives.
3. Ubatch-aware scheduling and sequence packing owned by the runtime rather than
   rebuilt ad hoc in the serving loop.
4. Backend-owned output selection, sampler coupling, and graph reserve identity.
5. A broader Metal kernel family for the quantized modes and grouped-id paths
   that this exact artifact actually exercises.
6. Less host-owned synchronization and fewer completed waits in the hot path.
7. Clear separation between one-token decode optimization and prompt-prefill
   optimization so prompt eval stops being the dominant failure.

## Bottom Line

Psionic is still not as good because it still behaves like a host-driven serve
engine with better kernels, while `llama.cpp` behaves like a reserved,
reused, batch-aware inference runtime with a much deeper Metal and scheduler
stack.

That is why:

- fixing native sparse FFN fallback was necessary but not sufficient
- tiny tuning passes keep moving numbers by tenths, not by multiples
- prompt eval is still catastrophically behind
- decode is still several times slower even after the fallback cliff is gone

The honest current state is:

- Psionic's Gemma 4 26B Metal lane is now real
- it is still not competitor-grade
- the next winning move is a full model-runtime architecture pass, not another
  small local tweak

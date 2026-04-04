# Gemma Throughput Audit Against MLX, `llama.cpp`, `vLLM`, and `rvLLM`

> Status: audit, 2026-04-04
>
> Scope: explain why current Psionic `gemma4:e4b` inference is still slower
> than the faster local and serving reference systems, even after `#896`
> correctness closure and the follow-on local Metal speed work.
>
> This is an audit doc. It is not the canonical inference spec. The canonical
> owner docs remain `docs/INFERENCE_ENGINE.md` and
> `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`.

## Why This Audit Exists

`#896` closed the first honest Gemma 4 execution bar:

- single-node Mac Metal works
- single-node RTX 4080 CUDA works
- mixed-device Mac+4080 distributed execution works

That closed correctness and honest publication.

It did not close the performance gap.

The measured gap is still large enough that it needs an architectural audit, not
another small tuning pass.

## Benchmark Gap Being Explained

The current best numbers in the `#896` work window were:

| Lane | Measurement shape | TTFT | Total time | End-to-end throughput |
| --- | --- | ---: | ---: | ---: |
| Psionic single-node Mac Metal, initial honest closeout mean | 3 warm runs | `184.0 ms` | `1.881 s` | `5.32 tok/s` |
| Psionic single-node Mac Metal, best kept follow-on tuning pass | warm mean from runs 2-5 | `47.78 ms` | `503.48 ms` | `19.862 tok/s` |
| Psionic single-node RTX 4080 CUDA | 3 warm runs | `50.1 ms` | `0.612 s` | `16.33 tok/s` |
| Psionic mixed Mac+4080 distributed, first clean proof run | 1 run | `2.912 s` | `4.272 s` | `2.34 tok/s` |
| Psionic mixed Mac+4080 distributed, later split sweep best observed | 1 run at `split=1` | not rederived | not rederived | `2.717 tok/s` |
| Ollama on the same local `gemma4-e4b-ollama.gguf` artifact | warm mean of runs 2-3 | `2.519 s` | `2.592 s` | `93.36 tok/s` |

Important measurement note:

- the direct same-artifact local comparator captured in this work window was
  `ollama`, not a successful direct `llama.cpp` run
- local upstream `llama.cpp` built cleanly but failed to load this exact GGUF
  with `wrong number of tensors; expected 2131, got 720`
- the Ollama throughput numbers use Ollama's own token accounting, so they are
  still useful as a same-artifact directional comparator, but they are not a
  tokenizer-neutral apples-to-apples metric against Psionic's
  `completion_tokens`

Even with that caveat, the direction is obvious:

- Psionic Metal improved materially
- Psionic Metal is still far behind the fast local reference path on the same
  model artifact
- Psionic mixed-device throughput is dominated by cross-device overhead and is
  nowhere near where a serious pipeline path should land

## Sources Reviewed

### Psionic

- `crates/psionic-serve/src/gguf.rs`
- `crates/psionic-backend-metal/src/lib.rs`
- `docs/INFERENCE_ENGINE.md`
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
- `docs/NON_GPT_OSS_GEMMA4_PILOT.md`
- `docs/METAL_GPT_OSS_MLX_LM_LESSONS.md`
- issue `#896` comments with the benchmark receipts and follow-on speed notes

### MLX

- `mlx/mlx/fast.cpp`
- `mlx/mlx/backend/metal/rope.cpp`
- `mlx/mlx/backend/metal/kernels/rms_norm.metal`
- `mlx/mlx/backend/metal/kernels/scaled_dot_product_attention.metal`
- `mlx/mlx/backend/metal/quantized.cpp`
- `mlx/mlx/backend/metal/kernels/gemv.metal`
- `mlx/mlx/memory.h`

### `llama.cpp`

- `llama.cpp/src/llama-batch.h`
- `llama.cpp/src/llama-memory-hybrid.cpp`
- `llama.cpp/src/llama-kv-cache.cpp`
- `llama.cpp/src/llama-graph.cpp`
- `llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m`
- `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`

### `vLLM`

- `vllm/README.md`
- `vllm/vllm/config/scheduler.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/core/block_pool.py`
- `vllm/vllm/compilation/passes/fusion/rope_kvcache_fusion.py`

### `rvLLM`

- `rvllm/README.md`
- `rvllm/crates/rvllm-block-manager/src/manager.rs`
- `rvllm/crates/rvllm-block-manager/src/prefix_cache.rs`
- `rvllm/kernels/paged_attention.cu`
- `rvllm/kernels/split_kv_attention.cu`
- `rvllm/docs/throughput-optimization-spec.md`

## Executive Summary

Psionic is currently slower for `Gemma 4` for structural reasons:

1. The hot Metal decode path is still host-owned between the projection kernels.
2. The backend-extension layer still reads device buffers to host for RMSNorm,
   RoPE, and SDPA instead of executing those operations natively on Metal.
3. The mixed-device path serializes full hidden-state and KV slices as host
   `Vec<f32>` over blocking HTTP on every token.
4. Psionic still lacks a serious serving scheduler for chunked prefill,
   continuous batching, prefix-block reuse, and async overlap.
5. Psionic still lacks a graph-owned decode path that keeps layer math, cache
   update, and attention in one device-resident plan.

The faster systems differ in the same direction:

- MLX keeps the model math inside tensor primitives that dispatch to real Metal
  kernels.
- `llama.cpp` builds explicit ubatches, explicit KV caches, and explicit compute
  graphs, then fuses or schedules them on Metal with concurrency.
- `vLLM` treats scheduling and KV ownership as first-class serving primitives,
  not incidental helpers around a single-request decode loop.
- `rvLLM` proves that these same design choices fit in a Rust codebase and are
  not blocked by the language choice.

The gap is not "we need one more fast GEMV kernel."

The gap is that Psionic accelerates selected matrix-vector steps while the
competitors accelerate the full decode system.

## Current Psionic Shape

### 1. The Metal Gemma4 hot path is still host-driven

The current Metal stage loop in `crates/psionic-serve/src/gguf.rs` still
materializes host vectors across most of the layer body.

Relevant lines:

- `gguf.rs:3095-3149`
- `gguf.rs:3158-3385`
- `gguf.rs:3387-3423`

What it does today:

- token embedding row decode into host memory
- `input_hidden.to_vec()` for forwarded stage input
- `forwarded_key.to_vec()` and `forwarded_value.to_vec()`
- `live_keys: Vec<Option<Vec<f32>>>`
- `live_values: Vec<Option<Vec<f32>>>`
- CPU `rms_norm(...)`
- CPU `per_head_rms_norm_in_place(...)`
- CPU `apply_rope_neox(...)`
- CPU `attend_impl(...)`
- CPU FFN activation and residual glue

What is accelerated:

- quantized projection matvecs such as Q/K/V, output projection, gate/up/down

What is not accelerated end to end:

- norm
- RoPE
- KV write and read orchestration
- attention score and weighted-value path
- FFN activation glue
- residual and post-norm glue
- per-layer staging and reuse behavior

This means the layer is not a device-resident decoder. It is a Rust host loop
that calls a handful of accelerated matrix kernels and then comes back to host
memory repeatedly.

That is enough to get correctness and some speed.

It is not enough to compete with a serious local engine.

### 2. The Metal backend-extension layer is still a host fallback

The names in `psionic-backend-metal` look like backend-native operations.

The implementation is still host fallback.

Relevant lines:

- `crates/psionic-backend-metal/src/lib.rs:2071-2163`
- `crates/psionic-backend-metal/src/lib.rs:3659-3765`

`compute_decode_attention_f32(...)` currently does this:

- `query.read_f32()`
- `key.read_f32()`
- `value.read_f32()`
- `cos.read_f32()`
- `sin.read_f32()`
- CPU RoPE application
- CPU cache expansion
- CPU SDPA
- return host `Vec<f32>` output

The backend extension ops for:

- `RmsNorm`
- `RotaryEmbedding`
- `ScaledDotProductAttention`

all read tensors back to host, compute there, then allocate and write back to a
Metal buffer.

That is not a device backend in the way MLX or `llama.cpp` mean it.

That is a device-shaped API wrapped around host execution.

### 3. The mixed-device path serializes the whole stage boundary per token

The current distributed Gemma4 front path in `crates/psionic-serve/src/gguf.rs`
is honest, but it is extremely expensive.

Relevant lines:

- `gguf.rs:4033-4049`
- `gguf.rs:4270-4303`
- `gguf.rs:4696-4717`
- `gguf.rs:5013-5069`

The front side runs:

- `forward_stage_step(...)` locally for layers `0..split_layer`
- then sends `input_hidden`, `forwarded_key`, and `forwarded_value` as JSON
  `Vec<f32>` fields over blocking HTTP

The remote side then:

- reconstructs those vectors on the host
- runs its own host-heavy `forward_stage_step(...)`
- returns logits to the front

The actual request struct is:

- `input_hidden: Vec<f32>`
- `forwarded_key: Vec<f32>`
- `forwarded_value: Vec<f32>`

That is a clean first proof surface.

It is not a high-throughput distributed runtime.

The mixed path has no:

- batched stage transport
- binary request transport for activations
- multiple tokens in flight
- overlap between local next-step work and remote suffix compute
- device-resident activation handoff
- stage-local cache ownership beyond host vectors

It pays serialization, allocation, and network overhead on every generated
token.

That is why the mixed path stalls near `2-3 tok/s` even when both endpoints are
working correctly.

## What MLX Does Differently

MLX matters most for the Apple-local lane.

The strongest lessons from MLX are not branding or syntax. The lessons are:

- model math stays inside tensor primitives
- those primitives dispatch to actual Metal kernels
- the runtime exposes real memory controls
- quantized and GEMV paths are device-native and specialized

### 1. MLX exposes RMSNorm, RoPE, and SDPA as real primitives

In `mlx/mlx/fast.cpp`:

- `rms_norm(...)` at `fast.cpp:53-120`
- `rope(...)` at `fast.cpp:474-528`
- `scaled_dot_product_attention(...)` at `fast.cpp:613-760`

These functions do not exist only as host helpers.

They build primitive nodes that dispatch to backend implementations when the
stream is on GPU.

That means the model code can stay graph-owned and tensor-owned while still
landing on Metal kernels.

Psionic currently does the opposite on Gemma4 Metal:

- it accelerates the projections
- then it leaves the graph and computes the rest in Rust on host vectors

### 2. MLX has actual Metal implementations for RoPE and RMSNorm

In `mlx/mlx/backend/metal/rope.cpp:14-163`, `RoPE::eval_gpu(...)`:

- selects a real Metal kernel
- handles single-token inference as a specific fast path
- dispatches threads directly on the Metal command encoder

In `mlx/mlx/backend/metal/kernels/rms_norm.metal:12-156`, the RMSNorm kernel:

- computes mean-square reduction in SIMD groups
- uses threadgroup memory for reduction
- writes the normalized output directly on device

This is the category difference that matters.

Psionic currently has an op called `rms_norm` in the Metal backend, but the
implementation in `crates/psionic-backend-metal/src/lib.rs:3679-3713` is still:

- read device buffer
- compute on CPU
- write device buffer

MLX does not pay that round trip for the normal path.

### 3. MLX has actual specialized SDPA kernels

`mlx/mlx/backend/metal/kernels/scaled_dot_product_attention.metal:9-44`
instantiates specialized SDPA kernels across head sizes and dtypes.

This matters for decode because attention is not a side detail. It is one of
the core costs per token.

Psionic currently computes decode attention in host memory for the Metal lane.

That alone keeps Psionic out of the same performance class.

### 4. MLX treats quantization and GEMV as tuned backend work

Relevant files:

- `mlx/mlx/backend/metal/quantized.cpp:1611-1725`
- `mlx/mlx/backend/metal/kernels/gemv.metal:448-546`

MLX does not just have a generic "matvec" concept.

It has:

- quantization-aware dispatch
- row-contiguity normalization
- kernel selection by mode and bit width
- instantiated GEMV families with different tile shapes

Psionic has started some of this work.

The important difference is that MLX combines this with device-resident
norm/rope/attention and memory management, while Psionic currently only has the
projection part of that closure.

### 5. MLX exposes memory and cache controls as runtime knobs

`mlx/mlx/memory.h:11-80` exposes:

- active memory
- peak memory
- cache memory
- memory limits
- cache limits
- explicit cache clearing
- wired memory limits for Metal

That is important because a serious Apple-local runtime is not only about one
kernel. It is also about allocator and residency policy.

Psionic currently has no comparable Gemma4-specific memory regime on the Metal
decode path. The execution shape is still dominated by temporary host vectors
and repeated buffer staging.

## What `llama.cpp` Does Differently

`llama.cpp` matters most as the portable local-engine and GGUF execution
reference.

The important differences are:

- ubatch splitting
- explicit KV ownership
- graph construction
- fused KV write and RoPE placement
- backend scheduler and Metal graph execution

### 1. `llama.cpp` explicitly builds ubatches before compute

`llama.cpp/src/llama-batch.h:71-121` defines `llama_batch_allocr` with
dedicated split strategies:

- `split_simple`
- `split_equal`
- `split_seq`
- `ubatch_reserve`

`llama.cpp/src/llama-memory-hybrid.cpp:62-112` then uses those ubatches to:

- split the batch
- prepare recurrent memory
- prepare attention cache
- only then return a batch context for compute

Psionic's current Gemma4 lane still behaves mostly like:

- one request
- one token step
- one host-owned decode loop

That is enough for proof-of-life serving.

It is not a serving scheduler.

### 2. `llama.cpp` owns KV cache as a first-class backend object

`llama.cpp/src/llama-kv-cache.cpp:79-259` allocates KV tensors by backend
buffer type, per layer, with optional offload and stream-aware views.

Important properties:

- buffer types are selected per backend device
- KV tensors are allocated in backend memory, not host-only containers
- per-stream views exist up front
- reuse and layer remapping are part of cache ownership

Psionic's Gemma4 Metal lane currently keeps live K/V in host `Vec<f32>` and
stores per-step slices into an in-memory host cache.

That is a large design gap.

### 3. `llama.cpp` places RoPE fusion and KV store inside the graph

`llama.cpp/src/llama-graph.cpp:2097-2113` is explicit:

- build Q/V/K into the graph together
- delay K expansion to enable RoPE fusion
- write directly into KV cache via `cpy_k(...)` and `cpy_v(...)`

This sentence in the source matters:

- `expand k later to enable rope fusion which directly writes into k-v cache`

That is exactly the sort of operation ordering Psionic does not yet have.

Psionic currently:

- computes Q/K/V projection
- pulls them to host vectors
- applies RoPE on host
- copies K/V into host cache vectors
- re-reads them for attention

That is not a fused cache update path.

### 4. `llama.cpp` has an actual backend graph scheduler on Metal

Relevant files:

- `llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:26-82`
- `llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:438-615`
- `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:28-220`

`ggml_metal_graph_compute(...)`:

- submits a graph, not a hand-written layer loop
- uses multiple command buffers
- optionally encodes work concurrently
- keeps graph execution asynchronous
- supports graph optimization and capture

`ggml-metal-ops.cpp`:

- filters nodes
- checks fusibility
- checks concurrency hazards with memory-range tracking
- encodes supported ops directly on the Metal backend

Psionic's current Gemma4 Metal decoder does not have any equivalent graph-owned
decode layer.

It has a host loop that happens to call Metal projection kernels.

### 5. `llama.cpp` also has deep Metal kernel coverage

The Metal backend is not a thin wrapper.

`llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` contains dedicated kernels and
function-constant specializations for:

- RoPE, starting around `ggml-metal.metal:4102-4152`
- flash attention variants, starting around
  `ggml-metal.metal:5532-5608`

This matters because the backend surface and the kernel surface align.

Psionic currently exposes backend-extension names for similar operations, but
those entrypoints still fall back to host execution.

## What `vLLM` Does Differently

`vLLM` matters most as the serving scheduler and KV-memory reference.

The big differences are:

- continuous batching is a first-class design assumption
- chunked prefill is built into scheduler config and policy
- KV cache is block-managed
- prefix caching is block-based
- fusion between RoPE and KV cache update is explicit
- async scheduling and connector paths exist for PD and disaggregation

### 1. `vLLM` treats throughput as a scheduler problem and a KV problem

`vllm/README.md:28-48` states the stack's speed claims directly:

- PagedAttention
- continuous batching
- CUDA/HIP graph
- optimized kernels
- chunked prefill
- prefix caching
- distributed parallelism

That list is useful because it names the real categories of work.

Psionic currently has pieces of the kernel story for selected local paths.

It does not yet have the scheduling and KV system that make those kernels add
up to a high-throughput server.

### 2. `vLLM` scheduler config exposes the real throughput knobs

`vllm/vllm/config/scheduler.py:42-154` defines:

- `max_num_batched_tokens`
- `max_num_seqs`
- `max_num_partial_prefills`
- `max_long_partial_prefills`
- `long_prefill_token_threshold`
- `enable_chunked_prefill`
- `scheduler_reserve_full_isl`
- `async_scheduling`
- `stream_interval`

This is not an afterthought.

It means prompt behavior, decode behavior, admission policy, and scheduling
overlap are part of the primary configuration model.

Psionic's current Gemma4 path has nothing equivalent for the live server lane.

### 3. `vLLM` scheduler owns running, waiting, KV, encoder, and connector state

`vllm/vllm/v1/core/sched/scheduler.py:67-243` initializes one scheduler that
owns:

- waiting queue
- skipped-waiting queue
- running requests
- finished request tracking
- KV connector
- encoder cache manager
- `KVCacheManager`
- optional pipeline/context-parallel implications

This is the right shape for a high-throughput server.

Psionic's current Gemma4 server path is still much closer to a direct request
executor than to a real scheduler.

### 4. `vLLM` KV cache manager and block pool are explicit subsystems

Relevant files:

- `vllm/vllm/v1/core/kv_cache_manager.py:106-310`
- `vllm/vllm/v1/core/block_pool.py:129-320`

Important properties:

- block pool owns physical KV blocks
- cached blocks are hash-addressed
- full-sequence fit is an admission gate
- prefix caching is block-based, not string-based folklore
- requests allocate slots against the block manager
- eviction and residency are explicit

Psionic currently does not have a Gemma-serving KV subsystem at this level.

It has request-local caches and per-token host vector management.

That is sufficient for a proof harness.

It is not sufficient for `vLLM`-class serving behavior.

### 5. `vLLM` fuses RoPE and KV cache update

`vllm/vllm/compilation/passes/fusion/rope_kvcache_fusion.py:33-229`
registers a fused custom op for:

- rotary embedding
- unified KV cache update

The docstring is explicit:

- the fused op eliminates separate kernel launches
- it removes intermediate memory operations between RoPE and cache update

Psionic is currently on the opposite side of that design:

- RoPE happens on host
- cache write happens on host
- attention then reconsumes host-side cache material

This is one of the clearest single differences between a serious decode path
and the current Psionic Gemma4 Metal path.

## What `rvLLM` Proves In Rust

`rvLLM` matters because it weakens the common excuse that Psionic is slow
because it is Rust-native.

The language is not the blocker.

The current system shape is the blocker.

### 1. `rvLLM` explicitly centers the scheduler, paged KV, CUDA graph, and JIT kernels

`rvllm/README.md:146-188` describes its inference pipeline as:

- Scheduler
- continuous batching
- block manager with paged KV
- CUDA graph replay
- JIT fused kernels

That is the right decomposition for throughput work.

Psionic's current Gemma4 story is still dominated by decoder correctness and
backend admission, not by a scheduler-first runtime.

### 2. `rvLLM` block manager has CoW, watermarking, and prefix sharing

`rvllm/crates/rvllm-block-manager/src/manager.rs:104-364` implements a real
block manager with:

- logical-to-physical block mapping
- GPU and CPU pools
- copy-on-write pending state
- watermark reservation
- optional prefix cache
- `can_allocate(...)`
- `allocate(...)`
- `register_prefix(...)`
- `free(...)`
- `evict_prefix_block(...)`
- `fork(...)`

This is what a serving substrate looks like when KV memory is treated as a real
system.

Psionic currently has no comparable block manager for the Gemma serving lane.

### 3. `rvLLM` prefix caching is block-boundary hashing with LRU and refcounts

`rvllm/crates/rvllm-block-manager/src/prefix_cache.rs:1-270` is explicit:

- hashes prompt token prefixes at block boundaries
- only reuses blocks when the full prefix matches
- increments refcounts on hit
- uses LRU eviction for inactive blocks

This is much stronger than ad hoc prompt reuse.

Psionic's current mixed and local Gemma paths have no equivalent prefix-block
cache.

### 4. `rvLLM` already has kernels shaped around paged and split-KV attention

Relevant files:

- `rvllm/kernels/paged_attention.cu:1-245`
- `rvllm/kernels/split_kv_attention.cu:1-260`

The paged attention kernel:

- operates over block tables
- uses online softmax
- handles variable-length sequences
- supports f16-KV cache reads promoted to f32 compute

The split-KV kernel:

- splits KV tiles across multiple thread blocks for one `(seq, head)`
- writes partial outputs and then combines them
- is shaped for higher decode throughput on large contexts

Psionic's distributed Gemma path is not in this class yet.

It ships whole activation and KV slices across HTTP and then continues a host
loop on the remote side.

### 5. `rvLLM`'s own throughput gap doc names the same categories Psionic is still missing

`rvllm/docs/throughput-optimization-spec.md:1-48` says its remaining gap to
Python `vLLM` is:

- per-step memory allocations
- missing CUDA graph replay in production decode
- separate Q/K/V and gate/up paths
- CPU/GPU synchronization points
- kernel launch overhead
- scheduler overlap problems
- KV access pattern issues

This matters because Psionic is behind both systems.

The categories that `rvLLM` still sees as the last `vLLM` gap are categories
that Psionic still lacks more fundamentally.

## Cross-System Difference Matrix

| System property | Psionic today | MLX / `llama.cpp` local references | `vLLM` / `rvLLM` serving references |
| --- | --- | --- | --- |
| Decoder ownership | host Rust loop with accelerated projections | graph or primitive owned | scheduler plus graph/kernels |
| RMSNorm | Metal backend host fallback | real backend kernel | fused or backend-native paths |
| RoPE | host math in Gemma path | real backend kernel or graph op | fused with cache update in `vLLM`, fused kernels in `rvLLM` |
| Attention | host SDPA in Metal Gemma path | backend-native or graph-native | paged attention / flash attention |
| KV cache | request-local host vectors and cache mirrors | explicit backend-owned KV tensors and slot logic | block-managed KV cache with admission and eviction |
| Prefix cache | none for Gemma serving path | some local reuse patterns, backend-aware | first-class block hash cache |
| Prefill | mostly decode-shaped replay in current Gemma lane | explicit prompt/prefill work | chunked prefill is core scheduler policy |
| Continuous batching | no real Gemma serving scheduler | limited local engine concern | core throughput primitive |
| Graph capture / replay | absent in Gemma live path | graph compute on Metal in `llama.cpp` | CUDA/HIP graph in `vLLM`, CUDA graph in `rvLLM` |
| Distributed stage handoff | JSON `Vec<f32>` over blocking HTTP per token | not the main local reference | KV connectors, PD, paged/block-aware transport |

## What Actually Needs To Change

### 1. Build a real device-resident Gemma4 Metal decoder

This is the first major job.

The target is not "make CPU glue faster."

The target is:

- Q/K/V projection on device
- RMSNorm on device
- RoPE on device
- KV write on device
- attention on device
- FFN activation on device
- residual and post-norm glue on device
- minimal host visibility until logits or explicit receipts are needed

Without this, Psionic will keep losing to MLX- and `llama.cpp`-class Apple
local engines regardless of small loop tuning.

### 2. Fuse RoPE and KV cache update on both Metal and CUDA decode paths

This is one of the cleanest common lessons from `llama.cpp` and `vLLM`.

The desired state is:

- do not materialize post-RoPE K on host
- do not write K/V into host mirrors before attention
- write directly into the cache structure that attention will consume

Until that exists, Psionic pays extra memory traffic and extra orchestration on
every token.

### 3. Split prompt/prefill runtime from decode runtime

The current Gemma path is still too decode-shaped.

Serious serving stacks do not treat prompt ingest and decode as the same loop
with a different stopping condition.

Psionic needs:

- prompt chunking
- prompt-side KV construction on device
- prefix-block reuse
- separate prompt and decode receipts
- separate prompt and decode benchmarking

This matters for both local and distributed serving.

### 4. Build a real KV block manager for the serving lane

Psionic needs a serving-owned KV subsystem with:

- block allocation
- free lists
- prefix-block hashing
- eviction policy
- full-sequence admission checks
- explicit ownership across requests and possibly across peers

Until then, every higher-level serving optimization is working around a missing
foundation.

### 5. Add a real Gemma-serving scheduler

Psionic needs a scheduler that owns:

- waiting requests
- running requests
- prompt chunking
- decode batching
- prefix hits
- preemption or admission policy
- async overlap with compute
- distributed-path aware scheduling

This is the category where `vLLM` is strongest and where Psionic is still
furthest behind.

### 6. Replace per-token host HTTP activation handoff with a device-aware transport

The current mixed-device path is good as a proof harness and bad as a runtime.

The next serious distributed stage needs:

- binary activation transport for requests, not JSON floats
- stage-owned cache continuity
- multiple tokens or requests in flight
- overlap between local prefix work and remote suffix work
- backpressure and batching
- eventually stage-local graph replay

The current `split_layer` sweep result is useful, but split choice will not
save the path while the transport stays host-vector and blocking.

### 7. Add graph-owned decode plans

Psionic needs a decode plan that is closer to:

- a per-layer or per-stack compiled plan
- capture or replay friendly
- explicit buffer liveness and reuse
- explicit backend residency

That is the only way to stop paying host orchestration cost on every layer,
every token.

## What Will Not Close The Gap By Itself

These changes can help, but they are not enough:

- replacing more scalar helpers with BLAS or vDSP
- tuning only the split layer for mixed-device runs
- adding one more projection kernel while norm/rope/attention stay on host
- renaming backend-extension ops without changing their execution ownership
- focusing only on TTFT while decode and transport remain host-heavy
- trying to out-benchmark `vLLM` without building a scheduler and KV manager

These are incremental improvements on top of a still-wrong system shape.

## What The Faster Systems Have In Common

The important common pattern is simple:

1. They keep decode math on device.
2. They make KV memory a first-class subsystem.
3. They separate prompt handling from decode handling.
4. They fuse RoPE, cache update, and attention-adjacent work where possible.
5. They treat scheduling and transport as part of inference performance, not as
   a layer above it.

Psionic currently does only part of item 1 and almost none of items 2 through
5 for the live Gemma4 lane.

## Concrete Program For Psionic

### Phase 1: local Apple closure

- replace the host fallback implementations of Metal `RmsNorm`,
  `RotaryEmbedding`, and `ScaledDotProductAttention` with real Metal kernels
- move Gemma4 layer glue out of `Vec<f32>` and into device buffers
- add fused RoPE plus KV write on Metal
- benchmark prompt and decode separately

Success bar:

- the Gemma4 Metal path no longer reads Q/K/V/cos/sin back to host for the
  normal decode step
- the gap to the fast local reference drops because the system shape changed,
  not because one scalar loop got shorter

### Phase 2: single-node serving closure

- introduce a serving-owned KV block manager
- introduce prefix-block reuse
- introduce chunked prefill
- introduce a scheduler for prompt and decode batching
- introduce graph-owned decode plans where the backend supports it

Success bar:

- the server is no longer "one request, one host loop"
- Psionic can make meaningful throughput claims on the serving surface instead
  of only correctness claims

### Phase 3: distributed closure

- replace JSON activation transport with binary transport
- allow more than one token or request in flight
- batch or pipeline stage work
- make distributed cache ownership explicit per stage
- connect distributed scheduling to the same KV and prompt/decode runtime

Success bar:

- mixed-device throughput is not dominated by per-token host serialization
- split execution is a real runtime, not only a clean proof harness

## Bottom Line

Psionic is slower than the faster reference systems because the hot path is
still too host-owned and too request-local.

MLX, `llama.cpp`, `vLLM`, and `rvLLM` differ from Psionic in the same direction:

- more work stays on device
- more of the runtime is graph-owned
- KV memory is a real subsystem
- prompt and decode are scheduled explicitly
- distributed execution is treated as a transport and scheduling problem, not
  only a correctness proof

The next real throughput jump for Psionic will come from copying that system
shape into Rust and into the owned backends.

It will not come from pretending the current Metal backend extensions are
already kernel-native or from doing one more scalar cleanup pass inside the
host loop.

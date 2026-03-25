# Psionic TurboQuant Integration Audit

> Status: written 2026-03-24 after reviewing `docs/turboquant.md`,
> `docs/ROADMAP.md`, `docs/ROADMAP_MLX.md`, `docs/INFERENCE_ENGINE.md`,
> `docs/deep-research-perf.md`, `docs/METAL_GPT_OSS_UNIFIED_PLAN.md`,
> `docs/QUANTIZATION_CAPABILITY_SEMANTICS.md`,
> `docs/EXTENSION_CONTRACT_SEMANTICS.md`,
> `crates/psionic-serve/src/lib.rs`, `crates/psionic-serve/src/gpt_oss.rs`,
> `crates/psionic-runtime/src/lib.rs`,
> `crates/psionic-backend-cuda/src/lib.rs`, `crates/psionic-models`,
> `crates/psionic-provider`, `crates/psionic-router`,
> `crates/psionic-datastream`, `crates/psionic-data`, and the live Psionic
> GitHub issue tracker.

## Summary

TurboQuant belongs in Psionic as an approximate KV-cache optimization for
served text generation.

The first valid landing surface is the CUDA decode path in:

- `psionic-serve`
- `psionic-backend-cuda`
- `psionic-provider`
- `psionic-eval`

It does not belong first in:

- weight quantization dispatch
- exact execution lanes
- Tassadar exactness claims
- training or optimizer state
- `psionic-datastream` generic tensor compression
- `psionic-data` retrieval work that does not exist yet

The current repo has embeddings surfaces but no public vector index or ANN
subsystem. Retrieval-side TurboQuant should stay out of scope until that
surface exists.

The current CUDA implementation also makes the integration boundary clear:

- host KV state is still stored as dense `Vec<f32>`
- the served CUDA path mirrors that state into device buffers
- decode attention kernels consume f16 KV buffers
- capability and receipt surfaces do not yet name KV-cache encoding

TurboQuant is therefore not a policy toggle. It requires:

- device-resident decode KV ownership
- explicit KV-cache encoding contracts
- native CUDA decode kernels for quantized KV
- benchmark and refusal publication before rollout

## Current State

### 1. Served CUDA decode still uses dense host KV plus an f16 device mirror

`crates/psionic-serve/src/lib.rs` stores served KV state in host memory as:

- `KvCacheEntry { key: Vec<f32>, value: Vec<f32> }`
- `InMemoryKvCache`

`crates/psionic-serve/src/gpt_oss.rs` then mirrors that host cache into CUDA
buffers through:

- `CudaKvCacheMirror::from_host_cache(...)`
- `forward_step_with_cuda_plan(...)`
- `encode_cuda_forward_step_submission(...)`

The backend kernel family is still `attention_decode_rope_cache_f16_kv*`.
There is no TurboQuant or generic quantized-KV kernel family in the repo.

TurboQuant cannot land as a request flag on this path. The active storage and
kernel contract both need to change.

### 2. The hot path still carries host-owned KV work

The CUDA generation loop still appends KV back into the host cache and can
materialize host KV from the device mirror. `docs/deep-research-perf.md`
already calls out the cost of that ownership model.

TurboQuant should not be layered on top of per-token host append and readback.
That would preserve the wrong memory boundary and reduce the value of the
optimization.

### 3. KV memory planning is still dense-float accounting

`default_kv_cache_policy(...)` and `default_decoder_memory_plan(...)` compute
bytes-per-token from dense widths and float element sizes.

That is correct for the current implementation. It is not correct for a cache
codec with separate host and device geometry.

TurboQuant needs explicit encoding-aware accounting.

### 4. Quantization vocabulary is weight-centric

Psionic already exposes weight quantization semantics through
`QuantizationMode`, dispatch decisions, and model metadata.

Those surfaces do not name KV-cache encoding. Reusing them for TurboQuant would
mix weight format and cache codec concerns into one namespace.

That would make compatibility, invalidation, and receipts harder to reason
about.

### 5. Capability and receipt publication is already strong enough to extend

Psionic already publishes:

- cache policy
- cache observations
- residency
- delivery proof
- execution profile

Those surfaces are the correct place to add cache-encoding truth. The gap is
not publication structure. The gap is missing cache-encoding vocabulary.

### 6. The router is a follow-on surface, not the first one

The router does not currently reason about cache codecs. That is acceptable for
an initial local serve integration.

Router work should follow explicit provider capability publication. It should
select on declared cache-encoding support, not infer from backend names.

### 7. The issue tracker does not yet represent the work

There is no current Psionic issue for:

- TurboQuant
- QJL
- PolarQuant
- KV-cache quantization
- retrieval-side quantized indexing

The closest existing issue evidence is indirect:

- `#14` `PMLX-701` treats quantized KV-cache behavior as a package-level text
  serving concern
- `#109` `TAS-045` establishes quantization-aware truth envelopes as a valid
  publication pattern
- `#308` `TAS-184A` establishes that KV state needs explicit semantic
  discipline
- `#468` and `#470` reinforce the need to remove host-owned hot-path work
  before making strong performance claims

## Integration Target

### Primary target

TurboQuant should land first on the served CUDA text stack:

- `psionic-serve`
- `psionic-backend-cuda`
- `psionic-provider`
- `psionic-eval`

This is where Psionic already has:

- decode kernels
- KV residency logic
- prefix-cache reuse
- receipts
- capability publication

This is also where KV memory and bandwidth pressure is already visible.

### Secondary target

Metal should follow only after:

- CUDA benchmark closure
- explicit device-resident KV ownership on the Metal path

`docs/METAL_GPT_OSS_UNIFIED_PLAN.md` still lists device-resident KV work as
planned.

### Deferred target

Retrieval-side TurboQuant should stay deferred until Psionic has a public
retrieval/index subsystem in `psionic-data` or another explicit retrieval
surface.

The current repo does not provide that.

## Required Workstreams

### Workstream 1: Make decode KV device-resident

Problem:

The active served decode path still treats dense host KV as the working source
of truth.

Hypothesis:

TurboQuant becomes practical only if active decode KV lives on device and host
materialization is deferred to persistence, replay, fallback, or debugging
paths.

Surface:

- `psionic-serve`
- `psionic-backend-cuda`
- `psionic-provider`

Claim class:

Systems prerequisite.

Benchmark / tests:

- unit tests for append and read behavior
- deterministic replay tests for cache reconstruction
- decode throughput benchmarks
- H2D and D2H traffic accounting
- prefix reuse correctness tests
- fallback-path tests

Exit criteria:

Supported CUDA decode no longer requires per-token host `Vec<f32>` KV append
or readback on the hot path.

### Workstream 2: Add explicit KV-cache encoding contracts

Problem:

Psionic can publish weight quantization, but it cannot yet publish KV-cache
encoding.

Hypothesis:

Psionic needs separate cache-specific contracts instead of overloading weight
quantization enums.

Surface:

- `psionic-runtime`
- `psionic-provider`
- `psionic-serve`

Recommended contract family:

- `KvCacheEncoding`
- `KvCacheEncodingPolicy`
- `KvCacheEncodingAccounting`

The policy should carry:

- algorithm id
- objective id
- bits per channel or effective bytes per token
- block or group shape
- outlier policy
- projection or rotation id
- codebook or build id
- model-family support bound
- context-length support bound
- host and device memory geometry

Claim class:

Capability and compatibility publication.

Benchmark / tests:

- serialization tests
- capability-matrix publication tests
- receipt publication tests
- cache invalidation tests on encoding mismatch
- route-selection tests once router support exists

Exit criteria:

Requests, receipts, and capability reports can state exactly which cache
encoding was used, downgraded, or refused.

### Workstream 3: Add CUDA TurboQuant KV storage and decode kernels

Problem:

Current CUDA attention kernels assume f16 KV buffers.

Hypothesis:

Psionic needs a native quantized-KV storage path and decode kernel family that
operate on TurboQuant-encoded cache state directly.

Surface:

- `psionic-backend-cuda`
- `psionic-serve`

Implementation shape:

- preserve the current f16 path as the dense baseline
- replace hard-coded `CudaKvCacheMirror` assumptions with a cache-storage
  abstraction for served decode
- add a TurboQuant storage layout and append path
- add TurboQuant decode attention kernels
- retain explicit dense fallback when the backend, model family, or context
  tier is unsupported

Design rule:

Do not add TurboQuant as a new weight `QuantizationMode`. This is a cache codec
lane.

Objective selection:

The paper family supports different optimization objectives. Psionic should
store the objective in the cache policy and benchmark it explicitly.

The safest starting assumption is:

- key cache uses a product-preserving objective
- value cache objective stays benchmark-gated until end-to-end ablations close

That key/value split is an engineering inference from the paper family. It is
not current Psionic evidence.

Claim class:

Approximate served-generation optimization.

Benchmark / tests:

- CUDA kernel correctness tests against dense reference decode
- long-context quality benchmarks against dense f16 KV
- memory reduction benchmarks
- decode tokens/sec benchmarks
- median and tail-latency benchmarks under concurrency
- OOM-threshold benchmarks
- fallback and refusal tests

Exit criteria:

At least one production model family shows a published memory win and a decode
throughput win with quality loss kept inside a declared benchmark envelope.

### Workstream 4: Publish refusal posture and benchmark package before rollout

Problem:

An approximate cache codec can silently leak into routes that require exactness
or unsupported model behavior.

Hypothesis:

TurboQuant should be published only as a bounded approximate served-inference
capability with explicit downgrade and refusal behavior.

Surface:

- `psionic-eval`
- `psionic-provider`
- `psionic-serve`
- `psionic-router`

Required refusal posture:

- do not publish TurboQuant for CPU-reference exact lanes
- do not publish TurboQuant for Tassadar exactness lanes
- do not reuse prefix artifacts across incompatible cache-encoding signatures
- do not silently fall through to unsupported kernels
- when a route requests an unsupported cache encoding, return explicit refusal
  or explicit dense downgrade in the receipt

Required benchmark package:

- exactness lane: state directly that TurboQuant is not an exactness feature
- quality lane: long-context tasks against dense baseline
- generalization lane: context tiers above the validated band
- cost lane: bytes per token, max live sessions, and decode throughput
- latency lane: median and tail latency under concurrency
- refusal lane: unsupported model, backend, route, and prefix-reuse cases

Claim class:

Capability publication and route discipline.

Benchmark / tests:

- benchmark package coverage tests
- refusal-path tests
- capability publication tests
- route-selection tests

Exit criteria:

TurboQuant appears as a separate approximate capability lane with explicit
downgrade and refusal behavior.

### Workstream 5: Add router support after serve capability exists

Problem:

The router cannot yet reason about KV-cache codecs.

Hypothesis:

Router support should follow successful local serve integration and explicit
provider capability publication.

Surface:

- `psionic-router`
- `psionic-provider`

Claim class:

Placement and rollout control.

Benchmark / tests:

- route-selection tests
- capability-matrix publication tests
- shared-prefix compatibility tests across pools

Exit criteria:

The router can prefer, require, or exclude TurboQuant-capable pools using
explicit capability data.

## What Should Not Happen

- Do not treat TurboQuant as evidence for exact execution or compiled bounded
  exactness.
- Do not claim generic quantization parity because a KV-cache codec exists.
- Do not add retrieval-side TurboQuant before a real public retrieval/index
  subsystem exists.
- Do not bury cache encoding inside backend-specific flags.
- Do not reuse weight quantization enums as a cache codec namespace.
- Do not promote a TurboQuant lane without a benchmark package that separates
  quality, length generalization, cost, latency, and refusal behavior.

## Recommended Issue Stack

1. `PLIB-303A` Make active GPT-OSS decode KV device-resident and defer host
   materialization.
2. `PLIB-303B` Add KV-cache encoding policy, accounting, capability
   publication, and receipt fields.
3. `PLIB-403A` Add CUDA TurboQuant KV storage and decode kernels for served
   generation.
4. `PLIB-308A` Add TurboQuant benchmark package and refusal-path coverage.
5. `PLIB-307A` Add cache-encoding-aware routing and capability publication.
6. `PMLX-follow-on` Add Metal TurboQuant parity only after device-resident KV
   and CUDA benchmark closure.

Those issues should map back to the existing roadmap docs instead of creating a
separate backlog.

## Bottom Line

TurboQuant belongs in Psionic as an approximate served-generation KV-cache
lane.

The right first integration is:

- `psionic-serve` for cache ownership, policy, provenance, and receipts
- `psionic-backend-cuda` for native quantized KV decode kernels
- `psionic-provider` for capability and receipt publication
- `psionic-eval` for benchmark and refusal coverage

The wrong first integration is:

- weight quantization dispatch
- generic datastream compression
- embeddings-only wiring with no retrieval engine
- exact execution claims

Psionic already has the right publication and cache-truth surfaces. It does not
yet have the right runtime ownership model, cache-encoding contract, or CUDA
kernel path. Those are the exact integration gaps that must close first.

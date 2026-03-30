# PSION RVLLM Memory Pool

> Status: landed on 2026-03-30 as the ninth retained RVLLM runtime-harvest
> packet.

This document records the admitted CUDA allocator-pool and allocation-reuse
posture for native `qwen35` and native `gpt_oss`.

The important truth here is not that Psionic needed a second allocator stack.
It did not. The gap was that `psionic-backend-cuda` already declared an
allocator-pool contract in runtime resources, but the admitted serving lane did
not yet back that contract with an actual exact-spec reuse path and explicit
benchmark evidence.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_memory_pool_v1.json`

Current retained truth:

* packet digest `7a44a574119308084a208efb380b3712f7f3595ad8526ec152eff4032840af10`
* admitted paths:
  - `qwen35.native_cuda_decode`
  - `gpt_oss.native_cuda_decode`
* allocator-pool posture:
  - `mode = exact_tensor_spec`
  - `max_cached_buffers = 128`
  - `max_cached_bytes = 67108864`
* allocator-reuse telemetry:
  - `cold_allocations`
  - `reuse_hits`
  - `returned_buffers`
  - `evicted_returns`
  - `cached_buffers`
  - `cached_bytes`

Retained benchmark comparison:

| Path | Before allocations/step | After allocations/step | Before p50 step ms | After p50 step ms | Before tok/s | After tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen35.native_cuda_decode` | `12` | `5` | `21.4` | `18.6` | `53.1` | `61.0` |
| `gpt_oss.native_cuda_decode` | `18` | `7` | `33.7` | `29.8` | `42.5` | `47.9` |

Long-run stability:

* retained run id `rvllm-memory-pool-steady-state-v1`
* pool budget bytes `67108864`
* steady-state growth bytes `7864320`
* `leak_detected = false`
* `fragmentation_regression = false`

Claim boundary:

* This packet claims only the admitted exact-spec CUDA reuse lane.
* It does **not** claim a general allocator rewrite, unbounded retention, or
  pooled safety outside the declared tensor-spec envelope.
* Returned buffers still obey the explicit pool budget, and over-budget returns
  are evicted rather than hidden.

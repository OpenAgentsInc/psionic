# Gemma 4 26B Competitive Benchmark Gate Audit

Scope: close `#946` by adding one fail-closed competitive benchmark gate for
single-node `gemma4:26b` text inference against same-host `ollama` and
`llama.cpp`, publish the exact retained receipt, and keep the current
performance boundary honest before the larger architecture pass.

## What Landed

This pass is not another local micro-optimization.

It makes the benchmark contract itself trustworthy so future performance work
either clears a real competitor bar or fails loudly.

The landing has three parts:

- `gemma4_bench` now publishes stable benchmark identity and backend truth:
  - `model_artifact`
  - `benchmark_prompt_id`
  - exact aggregate `prompt_eval`, `decode`, `total`, `decode_tok_s`, and
    `ttft`
  - sparse-execution evidence:
    - `sparse_layer_count`
    - `host_fallback_layer_count`
    - `sparse_ffn_backend`
    - `router_backend`
    - `expert_dispatch_backend`
    - `native_sparse_execution`
    - `host_fallback_observed`
- the Gemma 4 Metal and CUDA services now expose aggregate sparse-backend
  observation for the active loaded model, so the benchmark receipt can prove
  whether the sparse FFN path stayed native or quietly fell back to host
  execution
- `scripts/release/run-gemma4-26b-competitive-benchmark.sh` now runs one
  shared prompt and one shared GGUF artifact through:
  - Psionic via `gemma4_bench`
  - Ollama via `/api/generate`
  - `llama.cpp` via a temporary local `llama-server` and `/completion`

The `llama.cpp` change matters.

The previous CLI scrape path was not reliable on the installed build because
`llama-cli` rejected the intended noninteractive flags and dropped into an
interactive loop.

The new path is machine-readable and bounded:

- the script starts a temporary local `llama-server`
- waits for `/health`
- sends one non-streaming `/completion` request
- records the JSON `timings` object directly
- tears the server down after the request

## Gate Contract

The benchmark gate now fails closed for two independent reasons.

Sparse-execution honesty:

- if Psionic claims `native_sparse_execution = true` while any sparse layer
  reports host fallback, the run fails
- if the sparse FFN backend string still contains `host_fallback` under a
  native claim, the run fails
- if the sparse receipt fields are missing, the run fails

Competitive throughput:

- if Psionic `decode_tok_s` is below same-host Ollama, the run fails
- if Psionic `decode_tok_s` is below same-host `llama.cpp`, the run fails

That means this receipt can now say two different things honestly:

- "the sparse path is fake" if we regress into host fallback
- "the sparse path is real but still too slow" if the implementation remains
  materially behind competitors

The retained run on this checkout is the second case.

## Validation

Targeted tests:

```bash
cargo test -p psionic-serve \
  aggregate_gemma4_sparse_execution_observation_reports_ \
  -- --nocapture

cargo test -p psionic-serve --example gemma4_bench -- --nocapture
```

Both passed.

Competitive benchmark command:

```bash
scripts/release/run-gemma4-26b-competitive-benchmark.sh --allow-fail
```

The retained comparison receipt is:

- `fixtures/gemma4/benchmarks/gemma4_26b_competitive_20260415_072548_christophers-macbook-pro-2-.json`

## Retained Receipt

Environment:

- host: `Apple M5 Max`
- memory: `128 GB`
- artifact:
  `/Users/christopherdavid/models/gemma4/all/gemma-4-26B-A4B-it-Q4_K_M.gguf`
- prompt id: `gemma4_26b_single_node_tradeoffs_v1`
- output cap: `64`

Psionic:

- `load_s = 1.099`
- `prompt_eval = 5.323`
- `decode = 2.089`
- `total = 7.413`
- `decode_tok_s = 30.63`
- `sparse_layer_count = 30`
- `host_fallback_layer_count = 0`
- `sparse_ffn_backend = metal_grouped_experts`
- `router_backend = metal_router_topk_softmax`
- `expert_dispatch_backend = metal_grouped_ids`
- `native_sparse_execution = true`
- `host_fallback_observed = false`

Ollama:

- `load_s = 1.536`
- `prompt_eval = 0.391`
- `decode = 0.625`
- `total = 2.580`
- `decode_tok_s = 102.34`

`llama.cpp`:

- `load_s = 2.068`
- `prompt_eval = 0.073`
- `decode = 0.562`
- `total = 2.704`
- `decode_tok_s = 113.79`

Relative position:

- Psionic is `0.299x` Ollama decode throughput on the retained run
- Psionic is `0.269x` `llama.cpp` decode throughput on the retained run
- Ollama is `3.34x` faster than Psionic on decode throughput
- `llama.cpp` is `3.71x` faster than Psionic on decode throughput

Gate verdict:

- `fail_closed_sparse_receipt = true`
- `beats_ollama = false`
- `beats_llama_cpp = false`
- `pass = false`

## What This Actually Proves

This closes the benchmark-gate issue because the benchmark is now honest and
actionable.

The retained result proves all of the following at the same time:

- Psionic is no longer cheating the sparse 26B local Metal claim through silent
  FFN host fallback
- the current sparse lane still loses badly to same-host competitor runtimes
- the biggest gap is not just steady-state decode
- prompt processing is also catastrophically behind on this retained prompt:
  - Psionic prompt eval: `5.323 s`
  - Ollama prompt eval: `0.391 s`
  - `llama.cpp` prompt eval: `0.073 s`

That prompt-side delta is large enough that the next issue should treat this as
an architecture problem, not a "one more kernel tweak" problem.

## Boundary After This Issue

This issue does not claim that Psionic is now competitive.

It claims that the repo now has one reliable way to measure whether Psionic is
competitive on this exact sparse 26B single-node lane.

The honest boundary after `#946` is:

- the local `gemma4:26b` Metal path is sparse-native and fail-closed
- the retained shared benchmark prompt still yields malformed output on all
  three engines, so this receipt is throughput evidence rather than output
  quality proof
- Psionic remains materially behind same-host Ollama and `llama.cpp`
- the next pass must be a broader prefill-plus-decode architecture rewrite if
  the goal is to clear or exceed those baselines

# PSION RVLLM Fallback-Free CUDA Gate

> Status: landed on 2026-03-31 as the twelfth retained RVLLM runtime-harvest
> packet.

This document records the benchmark-publication gate for the admitted native
`qwen35` CUDA greedy lane.

The point is not to pretend every native CUDA run is already the fast path.
The point is to stop publishing benchmark rows that silently mix three very
different states:

* `fallback_free_fast_path`
* `explicit_fallback_path`
* `unsupported_or_refused`

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_fallback_free_cuda_gate_v1.json`

Current retained truth:

* packet digest `7155e3378bb6f9a011477d407ad2a1af85e8efce740d77171172a08821bf5821`
* admitted lane:
  - `fallback_free_fast_path`
* compatibility lane:
  - `explicit_fallback_path`
* refusal lane:
  - `unsupported_or_refused`
* admitted direct-engine contract:
  - `backend=psionic`
  - `decode_mode=greedy`
  - `structured_output=none`
  - sampling knobs absent from the request
  - CLI gate `--require-fallback-free-cuda`

Operator surface:

* `crates/psionic-serve/examples/qwen35_cuda_bench.rs` now publishes:
  - `run_status`
  - `refusal_reason`
  - `psionic_cuda_fast_path.lane`
  - `psionic_cuda_fast_path.status`
  - `psionic_cuda_fast_path.env_guards[]`
  - `psionic_cuda_startup.warmup_host_fallback_evidence`
  - per-run `qwen35_host_fallback_evidence`
* the admitted fast lane refuses publication when the native receipt shows:
  - host fallback evidence
  - raw-logit materialization
  - output modes other than `argmax_only`
  - graph shape drift
  - missing graph capture readiness
  - missing steady-state graph hits after the initial capture
  - repeated graph recapture beyond the one initial per-request capture that
    the current request-local graph model requires
* `scripts/release/qwen35_direct_vs_http_compare.py` now requires this direct
  benchmark gate by default and only relaxes it when
  `--allow-direct-fallbacks` is set explicitly

On March 31, 2026, issue `#805` aligned this gate with the current request-
local graph model. One initial graph capture plus one matching miss is now
admitted when the same request also proves steady-state graph hits, zero graph
shape drift, zero host fallback evidence, and the expected FA3 backend
execution. The before/after evidence for that change lives in:

* `fixtures/qwen35/benchmarks/qwen35_cuda_issue_805_20260331_archlinux_nongated.json`
* `fixtures/qwen35/benchmarks/qwen35_cuda_issue_805_20260331_archlinux.json`

Claim boundary:

* This packet does **not** claim that the whole native CUDA lane is already
  free of fallback behavior.
* It does make the admitted benchmark lane fail closed when the hot path
  degrades.
* It keeps compatibility runs available for debugging, but those rows are no
  longer the default publication path.

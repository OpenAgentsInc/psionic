# PSION RVLLM Direct Engine Comparator

> Status: landed on 2026-03-31 as the eleventh retained RVLLM runtime-harvest
> packet.

This document records the explicit comparator contract for the current native
`qwen35` CUDA serving lane.

The important truth here is not that Psionic suddenly learned how to compare
itself against HTTP or vLLM on March 31. It already had a direct native bench,
an OpenAI-compatible HTTP surface, and TTFT / ITL headers. The missing piece
was one repo-owned contract that keeps those numbers separated by benchmark
class instead of blending them into a single folklore tok/s claim.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_direct_engine_comparator_v1.json`

Current retained truth:

* packet digest `fc32d554dbfaec0f2b792fb8e67b3e4eeb0a508e1bcabf9ef02e2d771929232c`
* admitted benchmark classes:
  - `direct_engine`
  - `http`
  - `optional_reference_direct_engine`
* current explicit prompt contract:
  - `greedy_one_sentence`
* current admitted concurrency ladder:
  - `1`
  - `2`
  - `4`

Operator surface:

* `crates/psionic-serve/examples/qwen35_cuda_bench.rs` now keeps the native
  direct-engine row machine-readable with:
  - `benchmark_class = direct_engine`
  - `load_s`
  - `psionic_cuda_startup.warmup_*`
  - per-run `ttft_s` and `itl_s`
  - `mean_ttft_s`, `mean_itl_s`, `mean_total_s`, and `mean_decode_tok_s`
* `scripts/release/qwen35_direct_vs_http_compare.py` now owns the combined
  collector for:
  - the native direct-engine row
  - the native `psionic-openai-server` HTTP row
  - the explicit HTTP concurrency ladder
  - an optional direct `vllm` reference row when `--vllm-model` is supplied
  - the fallback-free direct publication gate by default, unless
    `--allow-direct-fallbacks` is set deliberately
* the HTTP row reuses the existing stable response-header contract from
  `openai_http.rs`:
  - `x-psionic-ttft-ns`
  - `x-psionic-itl-ns`
  - `x-psionic-scheduling-class`
  - `x-psionic-prefill-decode-mode`

Claim boundary:

* This packet does **not** widen the admitted model lane beyond the current
  native `qwen35` CUDA serving path.
* It does **not** treat direct-engine and HTTP numbers as one benchmark class.
* It does keep runtime load, warmup, TTFT, ITL, total latency, and throughput
  visible enough to tell whether a remaining gap lives in the runtime or in
  the server surface.
* The default direct row is now bounded by
  `docs/PSION_RVLLM_FALLBACK_FREE_CUDA_GATE.md` instead of publishing any
  compatibility receipt as if it were the admitted hot path.

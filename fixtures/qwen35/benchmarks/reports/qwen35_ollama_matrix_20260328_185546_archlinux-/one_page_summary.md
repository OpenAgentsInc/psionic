# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260328_185546_archlinux-`

Generated at: `2026-03-28T18:55:46Z`

Psionic commit: `aebe7685872aaf8c0cad8d0401d292e9321c7ef3`

Ollama version: `ollama version is 0.17.5`

Host label: `archlinux-`

Host GPU: `NVIDIA GeForce RTX 4080`

Host GPU memory total: `16376 MiB`

Host power limit: `320.00 W`

Host default power limit: `320.00 W`

Repeats per row: `3`

CARGO_INCREMENTAL: `0`

Change rationale: `Michel follow-up rerun with repo-owned per-run evidence capture`

Ollama comparison rationale: `Local Ollama comparison on the same host and prompt contract`

| Contract | Model | Psionic tok/s | Ollama tok/s | Ratio | Psionic output tokens | Ollama output tokens | Psionic termination | Ollama termination | First divergence | Strength |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `greedy` | `qwen3.5:4b` | `174.1205968064323` | `144.58749675017793` | `1.20` | `51,51,51` | `35,35,35` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `18,18,18` | `mismatched` |
| `sampled_topk40` | `qwen3.5:4b` | `175.25101746213434` | `143.82837420226693` | `1.22` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk100` | `qwen3.5:4b` | `171.0014325559372` | `145.81302227210253` | `1.17` | `41,41,41` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `5,5,5` | `mismatched` |

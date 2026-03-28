# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260328_182857_archlinux-`

Generated at: `2026-03-28T18:28:57Z`

Psionic commit: `bbac9e75257c8d2da8a4e01ac924fc90bcc118bd`

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
| `sampled_topk40` | `qwen3.5:0.8b` | `469.0576892626823` | `337.20465975810674` | `1.39` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:2b` | `243.37893840452864` | `207.0513017134083` | `1.18` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:4b` | `174.97645447573197` | `143.7487432001676` | `1.22` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:9b` | `108.36916950256959` | `96.68077710668395` | `1.12` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `1,3,3` | `weak_length_matched_only` |

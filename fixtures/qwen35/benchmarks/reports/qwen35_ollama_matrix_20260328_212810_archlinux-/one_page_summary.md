# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260328_212810_archlinux-`

Generated at: `2026-03-28T21:28:10Z`

Psionic commit: `d0bbea3240c2707ed333cfea3e5e8f3dd91ff3f8`

Ollama version: `ollama version is 0.17.5`

Host label: `archlinux-`

Host GPU: `NVIDIA GeForce RTX 4080`

Host GPU memory total: `16376 MiB`

Host power limit: `320.00 W`

Host default power limit: `320.00 W`

Repeats per row: `3`

CARGO_INCREMENTAL: `0`

Change rationale: `Presorted qwen35 candidate sampling plus row-by-row GPU isolation recheck`

Ollama comparison rationale: `Local Ollama comparison on the same host and prompt contract`

| Contract | Model | Psionic tok/s | Ollama tok/s | Ratio | Psionic output tokens | Ollama output tokens | Psionic termination | Ollama termination | First divergence | Strength |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `sampled_topk40` | `` | `252.83636417443302` | `118.05145058474967` | `2.14` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `` | `178.64811718735277` | `144.45868768445771` | `1.24` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk100` | `` | `250.56321043424634` | `206.8208763111463` | `1.21` | `27,27,27` | `38,38,38` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `2,2,2` | `mismatched` |
| `sampled_topk100` | `` | `178.1578042774727` | `144.1980690162475` | `1.24` | `41,41,41` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `5,5,5` | `mismatched` |

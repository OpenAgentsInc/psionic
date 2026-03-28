# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260328_210428_archlinux-`

Generated at: `2026-03-28T21:04:28Z`

Psionic commit: `d90cd0e55cc7759f5472672e6ab5266d89a7e963`

Ollama version: `ollama version is 0.17.5`

Host label: `archlinux-`

Host GPU: `NVIDIA GeForce RTX 4080`

Host GPU memory total: `16376 MiB`

Host power limit: `320.00 W`

Host default power limit: `320.00 W`

Repeats per row: `3`

CARGO_INCREMENTAL: `0`

Change rationale: `Adaptive qwen35 partitioned top-k block count on idle RTX 4080`

Ollama comparison rationale: `Local Ollama comparison on the same idle RTX 4080 host and prompt contract`

| Contract | Model | Psionic tok/s | Ollama tok/s | Ratio | Psionic output tokens | Ollama output tokens | Psionic termination | Ollama termination | First divergence | Strength |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `sampled_topk40` | `qwen3.5:0.8b` | `505.3277495575889` | `336.80750416509136` | `1.50` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:2b` | `252.94209867629218` | `206.30600337222134` | `1.23` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:4b` | `179.41631624753663` | `143.9918510537279` | `1.25` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:9b` | `110.12497156199328` | `83.80658025298482` | `1.31` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk100` | `qwen3.5:0.8b` | `501.50498768535834` | `327.1903812056087` | `1.53` | `33,33,33` | `20,20,20` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `4,4,4` | `mismatched` |
| `sampled_topk100` | `qwen3.5:2b` | `250.75671963584554` | `205.419243563329` | `1.22` | `27,27,27` | `38,38,38` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `2,2,2` | `mismatched` |
| `sampled_topk100` | `qwen3.5:4b` | `178.3075772194759` | `143.7236853202345` | `1.24` | `41,41,41` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `5,5,5` | `mismatched` |
| `sampled_topk100` | `qwen3.5:9b` | `109.41836720075342` | `97.7768528886583` | `1.12` | `34,34,34` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `6,6,6` | `mismatched` |

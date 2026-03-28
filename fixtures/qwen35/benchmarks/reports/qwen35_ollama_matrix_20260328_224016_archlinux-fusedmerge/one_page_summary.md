# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260328_224016_archlinux-fusedmerge`

Generated at: `2026-03-28T22:40:16Z`

Psionic commit: `351423c355eefef8e085b60e8dcc71da8c6926ca`

Ollama version: `ollama version is 0.17.5`

Host label: `archlinux-fusedmerge`

Host GPU: `NVIDIA GeForce RTX 4080`

Host GPU memory total: `16376 MiB`

Host power limit: `320.00 W`

Host default power limit: `320.00 W`

Repeats per row: `3`

CARGO_INCREMENTAL: `0`

Change rationale: `Partitioned top-k merge now preserves original indices without a third remap kernel.`

Ollama comparison rationale: `Local Ollama comparison on the same host and prompt contract`

| Contract | Model | Psionic tok/s | Ollama tok/s | Ratio | Psionic output tokens | Ollama output tokens | Psionic termination | Ollama termination | First divergence | Strength |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `sampled_topk40` | `qwen3.5:0.8b` | `505.7748742394745` | `338.500007753949` | `1.49` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:2b` | `253.10741293204418` | `206.42998359982008` | `1.23` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:4b` | `179.63047307189194` | `143.82313653601904` | `1.25` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:9b` | `110.15595569099285` | `96.54050370835364` | `1.14` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk100` | `qwen3.5:0.8b` | `502.080477737388` | `324.60783182488177` | `1.55` | `33,33,33` | `20,20,20` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `4,4,4` | `mismatched` |
| `sampled_topk100` | `qwen3.5:2b` | `250.22966843995266` | `205.27652767728412` | `1.22` | `27,27,27` | `38,38,38` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `2,2,2` | `mismatched` |
| `sampled_topk100` | `qwen3.5:4b` | `178.38051364823298` | `144.8008582354623` | `1.23` | `41,41,41` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `5,5,5` | `mismatched` |
| `sampled_topk100` | `qwen3.5:9b` | `109.46765738951727` | `97.84306883132142` | `1.12` | `34,34,34` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `6,6,6` | `mismatched` |

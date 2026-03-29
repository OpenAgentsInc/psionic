# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260329_032406_archlinux-`

Generated at: `2026-03-29T03:24:06Z`

Psionic commit: `7c09066b393f6dd8d4270c05765e3f82a52e214a`

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
| `sampled_topk100` | `qwen3.5:0.8b` | `509.946460187388` | `328.6594722228252` | `1.55` | `33,33,33` | `20,20,20` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `4,4,4` | `mismatched` |
| `sampled_topk100` | `qwen3.5:2b` | `255.25850160488014` | `205.7195889528672` | `1.24` | `27,27,27` | `38,38,38` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `2,2,2` | `mismatched` |
| `sampled_topk100` | `qwen3.5:4b` | `180.80470706902432` | `145.92383081402548` | `1.24` | `41,41,41` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `5,5,5` | `mismatched` |
| `sampled_topk100` | `qwen3.5:9b` | `110.38604589880434` | `97.7625675189439` | `1.13` | `34,34,34` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `6,6,6` | `mismatched` |

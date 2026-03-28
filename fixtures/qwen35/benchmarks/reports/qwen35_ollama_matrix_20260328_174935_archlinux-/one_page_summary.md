# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260328_174935_archlinux-`

Generated at: `2026-03-28T17:49:35Z`

Psionic commit: `613777d03b4202d61370c242e2467f6e875c1e36`

Ollama version: `ollama version is 0.17.5`

Host label: `archlinux-`

Host GPU: `NVIDIA GeForce RTX 4080`

Host GPU memory total: `16376 MiB`

Host power limit: `320.00 W`

Host default power limit: `320.00 W`

Repeats per row: `3`

CARGO_INCREMENTAL: `0`

Change rationale: `Clean RTX 4080 rerun after killing resident GPU jobs; includes termination and token-divergence evidence`

Ollama comparison rationale: `Local Ollama comparison on the same fresh GPU host using explicit greedy contract and repo-owned evidence capture`

| Contract | Model | Psionic tok/s | Ollama tok/s | Ratio | Psionic output tokens | Ollama output tokens | Psionic termination | Ollama termination | First divergence | Strength |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `greedy` | `qwen3.5:0.8b` | `529.7723749422624` | `336.2710799414608` | `1.58` | `26,25,25` | `28,28,28` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `1,3,3` | `mismatched` |
| `greedy` | `qwen3.5:2b` | `259.95252529440995` | `205.1492807900203` | `1.27` | `25,21,24` | `34,34,34` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `2,2,2` | `mismatched` |
| `greedy` | `qwen3.5:4b` | `173.56946901542173` | `146.2284178902544` | `1.19` | `128,29,30` | `35,35,35` | `max_output_tokens,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `1,6,6` | `mismatched` |
| `greedy` | `qwen3.5:9b` | `107.39672266956002` | `97.31954496369121` | `1.10` | `10,28,24` | `42,42,42` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `1,5,6` | `mismatched` |
| `sampled_topk40` | `qwen3.5:0.8b` | `270.07113593914613` | `336.9105693149369` | `0.80` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:2b` | `175.2393204875451` | `206.7784368456628` | `0.85` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:4b` | `136.8524202541421` | `144.01278131987593` | `0.95` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:9b` | `92.61361414622081` | `96.32059908347973` | `0.96` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `1,3,3` | `weak_length_matched_only` |
| `sampled_topk100` | `qwen3.5:0.8b` | `446.4942757627667` | `326.433886490558` | `1.37` | `12,32,26` | `20,20,20` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `1,3,4` | `mismatched` |
| `sampled_topk100` | `qwen3.5:2b` | `235.2776320933157` | `204.70450723344666` | `1.15` | `26,25,33` | `38,38,38` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `1,4,4` | `mismatched` |
| `sampled_topk100` | `qwen3.5:4b` | `170.96449766066453` | `144.39446208606083` | `1.18` | `128,25,34` | `37,37,37` | `max_output_tokens,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `1,6,5` | `mismatched` |
| `sampled_topk100` | `qwen3.5:9b` | `106.37090806342862` | `97.4443736095784` | `1.09` | `24,40,30` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `1,5,5` | `mismatched` |

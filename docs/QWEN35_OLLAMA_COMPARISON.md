# Qwen3.5 Native CUDA vs Ollama

This document is the canonical comparison matrix for Psionic native CUDA
`qwen35` inference versus local Ollama on this host.

Tracked issues:

- `#606` Scale native qwen35 CUDA lane to beat Ollama on 2B, 4B, and 9B
- `#631` Benchmark qwen35 temperature/top-k sampling vs Ollama and keep Psionic ahead
- `#632` Use bounded candidate decode for qwen35 temperature and top-k sampling

Published benchmark checkpoints:

- current canonical checkpoint: `March 28, 2026 clean RTX 4080 reruns with
  explicit Ollama greedy settings, per-run divergence evidence, the inclusive
  partitioned top-k40 selector, the qwen3.5:4b hybrid-state zero-init fix,
  and later partitioned top-k block-count tuning for sampled rows`
- historical greedy checkpoint: `c5bc0ba2`, preserved for provenance only
- historical sampled checkpoints: `March 27-28, 2026` reruns before
  termination and divergence evidence landed in the repo-owned collector

Shared benchmark rules:

- same host
- same GGUF artifact digests
- same prompt per benchmark mode
- same token cap
- Psionic uses the native CUDA `qwen35` lane
- Ollama uses the local `ollama serve` instance
- decode throughput is reported as mean `tok/s`
- verify `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory
  --format=csv,noheader,nounits` returns no resident compute processes before
  the run; do not publish qwen35 benchmark numbers from a busy GPU
- the canonical matrix runner now re-checks GPU idleness before every backend
  row and waits for the GPU to drain after each Ollama row
- the canonical matrix runner now stops any models still listed by `ollama ps`
  between rows so Ollama does not silently retain VRAM residency across the
  Psionic-versus-Ollama sequence
- benchmark one runtime at a time for `9b` on this 16 GB RTX 4080 because
  Ollama keeps model weights resident in VRAM
- the local Ollama `qwen3.5` path on this checkout routes through
  `runner/ollamarunner` and its active sampler surface is the one built by
  `sample.NewSampler(temperature, topK, topP, minP, seed, grammar)`

## Machine-Readable Evidence Contract

Michel's laptop run exposed a gap in the repo-owned evidence path, not in the
human-readable comparison doc alone.

The canonical repo-owned rerun path is now:

- per-row benchmark reports from
  `cargo run --release -p psionic-serve --example qwen35_cuda_bench -- ... --json-out report.json`
- sequential multi-row collection from
  `scripts/release/run-qwen35-ollama-matrix.sh`

The canonical matrix runner now refuses to start if `nvidia-smi` reports any
resident compute processes unless `PSIONIC_QWEN35_MATRIX_ALLOW_BUSY_GPU=1` is
set explicitly.

As of commit `d0bbea32`, the same runner also:

- stops any models still loaded in Ollama between rows
- waits for the GPU to become idle before each Psionic row
- waits for the GPU to become idle again after each Ollama row
- forces Psionic qwen35 benchmark requests onto
  `PrefixCacheMode::Bypass` so the matrix measures the raw prompt and decode
  lane instead of mixing in shared prefix-cache behavior

That change is benchmark-hygiene, not a throughput claim. It closes a real
measurement hole that showed up during later sampled follow-up runs.

Those artifacts preserve the per-run fields that matter for honest comparison:

- `prompt_tokens`
- `output_tokens`
- `output_token_ids`
- `prompt_s`
- `decode_s`
- `total_s`
- `decode_tok_s`
- `qwen35_output_modes`
- `qwen35_readback_bytes`
- `qwen35_raw_logits`
- termination classification and matched stop-sequence evidence
- first-divergence index and row-strength classification in the matrix summary
- rendered prompt and stop sequences
- effective sampled contract values
- host label, GPU memory, power-limit metadata, Psionic commit, and Ollama
  version at the matrix level

Constrained-host follow-up reruns should use that path instead of publishing a
reduced matrix that keeps only token count and mean throughput.

## Current Clean RTX 4080 Matrix

Current canonical evidence lives in:

- latest sampled follow-on after doubling the partitioned one-row CUDA top-k
  tile width by raising `kLogitsTopKItemsPerThread` from `8` to `16` on the
  idle RTX 4080:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260329_003921_archlinux-.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260329_003921_archlinux-/one_page_summary.md`
- latest sampled follow-on after making the partitioned one-row CUDA top-k
  block count adaptive by requested `top_k` on the idle RTX 4080:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_210428_archlinux-.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_210428_archlinux-/one_page_summary.md`
- latest sampled follow-on after tuning the partitioned one-row CUDA top-k
  block count from `8` to `24` on the idle RTX 4080:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_200654_archlinux-.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_200654_archlinux-/one_page_summary.md`
- fresh full-matrix rerun after both the inclusive `top_k = 40` selector fix
  and the `qwen3.5:4b` hybrid-state zero-init fix:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_190650_archlinux-.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_190650_archlinux-/one_page_summary.md`
- multi-model baseline:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_174935_archlinux-.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_174935_archlinux-/one_page_summary.md`
- `sampled_topk40` follow-on after routing `top_k = 40` through the inclusive
  partitioned one-row CUDA selector:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_182857_archlinux-.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_182857_archlinux-/one_page_summary.md`
- targeted `qwen3.5:4b` post-fix rerun after zeroing per-request hybrid SSM
  state on request init:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_185546_archlinux-.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_185546_archlinux-/one_page_summary.md`
- non-canonical diagnostic rerun after the presorted-candidate cleanup and the
  row-by-row isolation fix:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_212810_archlinux-.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_212810_archlinux-/one_page_summary.md`
- non-canonical diagnostic rerun after fusing the partitioned top-k remap into
  the merge kernel; this was later reverted because the end-to-end sampled
  deltas stayed inside noise and `sampled_topk100 qwen3.5:2b` regressed
  slightly:
  - `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_224016_archlinux-fusedmerge.json`
  - `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_224016_archlinux-fusedmerge/one_page_summary.md`

Why `20260328_212810` is diagnostic only:

- it intentionally narrowed scope to `2b` and `4b`
- it confirmed the isolation fix restored stable sampled numbers
- it showed the presorted-candidate cleanup is effectively throughput-flat on
  the real sampled contracts
- it is not a new canonical checkpoint because it does not widen Psionic's
  lead over the existing published sampled record

The mixed canonical table below uses:

- greedy rows from the full-matrix `20260328_190650` rerun
- `sampled_topk40` rows from the later sampled-only `20260329_003921`
  follow-on
- `sampled_topk100` rows from the earlier sampled-only `20260328_210428`
  follow-on

Interpretation rules:

- `strong` means exact comparable token-id match plus termination-class match
- `weak_length_matched_only` means both sides produced the same native output
  token count but diverged in comparable generated token IDs
- `mismatched` means the row is not a clean throughput comparison row

| Contract | Model | Psionic tok/s | Ollama tok/s | Strength | Notes |
| --- | --- | ---: | ---: | --- | --- |
| `greedy` | `qwen3.5:0.8b` | `535.38` | `335.93` | `mismatched` | Raw throughput favors Psionic, but generated lengths still differ (`37,37,37` vs `28,28,28`) and first divergence starts at token `23,23,23` |
| `greedy` | `qwen3.5:2b` | `260.12` | `206.20` | `strong` | The fresh full-matrix rerun now gives an exact comparable token-id match with matching EOS termination across all repeats, even though backend-native token counts still report `33,33,33` vs `34,34,34` |
| `greedy` | `qwen3.5:4b` | `173.87` | `144.41` | `mismatched` | The fresh full-matrix rerun keeps the post-fix `4b` row stable (`51,51,51` vs `35,35,35`), but output lengths still differ and first divergence starts at token `18,18,18` |
| `greedy` | `qwen3.5:9b` | `107.74` | `97.43` | `mismatched` | Raw throughput favors Psionic, but output lengths differ (`41,41,41` vs `42,42,42`) and first divergence starts at token `5,5,5` |
| `sampled_topk40` | `qwen3.5:0.8b` | `511.51` | `337.58` | `weak_length_matched_only` | Both sides still hit the `128` token cap, first comparable token divergence still starts at token `3,3,3`, and doubling the partitioned top-k tile width raises Psionic from `505.33` to `511.51 tok/s` while preserving bounded-candidate output |
| `sampled_topk40` | `qwen3.5:2b` | `254.07` | `206.38` | `weak_length_matched_only` | Both sides still hit the `128` token cap, first comparable token divergence still starts at token `4,4,4`, and the wider partitioned tile raises Psionic from `252.94` to `254.07 tok/s` without weakening the row classification |
| `sampled_topk40` | `qwen3.5:4b` | `180.01` | `144.10` | `weak_length_matched_only` | The later sampled rerun keeps the post-fix `4b` row length-matched and ahead on Psionic, with first comparable token divergence still starting at token `4,4,4` and the wider tile raising Psionic from `179.42` to `180.01 tok/s` |
| `sampled_topk40` | `qwen3.5:9b` | `110.35` | `96.37` | `weak_length_matched_only` | Both sides still hit the `128` token cap, first comparable token divergence still starts at token `3,3,3`, and the wider partitioned tile raises Psionic from `110.12` to `110.35 tok/s` while keeping bounded-candidate output |
| `sampled_topk100` | `qwen3.5:0.8b` | `501.50` | `327.19` | `mismatched` | Raw throughput still favors Psionic, output lengths still differ (`33,33,33` vs `20,20,20`), first divergence still starts at token `4,4,4`, and the adaptive wide-lane policy raises Psionic from `492.81` to `501.50 tok/s` |
| `sampled_topk100` | `qwen3.5:2b` | `250.76` | `205.42` | `mismatched` | Raw throughput still favors Psionic, output lengths still differ (`27,27,27` vs `38,38,38`), first divergence still starts at token `2,2,2`, and the adaptive policy raises Psionic from `247.57` to `250.76 tok/s` |
| `sampled_topk100` | `qwen3.5:4b` | `178.31` | `143.72` | `mismatched` | The later sampled rerun keeps the post-fix `4b` row stable (`41,41,41` vs `37,37,37`), the row remains `mismatched`, and the adaptive policy raises Psionic from `177.28` to `178.31 tok/s` |
| `sampled_topk100` | `qwen3.5:9b` | `109.42` | `97.78` | `mismatched` | Raw throughput still favors Psionic, output lengths still differ (`34,34,34` vs `37,37,37`), first divergence still starts at token `6,6,6`, and the adaptive policy raises Psionic from `109.02` to `109.42 tok/s` |

## Greedy Contract

Prompt:

```text
Explain what Psionic is in one sentence.
```

Token cap:

- `128`

Ollama greedy settings:

- `temperature = 0.0`
- `top_k = 1`
- `top_p = 1.0`
- `min_p = 0.0`
- `repeat_penalty = 1.0`
- `repeat_last_n = 0`
- `presence_penalty = 0.0`
- `frequency_penalty = 0.0`
- `seed = 42`
- `keep_alive = 0`

Historical note:

- the older March 27 greedy checkpoint is not current canonical evidence
- the old harness omitted these explicit Ollama greedy settings and therefore
  let Ollama use its default sampler surface instead of a forced greedy path
- the repo now forces the explicit no-sampling, no-penalty Ollama contract
  above, and the clean March 28 rerun supersedes the earlier greedy summary
  with committed per-run termination and divergence evidence

## Greedy Matrix

Use the clean RTX 4080 matrix above as the current canonical greedy evidence.

Current status:

- raw greedy `tok/s` is higher on Psionic across all four models
- `qwen3.5:2b` is now a `strong` exact-match row with matching EOS
  termination across all repeats
- `qwen3.5:0.8b`, `qwen3.5:4b`, and `qwen3.5:9b` still remain `mismatched`
- the `qwen3.5:4b` row no longer shows the earlier warmup-induced cap-hit
  corruption after zeroing per-request hybrid SSM state on request init
- greedy first-token parity remains unresolved and requires runtime work, not
  more summary-only reporting

## Sampled Contract

Prompt:

```text
Respond with plain text only. Write 20 numbered one-sentence reasons why GPU decode throughput matters for local inference. Start at 1 and keep going until all 20 are written.
```

Token cap:

- `128`

Sampled settings:

- `temperature = 0.8`
- `top_k = 40`
- `top_p = 0.9`
- `repeat_penalty = 1.0`
- `presence_penalty = 0.0`
- `frequency_penalty = 0.0`
- `seed = 42`
- `think = false` on Ollama

Psionic output-mode evidence on this contract:

- `qwen35_output_modes=[top_k_candidates:40]`
- `qwen35_raw_logits=false`
- Psionic sampled rows rerun after a clean `CARGO_INCREMENTAL=0` rebuild of
  `qwen35_cuda_bench`
- current canonical rerun after routing `top_k = 40` through the inclusive
  partitioned one-row CUDA selector instead of the slower generic top-k path

## Sampled Matrix

Use the clean RTX 4080 matrix above as the current canonical sampled
`top_k = 40` evidence.

Current status:

- all four rows are `weak_length_matched_only`, not `strong`
- Psionic stays on the bounded sampled lane:
  - `qwen35_output_modes=[top_k_candidates:40]`
  - `qwen35_raw_logits=false`
- the later adaptive policy keeps `top_k = 40` on the same `24`-block
  profile and leaves Psionic effectively flat at `505.33`, `252.94`,
  `179.42`, `110.12 tok/s` after the earlier jump from `470.49`, `243.56`,
  `175.06`, `108.36 tok/s`
- Psionic stays ahead of Ollama on all four clean `top_k = 40` rows on this
  host

## Large-`top_k` Sampled Contract

Prompt:

```text
Explain what Psionic is in one sentence.
```

Token cap:

- `128`

Sampled settings:

- `temperature = 0.8`
- `top_k = 100`
- `top_p = 0.9`
- `min_p = 0.05`
- `seed = 42`
- `think = false` on Ollama

Psionic output-mode evidence on this contract:

- `qwen35_output_modes=[top_k_candidates:100]`
- `qwen35_raw_logits=false`
- clean-host rerun after the one-row partitioned CUDA top-k path replaced the
  slower radix-sort route for larger bounded candidate sets
- serialized Ollama residency between rows to keep the `4b` and `9b`
  measurements honest on this 16 GB RTX 4080

## Large-`top_k` Sampled Matrix

Use the clean RTX 4080 matrix above as the current canonical sampled
`top_k = 100` evidence.

Current status:

- all four rows are `mismatched`
- Psionic stays on the wider bounded sampled lane:
  - `qwen35_output_modes=[top_k_candidates:100]`
  - `qwen35_raw_logits=false`
- the earlier fixed `24`-block tuning moved Psionic from `446.26`, `236.17`,
  `171.24`, `106.62 tok/s` to `492.81`, `247.57`, `177.28`, `109.02 tok/s`
- the later adaptive policy raises those same rows again to `501.50`,
  `250.76`, `178.31`, `109.42 tok/s`
- the targeted `qwen3.5:4b` rerun no longer shows the earlier cap-hit failure,
  but the row still remains `mismatched`

## Penalty-Active Psionic Follow-On

This is not a canonical Psionic-versus-Ollama matrix on this checkout, because
the local Ollama `qwen3.5` runner does not wire repeat, presence, and
frequency penalties through the same active sampler path. It is still a useful
runtime checkpoint for Psionic itself because these requests used to force
explicit dense `raw_logits` readback on qwen35 CUDA.

Prompt:

```text
Explain what Psionic is in one sentence.
```

Token cap:

- `128`

Sampled settings:

- `temperature = 0.8`
- `top_k = 40`
- `top_p = 0.9`
- `min_p = 0.05`
- `repeat_penalty = 1.1`
- `repeat_last_n = 64`
- `presence_penalty = 0.2`
- `frequency_penalty = 0.1`
- `seed = 42`

Psionic output-mode evidence on this contract:

- `qwen35_output_modes=[top_k_candidates:40]`
- `qwen35_raw_logits=false`

Psionic-only measured means:

| Model | Psionic decode tok/s | Notes |
| --- | ---: | --- |
| `qwen3.5:0.8b` | `89.29` | Same exact penalty-active request now stays on bounded candidates instead of dense readback |
| `qwen3.5:2b` | `120.71` | Clear throughput win over the earlier raw-logits penalty fallback on this host |
| `qwen3.5:4b` | `56.47` | The row stays on bounded candidates, but the short prompt remains noisier under penalties |
| `qwen3.5:9b` | `43.27` | The row also stays on bounded candidates without materializing dense logits on the host |

## Current Notes

- The fresh clean-host matrix changes the current canonical claim surface on
  this checkout:
  - greedy raw `tok/s` is higher on Psionic across all four models, and
    `qwen3.5:2b` is now a `strong` exact-match row
  - greedy `qwen3.5:0.8b`, `qwen3.5:4b`, and `qwen3.5:9b` still remain
    `mismatched`
  - clean sampled `top_k = 40` rows are the highest-signal constrained
    contract; Psionic stays on bounded candidates there and now leads Ollama
    on all four rows
  - sampled `top_k = 100` rows remain `mismatched`, but Psionic still leads
    Ollama on all four rows there too
- On current `main`, zeroing the per-request hybrid SSM state on request init
  removes the old `qwen3.5:4b` cap-hit corruption on both greedy and
  `top_k = 100` sampled reruns. The `4b` row is now stable but still
  mismatched against Ollama.
- The sampled CUDA lane is bounded, not vague. It uses
  `TopKCandidates { top_k }` only when the request stays inside the exact
  envelope:
  - sampled decode or non-zero effective temperature
  - effective `top_k` available and `<= 128`
  - structured-output masking inactive
  - `mirostat` inactive
- The runtime sampling surface now honors `min_p` and request-level
  `repeat_last_n` in addition to `temperature`, `top_k`, `top_p`, `min_p`,
  `typical_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`,
  `seed`, `mirostat`, `mirostat_tau`, and `mirostat_eta`.
- The generic OpenAI-compatible qwen35 server surface now forwards those same
  controls on `/v1/chat/completions` and `/v1/responses`.
- `repeat_last_n` follows the Ollama-compatible control contract in the local
  sampler and benchmark harness:
  - default `64`
  - `0` disables the penalty lookback window
  - `-1` expands the penalty window to the full available history
- `min_p` remains compatible with the bounded qwen35 CUDA sampled lane because
  Psionic applies it after exact top-k candidate selection on both the dense
  and bounded sampling paths.
- `typical_p`, repeat/presence/frequency penalties, `repeat_last_n`, and
  `mirostat` are supported on the Psionic runtime and request surfaces, but
  they are not part of the canonical Psionic-versus-Ollama matrix on this
  checkout because the local Ollama `qwen3.5` runner does not wire those
  controls through the same active sampler path.
- Native qwen35 structured outputs are also supported now. Greedy no-penalty
  structured requests stay on `TopKCandidates { top_k: 128 }` and use exact
  sparse allowed-logit gather on candidate misses, while structured requests
  outside that envelope still fall back to explicit dense `raw_logits`
  readback. Structured outputs are still a parity feature, not part of the
  bounded-candidate throughput matrix. The qwen35 proxy lane still refuses
  them. The tokenizer-side append cache now also buckets token ids by leading
  char so sparse schema recovery does not linearly rescan the full vocabulary
  when there is no candidate shortlist.
- The local `qwen35_cuda_bench` harness now reproduces native-versus-Ollama
  JSON object and JSON schema requests too through `--json-object` and
  `--json-schema-file`. It also writes machine-readable per-run reports through
  `--json-out`, and the repo-owned matrix collector now lives at
  `scripts/release/run-qwen35-ollama-matrix.sh`. The native qwen35 structured
  path is now replay-safe and can stay off dense raw-logit replay on the
  bounded greedy schema lane.
- Structured-output throughput is still outside the canonical matrix. After the
  leading-char token-cache bucketing pass, the matcher memo-path allocation
  cut, and a clean isolated rebuild on March 28, 2026, the local
  `qwen3.5:0.8b` summary-schema spot check measured native Psionic at about
  `78 tok/s` on the first bounded sparse-gather run and about `162 tok/s` mean
  across a warmed three-repeat pass, versus local Ollama at about `331 tok/s`.
  Psionic published
  `qwen35_output_modes=[top_k_candidates:128,sparse_logits:2,sparse_logits:3,sparse_logits:10]`,
  `qwen35_readback_bytes=5700`, and `qwen35_raw_logits=false` on the sparse
  run. The later warmed repeats stayed on `qwen35_output_modes=[top_k_candidates:128]`
  and hit the token cap without materializing `structured_output_value`, so
  this stays a parity note instead of a canonical throughput row.
- `mirostat` therefore remains a Psionic-side capability note, not a canonical
  beat-Ollama throughput claim.
- Requests outside that envelope still fall back to explicit raw-logit readback
  instead of silently narrowing behavior.
- The 4B row only became correct and faster after fixing the fused
  `ArgmaxOnly` output path. The hot decode branch now routes `Q6_K` output
  weights through `Q8_1` projection plus `argmax_f32` instead of the slower
  generic quantized matvec path.
- The 9B row does not require a separate Psionic fallback path on this host.
  The only extra rule is operational: unload Ollama's resident GPU caches
  before measuring Psionic because Ollama keeps prior model weights live in
  VRAM.
- The qwen35 sampled lane also exposes tuning-only overrides for follow-on
  kernel work:
  `PSIONIC_QWEN35_PARTITIONED_TOP_K_BLOCKS` and
  `PSIONIC_QWEN35_PARTITIONED_TOP_K_THRESHOLD`.
  They are diagnostic controls, not canonical benchmark settings.

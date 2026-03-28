# Qwen35 Partitioned Top-K Block Tuning Audit

Date: 2026-03-28

## Scope

This audit records the qwen35 sampled-decode tuning pass that followed the
clean RTX 4080 March 28 matrix rerun.

The goal was narrow:

- increase Psionic native CUDA sampled `tok/s`
- preserve the bounded candidate lane
- keep the benchmark host fully idle before every run
- land only changes that improved the repo-owned Psionic-versus-Ollama record

## Starting Point

Before this tuning pass, the current clean-host sampled record was:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_190650_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_190650_archlinux-/one_page_summary.md`

On that idle RTX 4080 rerun, Psionic was already ahead on all sampled rows,
but the remaining headroom was still obvious:

- `sampled_topk40`
  - `0.8b` `470.49` vs `336.97 tok/s`
  - `2b` `243.56` vs `207.17 tok/s`
  - `4b` `175.06` vs `144.17 tok/s`
  - `9b` `108.36` vs `96.48 tok/s`
- `sampled_topk100`
  - `0.8b` `446.26` vs `328.41 tok/s`
  - `2b` `236.17` vs `203.56 tok/s`
  - `4b` `171.24` vs `145.15 tok/s`
  - `9b` `106.62` vs `98.00 tok/s`

## Work Performed

### Rejected path: `Q6_K` fused q8_1 argmax hook-up

The CUDA backend already contained a `Q6_K` `q8_1` argmax kernel symbol, so
the first candidate change was wiring that path through Rust and allowing the
greedy qwen35 output head to use it.

That change was not landed.

On the real qwen35 benchmark rows it regressed throughput badly, especially on
`4b` and `9b`. The most severe observed greedy regressions on the idle RTX
4080 were:

- `qwen3.5:4b` from about `173.87` down to about `103.48 tok/s`
- `qwen3.5:9b` from about `107.74` down to about `63.43 tok/s`

The kernel symbol exists, but the actual end-to-end path was not profitable in
the live qwen35 runtime. That experiment was reverted locally and never
committed.

### Landed path: tune partitioned one-row top-k block count

The next candidate was the partitioned CUDA one-row top-k selector used on the
bounded sampled lane.

The runtime previously hard-coded:

- `QWEN35_CUDA_PARTITIONED_TOP_K_BLOCKS = 8`

This tuning pass:

- added a runtime helper so the block count can be overridden through
  `PSIONIC_QWEN35_PARTITIONED_TOP_K_BLOCKS`
- changed the default block count from `8` to `24`
- sized the partitioned top-k scratch buffers from the effective block count
  instead of the old fixed constant

The landed code change is:

- commit `e34c0b05` `Tune qwen35 partitioned top-k block count`

## Idle-GPU Sweep

The sweep ran on the idle RTX 4080 host with:

- Psionic only
- sampled `top_k = 40`
- `repeats = 3`
- models `0.8b`, `2b`, `4b`, `9b`

Sweep results:

| Blocks | `0.8b` | `2b` | `4b` | `9b` |
| --- | ---: | ---: | ---: | ---: |
| `4` | `425.78` | `231.10` | `168.60` | `105.98` |
| `6` | `454.42` | `239.12` | `172.87` | `107.49` |
| `8` | `470.51` | `243.49` | `174.75` | `108.30` |
| `10` | `480.40` | `246.19` | `175.97` | `108.80` |
| `12` | `487.41` | `247.91` | `176.78` | `109.16` |
| `16` | `496.50` | `250.27` | `177.83` | `109.57` |
| `20` | `500.80` | `251.81` | `178.79` | `109.89` |
| `24` | `506.06` | `252.60` | `178.99` | `109.80` |
| `28` | `495.23` | `249.94` | `179.45` | `110.15` |
| `32` | `497.33` | `250.09` | `176.97` | `110.23` |

Why `24` became the new default:

- it was the best overall setting across `0.8b`, `2b`, and `4b`
- it still improved `9b` over the old `8`-block baseline
- it did not require a model-specific branch in the live runtime

## Repo-Owned Follow-On Rerun

After landing `e34c0b05`, the sampled matrix was rerun again on the same idle
RTX 4080 host:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_200654_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_200654_archlinux-/one_page_summary.md`

That rerun preserved the same row-strength classifications as the earlier
clean-host matrix while widening Psionic's sampled lead.

### `sampled_topk40` deltas vs `20260328_190650`

| Model | Old Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `470.49` | `506.49` | `+7.65%` | `340.23` |
| `qwen3.5:2b` | `243.56` | `252.83` | `+3.80%` | `207.33` |
| `qwen3.5:4b` | `175.06` | `179.55` | `+2.57%` | `144.52` |
| `qwen3.5:9b` | `108.36` | `110.13` | `+1.63%` | `96.58` |

The clean sampled rows remain:

- `weak_length_matched_only`
- length-matched at `128/128`
- bounded-candidate on Psionic:
  - `qwen35_output_modes=[top_k_candidates:40]`
  - `qwen35_raw_logits=false`

### `sampled_topk100` deltas vs `20260328_190650`

| Model | Old Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `446.26` | `492.81` | `+10.43%` | `333.92` |
| `qwen3.5:2b` | `236.17` | `247.57` | `+4.83%` | `206.21` |
| `qwen3.5:4b` | `171.24` | `177.28` | `+3.53%` | `144.79` |
| `qwen3.5:9b` | `106.62` | `109.02` | `+2.25%` | `97.57` |

Those rows still remain:

- `mismatched`
- bounded-candidate on Psionic:
  - `qwen35_output_modes=[top_k_candidates:100]`
  - `qwen35_raw_logits=false`

## What This Tells Us

The bounded sampled lane still had kernel-launch shape headroom even after the
inclusive partitioned selector landed.

The improvement is real because:

- the host was idle
- the runner enforced the idle-GPU rule
- the rerun stayed on the bounded candidate lane instead of regressing to
  `raw_logits`
- the row-strength classifications did not get weaker

This is a real throughput gain, not just a measurement artifact.

## Next Steps

- tune the partitioned block count against the wider `top_k = 100` contract
  explicitly instead of assuming the `top_k = 40` optimum is globally optimal
- consider making the partitioned block count adaptive to `top_k` or model
  family once the wider sweep data exists
- keep benchmarking from the same remote staging worktree so future qwen35
  tuning passes do not pay path-sensitive rebuild cost again

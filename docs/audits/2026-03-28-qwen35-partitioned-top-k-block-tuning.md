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

The first landed code change was:

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

## Adaptive Follow-On For Wider `top_k`

The fixed `24`-block default was clearly right for the canonical
`sampled_topk40` contract, but a follow-on sweep showed the wider
`sampled_topk100` contract still wanted a larger block count.

Idle RTX 4080 `sampled_topk100` sweep:

| Blocks | `0.8b` | `2b` | `4b` | `9b` |
| --- | ---: | ---: | ---: | ---: |
| `16` | `473.13` | `245.94` | `176.22` | `108.60` |
| `24` | `493.70` | `248.26` | `177.45` | `109.11` |
| `32` | `498.93` | `250.04` | `178.03` | `109.32` |
| `40` | `501.72` | `250.84` | `178.46` | `109.49` |
| `48` | `501.85` | `250.07` | `178.29` | `109.44` |
| `64` | `501.85` | `250.77` | `178.33` | `109.42` |

That suggested a simple adaptive rule:

- keep the clean `top_k = 40` contract on the `24`-block profile
- widen the block count for larger bounded candidate sets

The second landed code change was:

- commit `d90cd0e5` `Adapt qwen35 partitioned top-k block counts`

That commit keeps the small sampled lane at `24` blocks and switches the wider
bounded lane to a larger block count for `top_k >= 96`.

## Commit-Pinned Adaptive Rerun

After landing `d90cd0e5`, the sampled matrix was rerun again on the same idle
RTX 4080 host:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_210428_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_210428_archlinux-/one_page_summary.md`

### `sampled_topk40` deltas vs `20260328_200654`

| Model | Prior Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `506.49` | `505.33` | `-0.23%` | `336.81` |
| `qwen3.5:2b` | `252.83` | `252.94` | `+0.04%` | `206.31` |
| `qwen3.5:4b` | `179.55` | `179.42` | `-0.08%` | `143.99` |
| `qwen3.5:9b` | `110.13` | `110.12` | `-0.00%` | `83.81` |

Interpretation:

- this is effectively flat
- the adaptive policy did not give back the `top_k = 40` win
- the clean sampled lane stayed on the same bounded-candidate profile

### `sampled_topk100` deltas vs `20260328_200654`

| Model | Prior Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `492.81` | `501.50` | `+1.76%` | `327.19` |
| `qwen3.5:2b` | `247.57` | `250.76` | `+1.29%` | `205.42` |
| `qwen3.5:4b` | `177.28` | `178.31` | `+0.58%` | `143.72` |
| `qwen3.5:9b` | `109.02` | `109.42` | `+0.36%` | `97.78` |

Interpretation:

- this is a smaller gain than the earlier `8 -> 24` jump, but it is still
  real and repeatable
- the gain is concentrated in the wider bounded lane, which is exactly what
  the sweep predicted
- the row-strength classifications stay unchanged

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
  beyond the current `24` versus `40` split and decide whether a richer
  adaptive rule is worth the added complexity
- keep benchmarking from the same remote staging worktree so future qwen35
  tuning passes do not pay path-sensitive rebuild cost again

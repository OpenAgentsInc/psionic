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

## Later Follow-On: Presorted Candidate Cleanup And Isolation Recheck

After the adaptive block-count tuning landed, the next sampled follow-on
looked at host-side work in the bounded candidate lane.

Two later commits matter here:

- `9990d5cf` `Skip redundant sort for qwen35 presorted candidates`
- `d0bbea32` `Tighten qwen35 benchmark GPU isolation`

### Presorted candidate cleanup

The bounded qwen35 sampled lane already receives CUDA top-k candidates sorted
by descending logit.

The cleanup in `9990d5cf`:

- adds a presorted-candidate sampling entrypoint
- skips the redundant host-side `top_p` sort for that path
- preserves the same bounded-candidate output mode

That change is valid, but on the real qwen35 sampled rows it did not produce a
meaningful throughput gain by itself.

### Why the first follow-on matrix was not trustworthy

The first full follow-on after `9990d5cf` produced:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_212306_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_212306_archlinux-/one_page_summary.md`

That artifact is diagnostic only and should not be treated as canonical
evidence.

What it exposed:

- startup-only idle-GPU checks were not enough
- later rows could still run after unrelated CUDA residency reappeared on the
  host
- some sampled rows looked contradictory even though token counts and sampled
  contracts were unchanged

The problem was measurement hygiene, not a newly validated runtime speedup or
regression.

### Isolation fix

Commit `d0bbea32` tightened the canonical matrix runner so it now:

- re-checks GPU idleness before every backend row
- stops any models still listed by `ollama ps` between rows
- waits for the GPU to become idle again after each Ollama row

This change is benchmark hygiene. It does not change qwen35 runtime speed.

### Clean diagnostic rerun

After landing `d0bbea32`, the narrowed rerun on the idle RTX 4080 host was:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_212810_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_212810_archlinux-/one_page_summary.md`

Scope:

- models `qwen3.5:2b` and `qwen3.5:4b`
- contracts `sampled_topk40` and `sampled_topk100`
- `repeats = 3`

What the clean rerun showed:

- `sampled_topk40`
  - `2b` `252.84 tok/s`
  - `4b` `178.65 tok/s`
- `sampled_topk100`
  - `2b` `250.56 tok/s`
  - `4b` `178.16 tok/s`

Interpretation:

- these numbers return to the previously published stable range
- the presorted-candidate cleanup is effectively throughput-flat on the real
  sampled contracts
- `9990d5cf` is reasonable to keep as bounded-candidate cleanup, but not as a
  new performance checkpoint
- `d0bbea32` was the materially important change because it makes later
  benchmark claims harder to contaminate

## Next Steps

## Later Follow-On: Host-Buffer Cleanup Stayed Flat

The next sampled follow-on tried to remove host-side heap materialization in the
bounded candidate path.

That code landed briefly as:

- commit `44b88722` `Avoid heap top-k materialization in qwen35`

The clean narrowed rerun on the idle RTX 4080 host was:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_214441_archlinux-.json`

Scope:

- models `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`
- contracts `sampled_topk40` and `sampled_topk100`
- `repeats = 3`

Relative to the prior clean sampled baseline from `20260328_210428`, the
observed deltas were effectively noise:

### `sampled_topk40` deltas vs `20260328_210428`

| Model | Prior Psionic | New Psionic | Delta |
| --- | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `505.33` | `506.21` | `+0.17%` |
| `qwen3.5:2b` | `252.94` | `253.10` | `+0.06%` |
| `qwen3.5:4b` | `179.42` | `179.59` | `+0.10%` |

### `sampled_topk100` deltas vs `20260328_210428`

| Model | Prior Psionic | New Psionic | Delta |
| --- | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `501.50` | `501.99` | `+0.10%` |
| `qwen3.5:2b` | `250.76` | `250.83` | `+0.03%` |
| `qwen3.5:4b` | `178.31` | `178.49` | `+0.10%` |

This did not justify a new canonical checkpoint.

One useful quality finding still came out of that pass:

- the matrix runner's Ollama cleanup loop was leaking a global `model`
  variable
- that bug blanked the `model` field in some narrowed JSON artifacts
- the runner now scopes that cleanup variable correctly so later matrices keep
  stable model labels again

## Later Follow-On: Presorted Sampler Fast Path Also Stayed Flat

The next sampled follow-on tried to bypass more runtime sampler work for the
presorted bounded-candidate path.

That experiment was benchmarked as:

- commit `da70c423` `Fast-path qwen35 presorted candidate sampling`

The clean narrowed rerun on the idle RTX 4080 host was:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_215855_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_215855_archlinux-/one_page_summary.md`

Relative to the same `20260328_210428` sampled baseline:

### `sampled_topk40` deltas vs `20260328_210428`

| Model | Prior Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `505.33` | `506.50` | `+0.23%` | `336.85` |
| `qwen3.5:2b` | `252.94` | `253.06` | `+0.05%` | `206.25` |
| `qwen3.5:4b` | `179.42` | `179.59` | `+0.10%` | `144.12` |

### `sampled_topk100` deltas vs `20260328_210428`

| Model | Prior Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `501.50` | `501.87` | `+0.07%` | `321.42` |
| `qwen3.5:2b` | `250.76` | `250.04` | `-0.28%` | `205.08` |
| `qwen3.5:4b` | `178.31` | `178.42` | `+0.06%` | `143.98` |

Interpretation:

- this path is effectively flat
- the bounded sampled lead over Ollama remains strong
- the extra runtime complexity did not buy a defensible throughput gain

That experiment should not be treated as the new optimization direction.

## Later Follow-On: Threshold Sweep Confirmed `top_k = 40` Must Stay Partitioned

After the flat host-side passes, the next question was whether the current
partitioned top-k crossover was wrong and `top_k = 40` should fall back to the
simple one-row top-k kernel instead.

Current main now exposes a tuning-only override for that crossover:

- commit `b926f2f7` `Make qwen35 top-k threshold tunable`
- env var `PSIONIC_QWEN35_PARTITIONED_TOP_K_THRESHOLD`

The decisive sweep was:

- idle RTX 4080 host
- Psionic only
- sampled `top_k = 40`
- `repeats = 3`
- models `0.8b`, `2b`, `4b`, `9b`
- forced override `PSIONIC_QWEN35_PARTITIONED_TOP_K_THRESHOLD=41`

That override routes `top_k = 40` off the partitioned kernel and back onto the
simple `top_k_f32` path.

Results:

| Model | Default Threshold `40` | Override Threshold `41` | Delta |
| --- | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `505.33` | `269.97` | `-46.58%` |
| `qwen3.5:2b` | `252.94` | `175.85` | `-30.48%` |
| `qwen3.5:4b` | `179.42` | `137.10` | `-23.59%` |
| `qwen3.5:9b` | `110.12` | `92.97` | `-15.58%` |

This answers the crossover question cleanly:

- the simple one-row top-k kernel is not a hidden win for the canonical
  `top_k = 40` contract
- `top_k = 40` must stay on the partitioned path on this host
- future qwen35 sampled tuning should stay inside the partitioned lane rather
  than toggling back to the non-partitioned selector

## Later Follow-On: Fused Remap Removal Also Stayed Flat And Was Reverted

The next partitioned-lane follow-on removed the third kernel in the
partitioned one-row top-k path by carrying original indices through the merge
stage instead of remapping them afterward.

That code landed briefly as:

- commit `351423c3` `Fuse qwen35 partitioned top-k remap into merge`

The clean narrowed rerun on the idle RTX 4080 host was:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_224016_archlinux-fusedmerge.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260328_224016_archlinux-fusedmerge/one_page_summary.md`

Scope:

- models `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`
- contracts `sampled_topk40` and `sampled_topk100`
- `repeats = 3`
- idle RTX 4080 host with the row-by-row isolation runner

The CUDA submission report did improve on paper because the partitioned top-k
path encoded `2` operations instead of `3`, but the end-to-end sampled rows did
not move in a defensible way.

Relative to the same `20260328_210428` sampled baseline:

### `sampled_topk40` deltas vs `20260328_210428`

| Model | Prior Psionic | New Psionic | Delta |
| --- | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `505.33` | `505.77` | `+0.09%` |
| `qwen3.5:2b` | `252.94` | `253.11` | `+0.07%` |
| `qwen3.5:4b` | `179.42` | `179.63` | `+0.12%` |
| `qwen3.5:9b` | `110.12` | `110.16` | `+0.03%` |

### `sampled_topk100` deltas vs `20260328_210428`

| Model | Prior Psionic | New Psionic | Delta |
| --- | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `501.50` | `502.08` | `+0.12%` |
| `qwen3.5:2b` | `250.76` | `250.23` | `-0.21%` |
| `qwen3.5:4b` | `178.31` | `178.38` | `+0.04%` |
| `qwen3.5:9b` | `109.42` | `109.47` | `+0.04%` |

Interpretation:

- this is still noise-level movement, not a meaningful qwen35 throughput gain
- reducing the partitioned top-k kernel count alone was not the active
  bottleneck in the sampled decode path
- one row regressed slightly, so there is no reason to keep the extra kernel
  rewrite on `main`

That experiment was reverted immediately as:

- commit `b8bfc453` `Revert "Fuse qwen35 partitioned top-k remap into merge"`

## Later Follow-On: Benchmark Harness Now Bypasses Shared Prefix Cache

After the later Hermes-driven qwen35 prefix-cache changes landed on `main`,
the canonical sampled `top_k = 40` row on the clean idle RTX 4080 stopped
being a stable benchmark input.

On current `main`, a narrowed sanity check at commit `88ebe571` reproduced:

- `sampled_topk40`
- `qwen3.5:0.8b`
- idle RTX 4080 host
- failure: `qwen35 cuda top-k returned a negative token index -1`

The same row immediately recovered once the benchmark harness forced Psionic
requests onto `PrefixCacheMode::Bypass`.

The clean rerun after that harness change was:

- `qwen3.5:0.8b`
- `sampled_topk40`
- `128/128` output tokens
- bounded-candidate output:
  - `qwen35_output_modes=[top_k_candidates:40]`
  - `qwen35_raw_logits=false`
- Psionic `507.12 tok/s`
- Ollama `330.42 tok/s`

What this means:

- the raw qwen35 sampled lane was still healthy on the clean host
- the new instability lived in shared prefix-cache participation, not in the
  bounded sampled selector itself
- benchmark requests should bypass shared prefix cache so the matrix keeps
  measuring prompt and decode throughput instead of unrelated cache behavior

This is a benchmark-discipline fix, not a new throughput claim.

## Later Follow-On: Wider Top-K Tiles Produced Another Real Sampled Gain

After the prefix-cache benchmark bypass restored a stable clean-host sampled
input again, the next partitioned-lane follow-on revisited kernel shape
instead of host-side cleanup.

The narrowed direct pilot on the idle RTX 4080 first checked whether reducing
the partitioned finalize kernel count mattered. It did not:

- current `main` `qwen3.5:0.8b sampled_topk40` pilot: `506.96 tok/s`
- fused-remap pilot on the same row: `505.40 tok/s`

That confirmed the active bottleneck was still inside the partitioned scan
itself.

The landed follow-on instead doubled the per-thread tile width inside the
partitioned selector:

- `kLogitsTopKItemsPerThread: 8 -> 16`
- tile width: `1024 -> 2048` logits per block pass

The clean full sampled rerun on the idle RTX 4080 host was:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260329_003921_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260329_003921_archlinux-/one_page_summary.md`

Scope:

- models `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`
- contract `sampled_topk40`
- `repeats = 3`
- idle RTX 4080 host with the row-by-row isolation runner

Relative to the same `20260328_210428` sampled baseline:

### `sampled_topk40` deltas vs `20260328_210428`

| Model | Prior Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `505.33` | `511.51` | `+1.22%` | `337.58` |
| `qwen3.5:2b` | `252.94` | `254.07` | `+0.45%` | `206.38` |
| `qwen3.5:4b` | `179.42` | `180.01` | `+0.33%` | `144.10` |
| `qwen3.5:9b` | `110.12` | `110.35` | `+0.21%` | `96.37` |

What this means:

- the partitioned sampled lane still had real headroom in the scan kernel
  itself
- widening the tile to `2048` logits per block pass reduced sampled-step cost
  enough to show up end-to-end on the clean host
- the row-strength classifications stayed unchanged
- bounded-candidate output stayed intact:
  - `qwen35_output_modes=[top_k_candidates:40]`
  - `qwen35_raw_logits=false`

This is a real throughput gain, not a measurement artifact.

## Later Follow-On: Wider Tiles Shifted The Small-Lane Block Optimum Again

After the tile-width increase landed, the next question was whether the old
small-lane `24`-block profile was still the best shape for the canonical
`sampled_topk40` contract.

The idle RTX 4080 resweep showed it was not.

The most sensitive row is `qwen3.5:0.8b sampled_topk40`, and after the wider
tile its observed sweep moved upward:

| Blocks | `0.8b` tok/s |
| --- | ---: |
| `24` | `512.02` |
| `32` | `516.51` |
| `36` | `516.83` |
| `40` | `516.69` |
| `48` | `519.56` |
| `64` | `519.50` |

That made the new conclusion straightforward:

- the earlier `24`-block default was tuned for the older `8`-item tile width
- after `kLogitsTopKItemsPerThread: 8 -> 16`, the best small-lane shape moved
  higher too
- `48` was the best observed small-lane setting and `64` did not improve it

The follow-on full sampled rerun used the runtime override equivalent to the
new default:

- `PSIONIC_QWEN35_PARTITIONED_TOP_K_BLOCKS=48`
- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260329_004540_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260329_004540_archlinux-/one_page_summary.md`

Scope:

- models `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`
- contract `sampled_topk40`
- `repeats = 3`
- idle RTX 4080 host with the row-by-row isolation runner

Relative to the earlier tile-width rerun from `20260329_003921`:

### `sampled_topk40` deltas vs `20260329_003921`

| Model | Prior Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `511.51` | `519.35` | `+1.53%` | `337.91` |
| `qwen3.5:2b` | `254.07` | `256.03` | `+0.77%` | `206.37` |
| `qwen3.5:4b` | `180.01` | `180.99` | `+0.54%` | `144.43` |
| `qwen3.5:9b` | `110.35` | `110.68` | `+0.30%` | `96.55` |

What this means:

- the tile-width gain exposed more headroom in the same partitioned lane
- the next profitable move was not a new algorithm, it was retuning the
  existing small-lane launch shape around the wider tile
- the row-strength classifications stayed unchanged
- bounded-candidate output stayed intact:
  - `qwen35_output_modes=[top_k_candidates:40]`
  - `qwen35_raw_logits=false`

## Later Follow-On: `256 x 8` Block Shape Kept Clean `top_k = 40` Flat And Lifted `top_k = 100`

The next partitioned-lane pass tested a different tradeoff inside the same
tile width:

- old shape: `128 threads x 16 items`
- new shape: `256 threads x 8 items`
- tile width stays `2048` logits per block pass in both cases

The direct idle-host gates were enough to keep this candidate alive:

- clean `sampled_topk40` direct pilots improved slightly on `2b`, `4b`, and
  `9b`
- clean `sampled_topk100` direct pilots stayed flat-to-up across the same
  host
- there was no obvious bounded-candidate regression or fallback to
  `raw_logits`

The landed code change was:

- commit `856e2176` `Adjust qwen35 top-k partitioned block shape`

The exact-commit full sampled rerun on the idle RTX 4080 host was:

- `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260329_011606_archlinux-.json`
- `fixtures/qwen35/benchmarks/reports/qwen35_ollama_matrix_20260329_011606_archlinux-/one_page_summary.md`

Scope:

- models `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`
- contracts `sampled_topk40` and `sampled_topk100`
- `repeats = 3`
- idle RTX 4080 host with the row-by-row isolation runner

### `sampled_topk40` deltas vs `20260329_004540`

| Model | Prior Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `519.35` | `518.74` | `-0.12%` | `338.97` |
| `qwen3.5:2b` | `256.03` | `256.25` | `+0.09%` | `206.22` |
| `qwen3.5:4b` | `180.99` | `181.13` | `+0.08%` | `144.45` |
| `qwen3.5:9b` | `110.68` | `110.75` | `+0.06%` | `96.60` |

Interpretation:

- this keeps the clean `top_k = 40` contract effectively flat overall
- `0.8b` gives back a negligible amount, while the larger models move up
  slightly
- the row-strength classifications stay unchanged
- bounded-candidate output stays intact:
  - `qwen35_output_modes=[top_k_candidates:40]`
  - `qwen35_raw_logits=false`

### `sampled_topk100` deltas vs `20260328_210428`

| Model | Prior Psionic | New Psionic | Delta | Ollama |
| --- | ---: | ---: | ---: | ---: |
| `qwen3.5:0.8b` | `501.50` | `513.06` | `+2.31%` | `322.06` |
| `qwen3.5:2b` | `250.76` | `253.55` | `+1.11%` | `205.61` |
| `qwen3.5:4b` | `178.31` | `179.74` | `+0.80%` | `144.73` |
| `qwen3.5:9b` | `109.42` | `109.98` | `+0.51%` | `97.59` |

Interpretation:

- the newer block shape is worth keeping because it lifts the wider sampled
  lane on all four models
- the earlier top-k tile-width and small-lane block-count gains still hold
- the new shape did not weaken the clean `top_k = 40` evidence enough to
  outweigh the wider-lane gain

## Later Follow-On: `256 x 12` Tile Shape Moved The Sampled Lane Again

After the `256 x 8` follow-on, the next question was whether the same
`256`-thread block could profitably carry a wider per-thread tile again without
giving back the clean bounded-candidate lane.

The next landed code change was:

- commit `317a65d3` `Adjust qwen35 top-k tile width`

What changed:

- `kLogitsTopKItemsPerThread: 8 -> 12`
- block size stays `256`
- tile width becomes `3072` logits per block pass

This pass was first gated by direct Psionic-only A/B pilots on the idle RTX
4080 host against the preserved `256 x 8` baseline binary.

Those pilots stayed on the bounded-candidate output path:

- `qwen35_output_modes=[top_k_candidates:40]` or `[top_k_candidates:100]`
- `qwen35_raw_logits=false`

### Direct pilot deltas vs the `256 x 8` baseline

| Contract | Model | Prior Psionic | New Psionic | Delta |
| --- | --- | ---: | ---: | ---: |
| `sampled_topk40` | `qwen3.5:0.8b` | `519.62` | `521.21` | `+0.31%` |
| `sampled_topk40` | `qwen3.5:2b` | `256.12` | `256.45` | `+0.13%` |
| `sampled_topk40` | `qwen3.5:4b` | `181.04` | `181.09` | `+0.03%` |
| `sampled_topk40` | `qwen3.5:9b` | `110.77` | `110.77` | `-0.00%` |
| `sampled_topk100` | `qwen3.5:0.8b` | `513.14` | `514.86` | `+0.34%` |
| `sampled_topk100` | `qwen3.5:2b` | `251.06` | `253.80` | `+1.09%` |
| `sampled_topk100` | `qwen3.5:4b` | `179.58` | `180.01` | `+0.24%` |
| `sampled_topk100` | `qwen3.5:9b` | `109.88` | `109.93` | `+0.05%` |

Interpretation:

- the new shape is modestly better, not dramatically better
- the gain is strongest on the wider sampled lane and smaller models
- the largest clean sampled row stayed effectively flat instead of regressing
- bounded-candidate output remained intact on every direct gate

### Why this landed before a fresh canonical matrix artifact

The first exact-commit full matrix rerun from a fresh remote worktree hit two
separate non-throughput problems:

- the idle-GPU guard correctly refused to start while a stray
  `psionic-openai-server` process still held CUDA residency
- after that process was killed, a fresh-worktree rerun still had to rebuild
  the full serve stack from a different filesystem path, which invalidated
  most of the shared target cache and turned the exact-commit matrix into a
  long build-bound step instead of a clean throughput rerun

That means the direct pilot numbers above are currently the best clean
evidence for `317a65d3`.

This is enough to justify keeping the code change because:

- all sampled pilot rows moved flat-to-up
- the wider `top_k = 100` lane improved on every screened model
- the clean `top_k = 40` lane did not suffer a meaningful regression
- the candidate stayed on bounded-candidate output instead of falling back to
  `raw_logits`

The remaining follow-up is benchmark publication discipline, not uncertainty
about whether the code change is directionally good.

## Updated Next Steps

- keep using the row-by-row isolation runner so future qwen35 tuning passes
  are measured on a genuinely idle GPU
- keep benchmark requests on `PrefixCacheMode::Bypass` until the later qwen35
  shared prefix-cache regression is classified and fixed separately
- stop spending time on host-side sampler cleanups that only move results by
  noise-level deltas
- stop spending time on non-partitioned `top_k = 40` crossover tuning on this
  host; that path is decisively slower
- publish one fresh repo-owned exact-commit sampled matrix for `317a65d3` once
  the remote rerun can start from an already-built or path-stable worktree
- focus the next optimization pass inside the partitioned bounded-candidate
  lane where it can plausibly reduce actual sampled-step cost further:
  larger structural changes inside the partitioned scan itself, better
  bounded-candidate output handling around the wider lane, or broader step
  fusion around the bounded-candidate output path

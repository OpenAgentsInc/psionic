# Michel RTX 4070 Qwen35 Audit

Date: 2026-03-28

## Scope

This audit records the current state of the external qwen35 native-CUDA versus
Ollama run published from `michellacle/psionic_local` on a constrained laptop
host and binds that result back to Psionic's current public qwen35 record.

Later on March 28, 2026, current `main` routed the canonical
`sampled_topk40` contract through the inclusive partitioned one-row CUDA
selector. A fresh clean RTX 4080 rerun at
`fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_182857_archlinux-.json`
restored Psionic's lead on all four clean length-matched sampled rows. This
audit still explains why Michel's originally published laptop artifact was not
definitive and why constrained-host reruns should happen only on that later
change or newer.

Later the same day, current `main` also zero-initialized the per-request
`qwen3.5:4b` hybrid SSM state and published a fresh full clean-host matrix at
`fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_190650_archlinux-.json`.
That rerun keeps Psionic ahead of local Ollama on all `12` raw `tok/s` rows,
upgrades greedy `qwen3.5:2b` to a `strong` exact-match row, and keeps all
four clean `sampled_topk40` rows ahead while still classifying them as
`weak_length_matched_only`.

Later the same day again, current `main` tuned the partitioned one-row CUDA
top-k block count from `8` to `24` and published a sampled follow-on at
`fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_200654_archlinux-.json`.
That rerun widened Psionic's clean-host sampled lead further without changing
the row-strength classifications: `sampled_topk40` moved to `506.49`, `252.83`,
`179.55`, `110.13 tok/s`, and `sampled_topk100` moved to `492.81`, `247.57`,
`177.28`, `109.02 tok/s`.

Tracked issue:

- `#650` Follow up Michel's RTX 4070 qwen35 run: benchmark-quality closure and
  constrained-host performance

Direct external artifacts reviewed for this audit:

- report:
  `https://github.com/michellacle/psionic_local/blob/main/fixtures/qwen35/benchmarks/reports/qwen35_ollama_20260328_rtx4070_8gb/one_page_summary.md`
- matrix:
  `https://github.com/michellacle/psionic_local/blob/main/fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_124310_rtx4070_laptop_8gb.json`

Repo-local surfaces reviewed for this audit:

- `docs/QWEN35_OLLAMA_COMPARISON.md`
- `docs/INFERENCE_ENGINE.md`
- `crates/psionic-serve/examples/qwen35_cuda_bench.rs`
- issues `#606`, `#630`, `#631`, `#632`, and `#650`

## Host And Artifact Facts

Michel's published run identifies this host:

- `NVIDIA GeForce RTX 4070 Laptop GPU`
- `8 GB` VRAM
- current power limit `55W`
- maximum power limit `90W`

The published matrix artifact records:

- `run_id = qwen35_full_benchmark_20260328_124310`
- `repeats_per_row = 1`
- one reduced row per `contract x backend x model`
- only these machine-readable row fields:
  - `contract`
  - `backend`
  - `model`
  - `output_tokens`
  - `mean_decode_tok_s`

The published one-page summary reports:

- Psionic ahead in `7` of `12` rows
- Psionic behind in `5` of `12` rows
- the worst comparability failure is `sampled_topk40` on `qwen3.5:9b`
  with `11` output tokens on Psionic versus `128` on Ollama

## Relationship To The Existing Psionic Record

This external run does not reopen the established 16 GB RTX 4080 qwen35 record.

The current repo-local comparison document still records:

- greedy qwen35 native CUDA ahead of local Ollama on `0.8b`, `2b`, `4b`, and
  `9b`
- sampled `top_k = 40` qwen35 native CUDA ahead of local Ollama on `0.8b`,
  `2b`, `4b`, and `9b`
- larger bounded-candidate `top_k = 100` qwen35 native CUDA ahead of local
  Ollama on `0.8b`, `2b`, `4b`, and `9b`

Those results are tied to a different host envelope and a stricter published
benchmark contract than Michel's laptop artifact.

The current canonical clean-host record is split by contract:

- the later March 28 full rerun at
  `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_190650_archlinux-.json`
  remains the full greedy-plus-sampled checkpoint
- the later March 28 sampled follow-on at
  `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_200654_archlinux-.json`
  is the latest sampled-only checkpoint after partitioned block-count tuning

This audit therefore treats Michel's run as a constrained-host follow-on, not
as a correction to `#606`, `#631`, or `#632`.

## Michel's Run Is Not Obviously A Stale Pre-Sampled-Lane Checkout

The first stale-checkout hypothesis does not fit the visible evidence.

What is true:

- `michellacle/psionic_local` is a fork of `OpenAgentsInc/psionic`
- the fork's canonical `docs/QWEN35_OLLAMA_COMPARISON.md` matches upstream
  byte-for-byte
- the fork's `qwen35_cuda_bench` already includes:
  - sampled controls beyond the original greedy path
  - bounded qwen35 output-mode reporting
  - `qwen35_output_modes`
  - `qwen35_readback_bytes`
  - `qwen35_raw_logits`
  - per-run `prompt_s`, `decode_s`, `total_s`, and `output_tokens`
- the fork's local-only commits are:
  - `f91c8844` `update documentation`
  - `6fe57414` `cuda: add CUB include fallback and record qwen35 matrix run`
  - `8dd1ba63` `bench: add one-page qwen35 vs ollama summary with graphs`
- the upstream-only commits currently missing from that fork are:
  - `dfbcb34c` `Log HOMEGOLF grad-accum queue correction`
  - `fcf76aaa` `Enable native qwen35 chat tool calling`
  - `d3918814` `Support multi-tool turns and parallel_tool_calls`
  - `bf897aeb` `Add track-aware v2 remote training visualization contracts`

This does not prove the fork is identical to upstream.

It does prove a narrower point:

- the external laptop run is not explained by Michel missing the original
  sampled qwen35 CUDA work from `#630`, `#631`, or `#632`
- the missing upstream commits are not an obvious explanation for the exact
  text-only decode results in the external matrix

## Benchmark-Quality Findings

The external artifact is honest enough to flag its own weak rows, but it is not
strong enough to settle the performance question by itself.

### 1. Most rows are not apples-to-apples decode workloads

`9` of `12` published rows have output-token mismatches between Psionic and
Ollama.

That matters because decode throughput is computed from generated output-token
count divided by decode time. If one backend emits a much shorter completion,
the two runtimes did not process the same decode workload.

The most severe mismatch is:

- `sampled_topk40` `qwen3.5:9b`
  - Psionic: `31.86 tok/s`, `11` output tokens
  - Ollama: `23.88 tok/s`, `128` output tokens

That row is not an honest throughput comparison.

### 2. The most informative rows are the clean sampled `top_k = 40` rows

Historical note:

- later the same day, current `main` closed this specific clean-host sampled
  gap by routing `top_k = 40` through the inclusive partitioned selector
- this section still describes Michel's originally published laptop artifact,
  not the later corrected RTX 4080 rerun

Only three rows in the published matrix keep the same output-token count on
both sides:

- `sampled_topk40` `qwen3.5:0.8b`
  - Psionic `150.66 tok/s`
  - Ollama `154.00 tok/s`
- `sampled_topk40` `qwen3.5:2b`
  - Psionic `81.61 tok/s`
  - Ollama `85.99 tok/s`
- `sampled_topk40` `qwen3.5:4b`
  - Psionic `53.62 tok/s`
  - Ollama `58.83 tok/s`

Psionic loses all three.

These are the primary rows that need real explanation and likely the primary
rows to optimize first if the loss survives a cleaner rerun.

### 3. The published matrix omits the runtime evidence needed to classify the rows

The current `qwen35_cuda_bench` already logs:

- `prompt_tokens`
- `output_tokens`
- `prompt_s`
- `decode_s`
- `total_s`
- `qwen35_output_modes`
- `qwen35_readback_bytes`
- `qwen35_raw_logits`

The published external matrix keeps only:

- `contract`
- `backend`
- `model`
- `output_tokens`
- `mean_decode_tok_s`

That omission blocks the central classification question:

- did Psionic stay on `TopKCandidates { top_k }` for the sampled rows on the
  laptop host, or did it fall back to dense `raw_logits`?

Without that evidence, the artifact cannot tell us whether we are looking at:

- benchmark-quality noise
- bounded-lane runtime losses
- explicit fallback behavior
- host operational constraints

### 4. The published external artifact is single-run evidence

The matrix records `repeats_per_row = 1`.

That is too weak for narrow host-sensitive deltas on a laptop GPU running at
`55W` instead of its published `90W` maximum. A `2%` to `9%` loss margin on
the clean sampled rows is not stable enough to explain without repeated reruns
and host telemetry.

## Output-Token Mismatch Analysis

An output-token mismatch means the two backends did not terminate at the same
point for what was intended to be the same benchmark row.

That can come from:

- EOS on one side earlier than the other
- stop-sequence match on one side earlier than the other
- divergent greedy token selection
- divergent sampled token selection
- any runtime-specific termination condition that changes the generated text

For the current qwen35 harness, the prompt path is already aligned more tightly
than the published matrix suggests:

- both backends use the same GGUF-derived prompt renderer
- the rendered stop sequences are forwarded to both sides
- sampled rows default to the explicit contract values when omitted:
  - `temperature = 0.8`
  - `top_k = 40`
  - `top_p = 0.9`
  - `repeat_penalty = 1.0`
  - `presence_penalty = 0.0`
  - `frequency_penalty = 0.0`
  - `seed = 42`

That means the remaining likely causes are runtime and numerical behavior, not
prompt drift.

Greedy mismatch has one main interpretation:

- the two runtimes picked a different next token at some step
- once that happens, the rest of the completion can diverge and terminate at a
  different length

Sampled mismatch has a broader interpretation:

- even with nominally aligned sampling controls, small differences in candidate
  sets, penalties, probability mass ordering, or token selection can diverge
  the completion very early
- once the first sampled difference appears, output length and finish reason
  can drift substantially

The external matrix does not record enough finish-state metadata to tell which
of those causes actually occurred on each mismatched row.

## What Michel's Run Proves

This external run proves these bounded points:

- Psionic still needs a constrained-host answer on smaller laptop CUDA systems
- the existing public matrix is not enough by itself to answer that question
- the clean sampled `top_k = 40` rows are weaker on Michel's host than on the
  16 GB RTX 4080 host
- the current benchmark-publication path is still too lossy for external host
  audits because it drops the most important qwen35 runtime evidence

## What Michel's Run Does Not Prove

This external run does not yet prove:

- a broad qwen35 native CUDA regression across hosts
- that the bounded sampled lane is broken on the laptop host
- that Psionic fell back to dense `raw_logits` on the weak sampled rows
- that the external host was measured under a stable full-power posture
- that the mismatched rows reflect comparable decode workloads

## Current Most Likely Interpretation

The best current interpretation is:

- the external laptop run is a real signal, but it is not yet a definitive
  benchmark verdict
- the clean sampled `top_k = 40` rows are the main actionable evidence
- the mismatched rows are mostly classification failures until they are rerun
  with finish-state and output-mode evidence preserved
- the host envelope is materially weaker than the 16 GB RTX 4080 baseline and
  may be exposing host-side or synchronization costs that were amortized more
  cleanly on the larger host

This audit therefore does not classify Michel's run as either:

- \"just noise\"
- or \"a proven qwen35 sampled-lane regression\"

The current honest classification is narrower:

- the result is strong enough to justify follow-up work
- the result is not strong enough to explain itself

## Path Forward

### 1. Fix the benchmark artifact before arguing about the rows

The first repo-owned correction should be benchmark-quality closure.

The qwen35 matrix artifact should preserve at least:

- `prompt_tokens`
- `output_tokens`
- `prompt_s`
- `decode_s`
- `total_s`
- `qwen35_output_modes`
- `qwen35_readback_bytes`
- `qwen35_raw_logits`
- finish reason if available
- exact command line or request contract
- exact commit
- exact artifact digest

Single-run published rows should stop being treated as decisive when margins
are narrow or output-token counts diverge.

### 2. Rerun the external host under a clean operational posture

The next rerun on Michel's laptop class should require:

- AC power connected
- highest vendor performance mode
- no unrelated CUDA work resident
- serialized runtime residency
- at least `3` repeats per row
- preferably `5` repeats per row

If practical, the rerun should also preserve:

- GPU clocks
- power draw
- thermals
- VRAM residency

### 3. Classify every materially mismatched row

Every row with large output-token divergence should be explicitly classified as:

- EOS divergence
- stop-sequence divergence
- greedy token divergence
- sampled token divergence
- other termination divergence

The most urgent mismatch case is still:

- `sampled_topk40` `qwen3.5:9b`

That row should not stay in any public throughput conclusion until it is
explained.

### 4. Optimize only after the weak rows are classified

If the clean sampled `top_k = 40` rows already stayed on:

- `qwen35_output_modes=[top_k_candidates:40]`
- `qwen35_raw_logits=false`

then the likely next optimization target is not \"recover the sampled lane.\"
The likely next target is:

- host-side sampling cost
- synchronization overhead
- bounded-candidate path efficiency on the smaller GPU
- another host-envelope cost exposed by the laptop posture

If instead those rows fell back to `raw_logits`, the next target is simpler:

- recover the bounded candidate lane on that host and request shape first

## Immediate Next Steps From This Audit

1. Keep `#650` as the catchall closure issue for the external laptop run.
2. Use the repo-owned qwen35 evidence path now landed in this repo:
   - `cargo run --release -p psionic-serve --example qwen35_cuda_bench -- ... --json-out report.json`
   - `scripts/release/run-qwen35-ollama-matrix.sh`
3. Rerun the external laptop contract with repeated rows and preserved output
   modes.
4. Split any surviving classified bottlenecks into bounded follow-on issues:
   - benchmark artifact quality
   - mismatched-row termination classification
   - clean sampled `top_k = 40` constrained-host performance
   - constrained-host operational benchmark posture if power or residency is
     the limiting factor

## Repo-Owned Closure Landed

The benchmark-quality gap identified in this audit is now closed at the
artifact layer inside Psionic itself.

The repo now has:

- machine-readable per-run qwen35 benchmark reports through
  `qwen35_cuda_bench --json-out`
- a sequential qwen35-versus-Ollama matrix collector at
  `scripts/release/run-qwen35-ollama-matrix.sh`
- a matrix manifest that embeds both backend reports together with:
  - output-token arrays
  - prompt, decode, and total timings
  - `qwen35_output_modes`
  - `qwen35_readback_bytes`
  - `qwen35_raw_logits`
  - host GPU memory and power-limit metadata
  - Psionic commit and Ollama version
- a repo-owned markdown summary that marks output-token comparability row by row

This closes the evidence-preservation gap.

It does not close the constrained-host performance question itself.

Michel-class reruns still need to answer:

- whether the clean sampled `top_k = 40` rows stay on bounded candidates
- whether output-token mismatches on the weak rows are EOS, stop, greedy, or
  sampled divergence
- whether any surviving sampled loss on the laptop host is runtime, host sync,
  or power-envelope limited

## Honest Boundary After This Audit

What is true:

- the current 16 GB RTX 4080 qwen35 record remains intact
- Michel's constrained-host laptop run is important enough to follow up
- the external artifact is not benchmark-complete yet
- the clean sampled `top_k = 40` rows are the main constrained-host weakness
  now visible in public evidence

What is not true:

- we do not yet have definitive proof of a bounded sampled-lane regression on
  the laptop host
- we do not yet know which mismatched rows ended by EOS, stop sequence, or
  other divergence
- we do not yet know whether the main constrained-host gap is runtime, host
  synchronization, or laptop power posture

This audit therefore closes no qwen35 performance issue by itself.

It establishes the current external-host state and the exact path needed to get
from anecdotal laptop evidence to a definitive constrained-host benchmark
record.

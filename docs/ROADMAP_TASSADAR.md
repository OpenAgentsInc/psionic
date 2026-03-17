# Psionic Tassadar Roadmap

> Status: written 2026-03-17 after reviewing `docs/ROADMAP.md`,
> `docs/ARCHITECTURE.md`, `docs/TRAIN_SYSTEM.md`, `README.md`,
> `docs/research/tassadar.md`,
> `docs/audits/2026-03-16-tassadar-phase-12-boundary-audit.md`,
> `docs/audits/2026-03-16-tassadar-phase-13-trainable-surface-audit.md`,
> `docs/audits/2026-03-16-tassadar-phase-14-promotion-green-audit.md`,
> and `docs/audits/2026-03-16-tassadar-phase-16-9x9-reference-run-audit.md`.
>
> This is the lane-specific roadmap for making the Psionic-owned `Tassadar`
> executor lane honest against the full shape of the Percepta
> "Can LLMs Be Computers?" article. It is intentionally narrower than
> `docs/ROADMAP.md`: it is about WebAssembly-first in-model execution,
> exact compiled/proof-backed executor lanes, learned executor lanes,
> long-trace fast decode, and article-class workload closure inside
> `crates/psionic-*`.

Agent execution instruction: implement this roadmap in dependency order, not by
whichever executor demo looks most exciting first. Keep compiled/proof-backed
and learned lanes separate, keep article claims subordinate to artifacts, and
do not let throughput numbers outrun exactness, compatibility, or refusal
truth.

Reference-first instruction: Tassadar work must not be implemented from memory.
Choose the reference that owns the layer being changed:

- start with `docs/ROADMAP.md` for the canonical full-program owner split and
  dependency order
- start with `docs/ARCHITECTURE.md` for the canonical current-state substrate
  and status vocabulary
- start with `docs/TRAIN_SYSTEM.md` for the learned-lane and train-substrate
  contract
- start with `README.md` for the current Tassadar inventory and crate map
- start with `docs/research/tassadar.md` for the literature-grounded design
  constraints, especially the separation between compiled/proof-backed and
  learned lanes
- start with committed artifact roots under `fixtures/tassadar/runs/` for the
  actual exactness, throughput, fit, failure, and proof posture instead of
  relying on remembered blog claims

Psionic-only execution rule: the shipped Tassadar lane must remain
Psionic-owned end to end. Do not close roadmap items by shelling out to a
Python interpreter, an external Wasm runtime, or a tool-use loop and then
pretending the transformer executed the program itself. If the computation did
not run through the Psionic-owned executor substrate and produce Psionic-owned
artifacts, it is not roadmap completion.

## Decision

`Tassadar` belongs in a separate `ROADMAP_TASSADAR.md`, not only as a long
section inside `docs/ROADMAP.md`.

That decision is deliberate for four reasons:

1. `docs/ROADMAP.md` is the canonical full-library roadmap and should remain
   the answer to "what is Psionic overall?"
2. Tassadar now spans runtime, models, serving, eval, train, and research, so
   the full article-shaped path is too detailed to keep only as one embedded
   section in the main roadmap.
3. The current repo already uses lane-specific deep dives such as
   `docs/ROADMAP_CLUSTER.md`, `docs/ROADMAP_FM.md`, `docs/ROADMAP_METAL.md`,
   and `docs/ROADMAP_MLX.md`.
4. The article-shaped Tassadar program needs its own acceptance language,
   separation between compiled/proof-backed and learned claims, and explicit
   long-trace closure path without implying that Tassadar is the whole Psionic
   program.

So the structural rule is:

- `docs/ROADMAP.md` remains canonical
- `docs/ROADMAP_TASSADAR.md` is the Tassadar-specific dependency-ordered deep
  dive

## Objective

Build Tassadar into an article-grade Psionic-owned executor lane with:

- a WebAssembly-first in-model execution substrate that is exact against a CPU
  reference lane
- digest-bound program artifacts, trace artifacts, proof bundles, runtime
  compatibility descriptors, and replay-stable lineage
- a fast decode path that is honest about when it is exact, when it falls
  back, and when it must refuse
- article-class benchmark packages and artifact roots for branch-heavy,
  memory-heavy, long-horizon computation workloads
- a bounded compiled/proof-backed lane that can execute matched workloads
  exactly inside the model-owned executor path
- a learned lane that is kept separate from the compiled lane and only widens
  claims when exactness artifacts justify it
- a path from bounded research candidates to a truthful "the model executes
  Wasm compute internally" claim for article-class workloads

This is not a plan to:

- outsource compute to tool use and describe it as in-model execution
- blur compiled exactness into learned generalization
- call bounded 4x4 proxy success "article parity"
- treat benchmark speed alone as proof that the executor is correct
- claim arbitrary-program or arbitrary-C closure before the instruction,
  profile, workload, and compile pipeline artifacts prove it

## Relationship To The Main Roadmap

This roadmap is subordinate to `docs/ROADMAP.md`.

It depends on and refines work already named there:

- framework-core extensibility and cache identity
- model/runtime-family truth
- serving and routing truth
- train/eval artifact truth
- research-runner and experiment-manifest truth

This roadmap does not widen product scope in `docs/MVP.md`, and it does not
move ownership boundaries out of `docs/OWNERSHIP.md`.

## Ownership Rules

This roadmap must continue to respect `docs/OWNERSHIP.md`:

- `crates/psionic-*` owns the reusable executor runtime, model, eval, train,
  serving, artifact, benchmark, proof, and research surfaces
- `apps/*` remain responsible for product UX and product control flows
- kernel and Nexus remain authority owners, not executor-runtime owners

More specifically:

- `psionic-runtime` owns Wasm execution semantics, trace artifacts, runtime
  compatibility, decode identity, and proof lineage
- `psionic-models` owns executor model families, decode-state contracts,
  compiled-weight artifacts, and truthful model descriptors
- `psionic-eval` owns exactness, benchmark, compatibility, and refusal reports
- `psionic-train` owns learned-lane manifests, training receipts, promotion
  gates, and fit reports
- `psionic-serve` owns explicit executor product and planner-routing surfaces
- `psionic-research` owns bounded architecture, compile-to-weights, and
  long-trace comparison families without widening product claims by itself

## Why This Roadmap Exists

The repo already has a real Tassadar lane, but it does not yet have a separate
document that answers the larger question:

> what is the full dependency-ordered path from today's bounded executor truth
> to an honest implementation of the Wasm-compute vision described in the
> article?

That gap matters because the current state is easy to misread.

The repo already has:

- a real CPU-reference executor lane
- proof-bearing trace artifacts
- a bounded `HullCache` fast path
- a bounded compiled/proof-backed exact Sudoku lane
- a bounded compiled/proof-backed exact Hungarian lane
- a bounded learned 4x4 lane with a green promotion gate
- an honest but still partial learned 9x9 lane

But it does not yet have:

- article-grade exactness on the full long-trace workload envelope it would
  take to justify "arbitrary C code executes inside the transformer"
- article-grade long-trace closure on branch-heavy, memory-heavy workloads
- a truthful learned long-horizon lane for 9x9 Sudoku or larger matching
  problems
- a generalized compile-to-weights or program-specialized deployment pipeline
  that is honest beyond the currently matched bounded corpora

So the right next move is not to widen the marketing language. The right next
move is to make the closure path explicit.

## Current Position

Tassadar already has substantial shipped substrate on `main`.

The current strongest committed artifacts are:

- `fixtures/tassadar/runs/sudoku_v0_promotion_v3`
  - learned 4x4 validation gate is green
- `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0`
  - learned 9x9 lane is honest but still partial
- `fixtures/tassadar/runs/sudoku_v0_supervision_ablation_v1`
  - bounded learned structural-supervision comparison proves richer targets
    improve instruction-pointer and stack-delta exactness without widening the
    learned claim boundary
- `fixtures/tassadar/runs/sudoku_9x9_v0_windowed_family_comparison_v1`
  - same-corpus flat-prefix-vs-windowed 9x9 comparison now proves the
    windowed family keeps bounded exactness parity while making the
    long-trace live-state contract explicit
- `fixtures/tassadar/runs/tassadar_trace_family_comparison_v1`
  - same-corpus sequential-vs-wavefront comparison now proves research-only
    Sudoku and Hungarian alternate trace families preserve final outputs
    exactly while shrinking max total tokens from `5335309` to `52969` on 9x9
    Sudoku and from `11532454` to `22050` on article-sized Hungarian-10x10;
    the sequential CPU trace remains the only full-trace authority
- `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0`
  - exact compiled/proof-backed 9x9 Sudoku lane now exists on the matched 9x9 corpus
- `fixtures/tassadar/runs/sudoku_v0_compiled_executor_v0`
  - bounded compiled/proof-backed Sudoku lane is exact on the matched corpus
- `fixtures/tassadar/runs/hungarian_v0_compiled_executor_v0`
  - bounded compiled/proof-backed Hungarian lane is exact on the matched
    corpus
- `fixtures/tassadar/runs/hungarian_10x10_v0_compiled_executor_v0`
  - exact compiled/proof-backed 10x10 Hungarian lane now exists on the matched
    article-sized corpus
- `fixtures/tassadar/runs/compiled_kernel_suite_v0`
  - exact compiled/proof-backed arithmetic, memory-update, forward-branch, and
    backward-loop kernel suite now exists with exactness-vs-trace-length
    reporting and proof-bearing deployments
- `fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json`
  - compiled article-class closure checker now exists and only turns green when
    the article-sized 9x9 Sudoku bundle, article-sized 10x10 Hungarian bundle,
    and generic compiled kernel suite are all present and the older bounded 4x4
    proxies are no longer the strongest exact compiled evidence in-tree

The current technical reality is:

- exact CPU-reference and proof-bearing runtime substrate exists
- the fast decode path exists, but its exact closure is still workload-shaped
  rather than general
- the compiled/proof-backed lane is stronger today than the learned long-trace
  lane and now includes article-sized 9x9 Sudoku, 10x10 Hungarian exactness,
  a generic compiled kernel suite over arithmetic, memory, branch, and
  loop-heavy workloads, and a dedicated article-closure checker
- the learned 4x4 lane is real
- research-only parallel and wavefront target families now exist beside the
  sequential trace for Sudoku and Hungarian workloads without widening learned
  execution claims
- the learned 9x9 lane still does not fit the full trace honestly under the
  current model contract

That means the honest article-shaped path is:

1. keep the compiled/proof-backed lane as the first exact long-trace answer
2. widen runtime/profile/benchmark truth for real Wasm workloads
3. make long-trace fast decode honest on the workload classes it claims
4. only then widen the learned lane toward article-class long-horizon compute

## Article Target

For this roadmap, "fully implementing the article-shaped Wasm compute lane"
means all of the following are true at once:

- Psionic can take a stable Wasm artifact from a canonical compile pipeline and
  execute it internally through the Tassadar executor path without an external
  interpreter round-trip.
- the same workload has exact CPU-reference parity, trace proof, runtime
  lineage, compatibility/refusal reporting, and benchmark receipts
- the fast decode path is exact on the long-trace workload class it claims,
  with trace-length scaling reports instead of one throughput screenshot
- compiled/proof-backed article workloads exist above today's 4x4 proxies,
  including 9x9 Sudoku, 10x10 Hungarian-class matching, and branch-heavy or
  memory-heavy kernel suites
- learned-lane claims, if made, clear their own exactness and fit bars and are
  never used to dilute compiled-lane truth
- "arbitrary C code" claims only become honest after instruction coverage,
  compile receipts, workload suites, and acceptance runners say so

## Claim Vocabulary

Tassadar uses one coarse persisted `claim_class` vocabulary on run bundles,
promotion bundles, and later acceptance artifacts. It does not replace:

- `claim_boundary`, which keeps the execution boundary honest
- `boundary_label`, which keeps train-side scope envelopes honest
- `serve_posture`, which keeps serving exposure honest

The canonical classes are:

| `claim_class` | Meaning | Required artifacts before the claim can appear |
| --- | --- | --- |
| `research_only` | bounded research candidate or comparison run | a persisted research bundle plus its training, family, or experiment reports |
| `learned_bounded` | learned executor result with an explicit bounded workload envelope | a learned run or promotion bundle plus exactness, failure, and fit artifacts that freeze the bounded envelope |
| `compiled_exact` | exact compiled/proof-backed executor on a matched bounded corpus | a compiled run bundle plus exactness, compatibility/refusal, deployment, and proof artifacts |
| `learned_article_class` | learned executor that clears the article-class acceptance bar | a learned bundle plus later-window exactness, fit closure, and the article acceptance artifacts from Epic 0 / `PTAS-003` |
| `compiled_article_class` | exact compiled/proof-backed executor that clears the article-class acceptance bar | a compiled bundle plus article workload-matrix and acceptance artifacts from Epic 0 / `PTAS-003` |

Until the acceptance artifacts in `PTAS-003` exist and turn green, no persisted
artifact in this repo should honestly use either article-class claim.

The machine-readable acceptance report in
`fixtures/tassadar/reports/tassadar_acceptance_report.json` now maps those
claim boundaries to live repo truth:

- `research_only = true` allows only `research_only` wording for the preserved
  research bundle
- `compiled_exact = true` allows `compiled_exact` wording for the bounded
  compiled/proof-backed Sudoku-v0 and Hungarian-v0 lanes
- `learned_bounded = true` allows `learned_bounded` wording for the green 4x4
  learned promotion bundle
- `fast_path_declared_workload_exact = true` allows only bounded fast-path
  equivalence wording for the declared Sudoku-v0 hull benchmark window; it does
  not authorize full-task or article-parity language by itself
- `compiled_article_class = true` is required before any
  `compiled_article_class` language can appear honestly
- `learned_article_class = true` is required before any
  `learned_article_class` language can appear honestly
- `article_closure = true` is required before article-parity or
  "the article claim is reproduced in-tree" wording can appear honestly

## Allowed Claim Transitions

Allowed claim movement is intentionally narrow:

- `research_only -> compiled_exact`
- `research_only -> learned_bounded`
- `compiled_exact -> compiled_article_class`
- `learned_bounded -> learned_article_class`
- staying in the same claim class for a new bounded run is allowed

Not allowed:

- any direct `research_only -> *_article_class` jump
- any automatic compiled-to-learned or learned-to-compiled upgrade
- treating `eval_only` serving posture as if it changed claim class by itself

## Explicit Non-Goals

These are not goals of this roadmap:

- generic chat-product capability claims
- replacing direct CPU execution as the authority lane
- pretending article screenshots are equivalent to reproducible artifacts
- claiming full softmax-attention parity before the executor-specific fast path
  is closed honestly
- claiming that bounded compiled/proof-backed success on a matched corpus means
  arbitrary-program closure
- claiming that a bounded learned 4x4 or 9x9 result proves a general learned
  Wasm computer

## Current Status Matrix

| Subsystem | Status | Truth today |
| --- | --- | --- |
| Wasm executor runtime, trace ABI, proof, and serving surfaces | `implemented_early` | The core runtime, trace proof, compatibility, benchmark, and serving substrate exists and is usable today inside `crates/psionic-*`. |
| Exact `HullCache` fast path | `partial` | Exact closure exists on bounded validated subsets, but the fast path is not yet honest as a general long-trace answer for all branch-heavy or memory-heavy workloads. |
| Compiled/proof-backed exact executor lane | `implemented_early` | Exact bounded Sudoku-v0 and Hungarian-v0 lanes exist with proof-bearing deployment bundles, but they remain matched-corpus and `eval_only` rather than article-grade general closure. |
| Learned 4x4 executor lane | `implemented_early` | The learned 4x4 promotion gate is green on the committed validation corpus. |
| Learned 9x9 executor lane | `partial` | The first honest 9x9 run is preserved, but the full learned trace still does not fit the current model contract and remains blocked. |
| Article-class long-trace benchmark closure | `partial` | Micro, Sudoku, and Hungarian families exist, but the repo does not yet have one acceptance bar that honestly closes the article-shaped long-trace claim. |
| Generalized program-specialized weights and compile-to-weights | `partial` | Program-specialized compiled executor artifacts exist, but the repo does not yet have generalized compile-to-weights closure for broader Wasm families. |
| Full article-parity Wasm compute claim | `planned` | Psionic is not yet at a truthful "arbitrary C/Wasm compute inside the transformer" claim boundary. |

## Dependency Order

This roadmap is organized into seven epics.

| Epic | Title | Primary outcome |
| --- | --- | --- |
| Epic 0 | Governance and acceptance | explicit claim vocabulary and one acceptance bar for article-shaped Tassadar work |
| Epic 1 | Wasm substrate closure | wider Wasm, compile, trace, and artifact truth for real article-class programs |
| Epic 2 | Fast decode closure | honest long-trace fast-path exactness, scaling, and fallback truth |
| Epic 3 | Compiled/proof-backed article closure | exact article-class workloads through the compiled lane first |
| Epic 4 | Learned executor substrate redesign | richer supervision, recurrence/windowing, and architecture baselines |
| Epic 5 | Learned long-trace closure | honest 9x9 and Hungarian-class learned-lane exactness or explicit bounded failure |
| Epic 6 | Compile-to-weights and hybrid systems | broader program-specialized weights, deployment truth, and planner-integrated article workflows |

## Epic 0: Governance And Acceptance

### Goal

Freeze the claim vocabulary and acceptance discipline so Tassadar work stops
oscillating between bounded research wins and implied article parity.

### Exit Criteria

- one canonical Tassadar claim vocabulary exists
- one repo-owned Tassadar acceptance runner exists
- compiled/proof-backed, learned, and research-only lanes are machine-legible
- article-parity language is tied to acceptance artifacts rather than audits or
  issue comments alone

### Issues

| ID | Status | Work |
| --- | --- | --- |
| `PTAS-001` | implemented | Write the lane-specific Tassadar roadmap. This document closes that issue. |
| `PTAS-002` | implemented | Freeze the Tassadar claim vocabulary: `compiled_exact`, `compiled_article_class`, `learned_bounded`, `learned_article_class`, and `research_only`. This document now defines the vocabulary and transition rules, and persisted bundles now carry `claim_class`. |
| `PTAS-003` | implemented | The repo now has a machine-readable Tassadar acceptance report at `fixtures/tassadar/reports/tassadar_acceptance_report.json`, a repo-owned checker command at `scripts/check-tassadar-acceptance.sh`, and explicit green/red mapping from acceptance fields to allowed claim language. |
| `PTAS-004` | implemented | The compact artifact index now lives at `docs/ROADMAP_TASSADAR_INDEX.md`, mapping the landed artifact-bearing Tassadar phases to their canonical roots, audits, validators, and current claim boundaries. |

## Epic 1: Wasm Substrate Closure

### Goal

Close the runtime, profile, compile, and artifact gaps between today's bounded
executor substrate and the article-shaped "compile low-level programs and run
them internally" target.

### Exit Criteria

- a canonical compile pipeline from source program to Wasm artifact exists
- the active Wasm profiles cover the article-class instruction and memory
  surface honestly
- long-trace workloads have benchmark packages and fixture suites that do not
  rely only on 4x4 proxies
- program compatibility and refusal reports stay explicit as coverage widens

### Issues

| ID | Status | Work |
| --- | --- | --- |
| `PTAS-101` | implemented | The repo now has the dedicated article-shaped `tassadar.wasm.article_i32_compute.v1` profile, aligned model/eval artifacts for the mixed workload suite, and a machine-readable Wasm instruction-coverage report at `fixtures/tassadar/reports/tassadar_wasm_instruction_coverage_report.json` with typed unsupported-opcode refusal examples. |
| `PTAS-102` | implemented | The repo now has a canonical `clang`-backed C-to-Wasm compile receipt at `fixtures/tassadar/reports/tassadar_c_to_wasm_compile_receipt.json`, rooted in `fixtures/tassadar/sources/tassadar_micro_wasm_kernel.c` and `fixtures/tassadar/wasm/tassadar_micro_wasm_kernel.wasm`, with explicit toolchain/config/source/output digests, typed compile refusals, and one canonical `TassadarProgramArtifact` lineage projection. |
| `PTAS-103` | implemented | The article-class benchmark package now widens beyond micro-kernel, Sudoku-v0, and Hungarian-v0 coverage with committed `branch_heavy_kernel`, `memory_heavy_kernel`, and `long_loop_kernel` cases, plus the report at `fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json` that records exactness, throughput, and direct-vs-fallback posture per workload family. The long-loop case materially widens long-horizon coverage under the current article profile ceiling, but the separate million-step decode closure remains `PTAS-204`. |
| `PTAS-104` | implemented | The repo now has the explicit long-horizon trace ABI decision report at `fixtures/tassadar/reports/tassadar_trace_abi_decision_report.json`, plus the committed long-loop article-class evidence bundle at `fixtures/tassadar/runs/long_loop_kernel_trace_abi_v0/execution_evidence_bundle.json`. The report freezes `tassadar.trace.v1` machine-truth authority, separates readable logs from canonical trace artifacts, and records validator-facing ABI pointers across benchmark, compiled, and long-horizon fixture artifacts. |
| `PTAS-105` | implemented | The repo now has the machine-readable workload capability matrix at `fixtures/tassadar/reports/tassadar_workload_capability_matrix.json`, generated from the committed article benchmark report plus compiled and learned bundle artifacts. It records runtime exact vs fallback-only posture per workload family and keeps compiled exact, bounded learned, and partial learned-long-horizon evidence separate instead of collapsing them into one article-parity summary. |

## Epic 2: Fast Decode Closure

### Goal

Make the fast path honest on the workload classes it claims, with scaling
artifacts that justify the "log-time retrieval beats full prefix scans" story.

### Exit Criteria

- exact fast-path parity is demonstrated on the workload class claimed
- fallback and refusal boundaries are explicit on workloads outside that class
- throughput, CPU-gap, and trace-length scaling are published together
- hull-based, sparse, and any successor decode paths are compared under the
  same artifact contract

### Issues

| ID | Status | Work |
| --- | --- | --- |
| `PTAS-201` | implemented | The repo now has the widened HullCache closure report at `fixtures/tassadar/reports/tassadar_hull_cache_closure_report.json`, proving direct exact HullCache closure on `MicroWasmKernel`, `BranchHeavyKernel`, `MemoryHeavyKernel`, and bounded `HungarianMatching` from the committed article benchmark artifact. The same report keeps `LongLoopKernel` and `SudokuClass` explicit as fallback-only families, so the widened closure stays honest about control-flow limits rather than muting refusal truth. |
| `PTAS-202` | implemented | The repo now has the SparseTopK comparison report at `fixtures/tassadar/reports/tassadar_sparse_top_k_comparison_report.json`, comparing SparseTopK against reference-linear and HullCache on the shared article workload set. The report keeps direct exact support on `MicroWasmKernel`, `MemoryHeavyKernel`, and bounded `HungarianMatching`, while leaving `BranchHeavyKernel`, `LongLoopKernel`, and `SudokuClass` explicit as fallback-only families under the current validation contract. |
| `PTAS-203` | implemented | A machine-readable decode-scaling report now exists at `fixtures/tassadar/reports/tassadar_decode_scaling_report.json`, comparing shared linear-memory, forward-branch, and backward-branch-loop workload families across requested reference-linear, HullCache, and SparseTopK execution while recording trace-artifact growth, throughput, CPU-gap, exactness, and direct-vs-fallback posture. |
| `PTAS-204` | implemented | The repo now has the canonical million-step decode benchmark bundle at `fixtures/tassadar/runs/million_step_loop_benchmark_v0/benchmark_bundle.json`, proving one reproducible 1,048,575-step backward-branch loop under the Psionic-owned reference-linear executor path. The bundle carries exactness, compact trace-summary proof lineage, runtime-manifest identity, serialized trace-byte growth, and direct reference-linear throughput receipts while keeping HullCache and SparseTopK explicit as fallback-only at that same horizon. |
| `PTAS-205` | implemented | The repo now has the geometric-variant comparison report at `fixtures/tassadar/reports/tassadar_geometric_variant_report.json`, comparing the promoted runtime HullCache lane against a research-only hierarchical-hull candidate on the shared article workload set. The report keeps current runtime fallback truth explicit on `LongLoopKernel` and `SudokuClass`, while showing that the research-only candidate stays direct and exact on those loop-heavy workloads without promoting that widened class to a runtime claim. |

## Epic 3: Compiled/Proof-Backed Article Closure

### Goal

Reach article-class exact in-model compute through the compiled/proof-backed
lane first, because that is the strongest current path in this repo.

### Exit Criteria

- exact compiled/proof-backed 9x9 Sudoku lane exists
- exact compiled/proof-backed 10x10 Hungarian-class lane exists
- generic branch-heavy and memory-heavy kernel suites exist with proof-bearing
  exactness and refusal reports
- compiled article-class workloads are benchmarked through the same executor
  runtime and artifact surfaces as the bounded lanes

### Issues

| ID | Status | Work |
| --- | --- | --- |
| `PTAS-301` | implemented | Widen the compiled/proof-backed lane from today's matched 4x4 corpora to article-class Sudoku and matching workloads. |
| `PTAS-302` | implemented | Land an exact compiled/proof-backed 9x9 Sudoku bundle with readable-log, token-trace, proof, and throughput artifacts. |
| `PTAS-303` | implemented | Land an exact compiled/proof-backed 10x10 Hungarian-class bundle with benchmark-package, proof, throughput, and explicit claim-boundary artifacts. |
| `PTAS-304` | implemented | Add a generic compiled kernel suite covering arithmetic, memory, branch, and loop-heavy programs with exactness-vs-trace-length reporting. |
| `PTAS-305` | implemented | Add one compiled article-closure checker that only turns green when the bounded 4x4 proxies are no longer the strongest exact compiled result in-tree. |

## Epic 4: Learned Executor Substrate Redesign

### Goal

Redesign the learned lane around what the research memo actually recommends:
more structure, better long-trace architectures, and clearer separation from
compiled-lane truth.

### Exit Criteria

- learned targets include more than next token
- recurrent/windowed and parallel-trace candidates exist beside flat-prefix
  next-token models
- architecture families are compared under the same dataset and artifact
  contract
- later-window evaluation exists so long-trace claims do not over-read early
  prefix wins

### Issues

| ID | Status | Work |
| --- | --- | --- |
| `PTAS-401` | implemented | Widen learned supervision beyond next token to instruction pointer, stack delta, memory diff, branch outcome, and workload-specific state such as Hungarian dual variables. |
| `PTAS-402` | implemented | Add recurrent or windowed executor families that can carry long-horizon state without pretending a flat growing prefix is the only honest option. |
| `PTAS-403` | implemented | Add parallel or wavefront trace families for Sudoku and Hungarian-class workloads and compare them against sequential CPU-style traces. |
| `PTAS-404` | planned | Add later-window and suffix-focused eval artifacts so learned long-trace progress is visible after the first bounded prefix. |
| `PTAS-405` | planned | Compare hull-specialized learned architectures against trainable sparse, hybrid, and recurrent baselines under the same artifact contract. |

## Epic 5: Learned Long-Trace Closure

### Goal

Either make the learned lane honestly article-class on long-horizon workloads
or produce explicit artifact-backed evidence that it remains a bounded or
research-only lane.

### Exit Criteria

- the current 9x9 learned fit blocker is removed honestly or replaced with a
  better long-trace contract
- learned 9x9 and Hungarian-class workloads have exactness-vs-length and fit
  reports
- million-step learned traces are either exact on a defined workload class or
  explicitly refused as unsupported
- article-grade learned claims only appear after acceptance artifacts exist

### Issues

| ID | Status | Work |
| --- | --- | --- |
| `PTAS-501` | partial | Remove the current learned 9x9 full-trace fit cliff without hiding it behind a bounded first-window metric. |
| `PTAS-502` | planned | Land a truthful learned 9x9 promotion gate with later-window and full-trace exactness criteria, not only first-prefix scores. |
| `PTAS-503` | planned | Land a learned Hungarian-class lane with explicit dual-state supervision and exactness-vs-trace-length reporting. |
| `PTAS-504` | planned | Add a million-step learned trace benchmark family or an explicit learned refusal policy for workloads beyond the supported horizon. |
| `PTAS-505` | planned | Add one learned article-closure audit that says either "exact article-class learned executor exists" or "the learned lane remains bounded" with no middle-ground marketing language. |

## Epic 6: Compile-To-Weights And Hybrid Systems

### Goal

Widen today's bounded program-specialized compiled executor artifacts into a
broader program-to-weights and hybrid execution system without collapsing the
compiled, learned, and planner-owned layers into one vague claim.

### Exit Criteria

- program-specialized weight deployments extend beyond the current matched
  corpora
- compile-to-weights truth stays bound to source program, compile evidence, and
  runtime contract
- planner and serving routes can invoke the executor lane for article-class
  compute while preserving evidence and refusal truth
- any hybrid article demos are reproducible from repo-owned artifacts

### Issues

| ID | Status | Work |
| --- | --- | --- |
| `PTAS-601` | partial | Widen the current program-specialized compiled-weight lane beyond today's bounded Sudoku and Hungarian corpora. |
| `PTAS-602` | planned | Add a generalized program-to-weights benchmark suite that compares direct tokenized execution against compiled-weight deployment on the same Wasm workloads. |
| `PTAS-603` | planned | Add a served article-workload executor session surface that emits readable-log, token-trace, proof, and benchmark identities without pretending to be ordinary tool use. |
| `PTAS-604` | planned | Add planner-owned hybrid workflows where larger reasoning systems route exact compute spans into Tassadar while preserving routing, proof, and refusal truth. |
| `PTAS-605` | planned | Add one final article-parity closeout audit that only turns green when the repo can reproduce article-class Wasm compute claims from local artifacts and commands. |

## Closing Rule

This roadmap only closes honestly when the repo can say all three of the
following with artifacts:

- compiled/proof-backed article-class Wasm compute exists inside the Psionic
  executor lane
- the fast decode path is honest about its exact workload class and scaling
- learned-lane claims, if any, are separately justified rather than borrowed
  from the compiled lane

Before that point, the repo should keep using narrower language:

- bounded compiled/proof-backed lane
- bounded learned lane
- research-only long-trace candidate
- partial article-shape substrate

# Tassadar Article Transformer Stack Boundary

This document is the canonical Transformer stack boundary for the
article-equivalence closure wave.

The final article route is not one monolithic new crate. It is one explicit
boundary spanning existing crates, with `psionic-transformer` as the
architecture anchor.

## Boundary Decision

The canonical owned article route must use this boundary:

- `psionic-core` plus `psionic-array` for tensor and bounded array ops
- `psionic-nn` plus `psionic-transformer` for layer, parameter-state, and
  reusable Transformer-architecture interfaces
- `psionic-models` for the canonical article-model wrapper, weight artifact
  format, and forward-pass-owned trace hook surface
- `psionic-runtime` for replay, trace ABI, proof identity, and receipt hooks

The final article route must not bypass `psionic-transformer` by reintroducing
an ad hoc mixed implementation spread directly across model and runtime crates.

## Ownership Diagram

```text
psionic-core + psionic-array
  -> tensor metadata, layouts, devices, bounded array ops

psionic-nn + psionic-transformer
  -> module state, primitive layers, reusable Transformer architecture

psionic-models
  -> canonical article model artifact, forward-pass hooks, weight bundle format

psionic-runtime
  -> replay, trace ABI, proof identity, and receipt hooks
```

## Required Interfaces

### Tensor and array ops

- owner modules:
  `crates/psionic-core/src/lib.rs`
  `crates/psionic-array/src/lib.rs`
- purpose:
  the article route consumes tensor metadata, layout truth, and bounded array
  execution through these shared surfaces

### Layer and parameter state

- owner modules:
  `crates/psionic-nn/src/lib.rs`
  `crates/psionic-nn/src/layers.rs`
  `crates/psionic-transformer/src/lib.rs`
  `crates/psionic-transformer/src/attention.rs`
  `crates/psionic-transformer/src/blocks.rs`
  `crates/psionic-transformer/src/encoder_decoder.rs`
- purpose:
  primitive layer semantics plus reusable Transformer attention, embeddings,
  feed-forward, residual and norm block composition, encoder-decoder stack
  assembly, masking, and probability-trace export stay here;
  `psionic-transformer` is the architecture anchor

### Model artifact format

- owner modules:
  `crates/psionic-models/src/lib.rs`
  `crates/psionic-models/src/tassadar_article_transformer.rs`
- purpose:
  the canonical article model descriptor, route selection, and later weight
  artifact identity all live in `psionic-models`

### Forward-pass trace hooks

- owner modules:
  `crates/psionic-models/src/tassadar_article_transformer.rs`
  `crates/psionic-runtime/src/tassadar.rs`
- purpose:
  forward-pass-owned trace hooks are emitted by the model boundary and
  serialized into runtime-owned trace ABI surfaces

## Canonical Wrapper Note

The canonical article wrapper is now
`crates/psionic-models/src/tassadar_article_transformer.rs`.

The older
`crates/psionic-models/src/tassadar_executor_transformer.rs`
remains a separate research and comparison lane. It is not the canonical
paper-faithful article route.

### Replay and receipt hooks

- owner modules:
  `crates/psionic-runtime/src/proof.rs`
  `crates/psionic-runtime/src/tassadar.rs`
- purpose:
  replay, proof identity, trace receipts, and execution challenge hooks remain
  runtime-owned

## Dependency Direction

The boundary is backed by real dependency checks:

- `psionic-models` must depend on `psionic-transformer`
- `psionic-transformer` must depend on `psionic-nn`
- `psionic-transformer` must not depend on `psionic-models` or
  `psionic-runtime`
- `psionic-nn` must not depend on `psionic-train`, `psionic-models`, or
  `psionic-runtime`
- `psionic-runtime` must not depend on `psionic-models`
- `psionic-models` must not depend on `psionic-eval` or `psionic-research`

Optimizer and training-facing orchestration can live above this boundary, but
it must not pull the lower-level `psionic-nn` layer substrate back into the
training or model crates. The repo now keeps that interop split in
`psionic-nn-optimizers`.

`TAS-164` now closes the first bounded article-Transformer training lane on
top of this split in
`crates/psionic-train/src/tassadar_article_transformer_training.rs`.
That lane consumes the canonical article wrapper from `psionic-models`,
keeps reusable encoder-decoder architecture ownership in
`psionic-transformer`, and leaves optimizer-loop orchestration in
`psionic-train` rather than reabsorbing Transformer ownership into the train
crate.

`TAS-165` now closes the first runtime receipt lane on top of the same split.
`psionic-models` now emits one canonical
`forward_with_runtime_evidence(...)` entrypoint from
`crates/psionic-models/src/tassadar_article_transformer.rs`, while
`psionic-runtime` now owns the article-Transformer runtime bundle in
`crates/psionic-runtime/src/tassadar_article_transformer_forward_pass.rs`.
That keeps forward-pass trace capture at the model boundary, reusable
encoder-decoder execution in `psionic-transformer`, and runtime-manifest plus
proof-bundle ownership in `psionic-runtime` instead of collapsing those
receipt hooks back into the model crate.

`TAS-166` now freezes the stronger distinction between "the actual owned
Transformer stack exists" and "final article equivalence is proven." The real
owned stack now spans reusable attention/block/encoder-decoder architecture in
`psionic-transformer`, the canonical article wrapper in `psionic-models`, the
bounded training lane in `psionic-train`, and the runtime evidence lane in
`psionic-runtime`. The older fixture-backed lane in
`crates/psionic-models/src/tassadar.rs` and the older
`crates/psionic-models/src/tassadar_executor_transformer.rs` comparison lane
remain explicit non-canonical surfaces, while `psionic-core`,
`psionic-array`, and `psionic-nn` remain lower substrate rather than
article-route proof by themselves.

`TAS-167` now adds the runtime-owned machine-step schema for the canonical
article trace domain in
`crates/psionic-runtime/src/tassadar_article_trace_schema.rs`
and binds that schema directly to the shared trace tokenizer in
`crates/psionic-models/src/tassadar_sequence.rs`
plus the canonical article wrapper in
`crates/psionic-models/src/tassadar_article_transformer.rs`.
That split keeps prompt/trace boundary truth, stack/local/memory channel truth,
and terminal halt-marker truth explicit at the runtime boundary while leaving
tokenization and source/target token batching in the model boundary instead of
collapsing those responsibilities back into the legacy
`tassadar_executor_transformer.rs` comparison lane.

`TAS-167A` now adds the first explicit prompt, tokenization, and
representation invariance gate on top of that trace-bound route.
`psionic-runtime` now owns one explicit prompt-field surface plus local-slot
remap and unreachable-suffix helpers in
`crates/psionic-runtime/src/tassadar_article_representation_invariance.rs`,
while `psionic-models` now owns symbolic retokenization and prompt/target
symbolic recomposition helpers in
`crates/psionic-models/src/tassadar_sequence.rs`.
That split lets the repo prove exact-trace stability for whitespace, prompt
field-order, and dead-code layout perturbations, while keeping local-renaming
cases explicitly representation-sensitive but canonically equivalent instead of
pretending every semantic-preserving perturbation must leave the raw trace
identical. The machine-readable gate also keeps bounded suppressions explicit
when an article case exceeds the current trace-domain reference-model position
window rather than widening the support boundary silently.

`TAS-168` now closes the artifact-backed descriptor tranche on top of the same
split. `psionic-models` now owns committed canonical and trace-bound
article-Transformer descriptors plus safetensors bundles under
`fixtures/tassadar/models/`, with explicit tensor inventory, save/load
roundtrip, and digest-bound artifact metadata in
`crates/psionic-models/src/tassadar_article_transformer.rs`. `psionic-runtime`
now binds the forward-pass evidence lane in
`crates/psionic-runtime/src/tassadar_article_transformer_forward_pass.rs` to
the descriptor digest, stable weight-bundle digest, and primary safetensors
SHA-256 from that model-owned artifact boundary instead of synthesizing model
identity from a fixture-only trainable surface. The legacy fixture-backed lane
in `crates/psionic-models/src/tassadar.rs` remains explicit non-canonical
rather than silently masquerading as the owned article model.

`TAS-169` now closes the first real trained-weight tranche on top of that same
split. `psionic-train` now owns a bounded article-class production lane in
`crates/psionic-train/src/tassadar_article_transformer_weight_production.rs`
that distills one explicit `32`-token trace-prefix window from the canonical
Hungarian article demo into the committed trace-bound article wrapper while
keeping the kernel-family cases as held-out evidence. `psionic-models` now
owns the resulting committed trained trace-bound descriptor and safetensors
artifact under `fixtures/tassadar/models/`, while `psionic-eval` and
`psionic-research` now freeze the corresponding production report and summary
without pretending that this first trained artifact is already the final
reference-linear proof route, full article-class exactness, or final
article-equivalence green status.

`TAS-169A` now closes the provenance hardening tranche on top of that same
split. `psionic-models` now points the trained trace-bound route at one
committed lineage contract under
`fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0_lineage_contract.json`,
while `psionic-eval` now audits that manifest against the exact workload set,
training-config snapshot, source inventory, checkpoint lineage, descriptor
digests, and committed artifact digests in
`fixtures/tassadar/reports/tassadar_article_transformer_weight_lineage_report.json`.
That keeps "weights exist" separate from "weight provenance is frozen and
challengeable" and still does not widen the public claim beyond this bounded
trained artifact.

`TAS-170` now closes the bounded replacement-certificate tranche on top of the
same split. `psionic-eval` now compares the historical fixture-backed
reference-linear lane against the committed trained trace-bound article wrapper
across the full canonical article corpus in
`fixtures/tassadar/reports/tassadar_article_fixture_transformer_parity_report.json`,
requiring routeability, exact trace parity, and exact output and terminal-state
parity on every declared case while keeping forward-bundle checks explicit only
for the cases that fit the current model window. `psionic-serve` now projects
that same result into one narrow served publication at
`fixtures/tassadar/reports/tassadar_article_transformer_replacement_publication.json`.
This is the point where the owned Transformer route is certified as the
replacement bounded truth carrier for the article corpus, but it is still not
yet the point where direct no-tool proof ownership has moved off the fixture
lane. That stronger handoff remains the next tranche.

`TAS-171` now closes that first direct-proof handoff without pretending the
entire article-exactness family is done. `psionic-serve` now binds the bounded
direct no-tool proof report in
`fixtures/tassadar/reports/tassadar_direct_model_weight_execution_proof_report.json`
to the committed trained trace-bound Transformer model rather than the
historical fixture model, and every receipt now carries the explicit trained
lineage contract plus the parity-certificate dependency that justifies the
rebind. The old fixture lane remains visible as the historical baseline, but
the canonical bounded direct-proof family for the three named workloads now
belongs to the Transformer-backed reference-linear route. Full declared-family
reference-linear exactness, anti-memorization, contamination independence, and
the stronger clean-room ownership verdict still remain later tranches starting
at `TAS-171A`.

`TAS-171A` now closes that next exactness tranche on top of the same route.
`psionic-eval` now commits one full-family exactness gate at
`fixtures/tassadar/reports/tassadar_article_transformer_reference_linear_exactness_gate_report.json`
that replays the full declared article corpus through the owned
Transformer-backed reference-linear trace-domain route, records explicit
exact/mismatch/refused rows, and separately verifies that the three bounded
direct-proof cases remain bound to the committed direct model-weight proof
report. `psionic-research` now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_transformer_reference_linear_exactness_summary.json`.
This is the point where the owned reference-linear route is now machine-readably
green on the full declared article family before fast-route promotion begins,
but it still does not close anti-memorization, contamination independence,
benchmark equivalence, single-run no-spill closure, or the final
article-equivalence claim.

`TAS-171B` now closes that next anti-memorization tranche on top of the same
reference-linear route. `psionic-eval` now commits one deterministic
generalization gate at
`fixtures/tassadar/reports/tassadar_article_transformer_generalization_gate_report.json`
covering held-out randomized bounded-family cases, adversarial article-scale
Sudoku and Hungarian variants, size/structure scaling across bounded and
article-scale programs, and mixed-order curriculum runs, all with explicit
out-of-distribution fingerprints against the declared article corpus and the
family corpora. `psionic-research` now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_transformer_generalization_summary.json`.
This is the point where the owned reference-linear route is now
machine-readably green on a held-out and adversarial article-envelope
generalization suite rather than only the declared corpus itself, but it still
does not close dataset contamination independence, fast-route promotion,
benchmark equivalence, single-run no-spill closure, clean-room weight
causality, or the final article-equivalence claim.

`TAS-171C` now closes that next dataset-contamination and
evaluation-independence tranche on top of the same route. `psionic-eval` now
commits one explicit audit at
`fixtures/tassadar/reports/tassadar_article_evaluation_independence_audit_report.json`
that ties the committed `TAS-171B` held-out and adversarial suite back to the
committed trained lineage contract, then requires exact case-id exclusion,
exact source/target/sequence digest exclusion, prefix-window near-duplicate
checks, generator-level separation, and profile-level feature-separation
review between the training slice and the evaluation slice. `psionic-research`
now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_evaluation_independence_summary.json`.
This is the point where the repo now machine-readably says the current
evaluation win cannot be explained by simple leaked training exposure inside
the bounded article slice, but it still does not close fast-route promotion,
benchmark equivalence, single-run no-spill closure, clean-room weight
causality, or the final article-equivalence claim.

`TAS-172` now closes the fast-route selection tranche on top of the same
owned route. `psionic-eval` now commits one machine-readable selection report
at
`fixtures/tassadar/reports/tassadar_article_fast_route_architecture_selection_report.json`
that compares the current promoted HullCache runtime lane against the
artifact-backed recurrent runtime baseline, the research hierarchical-hull
candidate, and the bounded 2D-head hard-max research lane while keeping
same-harness exactness, direct-versus-fallback posture, and route-contract fit
explicit. `psionic-research` now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_fast_route_architecture_selection_summary.json`.
This is the point where the repo now machine-readably chooses `HullCache` as
the canonical fast article route because it is the only fast family that is
both exact on the current article matrix and already compatible with the
canonical planner route contract, but it still does not claim that the owned
Transformer-backed model already runs on that fast path. That integration,
no-fallback closeout, and throughput-floor closeout remain `TAS-173` through
`TAS-175`.

`TAS-173` now closes that next integration tranche on top of the same owned
route. `psionic-models` now projects the trained
`tassadar-article-transformer-trace-bound-trained-v0` artifact into the served
executor descriptor boundary with explicit `ReferenceLinear` plus `HullCache`
decode support on the canonical article profile, and `psionic-serve` now uses
that trained model identity for the default article-session and hybrid article
workflow surfaces instead of keeping the historical article fixture as the
canonical served model. The committed
`fixtures/tassadar/reports/tassadar_direct_model_weight_execution_proof_report.json`
now binds one canonical Transformer model-descriptor digest plus one published
route digest into the direct proof family, and the committed
`fixtures/tassadar/reports/tassadar_article_fast_route_implementation_report.json`
plus
`fixtures/tassadar/reports/tassadar_article_fast_route_implementation_summary.json`
now freeze the descriptor, replacement-publication, live article-session,
hybrid-route, and direct-proof closure statement in one machine-readable
artifact pair. This is the point where the repo now machine-readably says the
selected `HullCache` fast path is owned by the canonical Transformer-backed
article route itself, but full no-fallback exactness and throughput-floor
closure still remain `TAS-174` and `TAS-175`.

`TAS-174` now closes that no-fallback exactness tranche on top of the same
owned route. `psionic-runtime` now keeps the canonical `HullCache` lane direct
for the committed article-class corpus rather than falling back on the long
loop, Sudoku, and Hungarian article families, while still refusing arbitrary
backward-branch programs outside that bounded corpus. The committed
`fixtures/tassadar/reports/tassadar_hull_cache_closure_report.json` now shows
LongLoopKernel, SudokuClass, and HungarianMatching as exact with zero fallback
rows, the committed dedicated fast-route exactness session and hybrid workflow
artifacts now expose representative long-loop, Sudoku, and Hungarian cases as
direct `HullCache` executions on the trained
`tassadar-article-transformer-trace-bound-trained-v0` model, and
`fixtures/tassadar/reports/tassadar_article_fast_route_exactness_report.json`
plus
`fixtures/tassadar/reports/tassadar_article_fast_route_exactness_summary.json`
now freeze that closure in one machine-readable artifact pair. This is the
point where the repo now machine-readably says the selected fast route is exact
and no-fallback on the declared canonical article workload families inside the
current article profile, but throughput-floor closure and the broader final
article-equivalence claim still remain `TAS-175` and later tranches.

`TAS-175` now closes that throughput-floor tranche on top of the same owned
route. `psionic-runtime` now commits the bounded
`fixtures/tassadar/runs/article_fast_route_throughput_v1/article_fast_route_throughput_bundle.json`
artifact, which measures direct `HullCache` throughput on the committed
Hungarian article run, the committed `sudoku_9x9_test_a` hard-Sudoku stand-in
run, and the bounded million-step and multi-million-step kernel set while
keeping the later Arto and benchmark-wide Sudoku closure explicitly out of
scope. `psionic-eval` now freezes the joined floor, prerequisite, and
cross-machine drift contract at
`fixtures/tassadar/reports/tassadar_article_fast_route_throughput_floor_report.json`,
and `psionic-research` mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_fast_route_throughput_floor_summary.json`.
This is the point where the repo now machine-readably says the selected fast
route clears the declared CPU throughput floor on the bounded committed route,
but the broader final article-equivalence claim still remains later frontend,
benchmark, single-run, and weight-ownership tranches.

`TAS-176` now closes the first frontend/compiler-boundary tranche on top of the
same owned route. `psionic-compiler` now commits the declared article
frontend/compiler envelope manifest at
`fixtures/tassadar/sources/tassadar_article_frontend_compiler_envelope_v1.json`,
freezing one Rust-source-only `rustc` -> `wasm32-unknown-unknown`
`#![no_std]` / `#![no_main]` envelope with explicit `core`-only source-surface
truth, explicit admitted ABI rows, and explicit refusal rows for C/C++ ingress,
std/alloc surface, host imports, syscall-dependent rows, UB-dependent rows, and
wider ABI shapes. `psionic-eval` now freezes the machine-readable closure
artifact at
`fixtures/tassadar/reports/tassadar_article_frontend_compiler_envelope_report.json`,
binding that declared envelope to the committed Rust source canon, current
toolchain identities, zero-import Wasm outputs on the admitted cases, and
representative refusal probes on the out-of-envelope cases. `psionic-research`
now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_frontend_compiler_envelope_summary.json`.
This is the point where the repo now machine-readably says article claims must
bind to one explicit frontend/compiler envelope instead of borrowing the older
Rust-only closeout path by implication, but corpus expansion and full
Hungarian/Sudoku demo parity still remain later frontend tranches.

`TAS-177` now closes the next frontend breadth tranche on top of that same
owned route. `psionic-eval` now commits the broader article-envelope frontend
corpus and compile matrix at
`fixtures/tassadar/reports/tassadar_article_frontend_corpus_compile_matrix_report.json`,
covering committed Rust sources across arithmetic, branch-heavy, state-machine,
allocator-backed-memory, Hungarian-like, and Sudoku-like support families while
keeping typed refusal and toolchain-failure rows explicit under the declared
Rust-only envelope. `psionic-research` now mirrors the operator-readable
summary at
`fixtures/tassadar/reports/tassadar_article_frontend_corpus_compile_matrix_summary.json`.
This is the point where the repo now machine-readably says the declared
frontend/compiler envelope is exercised by a broader committed corpus instead
of only the older narrow path, but full Hungarian/Sudoku demo-source parity and
the final frontend claim surface still remain later tranches.

`TAS-178` now closes that article-demo frontend parity tranche on top of the
same owned route. `psionic-eval` now commits the dedicated article-demo
frontend parity artifact at
`fixtures/tassadar/reports/tassadar_article_demo_frontend_parity_report.json`,
which recompiles the canonical Hungarian and Sudoku article demo sources
through the declared envelope into dedicated `_tas178` Wasm artifacts beside
the existing bounded route and verifies stable source, case-id, and
compiled-executor workload-id binding against the existing bounded reproducers.
`psionic-research` now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_demo_frontend_parity_summary.json`.
This is the point where the repo now machine-readably says the demo sources
themselves close through the declared frontend/compiler envelope, while later
interpreter-breadth, benchmark-wide, and final article-equivalence tranches
remain open.

`TAS-179` now closes the declared interpreter-breadth envelope tranche on top
of that same owned route. `psionic-data` now commits the machine-readable
article interpreter breadth manifest at
`fixtures/tassadar/sources/tassadar_article_interpreter_breadth_envelope_v1.json`,
freezing one explicit claim surface over the frozen core-Wasm floor, the
current named article i32 profiles, and the later required search-process,
long-horizon control, and module-scale Wasm-loop families while keeping
linked-program bundles research-only and import-mediated processes,
dynamic-memory resume, memory64, multi-memory, component-linking, exception
profiles, and broader float semantics explicitly outside the article envelope.
`psionic-eval` now commits the machine-readable closure artifact at
`fixtures/tassadar/reports/tassadar_article_interpreter_breadth_envelope_report.json`,
and `psionic-research` mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_interpreter_breadth_envelope_summary.json`.
This is the point where the repo now machine-readably says later
article-equivalence breadth claims must stay inside one declared envelope
instead of borrowing arbitrary-program language by implication, while the
follow-on breadth-suite, benchmark, single-run, and final article-equivalence
tranches remain open.

`TAS-179A` now closes that follow-on breadth-suite tranche on top of the same
owned route. `psionic-data` now commits one generic article-program family
suite manifest at
`fixtures/tassadar/sources/tassadar_article_interpreter_breadth_suite_v1.json`,
freezing the exact arithmetic, call-heavy, allocator-backed, indirect-call,
branch-heavy, loop-heavy, state-machine, and parser-style rows plus their
envelope anchors, authority refs, owner surfaces, and required evidence ids.
`psionic-eval` now commits one green suite gate at
`fixtures/tassadar/reports/tassadar_article_interpreter_breadth_suite_gate_report.json`,
proving those rows against the committed article frontend corpus, call-frame,
profile-completeness, runtime-closeout, module-scale, and trap/exception
surfaces without widening the claim outside the declared envelope.
`psionic-research` now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_interpreter_breadth_suite_gate_summary.json`.
This is the point where the repo now machine-readably says the declared
article interpreter breadth blocker itself is closed inside the bounded public
envelope, while benchmark-wide, single-run, and final article-equivalence
tranches remain open.

`TAS-180` now closes the Hungarian article-demo parity tranche on top of that
same owned route. `psionic-serve` now commits one direct article-session
artifact at
`fixtures/tassadar/reports/tassadar_article_hungarian_demo_fast_route_session_artifact.json`
and one planner-owned hybrid-workflow artifact at
`fixtures/tassadar/reports/tassadar_article_hungarian_demo_fast_route_hybrid_workflow_artifact.json`,
each keeping `hungarian_10x10_test_a` direct on `HullCache` with the trained
trace-bound article model and the committed article outputs. `psionic-eval`
now commits the joined parity artifact at
`fixtures/tassadar/reports/tassadar_article_hungarian_demo_parity_report.json`,
binding the canonical Hungarian source, the committed frontend parity row, the
existing no-tool reproducer proof, the new served fast-route artifacts, and the
declared throughput-floor receipt into one machine-readable TAS-180 closure
surface without claiming broader benchmark closure. `psionic-research` now
mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_hungarian_demo_parity_summary.json`.
This is the point where the repo now machine-readably says the canonical 10x10
Hungarian article demo is closed on the fast route inside the bounded public
envelope, while named hard-Sudoku closure, unified demo-and-benchmark parity,
single-run closure, and final article-equivalence tranches remain open.

`TAS-181` now closes the named hard-Sudoku tranche on top of that same owned
route. `psionic-runtime` now commits the bounded
`fixtures/tassadar/runs/article_hard_sudoku_benchmark_v1/article_hard_sudoku_benchmark_bundle.json`
bundle, which measures direct `HullCache` exactness plus the article's 180
second runtime ceiling across the declared hard-Sudoku suite and the new named
`sudoku_9x9_arto_inkala` case. `psionic-serve` now commits matching direct
article-session and planner-owned hybrid-workflow artifacts at
`fixtures/tassadar/reports/tassadar_article_hard_sudoku_fast_route_session_artifact.json`
and
`fixtures/tassadar/reports/tassadar_article_hard_sudoku_fast_route_hybrid_workflow_artifact.json`,
keeping both declared hard-Sudoku cases direct on the trained trace-bound
article model. `psionic-eval` now binds the declared suite manifest, the
existing Sudoku frontend/no-tool anchors, the two served fast-route artifacts,
and the runtime bundle into the joined TAS-181 closure artifact at
`fixtures/tassadar/reports/tassadar_article_hard_sudoku_benchmark_closure_report.json`,
while `psionic-research` mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_hard_sudoku_benchmark_closure_summary.json`.
This is the point where the repo now machine-readably says the named Arto
Inkala case and the declared hard-Sudoku suite are closed on the fast route
inside the bounded public envelope, while the later unified demo-and-benchmark
gate, single-run closure, and final article-equivalence tranches remain open.

`TAS-182` now closes that later unified demo-and-benchmark gate on top of the
same owned route. `psionic-eval` now commits the joined gate artifact at
`fixtures/tassadar/reports/tassadar_article_demo_benchmark_equivalence_gate_report.json`,
which binds the committed `TAS-180` Hungarian demo parity report and the
committed `TAS-181` hard-Sudoku benchmark closure report into one canonical
joined surface while explicitly anchoring the route boundary on
`crates/psionic-transformer/Cargo.toml` and this boundary document.
`psionic-research` now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_demo_benchmark_equivalence_gate_summary.json`,
and `psionic-provider` now projects the same gate into a provider-facing
receipt. This is the point where the repo now machine-readably says the
article's combined demo-and-benchmark surface is closed inside the bounded
public envelope, while no-spill single-run closure, clean-room weight
causality, route minimality, and final article-equivalence tranches remain
open.

`TAS-183` now closes that later single-run no-spill tranche on top of the same
owned route. `psionic-eval` now commits the joined closure artifact at
`fixtures/tassadar/reports/tassadar_article_single_run_no_spill_closure_report.json`,
binding the committed `TAS-182` demo-and-benchmark prerequisite, the canonical
article runtime closeout report, the fast-route throughput-floor report, the
trained trace-bound article model descriptor, and explicit negative-control
continuation lanes from execution checkpoints, spill or tape-store execution,
effect-safe resume, and dynamic-memory resume into one machine-readable
surface. That report keeps the selected fast route fixed to
`HullCache` under `tassadar.decode.hull_cache.v1`, requires deterministic exact
million-step and multi-million-step horizon closure, step and trace digest
alignment, bounded context-to-horizon posture, and explicit refusal of
stochastic retry-farming or teacher-forcing escape hatches. `psionic-research`
now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_single_run_no_spill_closure_summary.json`.
This is the point where the repo now machine-readably says the article's
single-run no-spill posture is closed inside the bounded public envelope,
while clean-room weight causality, route minimality, and final
article-equivalence tranches remain open.

`TAS-184` now closes that next clean-room interpreter-ownership tranche on top
of the same owned route. `psionic-eval` now commits the joined ownership gate
artifact at
`fixtures/tassadar/reports/tassadar_article_interpreter_ownership_gate_report.json`,
which binds the canonical Transformer boundary, the declared interpreter
breadth suite, a widened six-case generic direct no-tool proof suite, a
route-purity audit, a computation-mapping report, a breadth conformance
matrix, and concrete weight-perturbation and locality evidence into one
machine-readable ownership verdict. The widened direct-proof suite keeps the
three committed served proof receipts while adding the declared `micro`,
`branch-heavy`, and `memory-heavy` article families as explicit
Transformer-backed reference-linear receipts inside the eval-owned report
instead of silently treating the narrower demo set as sufficient. The same
gate keeps host substitution, external oracle use, preprocessing shortcuts,
runtime-owned control flow, helper-module mediation, and cache-owned decisive
steps explicit as audit questions instead of collapsing them into a generic
"the outputs were correct" claim. `psionic-research` now mirrors the
operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_interpreter_ownership_gate_summary.json`,
and `psionic-provider` now projects the same verdict into
`crates/psionic-provider/src/tassadar_article_interpreter_ownership_gate.rs`.
This is the point where the repo now machine-readably says the decisive
interpreter behavior for the declared article envelope belongs to the owned
Transformer forward-pass route itself rather than a hidden host-side control
plane.

`TAS-184A` now closes the next KV-cache and activation-state discipline tranche
on top of the same route. `psionic-eval` now commits the joined audit artifact
at
`fixtures/tassadar/reports/tassadar_article_kv_activation_discipline_audit_report.json`,
which makes the route's same-run state carriers explicit with analytic
KV-growth accounting, constrained-history sensitivity rows, acceptable versus
non-acceptable carrier boundaries, and one declared `mixed` verdict. That
verdict is stronger than hand-waving "weights do everything" and stronger than
"the cache is secretly the interpreter": the bounded article route is now
machine-readably declared as weight-sensitive and route-owned, while still
depending on request-local KV and activation state to carry same-run execution
history inside the admitted forward pass. `psionic-research` now mirrors the
operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_kv_activation_discipline_audit_summary.json`,
and `psionic-provider` now projects the same verdict into
`crates/psionic-provider/src/tassadar_article_kv_activation_discipline_audit.rs`.
Persisted or resumed cache state, hidden activation replay, and undeclared
history channels remain outside the admitted route. Cross-machine
reproducibility, route minimality, and final article-equivalence tranches
still remain open.

`TAS-185` now closes the next declared-machine reproducibility tranche on top
of the same route. `psionic-eval` now commits the joined matrix artifact at
`fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_matrix_report.json`,
which ties the selected `HullCache` fast route to the declared
`host_cpu_x86_64` and `host_cpu_aarch64` machine classes, the canonical
Hungarian and Sudoku demo evidence, the deterministic single-run long-horizon
closure, and the zero-drift throughput-floor policy already frozen for the
bounded article lane. `psionic-research` now mirrors the operator-readable
summary at
`fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_matrix_summary.json`,
`psionic-provider` now projects the same verdict into
`crates/psionic-provider/src/tassadar_article_cross_machine_reproducibility_matrix.rs`,
and `psionic-serve` now freezes the bounded served portability statement at
`fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_publication.json`.
The canonical fast route is now machine-readably reproducible on the declared
CPU machine envelope, while stochastic execution remains explicitly out of
scope for this article route and route-minimality plus the final claim checker
still remain open.

`TAS-185A` now closes that next route-minimality tranche on top of the same
route. `psionic-eval` now commits the joined audit artifact at
`fixtures/tassadar/reports/tassadar_article_route_minimality_audit_report.json`,
which fixes the canonical public claim route to the direct `HullCache` runtime
path on the canonical article model and excludes checkpoint restore, spill or
tape continuation, hidden helper mediation, planner-owned indirection, and
hybrid orchestration from that claim route. `psionic-research` now mirrors the
operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_route_minimality_audit_summary.json`,
`psionic-provider` now projects the same verdict into
`crates/psionic-provider/src/tassadar_article_route_minimality_audit.rs`, and
`psionic-serve` now freezes the explicit publication verdict split at
`fixtures/tassadar/reports/tassadar_article_route_minimality_publication_verdict.json`.
The bounded article route is now machine-readably minimal and operator-green,
and `TAS-186` now closes the final bounded article-equivalence verdict on top
of that same route. `psionic-eval` now commits the claim checker at
`fixtures/tassadar/reports/tassadar_article_equivalence_claim_checker_report.json`
plus the final audit at
`fixtures/tassadar/reports/tassadar_article_equivalence_final_audit_report.json`,
`psionic-research` now mirrors the operator-readable summary at
`fixtures/tassadar/reports/tassadar_article_equivalence_final_audit_summary.json`,
and the final checker script now lives at
`scripts/check-tassadar-article-equivalence-final-audit.sh`. The public repo
now has one bounded article-equivalence claim for the direct deterministic
`HullCache` route on the canonical article model and declared CPU machine
matrix, while arbitrary C ingress, arbitrary Wasm ingress, resumed execution,
stochastic execution, planner/hybrid canonical routes, generic
interpreter-in-weights claims outside the article envelope, and the post-article
universality bridge remain explicitly out of scope.

`TAS-187` now freezes the post-article universality bridge contract at
`fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_report.json`.
That bridge binds the canonical post-`TAS-186` article model, weight artifact,
and direct `HullCache` route identity to the preserved historical `TCM.v1`
runtime contract without rewriting the older closure artifacts. It keeps the
direct article-equivalent carrier distinct from the bounded resumable
universality carrier, leaves the later capability plane explicitly reserved,
and publishes the operator-readable summary at
`fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json`,
the provider receipt at
`crates/psionic-provider/src/tassadar_post_article_universality_bridge_contract.rs`,
and the dedicated checker at
`scripts/check-tassadar-post-article-universality-bridge-contract.sh`. This
bridge still is not a semantic-preservation proof, a control-plane provenance
proof, a served/public universality approval, a weighted plugin-control claim,
or an arbitrary software-capability claim.

`TAS-188` now closes the semantic-preservation and state-ownership tranche at
`fixtures/tassadar/reports/tassadar_post_article_canonical_route_semantic_preservation_audit_report.json`.
That audit binds the bridge machine identity to the current session-process,
spill/tape, and installed-process continuation evidence, freezes one declared
weights-owned versus ephemeral versus resumed versus durable state taxonomy,
and states machine-readably that host may execute declared continuation
mechanics but may not own workflow semantics. The operator-readable summary now
lives at
`fixtures/tassadar/reports/tassadar_post_article_canonical_route_semantic_preservation_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_canonical_route_semantic_preservation.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-canonical-route-semantic-preservation-audit.sh`.
This tranche still is not the full control-plane decision-provenance proof,
the final direct-versus-resumable carrier split publication, a rebased
Turing-completeness approval, a served/public universality approval, a
weighted plugin-control claim, or an arbitrary software-capability claim.

`TAS-188A` now closes the control-plane ownership and decision-provenance
tranche at
`fixtures/tassadar/reports/tassadar_post_article_control_plane_decision_provenance_proof_report.json`.
That proof binds branch, retry, and stop decisions to model outputs, canonical
route identity, and the bridge machine identity tuple while freezing one
machine-readable determinism contract, equivalent-choice relation,
failure-semantics lattice, time semantics contract, information boundary,
training-versus-inference boundary, hidden-state closure rule, observer model,
and hidden-control-channel review. The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_control_plane_decision_provenance_proof_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_control_plane_decision_provenance_proof.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-control-plane-decision-provenance-proof.sh`.
This tranche still is not the final direct-versus-resumable carrier split
publication, a rebased Turing-completeness approval, a served/public
universality approval, a weighted plugin-control claim, or an arbitrary
software-capability claim.

`TAS-189` now closes the carrier-split publication tranche at
`fixtures/tassadar/reports/tassadar_post_article_carrier_split_contract_report.json`.
That contract binds the direct article-equivalent truths and the resumable
bounded-universality truths to different carriers on the same bridge machine
identity, records that transfer by implication is blocked in both directions,
and keeps the reserved capability plane explicit instead of leaving later
capability widening implicit. The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_carrier_split_contract_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_carrier_split_contract.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-carrier-split-contract.sh`.
This tranche still does not rebind the universal-machine proof, reissue the
universality witness suite, publish the rebased verdict split, admit
served/public universality, admit weighted plugin control, or admit arbitrary
software capability.

`TAS-190` now closes the universal-machine proof-rebinding tranche at
`fixtures/tassadar/reports/tassadar_post_article_universal_machine_proof_rebinding_report.json`.
That report rebinds the historical universal-machine proof onto the bridge
machine identity, canonical model artifact, canonical weight artifact, and
canonical route id through one explicit proof-transport boundary instead of
treating rebinding as metadata relabeling. The operator-readable summary now
lives at
`fixtures/tassadar/reports/tassadar_post_article_universal_machine_proof_rebinding_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_universal_machine_proof_rebinding.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-universal-machine-proof-rebinding.sh`.
This tranche still does not reissue the broader universality witness suite,
enable the canonical-route universal-substrate gate, publish the rebased
theory/operator/served verdict split, admit served/public universality, admit
weighted plugin control, or admit arbitrary software capability.

`TAS-191` now closes the universality witness-suite reissue tranche at
`fixtures/tassadar/reports/tassadar_post_article_universality_witness_suite_reissue_report.json`.
That report reissues the older seven-row witness suite onto the bridge machine
identity, canonical model artifact, and canonical route id while keeping the
same exact and refusal-boundary families explicit and keeping helper
substitution, hidden cache-owned control flow, and resume-only cheating as
explicit negative rows. The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_universality_witness_suite_reissue_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_universality_witness_suite_reissue.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-universality-witness-suite-reissue.sh`.
This tranche still does not enable the canonical-route universal-substrate
gate, publish the rebased theory/operator/served verdict split, admit
served/public universality, admit weighted plugin control, or admit arbitrary
software capability.

`TAS-192` now closes the canonical-route universal-substrate gate tranche at
`fixtures/tassadar/reports/tassadar_post_article_canonical_route_universal_substrate_gate_report.json`.
That report joins the historical minimal universal-substrate gate,
article-equivalence closure, the bridge contract, semantic-preservation audit,
control-plane proof, carrier split, proof rebinding, and witness-suite reissue
into one machine-readable gate bound to the bridge machine identity while
keeping portability, refusal truth, helper substitution, route drift,
continuation abuse, semantic drift, and overclaim posture explicit. The
operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_canonical_route_universal_substrate_gate_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_canonical_route_universal_substrate_gate.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-canonical-route-universal-substrate-gate.sh`.
This tranche still does not publish the rebased theory/operator/served verdict
split, admit served/public universality, admit weighted plugin control, or
admit arbitrary software capability.

`TAS-193` now closes the post-article universality portability/minimality
matrix tranche at
`fixtures/tassadar/reports/tassadar_post_article_universality_portability_minimality_matrix_report.json`.
That report extends the rebased canonical-route universality lane across one
declared CPU machine matrix, one explicit route-carrier classification, and
one machine-level minimality contract instead of inheriting those claims from
adjacent green artifacts. The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_universality_portability_minimality_matrix_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_universality_portability_minimality_matrix.rs`,
the served conformance envelope now lives at
`fixtures/tassadar/reports/tassadar_post_article_universality_served_conformance_envelope.json`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-universality-portability-minimality-matrix.sh`.
This tranche still does not publish the rebased theory/operator/served verdict
split, admit served/public universality, admit weighted plugin control, or
admit arbitrary software capability.

`TAS-194` now closes the rebased theory/operator/served verdict-split tranche
at
`fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_split_report.json`.
That report reissues the older universality verdict split on the canonical
bridge machine identity, canonical route, canonical weight artifact, and
canonical portability/minimality envelope instead of inheriting the older split
by implication. The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_split_summary.json`,
the served publication now lives at
`fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_publication.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_rebased_universality_verdict_split.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-rebased-universality-verdict-split.sh`.
This tranche turns the rebased theory/operator claim on, but it still does not
admit weighted plugin control, served/public universality, or arbitrary
software capability.

`TAS-195` now closes the plugin-aware boundary tranche at
`fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_report.json`.
That report keeps `TCM.v1` as the bounded compute substrate below the canonical
owned route, binds the reserved plugin capability plane to the same bridge
machine identity, and states machine-readably that plugin execution is a
separate software-capability layer rather than an implicit extension of
compute truth. The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_plugin_capability_boundary.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-plugin-capability-boundary.sh`.
This tranche keeps plugin packet/state/receipt identity separate from the core
compute substrate, reserves choice-set integrity/resource transparency/
scheduling ownership plus the first closed-world operator-curated plugin
tranche, and still does not admit weighted plugin control, plugin publication,
served/public universality, or arbitrary software capability.

`TAS-196` now closes the post-article Turing-completeness closeout tranche at
`fixtures/tassadar/reports/tassadar_post_article_turing_completeness_closeout_audit_report.json`.
That report keeps the historical `TAS-156` closeout standing, states that the
canonical post-`TAS-186` route is now the truth carrier for the bounded
Turing-completeness claim, and makes control-plane ownership plus
decision-provenance proof part of that truth carrier instead of treating them
as adjacent but optional evidence. The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_turing_completeness_closeout_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_turing_completeness_closeout.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-turing-completeness-closeout-audit.sh`.
This tranche keeps the final canonical machine closure bundle separate for
`TAS-215`, and it still does not admit weighted plugin control, plugin
publication, served/public universality, or arbitrary software capability.

`TAS-197` now closes the plugin charter, authority boundary, and platform-law
tranche at
`fixtures/tassadar/reports/tassadar_post_article_plugin_charter_authority_boundary_report.json`.
That report freezes the plugin charter above the rebased carrier by inheriting
one canonical machine identity and computational-model statement from the
bridge, inheriting the pre-plugin control-plane proof as a hard dependency,
freezing the proof-versus-audit and observer-model boundary, fixing the
data/control/capability plane split and the packet-local/instance-local/
host-backed/weights-owned state split, and freezing semantic-preservation,
anti-interpreter-smuggling, and downward-non-influence laws without widening
the current claim surface. The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_plugin_charter_authority_boundary_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_plugin_charter_authority_boundary.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-plugin-charter-authority-boundary.sh`.
This tranche keeps current posture operator/internal-only, requires typed
governance receipts for any later envelope or publication widening, and still
does not admit weighted plugin control, plugin publication, served/public
universality, or arbitrary software capability.

## Route Requirement

Any canonical article-equivalence Transformer route must use this boundary.
Later issues may extend the implementations inside these owners, but they must
not redefine the ownership split or route around it.

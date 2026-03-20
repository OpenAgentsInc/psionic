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

## Route Requirement

Any canonical article-equivalence Transformer route must use this boundary.
Later issues may extend the implementations inside these owners, but they must
not redefine the ownership split or route around it.

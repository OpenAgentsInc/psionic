# Tassadar Wasm Runbook

> Status: canonical bounded-Wasm operator runbook, added 2026-03-18 after
> reviewing `README.md`, `docs/ARCHITECTURE.md`,
> `docs/ROADMAP_TASSADAR.md`, `docs/ROADMAP_TASSADAR_INDEX.md`, and the
> current committed Tassadar Wasm reports.

## Purpose

This runbook is the operator entrypoint for the current bounded Tassadar Wasm
flow.

It exists to answer a narrow question:

> from a clean checkout, what should I run to verify the real bounded Wasm lane
> and how should I interpret success, refusal, and drift?

If the narrower question is "what one command reproduces the full current
Rust-only article path end to end?", use
`docs/TASSADAR_RUST_ONLY_ARTICLE_RUNBOOK.md` instead.

## Scope

This runbook covers the current repo-owned bounded Wasm flow only:

- Rust-only article frontend canon
- Rust-to-Wasm article profile completeness matrix
- bounded Rust-only article ABI closure
- checkpointed multi-slice execution receipts
- linked module-graph and start-order runtime-support receipts
- deterministic-import and runtime-support subset promotion gate
- portable broad-family machine matrix and backend/toolchain envelopes
- memory ABI v2 bulk-memory exactness
- bounded dynamic-memory pause-and-resume receipts
- Rust-only article runtime closeout
- canonical Rust-only Hungarian-10x10 article reproducer
- canonical C-to-Wasm compile receipt
- source-to-Wasm-to-Tassadar compile-pipeline matrix
- frozen core-Wasm window declaration and official harness
- frozen core-Wasm semantic parity and closure gate
- bounded scalar-f32 semantics, NaN policy, and comparison matrix
- frozen mixed-numeric profile ladder
- numeric portability envelopes across backend and toolchain families
- float-enabled profile publication and route gate
- normalized Wasm-module ingress
- differential Wasm conformance against `wasmi`
- module-scale Wasm workload suite
- trap and exception parity

It does not claim:

- arbitrary Wasm closure
- general parameter-ABI closure
- generic resumable computation beyond the committed checkpointed workloads
- broad host-import closure
- broad C/C++ frontend closure

## Relation To The WebAssembly Spec

This runbook validates the current bounded Tassadar Wasm profile only.

Public claim discipline for this lane is:

- "supports Wasm" should mean a named Tassadar profile inside a frozen
  WebAssembly spec window
- the current bounded lane uses `wasmi` differential checks on its admitted
  subset, not a claim of general WebAssembly closure
- any future broad or full-Wasm closure should be tied to the official core
  tests and reference authority for the declared window
- the declared frozen-window artifact for the current target is
  `fixtures/tassadar/reports/tassadar_frozen_core_wasm_window_report.json`
- the declared closure-verdict artifact for the current target is
  `fixtures/tassadar/reports/tassadar_frozen_core_wasm_closure_gate_report.json`
- the current closure verdict is `not_closed`, so the frozen-window declaration
  must not be read as full core-Wasm closure
- the current bounded float-widening artifact is
  `fixtures/tassadar/reports/tassadar_float_semantics_comparison_matrix_report.json`
- that float artifact is scalar-`f32`, CPU-reference-only, and refusal-first
  outside its declared regimes; it is not generic Wasm float closure
- the current mixed-numeric widening artifact is
  `fixtures/tassadar/reports/tassadar_mixed_numeric_profile_ladder_report.json`
- that mixed-numeric artifact stages exact scalar-`f32`, exact mixed
  `i32`/`f32`, and bounded-approximate `f64 -> f32` conversion profiles; it
  is not generic Wasm numeric closure or full `f64` exactness
- the current numeric-portability artifact is
  `fixtures/tassadar/reports/tassadar_numeric_portability_report.json`
- that numeric-portability artifact freezes backend-, toolchain-, and
  machine-class envelopes for the bounded float and mixed-numeric lanes; it is
  not backend-invariant float exactness or broad served numeric publication
- the current float-profile gate artifacts are
  `fixtures/tassadar/reports/tassadar_float_profile_acceptance_gate_report.json`
  and `fixtures/tassadar/reports/tassadar_float_profile_route_policy_report.json`
- those gate artifacts allow exact cpu-reference numeric profiles to be named
  and routed in bounded profile-specific form only; they do not create a new
  default served float lane
- imports, effects, and host capability policy remain separate embedding
  contracts, not part of the bounded core language claim

## Preconditions

Start from a clean checkout:

```bash
git status --short --branch
```

Required local baseline:

- Rust toolchain that can build the workspace
- normal `cargo run` and `cargo test` access

Optional local baseline:

- `clang` with a working `wasm32-unknown-unknown` target if you want the
  historical C-source compile receipt to succeed locally

If that target is missing, the compile-receipt step should refuse with a typed
`toolchain_failure`. That is a local toolchain prerequisite failure, not proof
that the bounded Wasm lane regressed.

## Canonical Flow

### 1. Rust-only article frontend canon

```bash
cargo run -p psionic-eval --example tassadar_rust_source_canon_report
```

Read:

- `fixtures/tassadar/reports/tassadar_rust_source_canon_report.json`

Expected outcome:

- all canonical frontend cases are rooted in committed `.rs` fixtures
- the multi-export, memory-lookup, parameter-ABI, micro-kernel, heap-input,
  long-loop, Hungarian, and Sudoku Rust fixtures all compile successfully to
  committed Wasm outputs with stable source/toolchain/config/output lineage
- the `param_abi_fixture` and `heap_sum_article` source-canon rows now also
  freeze the admitted generalized-ABI exports `pair_add`, `dot_i32`, and
  `sum_and_max_into_buffer` instead of leaving the widened entrypoint surface
  implicit
- this report is the article-closure frontend anchor; it does not by itself
  imply arbitrary Rust closure or arbitrary Wasm lowering

### 2. Rust-to-Wasm article profile completeness matrix

```bash
cargo run -p psionic-eval --example tassadar_rust_article_profile_completeness_report
```

Read:

- `fixtures/tassadar/reports/tassadar_rust_article_profile_completeness_report.json`

Expected outcome:

- one machine-readable supported/refused matrix over module shape, control
  flow, tables/globals/indirect calls, numeric families, and ABI shape
- the current Rust-only article family is explicit as a bounded i32-first
  profile rather than inferred from several narrower artifacts
- the same profile boundary is what the Tassadar environment bundle and served
  capability publication now cite

### 3. Bounded Rust-only article ABI closure

```bash
cargo run -p psionic-eval --example tassadar_article_abi_closure_report
```

Read:

- `fixtures/tassadar/reports/tassadar_article_abi_closure_report.json`

Expected outcome:

- one machine-readable exact/refused report over the committed direct
  `param_abi_fixture` and `heap_sum_article` Rust sources
- exact direct scalar `i32 -> i32` entrypoint closure for `add_one`
- exact direct pointer-plus-length heap-input closure for `heap_sum_i32`,
  including a non-zero pointer offset case
- explicit refusal on floating-point params, multi-result returns, and
  out-of-range heap inputs
- this is the current honest direct-ABI closure lane; it does not imply that
  the generic Wasm-text or generic normalized-module lowering path now admits
  arbitrary parameterized exports

### 3A. Generalized ABI family

```bash
cargo run -p psionic-eval --example tassadar_generalized_abi_family_report
```

Read:

- `fixtures/tassadar/reports/tassadar_generalized_abi_family_report.json`

Expected outcome:

- one machine-readable exact/refused report over the widened generalized
  generalized ABI family
- exact multi-param scalar `i32, i32 -> i32` closure for `pair_add`
- exact multi-param scalar `i64, i64 -> i64` closure for `pair_add_i64`
- exact multiple pointer-length input closure for `dot_i32`
- exact homogeneous two-value `i32, i32 -> (i32, i32)` closure for
  `pair_sum_and_diff`
- exact caller-owned result-code-plus-output-buffer closure for
  `sum_and_max_into_buffer`
- exact caller-owned `i64` result-code-plus-output-buffer closure for
  `sum_and_max_i64_into_buffer`
- exact bounded multi-export program-shape closure for `pair_sum` and
  `local_double`
- explicit refusal on short output buffers, aliased output buffers,
  unaligned `i64` buffer layouts, floating-point params, mixed-width
  multi-result returns, host-handle callbacks, and callee-allocated returned
  buffers
- this is a benchmarked broader ABI family, not the current promoted
  article-closeout claim; the served internal-compute claim remains
  `tassadar.internal_compute.article_closeout.v1` even though the ladder now
  records both `tassadar.internal_compute.generalized_abi.v1` and
  `tassadar.internal_compute.wider_numeric_data_layout.v1` as implemented but
  still non-promoted

### 3B. Checkpointed multi-slice execution receipts

```bash
cargo run -p psionic-eval --example tassadar_execution_checkpoint_report
```

Read:

- `fixtures/tassadar/reports/tassadar_execution_checkpoint_report.json`
- `fixtures/tassadar/runs/tassadar_execution_checkpoint_v1/tassadar_execution_checkpoint_bundle.json`

Expected outcome:

- one machine-readable report and run root now bind the deterministic
  checkpointed multi-slice execution lane
- exact fresh-versus-resumed parity is explicit for the seeded
  `long_loop_kernel`, `state_machine_accumulator`, and
  `search_frontier_kernel` workloads
- persisted continuation artifacts now exist as checkpoint JSON plus
  datastream-manifest pairs under
  `fixtures/tassadar/runs/tassadar_execution_checkpoint_v1`
- resume safety is explicit: superseded checkpoints, oversized state,
  profile mismatches, and effect-state mismatches refuse with typed reasons
- this is benchmarked effective-unboundedness scaffolding only; it does not
  widen the current served claim beyond
  `tassadar.internal_compute.article_closeout.v1`

### 3B.1. Call-frame and memory exact checkpoint/resume promotion

```bash
cargo run -p psionic-eval --example tassadar_resumable_multi_slice_promotion_report
```

Read:

- `fixtures/tassadar/reports/tassadar_resumable_multi_slice_promotion_report.json`
- `fixtures/tassadar/runs/tassadar_call_frame_resume_v1/tassadar_call_frame_resume_bundle.json`

Expected outcome:

- one machine-readable joined promotion report now binds the call-frame
  resume lane to the earlier execution-checkpoint and dynamic-memory resume
  lanes
- exact fresh-versus-resumed parity is explicit for seeded recursive and
  multi-function call-heavy programs
- persisted continuation artifacts now exist as checkpoint JSON plus
  datastream-manifest pairs under
  `fixtures/tassadar/runs/tassadar_call_frame_resume_v1`
- resume safety is explicit: profile mismatches, stale checkpoints, and
  oversized state refuse with typed reasons
- the named profile `tassadar.internal_compute.resumable_multi_slice.v1` is
  now benchmarked and implemented in the internal-compute ladder, but it
  remains non-portable and non-promoted until broader portability evidence
  exists
- this does not widen the current served claim beyond
  `tassadar.internal_compute.article_closeout.v1`

### 3B.2. Deterministic import/effect taxonomy, receipts, and replay limits

```bash
cargo run -p psionic-router --example tassadar_effect_route_policy_report
cargo run -p psionic-eval --example tassadar_effect_taxonomy_report
```

Read:

- `fixtures/tassadar/reports/tassadar_effect_route_policy_report.json`
- `fixtures/tassadar/reports/tassadar_effect_taxonomy_report.json`

Expected outcome:

- one router-owned route-policy report now widens the old narrow import matrix
  into five explicit route kinds: internal exact, host-state snapshot-bound,
  sandbox delegation, receipt-bound input, and refused
- one eval-owned taxonomy report now exercises that widened surface with both
  admitted and refused negotiated cases
- deterministic internal stubs still stay internal-only
- durable host-backed state is now a bounded admitted lane, but only with
  explicit snapshot plus durable-state receipt evidence and only within the
  declared replay window
- sandbox delegation remains explicit delegation with challengeable evidence
  instead of being silently rebranded as internal exact compute
- nondeterministic relay input remains admissible only with an explicit input
  receipt window
- unsafe side effects remain typed refusals
- this closes effect taxonomy and replay-limit publication only; it does not
  grant general side-effect authority and it does not collapse host-backed or
  delegated behavior into the internal exact-compute claim

### 3B.3. Deterministic import-mediated continuation and effect-safe resume

```bash
cargo run -p psionic-eval --example tassadar_effect_safe_resume_report
```

Read:

- `fixtures/tassadar/reports/tassadar_effect_safe_resume_report.json`
- `fixtures/tassadar/runs/tassadar_effect_safe_resume_v1/tassadar_effect_safe_resume_bundle.json`

Expected outcome:

- one machine-readable joined report now promotes the resumable base lane only
  into the deterministic import subset
- exact resumable continuation is explicit only for deterministic internal stub
  receipts over the committed recursive and multi-function seeded checkpoints
- host-backed state, sandbox delegation, receipt-bound nondeterministic input,
  and unsafe side effects remain typed refusals for this target profile even
  when the broader effect taxonomy can classify them
- the sandbox effect boundary now makes continuation-safe versus
  continuation-refused effect refs machine-legible
- the named profile `tassadar.internal_compute.deterministic_import_subset.v1`
  is now benchmarked and implemented in the internal-compute ladder, but it
  remains non-portable and non-promoted until broader portability evidence
  exists
- this does not widen the current served claim beyond
  `tassadar.internal_compute.article_closeout.v1`

### 3B.4. Linked module graphs, start semantics, and bounded runtime support

```bash
cargo run -p psionic-runtime --example tassadar_linked_program_bundle_runtime_report
cargo run -p psionic-eval --example tassadar_linked_program_bundle_eval_report
cargo run -p psionic-research --example tassadar_linked_program_bundle_summary
```

Read:

- `fixtures/tassadar/reports/tassadar_linked_program_bundle_runtime_report.json`
- `fixtures/tassadar/reports/tassadar_linked_program_bundle_eval_report.json`
- `fixtures/tassadar/reports/tassadar_linked_program_bundle_summary.json`

Expected outcome:

- one runtime-owned report now freezes explicit bundle graph edges,
  helper-module lineage, start-order semantics, and typed refusal for
  unsupported shared-state cycles
- the admitted exact and rollback cases now keep helper lineage, graph-shape
  validity, and start-order replay parity machine-legible instead of leaving
  bundle topology implicit
- the research summary now names which bundles are graph-valid and start-safe
  under the current bounded runtime-support subset
- the named profile `tassadar.internal_compute.runtime_support_subset.v1` is
  now benchmarked and implemented in the internal-compute ladder, but it
  remains non-portable and non-promoted until broader portability and public
  publication gates go green
- this widens real linked-program behavior only inside the explicit bounded
  bundle family; it does not imply arbitrary linked software growth, general
  import/runtime-support closure, or a widened current served claim beyond
  `tassadar.internal_compute.article_closeout.v1`

### 3B.5. Deterministic-import and runtime-support subset promotion gate

```bash
cargo run -p psionic-eval --example tassadar_subset_profile_promotion_gate_report
```

Read:

- `fixtures/tassadar/reports/tassadar_subset_profile_promotion_gate_report.json`

Expected outcome:

- one machine-readable gate now names the two bounded post-article subset
  profiles that are evidence-complete enough to publish honestly as named,
  suppressed profiles:
  `tassadar.internal_compute.deterministic_import_subset.v1` and
  `tassadar.internal_compute.runtime_support_subset.v1`
- the gate is green only when deterministic-import replay proofs, linked-bundle
  replay proofs, negative refusal coverage, route suppression, and
  profile-specific world-mount plus accepted-outcome policy posture are all
  explicit
- the gate still records zero
  `served_publication_allowed_profile_ids`, which is the point: these profiles
  are safe to name publicly without being safe to widen into the current served
  lane
- `world-mounts`, `kernel-policy`, `nexus`, and `compute-market` remain
  explicit external dependency markers for any future profile-specific
  publication widening outside standalone `psionic`
- this does not widen the current served claim beyond
  `tassadar.internal_compute.article_closeout.v1`

### 3B.6. Portable broad-family machine matrix and backend/toolchain envelopes

```bash
cargo run -p psionic-eval --example tassadar_broad_internal_compute_portability_report
```

Read:

- `fixtures/tassadar/reports/tassadar_broad_internal_compute_portability_report.json`

Expected outcome:

- one machine-readable portability matrix now freezes broader internal-compute
  rows across declared backend families, stable toolchain families, and machine
  classes instead of collapsing portability into one current-host CPU fact
- the current explicit backend-family set is `cpu_reference`,
  `metal_served`, and `cuda_served`
- the current explicit toolchain-family set is
  `rustc:wasm32-unknown-unknown`,
  `rustc:wasm32-unknown-unknown+metal_served`, and
  `rustc:wasm32-unknown-unknown+cuda_served`
- `tassadar.internal_compute.article_closeout.v1` remains publication-allowed
  only on the CPU-reference envelope; the corresponding `metal_served` and
  `cuda_served` rows now stay explicit as
  `suppressed_backend_envelope_constrained` rather than being silently treated
  as portable
- rows outside the declared backend envelope or outside the declared profile
  envelope remain explicit `suppressed_drifted_outside_envelope`
- the same backend/toolchain family ids are now carried through served
  capability publication, provider receipts, and Tassadar environment metadata
  so operator review and publication review are looking at the same portability
  envelope
- this still does not widen the current served claim beyond
  `tassadar.internal_compute.article_closeout.v1`; it makes portability
  suppression more explicit instead of making broader publication greener

### 3B.7. Public broad-family route, mount, and accepted-outcome promotion

```bash
cargo run -p psionic-eval --example tassadar_broad_internal_compute_profile_publication_report
cargo run -p psionic-router --example tassadar_broad_internal_compute_route_policy_report
cargo run -p psionic-eval --example tassadar_subset_profile_promotion_gate_report
```

Read:

- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json`
- `fixtures/tassadar/reports/tassadar_subset_profile_promotion_gate_report.json`

Expected outcome:

- the current default served profile still stays
  `tassadar.internal_compute.article_closeout.v1`
- two broader named profiles are now explicit “publicly nameable but
  profile-specific” lanes instead of being flattened into generic suppression:
  `tassadar.internal_compute.deterministic_import_subset.v1` and
  `tassadar.internal_compute.runtime_support_subset.v1`
- in the publication report, those two profiles now carry
  `profile_specific_mount_template_available` and
  `profile_specific_accepted_outcome_template_available` instead of the older
  generic “needs policy later” posture
- in the route-policy report, those same profiles now carry
  `promoted_profile_specific`, which means they are explicitly public route
  targets under named profile policy without becoming the default served exact
  lane
- the subset gate remains green for those two profiles and still records zero
  `served_publication_allowed_profile_ids`
- `world-mounts`, `kernel-policy`, `nexus`, and `compute-market` remain
  explicit dependency markers for canonical task-scoped mounts, accepted
  outcomes, and market-wide broad-profile exposure outside standalone
  `psionic`
- this is bounded promotion discipline only; it does not make those subset
  profiles generic served exact compute and does not widen the current default
  served profile beyond `tassadar.internal_compute.article_closeout.v1`

### 3C. Hungarian-10x10 Rust-only article reproducer

```bash
cargo run -p psionic-research --example tassadar_hungarian_10x10_article_reproducer
```

Read:

- `fixtures/tassadar/runs/hungarian_10x10_article_reproducer_v1/reproducer_bundle.json`
- `fixtures/tassadar/reports/tassadar_hungarian_10x10_article_reproducer_report.json`

Expected outcome:

- one canonical Rust-only Hungarian-10x10 article root exists for
  `hungarian_10x10_test_a`
- the root binds the committed Rust source canon receipt to exact compiled
  deployment artifacts, readable log, compact token trace, compile evidence,
  runtime execution proof, and runtime trace proof
- the report keeps direct execution posture explicit:
  `reference_linear -> reference_linear`, `fallback_observed=false`, and
  `external_tool_surface_observed=false`
- this closes one concrete matching workload only; it does not yet imply
  hard-Sudoku or multi-million-step article closure

### 3D. Sudoku-9x9 Rust-only article reproducer

```bash
cargo run -p psionic-research --example tassadar_sudoku_9x9_article_reproducer
```

Read:

- `fixtures/tassadar/reports/tassadar_sudoku_9x9_article_reproducer_report.json`
- `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0/deployments/sudoku_9x9_test_a/token_trace_summary.json`
- `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0/deployments/sudoku_9x9_test_a/readable_log.txt`

Expected outcome:

- one canonical Rust-only Sudoku-9x9 article report binds the committed source
  canon receipt to the exact compiled `sudoku_9x9_test_a` search deployment
- the report freezes the committed 9x9 corpus case set, the canonical search
  trace artifacts, and the direct execution posture:
  `reference_linear -> reference_linear`, `fallback_observed=false`, and
  `external_tool_surface_observed=false`
- this closes one concrete backtracking-search workload family only; it does
  not yet imply Hungarian or multi-million-step article closure

### 3E. Rust-only article runtime closeout

```bash
cargo run -p psionic-eval --example tassadar_article_runtime_closeout_report
cargo run -p psionic-research --example tassadar_article_runtime_closeout_summary
```

Read:

- `fixtures/tassadar/runs/article_runtime_closeout_v1/article_runtime_closeout_bundle.json`
- `fixtures/tassadar/reports/tassadar_article_runtime_closeout_report.json`
- `fixtures/tassadar/reports/tassadar_article_runtime_closeout_summary.json`

Expected outcome:

- exact direct-reference-linear CPU execution for the committed
  `rust.long_loop_kernel` and `rust.state_machine_kernel` families at the
  declared `million_step` and `two_million_step` horizons
- explicit throughput-floor passes on all four committed horizon receipts
- explicit `HullCache` and `SparseTopK` fallback-only rows on those same
  backward-branch kernels instead of implied fast-path closure
- one served publication path now exists for this closeout summary, but it is
  benchmark-only and does not widen the generic served Wasm capability matrix

### 3F. Direct model-weight execution proof

```bash
cargo run -p psionic-serve --example tassadar_direct_model_weight_execution_proof_report
```

Read:

- `fixtures/tassadar/reports/tassadar_direct_model_weight_execution_proof_report.json`

Expected outcome:

- one operator-facing proof report freezes the current direct-execution claim
  over exactly three canonical article workloads:
  `long_loop_kernel`, `sudoku_v0_test_a`, and `hungarian_matching`
- each receipt in the report binds the article session to one
  direct-guaranteed planner route, one requested and effective decode mode,
  one model-weight bundle digest, one proof bundle, and one runtime-manifest
  lineage chain
- the report makes `fallback_observed=false`, `external_call_count=0`, and
  `cpu_result_substitution_observed=false` explicit instead of treating a
  completed session as sufficient evidence
- this closes the current "inside the model weights" claim only for the named
  workloads and only on the exact route digest in the report

### 4. Optional historical C-to-Wasm compile receipt

```bash
cargo run -p psionic-runtime --example tassadar_c_to_wasm_compile_receipt
```

Read:

- `fixtures/tassadar/reports/tassadar_c_to_wasm_compile_receipt.json`

Expected outcome:

- either a successful canonical lineage receipt over
  `fixtures/tassadar/sources/tassadar_micro_wasm_kernel.c`
- or a typed refusal if the local `clang` cannot target
  `wasm32-unknown-unknown`
- this step is not required for the Rust-only article-closure path

### 5. Compile-pipeline matrix

```bash
cargo run -p psionic-eval --example tassadar_compile_pipeline_matrix_report
```

Read:

- `fixtures/tassadar/reports/tassadar_compile_pipeline_matrix_report.json`

Expected outcome:

- exact lowering for the multi-export arithmetic and memory-lookup WAT cases
- explicit `unsupported_param_count` refusal for the parameter-ABI WAT case
- explicit toolchain refusal for the C-source path when the local toolchain is
  unavailable or incomplete
- that parameter-ABI refusal is still expected in this generic lowering path;
  the committed direct ABI closure now lives in the separate bounded
  Rust-only article ABI report above

### 5A. Memory ABI v2 bulk-memory exactness

```bash
cargo run -p psionic-eval --example tassadar_memory_abi_v2_report
```

Read:

- `fixtures/tassadar/reports/tassadar_memory_abi_v2_report.json`

Expected outcome:

- exact width-parity, sign-extension, `memory.size`, and `memory.grow` rows
- one explicit `copy_fill_exactness` row now freezes exact `memory.copy` and
  `memory.fill` behavior on the public byte-addressed lane
- the report still covers only bounded straight-line linear-memory programs; it
  does not by itself imply full module closure

### 5B. Dynamic-memory pause-and-resume receipts

```bash
cargo run -p psionic-eval --example tassadar_dynamic_memory_resume_report
```

Read:

- `fixtures/tassadar/reports/tassadar_dynamic_memory_resume_report.json`
- `fixtures/tassadar/runs/tassadar_dynamic_memory_resume_v1/tassadar_dynamic_memory_resume_bundle.json`

Expected outcome:

- one machine-readable report and run root now bind the bounded
  dynamic-memory pause-and-resume lane
- exact fresh-versus-resumed parity is explicit for the seeded
  `copy_fill_pause_after_copy` case
- persisted checkpoint artifacts now exist as checkpoint JSON plus
  datastream-manifest pairs under
  `fixtures/tassadar/runs/tassadar_dynamic_memory_resume_v1`
- this closes resumed-memory parity only for the committed bounded linear
  memory lane; it does not imply arbitrary Wasm checkpointing or broad served
  promotion

### 6. Wasm-module ingress

```bash
cargo run -p psionic-eval --example tassadar_wasm_module_ingress_report
```

Read:

- `fixtures/tassadar/reports/tassadar_wasm_module_ingress_report.json`

Expected outcome:

- the real committed Wasm binary parses, normalizes, and round-trips, but still
  refuses lowering because the exported function takes one parameter
- the seeded synthetic multi-function module lowers and executes exactly
- the widened module-execution lane now also has separate targeted test
  coverage for one admitted dynamic-memory module with active data segments,
  `memory.size`, `memory.grow`, `memory.copy`, and `memory.fill`, but that
  bounded module slice is still not equivalent to arbitrary Wasm

### 6B. Frozen core-Wasm window declaration and official harness

```bash
cargo run -p psionic-eval --example tassadar_frozen_core_wasm_window_report
```

Read:

- `fixtures/tassadar/reports/tassadar_frozen_core_wasm_window_report.json`

Expected outcome:

- one frozen int-first core-Wasm semantic window is declared explicitly
- the report binds the official text harness to the existing `.wat` fixtures
- the report binds the official binary harness to the existing compiled `.wasm`
  fixtures
- the report binds the official execution harness to the committed differential
  conformance report
- the report also freezes validator-backed out-of-window refusal rows for at
  least floating-point and multi-memory proposal families
- this sets the closure target and official harness only; it does not by itself
  claim full core-Wasm closure

### 6C. Frozen core-Wasm semantic parity and closure gate

```bash
cargo run -p psionic-eval --example tassadar_frozen_core_wasm_closure_gate_report
```

Read:

- `fixtures/tassadar/reports/tassadar_frozen_core_wasm_closure_gate_report.json`

Expected outcome:

- one machine-readable closure gate now joins the frozen-window declaration,
  differential conformance report, and trap-exception report into a single
  closure verdict
- `official_window_and_harness`, `differential_execution_parity`, and
  `trap_and_refusal_parity` should currently be green
- `target_feature_family_coverage` and `cross_machine_harness_replay` should
  currently remain red, which is why the honest current verdict is
  `not_closed`
- `served_publication_allowed` should currently be `false`
- this artifact controls interpretation: the frozen-window declaration defines
  the target, while the closure gate says whether the target is actually closed

### 6D. Bounded scalar-f32 semantics, NaN policy, and comparison matrix

```bash
cargo run -p psionic-eval --example tassadar_float_semantics_comparison_matrix_report
```

Read:

- `fixtures/tassadar/reports/tassadar_float_semantics_comparison_matrix_report.json`

Expected outcome:

- one machine-readable bounded float matrix now declares the scalar-`f32`
  policy for canonical quiet-NaN normalization, nearest-ties-to-even
  arithmetic, and ordered Wasm-style comparisons
- exact cases should currently cover finite arithmetic, NaN canonicalization,
  ordered finite comparisons, signed-zero equality, and NaN comparison posture
- explicit refusal cases should currently cover `f64_scalar` and
  `nan_payload_preservation`
- the policy should currently keep `cpu_reference` as the only supported
  backend family and keep `metal_served` plus `cuda_served` explicit as refused
  backend families
- this artifact widens bounded numeric semantics only; it does not make full
  Wasm float execution or served publication green

### 6E. Frozen mixed-numeric profile ladder

```bash
cargo run -p psionic-eval --example tassadar_mixed_numeric_profile_ladder_report
```

Read:

- `fixtures/tassadar/reports/tassadar_mixed_numeric_profile_ladder_report.json`

Expected outcome:

- one machine-readable ladder now stages numeric widening into exact
  scalar-`f32`, exact mixed `i32`/`f32`, and bounded-approximate
  `f64 -> f32` conversion profiles
- exact cases should currently cover in-range `i32 -> f32`,
  checked-truncation `f32 -> i32`, and one exact mixed scale-add case
- bounded-approximate cases should currently remain explicit rather than being
  flattened into exactness, especially for non-roundtripping `f64 -> f32`
  narrowing
- explicit refusal cases should currently cover non-exact `i32 -> f32`, NaN
  `f32 -> i32`, and out-of-range `f64 -> f32`
- this artifact widens one declared mixed-numeric family only; it does not
  make arbitrary Wasm numeric closure, generic mixed-numeric exactness, or
  full `f64` exactness green

### 6F. Numeric portability envelopes across backend and toolchain families

```bash
cargo run -p psionic-eval --example tassadar_numeric_portability_report
```

Read:

- `fixtures/tassadar/reports/tassadar_numeric_portability_report.json`

Expected outcome:

- one machine-readable portability matrix now freezes backend-, toolchain-,
  and machine-class envelopes for the bounded float and mixed-numeric lanes
- exact cpu-reference rows should currently be publication-allowed only for
  `tassadar.numeric_profile.f32_only.v1` and
  `tassadar.numeric_profile.mixed_i32_f32.v1`
- non-CPU backend rows should currently remain explicit backend-envelope
  suppressions instead of being flattened into generic numeric portability
- the bounded-approximate `tassadar.numeric_profile.bounded_f64_conversion.v1`
  row family should currently remain explicit as benchmarked but non-published
- this artifact narrows deployment truth for numeric widening only; it does
  not make backend-invariant float exactness, arbitrary Wasm numeric closure,
  or broader served publication green

### 6G. Float-enabled profile publication and route gate

```bash
cargo run -p psionic-eval --example tassadar_float_profile_acceptance_gate_report
cargo run -p psionic-router --example tassadar_float_profile_route_policy_report
```

Read:

- `fixtures/tassadar/reports/tassadar_float_profile_acceptance_gate_report.json`
- `fixtures/tassadar/reports/tassadar_float_profile_route_policy_report.json`

Expected outcome:

- exact `tassadar.numeric_profile.f32_only.v1` and
  `tassadar.numeric_profile.mixed_i32_f32.v1` should currently be green for
  named public profile posture
- `tassadar.numeric_profile.bounded_f64_conversion.v1` should currently remain
  explicit as suppressed
- `default_served_profile_allowed_profile_ids` should currently remain empty
- the route policy should currently promote only the two exact numeric
  profiles into cpu-reference profile-specific routes
- this gate narrows public numeric publication only; it does not create a new
  default served float lane or imply arbitrary Wasm float closure

### 7. Differential Wasm conformance

```bash
cargo run -p psionic-eval --example tassadar_wasm_conformance_report
```

Read:

- `fixtures/tassadar/reports/tassadar_wasm_conformance_report.json`

Expected outcome:

- exact success on the current supported i32 global/table/call-indirect subset
- exact trap parity on the seeded trap cases
- explicit boundary refusal on the unsupported host-import case

### 8. Module-scale Wasm workloads

```bash
cargo run -p psionic-eval --example tassadar_module_scale_workload_suite_report
```

Read:

- `fixtures/tassadar/reports/tassadar_module_scale_workload_suite_report.json`

Expected outcome:

- exact lowering for fixed-span memcpy, checksum, parsing, and VM-style cases
- explicit parameter-ABI refusal on the VM-style parameter case

### 9. Trap and exception parity

```bash
cargo run -p psionic-eval --example tassadar_trap_exception_report
```

Read:

- `fixtures/tassadar/reports/tassadar_trap_exception_report.json`
- `fixtures/tassadar/reports/tassadar_trap_exception_runtime_report.json`

Expected outcome:

- one success-parity control row
- explicit trap parity for bounds faults and indirect-call failures
- explicit refusal parity for malformed imports and unsupported profiles
- `drift_case_count = 0`

### 10. Bounded exceptions proposal profile

```bash
cargo run -p psionic-runtime --example tassadar_exception_profile_runtime_report
cargo run -p psionic-eval --example tassadar_exception_profile_report
```

Read:

- `fixtures/tassadar/reports/tassadar_exception_profile_runtime_report.json`
- `fixtures/tassadar/reports/tassadar_exception_profile_report.json`

Expected outcome:

- one named public exception profile:
  `tassadar.proposal_profile.exceptions_try_catch_rethrow.v1`
- exact typed throw/catch success parity
- exact trap parity on nested rethrow and handler-tag mismatch cases
- exact trap-stack parity on the non-success exception cases
- explicit malformed-handler refusal parity
- no default served exception profile ids

### 11. Bounded `memory64` continuation profile

```bash
cargo run -p psionic-runtime --example tassadar_memory64_resume_bundle
cargo run -p psionic-eval --example tassadar_memory64_profile_report
```

Read:

- `fixtures/tassadar/runs/tassadar_memory64_resume_v1/tassadar_memory64_resume_bundle.json`
- `fixtures/tassadar/reports/tassadar_memory64_profile_report.json`

Expected outcome:

- one named bounded `memory64` profile:
  `tassadar.proposal_profile.memory64_continuation.v1`
- exact sparse-window checkpoint and resume parity above the 4GiB boundary
- exact memory-growth checkpoint and resume parity above the 4GiB boundary
- persisted checkpoint and manifest artifacts under
  `fixtures/tassadar/runs/tassadar_memory64_resume_v1`
- typed datastream resume locators for the successful checkpoint rows
- explicit backend virtual-address-limit refusal parity

### 12. Bounded multi-memory routing profile

```bash
cargo run -p psionic-runtime --example tassadar_multi_memory_runtime_bundle
cargo run -p psionic-eval --example tassadar_multi_memory_profile_report
```

Read:

- `fixtures/tassadar/runs/tassadar_multi_memory_profile_v1/tassadar_multi_memory_runtime_bundle.json`
- `fixtures/tassadar/reports/tassadar_multi_memory_profile_report.json`

Expected outcome:

- one named bounded multi-memory profile:
  `tassadar.proposal_profile.multi_memory_routing.v1`
- exact routing parity for the `rodata_heap_output_split` topology
- exact routing and resume parity for the `scratch_heap_checkpoint_split`
  topology
- persisted per-memory checkpoint artifacts under
  `fixtures/tassadar/runs/tassadar_multi_memory_profile_v1`
- explicit malformed-topology refusal parity

### 13. Bounded component/linking proposal profile

```bash
cargo run -p psionic-runtime --example tassadar_component_linking_runtime_bundle
cargo run -p psionic-eval --example tassadar_component_linking_profile_report
```

Read:

- `fixtures/tassadar/runs/tassadar_component_linking_profile_v1/tassadar_component_linking_runtime_bundle.json`
- `fixtures/tassadar/reports/tassadar_component_linking_profile_report.json`

Expected outcome:

- one named bounded component/linking profile:
  `tassadar.proposal_profile.component_linking_interface_types.v1`
- exact interface-type lowering parity for the
  `utf8_decode_writer_component_pair` topology
- exact interface-type lowering parity for the
  `checkpoint_resume_component_pair` topology
- persisted lineage artifacts under
  `fixtures/tassadar/runs/tassadar_component_linking_profile_v1`
- explicit incompatible-component-interface refusal parity

### 14. Bounded SIMD deterministic profile

```bash
cargo run -p psionic-runtime --example tassadar_simd_profile_runtime_report
cargo run -p psionic-eval --example tassadar_simd_profile_report
```

Read:

- `fixtures/tassadar/reports/tassadar_simd_profile_runtime_report.json`
- `fixtures/tassadar/reports/tassadar_simd_profile_report.json`

Expected outcome:

- one named bounded SIMD profile:
  `tassadar.proposal_profile.simd_deterministic.v1`
- exact cpu-reference row
- explicit metal and cuda scalar-fallback rows
- explicit accelerator-specific refusal row
- no default served SIMD profile ids

### 15. Research-only threads profile with deterministic scheduler envelope

```bash
cargo run -p psionic-runtime --example tassadar_threads_research_profile_report
cargo run -p psionic-sandbox --example tassadar_threads_scheduler_sandbox_boundary_report
cargo run -p psionic-eval --example tassadar_threads_research_profile_report
cargo run -p psionic-research --example tassadar_threads_research_profile_summary
```

Read:

- `fixtures/tassadar/reports/tassadar_threads_research_profile_runtime_report.json`
- `fixtures/tassadar/reports/tassadar_threads_scheduler_sandbox_boundary_report.json`
- `fixtures/tassadar/reports/tassadar_threads_research_profile_report.json`
- `fixtures/tassadar/reports/tassadar_threads_research_profile_summary.json`

Expected outcome:

- one research-only threads profile:
  `tassadar.research_profile.threads_deterministic_scheduler.v1`
- exact deterministic parity for the bounded round-robin shared-counter row
- exact deterministic parity for the bounded barrier-then-reduce row
- explicit sandbox refusal on host-nondeterministic scheduling
- explicit runtime and sandbox refusal parity on relaxed shared-memory ordering
- `served_publication_allowed = false`

### 16. Durable process snapshot, tape, and work-queue object family

```bash
cargo run -p psionic-runtime --example tassadar_process_object_bundle
cargo run -p psionic-eval --example tassadar_process_object_report
```

Read:

- `fixtures/tassadar/runs/tassadar_process_objects_v1/tassadar_process_object_bundle.json`
- `fixtures/tassadar/reports/tassadar_process_object_report.json`
- persisted snapshot, tape, and work-queue artifacts under
  `fixtures/tassadar/runs/tassadar_process_objects_v1`

Expected outcome:

- one named durable process-object family:
  `tassadar.internal_compute.process_objects.v1`
- first-class snapshot, tape, and work-queue objects for the committed
  checkpoint-backed workload rows
- exact durable-process parity for all seeded rows
- typed datastream locators for process snapshots, tapes, and work queues
- explicit stale-snapshot, out-of-range tape cursor, and profile-mismatched
  queue refusals

### 17. Interactive message-loop and session-process compute profile

```bash
cargo run -p psionic-runtime --example tassadar_session_process_profile_runtime_report
cargo run -p psionic-eval --example tassadar_session_process_profile_report
cargo run -p psionic-router --example tassadar_session_process_route_policy_report
```

Read:

- `fixtures/tassadar/reports/tassadar_session_process_profile_runtime_report.json`
- `fixtures/tassadar/reports/tassadar_session_process_profile_report.json`
- `fixtures/tassadar/reports/tassadar_session_process_route_policy_report.json`

Expected outcome:

- one named bounded interactive session-process profile:
  `tassadar.internal_compute.session_process.v1`
- exact deterministic parity for the bounded echo-turn loop
- exact deterministic parity for the bounded stateful-counter loop
- named public profile posture only for the deterministic interaction surfaces
- zero default served session-process profiles
- explicit route-policy refusal on the open-ended external-event surface

### 18. Spill-aware memory extension and external tape-store semantics

```bash
cargo run -p psionic-runtime --example tassadar_spill_tape_store_bundle
cargo run -p psionic-eval --example tassadar_spill_tape_store_report
```

Read:

- `fixtures/tassadar/runs/tassadar_spill_tape_store_v1/tassadar_spill_tape_store_bundle.json`
- `fixtures/tassadar/reports/tassadar_spill_tape_store_report.json`
- persisted spill-segment and external-tape-store artifacts under
  `fixtures/tassadar/runs/tassadar_spill_tape_store_v1`

Expected outcome:

- one named bounded spill-aware continuation profile:
  `tassadar.internal_compute.spill_tape_store.v1`
- exact spill-vs-in-core parity on the bounded long-loop row
- exact spill-vs-in-core parity on the bounded search-frontier row
- exact external-tape resume parity on both bounded exact rows
- typed datastream locators for spill segments and external tape-store segments
- explicit oversize-state, missing-segment, and non-cpu portability refusals

### 19. Preemption, scheduling fairness, and resumable job receipts

```bash
cargo run -p psionic-runtime --example tassadar_preemptive_job_runtime_bundle
cargo run -p psionic-cluster --example tassadar_preemptive_job_fairness_report
cargo run -p psionic-eval --example tassadar_preemptive_job_report
```

Read:

- `fixtures/tassadar/runs/tassadar_preemptive_jobs_v1/tassadar_preemptive_job_runtime_bundle.json`
- `fixtures/tassadar/reports/tassadar_preemptive_job_fairness_report.json`
- `fixtures/tassadar/reports/tassadar_preemptive_job_report.json`

Expected outcome:

- one named bounded preemptive-job profile:
  `tassadar.internal_compute.preemptive_jobs.v1`
- exact slice-boundary and resume parity on the deterministic round-robin row
- exact slice-boundary and resume parity on the weighted-fair search row
- explicit green scheduler truth for deterministic round-robin and weighted fair
- explicit refusal on the host-nondeterministic scheduler regime
- zero served publication widening for the preemptive-job lane

### 20. Deterministic virtual filesystem and artifact-mount effect profile

```bash
cargo run -p psionic-runtime --example tassadar_virtual_fs_mount_runtime_bundle
cargo run -p psionic-sandbox --example tassadar_virtual_fs_mount_sandbox_boundary_report
cargo run -p psionic-eval --example tassadar_virtual_fs_mount_profile_report
```

Read:

- `fixtures/tassadar/runs/tassadar_virtual_fs_mounts_v1/tassadar_virtual_fs_mount_runtime_bundle.json`
- `fixtures/tassadar/reports/tassadar_virtual_fs_mount_sandbox_boundary_report.json`
- `fixtures/tassadar/reports/tassadar_virtual_fs_mount_profile_report.json`
- mounted-artifact and artifact-read-proof files under
  `fixtures/tassadar/runs/tassadar_virtual_fs_mounts_v1`

Expected outcome:

- one named bounded virtual-filesystem and artifact-mount profile:
  `tassadar.effect_profile.virtual_fs_mounts.v1`
- challengeable artifact-read proofs on the mounted dictionary and config rows
- deterministic bounded ephemeral workspace writes on the cache-write row
- explicit refusal on ambient host paths and path-traversal / undeclared-mount widening
- zero served publication widening for the virtual-filesystem lane

### 21. Deterministic clock, randomness, and network-simulator effect profile

```bash
cargo run -p psionic-runtime --example tassadar_simulator_effect_runtime_bundle
cargo run -p psionic-sandbox --example tassadar_simulator_effect_sandbox_boundary_report
cargo run -p psionic-eval --example tassadar_simulator_effect_profile_report
```

Read:

- `fixtures/tassadar/runs/tassadar_simulator_effects_v1/tassadar_simulator_effect_runtime_bundle.json`
- `fixtures/tassadar/reports/tassadar_simulator_effect_sandbox_boundary_report.json`
- `fixtures/tassadar/reports/tassadar_simulator_effect_profile_report.json`
- seeded simulator-trace files under
  `fixtures/tassadar/runs/tassadar_simulator_effects_v1`

Expected outcome:

- one named bounded simulator-backed effect profile:
  `tassadar.effect_profile.simulator_backed_io.v1`
- exact replay parity on the seeded clock, pseudo-random, and loopback-network rows
- explicit allowed-profile truth for the seeded simulator envelope
- explicit refusal on ambient system clock, OS entropy, and socket I/O
- zero served publication widening for the simulator-backed effect lane

### 22. Async call, interrupt, retry, and cancellation semantics

```bash
cargo run -p psionic-runtime --example tassadar_async_lifecycle_profile_runtime_report
cargo run -p psionic-eval --example tassadar_async_lifecycle_profile_report
cargo run -p psionic-router --example tassadar_async_lifecycle_route_policy_report
```

Read:

- `fixtures/tassadar/reports/tassadar_async_lifecycle_profile_runtime_report.json`
- `fixtures/tassadar/reports/tassadar_async_lifecycle_profile_report.json`
- `fixtures/tassadar/reports/tassadar_async_lifecycle_route_policy_report.json`

Expected outcome:

- one named bounded async-lifecycle profile:
  `tassadar.internal_compute.async_lifecycle.v1`
- exact parity on the interrupt, bounded retry, and safe-boundary cancellation rows
- explicit named-public and zero-default-served posture for the async-lifecycle lane
- explicit refusal on open-ended callbacks, mid-effect cancellation, and unbounded retry
- profile-specific route promotion only for the bounded lifecycle surfaces

### 23. Effectful replay, challenge, and audit receipts

```bash
cargo run -p psionic-runtime --example tassadar_effectful_replay_audit_bundle
cargo run -p psionic-eval --example tassadar_effectful_replay_audit_report
```

Read:

- `fixtures/tassadar/runs/tassadar_effectful_replay_audit_v1/tassadar_effectful_replay_audit_bundle.json`
- `fixtures/tassadar/reports/tassadar_effectful_replay_audit_report.json`
- effect-receipt and challenge-receipt files under
  `fixtures/tassadar/runs/tassadar_effectful_replay_audit_v1`

Expected outcome:

- one named bounded replay-and-challenge lane:
  `tassadar.effect_profile.replay_challenge_receipts.v1`
- exact replay-and-challenge parity on the seeded simulator, virtual-filesystem,
  and async-safe-cancel rows
- explicit refusal on missing effect receipts, missing challenge receipts, and
  unsafe effect families
- explicit `kernel-policy` and `nexus` dependency markers on the report rather
  than silent authority backfill inside standalone psionic
- zero served publication widening for the effectful replay lane

### 24. Internal component-model ABI and interface-type contracts

```bash
cargo run -p psionic-runtime --example tassadar_internal_component_abi_bundle
cargo run -p psionic-eval --example tassadar_internal_component_abi_report
```

Read:

- `fixtures/tassadar/runs/tassadar_internal_component_abi_v1/tassadar_internal_component_abi_bundle.json`
- `fixtures/tassadar/reports/tassadar_internal_component_abi_report.json`
- interface-manifest files under
  `fixtures/tassadar/runs/tassadar_internal_component_abi_v1`

Expected outcome:

- one named bounded internal-compute component-model ABI lane:
  `tassadar.internal_compute.component_model_abi.v1`
- exact interface parity on the session-checkpoint, artifact-reader-retry, and
  spill-resume component graphs
- persisted interface manifests for those three exact graphs
- explicit refusal on cross-profile handle mismatches and unsupported
  variant-union shapes
- explicit benchmark-only posture with `served_publication_allowed = false`

### 25. Internal compute package manager, catalog, and dependency solver

```bash
cargo run -p psionic-compiler --example tassadar_internal_compute_package_manager_report
cargo run -p psionic-router --example tassadar_internal_compute_package_route_policy_report
cargo run -p psionic-eval --example tassadar_internal_compute_package_manager_eval_report
```

Read:

- `fixtures/tassadar/reports/tassadar_internal_compute_package_manager_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_route_policy_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_manager_eval_report.json`

Expected outcome:

- three named public packages:
  `package.clrs_shortest_path_stack.v1`,
  `package.hungarian_matching_stack.v1`, and
  `package.verifier_search_stack.v1`
- three routeable package ids and zero default-served package ids
- explicit refusal on ambiguous solver, insufficient-evidence, and
  portability-mismatch package requests
- explicit benchmark-gated served-publication posture without widening this
  lane into a default served software platform

### 26. Cross-profile linking, downgrade planning, and compatibility refusal

```bash
cargo run -p psionic-compiler --example tassadar_cross_profile_link_compatibility_report
cargo run -p psionic-runtime --example tassadar_cross_profile_link_compatibility_bundle
cargo run -p psionic-router --example tassadar_cross_profile_link_route_policy_report
cargo run -p psionic-eval --example tassadar_cross_profile_link_eval_report
```

Read:

- `fixtures/tassadar/reports/tassadar_cross_profile_link_compatibility_report.json`
- `fixtures/tassadar/runs/tassadar_cross_profile_link_compatibility_v1/tassadar_cross_profile_link_compatibility_bundle.json`
- `fixtures/tassadar/reports/tassadar_cross_profile_link_route_policy_report.json`
- `fixtures/tassadar/reports/tassadar_cross_profile_link_eval_report.json`

Expected outcome:

- one bounded cross-profile software-platform compatibility lane
- one exact session-process -> spill-tape link under the current-host
  cpu-reference envelope
- one explicit generalized-ABI -> component-model downgrade plan that
  preserves deterministic parity
- explicit refusal on portability-envelope and effect-boundary mismatches
- explicit non-served posture with `served_publication_allowed = false`

### 27. Installed-process lifecycle

```bash
cargo run -p psionic-runtime --example tassadar_installed_process_lifecycle_bundle
cargo run -p psionic-eval --example tassadar_installed_process_lifecycle_report
```

Read:

- `fixtures/tassadar/runs/tassadar_installed_process_lifecycle_v1/tassadar_installed_process_lifecycle_bundle.json`
- `fixtures/tassadar/reports/tassadar_installed_process_lifecycle_report.json`
- installed-snapshot, migration-receipt, and rollback-receipt artifacts under
  `fixtures/tassadar/runs/tassadar_installed_process_lifecycle_v1`

Expected outcome:

- one bounded installed-process lifecycle lane
- one exact portable migration row and one exact rollback row over committed
  process objects
- typed datastream locators for installed snapshots, migration receipts, and
  rollback receipts
- explicit refusal on stale snapshots, portability mismatches, and missing
  rollback lineage
- explicit operator-only posture with `served_publication_allowed = false`

### 28. Learned call-stack and heap suite

```bash
cargo run -p psionic-train --example tassadar_learned_call_stack_heap_suite
cargo run -p psionic-eval --example tassadar_learned_call_stack_heap_suite_report
cargo run -p psionic-research --example tassadar_learned_call_stack_heap_suite_summary
```

Read:

- `fixtures/tassadar/runs/tassadar_learned_call_stack_heap_suite_v1/learned_call_stack_heap_suite_bundle.json`
- `fixtures/tassadar/reports/tassadar_learned_call_stack_heap_suite_report.json`
- `fixtures/tassadar/reports/tassadar_learned_call_stack_heap_suite_summary.json`

Expected outcome:

- one research-only learned benchmark suite over recursive, parser-frame,
  heap-allocator, resumable-process, and held-out-family workloads
- explicit later-window exactness, refusal calibration, max call depth, and
  max heap-cell limits across baseline versus structured-memory learned
  variants
- held-out-family fragility on the baseline lane and structured recoverability
  on the bounded suite remain explicit
- no broad learned process ownership, arbitrary Wasm, or served
  internal-compute promotion

Read:

- `fixtures/tassadar/runs/tassadar_program_family_frontier_v1/program_family_frontier_bundle.json`
- `fixtures/tassadar/reports/tassadar_program_family_frontier_report.json`
- `fixtures/tassadar/reports/tassadar_program_family_frontier_summary.json`

Expected outcome:

- one research-only program-family frontier over kernel-state, search-process,
  linked-program, effectful-resume, multi-module-workflow, and held-out
  virtual-machine/message-orchestrator workloads
- explicit held-out-family ladder rows, failure-mode taxonomy, and normalized
  cost-versus-frontier comparisons across compiled exact, learned
  structured-memory, and verifier-attached hybrid architecture families
- learned held-out fragility and hybrid recoverability remain explicit
- no broad internal compute, arbitrary Wasm, or served internal-compute
  promotion

Read:

- `fixtures/tassadar/runs/tassadar_hybrid_process_controller_v1/tassadar_hybrid_process_controller_runtime_bundle.json`
- `fixtures/tassadar/reports/tassadar_hybrid_process_controller_route_report.json`
- `fixtures/tassadar/reports/tassadar_hybrid_process_controller_report.json`

Expected outcome:

- one research-only verifier-attached hybrid controller over session-counter,
  search-frontier, linked-package-worker, and refused effectful-mailbox
  transitions
- explicit verifier-on versus verifier-off exactness deltas, challenge-path
  simulations, compiled-versus-hybrid route posture, and typed
  unsupported-transition refusal
- served publication remains false
- no arbitrary hybrid execution, broad internal compute, or served
  internal-compute promotion

Read:

- `fixtures/tassadar/reports/tassadar_broad_family_specialization_runtime_report.json`
- `fixtures/tassadar/reports/tassadar_broad_family_specialization_report.json`
- `fixtures/tassadar/reports/tassadar_broad_family_specialization_summary.json`

Expected outcome:

- one research-only broad-family specialization gate over state-machine,
  search-frontier, linked-worker, and effectful-resume bundles
- explicit stable-specializable, benchmark-only, and refused families plus
  lineage, decompilation, and safety-gate facts
- typed refusal on non-decompilable effect receipts and no served publication
- no arbitrary program-to-weights closure, broad internal compute, or served
  internal-compute promotion

## Validation Commands

Run the focused report checks after the flow:

```bash
cargo test -p psionic-eval article_abi -- --nocapture
cargo test -p psionic-runtime article_runtime_closeout -- --nocapture
cargo test -p psionic-eval article_runtime_closeout -- --nocapture
cargo test -p psionic-research article_runtime_closeout -- --nocapture
cargo test -p psionic-serve executor_service_publishes_rust_only_article_runtime_closeout_surface -- --nocapture
cargo test -p psionic-eval wasm_module_ingress -- --nocapture
cargo test -p psionic-eval frozen_core_wasm_window -- --nocapture
cargo test -p psionic-eval frozen_core_wasm_closure_gate -- --nocapture
cargo test -p psionic-eval float_semantics -- --nocapture
cargo test -p psionic-eval mixed_numeric -- --nocapture
cargo test -p psionic-eval numeric_portability -- --nocapture
cargo test -p psionic-eval float_profile_acceptance_gate -- --nocapture
cargo test -p psionic-router float_profile_route_policy -- --nocapture
cargo test -p psionic-eval wasm_conformance -- --nocapture
cargo test -p psionic-eval module_scale_workload_suite -- --nocapture
cargo test -p psionic-eval trap_exception -- --nocapture
cargo test -p psionic-runtime exception_profile -- --nocapture
cargo test -p psionic-eval exception_profile -- --nocapture
cargo test -p psionic-provider exception_profile_receipt_projects_report -- --nocapture
cargo test -p psionic-provider tassadar_capability_envelope_serializes_served_publication -- --nocapture
cargo test -p psionic-compiler memory64 -- --nocapture
cargo test -p psionic-runtime memory64 -- --nocapture
cargo test -p psionic-datastream memory64 -- --nocapture
cargo test -p psionic-eval memory64 -- --nocapture
cargo test -p psionic-ir multi_memory_profile -- --nocapture
cargo test -p psionic-compiler multi_memory_profile -- --nocapture
cargo test -p psionic-runtime multi_memory_profile -- --nocapture
cargo test -p psionic-eval multi_memory_profile -- --nocapture
cargo test -p psionic-provider multi_memory_profile -- --nocapture
cargo test -p psionic-ir component_linking_profile -- --nocapture
cargo test -p psionic-compiler component_linking_profile -- --nocapture
cargo test -p psionic-runtime component_linking_profile -- --nocapture
cargo test -p psionic-eval component_linking_profile -- --nocapture
cargo test -p psionic-provider component_linking_profile -- --nocapture
cargo test -p psionic-runtime simd_profile -- --nocapture
cargo test -p psionic-serve executor_service_capability_publication_serializes_benchmark_gated_matrix -- --nocapture
cargo test -p psionic-provider simd_profile -- --nocapture
cargo test -p psionic-provider tassadar_capability_envelope_serializes_served_publication -- --nocapture
cargo test -p psionic-runtime threads_research_profile -- --nocapture
cargo test -p psionic-sandbox threads_scheduler_boundary -- --nocapture
cargo test -p psionic-eval threads_research_profile -- --nocapture
cargo test -p psionic-provider threads_research_profile -- --nocapture
cargo test -p psionic-research threads_research_profile -- --nocapture
cargo test -p psionic-runtime process_object -- --nocapture
cargo test -p psionic-datastream process_ -- --nocapture
cargo test -p psionic-eval process_object -- --nocapture
cargo test -p psionic-provider process_object -- --nocapture
cargo test -p psionic-runtime session_process -- --nocapture
cargo test -p psionic-eval session_process -- --nocapture
cargo test -p psionic-router session_process -- --nocapture
cargo test -p psionic-serve executor_service_capability_publication_serializes_benchmark_gated_matrix -- --nocapture
cargo test -p psionic-provider session_process_profile_receipt_projects_report -- --nocapture
cargo test -p psionic-provider tassadar_capability_envelope_serializes_served_publication -- --nocapture
cargo test -p psionic-runtime spill_tape -- --nocapture
cargo test -p psionic-datastream spill -- --nocapture
cargo test -p psionic-datastream external_tape -- --nocapture
cargo test -p psionic-eval spill_tape -- --nocapture
cargo test -p psionic-provider spill_tape_store -- --nocapture
cargo test -p psionic-environments tassadar_environment_bundle_is_machine_legible -- --nocapture
cargo test -p psionic-runtime preemptive_job -- --nocapture
cargo test -p psionic-cluster preemptive_job -- --nocapture
cargo test -p psionic-eval preemptive_job -- --nocapture
cargo test -p psionic-provider preemptive_job -- --nocapture
cargo test -p psionic-runtime virtual_fs -- --nocapture
cargo test -p psionic-sandbox virtual_fs -- --nocapture
cargo test -p psionic-eval virtual_fs -- --nocapture
cargo test -p psionic-provider virtual_fs -- --nocapture
cargo test -p psionic-runtime simulator_effect -- --nocapture
cargo test -p psionic-sandbox simulator_effect -- --nocapture
cargo test -p psionic-eval simulator_effect -- --nocapture
cargo test -p psionic-provider simulator_effect -- --nocapture
cargo test -p psionic-runtime async_lifecycle -- --nocapture
cargo test -p psionic-eval async_lifecycle -- --nocapture
cargo test -p psionic-router async_lifecycle -- --nocapture
cargo test -p psionic-serve executor_service_capability_publication_serializes_benchmark_gated_matrix -- --nocapture
cargo test -p psionic-provider async_lifecycle -- --nocapture
cargo test -p psionic-provider tassadar_capability_envelope_serializes_served_publication -- --nocapture
```

These checks should keep the committed reports and generated truth aligned.

## How To Interpret Results

- If only the C-to-Wasm receipt refuses and the refusal is a typed local
  toolchain failure, the bounded Wasm lane can still be healthy.
- If the compile-pipeline matrix loses the exact WAT cases, that is a real
  lowering regression.
- If the frozen core-Wasm closure gate changes from its current explicit red
  posture without new supporting evidence, that is a real claim-discipline
  regression.
- If the float-semantics matrix starts claiming non-CPU backends, `f64`, or
  NaN-payload preservation without new evidence, that is a real claim-discipline
  regression.
- If the mixed-numeric ladder starts claiming arbitrary mixed-numeric closure
  or full `f64` exactness without new evidence, that is a real claim-discipline
  regression.
- If the numeric-portability matrix starts claiming non-CPU or
  bounded-approximate profiles as publication-allowed without new evidence,
  that is a real claim-discipline regression.
- If the float-profile gate starts allowing a default served float-enabled
  profile or unsuppresses bounded-approximate `f64` without new evidence, that
  is a real claim-discipline regression.
- If the ingress report stops lowering the seeded synthetic module exactly, that
  is a real bounded module-lane regression.
- If the conformance report loses exact success or trap parity on the supported
  subset, that is a real execution-truth regression.
- If the article runtime closeout loses direct exactness or its throughput
  floor on the committed long-horizon kernels, the runtime-performance part of
  the Rust-only article claim has regressed even if shorter bounded Wasm cases
  still look healthy.
- If the module-scale suite loses exactness on memcpy, checksum, parsing, or
  VM-style fixed-operand cases, that is a real workload-suite regression.
- If trap parity or refusal parity drifts in the trap report, the widened Wasm
  claim has a semantic hole even if success cases still look green.
- If the exception-profile report loses trap-stack parity, adds a default
  served exception profile, or widens beyond the current-host cpu-reference
  envelope without new evidence, that is a real claim-discipline regression.

## Cleanup Rule

The compile-receipt step can rewrite
`fixtures/tassadar/reports/tassadar_c_to_wasm_compile_receipt.json` if the
local toolchain differs from the committed environment.

If your task is only to validate the current bounded Wasm lane and not to
change the canonical receipt, restore that file before commit:

```bash
git restore --source=HEAD -- fixtures/tassadar/reports/tassadar_c_to_wasm_compile_receipt.json
```

## Exit Gate

Treat the current bounded Wasm flow as healthy only when all of the following
hold:

- the compile-pipeline, ingress, conformance, module-scale, and trap reports
  reproduce without unexpected drift
- the exception-profile runtime and eval reports reproduce without unexpected
  drift
- the article runtime closeout bundle, report, and summary reproduce without
  unexpected drift
- the focused tests above pass
- any local C-toolchain failure stays typed and explicit instead of silently
  widening or corrupting the lane
- `git status --short --branch` is clean before you leave the checkout

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
- memory ABI v2 bulk-memory exactness
- bounded dynamic-memory pause-and-resume receipts
- Rust-only article runtime closeout
- canonical Rust-only Hungarian-10x10 article reproducer
- canonical C-to-Wasm compile receipt
- source-to-Wasm-to-Tassadar compile-pipeline matrix
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

## Validation Commands

Run the focused report checks after the flow:

```bash
cargo test -p psionic-eval article_abi -- --nocapture
cargo test -p psionic-runtime article_runtime_closeout -- --nocapture
cargo test -p psionic-eval article_runtime_closeout -- --nocapture
cargo test -p psionic-research article_runtime_closeout -- --nocapture
cargo test -p psionic-serve executor_service_publishes_rust_only_article_runtime_closeout_surface -- --nocapture
cargo test -p psionic-eval wasm_module_ingress -- --nocapture
cargo test -p psionic-eval wasm_conformance -- --nocapture
cargo test -p psionic-eval module_scale_workload_suite -- --nocapture
cargo test -p psionic-eval trap_exception -- --nocapture
```

These checks should keep the committed reports and generated truth aligned.

## How To Interpret Results

- If only the C-to-Wasm receipt refuses and the refusal is a typed local
  toolchain failure, the bounded Wasm lane can still be healthy.
- If the compile-pipeline matrix loses the exact WAT cases, that is a real
  lowering regression.
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
- the article runtime closeout bundle, report, and summary reproduce without
  unexpected drift
- the focused tests above pass
- any local C-toolchain failure stays typed and explicit instead of silently
  widening or corrupting the lane
- `git status --short --branch` is clean before you leave the checkout

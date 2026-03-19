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
- broad host-import closure
- broad C/C++ frontend closure

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

### 3A. Hungarian-10x10 Rust-only article reproducer

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

### 3B. Sudoku-9x9 Rust-only article reproducer

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

### 3C. Rust-only article runtime closeout

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

### 3D. Direct model-weight execution proof

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

# Tassadar Rust-Only Article Runbook

> Status: canonical one-command operator runbook for the Rust-only article
> reproduction path.

## Purpose

This runbook answers one narrow question:

> from a clean checkout, what single command reproduces the current Rust-only
> article claim surface end to end?

It is subordinate to:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/TASSADAR_WASM_RUNBOOK.md`
- `fixtures/tassadar/reports/tassadar_acceptance_report.json`

## What This Runbook Actually Closes

This runbook closes the current bounded Rust-only article flow, not the
broadest possible reading of the Percepta article headline.

What it does prove:

- one-command reproduction of the committed Rust-only article path
- direct no-tool execution on the canonical workloads
  `long_loop_kernel`, `sudoku_v0_test_a`, and `hungarian_matching`
- direct model-weight proof receipts with zero external calls and no observed
  fallback on those committed workloads
- explicit CPU portability on `host_cpu_aarch64` and `host_cpu_x86_64`

What it does not prove:

- arbitrary C frontend closure
- arbitrary Wasm execution
- full frozen core-Wasm closure
- broad served/public universality
- or that the article's broad fast-attention headline is now the canonical
  public closeout route

The current audited closeout route remains the exact direct reference-linear
CPU lane for the committed workloads. Faster runtime families are tracked in
separate benchmark and research artifacts and should not be silently merged
into this runbook's claim boundary.

## Preconditions

Start from a clean checkout:

```bash
git status --short --branch
```

Required local baseline:

- a working Rust toolchain for this workspace
- normal `cargo run` access
- `jq` for the checker script

This runbook is Rust-only. It does not require the historical C-to-Wasm path.

## One Command

```bash
./scripts/check-tassadar-rust-only-article-reproduction.sh
```

That command runs the served orchestration harness:

```bash
cargo run -p psionic-serve --example tassadar_rust_only_article_reproduction
```

and then checks the committed reproduction report for an all-green result.

## Primary Output

Read:

- `fixtures/tassadar/reports/tassadar_rust_only_article_reproduction_report.json`

Expected outcome:

- `all_components_green=true`
- `component_count=9`
- `green_component_count=9`
- the report points at the canonical one-command script and this runbook
- the report covers the current Rust-only article surface:
  source canon, profile completeness, ABI closure, Hungarian reproducer,
  Sudoku reproducer, runtime closeout, runtime closeout summary, direct
  model-weight proof, and acceptance gate v2

## Component Outputs

The harness regenerates and validates these committed surfaces:

- `fixtures/tassadar/reports/tassadar_rust_source_canon_report.json`
- `fixtures/tassadar/reports/tassadar_rust_article_profile_completeness_report.json`
- `fixtures/tassadar/reports/tassadar_article_abi_closure_report.json`
- `fixtures/tassadar/runs/hungarian_10x10_article_reproducer_v1`
- `fixtures/tassadar/reports/tassadar_hungarian_10x10_article_reproducer_report.json`
- `fixtures/tassadar/reports/tassadar_sudoku_9x9_article_reproducer_report.json`
- `fixtures/tassadar/runs/article_runtime_closeout_v1/article_runtime_closeout_bundle.json`
- `fixtures/tassadar/reports/tassadar_article_runtime_closeout_report.json`
- `fixtures/tassadar/reports/tassadar_article_runtime_closeout_summary.json`
- `fixtures/tassadar/reports/tassadar_direct_model_weight_execution_proof_report.json`
- `fixtures/tassadar/reports/tassadar_rust_only_article_acceptance_gate_v2.json`
- `fixtures/tassadar/reports/tassadar_rust_only_article_acceptance_summary.json`
- `fixtures/tassadar/reports/tassadar_rust_only_article_closeout_audit_report.json`
- `fixtures/tassadar/reports/tassadar_rust_only_article_closeout_summary.json`

## Portability Companion

The one-command harness closes procedure for the Rust-only article path. The
cross-machine CPU portability surface is a separate companion check:

```bash
cargo run -p psionic-eval --example tassadar_article_cpu_reproducibility_report
cargo run -p psionic-research --example tassadar_article_cpu_reproducibility_summary
```

Run those commands sequentially, not in parallel. They both touch the
Rust-to-Wasm canon path, and sequential runs keep the committed portability
artifacts deterministic.

Read:

- `fixtures/tassadar/reports/tassadar_article_cpu_reproducibility_report.json`
- `fixtures/tassadar/reports/tassadar_article_cpu_reproducibility_summary.json`

Expected outcome on the current measured host:

- `current_host_machine_class_id=host_cpu_aarch64`
- `current_host_measured_green=true`
- supported classes are exactly `host_cpu_aarch64` and `host_cpu_x86_64`
- unsupported classes remain explicit under `other_host_cpu`
- the optional C-path compile row remains non-blocking for the Rust-only claim

## Acceptance Gate

The canonical prerequisite gate for the full Rust-only article claim is:

```bash
./scripts/check-tassadar-rust-only-article-acceptance-v2.sh
```

That checker runs:

```bash
cargo run -p psionic-serve --example tassadar_rust_only_article_acceptance_gate_v2
```

and then rejects any non-green result in:

- `fixtures/tassadar/reports/tassadar_rust_only_article_acceptance_gate_v2.json`

Expected outcome:

- `green=true`
- `prerequisite_count=8`
- `passed_prerequisite_count=8`
- `failed_prerequisite_ids=[]`

## Final Closeout Audit

The canonical final publication check for the full Rust-only article claim is:

```bash
./scripts/check-tassadar-rust-only-article-closeout-audit.sh
```

That checker runs:

```bash
cargo run -p psionic-eval --example tassadar_rust_only_article_closeout_audit_report
```

and then rejects any non-green result in:

- `fixtures/tassadar/reports/tassadar_rust_only_article_closeout_audit_report.json`

The companion operator summary is:

```bash
cargo run -p psionic-research --example tassadar_rust_only_article_closeout_summary
```

Read:

- `fixtures/tassadar/reports/tassadar_rust_only_article_closeout_audit_report.json`
- `fixtures/tassadar/reports/tassadar_rust_only_article_closeout_summary.json`

Expected outcome:

- `green=true`
- `all_surfaces_green=true`
- `harness_green=true`
- `acceptance_gate_green=true`
- `direct_model_weight_proof_green=true`
- `cpu_reproducibility_green=true`
- `direct_proof_case_ids` are exactly:
  `long_loop_kernel`, `sudoku_v0_test_a`, `hungarian_matching`
- `supported_machine_class_ids` are exactly:
  `host_cpu_aarch64`, `host_cpu_x86_64`

This closeout audit is the final publication boundary for the current
Rust-only article claim. It does not replace the underlying harness, gate,
proof, or CPU reports. It binds them together and turns the final claim red if
any of those surfaces regress.

## Interpretation

Green means:

- the committed Rust source canon still compiles for the bounded article path
- the explicit Rust-to-Wasm profile boundary still contains both supported and
  refused rows
- the bounded ABI lane still closes direct scalar and pointer-length heap input
- the canonical Hungarian and Sudoku article reproducers still stay exact and
  direct without observed external-tool posture
- the million-step runtime closeout still stays exact and above its declared
  floor on the direct reference-linear CPU path
- the CPU portability companion still keeps the current host green, the
  supported CPU classes explicit, and unsupported classes refused
- the direct model-weight proof still keeps the current served article cases on
  a direct-guaranteed route with zero external calls and no CPU substitution
- the acceptance gate still proves the full Rust-only article prerequisite set
  is present instead of inferring final closure from partial green artifacts
- the final closeout audit now keeps the full Rust-only article publication
  claim tied to the one-command harness, the prerequisite gate, the direct
  proof route, and the declared CPU classes instead of letting those surfaces
  drift apart in prose

Red means the Rust-only article claim has regressed, even if some narrower
bounded Wasm surfaces are still green.

## Claim Boundary

The current named internal-compute profile for this path is:

- `tassadar.internal_compute.article_closeout.v1`

The ladder now also records:

- `tassadar.internal_compute.generalized_abi.v1` as implemented but
  non-promoted

That broader generalized-ABI rung is benchmarked and challengeable, but it is
not the current served Rust-only article claim and is therefore out of scope
for this closeout runbook.

The canonical ladder and current claim-check artifacts are:

- `fixtures/tassadar/reports/tassadar_internal_compute_profile_ladder_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_profile_ladder_summary.json`

This runbook closes operator procedure only. It does not widen the current
claim surface beyond the committed Rust-only article workloads, runtime floors,
and direct proof receipts already cited by the acceptance gate.

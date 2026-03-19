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
- `component_count=8`
- `green_component_count=8`
- the report points at the canonical one-command script and this runbook
- the report covers the current Rust-only article surface:
  source canon, profile completeness, ABI closure, Hungarian reproducer,
  Sudoku reproducer, runtime closeout, runtime closeout summary, and direct
  model-weight proof

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
- the direct model-weight proof still keeps the current served article cases on
  a direct-guaranteed route with zero external calls and no CPU substitution

Red means the Rust-only article claim has regressed, even if some narrower
bounded Wasm surfaces are still green.

## Claim Boundary

This runbook closes operator procedure only. It does not widen the current
claim surface beyond the committed Rust-only article workloads, runtime floors,
and direct proof receipts already cited by the acceptance gate.

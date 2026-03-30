# Psion Executor MLX Decision-Grade Run

> Status: canonical `PSION-0204` / `#723` record, updated 2026-03-30 after
> landing the first retained MLX-local decision-grade packet and shared v2
> dashboard visibility packet.

This document records the first decision-grade MLX packet that the admitted Mac
profile may cite for explicitly MLX-local executor questions.

## Canonical Fixtures

- `fixtures/psion/executor/psion_executor_mlx_decision_grade_run_v1.json`
- `fixtures/training_visualization/psion_executor_mlx_decision_grade_remote_training_visualization_bundle_v2.json`
- `fixtures/training_visualization/psion_executor_mlx_decision_grade_remote_training_run_index_v2.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_mlx_decision_grade_run_fixtures
```

## What Landed

`psionic-train` now owns one typed MLX-local decision-grade packet that binds:

- the retained MLX smoke packet
- the retained MLX checkpoint-compatibility packet
- the retained same-node MLX report
- the retained admitted-device matrix report
- one executor-specific remote-training visualization `v2` bundle
- one executor-specific remote-training run index `v2`

That means the first MLX-local decision-grade run is no longer just a local
report path. It now appears inside the same typed dashboard grammar that the
repo already ships for the rest of the training surfaces.

## Current Retained Truth

- packet digest:
  `1b69444b0c1352bac9343426769bb2e3eb1d857e765bb8cd0eded6fc830ac1cd`
- visualization bundle digest:
  `eddf60778cdf2a5a46e50bca0bd1121a6feb3247a623e6e677a9995ae21b7fc4`
- run-index digest:
  `62d68d666f615ccdffc3152e7974d0d967f9696a52cb012b1c8bc0b72eba48cd`
- approved equivalent-subset digest:
  `91a74a9d1edc1ff1ea0bd3ed758a7e0aad7d264afeae38322f7595ad3b8cba8b`
- retained run:
  `same-node-wallclock-retained-mlx`
- retained device:
  `metal:0`
- completed steps:
  `93184`
- observed wallclock milliseconds:
  `573332`
- final mean loss:
  `0.0`
- retained dashboard entry count:
  `7`

## Approved Equivalent Local Subset

This packet does **not** pretend that the retained MLX run already emitted two
full checkpoint-time eval snapshots in the same way the later 4080 lane should.
Instead it uses one approved equivalent local subset for the explicitly
MLX-local question:

- `full_budget_retained_run_green`
- `checkpoint_restore_rehearsal_green`
- `export_smoke_green`
- `dashboard_visibility_green`

That equivalent subset is allowed only because:

- the retained run consumed most of the admitted `600` second local budget
- the checkpoint packet already keeps restore green on the retained bundle
- the export bundle already exists and stays tied to the same retained run
- the run is now visible in the shared dashboard/run-index family instead of
  hiding as a one-off local artifact

## Claim Boundary

This packet counts only for the **explicitly MLX-local** executor question.

It does **not** claim:

- Mac -> 4080 -> Mac local-cluster roundtrip truth
- remote launch truth
- shared checkpoint-writer authority
- cross-device convergence
- promotion readiness

Those remain later obligations, especially EPIC 3 and the promotion packet.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_mlx_decision_grade_run_fixtures`
- `cargo test -q -p psionic-train psion_executor_mlx_decision_grade_run -- --nocapture`

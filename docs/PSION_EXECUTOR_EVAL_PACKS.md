# Psion Executor Eval Packs

> Status: canonical `PSION-0104` / `#709` and `PSION-0105` / `#710` record,
> updated 2026-03-30 after freezing the first executor eval-pack catalog in
> `crates/psionic-train/src/psion_executor_eval_packs.rs`.

This document records the first frozen executor eval packs that phase-one local
smoke, decision-grade, and promotion work must use.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_eval_packs_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_eval_pack_fixtures
```

## Baseline Truth Binding

The frozen packs now have one committed baseline-truth overlay:

- `docs/PSION_EXECUTOR_BASELINE_TRUTH.md`
- `fixtures/psion/executor/psion_executor_baseline_truth_v1.json`

That packet reconstructs `trained-v0` truth across all frozen suite ids before
later variance or promotion-delta work starts arguing from the packs.

## Formatting Audit Binding

The frozen packs now also have one committed formatting and post-processing
audit:

- `docs/PSION_EXECUTOR_FORMATTING_AUDIT.md`
- `fixtures/psion/executor/psion_executor_formatting_audit_v1.json`

Decision-grade run admission for the executor lane assumes this audit packet
stays green, because unchecked prompt formatting, normalization, or
post-processing would invalidate the packs even if baseline truth stayed green.

## Decision-Threshold Binding

The promotion pack now also has one committed decision-threshold packet:

- `docs/PSION_EXECUTOR_DECISION_THRESHOLDS.md`
- `fixtures/psion/executor/psion_executor_decision_thresholds_v1.json`

That packet replays the retained baseline three times, records the current
promotion aggregates, and freezes the minimum meaningful deltas later same-
budget comparisons are allowed to claim.

## What Landed

`psionic-train` now owns one typed executor eval-pack catalog with two packs:

- `tassadar.eval.frequent.v0`
- `tassadar.eval.promotion.v0`

The catalog freezes:

- pack ids and versioned scope
- admitted profile ids that may cite the pack
- authority artifacts and digests
- suite classes and frozen case ids
- promotion throughput thresholds
- the acceptance-profile binding for promotion work

## `tassadar.eval.frequent.v0`

This is the checkpoint-time decision pack.

It freezes four required surfaces:

- exactness cases on the currently admitted executor workloads:
  `micro_wasm_kernel`, `branch_heavy_kernel`, `memory_heavy_kernel`,
  `long_loop_kernel`, `sudoku_v0_test_a`, `hungarian_matching`
- held-out exclusions from
  `fixtures/psion/isolation/psion_exclusion_manifest_v1.json`
- operator review cases:
  `artifact_packet_complete`, `checkpoint_restore_rehearsal_green`,
  `export_smoke_green`, `local_cluster_roundtrip_green`
- throughput blockers keyed to:
  `tassadar.reference_linear_steps_per_second`,
  `tassadar.hull_cache_steps_per_second`,
  `tassadar.hull_cache_speedup_over_reference_linear`,
  `tassadar.hull_cache_remaining_gap_vs_cpu_reference`

This pack does not decide promotion. It exists so checkpoint-time review uses
one frozen spine instead of one-off arguments.

### MLX Smoke Subset

Phase-one MLX smoke runs may cite one admitted local subset of
`tassadar.eval.frequent.v0` while the Mac profile is still proving substrate
closure rather than full executor-model training:

- subset id: `tassadar.eval.frequent.v0::mlx_smoke_subset_v1`
- suite id: `frequent_operator_review_cases_v0`
- included cases:
  `artifact_packet_complete`,
  `checkpoint_restore_rehearsal_green`,
  `export_smoke_green`
- deferred case:
  `local_cluster_roundtrip_green`

That keeps the smoke lane honest:

- local artifact packet completeness must be explicit
- local restore rehearsal must be explicit
- local export smoke must be explicit
- Mac -> 4080 -> Mac roundtrip truth remains an EPIC 3 obligation instead of a
  fake green check on a Mac-only smoke run

## `tassadar.eval.promotion.v0`

This is the first executor promotion pack.

It binds directly to:

- `docs/PSION_EXECUTOR_ACCEPTANCE_PROFILE.md`

It freezes all of the phase-one executor promotion surfaces:

- exactness suite on the admitted workload family
- held-out suite with both retained kernel-family and randomized hold-out rows
- adversarial suite with article-scale hostile variants
- runtime blockers covering CPU validation, restore rehearsal, and local
  cluster roundtrip truth
- serving blockers covering export, replacement, promoted-artifact
  compatibility, and rollback safety
- `reference_linear` anchor checks
- admitted-workload `hull_cache` fast-route checks

The first frozen throughput thresholds are:

- `tassadar.hull_cache_speedup_over_reference_linear >= 1.5`
- `tassadar.hull_cache_remaining_gap_vs_cpu_reference <= 3.0`

Those thresholds are deliberately conservative. They are derived from the
current retained article benchmark report and are meant to stop obvious
throughput regressions without pretending this first pack already closes every
later performance question.

## `reference_linear` vs `hull_cache`

The promotion pack now repeats the current claim boundary explicitly:

- `reference_linear` remains the measured baseline truth anchor
- `hull_cache` remains the admitted fast-route target on the executor workload
  family
- fast-route wins never override exactness, held-out, adversarial, runtime, or
  serving blockers

## Honest Boundary

These packs do not create a second eval spine and they do not widen the
executor workload family.

They freeze the first reviewable surfaces so later MLX, 4080, and local-cluster
runs can be compared honestly.

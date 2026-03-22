# Psion Architecture Reasoning Benchmark

> Status: canonical `PSION-21` / `#377` architecture-benchmark contract,
> written 2026-03-22 after landing the first typed architecture-reasoning
> package and direct acceptance-matrix binding for `Psion`.

This document freezes the first dedicated architecture-reasoning benchmark
package for the `Psion` learned-model lane.

It builds on the shared benchmark contracts in
`docs/PSION_BENCHMARK_PACKAGES.md` and
`docs/PSION_BENCHMARK_LABEL_GENERATION.md`.

## Canonical Artifacts

- `fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json` contains the
  canonical package row `psion_architecture_reasoning_benchmark_v1`.
- `fixtures/psion/benchmarks/psion_benchmark_receipt_set_v1.json` contains the
  canonical package receipt for that benchmark package.
- `fixtures/psion/benchmarks/psion_benchmark_label_generation_receipt_set_v1.json`
  contains the canonical label-generation receipt for that package.
- `fixtures/psion/acceptance/psion_acceptance_matrix_v1.json` now binds the
  architecture benchmark requirements directly to
  `psion_architecture_reasoning_benchmark_v1` for pilot and later scale-up
  gates.
- `crates/psionic-train/src/psion_benchmark_packages.rs` owns the typed
  architecture item payloads inside the shared package contract.
- `crates/psionic-train/src/psion_acceptance_matrix.rs` owns the direct
  benchmark-artifact binding used by the acceptance matrix.

## Package Identity

The first dedicated package is:

- package id: `psion_architecture_reasoning_benchmark_v1`
- package family: `architecture_reasoning`
- package digest:
  `703142f680a8a2702700fcdd18309b24d6e7889e07c4621e287178c9ac6af674`

The package is not a generic “systems design” prompt bucket. It is a bounded
architecture-reasoning benchmark package with typed items under one shared
rubric-backed grading surface.

## Typed Coverage

The canonical package now covers four typed probe kinds explicitly:

- dominant constraint
- bottleneck
- scheduling behavior
- tradeoff analysis

Each architecture item now preserves:

- `target_architecture`
- `workload_ref`
- `probe_kind`
- `dominant_constraint`
- `explicit_assumptions_required`
- `expected_focus`

This keeps the package focused on constraint-aware reasoning under stated
assumptions instead of rewarding generic fluent architecture prose.

## Labels And Receipts

The architecture package currently stays rubric-backed on the label side:

- the package receipt stays on the shared `PsionBenchmarkPackageReceipt`
  contract
- the label-generation receipt pins the rubric ref, rubric version, and
  generator reference for each architecture item
- the architecture receipt remains acceptance-ready through the shared
  `PsionBenchmarkEvidenceReceipt` shape with `PassRateBps`

## Contamination Attachment

The package is not treated as green through prompt shape or label quality
alone.

Its contamination posture is attached through the package’s committed
contamination-input bundle:

- benchmark-visible source ids remain explicit
- held-out and training-excluded review inputs remain explicit
- the near-duplicate review reference remains explicit
- later promotion gates still require a clean contamination review before the
  architecture package can count toward scale-up

## Acceptance Binding

`Psion` acceptance-matrix `v1` now binds the architecture family directly to
the concrete benchmark package artifact instead of only naming the family.

That direct binding is used on:

- `pilot`
- `broader_pretraining`
- `trusted_cluster_scale_up`

This means later promotion decisions cannot satisfy the architecture gate with
an arbitrary receipt from another package that happens to claim the same family
label. The receipt has to point at the frozen architecture package id and
digest above.

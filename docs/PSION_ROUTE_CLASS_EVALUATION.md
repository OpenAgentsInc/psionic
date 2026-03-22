# Psion Route-Class Evaluation

> Status: canonical `PSION-25` / `#381` route-class benchmark contract,
> written 2026-03-22 after landing the first four-class route package and
> route receipt for `Psion`.

This document freezes the first dedicated route-class evaluation package for
the `Psion` learned-model lane.

It is distinct from refusal calibration. The package is meant to test the
bounded route policy inside the supported learned lane: when to answer in
language, when to answer with uncertainty, when to request structured inputs,
and when to delegate to the exact executor.

## Canonical Artifacts

- `fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json` contains the
  canonical package row `psion_route_benchmark_v1`.
- `fixtures/psion/benchmarks/psion_benchmark_receipt_set_v1.json` contains the
  canonical package receipt for that benchmark package.
- `fixtures/psion/benchmarks/psion_benchmark_label_generation_receipt_set_v1.json`
  contains the canonical label-generation receipt for that package.
- `fixtures/psion/route/psion_route_class_evaluation_receipt_v1.json` contains
  the canonical route-class receipt that later serving and capability
  publication work can consume directly.
- `fixtures/psion/acceptance/psion_acceptance_matrix_v1.json` now binds the
  route-selection requirements directly to `psion_route_benchmark_v1` instead
  of leaving the route family artifact-agnostic.
- `crates/psionic-train/src/psion_benchmark_packages.rs` owns the typed route
  item payloads and exact route-class graders.
- `crates/psionic-train/src/psion_route_class_evaluation.rs` owns the route
  receipt with explicit false-positive and false-negative delegation fields.

## Package Identity

The first dedicated package is:

- package id: `psion_route_benchmark_v1`
- package family: `route_evaluation`
- package digest:
  `76c517290e29616bf424831b637833006dc16b9a70bd8b0162ca6b5a90efa954`

## Typed Coverage

The canonical package now covers four explicit route classes:

- `answer_in_language`
- `answer_with_uncertainty`
- `request_structured_inputs`
- `delegate_to_exact_executor`

Each route item now preserves:

- `route_class`
- `route_boundary_ref`
- `required_signal`
- `structured_input_schema_ref`
- `uncertainty_required`

This keeps route evaluation focused on route policy rather than one binary
route-or-no-route score.

## Route Receipt

The route receipt records one row per route class and keeps three things
explicit:

- observed route-selection accuracy for the class
- false-positive delegation when a class should have stayed in the learned lane
- false-negative delegation when a class should have handed off to the exact
  executor

The canonical receipt therefore distinguishes answer, uncertainty,
structured-input requests, and exact delegation directly.

## Acceptance Binding

`Psion` acceptance-matrix `v1` now binds the route-selection gate directly to
the concrete route benchmark package above.

The concrete artifact is required for:

- `sft_promotion`
- `internal_serving`
- `trusted_cluster_scale_up`

That keeps route claims tied to one committed benchmark package, while later
serving and capability-publication work can cite the route-class receipt for
class-specific route evidence instead of only an aggregate score.

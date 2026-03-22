# PSION Plugin Benchmark Packages

> Status: canonical `PSION_PLUGIN-7` shared benchmark-package contract for the
> plugin-conditioned convergence tranche, written 2026-03-22 after landing the
> first repo-owned plugin benchmark item schema and shared grader interfaces in
> `psionic-train`.

This document freezes the shared benchmark contract for plugin-conditioned
evaluation.

It is not the package-specific content for arguments, sequencing, refusal, or
interpretation.

It is the common contract that those package families must now share.

The first package-specific discovery-and-selection contract now lives in
`docs/PSION_PLUGIN_DISCOVERY_SELECTION_BENCHMARK.md` and reuses this shared
surface instead of inventing a new one.

## Canonical Artifacts

- `docs/PSION_PLUGIN_BENCHMARK_PACKAGES.md` is the canonical human-readable
  contract.
- `crates/psionic-train/src/psion_plugin_benchmark_packages.rs` owns the
  shared package shape, item schema, contamination attachment, receipt posture,
  task contracts, and grader interfaces.
- `docs/PSION_PLUGIN_DISCOVERY_SELECTION_BENCHMARK.md` is the first package-
  specific child contract built on top of this shared surface.

The stable schema version is
`psionic.psion.plugin_benchmark_package.v1`.

## What The Shared Contract Standardizes

The shared contract now freezes:

- one common outer package shape above the generic `psionic-eval`
  `BenchmarkPackage`
- one shared prompt-format surface for plugin-conditioned tasks
- one shared item schema that preserves contamination attachment and receipt
  posture per item
- one shared task-contract surface spanning:
  - discovery and selection
  - argument construction
  - sequencing and multi-call planning
  - refusal and request-for-structure
  - post-plugin result interpretation
- one shared grader-interface surface spanning:
  - selection decision
  - exact route
  - argument schema
  - sequencing plan
  - exact refusal
  - interpretation rubric

This means later plugin-use benchmark packages no longer get to invent their
own incompatible item or grader schemas.

## Contamination And Receipt Truth

Every benchmark item now carries:

- one contamination attachment bound to the `PSION_PLUGIN-6` parent-lineage bundle
- one receipt-posture block that says whether execution evidence is required
- one task payload and one grader binding

The shared contract validates that:

- contamination lineage ids resolve against the committed contamination bundle
- source-case and receipt refs actually come from those parent-lineage rows
- required receipt refs are a subset of the contamination attachment
- execution-backed tasks cannot silently drop their receipt requirement

This keeps plugin-use evaluation tied to the same provenance discipline as the
training substrate.

## Family Boundary

The first shared contract covers only the bounded host-native plugin tranche.

It does not yet claim:

- guest-artifact plugin benchmark coverage
- secret-backed or stateful plugin benchmark coverage
- broad controller-specific grading forks
- package-specific benchmark thresholds or green results

Those later issues must all reuse this shared contract rather than replacing
it.

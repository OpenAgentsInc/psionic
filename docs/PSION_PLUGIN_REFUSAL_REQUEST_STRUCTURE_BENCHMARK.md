# PSION Plugin Refusal And Request-For-Structure Benchmark

> Status: canonical `PSION_PLUGIN-11` refusal-and-request-for-structure
> benchmark package for the bounded host-native plugin tranche, written
> 2026-03-22 after landing the first repo-owned negative-case package and
> receipt on top of the shared `psion_plugin` benchmark contract.

This document freezes the first benchmark family for plugin refusal,
request-for-structure, and overdelegation-negative behavior.

The package is benchmark-authored on purpose.

The current held-out split does not yet carry the full unsupported-capability,
missing-input, and overdelegation breadth this package needs, so the benchmark
keeps authored prompt provenance explicit instead of pretending those rows came
from held-out execution lineage.

## Canonical Artifacts

- `docs/PSION_PLUGIN_REFUSAL_REQUEST_STRUCTURE_BENCHMARK.md` is the canonical
  human-readable contract.
- `crates/psionic-train/src/psion_plugin_refusal_request_structure_benchmark.rs`
  owns the package and receipt builders.
- `crates/psionic-train/examples/psion_plugin_refusal_request_structure_benchmark.rs`
  writes the canonical bundle.
- `fixtures/psion/benchmarks/psion_plugin_refusal_request_structure_benchmark_v1/`
  carries the first committed bundle.

## Coverage

The first package covers:

- unsupported-capability refusal
- missing-input request-for-structure
- direct-answer overdelegation negatives
- unsupported-capability overdelegation negatives

This is enough to close the first refusal/request-for-structure package
without claiming argument grading, execution-backed result interpretation, or
guest-artifact refusal coverage.

## Boundary

Each item now preserves:

- the expected refusal/request/answer route
- accepted refusal reason codes when explicit refusal is required
- missing argument paths when request-for-structure is required
- unsupported plugin ids when capability refusal is required
- whether the case is an overdelegation negative

That keeps refusal, request-for-structure, and overdelegation machine-readable
instead of collapsing them into one generic “negative case.”

## Receipt Surface

The package emits one shared plugin benchmark receipt with:

- route accuracy
- unsupported-capability refusal accuracy
- request-for-structure accuracy
- overdelegation rejection accuracy

That keeps unsupported capability, missing input, and overdelegation as
separate scored dimensions.

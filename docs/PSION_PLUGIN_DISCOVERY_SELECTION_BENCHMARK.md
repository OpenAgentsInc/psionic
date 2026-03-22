# PSION Plugin Discovery And Selection Benchmark

> Status: canonical `PSION_PLUGIN-8` discovery-and-selection benchmark package
> for the bounded host-native plugin tranche, written 2026-03-22 after landing
> the first repo-owned package and receipt on top of the shared
> `psion_plugin` benchmark contract.

This document freezes the first benchmark family for plugin discovery and
selection.

The package is benchmark-authored on purpose.

The current held-out trace substrate does not yet carry every route class that
the discovery benchmark needs.

So the package keeps authored prompt provenance explicit instead of pretending
those rows came directly from the current held-out training split.

## Canonical Artifacts

- `docs/PSION_PLUGIN_DISCOVERY_SELECTION_BENCHMARK.md` is the canonical
  human-readable contract.
- `crates/psionic-train/src/psion_plugin_discovery_selection_benchmark.rs`
  owns the package and receipt builders.
- `crates/psionic-train/examples/psion_plugin_discovery_selection_benchmark.rs`
  writes the canonical bundle.
- `fixtures/psion/benchmarks/psion_plugin_discovery_selection_benchmark_v1/`
  carries the first committed bundle.

## Coverage

The first package covers:

- direct-answer versus plugin-delegate routing
- single-plugin selection
- multi-plugin sequence selection
- wrong-tool negative cases
- unsupported-tool negative cases

This is enough to close the first discovery-and-selection package without
claiming broader argument, sequencing, refusal, or interpretation coverage.

## Provenance Boundary

Every item cites:

- the shared contamination bundle ref and digest
- one explicit authored benchmark prompt ref
- zero runtime receipt requirements when the item is route-only selection logic

That keeps the benchmark honest about what is benchmark-authored versus what is
derived from held-out runtime lineage.

## Receipt Surface

The package emits one shared plugin benchmark receipt with:

- route accuracy
- selection accuracy
- wrong-tool rejection accuracy
- unsupported-tool refusal accuracy

That keeps selection and route evidence visible without collapsing wrong-tool
and unsupported-capability outcomes into one score.

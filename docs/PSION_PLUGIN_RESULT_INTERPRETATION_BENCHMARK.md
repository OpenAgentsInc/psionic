# PSION Plugin Result Interpretation Benchmark

> Status: canonical `PSION_PLUGIN-12` result-interpretation benchmark package
> for the bounded host-native plugin tranche, written 2026-03-22 after landing
> the first repo-owned receipt-backed interpretation package and receipt on top
> of the shared `psion_plugin` benchmark contract.

This document freezes the first benchmark family for post-plugin result
interpretation.

The package is receipt-backed on purpose.

The first interpretation package should score execution-backed versus inferred
statement boundaries honestly, so every item is anchored to held-out runtime
receipts instead of benchmark-authored fake execution outputs.

## Canonical Artifacts

- `docs/PSION_PLUGIN_RESULT_INTERPRETATION_BENCHMARK.md` is the canonical
  human-readable contract.
- `crates/psionic-train/src/psion_plugin_result_interpretation_benchmark.rs`
  owns the package and receipt builders.
- `crates/psionic-train/examples/psion_plugin_result_interpretation_benchmark.rs`
  writes the canonical bundle.
- `fixtures/psion/benchmarks/psion_plugin_result_interpretation_benchmark_v1/`
  carries the first committed bundle.

## Coverage

The first package covers:

- interpretation of a successful execution-backed URL-extraction result
- interpretation of a later typed fetch refusal in the same held-out trace
- honest continuation after refusal without inventing hidden retries or
  successful unseen execution

This is enough to close the first interpretation package without claiming
broader guest-artifact interpretation or controller-specific post-processing
behavior.

## Evidence Boundary

Each item now preserves:

- one held-out parent-lineage row from the contamination bundle
- the source-case id from that row
- the exact receipt refs whose outputs or refusals must be interpreted
- execution-backed versus inferred boundary as an explicit graded contract
- whether continuation after refusal/failure is part of the task

That keeps interpretation tied to receipt-backed truth instead of turning it
into generic summarization.

## Receipt Surface

The package emits one shared plugin benchmark receipt with:

- interpretation score
- execution-backed boundary accuracy
- typed runtime-refusal accuracy

That keeps post-plugin reasoning quality, evidence discipline, and refusal
handling separate in the scoring surface.

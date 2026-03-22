# Psion Benchmark Packages

> Status: canonical `PSION-19` / `#375` benchmark-package contract, written
> 2026-03-22 after landing the first shared `Psion` benchmark schema and
> receipt set.

This document freezes the first shared package contract for `Psion`
benchmarks.

It does not finalize every benchmark threshold and it does not claim every
benchmark family is already green. It defines the package shapes that later
benchmark sets, aggregate receipts, and acceptance-matrix decisions must all
share.

Label-production provenance now lives in
`docs/PSION_BENCHMARK_LABEL_GENERATION.md`. This document stays focused on
package shape, not on how exact labels, rubric-backed labels, or derived-data
lineage receipts are produced.

The first family-specific package contract now also has a dedicated doc:
`docs/PSION_ARCHITECTURE_REASONING_BENCHMARK.md` freezes the typed
architecture item coverage and direct acceptance binding for
`psion_architecture_reasoning_benchmark_v1`.

`docs/PSION_NORMATIVE_SPEC_READING_BENCHMARK.md` now does the same for
`psion_normative_spec_benchmark_v1`, including the grounded-reading boundary
that keeps normative source text separate from later engineering inference.

`docs/PSION_ENGINEERING_SPEC_INTERPRETATION_BENCHMARK.md` does the same for
`psion_engineering_spec_benchmark_v1`, including the typed probe coverage that
keeps implementation implications, ambiguity, unspecified regions, and
portability risks separate from normative source reading.

`docs/PSION_MEMORIZATION_VS_REASONING_PROBES.md` now does the same for
`psion_memorization_reasoning_benchmark_v1`, including the typed probe
coverage that keeps recombination under altered constraints, paraphrases,
historical transfer, and spec-adjacent edge cases separate from stock recall.

`docs/PSION_ROUTE_CLASS_EVALUATION.md` now does the same for
`psion_route_benchmark_v1`, including the four explicit route classes and the
route receipt that keeps delegation, uncertainty, and structured-input requests
separate.

## Canonical Artifacts

- `crates/psionic-train/src/psion_benchmark_packages.rs` owns the shared
  benchmark package contract, grader interfaces, catalog, and receipt set.
- `crates/psionic-train/examples/psion_benchmark_package_fixtures.rs`
  regenerates the canonical package fixtures.
- `fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json` is the canonical
  package catalog spanning the main benchmark families.
- `fixtures/psion/benchmarks/psion_benchmark_receipt_set_v1.json` is the
  canonical receipt set for those packages.

The stable schema versions are:

- `psion.benchmark_package_contract.v1`
- `psion.benchmark_package_receipt.v1`
- `psion.benchmark_catalog.v1`
- `psion.benchmark_receipt_set.v1`

## What The Contract Standardizes

The shared contract now freezes:

- one common outer package shape over the generic `psionic-eval`
  `BenchmarkPackage`
- one shared prompt-format surface with explicit response-shape contracts
- one shared item schema with family-specific task payloads
- one typed grader interface surface that admits exact-label, rubric-backed,
  exact-route, and exact-refusal grading
- one contamination-input bundle per package, tied directly to the held-out and
  benchmark-isolation manifests
- one acceptance-ready receipt shape that can emit
  `PsionBenchmarkEvidenceReceipt` for the families that feed the acceptance
  matrix

The canonical catalog proves that the same contract shape covers:

- architecture reasoning
- normative spec reading
- engineering spec interpretation
- memorization-versus-reasoning probes
- route evaluation
- refusal evaluation

## Mechanical Enforcement

`psionic-train` now validates that:

- benchmark packages still match the generic `psionic-eval` benchmark package
  contract underneath them
- item ids still match the generic benchmark-case ids exactly
- every item still references a declared prompt format and grader interface
- route items cannot silently use freeform explanation prompts or generic
  graders
- refusal items cannot silently use route or reasoning graders
- package contamination inputs still point only at benchmark-visible and
  held-out source ids allowed by the benchmark-isolation manifest
- package receipts still expose the metric kinds the acceptance matrix needs
  for their family, such as route accuracy or refusal plus over-refusal
- families that map into the acceptance matrix still emit a matching
  `PsionBenchmarkEvidenceReceipt` instead of a custom one-off receipt shape

## Why This Matters

This closes the benchmark-contract step for the learned-model lane:

- later benchmark families no longer need to invent incompatible item schemas
- exact and rubric-backed grading now share one typed surface
- contamination review inputs sit inside the package model instead of being
  implied elsewhere
- acceptance-matrix benchmark evidence can be produced from one standardized
  receipt shape rather than a family-specific ad hoc format

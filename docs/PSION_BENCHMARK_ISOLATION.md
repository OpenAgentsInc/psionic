# Psion Benchmark Isolation

> Status: implemented_early as of 2026-03-22 via `PSION-4` / issue `#360`.

This document freezes the first contamination-control contract for the `Psion`
learned-model lane.

It defines:

- held-out and exclusion manifests keyed by stable source ids
- tokenizer-exposure reporting that stays distinct from model-training exposure
- section-range disjointness rules for the main source families
- required near-duplicate review checkpoints
- contamination consequences that invalidate affected benchmark claims instead
  of quietly weakening them

## Canonical Artifacts

- machine-readable isolation contract:
  `crates/psionic-data/src/psion_benchmark_isolation.rs`
- canonical exclusion manifest:
  `fixtures/psion/isolation/psion_exclusion_manifest_v1.json`

The stable schema version is `psion.benchmark_isolation.v1`.

## Loader Surfaces

The first isolation contract treats three loader surfaces separately:

- `tokenizer_training`
- `model_training`
- `benchmark_package`

`psionic-data` now exposes mechanical rejection helpers so these surfaces can
refuse excluded source ids instead of only logging them.

## Held-Out And Exclusion Policy

The first manifest keeps three explicit source-id sets:

- `held_out_source_ids`
- `training_excluded_source_ids`
- `benchmark_excluded_source_ids`

Held-out sources must also be training-excluded. This keeps benchmark and
held-out integrity explicit before tokenizer or dataset expansion proceeds.

## Tokenizer Exposure Reporting

Every source now carries a tokenizer-exposure record with:

- `tokenizer_exposed`
- `model_training_exposed`
- `benchmark_exposed`
- `detail`

This keeps `tokenizer_only` exposure and benchmark-only exposure separate from
model-training exposure, instead of pretending every held-out or benchmark
source is fully unseen once tokenization begins.

## Section-Range Disjointness

The first documented family rules are:

- `textbook -> chapter_section_disjoint`
- `specification -> chapter_section_disjoint`
- `manual -> page_range_disjoint`
- `paper -> page_range_disjoint`
- `technical_documentation -> entire_source_disjoint`

These rules are bound to explicit `boundary_kind` requirements such as chapter,
page, or record anchors. Later work may add more granular rules, but later work
may not erase the need for stable boundaries.

## Near-Duplicate Review

The first contract requires near-duplicate review:

- before model-training datasets are frozen
- before benchmark publication

This stays machine-legible in the manifest rather than living only in reviewer
folklore.

## Contamination Consequences

A contamination violation now has explicit minimum consequences:

- invalidate the affected benchmark
- trigger capability-matrix review
- trigger benchmark rebuild review

That keeps benchmark integrity connected to later capability publication rather
than treating contamination as a minor note.

## Mechanical Enforcement

`psionic-data` now exposes:

- `PsionExclusionManifest`
- `PsionLoaderSurface`
- `PsionTokenizerExposureRecord`
- `PsionSectionRangeDisjointnessRule`
- `PsionBenchmarkIsolationError`

The validation and loader path rejects:

- exclusion manifests that do not cover every lifecycle source with a tokenizer
  exposure record
- held-out sources that are not also training-excluded
- missing section-range rules for textbooks, specifications, manuals, or papers
- invalid exposure combinations such as restricted or evaluation-only sources
  being marked `model_training_exposed=true`
- source ids rejected by tokenizer, training, or benchmark loader policy

This gives the repo one explicit isolation boundary before tokenizer training
and benchmark publication expand under later `PSION-*` work.

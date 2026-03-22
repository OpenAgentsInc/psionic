# Psion Raw-Source Ingestion

> Status: canonical `PSION-7` / `#363` raw-source ingestion contract, written
> 2026-03-22 after landing the admission and lifecycle governance docs.

This document freezes the first repo-owned raw-source ingestion contract for the
`Psion` corpus.

It sits between governance and later tokenizer or training artifacts.

It is about reproducible import, normalization lineage, manifest emission, and
boundary preservation for reviewed sources.

## Canonical Artifacts

- `crates/psionic-data/src/psion_raw_source_ingestion.rs` owns the typed raw
  ingestion contract.
- `fixtures/psion/ingestion/psion_raw_source_manifest_v1.json` is the canonical
  machine-readable manifest example.

The stable schema version is `psion.raw_source_manifest.v1`.

## What The Manifest Carries

Each ingested source row now carries:

- `source_id`
- `source_family_id`
- `source_kind`
- current rights posture and lifecycle state at ingest time
- one raw source digest bound back to admission review
- one normalized source digest for the ingest output
- one or more imported documents with import references and per-document digests
- preserved section, page, file, or record boundaries

The manifest also carries one normalization profile with:

- `preprocessing_version`
- explicit normalization steps
- explicit document-boundary preservation
- explicit section-boundary preservation

## Boundary Contract

Raw ingestion must preserve the boundary kind already reviewed during admission.

That means:

- chapter-section sources stay chapter-section anchored
- page-range sources stay page-range anchored
- record-anchored sources stay record anchored

The raw manifest is rejected if normalization weakens that boundary information.

## Separation From Later Stages

This manifest is intentionally separate from:

- tokenizer manifests
- tokenized corpora
- benchmark packages
- training-stage manifests

Later issues can build on top of raw ingestion, but they do not get to replace
it with implicit preprocessing behavior or ad hoc notebook steps.

## Mechanical Enforcement

`psionic-data` now validates that:

- only admitted, restricted, or evaluation-only lifecycle states are ingested
- raw digests still match the admitted source digest
- source-family ids, source kinds, rights posture, and lifecycle state do not
  drift from reviewed truth
- normalization profiles preserve both document and section boundaries
- document and section order stays stable
- preserved section boundaries keep the reviewed boundary kind

That gives later tokenizer, dataset, and benchmark work one explicit raw-source
substrate instead of policy folklore or one-off preprocessing scripts.

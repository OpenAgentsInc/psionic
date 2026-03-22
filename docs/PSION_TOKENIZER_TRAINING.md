# Psion Tokenizer Training

> Status: canonical `PSION-8` / `#364` tokenizer manifest and artifact-bundle
> contract, written 2026-03-22 after landing raw-source ingestion and
> contamination-control contracts.

This document freezes the first tokenizer-training manifest and artifact bundle
for the `Psion` lane.

It is not model training.

It is the explicit provenance surface for building one reproducible tokenizer
artifact from admitted raw sources under the held-out isolation rules.

## Canonical Artifacts

- `crates/psionic-data/src/psion_tokenizer_training.rs` owns the typed manifest
  and artifact-bundle contracts.
- `fixtures/psion/tokenizer/psion_tokenizer_training_manifest_v1.json` is the
  canonical training-manifest example.
- `fixtures/psion/tokenizer/psion_tokenizer_artifact_bundle_v1.json` is the
  canonical artifact-bundle example.

The stable schema versions are:

- `psion.tokenizer_training_manifest.v1`
- `psion.tokenizer_artifact_bundle.v1`

## What The Manifest Records

The tokenizer-training manifest now freezes:

- tokenizer id and version
- raw-source and exclusion-manifest schema versions
- inherited preprocessing version from raw ingestion
- tokenizer config
- admitted source list
- excluded source list
- exposure audit rows for every raw ingested source

This makes tokenizer provenance explicit before any model checkpoint depends on
it.

## Tokenizer-Only Exposure

Tokenizer exposure is not the same thing as model-training exposure.

The manifest therefore keeps:

- `tokenizer_exposed`
- `tokenizer_only_exposure`
- `model_training_exposed`

on the same audit row.

That keeps restricted sources, such as tokenizer-only manuals, visible to later
audits instead of disappearing into one vague preprocessing step.

## Built Artifact Bundle

The artifact bundle derived from the manifest carries:

- tokenizer config digest
- tokenizer digest
- vocabulary artifact inventory
- admitted and excluded source lineage
- the same exposure audit rows carried by the manifest

Later model and dataset artifacts can bind to this tokenizer digest instead of
to an ad hoc experiment name.

## Mechanical Enforcement

`psionic-data` now validates that:

- admitted tokenizer sources are actually allowed on the tokenizer-training
  loader surface
- admitted and excluded source lists cover every raw-source row
- excluded sources stay explicit
- tokenizer-only exposure remains distinguishable from model-training exposure
- the built artifact bundle keeps the manifest lineage and version bindings

That gives the `Psion` lane one explicit tokenizer boundary before tokenized
datasets or model checkpoints expand on top of it.

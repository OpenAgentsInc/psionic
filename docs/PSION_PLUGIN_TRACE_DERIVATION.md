# PSION Plugin-Trace Normalization And Derivation

> Status: canonical `PSION_PLUGIN-4` derivation contract for normalizing the
> committed multi-plugin trace corpus into canonical plugin-training records,
> written 2026-03-22 after landing the first repo-owned derivation bundle in
> `psionic-data`.

This document freezes the first derivation path from real plugin-runtime traces
to the canonical Psion plugin-training record schema.

It does not define a second trace format.

It consumes the existing lane-neutral multi-plugin trace corpus and emits the
canonical training-record schema from `docs/PSION_PLUGIN_TRAINING_RECORD_SCHEMA.md`.

## Canonical Artifacts

- `docs/PSION_PLUGIN_TRACE_DERIVATION.md` is the canonical human-readable
  derivation contract.
- `crates/psionic-data/src/psion_plugin_training_derivation.rs` owns the
  typed derivation bundle, runtime-drift checks, and write helpers.
- `crates/psionic-data/examples/psion_plugin_training_derivation.rs` writes
  the canonical derived bundle.
- `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md` remains the canonical source
  corpus contract.

The stable derivation bundle schema version is
`psionic.psion.plugin_training_derivation_bundle.v1`.

## Current Sources

The current derivation lane consumes the already-committed multi-plugin trace
corpus built from:

- deterministic workflow traces
- router-owned `/v1/responses` plugin-loop traces
- local Apple FM plugin-session traces
- one deterministic-only digest-bound guest-artifact workflow trace

The derivation path does not special-case those lanes into separate training
schemas.

It normalizes them into one shared `PsionPluginTrainingRecord` surface while
preserving:

- controller surface id
- source bundle identity
- source case identity
- receipt refs and digests
- plugin class and runtime schema identity
- guest-artifact replay and digest linkage where present

## Drift Rejection

Derivation currently validates every projected tool row against the live
starter-plugin registration table before it emits training records.

That means derivation fails closed when the source trace corpus drifts from:

- plugin id
- tool name
- success-result schema id
- refusal schema ids
- replay class id

This is the main guarantee of `PSION_PLUGIN-4`:

the repo reuses runtime truth instead of exporting one stale controller-only
snapshot.

## Output Shape

The derivation bundle carries:

- one source corpus ref
- one source corpus digest
- one per-surface source-record count
- one ordered set of derived training records
- one stable bundle digest

The current bundle is the minimal bridge between:

- trace corpus
- dataset builder
- later benchmark and training stages

without inventing a second intermediate schema.

The current host-native dataset builder still filters the guest-artifact
workflow out explicitly. Derivation is broader than the first dataset on
purpose.

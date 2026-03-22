# PSION Plugin-Training Record Schema

> Status: canonical `PSION_PLUGIN-3` plugin-training record schema for the
> `Psion x Tassadar` convergence tranche, written 2026-03-22 after landing the
> first typed `psionic-data` contract for plugin-conditioned training records.

This document freezes the first machine-checkable record format that later
The Psion plugin derivation, dataset, benchmark, and training stages must reuse
this schema.

The purpose of the schema is simple:

- take real plugin-runtime truth
- keep controller-surface and plugin-class context attached
- preserve receipt linkage and replay posture
- make the result directly reusable for training and evaluation

It is not a free-form tool-call transcript format.

It is also not a controller-specific export.

## Canonical Artifacts

- `docs/PSION_PLUGIN_TRAINING_RECORD_SCHEMA.md` is the canonical
  human-readable contract.
- `crates/psionic-data/src/psion_plugin_training_record.rs` owns the typed
  schema, validator, and stable digest logic.
- `crates/psionic-data` re-exports the typed schema for later derivation and
  dataset builders.

The stable schema version is
`psionic.psion.plugin_training_record.v1`.

## Record Shape

Each record carries:

- one stable record id
- one task prompt or directive
- one admitted plugin set
- one controller-surface context block
- zero or more plugin invocation rows
- one route label
- one outcome label
- optional final response text
- one stable digest

The current route labels are:

- `answer_in_language`
- `delegate_to_admitted_plugin`
- `request_missing_structure_for_plugin_use`
- `refuse_unsupported_plugin_or_capability`

The current outcome labels are:

- `completed_success`
- `typed_runtime_refusal`
- `request_missing_structure`
- `refused_unsupported_capability`
- `runtime_failure`

## Admitted Plugin Set

The admitted plugin set is not optional context.

Each record must carry explicit rows for the plugins that were admissible for
that task, including:

- plugin id
- tool name
- plugin class
- capability class
- origin class
- input schema id
- result schema id
- refusal schema ids
- replay class id

That is the minimum information required to keep training examples tied to the
real plugin-runtime envelope instead of flattening them into one vague “tools
available” flag.

## Controller Context

The controller context keeps controller-specific provenance visible without
forking the schema by lane.

The current schema supports these controller surfaces:

- `deterministic_workflow`
- `router_responses`
- `apple_fm_session`
- `weighted_controller`
- `synthetic_validation`

Each record also keeps the source bundle ref, source bundle id, source bundle
digest, source case id, and optional workflow case id.

## Invocation Rows

Invocation rows are where receipt linkage becomes non-negotiable.

Each invocation row carries:

- invocation id
- controller decision ref
- plugin id
- tool name
- arguments payload
- receipt ref
- receipt digest
- invocation status
- optional result payload
- optional refusal schema id

Validation requires every invocation row to target a plugin already present in
the admitted plugin set.

That keeps later training and eval work from laundering unknown plugin rows
into the learned lane.

## Validation Rules

The validator currently enforces:

- schema version must match exactly
- record id, directive text, and detail must be non-empty
- admitted plugin rows must be present and unique by plugin id and tool name
- admitted plugin rows must have non-empty schema and replay fields
- invocation rows must target admitted plugins
- `delegate_to_admitted_plugin` records must contain at least one invocation
- refusal rows may not imply hidden execution without a receipt-backed
  invocation row
- final response text is required only for `completed_success`

That is enough structure for the next issue to build the real derivation
pipeline without inventing another schema.

# Compiled Agent Promoted Artifact Contract

> Status: first runtime-consumable compiled-agent artifact contract, updated 2026-03-29.

## Why This Exists

The compiled-agent learning loop is not product-authoritative until a runtime can
load a promoted artifact set without guessing which revisions are live,
candidate-only, or rollback-safe.

This contract closes that seam for the first compiled-agent graph.

It gives runtime consumers one retained object that answers:

- which artifact is promoted for each module slot
- which candidate labels are available for shadow comparison
- which artifact a clean rollback should restore
- which default learned row the artifacts were validated against
- which validator report and XTRAIN receipt justified promotion

## Canonical Fixture

- `fixtures/compiled_agent/compiled_agent_promoted_artifact_contract_v1.json`

Regenerate it with the rest of the bounded learning loop:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

## Schema Shape

Top-level fields:

- `schema_version`
- `ledger_id`
- `row_id`
- `evidence_class`
- `promoted_entry_count`
- `candidate_entry_count`
- `entries_by_module`
- `artifacts`
- `summary`
- `contract_digest`

Each artifact entry records:

- module slot identity
- stable module and signature names
- implementation family, label, and version
- lifecycle state: `promoted` or `candidate`
- optional candidate label
- compatibility version for the first graph runtime
- confidence floor
- artifact id and digest
- default learned row identity
- evidence class
- validator lineage
- predecessor and rollback artifact ids when applicable
- typed payload for either:
  - a retained revision set
  - a retained learned route model artifact

## Current Truth

The first retained contract carries:

- promoted route authority backed by `compiled_agent.route.multinomial_nb_v1`
- rollback-safe route fallback under `last_known_good`
- promoted baseline artifacts for:
  - `tool_policy`
  - `tool_arguments`
  - `grounded_answer`
  - `verify`
- candidate artifacts under `psionic_candidate` for:
  - `grounded_answer`
  - `verify`

This is intentionally narrow. The contract is not trying to describe every
future compiled-agent graph or every future module family yet.

## Runtime Expectations

Consumers should treat this contract as machine truth, not as a hint.

For the first graph:

- promoted authority loads from `lifecycle_state = promoted`
- shadow comparison loads from `lifecycle_state = candidate` plus the requested
  `candidate_label`
- rollback should route to `rollback_artifact_id` when present
- every artifact entry in the current contract stays inside `learned_lane`
- receipts should retain:
  - artifact id
  - artifact digest
  - manifest id
  - candidate label when used

Consumers must not merge later stronger-evidence artifacts into this learned
contract implicitly. If a future stronger-evidence lane is introduced, it needs
its own explicit contract rows and validator lineage rather than retroactively
changing the meaning of this learned-lane promotion surface.

## Honest Boundary

This contract is now retained in `psionic`, but `openagents` still needs to
consume it as the live compiled-agent authority surface. That runtime adoption
step is separate and deliberate.

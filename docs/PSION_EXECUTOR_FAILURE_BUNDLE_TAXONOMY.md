# Psion Executor Failure Bundle Taxonomy

> Status: canonical `PSION-0604` / `#750` record, updated 2026-03-30 after
> freezing the first admitted failure-bundle taxonomy for the executor lane.

This document records the first canonical failure-bundle taxonomy for the
admitted executor lane. It turns long-run failure posture into one retained
packet instead of leaving incident classes implicit across logs, review notes,
and operator memory.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_failure_bundle_taxonomy_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_failure_bundle_taxonomy_fixtures
```

## What Landed

`psionic-train` now owns one typed failure-bundle taxonomy packet that binds:

- the mandatory live-metrics packet
- the continue-vs-restart incident policy
- the weekly local-cluster review workflow
- seven canonical failure-bundle types
- two retained emitted bundle rows for the admitted MLX and 4080 ledger rows

The canonical bundle types are now frozen as:

- `optimizer_failure`
- `batch_failure`
- `dataloader_stall`
- `memory_pressure`
- `thermal_anomaly`
- `slow_node_behavior`
- `topology_failure`

That means the admitted executor lane now has one explicit answer to:

- which failure classes count as canonical long-run bundle types
- who owns the default response posture for each class
- which continue-vs-restart posture each class implies
- which current ledger rows actively emit a failure-bundle posture into review

## Current Retained Truth

- packet digest:
  `167de1726490b46baa6ab8dba39f1a10d19bec10180614bc2c72e515596ab0aa`
- live-metrics digest:
  `ed90d86a315b4c37427dcbc4353f6113cacc89862ffdd3ceefa8a7161c2d04c6`
- incident-policy digest:
  `40ef44b2c92651ab7e5ae3c7c039388bf472ca05467fc679054ffe4fb8413186`
- review-workflow digest:
  `c11b48bb9cb4381ccba810b5c154ffad6014c3b130c539a32b43dff4298078bf`
- taxonomy row count:
  `7`
- MLX emitted bundle row:
  `psion_executor_local_cluster_ledger_row_mlx_v1 ->
  no_active_failure_bundle -> green_no_failure_bundle`
- 4080 emitted bundle row:
  `psion_executor_local_cluster_ledger_row_4080_v1 ->
  slow_node_behavior -> watch_bundle_emitted`
- emitted review requirement:
  `review_references_current_bundle_type`

## Honest Current Meaning

The taxonomy does not claim the admitted lane is failure-free.

It does something stricter and more useful:

- the MLX candidate row explicitly emits no active failure bundle
- the retained 4080 current-best row explicitly emits
  `slow_node_behavior`
- weekly review now has to cite that active bundle posture directly instead of
  treating slower-node symptoms as informal commentary
- incident handling and review posture are now bound to the same failure-class
  vocabulary

That is the main closeout: long-run failure posture is now retained as typed
machine truth instead of getting flattened into generic "run unhealthy"
language.

## Follow-On Surfaces

The live-metrics packet that feeds this taxonomy now lives at:

- `docs/PSION_EXECUTOR_MANDATORY_LIVE_METRICS.md`
- `fixtures/psion/executor/psion_executor_mandatory_live_metrics_v1.json`

The incident-policy packet that governs continue-vs-restart posture now lives
at:

- `docs/PSION_EXECUTOR_CONTINUE_RESTART_POLICY.md`
- `fixtures/psion/executor/psion_executor_continue_restart_policy_v1.json`

The weekly review workflow that consumes emitted bundle posture now lives at:

- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_REVIEW_WORKFLOW.md`
- `fixtures/psion/executor/psion_executor_local_cluster_review_workflow_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_failure_bundle_taxonomy_fixtures`
- `cargo test -q -p psionic-train psion_executor_failure_bundle_taxonomy -- --nocapture`

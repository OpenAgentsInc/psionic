# Psion Run Observability

> Status: canonical `PSION-13` / `#369` run-observability contract, written
> 2026-03-22 after landing the explicit Psion pretrain stage.

This document freezes the first structured observability surface for `Psion`
pretraining runs.

It keeps cost, throughput, checkpoint size, hardware topology, and instability
markers in typed receipts instead of ad hoc operator notes.

## Canonical Artifacts

- `crates/psionic-train/src/psion_run_observability.rs` owns the typed
  pretrain-run observability receipts and the stage-level summary artifact.
- `crates/psionic-train/examples/psion_run_observability_fixtures.rs`
  regenerates the canonical observability fixtures.
- `fixtures/psion/observability/psion_pilot_pretrain_run_observability_receipt_v1.json`
  is the canonical pilot-run observability receipt.
- `fixtures/psion/observability/psion_broader_pretrain_run_observability_receipt_v1.json`
  is the canonical broader-pretraining observability receipt.
- `fixtures/psion/observability/psion_pretrain_stage_observability_summary_v1.json`
  is the canonical stage-level summary that compares the pilot and broader
  pretraining receipts directly.

The stable schema versions are:

- `psion.pretrain_run_observability_receipt.v1`
- `psion.pretrain_stage_observability_summary.v1`

The provider-neutral app-facing remote-training contract is separate from the
stage-level observability receipts:

- `crates/psionic-train/src/remote_training_visualization.rs` owns the typed
  shipped `v1` live bundle and run-index contract.
- `crates/psionic-train/src/remote_training_visualization_v2.rs` owns the
  track-aware `v2` follow-on contract.
- `crates/psionic-train/examples/remote_training_visualization_fixtures.rs`
  regenerates the canonical visualization fixtures.
- `fixtures/training_visualization/psion_google_summary_only_remote_training_visualization_bundle_v1.json`
  is the canonical summary-only example.
- `fixtures/training_visualization/parameter_golf_live_remote_training_visualization_bundle_v1.json`
  is the canonical full-series always-live example.
- `fixtures/training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json`
  is the canonical distributed always-live example.
- `fixtures/training_visualization/remote_training_run_index_v1.json` is the
  canonical run-index example.
- `fixtures/training_visualization/remote_training_run_index_v2.json` is the
  canonical track-aware run-index example used during the `v1` -> `v2`
  migration window.

## What The Receipt Freezes

The first observability receipt now binds:

- one explicit pretrain-stage receipt digest from `PSION-12`
- run-level cost breakdown in micro-USD
- token-rate, sequence-rate, step-latency, and checkpoint-write throughput
- promoted-checkpoint artifact size reporting
- realized delivered-execution and hardware-topology facts
- aggregated instability telemetry and structured policy markers when present

That gives later pilot-review and broader-pretraining decisions a repo-owned
artifact surface instead of screenshots or remembered logs.

## Mechanical Enforcement

`psionic-train` now validates that:

- every observability receipt still points to one exact pretrain-stage receipt
- cost components sum exactly to the declared total
- throughput fields stay non-zero and internally coherent
- checkpoint artifact identity still matches the promoted checkpoint from the
  stage receipt
- multi-device runs surface an explicit topology instead of hiding device shape
- instability markers map back to surfaced telemetry when a stability verdict
  exists
- the stage-level summary contains both the `pilot` and
  `broader_pretraining` rows and each row still matches the underlying run
  receipt

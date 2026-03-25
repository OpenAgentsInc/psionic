# Remote Training Visualization

This document freezes the first provider-neutral app-facing contract for remote
training runs.

The contract exists so Autopilot can render Google Cloud and RunPod runs from
one typed artifact family without scraping provider-shaped logs in pane code.

Psionic owns the machine-facing truth.

Autopilot owns rendering, refresh loops, and pane behavior.

## Canonical Artifacts

- `crates/psionic-train/src/remote_training_visualization.rs` owns the typed
  bundle and run-index contract.
- `crates/psionic-train/examples/remote_training_visualization_fixtures.rs`
  regenerates the canonical example fixtures.
- `fixtures/training_visualization/psion_google_summary_only_remote_training_visualization_bundle_v1.json`
  is the canonical summary-only example bundle.
- `fixtures/training_visualization/psion_google_live_remote_training_visualization_bundle_v1.json`
  is the canonical always-live Google single-node example bundle.
- `fixtures/training_visualization/parameter_golf_live_remote_training_visualization_bundle_v1.json`
  is the canonical full always-live example bundle.
- `fixtures/training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json`
  is the canonical RunPod distributed post-run example bundle.
- `fixtures/training_visualization/remote_training_run_index_v1.json` is the
  canonical run-index example.

The stable schema versions are:

- `psionic.remote_training_visualization_bundle.v1`
- `psionic.remote_training_run_index.v1`

## What The Bundle Freezes

The first bundle freezes one provider-neutral shape for:

- provider, profile, lane, run, and repo revision identity
- one explicit refresh contract with one-second active-run cadence semantics
- one explicit `series_status` plus `series_unavailable_reason`
- timeline rows for lifecycle boundaries
- heartbeat rows that explain what the runtime is doing now
- loss rows for the primary chartable training curve
- bounded optimizer and model-math diagnostics
- runtime pipeline timings
- GPU telemetry
- distributed telemetry when a run is multi-rank
- typed event rows
- provenance rows for the retained source artifacts

The contract is explicit about missing data.

If a lane only has summary truth and GPU samples, the bundle must say that
directly instead of inventing a loss curve.

## What The Run Index Freezes

The first run index freezes one discovery surface that can enumerate:

- summary-only lanes
- full-series lanes
- active runs
- completed runs
- refused runs
- rehearsal-only runs

That keeps the app out of provider-root walking and ad hoc manifest discovery.

## Live Requirement

The bundle is designed for a viewer that must stay visibly truthful every
second during an active run.

That means:

- active runs must not use `post_run_only` emission
- active runs must target `1000` ms or faster app refresh
- heartbeat freshness and stale-state posture remain explicit
- summary-only lanes stay honest about what they cannot yet show

The contract does not require raw tensor dumps or per-parameter state.

It requires bounded metrics that stay truthful and cheap enough to retain.

The repo now also ships a provider-neutral final evidence bundle family in
`crates/psionic-train/src/training_execution_evidence_bundle.rs`. Visualization
refs from this document are one typed section inside that final evidence family,
not a sidecar proof surface.

## Parameter Golf Single-H100

The first live lane now lands directly from the Rust-owned single-H100 trainer.

When the trainer runs with remote-training metadata in its environment, it
writes and rewrites these app-facing artifacts under the report root:

- `training_visualization/parameter_golf_single_h100_remote_training_visualization_bundle_v1.json`
- `training_visualization/remote_training_run_index_v1.json`

That lane now preserves:

- one-second heartbeat updates while the trainer is active
- retained per-step loss, optimizer-math, and runtime series from the trainer
  report
- local GPU samples from the training host when `nvidia-smi` is available
- explicit partial-series fallback from the retained trainer log when the
  trainer JSON never lands

The Google and RunPod single-H100 finalizers now re-materialize the same typed
bundle and run index at closeout instead of inventing provider-specific JSON.

## Google Single-Node Psion

The accelerated Google single-node Psion lanes now write their own provider-neutral
live bundle under the run output directory while the trainer is active.

The retained path is:

- `training_visualization/psion_google_single_node_remote_training_visualization_bundle_v1.json`
- `training_visualization/remote_training_run_index_v1.json`
- `training_visualization/snapshots/*.json`

The accelerated reference lane and the plugin-conditioned accelerated lane now
retain:

- one-second heartbeats while the trainer is active
- per-step train-loss samples plus retained validation checkpoints
- bounded optimizer math derived from the trainer step receipts
- runtime timings derived from the real CUDA batch, optimizer, and model materialization path
- local GPU samples from the training host when `nvidia-smi` is available
- checkpoint refs plus receipt and artifact provenance after the lane seals

The summary-only Google example fixture still exists because non-accelerated or
historical lanes may lack chartable series.

## RunPod Parameter Golf Distributed 8xH100

The RunPod `8xH100` Parameter Golf lane now seals the same provider-neutral
bundle family under the run root when the operator finalizer closes.

The retained paths are:

- `training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json`
- `training_visualization/remote_training_run_index_v1.json`
- `training_visualization/snapshots/finalized_bundle.json`

The current lane is still explicit about its boundary:

- the finalizer mirrors the retained `distributed_challenge_receipt` into the
  run root when one is only embedded inside the submission evidence report
- the bundle preserves GPU inventory, topology capture, exported-folder
  provenance, and any retained distributed receipt facts
- the bundle stays `post_run_only` today because this lane still lacks a
  coordinator-owned one-second live writer
- `series_status` remains `unavailable` until the lane retains a real loss
  curve or equivalent primary training series
- refusal posture stays explicit when the retained receipt is still a
  measurements-missing or inventory-mismatch refusal

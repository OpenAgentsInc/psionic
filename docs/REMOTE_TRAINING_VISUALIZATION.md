# Remote Training Visualization

This document freezes the provider-neutral app-facing contract family for
remote training runs.

The contract exists so Autopilot can render Google Cloud, RunPod, PGOLF,
HOMEGOLF, and bounded XTRAIN runs from typed retained artifacts instead of
scraping provider-shaped logs in pane code.

Psionic owns the machine-facing truth.

Autopilot owns rendering, refresh loops, and pane behavior.

## Canonical Artifacts

- `crates/psionic-train/src/remote_training_visualization.rs` owns the shipped
  `v1` bundle and run-index contract.
- `crates/psionic-train/src/remote_training_visualization_v2.rs` owns the
  track-aware `v2` follow-on bundle and run-index contract.
- `crates/psionic-train/examples/remote_training_visualization_fixtures.rs`
  regenerates the canonical example fixtures.
- `fixtures/training_visualization/psion_google_summary_only_remote_training_visualization_bundle_v1.json`
  is the canonical summary-only example bundle.
- `fixtures/training_visualization/psion_google_live_remote_training_visualization_bundle_v1.json`
  is the canonical always-live Google single-node example bundle.
- `fixtures/training_visualization/parameter_golf_live_remote_training_visualization_bundle_v1.json`
  is the canonical full always-live example bundle.
- `fixtures/training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json`
  is the canonical RunPod distributed always-live example bundle.
- `fixtures/training_visualization/remote_training_run_index_v1.json` is the
  canonical run-index example.
- `fixtures/training_visualization/psion_google_summary_only_remote_training_visualization_bundle_v2.json`
  is the canonical summary-only `v2` example bundle.
- `fixtures/training_visualization/psion_google_live_remote_training_visualization_bundle_v2.json`
  is the canonical accelerated Google `v2` example bundle.
- `fixtures/training_visualization/parameter_golf_live_remote_training_visualization_bundle_v2.json`
  is the canonical single-node PGOLF `v2` example bundle.
- `fixtures/training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v2.json`
  is the canonical distributed PGOLF `v2` example bundle.
- `fixtures/training_visualization/parameter_golf_homegolf_remote_training_visualization_bundle_v2.json`
  is the canonical HOMEGOLF score-closeout `v2` example bundle.
- `fixtures/training_visualization/remote_training_run_index_v2.json` is the
  canonical track-aware run-index example and now includes the HOMEGOLF lane.

The stable schema versions are:

- `psionic.remote_training_visualization_bundle.v1`
- `psionic.remote_training_run_index.v1`
- `psionic.remote_training_visualization_bundle.v2`
- `psionic.remote_training_run_index.v2`

## Migration Window

`v1` remains the shipped foundation and stays readable during the migration
window.

`v2` is the track-aware follow-on. It is not a stealth `v1` field extension.

That split is deliberate:

- `v1` keeps the already-shipped always-live provider-neutral substrate stable
- `v2` adds the machine-first track semantics needed for `HOMEGOLF`, bounded
  `XTRAIN`, and later compare mode without forcing the app to guess meaning

The new `v2` contract adds:

- `track_family`
- `track_id`
- `execution_class`
- `comparability_class`
- `proof_posture`
- `public_equivalence_class`
- `score_law_ref`
- explicit artifact and wallclock caps when the score law carries them
- machine-stable primary-score identity when a run already has a retained score
- optional score-closeout and promotion-gate posture plus retained score deltas
- concise machine-authored semantic summaries

The `v2` bundle stays machine-first. It does not move pane-local labels or
badge copy into Psionic.

## What The Bundle Freezes

The bundle family freezes one provider-neutral shape for:

- provider, profile, lane, run, and repo revision identity
- explicit track semantics once a lane upgrades to `v2`
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

The run-index family freezes one discovery surface that can enumerate:

- summary-only lanes
- full-series lanes
- active runs
- completed runs
- refused runs
- rehearsal-only runs

That keeps the app out of provider-root walking and ad hoc manifest discovery.

The `v2` run index also carries explicit track semantics per row so consumers
can tell the difference between:

- PGOLF live training truth
- HOMEGOLF score-closeout truth
- bounded XTRAIN train-to-infer truth
- non-score Psion training truth

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

## Track-Aware Semantics

`v2` makes one repo-owned distinction explicit:

- `series_status` answers whether live chartable telemetry exists
- track semantics answer what the run means

That lets one app-facing viewer stay honest across different run families.

Examples:

- a PGOLF live lane can be `available` with measured runtime truth but no final
  submission score yet
- a HOMEGOLF lane can carry score-law and cap semantics even when the retained
  score arrives at closeout rather than during active training, and it can
  surface held promotion posture plus retained public-comparison deltas through
  the shared `score_surface`
- a bounded XTRAIN lane can stay explicit about its proof posture instead of
  being mistaken for a public-leaderboard-equivalent contest run

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

The RunPod `8xH100` Parameter Golf lane now writes the same provider-neutral
bundle family under the run root while the Rust-owned distributed runtime is
active.

The retained paths are:

- `training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json`
- `training_visualization/remote_training_run_index_v1.json`
- `training_visualization/snapshots/*.json`
- `training_visualization/snapshots/finalized_bundle.json`

The current lane now retains:

- one-second heartbeats while the runtime is active, even during bootstrap or
  between explicit phase transitions
- explicit phase transitions for runtime bootstrap, train step, validation, and
  runtime completion
- retained train-loss, bounded math, runtime timing, distributed-skew, and
  checkpoint-reference samples as soon as the first aggregate train-step receipt
  lands
- local GPU samples when `nvidia-smi` is available on the coordinator host
- bring-up, runtime-manifest, train-step, distributed-receipt, current-model,
  inventory, topology, and finalizer provenance in the same bundle family

The finalizer still matters:

- it seals the existing live bundle instead of inventing a separate provider
  artifact family
- it mirrors the retained `distributed_challenge_receipt` into the run root
  when one is only embedded inside the submission evidence report
- it appends inventory, topology, exported-folder, and finalizer-owned
  provenance before writing `snapshots/finalized_bundle.json`

Historical or synthetic closeout-only runs stay explicit:

- if the finalizer runs on an older run root with no retained live bundle, the
  finalizer still emits the canonical bundle family with `post_run_only`
  emission
- if the runtime never retained a first train-step receipt, `series_status`
  stays `partial` while active and degrades to `unavailable` at closeout
- refusal posture stays explicit when the retained receipt is still a
  measurements-missing or inventory-mismatch refusal

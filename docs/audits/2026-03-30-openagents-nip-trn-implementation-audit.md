# OpenAgents NIP-TRN Implementation Audit

Date: 2026-03-30

This audit reviews the current `openagents` app and kernel surfaces to answer one question:

What has to change to fully implement `NIP-TRN`, meaning a real Nostr-facing AI model training coordination layer rather than the current mix of authority objects, mirrored JSON bundles, and operator-local training state?

This is a concept and architecture audit. It does not cover tests.

## Executive Summary

The current app does not implement `TRN` yet.

In the checkout under audit:

- `openagents/crates/nostr/nips/README.md` only lists `DS`, `SA`, `SKL`, and `AC`.
- `openagents/crates/nostr/core/src/lib.rs` exports `nip_ac`, `nip_ds`, `nip_sa`, and `nip_skl`, but no `nip_trn`.
- `apps/autopilot-desktop` already has major training surfaces, but they are split across three different data planes:
  - kernel-authority projected training state
  - `psionic` JSON mirror state for remote training dashboards
  - local operator state for Apple adapter training

That means the main missing piece is not a new pane. The main missing piece is one shared Nostr training record layer and one first-class app lane that publishes and consumes it.

The good news is that the kernel already carries most of the typed object model that `TRN` needs. `ComputeTrainingRun`, `ComputeAdapterTrainingWindow`, `ComputeAdapterContributionOutcome`, `ComputeAdapterCheckpointPointer`, and `ComputeAcceptedOutcome` are already close to the core records needed for training coordination. The missing work is publication, subscription, projection, and public recovery or fork semantics.

The right target is:

- `psionic` owns training execution, checkpoint production, proof files, artifact storage, and heavy telemetry.
- `openagents-kernel-core` and `apps/nexus-control` own canonical authority objects and policy enforcement.
- `TRN` becomes the shared Nostr coordination layer above that substrate.
- `apps/autopilot-desktop` consumes `TRN` through one explicit lane instead of stitching together authority snapshots and mirrored JSON files.

## Sources Reviewed

- `openagents/docs/OWNERSHIP.md`
- `openagents/docs/MVP.md`
- `openagents/crates/nostr/nips/README.md`
- `openagents/crates/nostr/core/src/lib.rs`
- `openagents/crates/nostr/client/src/lib.rs`
- `openagents/proto/openagents/compute/v1/compute_training.proto`
- `openagents/crates/openagents-kernel-core/src/authority.rs`
- `openagents/apps/nexus-control/src/lib.rs`
- `openagents/apps/nexus-control/src/kernel.rs`
- `openagents/apps/autopilot-desktop/src/desktop_control.rs`
- `openagents/apps/autopilot-desktop/src/remote_training_sync.rs`
- `openagents/apps/autopilot-desktop/src/xtrain_explorer_control.rs`
- `openagents/apps/autopilot-desktop/src/apple_adapter_training_control.rs`
- `openagents/apps/autopilot-desktop/src/runtime_lanes.rs`
- `openagents/apps/autopilot-desktop/src/render.rs`
- `openagents/apps/autopilot-desktop/src/pane_registry.rs`
- `psionic/docs/TRAIN_SYSTEM.md`

## What Exists Today

### 1. The app already has training UX, but not one training truth

`apps/autopilot-desktop` already exposes four distinct training surfaces or regimes:

- authority-projected training status through `desktop_control` and `autopilotctl`
- `Training Runs`, which is the mirror-backed remote training dashboard
- `XTRAIN Explorer`, which is a file-backed snapshot explorer
- `Apple Adapter Training`, which is an operator flow for launching and monitoring local training

The pane split is visible in the pane registry:

- `pane.psionic_remote_training` is described as a shared training dashboard in `openagents/apps/autopilot-desktop/src/pane_registry.rs:438`.
- `pane.xtrain_explorer` is a separate decentralized explorer in `openagents/apps/autopilot-desktop/src/pane_registry.rs:451`.
- `pane.apple_adapter_training` is a separate operator pane in `openagents/apps/autopilot-desktop/src/pane_registry.rs:556`.

The authority-projected training status is separate from those panes and is built in `desktop_control_training_status` in `openagents/apps/autopilot-desktop/src/desktop_control.rs:7129`.

At the state level, the app also keeps separate training-related state buckets in `RenderState`, including `desktop_control`, `xtrain_explorer`, `apple_adapter_training`, and Nostr runtime lanes for `SA`, `SKL`, and `AC`, but nothing parallel for `TRN` in `openagents/apps/autopilot-desktop/src/app_state.rs:15975` and `openagents/apps/autopilot-desktop/src/app_state.rs:16020`.

### 2. The authority path already has a strong typed training object model

The kernel training proto already defines major records that are close to `TRN`:

- `ComputeTrainingRun` in `openagents/proto/openagents/compute/v1/compute_training.proto:164`
- `ComputeAcceptedOutcome` in `openagents/proto/openagents/compute/v1/compute_training.proto:190`
- `ComputeAdapterCheckpointPointer` in `openagents/proto/openagents/compute/v1/compute_training.proto:213`
- `ComputeAdapterTrainingWindow` in `openagents/proto/openagents/compute/v1/compute_training.proto:230`
- `ComputeAdapterContributionOutcome` in `openagents/proto/openagents/compute/v1/compute_training.proto:266`

The authority client already supports create, finalize, record, list, and fetch operations for those records in `openagents/crates/openagents-kernel-core/src/authority.rs:78`, `openagents/crates/openagents-kernel-core/src/authority.rs:212`, `openagents/crates/openagents-kernel-core/src/authority.rs:1606`, and `openagents/crates/openagents-kernel-core/src/authority.rs:2131`.

`apps/nexus-control` already exposes those records over HTTP in `openagents/apps/nexus-control/src/lib.rs:3421`, `openagents/apps/nexus-control/src/lib.rs:3463`, `openagents/apps/nexus-control/src/lib.rs:3497`, `openagents/apps/nexus-control/src/lib.rs:3542`, `openagents/apps/nexus-control/src/lib.rs:3626`, and `openagents/apps/nexus-control/src/lib.rs:3694`.

The in-memory kernel already stores and filters the same records in `openagents/apps/nexus-control/src/kernel.rs:2552`, `openagents/apps/nexus-control/src/kernel.rs:2583`, `openagents/apps/nexus-control/src/kernel.rs:2614`, `openagents/apps/nexus-control/src/kernel.rs:2648`, `openagents/apps/nexus-control/src/kernel.rs:4702`, and `openagents/apps/nexus-control/src/kernel.rs:4856`.

That means the kernel is not the blocker. The kernel already has the shapes. The missing layer is Nostr publication and Nostr-native read models.

### 3. The desktop app already projects authority-backed training status

`DesktopControlTrainingStatus` in `openagents/apps/autopilot-desktop/src/desktop_control.rs:944` is already a substantial training read model.

The app refreshes compute history from the remote authority in `load_compute_history_from_authority` at `openagents/apps/autopilot-desktop/src/desktop_control.rs:6464`. That path loads:

- delivery proofs
- capacity instruments
- validator challenges
- training runs
- adapter windows
- contribution outcomes
- accepted outcomes

The app then derives a training dashboard from that cache in `desktop_control_training_status` at `openagents/apps/autopilot-desktop/src/desktop_control.rs:7129`.

This is important because it shows the app already knows how to present training coordination state. It just does it through one authority-specific projection instead of a `TRN` lane.

### 4. The remote training dashboard is file-mirror based, not relay based

The current remote training surface is not Nostr-backed.

`apps/autopilot-desktop/src/remote_training_sync.rs` reads a `psionic` fixture path, environment overrides, and a local cache:

- `OPENAGENTS_REMOTE_TRAINING_SOURCE_ROOT`
- `OPENAGENTS_REMOTE_TRAINING_INDEX_PATH`
- `OPENAGENTS_REMOTE_TRAINING_CACHE_ROOT`

Those entry points are defined in `openagents/apps/autopilot-desktop/src/remote_training_sync.rs:13` through `openagents/apps/autopilot-desktop/src/remote_training_sync.rs:22`.

The refresh path reads `RemoteTrainingRunIndexV2` and `RemoteTrainingVisualizationBundleV2` from disk in `openagents/apps/autopilot-desktop/src/remote_training_sync.rs:189` through `openagents/apps/autopilot-desktop/src/remote_training_sync.rs:260`.

The resulting `DesktopControlRemoteTrainingStatus` in `openagents/apps/autopilot-desktop/src/desktop_control.rs:1119` explicitly carries mirror-oriented fields like:

- `source_root`
- `source_index_path`
- `cache_root`
- `sync_state`

The projection even labels the source as `live_psionic_mirror` or `local_cache_mirror` in `openagents/apps/autopilot-desktop/src/desktop_control.rs:7505`.

This is useful for visualization, but it is not `TRN`.

### 5. The XTRAIN explorer is also file-backed, not relay-backed

`apps/autopilot-desktop/src/xtrain_explorer_control.rs` reads:

- `OPENAGENTS_XTRAIN_EXPLORER_SOURCE_ROOT`
- `OPENAGENTS_XTRAIN_EXPLORER_INDEX_PATH`

Those paths are defined in `openagents/apps/autopilot-desktop/src/xtrain_explorer_control.rs:9` through `openagents/apps/autopilot-desktop/src/xtrain_explorer_control.rs:15`.

The explorer loads `XtrainExplorerIndex` and `XtrainExplorerSnapshot` from files in `openagents/apps/autopilot-desktop/src/xtrain_explorer_control.rs:124` through `openagents/apps/autopilot-desktop/src/xtrain_explorer_control.rs:223`.

This means the current XTRAIN view is a static or mirrored artifact explorer, not a live Nostr coordination surface.

### 6. The Apple adapter operator flow is authority-backed and local-runtime backed

The Apple adapter training path is a real operator workflow in `openagents/apps/autopilot-desktop/src/apple_adapter_training_control.rs:1`.

It already integrates:

- `HttpKernelAuthorityClient`
- `CreateComputeTrainingRunRequest`
- `FinalizeComputeTrainingRunRequest`
- `AcceptComputeOutcomeRequest`
- `ComputeTrainingRun`
- `ComputeAcceptedOutcome`
- `psionic_train` execution and export helpers

That is strong substrate for a real `TRN` publisher, but today it is not publishing public training coordination records. It is talking to authority endpoints and local execution helpers.

### 7. The app already has custom Nostr lanes, but none for training

The app already spins up Nostr-backed runtime workers for `SA`, `SKL`, and `AC`:

- `SaLaneWorker::spawn()`
- `SklLaneWorker::spawn()`
- `AcLaneWorker::spawn()`

This happens in `openagents/apps/autopilot-desktop/src/render.rs:608` through `openagents/apps/autopilot-desktop/src/render.rs:611`.

The runtime lane state and command model live in `openagents/apps/autopilot-desktop/src/runtime_lanes.rs:1`.

There is no `TrnLaneWorker`, `TrnLaneSnapshot`, or training-specific Nostr command set.

This matters because the cleanest app implementation path is obvious:

Add `TRN` as one more first-class lane instead of hiding it in the mirror sync path or bolting it into the authority cache.

### 8. The Nostr crate does not expose `TRN` yet

The current in-repo Nostr draft index only lists:

- `DS`
- `SA`
- `SKL`
- `AC`

That is the entire list in `openagents/crates/nostr/nips/README.md:1`.

The Nostr core crate exports:

- `nip_ac`
- `nip_ds`
- `nip_sa`
- `nip_skl`

That is the export surface in `openagents/crates/nostr/core/src/lib.rs:48`.

There is no checked-in `TRN` draft or `nip_trn` module in this checkout.

## Coverage Against TRN

The proposed `TRN` shape is one umbrella training coordination profile with core records for:

- network
- node
- window
- receipt
- verdict
- artifact pointer
- closeout

And optional profiles for:

- discovery
- private coordination
- challenge jobs
- reputation

Here is the current coverage.

| TRN Surface | Current Status | Current Source | What Is Missing |
| --- | --- | --- | --- |
| Network record | Missing | No explicit public network or run-root record | Shared public network or program identity, governance revision, relay stance, recovery or fork lineage |
| Node record | Missing | Cluster members exist locally in desktop projections, but not as signed public training node records | Signed node capability publication, admitted roles, build digest, benchmark evidence, revocation or replacement semantics |
| Window record | Partial | `ComputeAdapterTrainingWindow` | Public event form, event ids, Nostr subscription model, resume or fork links, relay-visible sequencing |
| Receipt record | Partial | `ComputeAdapterContributionOutcome` | Separate public receipt event instead of only outcome object, clearer assignment and submission publication |
| Verdict record | Partial | Contribution disposition plus validator challenge state | Standalone public validator verdict record and challenge or replay references |
| Artifact pointer | Partial | `ComputeAdapterCheckpointPointer`, `final_checkpoint_ref`, `promotion_checkpoint_ref` | Public locator records for accepted checkpoints, final weights, optimizer state, manifest bundles, and fork or resume anchors |
| Closeout record | Partial | `ComputeAcceptedOutcome` | Public closeout event with reusable references, settlement or reputation linkage, explicit forkable accepted state |
| Discovery profile | Partial | General Nostr provider presence plus local desktop state | Training-specific network discovery, run discovery, and node discovery filters |
| Private coordination profile | Missing | None training-specific | NIP-44 or NIP-59 usage for assignments, validator coordination, or sensitive control messages |
| Challenge jobs profile | Partial | Kernel validator challenges | Public challenge task or replay coordination semantics |
| Reputation profile | Missing | No TRN-facing training reputation surface | NIP-32 labels or equivalent training-specific labels tied to verdicts or closeouts |

## The Main Conceptual Problem

The current training experience is built from three different truths:

1. Authority truth for runs, windows, contributions, and accepted outcomes.
2. Mirror truth for remote training bundles and explorer snapshots.
3. Local operator truth for running training, exporting artifacts, and accepting results.

That is why the app can show a lot of training information today while still not being close to real `TRN`.

`TRN` requires one public coordination language. Right now the app has three partial languages:

- kernel objects
- visualization bundles
- local operator stage state

Those need to stop competing.

The correct unification is:

- kernel objects remain canonical internal market or authority truth
- `TRN` becomes the shared public coordination and discovery truth
- visualization bundles become optional derived artifacts behind public pointers
- operator-local stage state becomes one local producer of future `TRN` records, not its own separate training protocol

## Required Changes By Owner

## 1. `openagents/crates/nostr/core`

This crate needs a real `nip_trn` module before the app can honestly claim `TRN` support.

Required work:

- Add `nip_trn.rs` or `nip_trn/mod.rs`.
- Export it from `crates/nostr/core/src/lib.rs`.
- Define kind constants for the core `TRN` records:
  - network
  - node
  - window
  - receipt
  - verdict
  - artifact pointer
  - closeout
- Define the canonical tag vocabulary for:
  - network id
  - run id
  - window id
  - node id
  - policy refs
  - parent or source run
  - resume or fork source
  - artifact role
  - manifest digest
  - weight pointer
  - verdict
  - replay requirement
  - closeout disposition
- Add typed parser and builder structs in the same style as `nip_ds`.
- Validate required references and reject invalid event shapes early.

This is the minimum protocol layer. Without it, the rest of the app has no stable `TRN` contract to target.

## 2. `openagents/crates/nostr/client`

The client crate is intentionally minimal today. That is fine, but full `TRN` support still needs client-side helpers.

Required work:

- Add `TRN`-specific subscription helpers for:
  - one training network
  - one run
  - one window
  - one node
  - artifact-pointer history
  - closeout history
- Add typed publish helpers so the app and authority services do not hand-roll raw events everywhere.
- Add projection utilities for ordering and deduping multiple relay views of the same training records.
- Add gap-recovery rules for `TRN` streams so the desktop can reconnect and rebuild state.

The client does not need to become a heavy training engine. It just needs enough typed support that `TRN` events can be published and consumed safely.

## 3. `openagents/crates/openagents-kernel-core` and proto contracts

The kernel object model is already close to `TRN`, but not identical.

Required work:

- Decide the authoritative mapping from kernel objects to `TRN` records.
- Add the missing public identifiers and lineage fields that `TRN` needs but the current kernel model does not expose cleanly.
- Normalize resume or fork semantics into typed fields instead of leaving them to ad hoc metadata blobs.
- Expose public artifact-locator fields for:
  - accepted checkpoint
  - final checkpoint
  - promotion checkpoint
  - model weight bundle
  - optimizer snapshot
  - config or manifest bundle
- Split "submission receipt", "validator verdict", and "closeout" more cleanly where the current object model collapses them into one contribution or outcome record.

The biggest gap here is not training windows. The biggest gap is public lineage:

- where did this run come from
- what accepted state can it resume from
- what accepted state can it fork from
- what exact artifact pointer is the public recovery anchor

That has to be explicit if `TRN` is supposed to support recovery and forking after operator or coordinator failure.

## 4. `apps/nexus-control`

`apps/nexus-control` is the natural place to publish `TRN` from canonical authority state.

Required work:

- Add a `TRN` publisher or projector beside the current HTTP authority routes.
- Publish `TRN` records when the kernel mutates training state:
  - training run created
  - training run finalized
  - adapter window recorded
  - contribution recorded or finalized
  - accepted outcome recorded
  - checkpoint pointer promoted
- Backfill existing kernel training history into `TRN` for relay-based recovery.
- Keep a mapping from kernel record ids to Nostr event ids if later projections need it.
- Decide what is public by default and what must move through optional encrypted coordination.

`nexus-control` should not become the place that stores checkpoint bytes or runs training. It should become the bridge between canonical authority truth and public coordination truth.

## 5. `apps/autopilot-desktop`

This is where the visible product change has to happen.

Required work:

- Add a first-class `TRN` runtime lane, parallel to the current `SA`, `SKL`, and `AC` lanes.
- Add `TrnLaneSnapshot`, `TrnLaneWorker`, command types, and reducer wiring.
- Subscribe to `TRN` records for the networks, runs, and nodes the user cares about.
- Build the training pane read model from `TRN` projections first, not from file mirrors.
- Keep the existing remote training bundles and explorer snapshots as optional detail artifacts, loaded only when a `TRN` artifact pointer says they exist.
- Teach `autopilotctl` to query and print `TRN` state, not only authority projections and mirror status.
- Add operator actions for:
  - publish node record
  - join or advertise a run
  - publish a receipt
  - inspect a verdict
  - open an artifact pointer
  - resume from accepted state
  - fork from accepted state

The current split between `training`, `remote_training`, `xtrain_explorer`, and `apple_adapter_training` should eventually collapse into one shared training coordination model:

- `training` becomes authority plus `TRN` status
- `remote_training` becomes a derived artifact and visualization view
- `xtrain_explorer` becomes a network or run explorer backed by `TRN`
- `apple_adapter_training` becomes one local operator path that publishes into the same training record system

Without that unification, the app will keep showing users multiple incompatible definitions of what "the training run" is.

## 6. `psionic`

`psionic` should stay out of the Nostr product plane except where it provides durable machine-readable artifact metadata.

Required work in `psionic`:

- Keep producing checkpoint manifests, proof files, and training visualization bundles.
- Make those artifacts easy to reference through stable digests and locators.
- Ensure accepted checkpoints, final weights, optimizer state, and config bundles have durable pointer metadata that `TRN` can publish.
- Keep heavy artifact transfer and live execution coordination outside `TRN`.

`TRN` should point to `psionic` artifacts. It should not carry them.

## What Should Stay Out Of TRN

To fully implement `TRN`, the team should be strict about what does not belong in it.

Do not move these into `TRN` events:

- checkpoint bytes
- model weights
- optimizer tensors
- step-by-step live gradient exchange
- high-frequency telemetry streams
- sandbox workspace files
- private credentials
- provider-local temporary file paths
- heavy visualization payloads when a stable artifact pointer is enough

`TRN` should standardize the public coordination records. It should not become a transport for the heavy training runtime.

## The Most Important Missing TRN Features

If the goal is not just "publish some training events" but "support recovery, self-healing, and forking," then these are the highest-value gaps:

### 1. Public recovery anchors

Another operator must be able to read the relay history and answer:

- what run is active
- what the current or last sealed window is
- what validator policy applies
- what artifact pointer is the accepted resume point
- what checkpoint or weight pointer is the best known public continuation state

Today the app does not expose that as one Nostr-readable chain of records.

### 2. Public fork lineage

If a run forks, the public record needs explicit lineage:

- source network or run
- source accepted checkpoint or weight pointer
- source window or closeout
- new policy revision or governance terms

Today the app and kernel have checkpoint references, but not a public fork model.

### 3. Public node capability publication

The desktop currently knows cluster members and provider status, but it does not publish signed training-node records that a third party can reuse for scheduling, trust, or audit.

That makes open training coordination impossible. A training network needs public node claims with at least:

- node id
- admitted roles
- software or build digest
- benchmark evidence
- supported artifact or format capabilities

### 4. Public validator verdict publication

Today validator state is partly visible through authority records and counters, but not through a dedicated relay-native verdict surface.

That means third parties cannot independently reconstruct which contribution was accepted, quarantined, rejected, or marked for replay without depending on private authority access.

## Recommended Target Architecture

The clean target is:

1. `psionic` produces artifacts and runtime evidence.
2. `apps/nexus-control` or another authority bridge writes canonical training truth.
3. The authority bridge publishes `TRN` records to relays.
4. `apps/autopilot-desktop` subscribes to `TRN`, builds a local training projection, and only fetches heavy artifacts when the user drills down.
5. Recovery or fork flows start from `TRN` lineage plus artifact locators, not from private dashboards or local cache directories.

That keeps the ownership split honest with `openagents/docs/OWNERSHIP.md`:

- `psionic` owns training execution substrate
- kernel and Nexus own authority and policy
- `openagents` app owns the UX and Nostr-facing product composition

## Implementation Sequence

## Phase 1: Freeze the TRN record model

- Add the in-repo `TRN` draft and index entry.
- Add `nip_trn` to `crates/nostr/core`.
- Freeze event kinds, tags, and validation rules.

## Phase 2: Map kernel training objects to TRN

- Define the mapping from current kernel objects to core `TRN` records.
- Add any missing fields needed for recovery, fork lineage, and public artifact locators.

## Phase 3: Publish TRN from authority state

- Add a `TRN` projector or publisher in `apps/nexus-control`.
- Backfill existing training history.
- Ensure the relay history is enough to reconstruct the public coordination state of a run.

## Phase 4: Add a TRN lane to the desktop

- Add `TrnLaneWorker`.
- Subscribe to training records.
- Build a local projection and expose it in `desktop_control`.
- Add `autopilotctl` support.

## Phase 5: Rebuild the panes around TRN

- Rebase `Training Runs` on `TRN`.
- Convert `XTRAIN Explorer` into a `TRN` explorer with optional artifact drill-down.
- Keep remote training visualization bundles as derived artifacts, not primary truth.
- Wire Apple adapter operator actions to publish or update `TRN` state.

## Phase 6: Add recovery and fork UX

- Publish accepted artifact pointers and closeouts as public continuation anchors.
- Add UI and CLI flows for resume and fork from accepted public state.

## Bottom Line

OpenAgents is not starting from zero.

The kernel already has most of the typed records that `TRN` needs, and the desktop already knows how to present training state. What is missing is the shared Nostr coordination layer and the decision to stop treating training as three separate protocols.

If the team wants full `TRN` support, the real job is:

- standardize the training records in `nostr/core`
- publish them from the authority path
- add one `TRN` lane in the desktop
- demote mirror bundles and local operator state from primary truth to supporting artifacts

That is the shortest path to real training coordination, public recovery, and forkable runs without turning `TRN` into a transport for heavy training runtime data.

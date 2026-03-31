# OpenAgents + Psionic NIP-TRN Implementation Audit

Date: 2026-03-30

This revision inspects actual `psionic` code, not just `psionic/docs/TRAIN_SYSTEM.md`.

The earlier version of this audit was too app-heavy. It described the `openagents` desktop and kernel situation correctly, but it undercounted how much `psionic` already models the training-coordination problem in code.

The real question is not "does Psionic have the concepts needed for `TRN`?"

It does.

The real question is:

How much of that existing Psionic vocabulary should be projected into a Nostr training profile, and what still has to change to turn those Rust-native contracts and artifact builders into a live relay-visible coordination layer?

This is a concept and architecture audit. It does not cover tests.

## Executive Summary

`openagents` still does not implement `TRN`.

In the current `openagents` checkout:

- there is no `TRN` draft in `crates/nostr/nips`
- there is no `nip_trn` module in `crates/nostr/core`
- there is no `TRN` client helper layer
- there is no `TRN` runtime lane in `apps/autopilot-desktop`

That part is unchanged.

What changed after inspecting actual `psionic` code is the conclusion about readiness.

`psionic` already has most of the semantic surface that `TRN` would need:

- run, participant, topology, contributor-set, and window state
- assignment, execution, upload, validation, aggregation, and promotion receipts
- checkpoint manifests, checkpoint pointers, restore receipts, and storage locators
- network, node, registry, assignment, validator, consensus, slashing, reward, settlement, and explorer contracts
- provider-neutral evidence bundles, remote-training run bundles, and explorer artifacts

The main gap is not missing nouns. The main gap is missing live publication.

Today these `psionic` surfaces are mostly expressed as:

- runtime structs
- canonical contract modules
- fixture-producing reference builders
- derived artifact builders

They are not yet a live Nostr event family.

That means the shortest path to `TRN` is not inventing a brand-new protocol from scratch. The shortest path is:

1. pick the existing `psionic` vocabulary that should become public coordination records
2. map it cleanly onto one `TRN` event family
3. bridge `openagents` kernel authority and `psionic` runtime truth into those events
4. make the desktop consume those events instead of mirror files

The architectural risk is duplication. Right now the same training system is described in four overlapping vocabularies:

- `psionic` runtime and contract types
- `openagents` kernel authority types
- remote-training and explorer artifact families
- desktop-local operator state

`TRN` should collapse those into one public coordination projection. It should not become a fifth competing vocabulary.

## Sources Reviewed

### OpenAgents

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

### Psionic

- `crates/psionic-train/src/lib.rs`
- `crates/psionic-train/src/run_graph.rs`
- `crates/psionic-train/src/orchestrator.rs`
- `crates/psionic-train/src/adapter_window.rs`
- `crates/psionic-train/src/artifact_storage.rs`
- `crates/psionic-train/src/checkpoint_recovery.rs`
- `crates/psionic-train/src/distributed_checkpoint_contract.rs`
- `crates/psionic-train/src/cross_provider_program_run_graph.rs`
- `crates/psionic-train/src/contributor_program_lineage.rs`
- `crates/psionic-train/src/decentralized_network_contract.rs`
- `crates/psionic-train/src/signed_node_identity_contract.rs`
- `crates/psionic-train/src/public_network_registry_contract.rs`
- `crates/psionic-train/src/public_work_assignment_contract.rs`
- `crates/psionic-train/src/validator_challenge_scoring_contract.rs`
- `crates/psionic-train/src/multi_validator_consensus_contract.rs`
- `crates/psionic-train/src/fraud_quarantine_slashing_contract.rs`
- `crates/psionic-train/src/reward_ledger_contract.rs`
- `crates/psionic-train/src/settlement_publication_contract.rs`
- `crates/psionic-train/src/public_run_explorer_contract.rs`
- `crates/psionic-train/src/training_execution_evidence_bundle.rs`
- `crates/psionic-train/src/remote_training_visualization_v2.rs`
- `crates/psionic-train/src/xtrain_explorer_artifacts.rs`
- `docs/TRAIN_SYSTEM.md`

## What Psionic Already Has

## 1. Live training substrate objects

The `psionic-train` crate already has real runtime-facing training structures, not just vague roadmap language.

`run_graph.rs` defines the core lifecycle vocabulary:

- `TrainingRunStatus`
- `TrainingParticipantRole`
- `TrainingParticipantAdmissionState`
- `TrainingParticipantReadinessState`
- `TrainingParticipantContributionState`
- `TrainingWindowStatus`
- `TrainingLifecycleEventKind`

That is already close to the `TRN` idea of network, node, window, and run state.

`orchestrator.rs` adds explicit control-plane behavior on top of that:

- off-policy admission budgets
- rollout-admission signals
- typed rollout receipts
- window planning and sealing constraints

This matters because `TRN` should not invent its own freshness and replay language when `psionic` already has one.

`adapter_window.rs` is even closer to a public receipt vocabulary. It already models:

- contribution assignment receipts
- execution summaries
- upload receipts
- validator receipts
- aggregation-eligibility receipts
- promotion requirements
- source checkpoint pointers and policy revisions

That file is basically a training receipt language already. What it does not have is a Nostr event form.

## 2. Strong artifact and recovery semantics

`artifact_storage.rs` and `checkpoint_recovery.rs` are the clearest evidence that `psionic` already knows how to support recovery and forkability.

`artifact_storage.rs` already defines:

- artifact classes
- storage tiers
- lifecycle states
- retention profiles
- explicit artifact locators for checkpoints, adapter contributions, promoted window checkpoints, rollout artifacts, eval artifacts, metrics bundles, and final evidence bundles

`checkpoint_recovery.rs` already defines:

- scope bindings for run, stage, and window
- checkpoint manifests
- checkpoint pointers
- restore attempts
- restore receipts
- uploader assignments

`distributed_checkpoint_contract.rs` builds on that with:

- shard placements
- upload receipts
- restore assignments
- restore plans

That means the "someone else can resume or fork the run if the original coordinator disappears" idea is already present in `psionic` as typed data. It is just not published as relay-native public history yet.

## 3. Whole-program and contributor-lineage state

`cross_provider_program_run_graph.rs` models whole-program training state under one shared run id:

- participants
- role windows
- transition log
- evidence bindings

`contributor_program_lineage.rs` binds contributor windows back to:

- dataset family
- dataset slice
- checkpoint family
- input policy revision
- candidate policy revision
- candidate checkpoint

This is the exact kind of lineage `TRN` needs if resumed or forked runs are supposed to be machine-readable instead of hand-waved in a dashboard.

## 4. Public-network contracts

`psionic` already has an explicit public-network contract family.

`decentralized_network_contract.rs` defines:

- `network_id`
- governance revision
- epoch cadence
- settlement backend posture
- checkpoint authority policy
- public role bindings

That is already a strong candidate for the `TRN` network record.

`signed_node_identity_contract.rs` defines signed node records with:

- wallet binding
- software attestation
- capability projection
- benchmark evidence
- admitted roles
- admitted execution classes
- revocation status
- detached signatures

That is already a strong candidate for the `TRN` node record.

`public_network_registry_contract.rs` adds:

- availability status
- relay posture
- endpoints
- compatibility policy
- discovery examples
- matchmaking offers

That is already a strong candidate for the `TRN` discovery profile.

`public_work_assignment_contract.rs` adds:

- public windows
- assignment ids
- assignment receipts
- late-window refusals
- explicit `public_validator_challenge` work kinds

That is already most of the `TRN` window and receipt story.

## 5. Validator, consensus, fraud, and settlement contracts

`validator_challenge_scoring_contract.rs` already models:

- replay rules
- score receipts
- challenge refusals
- shared validator dispositions

`multi_validator_consensus_contract.rs` already models:

- validator votes
- promotion decisions
- disagreement receipts

`fraud_quarantine_slashing_contract.rs` already models:

- fraud signals
- quarantine decisions
- slashing decisions
- appeal windows

`reward_ledger_contract.rs` already models:

- accounting periods
- contribution entries
- penalty entries
- final allocations

`settlement_publication_contract.rs` already models:

- validator-weight publications
- settlement records
- payout exports
- settlement refusals

`public_run_explorer_contract.rs` already models the public summary surface above those feeds:

- explorer panes
- explorer snapshot
- score rows
- stale-data policies

So the earlier conclusion that verdict, reputation, and settlement were mostly missing was wrong for `psionic`. They are not missing conceptually. They are present as structured contracts. The missing piece is live Nostr publication and app consumption.

## 6. Derived evidence and visualization artifacts

`training_execution_evidence_bundle.rs` ties retained evidence together and already links:

- launch facts
- runtime facts
- checkpoint facts
- metric facts
- visualization references
- validator results
- final artifacts
- after-action refs

It also explicitly understands:

- run bundles
- run indexes
- explorer snapshots
- explorer indexes

`remote_training_visualization_v2.rs` defines the track-aware provider-neutral remote-training bundle and run index. It is already rich enough for public score and comparison surfaces, but it is still a derived artifact family.

`xtrain_explorer_artifacts.rs` defines the XTRAIN snapshot and index artifacts, again as explorer-ready derived state.

These should not become the core `TRN` records. They should sit behind `TRN` artifact-pointer records.

## 7. The key Psionic limitation

The strongest limitation after reading the code is this:

Most of these `psionic` modules are contract and fixture surfaces, not live relay publishers.

That is visible across the files:

- stable `*_FIXTURE_PATH` constants
- stable `*_CHECK_SCRIPT_PATH` constants
- stable `*_DOC_PATH` constants
- `canonical_*` constructors
- `write_*` helpers that emit committed JSON artifacts

That is not a criticism. It means `psionic` has already done the hard work of naming and typing the training-coordination surface.

But it also means `psionic` does not yet implement a live public coordination plane. It implements the canonical shapes that such a coordination plane should publish.

## What OpenAgents Already Has

`openagents` is still the weak side of the `TRN` story.

In the checked-out `openagents` tree:

- `crates/nostr/nips/README.md` lists only `DS`, `SA`, `SKL`, and `AC`
- `crates/nostr/core/src/lib.rs` exports only `nip_ds`, `nip_sa`, `nip_skl`, and `nip_ac`
- `crates/nostr/client` has no `TRN`-specific helpers
- `apps/autopilot-desktop` has no `TRN` lane

The desktop already has training surfaces, but they are split:

- authority-projected training status via `desktop_control`
- mirror-backed remote training via `remote_training_sync.rs`
- file-backed XTRAIN explorer snapshots via `xtrain_explorer_control.rs`
- local operator state for Apple adapter training via `apple_adapter_training_control.rs`

The kernel authority already exposes strong typed objects:

- `ComputeTrainingRun`
- `ComputeAdapterTrainingWindow`
- `ComputeAdapterContributionOutcome`
- `ComputeAdapterCheckpointPointer`
- `ComputeAcceptedOutcome`

So `openagents` does not lack training truth entirely. It lacks a Nostr-native public projection of that truth.

## Revised TRN Coverage

The earlier version of this audit treated too many `TRN` surfaces as "missing." After reading actual `psionic` code, the honest picture is:

| TRN Surface | Psionic Reality | OpenAgents Reality | What Is Still Missing |
| --- | --- | --- | --- |
| Network record | Strong typed contract already exists in `decentralized_network_contract.rs` | No `TRN` network record | Nostr event kind, live publisher, and one chosen bridge from runtime or authority state |
| Node record | Strong typed signed node identity and registry records already exist in `signed_node_identity_contract.rs` and `public_network_registry_contract.rs` | No `TRN` node record | Nostr mapping, refresh cadence, and revocation publication |
| Window record | Strong typed runtime and public forms already exist in `run_graph.rs`, `adapter_window.rs`, and `public_work_assignment_contract.rs` | Kernel windows exist, but no `TRN` view | One unified public identity across psionic runtime, kernel authority, and relay history |
| Receipt record | Strong typed assignment, execution, upload, validator, aggregation, restore, and assignment receipts already exist | Contribution outcomes partly overlap | Decide which receipts become public core `TRN` records and which stay internal |
| Verdict record | Strong typed scoring, consensus, disagreement, slashing, and appeals already exist | Validator challenges and counters exist | Relay publication and stable public record linking between challenge, verdict, and closeout |
| Artifact pointer | Very strong in `artifact_storage.rs`, `checkpoint_recovery.rs`, and `distributed_checkpoint_contract.rs` | Kernel checkpoint pointer is partial | Standard `TRN` locator event family and explicit fork or resume parent links |
| Closeout record | Strong typed reward, settlement, payout, and explorer surfaces already exist | Accepted outcome is partial | Map final authority and public settlement state into one public closeout story |
| Discovery profile | Strong typed registry and matchmaking already exist | No training-specific Nostr discovery | Publish it as relay-native records instead of only fixture contracts |
| Private coordination profile | No Nostr implementation yet | No Nostr implementation yet | Encrypted assignment and validator coordination over `NIP-44` or `NIP-59` when needed |
| Challenge jobs profile | Strong typed public-validator challenge model already exists | Partial kernel validator challenges | Bind challenge work to actual Nostr request/result flow when needed |
| Reputation profile | Strong typed fraud, quarantine, slashing, and allocation surfaces already exist | No public reputation layer | Decide whether `TRN` carries this directly or references `NIP-32` labels and settlement feeds |
| Explorer and score surfaces | Strong derived artifact families already exist | Desktop consumes local files | Make them secondary artifacts behind `TRN` pointers rather than primary truth |

## The Main Architectural Problem

The core problem is not that the system lacks training concepts.

The core problem is that the same system is described in multiple incompatible vocabularies:

1. `psionic` runtime and contract types
2. `openagents` kernel authority types
3. remote-training and explorer artifact families
4. desktop-local operator state

`TRN` should not become a fifth independent vocabulary.

It should be a projection layer over the existing system.

That means the design bar is:

- do not invent new names when `psionic` already has a typed concept
- do not treat run bundles or explorer snapshots as the primary source of truth
- do not duplicate node, window, checkpoint, or validator language in `openagents` if `psionic` already has a stronger version

## What Should Map Into TRN Directly

The clean mapping is:

- `TRN` network record
  - derived from `decentralized_network_contract.rs`
  - augmented by current epoch and active discovery data from `public_network_registry_contract.rs`

- `TRN` node record
  - derived from `signed_node_identity_contract.rs`
  - optionally split into identity and live registry status if needed

- `TRN` window record
  - derived from `run_graph.rs`, `adapter_window.rs`, and `public_work_assignment_contract.rs`
  - must preserve assignment seed, policy revision, source checkpoint pointer, and lifecycle state

- `TRN` receipt record
  - derived from adapter-window receipts, public assignment receipts, and checkpoint restore receipts
  - must stay lightweight and pointer-based

- `TRN` verdict record
  - derived from validator score receipts, consensus votes, disagreement receipts, and fraud or slashing decisions

- `TRN` artifact-pointer record
  - derived from `artifact_storage.rs`, `checkpoint_recovery.rs`, and `distributed_checkpoint_contract.rs`
  - should point to manifests, checkpoint pointers, final evidence bundles, and explorer artifacts

- `TRN` closeout record
  - derived from reward-ledger, settlement-publication, and final authority acceptance state

## What Should Not Map Into TRN

Do not turn `TRN` into a transport for heavy runtime data.

Keep these out:

- checkpoint bytes
- model weights
- optimizer tensors
- shard payloads
- high-frequency telemetry streams
- raw rollout artifacts
- large explorer snapshots when a pointer is enough
- sandbox-local workspace state

`TRN` should carry identities, references, dispositions, and locators.

`psionic` should keep carrying the heavy runtime and artifact plane.

## What Is Still Missing Even After the Psionic Deep Dive

Even with the stronger `psionic` picture, the implementation gaps are still real.

## 1. No live Nostr publication path

The contracts are strong, but they are not live relay records.

There is no code today that takes the existing `psionic` network, node, window, receipt, verdict, artifact, and settlement surfaces and emits them as Nostr events.

## 2. No single bridge between Psionic and kernel authority

`psionic` and `openagents` both model training state, but they do not yet share one public projection contract.

That is the seam that will produce duplication unless it is made explicit.

The clean rule is:

- `psionic` should own execution facts, artifact facts, checkpoint facts, assignment facts, and recovery facts
- `openagents` kernel should own acceptance, adjudication, and payout authority
- `TRN` should publish both, but with a clear split between execution record and authority closeout

## 3. No explicit fork or resume parent record in the public plane

`psionic` already has checkpoint pointers, manifests, restore receipts, and contributor lineage.

What is still missing is one end-to-end public continuation record that says:

- this run resumes from this accepted pointer
- this run forks from this prior run or window
- this new policy revision supersedes that prior accepted state

The ingredients exist. The public projection does not.

## 4. No `TRN` lane in the desktop

The desktop still treats mirrored JSON and file-backed explorer artifacts as the training UI substrate.

That has to be inverted.

The desktop should subscribe to `TRN` records first, then optionally fetch:

- remote-training run bundles
- remote-training indexes
- XTRAIN explorer snapshots
- XTRAIN explorer indexes

from artifact pointers.

## 5. No training-specific Nostr client support

The minimal Nostr client in `openagents` does not yet expose:

- training subscriptions
- typed training publish helpers
- gap recovery for training feeds
- encrypted assignment or validator coordination helpers

## Required Changes By Owner

## 1. Psionic

`psionic` should be treated as the source of most `TRN` semantics.

Required changes:

- expose one projection layer from live runtime state into public record shapes
- stop relying only on canonical fixture writers for public training records
- define explicit public continuation or fork records above checkpoint pointers and contributor lineage
- keep artifact locators stable and small enough for relay publication
- keep heavy bytes and runtime transport out of the public plane

## 2. OpenAgents Nostr crates

Required changes:

- add a checked-in `TRN` draft in `crates/nostr/nips`
- add `nip_trn` in `crates/nostr/core`
- define kinds and tags by reusing `psionic` vocabulary, not inventing new one-off desktop names
- add typed publish and subscribe helpers in `crates/nostr/client`

## 3. Nexus and kernel authority

Required changes:

- define the exact bridge between kernel authority records and `psionic` execution records
- publish authority-owned closeout facts into `TRN`
- avoid creating a second public training schema beside the one derived from `psionic`

## 4. Desktop app

Required changes:

- add `TrnLaneWorker` and `TrnLaneSnapshot`
- move training panes to a `TRN`-first read model
- treat remote-training bundles and XTRAIN snapshots as drill-down artifacts
- let `autopilotctl` query live `TRN` state

## Recommended Implementation Order

1. Freeze the `TRN` event family around existing `psionic` concepts.
2. Build a projection layer from `psionic` runtime and contract state into `TRN` records.
3. Define the kernel-authority closeout bridge onto that same record family.
4. Add `TRN` publish and subscribe support in the Nostr crates.
5. Add a `TRN` lane in the desktop.
6. Rebase remote training and XTRAIN explorer onto artifact pointers instead of local file paths.

## Bottom Line

After reading actual `psionic` code, the honest conclusion is:

`TRN` is not blocked by missing training semantics.

It is blocked by missing projection and publication.

`psionic` already has a strong native vocabulary for:

- public network identity
- signed node identity
- discovery and matchmaking
- work windows and assignments
- validator challenge scoring
- multi-validator consensus
- fraud, quarantine, and slashing
- reward and settlement
- checkpoint pointers and restore plans
- evidence bundles and explorer artifacts

That means `TRN` should be built as a thin public event layer over that vocabulary, plus a clean bridge to `openagents` authority truth.

The work ahead is real, but it is more contained than the earlier audit implied. The team does not need to guess at the training nouns. It needs to standardize the publication of nouns it already owns.

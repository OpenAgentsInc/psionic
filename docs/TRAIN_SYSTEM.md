# Psionic Train System

> Status: updated 2026-04-01 after reviewing `docs/MVP.md`,
> `docs/OWNERSHIP.md`,
> `docs/audits/2026-03-13-intellect-lessons-for-psionic-train-audit.md`,
> `docs/audits/2026-03-14-covenant-code-lessons-for-psionic-train-audit.md`,
> `README.md`,
> `docs/ARCHITECTURE.md`,
> `docs/AUTORESEARCH_INTEGRATION_PLAN.md`,
> `docs/kernel/compute-training-authority.md`,
> `docs/headless-compute.md`,
> `crates/psionic-runtime/src/lib.rs`,
> `crates/psionic-datastream/src/lib.rs`,
> `crates/psionic-collectives/src/lib.rs`,
> `crates/psionic-distributed/src/lib.rs`,
> `crates/psionic-train/src/lib.rs`,
> `crates/psionic-environments/src/lib.rs`,
> `crates/psionic-eval/src/lib.rs`,
> `crates/psionic-adapters/src/lib.rs`,
> `crates/psionic-data/src/lib.rs`, and
> `crates/psionic-sandbox/src/lib.rs`,
> `apps/autopilot-desktop/src/desktop_control.rs`, and
> `apps/autopilot-desktop/src/bin/autopilotctl.rs`, plus the recently closed
> train-adjacent issue backlog through `#3643` and the decentralized adapter
> training issue program starting at `#3636`.

## Why This Doc Exists

The March 13 Intellect audit correctly described the shape Psionic should grow
toward, but one part of it is now stale: Psionic no longer lacks a
`psionic-train` crate entirely.

The tree now has:

- `psionic-train`
- `psionic-collectives`
- `psionic-adapters`

That means the right question is no longer "should Psionic have any train
subtree at all?"

The right question is:

> what does the Psionic train system honestly implement today, what does it
> still not implement, and what is the full Rust-native path from the current
> substrate to a real training system?

This doc answers that question.

The train system assumes the execution substrate defined in
`ARCHITECTURE.md` and does not redefine runtime, cluster, sandbox, or artifact
transport behavior.

The remote-training viewer contract now also has a canonical typed artifact
family. Psionic owns the shipped provider-neutral `v1` live bundle and run-index
truth in `crates/psionic-train/src/remote_training_visualization.rs`, the
track-aware `v2` follow-on in
`crates/psionic-train/src/remote_training_visualization_v2.rs`, and the
canonical fixtures under `fixtures/training_visualization/`. Autopilot
consumes that truth through its own cache, projection, and pane code instead
of moving renderer semantics into this repo. `v1` remains the shipped live
substrate. `v2` adds track family, execution class, comparability, proof
posture, public-equivalence, score-law, and cap semantics needed for
`HOMEGOLF` and bounded `XTRAIN`. The `HOMEGOLF` lane now also emits
score-closeout, retained score-delta, and promotion-gate posture through the
same `v2` artifact family instead of leaving that state in sidecar reports.
The bounded `XTRAIN -> PGOLF` lane now also emits one retained quick-eval
source report plus one shared `v2` run bundle with explicit bounded-train-to-
infer proof posture, closed-out local-reference BPB, and a held non-public
promotion gate.

The repo now also owns one canonical actual-lane identifier for `Psion`
pretraining in `crates/psionic-train/src/psion_actual_pretraining_lane.rs`,
the fixture generator
`crates/psionic-train/examples/psion_actual_pretraining_lane_fixtures.rs`, the
focused lane doc `docs/PSION_ACTUAL_PRETRAINING_LANE.md`, and the committed
fixture `fixtures/psion/pretrain/psion_actual_pretraining_lane_spec_v1.json`.
That surface freezes one named actual lane above the bounded
`psion_accelerated_reference_pilot` and ties it explicitly to the admitted
`broader_pretraining` trusted-cluster bundle instead of leaving "actual
pretraining" as prose.

The repo now also owns the canonical actual-lane recipe and admitted
topology/storage bundles in
`crates/psionic-train/src/psion_actual_pretraining_recipe_bundle.rs`, the
fixture generator
`crates/psionic-train/examples/psion_actual_pretraining_recipe_bundle_fixtures.rs`,
the focused recipe doc `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md`, and the
committed fixtures
`fixtures/psion/pretrain/psion_actual_pretraining_recipe_bundle_v1.json` and
`fixtures/psion/pretrain/psion_actual_pretraining_topology_storage_bundle_v1.json`.
That surface freezes one recipe id, one admitted four-node H100 topology and
storage bundle, env-declared credential sources, and one bounded continuation
path of `pretrain -> general_sft -> agentic_sft`.

The repo now also owns the canonical actual-lane output and evidence contract
in `crates/psionic-train/src/psion_actual_pretraining_evidence_contract.rs`,
the fixture generator
`crates/psionic-train/examples/psion_actual_pretraining_evidence_contract_fixtures.rs`,
the focused contract doc
`docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`, and the committed fixture
`fixtures/psion/pretrain/psion_actual_pretraining_evidence_contract_v1.json`.
That surface freezes one retained output family, one required provenance field
set, and one redaction policy for manifests, logs, alerts, resume pointers,
checkpoint manifests, eval receipts, exports, and closeout bundles.

The bounded `./TRAIN` operator path is now also explicitly separated from that
actual-lane truth. It writes bounded reference-pilot artifacts under
`psion_reference_pilot_runs/<run_id>` naming, uses
`reference_pilot_operator_manifest.json` and
`reference_pilot_operator_summary.json`, and states directly that it is not the
`psion_actual_pretraining_v1` launcher.

The repo now also owns the canonical actual-lane status and retained-summary
surface in `crates/psionic-train/src/psion_actual_pretraining_status_surface.rs`,
the fixture generator
`crates/psionic-train/examples/psion_actual_pretraining_status_surface_fixtures.rs`,
the operator helper `scripts/psion-actual-pretraining-status.sh`, the focused
doc `docs/PSION_ACTUAL_PRETRAINING_STATUS_SURFACE.md`, and the committed
fixtures `fixtures/psion/pretrain/psion_actual_pretraining_current_run_status_v1.json`
plus `fixtures/psion/pretrain/psion_actual_pretraining_retained_summary_v1.json`.
That surface now carries the real actual-lane status contract used by the
launcher, including explicit pre-first-checkpoint phases such as
`dry_run_planned` and `launch_staged`.

The repo now also owns the canonical actual-lane launcher and resume contract
in `crates/psionic-train/src/psion_actual_pretraining_launcher.rs`, the
fixture generators
`crates/psionic-train/examples/psion_actual_pretraining_launcher_fixtures.rs`
and `crates/psionic-train/examples/psion_actual_pretraining_operator.rs`, the
entrypoint wrapper `scripts/train-psion-actual-pretraining.sh`, the explicit
lane dispatch in `psionic/TRAIN`, the focused runbook
`docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md`, and the committed fixtures:

- `fixtures/psion/pretrain/psion_actual_pretraining_launch_manifest_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_resume_manifest_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_pointer_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_closeout_bundle_v1.json`

That surface gives the repo one real operator path for:

- `./TRAIN --lane actual_pretraining start`
- `./TRAIN --lane actual_pretraining resume --run-root <path>`
- `./TRAIN --lane actual_pretraining status --run-root <path>`

It consumes the frozen lane, recipe, topology/storage, evidence, and status
contracts directly; refuses dirty working trees by default; retains the
selected ref plus exact commit SHA; and repeats that provenance in the
provisional closeout bundle without claiming that later hardening issues are
already finished.

The repo now also owns the canonical accepted-checkpoint continuation handoff
contract in
`crates/psionic-train/src/psion_actual_pretraining_continuation_handoff.rs`,
the retained-path writer inside
`crates/psionic-train/examples/psion_actual_pretraining_operator.rs`, the
focused doc `docs/PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF.md`, and the
committed fixture
`fixtures/psion/pretrain/psion_actual_pretraining_continuation_handoff_v1.json`.
That surface binds one accepted actual-lane checkpoint to the frozen
`pretrain -> general_sft -> agentic_sft` continuation target and carries the
plugin benchmark-pack bindings already attached to the bounded continuation
stage without pretending that continuation-stage execution has already been
proved.

The repo now also owns a canonical provider-neutral training-program manifest in
`crates/psionic-train/src/cross_provider_training_program_manifest.rs`, the
binary `cross_provider_training_program_manifest`, the checker
`scripts/check-cross-provider-training-program-manifest.sh`, the focused
reference doc `docs/TRAIN_PROGRAM_MANIFEST_REFERENCE.md`, and the committed
fixture `fixtures/training/cross_provider_training_program_manifest_v1.json`.
That manifest freezes one root cross-provider pretraining authority over run id
template, stage authority, checkpoint family, environment key, artifact-root
layout, admitted compute-source classes, admitted execution classes, and one
reserved final-evidence surface, and it now binds its manifest id and digest
directly into `TrainingRunState` before the run graph may claim that program
authority.

The repo now also owns a canonical provider-neutral compute-source contract
family in `crates/psionic-train/src/cross_provider_compute_source_contract.rs`,
the binary `cross_provider_compute_source_contracts`, the checker
`scripts/check-cross-provider-compute-source-contracts.sh`, the focused
reference doc `docs/COMPUTE_SOURCE_CONTRACT_REFERENCE.md`, and the committed
fixtures under `fixtures/training/compute_sources/`. That surface freezes one
training-facing machine contract above the current Google, RunPod, local NVIDIA,
and local Apple artifacts with explicit provider, locality, accelerator,
backend, network, storage, cost, admitted execution classes, typed refusal
examples, and provider-neutral planner plus launch inputs. It keeps unsupported
role claims fail-closed instead of letting each provider lane widen its own
machine semantics.

The repo now also owns a canonical provider-neutral launch-contract family in
`crates/psionic-train/src/cross_provider_launch_contract.rs`, the binary
`cross_provider_launch_contracts`, the checker
`scripts/check-cross-provider-launch-contracts.sh`, the focused reference doc
`docs/LAUNCH_CONTRACT_REFERENCE.md`, and the committed fixtures under
`fixtures/training/launch_contracts/`. That surface freezes one shared runtime
envelope above the current Google single-node, Google swarm, RunPod, and local
trusted-LAN launchers: explicit runtime env, artifact roots, cluster-port
bindings, startup expectations, finalizer expectations, and projected
provider-specific step sequences. Resource creation remains provider-specific,
but the training-facing launch semantics are now typed in one place.

The repo now also owns the first provider-neutral runtime binder in
`crates/psionic-train/src/cross_provider_runtime_binder.rs`, the binary
`cross_provider_runtime_binder`, the checker
`scripts/check-cross-provider-runtime-binder.sh`, the focused reference doc
`docs/CROSS_PROVIDER_RUNTIME_BINDER_REFERENCE.md`, and the committed fixture
`fixtures/training/cross_provider_runtime_binder_v1.json`. That surface binds
the root program manifest, admitted compute sources, shared launch contracts,
shared runtime env, shared artifact backends, and provider-owned hooks into one
machine-legible launch-time authority above the current Google, RunPod, and
local adapters. Resource creation remains provider-specific, but the adapters
no longer define training-facing runtime truth on their own.

The Google lanes now also consume that binder explicitly through
`crates/psionic-train/src/google_training_binder_projection.rs`, the binary
`google_training_binder_projection`, the checker
`scripts/check-google-training-binder-projection.sh`, the focused reference doc
`docs/GOOGLE_TRAINING_BINDER_REFERENCE.md`, and the committed fixture
`fixtures/training/google_training_binder_projection_v1.json`. That projection
keeps the current single-node and two-node swarm Google operator surfaces
truthful while moving their runtime semantics, retained evidence surfaces, and
finalizer expectations onto the shared binder instead of Google-only training
truth.

The current RunPod and local trusted-LAN lanes now also consume that binder
explicitly through
`crates/psionic-train/src/runpod_local_training_binder_projection.rs`, the
binary `runpod_local_training_binder_projection`, the checker
`scripts/check-runpod-local-training-binder-projection.sh`, the focused
reference doc `docs/RUNPOD_LOCAL_TRAINING_BINDER_REFERENCE.md`, and the
committed fixture
`fixtures/training/runpod_local_training_binder_projection_v1.json`. That
projection keeps the current RunPod `8xH100` and first trusted-LAN swarm
operator surfaces bounded and truthful while moving their launch, runtime env,
artifact-root, and final evidence semantics onto the same shared binder used by
the Google lanes.

The repo now also owns the first cross-provider admission planner in
`crates/psionic-train/src/cross_provider_admission_planner.rs`, the binary
`cross_provider_admission_plan`, the checker
`scripts/check-cross-provider-admission-planner.sh`, the focused reference doc
`docs/CROSS_PROVIDER_ADMISSION_PLANNER_REFERENCE.md`, and the committed fixture
`fixtures/training/cross_provider_admission_plan_v1.json`. That surface turns
the retained source contracts plus the shared binder into one deterministic
role-placement policy with score breakdowns across trust, network, storage,
cost, backend, and binder alignment instead of leaving source admission to
runbook prose or operator instinct.

The repo now also owns the first contributor program-lineage bridge in
`crates/psionic-train/src/contributor_program_lineage.rs`, the binary
`contributor_program_lineage`, the checker
`scripts/check-contributor-program-lineage.sh`, the focused reference doc
`docs/CONTRIBUTOR_PROGRAM_LINEAGE_REFERENCE.md`, and the committed fixture
`fixtures/training/contributor_program_lineage_v1.json`. That surface binds the
current validated contributor windows to the same canonical dataset family,
checkpoint family, and shared policy revision used by the hybrid dense program,
and it freezes one promotion contract per contributor window so later accepted
and no-promotion outcomes stay machine-legible under one program identity.

The repo now also owns the first shared validator and promotion contract in
`crates/psionic-train/src/validator_promotion_contract.rs`, the binary
`shared_validator_promotion_contract`, the checker
`scripts/check-shared-validator-promotion-contract.sh`, the focused reference
doc `docs/SHARED_VALIDATOR_PROMOTION_CONTRACT_REFERENCE.md`, and the committed
fixture `fixtures/training/shared_validator_promotion_contract_v1.json`. That
surface freezes one shared vocabulary for `accepted`, `quarantined`,
`rejected`, `replay_required`, `promoted_revision`, `held_no_promotion`, and
`refused_promotion`, and the provider-neutral evidence bundle now carries that
contract id directly instead of treating validator and promotion language as
lane-local convention.

The repo now also owns the first whole-program cross-provider run graph in
`crates/psionic-train/src/cross_provider_program_run_graph.rs`, the binary
`cross_provider_program_run_graph`, the checker
`scripts/check-cross-provider-program-run-graph.sh`, the focused reference doc
`docs/CROSS_PROVIDER_PROGRAM_RUN_GRAPH_REFERENCE.md`, and the committed fixture
`fixtures/training/cross_provider_program_run_graph_v1.json`. That surface
reuses the existing run graph and orchestrator state, then layers typed
whole-program role participants and role-window composition over them so one
shared run id can carry dense ranks, validated contributor windows, validators,
checkpoint writers, eval workers, and data builders at the same time without
splitting provider-local side jobs into hidden program identities.

The repo now also owns the first decentralized network epoch, role, and
governance contract in
`crates/psionic-train/src/decentralized_network_contract.rs`, the binary
`decentralized_network_contract`, the checker
`scripts/check-decentralized-network-contract.sh`, the focused reference doc
`docs/DECENTRALIZED_NETWORK_CONTRACT_REFERENCE.md`, and the committed fixture
`fixtures/training/decentralized_network_contract_v1.json`. That surface binds
the current provider-neutral program manifest, whole-program run graph, and
shared validator-promotion vocabulary into one explicit decentralized network
object: network id, governance revision, permissioned-testnet registration
posture, fixed public epoch cadence, signed-ledger settlement posture,
checkpoint-authority quorum policy, and one retained public role set covering
`public_miner`, `public_validator`, `relay`, `checkpoint_authority`, and
`aggregator`. It keeps `relay` explicit as a network-only support role instead
of pretending the current run graph already ships a dedicated public relay
execution class.

The repo now also owns the first signed public-node identity contract set in
`crates/psionic-train/src/signed_node_identity_contract.rs`, the binary
`signed_node_identity_contract_set`, the checker
`scripts/check-signed-node-identity-contract-set.sh`, the focused reference doc
`docs/SIGNED_NODE_IDENTITY_CONTRACT_REFERENCE.md`, and the committed fixture
`fixtures/training/signed_node_identity_contract_set_v1.json`. That surface
binds each current canonical compute source to one signed node identity record:
wallet namespace, deterministic software build digest, capability projection
digests over accelerator or backend or network or storage posture, retained
benchmark evidence, admitted public roles, admitted execution classes, typed
refusal examples, and an explicit revocation-feed policy. It keeps the current
role gap honest: RunPod still does not claim `public_miner` because the current
network binds that role to `validated_contributor_window` rather than
`dense_full_model_rank`, and dense-rank capability remains outside the current
public role map until later decentralized runtime issues land.

The repo now also owns the first public-network registry, discovery, and
matchmaking contract in
`crates/psionic-train/src/public_network_registry_contract.rs`, the binary
`public_network_registry_contract`, the checker
`scripts/check-public-network-registry-contract.sh`, the focused reference doc
`docs/PUBLIC_NETWORK_REGISTRY_REFERENCE.md`, and the committed fixture
`fixtures/training/public_network_registry_contract_v1.json`. That surface
binds the signed node identity set into one permissioned-testnet registry:
registry record per node, current epoch id, compatibility policy over release
id plus environment plus manifest digest plus revocation posture, endpoint and
relay posture per node, typed discovery filters and refusal reasons, and typed
matchmaking offers for contributor-window miners, validator quorum, and
checkpoint promotion. It keeps current network shape explicit instead of hidden
in host lists: Google is the only current relay match, Google plus Apple MLX
close the current validator quorum, and Google plus RunPod remain the current
checkpoint-authority pair.

The repo now also owns the first elastic device mesh contract in
`crates/psionic-train/src/elastic_device_mesh_contract.rs`, the binary
`elastic_device_mesh_contract`, the checker
`scripts/check-elastic-device-mesh-contract.sh`, the focused reference doc
`docs/ELASTIC_DEVICE_MESH_REFERENCE.md`, and the committed fixture
`fixtures/training/elastic_device_mesh_contract_v1.json`. That surface turns
the registry into runtime-managed mesh truth: role-specific lease policy,
current member leases, retained heartbeat samples, explicit deathrattle
notices, and typed revision receipts for activation, replacement, and refusal.
It proves one real public-role replacement path, with Apple MLX promoted after
the RTX 4080 miner deathrattle, while keeping one critical honesty boundary:
live dense remove-without-replacement still refuses and remains explicitly
bound to the older refused topology and recovery scenarios until later runtime
and transport issues close.

The repo now also owns the first WAN overlay and relay route contract in
`crates/psionic-train/src/wan_overlay_route_contract.rs`, the binary
`wan_overlay_route_contract`, the checker
`scripts/check-wan-overlay-route-contract.sh`, the focused reference doc
`docs/WAN_OVERLAY_ROUTE_REFERENCE.md`, and the committed fixture
`fixtures/training/wan_overlay_route_contract_v1.json`. That surface turns the
public registry plus elastic mesh into internet-native path truth: one NAT
posture record per admitted node, retained route-quality samples, typed
direct-vs-relayed-vs-overlay route selection, and explicit failover receipts
for the same peer pair. It proves one honest transport move, from a relay-only
miner path to an overlay tunnel after packet-loss overflow, while still
refusing to pretend catch-up, outer sync, or public internet soak closure
already exist.

The repo now also owns the first live checkpoint catch-up contract in
`crates/psionic-train/src/live_checkpoint_catchup_contract.rs`, the binary
`live_checkpoint_catchup_contract`, the checker
`scripts/check-live-checkpoint-catchup-contract.sh`, the focused reference doc
`docs/LIVE_CHECKPOINT_CATCHUP_REFERENCE.md`, and the committed fixture
`fixtures/training/live_checkpoint_catchup_contract_v1.json`. That surface
turns the distributed checkpoint plus WAN route truth into a real join-time
recovery layer: admitted checkpoint advertisements, explicit freshness windows,
one completed replacement catch-up, and one refused stale or optimizer-thin
sidecar attempt. It proves Apple MLX can rejoin the public-miner window
through RunPod over the overlay path while keeping one critical honesty
boundary explicit: active-peer sidecars are not equivalent to full
checkpoint-authority recovery.

The repo now also owns the first quantized outer-sync contract in
`crates/psionic-train/src/quantized_outer_sync_contract.rs`, the binary
`quantized_outer_sync_contract`, the checker
`scripts/check-quantized-outer-sync-contract.sh`, the focused reference doc
`docs/QUANTIZED_OUTER_SYNC_REFERENCE.md`, and the committed fixture
`fixtures/training/quantized_outer_sync_contract_v1.json`. That surface turns
live catch-up into a WAN-feasible synchronization story: explicit quantized
delta policies, applied exchange receipts into a checkpoint authority,
bandwidth accounting, correctness receipts, and one refused full-precision WAN
path. It proves Google plus Apple MLX can both contribute compressed deltas to
RunPod after the MLX rejoin while keeping the dense all-reduce honesty
boundary explicit.

The repo now also owns the first internet fault and soak harness contract in
`crates/psionic-train/src/internet_fault_harness_contract.rs`, the binary
`internet_fault_harness_contract`, the checker
`scripts/check-internet-fault-harness-contract.sh`, the focused reference doc
`docs/INTERNET_FAULT_HARNESS_REFERENCE.md`, and the committed fixture
`fixtures/training/internet_fault_harness_contract_v1.json`. That surface turns
route, catch-up, and outer-sync truth into a real promotion gate: explicit
fault profiles, retained throughput baselines, suite-level pass thresholds,
repeated passed runs, and one held validator-loss result. It proves Psionic can
retain day plus night evidence for failover, catch-up, and throttled outer
sync while keeping one non-negotiable stop condition explicit: losing the MLX
validator collapses the current quorum and the run holds.

The repo now also owns the first public window clock and deterministic work
assignment contract in `crates/psionic-train/src/public_work_assignment_contract.rs`,
the binary `public_work_assignment_contract`, the checker
`scripts/check-public-work-assignment-contract.sh`, the focused reference doc
`docs/PUBLIC_WORK_ASSIGNMENT_REFERENCE.md`, and the committed fixture
`fixtures/training/public_work_assignment_contract_v1.json`. That surface turns
the public mesh into network-time work truth: explicit public windows,
deterministic miner assignments, deterministic validator challenges, assignment
receipts, and one late-window refusal. It proves the network can say exactly
why Google and Apple MLX worked on one page slice in one window while keeping
window-closure discipline explicit.

The repo now also owns the first public dataset authority contract in
`crates/psionic-train/src/public_dataset_authority_contract.rs`, the binary
`public_dataset_authority_contract`, the checker
`scripts/check-public-dataset-authority-contract.sh`, the focused reference doc
`docs/PUBLIC_DATASET_AUTHORITY_REFERENCE.md`, and the committed fixture
`fixtures/training/public_dataset_authority_contract_v1.json`. That surface
turns deterministic public work into replay-safe data truth: tokenizer and
packing digests, page definitions, page proofs over the committed tokenized
corpus, admitted miner data receipts, and one refused duplicate claim. It
proves public work can bind to real shard lineage instead of self-reported page
names.

The repo now also owns the first content-addressed artifact exchange contract
in `crates/psionic-train/src/content_addressed_artifact_exchange_contract.rs`,
the binary `content_addressed_artifact_exchange_contract`, the checker
`scripts/check-content-addressed-artifact-exchange-contract.sh`, the focused
reference doc `docs/CONTENT_ADDRESSED_ARTIFACT_EXCHANGE_REFERENCE.md`, and the
committed fixture
`fixtures/training/content_addressed_artifact_exchange_contract_v1.json`. That
surface turns public runtime receipts into portable artifact truth: peer seeds,
relay caches, authoritative stores, content ids for deltas, gradient slices,
checkpoints, and provisional score artifacts, plus one explicit
digest-mismatch refusal. It proves public artifact transport can fail closed on
corruption instead of quietly trusting whichever path returned bytes first.

The repo now also owns the first public miner protocol contract in
`crates/psionic-train/src/public_miner_protocol_contract.rs`, the binary
`public_miner_protocol_contract`, the checker
`scripts/check-public-miner-protocol-contract.sh`, the focused reference doc
`docs/PUBLIC_MINER_PROTOCOL_REFERENCE.md`, and the committed fixture
`fixtures/training/public_miner_protocol_contract_v1.json`. That surface turns
the public miner lane into a typed execution protocol: one execution-class
binding, one bounded retry policy, active miner sessions, local-step receipts,
delta publication receipts, checkpoint-sync receipts, and one explicit stale
standby refusal. It proves public miner behavior no longer lives only in
runbooks or implied runtime sequencing.

The repo now also owns the first validator challenge and scoring contract in
`crates/psionic-train/src/validator_challenge_scoring_contract.rs`, the binary
`validator_challenge_scoring_contract`, the checker
`scripts/check-validator-challenge-scoring-contract.sh`, the focused reference
doc `docs/VALIDATOR_CHALLENGE_SCORING_REFERENCE.md`, and the committed fixture
`fixtures/training/validator_challenge_scoring_contract_v1.json`. That surface
turns public validator work into typed scoring truth: replay rules, improvement
thresholds, validator receipts, and one stale-checkpoint refusal. It proves the
network can explain why a contribution was accepted or replay-required without
falling back to validator-local prose.

The repo now also owns the first multi-validator consensus contract in
`crates/psionic-train/src/multi_validator_consensus_contract.rs`, the binary
`multi_validator_consensus_contract`, the checker
`scripts/check-multi-validator-consensus-contract.sh`, the focused reference
doc `docs/MULTI_VALIDATOR_CONSENSUS_REFERENCE.md`, and the committed fixture
`fixtures/training/multi_validator_consensus_contract_v1.json`. That surface
turns checkpoint authority into explicit network governance: quorum policy,
weighted validator votes, held-no-promotion decisions, and disagreement
receipts. It proves checkpoint promotion no longer reduces to one validator or
one maintainer making an informal call.

The repo now also owns the first fraud, quarantine, and slashing contract in
`crates/psionic-train/src/fraud_quarantine_slashing_contract.rs`, the binary
`fraud_quarantine_slashing_contract`, the checker
`scripts/check-fraud-quarantine-slashing-contract.sh`, the focused reference
doc `docs/FRAUD_QUARANTINE_SLASHING_REFERENCE.md`, and the committed fixture
`fixtures/training/fraud_quarantine_slashing_contract_v1.json`. That surface
turns adversarial discipline into typed public truth: sybil-watch signals,
duplicate-work evidence, observation versus blocked quarantines, one slashing
decision, and one explicit appeal window. It proves the decentralized network
can now fail closed on known miner fraud modes without retreating to an
implicit maintainer allowlist.

The repo now also owns the first public contribution and reward ledger contract
in `crates/psionic-train/src/reward_ledger_contract.rs`, the binary
`reward_ledger_contract`, the checker
`scripts/check-reward-ledger-contract.sh`, the focused reference doc
`docs/REWARD_LEDGER_REFERENCE.md`, and the committed fixture
`fixtures/training/reward_ledger_contract_v1.json`. That surface turns public
scoring and penalties into one tamper-evident accounting period: retained
miner, validator, and checkpoint-authority work entries, retained penalty
entries, and payout-ready net allocations. It proves the network can now say
who earned what and who was penalized under one shared accounting surface.

The repo now also owns the first settlement publication contract in
`crates/psionic-train/src/settlement_publication_contract.rs`, the binary
`settlement_publication_contract`, the checker
`scripts/check-settlement-publication-contract.sh`, the focused reference doc
`docs/SETTLEMENT_PUBLICATION_REFERENCE.md`, and the committed fixture
`fixtures/training/settlement_publication_contract_v1.json`. That surface turns
closed-window accounting into publishable outcome truth: validator-weight
publication, one signed-ledger settlement record, payout exports bound to
wallet identities, and one explicit chain-adapter refusal. It proves Psionic
now has a truthful settlement surface before public dashboards or open
operator-facing packages land.

The repo now also owns the first public operator bootstrap package contract in
`crates/psionic-train/src/operator_bootstrap_package_contract.rs`, the binary
`operator_bootstrap_package_contract`, the checker
`scripts/check-operator-bootstrap-package-contract.sh`, the focused reference
doc `docs/OPERATOR_BOOTSTRAP_PACKAGE_REFERENCE.md`, and the committed fixture
`fixtures/training/operator_bootstrap_package_contract_v1.json`. That surface
turns public miner and validator onboarding into typed package truth:
reproducible images, env manifests, registration commands, dry-run commands,
and role-specific preflight checks. It proves the first public operator path no
longer depends on source-level patching or private setup lore.

The repo now also owns the first public run explorer contract in
`crates/psionic-train/src/public_run_explorer_contract.rs`, the binary
`public_run_explorer_contract`, the checker
`scripts/check-public-run-explorer-contract.sh`, the focused reference doc
`docs/PUBLIC_RUN_EXPLORER_REFERENCE.md`, and the committed fixture
`fixtures/training/public_run_explorer_contract_v1.json`. That surface turns
network health and scoring visibility into typed public truth: explorer panes,
one current network snapshot, score rows reconciled against the reward ledger,
and explicit stale-data policy. It proves Psionic now has a public-facing
status surface above raw logs and private maintainer dashboards.

The repo now also owns the first decentralized `XTRAIN` explorer artifact
family in `crates/psionic-train/src/xtrain_explorer_artifacts.rs`, the binary
`xtrain_explorer_artifacts`, the checker
`scripts/check-xtrain-explorer-artifacts.sh`, the focused reference doc
`docs/XTRAIN_EXPLORER_REFERENCE.md`, and the committed fixtures
`fixtures/training/xtrain_explorer_snapshot_v1.json` plus
`fixtures/training/xtrain_explorer_index_v1.json`. That surface turns the
public-explorer foundation into pane-ready decentralized ML truth: participant
graph state, one retained active window, one held checkpoint promotion, signed
settlement posture, explorer event rows, and one explicit sibling link back to
the bounded `XTRAIN -> PGOLF` run-centric visualization bundle. It proves
Psionic can drive the first honest `XTRAIN Explorer` pane without collapsing
decentralized network state into the bounded training-run dashboard family.

The repo now also owns the first staged public-testnet readiness contract in
`crates/psionic-train/src/public_testnet_readiness_contract.rs`, the binary
`public_testnet_readiness_contract`, the checker
`scripts/check-public-testnet-readiness-contract.sh`, the focused reference doc
`docs/PUBLIC_TESTNET_READINESS_REFERENCE.md`, and the committed fixture
`fixtures/training/public_testnet_readiness_contract_v1.json`. That surface
turns public participation rollout into typed gate truth: candidate records,
compliance receipts, reward-eligible versus canary decisions, and explicit
blocked admission tied to fraud policy. It proves Psionic can now graduate or
refuse public participants through one machine-legible staged onboarding path
instead of treating first contact with the internet as the only validation
mechanism.

The repo now also owns the first curated decentralized run contract in
`crates/psionic-train/src/curated_decentralized_run_contract.rs`, the binary
`curated_decentralized_run_contract`, the checker
`scripts/check-curated-decentralized-run-contract.sh`, the focused reference
doc `docs/CURATED_DECENTRALIZED_RUN_REFERENCE.md`, the committed fixture
`fixtures/training/curated_decentralized_run_contract_v1.json`, and the
after-action audit
`docs/audits/2026-03-26-curated-decentralized-run-after-action-audit.md`. That
surface turns the first permissioned internet run into retained proof: explicit
participants, one retained evidence bundle, and one honest after-action audit.

The repo now also owns the first open public decentralized run contract in
`crates/psionic-train/src/open_public_decentralized_run_contract.rs`, the
binary `open_public_decentralized_run_contract`, the checker
`scripts/check-open-public-decentralized-run-contract.sh`, the focused
reference doc `docs/OPEN_PUBLIC_DECENTRALIZED_RUN_REFERENCE.md`, the committed
fixture `fixtures/training/open_public_decentralized_run_contract_v1.json`, and
the public participation audit
`docs/audits/2026-03-26-open-public-miner-validator-run-audit.md`. That
surface turns the first outside-operator participation window into retained
proof: outside canary candidates, public score visibility, and explicit
blocked-fraud evidence.

The repo now also owns the first incentivized decentralized run contract in
`crates/psionic-train/src/incentivized_decentralized_run_contract.rs`, the
binary `incentivized_decentralized_run_contract`, the checker
`scripts/check-incentivized-decentralized-run-contract.sh`, the focused
reference doc `docs/INCENTIVIZED_DECENTRALIZED_RUN_REFERENCE.md`, the
committed fixture
`fixtures/training/incentivized_decentralized_run_contract_v1.json`, and the
incentives-focused audit
`docs/audits/2026-03-26-incentivized-decentralized-run-audit.md`. That surface
turns the first rewarded decentralized closeout into retained proof: paid
participants, payout publication, published validator weights, and one explicit
incentives audit.

The repo now also owns the first dense-rank recovery contract in
`crates/psionic-train/src/dense_rank_recovery_contract.rs`, the binary
`dense_rank_recovery_contract`, the checker
`scripts/check-dense-rank-recovery-contract.sh`, the focused reference doc
`docs/DENSE_RANK_RECOVERY_REFERENCE.md`, and the committed fixture
`fixtures/training/dense_rank_recovery_contract_v1.json`. That surface closes
the current admitted dense recovery stories against real substrate truth:
checkpoint shard restore assignments, checkpoint artifact placement and restore
authority, replay-order continuity for same-rank replacement, and an explicit
shrink-world refusal instead of a hidden best-effort fallback.

The repo now also owns the first controlled dense topology-revision contract in
`crates/psionic-train/src/dense_topology_revision_contract.rs`, the binary
`dense_topology_revision_contract`, the checker
`scripts/check-dense-topology-revision-contract.sh`, the focused reference doc
`docs/DENSE_TOPOLOGY_REVISION_REFERENCE.md`, and the committed fixture
`fixtures/training/dense_topology_revision_contract_v1.json`. That surface
keeps three revision classes explicit: hot replace-rank, checkpoint-barrier
grow-world, and checkpoint-barrier shrink-world. It still refuses live
remove-without-replacement instead of pretending the current fixed-world
data-feed path already does generic live elasticity.

The repo now also owns the first retained multi-provider dense CUDA proof-run
bundle in `crates/psionic-train/src/first_multi_provider_dense_cuda_run.rs`,
the binary `first_multi_provider_dense_cuda_run`, the checker
`scripts/check-first-multi-provider-dense-cuda-run.sh`, the committed fixture
`fixtures/training/first_multi_provider_dense_cuda_run_v1.json`, and the
after-action audit
`docs/audits/2026-03-25-first-multi-provider-dense-cuda-run-audit.md`. That
bundle records one bounded Google plus RunPod dense CUDA program that widened
through a checkpoint-barrier grow-world revision, retained a provider-loss
replace-rank recovery event, and closed as `bounded_success` under the shared
cross-provider contracts.

The repo now also owns the first generic dense-rank runtime layer in
`crates/psionic-train/src/dense_rank_runtime.rs`, the binary
`dense_rank_runtime_reference_contract`, the checker
`scripts/check-dense-rank-runtime-reference-contract.sh`, the focused reference
doc `docs/DENSE_RANK_RUNTIME_REFERENCE.md`, and the committed fixture
`fixtures/training/dense_rank_runtime_reference_contract_v1.json`. That surface
promotes the real PGOLF CUDA `8xH100` bootstrap and train-step path into one
shared dense-rank runtime receipt family with explicit runtime identity,
validation-hook contract, checkpoint-hook contract, and generic execution
receipt semantics. PGOLF remains one consumer lane, but it no longer owns the
only dense distributed runtime receipt model in the repo.

The repo now also owns the first hybrid pretraining planner in
`crates/psionic-train/src/hybrid_pretraining_planner.rs`, the binary
`hybrid_pretraining_plan`, the checker
`scripts/check-hybrid-pretraining-plan.sh`, the focused reference doc
`docs/HYBRID_PRETRAINING_PLANNER_REFERENCE.md`, and the committed fixture
`fixtures/training/hybrid_pretraining_plan_v1.json`. That surface freezes one
machine-legible planning layer above the current dense and contributor
substrates: explicit dense-rank assignments, validated contributor windows,
validators, eval workers, checkpoint writers, and shared lineage slots under
one dataset family and one checkpoint family. It keeps work classes explicit
instead of flattening contributor windows into dense ranks or inventing
provider-specific planner vocabularies.

The repo now also owns the first provider-neutral distributed checkpoint
contract in `crates/psionic-train/src/distributed_checkpoint_contract.rs`, the
binary `sharded_distributed_checkpoint_contract`, the checker
`scripts/check-sharded-distributed-checkpoint-contract.sh`, the focused
reference doc `docs/SHARDED_DISTRIBUTED_CHECKPOINT_REFERENCE.md`, and the
fixture `fixtures/training/sharded_distributed_checkpoint_contract_v1.json`.
That surface extends the older pointer-first checkpoint recovery layer with
typed parameter-shard and optimizer-shard placements, durable and refused shard
upload receipts, and deterministic dense-rank restore assignments under one
provider-neutral checkpoint family.

The repo now also owns the first provider-neutral remote artifact backend layer
in `crates/psionic-train/src/remote_artifact_backend_contract.rs`, the binary
`remote_train_artifact_backend_contract`, the checker
`scripts/check-remote-train-artifact-backend-contract.sh`, the focused
reference doc `docs/REMOTE_TRAIN_ARTIFACT_BACKEND_REFERENCE.md`, and the
fixture `fixtures/training/remote_train_artifact_backend_contract_v1.json`.
That surface adds one shared remote backend trait, concrete Google and RunPod
backends, byte-accounted placement policy, restore policy, and finalizer
projections for checkpoints, logs, metrics bundles, and final evidence
bundles.

The repo now also owns the first provider-neutral final evidence bundle family
in `crates/psionic-train/src/training_execution_evidence_bundle.rs`, the
binary `training_execution_evidence_bundle`, the checker
`scripts/check-training-execution-evidence-bundle.sh`, the focused reference
doc `docs/TRAINING_EXECUTION_EVIDENCE_REFERENCE.md`, and the fixture
`fixtures/training/provider_neutral_training_execution_evidence_bundle_v1.json`.
That surface seals launch facts, runtime facts, checkpoints, metrics,
visualization refs, validator results, and final disposition under one schema
family across single-node, dense-distributed, contributor-window,
validator-only, and hybrid runs. It now also carries explicit
surface-to-evidence links for the track-aware `v2` run bundles and the
decentralized `XTRAIN Explorer` artifact family, so score or explorer drilldown
can resolve retained proof mechanically from the evidence bundle itself.

The first local mixed-hardware swarm lane now also has a canonical machine-
legible contract in `crates/psionic-train/src/swarm_open_adapter.rs` plus the
committed fixture `fixtures/swarm/first_swarm_run_contract_v1.json`. That
contract freezes the first Mac MLX Metal plus Linux RTX 4080 CUDA lane as one
decentralized open-adapter delta program with explicit validator, replay,
local-snapshot-only publication, and no-full-model-overclaim posture.

That lane now also has one shared comparable receipt contract in
`crates/psionic-train/src/swarm_open_adapter_receipt.rs` plus the committed
fixture `fixtures/swarm/first_swarm_open_adapter_receipt_contract_v1.json`.
That contract freezes the first swarm lane to one `f32`-only open-adapter
receipt language with explicit backend label, logical-device identity, replay
identity, adapter family and format, tokenizer and base-model identity, and
hidden-state geometry truth before aggregation may accept both contributors.

The Mac node now also has a dedicated bring-up report seam in
`crates/psionic-train/src/swarm_mlx_bringup.rs`, the binary
`swarm_mac_mlx_bringup`, the verification runner
`scripts/check-swarm-mac-mlx-bringup.sh`, and the committed report
`fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json`. That report records
real local Mac identity plus the bounded Metal array surface and one bounded
same-node open-adapter overfit gate under the backend label
`open_adapter_backend.mlx.metal.gpt_oss_lm_head`. The gate keeps the fixed-
budget trainer host-owned, but it emits backend-tagged execution provenance,
artifact identity, explicit unsupported-precision refusal, and the shared first
swarm contributor receipt for the Mac swarm contributor lane.

The repo now also owns the first bounded MLX dense-rank runtime contract in
`crates/psionic-train/src/mlx_dense_rank_runtime.rs`, the binary
`mlx_dense_rank_runtime_contract`, the checker
`scripts/check-mlx-dense-rank-runtime-contract.sh`, the focused reference doc
`docs/MLX_DENSE_RANK_RUNTIME_REFERENCE.md`, and the committed fixture
`fixtures/training/mlx_dense_rank_runtime_contract_v1.json`. That surface lifts
the local Apple lane out of contributor-only status: one single-rank MLX Metal
dense runtime now emits the same generic dense-rank bootstrap and train-step
receipt family as the CUDA reference runtime, plus one retained single-rank
checkpoint manifest and pointer, one retained local metric-event set, and one
explicit final-evidence projection. It still refuses cross-host collectives,
same-job mixed-backend dense meshes, and sharded optimizer exchange.

The repo now also owns the first shared CUDA-plus-MLX dense mesh math contract
in `crates/psionic-train/src/cross_backend_cuda_mlx_dense_mesh.rs`, the binary
`cross_backend_cuda_mlx_dense_mesh_contract`, the checker
`scripts/check-cross-backend-cuda-mlx-dense-mesh-contract.sh`, the focused
reference doc `docs/CROSS_BACKEND_CUDA_MLX_DENSE_MESH_REFERENCE.md`, and the
committed fixture
`fixtures/training/cross_backend_cuda_mlx_dense_mesh_contract_v1.json`. That
surface freezes one explicit mixed-backend law above the generic CUDA dense
runtime and the MLX dense-rank runtime: fp32 gradient all-reduce, fp32
master-weight broadcast, mirrored fp32 AdamW state, and one explicit refusal
set for BF16 mixed precision, fp16 loss scaling, direct NCCL claims by MLX
ranks, split master-weight authority, and checkpointless optimizer migration.

The repo now also owns the first mixed-backend checkpoint and restore contract
in `crates/psionic-train/src/mixed_backend_checkpoint_contract.rs`, the binary
`mixed_backend_checkpoint_contract`, the checker
`scripts/check-mixed-backend-checkpoint-contract.sh`, the focused reference doc
`docs/MIXED_BACKEND_CHECKPOINT_REFERENCE.md`, and the committed fixture
`fixtures/training/mixed_backend_checkpoint_contract_v1.json`. That surface
freezes one shared checkpoint manifest and pointer, one portable fp32
safetensors-backed state receipt per backend, one restore ladder that covers
same-backend resume plus CUDA-to-MLX and MLX-to-CUDA restore, and one explicit
refusal set for BF16 optimizer-state migration, quantized checkpoint resume,
checkpointless migration, and incomplete portable group selection.

The repo now also owns the first bounded same-job MLX-plus-CUDA dense proof-run
bundle in
`crates/psionic-train/src/first_same_job_mixed_backend_dense_run.rs`, the
binary `first_same_job_mixed_backend_dense_run`, the checker
`scripts/check-first-same-job-mixed-backend-dense-run.sh`, the committed
fixture `fixtures/training/first_same_job_mixed_backend_dense_run_v1.json`, and
the acceptance audit
`docs/audits/2026-03-25-first-same-job-mlx-plus-cuda-dense-run-audit.md`. That
surface closes one bounded same-job dense pretraining proof across one local
MLX Metal rank and one RunPod CUDA dense participant under the shared fp32
cross-backend mesh law and the mixed-backend checkpoint family. It retains one
shared run id, one explicit checkpoint barrier plus resume event, inline step
metrics, and exact proof boundaries that still refuse BF16 mixed precision,
sharded optimizer exchange, local RTX 4080 dense closure, and broad production
rollout claims.

The Linux node now also has a dedicated RTX 4080 bring-up seam in
`crates/psionic-train/src/swarm_cuda_bringup.rs`, the binary
`swarm_linux_cuda_bringup`, the verification runner
`scripts/check-swarm-linux-4080-bringup.sh`, and the committed report
`fixtures/swarm/reports/swarm_linux_rtx4080_bringup_v1.json`. That report
binds the lane to retained RTX 4080 CUDA inventory truth plus a deterministic
open-adapter same-node harness with explicit unsupported-precision refusal and
the same comparable contributor receipt contract used by the Mac lane.

MLX planning for that lane no longer stops at package-local artifacts. The repo
now also owns a first swarm live planning bridge in
`crates/psionic-mlx-workflows/src/swarm_live_plan.rs`, the binary
`first_swarm_live_workflow_plan`, and the committed fixture
`fixtures/swarm/first_swarm_live_workflow_plan_v1.json`. That bridge consumes
one MLX recipe plan, one synthetic dataset artifact, and one local publish
config and feeds them into `AdapterTrainingClusterCoordinator` through explicit
mixed-backend contributor selection plus one shared capability policy that
admits both the Mac MLX Metal and Linux CUDA lanes without introducing a second
trainer or notebook side control plane.

That lane now also has one exact trusted-LAN cluster contract in
`crates/psionic-train/src/swarm_trusted_lan.rs`, the binaries
`first_swarm_trusted_lan_topology_contract` and
`first_swarm_trusted_lan_failure_drills`, the launcher
`scripts/first-swarm-launch-trusted-lan.sh`, the checker
`scripts/check-first-swarm-trusted-lan.sh`, the narrow runbook
`docs/FIRST_SWARM_TRUSTED_LAN_RUNBOOK.md`, and the committed fixtures
`fixtures/swarm/first_swarm_trusted_lan_topology_contract_v1.json` plus
`fixtures/swarm/reports/first_swarm_trusted_lan_failure_drills_v1.json`. That
surface freezes the exact two-node trusted-LAN topology, artifact staging,
heartbeat and stale-worker thresholds, per-host bring-up commands, and the
required stale-worker, upload-disagreement, contributor-loss, and skew drills
before the later rehearsal and live-run issues claim anything broader.

The Google follow-on now also has one exact configured-peer topology and
manifest contract in
`crates/psionic-train/src/psion_google_two_node_swarm_contract.rs`, the binary
`psion_google_two_node_swarm_contract`, the checker
`scripts/check-psion-google-two-node-swarm-contract.sh`, and the committed
fixture `fixtures/psion/google/psion_google_two_node_swarm_contract_v1.json`.
That surface freezes the first Google swarm lane to two CUDA-backed `g2` plus
`L4` nodes in `openagentsgemini`, one configured-peer cluster namespace, one
explicit zone-pair fallback order, distinct dedicated training subnetworks, the
reserved dual-node operator artifact paths, the admitted network-impairment
profile ids, and the exact bounded result classes before later launch, runbook,
or live-run issues claim the full operator path exists.

The repo now also owns the first Google dual-node operator-preflight surfaces
for that lane in
`fixtures/psion/google/psion_google_two_node_swarm_network_posture_v1.json`,
`fixtures/psion/google/psion_google_two_node_swarm_identity_profile_v1.json`,
`fixtures/psion/google/psion_google_two_node_swarm_operator_preflight_policy_v1.json`,
`scripts/psion-google-ensure-two-node-swarm-network.sh`,
`scripts/psion-google-ensure-two-node-swarm-service-account.sh`,
`scripts/psion-google-quota-preflight-two-node-swarm.sh`, and
`scripts/psion-google-operator-preflight-two-node-swarm.sh`. That surface
freezes two dedicated subnetworks, one swarm service account, one zone-pair
quota gate, and one operator preflight that rejects missing network, identity,
or pair-level headroom before the later launcher spends money.

The repo now also owns the first Google dual-node launch and runtime wiring for
that lane in
`fixtures/psion/google/psion_google_two_node_swarm_launch_profiles_v1.json`,
`scripts/psion-google-launch-two-node-swarm.sh`,
`scripts/psion-google-two-node-swarm-startup.sh`,
`scripts/psion-google-delete-two-node-swarm.sh`,
`crates/psionic-train/src/psion_google_two_node_swarm_runtime.rs`, and the
binary `psion_google_two_node_configured_peer_open_adapter_swarm`. That
surface freezes one repo-owned launch authority, one role-aware startup path,
one deterministic configured-peer cluster id, one exact coordinator versus
contributor node assignment, one explicit cluster-manifest plus launch-receipt
pair, and one bounded adapter-cluster runtime that uses the existing generic
worker-protocol, validation, and aggregation substrate instead of inventing a
second Google-only control plane. The live lane now also budgets ten minutes
for the contributor peer-connect loop so cold per-node compile skew on real
Google `g2` nodes does not produce a false cluster-port refusal before the
coordinator listener binds.

The repo now also owns explicit Google swarm impairment policy and host-side
transport shaping for that lane in
`fixtures/psion/google/psion_google_two_node_swarm_impairment_policy_v1.json`
and `scripts/psion-google-two-node-swarm-impair.sh`. That surface freezes the
admitted clean, mild-WAN, asymmetric-degraded, and temporary-partition drills,
keeps shaping scoped to the reserved cluster ports instead of the whole host,
and emits one machine-legible impairment receipt with the active profile,
affected ports, host identity, role-specific parameters, and observed `tc`
verification output.

The repo now also owns the first Google swarm bring-up, evidence, and finalizer
surfaces in `scripts/psion-google-two-node-swarm-startup.sh`,
`scripts/psion-google-finalize-two-node-swarm-run.sh`, and
`scripts/check-psion-google-two-node-swarm-evidence-bundle.sh`. That surface
emits one bring-up report per node before the runtime command begins, binds
those reports plus the runtime reports and optional impairment receipts into one
cluster-wide evidence bundle, uploads the evidence bundle and final manifest to
the dedicated training bucket, and keeps typed result classes explicit instead
of flattening every failure into generic launch or operator text.

The repo now also owns the dedicated operator runbook for that lane in
`docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md` plus the runbook checker
`scripts/check-psion-google-two-node-swarm-runbook.sh`. That surface freezes
the exact preflight, launch, monitoring, impairment, finalizer, checker, and
teardown commands and keeps the refusal boundary explicit that this lane is
still one bounded configured-peer adapter-delta Google rehearsal rather than a
broader cluster-training completion claim.

The repo now also owns one rehearsal-grade bottleneck report for that lane in
`crates/psionic-train/src/swarm_trusted_lan_rehearsal.rs`, the binary
`first_swarm_trusted_lan_rehearsal_report`, the checker
`scripts/check-first-swarm-trusted-lan-rehearsal.sh`, and the committed report
`fixtures/swarm/reports/first_swarm_trusted_lan_rehearsal_v1.json`. That
report measures the exact local operator-bundle and retained bring-up phases,
keeps contributor/upload/validator/aggregation timing explicitly simulated
where live receipts do not yet exist, produces a bottleneck map, and currently
ends with a `no_go` recommendation for a truthful live two-node attempt.

The repo now also owns one explicit live-attempt evidence bundle for that lane
in `crates/psionic-train/src/swarm_first_evidence_bundle.rs`, the binary
`first_swarm_trusted_lan_evidence_bundle`, the checker
`scripts/check-first-swarm-trusted-lan-evidence-bundle.sh`, and the committed
bundle `fixtures/swarm/reports/first_swarm_trusted_lan_evidence_bundle_v1.json`.
That bundle retains the exact contributor plan, launch digests, stage
summaries, and no-promotion truth for a refused first live attempt rather than
pretending execution, validator, aggregation, replay, or publication receipts
already exist.

The repo now also owns one deterministic closeout report for that lane in
`crates/psionic-train/src/swarm_first_closeout.rs`, the binary
`first_swarm_trusted_lan_closeout_report`, the checker
`scripts/check-first-swarm-trusted-lan-closeout.sh`, and the committed report
`fixtures/swarm/reports/first_swarm_trusted_lan_closeout_v1.json`. That report
keeps merge-or-no-merge truth explicit, keeps publish-or-refusal truth
explicit, binds the lane to the existing MLX local-snapshot publish surface,
and currently ends with `no_merge` plus `publish_refused` because the retained
first live attempt never earned accepted contributor, replay, aggregation, or
promotion receipts.

Apple-specific adapter work is no longer only later-family planning. The repo
now owns a canonical spec-and-fixture baseline for it in:

- `docs/APPLE_ADAPTER_DATASET_SPEC.md`
- `docs/APPLE_FMADAPTER_PACKAGE_SPEC.md`
- `docs/APPLE_ADAPTER_LINEAGE_SPEC.md`
- `fixtures/apple_adapter/`

and now also has:

- a repo-owned Apple training execution backend in `psionic-train`
- a first Rust-native Apple adapter SFT/export lane
- an optional Apple draft-model distillation lane
- an app-owned desktop-control and `autopilotctl` operator path that can
  launch, export, and accept one Apple training run into kernel authority

## Doc Authority

- `docs/TRAIN_SYSTEM.md` is the canonical training subsystem
  spec.
- `docs/ARCHITECTURE.md` is the canonical Psionic-wide system
  spec that defines the lower execution substrate this doc builds on.
- `docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md` is the canonical
  framework-core acceptance split; train acceptance must not be used as a
  substitute for framework-core parity claims.
- `docs/ARCHITECTURE_EXPLAINER_CLUSTER_BRINGUP_RUNBOOK.md` is
  the canonical operator guide for the first truthful multi-device clustered
  attempt around the `Psionic architecture explainer` path.
- `docs/REMOTE_TRAINING_VISUALIZATION.md` is the canonical app-facing remote
  training telemetry contract for Google Cloud and RunPod lanes and freezes
  that Psionic owns typed live bundle truth while Autopilot owns rendering and
  pane behavior.
- `docs/TRAIN_PROGRAM_MANIFEST_REFERENCE.md` is the canonical root
  cross-provider training-program manifest record and freezes the first
  provider-neutral training-program authority object before later compute-
  source, launch-binder, and hybrid-run issues widen the system.
- `docs/APPLE_ADAPTER_DATASET_SPEC.md`,
  `docs/APPLE_FMADAPTER_PACKAGE_SPEC.md`, and
  `docs/APPLE_ADAPTER_LINEAGE_SPEC.md` are the canonical
  Apple-adapter reference docs for dataset shape, package inventory, and
  lineage metadata.
- `docs/PSION_PROGRAM_MAP.md` is the canonical umbrella map for the `Psion`
  learned-model lane and freezes the dependency-ordered track split plus the
  learned-versus-executor claim boundary that all later `Psion` docs inherit.
- `docs/PSION_PLUGIN_PROGRAM_MAP.md` is the canonical convergence map for the next
  `Psion x Tassadar` plugin-conditioned training tranche and freezes the
  dependency-ordered training, benchmark, bounded `networked_read_only`
  substrate-proof, guest-artifact, and operator-proof split that later
  learned-plugin-use work must follow without widening publication or executor
  claims.
- `docs/PSION_PLUGIN_GUEST_ARTIFACT_DIRECTION.md` is the canonical product
  direction record for the bounded digest-bound guest-artifact starter-plugin
  lane and freezes that this class is now present-tense starter-plugin truth
  only in one trust-tiered, publication-blocked, operator-internal form rather
  than broad Wasm/plugin support.
- `docs/PSION_PLUGIN_GUEST_ARTIFACT_MANIFEST.md` is the canonical first guest-
  artifact manifest and identity contract for the bounded digest-bound lane and
  freezes the manifest fields, provenance fields, trust tier, publication
  posture, and fail-closed validation rules that runtime loading, shared
  catalog exposure, and receipt issues must reuse.
- `docs/PSION_PLUGIN_GUEST_ARTIFACT_RUNTIME_LOADING.md` is the canonical first
  bounded guest-artifact runtime-loading contract for the current digest-bound
  lane and
  freezes that load means manifest-bound byte admission, digest verification,
  minimal Wasm header checks, host-owned capability mediation, and typed
  load-time refusals rather than broad generic guest-plugin support.
- `docs/PSION_PLUGIN_GUEST_ARTIFACT_INVOCATION.md` is the canonical first
  bounded guest-artifact invocation contract for the current digest-bound lane
  and freezes one
  digest-bound Wasm packet invocation with host-native-equivalent receipt,
  replay, and typed-refusal evidence while still keeping generic guest-artifact
  breadth, publication, and arbitrary loading blocked.
- `docs/PSION_PLUGIN_CLAIM_BOUNDARY_AND_CAPABILITY_POSTURE.md` is the canonical
  claim-boundary and capability-posture contract for the convergence tranche
  and freezes which plugin classes are currently proved, not yet proved, later
  separate, or out of scope before any trained model capability matrix is
  honest.
- `docs/PSION_PLUGIN_TRAINING_RECORD_SCHEMA.md` is the canonical first
  plugin-conditioned training-record contract for the convergence tranche and
  freezes the admitted-plugin-set, controller-context, invocation-receipt,
  route-label, and outcome-label shape that later derivation, dataset, and
  benchmark work must reuse.
- `docs/PSION_PLUGIN_TRACE_DERIVATION.md` is the canonical first
  plugin-conditioned trace-normalization contract for the convergence tranche
  and freezes the runtime-drift-checked derivation path from the committed
  multi-plugin trace corpus into canonical training records.
- `docs/PSION_PLUGIN_CONDITIONED_DATASET.md` is the canonical first
  plugin-conditioned dataset-bundle contract for the convergence tranche and
  freezes the stable dataset identities, workflow-case-disjoint split rule, and
  preserved controller-surface plus plugin-class label contract for the first
  host-native reference build plus the first mixed host-native and
  guest-artifact dataset build.
- `docs/PSION_PLUGIN_CONTAMINATION_CONTROLS.md` is the canonical first
  plugin-aware contamination-control contract for the convergence tranche and
  freezes the parent-lineage rows, plugin-trace plus plugin-receipt exclusion
  manifest, and trace-disjoint train-vs-held-out review posture for the first
  host-native dataset build plus the first mixed dataset follow-on.
- `docs/PSION_PLUGIN_BENCHMARK_PACKAGES.md` is the canonical shared plugin
  benchmark contract for the convergence tranche and freezes the common
  item-schema, contamination-attachment, receipt-posture, task-contract, and
  grader-interface surface that later plugin benchmark families must reuse.
- `docs/PSION_PLUGIN_DISCOVERY_SELECTION_BENCHMARK.md` is the canonical first
  package-specific plugin benchmark doc for the convergence tranche and
  freezes the benchmark-authored discovery-versus-delegation package,
  wrong-tool-versus-unsupported-tool distinction, and shared receipt surface
  for the first host-native plugin-selection family.
- `docs/PSION_PLUGIN_ARGUMENT_CONSTRUCTION_BENCHMARK.md` is the canonical
  second package-specific plugin benchmark doc for the convergence tranche and
  freezes the packet-schema-aware argument package, missing-input versus
  malformed-structure distinction, and held-out typed-runtime-refusal evidence
  surface for the first host-native argument family.
- `docs/PSION_PLUGIN_SEQUENCING_BENCHMARK.md` is the canonical third
  package-specific plugin benchmark doc for the convergence tranche and
  freezes the serial-versus-parallel plan package, explicit continuation
  posture contract, and held-out refusal-stop evidence surface for the first
  host-native sequencing family.
- `docs/PSION_PLUGIN_REFUSAL_REQUEST_STRUCTURE_BENCHMARK.md` is the canonical
  fourth package-specific plugin benchmark doc for the convergence tranche and
  freezes the unsupported-capability refusal package, missing-input
  request-for-structure package, and separate overdelegation-negative scoring
  surface for the first host-native refusal family.
- `docs/PSION_PLUGIN_RESULT_INTERPRETATION_BENCHMARK.md` is the canonical
  fifth package-specific plugin benchmark doc for the convergence tranche and
  freezes the receipt-backed result-interpretation package, execution-backed
  versus inferred statement boundary, and refusal-continuation scoring surface
  for the first host-native interpretation family.
- `docs/PSION_PLUGIN_GUEST_PLUGIN_BENCHMARK.md` is the canonical sixth
  package-specific plugin benchmark doc for the convergence tranche and
  freezes the authored guest capability-boundary package, admitted guest use
  versus unsupported load/publication/arbitrary-binary/served-universality
  split, and the dedicated receipt metrics for the bounded guest lane.
- `docs/PSION_PLUGIN_CONDITIONED_SFT.md` is the canonical first
  plugin-conditioned `agentic_sft` stage doc for the convergence tranche and
  freezes the stage manifest, canonical dataset binding, benchmark-hook
  posture, replay-class coverage, stage receipt, and bounded output bundle for
  the first learned plugin-use stage contract.
- `docs/PSION_PLUGIN_CONDITIONED_COMPACT_DECODER_REFERENCE.md` is the
  canonical first plugin-conditioned compact-decoder config doc for the
  convergence tranche and freezes the lane-specific reference descriptor,
  context-budget assumptions, no-custom-plugin-token serialization posture,
  and checkpoint/export naming for the first learned plugin-use model config.
- `docs/PSION_PLUGIN_HOST_NATIVE_REFERENCE_LANE.md` is the canonical first
  bounded trained host-native plugin-conditioned lane doc for the convergence
  tranche and freezes the proved-class-only training boundary, the retained
  stage and learned-artifact receipts, and the benchmark-delta reporting
  posture against the named non-plugin baseline.
- `docs/PSION_PLUGIN_MIXED_REFERENCE_LANE.md` is the canonical first bounded
  mixed host-native plus guest-artifact plugin-conditioned lane doc for the
  convergence tranche and freezes the mixed dataset binding, retained
  guest-artifact training-example boundary, and explicit comparison posture
  against the committed host-native reference lane.
- `docs/PSION_PLUGIN_HOST_NATIVE_CAPABILITY_MATRIX_V1.md` is the canonical
  first publication doc for the host-native plugin-conditioned learned lane
  and freezes the served capability rows, explicit substrate-proved-but-outside
  the first learned publication versus unsupported versus blocked plugin-class
  split, and the executor-backed statement posture for outputs that cite the
  bounded host-native matrix.
- `docs/PSION_PLUGIN_MIXED_CAPABILITY_MATRIX_V2.md` is the canonical second
  publication doc for the mixed plugin-conditioned learned lane and freezes
  the supported host-native mixed rows, the one bounded guest admitted-use row,
  the still-blocked guest loading/publication/universality/arbitrary-software
  rows, and the learned-judgment-versus-executor-backed statement posture for
  outputs that cite the bounded mixed matrix.
- `docs/PSION_PLUGIN_ROUTE_REFUSAL_HARDENING.md` is the canonical hardening
  doc for the follow-on plugin-conditioned route/refusal tranche and freezes the
  committed regression rows, explicit zero-bps overdelegation budget, and
  no-implicit-execution failure cases that later operator or cluster decisions
  must cite instead of narrative-only confidence.
- `docs/PSION_PLUGIN_CLUSTER_SCALE_DECISION.md` is the canonical decision
  record for whether plugin-conditioned training should widen from the first
  single-node Google proofs to a trusted-cluster run and currently freezes the
  answer as `not_warranted_yet` even after the first generic and host-native
  accelerated single-node proofs, because a materially larger
  plugin-conditioned corpus and any mixed guest-artifact acceleration decision
  are still not closed even though the generic and host-native accelerated
  single-node lanes now retain machine-queryable bounded run-cost receipts.
- `docs/audits/2026-03-22-tassadar-full-plugin-system-state-audit.md` is the
  canonical current-state proof record for the bounded Tassadar plugin system
  and freezes the present authoring-class, publication, and guest-artifact
  boundaries that the Psion plugin convergence tranche must inherit rather than
  silently widen.
- `docs/PSION_CORPUS_ADMISSION.md` is the canonical first governance doc for
  the `Psion` learned-model lane and freezes the versioned source-admission
  contract that later ingestion, tokenizer, and training work must follow.
- `docs/PSION_SOURCE_LIFECYCLE.md` is the canonical first lifecycle and
  downstream-lineage doc for the `Psion` learned-model lane and freezes how
  source-state changes trigger artifact review, retraining review, and
  depublication review.
- `docs/PSION_BENCHMARK_ISOLATION.md` is the canonical first held-out and
  contamination-control doc for the `Psion` learned-model lane and freezes the
  exclusion-manifest, tokenizer-exposure, and benchmark-invalidation contract
  that later tokenizer and benchmark work must follow.
- `docs/PSION_RAW_SOURCE_INGESTION.md` is the canonical first raw-source import
  and normalization doc for the `Psion` learned-model lane and freezes the
  manifest, preprocessing-version, and boundary-preservation contract that
  later tokenizer and dataset stages must follow.
- `docs/PSION_TOKENIZER_TRAINING.md` is the canonical first tokenizer manifest
  and artifact-bundle doc for the `Psion` learned-model lane and freezes the
  admitted/excluded source lists, tokenizer digest, config, and tokenizer-only
  exposure reporting that later tokenized datasets and checkpoints must follow.
- `docs/PSION_TOKENIZED_CORPUS.md` is the canonical first tokenized dataset and
  replay-identity doc for the `Psion` learned-model lane and freezes the shard
  lineage, source-family binding, packing-policy version, and replay-safe
  dataset identity that later training stages must follow.
- `docs/PSION_SAMPLING_POLICY.md` is the canonical first sampling and mixture
  policy doc for the `Psion` learned-model lane and freezes the family
  weights, source caps, repetitive-region down-weighting, code-token ceiling,
  and regression-comparison contract that later pretraining runs must follow.
- `docs/PSION_COMPACT_DECODER.md` is the canonical first compact decoder-family
  doc for the `Psion` learned-model lane and freezes tokenizer binding,
  context-length configuration, checkpoint tensor naming, and export naming
  that later pretraining, evaluation, and serving work must follow.
- `docs/PSION_PRETRAIN_STAGE.md` is the canonical first pretrain-stage doc for
  the `Psion` learned-model lane and freezes the declared stage config,
  objective contract, source-family-aware reporting, replay receipt, and
  checkpoint-lineage receipt that later pilot and cluster runs must follow.
- `docs/PSION_RUN_OBSERVABILITY.md` is the canonical first run-observability
  doc for the `Psion` learned-model lane and freezes the cost, throughput,
  checkpoint-size, topology, and instability-marker receipts that later pilot
  and broader-pretraining decisions must cite directly.
- `docs/PSION_PILOT_PRETRAINING_RUN.md` is the canonical first pilot-run doc
  for the `Psion` learned-model lane and freezes the bounded pilot bundle,
  held-out-loss receipt, route/refusal probes, and acceptance-matrix promotion
  decision that must exist before broader pretraining is honest.
- `docs/PSION_REASONING_SFT.md` is the canonical first reasoning-SFT doc for
  the `Psion` learned-model lane and freezes the explicit assumption,
  uncertainty, normative-versus-inference, and style-plurality contract that
  bounded post-pretrain tuning must satisfy.
- `docs/PSION_BENCHMARK_PACKAGES.md` is the canonical first benchmark-package
  doc for the `Psion` learned-model lane and freezes the shared item schema,
  prompt-format, grader-interface, contamination-input, and receipt contract
  that later acceptance evidence must build on.
- `docs/PSION_BENCHMARK_LABEL_GENERATION.md` is the canonical first
  benchmark-label-generation doc for the `Psion` learned-model lane and
  freezes the exact-truth, rubric-version, label-generation-logic, and
  derived-data-lineage contract that later benchmark evidence and
  contamination review must preserve.
- `docs/PSION_ARCHITECTURE_REASONING_BENCHMARK.md` is the canonical first
  family-specific benchmark doc for the `Psion` learned-model lane and
  freezes the typed architecture item coverage, contamination attachment, and
  direct acceptance-matrix binding for the architecture-reasoning package.
- `docs/PSION_NORMATIVE_SPEC_READING_BENCHMARK.md` is the canonical first
  normative-reading benchmark doc for the `Psion` learned-model lane and
  freezes the typed normative item coverage, grounded-reading-versus-inference
  boundary, contamination attachment, and direct acceptance-matrix binding for
  the normative spec-reading package.
- `docs/PSION_ENGINEERING_SPEC_INTERPRETATION_BENCHMARK.md` is the canonical
  first engineering-interpretation benchmark doc for the `Psion` learned-model
  lane and freezes the typed implementation-implication, ambiguity, unspecified
  region, and portability-risk coverage plus the direct acceptance-matrix
  binding for the engineering spec package.
- `docs/PSION_MEMORIZATION_VS_REASONING_PROBES.md` is the canonical first
  memorization-versus-reasoning probe doc for the `Psion` learned-model lane
  and freezes the altered-constraint, unfamiliar-synthesis, historical
  transfer, paraphrase, and spec-adjacent-edge probe coverage plus the direct
  acceptance-matrix binding for the recombination package.
- `docs/PSION_ROUTE_CLASS_EVALUATION.md` is the canonical first route-class
  benchmark doc for the `Psion` learned-model lane and freezes the four route
  classes, typed route receipt, delegation-error accounting, and direct
  acceptance-matrix binding for the route benchmark package.
- `docs/PSION_REFUSAL_CALIBRATION.md` is the canonical first refusal-
  calibration doc for the `Psion` learned-model lane and freezes the
  unsupported-envelope refusal package, capability-matrix-bound refusal
  receipt, reason-code accounting, and direct acceptance binding for the
  unsupported-request refusal benchmark package.
- `docs/PSION_SERVED_EVIDENCE.md` is the canonical first served-evidence and
  provenance doc for the `Psion` learned-model lane and freezes the shared
  learned-judgment, source-grounded, executor-backed, and benchmark-backed
  schema plus the route/refusal and no-implicit-execution binding that later
  serving and provider work must reuse.
- `docs/PSION_SERVED_OUTPUT_CLAIMS.md` is the canonical first served-output
  claim-discipline doc for the `Psion` learned-model lane and freezes the
  explicit assumptions, visible route-or-refusal behavior, surfaced claim
  flags, and context/latency envelope binding that later serving work must
  preserve.
- `docs/PSION_CHECKPOINT_RECOVERY.md` is the canonical first dense-versus-
  sharded checkpoint-recovery doc for the `Psion` learned-model lane and
  freezes the explicit restart, rollback, corruption-detection, and
  invalidation bundle that later rented-cluster and trusted-cluster work must
  preserve.
- `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md` is the canonical first
  multi-plugin controller trace corpus and training-bootstrap doc for the
  Tassadar plugin-control lane and freezes the lane-neutral trace-record,
  parity-matrix, disagreement-retention, and receipt-identity contract that
  later weighted-controller work must follow.
- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md` is the canonical first rented-cluster
  runbook and failure-policy doc for the `Psion` learned-model lane and
  freezes the storage persistence, preemption downgrade, cost stop-condition,
  and infra-mode refusal contract that later trusted-cluster work must not
  silently widen.
- `docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md` is the canonical first Google
  single-region single-node operator runbook for the `Psion` learned-model
  lane and freezes the local preflight, launch, host evidence, checkpoint
  archive, cold-restore, and teardown procedure for the bounded Google pilot
  without widening the claim boundary to trusted-cluster or broader-pretrain
  posture, and explicitly freezes that the current CPU-reference pilot and
  plugin reference lanes are not valid accelerator-backed training proof
  targets; the canonical bounded accelerator-backed trainer now lives at
  `crates/psionic-train/examples/psion_accelerated_reference_pilot.rs`, and a
  purported Google accelerator-backed success now requires explicit `cuda`
  backend truth plus non-zero post-warmup GPU utilization and memory-residency
  evidence retained in the committed accelerator-validation receipt.
- `docs/audits/2026-03-23-openagentsgemini-first-google-accelerator-backed-single-node-psion-training-audit.md`
  is the canonical proof record for the first truthful accelerator-backed
  Google single-node `Psion` run and freezes the actual trainer path, backend
  truth, GPU-sample truth, throughput truth, checkpoint truth, and post-run VM
  deletion boundary for the bounded single-node accelerator lane.
- `docs/audits/2026-03-23-openagentsgemini-first-google-accelerator-backed-host-native-plugin-conditioned-run-audit.md`
  is the canonical proof record for the first truthful accelerator-backed
  Google host-native plugin-conditioned `Psion` run and freezes the actual
  plugin-conditioned trainer path, backend truth, GPU-sample truth, proved
  authoring-class boundary, benchmark provenance shift away from the old
  metadata-only reference artifact, and post-run VM deletion boundary for the
  bounded accelerated host-native plugin lane.
- `docs/audits/2026-03-23-openagentsgemini-query-backed-google-single-node-cost-receipt-audit.md`
  is the canonical follow-up proof record for the Google single-node
  machine-queryable run-cost receipt path and freezes the IAM root cause of the
  first failed pricing lookups, the identity and preflight hardening needed to
  close that gap, and the first retained priced follow-up runs for the generic
  and host-native accelerated lanes.
- `docs/audits/2026-03-22-openagentsgemini-first-google-single-gpu-pilot-run-audit.md`
  is the canonical proof record for the first bounded Google single-node
  `Psion` pilot and freezes the typed live-attempt history, retained evidence,
  and current no-go boundary against claiming effective GPU-backed broader
  pretraining from that run alone.
- `docs/PSION_TRUSTED_CLUSTER_RUN.md` is the canonical first trusted-cluster
  multi-host run doc for the `Psion` learned-model lane and freezes the
  bounded topology contract, distributed-group receipt, replay receipt, and
  checkpoint-restart coverage that later decentralized contribution work must
  build on instead of bypassing.
- `docs/PSION_ACCEPTANCE_MATRIX.md` is the canonical first phase-gate and
  promotion-decision doc for the `Psion` learned-model lane and freezes the
  acceptance-matrix plus evidence-bound promotion contract that later pilot,
  pretraining, SFT, serving, and cluster issues must satisfy.
- `docs/PSION_CAPABILITY_MATRIX.md` is the canonical first served capability
  and publication doc for the `Psion` learned-model lane and freezes the
  explicit supported, route-required, refusal-required, and unsupported
  publication surface that later serving issues must satisfy.
- `docs/PSION_CAPABILITY_WITHDRAWAL.md` is the canonical first rollback and
  downgrade-history doc for the `Psion` learned-model lane and freezes how
  rights, contamination, replay, route, and refusal regressions withdraw or
  narrow checkpoints, capability matrices, and served claim surfaces.
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md` is the canonical first bounded
  decentralized-contribution doc for the `Psion` learned-model lane and
  freezes how adapter-delta contribution windows inherit the same trusted-
  cluster, reasoning-SFT, acceptance, capability, and rollback discipline as
  the main lane.
- `docs/audits/2026-03-13-intellect-lessons-for-psionic-train-audit.md` is
  research rationale, not the canonical current-state spec.
- `docs/audits/2026-03-14-covenant-code-lessons-for-psionic-train-audit.md`
  is a code-grounded adaptation audit for windowed training, checkpoint
  protocol discipline, validator-owned benchmark truth, and bounded research
  loops.

## Status Vocabulary

This doc uses the canonical status vocabulary defined in `ARCHITECTURE.md`:
`implemented`, `implemented_early`, `partial`, `partial_outside_psionic`, and
`planned`.

## Short Definition

The Psionic train system is not one crate.

It is the Rust-native training-class execution stack inside this standalone
`psionic` workspace that should eventually own:

- training-session truth
- elastic membership and recovery
- collective planning
- checkpoint and weight movement
- environment-bound training and eval execution
- rollout ingestion and validation
- trainer and orchestrator control flow
- operator-inspectable receipts for the whole system

Today Psionic implements the lower half of that stack plus a first real
trainer-step core.

It already has real substrate for:

- reusable module, parameter, buffer, explicit trainable-versus-frozen
  posture, and deterministic state-tree semantics in `psionic-nn`
- deterministic `state_dict` naming plus bounded public `save_weights` /
  `load_weights` behavior with strict/non-strict keyed load posture and
  explicit size-mismatch refusal in `psionic-nn`
- a bounded reusable CPU-reference core layer surface in `psionic-nn`,
  covering linear, embedding, norms, activations, dropout, conv, and pooling
  families above the same module/state substrate
- a bounded reusable CPU-reference loss, initializer, and helper surface in
  `psionic-nn`, covering `mse_loss`, `l1_loss`, `binary_cross_entropy_loss`,
  `cross_entropy_loss`, `softmax_last_dim`, `log_softmax_last_dim`,
  `sigmoid`, `one_hot`, `init_tensor`, and `init_parameter`
- a bounded reusable public optimizer shell in `psionic-nn` that reuses
  `psionic-train` optimizer math while keeping module-path keyed state,
  explicit frozen-parameter handling, state snapshot restore, and per-step
  receipts in the framework-facing layer
- a bounded reusable public distributed-group, collective-helper, and
  launch/config shell in `psionic-distributed` that reuses current runtime
  mesh, cluster, and sandbox truth while keeping explicit mesh bootstrap,
  reusable global-group initialization, honest singleton fallback, rank/size
  identity, ordered member snapshots, explicit-plan subgroup split semantics,
  hostfile parsing, honest single-rank-per-node launch validation, per-rank
  bootstrap payloads and sandbox job plans, distributed reserved-environment
  synthesis, cluster execution evidence, and explicit `ring` / `mpi` / `nccl`
  backend-family capability mapping plus typed `jaccl` refusal over current
  topology profiles in the framework-facing layer
- bounded reusable tree-aware gradient reduction helpers in
  `psionic-distributed` that reuse the public collective layer while keeping
  deterministic tree structure, grouped small-leaf all-reduce, and
  floating-point `average_gradients` above the current reference-emulated
  multi-rank surface
- bounded reusable tensor-parallel linear wrappers in
  `psionic-distributed` that reuse bounded `psionic-nn::Linear` plus the
  public collective layer while keeping deterministic row/column sharding,
  inspectable shard layouts, local shard-input splitting, and explicit
  reference-emulated `ShardedToAllLinear` reconstruction above the current
  non-transport-backed public surface
- a bounded reusable public scheduler and parameter-group shell in
  `psionic-nn` that reuses `psionic-train` scheduler primitives while keeping
  scheduler bindings, group-level learning-rate and weight-decay scaling, and
  multi-optimizer composition in the framework-facing layer
- a bounded reusable eval-oriented quantized module shell in `psionic-nn`
  covering `Module::quantize(...)`, explicit keep-dense versus strict
  quantize reports, and `QuantizedLinear` plus `QuantizedEmbedding` wrappers
  over `int8_symmetric` block storage with explicit dequantize-to-`f32`
  forward semantics
- a seeded PyTorch-derived module parity matrix for normalized module-tree and
  `state_dict` semantics in `psionic-nn`, with an explicit refusal proof for
  registration-order-preserving `state_dict` parity
- a seeded PyTorch-derived optimizer parity matrix for SGD, Adam, AdamW,
  LARS, and LAMB single-step behavior in `psionic-train`, with an explicit
  refusal proof for state-kind mismatch
- training recovery posture
- checkpoint lineage
- elastic membership truth
- device-mesh and collective planning
- resumable dataset and checkpoint transport
- typed fixed-budget trainer steps
- per-group optimizer state, scaling semantics, scheduler bindings, and
  residency policy
- reusable optimizer contracts plus typed SGD, Adam, AdamW, LARS, and LAMB
  state/update semantics with explicit scheduler-driven learning-rate
  resolution
- reverse-mode autodiff, explicit detach, and training/no-grad gradient
  semantics over canonical IR primitives
- machine-legible step telemetry and checkpoint-anchored restore lineage
- checkpoint-aware policy revisions
- proof-bearing rollout artifacts and trainer-batch assembly
- versioned dataset manifests, tokenizer digests, split declarations, and
  long-context packing contracts
- environment package ABI and deterministic runtime sessions
- held-out eval runtime, benchmark packages, repeat-run aggregation, and local
  validator simulation
- bounded `AttnRes` tiny next-token training over a repo-owned fixture corpus,
  with stable parameter-group identity, fixed-budget step receipts, held-out
  loss plus routing-delta evaluation, and Psionic-native safetensors checkpoint
  lineage over the trained routing and LM-head parameter subset
- bounded `Tassadar` small-executor training over the validation benchmark
  package, using the fixed-budget training core plus proof-aware exactness
  comparison against the handcrafted reference lane
- bounded article-Transformer toy-task training over the canonical owned stack,
  with label-smoothed cross-entropy, Adam plus inverse-square-root warmup,
  finite-gradient checks, and deterministic checkpoint restore rooted in
  `psionic-transformer`
- bounded article-Transformer trained-weight production over a canonical
  article-class trace-prefix slice, with one explicit trained trace-bound
  safetensors artifact plus checkpoint and artifact-reload parity rooted in
  `psionic-transformer`
- frozen lineage contracts for trained article-Transformer artifacts, binding
  exact workload set, training-config snapshot, source inventory, checkpoint
  lineage, and committed artifact digests into one challengeable manifest
- adapter lineage

It does not yet implement the full distributed trainer-orchestrator-RL runtime.

## What Psionic Train Is Not

- It is not a promise that full general model training already works in the
  repo today beyond the current narrow Apple adapter and AttnRes reference
  lanes.
- It is not a Python trainer hidden behind Rust wrappers.
- It is not an app-owned workflow inside `apps/*`.
- It is not just "cluster execution, but for training."
- It is not just checkpoint files and background notes.

The honest description today is:

> Psionic already owns real training-class truth surfaces plus a bounded
> training-core reference loop, but it does not yet own the full distributed
> train system.

That now includes two AttnRes reference-training answers with distinct scope:

- `psionic-train::train_attnres_tiny_next_token(...)` can train the bounded
  CPU-reference AttnRes family over a repo-owned tiny next-token corpus without
  Burn, using stable named parameter groups for the routing pseudo-query and
  LM-head tensors rather than positional optimizer state
- `psionic-train::attnres_local_reference_training_config(...)`,
  `attnres_local_reference_training_corpus(...)`, and
  `train_attnres_local_reference_next_token(...)` now expose the full local
  reference run contract for the interactive AttnRes desktop lab: the same
  repo-owned corpus and training core, but a `320`-step local reference budget,
  non-`tiny` public naming, and logical run timing that matches the Burn demo's
  intended full-run bar rather than the earlier bounded smoke lane
- `psionic-train::AttnResTinyTrainingCorpus::reference()` now exposes that
  canonical tiny next-token corpus directly to repo-local consumers instead of
  forcing app code to duplicate sample construction
- `psionic-train::AttnResTinyTrainingRunner` plus
  `AttnResTinyTrainingUpdate` now expose the same bounded lane as a
  renderer-neutral stepwise contract, carrying per-step receipts, checkpoint
  refs, current diagnostics, and current loss/routing metrics so desktop or
  operator surfaces can pace the run without recreating AttnRes training math;
  the existing whole-run helper remains available as the convenience layer on
  top of that runner
- that same stepwise contract now also carries logical step duration, elapsed
  duration, and remaining duration, and the runner exposes the current model,
  baseline model, bound corpus/config, and accumulated step metrics so app or
  operator surfaces can render a full local reference run without replaying the
  entire training loop on every frame
- checkpoint export is Psionic-native and explicit: the lane persists
  safetensors parameter payloads plus a JSON manifest carrying config, base
  descriptor digest, dataset digests, checkpoint refs, and parent-checkpoint
  lineage instead of adopting Burn `.mpk` as a native storage contract
- restore is equally explicit: the lane reseeds the canonical Psionic AttnRes
  model family and reapplies the persisted named parameter subset, so
  checkpoint identity stays tied to the same descriptor/config surface the
  runtime and eval lanes already consume
- `psionic-research` now also owns an optional feature-gated Burn migration
  boundary for legacy `attnres` `.mpk` and `.bin` artifacts: it loads the old
  Burn model only inside that research/import surface, maps legacy tensor paths
  into canonical Psionic parameter ids once, and emits Psionic-native
  `safetensors` plus a machine-readable import manifest; path remaps and
  partial-load posture live only at that import boundary rather than leaking
  Burn semantics into the runtime or training contract
- `psionic-eval::evaluate_attnres_training_shift(...)` now gives the same lane
  a held-out machine-readable loss and routing-delta report, making it visible
  whether training changed the routing story in addition to the token loss
- the optional follow-on surfaces now stay on that same shared truth rather
  than growing a parallel AttnRes engine: `psionic-research` can run a bounded
  residual-vs-AttnRes comparison bundle over the committed tiny corpus and
  benchmark cases, while `psionic-serve` can expose the trained or seeded
  reference family through a local text-generation contract that emits the same
  routing diagnostics snapshots per decode step
- the claim remains intentionally narrow: this is a tiny CPU-reference
  next-token lane for the first AttnRes consumer, not a claim that generic
  AttnRes training, backend acceleration, or product-facing training UX is done

That now includes one intentionally narrow executor-training answer:

- `psionic-train::train_tassadar_small_executor(...)` can train a bounded small
  Tassadar model over package-backed validation-corpus supervision
- the learned lane uses the same fixed-budget training receipts as the rest of
  the train substrate rather than a sidecar research script
- evaluation remains proof-aware and baseline-aware: trained traces, outputs,
  and halt posture are checked against the handcrafted reference lane and keep
  the reference proof-bundle digests visible in the resulting report
- the resulting claim is intentionally scoped to the validation corpus only; it
  is not a claim that larger learned executors, broader Wasm coverage, or
  compile-to-weights work are already complete in Psionic
- the canonical coarse Tassadar claim vocabulary is now
  `compiled_exact`, `compiled_article_class`, `learned_bounded`,
  `learned_article_class`, and `research_only`; train-side learned bundles
  generated from the current code now persist `claim_class=learned_bounded`,
  while `boundary_label`,
  `claim_boundary`, and `serve_posture` keep the narrower learned-vs-compiled
  and serving limits explicit
- the runtime side now does carry the first honest broader-executor substrate
  for later training work: `tassadar.wasm.sudoku_v0_search.v1` can represent a
  real 4x4 backtracking Sudoku program on the CPU reference lane, but that is
  still substrate for later corpus/model/training issues rather than a claim
  that the trained executor already exists
- the benchmark side now also carries a real split-aware 4x4 Sudoku-v0 corpus
  with stable train/validation/test assignments and exact CPU-reference traces
  per puzzle, which replaces the earlier placeholder `SudokuClass` proxy and
  gives later tokenization/training issues an honest package-backed source
  corpus
- the tokenized-data side now also exists for that same corpus: Psionic can
  freeze deterministic program-plus-trace token sequences with explicit
  tokenizer/vocabulary digest lineage, split-stable dataset manifests, and
  generic packing plans for train/validation/test instead of leaving later
  training work to regenerate traces ad hoc
- the model side now also has a first honest train target above that corpus:
  `psionic-models` carries a real neural executor transformer family with
  explicit next-token logits, linear decode state, and 2D lookup-head geometry
  claims, while still keeping the claim boundary truthful that this is not yet
  the exact handcrafted executor path
- the train/eval loop now also exists for that model family: `psionic-train`
  can run teacher-forced next-token optimization over the frozen sequence
  manifest, and `psionic-eval` can score the trained model with exact-trace,
  final-output, and halt correctness against the same CPU-reference sequences
  that define the corpus
- the benchmark side now also includes the trained-model comparison the audit
  asked for: neural linear decode is measured directly against CPU reference
  execution with explicit decode-mode and KV-cache identity so the remaining
  performance and exactness gap is visible instead of being hidden behind the
  handcrafted runtime lanes
- the first persisted trained-run surface now also exists for that same lane:
  `psionic-train` can execute a canonical Sudoku-v0 reference run and persist
  the frozen training manifest, training report, linear benchmark report,
  checkpoint payload plus checkpoint manifest, and trained-model artifact under
  `fixtures/tassadar/runs/sudoku_v0_reference_run_v0`; the

That now also includes one intentionally narrow article-Transformer training
closure:

- `psionic-train::train_tassadar_article_transformer_toy_suite(...)` can train
  the canonical paper-faithful article wrapper over two toy selector tasks
  without bypassing `psionic-transformer`
- the recipe is explicit and machine-legible: label-smoothed cross-entropy,
  Adam, inverse-square-root warmup, finite-difference gradients, fixed-budget
  step receipts, and deterministic checkpoint restore all land in committed
  evidence
- the resulting committed artifact is intentionally narrow:
  `fixtures/tassadar/runs/tassadar_article_transformer_training_v1/article_transformer_training_evidence_bundle.json`
  proves the owned stack is trainable and restorable on bounded tasks, not
  that full article-model training, benchmark parity, or final
  article-equivalence closure are done
- that same evidence bundle now also preserves the runtime-visible checkpoint
  reference fields needed by the later forward-pass receipt lane:
  stream id, object digest, writer identity, cluster and topology digests,
  logical timing, step, and parent-checkpoint lineage are all committed so
  later runtime receipts can bind to real checkpoint truth rather than
  placeholder metadata
- the eval and research surfaces keep the boundary explicit:
  `psionic-eval` closes `TAS-164` through a dedicated training-closure report,
  while `psionic-research` mirrors that result in a summary without widening
  the public claim boundary
  current committed run is intentionally recorded as low exactness
  (`validation_exact_trace_case_count = 0/2`, aggregate target exactness
  `15` bps), which makes it useful as a real learning baseline rather than as
  benchmark theater
- `TAS-169` now adds the first real trained trace-bound article artifact on
  top of that same owned stack:
  `psionic-train::run_tassadar_article_transformer_weight_production(...)`
  now distills the committed trace-bound article wrapper over one explicit
  `32`-token Hungarian article-demo trace window while scoring the kernel
  family as held-out evidence, and the lane now persists the resulting bundle
  at
  `fixtures/tassadar/runs/tassadar_article_transformer_weight_production_v1/article_transformer_weight_production_bundle.json`
  plus the trained descriptor and safetensors artifact at
  `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0_descriptor.json`
  and
  `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0.safetensors`
  with explicit checkpoint-restore and artifact-reload parity
- the train-side learning loop artifacts now also exist for that same run:
  `psionic-train` can augment the persisted bundle with
  `training_telemetry.json`, `exactness_curve.json`,
  `trace_divergence_report.json`, and `failure_samples.json`, all bound to the
  same dataset/model/checkpoint identity; those artifacts currently show every
  decoded case diverging at target token 0, which is exactly the sort of
  machine-readable failure baseline the next curriculum/model iteration needs
- the post-run review loop now also exists for that same run:
  `psionic-train` can emit `postmortem.json` and `next_run_plan.json`, and the
  repo keeps the resulting plan bound to the same persisted run identity;
  `docs/audits/2026-03-16-tassadar-first-run-postmortem.md` is the
  human-readable companion review, and the current plan keeps the next run
  disciplined by prioritizing boundary curriculum and larger optimization
  budget first, and by explicitly gating 9x9 scale claims on better 4x4
  exactness evidence
- the neural fast-path benchmark loop now also exists for that same run:
  `psionic-models` now exposes explicit model-KV decode state plus
  machine-legible decode selection, `psionic-eval` can compare the trained
  model’s explicit linear-scan KV path against a real hull-cache KV path and
  direct CPU execution, and `psionic-train` can persist
  `neural_hull_benchmark_report.json` into the committed run bundle; the
  current committed Sudoku-v0 run shows `8/8` hull-vs-linear prefix agreement
  with no fallbacks or refusals and about `1.93x` hull speedup (`42,172` vs
  `21,860` target tok/s over a `4,096`-token per-case window), while exactness
  remains `0/8`, which is the right claim boundary for the lane today
- `TAS-169A` now hardens the same run into a challengeable provenance
  contract:
  `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0_lineage_contract.json`
  now freezes the exact workload set, training-config snapshot, source
  inventory, checkpoint lineage, descriptor digests, and committed artifact
  digests around the first trained trace-bound article weights, while
  `psionic-eval` and `psionic-research` mirror that state in
  `fixtures/tassadar/reports/tassadar_article_transformer_weight_lineage_report.json`
  and
  `fixtures/tassadar/reports/tassadar_article_transformer_weight_lineage_summary.json`
  without pretending that provenance closure itself is already reference-linear
  exactness, fast-route promotion, benchmark parity, or final
  article-equivalence green status
- `TAS-R1` now adds a research-only minimal-size frontier on top of that same
  bounded lane: `psionic-serve` materializes six reduced article-Transformer
  candidates under
  `fixtures/tassadar/runs/tassadar_article_transformer_minimal_frontier_v1/`,
  keeps candidate-specific base and trained descriptors, safetensors,
  weight-production bundles, and lineage manifests explicit, and freezes the
  aggregate comparison at
  `fixtures/tassadar/reports/tassadar_article_transformer_minimal_frontier_report.json`;
  that frontier reads the canonical acceptance-gate, generalization,
  fast-route-selection, and throughput-floor artifacts as anchors instead of
  mutating them, and the current report lands `frontier_green=false`, so the
  canonical `TAS-169A` lineage contract and final article-equivalence claim
  remain unchanged
- the Phase 11 scale-out substrate now also exists above that run:
  `psionic-runtime` owns a real `tassadar.wasm.sudoku_9x9_search.v1` profile
  plus a real split-aware 9x9 Sudoku-class corpus, `psionic-eval` and
  `psionic-train` can freeze that workload into a tokenized sequence dataset
  plus training manifest, `psionic-models` carries the matching 9x9
  executor-transformer descriptor, and `psionic-train` commits a machine-
  readable `scale_plan.json` fixture under
  `fixtures/tassadar/runs/sudoku_9x9_scale_plan_v0`; the same
  plan keeps the promotion gate explicit, so Phase 11 now means “real 9x9
  workload and curriculum path exist” rather than “the 9x9 trained executor is
  already good”
- the Phase 12 boundary-truth run now also exists above that baseline:
  `psionic-eval` emits first-target / first-8 / first-32 exactness plus
  first-divergence and confusion reports, `psionic-train` now supports an
  explicit boundary curriculum with per-epoch validation and boundary-ranked
  checkpoint selection, and the resulting follow-on run bundle at
  `fixtures/tassadar/runs/sudoku_v0_boundary_v1` records that
  the selected checkpoint clears the token-0 boundary (`10000` bps
  first-target exactness, no token-0 confusions, divergence bucket moved to
  target index `1`) while still failing the later gates (`5000` bps first-32
  exactness, `0/2` exact traces); `docs/audits/2026-03-16-tassadar-phase-12-boundary-audit.md`
  is the human-readable companion note for that run
- the Phase 13 trainable-surface ablation now also exists above that baseline:
  the lookup-style executor family now records a stable trainable surface in
  descriptors, manifests, checkpoints, and run bundles; `psionic-train` can
  update the output head alone, the output head plus token embeddings, the
  output head plus token and position embeddings, or those plus one small
  learned residual mixer; and `psionic-research` now materializes a
  same-corpus ablation root at
  `fixtures/tassadar/runs/sudoku_v0_trainable_surface_ablation_v1`
  with a machine-readable `trainable_surface_ablation.json`; that report keeps
  `output_head_only` as the preserved baseline and recommends only
  `output_head_embeddings_and_small_learned_mixer`, which improves the
  selected checkpoint to `3750` bps first-8 exactness and `5625` bps first-32
  exactness while still leaving `0/2` exact validation traces and the first
  divergence bucket at target index `1`; the companion human-readable audit is
  `docs/audits/2026-03-16-tassadar-phase-13-trainable-surface-audit.md`
- the preserved red Phase 14 promotion-truth run now also exists above that baseline:
  `psionic-train` can execute the canonical promotion config, stream live
  stage/epoch/batch/validation/checkpoint progress while it runs, and persist
  `best_checkpoint_manifest.json` plus `promotion_gate_report.json` under
  `fixtures/tassadar/runs/sudoku_v0_promotion_v1`; the repo also carries a
  standalone `scripts/check-tassadar-4x4-promotion-gate.sh` checker that
  revalidates persisted gate reports; that selected
  checkpoint stayed at `epoch_0006` from `prompt_to_first_16_tokens` with
  `10000` bps first-target exactness, `7500` bps first-8 exactness,
  `6875` bps first-32 exactness, and `0/2` exact validation traces, so that
  bundle remains preserved blocker evidence and the companion human-readable audit is
  `docs/audits/2026-03-16-tassadar-phase-14-blocker-audit.md`
- the Phase 14 teacher-forced continuation now also exists beside that
  baseline: `psionic-train` can execute the separate preserved config under
  `fixtures/tassadar/runs/sudoku_v0_promotion_v2`, keeping the
  same lookup-family surface and Phase 14 gate while removing greedy-rollout
  refinement and extending teacher-forced 16-/32-token supervision; the
  resulting selected checkpoint `epoch_0008` reproduces but does not beat the
  prior best (`10000` bps first-target, `7500` bps first-8, `6875` bps
  first-32, `0/2` exact traces), and later 32-token epochs still regress, so
  that bundle closes the “maybe this was just a schedule problem” question
  without pretending the learned 4x4 gate was any closer to green at the time; the
  companion audit is
  `docs/audits/2026-03-16-tassadar-promotion-v2-teacher-forced-audit.md`
- the learned 4x4 promotion gate is now green in
  `fixtures/tassadar/runs/sudoku_v0_promotion_v3`: `psionic-research` can now
  replay the bootstrap-plus-promotion attention continuation via
  `crates/psionic-research/examples/tassadar_executor_attention_promotion_run.rs`,
  persist `best_checkpoint_manifest.json`, `exactness_curve.json`,
  `failure_samples.json`, `exact_trace_samples.json`, and
  `promotion_gate_report.json` at the final run root, and keep the bootstrap
  checkpoint it used under `bootstrap_pc_boundary`; the selected checkpoint is
  `epoch_0015` from `prompt_to_first_32_tokens` with `10000` bps first-target
  exactness, `10000` bps first-8 exactness, `10000` bps first-32 exactness,
  and `2/2` exact validation traces, so the learned 4x4 lane is now
  promotable and the companion audit is
  `docs/audits/2026-03-16-tassadar-phase-14-promotion-green-audit.md`
- the Phase 15 executor-attention comparison now also exists beside that
  baseline: `psionic-models` now carries a distinct bounded
  `TassadarExecutorAttentionTransformer` family with layered causal hard-max
  attention, fixed 2D head geometry, explicit per-layer semantics, and hull
  fallback to reference-linear decode; `psionic-eval` and `psionic-research`
  now materialize a same-corpus comparison root at
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v1`
  with `architecture_comparison_report.json` plus per-family run bundles
  against the preserved Phase 13 lookup baseline; the committed report keeps
  the claim boundary honest by showing the new family is closer to the article
  structurally but still much worse on the bounded validation window (`0` bps
  first-target and first-32 exactness, `1333` target tok/s, hull fallback)
  than the lookup baseline (`10000` / `6563` bps, `32000` target tok/s, direct
  hull decode), so this phase is a research-family landing rather than a
  promotion or parity result
- the Phase 16 first honest 9x9 run now also exists beside that baseline:
  `psionic-train` now owns the canonical
  `crates/psionic-train/examples/tassadar_sudoku_9x9_reference_run.rs`
  replay path plus the committed bundle
  `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0`; the learned lane
  now records an explicit `incremental_decode_window` teacher-forced strategy
  and `incremental_decode_window` long-trace family contract in the training
  manifest, persists `sequence_fit_report.json`, `postmortem.json`,
  `next_run_plan.json`, `later_window_exactness_report.json`,
  `suffix_window_failure_report.json`, `best_checkpoint_manifest.json`,
  `promotion_bundle.json`, and `promotion_gate_report.json`, while the
  repo-owned `scripts/check-tassadar-9x9-promotion-gate.sh` checker
  revalidates the stored report as consistent; the selected checkpoint remains
  `epoch_0004` from `full_trace_supervision`, full 9x9 traces still do not fit
  the current `524288`-token model context (`4891222` to `5335309` total
  tokens, overflow `4366934` to `4811021`), and the new gate makes the
  remaining learned failure shape machine-readable: the early `512`-token
  prefix reaches `10000` bps first-target exactness but only `5938` bps
  first-32 exactness, the fixed later window at target token `262144` and the
  furthest fittable suffix window at target token `472240` both improve to
  `8438` bps first-32 exactness, all three gate windows stay `0/1` exact
  windows, and full-trace exactness across the declared gate windows remains
  `0`, so the correct audit statement remains “bounded and partial, not
  article-class” even though later-window truth is now explicit; the companion
  note is
  `docs/audits/2026-03-16-tassadar-phase-16-9x9-reference-run-audit.md`
- the first same-corpus flat-prefix-vs-windowed 9x9 comparison now also
  exists in `psionic-train` under
  `fixtures/tassadar/runs/sudoku_9x9_v0_windowed_family_comparison_v1`;
  those artifacts keep the learned claim bounded while making the long-trace
  family split machine-readable: the flat-prefix family and the windowed
  family both stay at `5938` bps first-32 and `0/1` exact validation traces
  over the first `512` target tokens, but the explicit contract live-bytes bar
  drops from `109715076` to `1459452`, which is the honest reason to keep the
  windowed family around even though it is not yet a green 9x9 learned lane
- the first same-corpus sequential-vs-wavefront target-family comparison now
  also exists under
  `fixtures/tassadar/runs/tassadar_trace_family_comparison_v1`; those
  artifacts freeze dataset manifests and training manifests for the sequential
  CPU trace plus alternate research-only families on shared Sudoku and
  Hungarian corpora, and they keep the claim boundary explicit by proving only
  final-output exactness for the alternates: 9x9 Sudoku drops from `5335309`
  max total tokens on the sequential trace to `52969` on the anti-diagonal
  wavefront family, while article-sized 10x10 Hungarian drops from
  `11532454` to `22050` on the parallel assignment frontier, and all of those
  alternate families remain `research_only` even though they preserve
  `10000` bps final-output exactness; the same lane now also carries a public
  comparable trace-family-set contract in `psionic-data`, a reproducible
  committed-truth check in `psionic-train`, and a repo-facing summary report
  at `fixtures/tassadar/reports/tassadar_trace_family_variant_report.json`
- the first public no-hint / self-supervised executor regime comparison now
  also exists beside that same bounded lane:
  `psionic-train` now materializes four seeded supervision regimes
  (`full_hint_trace`, `subroutine_hints`, `no_hint_output_only`, and
  `no_hint_self_supervised`) plus deterministic reusable-signal proxies for
  held-out sort / CLRS-shortest-path / sudoku-style workloads, and
  `psionic-research` now freezes the resulting architecture report at
  `fixtures/tassadar/reports/tassadar_no_hint_self_supervised_report.json`;
  on held-out CLRS shortest-path, reusable signal moves from `1666` bps on
  full-hint traces to `5000` on output-only no-hint and `8000` on no-hint
  plus self-supervised regularizers, while reusable subroutine hints remain
  the upper bound at `8333`, and served promotion is still explicitly refused
- the first public scratchpad / controlled-position executor framework
  comparison now also exists beside that same bounded lane:
  `psionic-ir` now owns bounded `flat_trace` and
  `delimited_chunk_scratchpad` formatting plus `absolute_monotonic`,
  `segment_reset`, and `trace_schema_buckets` controlled position-ID schemes,
  `psionic-models` now exposes public framework descriptors plus locality
  evidence inspection, and `psionic-train` now freezes the resulting
  arithmetic symbolic and algorithmic comparison at
  `fixtures/tassadar/reports/tassadar_scratchpad_framework_comparison_report.json`;
  the report keeps the lane explicitly `learned_bounded_success`, cuts
  arithmetic max output local position from `14` to `3`, cuts algorithmic max
  output local position from `11` to `3`, preserves final output tokens
  exactly, and keeps scratchpad overhead plus reset counts explicit instead of
  hiding them behind one aggregate score
- the learned-structure supervision follow-on now also exists beside that same
  bounded lane: `psionic-models` now derives structural target families for
  instruction pointer, branch outcome, stack delta, memory diff, and
  workload-specific state from the frozen trace ABI, `psionic-train` now
  persists structural-supervision weights and split-level coverage inventory in
  `TassadarSequenceTrainingManifest`, `psionic-eval` now emits
  `structural_supervision_report.json` for bounded validation decodes, and
  `psionic-research` now materializes the comparison root at
  `fixtures/tassadar/runs/sudoku_v0_supervision_ablation_v1`; those committed
  artifacts keep the claim bounded but prove richer supervision moves the
  learned lane on the same early-curriculum setup (`4570` to `7812` aggregate
  target-token exactness, `4375` to `6875` first-32 exactness,
  instruction-pointer `5000` to `7000` bps, stack-delta `2500` to `5833` bps,
  and no claim that branch/memory/workload-specific families are already green
  on the short validation window)
- the learned subroutine-library follow-on now also exists beside that bounded
  lane: `psionic-models` now exposes a public reusable subroutine library for
  sort, CLRS shortest-path, and sudoku-style workloads, `psionic-train` now
  materializes the same seeded corpus under explicit `full_trace` versus
  `subroutine_library` supervision modes and computes deterministic
  held-out-workload OOD target-reuse comparisons, and `psionic-research` now
  freezes the bounded comparison artifact at
  `fixtures/tassadar/reports/tassadar_subroutine_library_ablation_report.json`;
  those committed artifacts keep the claim bounded by proving only supervision
  target reuse deltas, not trained-model exactness
- the post-Phase-15 trained-attention follow-on now also exists beside that
  seeded comparison: `psionic-research` now runs a bounded attention-family
  output-head training loop and persists its artifacts under
  `fixtures/tassadar/runs/sudoku_v0_attention_training_v1`,
  while a trained-family comparison is now preserved under
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v2`;
  the resulting artifacts show real learning progress off the seeded
  attention-family floor (`6563` bps aggregate and first-32 exactness instead
  of `0`), but they also show the same remaining blocker plainly: the trained
  attention family still gets the first target token wrong (`0` bps
  first-target), still yields `0/2` exact bounded traces, and therefore still
  does not clear the open Phase 14 gate
- the post-Phase-15 boundary-adapter follow-on now also exists beside that
  trained attention floor: `psionic-models` now carries a bounded
  relative-target output-bias adapter, `psionic-research` now preserves the
  failed output-head-only boundary attempt under
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v1`,
  the improved adapter-backed run under
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v2`, and
  the later hidden-state projection-adapter follow-ons under
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v3` and
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v4`, the
  newer previous-token-conditioned transition-adapter follow-on under
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v5`, the
  later joint transition+projection fine-tune under
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v6`, the
  later trace-schema and per-position saturation runs under
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v7`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v8`, and
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v9`, and
  the current same-corpus comparison under
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v11`;
  the accepted `boundary_v2` run keeps the token-0 fix without destroying the
  bounded suffix (`10000` bps first-target, `7500` bps first-8, `6875` bps
  first-32), the later `boundary_v3` / `boundary_v4` follow-ons prove the
  remaining blocker is structural rather than vague, and the newer
  `boundary_v5` / `v7` pair proves the structural-transition surface can move
  that blocker deeper into the trace (`10000` bps first-target, `8750` bps
  first-8, `7188` bps first-32), while the later `boundary_v6` / `v8`
  joint-adapter fine-tune reproduces but does not beat that ceiling and the
  later `boundary_v7` / `boundary_v8` / `boundary_v9` saturation set preserves
  the last red attention-family ceiling before the green `promotion_v3`
  continuation
- the learned-family follow-on comparison now also exists under
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v12`; it keeps the
  same bounded Sudoku-v0 workload contract but widens the compared family set
  to hull-specialized lookup, direct sparse-top-k lookup, hybrid attention,
  and recurrent/windowed lookup, with per-family fit/exactness reports plus
  model descriptors and run bundles; the committed `v12` comparison remains
  honestly red and comparison-only, with all four seeded families at `0` bps
  first-target / first-8 / first-32 exactness, the recurrent family changing
  the long-trace contract explicitly, and the hybrid attention family fitting
  `0/2` full shared sequences under its `512`-token cap
- the separate Phase 17 compiled lane now also exists beside that learned
  stack: `psionic-models` now exposes a bounded typed
  `TassadarCompiledProgramExecutor` with compile-evidence bundles,
  `psionic-eval` now emits machine-readable exactness and
  compatibility/refusal reports for the real Sudoku-v0 corpus under the
  workload family id `tassadar.wasm.sudoku_v0_search.v1.compiled_executor`,
  and `psionic-research` now persists the canonical bundle root at
  `fixtures/tassadar/runs/sudoku_v0_compiled_executor_v0`; the
  committed artifacts prove only a bounded compiled/proof-backed lane on the
  matched corpus (`8/8` exact trace matches against CPU reference and `32/32`
  exact refusal matches), with explicit `eval_only` posture, so this does not
  close the open learned-lane promotion gate and does not unblock 9x9 by
  itself
- the separate Phase 18 compiled Hungarian lane now also exists beside that
  learned stack: `psionic-runtime` now carries a real bounded
  `tassadar.wasm.hungarian_v0_matching.v1` min-cost matching workload over 4x4
  cost matrices, `psionic-eval` now emits a real Hungarian-v0 benchmark
  package plus machine-readable compiled exactness/refusal and learned-vs-
  compiled lane-status reports, and `psionic-research` now persists the
  canonical bundle root at
  `fixtures/tassadar/runs/hungarian_v0_compiled_executor_v0`;
  the committed artifacts prove only a bounded Hungarian-class workload
  contract plus an exact compiled/proof-backed lane on the matched corpus
  (`8/8` exact trace matches and `32/32` exact refusal matches), so this does
  not make the learned lane green by association and does not justify article-
  parity language
- the separate learned Hungarian-v0 lane now also exists with explicit
  dual-state supervision at
  `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0`; the train-side
  bundle persists exactness-curve, failure-sample, structural-supervision,
  and fit artifacts and keeps token/state/final-result exactness separate
  (`aggregate=6839`, `first_target=0`, `first_32=6875`, `final_outputs=0`,
  `workload_specific_state=7568`), so the honest posture is `research_only`
  even though the full Hungarian-v0 traces fit the current learned model
  window
- the learned long-horizon boundary is now explicit rather than implicit:
  `psionic-research` persists
  `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`,
  which freezes a typed `unsupported_horizon` refusal for million-step and
  article-class learned traces until an exact learned long-horizon benchmark
  bundle replaces the current refusal posture
- `psionic-research` can now use that bounded trained-small receipt as an
  explicit comparator inside the learned-plus-compiled and learned-circuit
  Tassadar research family, but that does not expand the train-side claim
  boundary beyond `validation_corpus_only`

## Apple Adapter Reality

The Apple Foundation Models lane now has an honest training answer, and that
answer needs to stay precise.

Yes, the repo can now train LoRA-style Apple adapter patches and export valid
`.fmadapter` packages.

That does **not** mean:

- the Swift bridge is doing the training
- Apple exposes a repo-controlled Foundation Models training API
- the current Apple lane is already a generalized distributed trainer

The current path is:

1. `psionic-data` imports Apple adapter JSONL into the repo-owned dataset
   contract, preserving tokenizer, prompt-shaping, tool/schema augmentation,
   and long-context packing lineage. The app now derives that lineage from the
   live Apple FM bridge health profile plus dataset-aware default-instruction
   and locale posture instead of reusing hard-coded placeholder digests, and
   `psionic-data` now derives prompt/completion/tool/schema token captures from
   the full structured transcript path rather than from character counts.
2. `psionic-environments` binds that dataset to the Apple adapter
   train/eval/benchmark environment package family so the train, held-out eval,
   and runtime-smoke lanes all point at one versioned environment truth.
3. `psionic-train` now exposes the live Apple SFT executor through the
   repo-owned `AppleAdapterTrainingExecutionBackend` plus
   `run_apple_adapter_sft_export(...)`, which turns packed samples into
   token-sequence reference batches with turn-aware prompt pooling,
   tool/schema boundary encoding, completion-side token-sequence supervision,
   checkpoint emission, and staged `.fmadapter` package construction.
4. The current operator path in
   `apps/autopilot-desktop/src/apple_adapter_training_control.rs` now calls
   that Rust-native Psionic executor directly for the authoritative train and
   export path. A fresh live operator validation run
   (`rust-native-3769-validation-1773643126445`) completed the Rust train
   step, wrote repo-owned checkpoints, emitted a Rust-native Apple
   `adapter_weights.bin` blob-storage container, staged the final package, and
   then completed the bridge-backed held-out eval plus runtime-smoke path with
   `runtime_smoke_passed=true`. That means the narrow shipped Apple
   train/export lane is now Rust-native end to end, even though later issues
   still remain around eval fidelity, telemetry, UI, and cleanup.
5. `psionic-eval` runs the held-out and benchmark-style adapter harnesses, and
   the bridge-backed runtime-smoke path proves the exported package can be
   loaded and exercised against the live Apple FM runtime. That smoke path now
   checks the exported package's base-model, tokenizer, and template lineage
   against the expected runtime compatibility profile before acceptance. The
   Apple runtime parity pass now also normalizes raw guided-generation schemas
   through `AppleFmGenerationSchema::with_title_hint(...)`, uses bounded greedy
   generation options during live eval/smoke, and backs `lookup_doc` /
   `lookup_code` eval cases with real repo retrieval instead of echo tools.
   Benchmark reports now embed the full base/adapted `EvalRunState` receipts and
   a stable paired per-case receipt layer, so each weak case carries the request
   envelope, expected output, base/adapted outputs, structured-output payloads,
   observed tool-call transcripts, and copied model-request/runtime failure
   details instead of collapsing into aggregate benchmark deltas.
6. `autopilotctl training launch ...`, `autopilotctl training watch ...`,
   `autopilotctl training export ...`, and `autopilotctl training accept ...`
   provide the shipped app-owned operator flow, while
   `autopilotctl apple-fm load ...` and `autopilotctl apple-fm attach ...`
   exercise the resulting package through the retained bridge. That operator
   flow now runs the long Apple launch pipeline in the background, persists
   typed JSONL telemetry at `<run_directory>/telemetry.jsonl`, and projects the
   same phase, heartbeat, ETA, artifact-path, resource-summary, and
   failure-context fields through desktop-control so both CLI scripts and later
   WGPUI panes can inspect the run before it completes. The legacy toolkit
   compatibility wrapper in `psionic-train` is now quarantined behind the
   non-default `legacy-apple-toolkit-oracle` feature, and the packaged Apple
   release checks run
   `scripts/release/check-psionic-apple-rust-only-gate.sh`, which fails if the
   shipped Apple operator path regresses back to toolkit-root discovery,
   Python-interpreter discovery, or authoritative toolkit shell-outs.

That last bridge-backed load step is the authoritative export-validity gate.

A `.fmadapter` directory can be:

- inventory-valid
- metadata-valid
- and still not be an Apple-valid runtime asset

On 2026-03-15, GitHub issue `#3664` added the canonical parity and acceptance
program for this boundary:

- `scripts/release/check-psionic-apple-export-parity.sh`
- `fixtures/apple_adapter/TOOLKIT_ORACLE_RECIPE.md`

The train system must not treat package write success as equivalent to runtime
load success.

The shipped Apple reference lane is intentionally narrow:

- base-model weights stay frozen
- only adapter parameter groups are updated
- the current live operator precision posture is `f32_reference`
- the current live operator activation-checkpoint posture is `disabled`
- tokenizer and packing truth now come from a repo-owned Apple-compatible
  transcript preprocessor, not an Apple-exact tokenizer oracle
- the current live operator export is repo-owned and bridge-accepted for the
  narrow reference lane, with `adapter_weights.bin` emitted as the same
  64-byte-aligned Core ML blob-storage family Apple accepts rather than as raw
  concatenated fp16 bytes
- the higher-level optional follow-on is draft-model distillation, not a second
  generic full-model trainer

That is why the right current claim is "the repo now ships a Rust-native
authoritative Apple train/export lane for one narrow single-host reference
path" rather than "Psionic now has complete Rust-native distributed Apple FM
training."

### Relationship To Cluster And Distributed Training Substrate

The Apple lane already reuses real Psionic train substrate outside the narrow
execution backend itself:

- fixed-budget trainer-step execution
- reusable autodiff and optimizer layers
- dataset/tokenizer/packing contracts
- environment package bindings
- held-out eval and runtime-smoke harnesses
- training summaries, receipts, and accepted-outcome authority publication
- run-graph, orchestrator, and validator vocabulary used elsewhere in
  `psionic-train`

But the Apple lane does **not** yet consume the broader distributed-training
substrate as a live execution path:

- no real `psionic-cluster` multi-node Apple training run is claimed
- no collective-backed gradient exchange or sharded optimizer execution is used
  by the shipped Apple backend
- no production multi-device training kernel or memory-sharded Apple trainer is
  claimed
- no broader cluster scheduler is yet dispatching Apple training windows across
  multiple machines

What exists today is the reusable contract layer for those future steps:
`psionic-runtime` multi-device topology truth, `psionic-collectives`
collective planning, distributed-optimizer object models, orchestrator state,
and datastream/checkpoint movement. That substrate is meant to be reused later
for broader Psionic training lanes, including any future widened Apple path,
but it is not the current execution reality for the shipped Apple adapter
operator flow.

## Decentralized Adapter Training Program

Psionic is now explicitly planning decentralized adapter training as a
first-class train workload family.

That program sits on top of the already-retained train substrate:

- `TrainingRun`
- `TrainingStage`
- `TrainingWindow`
- `PolicyRevision`
- `RolloutArtifact`
- `TrainerBatch`
- `CheckpointPointer`
- `EvalRun`
- validator receipts and accepted-outcome authority projection

The new claim this doc freezes is not "the repo already has decentralized
adapter training."

The new claim is:

> the repo now has one canonical spec for decentralized adapter training, with
> Apple adapters as the first narrow lane and open adapter backends as the next
> generalized lane under the same control-plane vocabulary.

That widening step is now no longer only planned. `psionic-train` also owns a
first bounded non-Apple execution backend in
`crates/psionic-train/src/open_adapter.rs`. The implemented reference
target is intentionally narrow and explicit:

- admissible model family: `gpt_oss.decoder_lm_head_lora`
- adapter format: `safetensors`
- first concrete backend label: `open_adapter_backend.cuda.gpt_oss_lm_head`
- supervision shape: repo-owned hidden-state plus target-token batches
- export proof: the produced artifact roundtrips through
  `psionic-adapters::LmHeadLoraAdapterArtifact`

That is enough to make the decentralized adapter architecture honestly
non-Apple-only while still staying well short of a generalized full-model or
multi-node open-backend trainer claim.

That contract layer is no longer only planned. `psionic-train` now owns a
typed adapter-window state machine in
`crates/psionic-train/src/adapter_window.rs` that can represent one
window end to end with typed receipts for:

- assignment
- local execution summary
- upload completion
- validator disposition
- aggregation eligibility
- sealed-window aggregation or promotion

Those receipts bind adapter target identity, dataset-slice identity, source
policy revision, and source checkpoint pointer without ad hoc JSON sidecars,
and the crate now carries a runnable harness that proves the window lifecycle
from planning through reconciliation. What remains open after this step is the
live cluster, artifact, validator, authority, operator, and product projection
work in the later issue set.

The next control-plane layer is now implemented too. `psionic-train` also owns
an adapter cluster coordinator in
`crates/psionic-train/src/adapter_cluster.rs` that mirrors live
`psionic-cluster` membership and telemetry into adapter contributor eligibility,
derives deterministic contributor ranking from readiness plus capability facts,
and plans typed adapter windows with inspectable contributor-set revisions and
assignment seeds. The new reference harness proves membership churn can evict
or replace contributors for later windows without collapsing the whole run.

The worker-facing side of that program is also now real in
`crates/psionic-train/src/adapter_worker_protocol.rs`. The crate now
owns typed adapter-worker sessions, heartbeats, progress snapshots, assignment
claims, assignment acknowledgements, submission receipts, claim expiry, and
claim supersession on top of one active adapter window. Those transcripts bind
worker and session identity to window id, contribution id, source policy
revision, source checkpoint pointer, and upload-manifest expectations so late,
superseded, unauthorized, or mismatched submissions can be refused with
machine-legible outcomes instead of ad hoc local strings.

The artifact-staging layer is now real as well in
`crates/psionic-train/src/adapter_artifact_storage.rs`. That module
derives adapter-package datastream manifests from contribution payloads,
enforces manifest-digest and chunk-digest replay safety across resumable upload
sessions, registers completed contribution artifacts through the generic train
artifact controller, tracks reviewable/accepted/rejected retention posture, and
promotes sealed-window adapter state into typed checkpoint manifests plus
window-scoped checkpoint pointers. The included harness proves interrupted
uploads can resume from a committed cursor, corrupt chunks are refused without
advancing state, and the latest promoted checkpoint for a window can be
restored deterministically.

The provenance-security layer is now implemented too in
`crates/psionic-train/src/adapter_submission_security.rs`. The crate
now owns signed adapter-manifest envelopes that bind assignment digest, claim
digest, worker id, session id, auth subject, trust class, target policy
revision, target checkpoint pointer, upload expectation, upload reference, and
manifest/object digests under the worker session's submission-signing key.
Accepted contributions now preserve independently verifiable provenance bundles,
while signature mismatch, reassigned worker/session identity, and stale-session
checks surface typed reject or quarantine receipts for later validator and
aggregation stages.

The validator-owned adapter review layer is now implemented in
`crates/psionic-train/src/adapter_validation.rs`. That module consumes
submission receipts, staged artifacts, signed provenance bundles, and security
receipts; samples contributions for validator replay; emits typed
`accepted`/`quarantined`/`rejected`/`replay_required` verdicts; writes the
existing adapter-window validator and aggregation-eligibility receipts; and
seals the window with one scored summary over admitted, accepted, quarantined,
rejected, and replay-required work. Candidate window scoring now also consumes
held-out eval, benchmark aggregate summaries, and runtime-smoke eval runs, and
Apple-format windows can require runtime smoke before promotion-ready status is
true.

The first real aggregation-and-promotion path is now implemented in
`crates/psionic-train/src/adapter_aggregation.rs`. The crate now owns a
deterministic first rule for accepted adapter contributions,
`weighted_manifest_digest_merge_v1`, which preserves accepted artifact,
validator, security, provenance, and aggregation-weight lineage; emits a typed
promotion receipt; and either promotes a new `PolicyRevision` plus
`CheckpointPointer` or records an explicit held outcome when accepted work or
validator posture is insufficient. `AdapterTrainingClusterCoordinator` can now
consume that receipt, update its current input revision and checkpoint pointer,
reconcile the finished window, and plan the next window directly from the
promoted revision without local manual patching.

### First Lane, Next Lane, And Non-Goals

The first live execution lane remains the narrow single-host Apple adapter path
documented above.

The first decentralized lane should still begin with adapters, not full-model
weight updates, because the existing repo truth already has:

- adapter-only update semantics
- adapter package lineage
- benchmark and runtime-smoke validation posture
- accepted-outcome authority plumbing
- app-owned operator workflows that can expose contribution and review state

The first decentralized lane should therefore mean:

- one windowed multi-party adapter-training program
- one declared adapter target identity
- one declared dataset-slice identity per contribution
- one validator-owned disposition for each submitted contribution
- one aggregation decision that can produce a new policy revision or no-op

The first implementation has explicit non-goals:

- no world-scale synchronous all-reduce
- no full-model training claims
- no app-owned runtime or bridge logic inside Psionic train
- no generalized market/product claims before authority, validation, and
  operator truth exist
- no ad hoc JSON sidecars for contribution-state truth

### Canonical Workload Vocabulary

The decentralized adapter workload family should use the following additional
typed vocabulary on top of the generic train objects:

| Object | Purpose | Planned Scope |
| --- | --- | --- |
| `AdapterTrainingProgram` | Root program identity for one decentralized adapter-training family | one adapter target family, one policy family, one validator posture |
| `AdapterTrainingWindow` | One bounded contribution interval for one adapter target | contributor set, dataset-slice plan, policy revision in, seal state |
| `AdapterContributionAssignment` | One worker assignment into one adapter window | worker id, adapter target identity, dataset slice, replay budget |
| `AdapterContributionArtifact` | One uploaded local adapter delta or equivalent contribution bundle | manifest, delta/checkpoint pointer, local execution summary, provenance |
| `AdapterContributionReceipt` | One durable control-plane receipt for assignment, execution, upload, and disposition | worker id, window id, policy revision, validator result, aggregation eligibility |
| `AdapterAggregationReceipt` | One record of how accepted contributions were combined for one sealed window | accepted set, weights, output policy revision, checkpoint pointer |
| `PolicyPromotionReceipt` | One durable record of whether a sealed window promoted or held policy state | previous revision, next revision, acceptance basis, checkpoint lineage |

Every contribution receipt must bind:

- adapter target identity
- window id and contributor-set revision
- policy revision in
- dataset slice identity
- local execution summary
- uploaded artifact identity
- validator disposition
- aggregation eligibility decision
- resulting policy revision or no-promotion decision when the window seals

### Canonical Contribution Dispositions

The validator-owned contribution disposition vocabulary is:

| Disposition | Meaning |
| --- | --- |
| `accepted` | the contribution passed required checks and may participate in aggregation |
| `quarantined` | the contribution is retained for review but is not aggregation-eligible until explicitly released |
| `rejected` | the contribution failed required checks and is permanently excluded from aggregation |
| `replay_required` | the contribution cannot be trusted yet and must be regenerated or replayed under a fresh assignment |

These states are machine-legible control-plane truth, not UI-only labels.

### Canonical Window Flow

One decentralized adapter window should follow this explicit control flow:

1. The orchestrator plans one `AdapterTrainingWindow` with a sealed contributor
   set, dataset-slice plan, adapter target identity, and input policy revision.
2. Workers receive typed `AdapterContributionAssignment` records rather than
   free-form task text.
3. Each worker produces one `AdapterContributionArtifact` plus a local
   execution summary bound to the declared dataset slice and policy revision.
4. Datastream and artifact storage record upload completion as typed receipt
   state.
5. The validator emits one machine-legible disposition of `accepted`,
   `quarantined`, `rejected`, or `replay_required`.
6. The window seals only after the acceptance or replay policy is satisfied.
7. Aggregation consumes only aggregation-eligible accepted contributions and
   emits one `AdapterAggregationReceipt`.
8. Promotion emits one `PolicyPromotionReceipt` that either advances the policy
   revision or records an explicit no-promotion outcome.

This preserves deterministic replay because assignment, upload, disposition,
seal, aggregation, and promotion are all typed receipts on one window graph.

### Decentralized Adapter Acceptance Matrix

The repo must not claim decentralized adapter training is implemented until all
of the following rows are true:

| Claim Boundary | Required Truth |
| --- | --- |
| Spec frozen | this doc names decentralized adapter training as a first-class program, defines the object vocabulary above, and preserves the first-lane/non-goal boundary |
| Window contracts live | `psionic-train` represents adapter windows and contribution receipts as typed Rust objects without ad hoc JSON sidecars |
| Cluster selection live | contributor selection, membership, assignment, heartbeat, and window seal posture are bound to live `psionic-cluster` truth rather than local-only mocks |
| Artifact staging live | contribution uploads, manifests, delta/checkpoint pointers, and provenance are persisted through typed datastream/artifact contracts |
| Validator dispositions live | accepted, quarantined, rejected, and replay-required states are emitted by validator-owned code and preserved for replay or authority projection |
| Aggregation live | sealed windows can aggregate accepted contributions into a new policy revision or explicit no-promotion result with checkpoint lineage |
| Authority projection live | kernel/Nexus persist accepted window outcomes and policy-promotion truth as durable authority projections |
| Operator flow live | desktop and `autopilotctl` expose truthful contributor/operator state for window planning, submission, review, aggregation, and acceptance |
| Productization live | provider-substrate and compute-market claims are published only after the above control, validator, and authority rows are implemented |

Until every row above is true, the honest repo claim remains:

> single-host Apple adapter training and one bounded non-Apple open adapter
> backend are real, but decentralized adapter training is still an incomplete
> program rather than a finished productized system.

On 2026-03-15, GitHub issue `#3648` added the first repo-owned QA and
reference-program layer for this workload family in
`crates/psionic-train/src/adapter_reference_program.rs`.

That layer now makes the following acceptance proof explicit:

- one canonical two-window decentralized adapter reference run with multiple
  contributors and contributor churn between windows
- coverage for both the Apple adapter lane and the first non-Apple open
  adapter backend under the same control-plane path
- typed latency envelopes for window sealing, replay completion, and policy
  promotion
- explicit chaos rejection of stale uploads, manifest corruption, and
  replay-missing submissions before promotion

The canonical regression harness for that layer is now:

- `scripts/release/check-psionic-decentralized-adapter-reference-program.sh`

This does not close the overall decentralized adapter program by itself.
Authority projection, app-owned operator surfaces, and broader productization
rows from the acceptance matrix above still remain separate closure steps.

On 2026-03-15, GitHub issue `#3661` added the first concrete operator runbook
for the clustered follow-on:

- `docs/ARCHITECTURE_EXPLAINER_CLUSTER_BRINGUP_RUNBOOK.md`

That runbook is intentionally narrow and explicit about posture:

- it preserves the distinction between today's real single-host Apple operator
  path, today's cluster rehearsal truth, and later live multi-device ambition
- it names the preferred first topology as a small homogeneous Apple lab
  cluster
- it includes an explicitly experimental Apple Metal plus NVIDIA mixed-role
  path that is useful for cluster, staging, and receipt bring-up but does not
  overclaim Apple-valid mixed-backend training

On 2026-03-15, GitHub issue `#3662` tightened the first heterogeneous
mixed-backend experiment boundary:

- Apple remains the coordinator, `.fmadapter` export, and runtime-validation
  authority host for the Apple lane
- the first concrete non-Apple participant target is the CUDA-backed open
  adapter lane identified by `open_adapter_backend.cuda.gpt_oss_lm_head`
- mixed Apple plus NVIDIA participation is therefore shared at the cluster,
  artifact, validator, and replay layers first, not overclaimed as symmetric
  Apple training

## Canonical Train Objects

The full train system needs a formal object model. Today only some of these
objects have concrete repo types; the rest are planned and should become the
stable vocabulary for train-class execution.

| Object | Purpose | Current Repo Status |
| --- | --- | --- |
| `TrainingRun` | Root identity for one training program | `implemented_early` |
| `TrainingStage` | One named phase such as SFT, agentic SFT, or RL | `implemented_early` |
| `TrainingWindow` | One synchronized contribution or trainer interval with its own contributor set and transition state | `implemented_early` |
| `TrainerStep` | One optimizer update over one trainer batch | `implemented_early` |
| `PolicyRevision` | Versioned policy or weight state used by workers and trainer | `implemented_early` |
| `RolloutArtifact` | One worker-produced trajectory or completion bundle | `implemented_early` |
| `TrainerBatch` | One accepted batch of rollout or corpus inputs for a trainer step | `implemented_early` |
| `EnvironmentPackage` | One versioned environment definition used by training and eval | `implemented_early` |
| `BenchmarkPackage` | One validator-owned packaged benchmark or reference evaluation profile | `implemented_early` |
| `EvalRun` | One online or offline evaluation execution | `implemented_early` |
| `CheckpointPointer` | One stable pointer to the latest accepted checkpoint for a run, stage, or window | `implemented_early` |
| `CheckpointManifest` | One shard, digest, writer, and durability manifest for a checkpoint flush | `implemented_early` |
| `Checkpoint` | Recoverable training state and lineage anchor | `partial` |
| `ValidatorVerdict` | Verification result attached to one rollout, batch, or eval artifact | `implemented_early` |

Today the concrete object vocabulary is strongest around:

- `TrainingCheckpointReference`
- `TrainingRecoveryContext`
- `TrainingDeviceMeshContext`
- `TrainingCollectiveContext`
- `DatastreamManifest` and `DatastreamManifestRef`

Current checkpoint substrate is carried today by
`TrainingCheckpointReference`, explicit `CheckpointPointer` and
`CheckpointManifest` contracts, plus checkpoint-scoped datastream manifests.

The rest of the train object model still needs to be built explicitly.

What is still missing most clearly from the current vocabulary is:

- deeper checkpoint lineage policy such as checkpoint retention tiers,
  cross-window promotion rules, and cold-restore governance
- broader `ValidatorVerdict` families for trainer-batch and eval-class artifacts

### Current `RolloutArtifact` Shape

`RolloutArtifact` now exists in early form inside `psionic-train`. The current
shape already includes at least:

- `worker_id`
- `policy_revision`
- `environment_ref@version`
- `task_id` or task digest
- `token_ids`
- `logprobs`
- reward or rubric outputs
- termination reason
- proof or validator reference fields
- stable `artifact_digest`

## Current State At A Glance

| Subsystem | Current Status | What Is Real Today |
| --- | --- | --- |
| Runtime training truth | `implemented_early` | `TrainingRecoveryContext`, checkpoint refs, elastic-membership context, device-mesh context, collective context |
| Datastream | `implemented_early` | resumable manifests, checkpoint or dataset bindings, policy-weight control refs, freshness windows, and delivery receipts |
| Collectives | `implemented_early` | elastic mesh observation, bandwidth-aware local/global sync planning, transport-feedback replanning, and benchmark-gated quantized collective policy |
| Train session state | `implemented_early` | membership observation, async checkpoint state, durability transitions, live-recovery planning |
| Data contracts | `implemented_early` | `psionic-data` now owns versioned dataset manifests, tokenizer digests, split declarations, resumable iteration cursors, long-context packing policies, and Apple adapter JSONL import or validation with typed tool-schema augmentation plus tokenizer/prompt-shaping packing lineage |
| Adapters | `implemented_early` | adapter identity, package manifests, hosted adapter binding lineage, and first Apple `.fmadapter` reader/writer plus file-inventory validation |
| Sandbox for RL/train workloads | `implemented_early` | bounded execution, background jobs, warm reusable pools, staged loop inputs, pool acquisition receipts, and repeated agentic iteration receipts now exist in `psionic-sandbox` |
| Training core | `implemented_early` | `psionic-train` now has a typed fixed-budget trainer-step loop, `psionic-ir` now provides reusable reverse-mode autodiff plus explicit detach/training-mode gradient semantics beneath it, the repo-owned Apple adapter execution backend now turns packed Apple dataset batches into adapter-only gradient batches for that loop, the first higher-level Apple SFT lane closes the path through typed training summary plus `.fmadapter` export, and an explicitly separate optional Apple draft-model distillation lane now emits paired draft payloads plus latency or acceptance metadata; the crate also now owns a first non-Apple open adapter backend for `gpt_oss.decoder_lm_head_lora`, producing loadable LM-head LoRA `safetensors` artifacts from bounded hidden-state supervision under the same fixed-budget core; parameter-group scaling semantics, scheduler bindings, optimizer state/residency, step telemetry, model-IO roundtrip, and checkpoint restore lineage remain explicit over gradient batches |
| Training run graph | `implemented_early` | `psionic-train` now owns typed runs, contributor-set revisions, topology revisions, persistent participant ranking, heartbeats, departures, and window transitions |
| Orchestrator | `implemented_early` | `psionic-train` now owns typed window-control, assignment posture, rollout-assignment refs, rollout-admission receipts, bounded off-policy freshness budgets, rollout-worker heartbeats, claims, upload receipts, and trainer-batch assembly requests over the run graph |
| Live RL run service | `implemented_early` | `psionic-train` now also owns a bounded durable `LiveRlRunService` above the run graph, orchestrator, worker protocol, and validator state, with persistent run snapshots, current status or per-window artifacts, graceful draining or stop semantics, and restart recovery through a service-owned filesystem root |
| Training sampler service | `implemented_early` | `psionic-train` now owns the first bounded `TrainingSamplerService` above the repo-owned open-adapter lane, with health/readiness inspection, active policy revision status, completions/chat/logprob request surfaces, explicit hot-swap to newer promoted revisions, optional checkpoint and weight-broadcast identity in refresh/status, and fail-closed stale-revision refusal |
| Live RL update bridge | `implemented_early` | `psionic-train` now also owns a bounded `OpenAdapterLiveRlUpdateExecutor` that joins orchestrator batches to accepted rollout receipts plus prompt-side sequence inputs, preserves prompt/completion boundaries and per-token observed-versus-live logprobs, carries reward/advantage plus optional chosen-token teacher logprobs into one weighted adapter step, and emits a promoted served revision ready for sampler adoption |
| Environment ABI | `implemented_early` | `psionic-environments` now owns the package ABI, versioned key, workload/policy/difficulty/benchmark package shape, tool/rubric contracts, deterministic runtime session state machine, registry install/pin/group resolution, the first bounded live `EnvironmentRuntimeService` with worker/queue admission plus typed submission/activation/completion receipts, and a reusable Apple adapter train/eval/benchmark bundle with typed runtime refs plus train/eval parity receipts, while registry and authority truth remain in kernel/Nexus |
| Eval runtime | `implemented_early` | `psionic-eval` now owns held-out eval runs, rubric-scored sample/runtime contracts, benchmark packages, repeat-run aggregation, local validator simulation, and Apple adapter held-out plus benchmark harnesses with structured-output, tool-call, and runtime-smoke receipts, while kernel/Nexus still own canonical eval-run authority truth |
| Synthetic-data flows | `partial_outside_psionic` | synthetic-data job creation, append, finalize, and verification flows exist in kernel/Nexus, but no Psionic-native generation runtime exists yet |
| Rollout artifacts | `implemented_early` | `psionic-train` now has checkpoint-aware policy revisions, proof-bearing rollout artifacts, rollout-admission receipts, bounded stale-rollout pruning, and deterministic trainer-batch assembly with policy-lineage digests |
| Validator-aware RL verification | `implemented_early` | `psionic-train` now owns rollout-verification bundles, replay or duplicate detection, sampled benchmark checks, and typed validator verdicts; broader service productization is still later |
| Decentralized adapter window contracts | `implemented_early` | `psionic-train` now owns typed adapter-window receipts and a runnable window harness covering assignment, execution, upload, validator disposition, aggregation eligibility, seal, aggregation, and reconcile over one adapter-targeted window |
| Decentralized adapter cluster selection | `implemented_early` | `psionic-train` now owns a cluster-backed adapter coordinator that mirrors live membership and telemetry into contributor eligibility, deterministic ranking, contributor-set revisions, assignment seeds, and churn-safe window replanning |
| Decentralized adapter worker protocol | `implemented_early` | `psionic-train` now owns typed adapter-worker sessions, heartbeats, claim/ack flows, progress telemetry, superseded-claim retries, and contribution submission receipts bound to policy revision, checkpoint pointer, and upload expectations for one active adapter window |
| Decentralized adapter artifact staging | `implemented_early` | `psionic-train` now owns resumable adapter contribution uploads, manifest/chunk verification, typed contribution-artifact receipts, disposition-aware retention windows, and promoted window checkpoint manifests plus pointers for deterministic restore |
| Decentralized adapter provenance security | `implemented_early` | `psionic-train` now owns signed manifest envelopes, worker/session/auth-subject binding, independently verifiable accepted provenance bundles, and typed reject or quarantine receipts for signature mismatch, reassignment, or stale-session cases |
| Decentralized adapter validator and window scoring | `implemented_early` | `psionic-train` now owns sampled replay verification, typed validator dispositions, window sealing summaries over admitted/accepted/quarantined/rejected/replay-required work, and candidate held-out/benchmark/runtime-smoke gating for promotion readiness |
| Decentralized adapter aggregation and promotion | `implemented_early` | `psionic-train` now owns a deterministic accepted-delta aggregation rule, typed promotion receipts with artifact/validator/security/provenance lineage, hold-vs-promote gating, and coordinator-side adoption of the promoted revision plus checkpoint pointer for the next window |

## Current Crate Ownership

The current train-relevant ownership split in Psionic is:

- `psionic-runtime`
  - reusable runtime truth for training recovery, device meshes, collectives,
    and work classes such as `CollectiveStep` and `CheckpointFlush`
- `psionic-datastream`
  - resumable transport for datasets, checkpoints, served artifacts, and
    adapter packages
- `psionic-data`
  - versioned dataset manifests, tokenizer digests, split declarations,
    streamed iteration contracts, long-context packing rules, and Apple
    adapter dataset import or validation with typed schema/tool augmentation
- `psionic-collectives`
  - elastic mesh observation, local/global sync planning, transport-feedback
    replanning, and benchmark-gated collective policy
- `psionic-distributed`
  - bounded public framework-distributed group plus core collective-helper
    shell above runtime mesh truth, with explicit reference emulation and
    honest refusal where public backend transport has not landed yet
- `psionic-environments`
  - environment package ABI, execution entrypoints, tool and rubric hooks,
    artifact expectations, versioned dataset bindings, deterministic runtime
    sessions, and reusable Apple adapter train/eval/benchmark bundle helpers
- `psionic-eval`
  - held-out eval runs, rubric-scored sample/runtime contracts, benchmark
  packages, repeat-run aggregation, operator-local validator simulation, and
  Apple adapter held-out/benchmark/runtime-smoke harnesses
- `psionic-train`
  - training-session truth for checkpointing, live recovery,
    elastic-membership posture, typed run graphs, contributor-set revisions,
    window lifecycle, the fixed-budget training-core reference loop, the
    repo-owned Apple adapter reference execution backend, the higher-level
    Apple SFT/export lane, the optional Apple draft-model distillation lane,
    orchestrator state, and RL-facing rollout or batch contracts
- `psionic-adapters`
  - adapter package identity, Apple `.fmadapter` parsing or writing, file
    inventory validation, and hosted binding lineage
- `psionic-sandbox`
  - bounded sandbox execution substrate and background-job lifecycle
- `psionic-cluster`
  - durable ordered-state, cluster admission, catch-up, and topology truth

The broader OpenAgents tree now also has train-adjacent authority surfaces
outside Psionic for:

- environment package descriptors and registry behavior
- compute evaluation-run, training-run, and accepted-outcome lifecycle
- narrow Apple adapter-hosting and Apple-training provider/market projection
  surfaces
- synthetic-data job and verification lifecycle

This is already a meaningful substrate split. The missing work is higher in the
stack.

## What Is Implemented Today

### 1. Runtime-level training truth already exists

`psionic-runtime` already has typed training-class truth surfaces. The most
important ones are:

- `TrainingRecoveryPosture`
  - `SteadyState`
  - `LateJoinPending`
  - `Recovering`
  - `ElasticReconfiguration`
  - `AsyncCheckpointInFlight`
- `TrainingCheckpointAvailability`
  - `None`
  - `AsyncWriteInFlight`
  - `Durable`
- `TrainingElasticMembershipContext`
  - membership epoch
  - cluster-state digest
  - topology digest
  - active, joining, draining, and offline node sets
- `TrainingCheckpointReference`
  - checkpoint family
  - stream id
  - manifest digest
  - object digest
  - writer node
  - membership epoch and topology digests
  - optional logical step and durability timestamp
- `TrainingRecoveryContext`
  - current posture
  - checkpoint availability
  - elastic-membership facts
  - optional latest checkpoint
  - recovering and late-joiner node ids
- `TrainingDeviceMeshAxis`
  - data-parallel, tensor-parallel, pipeline-parallel, and expert-parallel axes
- `TrainingDeviceMeshContext`
  - mesh id, revision, backend, communication class, members, and axes
- `TrainingCollectiveContext`
  - collective kind
  - quantization mode
  - payload bytes
  - wire-byte estimate
  - benchmark justification

This matters because the train system does not start from nothing. The runtime
already has a typed language for recovery, checkpoints, meshes, and collectives.

### 2. Datastream already owns resumable training-class artifact movement

`psionic-datastream` is not training-specific, but it already covers several
training-critical artifact families.

Its subject model already includes:

- `TokenizedCorpus`
- `EvalBundle`
- `Checkpoint`
- `PolicyWeights`
- `ServedArtifact`
- `AdapterPackage`

Its manifests already support:

- payload digesting
- stable chunk descriptors
- dataset bindings
- checkpoint bindings
- policy-weight bindings
- control-plane-visible mirror metadata
- resumable transfer cursors
- restart-safe client progress
- final delivery receipts

That means the train system already has a real substrate for:

- dataset shard transport
- checkpoint transport
- policy-weight shard transport and lightweight control-plane refs
- eval-bundle movement
- adapter-package distribution

What is still missing is not "a data plane exists or not." The missing work is
the broader lifecycle policy over that data plane: richer retention classes,
cross-region mirror governance, and tighter integration with higher-level
orchestrator freshness rules.

### 3. Collective planning already exists

`psionic-collectives` already implements a real, inspectable collective
planning substrate.

The important current pieces are:

- `ElasticCollectivePlanner`
- `CollectiveMeshMember`
- `QuantizedCollectiveBenchmark`
- `CollectiveTransportFeedback`
- `CollectiveSyncCadencePolicy`
- `observe_mesh`
- `record_benchmark`
- `plan_collective`
- `observe_transport_feedback`
- `plan_sync`

The current planner already does several important things honestly:

- validates that declared mesh axes match member count
- ensures mesh members are actually active in the current membership set
- increments mesh revision only when mesh truth changes
- requires explicit benchmark approval before planning a quantized collective
- records transport feedback and surfaces typed replan triggers when bandwidth,
  latency, stream pressure, or mesh revision cross policy boundaries
- plans local subgroup sync separately from full-mesh sync when degraded
  transport and explicit subgroup topology justify it
- emits a `CollectiveExecutionPlan` with:
  - runtime-visible collective posture
  - explicit ring handoffs
  - a low-level `RuntimeWorkItem`
- emits a `CollectiveSyncCadenceReceipt` with:
  - cadence class
  - next global sync step
  - selected quantization
  - transport degradation posture
  - typed replan triggers

This is already enough to say Psionic has training-class collective truth.

It is not enough to say Psionic has a complete distributed optimizer or
end-to-end trainer.

### 4. `psionic-train` already implements session truth for checkpointing and recovery

`psionic-train` currently owns the most concrete part of the train system that
exists today.

Its public API centers on `TrainingSessionState`, which already supports:

- `new`
- `latest_durable_checkpoint`
- `active_checkpoint_write`
- `observe_membership`
- `begin_async_checkpoint`
- `mark_checkpoint_durable`
- `plan_live_recovery`

That session truth now also has a repo-owned writeback implementation under
`psionic-train`:

- `AsyncCheckpointWritebackWorker`
- `AsyncCheckpointWritebackPayload`
- `AsyncCheckpointWritebackReceipt`
- `write_checkpoint_payload_sync`
- `train_parameter_golf_local_reference_with_async_checkpoint_writeback`
- `LocalTrainMetricFanout`
- `LocalTrainMetricJsonlSink`
- `LocalTrainMetricProgressSink`
- `LocalTrainMetricStructuredLogSink`
- `train_parameter_golf_local_reference_with_metric_sink`

What that means in practice:

- Psionic can derive elastic-membership epochs from authoritative cluster truth.
- Psionic can begin an async checkpoint only from a checkpoint-scoped
  datastream manifest and only when the writer node is a known ready member.
- Psionic can surface in-flight checkpoint flush work as a typed runtime work
  item.
- Psionic can transition a checkpoint from writing to durable and update the
  durable recovery posture.
- Psionic can derive explicit live-recovery plans for recovering nodes and late
  joiners.
- Psionic can hand immutable checkpoint payloads to a bounded writer worker,
  publish checkpoint directories atomically, refuse queue overload explicitly,
  and flush or refuse pending writes during shutdown without making partial
  checkpoints look committed.
- The representative `Parameter Golf` local-reference lane now exercises that
  writeback path and proves restore equivalence plus lower train-loop stall at
  the handoff point relative to synchronous writes.
- Psionic can fan one typed local train metric into progress output,
  structured-log lines, JSONL telemetry, and in-memory pre-aggregation without
  turning those local streams into benchmark or receipt truth.

The current recovery action set is already meaningful:

- `ResumeFromDurableCheckpoint`
- `FenceRecoveringNodes`
- `StageCheckpointForLateJoiners`
- `RebalanceWorldSize`
- `BlockUntilDurableCheckpoint`
- `ContinueSteadyState`

That is real train-substrate behavior, not just placeholder nouns.

### 5. Adapter lineage already exists for later train outputs

`psionic-adapters` is not the core training loop, but it is relevant because a
train system eventually needs to emit attributable artifacts.

The adapter subtree already owns:

- `AdapterArtifactIdentity`
- `AdapterPackageManifest`
- target-family and residency semantics
- hosted binding lineage for adapter-backed serving

This means Psionic already has an artifact vocabulary for one class of training
outputs beyond full checkpoints.

### 6. Sandbox substrate exists, and the RL-oriented shape is now early rather than absent

`psionic-sandbox` already owns:

- runtime detection
- profile realization
- bounded job execution
- background jobs
- file transfer
- warm reusable pools
- staged loop inputs
- pool acquisition receipts
- repeated agentic iteration receipts
- execution receipts

This is enough to support bounded compiled runners plus early RL/post-training
iteration contracts.

It is not yet the mature high-throughput RL/post-training sandbox shape. The
remaining gaps are:

- productionized RL throughput and pool tuning
- broader environment-owned lifecycle and policy integration
- stronger operator and security hardening for long-running train workloads

### 7. Environment, eval, training-authority, and synthetic-data truth now spans Psionic runtime crates and authority-owned kernel surfaces

The recent issue closures matter because they changed both Psionic and the
broader system around it.

The tree now has Psionic-native execution crates for:

- environment package ABI and deterministic runtime sessions in
  `psionic-environments`
- held-out eval runs, benchmark packages, repeat-run aggregation, and local
  validator simulation in `psionic-eval`
- one repo-owned Apple adapter training execution lane in `psionic-train`
- one bounded MLX workflow package in `psionic-mlx-workflows` for
  deterministic synthetic SFT/preference dataset bundles, reward/judge helper
  plans, adapter merge/export artifacts, and a local safetensors-backed
  publish snapshot

The tree also has broader OpenAgents support for:

- environment package descriptors and registry behavior
- environment refs bound into compute products and delivery proofs
- evaluation-run creation, sample ingestion, and finalize flows
- training-policy registration, training-run create/finalize flows, and
  accepted-outcome publication
- narrow Apple adapter-hosting capability publication plus matching
  compute-market truth surfaces
- synthetic-data job creation, append, finalize, and verification flows

Those capabilities currently live in kernel/proto and Nexus-control surfaces.

So the accurate reading is:

- Psionic now has native environment and eval runtime clients plus one
  repo-owned Apple training lane inside the compute substrate
- the larger platform owns the canonical authority truth for environment, eval,
  training-run, and accepted-outcome records
- provider and market surfaces now expose one narrow Apple training and
  adapter-hosting projection on top of that authority truth
- synthetic-data lifecycle still remains `partial_outside_psionic` because
  there is no Psionic-native long-running generation/runtime authority path
  yet, even though `psionic-mlx-workflows` now owns one bounded local dataset
  materialization package above the shared data/train substrate

## What Psionic Can Honestly Claim Today

Today Psionic can honestly claim all of the following:

- training-class execution now has typed recovery, checkpoint, mesh, and
  collective truth in reusable crates
- clustered training recovery can be reasoned about with replay-safe session
  state rather than ad hoc logs
- checkpoint transport has a resumable data-plane substrate with delivery
  receipts
- collective planning already has benchmark-gated quantization and explicit mesh
  revisions
- fixed-budget trainer-step execution is real, with explicit optimizer-state
  ownership, residency transitions, and step telemetry
- reusable autodiff plus explicit detach or no-grad semantics now live in
  `psionic-ir` rather than trainer-private code
- reusable SGD, Adam, AdamW, LARS, and LAMB primitives plus distributed-
  optimizer contracts now live in `psionic-train`
- `psionic-distributed` now exposes a bounded public `fsdp_apply_gradients`
  helper above those distributed-optimizer contracts, with typed `zero_stage3`
  admission, mixed replicated/full-shard group handling, optional global-norm
  clipping, reference-emulated reduce-scatter/all-gather, and stable apply
  receipts
- rollout artifacts, trainer-batch assembly, policy revisions, and
  validator-aware verification are first-class typed contracts
- environment ABI and held-out eval runtime now exist in reusable Psionic
  crates
- sandbox execution now supports warm reusable pools and repeated agentic
  iteration receipts
- training-related artifact lineage is now materially first-class data rather
  than opaque side files
- the first repo-owned Apple adapter SFT lane is real in `psionic-train`, and
  the optional Apple draft-model distillation lane is now separate typed
  behavior instead of being implied by one generic training path
- the broader OpenAgents stack now has authority-layer environment, eval,
  training-run, and accepted-outcome flows that Psionic can target as
  execution clients
- the desktop app now has a truthful Apple training operator flow on top of the
  Psionic substrate, including explicit launch, eval, export, and
  accepted-outcome publication boundaries
- the broader stack also projects one narrow Apple adapter-hosting and
  Apple-training truth path into provider and compute-market surfaces, without
  pretending that broader train procurement is complete

That is a meaningful base.

## What Psionic Cannot Honestly Claim Yet

Psionic cannot honestly claim any of the following yet:

- full production-scale Rust-native model training across real multi-device
  runtime kernels
- full production-scale Rust-native RL or post-training throughput
- broad autodiff coverage across every future backend-extension and training op
- true multi-device execution kernels and ZeRO or FSDP transport and partition
  exchange
- fully mature checkpoint retention, promotion, and cold-restore governance
- broad kernel-backed accepted-outcome authority for every train artifact and
  lifecycle beyond the current Apple adapter path
- full security hardening, chaos coverage, and operator lifecycle for the train
  stack
- a broad provider-market training family or buyer-facing procurement surface
  on top of the current Apple reference lanes
- the broader research-loop or productization program beyond the current
  reference runs

Those are still planned.

## The Gap, Precisely

The gap is no longer "there is no train subtree."

The gap is:

> Psionic now has early trainer, orchestrator, rollout, environment, eval,
> validator, and reusable framework-core gradient or update substrate, but it
> still lacks the runtime breadth, hardening, and operator or product layers
> required for a complete distributed train system.

That gap is the main planning target for the rest of this doc.

## Target Train System

The target Psionic train system should be six explicit subsystems.

### 1. Training core

Owns:

- training graph or backward substrate
- optimizer state
- optimizer-state residency, offload, and prefetch policy
- gradient update policy
- checkpoint save and restore
- trainer step loop
- step-level training telemetry such as grad, update, and parameter norms

This is the engine that does the actual learning work.

### 2. Orchestrator

Owns:

- participant roles
- training-window creation, seal, score, and reconcile transitions
- rollout scheduling
- deterministic assignment for contributor, batch, and eval slices
- batch assembly
- off-policy budgeting
- policy revision tracking
- stage transitions
- online eval interleaving

This is the control plane for the train system.

### 3. Data plane

Owns:

- dataset transport
- checkpoint transport
- policy-weight broadcast
- eval-bundle transport
- artifact freshness and replay posture

This extends the current `psionic-datastream` substrate.

### 4. Environments and eval

Owns:

- environment package ABI
- benchmark package and validator-owned reference benchmark profiles
- rollout execution contracts
- tool and multi-turn abstractions
- reward and rubric contracts
- repeat-run scoring and robust aggregation rules
- operator-local validator simulation against the same packaged benchmark
  environment
- offline and online eval over the same environment definition

This is where environment-bound training becomes honest.

### 5. Validation and adjudication

Owns:

- rollout-verification bundles
- cheap universal checks
- sampled expensive checks
- stale or malformed rollout rejection
- timer, token-accounting, and final-state verification where a benchmark or
  validator package requires them
- declared execution-strategy verification for benchmark-class workloads
- validator verdict artifacts

This is the integrity loop for untrusted or semi-trusted rollout workers.

### 6. Operator and authority integration

Owns:

- training receipts
- topology and checkpoint inspection
- validator posture inspection
- environment version visibility
- accepted-outcome export into market or kernel truth when appropriate

This is how the train system becomes operable instead of remaining a research
toy.

## Canonical Planned Role Split

The full train system should separate these roles explicitly.

### Trainer

Trusted execution responsible for:

- reading trainer batches
- applying gradient updates
- producing new checkpoints or policy revisions
- emitting step and checkpoint receipts

### Orchestrator

Trusted control plane responsible for:

- scheduling rollouts
- assigning workers
- maintaining persistent participant ranking
- selecting bounded contributor sets from a wider active population
- enforcing freshness windows
- assembling trainer batches
- coordinating evaluation
- feeding the trainer the right artifacts

### Rollout workers

Untrusted or semi-trusted execution responsible for:

- generating trajectories or outputs against a declared policy revision
- returning typed rollout artifacts
- attaching enough metadata for validator review

### Validators

Integrity checkers responsible for:

- universal schema checks
- sampling-shape checks
- termination checks
- stale-policy checks
- duplicate or copycat detection
- contribution normalization and ranking feedback
- sampled high-cost verification when economics justify it

### Environment runtime

Trusted execution substrate responsible for:

- package loading
- stateful multi-turn task execution
- tool invocation
- reward or rubric application
- sandbox-bound execution where required

### Data-plane services

Responsible for:

- checkpoint and weight transfer
- resumable corpus delivery
- manifest and digest verification
- freshness and retention policy

### Contributor Selection And Ranking

The mature train system should treat active participants and contributing
participants as different sets.

That means:

- the system may keep a wider population admitted and heartbeat-visible
- only a bounded contributor set should actually produce work in a given round,
  interval, or trainer window
- contributor selection should consider freshness, persistent ranking, topology,
  and diversity rather than only "who asked first"
- duplicate or copycat behavior should reduce effective contribution weight and
  feed back into future participant ranking

This is the cleanest way to keep elastic membership open without letting every
active participant distort batch quality or network cost.

### Control Plane Versus Heavy Artifact Plane

The train control plane should not carry the heavy payloads.

The intended split is:

- the orchestrator, validators, and operator surfaces exchange run ids,
  artifact refs, digests, policy ids, and receipts
- checkpoints, policy weights, datasets, rollout payloads, and eval bundles
  move through the heavy artifact plane in `psionic-datastream`

This keeps control messages lightweight and replayable while the actual bytes
stay in the resumable artifact substrate.

## Canonical Planned Lifecycle

The mature Psionic train lifecycle should look like this:

1. A training run is created with stable run identity, policy, environment, and
   checkpoint lineage.
2. The orchestrator forms or revises the participant topology, contributor set,
   and current `TrainingWindow`.
3. The collective planner materializes the device mesh and collective posture.
4. The heavy artifact plane stages the active checkpoint, policy weights, and
   dataset or environment artifacts while the control plane carries only refs,
   digests, and policy posture.
5. Only the selected contributor subset begins rollout or trainer work under
   explicit policy, assignment, and freshness constraints.
6. The window transitions through explicit control states such as `planned`,
   `active`, `sealed`, `scored`, and `reconciled` as work is accepted and
   judged.
7. Rollout artifacts or trainer-step inputs are validated and assembled into
   trainer batches.
8. The trainer advances one or more steps and emits step-level metrics,
   receipts, and optional checkpoints.
9. Async checkpoint flushes begin and later transition to durable state.
10. Recovery, late join, reconfiguration, or eviction events update the run
   topology and checkpoint posture.
11. Online and offline eval may run against the same environment contract or
    benchmark package contract.
12. Accepted outcomes produce durable train and eval receipts and later, when
    market-relevant, can flow into kernel truth.

The current repository implements only pieces of steps 2, 3, 4, 9, and 10.

## Planned Run State Machine

The mature train system should give operators and controllers a small explicit
run-state machine.

| `TrainingRunStatus` | Meaning |
| --- | --- |
| `planned` | run identity exists but execution has not started |
| `initializing` | artifacts, participants, and execution substrate are still being prepared |
| `active` | trainer and rollout work are progressing normally |
| `recovering` | the run is reconfiguring or resuming from checkpoint-backed state |
| `paused` | the run is intentionally halted without being terminal |
| `completed` | the run reached a successful terminal outcome |
| `failed` | the run reached a terminal failure outcome |

The runtime and operator surfaces should not infer these states indirectly from
scattered logs. They should be first-class train truth.

## Training Time Semantics

Training execution depends on explicit time boundaries.

The most important ones are:

- policy freshness windows
- rollout expiry windows
- checkpoint cadence
- contributor reselection intervals
- validator sampling or adjudication intervals
- environment timeout limits
- sandbox reuse and pool lifetime limits

These time boundaries sit above the generic execution timing defined in
`ARCHITECTURE.md` and should be recorded in train policy and receipts where
they affect acceptance or rejection.

## Canonical Train Receipts

OpenAgents is receipt-first, so the train system needs explicit receipt
families rather than vague references to "logs" or "artifacts."

Today the repo already has some lower-level receipt substrate:

- `DatastreamDeliveryReceipt`
- sandbox execution receipts
- runtime execution-proof bundles
- checkpoint and recovery contexts that can feed later receipts

The first train-specific receipt family now exists through
`RolloutAdmissionReceipt`, but the mature train system should still emit at
least these broader receipts.

| Receipt | Purpose | Minimum Contents |
| --- | --- | --- |
| `TrainingRunReceipt` | One durable summary for a full run or run stage | run id, stage id, policy ids, environment refs, checkpoint lineage, validator posture, final outcome |
| `TrainingWindowReceipt` | One durable record for one contributor or trainer window transition | run id, stage id, window id, contributor-set revision, policy revision, transition state, validator posture |
| `TrainerStepReceipt` | One accepted optimizer step | run id, stage id, step id, trainer batch digest, policy revision in and out, optimizer policy, checkpoint linkage |
| `CheckpointReceipt` | One checkpoint creation or durability event | run id, stage id, checkpoint family, manifest digest, object digest, writer identity, durability state |
| `RolloutReceipt` | One rollout artifact and its acceptance result | run id, worker id, policy revision, environment version, rollout digest, reward and termination posture, acceptance result |
| `ValidatorReceipt` | One validator verdict over a rollout, batch, or eval artifact | validator policy id, sampled or universal check class, referenced artifact digests, verdict, reason codes |
| `EvalReceipt` | One online or offline evaluation result | eval run id, environment version, rubric version, policy revision, artifact digests, score summary |

The most important design rule is simple:

> every economically or operationally important train event should have a typed
> receipt family, not only a log line or an in-memory state transition.

Train objects define the durable execution vocabulary; receipts record accepted
state transitions and outcomes over those objects.

## Policy Surfaces

The full train system should make the configurable policy surfaces explicit.
The spec should say not only what happens, but what operators and higher-level
controllers are allowed to tune.

| Policy Surface | What It Governs |
| --- | --- |
| `TrainingPolicy` | trainer step budget, training-window cadence, checkpoint cadence, optimizer posture, gradient clipping, contributor caps, stage transitions, halt policy |
| `EnvironmentPolicy` | admissible environment packages, tool access, state persistence, reward and rubric posture |
| `ValidatorPolicy` | universal checks, sampled expensive checks, stale-policy tolerances, duplicate-detection posture, contribution normalization, benchmark verification posture, rejection posture, penalty posture |
| `CollectivePolicy` | mesh layout, sync cadence, quantization mode, replan triggers, communication class |
| `SandboxPolicy` | allowed profiles, warm-pool behavior, runtime limits, filesystem or network posture, retry behavior |
| `ArtifactPolicy` | artifact freshness windows, retention classes, replay rules, archival posture, provenance requirements |

Current repo truth only covers a small piece of this policy surface directly:

- collective quantization approval, benchmark posture, sync cadence, and
  transport thresholds
- cluster admission and readiness posture
- checkpoint durability posture
- sandbox profile realization

Most train policy remains to be formalized.

### Example Policy Values

The policy surfaces above become easier to reason about when rendered with
concrete examples.

| `TrainingPolicy` Field | Example Value |
| --- | --- |
| `max_policy_drift` | `3 revisions` |
| `checkpoint_interval` | `1000 steps` |
| `gradient_clip_norm` | `1.0` |
| `halt_on_entropy_drop` | `true` |
| `max_rollout_age_ms` | `30000` |
| `max_contributing_workers` | `256` |

### Policy Revision Propagation

Policy revisions should propagate through the data plane as staged artifacts,
not as implicit mutable state.

The intended model is:

- the trainer emits a new policy revision or checkpoint-backed weight state
- the revision is published through `psionic-datastream` as a staged artifact
- the orchestrator enforces freshness and admissibility before assigning work
- rollout workers and evaluators must bind their outputs to the specific policy
  revision they consumed

This keeps policy lineage replay-safe and validator-reviewable.

Control-plane coordination should carry refs, digests, and policy ids rather
than embedding the heavy policy payloads directly in orchestration messages.

## Training Failure Semantics

The train system needs explicit failure handling, not only a list of failure
classes. The table below describes the expected control policy for the mature
system.

| Failure Type | Expected System Response |
| --- | --- |
| rollout worker crash | replay or reassign the rollout task and mark prior claim incomplete |
| stale or mismatched policy revision | reject the rollout artifact and emit a stale-policy receipt |
| duplicate or copied rollout | reject or deweight the artifact, emit duplicate-detection reason codes, and update participant ranking |
| validator rejection | discard or quarantine the referenced rollout or batch and record reason codes |
| checkpoint flush failure | block any state transition that requires durability and keep the run in non-durable posture |
| orchestrator crash | resume from durable orchestrator state and latest accepted checkpoint lineage |
| trainer crash | restart from the latest durable checkpoint and replay admissible pending control-plane state |
| environment package mismatch | reject execution before rollout start and emit environment-mismatch reason codes |
| sandbox runtime failure | terminate the affected task, record runtime and profile identity, and apply retry or quarantine policy |
| topology shock or node loss | trigger elastic reconfiguration, recovery planning, and possibly world-size rebalance |
| datastream interruption | resume from the last committed cursor rather than restart blind transfer |

The system should never collapse these into one generic "training failed"
outcome. Failure handling is part of train truth.

Orchestrator durability and trainer durability are related but distinct; loss
of one must not silently imply loss of the other.

## Security Model

The train system explicitly allows for partially trusted and untrusted roles, so
the threat model belongs in the spec and not only in later issue descriptions.

| Threat | Mitigation Direction |
| --- | --- |
| malicious rollout workers | validator sampling, schema checks, stale-policy rejection, worker admission controls |
| artifact poisoning or tampering | manifest digests, object digests, provenance requirements, signed artifacts where policy requires |
| checkpoint tampering | datastream manifest verification plus checkpoint-family and writer identity linkage |
| environment compromise | signed or pinned packages, sandbox policy, version pinning, package admissibility policy |
| policy drift | explicit policy revisions, freshness windows, off-policy budget enforcement |
| copied or replayed rollouts | duplicate detection, artifact-digest lineage, contribution normalization, and participant-ranking penalties |
| worker spam or flooding | task-claim limits, admission control, rate limiting, and orchestrator-side pruning |
| orchestrator inconsistency | durable orchestrator state and replay-safe receipts |
| validator abuse or misconfiguration | validator policy versioning, sampled check receipts, adjudication reason codes |

The current repo already helps here in a limited way through:

- manifest and chunk digests in `psionic-datastream`
- explicit checkpoint identity and writer linkage in `psionic-runtime`
- benchmark-gated collective posture in `psionic-collectives`
- bounded profile and execution receipts in `psionic-sandbox`

The broader train security model is still planned.

## Train Artifact Retention Model

Retention policy affects reproducibility, cost, and later authority linkage, so
it should be named now even before enforcement exists.

| Artifact Class | Expected Retention |
| --- | --- |
| durable checkpoints | long-term or archival, because they anchor recovery and promotion lineage |
| trainer-step receipts | long-term, because they define accepted optimization history |
| rollout artifacts | medium-term by default, with longer retention for sampled, disputed, or promoted artifacts |
| validator receipts and proof refs | long-term, because they justify acceptance or rejection outcomes |
| eval summaries | long-term, because they anchor quality and release decisions |
| raw sandbox traces and transient logs | short-term by default unless attached to an incident or dispute |

The retention table does not imply the implementation already exists. It defines
the operating model the train stack should eventually enforce.

## What The Intellect Papers Change For Psionic

The March 13 audit remains directionally correct. The useful lessons from the
Intellect papers are still these.

### From INTELLECT-1

Psionic should take:

- explicit elastic topology as first-class truth
- join and recovery modes as policy rather than ad hoc behavior
- heartbeat and explicit departure semantics
- bandwidth-aware background replanning
- quantized sync only when benchmark-justified and receipt-bearing

Psionic should not copy:

- their exact Python or PyTorch stack
- their exact transport stack
- one specific pretraining topology as permanent architecture truth

### From INTELLECT-2

Psionic should take:

- trainer, orchestrator, rollout worker, and validator as distinct roles
- policy-weight distribution as its own data plane
- untrusted rollout validation with cheap universal checks and sampled expensive
  checks
- explicit off-policy budgets
- first-class curriculum and filtering
- instability telemetry as product truth

Psionic should not copy:

- one GRPO recipe as the permanent train contract
- one relay or firewall model as the only architecture
- one economic or ledger substrate as the product base layer

### From INTELLECT-3

Psionic should take:

- environment packages as independent products
- one environment contract for training and eval
- multi-turn and tool-using environments as first-class abstractions
- RL-oriented sandbox throughput
- stage transitions from SFT to agentic SFT to RL
- orchestrator state as core product truth

Psionic should not copy:

- their exact Python environment module system
- their exact Kubernetes control plane
- their exact optimizer or MoE decisions as architecture truth

## All-Rust Implication

If OpenAgents means "no Python trainer and no Python environment system," then
the completion bar is high.

An honest all-Rust Psionic train system now exists in early form across all of
these layers inside the Rust subtree:

- training core
- optimizer ownership
- rollout artifacts
- environment ABI
- data and corpus contracts
- eval runtime
- compiled runner and crash boundary

The completion bar is still high, though.

Psionic cannot honestly claim a finished all-Rust train system until
multi-device execution kernels, broader autodiff or operator coverage, mature
environment execution at scale, hardening, and operator-grade lifecycle
management all exist inside the Rust subtree.

## Planned Crate Shape

The most likely mature crate shape is:

- `psionic-train`
  - training core, run graph, checkpoint lineage, trainer state, orchestrator
    contracts
- `psionic-collectives`
  - mesh and collective planning, quantized sync policy
- `psionic-datastream`
  - dataset, checkpoint, policy-weight, and eval-bundle transport
- `psionic-eval`
  - shared online and offline evaluation runtime
- `psionic-data` or `psionic-datasets`
  - dataset manifests, tokenizer state, splits, packing, and curriculum facts
- `psionic-environments`
  - environment ABI, runtime sessions, and package-loading contracts
- `psionic-sandbox`
  - pooled execution substrate for environment-bound agentic workloads
- `psionic-adapters`
  - later train-output lineage for adapters and promoted derived artifacts

This is the architectural direction. It is not all implemented today.

The planned crate shape is canonical for current ownership direction, but it is
not a guarantee that every future subsystem lands under exactly these final
crate names.

## Current-To-Target Matrix

| Area | Current Repo Truth | Target Repo Truth |
| --- | --- | --- |
| Checkpoint lineage | present in `psionic-train` and `psionic-runtime` | durable checkpoint families, promotion, replay, and restore across full training programs |
| Elastic membership | present in `psionic-runtime` and `psionic-train` | full participant lifecycle with heartbeats, rejoin, eviction, and topology history |
| Collective planning | present in `psionic-collectives` | full local/global sync planning with distributed optimizer integration |
| Weight broadcast | present in `psionic-datastream` | staged policy-weight broadcast with freshness cutoffs and relay policy |
| Training steps | typed fixed-budget reference loop present | broader Rust-native trainer-step engine |
| RL rollouts | typed rollout, bounded stale-rollout budgeting, and worker-protocol contracts present | validator-ready lineage and sampled adjudication |
| Environment ABI | typed runtime ABI plus typed package shape present | broader package loading, composition, and environment system |
| Eval runtime | present in `psionic-eval` | shared online/offline eval and rubric runtime, benchmark packages, and local validator simulation |
| Sandbox throughput | bounded one-shot substrate exists | RL-throughput warm pools and repeated environment loops |
| Validators for RL | rollout-verification bundles and sampled adjudication contracts present | broader service productization, batch-level adjudication, and authority integration |
| Operator surfaces | app-owned desktop-control and `autopilotctl` surfaces now exist on top of Psionic, but Psionic still does not own its own operator crate or full train operator plane | inspection, diagnostics, and receipts across all train subsystems |

## Path To Completion

The path from the current repo to a real train system is best read in four
waves.

### Wave 0: implemented substrate

Already in tree:

- runtime training truth
- datastream manifests and receipts
- collective planning substrate
- session checkpoint and recovery substrate
- adapter lineage substrate
- bounded sandbox execution substrate

### Wave 1: implemented early all-Rust train platform

Now in tree:

- fixed-budget training core with reusable autodiff and optimizer layers under
  it
- rollout artifacts, policy-lineage contracts, worker protocol, and
  validator-ready verification bundles
- environment ABI plus environment registry helpers
- data, tokenizer, split, and packing contracts
- held-out eval runtime plus Apple held-out, benchmark, and runtime-smoke
  harnesses
- run graph, checkpoint lineage, orchestrator state machine, off-policy
  budgeting, and scheduling/economics contracts
- RL-throughput sandbox primitives
- repo-owned Apple training execution backend, Apple SFT/export lane, and
  optional Apple draft-model distillation lane

### Wave 2: remaining productization and scaling

Needed next:

- true multi-device execution kernels and distributed optimizer integration
- memory-sharding, partition exchange, and broader collective or optimizer
  integration at scale
- broader model, tokenizer, and artifact-format interoperability
- stronger security, provenance, and authority integration beyond the current
  Apple accepted-outcome path
- mature artifact retention, promotion, and cold-restore governance
- broader operator lifecycle and market/product surfaces beyond the current
  app-owned Apple reference workflow

### Later scope

After the above:

- model promotion and release governance
- human preference and critique ingestion

Those later items matter, but they are not prerequisites for the core
environment-first Intellect-style train stack.

## Proposed GitHub Issue Program

The issue program below is written from the current repository state, not from
the older "there is no `psionic-train` crate" assumption.

This program first landed as issues `#3564` through `#3593` and was later
extended by the framework-core follow-ons `#3602` and `#3603`, plus the
Apple-lane closures `#3616` through `#3631`.

### Core Platform Build-Out

### 1. `Psionic Train: complete the Rust-native training core beyond recovery substrate`

Status: implemented on 2026-03-14 via GitHub issue `#3564`.

Added `psionic-train` fixed-budget training-core types and behavior for:

- typed parameter groups
- explicit optimizer-state ownership
- optimizer-state residency policy and transitions
- machine-legible step telemetry for gradient, update, and parameter norms
- visible window and cadence scheduling
- checkpoint-anchored restore via `TrainingSessionState`

Issue `#3603` extends that core with a reusable optimizer layer in
`src/optimizer.rs` so SGD, Adam, AdamW, LARS, and LAMB step semantics are no
longer trainer-private. The fixed-budget loop now composes with the reusable
optimizer surface instead of carrying its own ad hoc update implementation.

Issue `#3602` adds reusable autodiff underneath that loop in `psionic-ir`:
explicit gradient-bearing graph construction, an IR-level `detach` op,
training/evaluation plus no-grad posture, symbolic backward plans, dense
reference materialization, and a trainer-integration proof that the resulting
gradients can feed the fixed-budget training core without trainer-local
gradient logic.

The canonical runbook and harness are now:

- `docs/TRAINING_CORE_FIXED_BUDGET_REFERENCE.md`
- `scripts/release/check-psionic-training-core.sh`

The current step path is intentionally an explicit-gradient reference loop over
`f32` tensor payloads, but it no longer implies trainer-private gradient logic.
Autodiff and optimizer behavior now live in reusable lower Psionic layers,
while broader operator-family coverage and higher-order training behavior still
remain future work.

On 2026-03-15, GitHub issue `#3631` added the missing repo-owned Apple
training execution backend inside `psionic-train`:

- validation over the repo-owned Apple dataset tokenizer and prompt-shaping
  lineage plus the SFT-capable Apple environment bundle
- deterministic Apple sample batching on top of the packed dataset contract
- adapter-only parameter selection with frozen-base semantics
- repo-owned forward/loss and low-rank gradient production that feeds
  `TrainingGradientBatch` and `FixedBudgetTrainingRun`
- explicit training-posture declaration for the currently supported `f32`
  reference precision path, with graph-level checkpoint transforms available
  in `psionic-ir` but activation checkpointing still disabled in this shipped
  Apple reference lane

This lands the learning computation itself for the first Apple lane.

On 2026-03-15, GitHub issue `#3625` then added the higher-level Apple SFT lane
on top of that backend:

- fixed-budget step execution across the repo-owned Apple batches
- typed step receipts and final training summary for the Apple run
- initial/final adapter-only portable bundle snapshots plus derived adapter
  delta
- reproducibility metadata suitable for later authority publication
- valid `.fmadapter` export through `psionic-adapters`
- bounded long-run retention in the operator-facing SFT path: the optimizer step
  still consumes explicit gradient tensors, but completed-step records now
  redact those tensors after application so step history does not keep one full
  gradient snapshot per step in memory during longer Apple runs

That means the first honest Rust-native Apple adapter SFT path is now real in
repo code.

On 2026-03-15, GitHub issue `#3626` added the explicitly separate optional
Apple draft-model distillation lane on top of that SFT path:

- fixed-budget teacher/student distillation over the repo-owned Apple batches
- explicit teacher/draft runtime pairing and dual-precision posture capture
- deterministic latency and speculative-acceptance accounting in typed batch
  records and summary output
- portable draft checkpoint export plus paired `draft.mil` or
  `draft_weights.bin` payload emission inside `.fmadapter`

Authority publication and desktop workflow are now real through the later
Apple-lane issue closures. Broader provider-market training truth and generic
market claims beyond the current Apple reference path remain later work.

### 2. `Psionic RL: define rollout artifacts, trainer batches, and policy-lineage contracts`

Status: implemented on 2026-03-14 via GitHub issue `#3565`.

Added `psionic-train` RL-facing contracts for:

- checkpoint-aware `PolicyRevision`
- proof-bearing `RolloutArtifact`
- deterministic `TrainerBatch` assembly
- explicit `PolicyRevisionLineage`

The canonical runbook and harness are now:

- `docs/ROLLOUT_ARTIFACT_POLICY_LINEAGE_REFERENCE.md`
- `scripts/release/check-psionic-rl-rollout-artifacts.sh`

This issue makes rollout payloads, trainer-batch assembly, and policy lineage
real and reusable. It does not yet claim freshness enforcement, worker
protocols, validator adjudication, or full orchestration.

### 3. `Environments: define a Rust-native environment ABI and runtime contract`

Status: implemented on 2026-03-14 via GitHub issue `#3566`.

Added the `psionic-environments` crate for:

- canonical `environment_ref@version` package identity
- Rust-native environment package ABI
- execution entrypoints, tool interfaces, rubric hooks, and artifact
  expectations
- deterministic runtime sessions with turn, tool, artifact, and rubric receipts

The canonical runbook and harness are now:

- `docs/ENVIRONMENT_ABI_REFERENCE.md`
- `scripts/release/check-psionic-environment-abi.sh`

Kernel and Nexus still own registry and authority truth. This issue lands the
Psionic-side runtime and contract layer only.

On 2026-03-15, GitHub issue `#3622` extended the same crate with a reusable
Apple adapter environment bundle:

- a shared train/eval core package plus a benchmark-only package over the same
  typed environment ABI
- explicit Apple session/runtime, tool-bundle, rubric-binding, and
  structured-output refs carried as package metadata that now affects package
  digests
- train/eval group composition that proves the same pinned core package is
  reused across both surfaces through an explicit parity receipt

### 4. `Psionic Data: add Rust-native dataset, tokenizer, split, and packing contracts`

Status: implemented on 2026-03-14 via GitHub issue `#3567`.

Added the `psionic-data` crate for:

- canonical `dataset_ref@version` identity through `DatasetKey`
- typed dataset manifests bound to tokenizer digests and tokenized shard refs
- split declarations over `psionic-datastream` manifest refs with explicit
  shard-level sequence and token counts
- resumable streamed iteration contracts with deterministic shard ordering and
  epoch-wrap semantics
- sequence-packing and batch-packing policies for long-context workloads

The canonical runbook and harness are now:

- `docs/DATASET_TOKENIZER_PACKING_REFERENCE.md`
- `scripts/release/check-psionic-data-contracts.sh`

This issue keeps byte movement in `psionic-datastream` but makes data lineage,
iteration, and packing policy first-class typed Psionic contracts. The
environment ABI now binds versioned dataset keys from this layer instead of
free-form dataset refs.

On 2026-03-15, GitHub issue `#3621` extended that same crate with the first
repo-owned Apple adapter dataset path:

- UTF-8 JSONL import into typed Apple message, tool, and guided-generation
  records
- fixture-backed validation of message ordering, tool definitions, and
  `response_format` schema requirements
- explicit tokenizer and prompt-shaping lineage metadata for later train/eval
  reuse
- deterministic Apple sample packing over explicit prompt/completion/tool/schema
  token captures, with typed refusal on tokenizer or prompt-shaping drift

On 2026-03-15, GitHub issue `#3651` added the first reviewed real-run corpus
on top of that same Apple dataset contract for the `Psionic architecture
explainer` target:

- curated `train`, `held_out`, and `benchmark` JSONL splits under
  `fixtures/apple_adapter/datasets/psionic_architecture_explainer/`
- a repo-owned curation manifest that tags every split-local sample with task
  family, expected behavior, review posture, and source provenance
- machine-checkable split-leakage validation so benchmark rows remain distinct
  from train and held-out rows even when they draw from the same stable docs

This is still a first reviewed positive-path corpus, not yet the full
truthfulness or refusal slice for the first real run.

On 2026-03-15, GitHub issue `#3652` extended that same corpus with explicit
negative, correction, and retrieval-style refusal rows across `train`,
`held_out`, and `benchmark`:

- Apple-lane overclaim correction for single-host versus distributed-training
  claims
- bridge-versus-training-engine correction so the runtime sidecar is not
  confused with the Rust-owned training path
- ownership-boundary correction for pane-facing UX versus reusable Psionic or
  provider-substrate code
- retrieval-style refusal when the answer depends on current run artifacts,
  current Apple runtime state, or other stale-able evidence

That means the first reviewed real-run corpus no longer teaches only happy-path
architecture answers; it now explicitly teaches the adapter when to correct,
refuse, or avoid overclaiming.

### 5. `Psionic Eval: create the Rust-native eval and rubric runtime`

Status: implemented on 2026-03-14 via GitHub issue `#3568`.

Added the `psionic-eval` crate for:

- held-out eval-run contracts and local eval-run state machines
- rubric-scored sample construction directly from `psionic-environments`
  session summaries
- durable eval summaries with machine-legible aggregate metrics and artifacts
- explicit online/offline parity through one shared sample/runtime contract
- validator-style `BenchmarkPackage` contracts with repeat-run aggregation and
  operator-local validator simulation
- typed verification facts for timer integrity, token accounting, final-state
  capture, and declared execution strategy

The canonical runbook and harness are now:

- `docs/EVAL_RUNTIME_REFERENCE.md`
- `scripts/release/check-psionic-eval-runtime.sh`

Kernel and Nexus still own canonical eval-run authority truth. This issue lands
the reusable Psionic-side runtime and benchmark-contract layer only.

On 2026-03-15, GitHub issue `#3623` extended that same crate with repo-owned
Apple adapter eval harnesses:

- held-out and benchmark scoring over imported Apple dataset fixtures plus
  observed candidate outputs
- explicit structured-output conformance and tool-call coverage checks
- bridge-backed runtime-smoke receipts that prove a `.fmadapter` package parses,
  loads, attaches, and runs against the Apple lane
- typed failure separation between dataset/config problems, package
  incompatibility, and bridge/runtime refusal

On 2026-03-15, GitHub issue `#3653` added the first real-run benchmark and
acceptance gate for the `Psionic architecture explainer` target:

- the benchmark package can now be enriched from the curated corpus so each
  benchmark case carries task-family, expected-behavior, and source-provenance
  metadata
- `psionic-eval` now compares base-model and adapted-model benchmark runs over
  the same cases and emits machine-legible per-case, per-task-family, aggregate,
  and improved-case deltas
- a reproducible acceptance policy now blocks calling a run "real" unless the
  adapted model beats the base model by the declared score, pass-rate, and
  improved-case thresholds
- the frozen architecture-explainer experiment manifest now carries one
  explicit `useful_adapter_gate` contract with:
  - the standard benchmark-improving bar for the normal reference run
  - a weaker `overfit_non_zero` benchmark gate that still requires non-zero
    movement before broader claims are allowed
  - an explicit `runtime_smoke_required` truth bit so reports can distinguish
    `accepted` from `exported_but_not_useful`

The canonical gate for that layer is now:

- historical operator gate in `openagents`:
  `scripts/release/check-psionic-apple-architecture-explainer-benchmark.sh`
- standalone `psionic` repo evidence:
  `fixtures/apple_adapter/runs/psionic_architecture_explainer_reference_overfit_report.json`

On 2026-03-15, GitHub issue `#3656` added the experiment-management layer for
that same first real-run target:

- `psionic-train` now exposes typed Apple adapter experiment manifests,
  checkpoint candidates, selection records, trend ledgers, and regression
  reason codes for the `Psionic architecture explainer` program
- the first reviewed real-run manifest now freezes dataset version, train,
  held-out, and benchmark split digests, benchmark ref, environment ref,
  runtime compatibility anchor, LoRA targets, feature widths, and acceptance
  policy in repo fixtures
- checkpoint choice is now intentional rather than log-only: accepted
  candidates sort ahead of rejected ones, then by benchmark quality and stable
  candidate id
- later iterations can now surface regression explicitly against the best prior
  accepted run instead of silently overwriting "the latest good package"

The canonical experiment-program gate for that layer is now:

- historical operator gate in `openagents`:
  `scripts/release/check-psionic-apple-experiment-program.sh`
- standalone `psionic` repo contract:
  `fixtures/apple_adapter/experiments/psionic_architecture_explainer_reference_overfit_v1.json`

On 2026-03-17, GitHub issue `psionic#2` closed the standalone repo-local
benchmark-effectiveness gap for the bounded Apple reference lane:

- `psionic-train` now exposes one repo-local overfit runner:
  `run_apple_adapter_reference_overfit(...)`
- the runner keeps the machine-legible truth in-repo:
  - the frozen contract lives at
    `fixtures/apple_adapter/experiments/psionic_architecture_explainer_reference_overfit_v1.json`
  - the canonical generator lives at
    `crates/psionic-train/examples/apple_architecture_explainer_reference_overfit.rs`
  - the canonical report lives at
    `fixtures/apple_adapter/runs/psionic_architecture_explainer_reference_overfit_report.json`
- the current report uses the bounded Rust-only Apple backend, the reviewed
  benchmark corpus reused as train plus held-out, manifest-pinned live target
  ids, repo-local negative anchors derived from the reviewed base-output oracle,
  and an explicit `nearest_target_prototype` reference decoder over the learned
  feature space
- the current truthful repo-local result is:
  - benchmark mode `overfit_non_zero`
  - base score `4997` bps
  - adapted score `10000` bps
  - aggregate score delta `5003` bps
  - base pass rate `4285` bps
  - adapted pass rate `10000` bps
  - aggregate pass-rate delta `5715` bps
  - improved case count `4`
- that report now gives this standalone repo one committed, replayable proof
  that the bounded Rust-owned Apple lane can move the benchmark materially off
  the zero floor while still exporting a valid `.fmadapter`
- it does not prove the stronger standard usefulness bar or replace the
  cross-repo live bridge or authority harness; those older `openagents`
  references remain historical context only

On 2026-03-16, GitHub issue `#3900` added the full-path acceptance harness for
that same Rust-only Apple reference lane:

- `apps/autopilot-desktop` now exposes one repo-owned acceptance entrypoint for
  the architecture-explainer lane:
  `run_architecture_explainer_acceptance_harness(...)`
- the harness runs the full path twice:
  - `overfit_non_zero`, which reuses `benchmark.jsonl` as train plus held-out
    and still requires runtime smoke, export completion, and the weak non-zero
    benchmark gate to pass
  - `standard`, which reruns the normal train/held-out/benchmark path and still
    requires runtime smoke, export completion, and the stricter benchmark bar
- the canonical operator entrypoints for that harness are now:
  - `cargo run -p autopilot-desktop --bin apple_architecture_explainer_acceptance_harness -- ...`
  - `scripts/release/check-psionic-apple-architecture-explainer-acceptance.sh`
- the current frozen manifest for that acceptance lane is now:
  - `fixtures/apple_adapter/experiments/psionic_architecture_explainer_acceptance_reference_v2.json`
- the harness always writes a machine-readable acceptance receipt with:
  - top-level `acceptance_passed`
  - stage-specific reports for `overfit_non_zero` and `standard`
  - exact reason codes when a stage fails on launch, runtime smoke, export
    completion, or benchmark usefulness
- the stage-specific report tree is deterministic under the chosen report
  directory:
  - `overfit_non_zero/report.json`
  - `overfit_non_zero/report.md`
  - `overfit_non_zero/overfit_non_zero.fmadapter`
  - `standard/report.json`
  - `standard/report.md`
  - `standard/standard.fmadapter`
- the release script and harness binary both exit non-zero if either stage
  fails, so zero-improvement adapters stop at the acceptance boundary instead of
  being mistaken for complete Apple-lane success

On 2026-03-16, GitHub issue `#3901` updated the operator wording and the
canonical docs to match the exact live acceptance-harness result instead of
letting export/runtime truth imply broader Apple-lane success:

- `autopilotctl training status` and related text output now label the
  authority boundary explicitly as `authority_accept` and `authority_outcome`
  instead of the looser `accept` or `accepted_outcome`
- when Apple operator runs are present, the text output now also prints an
  explicit note that export, runtime smoke, and authority acceptance do not by
  themselves prove benchmark-useful adapter quality
- the latest live acceptance harness receipt on 2026-03-16 was:
  - top-level `acceptance_passed = false`
  - overfit stage run
    `psionic-architecture-explainer-first-real-run-overfit-non-zero-1773694818159`
    passed the weak gate with:
    - aggregate score `520` bps
    - aggregate pass rate `1428` bps
    - improved case count `1`
  - standard stage run
    `psionic-architecture-explainer-first-real-run-standard-1773694877205`
    remained rejected with:
    - aggregate score `571` bps
    - aggregate pass rate `1428` bps
    - improved case count `1`
    - reason codes:
      - `adapter_score_below_minimum`
      - `adapter_pass_rate_below_minimum`
      - `score_delta_below_minimum`
      - `pass_rate_delta_below_minimum`
      - `improved_case_count_below_minimum`
- the resulting claim boundary is now explicit:
  - the Rust-only Apple lane can train, export, load, and runtime-smoke valid
    `.fmadapter` packages
  - the same lane now clears the weak overfit non-zero gate
  - the standard benchmark-useful gate is still not met
  - operator status and authority publication are lifecycle truth, not proof
    that the adapter is already useful in the stronger benchmark sense
- the canonical human-readable companion record for that status update is now:
  - `docs/audits/2026-03-16-psionic-apple-acceptance-harness-status.md`

On 2026-03-16, GitHub issue `#3893` moved that same Apple overfit path off the
flat-zero benchmark floor without weakening the structured or tool contract:

- the Rust-only Apple training backend now adds contrastive target-bank
  alignment plus runtime-derived negative-anchor rejection on top of the older
  pooled reconstruction objective
- Apple eval samples now retain raw expected and observed text in
  machine-legible metadata so benchmark comparison can reason about behavior
  instead of only string identity
- the curated base-vs-adapter benchmark now uses behavior-aware text scoring
  for plain-text benchmark rows keyed by the reviewed corpus annotation:
  - `direct_answer` rows still zero out on policy-refusal or safety-hallucinated
    non-answers
  - `correction` rows require the right `yes` or `no` posture before they can
    score
  - `refusal` rows can now distinguish a grounded "needs current runtime
    validation" answer from a generic "`can't assist`" refusal
- structured-output rows and tool-routing rows remain strict; those still score
  from exact structured conformance and tool-call coverage rather than the new
  text rubric
- the live Apple overfit-non-zero run now clears the frozen weak gate with a
  real adapted benchmark delta:
  - aggregate score `520` bps
  - aggregate pass rate `1428` bps
  - improved case count `1`
  - the improved case is the reviewed stale-evidence refusal row
    `sample-000007`
- that is intentionally not the same thing as saying the Apple lane is now
  broadly useful:
  - the standard benchmark-improving bar is still not met
  - structured summary and tool-routing rows remain unresolved
- multiple plain-text rows still fail outright

Later 2026-03-16 follow-up tuning on that same Rust-only lane tightened both
the trainer and the live benchmark harness again without reverting to Python or
toolkit ownership:

- the Rust-owned Apple backend now uses stronger target-alignment and
  runtime-negative weighting, heavier reviewed-sample weighting, richer exact
  text-signature features, and posture-sensitive answer features for direct
  answer, correction, tool, schema, and stale-evidence refusal rows
- the live benchmark harness now uses a raw-JSON fallback prompt for schema
  rows and a retry path that feeds concrete suggested repo paths back into tool
  rows after model-request failures, so structured/tool benchmark failures are
  less contaminated by avoidable harness behavior
- the honest result is still not "useful adapter achieved":
  - local overfit reruns can now reach materially higher aggregate score than
    the earlier flat-zero floor while still exporting, loading, and runtime
    smoking valid `.fmadapter` packages
  - but the useful-adapter gate is still not met because the adapted path still
    does not produce a durable pass-rate improvement over base
  - the remaining stubborn benchmark rows are still the kernel-authority direct
    answer, the structured lane-summary JSON row, and the exact tool-routing
    ownership-path row

On 2026-03-16, GitHub issue `#3894` closed the manifest-to-live parity gap for
that same first reviewed Apple lane:

- the frozen architecture-explainer manifest fixture is now pinned to the
  actual live exportable lane instead of claiming a broader geometry or target
  family:
  - symbolic targets: `decoder.attn.q_proj`
  - feature width: `2048x2048`
  - LoRA rank: `32`
- the operator no longer silently narrows unsupported manifest requests:
  - unsupported symbolic target families such as `decoder.ffn.up_proj` now
    fail before training with an explicit contract error
  - geometry or rank mismatches now fail before training with an explicit
    live-lane requirement error
- operator receipts and lineage metadata now record both the requested and the
  executed target families plus geometry, so reports show exactly what the run
  asked for and what the live lane actually executed
- the current truthful overfit report at
  `psionic-architecture-explainer-first-real-run-1773687000518` shows those
  requested and executed fields matching exactly for the frozen manifest

On 2026-03-16, GitHub issue `#3895` made the Apple training policy explicit,
inspectable, and field-sweepable instead of leaving optimizer behavior as an
implicit operator default:

- `psionic-train` experiment manifests can now carry an explicit
  `training_policy` block that freezes:
  - optimizer family plus optimizer-family fields such as `learning_rate`,
    `weight_decay`, `beta1`, `beta2`, `epsilon`, and optional
    `gradient_clip_norm`
  - optimizer residency posture
  - optional scheduler binding
  - precision policy
  - activation-checkpoint policy
  - packing policy
  - `max_steps`
  - `gradient_accumulation_steps`
- manifest validation now rejects obviously invalid policy shapes before launch,
  including:
  - `learning_rate <= 0`
  - `gradient_accumulation_steps == 0`
  - invalid packing-policy windows
- the current Rust-native Apple lane remains truthful about its live limit:
  `gradient_accumulation_steps` must still be `1`, and other values fail before
  training instead of being silently approximated
- `autopilotctl training launch ...` and
  `apple-architecture-explainer-reference-run ...` now both accept an optional
  `training_policy_override_path`, which applies field-by-field CLI overrides
  on top of the frozen manifest policy rather than replacing the whole policy
  blob
- operator-local summaries and Apple lineage metadata now persist the fully
  resolved training policy with per-field source attribution:
  - `repo_default`
  - `experiment_manifest`
  - `cli_override`
- the override-backed proof run
  `psionic-architecture-explainer-first-real-run-1773690239110` demonstrates
  the intended behavior:
  - optimizer family, residency, scheduler, precision, activation policy, and
    `gradient_accumulation_steps` remained `experiment_manifest` sourced
  - only `learning_rate`, `weight_decay`, `max_steps`, and the widened packing
    window were `cli_override` sourced
  - the run exported, loaded, and runtime-smoked successfully while keeping the
    sourced policy block visible in the resulting receipt

On 2026-03-16, GitHub issue `#3896` fixed the structured-output benchmark
contract so structured rows are judged on structured truth first instead of
being vulnerable to harness-format noise:

- Apple structured benchmark rows now canonicalize JSON semantics rather than
  requiring byte-for-byte raw text identity for the emitted JSON string
- the eval harness now accepts semantically equal structured payloads even when
  the raw JSON field order or whitespace differs
- when a structured request fails at the harness or bridge contract layer, the
  observed sample now carries explicit structured-contract metadata instead of
  forcing everything into a generic text-mismatch bucket
- benchmark failure reasons now distinguish:
  - `harness_contract_failure:structured_generation:*`
  - `runtime_failure:*`
  - true `model_output_mismatch:structured`
- this keeps the architecture-explainer structured row truthful:
  - if the model emits the wrong JSON values, it still fails as a model error
  - if the bridge or schema path is the problem, the receipt now says so

On 2026-03-16, GitHub issue `#3897` fixed the Apple repo-lookup tool contract
so tool benchmark rows are no longer suppressed by avoidable harness aborts:

- repo lookup tools now expose a tighter path contract directly in their schema
  surface:
  - repo-relative concrete file paths only
  - no directories
  - no globs
  - no absolute paths
  - no `..` traversal
- model-request mistakes on the lookup tools are now recoverable tool results
  instead of hard tool exceptions that abort the whole Apple FM turn:
  - directory requests
  - glob requests
  - wrong lookup-kind requests
  - other invalid-path proposals
- recoverable repo-lookup responses now carry retry guidance, suggested tool
  family, and suggested repo-relative paths, and the same details are persisted
  in per-sample repo-lookup metadata
- the eval harness now distinguishes the benchmark categories the issue required:
  - `model_behavior:tool_not_chosen:*`
  - `model_behavior:wrong_tool_chosen:*`
  - `model_behavior:invalid_path_proposed:*`
  - `harness_failure:*`
  - `model_behavior:true_execution_failure:*`
- this keeps the tool benchmark honest:
  - avoidable path-policy mistakes no longer look like a generic runtime crash
  - truly missing or wrong tool behavior still remains visible as model failure

On 2026-03-15, GitHub issue `#3657` tightened the Apple runtime-validation
layer around that same run:

- runtime-smoke receipts now carry the bridge-reported compatibility snapshot
  used during validation, including the current base-model family anchor,
  bridge version/platform, availability state, and adapter inventory or attach
  capability posture
- bridge-backed runtime validation now fails with explicit reasons when the
  Apple runtime is unavailable, when adapter inventory or attach support is
  missing, or when the bridge rejects the package
- `autopilotctl training accept ...` now reruns a drift check against the live
  Apple bridge before publishing authority truth, so runtime or Background
  Assets drift is surfaced explicitly instead of being hidden behind a stale
  earlier smoke receipt

The canonical runtime-validation gate for that layer is now:

- `scripts/release/check-psionic-apple-runtime-validation.sh`

### 6. `Psionic Train: define canonical run graph, topology revisions, and participant lifecycle`

Status: implemented on 2026-03-14 via GitHub issue `#3569`.

Added run-graph contracts inside `psionic-train` for:

- stable run ids, stage ids, topology revisions, contributor-set revisions, and
  `TrainingWindow` ids
- explicit participant admission, readiness, contribution, departure, and
  suspension state
- persistent participant ranking and deterministic contributor reselection
- heartbeat, departure, rejoin, and contributor-suspension lifecycle events
- replay-safe window planning with deterministic batch/eval slice assignment
- machine-legible window transitions through `planned`, `active`, `sealed`,
  `scored`, and `reconciled`

The canonical runbook and harness are now:

- `docs/TRAIN_RUN_GRAPH_REFERENCE.md`
- `scripts/release/check-psionic-train-run-graph.sh`

This issue makes the run graph and participant lifecycle explicit typed Psionic
truth instead of a scheduler convention. It does not yet land full
orchestrator, checkpoint-pointer, or batch-propagation policy.

### 7. `Psionic Train: extend checkpoint lineage, recovery modes, and catch-up receipts`

Status: implemented on 2026-03-14 via GitHub issue `#3570`.

Added checkpoint-lineage and restore-ladder contracts inside `psionic-train`
for:

- typed `CheckpointPointer` and `CheckpointManifest` objects over explicit run,
  stage, or window scope
- explicit durability posture on checkpoint manifests, including partial-upload
  versus durable restore eligibility
- declared `TrainingRecoveryMode` choices for blocking catch-up, overlapped
  catch-up, and resume-from-last-stable-checkpoint
- pointer-first restore planning with manifest-listing fallback when the latest
  pointer is missing, stale, or references non-durable state
- deterministic shard-uploader assignment over the accepted restore manifest
- fake object-store tests covering missing pointer, stale pointer,
  partial-upload, and listing-limit failure paths

The canonical runbook and harness are now:

- `docs/TRAIN_CHECKPOINT_RECOVERY_REFERENCE.md`
- `scripts/release/check-psionic-train-checkpoint-recovery.sh`

This issue turns checkpoint recovery from implicit latest-checkpoint heuristics
into typed restore receipts that can explain why one source was preferred over
another. It does not yet land retention policy, cold-restore classes, or
cross-window checkpoint governance.

### 8. `Psionic Collectives: add bandwidth-aware elastic sync planning and quantized policy surfaces`

Status: implemented on 2026-03-14 via GitHub issue `#3571`.

Added sync-planning and cadence-policy contracts inside `psionic-collectives`
for:

- mesh-wide `CollectiveTransportFeedback` observations with stable digests and
  explicit bandwidth, latency, and stream-pressure metrics
- `CollectiveSyncCadencePolicy` over healthy and degraded global-sync
  intervals, transport thresholds, and local/global quantization posture
- `CollectiveSyncExecutionPlan` and `CollectiveSyncStage` so local subgroup
  sync and full-mesh sync are planned as explicit ordered stages
- `CollectiveSyncCadenceReceipt` and `CollectiveReplanTrigger` so cadence,
  transport degradation, quantization fallback, and interval-elapse decisions
  stay machine-legible
- planner-owned local-group selection, interval-based deferred global sync, and
  mesh-revision replanning over the existing benchmark-gated quantized
  collective substrate

The canonical runbook and harness are now:

- `docs/COLLECTIVE_SYNC_POLICY_REFERENCE.md`
- `scripts/release/check-psionic-collective-sync.sh`

This issue makes collective sync cadence explicit Psionic truth instead of a
hidden optimizer-side heuristic. It does not yet land distributed optimizer
state integration or parameter-shard accounting.

### 9. `Psionic Datastream: add sharded policy-weight broadcast and freshness control`

Status: implemented on 2026-03-14 via GitHub issue `#3572`.

Added policy-weight broadcast contracts inside `psionic-datastream` for:

- explicit `PolicyWeights` subject identity plus `DatastreamPolicyWeightBinding`
  over policy id, revision, shard identity, assembled-artifact digest, and
  freshness window
- lightweight `DatastreamPolicyWeightControlPlaneRef` and
  `DatastreamPolicyWeightBroadcastManifest` objects so orchestrators can carry
  refs, digests, and mirror metadata instead of heavy payload bytes
- mirror or relay metadata through `DatastreamMirrorLocator`
- stale-artifact rejection at control-plane-ref export time
- `InMemoryPolicyWeightBroadcast` and
  `DatastreamPolicyWeightBroadcastReceipt` for pipelined multi-shard delivery
  over the existing resumable chunk path
- tests proving the control-plane summary stays smaller than the heavy artifact
  bytes while the heavy artifact plane remains resumable and byte-accountable

The canonical runbook and harness are now:

- `docs/POLICY_WEIGHT_BROADCAST_REFERENCE.md`
- `scripts/release/check-psionic-policy-weight-broadcast.sh`

This issue makes the heavy artifact plane versus lightweight control plane
split explicit for policy weights. It does not yet land orchestrator-owned
assignment or rollout freshness budgets.

### 10. `Psionic Train: build the orchestrator state machine and trainer-batch assembly contracts`

Status: implemented on 2026-03-14 via GitHub issue `#3573`.

Added the first orchestrator module inside `psionic-train` for:

- typed `TrainingOrchestratorState` over the existing run graph, target policy
  revision, and lightweight policy-weight broadcast manifest
- orchestrator ownership of contributor selection, window planning, window
  activation, sealing, scoring, and reconciliation transitions
- deterministic `TrainingWindowAssignmentPosture` carrying assignment seed,
  policy revision id, and weight-broadcast digest
- lightweight rollout and sampled-eval assignments that exchange only ids,
  digests, policy ids, and weight-broadcast refs
- lightweight `RolloutArtifactRef` and `TrainerBatchAssemblyRequest` contracts
  so trainer-batch assembly stays control-plane-safe while still composing with
  full `RolloutArtifact` and `TrainerBatch` substrate
- replay-safe tests proving admitted participants, contributing participants,
  and resulting trainer batches can all differ in one orchestrated window

The canonical runbook and harness are now:

- `docs/TRAIN_ORCHESTRATOR_REFERENCE.md`
- `scripts/release/check-psionic-train-orchestrator.sh`

This issue makes the orchestrator a first-class Psionic control plane instead
of a loose pile of helpers around the run graph. It does not yet land
off-policy pruning or worker protocol completion.

### 11. `Psionic Train: implement off-policy budget rules and stale-rollout pruning`

Status: implemented on 2026-03-14 via GitHub issue `#3574`.

Added bounded rollout-admission contracts inside `psionic-train` for:

- explicit `TrainingOffPolicyBudget` policy over revision drift, policy age,
  rollout age, and quarantine thresholds
- typed `RolloutAdmissionReceipt` outcomes for accepted exact, accepted
  off-policy, quarantined, and discarded rollouts
- machine-readable `RolloutAdmissionSignal` reason codes so freshness and drift
  violations stay inspectable rather than log-only
- per-window `RolloutIngestionTelemetry` and retained quarantined-versus-
  discarded rollout state on the orchestrator
- replay-safe tests proving exact acceptance, bounded off-policy acceptance,
  quarantine outside direct-accept budgets, and hard discard beyond quarantine
  budgets

The canonical runbook and harness are now:

- `docs/TRAIN_OFF_POLICY_BUDGET_REFERENCE.md`
- `scripts/release/check-psionic-train-off-policy-budget.sh`

This issue makes stale-rollout accounting first-class train control-plane truth
instead of a batch-filtering convention. Worker claim protocol completion and
validator-owned rollout adjudication now live in the follow-on records for
issues `#3575` and `#3576`.

### 12. `Psionic Train: define the inference-worker protocol for trustless rollout generation`

Status: implemented on 2026-03-14 via GitHub issue `#3575`.

Added rollout-worker protocol contracts inside `psionic-train` for:

- explicit `RolloutWorkerTrustClass` and `RolloutWorkerIdentity` so trusted
  trainer nodes are protocol-distinct from semi-trusted or untrusted rollout
  workers
- `RolloutWorkerHeartbeatReceipt` and `RolloutTaskClaim` over heartbeat
  freshness, claim TTL, deterministic sample-selection seed, and assignment
  binding
- `RolloutUploadLocator` and upload-policy enforcement for inline versus
  external artifact delivery
- `RolloutWorkerOutcomeReceipt` that wraps local claim-expiry or upload-policy
  outcomes plus orchestrator-provided rollout-admission receipts
- replay-safe tests proving fresh-heartbeat claims, bounded off-policy upload
  handling, and local receipts for expired claims or oversized uploads

The canonical runbook and harness are now:

- `docs/TRAIN_ROLLOUT_WORKER_PROTOCOL_REFERENCE.md`
- `scripts/release/check-psionic-train-rollout-worker-protocol.sh`

This issue makes rollout-worker coordination a first-class typed protocol
inside Psionic instead of a trainer-local convention. Validator-owned rollout
verification and sampled adjudication now live in the follow-on record for
issue `#3576`.

### 13. `Validator Service: add rollout-verification bundles and sampled adjudication protocols`

Status: implemented on 2026-03-14 via GitHub issue `#3576`.

Added rollout-validation contracts inside `psionic-train` for:

- `RolloutVerificationBundle` over one rollout artifact, worker outcome, and
  optional benchmark observation or expectation
- `RolloutValidatorPolicy` with execution-proof requirements, deterministic
  sampled expensive-check posture, benchmark-check posture, and duplicate
  normalization policy
- `ValidatorVerdict` with typed replay-detected, duplicate-detected,
  stale-policy-rejected, contribution-normalized, timer-integrity,
  token-accounting, final-state, and execution-strategy reason codes
- stateful replay and duplicate detection through artifact-digest and
  response-signature history
- benchmark-gated sampled adjudication for timer, token, final-state, and
  declared-execution-strategy checks

The canonical runbook and harness are now:

- `docs/TRAIN_ROLLOUT_VALIDATION_REFERENCE.md`
- `scripts/release/check-psionic-train-rollout-validation.sh`

This issue makes validator-ready rollout integrity first-class typed Psionic
truth. Broader external validator services, batch-level verdicts, and authority
integration are still later layers.

On 2026-04-01, GitHub issue `#816` added the first bounded live RL update
bridge on top of that rollout and validator substrate:

- `OpenAdapterLiveRlUpdateExecutor` over the repo-owned open-adapter lane
- typed `LiveRlMaterializedBatch`, `LiveRlMaterializedRollout`, and
  `LiveRlMaterializedToken` records that keep prompt/completion boundaries,
  observed-versus-live logprobs, importance ratios, reward, advantage, and
  optional chosen-token teacher logprobs explicit
- one live update receipt plus one promoted adapter revision ready for
  `TrainingSamplerService::refresh_revision`
- a focused end-to-end harness proving admitted rollout batch -> weighted
  trainer step -> promoted revision

The canonical runbook and harness are now:

- `docs/TRAIN_LIVE_RL_UPDATE_REFERENCE.md`
- `scripts/release/check-psionic-train-live-rl-update.sh`

This closes the earlier gap between typed RL control-plane truth and one honest
live adapter update path. The teacher input is still a bounded chosen-token
auxiliary term, not a full-distribution distillation runtime.

On 2026-04-01, GitHub issue `#813` added the first durable live RL run service
above that substrate:

- `LiveRlRunService` over `TrainingRunState`, `TrainingOrchestratorState`,
  `RolloutWorkerProtocolState`, and `RolloutValidatorState`
- durable `state.json`, `status.json`, per-window status artifacts, and
  failure artifacts under a service-owned run root
- run creation, graceful draining, and terminal stop semantics
- worker heartbeat, claim, upload, validator-ingestion, and trainer-batch flow
  against live service state rather than harness-only in-memory state
- restart recovery by loading persisted state back into the service

The canonical runbook and harness are now:

- `docs/TRAIN_LIVE_RL_RUN_SERVICE_REFERENCE.md`
- `scripts/release/check-psionic-train-live-rl-run-service.sh`

This closes the durability gap above the existing RL control-plane substrate.
It is still an in-process bounded service. Automatic environment execution,
sampler-owned rollout generation, and automatic promoted-revision adoption
remain explicit follow-on layers above it.

### 14. `Environments: define a package contract for SFT, RL, and eval`

Status: implemented on 2026-03-14 via GitHub issue `#3577`.

Added package-shape contracts inside `psionic-environments` for:

- `EnvironmentWorkloadClass` so one package can declare SFT, RL, online-eval,
  offline-eval, and validator-benchmark use explicitly
- typed `EnvironmentPolicyReference` and `EnvironmentDifficultyMetadata`
  instead of burying those semantics in free-form metadata
- `EnvironmentBenchmarkProfile` for validator-owned benchmark identity,
  runtime-profile identity, verification posture, and declared
  execution-strategy expectations
- package validation and digest coverage for workload classes, policy refs,
  difficulty metadata, and benchmark profiles
- replay-safe tests proving one package can carry both ordinary environment
  execution contracts and a reusable benchmark profile

The canonical runbook and harness are now:

- `docs/ENVIRONMENT_PACKAGE_CONTRACT_REFERENCE.md`
- `scripts/release/check-psionic-environment-package-contract.sh`

This issue makes environment packages composable across training, eval, and
validator-local simulation instead of relying on raw metadata blobs or hidden
side settings. Registry install and composition flows remain the next issue.

### 15. `Environments Registry: add package install, version pinning, composition, and eval parity`

Status: implemented on 2026-03-14 via GitHub issue `#3578`.

Added the first Psionic-native registry and composition layer inside
`psionic-environments`:

- typed install requests and install receipts for versioned environment package
  materialization
- digest-bound pin aliases so train and eval code resolve immutable package
  versions instead of floating refs
- mixed-surface composition groups and group-member contracts across train,
  eval, and benchmark surfaces
- dependency-aware group resolution and benchmark-profile validation
- explicit train/eval parity receipts for shared group members

The canonical runbook and harness are now:

- `docs/ENVIRONMENT_REGISTRY_REFERENCE.md`
- `scripts/release/check-psionic-environment-registry.sh`

This issue removes the need for bespoke environment-mix glue in the
orchestrator for the first train/eval/benchmark package groups. Persistent
authority sync, package publication, and richer eval-policy productization
remain later layers.

On 2026-03-15, GitHub issue `#3622` added the first repo-owned Apple adapter
specialization on top of that registry substrate: one helper now materializes
the shared Apple core package, benchmark package, mixed-surface group, and the
train/eval parity receipt together so later train and eval layers do not have
to rebuild Apple environment wiring from app-local config.

### 16. `Psionic Sandbox: add RL-throughput primitives for pooled, repeated agentic execution`

Status: implemented on 2026-03-14 via GitHub issue `#3579`.

Added the first RL-throughput sandbox control plane inside `psionic-sandbox`:

- typed warm-pool specs, snapshots, warm receipts, and acquisition receipts
- staged-input receipts for command inputs, image frames, and context artifacts
- repeated bounded loop execution on the same acquired workspace
- explicit reuse accounting so pool health and acquisition latency are visible
  to later train or operator layers

The canonical runbook and harness are now:

- `docs/SANDBOX_RL_THROUGHPUT_REFERENCE.md`
- `scripts/release/check-psionic-sandbox-rl-throughput.sh`

This issue makes the sandbox layer usable for RL-style short-lived environment
actions without forcing one bespoke background-job flow per environment.
Distributed pool management and higher-level train scheduling still remain
later layers.

### 17. `Psionic Train: add SFT trace ingestion, stage transitions, and agentic pre-RL flows`

Status: implemented on 2026-03-14 via GitHub issue `#3580`.

Added the first multi-stage train-program layer inside `psionic-train`:

- typed `TrainingStageKind` identity for `general_sft`, `agentic_sft`, and `rl`
- typed SFT trace artifacts with tool-call and long-context lineage
- stage completion receipts, checkpoint-promotion receipts, and stage-transition
  receipts
- a stage-program state machine that owns `general_sft -> agentic_sft -> rl`
  sequencing

The canonical runbook and harness are now:

- `docs/TRAIN_STAGE_PROGRAM_REFERENCE.md`
- `scripts/release/check-psionic-train-stage-program.sh`

This issue makes stage sequencing first-class Psionic truth instead of operator
glue. Curriculum, filtering, and instability policy remain the next train
issues.

### 18. `Psionic Train: implement curriculum, filtering, and non-zero-advantage gates`

Status: implemented on 2026-03-14 via GitHub issue `#3581`.

Added the first train-side curriculum controller inside `psionic-train`:

- digest-bound curriculum policy with online and offline sampling filters
- typed training candidates constructed from SFT traces and rollout artifacts
- explicit filter receipts and batch selection receipts
- difficulty-tier consumption, trivial-reward suppression, source-budget
  suppression, and non-zero-advantage gates

The canonical runbook and harness are now:

- `docs/TRAIN_CURRICULUM_REFERENCE.md`
- `scripts/release/check-psionic-train-curriculum.sh`

This issue makes training-sample selection inspectable and reproducible.
Instability telemetry and halt policy remain the next train issue.

### 19. `Psionic Train: add instability telemetry, halt policies, and risky-optimization gating`

Status: implemented on 2026-03-14 via GitHub issue `#3582`.

Added the first train-safety controller inside `psionic-train`:

- aggregated instability telemetry over gradient norms, clipping ratios, and
  rollout-drop rate, with explicit extension points for entropy drift,
  checkpoint catch-up latency, topology churn, and failure rates
- digest-bound threshold rules that map signals to `continue`, `quarantine`, or
  `halt`
- explicit risky-optimization rules so dangerous runtime shortcuts are policy,
  not hidden flags
- final typed verdicts carrying both signal receipts and optimization receipts

The canonical runbook and harness are now:

- `docs/TRAIN_STABILITY_REFERENCE.md`
- `scripts/release/check-psionic-train-stability.sh`

This issue makes halt/quarantine policy machine-legible. Broader operator
product surfaces beyond the current app-owned Apple workflow remain later
layers.

### 20. `Kernel and Nexus: add training and eval receipt families, policy registries, and read models`

Once train and eval become economic or productized objects, their outcomes
need authority-facing truth. This issue should add durable receipt families,
read models, and policy registries for environment packages, checkpoint
families, validator posture, and accepted train or eval outcomes. It is the
bridge from Psionic-local execution truth into higher-level OpenAgents market
or authority truth. It should also prefer typed Rust client and payload-builder
surfaces for those train, eval, and validator-facing authority contracts rather
than ad hoc JSON glue.

Status: implemented on 2026-03-14 via GitHub issue `#3583`.

The canonical authority docs are now:

- `docs/kernel/compute-evaluation-runs.md`
- `docs/kernel/compute-training-authority.md`

The generated or typed authority path now exists in `openagents-kernel-core`
and `apps/nexus-control` for:

- checkpoint-family policy registry
- validator-policy registry
- benchmark-package registry
- training-policy registry
- training-run create/finalize/list/get
- accepted eval or training outcomes

On 2026-03-15, GitHub issue `#3624` extended the same authority surface with
Apple-specific benchmark adapter kinds plus typed training-policy and
training-run metadata validation, so Apple packages and runs now have to carry
consistent benchmark, validator, and environment bindings before Nexus accepts
them.

On 2026-03-15, GitHub issue `#3627` then extended Nexus with the first
canonical Apple training-run and accepted-outcome projection path, including
held-out eval gating, optional runtime-smoke gating, and persistence of typed
Apple package lineage on both the finalized training run and accepted outcome.

### 21. `Desktop and autopilotctl: expose training operator surfaces and diagnostics`

Implemented on Sunday, March 15, 2026 via GitHub issue `#3629`, after the
earlier Apple adapter inventory/control plumbing from issue `#3620`.

The app-owned desktop-control surface and `autopilotctl` now expose a typed
training operator view. The current projection is intentionally truthful about
what is authority-backed versus what is not yet wired from a live train
controller:

- authority-backed training runs and accepted outcomes are loaded into the
  desktop-control compute-history cache alongside proof and challenge truth
- the snapshot now exposes a dedicated `training` domain with explicit
  `control_plane_state` versus `artifact_plane_state`
- operator output includes environment versions, checkpoint refs,
  contributor-set revision hints, contributor reselection timing, stale-rollout
  discard counts, duplicate quarantine or deweight counts, validator verdict
  totals, sandbox pool readiness, and visible run-level diagnostics
- the same surface now carries an app-owned Apple operator sub-view with
  explicit `launch`, `evaluation`, `export`, and `acceptance` stage state plus
  persisted run logs and authority refs
- `autopilotctl training launch`, `training export`, and `training accept`
  drive the same desktop-control mutations instead of relying on ad hoc scripts
- `autopilotctl training status` prints the same app-owned projection directly,
  while `autopilotctl status` includes a condensed training summary

This does not claim a live Psionic train orchestrator is embedded in the
desktop app yet. It does make the currently available training truth
inspectable without reconstructing it from logs or ad hoc scripts.

On 2026-03-15, GitHub issue `#3628` projected the same accepted Apple adapter
truth into `openagents-provider-substrate` as a narrow provider-hosted
adapter-family capability, and GitHub issue `#3630` updated the compute-market
docs to match that narrow truth boundary. Those surfaces sit above Psionic and
do not yet imply a broad training procurement product.

### 22. `Reference Program: run one end-to-end agentic SFT plus RL pilot on the full stack`

Implemented on Saturday, March 14, 2026.

`psionic-train` now ships a typed reference-program runner in
`src/reference_program.rs` plus the runnable harness
`scripts/release/check-psionic-agentic-sft-rl-reference-program.sh`.

The pilot intentionally crosses the currently implemented Rust-owned stack
instead of claiming completion from isolated subsystem tests:

- one versioned weather-agent environment package is reused across SFT, RL,
  online eval, and benchmark-mode eval
- dataset lineage remains explicit through environment bindings, trace source
  refs, and eval contracts
- stage-program lineage crosses `general_sft -> agentic_sft -> rl` with
  explicit checkpoint-promotion receipts
- policy weights are delivered through `psionic-datastream` broadcast receipts
- sandbox warm-pool reuse is proven through staged-input and iteration receipts
- rollout-worker heartbeat, claim, upload, and outcome receipts run against the
  real train orchestrator state
- validator-aware adjudication emits typed verdicts over rollout bundles
- benchmark aggregation and online eval both remain machine-legible
- the trainer step consumes the orchestrator-produced trainer batch rather than
  a disconnected toy batch
- the final report includes a condensed operator view without discarding the
  underlying typed receipts, lineage, and summaries

This is the current main integration gate for the early train stack. It does
not claim that replay guarantees, security hardening, artifact lifecycle, or
research-loop layers are complete, and it does not turn the landed
distributed-optimizer or model-IO contracts into proof that the full
multi-device runtime is complete.

### Production Completion And Hardening

### 23. `Psionic Train: define distributed optimizer, precision, and memory-sharding contracts`

Implemented on Saturday, March 14, 2026.

`psionic-train` now owns an explicit distributed-optimizer layer in
`src/distributed_optimizer.rs` on top of the existing fixed-budget core.

The new contract makes all of the following first-class:

- distributed optimizer family selection
- parameter sharding per group
- gradient-buffer sharding per group
- optimizer-state sharding plus residency
- master-weight residency
- precision policy across parameter, gradient, optimizer-state, master-weight,
  and reduction paths
- activation checkpointing or rematerialization policy
- long-run host/device memory budgeting and derived memory-plan receipts
- microbatch accumulation and flush discipline
- collective sync-plan attachment to the optimizer contract itself

The runtime wrapper is still intentionally bounded. It buffers microbatches,
refuses incomplete flushes, derives an explicit memory plan, and then flushes
one accumulated step through the existing fixed-budget trainer core while
preserving the higher-level distributed receipt.

This does not claim that the full multi-device runtime already exists. It does
mean the distributed optimizer, precision, and memory-sharding model is now
typed and inspectable instead of implied by future plans.

The distributed layer now composes with the reusable optimizer surface in
`src/optimizer.rs`, so local optimizer-family step semantics are inspectable
without being trapped inside one trainer implementation.

### 24. `Model IO: add Rust-native checkpoint, tokenizer, and model-format interoperability`

Implemented on Saturday, March 14, 2026.

`psionic-train` now owns a typed model-IO portability layer in
`src/model_io.rs`.

The new layer makes these train-to-serve seams explicit:

- named state-dict traversal and assignment contracts
- portable training-group reconstruction from state-dict artifacts
- machine-readable compatibility boundaries for Psionic-native state dicts,
  manifest-carrying safetensors, typed JSON state dicts, GGUF import, and
  intentionally unsupported opaque checkpoint families
- tokenizer family, digest, special-token, and version binding
- dense safetensors export and import with embedded Psionic manifest metadata
- JSON torch-style state-dict artifacts for Rust-native portability
- GGUF import with tensor inventory, tokenizer binding, and chat-template
  digest extraction
- additive adapter merge and unmerge over parameter tensors

The scope is still intentionally bounded. The current torch-compatible surface
is typed JSON rather than opaque Python checkpoint loading, and GGUF support is
currently import-focused rather than full re-export. That is still a material
shift: trained or served artifacts are now portable through one Rust-owned
contract instead of bespoke scripts or disconnected side files.

General-purpose array artifact IO now lives separately in `psionic-array-io`,
and general-purpose native function artifact IO now lives separately in
`psionic-function-io`. That split is intentional: `psionic-train::model_io`
still owns checkpoint, tokenizer, state-dict, and model-family portability,
while `psionic-array-io` owns public framework-facing `npy` / `npz` /
`safetensors` plus bounded GGUF array save/load semantics above the lazy-array
surface, and `psionic-function-io` owns public `.psifn` export/import
artifacts plus a bounded `.mlxfn` compatibility shell above export-safe graph
and compiler contracts instead of burying that boundary inside train-local
packaging code; `psionic-array` now also owns bounded public runtime memory
reporting with active/peak/cache counters plus explicit cache-limit and reset
controls above the reference eval substrate.

### 25. `Training Truth: add deterministic replay and reproducibility guarantees`

Implemented on Saturday, March 14, 2026.

`psionic-train` now owns a deterministic replay-truth layer in
`src/replay_truth.rs`.

The new contract makes these reproducibility seams explicit:

- assignment, trainer, and eval seed discipline
- deterministic sample-selection rules with stable worker and attempt identity
- replayable trainer-batch anchoring
- pinned environment package and tool contracts
- pinned tool-version labels
- reproducible eval posture with deterministic scheduler enforcement
- typed replay-verification receipts and drift signals

`PLIB-212` / `#3727` extends that foundation by publishing one
machine-readable `ReproducibilitySemanticsReport` in `src/replay_truth.rs`.
That report binds assignment, trainer, and eval seeds into runtime
determinism contracts, proves stable local-device and distributed-rank
generator derivation, proves checkpoint-stable RNG restore, and carries typed
refusal for missing strict generators or invalid distributed-rank bounds.

`PLIB-213` / `#3728` now complements that replay truth with one bounded
autocast-style `AutocastPolicyMatrixReport` in `psionic-core`. The report
keeps backend-aware low-precision policy, stability-preserving no-downcast
rules, float8 meta-only posture, and explicit unsupported mixed-precision
requests machine-legible before train-class grad scaling lands.

`PLIB-214` / `#3729` now lands that bounded train-class grad-scaling layer in
`psionic-train::mixed_precision`. The new
`GradientScalingSemanticsReport` makes fp16 dynamic loss scaling, overflow
backoff plus step-skip, underflow-driven scale growth, bf16 no-scaling
posture, and unsupported mixed-precision refusal machine-legible instead of
burying those decisions inside one trainer loop.

`PLIB-215` / `#3730` now lands one bounded
`QuantizationCapabilitySemanticsReport` in `psionic-core`. That report
separates PTQ, QAT, quantized runtime execution, compiler-lowering posture,
and export-aware quantization intent above raw decode so train- and
deployment-class quantization claims stop collapsing into "the loader can read
GGUF."

`PLIB-218` / `#3733` now lands one bounded `DataIngressSemanticsReport` in
`psionic-data`. That report makes local dataset source, iterable-streaming,
sampler, batch-sampler, and host-device staging contracts machine-legible
instead of leaving them as train-loop glue.

`PLIB-219` / `#3734` now layers one bounded
`DistributedDataFeedSemanticsReport` on top of that local ingress surface in
`psionic-data`. The new report makes fixed-world-size shard partitioning,
epoch-barrier or step-barrier worker coordination, and runtime-derived
replay-safe per-rank ordering machine-legible, while explicitly refusing
elastic membership until a higher-level distributed run-control contract lands.

This is still not the claim that the full train system can be re-executed from
one receipt without more runtime work. It is the claim that replay-compatible
inputs, pins, and verification are now explicit enough to support "same
receipt, same recomputation rules" instead of best-effort repeatability.

### 26. `Security: harden environment packages, artifact provenance, and untrusted worker admission`

Implemented on Saturday, March 14, 2026.

`psionic-train` now owns a train-security posture layer in
`src/security_posture.rs`.

The new contract makes these hardening seams explicit:

- environment package identity and digest verification
- required environment verification and safety policy references
- artifact signing contracts plus trust roots
- minimum signature counts for admitted artifacts
- untrusted-worker rate limits and burst controls
- required execution-proof posture for untrusted workers
- duplicate-artifact rejection and duplicate-response-signature quarantine
- validator-bound security receipts with typed reason codes

This does not replace the validator loop. It does connect rollout validation to
the broader train security posture instead of leaving environment trust,
artifact provenance, and untrusted-worker admission as implicit assumptions.

### 27. `Artifact Storage: define retention, garbage collection, archival, and cold-restore policy`

Status: implemented on 2026-03-14 via GitHub issue `#3590`.

`psionic-train` now owns an explicit artifact-storage lifecycle layer in
`src/artifact_storage.rs`.

The new contract makes these storage seams explicit:

- per-artifact-class retention profiles with hot and warm thresholds
- archive classes for ephemeral, restorable, and immutable artifacts
- digest-aware deduplication for rollout or other repeatable artifact classes
- typed records for checkpoint, rollout, eval, and log bundle artifacts
- explicit sweep receipts for warm migration, archival, deduplication, and
  garbage collection
- cold-restore request and completion receipts bound to restore objectives

The canonical runbook and harness are now:

- `docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md`
- `scripts/release/check-psionic-train-artifact-storage.sh`

This issue makes train artifact retention part of typed Psionic truth instead
of operator-local scripts. Scheduler budgeting, queue preemption, and broader
economic accounting remain the next layer.

### 28. `Scheduling and Accounting: add budget, priority, preemption, and cost attribution`

Status: implemented on 2026-03-14 via GitHub issue `#3591`.

`psionic-train` now owns an explicit scheduling and accounting layer in
`src/scheduling_accounting.rs`, and `psionic-runtime` now surfaces train-owned
runtime work classes for trainer, rollout, eval, sandbox, and validator work.

The new contract makes these operator seams explicit:

- global active-work budget caps over work units, bytes, and estimated cost
- queue classes with inspectable priority and preemption policy
- role-specific cost rates for trainer, rollout, eval, sandbox, and validator
  work
- typed admission, preemption, queueing, completion, and snapshot receipts
- validator-scoped and environment-scoped cost attribution
- queue draining after completion so queued work becomes active through typed
  state transitions rather than implicit retries

The canonical runbook and harness are now:

- `docs/TRAIN_SCHEDULING_ACCOUNTING_REFERENCE.md`
- `scripts/release/check-psionic-train-scheduling-accounting.sh`

This issue makes train-side operator economics first-class typed Psionic truth.
Chaos testing and benchmark thresholds remain the final follow-on issues in the
train program.

### 29. `Reliability: add chaos and failure-injection suites for topology, checkpoint, and validator flows`

Status: implemented on 2026-03-14 via GitHub issue `#3592`.

`psionic-train` now owns an explicit reliability suite in
`src/reliability.rs` that runs typed chaos scenarios over existing checkpoint,
collective, orchestrator, and validator contracts.

The new contract makes these reliability seams explicit:

- topology churn drills over elastic membership and checkpoint-backed recovery
- network degradation drills over collective cadence fallback
- stale-weight flood containment over rollout admission
- checkpoint corruption drills over stale-pointer fallback
- validator sampling stress over accepted, normalized, and rejected verdicts
- orchestrator restart roundtrips that resume window control after state restore

The canonical runbook and harness are now:

- `docs/TRAIN_RELIABILITY_REFERENCE.md`
- `scripts/release/check-psionic-train-reliability.sh`

This issue makes reliability claims a machine-checkable suite instead of a
collection of unrelated unit tests. Quantitative benchmark thresholds remain
the final train-program gap.

### 30. `Benchmarking: define performance acceptance thresholds for trainer, sandbox, datastream, and validation`

Status: implemented on 2026-03-14 via GitHub issue `#3593`.

`psionic-train` now owns a typed quantitative acceptance layer in
`src/benchmarking.rs` instead of leaving train performance closure to ad hoc
notes or one-off benchmark scripts.

The new benchmark contract makes these production thresholds explicit:

- fixed-budget trainer throughput
- rollout ingestion throughput at the orchestrator boundary
- warm sandbox reuse latency and reuse ratio
- checkpoint restore latency plus resumable datastream recovery throughput
- validator verification cost and sampled benchmark-check share
- elastic scaling curves from two to four members, including degraded transport
  fallback

The canonical runbook and harness are now:

- `docs/TRAIN_BENCHMARK_ACCEPTANCE_REFERENCE.md`
- `scripts/release/check-psionic-train-benchmark-acceptance.sh`

This issue closes the last train-system gap called out at the end of the issue
program: Psionic now has both chaos-style reliability drills and one owned
acceptance profile for deciding whether the current train substrate is fast and
stable enough to claim seriously.

## Later-Scope Issues

These are valid future issues, but they are not part of the minimum path above.

### Model promotion and release governance

Later the system will also need:

- candidate promotion gates
- release thresholds
- rollback policy
- checkpoint-to-release lineage
- human signoff hooks

### Human preference, critique, and label-ingestion pipelines

If OpenAgents expands into broader RLHF-style or critique-driven post-training,
the system will also need:

- critique and preference record schemas
- provenance and adjudication for noisy labels
- human-score and rubric blending
- reviewer-tooling integration

## Bottom Line

The current Psionic tree already contains real train-substrate work:

- runtime training truth
- datastream movement for train-relevant artifacts
- collective planning
- checkpoint and recovery session state
- early training-output lineage through adapters
- reusable environment, eval, validator, and orchestrator crates
- one real repo-owned Apple training lane with app-owned operator and accepted-
  outcome integration around it

That means the train system is no longer hypothetical.

But the current tree still stops short of a scaled, generalized, fully hardened
all-Rust training system.

The missing center of gravity is now:

- multi-device execution kernels and distributed optimizer execution at scale
- broader format interoperability and retention/promotion governance
- stronger provenance, security, and authority integration beyond the current
  Apple path
- broader operator and product surfaces beyond the current Apple reference
  workflow

That is the path Psionic still has to build from its now-real early train
system.

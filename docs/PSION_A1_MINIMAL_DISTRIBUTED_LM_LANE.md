# Psion A1 Minimal Distributed LM Lane

> Status: contract fixture landed 2026-04-27 as the smallest real
> language-model training lane that can later feed Pylon/Nexus participant and
> model-progress accounting.

This document records the Psionic lane contract for
`a1_minimal_distributed_lm_001`.

It sits between two existing lanes:

- `docs/PSION_CS336_A1_REFERENCE_LANE.md` proves bounded Stanford CS336 A1
  mechanics inside owned Rust.
- `docs/PSION_CS336_A1_DEMO_LANE.md` packages the A1 bounded reference trainer
  as one four-step host-CPU Pylon/Nexus demo.
- The actual `Psion` pretraining lane remains the broader pretraining program
  documented in `docs/TRAIN_SYSTEM.md`.

The minimal distributed LM lane is not a replacement for any of those. Its job
is to freeze the smallest shared-model contract that can support many Pylons
doing real assigned compute under one run id.

## Canonical Identity

- lane id: `a1_minimal_distributed_lm_001`
- contract schema:
  `psion.a1_minimal_distributed_lm.lane_contract.v1`
- release id: `psionic-train.a1_minimal_distributed_lm.release.v1`
- environment ref:
  `psionic.environment.a1_minimal_distributed_lm.tiny_lm.operator@v1`
- run id family: `a1_minimal_distributed_lm_001`
- retained fixture:
  `fixtures/training/a1_minimal_distributed_lm_lane_contract_v1.json`
- generator:
  `crates/psionic-train/examples/a1_minimal_distributed_lm_lane_contract_fixture.rs`
- checker:
  `scripts/check-a1-minimal-distributed-lm-lane-contract.sh`
- tokenizer/dataset bundle:
  `fixtures/psion/tokenized/a1_minimal_distributed_lm_tokenizer_dataset_bundle_v1.json`
- tokenizer/dataset generator:
  `crates/psionic-train/examples/a1_minimal_distributed_lm_tokenizer_dataset_bundle_fixture.rs`
- tokenizer/dataset checker:
  `scripts/check-a1-minimal-distributed-lm-tokenizer-dataset-bundle.sh`
- local-update proof:
  `fixtures/psion/a1_minimal_distributed_lm/local_update_report_v1.json`
- local-update checkpoints:
  `fixtures/psion/a1_minimal_distributed_lm/local_update_checkpoint_step2_v1.json`
  and
  `fixtures/psion/a1_minimal_distributed_lm/local_update_checkpoint_step4_v1.json`
- local-update artifact manifest:
  `fixtures/psion/a1_minimal_distributed_lm/local_update_artifact_manifest_v1.json`
- local-update contribution receipt:
  `fixtures/psion/a1_minimal_distributed_lm/local_update_contribution_receipt_v1.json`
- trusted aggregation report:
  `fixtures/psion/a1_minimal_distributed_lm/trusted_aggregation_report_v1.json`
- aggregate checkpoint:
  `fixtures/psion/a1_minimal_distributed_lm/aggregate_checkpoint_v1.json`
- promotion receipt:
  `fixtures/psion/a1_minimal_distributed_lm/promotion_receipt_v1.json`
- local-update generator:
  `crates/psionic-train/examples/a1_minimal_distributed_lm_local_update_fixture.rs`
- trusted aggregation generator:
  `crates/psionic-train/examples/a1_minimal_distributed_lm_trusted_aggregation_fixture.rs`
- local-update checker:
  `scripts/check-a1-minimal-distributed-lm-local-update.sh`
- trusted aggregation checker:
  `scripts/check-a1-minimal-distributed-lm-trusted-aggregation.sh`

The contract is typed in
`crates/psionic-train/src/a1_minimal_distributed_lm_lane.rs` and validates
deterministically without launching a training job.

## Frozen Contract Fields

The fixture names every field OpenAgents needs before it can define a matching
run:

- tokenizer artifact digest
- tokenized dataset digest
- validation set digest
- tiny Transformer LM model config
- AdamW optimizer config
- warmup plus cosine scheduler config
- checkpoint family
- trusted weighted-delta aggregation rule
- aggregation weight basis
- contribution receipt schema
- validator acceptance policy
- Nexus closeout and checkpoint-promotion semantics

The fixture now points at the dedicated frozen tokenizer and tokenized dataset
bundle typed in
`crates/psionic-train/src/a1_minimal_distributed_lm_tokenizer_dataset_bundle.rs`.
That bundle uses the existing CS336 A1 byte-level BPE trainer and tokenizer
runtime, but it does not attempt distributed BPE merge training. The committed
corpus fixture is synthetic Psionic text at
`fixtures/training/a1_minimal_distributed_lm_corpus.txt`.

Current frozen bundle values:

- tokenizer digest:
  `sha256:e32b619b67029aba5de26391f9a5f4a32801220ca690ae2c89d565e61069cf63`
- tokenized training dataset digest:
  `sha256:f0b92dc6301fc72e05a4ead6d85a4b5706e51267c116ecd72025a90c43a37905`
- validation set digest:
  `sha256:6c1c6ca83a2d8eca6cb133f3ec719e822f134452723e72e5201407b28cd3d228`
- tokenizer requested vocab size: `272`
- token counts: `201` training tokens and `92` validation tokens

The bundle records corpus source, source shard digests, tokenizer digest,
training and validation shard manifests, token counts, replay samples, and a
bundle digest. Validation replay samples decode back to the raw validation text
through the admitted tokenizer.

## Production Local Update

The first production local-update proof is typed in
`crates/psionic-train/src/a1_minimal_distributed_lm_local_update.rs`.

It runs the fixed bundle through the A1 Transformer forward path, materializes
final hidden states, computes an analytic cross-entropy backward pass for
`lm_head.weight`, applies global gradient clipping plus AdamW, and writes
checkpoint state with optimizer state and deterministic cursor state. It does
not use finite-difference gradients.

Current retained proof values:

- backward path: `analytic_lm_head_cross_entropy_backward_v1`
- finite-difference used: `false`
- trained parameters: `lm_head.weight`
- local steps: `4`
- consumed tokens: `8`
- training loss: `5.613701 -> 5.565771`
- validation loss: `5.6042047 -> 5.559738`
- resume exactness: `true`
- delta digest:
  `sha256:2ccbf7ea2a89e8403212e2985c6d0a158568bfafe5141eed8a0214864092d7c1`
- report digest:
  `sha256:f643547ad6002ea4028b9cf3f0993f02679cf974eb02ac002606e40a8a3ceec8`
- artifact manifest digest:
  `sha256:dedf62d211a2ba66ca45642016350dceb718af4fe655a575b6ba0619b0effba3`
- contribution receipt digest:
  `sha256:319e1fb5de6671fe9718d18ea74a4dca442867959e0427a983c657ad02767b91`

This is intentionally a narrow first production backward path. It proves a real
local update for the tiny shared LM lane, but it is LM-head-only and does not
claim full Transformer backward coverage for attention, MLP, embeddings, or
normalization parameters.

## Local-Update Receipt

The retained local-update proof now writes a lane-specific contribution receipt
and artifact manifest. They bind the facts OpenAgents needs to produce a
`ComputeAdapterContributionOutcome` without parsing logs:

- `training_run_id`, `stage_id`, `window_id`, `assignment_id`, `contribution_id`
- `worker_id`, `node_pubkey`, and `contributor_node_id`
- tokenizer, tokenized-dataset, validation-set, input-shard, and token-range
  digests
- base checkpoint ref and digest
- local step count, consumed token count, training loss, and validation loss
- output checkpoint ref and digest, output delta ref and digest, object digest,
  manifest digest, and local-update report digest
- validator disposition, validator verdict binding, aggregation eligibility,
  aggregation weight, and closeout binding

The fixture receipt is deliberately `validator_disposition: replay_required`,
`aggregation_eligibility: eligible`, and `accepted_for_aggregation: false`.
That means the local update is model-progress eligible, but it does not
pre-claim Nexus acceptance. Nexus closeout truth must accept the receipt before
OpenAgents may set `accepted_for_aggregation` or count the worker as a
model-progress participant.

Materialized file paths are carried only as manifest location hints. The stable
artifact-manifest digest clears those paths before hashing, so the logical
artifact identity remains stable across machines.

## Trusted Aggregation

The first trusted aggregate fixture consumes two accepted local-update
contributions from the same base checkpoint and aggregate window. The two
updates use distinct assignment, worker, node, and contribution ids and consume
different deterministic token windows: `[0, 1, 2, 3]` and `[4, 5, 6, 7]`.

Current retained aggregate values:

- accepted aggregate id:
  `aggregate.a1_minimal_distributed_lm_001.trusted_fixture.000001`
- accepted local updates: `2`
- model-progress participants: `2`
- total aggregation weight: `16` tokens
- aggregation rule: `trusted_weighted_delta_average_v1`
- aggregated delta digest:
  `sha256:856619da177eb5a276435e3f3ad38882eec016afa680c440dc83d793c2c23700`
- output checkpoint digest:
  `sha256:234460c819f2d4cf56918e774a2e23f2a65562bb925e359e2735ad0980070e8d`
- promoted checkpoint ref:
  `checkpoint://a1_minimal_distributed_lm_001.trusted_aggregation_fixture/promoted/step-000004`
- validation loss: `5.6042047 -> 5.562639`
- promotion receipt digest:
  `sha256:e7750512afe59d5980d561619d17ab1281b430155fa0011b3580e0a152fddb44`
- aggregation report digest:
  `sha256:2886b04948c2ddb312644854b001a37c127a7a1aa880459a215bd1ff97b1c704`

This is still trusted aggregation, not permissionless public model-progress
acceptance. The retained tests reject mismatched tokenizer and stale-window
inputs fail-closed before promotion.

## Contribution Semantics

The lane distinguishes participant truth from model-progress truth.

Accepted verifier/support work may count toward public participant totals when
it was assigned under this run id, executed through Psionic/Pylon, tied to
explicit inputs and artifacts, and accepted by Nexus closeout truth.

Only accepted `local_update_training` artifacts that enter the promoted
aggregate checkpoint can count as model-progress participants.

The fixture therefore maps:

- public label `participants` to internal
  `training_accepted_contributors`
- public label `model-progress participants` to internal
  `training_model_progress_contributors`

Support work has zero aggregation weight. It must never be used to claim that a
worker advanced the canonical model checkpoint.

## Claim Boundary

This lane honestly claims only a fixed tiny LM contract for distributed
local-update and verifier/support work.

It does not claim:

- OpenWebText leaderboard parity
- broad pretraining
- model-size leadership
- token-budget leadership
- total-FLOP leadership
- permissionless model-progress training
- that the bounded four-step `psion_cs336_a1_demo_v1` lane became distributed

The phrase "by number of participants" is allowed only when "participant" means
accepted real compute work under one run id. It must never be inferred from
online Pylons, seen-in-24h Pylons, sellable Pylons, generic payout totals,
Discord members, downloads, or app sessions.

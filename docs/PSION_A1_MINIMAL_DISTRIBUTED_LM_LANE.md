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

The first fixture intentionally uses the tiny CS336 A1 reference tokenizer and
tokenized dataset digests as anchors. Issue #949 is expected to replace those
with a dedicated frozen tokenizer and tokenized dataset bundle for this lane,
without changing the lane identity.

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

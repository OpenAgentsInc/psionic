# Validator Challenge Scoring Reference

> Status: canonical `XTRAIN-37` / `#582` record, updated 2026-03-26 after
> landing the first public validator challenge and scoring contract.

## What This Closes

Psionic now owns one typed validator challenge and improvement-scoring surface
above the public miner protocol.

The new contract lives in:

- `crates/psionic-train/src/validator_challenge_scoring_contract.rs`
- `crates/psionic-train/src/bin/validator_challenge_scoring_contract.rs`
- `fixtures/training/validator_challenge_scoring_contract_v1.json`
- `scripts/check-validator-challenge-scoring-contract.sh`

This issue closes the first truthful answer to:

- which validator assignment challenged which miner session
- which dataset receipt, delta artifact, and checkpoint reference a replay had
  to bind to
- how improvement is scored
- when the validator must accept versus require replay
- how stale-checkpoint submissions are refused instead of being hand-scored

## Contract Shape

The canonical contract freezes:

- one validator improvement-scoring policy
- two replay rules
- two score receipts
- one stale-checkpoint refusal

## Current Canonical Binding

The contract binds directly to:

- the public work-assignment contract
- the public miner protocol contract
- the shared validator-promotion contract

That means public validator scoring now has one typed path from challenge
assignment to replay rule to final validator disposition.

## Canonical Score Receipts

The current score receipts are:

- `score.public_validator.google.local_mlx.window1231`
- `score.public_validator.local_mlx.google.window1231`

Those two receipts keep the first public scoring boundary explicit:

- one accepted contribution with low replay error and clear improvement
- one replay-required contribution with replay error above the admitted ceiling

## Honest Refusal Boundary

The first explicit validator refusal is:

- `refusal.public_validator.google.local_rtx4080.stale`

That refusal keeps a stale miner submission out of scoring because the miner
protocol already failed checkpoint recovery.

## Pass Criteria

The contract is green only if all of the following stay true:

- replay rules still bind validator assignments to the exact challenged miner
  sessions
- score receipts still recompute their improvement basis points from the stored
  before/after losses
- high replay error still forces `replay_required`
- stale checkpoint refusals still point at a real miner-protocol refusal

## Current Limits

This issue intentionally does not claim:

- multi-validator consensus
- checkpoint promotion
- fraud penalties or slashing

This issue freezes challenge truth first: replay rules, improvement thresholds,
validator receipts, and refusal posture.

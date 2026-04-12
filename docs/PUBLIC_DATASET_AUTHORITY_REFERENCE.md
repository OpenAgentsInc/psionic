# Public Dataset Authority Reference

> Status: canonical `XTRAIN-34` / `#578` record, updated 2026-03-26 after
> landing the first page-proofed public data-truth contract.

## What This Closes

Psionic now owns one page-proofed public dataset authority surface above the
public work-assignment contract.

The new contract lives in:

- `crates/psionic-train/src/public_dataset_authority_contract.rs`
- `crates/psionic-train/src/bin/public_dataset_authority_contract.rs`
- `fixtures/training/public_dataset_authority_contract_v1.json`
- `scripts/check-public-dataset-authority-contract.sh`

This issue closes the first truthful answer to:

- which tokenized corpus and tokenizer revision public work used
- which page slices were assigned to miners and validators
- how those page slices bind back to shard lineage and replay identity
- how duplicate miner claims are refused

## Contract Shape

The canonical contract freezes:

- eight dataset pages aligned to public assignment ids
- eight page proofs
- four admitted miner data receipts
- one refused duplicate receipt

## Current Canonical Binding

The contract binds directly to the committed Psion tokenizer and tokenized
corpus fixtures, plus the public work-assignment contract.

That means public data truth now carries:

- tokenizer id and version
- tokenizer digest and config digest
- tokenized corpus dataset id and version
- replay identity
- packing-policy digest

## Page Proofs

The current page proofs cover:

- four train pages on `psion_train_shard_0001`
- four validator challenge pages on `psion_validation_shard_0001`

So later validator and reward surfaces can point at machine-legible page truth
instead of only human-readable page names.

## Honest Anti-Replay Boundary

The first duplicate refusal is explicit:

- `anti_replay.assignment.public_miner.window1230.google.duplicate`

That receipt proves a replayed Google miner claim with the same fingerprint is
refused as duplicate work rather than silently counted again.

## Pass Criteria

The contract is green only if all of the following stay true:

- dataset pages still match the public assignment page selectors exactly
- every page retains one proof
- tokenizer, packing, and replay digests remain aligned to the committed data
  fixtures
- four admitted miner receipts remain present
- one duplicate refusal remains present

## Current Limits

This issue intentionally does not claim:

- content-addressed artifact exchange
- public miner execution protocol closure
- validator scoring or reward accounting

This issue freezes public data truth first: tokenizer binding, packing binding,
page proofs, and duplicate-work refusal.

## Relation To Weak-Device Validation Replay

The newer weak-device validation replay proof surface in
`crates/psionic-train/src/weak_device_accepted_outcome_proof.rs` builds on this
dataset-authority contract; it does not replace it.

When Apple / Metal validator replay emits
`weak_device_validation_replay_proof.json`, that proof cites one accepted weak-
device replay outcome plus the validator score, quality-drift, rollback, and
artifact-lineage evidence that bounded replay consumed. The proof is only
honest because the public dataset authority already froze tokenizer, packing,
page-slice, and duplicate-work truth for the cited public replay inputs. The
new proof therefore packages one accepted validator-side replay outcome above
the retained dataset/page authority surface; it does not independently prove
dataset provenance, public-run policy, payout closeout, or checkpoint finality.

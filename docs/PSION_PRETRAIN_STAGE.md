# Psion Pretrain Stage

> Status: canonical `PSION-12` / `#368` pretrain-stage contract, written
> 2026-03-22 after landing the first Psion compact decoder family.

This document freezes the first explicit `pretrain` stage for the `Psion`
learned-model lane.

It keeps pretraining as a named stage with its own receipts instead of
overloading SFT or relying on notebook-local state.

## Canonical Artifacts

- `crates/psionic-train/src/psion_pretrain_stage.rs` owns the typed pretrain
  stage config and run receipt.
- `crates/psionic-train/examples/psion_pretrain_stage_fixtures.rs`
  regenerates the canonical config and receipt fixtures.
- `fixtures/psion/pretrain/psion_pretrain_stage_config_v1.json` is the
  canonical declared pretrain-stage config fixture.
- `fixtures/psion/pretrain/psion_pretrain_stage_receipt_v1.json` is the
  canonical pretrain-stage receipt fixture.

The stable schema versions are:

- `psion.pretrain_stage_config.v1`
- `psion.pretrain_stage_receipt.v1`

## What The Stage Freezes

The pretrain stage now binds:

- one explicit `pretrain` stage kind in the stage program
- one next-token objective config tied to model, tokenizer, and dataset
  identity
- source-family-aware reporting across train, validation, and held-out splits
- replay receipts tied back to the tokenized corpus replay contract
- checkpoint-lineage receipts tied back to the promoted stage checkpoint and
  model descriptor

That gives the first Psion pretrain lane one repo-owned stage surface instead of
smuggling broad-model training through existing SFT types.

## Mechanical Enforcement

`psionic-train` now validates that:

- the declared stage is really `pretrain`
- objective config uses the model descriptor's tokenizer binding and context
  length
- sampling policy, dataset identity, and model descriptor all stay aligned
- source-family reporting covers every split-family pair represented by the
  tokenized corpus
- replay receipts preserve the tokenized corpus replay contract
- checkpoint lineage stays tied to one promoted checkpoint family and one model
  descriptor digest
- the generic stage program will not complete a `pretrain` stage until at least
  one validated pretrain receipt has been recorded

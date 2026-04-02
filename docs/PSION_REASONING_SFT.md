# Psion Reasoning SFT

> Status: canonical `PSION-18` / `#374` reasoning-SFT contract, written
> 2026-03-22 after landing the first bounded `Psion` reasoning-SFT bundle.

This document freezes the first truthful supervised reasoning-tuning lane for
`Psion`.

It does not claim generic chat tuning, arbitrary answer formatting, or model
execution it did not perform. It records one bounded `general_sft` stage that
starts from the canonical `Psion` pretrain receipt, keeps assumptions and
uncertainty explicit, and preserves multiple valid reasoning styles instead of
rewarding one canned narration shape.

The actual-lane recipe freeze in `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md` now
declares this reasoning-SFT bundle as the canonical bounded `general_sft`
bridge on the actual-lane continuation path
`pretrain -> general_sft -> agentic_sft`. That makes this bundle the declared
handoff between the actual pretrain lane and the later bounded
plugin-conditioned stage, without implying that broader plugin-conditioned
cluster operation is already closed.

That declared bridge is now also retained explicitly in
`docs/PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF.md` and
`fixtures/psion/pretrain/psion_actual_pretraining_continuation_handoff_v1.json`.
The actual lane therefore closes into this reasoning-SFT bridge through one
accepted-checkpoint handoff contract instead of only through the recipe prose.

## Canonical Artifacts

- `crates/psionic-train/src/psion_reasoning_sft.rs` owns the reasoning-SFT
  dataset bundle, stage receipt, evaluation receipt, and full run bundle.
- `crates/psionic-train/examples/psion_reasoning_sft_fixtures.rs` regenerates
  the canonical reasoning-SFT fixtures.
- `fixtures/psion/sft/psion_reasoning_sft_dataset_bundle_v1.json` is the
  canonical derived-trace and lineage bundle.
- `fixtures/psion/sft/psion_reasoning_sft_stage_receipt_v1.json` is the
  canonical bounded reasoning-SFT stage receipt.
- `fixtures/psion/sft/psion_reasoning_sft_evaluation_receipt_v1.json` is the
  canonical style-plurality and truth-control evaluation receipt.
- `fixtures/psion/sft/psion_reasoning_sft_run_bundle_v1.json` is the canonical
  self-contained reasoning-SFT run bundle.

The stable schema versions are:

- `psion.reasoning_sft_dataset_bundle.v1`
- `psion.reasoning_sft_stage_receipt.v1`
- `psion.reasoning_sft_evaluation_receipt.v1`
- `psion.reasoning_sft_run_bundle.v1`

## What The Bundle Freezes

The first reasoning-SFT bundle now binds:

- one explicit `pretrain -> general_sft` stage-program path in
  `psionic-train`
- one canonical reasoning-SFT dataset bundle tied to the existing
  `psion_reasoning_sft_seed_v1` lineage row
- one truth-control surface that requires explicit assumptions, explicit
  uncertainty language, and explicit separation between normative statements
  and engineering inference
- three admissible style profiles spanning decomposition strategy, explanation
  order, and abstraction level
- one stage receipt that rejects style collapse when any single narration style
  dominates the bounded trace mix
- one evaluation receipt that accepts multiple valid styles per case and
  requires all declared style profiles to remain live in the observed output
  set

That makes reasoning SFT a repo-owned machine-checkable lane instead of a vague
"make the model sound smarter" tuning claim.

## Mechanical Enforcement

`psionic-train` now validates that:

- the bound stage graph is still exactly `pretrain -> general_sft`
- the general-SFT stage still points back to the same canonical pretrain
  receipt and promoted checkpoint lineage
- the reasoning-SFT dataset bundle still matches the canonical SFT artifact
  lineage row in `fixtures/psion/lifecycle/psion_artifact_lineage_manifest_v1.json`
- derived traces still carry parent source ids and parent tokenized-corpus ids
- the exclusion manifest still admits those source ids for model training, so
  held-out or training-excluded material cannot silently enter the SFT lane
- the stage receipt still carries a multi-style trace distribution under the
  dominant-style ceiling and still reports explicit assumption, uncertainty,
  and normative-versus-inference retention
- the evaluation receipt still accepts multiple valid style profiles per case
  instead of one mandatory answer outline

## What This Does Not Claim

This closes one bounded reasoning-SFT step. It does not claim:

- generic chat-breadth finetuning
- public serving readiness
- executor substitution
- hidden execution or tool use
- that every future SFT stage may ignore the lifecycle, exclusion, or artifact
  lineage contracts this bundle depends on

`PSION-30` now binds this reasoning-SFT run into the first bounded
decentralized contribution bundle in
`docs/PSION_DECENTRALIZED_CONTRIBUTION.md`. Contributed outputs sit above this
SFT baseline, but they still need the same later acceptance, capability, and
rollback discipline before they can become served `Psion` outputs.

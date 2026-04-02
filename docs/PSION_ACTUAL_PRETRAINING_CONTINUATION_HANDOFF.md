# Psion Actual Pretraining Continuation Handoff

> Status: canonical accepted-checkpoint handoff contract for the actual
> `Psion` pretraining lane, written 2026-04-02 after binding the frozen
> broader-pretraining checkpoint family to one named continuation target.

This document freezes the operator-owned handoff from an accepted actual
pretraining checkpoint into the declared bounded continuation path.

It does not claim continuation-stage execution. It does make the base lane end
in one named target instead of an unnamed generic checkpoint.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_continuation_handoff.rs`
  owns the typed handoff contract.
- `crates/psionic-train/examples/psion_actual_pretraining_launcher_fixtures.rs`
  regenerates the committed handoff fixture and the retained resume example.
- `fixtures/psion/pretrain/psion_actual_pretraining_continuation_handoff_v1.json`
  is the canonical committed handoff contract.
- `fixtures/psion/pretrain/psion_actual_pretraining_launcher_example/resume/run-psion-actual-20260402t020000z/continuation/accepted_checkpoint_handoff.json`
  is the retained example under the actual-lane evidence family.

Stable schema version:

- `psion.actual_pretraining_continuation_handoff.v1`

## What The Handoff Binds

The contract binds:

- one accepted actual-lane checkpoint pointer
- one continuation target id:
  `psion_actual_pretraining_general_sft_agentic_sft_v1`
- one ordered stage path:
  `pretrain -> general_sft -> agentic_sft`
- one canonical reasoning-SFT run bundle
- one canonical plugin-conditioned stage manifest
- one canonical plugin-conditioned run bundle
- one canonical continuation-stage eval pack
- the plugin benchmark-pack bindings and eval hooks already attached to that
  bounded continuation target

That means the accepted checkpoint now feeds one declared path for later
review instead of reading as an unowned checkpoint export.

## Retained Path

The actual-lane retained evidence family now reserves:

```text
continuation/accepted_checkpoint_handoff.json
```

The operator path writes that artifact when resume selects an accepted
checkpoint. Start and dry-run still reserve the path through the retained-path
set, but they do not fabricate a handoff before an accepted checkpoint exists.

## Claim Boundary

This handoff claims:

- one explicit base-lane to continuation-lane binding
- one benchmark-pack attachment for later continuation review
- one retained operator artifact proving which accepted checkpoint is allowed
  to feed the bounded continuation target

It does not claim:

- plugin-conditioned cluster training
- completed continuation-stage rehearsal
- promotion beyond the bounded continuation target

The current repo-owned continuation-review surface above this handoff also
lives in
`fixtures/psion/pretrain/psion_actual_pretraining_continuation_alignment_bundle_v1.json`.
That alignment bundle keeps the reasoning bridge, bounded plugin-conditioned
stage, and current `agentic_sft -> rl` reference-program lineage together for
later review, but it still does not claim actual continuation execution.

## Related Docs

- `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md`
- `docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`
- `docs/PSION_REASONING_SFT.md`
- `docs/PSION_PLUGIN_PROGRAM_MAP.md`

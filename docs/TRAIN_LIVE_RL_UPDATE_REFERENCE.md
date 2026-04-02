# Training Live RL Update Reference

This document describes the first bounded live rollout-to-trainer bridge in
`psionic-train`.

## Scope

The current path is intentionally narrow:

- backend: repo-owned open-adapter LM-head LoRA lane
- active policy surface: `TrainingSamplerServedRevision`
- control-plane source: orchestrator-owned `TrainingOrchestratorBatchRecord`
  plus `TrainingOrchestratorWindow`
- prompt/completion source: explicit `LiveRlRolloutInput` records keyed by
  rollout artifact id
- output: one promoted adapter revision ready for
  `TrainingSamplerService::refresh_revision`

It does not claim full dense-model RL training, full-distribution KL
distillation, or distributed trainer execution.

## What Is Implemented

`psionic-train` now owns:

- `OpenAdapterLiveRlUpdateExecutor`
- `LiveRlUpdateRequest`
- `LiveRlMaterializedBatch`
- `LiveRlMaterializedRollout`
- `LiveRlMaterializedToken`
- `LiveRlUpdateReceipt`

The live path does all of the following in one bounded cycle:

- checks that the current served revision matches the trainer batch target
- joins the trainer batch to accepted rollout receipts in the owning window
- keeps exact versus admitted-off-policy rollouts separate in typed state
- keeps prompt text outside completion tokens so the prompt/completion boundary
  stays explicit
- queries the current sampler revision for live chosen-token logprobs
- computes per-token importance ratios against the rollout artifact logprobs
- clips those ratios under an explicit bounded update policy
- carries reward and advantage into one scalar chosen-token update weight
- optionally carries chosen-token teacher logprobs into the same typed batch
- turns the materialized tokens into one weighted adapter trainer step
- exports one promoted LoRA artifact and wraps it as a new served revision

## Update Semantics

The live bridge uses a chosen-token update rule over the open-adapter LM-head
target.

The current bounded weight per token is:

- `clipped_importance_ratio * advantage + reward_mix * reward`

The optional teacher input is also bounded:

- `teacher_target_blend * exp(teacher_logprob)` is added as an auxiliary
  chosen-token confidence term

This matters because the current path is not a full-distribution KL or
distillation runtime. It is a typed single-token auxiliary signal that keeps
teacher compatibility explicit without pretending the broader distillation path
already exists.

## Operator Surfaces

`LiveRlMaterializedBatch` keeps the important state inspectable:

- exact versus off-policy rollout counts
- prompt digests and per-token prefix digests
- rollout-time versus live logprobs
- importance ratios and clipped ratios
- reward, advantage, and final loss weight
- teacher-token count

`LiveRlUpdateReceipt` keeps the promoted update inspectable:

- source and promoted revision ids
- promoted revision number
- rollout and token counts
- mean observed versus live logprob
- mean loss weight and mean weighted loss
- adapter identity digest

## Release Check

Run:

```bash
scripts/release/check-psionic-train-live-rl-update.sh
```

That check runs the focused `rl_online_update` tests, including:

- one end-to-end update from admitted rollout batch to promoted revision
- one validation failure for teacher-logprob alignment

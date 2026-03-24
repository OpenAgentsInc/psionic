# Psionic MLX Recipe Package

This document defines the bounded `psionic-mlx-recipes` package that closes
`PMLX-706`.

## Scope

`psionic-mlx-recipes` is the MLX-facing training recipe layer above
`psionic-train`.

It owns:

- ergonomic MLX-style method selection for SFT, LoRA, DoRA, QLoRA, DPO, CPO,
  ORPO, GRPO, online DPO, XPO, and PPO
- recipe planning into the existing train run graph, budget, policy, optimizer,
  adapter, and rollout-validator primitives
- machine-readable method inventory plus package-facing CLI entrypoints for
  method listing and plan emission

It does not own:

- a second trainer runtime
- notebook-only recipe state
- hidden policy or stage transitions outside the existing train substrate

## Current Truth

The package is a planner, not a second trainer.

It compiles MLX-style recipe choices into existing Psionic train primitives:

- `TrainingRunState`
- `TrainingLoopBudget`
- `TrainingOptimizerConfig`
- `PolicyRevision`
- `OpenAdapterAdmissibleModelFamily`
- `RolloutValidatorPolicy`

That keeps one training architecture in the repo.

The first swarm lane now also consumes the emitted recipe plan directly inside
`psionic-mlx-workflows::plan_first_swarm_live_adapter_cluster`. The recipe
package still does not own a second trainer runtime. It now provides the plan
digest, adapter geometry, and stage truth that seed one real mixed-backend
adapter-cluster window through `psionic-train`.

## Stage Mapping

The current bounded stage mapping is:

- `sft`, `lora`, `dora`, `qlora` -> `general_sft`
- `dpo`, `cpo`, `orpo` -> `general_sft -> agentic_sft`
- `grpo`, `online_dpo`, `xpo`, `ppo` -> `general_sft -> agentic_sft -> rl`

That mapping is explicit in the recipe plan and must not be hidden behind
method-specific side effects.

## Adapter Methods

Adapter methods reuse the existing open adapter lane:

- admissible family:
  `OpenAdapterAdmissibleModelFamily::GptOssDecoderLmHeadLora`
- adapter family:
  `gpt_oss.decoder_lm_head_lora`
- adapter format:
  `safetensors`

The first mixed-hardware swarm lane now also has one explicit MLX-backed
backend label for Mac contributors:

- `open_adapter_backend.mlx.metal.gpt_oss_lm_head`

That label is consumed by `psionic-train` runtime truth. It does not change the
planner boundary here: `psionic-mlx-recipes` still selects and parameterizes
the existing open-adapter lane instead of introducing a second trainer runtime.

`qlora` keeps quantization explicit in the adapter execution plan.

## RL Methods

RL-style methods reuse `RolloutValidatorPolicy` and emit an explicit rollout
validator posture in the plan rather than hiding it behind implicit defaults.

## CLI

The package CLI is:

- `psionic-mlx-recipes methods`
- `psionic-mlx-recipes plan`

`methods` emits one machine-readable inventory with stage sequence,
adapter-required posture, rollout-validator posture, and bounded notes per
method family.

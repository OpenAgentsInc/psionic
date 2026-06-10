# Psion CS336 A5 Alignment Reference Lane

> Status: `implemented_early` bounded reference lane, landed 2026-06-10
> under issue #1101. Companion to the A1 and A2 reference lanes
> (`PSION_CS336_A1_REFERENCE_LANE.md`, `PSION_CS336_A2_REFERENCE_LANE.md`).

This document records the owned `psionic` surfaces for the bounded
Stanford CS336 Assignment 5 (alignment and reasoning RL) port program.
It exists as the psionic-side answer to the OpenAgents homework epic's
external ask (openagents issue #4682).

## Identity

- lane id: `psion_cs336_a5_alignment_reference_v1`
- owned surface: `crates/psionic-train/src/cs336_a5_alignment_reference.rs`
- claim boundary: bounded deterministic f64 reference math for the
  portable Stanford CS336 A5 adapter surface only; no model execution, no
  RL training run, no tokenizer coupling, and no claim of full A5 parity
  against Stanford fixtures.

## Adapter Matrix

This matrix maps the adapter families declared in the Stanford reference
`assignment5-alignment/tests/adapters.py` onto owned surfaces. It is the
hard truth bar for discussing A5 coverage in `psionic`.

| Stanford adapter | Owned surface | Status | Notes |
| --- | --- | --- | --- |
| `run_tokenize_prompt_and_output` | `cs336_a5_tokenize_prompt_and_output` | `partial` | Concatenation, shifting, padding, and response-mask construction over pre-tokenized id sequences; the HF string tokenizer step is out of scope |
| `run_compute_rollout_rewards` | `cs336_a5_rollout_rewards` | `partial` | Aggregation over precomputed reward components; the reward-callable invocation belongs to the caller |
| `run_compute_group_normalized_rewards` | `cs336_a5_group_normalized_rewards` | `implemented_early` | mean/none baseline; std/none/mean normalizer; sample std (n−1) matching torch defaults |
| `run_compute_policy_gradient_loss` | `cs336_a5_policy_gradient_loss` | `implemented_early` | All four methods: none, noclip, GRPO token-level clip, GSPO sequence-level masked ratio; clip-fraction metadata; typed refusals for missing arguments |
| `run_aggregate_loss_across_microbatch` | `cs336_a5_aggregate_loss_across_microbatch` | `implemented_early` | sequence and constant normalization |
| `run_grpo_train_step` | composition of the above | `partial_outside_psionic` | The forward/backward/optimizer execution belongs to the training boundary; the loss/advantage/aggregation core is owned here |
| `run_get_response_log_probs` | — | `partial_outside_psionic` | Model-coupled scoring; consumes the tokenized layout from this lane |
| `get_packed_sft_dataset` | `cs336_a5_pack_sft_sequences` | `partial` | Concat-and-chunk packing; exact conformance against Stanford packed-SFT fixtures unverified |
| `run_iterate_batches` | — | `planned` | Batch iteration utility; port with the training-boundary integration |
| `run_parse_mmlu_response` | `cs336_a5_parse_mmlu_response` | `partial` | Bounded heuristic (explicit-pattern preference, then first standalone A–D token) |
| `run_parse_gsm8k_response` | `cs336_a5_parse_gsm8k_response` | `partial` | Bounded heuristic (last numeric token, commas/dollars stripped) |
| `run_compute_per_instance_dpo_loss` | `cs336_a5_per_instance_dpo_loss` | `partial` | Bradley–Terry loss over supplied summed log-probs with numerically stable softplus; the two-model/tokenizer execution is out of scope |

## Landed Tests

13 unit tests: tokenization layout and response masks; group
normalization against hand computation (including the zero-deviation
group); group-size refusal; policy-gradient `none` closed form; GRPO
clipping with clip-fraction accounting; GSPO sequence-level masked ratio;
typed missing-argument refusals; sequence and constant aggregation; DPO
closed form, preference ordering; rollout reward summary; MMLU/GSM8K
parsing; SFT packing.

## Relation To The Homework Epic

The OpenAgents CS336 epic (#4682) decomposes A5 into network work:
rollout generation as inference homework, reward/eval grading as
deterministic CPU homework, and the policy-gradient update behind the
training boundary. This lane is the math those job kinds and the update
step share. The grading heuristics here are bounded reference parsers;
production graders for paid homework must be conformance-tested per the
epic's verification rules before any payout depends on them.

# XTRAIN PGOLF Token Admission Audit

Date: 2026-03-27

## Goal

Improve the first real XTRAIN-trained PGOLF-shaped inference lane without changing
the training math, by tightening decode so the promoted tokenizer contract can
refuse tokens that should never be emitted in normal text generation.

## Problem Found

The previous promoted PGOLF inference path was allowed to sample any logit in the
full SentencePiece table, including:

- `unused` reserved ids such as `794`, `862`, and `952`
- `unknown` ids
- non-EOS `control` ids

That meant train and infer were structurally working, but the first emitted tokens
could still land in dead vocabulary slots and produce nonsense output even when
the model had learned something real about the tiny cycle task.

Concrete evidence from the prior retained XTRAIN report:

- report: `/tmp/psionic_xtrain_pgolf_eval_1774599196/xtrain_parameter_golf_train_infer_report.json`
- prior XTRAIN generated tokens: `[952, 862, 862, 794, 794, 794, 794, 794]`
- those ids resolve to tokenizer pieces:
  - `952 -> <reserved_0952> (unused)`
  - `862 -> <reserved_0862> (unused)`
  - `794 -> <reserved_0794> (unused)`

## Change

Decode now enforces one explicit token-admission policy for promoted PGOLF bundles:

- allow `normal` tokens
- allow `byte` tokens
- allow EOS control ids when present
- refuse `unknown`
- refuse `unused`
- refuse non-EOS `control`

This landed in three places so direct runtime, served runtime, and the benchmark
path all agree:

- `psionic-runtime::TokenSampler` gained `select_next_token_with_disallowed`
- `ParameterGolfPromotedRuntimeTokenizer` now computes stable
  `generation_disallowed_token_ids`
- promoted PGOLF direct decode, served decode, and the train/infer proof example
  all use that refusal list

## Verification

Commands run:

```bash
cargo test -q -p psionic-runtime token_sampler_can_refuse_disallowed_ids -- --nocapture
cargo test -q -p psionic-models promoted_runtime_tokenizer_refuses_unknown_non_eos_control_and_unused_ids -- --nocapture
cargo build -q -p psionic-serve --example xtrain_parameter_golf_train_infer
target/debug/examples/xtrain_parameter_golf_train_infer /tmp/psionic_xtrain_pgolf_eval_1774601500
```

New retained report:

- `/tmp/psionic_xtrain_pgolf_eval_1774601500/xtrain_parameter_golf_train_infer_report.json`

## Results

Training quality metrics are unchanged, as expected, because this pass only changed
decode admission:

- XTRAIN loss: `7.634324550628662`
- XTRAIN BPB: `8.811201735783069`

Output quality improved materially at the inference layer:

- previous XTRAIN generated tokens:
  - `[952, 862, 862, 794, 794, 794, 794, 794]`
- current XTRAIN generated tokens:
  - `[7, 6, 8, 8, 8, 8, 7, 7]`

That is still not the exact target cycle `abcd -> efghabcd`, but it is now using
live task vocabulary instead of dead reserved ids.

Throughput stayed effectively flat to slightly better relative to the immediately
prior retained runtime-allocation pass:

- prior direct runtime: `24.827612078552526 tok/s`
- current direct runtime: `24.86396449115071 tok/s`
- prior served runtime: `24.70252483344532 tok/s`
- current served runtime: `25.118654325792047 tok/s`
- direct/served parity: still `true`

## Current Honest State

The system now has three distinct truths:

1. XTRAIN training works and produces a promoted PGOLF-shaped bundle.
2. PGOLF-shaped inference works in both direct and served paths.
3. The inference path no longer lies to itself by emitting tokenizer-reserved
   garbage ids.

What is still missing is stronger learned task fidelity. The next iteration should
target training capacity or coordinate budget so the first emitted token becomes
`5` instead of `7`.

# XTRAIN To PGOLF Quality Tuning Audit

Date: 2026-03-27

## Purpose

This audit records the first post-proof tuning pass on the working
XTRAIN-promoted PGOLF train/infer lane.

The previous retained baseline already proved:

- XTRAIN training completes
- a promoted PGOLF-shaped bundle is emitted
- the bundle runs through both local runtime and `psionic-serve`
- direct runtime and served runtime agree
- bounded-window decode is much faster than the old legacy full-history path

This iteration only tuned the bounded local-reference XTRAIN hyperparameters. It
did not widen the trainable surface and it did not change the runtime
architecture.

## Config Change

Retained tuned deltas inside
`ParameterGolfReferenceTrainingConfig::xtrain_promoted_general_small_decoder_baseline()`:

- `finite_difference_epsilon`: `0.01 -> 0.005`
- `tied_embed_lr`: `0.05 -> 0.2`
- `matrix_lr`: `0.04 -> 0.08`
- `scalar_lr`: `0.04 -> 0.08`

Unchanged important constraints:

- `max_steps = 8`
- `train_sequence_length = 4`
- `bounded_attention_window_tokens = 4`
- same selected coordinate budget
- same direct-runtime vs served-runtime inference path

## Verification

Build:

```bash
cargo build -q -p psionic-serve --example xtrain_parameter_golf_train_infer
```

Focused config test:

```bash
cargo test -q -p psionic-train xtrain_promoted_parameter_golf_config_expands_the_inferable_budget -- --nocapture
```

End-to-end proof:

```bash
target/debug/examples/xtrain_parameter_golf_train_infer /tmp/psionic_xtrain_pgolf_eval_1774598142
```

Generated report:

- `/tmp/psionic_xtrain_pgolf_eval_1774598142/xtrain_parameter_golf_train_infer_report.json`

Observed completion line:

```text
xtrain PGOLF train->infer completed: report=/tmp/psionic_xtrain_pgolf_eval_1774598142/xtrain_parameter_golf_train_infer_report.json xtrain_loss=7.634325 direct_tps=24.49
```

## Measured Delta Versus Prior Retained Baseline

Previous retained XTRAIN baseline:

- validation loss: `8.447443962097168`
- validation BPB: `9.749668409844002`
- direct runtime throughput: `24.8937880452915 tok/s`

Tuned XTRAIN result:

- validation loss: `7.634324550628662`
- validation BPB: `8.811201735783069`
- direct runtime throughput: `24.48918946732971 tok/s`

Delta from previous retained XTRAIN baseline:

- loss improvement: `0.8131194114685059`
- BPB improvement: `0.938466674060933`
- throughput delta: `-0.404598577961792 tok/s`

Interpretation:

- quality improved materially
- throughput stayed in the same bounded-window band and appears slightly lower,
  but only by about `1.6%`
- the runtime implementation itself did not change, so this throughput change is
  best treated as measurement noise until repeated benchmark runs say otherwise

## Full Proof Comparison Against The Original Proof Run

Proof baseline:

- validation loss: `8.60598874092102`
- validation BPB: `9.93265382277841`
- generated tokens: `952,1005,951,900,884,862,1005,862`

Tuned XTRAIN:

- validation loss: `7.634324550628662`
- validation BPB: `8.811201735783069`
- generated tokens: `952,862,862,794,794,794,794,794`

Improvement over proof baseline:

- loss delta: `0.9716641902923584`
- BPB delta: `1.1214520869953404`

Throughput on the tuned XTRAIN bundle:

- legacy full-history decode: `2.730272440558336 tok/s`
- bounded direct runtime: `24.48918946732971 tok/s`
- served runtime: `24.429709034170017 tok/s`
- direct over legacy improvement: `796.9503959949766%`
- direct runtime vs served runtime parity: `true`

## Honest Boundary

This tune is worth retaining because it improves the measurable model-quality
metrics a lot without changing the runtime architecture or collapsing the
throughput posture.

What it still does not solve:

- exact prefix gain remains `0`
- exact cycle match remains `false`
- the model still emits reserved tokens instead of the intended
  `abcd -> efghabcd` continuation

So the current truth is:

- train/infer path: working
- bounded runtime parity: working
- throughput posture: working
- quality metrics: improved materially
- exact inferable toy-cycle correctness: still not working

## Next Likely Move

The next useful quality step is probably no longer another simple hyperparameter
change. The remaining gap now looks structural:

- either the trainable coordinate surface is still too small for exact-token
  control
- or the finite-difference local-reference trainer has reached the limit of what
  it can realistically prove on this toy lane

The next quality iteration should therefore target exact-token behavior
explicitly, not just lower loss:

1. add a tiny exact-cycle tuning profile with a slightly richer token-to-logit
   trainable surface
2. measure exact prefix gain as the primary acceptance gate
3. keep the current tuned bounded-window runtime as the throughput reference

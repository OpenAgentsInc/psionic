# XTRAIN PGOLF Sparse Budget And Allowed-Decode Audit

Date: 2026-03-27

## Goal

Continue improving the working XTRAIN -> promoted PGOLF train/infer lane without
changing the frozen promoted model-family contract.

This pass retained two changes:

- a faster decode-selection path that works from the admitted token set instead of
  scanning through a large rejected vocabulary
- a wider **sparse** XTRAIN training budget that stays on token/control surfaces
  instead of widening into dense matrix updates

## Retained Changes

### 1. Allowed-token decode fast path

Promoted PGOLF runtime tokenizers now expose both:

- `generation_allowed_token_ids`
- `generation_disallowed_token_ids`

`psionic-runtime::TokenSampler` now has
`select_next_token_with_allowed`, with one greedy/no-penalty fast path that scans
only the admitted token ids and avoids the old clone-and-mask posture.

The promoted decode callers were updated to use that admitted-token path:

- direct bundle inference
- served promoted PGOLF inference
- the `xtrain_parameter_golf_train_infer` proof example

### 2. Wider sparse XTRAIN budget

The bounded XTRAIN promoted config still keeps:

- `max_steps = 12`
- the same frozen baseline PGOLF-shaped model config

But the selected sparse coordinates were widened further on cheap surfaces:

- token embedding rows for the 8 task tokens: `2 -> 4` dims per token
- `skip_weights`: `2 -> 4`
- `blocks.0.attn.q_gain`: `4 -> 6`
- `blocks.0.attn_scale`: `2 -> 4`
- `blocks.0.mlp_scale`: `2 -> 4`
- `blocks.0.resid_mix`: `2 -> 4`

This lifts the sparse coordinate budget from `32` to `54` without widening into
the dense Muon matrix path.

## Rejected Local Experiment

I also tried widening the XTRAIN budget into `c_k`, `c_v`, and MLP matrix weights.
That was **not retained**. In this finite-difference local lane, touching those
matrix tensors pushes the run into dense Muon-sized work and blows wall time up
too much for the local proof lane.

## Verification

Commands run:

```bash
cargo test -q -p psionic-runtime token_sampler_can_select_from_explicit_allowed_ids -- --nocapture
cargo test -q -p psionic-models promoted_runtime_tokenizer_refuses_unknown_non_eos_control_and_unused_ids -- --nocapture
cargo test -q -p psionic-train xtrain_promoted_parameter_golf_config_expands_the_inferable_budget -- --nocapture
cargo build -q -p psionic-serve --example xtrain_parameter_golf_train_infer
target/debug/examples/xtrain_parameter_golf_train_infer /tmp/psionic_xtrain_pgolf_eval_1774604300
```

Comparison baseline:

- previous retained report:
  `/tmp/psionic_xtrain_pgolf_eval_1774602400/xtrain_parameter_golf_train_infer_report.json`

Current retained report:

- `/tmp/psionic_xtrain_pgolf_eval_1774604300/xtrain_parameter_golf_train_infer_report.json`

## Results

### Quality

Quality improved substantially again:

- previous XTRAIN loss: `6.090883255004883`
- current XTRAIN loss: `4.380941867828369`
- loss improvement: `1.7099413871765137`

- previous XTRAIN BPB: `7.029829653303338`
- current XTRAIN BPB: `5.056290485711057`
- BPB improvement: `1.973539167592281`

Generated tokens also moved closer to the tiny task, though still not exact:

- previous XTRAIN generated tokens:
  `[7, 7, 8, 8, 8, 8, 7, 7]`
- current XTRAIN generated tokens:
  `[6, 8, 8, 8, 8, 7, 7, 8]`

That means the first emitted token moved from `7` to `6`, but the exact target
is still:

- `[5, 6, 7, 8, 1, 2, 3, 4]`

### Throughput

The decode fast path preserved the current throughput band and improved served
runtime slightly:

- previous direct runtime: `25.088079657342355 tok/s`
- current direct runtime: `24.852325009270356 tok/s`

- previous served runtime: `24.848751585620388 tok/s`
- current served runtime: `25.080392291692444 tok/s`

- direct/served parity: still `true`

So the honest throughput read is:

- served runtime improved
- direct runtime stayed effectively flat, slightly lower than the prior retained
  run and still within the same ~25 tok/s band

## Conclusion

The train/infer stack is now materially stronger than the prior retained pass:

- same working XTRAIN -> promoted PGOLF bundle -> direct/served inference path
- much better loss and BPB
- first-token behavior moved in the right direction
- served inference got slightly faster
- the local lane stayed on sparse surfaces and avoided the dense-matrix runtime cliff

The remaining gap is now almost entirely task fidelity in the tiny bounded
training lane, not infrastructure or bundle/runtime legitimacy.

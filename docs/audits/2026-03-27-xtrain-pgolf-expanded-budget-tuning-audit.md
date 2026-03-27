# XTRAIN PGOLF Expanded Budget Tuning Audit

Date: 2026-03-27

## Goal

Push the first XTRAIN-trained PGOLF-shaped model further now that the inference
path is stable and decode is gated to admitted tokenizer ids.

This pass changes training capacity only:

- larger XTRAIN step budget
- wider promoted coordinate budget
- no new inference semantics

## Change

Updated `ParameterGolfReferenceTrainingConfig::xtrain_promoted_general_small_decoder_baseline()`
and the promoted XTRAIN coordinate budget in
`crates/psionic-train/src/parameter_golf_reference.rs`.

New bounded XTRAIN posture:

- `max_steps: 8 -> 12`
- promoted trainable coordinate count: `16 -> 32`

The widened coordinate budget now trains more of the tiny task-relevant surface:

- `tok_emb.weight`: `8 -> 16` coordinates
- `skip_weights`: `1 -> 2`
- `blocks.0.attn.q_gain`: `2 -> 4`
- `blocks.0.attn_scale`: `1 -> 2`
- `blocks.0.mlp_scale`: `1 -> 2`
- `blocks.0.resid_mix`: `1 -> 2`
- `blocks.0.attn.c_q.weight`: `1 -> 2`
- `blocks.0.attn.proj.weight`: `1 -> 2`

## Verification

Commands run:

```bash
cargo test -q -p psionic-train xtrain_promoted_parameter_golf_config_expands_the_inferable_budget -- --nocapture
cargo build -q -p psionic-serve --example xtrain_parameter_golf_train_infer
target/debug/examples/xtrain_parameter_golf_train_infer /tmp/psionic_xtrain_pgolf_eval_1774602400
```

Retained reports used for comparison:

- prior token-admission run:
  `/tmp/psionic_xtrain_pgolf_eval_1774601500/xtrain_parameter_golf_train_infer_report.json`
- new expanded-budget run:
  `/tmp/psionic_xtrain_pgolf_eval_1774602400/xtrain_parameter_golf_train_infer_report.json`

## Results

Quality improved materially again:

- prior XTRAIN loss: `7.634324550628662`
- new XTRAIN loss: `6.090883255004883`
- loss improvement: `1.5434412956237793`

- prior XTRAIN BPB: `8.811201735783069`
- new XTRAIN BPB: `7.029829653303338`
- BPB improvement: `1.781372082479731`

Decode throughput also improved slightly on the direct runtime:

- prior direct runtime: `24.86396449115071 tok/s`
- new direct runtime: `25.088079657342355 tok/s`

Served runtime stayed in the same band:

- prior served runtime: `25.118654325792047 tok/s`
- new served runtime: `24.848751585620388 tok/s`

Direct/served output parity remains intact:

- `direct_and_served_match = true`

## Output Quality Boundary

The model is still not solving the full toy cycle exactly.

Current XTRAIN generated tokens:

- `[7, 7, 8, 8, 8, 8, 7, 7]`

Expected tokens:

- `[5, 6, 7, 8, 1, 2, 3, 4]`

So the honest state after this pass is:

- train/infer works
- decode stays inside the live tokenizer vocabulary
- loss and BPB are substantially better
- tok/s is still strong
- exact first-token fidelity is still missing

## Conclusion

The remaining problem is no longer “can Psionic train and infer one PGOLF-shaped
bundle” and no longer “is decode corrupted by reserved ids.” The remaining gap is
task-specific fidelity inside the tiny bounded training lane.

The next reasonable experiments are:

- increase task-relevant token-embedding coverage again, but without another
  large runtime hit
- add one tiny output-head-adjacent surface if the tied-embedding budget is still
  too weak
- inspect whether the local objective needs one stronger curriculum or sequence
  presentation change to force the `5` token first

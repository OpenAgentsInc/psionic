# HOMEGOLF Local Honest Loop And Artifact Prompt Audit

Date: 2026-03-28

## Summary

This audit records one follow-on HOMEGOLF integration pass after the earlier
2026-03-28 score-claim correction.

Two concrete gaps were closed:

- the local `HOMEGOLF` CUDA profile no longer inherits the single-H100
  `warmup_steps=20` prelude, so the default local challenge loop now keeps the
  public `600` second training cap honest on the training side
- the HOMEGOLF prompt utility can now load a persisted quantized artifact
  directly, so local text-generation checks no longer have to wait for the full
  score report to finish

What did **not** change:

- the public `parameter-golf` scoring contract still requires full FineWeb
  validation
- the current real retained full-validation score in-repo is still the RunPod
  single-H100 receipt at `6.306931747817168`
- the home `RTX 4080` lane is still not a practical path to a same-turn full
  PGOLF score because full validation remains hours long on that hardware

## Code Changes

Files changed:

- `crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_prompt.rs`

Behavioral changes:

- `apply_homegolf_local_cuda_profile()` now forces `warmup_steps=0`
- the new test
  `homegolf_local_cuda_defaults_drop_warmup_but_keep_wallclock_cap`
  freezes that contract
- `parameter_golf_homegolf_prompt` now supports an artifact-only mode:
  - `--artifact-path`
  - `--tokenizer-path`
  - `--model-variant`
  - optional `--run-id`, `--machine-profile`, `--prompt`,
    `--max-new-tokens`, and `--output`

That new prompt mode matters because the current trainer persists the final
`.st` artifact before the long roundtrip score pass completes.

## Validation

The following local validations passed:

```bash
bash scripts/check-parameter-golf-google-single-h100-lane.sh --report /tmp/parameter_golf_google_single_h100_operator_rehearsal_20260328.json
cargo check -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train
cargo check -q -p psionic-train --bin parameter_golf_homegolf_prompt
cargo test -q -p psionic-train homegolf_local_cuda_defaults_drop_warmup_but_keep_wallclock_cap -- --nocapture
```

The Google checker pass matters here because the HOMEGOLF and remote PGOLF
loops share the same operational score path: real package materialization,
real bringup, and real operator wrappers.

## Live `archlinux` HOMEGOLF Run

Machine:

- `archlinux`
- `NVIDIA GeForce RTX 4080`

Command:

```bash
PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_ACCUM_STEPS=64 \
cargo run -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/psionic_homegolf_runs/homegolf_baseline_actual_20260328.json \
  1 \
  roundtrip_only
```

Observed training facts from the live log:

- `machine_profile=homegolf_local_cuda`
- `warmup_steps=0`
- `grad_accum_steps=64`
- `validation_eval_mode=non_overlapping`
- `validation_batch_sequences=64`
- `mean_microbatch_loss=8.29203224`
- `train_time_ms=278709`

Observed artifact facts:

- `compressed_model_bytes=4073137`
- persisted artifact path:
  `/tmp/psionic_homegolf_runs/homegolf_baseline_actual_20260328.final_model.st`

## Immediate Text Generation From The Exported Artifact

With the new artifact-only prompt mode, the exported local HOMEGOLF artifact was
prompted immediately without waiting for the score report to finish.

Commands:

```bash
cargo run -q -p psionic-train --bin parameter_golf_homegolf_prompt -- \
  --artifact-path /tmp/homegolf_baseline_actual_20260328.final_model.st \
  --tokenizer-path /tmp/fineweb_1024_bpe.model \
  --model-variant baseline_sp1024_9x512 \
  --run-id homegolf_baseline_actual_20260328 \
  --machine-profile homegolf_local_cuda \
  --prompt "the meaning of life is" \
  --max-new-tokens 32 \
  --output /tmp/homegolf_baseline_actual_20260328_prompt.json

cargo run -q -p psionic-train --bin parameter_golf_homegolf_prompt -- \
  --artifact-path /tmp/homegolf_baseline_actual_20260328.final_model.st \
  --tokenizer-path /tmp/fineweb_1024_bpe.model \
  --model-variant baseline_sp1024_9x512 \
  --run-id homegolf_baseline_actual_20260328 \
  --machine-profile homegolf_local_cuda \
  --prompt "Once upon a time" \
  --max-new-tokens 24 \
  --output /tmp/homegolf_baseline_actual_20260328_prompt_story.json
```

Observed outputs:

- prompt `the meaning of life is`
  - generated text:
    `iiildKild loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc`
- prompt `Once upon a time`
  - generated text:
    `ildKiild partiild loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc loc`

This is not good model quality, but it **is** real text generation from the
actual exported HOMEGOLF artifact produced in this iteration.

## Why The Local Lane Still Does Not Yield A Same-Turn PGOLF Score

The retained live log also showed the full score boundary clearly:

- roundtrip evaluation started with `sequences=60568`
- `batch_sequences=64`
- `947` non-overlapping validation batches were scheduled
- batch `1/947` took `25086 ms`

At that observed rate, the full roundtrip score pass is still on the order of
hours on the `4080`.

So the local lane now has:

- honest training-time semantics
- immediate artifact promptability
- real artifact-byte accounting

But it still does not have:

- a practical same-turn full FineWeb score path on local `RTX 4080`

## Improvement Over The Earlier 2026-03-28 Correction

Compared with the earlier correction audit, this pass improves the integration
in two concrete ways:

1. The default local challenge loop is no longer dishonest about the 10-minute
   training budget.
2. The train-to-infer proof can now be checked directly from the live exported
   artifact instead of waiting for the long score closeout.

That does not improve the retained actual PGOLF score yet, but it does remove
two misleading seams in the HOMEGOLF iteration loop.

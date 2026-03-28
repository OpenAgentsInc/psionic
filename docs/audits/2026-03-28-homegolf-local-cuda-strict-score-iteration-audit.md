# HOMEGOLF Local-CUDA Strict Score Iteration Audit

Date: 2026-03-28

## Summary

This audit records the first local `RTX 4080` HOMEGOLF strict-score iteration
that reached a finite exact-score result on the real FineWeb `SP1024` lane.

It also records two correctness fixes that were required before the local-CUDA
fit search was trustworthy:

- the HOMEGOLF runner no longer overwrites the local-CUDA validation-batch fit
  profile when `sliding_window:64` is selected
- the single-device trainer now reuses the shared batch-geometry validator, so
  non-divisor `grad_accum_steps` values are refused instead of silently
  shrinking the real train-token budget

## Code Changes

Files changed:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `crates/psionic-train/src/parameter_golf_reference.rs`
- `crates/psionic-train/src/parameter_golf_single_h100_training.rs`

Behavioral effect:

- `validation_eval_mode=sliding_window:64` now preserves the already-selected
  local-CUDA validation batch unless the previous value was still just the old
  default
- HOMEGOLF local-CUDA fit trials now reject invalid profiles like
  `grad_accum_steps=24` instead of running a token-dropped approximation

## Validation

Build validation that passed:

```bash
cargo check -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train
```

Targeted test attempt:

```bash
cargo test -q -p psionic-train homegolf_local_cuda_rejects_non_sequence_exact_grad_accum -- --nocapture
```

Current result on this Mac:

- blocked by an unrelated existing arm64/macOS linker failure in the broader
  `psionic-train` test binary

That linker issue did not block the real remote HOMEGOLF runs below.

## Remote Fit Search

Machine used:

- `archlinux`
- `NVIDIA GeForce RTX 4080`
- exact dataset root:
  `~/code/parameter-golf/data/datasets/fineweb10B_sp1024`
- exact tokenizer path:
  `~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`

Observed initial blocker:

- unrelated `ollama.service` runners temporarily occupied about `11.7 GiB`
  before the clean-card fit search

Observed divisor-safe fit results after the card cleared:

- `grad_accum_steps=16`
  - exact lane still OOM
- `grad_accum_steps=32`
  - live-only train step admitted
  - strict scored run still OOMed partway through training
- `grad_accum_steps=64`
  - strict scored run admitted one full optimizer step and the strict scorer

## First Finite Strict Score

Command:

```bash
PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_ACCUM_STEPS=64 \
PSIONIC_PARAMETER_GOLF_HOMEGOLF_VALIDATION_BATCH_SEQUENCES=8 \
cargo run -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/psionic_homegolf_runs/homegolf_exact_fitprobe_g64_v8_ttt4_report.json \
  1 both sliding_window:64 legal_score_first_ttt:batch_sequences=4
```

Observed strict result from the live trainer logs:

- `grad_accum_steps=64`
- `validation_batch_sequences=8`
- `score_first_ttt.batch_sequences=4`
- `mean_microbatch_loss=8.29200554`
- `train_time_ms=321606`
- `final_validation_mean_loss=11.74549673`
- `final_validation_bits_per_byte=6.80014851`
- `final_validation_elapsed_ms=187517`

Observed combined train-plus-score wallclock before export/closeout:

- `321606 + 187517 = 509123 ms`

That is inside the `600000 ms` HOMEGOLF wallclock cap.

## Improvement Over The Previous HOMEGOLF Runnable Truth

The previously retained exact-family HOMEGOLF proof lane reported:

- `final_validation_bits_per_byte=9.93265382277841`

The new strict local-CUDA observed score reached:

- `final_validation_bits_per_byte=6.80014851`

Observed improvement:

- `3.13250531277841` BPB better than the old bounded exact-family proof lane

## Honest Boundary

This iteration is materially stronger than the old strict-lane refusal and the
older bounded proof lane.

It is still not fully closed.

What is true now:

- the exact local-CUDA strict scorer can reach a finite BPB on the home `4080`
- one divisor-safe strict score path is now inside the `10` minute envelope
  before export/closeout
- the runner and trainer no longer hide two fit-profile correctness bugs

What is still not true:

- the exact local-CUDA strict run did not finish writing its final retained
  report/artifact pair within this iteration window
- exact prompt-generation proof on that scored local-CUDA artifact is therefore
  still pending
- this is still not a public-leaderboard-equivalent result

## Next Moves

1. Make the post-score export/closeout path emit the retained report/artifact
   promptly after the finite strict score is already known.
2. Run `parameter_golf_homegolf_prompt` directly on that retained scored
   artifact to prove real text generation on the exact local-CUDA strict lane.
3. Keep the current `g64 / v8 / ttt4` posture as the baseline until one
   smaller, faster, still-finite strict profile beats it honestly.

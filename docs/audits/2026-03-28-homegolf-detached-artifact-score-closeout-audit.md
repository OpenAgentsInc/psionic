# HOMEGOLF Detached Artifact Score Closeout Audit

Date: 2026-03-28

## What Changed

The local HOMEGOLF CUDA operator loop now supports a detached score-closeout
path.

Code landed at `psionic` commit `8660af4c`:

- `crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_artifact_score.rs`
- `scripts/run-parameter-golf-homegolf-local-cuda.sh`
- `scripts/queue-parameter-golf-homegolf-local-cuda.sh`
- `scripts/wait-parameter-golf-homegolf-artifact-score.sh`

The trainer now accepts one additional explicit final-validation posture:

- `artifact_only`

That mode preserves the normal training contract and exported quantized artifact
path, but skips both:

- final live-model validation
- final quantized roundtrip validation

The new detached scorer restores the exported artifact into the full-precision
reference family and measures non-overlapping validation as a separate closeout
receipt.

## Why This Matters

The previous retained operator shape was still serialized:

- `20260328d` finished its honest one-step train phase in `274570ms`
- then the same GPU entered `947` roundtrip validation batches
- at `2026-03-28 06:38:27 CDT`, the retained active run had only reached
  `batch=60/947` with `elapsed_ms=1476055`

So the real training bottleneck was no longer artifact export or text
generation. It was score closeout occupying the same training device for hours.

The detached closeout path improves on the previous `20260328e` queue posture
because it removes full validation from the critical path of the next training
attempt.

## Operator Surface

New operator surfaces:

- trainer validation mode:
  `artifact_only`
- runner flag:
  `scripts/run-parameter-golf-homegolf-local-cuda.sh --attach-detached-score-closeout`
- detached watcher:
  `scripts/wait-parameter-golf-homegolf-artifact-score.sh`

Queued next run after this change:

- run id:
  `homegolf-baseline-g64-stepcap2-clip1-lr075-artifactonly-600s-20260328f`
- queue pid:
  `778424`
- queue log:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328f_repo.log`

Queued posture:

- `challenge_max_steps=2`
- `grad_clip_norm=1.0`
- `learning_rate_scale=0.75`
- `final_validation_mode=artifact_only`
- `validation_eval_mode=non_overlapping`
- prompt closeout attached
- detached score closeout attached

## Validation

Local validation:

- `bash -n scripts/run-parameter-golf-homegolf-local-cuda.sh scripts/queue-parameter-golf-homegolf-local-cuda.sh scripts/wait-parameter-golf-homegolf-artifact-score.sh`
- `cargo check -q -p psionic-train --bin parameter_golf_homegolf_artifact_score --bin parameter_golf_homegolf_single_cuda_train`
- `cargo test -q -p psionic-train --lib validation_mode_parser_accepts_all_supported_labels`
- `cargo test -q -p psionic-train --lib live_validation_checkpoints_skip_step_zero_but_keep_later_and_final_validation`

Remote validation:

- `archlinux` pulled `8660af4c`
- the old queue process was replaced with the new repo-owned queued
  `artifact_only` run
- a detached scorer smoke pass was launched against the live exported
  `20260328d` artifact through a synthetic report rooted in the retained
  single-H100 schema

Smoke output target:

- `/home/christopherdavid/scratch/psionic_homegolf_runs/detached_score_smoke_20260328/artifact_score_report.json`

At `2026-03-28 06:38:27 CDT`, that smoke output had not completed yet.

## Honest Boundary

This change does not create a new completed PGOLF score by itself.

Current retained truth after this audit:

- the retained actual public PGOLF score is still `6.306931747817168`
- `20260328d` is still completing inline full validation
- `20260328f` is queued to use detached score closeout after artifact export
- XTRAIN did not change in this iteration

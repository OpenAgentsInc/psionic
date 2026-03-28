# HOMEGOLF EMA Export Queue Audit

Date: 2026-03-28

## Scope

This audit records the next local HOMEGOLF queue extension after the chained
raw-surface LR sweep.

The chained sweep solved queue depth, but every queued run still exported the
raw post-step model surface.

That left one obvious missing quality comparison:

- same honest `600s` loop
- same detached score closeout
- different final export surface

## What Changed

Updated:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `scripts/run-parameter-golf-homegolf-local-cuda.sh`

New operator controls:

- `PSIONIC_PARAMETER_GOLF_EMA_DECAY=<f32>`
- `--ema-decay <f32>`
- `--final-model-surface <surface>`

That landed at commit `5f6b2f10`.

The trainer already supported explicit final-model-surface selection. The new
work makes that surface reachable from the repo-owned local HOMEGOLF runner and
binds one explicit EMA config into the same path.

## Why This Improves The System

Before this change:

- the queued sweep could compare LR variants only
- every queued candidate still exported the raw surface

After this change:

- the queued sweep can include one EMA-backed export without leaving the same
  honest local HOMEGOLF control loop
- prompt proof and detached score closeout remain attached
- the queue now carries one real quality hypothesis beyond pure LR reduction

## Live Queue Extension

At `2026-03-28 06:44:43 CDT` on `archlinux`, the new queued EMA run was:

- run id:
  `homegolf-baseline-g64-stepcap2-clip1-lr050-ema997-artifactonly-600s-20260328i`
- queue pid:
  `781762`
- queue log:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328i_repo.log`

Queued settings:

- `challenge_max_steps=2`
- `grad_clip_norm=1.0`
- `learning_rate_scale=0.50`
- `ema_decay=0.997`
- `final_model_surface=ema`
- `final_validation_mode=artifact_only`
- `validation_eval_mode=non_overlapping`

This run waits behind:

- `/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap2-clip1-lr035-artifactonly-600s-20260328h/train.pid`

## Honest Boundary

This audit does not add a completed new score receipt.

Current retained truth:

- the active inline-validation run `20260328d` was still alive
- the raw-surface detached-score sweep remained queued as `f`, `g`, and `h`
- the EMA-backed comparison run `i` is now queued behind that chain
- the retained actual public PGOLF score is still `6.306931747817168`
- XTRAIN did not change in this pass

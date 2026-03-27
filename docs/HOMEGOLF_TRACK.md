# HOMEGOLF Track

> Status: canonical HOMEGOLF benchmark-track contract, updated 2026-03-27
> after landing `HOMEGOLF-0`.

This document freezes the first honest Psionic answer to:

- how to compete against Parameter Golf from a custom clustered home setup
- without pretending the hardware is the same as the public leaderboard

## What HOMEGOLF Means

HOMEGOLF keeps as much of the public Parameter Golf contract as possible:

- exact dense `9x512` baseline family as the initial target
- exact FineWeb `SP1024` identity in strict mode
- exact `10` minute wallclock cap
- exact tokenizer-agnostic `val_bpb` score reporting
- exact contest-style artifact-byte accounting toward the decimal `16MB` cap
- required train-to-infer bundle closure

HOMEGOLF changes one thing only:

- the compute posture is a declared custom cluster instead of public `8xH100`

So the honest reading is:

- HOMEGOLF can be compared against the public baseline and leaderboard
- HOMEGOLF is not a public leaderboard-equivalent result unless rerun under the
  official `8xH100` posture

## Canonical Contract

Machine-readable contract:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_track_contract.json`

Generator:

- `crates/psionic-train/src/parameter_golf_homegolf.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_track_contract.rs`

Checker:

- `scripts/check-parameter-golf-homegolf-track-contract.sh`
- `scripts/check-parameter-golf-homegolf-dense-baseline-surface.sh`
- `scripts/check-parameter-golf-homegolf-dense-bundle-proof.sh`

## Current Baseline Surfaces

The HOMEGOLF contract deliberately binds to existing Psionic surfaces instead of
inventing a second Parameter Golf stack:

- exact dense trainer entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_single_h100_train.rs`
- exact dense trainer implementation:
  `crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- promoted runtime bundle handoff:
  `crates/psionic-models/src/parameter_golf_promoted_bundle.rs`
- strict family contract:
  `docs/PARAMETER_GOLF_PROMOTED_FAMILY_CONTRACT.md`
- contest-style artifact accounting:
  `docs/PARAMETER_GOLF_NON_RECORD_SUBMISSION.md`

## First Honest Dense Baseline Surface

HOMEGOLF now has one canonical retained dense-baseline surface proving that the
exact public naive-baseline trainer already runs inside Psionic:

- source dense run:
  `fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_training_report.json`
- HOMEGOLF baseline surface report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_baseline_surface.json`
- HOMEGOLF baseline surface generator:
  `crates/psionic-train/src/parameter_golf_homegolf_dense_baseline.rs`
- HOMEGOLF baseline surface entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_dense_baseline_surface.rs`

What this freezes:

- the exact `SP1024` `9x512` config, not an approximation
- one completed dense single-device Psionic run
- real retained `val_loss`, `val_bpb`, train-token, wallclock, and artifact-byte
  outputs
- the honest claim boundary that this is the first HOMEGOLF dense baseline
  surface, not yet a mixed-device 10-minute HOMEGOLF score

## Train-To-Infer Closure

HOMEGOLF now also has one retained train-to-infer closure proof for the exact
`9x512` family:

- retained proof report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json`
- proof runner:
  `crates/psionic-serve/examples/parameter_golf_homegolf_dense_bundle_proof.rs`
- checker:
  `scripts/check-parameter-golf-homegolf-dense-bundle-proof.sh`

What this proves:

- Psionic can emit a real promoted runtime bundle from an exact-family
  HOMEGOLF-compatible bounded run
- that bundle loads directly for inference
- that same bundle loads through `psionic-serve`
- direct and served generation agree on the retained proof prompt

What it does not prove:

- that the retained single-H100 dense source report already shipped committed
  model bytes
- that the public FineWeb/SP1024 scorepath now closes locally
- that mixed-device `10` minute HOMEGOLF execution is already solved

## First Clustered HOMEGOLF Surface

HOMEGOLF now also has a first honest clustered score surface:

- retained clustered surface report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_clustered.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_clustered_run_surface.rs`
- checker:
  `scripts/check-parameter-golf-homegolf-clustered-run-surface.sh`

What this proves:

- one real two-device home-Tailnet admitted run stayed inside the `600s` budget
- the current admitted device inventory and per-device contribution receipts are
  frozen in one HOMEGOLF report
- that same report binds the current exact-family HOMEGOLF train-to-infer bundle
  proof and final `val_bpb`
- operators can now compare real clustered home-device work against exact-family
  score progress without pretending the dense exact-baseline trainer already ran
  live across the admitted home cluster

What it does not prove:

- that one exact dense mixed-device home-cluster run directly produced the
  scored bundle
- that the current exact dense HOMEGOLF trainer already has live MLX plus home
  CUDA execution parity

## Comparison Policy

Allowed wording:

- `public-baseline comparable`
- `not public-leaderboard equivalent`

Forbidden shortcut:

- calling a HOMEGOLF result a public leaderboard-equivalent score without
  rerunning it under the official `8xH100` execution posture

## Hardware Manifest Law

The HOMEGOLF contract admits one mixed-device cluster family:

- local Apple Silicon Metal workstation
- home consumer CUDA node
- secondary Apple Silicon peer when stable
- optional future H100 nodes

The hardware mix may change over time.

The score semantics may not.

That means:

- same model family
- same scorepath
- same wallclock budget
- same artifact accounting
- richer declared hardware manifest

## Current Honest Boundary

HOMEGOLF is frozen as a contract now, but one important surface is still
blocked:

- the exact dense PGOLF baseline is still H100-only in its live trainer
  execution surface for retained runs
- the first honest dense baseline surface is real now, but it is not yet a
  clustered `10` minute HOMEGOLF score

So HOMEGOLF is ready to guide implementation, but it does not yet claim that
the dense strict-baseline run can already execute across the full admitted home
cluster.

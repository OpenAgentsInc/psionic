# HOMEGOLF Track

> Status: canonical HOMEGOLF benchmark-track contract, updated 2026-03-28
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
- `scripts/check-parameter-golf-homegolf-strict-challenge-lane.sh`
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

## Strict Challenge Runnable Lane

HOMEGOLF now has one canonical strict runnable lane surface:

- retained lane report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_strict_challenge_lane.json`
- lane generator:
  `crates/psionic-train/src/parameter_golf_homegolf_strict_challenge.rs`
- lane entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_strict_challenge_lane.rs`
- latest audit:
  `docs/audits/2026-03-28-homegolf-strict-challenge-path-identity-audit.md`
- checker:
  `scripts/check-parameter-golf-homegolf-strict-challenge-lane.sh`

What this proves:

- the canonical runnable HOMEGOLF lane now binds the strict PGOLF challenge
  overlay instead of the old general local-reference profile
- the emitted lane surface keeps:
  - exact challenge tokenizer identity
  - exact FineWeb `SP1024` data-lane requirement
  - `sliding_window:64` evaluation
  - legal score-first TTT
  - contest bits-per-byte accounting
  - exact `16,000,000`-byte artifact-cap law
- fake temp paths with the right final basename no longer produce
  `ready_to_execute`; the strict lane now requires the real expected
  `~/code/parameter-golf/...` path identity
- missing challenge inputs now produce an explicit typed refusal instead of a
  silent local-reference fallback

What it does not prove:

- that the exact challenge inputs are already present on every machine
- that the live dense strict HOMEGOLF run has already been retained
- that the mixed-device home cluster already produces the scored bundle

## Current Local-CUDA Score Boundary

HOMEGOLF's current home `RTX 4080` lane is still useful, but the 2026-03-28
follow-up review showed that the earlier "local strict score" language was too
strong.

- latest correction audit:
  `docs/audits/2026-03-28-homegolf-local-cuda-strict-score-iteration-audit.md`
- latest follow-on integration audit:
  `docs/audits/2026-03-28-homegolf-local-honest-loop-and-artifact-prompt-audit.md`
- latest competitive override audit:
  `docs/audits/2026-03-28-homegolf-local-competitive-override-lane-audit.md`
- latest competitive BigramHash fix audit:
  `docs/audits/2026-03-28-homegolf-competitive-bigram-input-fix-audit.md`
- latest step-zero validation fix audit:
  `docs/audits/2026-03-28-homegolf-initial-validation-step-zero-fix-audit.md`
- latest wallclock projection refusal audit:
  `docs/audits/2026-03-28-homegolf-local-wallclock-projection-refusal-audit.md`

What is true now:

- the runner and trainer fixes from that audit still stand:
  - `sliding_window:64` no longer silently overwrites the local-CUDA
    validation-batch fit profile
  - invalid local fit trials like `grad_accum_steps=24` are now refused by the
    shared batch-geometry validator instead of silently dropping train tokens
- the default local HOMEGOLF CUDA profile now forces `warmup_steps=0` while
  preserving the public `600` second wallclock cap
- one bounded local one-step fit posture is real on the `4080`:
  - `max_steps=1`
  - `warmup_steps=0`
  - `grad_accum_steps=64`
  - `validation_batch_sequences=64`
- that bounded local one-step rerun reached:
  - `train_time_ms=278709`
  - `mean_microbatch_loss=8.29203224`
- that same rerun exported a real local quantized artifact:
  - `compressed_model_bytes=4073137`
  - persisted path:
    `/tmp/psionic_homegolf_runs/homegolf_baseline_actual_20260328.final_model.st`
- HOMEGOLF prompt validation is no longer blocked on full report closeout:
  - `parameter_golf_homegolf_prompt` can now load the persisted artifact
    directly with explicit `--artifact-path`, `--tokenizer-path`, and
    `--model-variant`
- one artifact-only prompt proof from that local export now exists:
  - prompt `the meaning of life is`
  - generated text begins:
    `iiildKild loc loc loc ...`
- the local competitive entrypoint is now tunable without source edits for
  honest `4080` iteration:
  - `PSIONIC_PARAMETER_GOLF_MODEL_VARIANT=competitive_homegolf_v1`
  - `PSIONIC_PARAMETER_GOLF_DISABLE_SCORE_FIRST_TTT=1`
  - `PSIONIC_PARAMETER_GOLF_FINAL_MODEL_SURFACE=raw|ema|swa`
  - `PSIONIC_PARAMETER_GOLF_SWA_EVERY_STEPS=<n>`
- that matters because the retained competitive defaults still enable
  leaderboard-style score-first TTT and `final_model_surface=swa` with the
  default `every_steps=50`, which can waste or under-sample the local `600`
  second loop on `RTX 4080`
- the local competitive device-resident CUDA lane now also binds the optional
  BigramHash graph input correctly during both validation and training, so
  `competitive_homegolf_v1` no longer dies immediately with
  `missing input tensor t1` on the first validation batch
- the local HOMEGOLF control loop no longer fires a full live validation pass
  at `step=0`, so the `600` second training budget now starts with real
  optimizer work instead of an accidental full validation sweep
- periodic live validation now begins only after real training progress, while
  final raw-surface validation semantics remain unchanged
- the local competitive exact lane now fails fast when the declared `600`
  second wallclock is impossible on the current `4080` posture
- the latest retained competitive refusal on `archlinux` proved that directly:
  - posture:
    `competitive_homegolf_v1 + grad_accum_steps=64 + ema + roundtrip_only + non_overlapping + no-TTT`
  - observed first micro-step:
    `observed_step_wallclock_ms=29584`
  - projected exact full step:
    `projected_full_step_wallclock_ms=1893376`
  - remaining training wallclock:
    `600000`
  - retained report:
    `/tmp/psionic_homegolf_runs/homegolf_competitive_nottt_ema_projection_refusal_8dad683d_20260328.json`
- that matters because the exact local loop no longer burns tens of minutes on
  an already-impossible first optimizer step just to rediscover that the lane
  is not `10`-minute honest on this hardware

What is not true:

- the observed `6.80014851` BPB from the earlier 2026-03-28 local receipts is
  not an actual PGOLF score
- the public `parameter-golf` README still requires scoring on the full
  FineWeb validation split, and the current single-device trainer loads that
  full split
- the leaderboard-default `legal_score_first_ttt` overlay expands to
  `chunk_tokens=32768` and `epochs=3`, which is also not `10` minute honest on
  that local lane
- the current full local roundtrip score path is still not practical on the
  `4080`; one observed `non_overlapping` validation pass scheduled `947` full
  batches, needed about `58979 ms` for batch `1/947`, and was started by the
  old pre-fix `step=0` validation behavior
- the current exact competitive training path is also not practical on the
  `4080`; the latest live refusal projected step `1` to about `1893` seconds
  after just `1/64` micro-steps
- no new retained local full-validation report/artifact pair was produced on
  the `4080` during this iteration loop

Current real full-score truth retained in-repo is still:

- `fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_training_report.json`
  - `final_validation_bits_per_byte=6.306931747817168`
  - `evaluated_token_count=62021632`
  - `compressed_model_bytes=4732744`

That retained score is still a real full-validation single-H100 result, not a
public `8xH100` leaderboard-equivalent run.

## Train-To-Infer Closure

HOMEGOLF now also has one retained train-to-infer closure proof for the exact
`9x512` family:

- retained proof report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json`
- proof runner:
  `crates/psionic-serve/examples/parameter_golf_homegolf_dense_bundle_proof.rs`
- latest rerun audit:
  `docs/audits/2026-03-27-homegolf-fresh-rerun-results-audit.md`
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

After `HOMEGOLF-8`, this is no longer the canonical runnable contest-lane
surface. It remains the secondary exact-family train-to-infer proof, while the
strict challenge runnable lane is the canonical HOMEGOLF command surface.

## Live Dense HOMEGOLF Surface

HOMEGOLF now also has a first upgraded live dense mixed-device score surface:

- retained clustered surface report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_clustered.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_clustered_run_surface.rs`
- audit:
  `docs/audits/2026-03-27-homegolf-live-dense-run-surface.md`
- checker:
  `scripts/check-parameter-golf-homegolf-clustered-run-surface.sh`

What this proves:

- one real same-job dense mixed-device runtime surface now exists inside the
  HOMEGOLF track
- the retained dense participants are:
  - one local Apple Silicon MLX rank
  - one optional-H100 CUDA submesh
- the retained dense surface keeps explicit per-device contribution receipts and
  dense wallclock facts inside one HOMEGOLF report
- that same report now binds the exact dense challenge export bytes and final
  roundtrip `val_bpb`
- the retained surface keeps:
  - `observed_cluster_wallclock_ms=3611`
  - `final_validation_bits_per_byte=6.306931747817168`
  - `model_artifact_bytes=4732744`

What it does not prove:

- admitted home-RTX dense closure on the local home cluster
- one single retained run id that already binds the mixed-device dense runtime
  receipts and the final scored export bytes in one artifact family
- official public-leaderboard equivalence

## Canonical Score-Relevant Dense Runtime

HOMEGOLF now also has one canonical score-relevant dense runtime surface:

- retained runtime report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_score_relevant_runtime.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_score_runtime.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_score_relevant_runtime.rs`
- audit:
  `docs/audits/2026-03-27-homegolf-score-relevant-runtime-audit.md`
- checker:
  `scripts/check-parameter-golf-homegolf-score-relevant-runtime.sh`

What this proves:

- the canonical HOMEGOLF dense runtime is no longer a symbolic bounded proof
  updater
- the retained mixed-device dense lane keeps resident state across real dense
  steps
- the runtime now publishes:
  - cluster steps per second
  - cluster train tokens per second
  - per-device dense throughput
  - average phase timing for CUDA work, MLX work, bridge time, and optimizer
    time
  - projected `600s` training volume
- the retained runtime clears more than one full training-dataset pass inside
  the `600s` cap, so quality comparisons are now grounded in real dense
  training volume instead of symbolic proof traffic

What it does not prove:

- admitted Apple-plus-home-RTX dense closure
- one locally produced scored bundle from that admitted home cluster
- public-leaderboard-equivalent hardware or score

## Public Comparison Report

HOMEGOLF now also has one frozen public comparison surface:

- retained comparison report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_public_comparison.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_comparison.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_public_comparison.rs`
- checker:
  `scripts/check-parameter-golf-homegolf-public-comparison.sh`

What this keeps explicit:

- delta from the public naive baseline
- delta from the current public best leaderboard row
- exact `val_bpb` delta
- exact surfaced artifact-byte delta
- unchanged `600s` wallclock-cap posture
- required language:
  - `public-baseline comparable`
  - `not public-leaderboard equivalent`

Current boundary:

- the public references are frozen from the reviewed Parameter Golf repo
  snapshot

## Competitive Exact-Lane Ablation Surface

HOMEGOLF now also has one retained competitive exact-lane ablation surface:

- retained ablation report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_competitive_ablation.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_competitive_ablation.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_competitive_ablation.rs`
- audit:
  `docs/audits/2026-03-27-homegolf-competitive-ablation-audit.md`
- checker:
  `scripts/check-parameter-golf-homegolf-competitive-ablation.sh`

What this proves:

- the exact HOMEGOLF lane is no longer hard-wired to only the naive baseline
  model shape
- the canonical exact trainer now admits one explicit best-known competitive
  variant: `competitive_homegolf_v1`
- that best-known exact-lane variant already wires in:
  - `BigramHash`
  - partial RoPE
  - deep-layer XSA
  - `LeakyReLU(0.5)^2`
  - late-layer VE
  - EMA
  - SWA sourced from EMA
  - legal score-first TTT
  - competitive export defaults
- the retained report also keeps explicit refusals for techniques that are still
  out of scope on the exact HOMEGOLF lane today

What it does not prove:

- that `competitive_homegolf_v1` has already produced the best retained
  HOMEGOLF score on local hardware
- that every public-winning technique is already admitted into the exact
  Psionic trainer
- that the admitted home-cluster dense runtime has already been retuned around
  this competitive variant
- that the current local competitive overrides already yielded one retained
  improved full-validation PGOLF report
- the HOMEGOLF side now compares using the live dense mixed-device surface, the
  canonical score-relevant runtime report, and the retained exact dense
  challenge export bytes
- the result is materially stronger than the older open-adapter composition,
  but still not near the public leaderboard on score

## Artifact Accounting Report

HOMEGOLF now also has one explicit counted-byte surface:

- retained accounting report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_artifact_accounting.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_accounting.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_artifact_accounting.rs`
- checker:
  `scripts/check-parameter-golf-homegolf-artifact-accounting.sh`

What this keeps explicit:

- counted code bytes
- scored model-artifact bytes
- total counted bytes
- exact delta versus the `16,000,000`-byte cap
- explicit pass/refusal status
- the under-cap dense export remains bound to the same exact-family train-to-
  infer proof surface already used for direct runtime and `psionic-serve`
  closure

Current truth:

- the current HOMEGOLF accounting answer is now a pass, not a refusal
- Psionic's current counted code plus the retained compressed dense export stays
  inside the contest budget
- the retained counted values are:
  - `counted_code_bytes=7188700`
  - `scored_model_artifact_bytes=4732744`
  - `total_counted_bytes=11921444`
  - `cap_delta_bytes=-4078556`
- the retained under-cap export stays compatible with:
  - direct exact-family inference
  - `psionic-serve`
  - the retained exact dense bundle proof:
    `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json`

## Multi-Seed Package

HOMEGOLF now also has one retained repeated-run package:

- retained package report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_multiseed_package.json`
- retained per-run receipts:
  - `fixtures/parameter_golf/reports/homegolf_multiseed/parameter_golf_homegolf_dense_bundle_proof_seed_000.json`
  - `fixtures/parameter_golf/reports/homegolf_multiseed/parameter_golf_homegolf_dense_bundle_proof_seed_001.json`
  - `fixtures/parameter_golf/reports/homegolf_multiseed/parameter_golf_homegolf_dense_bundle_proof_seed_002.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_multiseed_package.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_multiseed_package.rs`
- checker:
  `scripts/check-parameter-golf-homegolf-multiseed-package.sh`
- audit:
  `docs/audits/2026-03-27-homegolf-multiseed-package-audit.md`

What this keeps explicit:

- three repeated exact HOMEGOLF proof-lane receipts
- exact per-run `val_bpb`, model bytes, descriptor/tokenizer digests, and
  direct-versus-served parity
- mean and spread across the repeated package
- mean delta versus the public naive baseline and current public best
- the stronger-claim boundary:
  - `public-baseline comparable`
  - `not public-leaderboard equivalent`
  - no beat claim support yet

Current truth:

- the current HOMEGOLF repeated package is reproducibility-grade, not
  competitiveness-grade
- the repeated package is deterministic today:
  - `mean_validation_bits_per_byte=9.93265382277841`
  - `stddev_validation_bits_per_byte=0.0`
- that zero spread is honest because the current exact-family proof lane is
  deterministic
- the repeated package is strong enough for honest public-comparison language,
  but not strong enough for win-style contest rhetoric

## Mixed Hardware Manifest Example

HOMEGOLF now also has one explicit retained mixed-hardware manifest surface:

- retained manifest:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_mixed_hardware_manifest.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_manifest.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_mixed_hardware_manifest.rs`
- checker:
  `scripts/check-parameter-golf-homegolf-mixed-hardware-manifest.sh`

What it keeps explicit:

- the current admitted Apple Silicon and consumer CUDA nodes
- the secondary Apple Silicon peer as an optional offline member
- one optional future H100 slot
- unchanged `600s` wallclock semantics
- unchanged `16,000,000`-byte artifact accounting semantics
- unchanged comparison language:
  - `public-baseline comparable`
  - `not public-leaderboard equivalent`

Current truth:

- HOMEGOLF now has one committed mixed-manifest example that includes a future
  H100 slot
- the H100 slot is admitted without creating a second benchmark philosophy
- the manifest is no longer the only H100 surface:
  - the retained live dense HOMEGOLF run already includes `optional_h100_node`
    in `fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json`

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

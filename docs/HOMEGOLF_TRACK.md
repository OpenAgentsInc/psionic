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

## Strict Challenge Preflight Lane

HOMEGOLF now has one canonical strict preflight lane surface:

- retained lane report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_strict_challenge_lane.json`
- lane generator:
  `crates/psionic-train/src/parameter_golf_homegolf_strict_challenge.rs`
- lane entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_strict_challenge_lane.rs`
- latest audit:
  `docs/audits/2026-03-28-homegolf-strict-preflight-semantics-and-local-entrypoint-audit.md`
- checker:
  `scripts/check-parameter-golf-homegolf-strict-challenge-lane.sh`

What this proves:

- the canonical strict HOMEGOLF preflight now binds the strict PGOLF challenge
  overlay instead of the old general local-reference profile
- the emitted lane surface keeps:
  - exact challenge tokenizer identity
  - exact FineWeb `SP1024` data-lane requirement
  - `sliding_window:64` evaluation
  - legal score-first TTT
  - contest bits-per-byte accounting
  - exact `16,000,000`-byte artifact-cap law
- fake temp paths with the right final basename no longer produce
  `preflight_satisfied`; the strict lane now requires the real expected
  `~/code/parameter-golf/...` path identity
- when exact inputs are present, the strict lane now reports
  `preflight_satisfied` instead of `ready_to_execute`, because runtime
  viability still belongs to the actual HOMEGOLF trainer path
- the retained strict command template now targets
  `parameter_golf_homegolf_single_cuda_train` instead of
  `parameter_golf_single_h100_train`
- missing challenge inputs now produce an explicit typed refusal instead of a
  silent local-reference fallback

What it does not prove:

- that the exact challenge inputs are already present on every machine
- that the current host can already satisfy the `600` second runtime contract
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
- the local HOMEGOLF CUDA lane now also has one repo-owned scratch-first
  operator wrapper:
  - `scripts/run-parameter-golf-homegolf-local-cuda.sh`
- HOMEGOLF local score runs now also have one repo-owned post-closeout prompt
  watcher:
  - `scripts/wait-parameter-golf-homegolf-prompt-closeout.sh`
- that wrapper refuses dirty non-`main` launches by default, can fast-forward
  clean `main`, writes reports and logs under
  `~/scratch/psionic_homegolf_runs/<run_id>`, and keeps `TMPDIR` off `/tmp`
- the prompt watcher waits for the final training report, then emits one bound
  text-generation proof from that same retained artifact family without another
  manual operator step
- that matters because the latest live `archlinux` rerun attempts exposed
  quota-sensitive failures on the older ad-hoc `/tmp` posture even though the
  reachable local CUDA lane itself is still valid for bounded and long-closeout
  work
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

HOMEGOLF now also has a first upgraded retained H100-backed live dense mixed-device score surface:

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
- the retained mixed-device source here is still:
  - one local Apple Silicon MLX rank
  - one optional-H100 CUDA submesh
  - not the currently reachable Apple-plus-home-RTX cluster

What it does not prove:

- admitted home-RTX dense closure on the local home cluster
- that the currently reachable Apple-plus-home-RTX cluster already produced
  this retained scored surface
- one single retained run id that already binds the mixed-device dense runtime
  receipts and the final scored export bytes in one artifact family
- official public-leaderboard equivalence

## Canonical Score-Relevant Dense Runtime

HOMEGOLF now also has one canonical retained H100-backed score-relevant dense runtime surface:

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
- that this retained runtime is already the current Apple-plus-home-RTX home
  cluster lane
- one locally produced scored bundle from that admitted home cluster
- public-leaderboard-equivalent hardware or score

## Reachable Home-Cluster Daily Loop

The currently reachable consumer home-cluster iteration loop is the admitted
Tailnet daily runner:

- entrypoint:
  `scripts/run-tailrun-daily-loop.sh`
- latest repo-local audit:
  `docs/audits/2026-03-28-first-swarm-exact-mean-delta-merged-bundle-audit.md`
- latest retained mixed-device run:
  `tailrun-home-admitted-20260328k`
- retained mixed-device merged portable bundle:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/retained_artifacts/merged_portable_bundle.safetensors`
- latest retained run:
  `tailrun-daily-20260328e`

What this proves:

- Psionic now has one end-to-end retained admitted-device daily loop over:
  - local M5 MLX
  - remote `archlinux` RTX 4080 CUDA
- that loop now completes:
  - same-node fixed-budget training on both admitted devices
  - PGOLF-ish held-out quality comparison on the produced bundles
  - near-equivalent direct-versus-served inference closeout
- the admitted mixed-device Tailnet runtime no longer stops at summary-only
  contribution truth
- that runtime now retains:
  - the real local contributor adapter payload
  - the real remote contributor adapter payload
  - one exact mean-delta merged adapter artifact
  - one inferable merged portable bundle
- the retained mixed-device merged artifact closes the canonical first-swarm
  profile at:
  - mean loss: `1.788139627478813e-07`
  - bits per token: `2.579740172980722e-07`
  - deterministic probe top token: `2`
- the latest retained daily scoreboard is:
  - overall verdict: `throughput_improved`
  - M5 throughput verdict: `meaningful_improvement`
  - RTX 4080 throughput verdict: `meaningful_improvement`
  - held-out quality verdict: `noise_band`
  - near-equivalent bridge verdict: `passed`

What it does not prove:

- exact HOMEGOLF or exact Parameter Golf score closure
- one mixed-device promoted served runtime bundle emitted directly from the
  admitted home cluster
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
- the HOMEGOLF side now compares using the retained H100-backed live dense
  mixed-device surface, the canonical score-relevant runtime report, and the
  retained exact dense challenge export bytes
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

## Current-Main Local Honest Relaunch

After the first current-main local score launch, one important correction became
explicit:

- `homegolf-baseline-g64-2step-roundtrip-20260328a` was a bounded probe, not
  the honest local `600s` score loop
- the reason is simple:
  - that launch passed `--max-steps 2`
  - the trainer then surfaced `max_wallclock_seconds=0`
- that bounded run is still useful for short-step timing and loss-shape checks
- it is not an honest retained local score receipt

The corrected current-main local score run is now:

- host: `archlinux`
- repo revision at launch: `60c97bf0`
- run id: `homegolf-baseline-g64-roundtrip-600s-20260328b`
- output root:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-roundtrip-600s-20260328b`
- attached watcher outputs:
  - `prompt_report.json`
  - `closeout_summary.json`
  - `prompt_closeout.log`

Observed live contract from the retained log:

- `machine_profile=homegolf_local_cuda`
- `warmup_steps=0`
- `grad_accum_steps=64`
- `final_validation_mode=roundtrip_only`
- `validation_eval_mode=non_overlapping`
- `max_wallclock_seconds=600`
- `train_step_start step=1/20000`

Observed early training movement from that same live log:

- `micro_step=1/64 train_loss=8.35887241`
- `micro_step=2/64 train_loss=8.34251022`
- `micro_step=3/64 train_loss=8.30022907`
- `micro_step=4/64 train_loss=8.27811050`
- `micro_step=9/64 train_loss=8.33421516`

The operator path is tighter now too:

- `scripts/run-parameter-golf-homegolf-local-cuda.sh` can now attach the prompt
  closeout watcher directly with `--attach-prompt-closeout`
- `scripts/wait-parameter-golf-homegolf-prompt-closeout.sh` now accepts
  `--tmpdir` and `--cargo-target-dir`
- that keeps both the trainer and the prompt closeout off brittle `/tmp`
  surfaces without a hand-built remote shell

Current truth:

- the live honest current-main local score loop is `20260328b`, not `20260328a`
- the attached closeout path now remains repo-owned end to end
- no new completed local or public PGOLF score has landed from this run yet

One more live truth became explicit during `20260328b`:

- step `1` did complete honestly on the local `600s` contract
- retained live facts:
  - `train_step_complete step=1 mean_microbatch_loss=8.29199219`
  - `optimizer_step_ms=3803`
- but step `2` immediately diverged:
  - `micro_step=1/64 train_loss=13.90855503`
  - `micro_step=2/64 train_loss=13.96454716`
  - `micro_step=3/64 train_loss=13.86529732`
  - `micro_step=4/64 train_loss=14.02747917`

That means the current local `g64` baseline is not blocked on fitting the first
step anymore.

It is blocked on keeping quality once the second optimizer step begins.

The operator path now exposes one explicit response:

- `PSIONIC_PARAMETER_GOLF_HOMEGOLF_MAX_CHALLENGE_STEPS=<n>`
- `scripts/run-parameter-golf-homegolf-local-cuda.sh --challenge-max-steps <n>`

Unlike the old positional `--max-steps`, this challenge-step cap preserves:

- `challenge_homegolf_local_cuda_defaults`
- `warmup_steps=0`
- `max_wallclock_seconds=600`
- the normal honest local HOMEGOLF score semantics

So the local loop can now stop at one stable optimizer step without dropping
into the bounded proof lane.

## Honest Grad-Clip Closeout Unlock

The next retained local iteration answered the immediate question from the
challenge-step-cap pass:

- was the capped honest `600s` lane still blocked on NaN artifact export after
  the one good optimizer step?

The answer is no.

The local HOMEGOLF lane now has one repo-owned stability override:

- env:
  `PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_CLIP_NORM=<f32>`
- runner flag:
  `scripts/run-parameter-golf-homegolf-local-cuda.sh --grad-clip-norm <f32>`

That override landed in:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `scripts/run-parameter-golf-homegolf-local-cuda.sh`

The retained clipped honest run is:

- host: `archlinux`
- repo revision at launch: `68e46b82`
- run id: `homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`
- output root:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`

Retained honest step facts from that live log:

- `machine_profile=homegolf_local_cuda`
- `max_steps=1`
- `warmup_steps=0`
- `grad_accum_steps=64`
- `max_wallclock_seconds=600`
- `train_step_complete step=1 mean_microbatch_loss=8.29197598`
- `optimizer_step_ms=3927`
- `step:1/1 train_loss:8.2920 train_time:274570ms`

Compared with the prior honest capped run:

- step time improved from `304562ms` to `274570ms`
- that is about `9.8%` faster
- mean microbatch loss stayed effectively flat while slightly better:
  `8.29201698 -> 8.29197598`

The more important fix is closeout integrity:

- `20260328c` panicked during artifact export with `min = NaN, max = NaN`
- `20260328d` reached:
  - `final_artifact_export_complete`
  - `final_artifact_persist_complete`
- retained artifact:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap1-clip1-600s-20260328d/training_report.final_model.st`

So the clipped honest lane is no longer blocked on post-step non-finite export.

It is now blocked on the next honest surface:

- the full `int8_zlib_roundtrip` validation sweep

Retained live progress from that same run:

- `validation_batch_start stage=int8_zlib_roundtrip ... batch=10/947`
- `evaluated_tokens=589824`
- `elapsed_ms=225066`

Inference from those retained first batches:

- the local full roundtrip closeout is still multi-hour on the `RTX 4080`
- the blocker moved from artifact creation failure to slow honest score
  completion

The clipped artifact now also has fresh actual-text proof on the exact retained
model file, using the artifact-only prompt path:

- local prompt report:
  `/tmp/psionic_homegolf_prompt_20260328d/prompt_report.json`
- prompt:
  `the meaning of life is`
- generated text:
  `do do hased do do do do do do do do do do do do do has has hased hased hasedededededededed`
- compressed model bytes:
  `4132303`

## Queued LR-Scale Follow-On

After the grad-clip pass, the next missing operator surface was learning-rate
control.

The local HOMEGOLF lane now also exposes one repo-owned uniform LR scale:

- env:
  `PSIONIC_PARAMETER_GOLF_HOMEGOLF_LR_SCALE=<f32>`
- runner flag:
  `scripts/run-parameter-golf-homegolf-local-cuda.sh --learning-rate-scale <f32>`

That surface landed in:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `scripts/run-parameter-golf-homegolf-local-cuda.sh`

Current retained reason for adding it:

- unclipped `20260328b` diverged immediately at step `2`
- clipped `20260328d` fixed export and artifact promptability
- the next honest quality question is whether a lower LR can keep two local
  optimizer steps stable inside the same `600s` contract

The queued next run is already staged on `archlinux` and will start
automatically when the current `20260328d` full-validation pass releases the
GPU:

- queued run id:
  `homegolf-baseline-g64-stepcap2-clip1-lr075-600s-20260328e`
- queue script:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328e.sh`
- queue log:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328e.log`
- queue pid:
  `773908`

Queued run posture:

- `grad_accum_steps=64`
- `challenge_max_steps=2`
- `grad_clip_norm=1.0`
- `learning_rate_scale=0.75`
- `final_validation_mode=roundtrip_only`
- `validation_eval_mode=non_overlapping`

This is still an honest local HOMEGOLF score attempt:

- it preserves `max_wallclock_seconds=600`
- it stays on clean `main`
- it keeps the same repo-owned closeout path

## Repo-Owned Queue Runner

The ad hoc remote scratch queue for `20260328e` is now replaced by one
committed repo-owned operator script:

- script:
  `scripts/queue-parameter-golf-homegolf-local-cuda.sh`

What it does:

- waits on an existing HOMEGOLF output root's `train.pid`
- polls until the active trainer exits
- then launches the normal repo-owned local CUDA runner
- defaults to `--sync-main` before the queued launch

The active queued follow-on is now staged through that script on `archlinux`:

- queue pid:
  `775656`
- queue log:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328e_repo.log`
- retained log head:
  - `queue_wait_root=/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`
  - `queue_wait_pid=768560`
  - `queue_waiting timestamp=2026-03-28T06:25:29-05:00 pid=768560`

This improves the operator path because the next run no longer depends on a
one-off scratch shell file outside the repo.

It now depends on committed repo code that future operators and agents can
reuse directly.

## Detached Artifact Score Closeout

The next honest bottleneck after `20260328d` was not training-step stability.

It was serialized score closeout:

- step `1/1` now trains in `274570ms`
- then the same GPU stays occupied for hours in
  `int8_zlib_roundtrip` validation
- at `2026-03-28 06:38:27 CDT`, the retained live log had only reached
  `batch=60/947` with `elapsed_ms=1476055`

That meant the training lane could already export one real promptable artifact,
but the next training attempt still had to wait behind the current full score
sweep.

This repo now exposes one explicit detached-closeout posture for local HOMEGOLF:

- trainer validation mode:
  `final_validation_mode=artifact_only`
- new detached scorer binary:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_artifact_score.rs`
- new detached score watcher:
  `scripts/wait-parameter-golf-homegolf-artifact-score.sh`
- new runner control:
  `scripts/run-parameter-golf-homegolf-local-cuda.sh --attach-detached-score-closeout`

What changed in behavior:

- the trainer can now stop after:
  - `final_artifact_export_complete`
  - `final_artifact_persist_complete`
  - writing `training_report.json` with
    `final_validation_mode=artifact_only`
- final score closeout can then run in a separate process against the exported
  artifact
- the queued next run no longer needs to wait for the previous run's full
  validation sweep before it may begin

This is a real improvement over the prior `20260328e` queue posture:

- old queued run:
  `homegolf-baseline-g64-stepcap2-clip1-lr075-600s-20260328e`
- old score posture:
  `final_validation_mode=roundtrip_only`
- old effect:
  the next run would still inherit the same multi-hour serialized GPU-bound
  closeout

The queued follow-on is now replaced on `archlinux` with:

- queued run id:
  `homegolf-baseline-g64-stepcap2-clip1-lr075-artifactonly-600s-20260328f`
- queue pid:
  `778424`
- queue log:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328f_repo.log`
- retained queue state:
  - `queue_wait_root=/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`
  - `queue_wait_pid=768560`
  - `queue_waiting timestamp=2026-03-28T06:38:05-05:00 pid=768560`

Queued `20260328f` posture:

- `grad_accum_steps=64`
- `challenge_max_steps=2`
- `grad_clip_norm=1.0`
- `learning_rate_scale=0.75`
- `final_validation_mode=artifact_only`
- `validation_eval_mode=non_overlapping`
- `attach_prompt_closeout=true`
- `attach_detached_score_closeout=true`

The detached scorer was also launched in one real smoke pass on `archlinux`
against the live exported `20260328d` artifact through a synthetic training
report rooted in the retained single-H100 schema:

- smoke output target:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/detached_score_smoke_20260328/artifact_score_report.json`

At `2026-03-28 06:38:27 CDT`, that detached score receipt had not landed yet.
That is acceptable and honest. The point of this change is not that detached
scoring is suddenly cheap. The point is that detached scoring is no longer on
the critical path for the next GPU training run.

## Chained LR Sweep Queue

The next integration gap after detached closeout was queue depth.

Before this pass, the repo-owned queue runner could only wait on an already
existing `train.pid`.

That meant agents could queue one follow-on behind the active run, but they
could not stage a longer honest sweep behind future run roots that did not
exist yet.

The queue runner now has one explicit future-run waiting mode:

- `scripts/queue-parameter-golf-homegolf-local-cuda.sh --wait-for-pid-file`

When enabled, the queue runner:

- waits for `<run-root>/train.pid` to appear
- then waits for that process to exit
- then launches the next queued repo-owned HOMEGOLF run

That improvement landed at `psionic` commit `12f60f5b`.

At `2026-03-28 06:42:19 CDT`, the staged remote chain on `archlinux` was:

- current active inline-validation run:
  `homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`
- first artifact-only queued run:
  `homegolf-baseline-g64-stepcap2-clip1-lr075-artifactonly-600s-20260328f`
- second queued run:
  `homegolf-baseline-g64-stepcap2-clip1-lr050-artifactonly-600s-20260328g`
- third queued run:
  `homegolf-baseline-g64-stepcap2-clip1-lr035-artifactonly-600s-20260328h`

Retained remote queue receipts:

- `20260328f` queue pid:
  `778424`
- `20260328g` queue pid:
  `780303`
- `20260328h` queue pid:
  `780408`

Retained queue logs:

- `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328f_repo.log`
- `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328g_repo.log`
- `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328h_repo.log`

Retained waiting evidence:

- `20260328g`:
  `queue_waiting_for_pid_file timestamp=2026-03-28T06:42:19-05:00 path=/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap2-clip1-lr075-artifactonly-600s-20260328f/train.pid`
- `20260328h`:
  `queue_waiting_for_pid_file timestamp=2026-03-28T06:42:19-05:00 path=/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap2-clip1-lr050-artifactonly-600s-20260328g/train.pid`

This is an actual loop improvement over the prior single-follow-on posture:

- the system no longer depends on a human or agent to requeue every next local
  LR attempt
- the next several honest `600s` candidate runs are already staged on clean
  `main`
- detached scoring remains attached to each queued run, so the GPU chain does
  not wait behind inline full validation after `20260328f`

## EMA Export Queue Extension

The chained LR sweep still only exercised raw-surface exports.

That meant the queued local loop could compare learning-rate variants, but it
could not test whether a smoothed final export surface improves the same
two-step honest `600s` posture.

The repo-owned runner now exposes one explicit EMA export control:

- env:
  `PSIONIC_PARAMETER_GOLF_EMA_DECAY=<f32>`
- runner flags:
  - `--ema-decay <f32>`
  - `--final-model-surface <surface>`

That landed at `psionic` commit `5f6b2f10`.

This does not invent a new trainer lane.

It simply makes one existing substrate surface operator-reachable in the
HOMEGOLF loop:

- explicit EMA state
- explicit final exported surface selection

At `2026-03-28 06:44:43 CDT`, the next queued EMA-backed local run was staged
on `archlinux` behind the existing raw-surface chain:

- queued run id:
  `homegolf-baseline-g64-stepcap2-clip1-lr050-ema997-artifactonly-600s-20260328i`
- queue pid:
  `781762`
- queue log:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328i_repo.log`

Queued posture:

- `challenge_max_steps=2`
- `grad_clip_norm=1.0`
- `learning_rate_scale=0.50`
- `ema_decay=0.997`
- `final_model_surface=ema`
- `final_validation_mode=artifact_only`
- `validation_eval_mode=non_overlapping`
- prompt closeout attached
- detached score closeout attached

Retained waiting evidence:

- `queue_waiting_for_pid_file timestamp=2026-03-28T06:44:43-05:00 path=/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap2-clip1-lr035-artifactonly-600s-20260328h/train.pid`

So the current live queue chain is now:

- `20260328f` raw, `lr=0.75`
- `20260328g` raw, `lr=0.50`
- `20260328h` raw, `lr=0.35`
- `20260328i` ema, `lr=0.50`, `ema_decay=0.997`

This is the first queued pass that can compare one smoothed final export
surface against the raw-surface local chain without another operator
intervention.

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

One more current truth is explicit after `20260328d`:

- the local honest capped lane can now complete one optimizer step, export the
  retained artifact, and generate actual text from that artifact
- the next queued local honest run can now stop at artifact export and hand the
  final score closeout to a detached watcher instead of holding the GPU behind
  inline full validation
- the next two LR variants after that are also already staged through the
  repo-owned future-run queue and will launch automatically
- one EMA-backed export variant is now also staged behind that raw-surface
  chain
- the retained actual public PGOLF score is still `6.306931747817168`
- the active local clipped run is still closing its full roundtrip validation,
  so it does not yet have a completed new score receipt
- XTRAIN is unchanged in this iteration

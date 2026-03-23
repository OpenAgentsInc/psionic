# Psionic Parameter Golf Contest Readiness Audit

> Status: current-state audit written on 2026-03-23 after reading
> `~/code/parameter-golf/README.md`, the committed Parameter Golf docs and
> reports in `psionic`, the closed `PGOLF-*` issue history, and the current
> `parameter_golf` code paths in `psionic-models`, `psionic-data`,
> `psionic-eval`, and `psionic-train`.

## Scope

This audit answers one specific question:

> how ready is `psionic` to compete in the OpenAI Parameter Golf contest using
> Psionic-owned code paths rather than the upstream PyTorch trainer, and what
> still has to happen before real Parameter Golf-style runs are honest again?

It covers:

- the current public contest contract from `~/code/parameter-golf/README.md`
- the committed Parameter Golf surfaces already present in `psionic`
- the closed `PGOLF-*` issue history that created those surfaces
- the current stop record and acceptance posture
- what the repo can do today
- what is still missing for real single-H100, `8xH100`, and record-track runs

It does not reopen the old sprint automatically.

The historical stop record remains:

- `docs/PARAMETER_GOLF_AFTER_ACTION.md`
- `fixtures/parameter_golf/reports/parameter_golf_after_action_report.json`

## Current Contest Contract

The current `parameter-golf` README defines a narrow public target:

- the submission artifact must fit within `16,000,000` decimal bytes
- the counted artifact is code bytes plus compressed model bytes
- leaderboard submissions must train in under `10` minutes on `8xH100`
- evaluation must be self-contained and offline
- the public baseline path is still rooted in `train_gpt.py`
- local smoke work may happen on smaller machines, but leaderboard truth is
  still tied to the `8xH100` bar

That means a real Psionic contest claim needs more than model code. It needs:

- exact data and metric parity
- a real training path on the challenge geometry
- a defensible counted-runtime story
- a real exported submission surface
- reproducible `8xH100` evidence

## What In Psionic Directly Relates To Parameter Golf

### 1. Challenge oracle parity and data contract

This part is real and still strong.

The repo already owns:

- FineWeb shard loading and deterministic token streaming in `psionic-data`
- SentencePiece byte-accounting parity
- exact `val_loss` and `val_bpb` oracle fixtures
- fixed validation-split identity and dataset manifests

The acceptance report still marks this category as:

- `challenge-oracle-parity = implemented`

That is the foundation needed for any honest contest work.

### 2. A Psionic-owned compact decoder family

The repo already ships a dedicated Parameter Golf model family in:

- `crates/psionic-models/src/parameter_golf.rs`

This includes:

- the public baseline `9x512` family
- stable tensor naming
- baseline parameter accounting
- descriptor digests and export helpers

This is real model ownership, not only an adapter around the upstream Python
code.

### 3. A bounded Psionic trainer and eval lane

The repo already has a real Parameter Golf reference trainer in:

- `crates/psionic-train/src/parameter_golf_reference.rs`

It owns:

- a local-reference training lane
- checkpoint and restart state
- raw model export
- int8+zlib roundtrip restore
- validation re-eval

But this lane is still deliberately bounded. It is not the real H100 training
critical path.

The local-reference geometry is small and repo-owned, while the challenge
geometry in the same file is:

- single-device defaults: `world_size=1`, `train_batch_tokens=524288`,
  `validation_batch_tokens=524288`, `train_sequence_length=1024`,
  `grad_accum_steps=8`
- distributed defaults: `world_size=8`, same token budgets, and
  `grad_accum_steps=1`

### 4. CUDA and distributed contest substrate

The repo already landed significant CUDA and distributed work:

- typed `8xH100` topology, timing, communication, and memory receipts in
  `crates/psionic-train/src/parameter_golf_distributed.rs`
- a dedicated CUDA train-path coverage report in
  `docs/PARAMETER_GOLF_CUDA_TRAINING_COVERAGE.md`
- widened CUDA family support for the baseline decoder path, including BF16,
  RMSNorm, residual-mix, RoPE or GQA decoder-block surfaces, and CUDA Muon

The important boundary is that this is substrate closure, not end-to-end
contest closure.

The distributed lane is still marked:

- `distributed-throughput-closure = partial`

The coverage doc now says the family-level blocker list is empty, but it also
explicitly refuses to turn that into a challenge-speed claim. That distinction
matters: kernel-family coverage is better than it was, but the actual contest
trainer path is still not proved.

### 5. Submission and packaging surfaces

This part is also real.

The repo already ships:

- a non-record submission package builder
- a root-local `train_gpt.py` launcher
- a shipped Psionic runtime payload
- explicit counted-byte accounting
- record-folder compatibility verification
- local challenge-clone dry-run support
- promotion and PR-bundle helpers

This is enough for:

- self-contained non-record packaging
- local replay and review
- challenge-repo folder compatibility

It is not enough for:

- a defended record-track counted-runtime story
- a real `8xH100` exported-folder run

### 6. Historical hardware evidence

The repo also kept the last honest H100-adjacent findings:

- single-H100 bring-up exists as a Rust-native seam
- the command owns dataset, tokenizer, model, optimizer, and machine-contract
  truth
- the command can refuse honestly on a non-H100 machine
- the last bounded real H100 smoke reached the CUDA microbatch path and showed
  heavy cost in attention plus host-materialized view-style ops

The after-action record preserves the next unlanded engineering step as:

- reduce host-materialized view-op cost and continue forward-path profiling

That is a real implementation clue, not generic roadmap prose.

## Current Readiness

The strongest current machine-readable posture in the repo is still:

- `current_claim_posture = non_record_submission`

That is still the correct answer today.

The acceptance categories remain:

- `challenge-oracle-parity = implemented`
- `single-device-trainer-parity = implemented_early`
- `distributed-throughput-closure = partial`
- `packaging-readiness = implemented`
- `record-track-readiness = partial_outside_psionic`

So the honest readiness split is:

### Ready today

- challenge-oracle parity work
- offline data and metric parity checks
- bounded local-reference training and roundtrip evaluation
- non-record submission packaging
- record-folder compatibility verification against the local
  `~/code/parameter-golf` clone
- research-only architecture and compression experiments on the bounded local
  reference lane
- Rust-native machine-admission validation for the single-H100 contract

### Not ready today

- a true Psionic-only single-H100 challenge training run
- a true Psionic-only `8xH100` contest run
- a record-track submission defended under the current challenge rules
- a real record-candidate campaign

## Why Psionic Is Not Ready To Compete Yet

### 1. The real Rust-only H100 training critical path is still missing

This is the biggest blocker.

The current single-H100 entrypoint is:

- `crates/psionic-train/src/bin/parameter_golf_single_h100_bringup.rs`

But its current posture is still bring-up and validation, not full training.

The committed bring-up report says:

- `execution_posture = contract_validation_only`
- `disposition = refused_machine_contract` on the committed non-H100 host
- `final_val_loss`, `final_val_bpb`, and `compressed_model_bytes` are absent
  because no real training artifact was produced

Issue `PGOLF-604` existed exactly to close this gap and was closed
`not_planned` when the lane stopped.

### 2. The shipped submission path is still not a real contest training path

The current exported submission surface still uses:

- a top-level `train_gpt.py` launcher
- a shipped Psionic runtime payload

That payload replays bounded validation over a shipped local-reference fixture.
It does not perform the real challenge training loop.

So the current exported folder is good for:

- non-record packaging
- replay
- accounting

It is not the real Rust-only contest execution payload.

### 3. There is still no real exported-folder `8xH100` evidence bundle

The distributed receipt lane is committed, but the real hardware evidence from
the exported submission entrypoint never landed.

Issue `PGOLF-602` existed to capture:

- run bundles
- train logs
- wallclock receipts
- memory receipts
- artifact-size receipts

from the exported folder on actual `8xH100` hardware.

That issue also closed `not_planned` when the sprint stopped.

So today `psionic` has distributed contract truth, not reproducible contest-run
truth.

### 4. Record-track accounting is still blocked

The record-track contract still names two blockers:

- no defended counted-runtime or build-dependency story for a real record-track
  execution payload
- no reproducible challenge-speed `8xH100` execution

This matters especially for the user's "using psionic not pytorch" requirement.

If the real execution path is a Psionic runtime, then the repo must defend:

- what bytes count
- what shipped runtime counts
- what build-time dependencies count
- how the entrypoint remains self-contained under the contest rules

The repo is honest about this and does not pretend the current non-record
launcher solves record-track accounting.

### 5. There is no current PGOLF operator lane

Search of the current Google operator scripts and Google launch fixtures found
no Parameter Golf wiring.

That means the newer Psion Google single-node operator work does not currently
make the Parameter Golf lane active again.

Today there is no committed:

- PGOLF-specific Google launch profile
- PGOLF-specific H100 cloud operator runbook
- PGOLF-specific real hardware execution lane tied to current cloud tooling

So the Parameter Golf substrate exists, but it is not connected to the active
operator infrastructure the rest of the repo now uses.

### 6. The lane is still historically paused

This is not just a technical gap. It is also a program-state fact.

The canonical after-action record says:

- `lane_status = stopped_not_planned`

and the stopped sprint explicitly closed the unfinished execution issues:

- `PGOLF-500` / `#183`
- `PGOLF-602` / `#189`
- `PGOLF-604` / `#194`
- `PGOLF-606` / `#250`
- `PGOLF-609` / `#253`

So this lane should be read as preserved substrate plus preserved evidence,
not as an active contest campaign.

## What The Closed PGOLF Issue History Says

The issue history splits cleanly into five tranches.

### 1. Foundation closed successfully

These landed and still matter:

- `PGOLF-101` to `PGOLF-103`: FineWeb, tokenizer, and oracle parity
- `PGOLF-201` to `PGOLF-203`: compact decoder family plus bounded trainer and
  roundtrip export or eval

This is the part of the lane that feels genuinely finished.

### 2. Distributed and CUDA substrate mostly landed

These also landed:

- `PGOLF-301` to `PGOLF-303`: distributed receipt lane and CUDA coverage
- `PGOLF-601` to `PGOLF-628`: wide CUDA family work for the baseline path

This means the lane is much stronger than an idea or benchmark stub. The repo
does own real Parameter Golf-specific runtime and backend work.

### 3. Packaging and challenge-repo compatibility landed

These landed too:

- `PGOLF-401`
- `PGOLF-402`
- `PGOLF-403`
- `PGOLF-502`
- `PGOLF-503`
- `PGOLF-504`
- `PGOLF-603`
- `PGOLF-701`
- `PGOLF-702`
- `PGOLF-703`

This is why the repo can honestly say it has a real non-record submission path.

### 4. The real contest-execution tranche did not close

The missing run-critical issues were:

- `PGOLF-602`: real exported-folder `8xH100` run evidence
- `PGOLF-604`: Rust-native `1xH100` baseline trainer entrypoint
- `PGOLF-609`: one real record-candidate campaign

Those are exactly the pieces still missing today.

### 5. The sprint stopped before promotion

`PGOLF-606` existed to promote the acceptance posture once the runtime and
hardware evidence were real.

That never happened, because the runtime and hardware evidence never became
strong enough before the lane stopped.

## What Psionic Can Honestly Do Today For PGOLF-Style Runs

If the user wants to do Parameter Golf-style work today, `psionic` is ready for
only the following classes of work:

- parity checks against the current challenge data and metrics
- local bounded research runs on the repo-owned local-reference fixture
- non-record package generation and local challenge-clone verification
- machine-admission and first-microbatch bring-up on a qualifying H100 host
- continued CUDA or graph profiling work toward the baseline path
- architecture and compression research under the committed research harness

It is not ready today for:

- a real leaderboard attempt
- a pure Psionic single-H100 baseline training run
- a pure Psionic `8xH100` run campaign

## What Still Needs To Happen To Resume Real PGOLF Work

The shortest honest restart path is:

1. Restore the Rust-only single-H100 trainer path from the `PGOLF-604` scope.
   That means training end to end on cached FineWeb or SP1024 data with the
   public single-device geometry, without invoking `python3 train_gpt.py` on
   the actual training critical path.

2. Reconnect the real training path to real hardware profiling.
   The after-action record says the next concrete engineering move is reducing
   host-materialized view-op cost and continuing forward-path profiling on H100.

3. Emit challenge-comparable outputs directly from the Rust path.
   That includes:
   - step logs
   - final `val_loss`
   - final `val_bpb`
   - compressed-model bytes
   - artifact-byte accounting

4. Build one real `8xH100` execution lane from the exported submission
   surface, not just from internal receipt builders.
   That is the missing `PGOLF-602` class of evidence.

5. Defend the counted-runtime story for record-track.
   The repo already knows this blocker by name. It just has not solved it yet.

6. Freeze one real record-candidate family and run repeated evidence bundles.
   That is the missing `PGOLF-609` campaign step.

7. Only after those steps, revisit record-track promotion.
   That is the role the old `PGOLF-606` promotion issue was meant to play.

## Suggested Google Issue Program

The repo now has a real Google operator substrate for bounded `Psion` training,
but none of that is currently wired to the Parameter Golf lane.

The right next tranche is therefore not "invent new cloud infrastructure from
scratch." It is:

- reuse the existing `openagentsgemini` Google project, bucket, service
  account, preflight, launch, finalizer, and cost-receipt substrate where it
  fits
- add the missing Parameter Golf-specific trainer and H100 operator surfaces
- only then attempt real Google-backed Parameter Golf evidence

The issue set below is designed to be copied into GitHub as the new
Google-resumption tranche.

Recommended order:

1. `PGOLF_GOOGLE-1`
2. `PGOLF_GOOGLE-2`
3. `PGOLF_GOOGLE-3`
4. `PGOLF_GOOGLE-4`
5. `PGOLF_GOOGLE-5`
6. `PGOLF_GOOGLE-6`
7. `PGOLF_GOOGLE-7`
8. `PGOLF_GOOGLE-8`
9. `PGOLF_GOOGLE-9`
10. `PGOLF_GOOGLE-10`
11. `PGOLF_GOOGLE-11`

### `PGOLF_GOOGLE-1: Restore The Rust-Only Single-H100 Parameter Golf Baseline Trainer`

**Summary**

Restore the unfinished `PGOLF-604` execution gap by turning the current
single-H100 bring-up seam into a real Psionic-owned single-H100 baseline
trainer that consumes the cached FineWeb or SP1024 challenge data, trains on
CUDA, and emits challenge-comparable outputs without invoking
`python3 train_gpt.py` on the actual training critical path.

**Why**

The current repo already has:

- challenge-oracle parity
- the compact Parameter Golf decoder family
- a bounded local-reference trainer
- the single-H100 bring-up seam

But it still does not have the most important thing needed for a real contest
attempt:

- one documented Rust-only H100 training path that produces real training
  artifacts, `val_loss`, `val_bpb`, and compressed-model bytes directly from
  `psionic`

Without this issue, the Google tranche would still be built on a missing core
trainer path.

**Depends On**

- current committed Parameter Golf foundation in `psionic-data`,
  `psionic-models`, `psionic-eval`, and `psionic-train`

**Scope**

- extend `parameter_golf_single_h100_bringup` or add one new Rust entrypoint
  that performs the real baseline train loop end to end
- bind the command to the cached FineWeb `sp1024` shard contract and tokenizer
  contract already modeled in `psionic-data`
- run with
  `ParameterGolfBatchGeometry::challenge_single_device_defaults()`
- emit challenge-style outputs directly from the Rust path:
  - step logs
  - final `val_loss`
  - final `val_bpb`
  - compressed-model bytes
  - artifact-byte accounting inputs
- preserve explicit refusal when the machine contract, dataset root, tokenizer
  path, or CUDA capability is invalid

**Non-Goals**

- no `8xH100` work yet
- no record-track promotion yet
- no new submission-folder contract yet

**Acceptance Criteria**

- one documented `cargo run` path on a qualifying H100 performs the baseline
  training run end to end without `python3 train_gpt.py`
- the run consumes cached FineWeb `sp1024` data rather than the bounded
  local-reference fixture
- the Rust path emits challenge-comparable logs and final metrics directly
- the path still refuses explicitly on non-H100 or invalid-machine contracts

### `PGOLF_GOOGLE-2: Profile And Reduce H100 View-Op And Host-Materialization Cost On The Baseline Path`

**Summary**

Resume the exact next engineering step preserved by the after-action record:
profile and reduce host-materialized view-op cost on the real Parameter Golf
baseline path so the restored single-H100 trainer is not bottlenecked by
attention-adjacent data movement and view orchestration.

**Why**

The last honest H100 findings in the repo already identified the dominant next
optimization direction:

- `scaled_dot_product_attention`
- `rotary_embedding`
- `permute`
- `expand`
- `reshape`

The lane stopped before the next optimization cycle landed. If this work stays
undone, a real H100 trainer may exist in name but still be too inefficient to
matter.

**Depends On**

- `PGOLF_GOOGLE-1`

**Scope**

- restore the bounded H100 trace workflow on the real Psionic baseline path
- capture fresh forward-path and microbatch profiles on a qualifying H100
- reduce host materialization or unnecessary copy costs where the current path
  still widens tensors or views expensively
- retain machine-readable profiling receipts or reports for the same path
- keep any remaining runtime bottlenecks explicit rather than implying speed
  closure

**Non-Goals**

- no `8xH100` distributed run yet
- no claim that one optimization cycle alone reaches leaderboard speed

**Acceptance Criteria**

- one fresh H100 profile exists for the real Rust-only baseline trainer path
- the current dominant view-op or host-materialization costs are materially
  reduced or replaced by an explicit narrowed blocker list
- the resulting path is the one later Google single-H100 runs actually use

### `PGOLF_GOOGLE-3: Package Immutable FineWeb SP1024 Inputs For Google Parameter Golf Execution`

**Summary**

Add one repo-owned immutable Google input-package contract for Parameter Golf
that binds the cached FineWeb `sp1024` shards, tokenizer identity, validation
split identity, and dataset or tokenizer digests into a remote-execution
package that the Google operator lane can consume without ambiguity.

**Why**

The current Google operator lane has no Parameter Golf input package at all.
The contest lane also has stricter offline and reproducibility requirements than
the current bounded Psion pilot lanes.

Parameter Golf on Google therefore needs one explicit remote-input contract
before any honest cloud runs begin.

**Depends On**

- current Parameter Golf data and oracle parity substrate in `psionic-data`

**Scope**

- define one immutable Google input-package descriptor for Parameter Golf
  single-H100 runs
- define one matching descriptor for later `8xH100` runs if the content shape
  differs materially
- bind dataset manifest digest, tokenizer digest, shard selection posture, and
  fixed validation identity into the package
- upload the package into the existing Google training bucket with stable
  digests
- make the input package usable by later Google manifests and final receipts

**Non-Goals**

- no contest submission-folder packaging here
- no claim that the Google operator lane is already record-ready

**Acceptance Criteria**

- one committed Google Parameter Golf input-package descriptor exists
- the descriptor binds the FineWeb `sp1024` contract and tokenizer identity by
  digest
- the package is materialized in GCS and can be referenced by later Google
  launch profiles

### `PGOLF_GOOGLE-4: Add A Google Single-H100 Operator Lane For The Psionic Parameter Golf Baseline`

**Summary**

Extend the existing `openagentsgemini` Google operator substrate with one
Parameter Golf-specific single-H100 launch profile, runbook slice, preflight,
and final-manifest surface that launches the real Rust-only Parameter Golf
baseline trainer rather than the current generic Psion lanes.

**Why**

The repo already has Google operator tooling, but it is currently wired only to
the `Psion` training lanes. Parameter Golf has:

- no Google launch profile
- no Google H100 preflight
- no Google H100 runbook
- no Google final-manifest authority for its own trainer path

This is the first cloud integration step.

**Depends On**

- `PGOLF_GOOGLE-1`
- `PGOLF_GOOGLE-3`

**Scope**

- add one Parameter Golf Google single-H100 launch profile
- bind that profile to the Rust-only Parameter Golf trainer command
- add preflight checks for the required H100 machine contract, quota, and cost
  ceiling
- preserve existing Google bucket, finalizer, accelerator-validation, and
  cost-receipt seams where they fit
- document the exact runbook for the bounded single-H100 Parameter Golf lane
- keep machine-family, zone, and fallback posture explicit instead of implicit

**Non-Goals**

- no `8xH100` lane yet
- no record-track claim yet

**Acceptance Criteria**

- one committed Google launch profile exists for the Parameter Golf H100 lane
- the launch manifest records the Rust-only PGOLF trainer command explicitly
- local preflight and manifest-only rehearsal both work for the new profile

### `PGOLF_GOOGLE-5: Run And Audit The First Real Google Single-H100 Psionic Parameter Golf Baseline`

**Summary**

Execute and audit the first real Google-hosted single-H100 Parameter Golf
baseline run using the new Psionic-owned Rust trainer path, the committed
Google input package, and the Google operator lane.

**Why**

Until this run exists, the repo still lacks a truthful cloud proof that the
Rust-only Parameter Golf training path works on real remote H100 hardware.

This is the cloud equivalent of closing the old `PGOLF-604` gap.

**Depends On**

- `PGOLF_GOOGLE-2`
- `PGOLF_GOOGLE-4`

**Scope**

- one bounded Google single-H100 Parameter Golf run
- full retained evidence:
  - launch manifest
  - training logs
  - final metrics
  - compressed-model bytes
  - accelerator evidence
  - run-cost receipt
  - teardown proof
- one follow-up audit that says clearly whether the single-H100 lane is now
  real or still blocked

**Non-Goals**

- no `8xH100` distributed attempt yet
- no leaderboard or record-track claim yet

**Acceptance Criteria**

- one real Google single-H100 run completes or fails with preserved cause
- successful runs emit final `val_loss`, final `val_bpb`, and compressed-model
  bytes from the Rust path itself
- one audit records the exact outcome and current claim boundary

### `PGOLF_GOOGLE-6: Bind The Exported Submission Folder To A Real Psionic Training Runtime`

**Summary**

Replace the current replay-only exported-folder runtime posture with a real
Psionic execution payload that can perform the Parameter Golf training or eval
workflow needed by later Google contest runs from the exported submission
surface itself.

**Why**

The current exported folder is honest but deliberately narrow:

- `train_gpt.py` launches a shipped Psionic runtime
- that runtime replays bounded validation on a shipped local-reference fixture
- it does not perform the real challenge training loop

The full contest path needs the exported-folder runtime to represent the actual
Psionic execution path, not only a review-time replay path.

**Depends On**

- `PGOLF_GOOGLE-1`
- `PGOLF_GOOGLE-5`

**Scope**

- upgrade the exported-folder runtime payload so it can perform the real
  bounded Parameter Golf training or evaluation path needed for later Google
  evidence
- keep the folder self-contained and compatible with the contest repo contract
- preserve explicit counted-byte accounting for the shipped runtime and model
- keep offline evaluation posture explicit

**Non-Goals**

- no immediate record-track approval
- no external PR yet

**Acceptance Criteria**

- one exported folder can invoke a real Psionic Parameter Golf execution path
  instead of replay-only local-reference validation
- the challenge-clone compatibility and replay-verification gates still pass
- the runtime payload and accounting receipts stay machine-readable

### `PGOLF_GOOGLE-7: Add A Google 8xH100 Parameter Golf Operator Lane`

**Summary**

Add one Google `8xH100` Parameter Golf operator lane that binds the public
challenge `WORLD_SIZE=8` posture, the existing distributed receipt lane, the
real execution payload, and the current Google launch and finalizer substrate
into one committed cloud operator path.

**Why**

The contest bar is still defined by:

- training in under `10` minutes on `8xH100`

The repo already has distributed receipts and distributed geometry, but there
is still no real Google `8xH100` operator lane.

**Depends On**

- `PGOLF_GOOGLE-3`
- `PGOLF_GOOGLE-6`

**Scope**

- add one committed Google `8xH100` launch profile for Parameter Golf
- preserve the public `WORLD_SIZE=8`, `grad_accum_steps=1` posture in the
  launch manifest and distributed receipts
- add preflight for H100 inventory, zonal capacity, runtime ceiling, and cost
  ceiling
- preserve topology, communication, timing, memory, and accelerator evidence
  in the final bundle
- make the machine-family or topology choice explicit rather than implicit; if
  Google requires a specific `8xH100` family or pod shape, record that exact
  choice in the profile

**Non-Goals**

- no record-track promotion yet
- no silent fallback to weaker machines

**Acceptance Criteria**

- one committed Google `8xH100` PGOLF launch profile exists
- local preflight can validate it
- manifest-only rehearsal works
- the finalizer and receipts preserve distributed evidence for this lane

### `PGOLF_GOOGLE-8: Capture The First Real Google 8xH100 Exported-Folder Evidence Bundle`

**Summary**

Run the real exported submission surface on the Google `8xH100` lane and retain
the full evidence bundle that the old `PGOLF-602` issue was intended to
capture.

**Why**

This is the missing proof that turns the current distributed lane from contract
truth into real contest-execution evidence.

The key requirement is that the run be tied to:

- the exact exported entrypoint
- the exact shipped artifact bytes
- the actual `8xH100` Google execution

**Depends On**

- `PGOLF_GOOGLE-6`
- `PGOLF_GOOGLE-7`

**Scope**

- run the exported submission surface on real Google `8xH100` hardware
- retain run bundles, train logs, wallclock receipts, memory receipts, final
  metrics, and artifact-size receipts
- bind the evidence to the exact exported-folder entrypoint digest and shipped
  artifact bytes
- preserve explicit measured-or-refused posture if the run misses the challenge
  bar

**Non-Goals**

- no SOTA or leaderboard claim yet
- no significance campaign yet

**Acceptance Criteria**

- one real `8xH100` evidence bundle exists for the exported-folder path
- the evidence cites the exact entrypoint and artifact bytes used by the run
- the run outcome is explicit about whether it clears or misses the challenge
  bar

### `PGOLF_GOOGLE-9: Defend Record-Track Counted-Runtime And Build-Dependency Posture For The Google Execution Payload`

**Summary**

Retire the current record-track accounting blocker by defending the exact
counted-runtime and build-dependency story for the real Psionic execution
payload used on the Google Parameter Golf lane.

**Why**

The current record-track contract still blocks on:

- counted-runtime posture
- build-dependency posture

For a real pure-Psionic contest path, the repo must answer exactly what bytes
count when the exported folder carries a Psionic runtime payload and any
supporting files required for evaluation.

**Depends On**

- `PGOLF_GOOGLE-6`
- `PGOLF_GOOGLE-8`

**Scope**

- enumerate the exact shipped runtime payload used by the Google-backed
  Parameter Golf exported folder
- defend which files count as code bytes and which count as compressed-model
  bytes
- defend whether any build-time dependencies are shipped, required, or avoided
- update the record-track contract, accounting docs, and receipts accordingly
- keep the answer explicit and machine-readable

**Non-Goals**

- no performance claim by accounting alone
- no external maintainer claim if the accounting story remains ambiguous

**Acceptance Criteria**

- the record-track contract no longer treats counted-runtime posture as an
  unnamed blocker
- the accounting answer is explicit for the Google-backed execution payload
- later record-track claims can point to one clear counted-runtime contract

### `PGOLF_GOOGLE-10: Freeze One Google-Backed Record Candidate And Run A Repeated Evidence Campaign`

**Summary**

Freeze one exact Google-backed Parameter Golf candidate family and run a real
campaign with repeated evidence bundles, explicit promotion receipts, and
stable campaign identity.

**Why**

The repo already named this missing step in the old `PGOLF-609` issue:

- there is still no single frozen candidate configuration
- no repeated evidence bundle set
- no promotion logic tied to one stable candidate

Without this step, later runs remain ad hoc.

**Depends On**

- `PGOLF_GOOGLE-8`
- `PGOLF_GOOGLE-9`

**Scope**

- choose one exact candidate family for the first serious campaign
- freeze architecture, tokenizer, accounting posture, and run recipe
- run repeated Google-backed evidence bundles for that exact candidate
- retain final metrics, wallclock, artifact bytes, and promotion receipts
- tie any significance or systems-only waiver posture to this exact candidate

**Non-Goals**

- no parallel multi-candidate zoo
- no external submission until the campaign evidence is internally coherent

**Acceptance Criteria**

- one named record-candidate campaign exists with a frozen config
- repeated evidence bundles are preserved for that candidate
- promotion decisions are tied to the frozen candidate and explicit receipts

### `PGOLF_GOOGLE-11: Produce The Full Google-Hosted Parameter Golf Submission Dry Run And Final Readiness Audit`

**Summary**

Produce the final Google-hosted end-to-end submission dry run for Parameter
Golf, including the exported record folder, local challenge-clone dry run,
promotion receipts, and a final readiness audit that says explicitly whether
the repo should submit or still hold.

**Why**

A full Google-backed contest path is only complete when the repo can point to
one integrated dry run that joins:

- the real execution evidence
- the exported folder
- the counted-runtime contract
- the promotion posture
- the final do-submit or do-not-submit decision

**Depends On**

- `PGOLF_GOOGLE-10`

**Scope**

- generate the final exported submission folder from the Google-backed evidence
- run the local `parameter-golf` clone dry run again on that final folder
- preserve final PR-bundle or maintainer-facing review artifacts
- write one final readiness audit that compares the result against the current
  public contest README requirements
- keep the external submission itself out of scope unless the user explicitly
  asks for it

**Non-Goals**

- no automatic PR to `openai/parameter-golf`
- no claim stronger than the final evidence bundle supports

**Acceptance Criteria**

- one final Google-hosted submission dry run exists with linked evidence
- the local challenge-clone dry run passes on the final folder
- one final readiness audit says explicitly whether the repo is ready to submit
  or still blocked

## Validation Performed For This Audit

I validated the current committed lane with:

- `scripts/check-parameter-golf-acceptance.sh`
- `scripts/check-parameter-golf-record-folder-compatibility.sh --parameter-golf-root /Users/christopherdavid/code/parameter-golf`
- `cargo test -p psionic-train parameter_golf_single_h100_bringup -- --nocapture`
- `cargo test -p psionic-train parameter_golf_distributed_8xh100_lane -- --nocapture`

Current observed results:

- acceptance report still resolves to `current_claim_posture = non_record_submission`
- record-folder compatibility still resolves to `compatibility_status = compatible`
- single-H100 bring-up tests passed
- distributed `8xH100` receipt-lane tests passed

That means the preserved PGOLF substrate is still coherent and buildable.
It does **not** change the higher-level readiness conclusion.

## Bottom Line

`psionic` is not currently ready to compete seriously in Parameter Golf using a
fully Psionic-owned contest path.

The honest current posture is still:

- good research substrate
- good non-record packaging substrate
- partial distributed and CUDA contest substrate
- not yet a real Rust-only H100 trainer
- not yet a real `8xH100` contest lane
- not yet record-ready

The repo is much closer to:

- "we preserved a strong paused PGOLF substrate that can be resumed"

than to:

- "we can launch a serious pure-Psionic contest run right now"

If resumed, the next step is not new roadmap prose. It is finishing the real
Rust-only `1xH100` trainer path and then collecting real exported-folder
`8xH100` evidence from that path.

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

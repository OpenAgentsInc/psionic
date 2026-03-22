# 2026-03-22 Psion Training System Full State Audit

This audit reviews the current `Psion` training system in
`OpenAgentsInc/psionic` after the full `PSION-*` issue program and the Google
single-node follow-on issues were closed.

It also compares the current system to the original planning documents in:

- `~/code/alpha/training/initial-psionic-model-training-spec.md`
- `~/code/alpha/training/model-training-chat.md`

## Executive Verdict

The `Psion` training system is now issue-complete and substantially more real
than it was at the start of the program.

The repo now has:

- a completed `PSION-1` through `PSION-45` issue record
- no open GitHub issues in `OpenAgentsInc/psionic`
- a real executable reference corpus build path
- a real executable reference pilot pretraining bundle
- real checkpoint archive and cold-restore coverage
- a real Google single-node operator lane
- one actual Google-hosted bounded pilot run with preserved evidence

But the important boundary is this:

- the system is now **reference-lane real**
- it is **not yet broad-pretraining complete**
- it is **not yet accelerator-throughput proved**
- it is **not yet cluster-scale proved by a live committed run audit**

So the correct statement today is:

- `Psion` is ready for the next serious implementation tranche toward real
  accelerator-backed curated-corpus pretraining
- `Psion` is not yet in a state where it can honestly claim broad-model
  GPU-efficient pretraining closure

## Scope And Sources

I consulted all of the following:

- the original planning docs in `~/code/alpha/training`
- the current canonical training spec in `docs/TRAIN_SYSTEM.md`
- the canonical `Psion` track docs under `docs/PSION_*.md`
- the current `Psion` code in `crates/psionic-data/` and `crates/psionic-train/`
- the Google operator scripts in `scripts/psion-google-*.sh`
- the recent Google pilot audits in `docs/audits/`
- the current GitHub issue state via `gh issue list`

I also reran focused validation on the current tree:

- `cargo test -p psionic-data psion -- --nocapture`
- `cargo test -p psionic-train reference_pilot -- --nocapture`
- `cargo run -p psionic-data --example psion_reference_corpus_build -- <tmpdir>`
- `cargo run -p psionic-train --example psion_reference_pilot_bundle -- <tmpdir>`
- `bash scripts/psion-google-operator-preflight.sh --profile g2_l4_single_node --zone us-central1-a`

Observed results:

- the `psionic-data` Psion tests passed
- the `psionic-train` reference-pilot tests passed
- the reference corpus build emitted dataset identity
  `psion_reference_tokenized@v1`
- the reference pilot bundle completed with benchmark pass rates
  `architecture=10000`, `normative_specs=10000`, `held_out=10000`,
  `refusal=10000`
- the Google operator preflight returned `result=ready`

## Issue State

Current GitHub issue state for `OpenAgentsInc/psionic`:

- open issues: `0`
- original `Psion` learned-model program: `PSION-1` through `PSION-30` all
  closed
- follow-on hardening tranche: `PSION-31` through `PSION-35` all closed
- Google training tranche: `PSION-36` through `PSION-45` all closed

That means the issue program is complete as a repo planning and implementation
record.

It does **not** mean the most ambitious reading of the original plan is fully
realized at scale.

## What The Original Plan Asked For

The original plan in `~/code/alpha/training/initial-psionic-model-training-spec.md`
and `~/code/alpha/training/model-training-chat.md` was unusually disciplined.

It asked for a learned lane that:

- is trained from scratch on a curated technical corpus
- is explicitly not an executor-claim widening of `Tassadar`
- emphasizes historical and first-principles reasoning about systems
- preserves rights posture, source lineage, and contamination boundaries
- controls code-token dominance so the model does not collapse into a generic
  coding assistant
- uses explicit route classes and refusal boundaries
- treats benchmark, route, refusal, and capability publication as separate
  evidence surfaces
- climbs in phases:
  corpus hygiene, tokenizer closure, pilot pretraining, broader curated
  pretraining, reasoning SFT, route/refusal alignment, trusted-cluster scale,
  then bounded decentralized follow-on

The planning docs also clearly preferred:

- books, specs, manuals, papers, and selected systems code over broad web text
- architecture as tradeoffs under constraints
- symbolic-AI and operating-system materials as reasoning scaffolds
- a compact decoder family before any more ambitious architecture
- a small pilot before broader scale

That planning posture survives very strongly in the current repo.

## Where The Current System Matches The Original Plan Well

### 1. Governance and corpus discipline

This part is now strong.

The repo has explicit canonical docs and code for:

- corpus admission
- rights posture
- source lifecycle and removal traceability
- benchmark isolation
- acceptance and promotion
- capability publication
- rollback and withdrawal

This is one of the most faithful parts of the implementation relative to the
original plan. The repo clearly did not drift into "just train something and
see vibes."

### 2. Data and tokenizer closure

This part is also strong at reference scale.

`psionic-data` now owns:

- raw-source ingestion contracts
- tokenizer-training manifests and artifact bundles
- tokenized corpus manifests
- sampling-policy and code-dominance controls
- a real executable reference corpus build path

The current reference corpus run proves that the repo can go from admitted raw
sources to tokenized corpus artifacts with explicit dataset identity and held-
out separation.

### 3. Model and pilot-run closure

This part is now real for the bounded reference lane.

The repo now has:

- a `Psion` compact decoder family
- an explicit pretrain-stage contract
- observability receipts
- checkpoint recovery semantics
- a real executable reference pilot bundle
- a real resume-from-last-stable-checkpoint probe

This is a meaningful shift from the earlier state where the lane was mostly
schema and fixture theater.

### 4. Benchmarks, route, refusal, and serving posture

This part is stronger than most training repos.

The repo now has explicit contracts and artifact families for:

- architecture reasoning
- normative spec reading
- engineering spec interpretation
- memorization-vs-reasoning probes
- route-class evaluation
- refusal calibration
- served evidence and claim posture
- capability withdrawal

This is very aligned with the original planning documents, which were explicit
that route quality, refusal quality, and reasoning quality should not collapse
into one scalar.

### 5. Google single-node operator lane

This part is now real, not inferred.

The repo now has:

- launch, preflight, startup, finalization, archive, restore, and delete
  scripts
- dedicated bucket, service-account, network, and quota/budget posture
- one real bounded Google single-node pilot audit

That audit proved:

- real host allocation
- real bootstrap
- real run evidence retention
- real checkpoint archive
- real cold restore
- real teardown

This is a genuine operator milestone.

## Where The Current System Is Narrower Than The Closed Issue List Suggests

This is the most important section in the audit.

### 1. The reference lane is still tiny

The current executable corpus and pilot are still a reference-scale lane, not
the large curated pretraining lane described in the original plan.

The original plan talked about:

- pilot corpora on the order of `5B` to `20B` tokens
- broader pretraining on the order of `20B` to `80B` tokens
- pilot models around `50M` to `150M` parameters
- later serious internal models around `300M` to `1B`

The current repo evidence does **not** show that scale.

What it does show is:

- a small curated reference corpus
- a bounded compact-decoder pilot
- correct receipts and validation at small scale

That is progress, but it is still much narrower than the original ambition.

### 2. The first real Google run was not real GPU training

This is the largest remaining reality leak.

The first real Google-hosted pilot run completed successfully, but the retained
GPU summary showed:

- average GPU utilization: `0%`
- max GPU utilization: `0%`
- max GPU memory used: `0 MiB`

So the current Google proof is:

- real cloud execution proof
- real artifact-retention proof
- real checkpoint/cold-restore proof

It is **not**:

- proof that the current `Psion` training lane actually uses the GPU
- proof of realistic accelerator throughput
- proof of cost/performance behavior for a serious pretraining run

### 3. Later phases are more contract-complete than run-proven

The repo has strong bounded contracts for:

- rented-cluster operation
- trusted-cluster operation
- reasoning SFT
- decentralized contribution

But those lanes are not all equally proved in the same way the Google
single-node lane is proved.

What I found:

- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md` is explicit that it is a runbook and
  failure-policy contract
- `docs/PSION_TRUSTED_CLUSTER_RUN.md` freezes a bounded trusted-cluster bundle
  and topology contract
- `docs/PSION_REASONING_SFT.md` freezes a bounded reasoning-SFT bundle
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md` freezes a bounded decentralized
  adapter-delta contribution lane

That is real work, and the corresponding Rust modules exist.

But I did **not** find a committed real-run audit for:

- a live multi-host `Psion` trusted-cluster training execution
- a live run-derived `Psion` reasoning-SFT campaign on meaningful model weights
- a live decentralized `Psion` contribution campaign

So the honest posture is:

- these later phases are implemented as machine-checkable bundle and contract
  surfaces
- they are not yet all equally proved by live operational evidence

### 4. Some issue closures are spec closure, not scale closure

This is not a bug, but it matters.

Many `PSION-*` issues landed the correct thing:

- schemas
- validators
- canonical fixtures
- machine-legible receipts
- runbooks

That is valuable because it prevents future claim drift.

But it also means:

- issue closure does not always equal "this lane is now demonstrated at serious
  training scale"

The repo today is best described as:

- strongly contract-complete
- reference-run complete in several key places
- still only partially scale-complete

## Phase-By-Phase Status Against The Original Plan

### Phase 0: corpus assembly and source hygiene

Status: `implemented_early`

Assessment:

- strong
- clearly aligned with the original intent
- one of the best-completed parts of the program

### Phase 1: tokenizer and pilot dataset closure

Status: `implemented_early`

Assessment:

- strong at bounded reference scale
- executable and tested
- still not evidence of a large private production corpus

### Phase 2: pilot pretraining run

Status: `implemented_early`

Assessment:

- strong for the bounded reference lane
- local run path is real
- Google single-node pilot proof is real
- still reference-scale and CPU-bound on the cloud host

### Phase 3: broader curated pretraining

Status: `partial`

Assessment:

- the governance and control surfaces needed for broader pretraining mostly
  exist
- the repo does not yet show a broader curated-corpus, accelerator-using,
  serious-size pretraining run

This is the main unfinished training-system milestone.

### Phase 4: reasoning SFT

Status: between `implemented_early` and `partial`

Assessment:

- bundle, lineage, stage, and evaluation contracts are real
- the bounded lane exists in code
- I did not find a live run audit comparable to the Google single-node pilot

So this phase is structurally real, but not yet equally operationally proved.

### Phase 5: route-selection and refusal alignment

Status: `implemented_early`

Assessment:

- route and refusal are among the clearest success areas of the `Psion` lane
- the benchmark and receipt discipline here is much stronger than in most model
  repos
- the reference pilot bundle already exercises these surfaces

### Phase 6: trusted-cluster scale-up

Status: `partial`

Assessment:

- trusted-cluster contracts and bounded bundle surfaces exist
- the repo is much better prepared for this than it was before
- I did not find committed live multi-host proof on the same footing as the
  Google single-node pilot audit

### Phase 7: decentralized follow-on

Status: `implemented_early` for bounded contract surface, `partial` for live
operational proof

Assessment:

- the adapter-delta bounded decentralized lane exists as a machine-checkable
  contract
- that is consistent with the original plan, which explicitly wanted bounded
  follow-on rather than fake world-scale all-reduce
- a live `Psion` decentralized campaign is still not proved in the same way the
  single-node lane is proved

## Current Best Reading Of The System

The best current reading is:

1. The repo now has a serious train-system substrate.
2. The `Psion` learned-model lane is no longer speculative.
3. The `Psion` program map has been materially implemented, not just written.
4. The strongest real evidence is in:
   governance, data lineage, reference corpus build, reference pilot bundle,
   checkpoint recovery, route/refusal receipts, and Google single-node
   operations.
5. The weakest remaining evidence is in:
   accelerator utilization, broader curated pretraining, and live cluster-scale
   execution proof.

## Residual Risks

### 1. Accelerator illusion risk

The repo now has cloud proof without GPU-training proof. That is useful, but it
can mislead readers if the distinction is not kept explicit.

### 2. Scale illusion risk

The closed issue program may read like end-to-end closure to someone who does
not distinguish fixture-valid contracts from large-scale operational proof.

### 3. Cost-truth gap

The Google lane now has bounded budget and evidence posture, but it still does
not have repo-local invoice-grade Cloud Billing truth.

### 4. Warning debt

The focused `psionic-train` validation still emits several Rust warnings around
unused imports and dead code in unrelated train modules. That is not a launch
blocker, but it is real maintenance debt in a large training crate.

## Go / No-Go

### What is a truthful "go" today

It is truthful to say:

- the `Psion` training system is now real enough to support the next
  implementation tranche toward serious learned-model training
- the issue program is complete
- the bounded reference lane is executable
- the single-node Google operator lane is real
- the repo can preserve evidence honestly

### What is still a "no-go" today

It is **not** truthful to say:

- the entire original training ambition is finished
- `Psion` has already proved broad GPU-backed pretraining
- `Psion` has already proved cluster-scale training in live operational
  evidence
- the system is ready to market as a finished serious pretraining platform

## Recommended Next Steps

These are the highest-value next steps now that the issue program itself is
closed.

### 1. Replace the CPU-bound reference pilot command with the first real
accelerator-using `Psion` training lane

This is the single highest-priority next step.

Until this exists, every GPU audit remains partly an infra audit.

### 2. Materialize the first real curated private corpus beyond the reference
fixtures

The original planning documents were explicit that corpus quality is the whole
game.

The repo now has the governance and ingestion contracts to do this honestly.
The next missing step is to actually populate a larger rights-reviewed corpus
through those contracts.

### 3. Run the first real accelerator-backed bounded pretraining audit

The next audit should prove all of the following together:

- real GPU utilization
- real checkpoint growth under the actual training lane
- real cost and throughput posture
- real held-out and benchmark behavior on a non-reference corpus

### 4. Produce a real run-derived reasoning-SFT audit

The bundle and validation surfaces exist.
The next missing proof is a real run that derives those artifacts from a live
training path instead of only canonical fixture state.

### 5. Produce a real multi-host trusted-cluster audit

The trusted-cluster contract is now good enough that the next meaningful step
is not more schema widening. It is a real bounded multi-host execution proof.

### 6. Tighten train-crate warning debt

The warnings seen during focused validation are not catastrophic, but the
training stack is now large enough that warning debt will eventually hide real
mistakes.

## Final Verdict

Compared to the original plans in `~/code/alpha/training`, the repo is now in a
good state structurally and a mixed state operationally.

Structurally:

- the implementation is strong
- the claim discipline survived
- the data and benchmark posture are unusually rigorous
- the issue program is fully closed

Operationally:

- the bounded reference lane is real
- the Google single-node lane is real
- the broader accelerator-backed and cluster-backed pretraining story is still
  the main unfinished frontier

So the most honest one-line conclusion is:

- `Psion` is no longer blocked on missing training-system shape; it is now
  blocked on turning the bounded reference lane into a real accelerator-backed
  curated-corpus pretraining lane.

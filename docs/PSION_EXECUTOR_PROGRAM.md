# Psion Executor Program

> Status: canonical `PSION-0001` / `#700` executor-capable `Psion` naming and
> program-split contract, updated 2026-03-30.

## Why This Doc Exists

The repo already has real `Psion` docs, real `Tassadar` docs, and real
executor-lane artifacts.

What it did not have in one durable place was the naming rule that keeps those
lanes from being flattened into each other.

This doc exists to make three things explicit:

- `Psion` is the umbrella learned-model family inside `psionic`
- `Tassadar` names the executor-capable bounded `Psion` profile and route lane
- the generic compact-decoder `Psion` lane and the executor-capable `Psion`
  lane are different lanes with different acceptance truth

## Umbrella And Profile Rule

For the current program:

- `Psion` is the umbrella family name
- generic compact-decoder `Psion` remains a real learned lane with its own
  route, refusal, serving, and training contracts
- `Tassadar` is the executor-capable bounded `Psion` profile and route family
- the current executor-capable implementation remains the bounded
  article-transformer route and artifact family already shipped in this repo

This means phase one does not rename the live executor lane into some new
artifact family just to make the umbrella cleaner on paper.

## Current Program Split

The active `Psion` program is multi-lane:

### Generic compact-decoder `Psion`

This is the general learned lane documented through surfaces such as:

- `docs/PSION_PROGRAM_MAP.md`
- `docs/PSION_ACCEPTANCE_MATRIX.md`
- `docs/PSION_CAPABILITY_MATRIX.md`

This lane is about bounded learned route, refusal, serving, and training
truth. It is not implicitly executor-capable just because it shares the
umbrella family name.

### Executor-capable `Psion` / `Tassadar`

This is the bounded executor lane documented through surfaces such as:

- `docs/ROADMAP_TASSADAR.md`
- `docs/ROADMAP_TASSADAR_INDEX.md`
- the workspace-level umbrella roadmap in `docs/ROADMAP_PSION.md`

This lane is the current path toward bounded Percepta/Tassadar-style executor
closure, local-first training runs, export, replacement, and fast-route truth.

### Shared substrate

Both lanes still reuse the same `psionic` substrate for:

- training manifests and launch contracts
- runtime binder and admission
- validator and promotion contracts
- cluster and Tailnet execution surfaces
- receipts, lineage, and replay-safe evidence

## Hard Naming Rules

- Do not describe every `Psion` artifact as already executor-capable.
- Do not describe `Tassadar` as a separate top-level product family inside
  `psionic`.
- Do not flatten generic compact-decoder acceptance into executor acceptance.
- Do not hide the current executor lane behind generic learned-lane prose.

## What This Does Not Claim

This doc does not claim:

- that every `Psion` lane can already execute bounded programs
- that generic compact-decoder `Psion` is already executor-capable
- that the current executor lane is broader than the bounded workload family
  and route truth already retained in repo artifacts

## Canonical Follow-On Docs

- `docs/PSION_EXECUTOR_BASELINE.md` owns the frozen executor baseline record
- `docs/PSION_EXECUTOR_ARTIFACT_NAMING.md` owns the phase-one executor
  artifact naming policy
- `docs/PSION_EXECUTOR_ACCEPTANCE_PROFILE.md` owns the executor-capable
  acceptance profile
- `docs/PSION_EXECUTOR_EVAL_PACKS.md` owns the frozen frequent and promotion
  eval packs
- `docs/PSION_EXECUTOR_BASELINE_TRUTH.md` owns the frozen `trained-v0`
  baseline-truth packet for those packs
- `docs/PSION_EXECUTOR_FORMATTING_AUDIT.md` owns the suite-by-suite prompt,
  normalization, and post-processing audit for those packs
- `docs/PSION_EXECUTOR_DECISION_THRESHOLDS.md` owns the retained variance and
  minimum-meaningful-delta packet for promotion comparisons
- `docs/PSION_EXECUTOR_OWNERSHIP.md` owns named owners and review cadence
- `docs/PSION_EXECUTOR_LOCAL_PROFILE_REFERENCE.md` owns the admitted local
  executor profile catalog
- `docs/PSION_EXECUTOR_MLX_FORWARD_LOAD_PARITY.md` owns the retained MLX
  forward/load parity packet for the admitted Mac profile
- `docs/PSION_EXECUTOR_MLX_CHECKPOINT_COMPATIBILITY.md` owns the retained MLX
  checkpoint save/load compatibility packet for the admitted Mac profile
- `docs/PSION_EXECUTOR_MLX_SMOKE_RUN.md` owns the retained MLX smoke-run
  packet and admitted frequent-pack subset for the admitted Mac profile
- `docs/PSION_EXECUTOR_MLX_DECISION_GRADE_RUN.md` owns the retained MLX-local
  decision-grade packet plus the shared v2 dashboard visibility packet for the
  admitted Mac profile
- `docs/PSION_EXECUTOR_MAC_EXPORT_INSPECTION.md` owns the retained Mac-local
  export inspection and CPU-route validation packet for the admitted Mac
  profile
- `docs/PSION_PROGRAM_MAP.md` owns the generic learned `Psion` family map
- `docs/PSION_ACCEPTANCE_MATRIX.md` owns generic compact-decoder acceptance
- `docs/ROADMAP_TASSADAR.md` remains the repo-local executor-lane bridge
- `docs/ROADMAP_TASSADAR_INDEX.md` remains the executor artifact index

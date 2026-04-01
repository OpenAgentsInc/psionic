# Optimizer Substrate

> Status: `implemented_early` for run specs, candidate manifests, lineage
> state, case and batch evaluation contracts, unified case-eval cache, frontier
> snapshots, persisted search state, cheap-first iteration receipts, reflection
> datasets and prompt builders, proposer receipts, lineage-aware merge receipts,
> and top-level run receipts. Proof lanes land in follow-on issues.

## Purpose

Psionic needs one Rust-native optimizer substrate for bounded offline search.

That substrate is for:

- manifest-backed candidate families
- retained offline datasets
- lineage and frontier tracking
- resumable optimizer state
- machine-readable optimizer receipts

It is not:

- runtime promotion inside Probe
- a hidden Python control plane
- a whole-agent online training loop

Probe consumes this substrate later through explicit artifacts and receipts.

## Current Artifact Surface

The first landed crate is:

- `crates/psionic-optimize`

The current explicit artifact contracts are:

- `OptimizationRunSpec`
  - run id, family id, dataset refs, frontier mode, optional issue ref, and
    coarse search budgets
- `OptimizationCandidateManifest`
  - candidate id, family id, component map, parent ids, provenance refs, and
    stable digest
- `OptimizationCaseManifest`
  - retained case identity, split membership, optional label, metadata,
    evidence refs, and stable digest
- `OptimizationCaseEvaluationReceipt`
  - scalar score, named objective scores, shared feedback, per-component
    feedback, unified cache key, and stable digest
- `OptimizationBatchEvaluationReceipt`
  - case receipts, aggregate scalar and objective totals, cache hit or miss
    accounting, and stable digest
- `OptimizationEvaluationCache`
  - one cache keyed by candidate-manifest digest plus case digest instead of
    duplicated adapter and engine caches
- `OptimizationFrontierSnapshot`
  - per-case winners, per-objective winners, optional cartesian winners, and
    hybrid retained candidate ids
- `OptimizationSearchState`
  - current candidate, accepted validation batches, unified cache, iteration
    receipts, persisted JSON state, and stable digest
- `OptimizationEngine`
  - deterministic cheap-first loop that minibatch-evaluates the current and
    proposed candidates, full-evaluates only accepted proposals on validation,
    and emits iteration receipts plus a final run receipt
- `OptimizationReflectiveDataset`
  - ASI-style reflective rows built from typed evaluator feedback plus retained
    case labels and selected component snapshots
- `OptimizationReflectionPrompt`
  - component-scoped prompt artifacts rendered from reflective datasets
- `OptimizationProposerReceipt`
  - proposer kind, selected components, rendered prompts, candidate diffs, and
    stable receipt identity for each mutation attempt
- `OptimizationReflectiveMutationProposer`
  - generic reflection pipeline over component selection, typed-feedback
    dataset building, prompt rendering, and bounded mutation strategies
- `OptimizationMergeReceipt`
  - merge attempt, accept, and reject receipts with explicit parent ids,
    ancestor ids, component-source maps, and gating score floors
- `OptimizationLineageAwareMergeProposer`
  - shared-ancestor merge pipeline that recombines compatible retained
    candidates and forces merged candidates to beat both parents on the same
    minibatch
- `OptimizationLineageState`
  - materialized candidates, discovery order, retained candidates, persisted
    JSON state, and stable digest
- `OptimizationRunReceipt`
  - run id, run spec digest, lineage state digest, retained candidates, frontier
    refs, stop reason, and claim boundary

## Boundaries

Psionic owns:

- optimizer artifact identity
- lineage state
- resumable run state
- later evaluation and search receipts

Probe owns:

- runtime transcripts and approvals
- retained runtime-derived dataset export
- candidate family definitions for Probe behavior
- final promotion and runtime adoption

## Next Work

Follow-on optimizer issues extend this substrate with:

- proof reports over bounded module families

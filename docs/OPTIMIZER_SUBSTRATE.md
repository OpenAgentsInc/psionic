# Optimizer Substrate

> Status: `implemented_early` for run specs, candidate manifests, lineage
> state, and top-level run receipts. Search, evaluation, reflection, and merge
> land in follow-on issues.

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

- typed evaluation contracts and feedback
- one unified optimizer cache
- frontier snapshots across cases and objectives
- cheap-first search loop
- component-aware reflection
- lineage-aware merge
- proof reports over bounded module families

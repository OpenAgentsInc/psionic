# Training Execution Evidence Reference

> Status: canonical `TVIZ-PSI-5` / `#644` record, updated 2026-03-28 after
> aligning the provider-neutral final evidence bundle family with the new
> track-aware visualization and explorer surfaces in
> `crates/psionic-train/src/training_execution_evidence_bundle.rs`.

This document records the first shared final-evidence schema family across the
current training execution classes.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-training-execution-evidence-bundle.sh
```

## What Landed

`psionic-train` now owns one final evidence bundle family with:

- one shared `TrainingExecutionEvidenceBundle`
- per-segment proof for single-node, dense-distributed, contributor-window,
  validator-only, and hybrid runs
- explicit `visualization_surface_links` that map one retained score or
  explorer surface artifact back to the supporting evidence refs already
  present in the bundle
- explicit launch, runtime, checkpoint, metric, visualization, validator, and
  after-action references
- one shared validator and promotion contract id carried by the bundle
- explicit validator dispositions of `accepted`, `quarantined`, `rejected`,
  and `replay_required`
- explicit promotion outcomes of `promoted_revision`, `held_no_promotion`, and
  `refused_promotion`
- explicit successful, degraded-success, refused, and failed dispositions
- the generator binary `training_execution_evidence_bundle`
- the checker `scripts/check-training-execution-evidence-bundle.sh`
- the fixture `fixtures/training/provider_neutral_training_execution_evidence_bundle_v1.json`

## Surface Links

The evidence bundle now makes one machine-legible jump contract explicit:

- start from the exact retained visualization or explorer artifact path
- resolve the matching `visualization_surface_link`
- use `supporting_evidence_paths` to open the retained proof set without
  pane-local heuristics

The canonical bundle now carries explicit links for:

- live Psion single-node `v2` score surfaces
- distributed PGOLF `v2` score surfaces
- summary-only validator `v2` score surfaces
- the shared `v2` run index
- the HOMEGOLF score-closeout `v2` surface
- the bounded XTRAIN `v2` score lane
- the decentralized XTRAIN explorer snapshot and explorer index

## Current Honest Boundary

This issue closes the schema family, not the entire runtime stack.

It proves:

- finalizers can emit one bundle family across differing execution classes
- refusal and degraded-success posture stay explicit
- hybrid runs can carry multiple execution classes without a hybrid-only proof
  family
- track-aware score surfaces and decentralized explorer surfaces can be treated
  as first-class evidence jump points instead of viewer-only sidecars

It does not prove:

- that every segment in the canonical example has already happened in one real
  production run
- app rendering behavior
- mixed-backend dense portability

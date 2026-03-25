# Contributor Program Lineage Reference

> Status: canonical `XTRAIN-14` / `#530` record, updated 2026-03-25 after
> landing the contributor program-lineage bridge in
> `crates/psionic-train/src/contributor_program_lineage.rs`.

This document records the bridge that ties validated contributor windows back
to the same dataset family, checkpoint family, and shared policy revision used
by the cross-provider pretraining program.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-contributor-program-lineage.sh
```

## What Landed

`psionic-train` now owns one contributor program-lineage contract that freezes:

- one shared input policy revision for all admitted contributor windows
- the canonical pretraining dataset family and checkpoint family those windows
  must use
- dense-rank lineage anchors that contributor windows trace back to
- one promotion contract per contributor window with explicit no-promotion,
  quarantine, reject, and replay-required posture

The landed surface includes:

- `ContributorProgramLineageContract`
- `ContributorWindowProgramBinding`
- `ContributorPromotionContract`
- `DenseProgramLineageAnchor`
- the binary `contributor_program_lineage`
- the checker `scripts/check-contributor-program-lineage.sh`
- the committed fixture `fixtures/training/contributor_program_lineage_v1.json`

## Why This Matters

Before this issue, contributor windows and dense-rank work were related in the
hybrid plan, but the contributor side still did not freeze one shared program
policy revision and one shared promotion contract.

After this issue, contributor work can be traced back to the same shared
pretraining lineage instead of floating beside it.

## Current Limits

This issue intentionally does not claim:

- that bounded local swarm lanes have all already migrated to the canonical
  pretraining dataset family
- that contributor windows replace dense-rank training
- same-job mixed-backend dense closure


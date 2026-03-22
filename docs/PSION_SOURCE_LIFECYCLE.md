# Psion Source Lifecycle

> Status: implemented_early as of 2026-03-22 via `PSION-3` / issue `#359`.

This document freezes the first source-lifecycle and downstream lineage
contract for the `Psion` learned-model lane.

It builds on `docs/PSION_CORPUS_ADMISSION.md` and answers four questions:

- what current lifecycle state each reviewed source can occupy
- which transitions between those states are allowed
- how downstream tokenizer, corpus, SFT, checkpoint, and benchmark artifacts
  retain source lineage
- what receipt must be produced when a source posture change forces retraining,
  capability withdrawal review, or depublication review

## Canonical Artifacts

- machine-readable lifecycle contract:
  `crates/psionic-data/src/psion_source_lifecycle.rs`
- canonical lifecycle fixture:
  `fixtures/psion/lifecycle/psion_source_lifecycle_manifest_v1.json`
- canonical downstream-lineage fixture:
  `fixtures/psion/lifecycle/psion_artifact_lineage_manifest_v1.json`

The stable schema versions are:

- `psion.source_lifecycle.v1`
- `psion.artifact_lineage.v1`
- `psion.source_impact_analysis.v1`

## Lifecycle States

Each reviewed source must occupy one current lifecycle state:

- `admitted`
- `restricted`
- `evaluation_only`
- `withdrawn`
- `rejected`

This state is separate from the source's original admission review. A source
can therefore keep:

- its original `admission_rights_posture`
- its current `current_rights_posture`
- the latest transition trigger and rationale

without pretending that rights posture never changes after initial review.

## Allowed Transitions

The first explicit transition graph is:

- `admitted -> admitted | restricted | evaluation_only | withdrawn`
- `restricted -> restricted | admitted | evaluation_only | withdrawn`
- `evaluation_only -> evaluation_only | restricted | withdrawn`
- `withdrawn -> withdrawn | restricted | evaluation_only | rejected`
- `rejected -> rejected | restricted | admitted`

This is intentionally explicit and narrower than "operators can change fields as
needed."

## Downstream Lineage

The first lineage manifest retains source ids through:

- tokenizer artifacts
- tokenized corpora
- SFT artifacts
- promoted checkpoints
- benchmark artifacts

Each downstream artifact row carries:

- one stable artifact id
- one stable digest
- explicit parent artifact references where applicable
- explicit `source_ids`

This gives the repo one bounded answer to "what downstream artifacts depend on
source `X`?" without forcing operators to reconstruct the graph from notes or
logs.

## Impact Analysis Receipt

When a source changes posture or state, the repo now has one explicit impact
receipt shape:

- previous state and next state
- previous rights posture and next rights posture
- the change trigger
- affected tokenizer artifacts
- affected tokenized corpora
- affected SFT artifacts
- affected checkpoints
- affected benchmark artifacts
- required downstream actions

The required actions can include:

- raw-source re-ingest
- tokenizer rebuild
- tokenized-corpus rebuild
- retraining review
- benchmark invalidation review
- capability-matrix review
- depublication review

## Mechanical Enforcement

`psionic-data` now exposes:

- `PsionSourceLifecycleManifest`
- `PsionArtifactLineageManifest`
- `PsionSourceImpactAnalysisReceipt`
- `PsionSourceLifecycleState`
- `PsionLifecycleChangeTrigger`
- `PsionLifecycleImpactAction`

The validation path rejects:

- lifecycle manifests that do not cover every reviewed admission source
- lifecycle rows whose admission posture snapshot no longer matches the
  admission manifest
- invalid state and posture combinations such as `evaluation_only` with a
  non-`evaluation_only` posture
- artifact-lineage rows that reference unknown sources, tokenizers, corpora, or
  SFT artifacts
- lifecycle transitions outside the declared graph

The impact-analysis path turns one source change into a bounded action list
instead of leaving retraining, benchmark invalidation, or served-claim
withdrawal to operator memory.

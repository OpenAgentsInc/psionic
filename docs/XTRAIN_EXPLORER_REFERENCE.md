# XTRAIN Explorer Reference

> Status: canonical `TVIZ-PSI-4` / `#643` record, updated 2026-03-28 after
> landing the first decentralized XTRAIN explorer artifact family.

## What This Closes

Psionic now owns a dedicated explorer-oriented artifact family for decentralized
`XTRAIN`.

The new family exists because decentralized `XTRAIN` should not be squeezed
into the same run-centric contract as bounded training runs. The explorer
artifact family keeps participant, window, checkpoint, settlement, and
promotion truth explicit, then links to the bounded run-centric score lane as a
sibling surface where relevant.

The canonical artifacts live in:

- `crates/psionic-train/src/xtrain_explorer_artifacts.rs`
- `crates/psionic-train/src/bin/xtrain_explorer_artifacts.rs`
- `scripts/check-xtrain-explorer-artifacts.sh`
- `fixtures/training/xtrain_explorer_snapshot_v1.json`
- `fixtures/training/xtrain_explorer_index_v1.json`

## Artifact Shape

The family freezes two typed artifacts:

- `psionic.xtrain_explorer_snapshot.v1`
- `psionic.xtrain_explorer_index.v1`

The snapshot carries the minimum explorer state required by the roadmap spec:

- participant nodes with role, execution-class, availability, and state truth
- participant edges for validator scoring, checkpoint sync, and explicit
  refusals
- one retained active-window row with dataset, miner-session, vote, checkpoint,
  and settlement bindings
- one retained checkpoint row with promotion outcome and disagreement receipts
- explorer event rows for session closeout, held promotion, settlement
  publication, and participant refusal
- source artifacts binding the snapshot back to registry, miner-protocol,
  consensus, settlement, curated-run, and bounded-XTRAIN bundle truth

The index stays separate from the run-centric remote-training index and only
does decentralized explorer discovery:

- current snapshot identity and digest
- network and epoch identity
- active window id
- participant, held-checkpoint, and settlement-publication counts
- one semantic summary for explorer discovery

## Relationship To The Shared Run-Centric Bundle Family

The explorer family intentionally does not replace
`psionic.remote_training_visualization_bundle.v2`.

Instead:

- the explorer snapshot owns decentralized network state
- the bounded `XTRAIN -> PGOLF` bundle owns the local-reference score lane
- the snapshot links back to the bounded bundle through one explicit
  `run_surface_link`

That split keeps the UI honest:

- operator graph and window questions come from the explorer snapshot
- bounded score-band questions come from the run-centric bundle
- discovery of decentralized explorer snapshots stays separate from discovery
  of bounded training runs

The final provider-neutral evidence bundle now also carries explicit
surface-to-evidence links for the explorer snapshot and index, so an explorer
consumer can jump into registry, miner-protocol, consensus, settlement, and
sibling bounded-XTRAIN proof without ad hoc mapping logic.

## Canonical First View

The committed canonical snapshot is designed to drive the first required
`XTRAIN Explorer` pane:

- participant graph:
  - Google as active miner or validator or relay participant
  - RunPod as active checkpoint-authority mirror
  - local RTX 4080 as an explicit refused standby participant
  - local Apple MLX as a held miner or validator participant
- active window status:
  - current window `window1231`
  - retained miner sessions
  - retained validator votes
  - held-no-promotion checkpoint candidate
  - signed settlement publication still present

This gives the app enough machine truth to render:

- participant graph plus role posture
- active window summary
- held checkpoint promotion state
- settlement publication posture
- event drilldown and provenance inspection

## What This Does Not Claim

This family does not claim:

- one generic explorer for every future decentralized topology
- leaderboard-style score comparison across decentralized and bounded surfaces
- that settlement publication implies checkpoint promotion succeeded

It only claims the minimum explorer-oriented machine truth required to power the
first honest decentralized `XTRAIN Explorer` pane.

# PSION Plugin-Conditioned SFT

> Status: canonical `PSION_PLUGIN-13` stage contract for the first
> plugin-conditioned `agentic_sft` lane, written 2026-03-22 after landing the
> stage manifest, receipt, and bounded output bundle in `psionic-train`.

This document freezes the first machine-checkable stage contract for
plugin-conditioned supervised fine-tuning.

It sits above the already-landed canonical plugin-conditioned datasets and the
five host-native plugin benchmark packages.

It is now reused by both the bounded host-native reference lane and the bounded
mixed host-native plus guest-artifact reference lane.

## Canonical Artifacts

- `crates/psionic-train/src/psion_plugin_conditioned_sft.rs` owns the stage
  manifest, trace binding, benchmark binding, evaluation-hook, stage receipt,
  and bounded run-bundle contracts.
- `crates/psionic-train/examples/psion_plugin_conditioned_sft_fixtures.rs`
  writes the committed reference manifest, receipt, and run bundle.
- `fixtures/psion/plugins/training/psion_plugin_conditioned_sft_v1/` carries
  the committed reference outputs for the first stage contract.

Stable schema versions:

- `psionic.psion.plugin_conditioned_sft_stage_manifest.v1`
- `psionic.psion.plugin_conditioned_sft_stage_receipt.v1`
- `psionic.psion.plugin_conditioned_sft_run_bundle.v1`

## What This Stage Freezes

The first stage contract now freezes:

- one dedicated plugin-conditioned `agentic_sft` stage identity
- direct binding to the committed dataset ref, stable dataset identity, and
  bundle digest
- one trace-binding row per committed train record
- one benchmark-binding row per host-native benchmark family
- explicit later evaluation hooks instead of implicit “we will evaluate later”
- explicit replay-class and receipt-ref posture per accepted trace
- one bounded stage receipt and one bounded output bundle

The stage is intentionally narrow:

- it assumes the current stage graph ends in `general_sft -> agentic_sft`
- it binds one committed plugin-conditioned dataset bundle at a time
- it binds only the five current host-native benchmark families
- it keeps later held-out benchmark and replay review hooks explicit

## Stage Graph Boundary

The stage contract does not create a new training-stage taxonomy.

It reuses the existing train-program stage graph:

- `general_sft`
- `agentic_sft`

The plugin-conditioned stage is the canonical bounded `agentic_sft` contract
for learned plugin use.

That means:

- the preceding `general_sft` stage must already be completed
- the current `agentic_sft` stage must own the accepted plugin-conditioned
  traces
- the stage receipt is only valid once that `agentic_sft` stage is completed

## Input Truth

The first manifest binds directly to:

- `fixtures/psion/plugins/datasets/psion_plugin_conditioned_dataset_v1/psion_plugin_conditioned_dataset_bundle.json`
- committed dataset ref
  `dataset://openagents/psion/plugin_conditioned_host_native_reference`
- committed stable dataset identity
  `dataset://openagents/psion/plugin_conditioned_host_native_reference@2026.03.22.v1`

The same stage contract is also now reused by the mixed reference lane against:

- `fixtures/psion/plugins/datasets/psion_plugin_mixed_conditioned_dataset_v1/psion_plugin_mixed_conditioned_dataset_bundle.json`
- committed stable dataset identity
  `dataset://openagents/psion/plugin_conditioned_mixed_reference@2026.03.22.v1`

For each accepted trace, the manifest preserves:

- canonical train-record id
- accepted agentic trace id
- accepted trace-lineage digest
- controller surface
- route label
- outcome label
- replay-class ids for the preserved full record or bounded admitted subtrace
- runtime receipt refs for the preserved full record or bounded admitted
  subtrace

That keeps stage inputs tied to real plugin-runtime truth rather than synthetic
string tool calls.

## Evaluation And Replay Posture

The first manifest also binds:

- all five host-native plugin benchmark families by bundle ref and receipt
  digest
- post-stage held-out benchmark hooks
- pre-promotion benchmark hooks
- a pre-promotion replay-and-receipt review hook

The stage config now requires:

- explicit receipt boundaries
- explicit replay-class coverage
- explicit held-out benchmark hooks before promotion

That means later audits no longer need to guess what “evaluate the
plugin-conditioned stage” means.

## Output Boundary

The bounded output bundle for this issue contains:

- the stage-program state
- the plugin-conditioned stage manifest
- the plugin-conditioned stage receipt

It does not yet claim:

- model-family closure
- trained checkpoint quality
- `networked_read_only` authoring closure
- served capability-matrix closure
- Google operator proof

Guest-artifact training coverage is now exercised by the mixed reference lane,
but broader guest-artifact publication and operator proof remain later
`PSION_PLUGIN-*` issues.

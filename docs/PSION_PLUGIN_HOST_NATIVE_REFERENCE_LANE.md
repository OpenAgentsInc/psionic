# PSION Plugin Host-Native Reference Lane

> Status: canonical `PSION_PLUGIN-15` record for the first bounded trained
> host-native plugin-conditioned lane, written 2026-03-22 after landing the
> runnable reference bundle in `psionic-train`.

This document records the first executable learned plugin-use lane in the repo.

It is intentionally narrow.

The lane is explicitly limited to the currently fully proved starter-plugin
authoring class:

- host-native
- capability-free
- local deterministic

It does **not** claim `networked_read_only`, secret-backed, stateful, or
guest-artifact competence.

It also does **not** claim accelerator-backed plugin-conditioned training. The
current host-native reference lane is a bounded learned reference artifact and
evaluation lane, not the canonical GPU-training target for Google audits.

## Canonical Artifacts

- `crates/psionic-train/src/psion_plugin_host_native_reference_lane.rs` owns
  the bounded run bundle, learned artifact, and benchmark-delta receipt.
- `crates/psionic-train/examples/psion_plugin_host_native_reference_lane.rs`
  writes the committed reference bundle.
- `fixtures/psion/plugins/training/psion_plugin_host_native_reference_lane_v1/psion_plugin_host_native_reference_run_bundle.json`
  is the committed reference output.

Stable schema versions:

- `psionic.psion.plugin_host_native_reference_model_artifact.v1`
- `psionic.psion.plugin_host_native_reference_evaluation_receipt.v1`
- `psionic.psion.plugin_host_native_reference_run_bundle.v1`

## What The Lane Actually Does

The first host-native reference lane now:

- derives bounded training examples from the canonical plugin-conditioned train
  split
- removes out-of-scope `networked_read_only` invocations from those examples
- runs a real bounded `general_sft -> agentic_sft` stage program for the
  filtered local-deterministic subtraces
- binds the stage to the plugin-conditioned compact-decoder reference config
- emits one learned artifact over the proved local-deterministic plugin set
- reports benchmark deltas against a named non-plugin baseline
- keeps out-of-scope benchmark items explicit instead of flattening them into
  competence claims

Its retained stage bundle now also binds:

- the unversioned dataset ref for the host-native dataset family
- the committed stable dataset identity
  `dataset://openagents/psion/plugin_conditioned_host_native_reference@2026.03.22.v1`
- a max-plugin-call ceiling derived from the retained local-deterministic trace
  bindings instead of a stale hand-written constant

## Boundary

The learned artifact is not presented as broad plugin competence.

It is only the first bounded host-native reference lane.

Its claim boundary is:

- supported only for the proved local-deterministic plugin class
- benchmark deltas reported only on eligible in-boundary items
- networked benchmark items excluded explicitly
- no publication widening from this lane alone

## Why The Out-Of-Scope Split Matters

The current canonical plugin-conditioned dataset and benchmark packages already
contain some `networked_read_only` truth surfaces.

This issue does not pretend those are already learned.

Instead, the reference lane:

- trains only on the capability-free local-deterministic subtraces
- keeps the stage receipt and learned artifact inside that boundary
- reports benchmark coverage and exclusions per family

That makes the first trained lane honest enough to support the later capability
matrix issue and the later mixed-lane comparison issue.

The current lane remains a valid bounded reference surface for receipts,
datasets, and benchmark deltas. It is not a valid GPU proof target until a
later accelerated plugin-conditioned lane exists and emits explicit CUDA-backed
training evidence.

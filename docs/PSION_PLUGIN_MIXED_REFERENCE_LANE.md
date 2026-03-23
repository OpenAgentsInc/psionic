# PSION Plugin Mixed Reference Lane

> Status: canonical `PSION_PLUGIN-26` record for the first bounded trained
> mixed plugin-conditioned lane, written 2026-03-22 after landing the runnable
> mixed reference bundle in `psionic-train`.

This document records the first executable learned lane that keeps both
host-native plugin truth and one bounded guest-artifact training example in the
same retained run bundle.

It is still intentionally narrow.

The lane does **not** widen publication posture or claim broad guest-artifact
competence. It only proves that the current bounded training substrate can bind
to the mixed dataset identity, train one mixed learned artifact, and compare
that bounded result against the committed host-native reference lane.

It also does **not** claim accelerator-backed plugin-conditioned training. The
current mixed reference lane is a bounded learned comparison artifact and
evidence lane, not a valid Google GPU training proof target.

## Canonical Artifacts

- `crates/psionic-train/src/psion_plugin_mixed_reference_lane.rs` owns the
  bounded mixed run bundle, learned artifact, and comparison receipt.
- `crates/psionic-train/examples/psion_plugin_mixed_reference_lane.rs` writes
  the committed mixed run bundle.
- `crates/psionic-train/examples/psion_plugin_conditioned_mixed_compact_decoder_fixtures.rs`
  writes the committed mixed compact-decoder config.
- `fixtures/psion/plugins/training/psion_plugin_mixed_reference_lane_v1/psion_plugin_mixed_reference_run_bundle.json`
  is the committed mixed reference output.
- `fixtures/psion/plugins/models/psion_plugin_conditioned_mixed_compact_decoder_reference_config_v1.json`
  is the committed mixed compact-decoder config.

Stable schema versions:

- `psionic.psion.plugin_mixed_reference_model_artifact.v1`
- `psionic.psion.plugin_mixed_reference_evaluation_receipt.v1`
- `psionic.psion.plugin_mixed_reference_run_bundle.v1`

## What The Lane Actually Does

The first mixed reference lane now:

- derives bounded training examples from the mixed plugin-conditioned train
  split
- keeps the admitted host-native and guest-artifact receipt refs explicit
- binds the shared plugin-conditioned stage contract directly to the mixed
  stable dataset identity
- reuses the compact-decoder reference family with a dedicated mixed lane id,
  model id, checkpoint prefix, and export directory
- emits one learned artifact that records guest-artifact participation
- reports benchmark comparison rows against the committed host-native reference
  lane instead of flattening the result into architecture intent

## Comparison Boundary

The comparison receipt is intentionally explicit:

- comparison label:
  `psion_plugin_host_native_reference`
- comparison run bundle:
  `fixtures/psion/plugins/training/psion_plugin_host_native_reference_lane_v1/psion_plugin_host_native_reference_run_bundle.json`
- mixed dataset identity:
  `dataset://openagents/psion/plugin_conditioned_mixed_reference@2026.03.22.v1`

That keeps the mixed lane honest about what changed:

- one additional admitted guest-artifact train example
- the same held-out host-native benchmark suite
- benchmark deltas measured relative to the host-native reference lane rather
  than a vague “should be better” claim

## Boundary

This issue does not yet claim:

- broad guest-artifact publication readiness
- `networked_read_only` authoring closure
- secret-backed or stateful plugin competence
- Google operator proof for the mixed lane
- cluster-scale plugin-conditioned training readiness

It is only the first truthful bounded mixed reference lane.

## First Google Operator Proof

The first real Google-hosted mixed operator proof now lives at:

- `docs/audits/2026-03-22-openagentsgemini-first-google-mixed-plugin-conditioned-run-audit.md`

That audit proves the bounded mixed lane on real operator infrastructure,
retains the guest benchmark bundle plus the run-derived mixed capability matrix
and served posture, and keeps the claim boundary explicit:

- real single-node Google operator proof now exists
- mixed host-native plus guest-artifact evidence is retained end to end
- the lane was still CPU-bound on the L4 host, so this is not yet
  accelerator-backed throughput proof

That CPU-bound mixed run remains valid as an operator and boundary proof. It is
not a valid GPU-backed training proof until a later accelerated mixed or
host-native lane is wired to a real CUDA trainer.

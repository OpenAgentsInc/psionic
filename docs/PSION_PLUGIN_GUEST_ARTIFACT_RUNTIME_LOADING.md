# PSION Plugin Guest-Artifact Runtime Loading

> Status: canonical `PSION_PLUGIN-20` bounded guest-artifact runtime-loading
> contract, written 2026-03-22.

This document freezes the first runtime-loading path for the later
guest-artifact lane.

It is intentionally narrow.

The current loader only proves:

- manifest-bound byte admission
- digest verification
- minimal Wasm header and version checks
- host-owned capability mediation
- typed refusal paths for load-time failures

It does **not** yet prove:

- bridge exposure
- catalog exposure
- controller use
- public plugin support

## Canonical Artifacts

- `docs/PSION_PLUGIN_GUEST_ARTIFACT_RUNTIME_LOADING.md` is the canonical
  human-readable contract.
- `crates/psionic-runtime/src/psion_plugin_guest_artifact_runtime_loading.rs`
  owns the typed loader, refusal codes, and runtime-loading bundle.
- `crates/psionic-runtime/examples/psion_plugin_guest_artifact_runtime_loading.rs`
  writes the committed loading bundle fixture.
- `fixtures/psion/plugins/guest_artifact/psion_plugin_guest_artifact_runtime_loading_v1.json`
  is the canonical runtime-loading bundle.

Stable schema version:

- `psionic.psion.plugin_guest_artifact_runtime_loading.v1`

## What Counts As Load Here

For this tranche, “load” means:

1. validate the guest-artifact manifest
2. verify the artifact bytes against the manifest digest
3. verify the admitted Wasm magic and version header
4. keep mount and capability mediation host-owned
5. produce one typed loaded-artifact identity surface or one typed refusal

That is enough to prove a bounded runtime-loading path without jumping ahead to
execution.

## Typed Refusal Surface

The loader now carries explicit refusal codes for:

- `digest_mismatch`
- `malformed_artifact`
- `unsupported_format`
- `capability_mount_denied`

Those are load-time refusals, not generic host errors.

## Host-Owned Boundaries

The loader keeps all of the following host-owned:

- capability mediation
- mount policy
- publication posture
- artifact-admission boundary

That means the guest artifact still may not:

- self-authorize capabilities
- self-publish
- widen itself into arbitrary binary loading

## Current Claim Boundary

This runtime-loading proof does not claim:

- guest-artifact execution
- receipt-equivalent invocation
- admitted guest-plugin controller use
- present-tense Wasm plugin support

It only claims that one digest-bound guest artifact can be admitted into a
bounded host-owned loader with explicit typed refusal paths.

The next layer above this contract now lives in
`docs/PSION_PLUGIN_GUEST_ARTIFACT_INVOCATION.md`.

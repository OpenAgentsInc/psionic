# PSION Plugin Guest-Artifact Invocation

> Status: canonical `PSION_PLUGIN-21` bounded guest-artifact invocation
> contract, written 2026-03-22.

This document freezes the first receipt-equivalent invocation path for the
later guest-artifact plugin lane.

It is intentionally narrow.

The current invocation proof closes only:

- one digest-bound Wasm guest artifact
- one bounded packet echo invocation
- one host-native-equivalent invocation receipt shape
- one replay-explicit projected tool-result envelope
- explicit typed refusal paths for schema, size, load, and runtime refusal

It does **not** yet close:

- shared catalog admission
- weighted-controller breadth
- publication
- arbitrary guest-artifact admission
- broad third-party Wasm plugin support

## Canonical Artifacts

- `docs/PSION_PLUGIN_GUEST_ARTIFACT_INVOCATION.md` is the canonical
  human-readable contract.
- `crates/psionic-runtime/src/psion_plugin_guest_artifact_invocation.rs`
  owns the bounded invocation path, tool projection, typed refusal mapping,
  receipt-equivalent evidence, and invocation bundle.
- `crates/psionic-runtime/examples/psion_plugin_guest_artifact_invocation.rs`
  writes the committed invocation bundle fixture.
- `fixtures/psion/plugins/guest_artifact/psion_plugin_guest_artifact_invocation_v1.json`
  is the canonical invocation bundle.

Stable schema version:

- `psionic.psion.plugin_guest_artifact_invocation.v1`

## What Counts As Invocation Here

For this tranche, “invocation” means:

1. validate the guest-artifact manifest
2. validate digest-bound loading against the declared artifact bytes
3. instantiate the guest Wasm module under host-owned bounds
4. write one bounded packet into guest memory
5. call the canonical `handle_packet` export
6. read one bounded output packet from guest memory
7. emit the same class of invocation receipt and projected tool-result envelope
   used by host-native starter plugins

That is enough to prove receipt-equivalent guest invocation without claiming
broader plugin-surface closure.

## Receipt And Bridge Equivalence

The guest-artifact invocation bundle now reuses:

- `StarterPluginInvocationReceipt`
- `StarterPluginInvocationStatus`
- `StarterPluginProjectedToolResultEnvelope`
- `StarterPluginToolProjection`

That means the guest lane is not allowed to invent weaker receipt evidence or a
separate replay surface.

The current reference tool is:

- `plugin_example_echo_guest`

And its replay class remains explicit:

- `guest_artifact_digest_replay_only.v1`

## Typed Refusal Surface

The invocation path now keeps typed refusals explicit for:

- `schema_invalid`
- `packet_too_large`
- `guest_artifact_load_refusal`
- `guest_artifact_runtime_unavailable`

Those are emitted as receipt-bound refusal results, not dropped as host-local
errors.

## Current Claim Boundary

This invocation proof does not claim:

- guest-artifact catalog exposure
- controller parity with the full starter-plugin lane
- weighted-controller promotion breadth
- user-facing plugin publication
- generic Wasm plugin support

It only claims that one digest-bound guest artifact can execute one bounded
packet invocation while preserving the host-native receipt, replay, and typed
refusal discipline.

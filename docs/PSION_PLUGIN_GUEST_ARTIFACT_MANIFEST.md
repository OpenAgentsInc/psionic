# PSION Plugin Guest-Artifact Manifest

> Status: canonical `PSION_PLUGIN-19` guest-artifact manifest and identity
> contract, written 2026-03-22.

This document freezes the first digest-bound manifest contract for the bounded
guest-artifact starter-plugin lane.

It is a contract-only tranche.

It does **not** add runtime loading, controller integration, or publication.

## Canonical Artifacts

- `docs/PSION_PLUGIN_GUEST_ARTIFACT_MANIFEST.md` is the canonical
  human-readable contract.
- `crates/psionic-runtime/src/psion_plugin_guest_artifact_manifest.rs` owns the
  typed manifest schema, identity surface, and validators.
- `crates/psionic-runtime/examples/psion_plugin_guest_artifact_manifest.rs`
  writes the committed reference fixtures.
- `fixtures/psion/plugins/guest_artifact/psion_plugin_guest_artifact_manifest_v1.json`
  is the canonical reference manifest fixture.
- `fixtures/psion/plugins/guest_artifact/psion_plugin_guest_artifact_identity_v1.json`
  is the canonical digest identity surface derived from that manifest.

Stable schema versions:

- `psionic.psion.plugin_guest_artifact_manifest.v1`
- `psionic.psion.plugin_guest_artifact_identity.v1`

## What This Contract Freezes

The first guest-artifact contract requires all of the following to be explicit:

- plugin id
- plugin version
- artifact id
- artifact format
- artifact digest
- artifact byte length
- packet ABI version
- guest export name
- input schema id
- success output schema id
- typed refusal schema ids
- replay class id
- trust tier
- publication posture
- provenance digests
- evidence settings

That is the minimum surface needed to keep the bounded guest-artifact starter-
plugin lane digest-bound and attributable instead of drifting into vague binary
loading.

## Digest Identity Surface

The companion identity surface exists so later runtime and training work can
bind one guest artifact to:

- manifest id
- manifest digest
- plugin id and version
- artifact id and digest
- packet ABI version
- guest export name
- trust tier
- publication posture

The identity surface is intentionally narrower than the full manifest. It is
the digest-bound join surface later issues may reuse.

## Validation Rules

The validators currently reject manifests that are:

- missing any required identity field
- missing the artifact digest
- missing typed refusal schema ids
- missing provenance digests
- missing evidence settings
- not `packet.v1`
- not `handle_packet`
- not operator-internal and publication-blocked
- digest-drifted from the declared manifest contents

This keeps the current contract fail-closed.

## Current Boundary

This contract does not claim:

- runtime loading
- execution
- bridge exposure
- catalog exposure
- controller admission
- public plugin publication
- generic Wasm plugin support

The committed reference manifest is only a manifest-and-identity proof object.
It is not evidence that the guest artifact is already runnable.

## Relationship To The Direction Record

`docs/PSION_PLUGIN_GUEST_ARTIFACT_DIRECTION.md` remains the higher-level
product-direction record.

This manifest doc is the first concrete follow-on contract inside that bounded
direction:

- guest artifacts remain later and separate
- the lane remains digest-bound
- the lane remains trust-tiered
- the lane remains publication-blocked

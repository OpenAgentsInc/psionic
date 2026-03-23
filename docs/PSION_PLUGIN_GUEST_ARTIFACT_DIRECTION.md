# PSION Plugin Guest-Artifact Direction

> Status: canonical `PSION_PLUGIN-18` product-direction record for the bounded
> digest-bound guest-artifact starter-plugin lane, written 2026-03-22 after the host-native
> `networked_read_only` substrate proof closed.

This document freezes the product direction for guest-artifact plugin support
in the Psion plugin convergence program.

## Decision

One bounded digest-bound guest-artifact starter-plugin lane is now live.

It remains intentionally narrow and internal.

It does **not** authorize generic Wasm/plugin support.

The current live plugin substrate is:

- host-native
- operator-internal
- runtime-owned
- receipt-backed
- publication-blocked

That substrate now includes:

- the capability-free local deterministic user-authored path
- one manual `networked_read_only` user-authored proof
- one digest-bound guest-artifact starter-plugin proof

It does not include arbitrary guest-artifact loading or broad user-provided
binary admission.

## Why This Decision Exists

The historical alpha planning note kept a future user-provided Wasm lane open.
The current repo, however, has converged on a host-native starter-plugin
platform first.

Without this decision record, historical guest-artifact planning language can
be misread as current product truth.

This doc closes that ambiguity:

- one digest-bound guest-artifact starter-plugin lane is now real
- host-native starter plugins remain the broad present-tense user-authoring
  substrate
- any broader future guest-artifact tranche must earn its own bounded contracts
  and evidence

## Required Boundary For Any Future Guest-Artifact Lane

Any guest-artifact lane beyond the current bounded proof must remain:

- digest-bound
- trust-tiered
- authority-gated
- receipt-equivalent
- replay-explicit
- publication-blocked
- operator-internal until a later explicit publication tranche says otherwise

The future lane must not rely on:

- floating artifact identity
- implicit runtime loading
- silent trust inheritance from host-native starter plugins
- public plugin marketplace language
- arbitrary external binary execution claims

## Direction For The Follow-On Issues

The next guest-artifact issues remain valid only inside this bounded direction:

1. digest-bound manifest and identity contract
2. bounded runtime loading path
3. receipt-equivalent invocation path
4. one user-provided Wasm plugin end to end
5. trust-tier and authority-gate rows

Those issues are for one narrow admitted guest-artifact class only.

The first concrete contract for that lane now lives in:

- `docs/PSION_PLUGIN_GUEST_ARTIFACT_MANIFEST.md`
- `crates/psionic-runtime/src/psion_plugin_guest_artifact_manifest.rs`
- `docs/PSION_PLUGIN_GUEST_ARTIFACT_RUNTIME_LOADING.md`
- `crates/psionic-runtime/src/psion_plugin_guest_artifact_runtime_loading.rs`
- `docs/PSION_PLUGIN_GUEST_ARTIFACT_INVOCATION.md`
- `crates/psionic-runtime/src/psion_plugin_guest_artifact_invocation.rs`

The current proof is not authorization for:

- generic Wasm plugin support
- arbitrary user binary support
- publication widening
- universality claims

## Non-Goals

This decision record does not:

- authorize generic guest-artifact runtime loading
- authorize arbitrary manifest admission
- authorize user-provided Wasm plugins beyond the single digest-bound admitted
  class
- change the current served capability matrix
- widen publication or serving posture

## Current Honest Summary Language

Allowed:

- “one bounded digest-bound guest-artifact starter-plugin lane now exists”
- “current broader plugin truth is still host-native starter plugins plus one
  narrow digest-bound guest-artifact exception”
- “any future guest-artifact path must be digest-bound, trust-tiered, and
  publication-blocked”

Disallowed:

- “the repo supports generic Wasm plugins” in the present tense
- “user-provided plugins are supported” without class qualifiers
- “guest plugins are broadly part of the current starter-plugin platform”

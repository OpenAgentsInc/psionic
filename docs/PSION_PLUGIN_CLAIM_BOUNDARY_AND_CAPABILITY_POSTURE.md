# PSION Plugin Claim Boundary And Capability Posture

> Status: canonical `PSION_PLUGIN-2` claim-boundary and capability-posture
> contract for the `Psion x Tassadar` plugin-conditioned convergence tranche,
> written 2026-03-22 from `docs/PSION_PLUGIN_PROGRAM_MAP.md`,
> `docs/TRAIN_SYSTEM.md`, and
> `docs/audits/2026-03-22-tassadar-full-plugin-system-state-audit.md`.

This document freezes the combined claim boundary for the plugin-conditioned
learned lane and the first capability-posture language that later
`PSION_PLUGIN-*` issues must inherit.

It is not the later served capability matrix for a trained model.

That later publication is owned by:

- `PSION_PLUGIN-16` for the first host-native plugin-conditioned model
- `PSION_PLUGIN-28` for the mixed host-native plus guest-artifact model

The first `PSION_PLUGIN-16` publication now lives at:

- `docs/PSION_PLUGIN_HOST_NATIVE_CAPABILITY_MATRIX_V1.md`
- `fixtures/psion/plugins/capability/psion_plugin_host_native_capability_matrix_v1.json`
- `fixtures/psion/plugins/serve/psion_plugin_host_native_served_posture_v1.json`

The mixed `PSION_PLUGIN-28` publication now lives at:

- `docs/PSION_PLUGIN_MIXED_CAPABILITY_MATRIX_V2.md`
- `fixtures/psion/plugins/capability/psion_plugin_mixed_capability_matrix_v2.json`
- `fixtures/psion/plugins/serve/psion_plugin_mixed_served_posture_v2.json`

This document exists earlier because the convergence tranche already needs one
explicit answer to a simpler question:

> what may the repo honestly claim today, what is not yet proved, and what is
> still outside the bounded convergence lane entirely?

## Canonical Artifacts

- `docs/PSION_PLUGIN_CLAIM_BOUNDARY_AND_CAPABILITY_POSTURE.md` is the canonical
  human-readable contract.
- `docs/PSION_PLUGIN_PROGRAM_MAP.md` is the canonical dependency map and track split
  that this document sharpens into publication and route/refusal posture.
- `docs/PSION_PLUGIN_GUEST_ARTIFACT_DIRECTION.md` is the canonical bounded-lane
  direction record for the digest-bound guest-artifact starter-plugin proof and
  freezes that this class is now real only in one narrow operator-internal
  form, not as generic Wasm/plugin support.
- `docs/TRAIN_SYSTEM.md` is the canonical training-system authority doc that
  inherits this bounded posture for the convergence tranche.
- `docs/PSION_SERVED_EVIDENCE.md` and `docs/PSION_SERVED_OUTPUT_CLAIMS.md`
  remain the lower served-evidence and claim-posture contracts that later
  plugin-conditioned serving work must reuse instead of replacing.
- `docs/PSION_FAMILY_SERVE_VOCABULARY.md` now records how the
  plugin-conditioned lane sits inside the broader family serve vocabulary
  without flattening this lane into either the generic learned row or the
  executor-capable row.

## Combined Claim Boundary

The convergence tranche is for a **plugin-conditioned learned lane**.

It may claim:

- plugin recognition over the admitted plugin set
- plugin choice over the admitted plugin set
- typed plugin-argument planning
- plugin sequencing and stop choice
- request-for-structure behavior when required inputs are missing
- refusal when capability is outside the admitted set
- post-plugin interpretation of receipt-backed results

It may not claim:

- plugin publication closure
- arbitrary software capability closure
- public plugin universality
- arbitrary binary loading
- exact execution without an explicit runtime receipt
- broad GPU-scale pretraining closure
- trusted-cluster readiness by implication

The runtime still owns:

- plugin admission
- plugin loading
- capability mediation
- mount policy
- execution
- receipts
- replay posture
- runtime-side refusal

## Plugin-Class Posture

The current convergence tranche distinguishes four plugin classes:

1. `host_native_capability_free_local_deterministic`
2. `host_native_networked_read_only`
3. `host_native_secret_backed_or_stateful`
4. `guest_artifact_digest_bound`

The current posture for each class is:

| Plugin class | Current posture | Meaning |
| --- | --- | --- |
| `host_native_capability_free_local_deterministic` | `proved_supported_substrate` | real runtime, bridge, catalog, controller, weighted-controller, and user-authoring proof already exist |
| `host_native_networked_read_only` | `proved_supported_substrate` | one bounded manual user-authored proof now exists across runtime, bridge, catalog, controller, weighted-controller, and receipt truth, but the first trained publication still excludes this class |
| `host_native_secret_backed_or_stateful` | `out_of_scope_for_this_tranche` | class remains later bounded substrate work and must not be implied by the current convergence lane |
| `guest_artifact_digest_bound` | `proved_supported_substrate` | one bounded digest-bound user-provided Wasm proof now exists across runtime, bridge, catalog, deterministic controller, weighted-controller, and receipt truth, but generic Wasm/plugin support remains blocked and the first trained publication still excludes this class |

That table is the most important posture split in this tranche.

Later issues may move classes forward, but they must not flatten the rows into
one vague “plugin support” statement.

## Route And Refusal Variants

Plugin-conditioned route and refusal work must preserve the existing `Psion`
discipline while adding plugin-aware variants.

The learned lane now recognizes four plugin-conditioned route variants:

- `answer_in_language`
- `delegate_to_admitted_plugin`
- `request_missing_structure_for_plugin_use`
- `refuse_unsupported_plugin_or_capability`

The route side may only classify a response as `delegate_to_admitted_plugin`
when:

- the plugin is inside the declared admitted set
- the plugin class is within the current proved or explicitly enabled tranche
- execution later emits a real runtime receipt

The refusal side must keep at least these typed boundaries explicit:

- `plugin_class_not_yet_proved`
- `plugin_capability_outside_admitted_set`
- `missing_required_structured_input`
- `publication_or_arbitrary_loading_claim_blocked`
- `secret_backed_or_stateful_class_not_enabled`

That keeps plugin-conditioned route and refusal evidence compatible with the
later package work in `PSION_PLUGIN-7` through `PSION_PLUGIN-12` without inventing a
second route/refusal vocabulary.

## Served Posture For This Tranche

Any served description of the convergence tranche must stay strictly inside the
following posture:

- the lane is operator-internal
- plugin publication remains blocked
- the first trained host-native capability publication remains limited to the
  capability-free local deterministic class
- one manual `networked_read_only` substrate proof now exists, but it remains
  outside the first trained capability publication
- secret-backed and stateful classes remain unsupported in this tranche
- one bounded digest-bound guest-artifact substrate proof now exists, but it
  remains outside the first trained capability publication and does not imply
  generic Wasm/plugin support
- execution claims require explicit runtime receipts
- learned outputs may reason about plugin use, but may not imply hidden
  execution without receipts

This means allowed summary language today looks like:

- “plugin-conditioned learned lane over the bounded admitted host-native
  starter-plugin substrate”
- “operator-internal bounded plugin-use training tranche”
- “one manual networked_read_only substrate proof exists, but the first trained
  capability publication remains local-deterministic only”
- “one bounded digest-bound guest-artifact substrate proof exists, but generic
  Wasm/plugin support remains blocked and the mixed capability publication only
  admits one bounded digest-bound guest row”
- “receipt-backed plugin execution remains runtime-owned”

Disallowed summary language today includes:

- “plugin-supporting model” without class qualifiers
- “user-plugin-capable” without class qualifiers
- “generic Wasm plugin support” in the present tense
- “general plugin platform” or “plugin marketplace”
- “software execution in weights”

## Why This Boundary Matters

The convergence lane is only honest if it keeps three things separate:

- learned plugin reasoning
- runtime-owned plugin execution
- product- and publication-level plugin claims

That separation is what lets the repo improve learned plugin use without
quietly widening the platform story at the same time.

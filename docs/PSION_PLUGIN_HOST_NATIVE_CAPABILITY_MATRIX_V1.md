# PSION Plugin Host-Native Capability Matrix V1

> Status: canonical `PSION_PLUGIN-16` publication record for the first
> host-native plugin-conditioned capability matrix and served posture, written
> 2026-03-22 after landing the machine-readable v1 artifacts in `psionic-train`.

This document freezes the first explicit capability publication for the learned
host-native plugin-conditioned lane.

It is intentionally narrower than generic "plugin support."

The v1 publication is only for the first trained lane over the currently
trained subset of the proved starter-plugin substrate:

- host-native
- capability-free
- local deterministic

It does not silently widen `networked_read_only`, secret-backed, stateful,
guest-artifact, publication, or arbitrary software claims.

## Canonical Artifacts

- `docs/PSION_PLUGIN_HOST_NATIVE_CAPABILITY_MATRIX_V1.md` is the canonical
  human-readable publication record.
- `crates/psionic-train/src/psion_plugin_host_native_capability_matrix.rs`
  owns the typed machine-readable capability matrix and served-posture
  builders.
- `crates/psionic-train/examples/psion_plugin_host_native_capability_matrix.rs`
  writes the committed publication fixtures.
- `fixtures/psion/plugins/capability/psion_plugin_host_native_capability_matrix_v1.json`
  is the canonical machine-readable capability matrix.
- `fixtures/psion/plugins/serve/psion_plugin_host_native_served_posture_v1.json`
  is the canonical machine-readable served-posture artifact.

Stable schema versions:

- `psionic.psion.plugin_host_native_capability_matrix.v1`
- `psionic.psion.plugin_host_native_served_posture.v1`

## What V1 Actually Publishes

The v1 matrix publishes only four supported learned regions:

1. discovery and selection among the admitted local-deterministic starter
   plugins
2. typed argument construction and request-for-structure behavior for the
   admitted local-deterministic starter plugins
3. unsupported-capability refusal and missing-structure request behavior inside
   the proved host-native boundary
4. receipt-backed result interpretation over admitted local-deterministic
   plugin outputs or typed refusals

The admitted plugin ids in this first publication are exactly:

- `plugin.feed.rss_atom_parse`
- `plugin.html.extract_readable`
- `plugin.text.url_extract`

`plugin.http.fetch_text` is intentionally absent from the supported v1 surface
even though the repo now has one bounded manual `networked_read_only`
substrate proof for it.

## Explicit Non-Supported Rows

The matrix also publishes the rows it is not allowed to flatten away:

- `host_native_networked_read_only` is `substrate_proved_but_outside_supported_v1`
- local-deterministic `sequencing_multi_call` remains outside supported v1
  because the current bounded evaluation receipt has zero eligible in-boundary
  items
- `host_native_secret_backed_or_stateful` remains `unsupported`
- `guest_artifact_digest_bound` remains `unsupported`
- plugin publication or marketplace claims remain `blocked`
- public plugin universality remains `blocked`
- arbitrary software-capability claims remain `blocked`

That split is the point of the publication. The matrix is only honest if it
keeps supported, substrate-proved-but-outside-v1, unsupported, and blocked
rows explicit.

## Served Posture

The paired served-posture artifact freezes the statement boundary for outputs
that cite this matrix.

It keeps the lane:

- operator-internal only
- explicit about all four learned route labels:
  `answer_in_language`, `delegate_to_admitted_plugin`,
  `request_missing_structure_for_plugin_use`, and
  `refuse_unsupported_plugin_or_capability`
- allowed to surface `learned_judgment`,
  `benchmark_backed_capability_claim`, and `executor_backed_result`
- blocked from implying source grounding, verification, plugin publication,
  public plugin universality, arbitrary software capability, or hidden
  execution without runtime receipts

## Learned Versus Executor-Backed Statements

The served posture inherits `docs/PSION_SERVED_EVIDENCE.md` and
`docs/PSION_SERVED_OUTPUT_CLAIMS.md` directly.

For this lane, the practical rule is:

- learned plugin-use behavior may be served as learned judgment
- benchmark-backed capability claims may cite only supported rows from the v1
  matrix
- executor-backed results require explicit runtime receipt references
- the trained lane may not imply hidden execution when those receipts are absent

That is the exact statement boundary `PSION_PLUGIN-16` exists to freeze.

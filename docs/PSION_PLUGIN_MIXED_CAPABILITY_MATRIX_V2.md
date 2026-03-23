# PSION Plugin Mixed Capability Matrix V2

> Status: canonical `PSION_PLUGIN-28` publication record for the mixed
> host-native plus guest-artifact plugin-conditioned capability matrix and
> served posture, written 2026-03-22 after landing the machine-readable v2
> artifacts in `psionic-train`.

This document freezes the second explicit capability publication for the
learned plugin-conditioned lane.

It widens the first host-native-only publication in exactly two bounded ways:

- it now includes the admitted `host_native_networked_read_only` plugin class
- it now includes one bounded digest-bound guest-artifact admitted-use region

It still does not silently widen secret-backed, stateful, publication,
marketplace, public-universality, arbitrary-binary, or arbitrary-software
claims.

## Canonical Artifacts

- `docs/PSION_PLUGIN_MIXED_CAPABILITY_MATRIX_V2.md` is the canonical
  human-readable publication record.
- `crates/psionic-train/src/psion_plugin_mixed_capability_matrix.rs` owns the
  typed machine-readable capability matrix and served-posture builders.
- `crates/psionic-train/examples/psion_plugin_mixed_capability_matrix.rs`
  writes the committed publication fixtures.
- `fixtures/psion/plugins/capability/psion_plugin_mixed_capability_matrix_v2.json`
  is the canonical machine-readable capability matrix.
- `fixtures/psion/plugins/serve/psion_plugin_mixed_served_posture_v2.json`
  is the canonical machine-readable served-posture artifact.

Stable schema versions:

- `psionic.psion.plugin_mixed_capability_matrix.v2`
- `psionic.psion.plugin_mixed_served_posture.v2`

## What V2 Actually Publishes

The v2 matrix publishes these supported learned regions and class boundaries:

1. discovery and selection over the admitted host-native plugin set
2. typed argument construction and request-for-structure behavior over the
   admitted host-native plugin set
3. bounded multi-call sequencing over the admitted host-native plugin set
4. refusal and request-for-structure behavior over the admitted host-native
   plugin set
5. receipt-backed result interpretation over the admitted host-native plugin
   set
6. the `host_native_capability_free_local_deterministic` class boundary
7. the `host_native_networked_read_only` class boundary
8. the `guest_artifact_digest_bound` class boundary
9. one bounded `guest_artifact_digest_bound.admitted_use` region

The admitted host-native plugin ids in this publication are exactly:

- `plugin.feed.rss_atom_parse`
- `plugin.html.extract_readable`
- `plugin.http.fetch_text`
- `plugin.text.stats`
- `plugin.text.url_extract`

The admitted guest-artifact plugin id in this publication is exactly:

- `plugin.example.echo_guest`

This v2 publication is still operator-internal. It is not a generic "plugins in
weights" statement.

## Evidence Binding

The host-native rows in v2 are bound directly to the mixed-lane comparison
receipt from:

- `fixtures/psion/plugins/training/psion_plugin_mixed_reference_lane_v1/psion_plugin_mixed_reference_run_bundle.json`

That receipt preserves the mixed-versus-host-native comparison rows for:

- `discovery_selection`
- `argument_construction`
- `sequencing_multi_call`
- `refusal_request_structure`
- `result_interpretation`

The guest-artifact rows in v2 are bound directly to the dedicated guest-plugin
benchmark receipt from:

- `fixtures/psion/benchmarks/psion_plugin_guest_plugin_benchmark_v1/psion_plugin_guest_plugin_benchmark_bundle.json`

That benchmark is what lets the matrix distinguish:

- supported bounded guest admitted use
- blocked generic guest loading or unadmitted digest claims
- blocked publication or marketplace claims
- blocked public-served plugin universality claims
- blocked arbitrary-binary or arbitrary-software claims

## Explicit Blocked And Unsupported Rows

The v2 matrix also publishes the rows it is not allowed to flatten away:

- `host_native_secret_backed_or_stateful` remains `unsupported`
- `guest_artifact_generic_loading_or_unadmitted_digest` is `blocked`
- `plugin_publication_or_marketplace` is `blocked`
- `public_plugin_universality` is `blocked`
- `arbitrary_software_capability` is `blocked`

That split matters because the mixed lane is still only honest if it preserves:

- execution-backed capability evidence
- learned judgment over bounded admitted classes
- benchmark-backed refusal boundaries for everything it still does not claim

## Learned Judgment Versus Executor-Backed Result

The paired served-posture artifact keeps the statement boundary explicit.

The lane may serve:

- `learned_judgment`
- `benchmark_backed_capability_claim`
- `executor_backed_result`

The lane remains blocked from implying:

- source grounding
- verification
- plugin publication
- public plugin universality
- arbitrary software capability
- hidden execution without runtime receipts

The practical rule in v2 is:

- learned plugin-use behavior may be served as learned judgment
- benchmark-backed capability claims may cite supported host-native mixed rows
  and the one bounded guest admitted-use row
- executor-backed results still require explicit runtime receipt references
- guest capability rows do not imply hidden guest execution when those receipts
  are absent

That is the exact statement boundary `PSION_PLUGIN-28` exists to freeze.

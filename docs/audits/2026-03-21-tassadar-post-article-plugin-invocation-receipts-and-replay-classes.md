# Tassadar Post-Article Plugin Invocation Receipts And Replay Classes

`TAS-201` closes the next bounded plugin-runtime tranche above the canonical
post-`TAS-186` machine by freezing canonical invocation-receipt identity and
replay posture on top of the already-closed host-owned runtime API.

The committed runtime bundle,
`fixtures/tassadar/runs/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_v1/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_bundle.json`,
now states machine-readably that the invocation receipt contract must:

- keep explicit receipt, invocation, install, plugin, version, artifact,
  export, packet, mount-envelope, capability-envelope, and backend identity
- keep optional output digest, optional failure class, and optional challenge
  receipt fields typed instead of implied
- require a resource summary with logical start, logical duration, timeout
  ceiling, memory ceiling, queue wait, and bounded usage signals
- freeze four replay classes:
  `deterministic_replayable`, `replayable_with_snapshots`,
  `operator_replay_only`, and
  `non_replayable_refused_for_publication`
- freeze twelve typed refusal and failure classes, including policy, schema,
  capability, timeout, memory-limit, crash, artifact-mismatch, replay-posture,
  trust-posture, and publication-posture lanes
- bind every receipt to explicit route evidence
- bind accepted and snapshot-replayable failure lanes to explicit challenge
  receipts

The eval-owned report,
`fixtures/tassadar/reports/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_report.json`,
binds that bundle to the same canonical bridge machine identity, canonical
route, and computational-model statement as the earlier plugin tranche. It
also cites the closed runtime-API contract plus the earlier effectful replay,
installed-module evidence, and module-promotion reports so replay posture and
receipt lineage are not claimed by implication from the runtime bundle alone.

The operator summary, provider projection, and checker now live at:

- `fixtures/tassadar/reports/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_summary.json`
- `crates/psionic-provider/src/tassadar_post_article_plugin_invocation_receipts_and_replay_classes.rs`
- `scripts/check-tassadar-post-article-plugin-invocation-receipts-and-replay-classes.sh`

This issue also refreshes the earlier runtime-API artifacts so their defer
pointer is now empty:

- `fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary.json`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_publication.json`

Claim boundary:

- this tranche allows the rebased machine claim to carry one canonical
  invocation-receipt and replay-class contract
- it does not by itself allow weighted plugin control
- it does not allow plugin publication
- it does not widen served/public universality
- it does not imply arbitrary software capability

The deferred frontier now moves to `TAS-202`, where route and mount policy
must be compiled into explicit admissibility and world-mount envelopes.

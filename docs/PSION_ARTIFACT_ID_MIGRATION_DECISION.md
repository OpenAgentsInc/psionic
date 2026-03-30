# Psion Artifact-Id Migration Decision

> Status: canonical `PSION-0903` / `#787` artifact-id migration decision memo,
> written 2026-03-30 after the generic serve packet, family serve vocabulary,
> and bounded executor `trained-v1` promotion all landed.

## Decision

`do_not_migrate`

The repo should not rename the live executor-capable artifact family just to
make the umbrella `Psion` family look cleaner.

## Why

The current `tassadar-*` identifiers are not stale decorations. They are the
actual retained identity surface for the bounded executor-capable lane.

The executor lane now has:

- one retained `trained-v1` promotion packet
- one retained replacement report
- one retained bounded closeout status
- one admitted fast-route identity
- one live consumer seam in `openagents` built around stable promoted and
  rollback artifact ids

Renaming that identity family now would introduce churn across real artifacts,
real route ids, real reports, and real consumer manifests without buying new
capability or resolving a live ambiguity that the current docs cannot already
handle.

## Evaluated Live-Route Impact

### Current executor model identity

The current bounded executor lane already publishes the promoted candidate under
the retained model id:

- `tassadar-article-transformer-trace-bound-trained-v1`

That id is carried in:

- `docs/PSION_EXECUTOR_TRAINED_V1_PROMOTION.md`
- `docs/PSION_EXECUTOR_TRAINED_V1_REPLACEMENT_REPORT.md`
- `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v1_descriptor.json`
- `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v1_artifact_manifest.json`
- `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v1_lineage_contract.json`

### Current executor route identity

The current bounded fast-route target already publishes the retained route id:

- `tassadar.article_route.direct_hull_cache_runtime.v1`

That id is carried across retained bounded closeout and conformance surfaces,
including:

- `fixtures/tassadar/reports/tassadar_article_route_minimality_audit_summary.json`
- `fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json`
- `fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_summary.json`

### Current consumer seam impact

The current `openagents` compiled-agent consumer seam already treats artifact
ids as stable runtime lineage, shadow, and rollback inputs:

- `crates/openagents-compiled-agent/src/manifest.rs`
- `crates/openagents-compiled-agent/src/hub.rs`
- `docs/compiled-agent-vertical-slice.md`

Those surfaces already keep:

- promoted artifact ids
- candidate artifact ids
- rollback artifact ids
- shadow manifest ids

explicit in the runtime contract.

Forcing an executor artifact-id rename now would therefore require a dual-name
or migration pass across:

- promotion packets
- replacement reports
- route identity summaries
- artifact manifests
- consumer manifests
- shadow and rollback seams

That is real compatibility work, not harmless cleanup.

## What Makes Migration Not Worth It Today

Migration is not worth it today because:

1. the current docs already solve the naming ambiguity in prose:
   `Psion` is the umbrella family and `Tassadar` is the executor-capable
   bounded profile
2. no retained capability claim is blocked on the current ids
3. no live route malfunction is caused by the current ids
4. the consumer seam still benefits from stable artifact and rollback identity
5. the migration would touch many retained artifacts while producing no new
   execution, serving, or evaluation truth

## What The Repo Should Do Instead

- keep the current executor artifact ids valid
- keep the current executor route ids valid
- continue describing them as executor-capable `Psion` artifacts in prose
- keep the generic family and executor-capable profile split explicit in docs
- revisit migration only if a real compatibility or operator-cost problem
  appears

## Conditions That Could Reopen This Decision

Revisit artifact-id migration only if at least one of the following becomes
true:

- the current umbrella/family split still causes real operator confusion after
  the lane docs and vocabulary packet are in place
- a public consumer API needs one normalized family-level artifact namespace
- the repo is ready to ship a dual-acceptance migration window with explicit
  rollback and lineage preservation
- the current `tassadar-*` ids block multi-profile publication or route
  compatibility in a way docs alone cannot solve

Until then, the correct program answer is to leave the ids alone.

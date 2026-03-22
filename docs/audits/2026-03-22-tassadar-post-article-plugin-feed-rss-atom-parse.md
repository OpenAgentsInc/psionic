# 2026-03-22 Tassadar Post-Article Plugin Feed RSS Atom Parse

`TAS-220` closes the first deterministic structured-ingest starter plugin above
the fetch-text lane.

## Landed Surfaces

- `psionic-runtime` now owns a real `plugin.feed.rss_atom_parse` runtime path in
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`
- the same runtime writes committed fixture truth to
  `fixtures/tassadar/runs/tassadar_post_article_plugin_feed_rss_atom_parse_v1/tassadar_post_article_plugin_feed_rss_atom_parse_bundle.json`
- `scripts/check-tassadar-post-article-plugin-feed-rss-atom-parse.sh` now acts
  as the dedicated checker over the runtime bundle and targeted tests
- `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md` now tracks the fetch-to-feed-parse
  composition lane as first-class starter-runtime truth

## What Is Green

- one canonical plugin id: `plugin.feed.rss_atom_parse`
- one stable tool projection: `plugin_feed_rss_atom_parse`
- one explicit no-capability mount envelope:
  `mount.plugin.feed.rss_atom_parse.no_capabilities.v1`
- deterministic replay posture over already-fetched content
- one bounded explicit format window for RSS 2.0 and Atom 1.0
- typed refusals for schema invalid, input too large, and unsupported feed
  format
- one green composition case proving `plugin.http.fetch_text ->
  plugin.feed.rss_atom_parse` without hidden host schema repair or host-side
  feed parsing

## What Is Still Refused

- arbitrary XML support
- OPML support
- general document-parsing closure
- broader multi-plugin controller closure

## Next Frontier

`TAS-224` is now also closed by the router-owned served pilot in
`docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`.

The next open orchestration frontier above the deterministic starter controller
is `TAS-221`: publish the first real-run multi-plugin orchestration wave.

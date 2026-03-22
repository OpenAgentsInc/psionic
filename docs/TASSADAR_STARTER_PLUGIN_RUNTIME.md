# Tassadar Starter Plugin Runtime

This document tracks the first real runtime-owned starter plugin implementations
above the post-article plugin manifest, packet ABI, receipt, and world-mount
contracts.

The boundary is narrow on purpose:

- starter plugins are operator-curated and operator-internal
- packet schemas, refusal classes, replay posture, and mount envelopes stay
  explicit
- local deterministic plugins do not imply URL validity, browser execution,
  JavaScript, DNS, cookies, auth sessions, or general web-agent closure
- read-only network plugins do not imply unrestricted network access or replay
  stability unless the mounted backend is snapshot-backed

## Implemented

### `plugin.text.url_extract`

- runtime bundle:
  `fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1/tassadar_post_article_plugin_text_url_extract_bundle.json`
- example writer:
  `cargo run -p psionic-runtime --example tassadar_post_article_plugin_text_url_extract_bundle`
- checker:
  `scripts/check-tassadar-post-article-plugin-text-url-extract.sh`

`plugin.text.url_extract` is now a real capability-free runtime entry. It
accepts one JSON packet shaped like `{ "text": string }`, applies the bounded
legacy match rule `https?://[^\\s]+`, preserves left-to-right order, preserves
duplicates, and returns `{ "urls": [...] }`.

Typed refusal surface:

- `plugin.refusal.schema_invalid.v1`
- `plugin.refusal.packet_too_large.v1`
- `plugin.refusal.unsupported_codec.v1`
- `plugin.refusal.runtime_resource_limit.v1`

Tool projection is explicit and stable:

- tool name: `plugin_text_url_extract`
- argument schema remains JSON-schema-shaped and packet-derived
- replay class remains `deterministic_replayable`
- mount envelope remains
  `mount.plugin.text.url_extract.no_capabilities.v1`

Negative claims stay explicit:

- no URL validation truth
- no DNS truth
- no redirect truth
- no network reachability truth

## Planned

- `plugin.http.fetch_text`
- `plugin.html.extract_readable`
- `plugin.feed.rss_atom_parse`
- shared plugin-to-tool projection across deterministic, router-owned, and Apple
  FM controller lanes
- deterministic, served, and Apple FM multi-plugin pilot traces above the same
  runtime-owned starter plugin substrate

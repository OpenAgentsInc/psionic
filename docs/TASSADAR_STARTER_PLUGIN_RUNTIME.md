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

The runtime now keeps starter-plugin registration in one runtime-owned source
of truth. Plugin identity, tool name, packet schemas, refusal schemas, replay
class, capability posture, mount envelope identity, runtime bundle identity,
and per-surface publication flags derive from that one registry so the runtime,
shared bridge, and starter catalog do not drift independently.

The human-readable contributor contract for adding a new capability-free
starter plugin now lives in `docs/TASSADAR_STARTER_PLUGIN_AUTHORING.md`.

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

### `plugin.text.stats`

- runtime bundle:
  `fixtures/tassadar/runs/tassadar_post_article_plugin_text_stats_v1/tassadar_post_article_plugin_text_stats_bundle.json`
- example writer:
  `cargo run -p psionic-runtime --example tassadar_post_article_plugin_text_stats_bundle`
- user demo:
  `cargo run -p psionic-runtime --example tassadar_post_article_plugin_text_stats_demo`
- checker:
  `scripts/check-tassadar-post-article-plugin-text-stats.sh`

`plugin.text.stats` is now a real capability-free runtime entry for a simple
user-authored host-native plugin. It accepts one JSON packet shaped like
`{ "text": string }` and returns bounded packet-local count truth:

- `byte_count`
- `unicode_scalar_count`
- `line_count`
- `non_empty_line_count`
- `word_count`

The counting rules stay explicit and deterministic: Rust byte length,
`chars()`, `lines()`, and `split_whitespace()` semantics only.

Typed refusal surface:

- `plugin.refusal.schema_invalid.v1`
- `plugin.refusal.packet_too_large.v1`
- `plugin.refusal.unsupported_codec.v1`

Tool projection is explicit and stable:

- tool name: `plugin_text_stats`
- argument schema remains JSON-schema-shaped and packet-derived
- replay class remains `deterministic_replayable`
- mount envelope remains `mount.plugin.text.stats.no_capabilities.v1`

Negative claims stay explicit:

- no tokenizer truth
- no language detection truth
- no sentence boundary truth
- no semantic structure truth

### `plugin.http.fetch_text`

- runtime bundle:
  `fixtures/tassadar/runs/tassadar_post_article_plugin_http_fetch_text_v1/tassadar_post_article_plugin_http_fetch_text_bundle.json`
- example writer:
  `cargo run -p psionic-runtime --example tassadar_post_article_plugin_http_fetch_text_bundle`
- checker:
  `scripts/check-tassadar-post-article-plugin-http-fetch-text.sh`

`plugin.http.fetch_text` is now a real read-only network starter plugin. It
accepts one JSON packet shaped like `{ "url": string }`, enforces URL policy
from the mounted envelope instead of the guest packet, and returns structured
fetch truth:

- `final_url`
- `status_code`
- `content_type`
- `charset`
- `body_text`
- `truncated`

The current committed evidence is snapshot-backed so replay posture is honest
and deterministic. The runtime also carries a live host-HTTP path that is
explicitly classified as `operator_replay_only`.

Typed refusal surface:

- `plugin.refusal.schema_invalid.v1`
- `plugin.refusal.unsupported_codec.v1`
- `plugin.refusal.network_denied.v1`
- `plugin.refusal.url_not_permitted.v1`
- `plugin.refusal.timeout.v1`
- `plugin.refusal.response_too_large.v1`
- `plugin.refusal.content_type_unsupported.v1`
- `plugin.refusal.decode_failed.v1`
- `plugin.refusal.upstream_failure.v1`

Tool projection is explicit and stable:

- tool name: `plugin_http_fetch_text`
- argument schema remains JSON-schema-shaped and packet-derived
- sample mount envelope remains
  `mount.plugin.http.fetch_text.read_only_http_allowlist.v1`
- replay classes remain explicit:
  `replayable_with_snapshots` and `operator_replay_only`

Negative claims stay explicit:

- no browser execution
- no JavaScript execution
- no cookie or auth-session support
- no arbitrary header surface
- no unrestricted web access

### `plugin.html.extract_readable`

- runtime bundle:
  `fixtures/tassadar/runs/tassadar_post_article_plugin_html_extract_readable_v1/tassadar_post_article_plugin_html_extract_readable_bundle.json`
- example writer:
  `cargo run -p psionic-runtime --example tassadar_post_article_plugin_html_extract_readable_bundle`
- checker:
  `scripts/check-tassadar-post-article-plugin-html-extract-readable.sh`

`plugin.html.extract_readable` is now a real local deterministic starter
plugin. It accepts already-fetched HTML-shaped input:

- `source_url`
- `content_type`
- `body_text`

and returns bounded readability-oriented output:

- `title`
- `canonical_url`
- `site_name`
- `excerpt`
- `readable_text`
- `harvested_links`
- `content_language`

The committed bundle also freezes one green composition case where
`plugin.http.fetch_text` output feeds this plugin without hidden host schema
repair.

Typed refusal surface:

- `plugin.refusal.schema_invalid.v1`
- `plugin.refusal.unsupported_codec.v1`
- `plugin.refusal.input_too_large.v1`
- `plugin.refusal.content_type_unsupported.v1`

Tool projection is explicit and stable:

- tool name: `plugin_html_extract_readable`
- argument schema remains JSON-schema-shaped and packet-derived
- replay class remains `deterministic_replayable`
- mount envelope remains
  `mount.plugin.html.extract_readable.no_capabilities.v1`

Negative claims stay explicit:

- no browser rendering
- no JavaScript evaluation
- no CSS layout truth
- no full DOM-semantics claim

### `plugin.feed.rss_atom_parse`

- runtime bundle:
  `fixtures/tassadar/runs/tassadar_post_article_plugin_feed_rss_atom_parse_v1/tassadar_post_article_plugin_feed_rss_atom_parse_bundle.json`
- example writer:
  `cargo run -p psionic-runtime --example tassadar_post_article_plugin_feed_rss_atom_parse_bundle`
- checker:
  `scripts/check-tassadar-post-article-plugin-feed-rss-atom-parse.sh`

`plugin.feed.rss_atom_parse` is now a real local deterministic structured-ingest
starter plugin. It accepts already-fetched feed-shaped input:

- `source_url`
- `content_type`
- `feed_text`

and returns bounded feed metadata plus normalized entry rows:

- `feed_title`
- `feed_homepage_url`
- `feed_description`
- `entries[]` with title, link, published time, summary, and content excerpt

The committed bundle also freezes one green composition case where
`plugin.http.fetch_text` output feeds this parser without hidden host schema
repair or host-side feed parsing.

Typed refusal surface:

- `plugin.refusal.schema_invalid.v1`
- `plugin.refusal.input_too_large.v1`
- `plugin.refusal.unsupported_feed_format.v1`

Tool projection is explicit and stable:

- tool name: `plugin_feed_rss_atom_parse`
- argument schema remains JSON-schema-shaped and packet-derived
- replay class remains `deterministic_replayable`
- mount envelope remains `mount.plugin.feed.rss_atom_parse.no_capabilities.v1`

Negative claims stay explicit:

- no arbitrary XML support
- no OPML support
- no general document-parsing closure

## Adjacent Surface

The shared controller-facing bridge above these runtime entries now lives in
`docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md`. That bridge keeps tool
definitions, structured result envelopes, typed refusals, and plugin receipt
identity stable across deterministic, router-owned, and Apple FM controller
lanes without widening this runtime tranche into weighted control claims.

## Planned

- deterministic, served, and Apple FM multi-plugin pilot traces above the same
  runtime-owned starter plugin substrate

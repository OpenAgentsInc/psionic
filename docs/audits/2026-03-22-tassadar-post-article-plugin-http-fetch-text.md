# 2026-03-22 Tassadar Post-Article Plugin HTTP Fetch Text

`TAS-218` closes the first read-only network starter plugin above the earlier
catalog shell and the new starter-runtime substrate.

## Landed Surfaces

- `psionic-runtime` now owns a real `plugin.http.fetch_text` runtime path in
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`
- the same runtime writes committed fixture truth to
  `fixtures/tassadar/runs/tassadar_post_article_plugin_http_fetch_text_v1/tassadar_post_article_plugin_http_fetch_text_bundle.json`
- `scripts/check-tassadar-post-article-plugin-http-fetch-text.sh` now acts as
  the dedicated checker over the runtime bundle and targeted tests
- `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md` now tracks both the local
  deterministic and read-only network starter-runtime surfaces

## What Is Green

- one canonical plugin id: `plugin.http.fetch_text`
- one stable tool projection: `plugin_http_fetch_text`
- one explicit sample host-mediated mount envelope:
  `mount.plugin.http.fetch_text.read_only_http_allowlist.v1`
- replay posture is explicit:
  `replayable_with_snapshots` for committed snapshot evidence and
  `operator_replay_only` for the live host-HTTP backend
- typed refusals for schema invalid, unsupported codec, network denied, URL not
  permitted, timeout, response too large, content type unsupported, decode
  failed, and upstream failure

## What Is Still Refused

- browser execution
- JavaScript execution
- cookie or auth-session semantics
- arbitrary header control
- unrestricted web access
- broader multi-plugin orchestration closure

## Next Frontier

`TAS-219` is now also closed by the dedicated
`plugin.html.extract_readable` runtime bundle in
`fixtures/tassadar/runs/tassadar_post_article_plugin_html_extract_readable_v1/tassadar_post_article_plugin_html_extract_readable_bundle.json`.

The next starter-plugin runtime frontier is `TAS-220`:
`plugin.feed.rss_atom_parse`.

# 2026-03-22 Tassadar Post-Article Plugin Text URL Extract

`TAS-217` closes the first dedicated starter-plugin runtime implementation above
the earlier catalog shell.

## Landed Surfaces

- `psionic-runtime` now owns a real `plugin.text.url_extract` packet runtime in
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`
- the same runtime writes committed fixture truth to
  `fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1/tassadar_post_article_plugin_text_url_extract_bundle.json`
- `scripts/check-tassadar-post-article-plugin-text-url-extract.sh` now acts as
  the dedicated checker over the runtime bundle and targeted tests
- `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md` now tracks the growing starter
  runtime surface

## What Is Green

- one canonical plugin id: `plugin.text.url_extract`
- one stable tool projection: `plugin_text_url_extract`
- one capability-free mount envelope:
  `mount.plugin.text.url_extract.no_capabilities.v1`
- one deterministic replay class: `deterministic_replayable`
- typed refusals for schema invalid, packet too large, unsupported codec, and
  runtime resource limit
- explicit negative claims for URL validation, DNS, redirects, and network
  reachability

## What Is Still Refused

- browser rendering
- JavaScript execution
- DNS or network truth
- general URL understanding
- broader multi-plugin orchestration closure

## Next Frontier

The next starter-plugin runtime frontier is `TAS-218`:
`plugin.http.fetch_text`.

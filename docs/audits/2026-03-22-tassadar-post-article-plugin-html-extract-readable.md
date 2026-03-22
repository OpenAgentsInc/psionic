# 2026-03-22 Tassadar Post-Article Plugin HTML Extract Readable

`TAS-219` closes the first local deterministic readability-transform starter
plugin above the fetch-text lane.

## Landed Surfaces

- `psionic-runtime` now owns a real `plugin.html.extract_readable` runtime path
  in `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`
- the same runtime writes committed fixture truth to
  `fixtures/tassadar/runs/tassadar_post_article_plugin_html_extract_readable_v1/tassadar_post_article_plugin_html_extract_readable_bundle.json`
- `scripts/check-tassadar-post-article-plugin-html-extract-readable.sh` now
  acts as the dedicated checker over the runtime bundle and targeted tests
- `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md` now tracks the fetch-to-extract
  composition lane as first-class starter-runtime truth

## What Is Green

- one canonical plugin id: `plugin.html.extract_readable`
- one stable tool projection: `plugin_html_extract_readable`
- one explicit no-capability mount envelope:
  `mount.plugin.html.extract_readable.no_capabilities.v1`
- deterministic replay posture over already-fetched content
- typed refusals for schema invalid, unsupported codec, input too large, and
  unsupported content type
- one green composition case proving `plugin.http.fetch_text ->
  plugin.html.extract_readable` without hidden host schema repair

## What Is Still Refused

- browser rendering
- JavaScript execution
- CSS layout truth
- full DOM semantics
- broader multi-plugin orchestration closure

## Next Frontier

`TAS-224` is now also closed by the router-owned served pilot in
`docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`.

The next open orchestration frontier above the deterministic starter controller
is the later `TAS-204` weighted-controller lane above the published
orchestration wave.

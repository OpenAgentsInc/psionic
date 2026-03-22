# 2026-03-21 Tassadar Post-Article Starter Plugin Catalog

## Intent

`TAS-216` closes the first starter-catalog publication layer above the bounded
weighted-plugin platform and the canonical machine closure bundle.

The point of this tranche is not to widen publication posture. It is to make
the plugin platform concretely useful by publishing one small, explicit,
operator-curated catalog with bounded composition flows and explicit machine
legibility.

## Landed Surfaces

- `psionic-runtime` now owns the starter-catalog runtime bundle at
  `fixtures/tassadar/runs/tassadar_post_article_starter_plugin_catalog_v1/tassadar_post_article_starter_plugin_catalog_bundle.json`
  plus one descriptor, one fixture bundle, and one sample mount envelope for:
  - `plugin.text.url_extract@v1`
  - `plugin.http.fetch_text@v1`
  - `plugin.html.extract_readable@v1`
  - `plugin.feed.rss_atom_parse@v1`
- `psionic-catalog` now binds that runtime-owned artifact to the bounded
  platform, authority gate, world-mount precedent, and canonical closure bundle
  in
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_report.json`.
- `psionic-eval` now projects the same starter-catalog posture into
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_eval_report.json`.
- `psionic-research` now commits the disclosure-safe summary at
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_summary.json`.
- `psionic-provider` now projects the same bounded publication posture into
  `crates/psionic-provider/src/tassadar_post_article_starter_plugin_catalog.rs`.
- `scripts/check-tassadar-post-article-starter-plugin-catalog.sh` now acts as
  the dedicated checker over the runtime bundle, catalog report, eval report,
  research summary, and provider projection.

## What Is Explicitly Green

- four bounded starter plugins are named machine-readably
- three starter plugins are local deterministic transforms
- one starter plugin is read-only network access through an explicit host-owned
  mount envelope
- every starter plugin carries descriptor, fixture, and sample mount references
- two bounded starter flows are machine-readable:
  - discover URL strings, then fetch and extract readable page content
  - fetch a feed document, then parse RSS or Atom metadata and entries
- the starter catalog remains bound to the canonical machine closure bundle by
  digest

## What Is Still Refused

- public plugin publication
- public plugin marketplace or discovery posture
- served or public universality widening
- arbitrary public Wasm execution
- arbitrary public tool use
- broader multi-plugin real-run closure

## Next Frontier

`TAS-217` is now closed by the dedicated
`plugin.text.url_extract` runtime implementation in
`crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`
and its committed bundle at
`fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1/tassadar_post_article_plugin_text_url_extract_bundle.json`.

`TAS-218` is now also closed by the dedicated `plugin.http.fetch_text` runtime
implementation in the same starter-runtime module and its committed bundle at
`fixtures/tassadar/runs/tassadar_post_article_plugin_http_fetch_text_v1/tassadar_post_article_plugin_http_fetch_text_bundle.json`.

`TAS-219` is now also closed by the dedicated
`plugin.html.extract_readable` runtime implementation in the same
starter-runtime module and its committed bundle at
`fixtures/tassadar/runs/tassadar_post_article_plugin_html_extract_readable_v1/tassadar_post_article_plugin_html_extract_readable_bundle.json`.

The next open starter-plugin frontier is `TAS-220`, which continues the
plugin-by-plugin runtime wave on top of this catalog shell with
`plugin.feed.rss_atom_parse`.

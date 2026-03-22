# Tassadar Apple FM Plugin Session

This document tracks the first local Apple FM starter-plugin controller lane
above the shared starter-plugin bridge.

The boundary is narrow on purpose:

- admissible starter plugins are projected into `AppleFmToolDefinition`s from
  the shared bridge instead of an Apple-only schema system
- tool callbacks stay session-aware and dispatch into the shared
  starter-plugin runtime
- transcript truth and plugin receipt truth stay explicit in the committed
  pilot bundle
- this lane is local Apple FM plugin orchestration, not served-model closure
  and not weighted controller closure

## Implemented

- Apple FM tool projection:
  `crates/psionic-apple-fm/src/tassadar_post_article_starter_plugin_tools.rs`
- local pilot bundle:
  `crates/psionic-apple-fm/src/client/tassadar_post_article_starter_plugin_session_pilot.rs`
- committed bundle:
  `fixtures/tassadar/runs/tassadar_post_article_apple_fm_plugin_session_pilot_v1/tassadar_post_article_apple_fm_plugin_session_pilot_bundle.json`
- example writer:
  `cargo run -p psionic-apple-fm --example tassadar_post_article_apple_fm_plugin_session_pilot_bundle`
- checker:
  `scripts/check-tassadar-post-article-apple-fm-plugin-pilot.sh`

`psionic-apple-fm` now owns one reusable Apple FM starter-plugin projection
surface that projects the four starter plugins into Apple FM tool definitions
and routes callback execution into the shared starter-plugin runtime bridge.

The committed local pilot bundle freezes two bounded cases:

- one success case with five sequential plugin steps and explicit transcript
  truth
- one typed-refusal case where `plugin_http_fetch_text` returns a refusal on a
  binary URL and the refusal remains receipt-bound and structured

## What Is Green

- one Apple FM tool-definition set covering all four shared starter plugins
- one session-aware local callback registry reused across multiple tool calls
- rendered tool outputs that keep the full projected plugin envelope visible,
  including plugin receipt identity and typed refusal shape
- one success pilot with `plugin_text_url_extract`,
  `plugin_http_fetch_text`, `plugin_html_extract_readable`,
  `plugin_http_fetch_text`, and `plugin_feed_rss_atom_parse`
- one refusal pilot with session-token binding preserved across tool calls and
  typed refusal preservation through the transcripted callback lane

## Adjacent Surface

The shared projection and receipt contract below this lane lives in
`docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md`.

The deterministic controller lane above the same bridge lives in
`docs/TASSADAR_STARTER_PLUGIN_WORKFLOW_CONTROLLER.md`.

The router-owned served controller lane above the same bridge lives in
`docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`.

## Planned

- cross-lane trace corpus and parity matrix now land separately in
  `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md`

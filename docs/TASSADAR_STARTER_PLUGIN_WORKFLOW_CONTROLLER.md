# Tassadar Starter Plugin Workflow Controller

This document tracks the first host-owned deterministic controller above the
shared starter-plugin bridge.

The boundary is narrow on purpose:

- the workflow graph is explicit and host-owned
- branch and stop decisions stay machine-readable and challengeable
- the controller reuses the shared starter-plugin bridge instead of bespoke
  per-plugin glue
- this controller is a bounded pilot lane, not open-ended planning and not
  weighted controller closure

## Implemented

- runtime bundle:
  `fixtures/tassadar/runs/tassadar_post_article_starter_plugin_workflow_controller_v1/tassadar_post_article_starter_plugin_workflow_controller_bundle.json`
- example writer:
  `cargo run -p psionic-runtime --example tassadar_post_article_starter_plugin_workflow_controller_bundle`
- checker:
  `scripts/check-tassadar-post-article-starter-plugin-workflow-controller.sh`

`psionic-runtime` now owns one deterministic workflow controller in
`crates/psionic-runtime/src/tassadar_post_article_starter_plugin_workflow_controller.rs`
for one bounded web-content intake graph:

- extract URLs from one directive with `plugin_text_url_extract`
- fetch each extracted URL with `plugin_http_fetch_text`
- branch on fetched content type into `plugin_html_extract_readable` or
  `plugin_feed_rss_atom_parse`
- stop explicitly on typed refusal or when the extracted URL set is exhausted

The committed bundle freezes two pilot cases:

- `web_content_intake_success`
- `web_content_intake_fetch_refusal`

Each case carries:

- host-owned decision rows
- per-step projected tool results
- per-step plugin receipts
- explicit refusal rows
- explicit stop condition

## What Is Green

- one reproducible multi-plugin success pilot with five sequential plugin steps
- one refusal pilot that stops on typed fetch refusal instead of hiding retry
  logic
- explicit html-versus-feed branch decisions
- explicit final stop conditions for success and refusal
- direct reuse of the shared starter-plugin bridge for every tool call

## Planned

- router-owned `/v1/responses` pilot now lands separately in
  `docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`
- Apple FM local plugin-session integration now lands separately in
  `docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md`
- cross-lane trace corpus and parity matrix now land separately in
  `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md`

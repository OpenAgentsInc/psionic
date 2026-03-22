# Tassadar Starter Plugin Tool Bridge

This document tracks the shared adapter layer that projects starter-plugin
runtime definitions into controller-facing tool definitions and binds
controller-visible tool results back to the underlying plugin invocation
receipts.

The boundary is narrow on purpose:

- one shared bridge sits above the starter-plugin runtime and below controller
  lanes
- the bridge does not invent a second schema vocabulary; it derives tool
  definitions from the runtime-owned plugin projections
- tool results keep structured payloads, typed refusals, and plugin receipt
  identity explicit instead of flattening them into free-form strings
- this bridge is reusable by deterministic workflows, router-owned
  `/v1/responses` loops, and Apple FM tool callbacks, but it is not itself a
  weighted controller

## Implemented

- runtime bundle:
  `fixtures/tassadar/runs/tassadar_post_article_starter_plugin_tool_bridge_v1/tassadar_post_article_starter_plugin_tool_bridge_bundle.json`
- example writer:
  `cargo run -p psionic-runtime --example tassadar_post_article_starter_plugin_tool_bridge_bundle`
- checker:
  `scripts/check-tassadar-post-article-starter-plugin-tool-bridge.sh`

`psionic-runtime` now owns one shared bridge in
`crates/psionic-runtime/src/tassadar_post_article_starter_plugin_tool_bridge.rs`
with three main surfaces:

- stable per-plugin tool-definition projection for deterministic workflows
- OpenAI-compatible router-facing function-tool projection
- Apple FM-facing tool-definition projection over the same argument schemas

The current bridge freezes four starter-plugin definitions:

- `plugin_text_url_extract`
- `plugin_http_fetch_text`
- `plugin_html_extract_readable`
- `plugin_feed_rss_atom_parse`

The bridge also freezes one receipt-bound projected result envelope carrying:

- tool name and plugin identity
- success or refusal status
- output or refusal schema id
- structured payload
- full plugin invocation receipt

The rendered tool output path is the JSON serialization of that envelope, which
keeps receipt references and typed refusals visible even in text-only callback
lanes.

## What Is Green

- one shared projection layer above the starter-plugin runtime
- stable argument-schema digests across deterministic, router-owned, and Apple
  FM surfaces for all current starter plugins
- shared execution entrypoint that dispatches projected tool calls into the real
  starter-plugin runtime
- projected tool results that preserve plugin receipt identity at the tool
  boundary
- typed refusals that remain structured instead of collapsing into free-form
  tool text

## Planned

- deterministic starter-workflow controller and pilot above this shared bridge
- router-owned `/v1/responses` plugin tool-loop integration above this shared
  bridge
- Apple FM session tool integration above this shared bridge
- trace-corpus and parity-matrix freezing above the deterministic, router, and
  Apple FM controller lanes

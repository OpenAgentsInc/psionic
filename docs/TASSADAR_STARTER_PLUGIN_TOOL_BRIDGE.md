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

The current bridge freezes six starter-plugin definitions:

- `plugin_text_url_extract`
- `plugin_text_stats`
- `plugin_http_fetch_text`
- `plugin_example_echo_guest`
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
- stable argument-schema digests across deterministic and router-owned surfaces
  for all current starter plugins, including the digest-bound guest-artifact
  row
- Apple FM reuses the same bridge contract for the currently admitted host-
  native subset without widening to generic guest-artifact callbacks
- shared execution entrypoint that dispatches projected tool calls into the real
  starter-plugin runtime
- projected tool results that preserve plugin receipt identity at the tool
  boundary
- typed refusals that remain structured instead of collapsing into free-form
  tool text

## Adjacent Surface

The first deterministic controller above this bridge now lives in
`docs/TASSADAR_STARTER_PLUGIN_WORKFLOW_CONTROLLER.md`. That controller keeps
its workflow graph, branch rows, refusal rows, and stop conditions host-owned
and explicit while reusing this bridge for every tool call.

The first router-owned served controller above the same bridge now lives in
`docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`. That pilot keeps projected tool
definitions, router-owned tool execution, structured envelopes, plugin
receipts, and response-state continuation explicit without claiming weighted
controller closure.

The first local Apple FM controller above the same bridge now lives in
`docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md`. That pilot keeps session-aware tool
callbacks, transcript truth, projected envelopes, and plugin receipts explicit
without claiming served-model closure or weighted controller closure.

## Planned

- cross-lane trace corpus and parity matrix now land separately in
  `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md`

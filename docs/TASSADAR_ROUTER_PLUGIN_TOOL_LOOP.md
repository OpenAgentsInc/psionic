# Tassadar Router Plugin Tool Loop

This document tracks the first router-owned starter-plugin tool loop on
`/v1/responses`.

The boundary is narrow on purpose:

- admissible starter plugins are projected into one bounded `/v1/responses`
  tool set
- tool execution stays inside the router-owned `ToolGateway` and dispatches
  into the shared starter-plugin runtime bridge
- each tool-loop step preserves the structured plugin envelope and the
  underlying plugin invocation receipt
- the pilot is model-assisted router-owned orchestration, not weighted-plugin
  closure

## Implemented

- serve bundle:
  `fixtures/tassadar/runs/tassadar_post_article_router_plugin_tool_loop_pilot_v1/tassadar_post_article_router_plugin_tool_loop_pilot_bundle.json`
- example writer:
  `cargo run -p psionic-serve --example tassadar_post_article_router_plugin_tool_loop_pilot_bundle`
- checker:
  `scripts/check-tassadar-post-article-router-plugin-tool-loop.sh`

`psionic-router` now owns the reusable starter-plugin tool gateway in
`crates/psionic-router/src/tassadar_post_article_starter_plugin_tool_loop.rs`,
which registers the four starter plugins as native router tools and routes each
tool call into the shared starter-plugin runtime bridge.

`psionic-serve` now owns the served pilot bundle in
`crates/psionic-serve/src/openai_http/tassadar_post_article_router_plugin_tool_loop_pilot.rs`.
The committed bundle freezes:

- one success pilot with six bounded turns and five plugin receipt rows
- one typed-refusal pilot with three bounded turns and structured refusal
  preservation
- one continuation row proving response-state replay, stable conversation id,
  and explicit route truth after the first served response

## What Is Green

- one real `/v1/responses` seed response that exposes `plugin_text_url_extract`
  from projected starter-plugin tool definitions
- one router-owned tool gateway that executes all four starter plugins through
  the shared bridge instead of hand-wired executors
- per-step structured tool results that preserve full plugin receipts
- typed refusal survival through the router-owned tool loop
- bounded step count, explicit route truth, and explicit response-state
  continuation in one committed pilot artifact

## Planned

- Apple FM local plugin-session integration now lands separately in
  `docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md`
- cross-lane trace corpus and parity matrix now land separately in
  `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md`

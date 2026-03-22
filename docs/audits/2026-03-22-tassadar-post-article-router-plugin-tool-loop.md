# 2026-03-22 Tassadar Post-Article Router Plugin Tool Loop

`TAS-224` closes the first router-owned starter-plugin tool loop on
`/v1/responses`.

## Landed Surfaces

- `psionic-router` now owns the reusable starter-plugin gateway in
  `crates/psionic-router/src/tassadar_post_article_starter_plugin_tool_loop.rs`
- `psionic-serve` now owns the served pilot bundle in
  `crates/psionic-serve/src/openai_http/tassadar_post_article_router_plugin_tool_loop_pilot.rs`
- committed fixture truth now lives at
  `fixtures/tassadar/runs/tassadar_post_article_router_plugin_tool_loop_pilot_v1/tassadar_post_article_router_plugin_tool_loop_pilot_bundle.json`
- `scripts/check-tassadar-post-article-router-plugin-tool-loop.sh` now acts as
  the dedicated checker over the router gateway, served pilot bundle, and
  targeted tests
- `docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md` now tracks the router-owned
  controller boundary explicitly

## What Is Green

- one real `/v1/responses` seed response that exposes the projected
  `plugin_text_url_extract` tool call
- one router-owned tool gateway that dispatches projected tool calls into the
  shared starter-plugin runtime bridge
- one success pilot where five plugin calls execute in sequence through
  `ToolLoopController`
- one refusal pilot where a typed fetch refusal remains structured and
  receipt-bound after crossing the tool loop
- one continuation row proving explicit response-state replay, conversation
  revision advance, and route truth

## What Is Still Refused

- weighted controller closure
- open-ended model planning claims
- Apple FM session completion before the Apple FM lane lands
- cross-lane publication or controller-closure claims before the umbrella
  closeout lands

## Next Frontier

The next open orchestration frontier above the router-owned served lane is
`TAS-221`: publish the first real-run multi-plugin orchestration wave.

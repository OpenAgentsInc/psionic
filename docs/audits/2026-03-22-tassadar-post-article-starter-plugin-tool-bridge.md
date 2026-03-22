# 2026-03-22 Tassadar Post-Article Starter Plugin Tool Bridge

`TAS-222` closes the shared starter-plugin projection and receipt bridge above
the runtime-owned starter plugins.

## Landed Surfaces

- `psionic-runtime` now owns the shared bridge in
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_tool_bridge.rs`
- the same runtime writes committed fixture truth to
  `fixtures/tassadar/runs/tassadar_post_article_starter_plugin_tool_bridge_v1/tassadar_post_article_starter_plugin_tool_bridge_bundle.json`
- `scripts/check-tassadar-post-article-starter-plugin-tool-bridge.sh` now acts
  as the dedicated checker over the bridge bundle and targeted tests
- `docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md` now tracks the shared
  deterministic or router-owned or Apple FM projection boundary explicitly

## What Is Green

- one shared tool-definition projection surface for deterministic workflows,
  router-owned `/v1/responses`, and Apple FM sessions
- stable schema digests across all three controller surfaces for the current
  four starter plugins
- one shared execution adapter that dispatches projected tool calls into the
  real starter-plugin runtime
- one receipt-bound projected result envelope that preserves structured payloads
  and plugin invocation receipts
- one typed refusal bridge row proving plugin refusals do not degrade into
  free-form tool text

## What Is Still Refused

- weighted controller closure
- router integration claims before the router lane actually consumes this bridge
- Apple FM integration claims before the Apple FM lane actually consumes this
  bridge
- broader multi-plugin controller parity before the later pilot and corpus
  issues land

## Next Frontier

The next open orchestration frontier above the shared bridge is `TAS-223`: the
deterministic starter workflow controller and web-content pilot.

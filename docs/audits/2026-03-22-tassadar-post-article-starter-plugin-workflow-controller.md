# 2026-03-22 Tassadar Post-Article Starter Plugin Workflow Controller

`TAS-223` closes the first host-owned deterministic multi-plugin controller
above the shared starter-plugin bridge.

## Landed Surfaces

- `psionic-runtime` now owns the deterministic controller in
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_workflow_controller.rs`
- the same runtime writes committed fixture truth to
  `fixtures/tassadar/runs/tassadar_post_article_starter_plugin_workflow_controller_v1/tassadar_post_article_starter_plugin_workflow_controller_bundle.json`
- `scripts/check-tassadar-post-article-starter-plugin-workflow-controller.sh`
  now acts as the dedicated checker over the controller bundle and targeted
  tests
- `docs/TASSADAR_STARTER_PLUGIN_WORKFLOW_CONTROLLER.md` now tracks the bounded
  deterministic controller boundary explicitly

## What Is Green

- one bounded host-owned web-content intake graph above the shared bridge
- one reproducible success pilot where five plugin steps execute in sequence
- one refusal pilot where the controller stops on typed fetch refusal
- explicit content-type branch decisions between readability extraction and feed
  parsing
- explicit per-step plugin receipt truth and refusal rows in the committed
  artifact

## What Is Still Refused

- open-ended agent planning
- weighted controller closure
- router-lane completion before the router tool loop consumes this graph
- Apple FM completion before the Apple FM lane consumes the same bridge

## Next Frontier

`TAS-224` is now also closed by the router-owned served pilot in
`docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md` and its committed bundle at
`fixtures/tassadar/runs/tassadar_post_article_router_plugin_tool_loop_pilot_v1/tassadar_post_article_router_plugin_tool_loop_pilot_bundle.json`.

The next open orchestration frontier above the deterministic controller is now
`TAS-221`: publish the first real-run multi-plugin orchestration wave.

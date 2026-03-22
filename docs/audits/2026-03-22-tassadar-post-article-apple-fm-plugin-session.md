# 2026-03-22 Tassadar Post-Article Apple FM Plugin Session

`TAS-225` closes the first session-aware local Apple FM starter-plugin
controller lane above the shared starter-plugin bridge.

## Landed Surfaces

- `psionic-apple-fm` now owns the reusable Apple FM starter-plugin projection
  in `crates/psionic-apple-fm/src/tassadar_post_article_starter_plugin_tools.rs`
- `psionic-apple-fm` now owns the session-aware local pilot bundle in
  `crates/psionic-apple-fm/src/client/tassadar_post_article_starter_plugin_session_pilot.rs`
- committed fixture truth now lives at
  `fixtures/tassadar/runs/tassadar_post_article_apple_fm_plugin_session_pilot_v1/tassadar_post_article_apple_fm_plugin_session_pilot_bundle.json`
- `scripts/check-tassadar-post-article-apple-fm-plugin-pilot.sh` now acts as
  the dedicated checker over the Apple FM projection, local pilot bundle, and
  targeted tests
- `docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md` now tracks the local Apple FM
  controller boundary explicitly

## What Is Green

- one Apple FM tool-definition surface covering the four shared starter plugins
- one session-aware callback runtime that can execute more than one projected
  plugin in sequence
- one success pilot where five plugin calls execute through the shared runtime
  while transcript truth stays explicit
- one refusal pilot where a typed fetch refusal remains structured and
  receipt-bound after crossing the Apple FM callback lane
- one deterministic committed bundle that records session-token binding truth
  without recording callback URL or raw session token

## What Is Still Refused

- served-model closure
- cross-platform serving claims
- weighted controller closure
- cross-lane publication or controller-closure claims before the umbrella
  closeout lands

## Next Frontier

The next open orchestration frontier above the three current controller lanes
is the later `TAS-204` weighted-controller lane above the published
orchestration wave.

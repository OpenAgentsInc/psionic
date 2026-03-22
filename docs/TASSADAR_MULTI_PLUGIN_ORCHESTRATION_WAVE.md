# Tassadar Multi-Plugin Orchestration Wave

This document tracks the first published real-run multi-plugin orchestration
wave above the shared starter-plugin runtime.

The boundary is narrow on purpose:

- the wave ties together the shared tool bridge, deterministic controller,
  router-owned served controller, local Apple FM controller, and trace corpus
- the tranche is operator-internal experimental controller work above the
  shared plugin runtime, not proof-bearing weights-only controller closure
- every lane keeps machine-legible decision, tool, receipt, refusal, or parity
  truth explicit
- completion of this wave still leaves later `TAS-204` weighted-controller
  work, publication widening, and platform closeout as separate tasks

## Implemented

- shared projection and receipt bridge:
  `docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md`
- deterministic workflow controller:
  `docs/TASSADAR_STARTER_PLUGIN_WORKFLOW_CONTROLLER.md`
- router-owned served tool loop:
  `docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`
- local Apple FM plugin session:
  `docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md`
- lane-neutral trace corpus and parity matrix:
  `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md`

The wave now publishes one reviewable sequence above the shared runtime:

1. one shared starter-plugin projection surface
2. one deterministic host-owned multi-plugin workflow
3. one router-owned `/v1/responses` multi-plugin tool loop
4. one local Apple FM session-aware multi-plugin tool lane
5. one repo-owned trace corpus and parity matrix across those controller lanes

## What Is Green

- the same bounded starter-plugin catalog is reused across deterministic,
  router-owned, and Apple FM controller surfaces
- all three controller lanes execute more than one plugin in sequence on real
  committed pilots
- typed refusals remain explicit rather than hidden behind retry or summary
  text
- the trace corpus keeps disagreement rows explicit and receipt-bound instead
  of inventing a fake controller consensus
- the wave stays clearly below `TAS-204` weights-only controller claims

## What Is Still Refused

- proof-bearing weights-only plugin control
- any widening of article-equivalence or served universality claims
- open plugin publication or marketplace closure
- app-local glue disguised as plugin-platform completion

## Planned

- later `TAS-204` weighted-controller work remains separate and must consume
  the published bridge, controller, and corpus surfaces honestly

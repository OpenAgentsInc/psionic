# Tassadar Starter-Plugin User Authoring Wave

This document tracks the first published starter-plugin user-authoring wave
above the shared starter-plugin runtime.

The boundary is narrow on purpose:

- the wave is about repo users adding bounded host-native starter plugins that
  actually work across the runtime, bridge, catalog, controller, and weighted
  controller surfaces
- the first-class path remains capability-free local deterministic starter
  plugins
- one manual `networked_read_only` proof now exists, but it remains narrower
  and more policy-heavy than the capability-free path
- every surface keeps registration, schema, refusal, receipt, and claim
  boundaries explicit
- completion of this wave still does not claim a public plugin marketplace,
  arbitrary external binary loading, or broad automatic admission

## Implemented

- central runtime-owned starter-plugin registry:
  `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md`
- bounded authoring contract and template:
  `docs/TASSADAR_STARTER_PLUGIN_AUTHORING.md`
- shared projection and receipt bridge:
  `docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md`
- starter catalog and downstream report surface:
  `docs/TASSADAR_STARTER_PLUGIN_CATALOG.md`
- deterministic host-owned workflow controller:
  `docs/TASSADAR_STARTER_PLUGIN_WORKFLOW_CONTROLLER.md`

The wave now publishes one dependency-ordered user-authoring path:

1. add a starter plugin once in the shared runtime registry
2. derive shared bridge exposure from that registry
3. derive starter-catalog exposure from that registry
4. prove one host-owned controller composition case
5. publish the bounded authoring template and scaffold helper
6. keep capability-free and networked authoring classes explicitly separate
7. admit the bounded user-added capability-free path into the canonical
   weighted Tassadar controller lane
8. keep the proved user-added `networked_read_only` path narrow and explicit
   instead of treating it as broad networked authoring closure

## What Is Green

- one central starter-plugin registration source instead of repeated per-surface
  tables
- two proven user-added starter plugins:
  `plugin.text.stats` for the capability-free class and
  `plugin.http.fetch_text` for the first manual `networked_read_only` class
- both user-added plugins now run from runtime truth through the shared bridge,
  starter catalog, deterministic workflow controller, and canonical weighted
  controller lane
- one published authoring contract and one narrow scaffold helper for the
  capability-free class
- one explicit boundary keeping networked starter-plugin authoring manual and
  policy-heavy instead of inheriting unsafe defaults from the capability-free
  path

## What Is Still Refused

- open plugin publication or marketplace closure
- arbitrary external binary loading
- automatic admission of every new plugin into every controller surface
- collapsing networked and capability-free authoring into one default path

## Planned

- future user-added capability-free starter plugins should follow the same
  shared-registry path and reach later controller surfaces without new parallel
  metadata tables
- future `networked_read_only` plugins should follow the same manual shared
  path only when their mount, replay, and refusal contracts are equally
  explicit
- broader publication, trust-tier widening, secret-backed classes, stateful
  classes, or networked-authoring automation remain later separate issues

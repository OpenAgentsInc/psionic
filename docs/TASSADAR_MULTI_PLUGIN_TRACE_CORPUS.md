# Tassadar Multi-Plugin Trace Corpus

This document tracks the first repo-owned, lane-neutral multi-plugin trace
corpus and parity matrix above the deterministic, router-owned, and local
Apple FM controller lanes.

The boundary is narrow on purpose:

- the record shape is lane-neutral rather than Apple-only or router-only
- each trace record keeps directive text, admissible tool set, projected tool
  schemas, controller decisions, plugin arguments, projected outputs or
  refusals, receipt identity, and stop condition explicit
- the parity matrix keeps disagreement rows explicit instead of smoothing them
  into a synthetic consensus result
- this bundle is bootstrap input to later `TAS-204` weighted-controller work,
  not proof that `TAS-204` is already solved

## Implemented

- corpus builder:
  `crates/psionic-data/src/tassadar_multi_plugin_trace_corpus.rs`
- committed bundle:
  `fixtures/tassadar/datasets/tassadar_multi_plugin_trace_corpus_v1/tassadar_multi_plugin_trace_corpus_bundle.json`
- example writer:
  `cargo run -p psionic-data --example tassadar_multi_plugin_trace_corpus_bundle`
- checker:
  `scripts/check-tassadar-multi-plugin-trace-corpus.sh`

`psionic-data` now owns one repo-native trace corpus bundle that consumes the
committed deterministic workflow bundle, router-owned served pilot bundle, and
Apple FM local session bundle without collapsing them into one fake canonical
controller trace.

The committed bundle freezes:

- three source controller bundles
- six normalized trace records across the two shared workflow families
- four projected tool-schema rows
- two workflow parity rows
- fifteen explicit disagreement rows
- one bootstrap contract that keeps receipt identity and disagreement
  retention mandatory

## What Is Green

- one machine-legible trace record shape that is not Apple-only in schema or
  semantics
- one explicit parity surface comparing deterministic, router-owned, and Apple
  FM traces on the same bounded workflows
- per-step plugin receipt identity retained inside every normalized trace
  record
- explicit disagreement rows for directive drift, payload drift, receipt drift,
  and stop-condition drift instead of synthetic consensus
- one honest bootstrap boundary to later weighted-controller work with
  `bootstrap_ready = true` while still refusing weighted-controller closure

## Adjacent Surface

The deterministic controller source lane lives in
`docs/TASSADAR_STARTER_PLUGIN_WORKFLOW_CONTROLLER.md`.

The router-owned served source lane lives in
`docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`.

The local Apple FM source lane lives in
`docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md`.

## Planned

- later `TAS-204` weighted-controller work remains separate and must preserve
  receipt identity plus explicit disagreement rows

# 2026-03-22 Tassadar Multi-Plugin Trace Corpus

`TAS-226` closes the first repo-owned multi-plugin trace corpus, parity
matrix, and training-bootstrap contract above the deterministic, router-owned,
and local Apple FM controller lanes.

## Landed Surfaces

- `psionic-data` now owns the lane-neutral corpus builder in
  `crates/psionic-data/src/tassadar_multi_plugin_trace_corpus.rs`
- committed fixture truth now lives at
  `fixtures/tassadar/datasets/tassadar_multi_plugin_trace_corpus_v1/tassadar_multi_plugin_trace_corpus_bundle.json`
- `scripts/check-tassadar-multi-plugin-trace-corpus.sh` now acts as the
  dedicated checker over the corpus bundle, the example writer, and targeted
  tests
- `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md` now tracks the corpus and
  bootstrap boundary explicitly

## What Is Green

- one repo-owned trace record schema covering deterministic, router-owned, and
  Apple FM controller traces without using an Apple-only dataset shape
- six normalized trace records bound back to three committed source bundles
- one parity matrix that keeps directive drift, payload drift, receipt drift,
  and stop-condition drift explicit
- one bootstrap contract that requires receipt identity plus disagreement-row
  retention before later weighted-controller work is honest

## What Is Still Refused

- weighted-controller closure
- any served claim that `TAS-204` is already solved
- synthetic controller consensus that discards real lane disagreements
- proof-bearing Tassadar controller claims beyond the bounded source bundles

## Next Frontier

The next later frontier above the now-published controller and corpus lanes is
the separate `TAS-204` weighted-controller lane.

# Tassadar ALM Futamura Specializer

> Status: `implemented_early` for executor-compiler phase E5 (pulled ahead
> of E3/E4 because it operates purely on the landed IR and backend),
> landed 2026-06-10 under issue #1100. E1 is `docs/TASSADAR_ALM_GRAPH.md`;
> E2 is `docs/TASSADAR_ALM_BACKEND.md`.

This document is the contract for the first-Futamura-projection
specializer in `crates/psionic-compiler/src/tassadar_alm_specializer.rs`.

## Identity

- specializer family: `tassadar_alm_first_futamura`
- specializer version: `v1`
- claim boundary: ALM specialization rewrites reads of one static seeded
  channel into exact ReGLU step-function fetches and removes the channel;
  the rewrite is claimed only for programs whose reads query seeded keys,
  because the step-function fetch totalizes the partial keyed-read
  function instead of refusing between keys; no tensor weights, serving,
  or public capability copy is created.

## What Specialization Does

`specialize_tassadar_alm_graph(graph, channel)` applies classical partial
evaluation to one **static** keyed channel — a channel populated only by
seed writes and never written by any gate. Every `ChannelRead` on that
channel becomes the construction post's step-function fetch over the
channel's sorted seed entries `(k_i, c_i)`:

```
fetched(q) = c_0 + Σ_{i>=1} (c_i − c_{i−1}) · 1[q >= k_i]
1[q >= k]  = relu(q − k + 1) − relu(q − k)
```

with each indicator built from two ReGLU gates over a shared constant-one
gate (zero-delta entries are skipped). The channel declaration and its
seed writes disappear from the specialized graph. The program has moved
from channel state into gate structure — the IR-level analog of moving
the instruction table from the prompt prefix into feed-forward weights,
with the same O(N)-gates-per-read accounting as the post's 2N shared
neurons.

Typed refusals: unknown channel, accumulator channel, dynamic channel
(any gate writes it — refused with the offending gate index), and empty
seed set.

Every run produces a digest-pinned `TassadarAlmSpecializationReport`
binding source and specialized graph digests, entry count, rewritten-read
count, and gate counts.

## Semantics Boundary

The keyed read is a partial function (missing key refuses). The
step-function fetch is total: queries between seeded keys resolve to the
value at the largest seeded key below, and queries below the first key
resolve to `c_0`. Specialization is therefore claimed only for programs
whose reads always hit seeded keys — the cursor-in-range situation. The
parity tests are the check; no static analysis of query ranges is
attempted in v1.

## Parity Bar (Landed)

On the delta-program workload (a four-instruction static program fetched
by cursor and accumulated): specialized-graph evaluator outputs equal the
original step-for-step, and the specialized graph compiles through the E2
backend with the compiled bundle reproducing the same outputs — the full
IR → specialize → schedule → compiled-execution pipeline. The
verb-parity workload's parity channel correctly refuses specialization
(it is dynamic), as do accumulators, unknown channels, and empty seeds.

## What This Phase Does Not Do

No tensor weight materialization, no shared-neuron deduplication across
multiple reads (each read inlines its own indicators in v1; sharing is an
optimization for the MILP phase), no Wasm intake, no serving.

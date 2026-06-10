# Tassadar ALM Gate-Graph IR And Exact Evaluator

> Status: `implemented_early` for executor-compiler phase E1, landed
> 2026-06-10 under issue #1098.

This document is the contract for the bounded Append-only Lookup Machine
(ALM) gate-graph IR and its exact reference evaluator in
`crates/psionic-ir/src/tassadar_alm_graph.rs`.

It is the first phase of the executor-compiler program: a general
program-to-weights pipeline for the Tassadar lane, following the published
Percepta construction (ALM/CALM → gate graph → scheduled layers →
analytical weights). The cross-repo design sketch lives in
`openagents/docs/tassadar/2026-06-10-psionic-alm-compiler-design-speculation.md`.

## Identity

- language id: `tassadar.alm_gate_graph.v1`
- schema version: `1`
- claim class: `exact_integer_reference_semantics`
- claim boundary: the ALM gate graph and its exact evaluator define
  executor-compiler reference semantics over checked 64-bit integers only;
  they do not emit transformer weights, do not schedule layers, do not
  accept Wasm, and do not create any learned-model, served-route, or
  transformer-execution claim.

## The Machine

One `TassadarAlmGraph` is a per-step gate program executed once per token
step over declared channels. The gate families are exactly the five ALM
primitives, each chosen because a transformer component implements it
exactly:

| Gate | Meaning | Transformer realization (E2+) |
| --- | --- | --- |
| `Input` | per-step input field | token embedding |
| `Const` / `Linear` | exact linear combination | residual wiring |
| `ReGlu` | `value * max(gate, 0)` | gated FFN neuron |
| `ChannelWrite` / `ChannelRead` | keyed memory, latest write wins | 2D lookup attention head |
| `CumSum` | exact running sum including the current step | uniform-key attention head |

Channels are declared `Keyed` or `Accumulator` and gates must match the
declared kind.

## Semantics

- Gates execute in declaration order; a gate may reference only values
  produced by earlier gates (forward references are validation errors, so
  the per-step program is a DAG by construction).
- **Visibility rule:** keyed reads at step `t` see seed writes plus writes
  from steps `<= t - 1`. Writes made at step `t` become visible at step
  `t + 1`. There is no same-step read-after-write hazard by construction.
  This mirrors the transformer reality that a position's keys and values
  serve later positions.
- Latest write wins per key. Reading a key that has never been written is
  a typed refusal (`MissingKey`), not a default value.
- `CumSum` returns the exact prefix sum including the current step's
  contribution.
- Seed writes initialize keyed channels before step zero, covering the
  previous-token reads that step zero would otherwise miss.
- All arithmetic is checked `i64`; overflow is a typed refusal.

## Bounds

- gates per graph: 65,536
- channels per graph: 256
- linear terms per gate: 64
- evaluation steps: 1,048,576

## Evaluator

`TassadarAlmEvaluator::evaluate(graph, steps)` validates the graph and
runs it deterministically, producing a `TassadarAlmTrace`: per-step output
rows, the keyed-write count, the graph digest, and a stable trace digest
over the output rows. The evaluator is the reference leg for all later
phases: E2's emitted weights must reproduce these traces exactly before
any parity claim.

## Landed Test Workloads

- running sum (accumulator semantics, digest determinism and input
  sensitivity)
- the Percepta post's verb-parity toy (seeded previous-token read, XOR via
  `a + b - 2ab` with the product through `ReGlu`)
- a bounded three-instruction stack micro (`PUSH 3, PUSH 5, ADD, OUT`)
  with masked writes built from the ReLU step identity
  `1[x >= c] = relu(x - c + 1) - relu(x - c)` — depth as an accumulator,
  stack cells as a keyed channel
- typed-refusal coverage: forward references, channel-kind mismatch,
  missing keys, overflow, and the same-step write-visibility rule

## What This Phase Does Not Do

No scheduling, no slot allocation, no weight emission, no Wasm frontend,
no specialization, no serving. Those are phases E2–E6 in the design
sketch and land behind the existing Tassadar disclosure gates.

# Tassadar ALM Numeric Model Materialization

> Status: `implemented_early` for executor-compiler phase E6-numeric
> (bounded), landed 2026-06-10 under issue #1113.

This document is the contract for the numeric materialization lane in
`crates/psionic-compiler/src/tassadar_alm_numeric.rs`.

## Identity

- model schema: `TassadarAlmNumericModel` v1
- executor id: `tassadar.alm_numeric_executor.v1`
- claim boundary: the numeric model is a faithful f64 re-encoding of one
  compiled ALM bundle — explicit coefficient arrays executed with
  hard-max attention inside a checked exactness window of 2^53 — not a
  trained transformer; it claims integer parity only while every
  intermediate stays inside the window, refuses when one does not, and
  makes no softmax, learning, or served-route claim.

## What This Phase Adds

Every previous leg executes the compiled bundle as *code*. This leg
re-encodes the bundle as *data*: a serde-serializable, digest-pinned
model whose layers are explicit f64 coefficient arrays —

- wiring rows as sparse linear maps over the residual vector (the
  weights-shaped artifact: `out = bias + Σ coeffᵢ · residual[slotᵢ]`),
- FFN rows as gated neurons `value · max(gate, 0)`,
- attention rows as parabolic-point hard-max reads and accumulator sums,
- write emissions and output slots,

bound to the source graph and bundle digests. The lowering loses no
information; the model is the bundle re-expressed as numbers, and the
portability test proves it: serialize to JSON, deserialize, execute,
identical outputs.

## The Exactness Window

f64 represents every integer with |v| ≤ 2^53 exactly. The executor
checks every computed intermediate against that window and refuses with
a typed `ExactnessWindowExceeded` when breached — so the integer-parity
claim is *checked at runtime*, never assumed. The window is narrower
than i64 by design; the bounded harness treats a window breach as an
acceptable domain demotion, and inside the window the numeric leg must
agree exactly.

## Parity Bars (Landed)

4 tests: digest equality with the integer legs on the committed
workloads; **a real runtime `TassadarProgram`** (the backward-branch
loop) materialized through interpreter → bundle → numeric model and
matching the production CPU reference runner's outputs; the
serde-roundtrip portability proof; and the window refusal. The bounded
differential harness now runs **five legs** (evaluator / row /
geometric / hull / numeric) over its 400 generated graphs with zero
failures.

## Run-Facing Program Corpus

`build_tassadar_alm_numeric_program_corpus_fixture_v1` emits the first
run-facing compiled-program corpus for OpenAgents C1. It compiles four
distinct `TassadarProgram`s through the same owned path:

- backward-branch loop sum,
- straight-line arithmetic,
- memory load/store roundtrip,
- factorial countdown state machine.

For each program, the builder validates CPU-reference outputs, lowers
through the ALM Wasm interpreter, schedules the graph, materializes the
numeric model, executes it, and stores the program digest, model digest,
trace digest, expected outputs, and public-safe compile receipt refs.
The committed fixture can be regenerated with the ignored
`dump_numeric_program_corpus_fixture` test in
`crates/psionic-compiler/src/tassadar_alm_numeric.rs`.

## What This Phase Does Not Do

Hard-max attention only — no softmax, no temperature, no approximation
error analysis (that is the post's carry-over-bounds territory and a
later phase). No trained weights, no learning, no d_model packing of
slots into dense matrices (slots remain scalar lanes), no serving.

Dense W1.2 materialization is layered on top of this numeric contract in
`docs/TASSADAR_ALM_DENSE_MODULE.md`: it packs one compiled numeric model
into loadable full-width matrices while preserving this phase's hard-max,
bounded-replay claim boundary.

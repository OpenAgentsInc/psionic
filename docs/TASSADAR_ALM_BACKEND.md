# Tassadar ALM Backend Phase 1: Scheduler, Slot Allocator, Compiled Bundle

> Status: `implemented_early` for executor-compiler phase E2 (feasible-first
> backend), landed 2026-06-10 under issue #1099. E1 (the ALM IR and exact
> evaluator) is `docs/TASSADAR_ALM_GRAPH.md`.

This document is the contract for the feasible-first ALM backend in
`crates/psionic-compiler/src/tassadar_alm_backend.rs`.

## Identity

- compiler family: `tassadar_alm_backend_list_schedule`
- compiler version: `v1`
- bundle schema version: `1`
- claim boundary: the compiled ALM bundle executes integer-exact analytical
  rows produced by a feasible-first list scheduler and an interval-coloring
  slot allocator; it proves evaluator parity for committed workloads only
  and does not claim optimal scheduling, tensor weight materialization,
  hull-cache decode, Wasm intake, or any served-route capability.

## What The Backend Does

`compile_tassadar_alm_graph` lowers one validated `TassadarAlmGraph` into a
digest-pinned `TassadarAlmCompiledBundle`:

1. **List scheduling** over the four-phase layer structure. Global phase
   indices: phase 0 is the embedding phase (inputs, constants); layer `L`
   occupies phases `4L+1..=4L+4` as attention / persist / FFN / persist.
   Type constraints: keyed reads and accumulator sums go to attention
   phases, ReGLU products to FFN phases, linear wiring to persist phases.
   Every dependency is strictly earlier than its consumer's phase. Channel
   writes are end-of-step emissions: their key and value operands stay
   live to end-of-step, and the write gate's own value aliases its value
   operand's slot.
2. **Interval-coloring slot allocation.** Value lifetimes run from birth
   phase to last-consumer phase (end-of-step for outputs and write
   operands). Slots are reused greedily when lifetimes do not overlap, and
   every reuse carries an explicit `TassadarAlmSlotSubtraction` record,
   because the residual stream is additive and a reused slot must have its
   stale value cleared.
3. **Row emission.** The bundle carries wiring rows (input/const/linear),
   attention rows (keyed read, cumsum), FFN rows (ReGLU), end-of-step
   write rows, output slots, placements, and subtraction records — all in
   phase order, all referring only to slots.

A schedule validity checker (`validate_schedule`) enforces phase-kind and
strict-precedence invariants before any bundle is produced.

## The Compiled Executor

`TassadarAlmCompiledExecutor::execute` runs a bundle from its own rows
only: an explicit `i64` residual vector of `slot_count` slots per step,
plus keyed-channel and accumulator state fed exclusively by bundle rows.
It has no access to the source graph. Checked arithmetic and typed
refusals (`MissingKey`, `Overflow`, input arity) mirror the evaluator;
`tassadar_alm_errors_match` maps the two error families for parity
assertions.

## Parity Bar (Landed)

For the committed E1 workloads — running sum, verb parity, stack micro —
the compiled bundle's step outputs and trace digest equal the E1
evaluator's exactly. The stack-micro bundle demonstrates real slot reuse
(`slot_count` strictly below gate count, non-empty subtraction records).
Refusal parity is covered for the missing-key case.

## What This Phase Does Not Do

- No MILP optimal scheduling (that is phase E4; this scheduler is
  feasible-first and stays as the fallback and cross-check when the
  optimal backend lands).
- No f32 tensor materialization into
  `psionic-models::TassadarExecutorAttentionWeightBundle`; the bundle is
  integer-exact analytical rows.
- No hull-cache decode, no Wasm frontend, no specialization, no serving.

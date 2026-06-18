# Tassadar ALM Backend: E4 Scheduler, Slot Allocator, Compiled Bundle

> Status: `implemented` for executor-compiler phase E4 scheduling over the
> E2 compiled-bundle row format. E1 (the ALM IR and exact evaluator) is
> `docs/TASSADAR_ALM_GRAPH.md`.

This document is the contract for the ALM backend in
`crates/psionic-compiler/src/tassadar_alm_backend.rs`.

## Identity

- default compiler family: `tassadar_alm_backend_e4_milp_schedule`
- default compiler version: `v2`
- legacy comparison compiler family: `tassadar_alm_backend_list_schedule`
- legacy comparison compiler version: `v1`
- bundle schema version: `1`
- claim boundary: the compiled ALM bundle executes integer-exact analytical
  rows produced by the E4 finite-horizon MILP scheduler and an
  interval-coloring slot allocator; it minimizes peak liveness exactly for
  bounded graph windows, never accepts a schedule wider than the legacy
  feasible-first scheduler, and does not claim tensor weight
  materialization, softmax approximation, Wasm-window expansion, or any
  served-route capability.

## What The Backend Does

`compile_tassadar_alm_graph` lowers one validated `TassadarAlmGraph` into a
digest-pinned `TassadarAlmCompiledBundle`:

1. **E4 phase scheduling** over the four-phase layer structure. Global
   phase indices: phase 0 is the embedding phase (inputs, constants);
   layer `L` occupies phases `4L+1..=4L+4` as attention / persist / FFN /
   persist. Type constraints place keyed reads and accumulator sums in
   attention phases, ReGLU products in FFN phases, and linear wiring in
   persist phases. Every non-write dependency is strictly earlier than its
   consumer's phase. Channel writes are end-of-step emissions: their key
   and value operands stay live to end-of-step, and the write gate's own
   value aliases its value operand's slot.
2. **Finite-horizon peak-liveness optimization.** The legacy list
   scheduler remains available as `compile_tassadar_alm_graph_greedy`.
   The default E4 scheduler uses that schedule as the horizon and
   incumbent, then evaluates ALAP and local integer candidates plus an
   exact branch-and-bound MILP search for bounded graph windows
   (`<=12` gates, capped node count). Candidate schedules that violate
   phase kind, precedence, or cumsum ordering are discarded. The selected
   schedule must validate and must have `slot_count <=` the greedy
   schedule on the same graph.
3. **Interval-coloring slot allocation.** Value lifetimes run from birth
   phase to last-consumer phase (end-of-step for outputs and write
   operands). Slots are reused greedily when lifetimes do not overlap, and
   every reuse carries an explicit `TassadarAlmSlotSubtraction` record,
   because the residual stream is additive and a reused slot must have its
   stale value cleared.
4. **Row emission.** The bundle carries wiring rows (input/const/linear),
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

The committed C2 bar also compares the default E4 scheduler against the
legacy greedy scheduler over the E1 workloads and the four run-facing Wasm
program corpus. The regenerated corpus fixture
`fixtures/tassadar/tassadar-compiled-program-corpus-v1.json` is emitted by
the E4 compiler identity and has corpus digest
`1b7babcd0c3ce63e43212f3e4f07480969a7a9612a237b117f8de7fb8a828d6a`.

## What This Phase Does Not Do

- No f32 tensor materialization into
  `psionic-models::TassadarExecutorAttentionWeightBundle`; the bundle is
  integer-exact analytical rows.
- No softmax approximation bounds or Wasm-window expansion beyond the
  current frontend support.
- No hull-cache decode, no serving.

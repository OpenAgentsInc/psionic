# Tassadar ALM Geometric Attention Execution Leg

> Status: `implemented_early` for executor-compiler phase E2b, landed
> 2026-06-10 under issue #1109.

This document is the contract for the geometric execution leg in
`crates/psionic-compiler/src/tassadar_alm_geometric.rs`.

## Identity

- executor id: `tassadar.alm_geometric_executor.v1`
- claim boundary: the geometric executor realizes keyed reads as
  parabolic-key argmax over append-only point lists and accumulators as
  uniform-attention sums, in exact integers with linear-scan argmax; it
  proves mechanism parity with the evaluator and row executor only and
  makes no f32, softmax, hull-fast-path, or served-route claim.

## What This Leg Proves

The previous executors realized keyed channels as map lookups — correct,
but not the construction's mechanism. This leg executes compiled bundles
through the **actual geometry**:

- Every write appends a point `(2k, −k²; value, write_order)` to its
  channel's list; seed writes seed the lists.
- A read scores every point against the direction `(q, 1)`:
  `score = 2qk − k² = −(k − q)² + q²`, uniquely maximal at `k = q` among
  distinct keys. Duplicate keys tie exactly, and the tie breaks to the
  **latest write order** — the executable analog of the construction's
  position-dependent key perturbation, stated as such.
- The winning key is verified against the query: a near-miss argmax
  (closest-but-not-equal key) is a typed `MissingKey` refusal, never an
  interpolation — preserving the partial keyed-read semantics that the
  evaluator defines and that the E5 specializer deliberately totalizes.
- Accumulators sum their contribution lists through the
  uniform-attention formulation (average × count = sum, exact in
  integers).

## Parity Bars (Landed)

- Trace-digest equality with the E1 evaluator and the E2 row executor on
  every committed workload: running sum, verb parity, stack micro, the
  stack-ISA program (universal **and** E5-specialized), and all bridged
  symbolic examples.
- The bounded differential harness (#1107) now runs **three legs** —
  evaluator / row executor / geometric executor — over its 400 generated
  graphs with zero failures, refusal parity included.
- Dedicated tests pin the near-miss refusal and the duplicate-key
  latest-write tie-break.

The trace carries `argmax_comparisons` so the E2c hull fast path has a
measured baseline to beat.

## What This Phase Does Not Do

Linear-scan argmax only (the Li Chao / convex-hull-trick fast path is
E2c); no f32 tensor materialization; no softmax or hard-max numerics
beyond exact integer scores; no serving.

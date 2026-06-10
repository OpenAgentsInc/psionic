# Tassadar ALM Hull Fast Path

> Status: `implemented_early` for executor-compiler phase E2c, landed
> 2026-06-10 under issue #1110. E2b (the geometric leg this accelerates)
> is `docs/TASSADAR_ALM_GEOMETRIC.md`.

This document is the contract for the hull fast path in
`crates/psionic-compiler/src/tassadar_alm_hull.rs`.

## Identity

- executor id: `tassadar.alm_hull_executor.v1`
- claim boundary: the hull executor accelerates exact integer
  parabolic-key argmax with a Li Chao tree over a declared query window,
  demoting out-of-window reads to the linear-scan fallback with counts
  reported; it changes retrieval cost only, proves digest parity with the
  evaluator, row, and geometric legs, and makes no f32, softmax, or
  served-route claim.

## The Mechanism

The geometric score `2kq − k²` is the line `y = (2k)·x + (−k²)` evaluated
at `x = q`, so dynamic-insert argmax over keys is the convex-hull-trick
problem. A Li Chao tree over the declared window (|key|, |query| ≤ 2³¹,
scores evaluated in i128 so they cannot overflow) answers which key
maximizes in O(log W) node visits. A per-channel latest-write-wins value
map carries retrieval and the exact-match check: the hull names the
maximizing key; only an exact hit retrieves; a near-miss is a typed
`MissingKey` refusal, identical to every other leg.

## Posture

Borrowed from the existing Tassadar hull lane: in-window reads are
`direct`; an out-of-window key on any write demotes the channel to the
linear-scan `fallback` path, and out-of-window queries fall back per
read. Direct/fallback counts are reported in the trace, never hidden.

## Measured Result (Landed, Deterministic Counts)

On the committed 2,000-step chain workload (one write and one read per
step, so channels grow linearly), the linear-scan baseline exceeds one
million comparisons while hull node visits stay more than an order of
magnitude below it — asserted in the test, not benchmarked by wall
clock, so the number is deterministic and machine-independent.

## Parity Bars (Landed)

Digest equality with the evaluator, row executor, and geometric executor
on the committed workloads; the bounded differential harness (#1107) now
runs **four legs** over its 400 generated graphs with zero failures,
refusal parity included; fallback correctness and missing-key refusals
pinned by dedicated tests.

## What This Phase Does Not Do

No f32 tensors, no softmax or k-sparse approximation, no checkpointed
hull state across executions, no serving. The window is a declared
constant in v1; per-profile windows are a later refinement.

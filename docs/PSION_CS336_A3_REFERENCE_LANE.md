# Psion CS336 A3 Scaling Reference Lane

> Status: `implemented_early` bounded reference lane, landed 2026-06-10
> under issue #1103. Companion to the A1, A2, A4, and A5 reference lanes.

This document records the owned `psionic` surfaces for the bounded
Stanford CS336 Assignment 3 (scaling) port program. It is the
psionic-side answer to the OpenAgents homework epic's external ask
(openagents issue #4679).

## Identity

- lane id: `psion_cs336_a3_scaling_reference_v1`
- owned surface: `crates/psionic-train/src/cs336_a3_scaling_reference.rs`
- claim boundary: bounded deterministic IsoFLOP analysis math over
  supplied sweep cells only; no training runs, no dispatch authority, and
  no claim of fitted-law validity beyond the committed synthetic recovery
  test.

## What The Lane Owns

Stanford A3 is a training-API client exercise: submit `(N, D)` configs
under a compute budget, observe losses, fit scaling laws. The course
hosts the dispatcher; in the OpenAgents homework epic the worker is the
dispatcher and the Pylon network is the cluster. This lane owns the two
analysis halves both sides share:

- **`cs336_a3_plan_isoflop_sweep`** — the dispatchable grid: geometric N
  spacing within declared bounds per compute budget, `D = C / (6N)` per
  cell, validated bounds with typed refusals. One planned run is exactly
  one homework cell in openagents #4679.
- **`cs336_a3_fit_isoflop`** — the Chinchilla approach-2 fit: cells
  grouped by budget; per-budget least-squares parabola of loss against
  `ln N` requiring an interior minimum (typed refusal otherwise); the
  per-budget optima regressed log-log against budget to produce the
  parameter exponent `a` in `N_opt = k·C^a`, its coefficient, and the
  implied token exponent `b = 1 − a`. The digest-pinned
  `Cs336A3IsoflopFit` predicts compute-optimal `(N, D)` for new budgets.

Typed refusals: insufficient cells per budget (< 3), no interior minimum
(rising or degenerate curves), insufficient budgets (< 2), non-positive
cell quantities, invalid planner bounds.

## Landed Tests

6 unit tests, including the load-bearing one: a full
planner → synthetic-loss → fit pipeline over a Chinchilla-form law
(`L = E + A/N^0.5 + B/D^0.5`, analytic optimal exponent 0.5) recovers
the exponent within 0.05 and predicts optima within tolerance of the
analytic value. Plus grid-shape/invariant checks (`6·N·D = C` per cell,
geometric monotonicity), refusal coverage for thin/degenerate inputs and
rising curves, and digest stability.

## Relation To The Homework Epic

OpenAgents #4679 dispatches planned cells as homework, verifies sampled
cells by re-run, and publishes the IsoFLOP dashboard from receipts. The
fit's `budget_optima` and exponents are the dashboard's payload. The
fitted-law outputs are analysis artifacts, not capability claims: any
public copy about "our scaling law" must cite cell receipts and carry the
fit digest.

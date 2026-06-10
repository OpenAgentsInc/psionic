# Tassadar ALM Bounded Differential Check

> Status: `implemented_early`, landed 2026-06-10 under issue #1107. This is
> the formal follow-on promised when E1 landed (#1098).

This document is the contract for the bounded check harness in
`crates/psionic-compiler/src/tassadar_alm_bounded_check.rs`.

## Identity

- claim boundary: the bounded check harness exercises evaluator/compiled
  parity (including refusal parity) and independent allocator-safety
  invariants over seeded generated graphs within a fixed size budget; it
  is strong bounded evidence, not a proof, and creates no capability
  claim.

## What It Checks

1. **Differential parity.** A deterministic seeded enumerator generates
   small ALM graphs over the full gate grammar (valid by construction)
   plus input rows, then requires the E1 evaluator and the E2 compiled
   executor to either produce identical step outputs or refuse with
   matching typed error families — so missing-key and overflow paths are
   exercised, not avoided. The committed test pins one seed across 400
   graphs and requires zero failures, with both the success and refusal
   paths demonstrably hit.
2. **Independent allocator safety.**
   `tassadar_alm_check_bundle_invariants` recomputes lifetimes from the
   source graph and the bundle's placements — not the scheduler's
   internals — and verifies producer-before-consumer phases, phase-kind
   discipline, slot-lifetime disjointness, end-of-step protection for
   outputs and write operands, subtraction-record completeness for every
   reuse, and same-channel cumsum ordering. Corruption tests prove the
   checker catches forced slot collisions, dropped subtractions, and
   truncated placement tables.

## The Bug It Already Found

On its first run the harness found a real scheduler bug: two `CumSum`
gates on one accumulator channel could be reordered across layers
(accumulator contributions are order-sensitive side effects, and a
dependency on an FFN-phase value pushed the earlier gate to a later
layer), silently breaking gate-order accumulation in compiled execution.
The fix chains same-channel cumsums as scheduling dependencies, enforced
in `validate_schedule`, in the independent invariant checker, and by a
pinned regression test. A second latent hazard — same-step writes to one
key resolving in schedule order instead of gate order — was fixed in the
same pass by emitting write rows in source-gate order.

This is the concrete argument for the harness: two semantics bugs that
all twelve hand-written workload tests missed, surfaced by the first 400
generated graphs.

## Boundary

Bounded checking over the committed seed and size budget only. It is not
exhaustive and not a proof; widening budgets, adding shrinking, or
modeling the ALM in an external checker are open follow-ons.

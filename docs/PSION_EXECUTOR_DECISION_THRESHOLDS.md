# Psion Executor Decision Thresholds

> Status: canonical `PSION-0108` / `#713` retained variance and threshold
> packet for phase-one executor promotion review, updated 2026-03-30.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_decision_thresholds_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_decision_threshold_fixtures
```

## What This Packet Does

This packet does not claim fresh GPU benchmark variance.

It replays the committed baseline-truth packet and retained executor reports
three times, records the aggregate promotion metrics that phase-one actually
uses, and then freezes conservative non-zero decision floors for later same-
budget comparisons.

That keeps the current boundary honest:

- retained replay noise is currently zero
- promotion still requires non-zero change floors before claiming improvement
- regressions stay stricter than wins

## Replayed Baseline Metrics

The retained packet replays these promotion-facing aggregates:

- `promotion_exactness_regression_count`
- `promotion_held_out_regression_count`
- `promotion_adversarial_regression_count`
- `promotion_runtime_blocker_red_count`
- `promotion_serving_blocker_red_count`
- `promotion_reference_linear_anchor_median_steps_per_second`
- `promotion_hull_cache_median_steps_per_second`
- `promotion_hull_cache_min_speedup_over_reference_linear`
- `promotion_hull_cache_max_remaining_gap_vs_cpu_reference`

Current retained posture is simple:

- exactness regressions: `0`
- held-out regressions: `0`
- adversarial regressions: `0`
- runtime blocker red rows: `0`
- serving blocker red rows: `0`
- replay span: `0` for every retained aggregate

## Frozen Decision Floors

The first retained floors are:

- exactness regressions: any increase of `1` or more holds promotion
- held-out regressions: any increase of `1` or more holds promotion
- adversarial regressions: any increase of `1` or more holds promotion
- runtime blocker red rows: any increase of `1` or more holds promotion
- serving blocker red rows: any increase of `1` or more holds promotion
- `reference_linear` anchor median throughput:
  candidate must move by at least `5%` to claim improvement, and an equal drop
  holds promotion
- `hull_cache` median throughput:
  candidate must move by at least `5%` to claim improvement, and an equal drop
  holds promotion
- minimum `hull_cache` speedup over `reference_linear`:
  candidate must improve by at least `0.05` to claim a win, and an equal drop
  holds promotion before the absolute `1.5` floor is even considered
- maximum `hull_cache` remaining gap vs CPU reference:
  candidate must lower the worst gap by at least `0.05` to claim a win, and an
  equal increase holds promotion before the absolute `3.0` ceiling is even
  considered

## Promotion Use Rule

These thresholds are the minimum meaningful deltas for phase-one executor
promotion review.

That means:

- candidate changes inside the threshold band count as no material change
- candidate regressions outside the band count as promotion holds
- exactness, held-out, and adversarial regressions remain zero-tolerance

## Honest Boundary

This packet is intentionally conservative.

It is a retained comparison spine for phase one, not a claim that live MLX or
4080 hardware variance has already been exhaustively measured. When later runs
produce real repeated hardware evidence, this packet should be revised from
those runs instead of from retained replay alone.

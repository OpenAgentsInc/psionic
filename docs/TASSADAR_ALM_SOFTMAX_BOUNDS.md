# Tassadar W1.4 Softmax Bounds

> Status: implemented for C4/#5324 on 2026-06-18.

This document records the W1.4 approximation certificate in
`crates/psionic-compiler/src/tassadar_alm_softmax_bounds.rs`.

## Boundary

The current ALM numeric and dense replay paths still execute hard-max
keyed reads. W1.4 adds an analytic bound for comparing that hard-max
winner to a hypothetical softmax with inverse temperature `beta`; it
does not introduce softmax execution, learned weights, f32 serving, or
live-route behavior.

## Bound

For `n` candidates, winner logit margin `Delta > 0`, and inverse
temperature `beta > 0`, define:

`T = (n - 1) * exp(-beta * Delta)`

Then the total probability mass assigned by softmax to non-winning keys
is bounded by:

`T / (1 + T)`

The hardmax winner probability is at least `1 / (1 + T)`, and the L1
distance from the one-hot hardmax distribution is at most
`2 * T / (1 + T)`.

For the ALM keyed-read score family `score(q, k) = 2*q*k - k^2`, when
`q` equals the winning integer key and every other key is at least
distance `d` away, the logit gap is at least `d^2`.

The canonical report builder uses `n = 1024`, `d = 1`, and `beta = 32`.
That certifies:

- non-winner mass `< 1.4e-11`
- L1 distance to hardmax `< 2.7e-11`

Focused verification:

```bash
cargo test -p psionic-compiler tassadar_alm_softmax_bounds -- --nocapture
```

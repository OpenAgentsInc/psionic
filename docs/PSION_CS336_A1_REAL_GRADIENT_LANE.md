# Psion CS336 A1 Real-Gradient Reference Lane

> Status: `implemented_early` bounded reference lane, landed 2026-06-10
> under issue #1114. Companion to the full-port matrix
> (`docs/PSION_CS336_A1_FULL_PORT_MATRIX.md`), whose claim boundary this
> lane narrows: the tiny trainer is no longer finite-difference-only.

This document records the owned `psionic` surface for analytic-gradient
training on the A1 architecture shape, the psionic-side ask of the
OpenAgents homework epic's real-training kind (openagents #4678).

## Identity

- lane id: `psion_cs336_a1_real_gradient_reference_v1`
- owned surface:
  `crates/psionic-train/src/cs336_a1_real_gradient_reference.rs`
- claim boundary: a bounded f64 reference trainer proving
  analytic-gradient correctness on the A1 architecture shape (embedding,
  RMSNorm, single-head causal attention, SwiGLU, cross-entropy) at tiny
  scale, gradient-checked against central differences; no
  scalable-pretraining claim, no GPU claim, and no promotion into the
  actual-pretraining operator lane.

## What The Lane Owns

A self-contained tiny A1-shaped LM in f64 with **hand-derived analytic
backprop** for every parameter tensor:

- forward: embedding → RMSNorm → single-head causal softmax attention
  (Wq/Wk/Wv/Wo, scaled scores, numerically stable softmax) with residual
  → RMSNorm → SwiGLU FFN (W1 gate / W3 up / W2 down) with residual →
  unembedding → mean next-token cross-entropy;
- backward: exact derivatives through cross-entropy-from-logits, the
  unembedding, both residual branches, SwiGLU
  (`silu'(a) = σ(a)(1 + a(1 − σ(a)))`), both RMSNorms
  (`dx = g⊙dy/r − x·(Σ dy⊙g⊙x)/(d·r³)`), causal softmax attention
  (row-wise softmax Jacobian), the three input projections, and
  embedding rows;
- an inline AdamW loop with deterministic seeded initialization and a
  digest-pinned training report.

## The Load-Bearing Bar

`analytic_gradients_match_central_differences_for_every_tensor`: every
parameter tensor's analytic gradient is checked against central
differences at 1e-5 relative tolerance in f64 — the finite-difference
machinery graduates from being the trainer to being the proof. Training
on the tiny corpus must at least halve the loss in 40 AdamW steps, and
training is bitwise deterministic across runs.

## Relation To The Full-Port Matrix And The Epic

The matrix's claim boundary previously said gradients existed only as a
tiny finite-difference reference. This lane upgrades that statement for
the bounded config: real analytic gradients exist and are verified. What
remains honestly open: RoPE and multi-head attention backward, batched
training, any scale beyond the tiny config, and promotion into the
actual-pretraining operator lane. The homework epic's real-training
kind (#4678) can dispatch tiny analytic-gradient steps against this
contract; statistical_cross_check / seeded_replication remain the
verification grades for training work.

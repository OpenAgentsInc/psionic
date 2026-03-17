# Psionic MLX Benchmark Package

This document defines the bounded `psionic-mlx-bench` package that closes
`PMLX-708`.

## Scope

`psionic-mlx-bench` is the MLX-facing benchmark package layer above
`psionic-eval`, `psionic-mlx-lm`, and `psionic-mlx-vlm`.

It owns:

- machine-readable benchmark suite manifests above `BenchmarkPackage`
- repeated benchmark receipts that reuse `psionic-eval`
- local text and served provider adapters for MLX-class lanes
- multimodal prompt projection through `psionic-mlx-vlm`
- bounded text and structured-output scoring contracts for local or private
  eval suites

It does not own:

- a second eval runtime
- benchmark registry authority
- benchmark transport or hosted execution authority
- a native multimodal encoder runtime

## Current Truth

The package is a benchmark and adapter layer, not a second scoring stack.

It reuses:

- `psionic-eval` for benchmark packages, eval runs, verification facts, and
  repeat-run aggregation
- `psionic-mlx-lm` for local text request shaping
- `psionic-mlx-vlm` for multimodal prompt projection and served-request
  planning

That keeps MLX-facing benchmark work tied to the same receipts and aggregate
truth as the rest of Psionic.

## Provider Adapters

`psionic-mlx-bench` now exposes two bounded adapter families:

- a local text adapter that renders prompt messages into one deterministic text
  request before calling the caller-owned local generation lane
- a served adapter that emits machine-readable JSON requests for
  `/v1/responses` or `/v1/chat/completions` so caller-owned harnesses can score
  local or private served lanes

Both adapters return the same provider-response contract, which carries output
text or structured data, token accounting, timer facts, final-state digests,
execution-strategy facts, and optional artifacts.

## Multimodal Posture

Multimodal benchmark cases stay bounded to prompt projection.

`psionic-mlx-bench` does not claim a native image, audio, or video encoder.
Instead it reuses `psionic-mlx-vlm` to turn multimodal inputs into digest-bound
projection receipts plus either:

- one local text prompt for the local adapter
- one served JSON request for the served adapter

## Scoring And Receipts

The package currently ships bounded expectation families for:

- exact text matches
- required-fragment text matches
- exact structured JSON matches

Those expectations produce normal `psionic-eval` sample metrics, artifacts,
verification facts, finalized eval runs, and repeated aggregate summaries.

That means benchmark receipts remain machine-legible and comparable with the
rest of the repo instead of becoming notebook-only side effects.

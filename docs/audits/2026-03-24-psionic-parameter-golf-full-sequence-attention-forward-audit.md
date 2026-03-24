# Psionic Parameter Golf Full-Sequence Attention Forward Audit

> Status: written 2026-03-24 after landing the bounded full-sequence causal
> attention forward path in
> `crates/psionic-backend-cuda/src/kernels/quantized_matvec.cu`,
> `crates/psionic-backend-cuda/src/lib.rs`, and rerunning the local validation
> runtime comparison harness on the RTX 4080 node.

## Summary

The admitted Parameter Golf forward path no longer runs causal self-attention
through the old host-orchestrated token loop.

The CUDA backend now executes the admitted rank-4 grouped-query attention lane
through one full-sequence causal kernel for both contiguous `f32` and
contiguous `bf16` query, key, value, and output tensors.

This changes the current honest bottleneck split:

- the admitted forward runtime is no longer dominated by the per-position
  `attention_decode(...)` host loop
- the remaining dominant train-path slowness still lives in backward replay
  plus the later validation-structure issues tracked separately

## What Changed

`execute_scaled_dot_product_attention_step(...)` in
`crates/psionic-backend-cuda/src/lib.rs` now dispatches admitted Parameter Golf
attention shapes to the new CUDA sequence kernel instead of:

- packing one token into scratch
- calling `attention_decode(...)`
- scattering one token back out
- copying one cache row per position

The new bounded kernel keeps the existing public boundary explicit:

- causal only
- rank-4 only
- grouped-query head posture only
- matching batch and sequence extents only
- sequence lengths bounded by the existing public max
- baseline `1/sqrt(head_dim)` scale only

Unsupported shapes still fail explicitly instead of silently widening support.

## Local Evidence

The pre-change committed bounded validation comparison receipt in
`fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison.json`
recorded:

- legacy average batch time: `7574.50 ms`
- device-resident average batch time: `7392.00 ms`

After the full-sequence attention kernel landed, the same bounded comparison
harness on the same RTX 4080 node produced
`fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison_sequence_attention.json`
with:

- legacy average batch time: `3439.00 ms`
- device-resident average batch time: `3253.50 ms`

That means the bounded same-node validation comparison moved by:

- `7574.50 ms -> 3439.00 ms` on the legacy lane
- `7392.00 ms -> 3253.50 ms` on the device-resident lane

This is large enough to attribute the improvement to the forward runtime
change, not to receipt noise.

## What This Proves

- the admitted Parameter Golf forward hot path is no longer using the old
  per-position decode loop
- the forward runtime improvement is measurable on the local CUDA node even
  before a new H100 rerun
- the eval-stack work that follows now has a cleaner substrate to build on

## Fresh H100 Evidence

The next same-node H100 rerun is now committed too.

The bounded H100 validation comparison in
`fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison_sequence_attention_h100.json`
records:

- legacy average batch time: `9219.00 ms`
- device-resident eval-graph average batch time: `9048.00 ms`
- runtime receipt:
  - `path=device_resident_cuda_eval_graph_v1`
  - `graph_surface=parameter_golf_baseline_eval_graph_v1`

That H100 receipt is narrower than the full end-to-end trainer run, but it
does prove the admitted validation forward path is now executing the eval graph
through the real full-sequence attention kernel on the current single-H100
lane.

The live profiled H100 trainer rerun on current `main` also moved the first
real microstep forward wallclock from `126979 ms` down to `101823 ms` while
keeping forward host fallback to a non-dominant `159 ms` `permute` receipt on
the same bounded lane.

## What This Still Does Not Prove

This audit still does not prove:

- the final single-H100 validation wallclock after all validation-structure
  work lands
- that the remaining train-path blocker is gone, because
  `scaled_dot_product_attention_query_backward` still dominates the backward
  host-fallback receipt
- anything about the later distributed `8xH100` lane

Those stay with `#470`, `#454`, `#458`, and `#473`.

# Psionic Parameter Golf Single-H100 Host-Fallback Profile Audit

> Status: bounded H100 profile note updated on 2026-03-23 after preserving the
> last full micro-step fallback receipt from the real Rust-only trainer and a
> same-node forward before/after comparison on a fresh RunPod `NVIDIA H100 NVL`
> after the layout-transform parallelization landed.

## Scope

This audit records two complementary H100 fallback receipts for the real
`parameter_golf_single_h100_train` path:

- the last full micro-step receipt captured before the layout-transform
  parallelization landed, so the backward blocker family remains explicit
- one same-node forward before/after comparison after three runtime fixes
  landed:
  - zero-copy CUDA liveness retention across execution plans
  - zero-copy aliasing for `detach` and `reshape`
  - chunked parallel host fallback for `expand`, `permute`, and `reduce_sum`

It is not an end-to-end training claim.

The raw machine-readable receipts live at:

- `fixtures/parameter_golf/reports/parameter_golf_single_h100_host_fallback_profile_forward_before_after.jsonl`
- `fixtures/parameter_golf/reports/parameter_golf_single_h100_host_fallback_profile_full_microstep_pre_layout_parallelization.jsonl`

## What Changed

The new same-node comparison was captured on one fresh single-H100 RunPod pod
by stopping the trainer after the first forward host-fallback receipt before
and after commit `9c6e7b80`.

On the same H100 node, the first forward fallback receipt moved from:

- `total_host_fallback_ms = 158060` to `20667`
- `expand = 85344 ms` to `10244 ms`
- `permute = 72716 ms` to `10423 ms`

That is approximately:

- `86.9%` lower total forward host-fallback time
- `88.0%` lower `expand` time
- `85.7%` lower `permute` time

## Fresh H100 Fallback Receipt

The same-node forward before/after comparison recorded:

- before layout-transform parallelization:
  - `total_host_fallback_ms = 158060`
  - `expand = 85344 ms`
  - `permute = 72716 ms`
- after layout-transform parallelization:
  - `total_host_fallback_ms = 20667`
  - `expand = 10244 ms`
  - `permute = 10423 ms`

The last full micro-step receipt preserved the remaining backward replay family:

- `permute = 152234 ms`
- `reduce_sum = 32010 ms`
- `scaled_dot_product_attention_query_backward = 35849 ms`
- `scaled_dot_product_attention_key_backward = 34861 ms`
- `scaled_dot_product_attention_value_backward = 35538 ms`
- `rotary_embedding_backward = 4463 ms`

The important narrowed conclusion is:

- forward `expand` and `permute` are still present, but no longer dominate at
  the same order of magnitude after the chunked parallel layout path landed
- `reshape` is no longer a measured blocker on this path after the zero-copy
  alias change
- `detach` is also no longer part of the measured fallback cost
- the practical remaining dominant host fallback costs are now the backward
  replay family: `permute`, `reduce_sum`, the attention backward family, and
  `rotary_embedding_backward`

## Honest Boundary

This does not close `#454`.

The trainer still has to finish one full optimizer step and emit final
`val_loss` / `val_bpb` to close the single-H100 issue honestly.

It does mean `#455` now has a same-node H100 before/after comparison showing a
material reduction in the dominant forward view-op costs, while keeping the
remaining backward blocker list explicit instead of implying speed closure.

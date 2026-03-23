# Psionic Parameter Golf Single-H100 Host-Fallback Profile Audit

> Status: bounded H100 profile note written on 2026-03-23 after running the
> Rust-only single-H100 trainer on one RunPod `NVIDIA H100 NVL` with the
> opt-in CUDA host-fallback profile sink enabled and stopping after the first
> completed micro-step.

## Scope

This audit records one fresh H100 fallback profile for the real
`parameter_golf_single_h100_train` path after two runtime fixes landed:

- zero-copy CUDA liveness retention across execution plans
- zero-copy aliasing for `detach` and `reshape`

It is not an end-to-end training claim.

The raw machine-readable receipt lives at:

- `fixtures/parameter_golf/reports/parameter_golf_single_h100_host_fallback_profile.jsonl`

## What Changed

The fresh profiled run completed one real micro-step and then terminated
intentionally so the fallback receipt could be inspected without paying the
full validation cost on the current slow path.

Compared with the immediately preceding unprofiled H100 micro-step on the same
trainer path:

- forward wall time improved from `300676 ms` to `250265 ms`
- backward wall time improved from `416203 ms` to `370247 ms`
- host gradient materialization stayed negligible at about `726 ms`

That is approximately:

- `16.7%` faster forward
- `11.0%` faster backward

## Fresh H100 Fallback Receipt

The forward retained-graph replay recorded:

- `expand = 65136 ms`
- `permute = 47219 ms`

The backward replay recorded:

- `permute = 152234 ms`
- `reduce_sum = 32010 ms`
- `scaled_dot_product_attention_query_backward = 35849 ms`
- `scaled_dot_product_attention_key_backward = 34861 ms`
- `scaled_dot_product_attention_value_backward = 35538 ms`
- `rotary_embedding_backward = 4463 ms`

The important narrowed conclusion is:

- `reshape` is no longer a measured blocker on this path after the zero-copy
  alias change
- `detach` is also no longer part of the measured fallback cost
- the remaining dominant host fallback costs are now `expand`, `permute`, the
  attention backward family, and `reduce_sum`

## Honest Boundary

This does not close `#454`.

The trainer still has to finish one full optimizer step and emit final
`val_loss` / `val_bpb` to close the single-H100 issue honestly.

It does mean `#455` now has one real H100 fallback receipt and one concrete
runtime improvement, with the remaining blocker list narrowed to the ops above
instead of the older broader "view-op cost" wording.

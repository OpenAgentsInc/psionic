# Psionic Parameter Golf CUDA Training Coverage

> Status: canonical `PGOLF-303` / `#171` CUDA-training coverage record,
> updated 2026-03-19 after landing the machine-readable Parameter Golf CUDA
> coverage report in
> `crates/psionic-train/src/parameter_golf_cuda_coverage.rs`.

This document records the current honest CUDA training posture for the
Parameter Golf lane.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfCudaTrainingFamily`
- `ParameterGolfCudaTrainingCoverageStatus`
- `ParameterGolfCudaTrainingCoverageCase`
- `ParameterGolfCudaTrainingCapabilityReport`
- `builtin_parameter_golf_cuda_training_capability_report()`
- `challenge_readiness_refusal()`

The distributed `8xH100` receipt lane now also carries:

- `training_capability_report_digest`
- `challenge_kernel_blockers`
- `boundary_notes` derived from the same typed coverage cases

The public CUDA execution backend now also owns two real forward-surface
widens needed by the baseline decoder lane:

- dense contiguous `f32` pointwise `mul`
- backend-specialized dense contiguous `f32` `rms_norm` forward execution plus
  declared runtime extension support

`psionic-ir` now also owns one real train-visible RMSNorm graph seam above the
backend:

- dense `f32` reference evaluation for `rms_norm`
- bounded reverse-mode support for `rms_norm`, with explicit continued refusal
  for the other backend-extension families

The public CUDA backend now also executes the bounded RMSNorm backward graph
ops introduced by that autodiff seam, so RMSNorm itself is no longer part of
the remaining blocker set.

The public CUDA backend now also executes one bounded residual-mix training
graph directly through dense `add`/`mul` plus CUDA backward graphs when the
residual-control tensors are already materialized to activation shape, so the
explicit residual-mix blocker is also retired.

The public CUDA backend now also executes one bounded dense `f32`
non-interleaved RoPE plus causal grouped-query decoder block through a real
full-sequence causal attention kernel for the admitted Parameter Golf shapes,
so the old "no public decoder-block execution" gap is no longer the honest
attention blocker.

`psionic-ir` now also owns one bounded decoder reverse-mode seam above that
forward runtime:

- dense `f32` reference evaluation for RoPE and grouped-query attention
- bounded reverse-mode lowering through dedicated
  `rotary_embedding_backward` and
  `scaled_dot_product_attention_{query,key,value}_backward` ops
- explicit continued refusal for RoPE table gradients, which are not part of
  the Parameter Golf trainer lane

That narrows the attention blocker from "no public decoder reverse-mode" to the
remaining public CUDA backward-runtime gap for that decoder block.

The public CUDA backend now also owns one bounded BF16 runtime seam needed by
the baseline mixed-precision lane:

- dense BF16 buffer residency on the public CUDA dense surface
- bounded row-major BF16xBF16-to-F32 matmul execution through cuBLAS
- bounded Parameter Golf token-embedding forward and backward admission with
  BF16 train-visible embedding tables while preserving the current F32
  embedding-activation and grad-output posture

That first BF16 runtime seam is now followed by one bounded public CUDA BF16
master-weight optimizer step over train-visible BF16 parameter and gradient
buffers:

- train-visible parameter buffers are staged through CUDA `bf16` storage
- train-visible gradient buffers are staged through CUDA `bf16` storage
- optimizer math runs through the shared FP32 reusable optimizer surface on
  FP32 master weights and FP32 optimizer state
- updated train-visible parameter values are re-materialized through CUDA
  `bf16` storage instead of being implied

That retires the last family-level BF16 blocker from the canonical coverage
report while keeping the mixed-precision boundary honest.

`psionic-train` now also owns one bounded public CUDA Muon step over the same
matrix-shaped parameter groups used by the baseline optimizer split:

- exact momentum and scale-correction semantics stay shared with the CPU
  reference step
- the Newton-Schulz BF16 matmul family now runs through the public CUDA dense
  surface
- transpose, norm, and scalar orchestration stay explicit in Rust instead of
  being hidden behind a fake fused-kernel claim

That retires the explicit "no public CUDA Muon path" blocker while preserving
the boundary that the current step is still host-orchestrated around CUDA
matmuls.

That means the canonical CUDA train-path blocker list is now empty on the same
benchmark seam that already carries topology, communication, wallclock, and
memory facts.

## Covered Requirement Families

The report now keeps the following families explicit:

- BF16 train precision posture
- RoPE plus GQA attention block support
- RMSNorm train-path support
- residual or residual-mix train-path support
- Muon optimizer support on CUDA
- post-train int8 plus zlib export or roundtrip support

The current canonical blocker set is empty.

That statement is about family-level runtime coverage, not contest-speed
closure.

Fresh H100 fallback profiling on 2026-03-23 still found substantial
host-executed cost on the real single-H100 trainer path, recorded in:

- `fixtures/parameter_golf/reports/parameter_golf_single_h100_host_fallback_profile_forward_before_after.jsonl`
- `fixtures/parameter_golf/reports/parameter_golf_single_h100_host_fallback_profile_full_microstep_pre_layout_parallelization.jsonl`
- `docs/audits/2026-03-23-psionic-parameter-golf-single-h100-host-fallback-profile-audit.md`

The same-node forward comparison after the chunked layout-transform path landed
reduced:

- total forward host fallback from `158060 ms` to `20667 ms`
- `expand` from `85344 ms` to `10244 ms`
- `permute` from `72716 ms` to `10423 ms`

The fresh narrowed blocker list is now:

- backward replay: `scaled_dot_product_attention_{query,key,value}_backward`

Forward `expand` and `permute` still exist on the fallback surface, but they no
longer dominate at the same order of magnitude after the parallel layout path
landed. The same receipts also keep `reshape` and `detach` retired from the
measured fallback cost after the zero-copy alias path landed for those ops.

A local 2026-03-24 follow-on validation slice on an RTX 4080 also tightened
the same bounded public lane before the next H100 rerun:

- `rotary_embedding_backward` now runs through one bounded CUDA kernel instead
  of forcing a full host readback and replay path
- a fresh local profiled decoder-backward run on the RTX 4080 now reports only
  one replayed `scaled_dot_product_attention_query_backward` label on the
  fallback surface after sibling query/key/value gradient reuse collapsed the
  duplicate replays; `rotary_embedding_backward` is absent from the receipt
- `scaled_dot_product_attention_{query,key,value}_backward` host replay now
  reuses per-position gradient-weight scratch instead of recomputing the same
  inner products twice
- one bounded CUDA causal full-sequence attention-backward kernel now writes
  the query, key, and value gradients for the admitted `f32` PGOLF shapes
  directly on-device
- a fresh local profiled decoder-backward run on the RTX 4080 now leaves the
  fallback profile sink empty, so the bounded decoder backward path no longer
  touches host fallback at all on that lane
- CUDA plan validation now admits the host-fallback or alias view family
  (`reshape`, `detach`, `permute`, `slice`, `select`, `concat`, `expand`,
  `reduce_sum`) that the runtime already executes on the bounded lane
- the CUDA dense allocator now truly admits `I32` buffers, so the PGOLF token
  id and target id path is real instead of only implied by helper names
- bounded CUDA kernels now execute the two hottest PGOLF `permute` shapes
  directly on-device:
  - rank-2 transpose
  - decoder-attention `[0, 2, 1, 3]` layout swap
- bounded CUDA kernels now also execute the hottest PGOLF `reduce_sum` shapes
  directly on-device:
  - full reduction
  - first-axis reduction
  - last-axis reduction
  - rank-3 middle-axis reduction
- bounded CUDA kernels now also execute the dominant PGOLF contiguous `expand`
  broadcasts directly on-device:
  - rank-3 `[1, 1, model_dim] -> [batch, seq, model_dim]`
  - rank-4 `[1, num_heads, 1, 1] -> [batch, num_heads, seq, head_dim]`
- bounded CUDA kernels now also execute the admitted PGOLF full-sequence
  causal grouped-query attention forward lane directly on-device for both:
  - contiguous `f32` rank-4 query, key, value, and output tensors
  - contiguous `bf16` rank-4 query, key, value, and output tensors
- the old host-orchestrated per-position `attention_decode(...)` loop is no
  longer the forward runtime for the admitted Parameter Golf attention shapes
- a fresh same-node bounded validation receipt on the RTX 4080 moved average
  device-resident validation batch time from `7392.00 ms` in
  `fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison.json`
  down to `3253.50 ms` in
  `fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison_sequence_attention.json`
- a fresh same-node H100 validation receipt now also records the admitted
  forward lane on the real single-H100 pod in
  `fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison_sequence_attention_h100.json`:
  - legacy average batch time: `9219.00 ms`
  - device-resident eval-graph average batch time: `9048.00 ms`
  - runtime receipt: `path=device_resident_cuda_eval_graph_v1`,
    `graph_surface=parameter_golf_baseline_eval_graph_v1`
- bounded PGOLF residual `add`/`mul` execution now preserves the IR broadcast
  contract when exact CUDA input specs do not match:
  - exact dense `f32` peers still encode directly on-device
  - bounded broadcast or view-shaped peers now materialize through the profiled
    host-fallback lane instead of aborting the trainer on a spec-mismatch
- fresh local profiled permute and reduce-sum execution tests on the RTX 4080
  leave the fallback profile sink at `0` bytes, so those bounded shapes no
  longer touch the host-fallback surface at all
- the CUDA host-fallback profile sink now emits
  `psionic_cuda_host_fallback_profile_v2`, which preserves per-op totals plus
  the concrete op-detail, output-shape, and output-dtype cases inside each
  fallback family for later H100 reruns

That 4080 slice is not an H100 benchmark receipt and does not close `#470` by
itself, but it does mean the next H100 profile will measure the narrowed
fallback family against the current runtime truth rather than against stale
validator or integer-buffer failures.

## Current Honest Boundary

The report is intentionally not a fake green badge.

Today it keeps these truths separate:

- `implemented_early`
  - bounded dense contiguous `f32` RMSNorm forward plus backward graph
    execution is real on the public CUDA path
  - one bounded full-shape residual-mix train graph is real on the public CUDA
    path
  - bounded dense `f32` decoder reverse-mode graph semantics are real on the
    reference path for non-interleaved RoPE plus causal grouped-query
    attention
  - one bounded full-sequence causal grouped-query CUDA attention forward path
    is real on the public lane for admitted `f32` and `bf16` Parameter Golf
    shapes
  - one bounded CUDA decoder backward kernel is real on the public lane for
    `rotary_embedding_backward`
  - one bounded CUDA decoder backward kernel is also real on the public lane
    for `scaled_dot_product_attention_{query,key,value}_backward` on admitted
    `f32` Parameter Golf shapes
  - one bounded host-orchestrated CUDA BF16 master-weight optimizer step over
    BF16 train-visible parameter and gradient buffers with FP32 master weights
    and FP32 optimizer state is real on the public lane
  - one bounded host-orchestrated CUDA Muon step is real on the public lane
  - one bounded device-resident single-H100 train runner now keeps the stable
    Parameter Golf graph-input weight surface resident across repeated train
    batches for one admitted batch shape and refreshes those buffers once per
    optimizer step instead of rebuilding full host-side graph inputs every batch
  - post-train quantized export or roundtrip support is real

This is the intended contract for the issue: do not hide missing CUDA kernels
behind broader model or distributed receipts.

The distributed receipt now links back to this exact coverage report by
digest, and that keeps the `8xH100` lane reviewable without pretending the
public CUDA surface is already a fused, fully device-resident, or
challenge-speed trainer.

## Why This Matters

Without this report, the repo could say all of these misleading things:

- the `8xH100` receipt lane means the CUDA train path is already broad enough
- an IR or meta-program proof means the direct CUDA kernel exists
- one forward CUDA surface widening means the whole decoder block now trains on
  CUDA
- one bounded decoder reverse-mode graph seam means direct CUDA backward
  execution now exists for that block
- one bounded host-orchestrated CUDA decoder backward path means fused or
  challenge-speed decoder training closure is done
- one bounded RMSNorm closure means the whole decoder block now trains on CUDA
- one bounded full-shape residual-mix graph means generic broadcast or fused
  decoder closure now exists
- one bounded host-orchestrated CUDA BF16 master-weight step means generic
  BF16 graph execution, fused optimizer kernels, or challenge-speed
  mixed-precision closure is done
- one bounded host-orchestrated CUDA Muon step means fused or fully
  device-resident optimizer closure is done
- artifact quantization means train-time low-precision closure is done

The new report prevents that. It turns the remaining CUDA blockers into one
stable typed contract that runtime or backend work can retire without
over-claiming the full trainer.

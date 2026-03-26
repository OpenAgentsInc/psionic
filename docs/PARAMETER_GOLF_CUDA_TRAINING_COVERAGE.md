# Psionic Parameter Golf CUDA Training Coverage

> Status: canonical `PGOLF-303` / `#171` CUDA-training coverage record,
> updated 2026-03-26 after landing the same-node H100 banked-attribution
> receipts and tightening the measured wallclock boundary language.

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
- bounded on-device `f32 -> bf16` cast execution on the public CUDA dense
  surface, so later PGOLF graph slices can lower explicit BF16 boundaries
  without forcing host replay
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

The public CUDA dense surface now also owns the first transpose-aware
strided-batched cuBLAS matmul lane needed by the score-path attention rewrite:

- bounded row-major `bf16 x bf16 -> f32` strided-batched GEMM with explicit
  transpose control on the row-major logical matrices plus explicit
  left/right/output batch strides, so the score lane can walk real
  `[batch, head, seq, dim]` and GQA-owned KV layouts without repacking them
- bounded row-major `f32 x bf16 -> f32` batched GEMM helper with the same
  transpose and stride contract, currently encoded as `f32 -> bf16` staging
  plus the real BF16 strided-batched lane because the mixed-type
  strided-batched cuBLAS posture was not admitted on this lane

That does not retire the attention hot-path blocker by itself. It removes one
runtime-surface gap for the `#562` forward replacement work so the next score
lane can encode QK and AV batches without falling back to one-position-at-a-
time kernels or one-GEMM-at-a-time host orchestration.

Current `main` now also owns one direct banked PGOLF train-path seam above the
same CUDA matmul surface:

- the banked PGOLF matrix families (`qo_bank`, `kv_bank`, `mlp_up_bank`,
  `mlp_down_bank`) no longer drop back to graph-level `slice` plus rank-2
  `matmul` when the graph requires gradients
- `psionic-ir` now lowers explicit banked input-gradient and bank-weight-
  gradient ops for `parameter_golf_banked_linear`
- the public CUDA backend executes those backward ops directly, so the banked
  training graph can keep the same rank-3 bank tensor surface across forward,
  backward, and resident training-session execution
- the admitted CUDA banked graph now also casts direct banked-linear inputs
  down to BF16 before those `parameter_golf_banked_linear` sites, so the
  score-lane matrix-input surface no longer stays wide F32 by default when the
  runtime is already on the banked BF16 matmul path
- the direct banked CUDA path now also binds contiguous zero-copy sub-buffers
  over the owned bank slice for forward and input-backward GEMMs, and writes
  banked weight gradients directly into the destination bank slice instead of
  copying a standalone rank-2 slice in and out around every direct banked op

That retires the old "banked storage but split-matrix training execution"
boundary from the admitted CUDA train lane. It does not prove score closure by
itself. Fresh H100 or `8xH100` receipts are still required for the wallclock
claim.

Current `main` now also exposes one explicit same-node H100 attribution seam
above that runtime surface:

- the single-H100 PGOLF trainer and its runtime receipts now carry an explicit
  `matrix_execution_mode` with `direct_banked` as the default score-path lane
  and `split_sliced` as the forced comparator
- that lets the repo retain exact-shape same-node H100 A/B receipts that
  isolate the banked matrix-execution effect instead of inferring it from a
  broader moving target

That attribution seam now has one retained exact-shape H100 proof in
`docs/audits/2026-03-26-psionic-parameter-golf-single-h100-banked-vs-split-audit.md`:

- `matrix_execution_mode=direct_banked` reached `train_runtime_receipt` in
  `478.511s`
- the explicit `matrix_execution_mode=split_sliced` comparator still had no
  `train_runtime_receipt` after `970s`, had only completed `6/8` micro-steps,
  and was terminated after crossing `2x` the direct-banked receipt time

That same-node proof is enough to attribute a material train-step wallclock
reduction to the banked execution path itself.

Current `main` now also moves the admitted BF16 full-sequence PGOLF attention
forward lane onto that batched scorepath:

- bounded BF16 `scaled_dot_product_attention` now computes grouped-query
  `QK^T` through the strided-batched cuBLAS lane over `(batch, kv_head)`
  batches with the query-head group folded into the row dimension
- causal masking and row-softmax now run through one explicit CUDA
  row-softmax kernel over the retained score matrix instead of the old
  per-position scalar-loop forward kernel
- the attended value pass now runs through the same batched GEMM surface as
  `softmax(QK^T) @ V`, followed by the existing on-device `f32 -> bf16` cast
  back onto the graph surface

That retires the old naive full-sequence forward kernel from the admitted BF16
score lane. The compatibility `f32` full-sequence attention lane still exists,
but it is no longer the hot scoreboard path.

Current `main` now also moves the admitted BF16 full-sequence PGOLF attention
backward lane onto the same batched scorepath:

- the BF16 backward lane now recomputes grouped-query `QK^T` through the
  strided-batched cuBLAS surface, derives `P`, forms `dP = dO @ V^T`, applies
  the row-wise softmax Jacobian through one explicit CUDA backward kernel, and
  then computes `dQ`, `dK`, and `dV` through batched GEMMs
- the old `attention_causal_sequence_backward_to_f32_kernel` and its
  atomic-add K/V accumulation are no longer the admitted BF16 scoreboard lane
- the compatibility `f32` backward lane still exists as a bounded fallback
  surface while the score lane stays on the BF16 GEMM-backed path
- one CUDA submission-scoped scratch cache now reuses the large BF16 attention
  score, output, gradient, and staged mixed-matmul scratch buffers across
  attention ops on the same stream instead of retaining a fresh multi-gigabyte
  scratch allocation per layer until commit; this is the memory-surface fix
  that let the exact public-shape same-node H100 train step fit again on
  current `main`

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

That retires the family-level CUDA runtime blockers on the admitted PGOLF train
lane. It does not retire the measured same-node or distributed score-lane
wallclock blockers.

## Covered Requirement Families

The report now keeps the following families explicit:

- BF16 train precision posture
- RoPE plus GQA attention block support
- RMSNorm train-path support
- residual or residual-mix train-path support
- Muon optimizer support on CUDA
- post-train int8 plus zlib export or roundtrip support

The current canonical family-level blocker set is empty.

That statement is about runtime-surface admission only. It does not mean the
live score-path blocker set is empty.

Current measured wallclock blockers remain open and are retained in separate
receipts and issues:

- same-node exact public-shape H100:
  `matrix_execution_mode=direct_banked` reached `train_runtime_receipt` in
  `478.511s` in
  `docs/audits/2026-03-26-psionic-parameter-golf-single-h100-banked-vs-split-audit.md`,
  and the later bank-offset experiment regressed to `504.978s` in
  `docs/audits/2026-03-26-psionic-parameter-golf-single-h100-banked-offset-regression-audit.md`;
  `#546` remains open because the admitted CUDA forward path is still the
  dominant catastrophic wallclock source on the exact public shapes
- real `8xH100` score lane:
  the retained distributed proof still showed step-2 rank-0 timings of about
  `53.3 s` forward, `123.6 s` backward, `7.4 s` gradient sync, and `0.9 s`
  host gradient materialization; `#541`, `#545`, `#550`, and `#552` remain
  open because the public under-10-minute train or validation budgets are
  still unmet on the live distributed lane

## Historical Narrowing That Retired The Old Family Blockers

Fresh H100 fallback profiling on 2026-03-23 still found substantial
host-executed cost on the real single-H100 trainer path at that earlier stage,
recorded in:

- `fixtures/parameter_golf/reports/parameter_golf_single_h100_host_fallback_profile_forward_before_after.jsonl`
- `fixtures/parameter_golf/reports/parameter_golf_single_h100_host_fallback_profile_full_microstep_pre_layout_parallelization.jsonl`
- `docs/audits/2026-03-23-psionic-parameter-golf-single-h100-host-fallback-profile-audit.md`

The same-node forward comparison after the chunked layout-transform path landed
reduced:

- total forward host fallback from `158060 ms` to `20667 ms`
- `expand` from `85344 ms` to `10244 ms`
- `permute` from `72716 ms` to `10423 ms`

At that point the remaining family-level blocker was backward replay:

- `scaled_dot_product_attention_{query,key,value}_backward`

That family-level gap is now retired by the GEMM-backed BF16 attention
backward lane described above.

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
  the query, key, and value gradients directly on-device for both:
  - admitted `f32` PGOLF shapes
  - admitted `bf16` PGOLF shapes through explicit `f32` accumulation followed
    by on-device `f32 -> bf16` casts onto the graph surface
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
- the CUDA training and eval graphs now lower the linear hot path more
  explicitly on CUDA devices:
  - CUDA linear hot-path matmuls now route through the admitted
    `f32 x bf16 -> f32` lane in the lowered graph instead of leaving the whole
    linear surface at pure `f32`
  - reverse-mode autodiff now admits the bounded float-cast family, so the
    graph layer no longer blocks mixed-precision train-path lowering
  - CUDA train and eval now both cast admitted attention `q`, `k`, and `v`
    inputs onto the bounded BF16 full-sequence attention lane before
    `scaled_dot_product_attention`
- the latest real `8xH100` scoreproof still retained the remaining
  training-side runtime bottleneck honestly:
  - wallclock cap hit after only `3` steps
  - step `2` rank-0 timings were about `53.3 s` forward, `123.6 s` backward,
    `7.4 s` gradient sync, and `0.9 s` host gradient materialization
  - that is still far outside the upstream `10 minute` scoreboard bar even
    after the resident worker mesh and resident-parameter refresh slices landed
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
    `f32` and `bf16` Parameter Golf shapes
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

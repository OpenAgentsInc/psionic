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

That means the remaining CUDA train-path blockers are now machine-readable on
the same benchmark seam that already carries topology, communication,
wallclock, and memory facts.

## Covered Requirement Families

The report now keeps the following families explicit:

- BF16 train precision posture
- RoPE plus GQA attention block support
- RMSNorm train-path support
- residual or residual-mix train-path support
- Muon optimizer support on CUDA
- post-train int8 plus zlib export or roundtrip support

The current canonical blocker set is:

- `cuda_bf16_train_precision_contract`
- `cuda_rope_gqa_attention_block`
- `cuda_muon_optimizer_path`

## Current Honest Boundary

The report is intentionally not a fake green badge.

Today it keeps these truths separate:

- `implemented_early`
  - bounded dense contiguous `f32` RMSNorm forward plus backward graph
    execution is real on the public CUDA path
  - one bounded full-shape residual-mix train graph is real on the public CUDA
    path
  - post-train quantized export or roundtrip support is real
- `partial`
  - BF16 policy, attention or RoPE program shape, and Muon semantics all have
    explicit substrate or refusal contracts
  - the public CUDA execution backend now genuinely owns dense `f32`
    pointwise `mul` plus bounded dense contiguous `f32` RMSNorm forward and
    backward execution, plus one bounded residual-mix graph, but the full
    decoder or optimizer train path is still narrower than the Parameter Golf
    challenge lane

This is the intended contract for the issue: do not hide missing CUDA kernels
behind broader model or distributed receipts.

The distributed receipt now links back to this exact blocker list by digest.
That keeps the `8xH100` lane reviewable without pretending the public CUDA
surface is already fully widened.

## Why This Matters

Without this report, the repo could say all of these misleading things:

- the `8xH100` receipt lane means the CUDA train path is already broad enough
- an IR or meta-program proof means the direct CUDA kernel exists
- one forward CUDA surface widening means the whole decoder block now trains on
  CUDA
- one bounded RMSNorm closure means the whole decoder block now trains on CUDA
- one bounded full-shape residual-mix graph means generic broadcast or fused
  decoder closure now exists
- a CPU-reference Muon implementation means the CUDA optimizer surface is done
- artifact quantization means train-time low-precision closure is done

The new report prevents that. It turns the remaining CUDA blockers into one
stable typed contract that later runtime or backend work can actually retire.

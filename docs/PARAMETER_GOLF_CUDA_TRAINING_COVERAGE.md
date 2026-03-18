# Psionic Parameter Golf CUDA Training Coverage

> Status: canonical `PGOLF-303` / `#171` CUDA-training coverage record,
> updated 2026-03-18 after landing the machine-readable Parameter Golf CUDA
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
- `cuda_rms_norm_train_path`
- `cuda_residual_mix_train_path`
- `cuda_muon_optimizer_path`

## Current Honest Boundary

The report is intentionally not a fake green badge.

Today it keeps these truths separate:

- `implemented_early`
  - post-train quantized export or roundtrip support is real
- `partial`
  - BF16 policy, attention or RoPE program shape, RMSNorm semantics, residual
    semantics, and Muon semantics all have explicit substrate or refusal
    contracts, but the public CUDA train path is still narrower than the full
    Parameter Golf challenge lane

This is the intended contract for the issue: do not hide missing CUDA kernels
behind broader model or distributed receipts.

The distributed receipt now links back to this exact blocker list by digest.
That keeps the `8xH100` lane reviewable without pretending the public CUDA
surface is already fully widened.

## Why This Matters

Without this report, the repo could say all of these misleading things:

- the `8xH100` receipt lane means the CUDA train path is already broad enough
- an IR or meta-program proof means the direct CUDA kernel exists
- a CPU-reference Muon implementation means the CUDA optimizer surface is done
- artifact quantization means train-time low-precision closure is done

The new report prevents that. It turns the remaining CUDA blockers into one
stable typed contract that later runtime or backend work can actually retire.

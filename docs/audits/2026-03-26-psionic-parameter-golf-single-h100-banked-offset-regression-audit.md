# Psionic Parameter Golf Single-H100 Banked-Offset Regression Audit

> Date: 2026-03-26
> Scope: same-node H100 check of one offset-aware banked-linear CUDA experiment
> against the current direct-banked single-H100 `#546` baseline.

## Summary

This audit retains one exact public-shape same-node H100 result for an
offset-aware banked-linear experiment that was **not** landed on `main`.

The tested code path tried to avoid copying `bf16` bank slices for direct banked
PGOLF linears by:

- adding offset-aware CUDA GEMM entrypoints over bank storage
- replacing the forward-path bank-slice copy with an `f32 -> bf16` lhs cast plus
  direct `bf16` bank-offset GEMM
- writing the banked weight-gradient result directly into the destination bank
  slice

That posture regressed the exact public-shape same-node H100 train step instead
of improving it.

## Exact H100 Result

- proof pod: RunPod `NVIDIA H100 80GB HBM3`
- proof tree:
  `/root/psionic-h100-proof-546-clean`
- binary:
  `/root/psionic-target-proof/release/parameter_golf_single_h100_train`
- run root:
  `/workspace/parameter-golf-runpod/pgolf-546-offset-proof`
- log:
  `/tmp/pgolf_546_offset_run.log`
- bounded command:
  `/root/psionic-target-proof/release/parameter_golf_single_h100_train /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model <report> 1 roundtrip_only non_overlapping`

Retained train-step facts:

- `matrix_execution_mode=direct_banked`
- `train_runtime_receipt` reached in `504.978s`
- `micro_step_count=8`
- final cumulative `forward_ms=386084`
- final cumulative `backward_ms=113097`
- final cumulative `host_materialization_ms=613`
- final cumulative `retained_binding_f32=36642059168`
- `resident_parameter_upload_us=33684`

The trainer was interrupted after the train-step receipt and the early
int8-zlib validation batches. No final JSON report was retained for this audit.

## Baseline Comparison

The retained direct-banked same-node H100 baseline on current `main` is:

- `docs/audits/2026-03-26-psionic-parameter-golf-single-h100-banked-vs-split-audit.md`
- `train_runtime_receipt=478.511s`

Relative to that retained baseline, this offset-aware banked-linear experiment
was slower by:

- `26.467s`
- `1.055x`

The final cumulative backward and host-materialization counters stayed close to
the earlier retained same-node H100 posture. The regression showed up in the
forward lane.

## Interpretation

This experiment is useful because it narrows one tempting but wrong direction:

- avoiding the `bf16` bank-slice copy by casting `f32` lhs activations into a
  fresh `bf16` scratch per banked forward matmul is a worse trade on the exact
  public-shape single-H100 lane
- the attempted direct mixed `f32 x bf16 -> f32` cuBLAS offset path still
  rejected real bank-offset or transpose postures with `cublasGemmEx` status
  `15`, so it could not replace the fallback safely
- current `#546` wallclock is still dominated by the admitted CUDA forward
  path, not by host materialization

## Conclusion

Do not land this offset-aware banked-linear experiment.

It does not close `#546`. The useful outcome is narrower:

- the repo now has one exact same-node H100 proof that this bank-offset rewrite
  is a regression
- the next `#546` slice should stay focused on the admitted forward CUDA path
  rather than on replacing the existing bank-slice copy with lhs-cast scratch

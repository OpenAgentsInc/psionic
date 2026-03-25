# Cross-Backend CUDA Plus MLX Dense Mesh Reference

> Status: canonical `XTRAIN-21` / `#537` record, updated 2026-03-25 after
> landing the first shared CUDA-plus-MLX dense mesh contract in
> `crates/psionic-train/src/cross_backend_cuda_mlx_dense_mesh.rs`.

This document records the first explicit mixed CUDA-plus-MLX dense mesh math
contract inside `psionic-train`.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-cross-backend-cuda-mlx-dense-mesh-contract.sh
```

## What Landed

`psionic-train` now owns one shared CUDA-plus-MLX dense mesh contract with:

- one typed backend participant set above the generic CUDA dense runtime and
  the bounded MLX dense-rank runtime
- one explicit fp32 gradient all-reduce contract
- one explicit fp32 master-weight broadcast contract
- one explicit fp32-only shared precision policy
- one explicit mirrored fp32 AdamW optimizer and master-weight ownership law
- one explicit refusal set for BF16 mixed precision, fp16 loss scaling, direct
  NCCL participation by MLX ranks, split master-weight authority, and
  checkpointless optimizer migration
- the binary `cross_backend_cuda_mlx_dense_mesh_contract`
- the checker `scripts/check-cross-backend-cuda-mlx-dense-mesh-contract.sh`
- the fixture `fixtures/training/cross_backend_cuda_mlx_dense_mesh_contract_v1.json`

## Current Honest Boundary

This issue freezes the shared math and ownership law. It does not claim:

- a real same-job mixed-backend proof run
- mixed-backend checkpoint portability
- BF16 or fp16 mixed precision across CUDA and MLX
- direct NCCL participation by MLX Metal ranks
- sharded mixed-backend optimizer ownership

The first shared mixed-backend contract stays on one explicit fp32 reference law
until the checkpoint and proof-run issues land.

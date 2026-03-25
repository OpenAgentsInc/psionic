# First Same-Job MLX Plus CUDA Dense Run Audit

Date: March 25, 2026

## Summary

This audit records the first bounded same-job dense pretraining proof run that
spans one MLX-backed Mac rank and one CUDA-backed dense participant under the
shared mixed-backend contracts.

Run id: `psion-xprovider-pretrain-mixed-backend-20260325`

Retained machine-legible bundle:

- `fixtures/training/first_same_job_mixed_backend_dense_run_v1.json`

## What The Run Did

The retained proof run used one shared run id and one shared dense execution
class across two backend families:

- one local Mac MLX Metal rank admitted through
  `local_mlx_mac_workstation`
- one RunPod CUDA dense participant admitted through
  `runpod_8xh100_dense_node`
- one shared fp32 cross-backend mesh law from
  `fixtures/training/cross_backend_cuda_mlx_dense_mesh_contract_v1.json`
- one shared mixed-backend checkpoint family from
  `fixtures/training/mixed_backend_checkpoint_contract_v1.json`

The proof surface keeps four bounded retained steps:

1. step `4094` steady-state mixed-backend execution
2. step `4095` steady-state mixed-backend execution
3. step `4096` durable mixed-backend checkpoint barrier
4. step `4097` resumed mixed-backend execution under the same run id

The bundle also retains:

- the admitted source bindings for the local MLX rank and the RunPod CUDA
  dense participant
- the participant rank layout for a `9`-rank world
- inline step metrics for the CUDA submesh, MLX rank, bridge collective, and
  optimizer step
- one explicit checkpoint barrier and resume event
- the shared provider-neutral execution-evidence bundle id and digest

## What This Proved

The repo can now truthfully cite one bounded same-job mixed-backend dense run
that:

- used one run id instead of backend-local side jobs
- admitted one MLX-backed Mac rank and one CUDA dense participant under the
  same cross-provider program manifest
- kept one explicit fp32 gradient and master-weight contract across the CUDA
  and MLX participants
- emitted one mixed-backend durable checkpoint and resumed after that barrier
- retained one machine-legible proof bundle plus this acceptance audit

## What This Did Not Prove

This run still does not prove:

- BF16 or FP16 mixed-backend dense training
- sharded optimizer-state exchange across CUDA and MLX
- elastic mixed-backend world-size growth or shrink during live execution
- same-job dense closure for the local RTX 4080 workstation
- broad production readiness across arbitrary mixed-backend topologies

## Main Constraint

The proof run stays intentionally narrow:

- one MLX rank
- one CUDA participant backed by the existing RunPod dense lane
- one shared fp32-only math and optimizer law
- one durable checkpoint and one post-checkpoint resume step

That is enough to close same-job mixed-backend dense training as a bounded
proof surface. It is not a claim that every local or cross-provider mixed
hardware lane now inherits production-ready dense closure automatically.

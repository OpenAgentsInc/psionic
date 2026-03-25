# Mixed-Backend Checkpoint Reference

> Status: canonical `XTRAIN-22` / `#538` record, updated 2026-03-25 after
> landing the first mixed-backend checkpoint and restore contract in
> `crates/psionic-train/src/mixed_backend_checkpoint_contract.rs`.

This document records the first explicit checkpoint, restore, and optimizer-state
portability contract for a CUDA-plus-MLX dense run.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-mixed-backend-checkpoint-contract.sh
```

## What Landed

`psionic-train` now owns one mixed-backend checkpoint contract with:

- one shared checkpoint manifest and pointer under the canonical pretraining
  checkpoint family
- one CUDA portable state receipt and one MLX portable state receipt
- one shared safetensors-backed fp32 portable state law for parameters and
  optimizer state
- one restore ladder that includes native CUDA resume, native MLX resume,
  CUDA-to-MLX restore, MLX-to-CUDA restore, and one refused BF16 migration
- one explicit refusal set for BF16 optimizer-state migration, quantized
  checkpoint resume, checkpointless migration, and incomplete portable group
  selection
- the binary `mixed_backend_checkpoint_contract`
- the checker `scripts/check-mixed-backend-checkpoint-contract.sh`
- the fixture `fixtures/training/mixed_backend_checkpoint_contract_v1.json`

## Current Honest Boundary

This issue does not claim:

- low-precision mixed-backend portability
- quantized mixed-backend checkpoint interchange
- sharded mixed-backend optimizer ownership
- a real same-job mixed-backend proof run

This issue closes the checkpoint family and restore ladder first. The proof run
still remains a later issue.

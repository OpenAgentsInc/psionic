# Dense-Rank Runtime Reference

> Status: canonical `XTRAIN-4` / `#520` record, updated 2026-03-25 after
> promoting the PGOLF CUDA `8xH100` runtime into one shared dense-rank runtime
> substrate in `crates/psionic-train/src/dense_rank_runtime.rs`.

This document records the first generic dense-rank runtime surface inside
`psionic-train`.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-dense-rank-runtime-reference-contract.sh
```

## What Landed

`psionic-train` now owns one dense-rank runtime layer above the current PGOLF
distributed CUDA lane.

The landed surface includes:

- `DenseRankRuntimeIdentity`
- `DenseRankValidationHookContract`
- `DenseRankCheckpointHookContract`
- `DenseRankRuntimeBootstrapReceipt`
- `DenseRankRuntimeTrainStepReceipt`
- `DenseRankRuntimeExecutionReceipt`
- `ParameterGolfBackedDenseRankRuntimeOutcome`
- the binary `dense_rank_runtime_reference_contract`
- the checker `scripts/check-dense-rank-runtime-reference-contract.sh`
- the canonical fixture `fixtures/training/dense_rank_runtime_reference_contract_v1.json`

## Ownership Boundary

Before this change, the only real dense distributed runtime in the repo was
buried inside PGOLF-specific bootstrap and train-step wrappers.

After this change:

- PGOLF still owns its lane-specific bring-up, bootstrap, and train-step
  receipts
- the shared dense-rank runtime now owns the provider-neutral execution receipt
  family above those lane-specific receipts
- validation and checkpoint behavior now have explicit hook contracts at the
  shared runtime layer instead of remaining implicit PGOLF-only assumptions

The exported-folder distributed execution path now emits both:

- the existing PGOLF lane receipts
- one shared `dense_rank_runtime_execution_receipt_v1.json`

That keeps current contest evidence truthful while giving later pretraining
lanes one reusable dense-rank receipt substrate.

## Current Consumer

The first consumer is still the PGOLF CUDA `8xH100` lane.

The shared runtime receipt explicitly records that fact through the consumer
lane id `parameter_golf.distributed_8xh100`. The reference contract uses the
generic lane id `psion.cross_provider_pretraining_dense_reference` to keep the
runtime family itself separate from any one consumer.

## Current Limits

This issue intentionally does not claim:

- mixed-backend dense training
- distributed checkpoint shard closure
- post-train validation closure
- provider-specific launch or quota automation
- public or adversarial swarm compute

This issue closes the dense-rank runtime ownership boundary first.

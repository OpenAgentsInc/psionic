# Psionic Parameter Golf Distributed 8xH100 Lane

> Status: canonical `PGOLF-302` / `#170` distributed-lane contract, updated
> 2026-03-18 after landing the typed `8xH100` admission, topology,
> communication, wallclock, and memory receipt path in
> `crates/psionic-train/src/parameter_golf_distributed.rs` and
> `crates/psionic-eval/src/parameter_golf_distributed.rs`.

This document freezes the exact distributed execution posture Psionic now uses
for the public Parameter Golf lane.

## Public Baseline Alignment

The public `train_gpt.py` baseline currently implies this distributed shape:

- `WORLD_SIZE=8`
- `grad_accum_steps=1`
- replicated DDP-style training across one `8xH100` CUDA pod
- NCCL-style `all_reduce` for the full BF16 gradient surface
- an additional Muon `all_reduce` over the flattened matrix-update buffer
- `all_reduce` over validation loss, token count, and byte count

Psionic now encodes that exact posture explicitly instead of treating
"distributed closure" as free-form prose.

## What Landed

- `psionic-train` now ships
  `benchmark_parameter_golf_distributed_8xh100(...)`
- `psionic-train` now exposes
  `ParameterGolfDistributed8xH100Config` and
  `ParameterGolfDistributedStepObservation`
- `psionic-eval` now exposes
  `ParameterGolfDistributedThroughputReceipt` plus the supporting topology,
  communication, timing, memory, threshold, and refusal types
- the distributed receipt now also preserves the aligned
  `training_capability_report_digest`, ordered `challenge_kernel_blockers`,
  and derived CUDA boundary notes from the typed `PGOLF-303` coverage report
- the lane now emits either a measured receipt or an explicit refusal with the
  local-reference benchmark preserved as the fallback review lane

## Admission Gate

The distributed lane is admitted only when all of the following are true:

- backend is `cuda`
- the selected inventory contains exactly `8` devices
- every selected device name matches `H100`
- devices are not MIG-partitioned
- the cluster capability profile advertises
  `tensor_collective_mesh`

If any of those checks fail, the lane emits a refusal receipt instead of
pretending the run is comparable.

## Topology And Communication Posture

The landed topology is:

- replicated `8`-way execution topology
- one data-parallel mesh axis `dp` with extent `8`
- loopback or single-pod transport posture with tensor-collective mesh support

The landed communication receipt preserves three concrete stages:

- `ddp_gradient_all_reduce`
- `muon_matrix_update_all_reduce`
- `validation_metric_all_reduce`

These stages mirror the current public Python baseline instead of inventing a
different sharding story.

## Timing And Memory Receipts

The lane now preserves:

- observed per-step timings
- observed validation duration
- observed export or roundtrip duration
- total wallclock versus the declared challenge cap
- analytic optimizer-contract memory facts for parameters, gradients,
  optimizer state, master weights, and activation upper bound

The memory lane is deliberately explicit about being an analytic upper bound,
not a direct CUDA allocator trace.

## Refusal Posture

The typed refusal surface now distinguishes:

- `device_inventory_mismatch`
- `capability_mismatch`
- `measurements_missing`
- `memory_budget_exceeded`
- `wallclock_exceeded`

This is the intended boundary for the current repo: when Psionic cannot defend
the declared `8xH100` bar, it refuses instead of silently falling back.

## Current Honest Boundary

This issue does not claim full challenge closure by itself.

What is now explicit:

- the exact public `8xH100` topology
- the DDP or Muon communication posture
- measured-or-refused timing receipts
- measured-or-refused memory receipts
- the digest and blocker list for the current CUDA train-path coverage report

What is still separate work:

- retiring the explicit blocker list carried by the CUDA training coverage
  report
- widening the public CUDA train path until the decoder-block, precision, and
  optimizer surfaces no longer need those blockers
- broader proof that the public array surface owns every required train-time
  kernel directly rather than through partial IR or semantic evidence

That remaining closure stays with `PGOLF-303` / `#171`.

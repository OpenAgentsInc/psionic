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
  `ParameterGolfDistributed8xH100Config`,
  `ParameterGolfDistributedStepObservation`, and
  `ParameterGolfDistributedMemoryObservation`
- `psionic-train` now also ships the example
  `crates/psionic-train/examples/parameter_golf_distributed_8xh100_receipt.rs`
  so later pod runs can bind JSON-collected device inventory, clustered
  capability profile, and observed timing or memory telemetry directly into the
  typed receipt without another schema pass
- `psionic-train` now also ships
  `ParameterGolfRunPod8xH100Measurements`,
  `benchmark_parameter_golf_runpod_8xh100_from_measurements(...)`, and the
  example `crates/psionic-train/examples/parameter_golf_runpod_8xh100_receipt.rs`
  so a real RunPod run root can be lifted directly from
  `nvidia_smi_inventory.txt` plus one minimal operator-measurement JSON into
  the typed distributed receipt without hand-written intermediate JSON
- `psionic-train` now also ships
  `build_parameter_golf_runpod_8xh100_measurements_from_train_log(...)`, the
  example
  `crates/psionic-train/examples/parameter_golf_runpod_8xh100_measurements_from_log.rs`,
  and the wrapper
  `scripts/parameter-golf-runpod-build-8xh100-measurements.sh` so a retained
  RunPod `execution.log` can be lifted directly into
  `parameter_golf_distributed_8xh100_measurements.json` before the typed
  receipt builder runs
- `psionic-train` now also ships typed distributed validation shard inputs via
  `ParameterGolfDistributedValidationShardObservation`, and the receipt now
  preserves a typed `validation_aggregation` section when rank-local shard
  facts are available
- `psionic-train` now also ships the Rust-owned admission-plus-bootstrap seam
  `ParameterGolfDistributed8xH100BringupConfig`,
  `build_parameter_golf_distributed_8xh100_bringup_report(...)`,
  `write_parameter_golf_distributed_8xh100_bringup_report(...)`, and the CLI
  binary `crates/psionic-train/src/bin/parameter_golf_distributed_8xh100_bringup.rs`
  so the repo can emit one machine-readable `8xH100` admission report and
  optional measured-receipt lift without going through the RunPod finalizer
  path first
- `psionic-train` now also ships the runtime bootstrap contract
  `ParameterGolfDistributed8xH100RuntimeBootstrapReceipt` plus retained
  per-rank bootstrap receipts and logs under the exported-folder distributed
  mode, so a real `8xH100` execution log can now cross from machine admission
  into explicit rank fanout and one runtime-owned mesh/bootstrap contract
  before the later train-step work begins
- `psionic-train` now also ships the runtime-owned one-step execution seam
  `ParameterGolfDistributed8xH100TrainStepReceipt`,
  `ParameterGolfDistributed8xH100TrainStepRankReceipt`, retained per-rank
  train-step windows, retained per-rank train-step gradient artifacts, and one
  measured distributed receipt lift from that step, so the exported-folder
  `distributed_8xh100_train` mode no longer stops at bring-up only
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
- each device still satisfies the `80GB` class after allowing the normal
  `nvidia-smi` inventory delta seen on real RunPod `H100 80GB HBM3` hosts
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
- one real measured train-step receipt from the exported-folder runtime when
  the machine contract is satisfied
- observed validation duration
- optional typed rank-local validation shard receipts plus aggregated
  `loss_sum`, `token_count`, and `byte_count`
- observed export or roundtrip duration
- total wallclock versus the declared challenge cap
- either:
  - observed runtime peak device or host bytes per worker plus the analytic
    logical tensor-state breakdown, when real runtime memory telemetry exists
  - or the older analytic optimizer-contract upper bound when no runtime
    memory telemetry exists yet

The memory lane stays deliberately explicit about which parts are measured and
which parts remain analytic:

- runtime peaks can now be observed directly
- logical parameter, gradient, optimizer-state, master-weight, and activation
  accounting still come from the distributed optimizer contract

When the runtime preserves rank-local validation shard facts, the receipt now
also records:

- one contiguous shard layout across the full validation sequence space
- per-rank `sequence_start`, `sequence_count`, `loss_sum`, `token_count`,
  `byte_count`, and `observed_ms`
- one aggregated `mean_loss` and `bits_per_byte`
- one honest distributed validation wallclock as the slowest participating rank

You can now build one distributed receipt directly from JSON-collected runtime
facts:

```bash
cargo run -p psionic-train --example parameter_golf_distributed_8xh100_receipt -- \
  /tmp/parameter_golf_distributed_devices.json \
  /tmp/parameter_golf_cluster_capability_profile.json \
  /tmp/parameter_golf_distributed_8xh100_config.json \
  /tmp/parameter_golf_distributed_8xh100_receipt.json
```

For the later real RunPod lane, the repo now also owns a one-command
run-root bridge for measurements plus the existing receipt bridge:

```bash
bash scripts/parameter-golf-runpod-build-8xh100-measurements.sh \
  --run-root /workspace/parameter-golf-runpod-8xh100-20260324T000000Z

bash scripts/parameter-golf-runpod-build-8xh100-receipt.sh \
  --run-root /workspace/parameter-golf-runpod-8xh100-20260324T000000Z
```

You can also build one Rust-owned distributed bring-up report directly on the
current machine, with or without one retained measurements JSON:

```bash
cargo run -p psionic-train --bin parameter_golf_distributed_8xh100_bringup -- \
  --output /tmp/parameter_golf_distributed_8xh100_bringup.json

cargo run -p psionic-train --bin parameter_golf_distributed_8xh100_bringup -- \
  --measurements /tmp/parameter_golf_distributed_8xh100_measurements.json \
  --output /tmp/parameter_golf_distributed_8xh100_bringup.json
```

The measurements builder expects:

- `/workspace/.../execution.log`

and emits:

- `/workspace/.../parameter_golf_distributed_8xh100_measurements.json`

The receipt builder then expects:

- `/workspace/.../nvidia_smi_inventory.txt`
- `/workspace/.../parameter_golf_distributed_8xh100_measurements.json`

The measurements JSON preserves only the runtime facts that are not already in
the finalizer-owned run root:

- ordered `step_observations`
- `validation_observed_ms`
- `export_observed_ms`
- optional `memory_observation`
- optional ordered distributed validation shard observations lifted from
  `distributed_validation_rank_complete ...` log lines when the runtime emits
  them

The device inventory and capability profile are derived by Psionic itself from
the run-root inventory contract plus the canonical RunPod `8xH100` lane
assumptions.

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
- the exported-folder runtime can now cross from machine admission into one
  real `WORLD_SIZE=8` train step with retained per-rank receipts, logs,
  windows, gradient artifacts, one aggregate train-step receipt, and one typed
  measured distributed receipt
- measured-or-refused timing receipts
- measured-or-refused memory receipts
- the Rust-owned local bring-up seam for exact `8xH100` admission and
  optional measured-receipt lifting
- the digest and blocker list for the current CUDA train-path coverage report

What is still separate work:

- the later distributed validation path and final challenge-metric aggregation
- retiring the explicit blocker list carried by the CUDA training coverage
  report
- widening the public CUDA train path until the decoder-block, precision, and
  optimizer surfaces no longer need those blockers
- broader proof that the public array surface owns every required train-time
  kernel directly rather than through partial IR or semantic evidence

That remaining closure stays with `PGOLF-303` / `#171`.

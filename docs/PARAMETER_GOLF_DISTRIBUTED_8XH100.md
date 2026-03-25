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
- `psionic-train` now also ships the runtime-owned repeated train-loop seam
  behind `ParameterGolfDistributed8xH100TrainStepReceipt` and
  `ParameterGolfDistributed8xH100TrainStepRankReceipt`; the exported-folder
  `distributed_8xh100_train` mode now launches one persistent `WORLD_SIZE=8`
  worker mesh, keeps model and optimizer state resident inside those workers,
  performs rank-local synchronization over the in-memory
  `rank0_loopback_tcp_non_muon_mean_all_reduce_plus_parallel_muon_reduce_scatter_all_gather_v1`
  transport instead of per-step gradient `safetensors`, emits
  wallclock-bounded step progress lines during execution, and preserves
  ordered `step_observations` plus the honest loop stop reason in the
  aggregate train receipt instead of pretending the final retained step was
  the whole run. The resident score lane now splits optimizer families
  explicitly: non-banked optimizer groups still mean-all-reduce, while the
  rank-3 banked Muon groups now reduce-scatter by owned bank-slice shard
  through rank 0, update locally on the owning rank, and then all-gather the
  updated bank values back across the resident mesh. That ownership and
  reduce-scatter surface is now explicit in `parallel_muon_receipt` instead of
  being hidden behind one generic communication label. The live score lane now
  exports only the final runtime-owned model surfaces after the repeated loop
  instead of using file-artifact gradient handoff in the hot path. The
  repeated worker-mesh loop now also prunes superseded step-scope directories
  once the next step succeeds, so the resident scoreproof lane stops
  accumulating dead per-step receipt directories indefinitely. The
  resident train-session refresh path now also reuses prepacked host `bf16`
  staging for BF16-visible parameter banks instead of repacking those large
  tensors on every optimizer step. The shared CUDA train path now also narrows
  both the backward graph output surface and the retained primal-binding
  surface to the parameter-only backward-live subset, binds graph-input
  primals back into backward directly from the original input buffers, no
  longer counts those input-bound parameter tensors as retained forward
  outputs, and now binds the default PGOLF `relu_squared` backward rule
  against the activation output instead of the pre-activation hidden tensor.
- `psionic-train` now also ships retained per-rank distributed validation
  receipts plus one completion receipt bound to the trained runtime-produced
  int8+zlib artifact, so the exported-folder `distributed_8xh100_train` mode
  can finish with a real runtime outcome when the measured distributed
  validation surface exists; the current validation path now reuses the same
  persistent rank workers instead of respawning a second validation-only child
  fanout, and the coordinator now dispatches validation work to all resident
  ranks before waiting on any one receipt so the retained validation wallclock
  can reflect real parallel rank execution instead of accidental rank-serial
  receipt collection; the resident workers now also preload the fixed
  validation token surface once at startup instead of reopening the full
  validation shards inside every validation call; the non-score-first-TTT
  validation path now also reuses the resident train-session parameter buffers
  directly instead of materializing a second post-step model and reuploading a
  second parameter surface per rank-local validation batch shape
- `psionic-models` now also ships the upstream-style banked PGOLF matrix
  surface under `ParameterGolfBankedWeights`, with the exact public bank tensor
  ids `qo_bank`, `kv_bank`, `mlp_up_bank`, and `mlp_down_bank`; the optimizer
  planner now classifies those `3D` matrix banks as Muon-owned tensors instead
  of forcing the runtime to stay on the fully split per-layer matrix surface,
  and the baseline graph plus trainer state now execute that banked runtime
  descriptor directly instead of treating it as metadata only
- the PGOLF banked forward, eval, and train lanes now lower
  `q/k/v/out/fc/proj` through one explicit `parameter_golf_banked_linear`
  backend extension instead of graph-level `slice` plus rank-2 `matmul`,
  which matters on CUDA because the old bank-slice view path was still
  materializing those rank-3 bank slices through host fallback before the
  matmul; the train lane now also lowers direct banked input-gradient and
  bank-weight-gradient ops instead of bouncing those matrix families back
  through the older sliced-bank backward posture
- `psionic-models` plus the lowered PGOLF baseline graph now also admit two
  architecture-pack slices under the shared config surface:
  `rope_rotary_dim` for Partial RoPE,
  `layer_norm_scale=inverse_sqrt_layer_index_plus_one` for the public
  inverse-sqrt per-layer RMSNorm output scale, and `xsa_last_n` for
  GQA-aware XSA-style self-value orthogonalization on the deepest layers.
  The same shared surface now also admits VE-style late-layer features through
  `ve_dim` plus ordered `ve_layer_indices`, with one shared token-id value
  embedding table, one optional projection into `kv_dim`, one shared scale,
  and one per-layer scale applied before the late-layer value injection into
  `v_proj`. The optimizer planner and lowered graph now classify and bind
  `ve_shared.*` and `ve_layer_scales.*` explicitly instead of treating them as
  unknown tensors.
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

The landed communication receipt now preserves five concrete stages:

- `non_muon_gradient_all_reduce`
- `parallel_muon_gradient_reduce_scatter`
- `parallel_muon_bank_shard_update`
- `parallel_muon_bank_value_all_gather`
- `validation_metric_all_reduce`

These stages describe the current landed Psionic topology directly. The Muon
path is no longer represented as one extra full all-reduce because the
resident runtime now reduce-scatters the owned bank-shard gradients, applies
those shards locally, and then all-gathers the updated bank values.

## Current Validation Contract

The distributed lane now carries an explicit validation-eval contract instead
of treating validation shards as anonymous sequence slices.

The shipped runtime now preserves, validates, and lifts:

- `eval_mode`
- `local_batch_sequences`
- `validation_batch_sequences`
- `evaluation_unit_start`
- `evaluation_unit_count`
- `scored_token_start`
- `scored_token_count`

For the scoreboard-grade lane, Psionic now uses sliding-window validation with:

- `eval_mode=sliding_window_stride_64`
- one global ordered `window_starts` list derived from the full validation
  token stream
- one contiguous partition of those evaluation windows across ranks
- one explicit eval-batch geometry surface that is independent from the train
  batch geometry
- one manifest-backed `validation_batch_sequences` value that the runtime now
  preserves into shard plans and receipts instead of silently re-deriving later
- default `batch_sequences=1024` windows per rank-local forward batch only for
  the sliding-window score path, matching accepted public sliding-window record
  posture instead of the baseline non-overlapping token cap
- optional override through
  `PSIONIC_PARAMETER_GOLF_VALIDATION_BATCH_SEQUENCES=<positive integer>` when
  operator experiments need a different rank-local eval batch size
- scored-token ranges that match the upstream score-only suffix contract

This matters because the distributed receipt can now defend the real eval
surface: which windows each rank evaluated, which scored-token interval each
rank owned, which explicit `validation_batch_sequences` contract was active,
and how the aggregated `loss_sum`, `token_count`, and `byte_count` were reduced
back into one distributed validation result.

For the resident non-TTT path, rank-local validation now binds those eval
graphs against the already refreshed train-session parameter buffers. That
keeps the validation parameter surface on device after each optimizer step and
removes the extra post-step model materialization plus eval-only parameter
upload from the per-rank validation startup path.

The distributed runtime now also admits the same legal `score_first_ttt`
contract that the single-H100 CUDA lane already owned. The worker mesh keeps
global sliding-window shard ownership per rank, scores each score-first chunk
on the subset of windows owned by that rank, then runs the chunk-local
adaptation step on every resident worker so the mesh stays synchronized without
introducing a second optimizer surface. The resulting train-step receipt can
now preserve one aggregated `score_first_ttt_receipt` beside the existing
distributed validation observations. When the resident training-session cache
already exists, that score-first chunk scoring path now reuses the refreshed
resident parameter buffers instead of forcing a second eager eval-only
parameter upload per chunk.

What is still missing is retained real `8xH100` evidence for that path under
the public wallclock bar. The code path exists. The scoreboard proof does not.

## Current Matrix Banking Boundary

Psionic now owns the same four-bank matrix vocabulary cited by the public top
record:

- `qo_bank`
- `kv_bank`
- `mlp_up_bank`
- `mlp_down_bank`

That banked surface is no longer metadata only. The single-H100 and runtime
train-step code now admit:

- banked graph bindings for `q/k/v/out/fc/proj`
- trainer-state seeding and materialization from bank tensor ids
- slice-wise Muon updates over rank-3 bank tensors
- banked safetensors restore back into one explicit `ParameterGolfBankedWeights`
  runtime surface for the distributed train-step child
- direct graph-input binding and resident training-session refresh from that
  explicit banked runtime surface instead of regenerating the bank tensors from
  the split model on every train-step bind

What is still missing is the scoreboard-grade proof:

- no fresh `1xH100` or `8xH100` receipt yet proves that the banked path
  materially improves real train or validation wallclock
- the later score-path issues still need to combine this surface with the
  persistent worker mesh, legal score-first TTT, and the remaining hot-kernel
  work before the repo can claim competitive scoreboard posture

## Timing And Memory Receipts

The lane now preserves:

- observed per-step timings
- one real measured train-step receipt from the exported-folder runtime when
  the machine contract is satisfied
- observed validation duration
- optional typed rank-local validation shard receipts plus aggregated
  `loss_sum`, `token_count`, and `byte_count`
- optional aggregated legal `score_first_ttt_receipt` with ordered chunk
  receipts and adaptation-step facts when the shipped manifest requests it
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

- one explicit `eval_mode`
- one contiguous shard layout across the validation evaluation-unit space
- per-rank `sequence_start`, `sequence_count`
- per-rank `evaluation_unit_start`, `evaluation_unit_count`
- per-rank `scored_token_start`, `scored_token_count`
- per-rank `local_batch_sequences`, `loss_sum`, `token_count`, `byte_count`,
  and `observed_ms`
- optional per-rank validation `runtime_receipt` when the CUDA eval lane emits
  one, so retained validation proofs keep the resident parameter-buffer and
  token-write facts instead of collapsing them into wallclock only
- one aggregated `mean_loss` and `bits_per_byte`
- one honest distributed validation wallclock as the slowest participating rank
- optional legal `score_first_ttt` chunk receipts layered over that same global
  shard layout instead of a second incompatible validation accounting surface

The resident worker train-step receipts now also preserve the rank-local
`runtime_receipt` from the device-resident train runner when that hot path is
active. That makes the retained `8xH100` proof explicit about resident buffer
counts, resident upload cost, parameter-refresh cost, and mutable token-write
cost instead of forcing later audits to infer those facts only from phase
timings.
- one aggregated `total_evaluation_unit_count` beside
  `total_sequence_count`

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
- `validation_eval_mode`
- `validation_batch_sequences`
- `validation_observed_ms`
- `export_observed_ms`
- optional `memory_observation`
- optional ordered distributed validation shard observations lifted from
  `distributed_validation_rank_complete ...` log lines when the runtime emits
  them
- retained per-rank validation receipt files now also preserve the optional
  CUDA eval `runtime_receipt` when present
- optional aggregated `score_first_ttt_receipt` when the resident worker mesh
  executes the legal chunk-ordered overlay

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
- the exported-folder runtime can now cross from machine admission into a real
  repeated `WORLD_SIZE=8` persistent worker mesh with resident train state,
  in-memory rank reduction, retained per-rank receipts and logs, one aggregate
  train-step receipt, and one typed measured distributed receipt
- the later distributed validation path now reuses those same persistent rank
  workers and no longer requires a second spawn-per-pass runtime fanout
- measured-or-refused timing receipts
- measured-or-refused memory receipts
- the Rust-owned local bring-up seam for exact `8xH100` admission and
  optional measured-receipt lifting
- the digest and blocker list for the current CUDA train-path coverage report

What is still separate work:

- proving on a real `8xH100` pod that the persistent worker mesh materially
  lowers train-step and validation wallclock
- replacing the current loopback proof transport with scoreboard-grade device
  collectives for the public lane
- retiring the explicit blocker list carried by the CUDA training coverage
  report
- widening the public CUDA train path until the decoder-block, precision, and
  optimizer surfaces no longer need those blockers
- broader proof that the public array surface owns every required train-time
  kernel directly rather than through partial IR or semantic evidence

That remaining closure stays with `PGOLF-303` / `#171`.

# Actual Pretraining Tri-Host Efficiency Optimization Audit

Date: 2026-04-13

## Goal

Improve the shipped tri-host actual-lane production-candidate canary so the
RTX `4080` contributes more useful work per wall-clock minute and the two
remote contributors stop serializing each other.

The topology stayed the same:

- local M5 MacBook Pro as control plane and CPU contributor
- remote `archlinux` Tailnet host as CUDA contributor on the RTX `4080`
- remote `macbook-pro-m2` Tailnet host as CPU contributor

The workload identity also stayed the same:

- model id: `psion-compact-decoder-internal-v1`
- dataset identity: `psion_corpus_tokenized@v1`

## What Changed

Two concrete changes landed in the shipped actual-lane bringup path.

### Remote contributors now overlap

Before this change, the local operator requested one remote contribution,
waited for it to finish, then requested the second remote contribution.

After this change:

- the `archlinux` CUDA contribution
- the `macbook-pro-m2` CPU contribution
- and the local M5 CPU contribution

all run in the same step window and are joined together before merge.

That change landed in:

- `crates/psionic-train/src/psion_reference_pilot.rs`

### Default batch sizing now matches backend strength

Before this change, the actual-lane distributed bringup gave all three hosts
the same tiny batch-row default:

- local M5 CPU: `2`
- remote RTX `4080`: `2`
- remote M2 CPU: `2`

After this change, the shipped actual-lane bringup defaults are:

- local M5 CPU: `2`
- remote RTX `4080`: `12`
- remote M2 CPU: `2`

That change landed in:

- `crates/psionic-train/examples/psion_distributed_actual_pretraining_bringup.rs`

The same backend-aware posture was mirrored into:

- `crates/psionic-train/examples/psion_distributed_reference_pilot.rs`

## Verification Before Live Run

Focused verification in the clean worktree:

```bash
cargo test -q -p psionic-train --lib remote_joint_contributions_are_collected_concurrently
cargo test -q -p psionic-train --lib actual_pretraining_bringup_runs_tri_host_cpu_merge_end_to_end
cargo build -q -p psionic-train --example psion_distributed_actual_pretraining_bringup
```

The concurrency test proves the two remote requests overlap in the same scoped
execution block instead of serializing.

## Source-Of-Truth Command

```bash
PSION_REFERENCE_PILOT_MAX_STEPS=12 \
PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=3 \
PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=2 \
./TRAIN rehearse-base-lane \
  --run-id psion-actual-pretraining-tri-host-actual-prodcanary-optimized-20260413t101910Z \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote
```

Wrapper timing:

```text
real 1440.84
user 505.15
sys 31.11
```

Run root:

- `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-optimized-20260413t101910Z`

## Baseline Comparison

Previous clean baseline:

- run id:
  `psion-actual-pretraining-tri-host-actual-prodcanary-zstd-clean-20260413t134400Z`
- run root:
  `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-zstd-clean-20260413t134400Z`

Both runs used:

- the same actual-lane tri-host topology
- the same `12` optimizer steps
- the same `3` steps per window
- the same `2` windows per cadence
- the same real workload identity

### Retained before / after

- optimizer steps: `12` -> `12`
- contribution receipt count: `36` -> `36`
- progress checkpoint count: `4` -> `4`
- cumulative train tokens processed: `775` -> `2194`
- retained mean tokens per second: `16` -> `45`

### Real elapsed before / after

From retained train-log timestamps:

- old full elapsed: `2285s`
- new full elapsed: `1421s`
- full elapsed reduction: about `38%`

From distributed launch to cleanup:

- old distributed execution elapsed: `2243s`
- new distributed execution elapsed: `1388s`
- distributed execution reduction: about `38%`

### Why the numbers moved

The improvement is explained by the code change, not by a different workload:

- the two remote contributors no longer wait on each other
- the RTX `4080` now gets a materially larger shard by default

The retained contribution receipts confirm the changed shard allocation:

- baseline first-step batch rows:
  - local CPU `2`
  - RTX `4080` `2`
  - remote M2 CPU `2`
- optimized first-step batch rows:
  - local CPU `2`
  - RTX `4080` `12`
  - remote M2 CPU `2`

## Result

Terminal outcome:

```text
status=base_lane_rehearsal_complete
run_id=psion-actual-pretraining-tri-host-actual-prodcanary-optimized-20260413t101910Z
checkpoint_eval_decision=continue
continue_restart_decision=continue
distributed_topology=multi_host_joint_gradient_average
```

Retained facts:

- topology: `multi_host_joint_gradient_average`
- contributor count: `3`
- worker hosts:
  - `Christophers-MacBook-Pro-2`
  - `archlinux`
  - `macbook-pro-m2`
- runtime backends:
  - `cpu`
  - `cuda`
  - `cpu`
- optimizer steps: `12`
- contribution receipt count: `36`
- progress checkpoint count: `4`
- progress window count: `4`
- progress cadence count: `2`
- final cumulative train tokens processed: `2194`
- final cumulative mean tokens per second: `45`
- accepted checkpoint label: `bounded-actual-pretraining-bringup-step-12`
- checkpoint ref: `psion-reference-pilot-step-12`
- checkpoint object digest:
  `2c301b75caa8aba0463d048593ebb8dc74c7d89e448b47266a8db80df2c7ec22`

## Throughput Claim Boundary

The retained `final_cumulative_mean_tokens_per_second` field is useful inside
the bounded training lane, but it is not the right field for end-to-end
operator wall-clock claims.

For honest canary comparison:

- use retained train-log timestamps or shell timing for elapsed duration
- use retained token/accounting fields for relative model-progress volume

Do not present the retained mean-tokens-per-second field as the literal
wall-clock throughput of the whole operator run.

## Retained Evidence

- `distributed_execution/distributed_actual_pretraining_bringup.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_operator_manifest.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_operator_summary.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_train.log`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_topology_receipt.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_contribution_receipts.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_progress_checkpoint_receipts.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_checkpoint_manifest.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_checkpoint.safetensors`
- `closeout/closeout_bundle.json`
- `status/current_run_status.json`
- `status/retained_summary.json`

## What This Means Now

The shipped actual-lane production-candidate canary is materially better than
the previous clean baseline on the same three-machine shape.

What is now true:

- the 4080 carries a larger default training shard
- the two remotes contribute in parallel instead of serial order
- the same `12`-step actual-lane canary finishes much faster
- the retained model-progress accounting moves up with the faster run

What is still not true:

- this is still a bounded canary, not a continuously live production lane
- first-run remote compile and staging cost is still significant
- the operator path still needs a longer-duration live execution before it can
  honestly be called production-ready

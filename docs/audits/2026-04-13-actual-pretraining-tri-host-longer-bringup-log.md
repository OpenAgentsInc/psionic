# Actual Pretraining Tri-Host Longer Bringup Log

Date: 2026-04-13

## Goal

Run a longer bounded actual-workload rehearsal on the shipped actual-lane
operator path using the same three-device topology:

- local M5 MacBook Pro as control plane and CPU contributor
- remote `archlinux` Tailnet host as CUDA contributor on the RTX 4080
- remote `macbook-pro-m2` Tailnet host as CPU contributor

This run is closer to the live actual-pretraining shape than the earlier
two-step proof because it widens the bounded distributed segment to six
optimizer steps across two logical windows.

## Command

```bash
PSION_REFERENCE_PILOT_MAX_STEPS=6 \
PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=3 \
PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=2 \
./TRAIN rehearse-base-lane \
  --run-id psion-actual-pretraining-tri-host-actual-long-20260413t003600Z \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote
```

## Result

Run root:

- `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-long-20260413t003600Z`

Terminal outcome:

```text
status=base_lane_rehearsal_complete
run_id=psion-actual-pretraining-tri-host-actual-long-20260413t003600Z
distributed_bringup=/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-long-20260413t003600Z/distributed_execution/distributed_actual_pretraining_bringup.json
distributed_topology=multi_host_joint_gradient_average
checkpoint_eval_decision=continue
continue_restart_decision=continue
```

## Retained Facts

- truth surface kind: `bounded_actual_pretraining_bringup`
- actual-lane relation: `bounded_actual_pretraining_workload`
- execution topology classification: `multi_host_joint_gradient_average`
- contributor count: `3`
- worker hosts:
  - `Christophers-MacBook-Pro-2`
  - `archlinux`
  - `macbook-pro-m2`
- runtime backends:
  - `cpu`
  - `cuda`
  - `cpu`
- optimizer steps: `6`
- contribution receipt count: `18`
- checkpoint ref: `psion-reference-pilot-step-6`
- accepted checkpoint label: `bounded-actual-pretraining-bringup-step-6`
- model id: `psion-compact-decoder-internal-v1`
- dataset identity: `psion_corpus_tokenized@v1`
- train example count: `123`
- validation example count: `36`

## Retained Evidence

Distributed bringup summary:

- `distributed_execution/distributed_actual_pretraining_bringup.json`

Operator summary:

- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_operator_summary.json`

Checkpoint manifest:

- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_checkpoint_manifest.json`

Topology and contribution receipts:

- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_topology_receipt.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_step_receipts.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_contribution_receipts.json`

Per-step exchange evidence:

- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0001/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0002/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0003/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0004/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0005/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0006/...`

Actual-lane closeout:

- `closeout/closeout_bundle.json`

Status surfaces:

- `status/current_run_status.json`
- `status/retained_summary.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`

## Verification

Status verification:

```bash
./TRAIN status --run-root /Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-long-20260413t003600Z
```

Observed status output:

- `phase=base_lane_rehearsal_complete`
- `last_completed_step=6`
- `latest_checkpoint_label=bounded-actual-pretraining-bringup-step-6`
- `git_commit_sha=202e5d117cf51d3576ce84129b108d459729fa62`
- `checkpoint_eval_decision_state=continue`
- `continue_restart_decision_state=continue`

Operational verification:

- `archlinux` was reachable over Tailnet SSH and had no resident CUDA compute
  process before launch
- `macbook-pro-m2` was reachable over Tailnet SSH before launch
- `--cleanup-remote` remained enabled for the whole run

## What This Proves

This run proves the current actual-pretraining operator path can retain a
longer bounded distributed segment on the real actual workload, not just the
minimal two-step proof:

- six optimizer-bearing steps instead of two
- eighteen retained contribution receipts instead of six
- the same three-device merged training path
- the same actual-lane checkpoint, backup, evaluation, continue, and resume
  surfaces around that longer segment

## Claim Boundary

This is still a bounded rehearsal, not the full long-running production
cluster lane.

It proves a longer, more realistic actual-workload segment on the shipped
actual-lane path. It does not by itself prove:

- multi-window public scheduler coordination
- validator or payout finality
- an indefinitely running live actual-pretraining cluster

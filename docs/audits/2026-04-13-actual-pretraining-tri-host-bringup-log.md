# Actual Pretraining Tri-Host Bringup Log

Date: 2026-04-13

## Goal

Prove that the shipped actual pretraining operator path can retain one bounded
multi-host optimizer-bearing segment using the larger actual workload instead
of delegating to the older reference-only workload.

Target topology:

- local M5 MacBook Pro as control plane and CPU contributor
- remote `archlinux` Tailnet host as CUDA contributor on the RTX 4080
- remote `macbook-pro-m2` Tailnet host as CPU contributor

## Command

```bash
./TRAIN rehearse-base-lane \
  --run-id psion-actual-pretraining-tri-host-actual-20260413t001900Z \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote
```

## Result

Run root:

- `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-20260413t001900Z`

Terminal outcome:

```text
status=base_lane_rehearsal_complete
run_id=psion-actual-pretraining-tri-host-actual-20260413t001900Z
distributed_bringup=/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-20260413t001900Z/distributed_execution/distributed_actual_pretraining_bringup.json
distributed_topology=multi_host_joint_gradient_average
checkpoint_eval_decision=continue
continue_restart_decision=continue
```

## Retained Facts

The bounded distributed segment now proves the actual workload rather than the
reference-only workload:

- truth surface kind: `bounded_actual_pretraining_bringup`
- actual-lane relation: `bounded_actual_pretraining_workload`
- model id: `psion-compact-decoder-internal-v1`
- dataset identity: `psion_corpus_tokenized@v1`
- train example count: `123`
- validation example count: `36`
- optimizer steps: `2`
- contributor count: `3`
- execution topology classification: `multi_host_joint_gradient_average`
- runtime backends:
  - `cpu`
  - `cuda`
  - `cpu`
- worker hosts:
  - `Christophers-MacBook-Pro-2`
  - `archlinux`
  - `macbook-pro-m2`
- retained checkpoint ref: `psion-reference-pilot-step-2`
- actual-lane accepted checkpoint label:
  `bounded-actual-pretraining-bringup-step-2`

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

Actual-lane closeout:

- `closeout/closeout_bundle.json`

Status surfaces:

- `status/current_run_status.json`
- `status/retained_summary.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`

## Verification

Focused code verification:

```bash
cargo test -q -p psionic-data actual_pretraining_reconstructed_corpus_builds_validated_inputs
cargo test -q -p psionic-train actual_pretraining_bringup_runs_tri_host_cpu_merge_end_to_end
cargo test -q -p psionic-train --example psion_actual_pretraining_operator
```

Status verification:

```bash
./TRAIN status --run-root /Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-20260413t001900Z
```

Observed status output:

- `phase=base_lane_rehearsal_complete`
- `last_completed_step=2`
- `latest_checkpoint_label=bounded-actual-pretraining-bringup-step-2`
- `git_commit_sha=5ce5139d9f6198e2de88fa30b30939a460061cde`
- `checkpoint_eval_decision_state=continue`
- `continue_restart_decision_state=continue`

## Issue Found And Fixed

The first actual-workload bringup run showed the correct larger model id but
the wrong dataset identity in the retained checkpoint manifest:

- observed wrong value: `psion_reference_tokenized@v1`
- required value: `psion_corpus_tokenized@v1`

Root cause:

- `export_checkpoint(...)` still hardcoded the reference dataset identity

Fix:

- pass the corpus bundle's stable dataset identity through checkpoint export
- rerun the full tri-host actual-workload bringup from the pushed fix SHA

## Claim Boundary

This run proves:

- the actual pretraining operator path now delegates one bounded multi-host
  optimizer-bearing segment through the actual workload
- all three machines contributed to the same optimizer-bearing segment
- the retained checkpoint and operator summary now carry the actual model id
  and canonical actual dataset identity
- the same actual-lane path still retains checkpoint, backup, continue, and
  resume truth around that segment

This run does not prove:

- full broader long-running actual-pretraining cluster execution
- network-coordinated multi-window scheduling
- public control-plane admission, payout, or validator finality for this run

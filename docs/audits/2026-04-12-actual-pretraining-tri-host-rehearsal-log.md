# Actual Pretraining Tri-Host Rehearsal Log

Date: 2026-04-12

## Goal

Prove that the shipped actual pretraining operator path can retain one bounded
multi-host model-progress-bearing segment inside `./TRAIN rehearse-base-lane`
without introducing a second runtime.

Target topology:

- local M5 MacBook Pro as control plane and CPU contributor
- remote `archlinux` Tailnet host as CUDA contributor on the RTX 4080
- remote `macbook-pro-m2` Tailnet host as CPU contributor

## Code Path

Command path:

```bash
./TRAIN rehearse-base-lane \
  --run-id psion-actual-pretraining-tri-host-20260412t224000Z \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote \
  --allow-dirty-tree
```

The operator executed the existing actual-lane sequence:

- `start`
- bounded distributed reference rehearsal
- `record-checkpoint`
- `backup --inject-failed-upload`
- `backup`
- `decide-continue-restart`
- `resume`

The distributed segment was delegated through the shipped
`scripts/train-psion-local-first.sh --mode distributed_reference` path and then
folded back into the actual-lane checkpoint, status, and closeout evidence.

## First Failure And Fix

The first live attempt used:

- run id `psion-actual-pretraining-tri-host-20260412t221500Z`

That run failed inside the distributed rehearsal helper. The retained
`reference_pilot_train.log` showed that `archlinux` staged `refs/heads/main` at
`91d42a8e`, while `macbook-pro-m2` staged the same symbolic ref at
`a16b1e21`. The helper had delegated the rehearsal with the symbolic branch ref
instead of one resolved commit SHA, so the two remote seed repos could stage
different commits.

Fix:

- resolve `selected_git_ref` to one exact SHA inside
  `run_actual_pretraining_distributed_reference_rehearsal()`
- pass that exact SHA to `train-psion-local-first.sh`
- include the retained local rehearsal log tail in the helper error when the
  delegated command fails

## Successful Run

Successful run root:

- `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-20260412t224000Z`

Terminal outcome:

```text
status=base_lane_rehearsal_complete
run_id=psion-actual-pretraining-tri-host-20260412t224000Z
distributed_topology=multi_host_joint_gradient_average
checkpoint_eval_decision=continue
continue_restart_decision=continue
```

Key retained facts from the run:

- distributed topology classification: `multi_host_joint_gradient_average`
- contributor count: `3`
- worker hosts:
  - `Christophers-MacBook-Pro-2`
  - `archlinux`
  - `macbook-pro-m2`
- runtime backends:
  - `cpu`
  - `cuda`
  - `cpu`
- bounded distributed checkpoint ref: `psion-reference-pilot-step-4`
- actual-lane accepted checkpoint label:
  `bounded-distributed-reference-step-4`
- actual-lane accepted checkpoint ref:
  `checkpoint://psion/actual-pretraining/rehearsal/psion-reference-pilot-step-4`
- distributed contribution receipt count: `12`

## Retained Evidence

Actual-lane distributed summary:

- `distributed_execution/distributed_reference_rehearsal.json`

Actual-lane closeout bundle:

- `closeout/closeout_bundle.json`

Actual-lane status and summary:

- `status/current_run_status.json`
- `status/retained_summary.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`

Retained distributed-reference artifacts inside the actual run root:

- `distributed_reference_rehearsal/reference_pilot_artifacts/psion_reference_pilot_cluster_topology_receipt.json`
- `distributed_reference_rehearsal/reference_pilot_artifacts/psion_reference_pilot_cluster_contribution_receipts.json`
- `distributed_reference_rehearsal/reference_pilot_artifacts/psion_reference_pilot_cluster_step_receipts.json`
- `distributed_reference_rehearsal/reference_pilot_artifacts/psion_reference_pilot_checkpoint_manifest.json`
- `distributed_reference_rehearsal/reference_pilot_artifacts/psion_reference_pilot_checkpoint.safetensors`

The actual-lane closeout bundle now carries these additional distributed
artifact kinds:

- `distributed_reference_rehearsal`
- `distributed_reference_topology_receipt`
- `distributed_reference_contribution_receipts`
- `distributed_reference_checkpoint_manifest`

It also adds one distributed proof gate:

- `bounded_distributed_training_proof_retained`

## Verification

Focused code verification:

```bash
cargo test -q -p psionic-train --example psion_actual_pretraining_operator
cargo test -q -p psionic-train rehearsal_uses_actual_pretraining_prefix_for_top_level_binary
```

Retained status verification:

```bash
./TRAIN status --run-root /Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-20260412t224000Z
```

Observed status output:

- `phase=base_lane_rehearsal_complete`
- `last_completed_step=4`
- `latest_checkpoint_label=bounded-distributed-reference-step-4`
- `checkpoint_eval_decision_state=continue`
- `continue_restart_decision_state=continue`

## Operational Notes

- The first remote compile on `archlinux` and `macbook-pro-m2` took several
  minutes. That was expected cold-build latency, not a deadlock.
- `--cleanup-remote` removed the staged remote rehearsal repos and outputs
  after completion.
- This run used `--allow-dirty-tree` because the operator and runbook changes
  were still in progress locally while the live proof was being generated. The
  retained actual-lane surfaces recorded that posture as
  `dirty_tree_admission=allowed_by_operator_override`.

## Claim Boundary

This run proves:

- the actual pretraining operator path can retain one bounded multi-host
  model-progress-bearing segment
- that bounded segment can feed the actual-lane accepted checkpoint lineage
- the same actual-lane path can still retain backup recovery, automatic eval,
  continue-vs-restart decision, and resume after that segment

This run does not prove:

- full broader actual-pretraining distributed cluster execution beyond the
  bounded rehearsal segment
- external alert delivery
- streaming dashboard publication
- plugin-conditioned continuation execution

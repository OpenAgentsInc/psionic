# Actual Pretraining Tri-Host Recovery Drill Log

Date: 2026-04-13

## Goal

Prove one planned interruption during the longer tri-host actual-lane execution
and recover on the same retained checkpoint and artifact lineage without manual
path edits.

Target topology:

- local M5 MacBook Pro as control plane and CPU contributor
- remote `archlinux` Tailnet host as CUDA contributor on the RTX `4080`
- remote `macbook-pro-m2` Tailnet host as CPU contributor

Target workload identity:

- model id: `psion-compact-decoder-internal-v1`
- dataset identity: `psion_corpus_tokenized@v1`

## Source-Of-Truth Command

```bash
PSION_REFERENCE_PILOT_MAX_STEPS=6 \
PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=3 \
PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=1 \
./TRAIN rehearse-base-lane \
  --run-id psion-actual-pretraining-tri-host-actual-recovery-20260413t154800Z \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --planned-interruption-step 3 \
  --cleanup-remote \
  --allow-dirty-tree
```

## Result

Run root:

- `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-recovery-20260413t154800Z`

Terminal outcome:

```text
phase=base_lane_rehearsal_complete
run_id=psion-actual-pretraining-tri-host-actual-recovery-20260413t154800Z
distributed_topology=multi_host_joint_gradient_average
last_completed_step=6
latest_checkpoint_label=bounded-actual-pretraining-bringup-step-6
checkpoint_eval_decision_state=continue
continue_restart_decision_state=continue
continue_restart_operator_action=continue_long_run
distributed_recovery_drill=/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-recovery-20260413t154800Z/distributed_actual_pretraining_bringup/actual_pretraining_bringup_recovery_drill.json
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
- interruption kind: `planned_checkpoint_boundary_interrupt`
- interruption step: `3`
- total target optimizer steps: `6`
- interrupted checkpoint ref: `psion-reference-pilot-step-3`
- interrupted checkpoint object digest:
  `6146b995bcc9b4169be93f01c5f45164c6ecd3da66e4b50701ee5b8a751d2d11`
- recovered checkpoint ref: `psion-reference-pilot-step-6`
- recovered checkpoint object digest:
  `da3a69a2007cc6492eae508c15cf57f2437731d1a0073d9c56310269d4f29743`
- lineage preserved: `true`
- recovery state: `recovered`
- contribution receipt count: `18`
- progress checkpoint count: `2`
- progress window count: `2`
- progress cadence count: `2`
- final cumulative train tokens processed: `394`
- final cumulative mean tokens per second: `16`
- checkpoint eval decision: `continue`
- continue/restart decision: `continue`

## What The Run Proved

- the interrupted segment retained a coherent actual-lane checkpoint family
- the recovered segment resumed from that retained family without manual path
  edits
- the combined proof root kept one coherent checkpoint lineage and one combined
  topology, contribution, continuity, and progress family
- the operator closeout bundle now carries one retained distributed recovery
  drill artifact and one explicit gate:
  `bounded_distributed_recovery_drill_retained`

## Important Runtime Detail

One oversized attempt at the same proof with a `12`-step total target
(`psion-actual-pretraining-tri-host-actual-recovery-20260413t145500Z`) was
stopped after it became clear that first-segment request serialization on the
local M5 was dominating wallclock time. That did not expose a lineage or
resume bug; it exposed a remaining performance boundary on the larger recovery
shape.

The accepted source-of-truth recovery proof for this issue is therefore the
cleaner `6`-step interruption drill above, which still exercises:

- real actual-workload tri-host contribution
- one interrupted checkpoint boundary
- one resumed segment on the same checkpoint family
- one final accepted checkpoint chain and closeout packet

## Verification

Commands run:

```bash
cargo test -q -p psionic-train --example psion_distributed_actual_pretraining_bringup
cargo test -q -p psionic-train --example psion_actual_pretraining_operator
./TRAIN status --run-root /Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-recovery-20260413t154800Z
```

Key retained outputs:

- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_recovery_drill.json`
- `distributed_execution/distributed_actual_pretraining_bringup.json`
- `closeout/closeout_bundle.json`

Those files are the operator-readable proof that the actual lane can survive
one planned checkpoint-boundary interruption and recover on the same retained
artifact family.

# Actual Pretraining Tri-Host Production-Candidate Bringup Log

Date: 2026-04-13

## Goal

Extend the real actual-lane tri-host bringup from the earlier `2`-step and
`6`-step proofs into a materially longer production-candidate canary on the
same shipped operator path.

Target topology:

- local M5 MacBook Pro as control plane and CPU contributor
- remote `archlinux` Tailnet host as CUDA contributor on the RTX `4080`
- remote `macbook-pro-m2` Tailnet host as CPU contributor

Target workload identity:

- model id: `psion-compact-decoder-internal-v1`
- dataset identity: `psion_corpus_tokenized@v1`

## Earlier Failures That Drove The Fixes

Before the final clean canary, the longer run exposed two real bottlenecks.

### Oversized retained receipt serialization

Run:

- `psion-actual-pretraining-tri-host-actual-prodcanary-20260413t093800Z`

Observed behavior:

- all `12` optimizer steps completed
- the run stalled while writing retained full JSON receipt payloads

Fix that landed:

- retain compact summary receipts instead of full raw gradient payloads

### Oversized step-exchange transport

Run:

- `psion-actual-pretraining-tri-host-actual-prodcanary-rerun-20260413t105400Z`

Observed behavior:

- the longer canary now got past retained summary writing
- step requests and responses were still too large as plain JSON transport
- the request payload was roughly `544 MB`
- the response payload was roughly `732 MB`

Fixes that landed:

- compress per-step request and response transport to `.json.zst`
- compress SSH and SCP transport
- stage the repo to remotes as `.tar.gz` instead of raw tar

Those changes are now part of the shipped path for the actual-lane distributed
bringup.

## Source-Of-Truth Command

```bash
PSION_REFERENCE_PILOT_MAX_STEPS=12 \
PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=3 \
PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=2 \
./TRAIN rehearse-base-lane \
  --run-id psion-actual-pretraining-tri-host-actual-prodcanary-zstd-clean-20260413t134400Z \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote
```

## Result

Run root:

- `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-zstd-clean-20260413t134400Z`

Terminal outcome:

```text
phase=base_lane_rehearsal_complete
run_id=psion-actual-pretraining-tri-host-actual-prodcanary-zstd-clean-20260413t134400Z
distributed_topology=multi_host_joint_gradient_average
last_completed_step=12
latest_checkpoint_label=bounded-actual-pretraining-bringup-step-12
checkpoint_eval_decision_state=continue
continue_restart_decision_state=continue
continue_restart_operator_action=continue_long_run
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
- optimizer steps: `12`
- contribution receipt count: `36`
- progress checkpoint count: `4`
- progress window count: `4`
- progress cadence count: `2`
- final cumulative train tokens processed: `775`
- final cumulative mean tokens per second: `16`
- checkpoint ref: `psion-reference-pilot-step-12`
- checkpoint object digest:
  `b109c42c2380cf0520538ad8d484386ebe5ecf758c9faf23c7dffcc0e0547f66`
- checkpoint total bytes: `125960488`
- accepted checkpoint label: `bounded-actual-pretraining-bringup-step-12`
- model id: `psion-compact-decoder-internal-v1`
- dataset identity: `psion_corpus_tokenized@v1`

## Retained Evidence

Distributed bringup summary:

- `distributed_execution/distributed_actual_pretraining_bringup.json`

Actual-lane status and checkpoint truth:

- `status/current_run_status.json`
- `status/retained_summary.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `closeout/closeout_bundle.json`

Retained distributed evidence family:

- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_topology_receipt.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_step_receipts.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_contribution_receipts.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_contributor_continuity_receipt.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_progress_checkpoint_receipts.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_progress_checkpoints/`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_checkpoint_manifest.json`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_checkpoint.safetensors`

Compressed per-step exchange evidence:

- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0001/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0002/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0003/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0004/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0005/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0006/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0007/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0008/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0009/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0010/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0011/...`
- `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_exchange/step-0012/...`

## Verification

Focused code verification:

```bash
cargo test -q -p psionic-train --example psion_actual_pretraining_operator
cargo test -q -p psionic-train --example psion_distributed_actual_pretraining_bringup
cargo test -q -p psionic-train --example psion_actual_pretraining_joint_contribution
```

Status verification:

```bash
./TRAIN status --run-root /Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-zstd-clean-20260413t134400Z
```

Observed status output:

- `phase=base_lane_rehearsal_complete`
- `last_completed_step=12`
- `latest_checkpoint_label=bounded-actual-pretraining-bringup-step-12`
- `selected_git_ref=refs/heads/issue-937`
- `git_commit_sha=033c09a8197d6e69b7f7dd3a3475f3062691ead4`
- `checkpoint_eval_decision_state=continue`
- `checkpoint_eval_score_bps=8532`
- `continue_restart_decision_state=continue`
- `continue_restart_operator_action=continue_long_run`

Remote cleanup verification:

- `archlinux` staged run root removed after completion
- `macbook-pro-m2` staged run root removed after completion

## What This Proves

This run proves the shipped actual-lane operator path now supports one
materially longer bounded tri-host production-candidate canary on the real
actual workload:

- `12` optimizer-bearing steps instead of `2` or `6`
- `36` retained contribution receipts
- four retained progress checkpoints across four retained windows and two
  cadences
- explicit contributor continuity proof across the whole run
- compressed request/response transport that keeps the longer canary finishable
  end to end on the same path

## Residual Risk

The remaining risk is no longer basic correctness of the bounded canary. The
remaining risk is launch posture:

- cold remote compile time still dominates startup
- repo staging for a cold remote is still noticeable even after tarball
  compression
- this is still a bounded production-candidate canary, not a continuously live
  longer actual-lane execution with in-flight recovery and control-plane truth

Those risks are the next phase, not a blocker to saying the longer bounded
actual-lane canary is now proven.

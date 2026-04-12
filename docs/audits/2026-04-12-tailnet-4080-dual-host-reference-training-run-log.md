# Tailnet 4080 Dual-Host Reference Training Run Log

This log records the first completed bounded `reference_pilot` run where the
local M5 host and the Tailnet-connected RTX 4080 host both remained in the
optimizer path for the same retained run.

## Scope

- lane: `reference_pilot`
- mode: `distributed_reference`
- control plane: `Christophers-MacBook-Pro-2`
- remote worker: `archlinux`
- remote worker GPU: `NVIDIA GeForce RTX 4080`
- execution classification: `dual_host_joint_gradient_average`
- claim boundary: bounded reference-pilot proof only, not the actual broader-
  pretraining lane

## Preflight

Direct Tailnet SSH to `100.108.56.85` was required. The host had just been
rebooted. Before launch:

- `tailscale status` showed `archlinux` active again
- `nc -vz -G 3 100.108.56.85 22` succeeded
- `ssh christopherdavid@100.108.56.85 'bash -ic "hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"'`
  succeeded
- `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits`
  returned no resident compute processes

## Command

Run from `/Users/christopherdavid/work/psionic`:

```bash
./TRAIN --lane reference_pilot --mode distributed_reference --cleanup-remote \
  --run-id tailnet-4080-dual-host-reference-20260412t193400Z
```

## What Happened

The launcher wrote the operator manifest immediately and staged a clean remote
worktree on `archlinux` under:

- `$HOME/code/psion-reference-pilot/tailnet-4080-dual-host-reference-20260412t193400Z/repo`

The first optimizer step then blocked on the remote cold build of
`psion_reference_pilot_joint_contribution`. That was expected because the
remote target directory had been cleared by the reboot. After the first remote
build completed, the four-step bounded run finished successfully.

Final launcher output:

```text
status=completed
mode=distributed_reference
run_id=tailnet-4080-dual-host-reference-20260412t193400Z
output_root=/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-dual-host-reference-20260412t193400Z
reference_pilot_artifact_dir=/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-dual-host-reference-20260412t193400Z/reference_pilot_artifacts
```

## Retained Outputs

Local run root:

- `/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-dual-host-reference-20260412t193400Z`

Canonical operator files:

- `reference_pilot_operator_manifest.json`
- `reference_pilot_operator_summary.json`
- `reference_pilot_train.log`

Canonical retained artifacts:

- `reference_pilot_artifacts/psion_reference_pilot_stage_receipt.json`
- `reference_pilot_artifacts/psion_reference_pilot_observability_receipt.json`
- `reference_pilot_artifacts/psion_reference_pilot_checkpoint_manifest.json`
- `reference_pilot_artifacts/psion_reference_pilot_dual_host_topology_receipt.json`
- `reference_pilot_artifacts/psion_reference_pilot_dual_host_step_receipts.json`
- `reference_pilot_artifacts/psion_reference_pilot_dual_host_exchange/`

## What The Retained Proof Shows

`reference_pilot_operator_summary.json` records:

- `selected_mode = distributed_reference`
- `status = completed`
- `worker_count = 2`
- `execution_location = hybrid_cluster`
- `execution_topology_classification = dual_host_joint_gradient_average`
- `local_runtime_backend = cpu`
- `remote_runtime_backend = cuda`
- `checkpoint_ref = psion-reference-pilot-step-4`
- `checkpoint_parameter_state_digest = 3721f39a289cdef21e6c81ad8914bb526767194f8c11a3cfc8a56dc218adaba5`

`psion_reference_pilot_dual_host_topology_receipt.json` records:

- `control_plane_host = Christophers-MacBook-Pro-2`
- `remote_worker_host = archlinux`
- `execution_topology_classification = dual_host_joint_gradient_average`
- two selected nodes in `cluster_execution.selected_nodes`
- one local CPU trainer and one remote CUDA trainer in the same retained run
- `checkpoint_ref = psion-reference-pilot-step-4`

The receipt detail is explicit:

> one local CPU contribution per optimizer step and one remote cuda
> contribution per optimizer step before the merged update advanced the shared
> checkpoint

## Resume Verification

Run from `/Users/christopherdavid/work/psionic`:

```bash
cargo run -q -p psionic-train --example psion_reference_pilot_resume_probe -- \
  /Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-dual-host-reference-20260412t193400Z/reference_pilot_artifacts \
  /tmp/tailnet-4080-dual-host-reference-resume-probe-20260412t193400Z
```

Result:

```text
psion reference pilot resume probe completed: run=psion-distributed-reference-pilot-run checkpoint=psion-reference-pilot-step-4 output=/tmp/tailnet-4080-dual-host-reference-resume-probe-20260412t193400Z/psion_reference_pilot_resume_probe.json
```

Resume receipt:

- `/tmp/tailnet-4080-dual-host-reference-resume-probe-20260412t193400Z/psion_reference_pilot_resume_probe.json`

## Remote Cleanup Verification

Because the run used `--cleanup-remote`, the staged remote run root was removed
after completion:

```bash
ssh christopherdavid@100.108.56.85 'bash -ic "test ! -e $HOME/code/psion-reference-pilot/tailnet-4080-dual-host-reference-20260412t193400Z && echo cleaned || echo still_present"'
```

Observed result:

```text
cleaned
```

## Conclusion

This run closes the bounded dual-host reference proof. The local M5 host and
the Tailnet RTX 4080 host both contributed model-progress-bearing work to the
same retained run, the shared checkpoint advanced to step 4, the retained
topology receipt names both machines as trainers, and the checkpoint restored
through the shipped resume-probe path.

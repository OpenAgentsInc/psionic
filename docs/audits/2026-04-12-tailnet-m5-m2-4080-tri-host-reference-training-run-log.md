# 2026-04-12 Tailnet M5 + M2 + 4080 Tri-Host Reference Training Run

## Goal

Run one bounded `reference_pilot` training job where all available machines
stay in the optimizer path:

- local `macbook-pro-m5` CPU contributor and control plane
- remote `archlinux` RTX 4080 CUDA contributor
- remote `macbook-pro-m2` CPU contributor

This run needed to remain inside the shipped Psionic bounded reference lane. It
could not invent a second training runtime.

## Preconditions

- live Tailnet reachability to:
  - `archlinux`
  - `macbook-pro-m2`
- staged code from the current local `psionic` checkout
- `cargo` available on both remotes

## First Failure

The first tri-host launch used:

```bash
./TRAIN --lane reference_pilot \
  --mode distributed_reference \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote \
  --run-id tailnet-m5-m2-4080-reference-20260412t210500Z
```

That run reached `archlinux`, completed the first CUDA contribution, then
failed on `macbook-pro-m2` with:

```text
zsh:1: command not found: cargo
Error: RemoteContribution { message: "remote contributor command failed with status exit status: 127" }
```

The root cause was the remote helper in
`crates/psionic-train/examples/psion_distributed_reference_pilot.rs`. It sent
`bash -lc` to SSH as split argv. SSH re-stringified that command on the remote
side, which made the quoting unreliable on macOS. The remote shell reached the
machine, but it did not execute the intended `bash -lc '<full command>'`
string.

## Fix

The helper now wraps the full remote command in one quoted shell string:

- before: split `ssh ... bash -lc <command>`
- after: `ssh ... "bash -lc '<command>'"`

That change landed in:

- `crates/psionic-train/examples/psion_distributed_reference_pilot.rs`

## Successful Tri-Host Run

After the remote shell fix, the successful run used:

```bash
./TRAIN --lane reference_pilot \
  --mode distributed_reference \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote \
  --run-id tailnet-m5-m2-4080-reference-20260412t213600Z
```

Run root:

- `/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-m5-m2-4080-reference-20260412t213600Z`

Result:

- status: `completed`
- execution classification: `multi_host_joint_gradient_average`
- contributor count: `3`
- checkpoint ref: `psion-reference-pilot-step-4`
- optimizer steps: `4`

## Proof Artifacts

Canonical retained proof files:

- `reference_pilot_artifacts/psion_reference_pilot_cluster_topology_receipt.json`
- `reference_pilot_artifacts/psion_reference_pilot_cluster_step_receipts.json`
- `reference_pilot_artifacts/psion_reference_pilot_cluster_contribution_receipts.json`
- `reference_pilot_artifacts/psion_reference_pilot_checkpoint_manifest.json`

Topology receipt facts:

- `execution_topology_classification`: `multi_host_joint_gradient_average`
- `contributor_count`: `3`
- `worker_hosts`:
  - `Christophers-MacBook-Pro-2`
  - `archlinux`
  - `macbook-pro-m2`
- `runtime_backends`:
  - `cpu`
  - `cuda`
  - `cpu`

Contribution receipt facts:

- `4` optimizer steps
- `12` contribution receipts total
- each step includes contributions from:
  - local M5 CPU
  - remote 4080 CUDA
  - remote M2 CPU
- each contribution used `256` samples

That means each shared optimizer step merged one gradient contribution from all
three machines before the checkpoint advanced.

## Resume Verification

Retained checkpoint restore verification used:

```bash
cargo run -q -p psionic-train --example psion_reference_pilot_resume_probe -- \
  /Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-m5-m2-4080-reference-20260412t213600Z/reference_pilot_artifacts \
  /tmp/tailnet-m5-m2-4080-reference-resume-probe-20260412t213600Z
```

Result:

- completed successfully
- receipt:
  `/tmp/tailnet-m5-m2-4080-reference-resume-probe-20260412t213600Z/psion_reference_pilot_resume_probe.json`

## Remote Cleanup Verification

Because the run used `--cleanup-remote`, both staged remote run roots were
removed after completion:

- `archlinux`: removed
- `macbook-pro-m2`: removed

## Verification Commands

Code verification before the successful rerun:

```bash
rustfmt crates/psionic-train/examples/psion_distributed_reference_pilot.rs
cargo build -q -p psionic-train --example psion_distributed_reference_pilot
cargo test -q -p psionic-train distributed_reference_pilot_runs_dual_host_cpu_merge_end_to_end
cargo test -q -p psionic-train distributed_reference_pilot_runs_tri_host_cpu_merge_end_to_end
```

Run verification after completion:

```bash
ssh christopherdavid@100.108.56.85 'bash -ic "test -e \$HOME/code/psion-reference-pilot/tailnet-m5-m2-4080-reference-20260412t213600Z && echo ARCH_EXISTS=1 || echo ARCH_EXISTS=0"'
ssh christopherdavid@100.72.151.98 'bash -lc "test -e \$HOME/code/psion-reference-pilot/tailnet-m5-m2-4080-reference-20260412t213600Z && echo M2_EXISTS=1 || echo M2_EXISTS=0"'
```

Both returned `0`, which confirms cleanup succeeded.

## Current Boundary

This is a real three-contributor bounded training proof. It is still the
`reference_pilot` lane, not the broader actual-pretraining lane. The next step
is to carry the same retained multi-host proof shape into the broader shipped
training surface without dropping checkpoint, replay, or observability truth.

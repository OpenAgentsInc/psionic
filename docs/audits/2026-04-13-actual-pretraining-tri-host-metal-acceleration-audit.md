# Actual Pretraining Tri-Host Metal Acceleration Audit

Date: 2026-04-13
Repo: `psionic`
Issue: `OpenAgentsInc/psionic#940`

## Scope

This audit explains the move from CPU-only Apple hosts to Metal-backed Apple
hosts in the bounded tri-host actual-pretraining bringup lane. The admitted
cluster shape is:

- local M5 MacBook Pro
- remote Tailnet RTX `4080` host `archlinux`
- remote Tailnet M2 MacBook Pro

The goal was not a new lane. The goal was to keep the existing shipped
`./TRAIN rehearse-base-lane` actual-lane path, make the Apple machines use the
repo's real Metal backend, and improve useful model-progress throughput on that
same retained proof surface.

## What Changed

The code changes were:

- added `psionic-backend-metal` to `psionic-train`
- extended the joint-contribution backend enum to accept `metal`
- defaulted macOS control-plane and secondary Apple contributors to `metal`
  instead of `cpu`
- raised the default Metal batch sizing on the actual bringup path
  from `4/6/4` style conservative sizing to `8/12/8` across
  control-plane Apple / CUDA / secondary Apple contributors
- fixed the retained cluster-topology `selected_nodes` view so the local
  Apple-silicon contributor is recorded as `metal` instead of one stale nested
  `cpu` label

The shipped user path stayed the same:

```bash
./TRAIN rehearse-base-lane \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote
```

## Important Operational Finding

Remote staging behavior matters.

When the run commit exists only in the local checkout, the remotes cannot stage
from their seed repos. In that case the launcher falls back to
`archive_tarball`.

When the exact run commit is on `origin/main`, the remotes can fetch `main` and
stage with `remote_git_worktree`.

That distinction does not change retained optimizer outputs. It does change
real wall-clock elapsed time by cutting stage and unpack overhead from the front
of the run.

## Source Runs

### 1. CPU-Mac baseline

- run id:
  `psion-actual-pretraining-tri-host-actual-prodcanary-optimized-20260413t101910Z`
- backends: `cpu`, `cuda`, `cpu`
- stage strategy: `archive_tarball`
- retained train tokens: `2194`
- retained observability mean tokens/sec: `56`
- shell elapsed: `1421.00s`
- effective train tokens / real elapsed second: `1.54`

### 2. First Metal bringup

- run id:
  `psion-actual-pretraining-tri-host-actual-prodcanary-metal-rerun-20260413t151200Z`
- backends: `metal`, `cuda`, `metal`
- stage strategy: `archive_tarball`
- retained train tokens: `2827`
- retained observability mean tokens/sec: `70`
- shell elapsed: `1441.71s`
- effective train tokens / real elapsed second: `1.96`

### 3. Tuned Metal bringup, local-only commit staging

- run id:
  `psion-actual-pretraining-tri-host-actual-prodcanary-metal-tuned-20260413t161900Z`
- backends: `metal`, `cuda`, `metal`
- stage strategy: `archive_tarball`
- retained train tokens: `3992`
- retained observability mean tokens/sec: `94`
- shell elapsed: `1525.31s`
- effective train tokens / real elapsed second: `2.62`

### 4. Tuned Metal bringup, pushed `origin/main` seed-repo staging

- run id:
  `psion-actual-pretraining-tri-host-actual-prodcanary-metal-mainseed-20260413t183100Z`
- backends: `metal`, `cuda`, `metal`
- stage strategy: `remote_git_worktree`
- retained train tokens: `3992`
- retained observability mean tokens/sec: `94`
- shell elapsed: `1459.39s`
- effective train tokens / real elapsed second: `2.74`

## What Is True Now

The Apple machines are no longer CPU-only in the tri-host actual-lane bringup.

That is proven by:

- operator summaries with `runtime_backends = ["metal", "cuda", "metal"]`
- contribution receipts for:
  - `Christophers-MacBook-Pro-2-metal`
  - `macbook-pro-m2-metal`
- cluster topology receipts whose `selected_nodes` now report the local Apple
  host as `metal`

The actual-lane tri-host proof now does materially more training work per
bounded run than the prior CPU-Mac version.

Compared to the clean CPU-Mac baseline:

- retained train tokens increased from `2194` to `3992`
- retained observability mean tokens/sec increased from `56` to `94`
- effective train tokens per real elapsed second increased from `1.54` to
  `2.74`

That last number is the useful plain-language result. The current tuned
Metal-backed tri-host actual-lane canary gets about `78%` more retained model
progress per real elapsed second than the previous CPU-Mac version.

## What Did Not Improve Automatically

The first Metal run did not magically collapse total wall-clock time.

Reasons:

- first-run compile on all three machines still costs time
- when the commit is not yet on `origin/main`, remote staging falls back to
  tarball archive/unpack work
- the tuned Metal configuration also carries a larger workload, so doing more
  useful training work can still keep the total run near the old elapsed band

This means two throughput views must stay separate:

- retained model-progress throughput
- shell-observed end-to-end elapsed time

The system improved strongly on the first measure. It improved moderately on
the second once remote staging switched to `remote_git_worktree`.

## Proof Artifacts

The strongest current proof run is:

- run root:
  `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-metal-mainseed-20260413t183100Z`
- operator summary:
  `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-metal-mainseed-20260413t183100Z/distributed_actual_pretraining_bringup/actual_pretraining_bringup_operator_summary.json`
- observability receipt:
  `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-metal-mainseed-20260413t183100Z/distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_observability_receipt.json`
- cluster topology receipt:
  `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-metal-mainseed-20260413t183100Z/distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/psion_actual_pretraining_bringup_cluster_topology_receipt.json`

## Remaining Gap

This is still a bounded bringup proof inside the actual-lane operator path.

It is not yet a long-duration continuous production run with the full ongoing
operational surface left live the entire time.

The next honest step after this audit is:

- keep the same Metal-backed tri-host shape
- extend duration
- keep more of the actual-lane operational surface live continuously
- keep using pushed release refs so remote stage uses `remote_git_worktree`
  instead of tarball fallback

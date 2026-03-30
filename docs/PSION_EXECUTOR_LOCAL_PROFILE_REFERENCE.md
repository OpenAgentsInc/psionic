# Psion Executor Local Profile Reference

> Status: canonical `PSION-0101` / `#706` and `PSION-0102` / `#707` record,
> updated 2026-03-30 after landing the first typed executor admitted-profile
> catalog in
> `crates/psionic-train/src/psion_executor_admitted_profiles.rs`.

This document records the local-first executor profile authority that now sits
under the workspace roadmap.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_admitted_profiles_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_admitted_profile_fixtures
```

## What Landed

`psionic-train` now owns one typed admitted-profile catalog for the executor
lane.

The first landed profiles are:

- `local_mac_mlx_aarch64`
- `local_4080_cuda_tailnet_x86_64`
- `local_tailnet_cluster_control_plane`

The catalog freezes:

- the stable profile id
- admitted run-type posture
- local requirements
- checkpoint expectations
- connectivity expectations
- shipped entrypoints
- authority artifacts and digests
- claim boundary

## Current Admitted Mac Profile

`local_mac_mlx_aarch64` is now the explicit local Apple Silicon executor
profile for:

- MLX smoke runs
- MLX-local decision-grade runs when the question is explicitly MLX-specific
- eval-pack execution
- checkpoint restore rehearsal
- export inspection
- CPU-validation ownership

It is grounded in the already-shipped surfaces:

- `fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json`
- `fixtures/psion/executor/psion_executor_mlx_forward_load_parity_v1.json`
- `fixtures/training/compute_sources/local_mlx_mac_workstation_v1.json`
- `scripts/check-swarm-mac-mlx-bringup.sh`
- `crates/psionic-train/src/swarm_mlx_bringup.rs`
- `crates/psionic-train/src/bin/swarm_mac_mlx_bringup.rs`

The first executor-lane MLX load/forward boundary is now also explicit:

- `docs/PSION_EXECUTOR_MLX_FORWARD_LOAD_PARITY.md`
- `docs/PSION_EXECUTOR_MLX_CHECKPOINT_COMPATIBILITY.md`
- `docs/PSION_EXECUTOR_MLX_SMOKE_RUN.md`
- `docs/PSION_EXECUTOR_MLX_DECISION_GRADE_RUN.md`
- `docs/PSION_EXECUTOR_MAC_EXPORT_INSPECTION.md`

That packet keeps the shipped MLX entrypoint, admitted forward probe, bounded
converted-equivalent load lane, and explicit parity gaps in one reviewable
place before smoke-run and checkpoint work starts citing the Mac profile. The
checkpoint packet then binds the retained same-node MLX bundle back into the
canonical model-IO import and compatibility contract instead of leaving MLX
checkpoint claims as a freeform note. The smoke-run packet then binds the
retained same-node run to the approved local subset of
`tassadar.eval.frequent.v0` without claiming the Mac already closes the
local-cluster roundtrip. The decision-grade packet then upgrades that same run
into one explicit MLX-local decision packet by binding the retained same-node
report, the admitted-device matrix report, and one executor-specific
remote-training `v2` bundle plus run-index entry together without pretending
the Mac already owns the cross-device lane. The Mac export-inspection packet
then imports that retained MLX bundle locally, emits one torch-style
compatibility artifact, keeps `host_cpu_aarch64` explicit as the admitted CPU
validation class, and rechecks the `reference_linear` versus `hull_cache`
claim boundary before later replacement packets can cite the Mac profile.

## Current Admitted 4080 Tailnet Profile

`local_4080_cuda_tailnet_x86_64` is now the explicit local accelerator profile
for:

- 4080 smoke runs
- 4080 decision-grade runs
- 4080 confirmation reruns
- replay-accounted eval participation inside the admitted Tailnet lane

It is grounded in the already-shipped surfaces:

- `fixtures/swarm/reports/swarm_linux_rtx4080_bringup_v1.json`
- `fixtures/training/compute_sources/local_rtx4080_workstation_v1.json`
- `fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json`
- `docs/audits/2026-03-27-tailnet-short-run-device-audit.md`
- `scripts/check-swarm-linux-4080-bringup.sh`
- `scripts/run-first-swarm-tailnet-admitted-live.sh`
- `crates/psionic-train/src/swarm_cuda_bringup.rs`
- `crates/psionic-train/src/bin/swarm_linux_cuda_bringup.rs`

The retained throughput band is now explicit:

- metric: `same_node_open_adapter_steps_per_second`
- minimum admitted value: `80.0`
- expected retained value: `122.8920`
- declared band ceiling: `125.0`

Checkpoint and connectivity expectations are now explicit too:

- worker-local scratch can live under staged remote bundle roots such as
  `$HOME/code/psionic-tailrun/<run_id>/linux`
- counted checkpoints and retained artifact packets still have to come back
  through the controller-owned bundle under `fixtures/swarm/runs/<run_id>/`
- Tailnet SSH reachability to `archlinux` is required before the profile counts
- coordinator and contributor ports remain operator-selected and bounded

The first admitted remote-launch packet for this profile is now explicit too:

- `docs/PSION_EXECUTOR_4080_REMOTE_LAUNCH.md`
- `docs/PSION_EXECUTOR_4080_DURABLE_CHECKPOINT.md`

That packet binds the shipped Tailnet operator script, the retained operator
manifest, the retained admitted run bundle, and the Linux RTX 4080
acknowledgement receipt into one reviewable launch packet before later
checkpoint, eval, smoke, and recovery packets start citing the 4080 lane. The
durable-checkpoint packet then upgrades that same lane into one explicit
checkpoint path by binding the retained pointer digest, both worker submission
receipts, the remote contributor checkpoint family, and the merged portable
bundle import path back onto the Mac control plane without pretending the live
interruption drill is already done.

## Current Admitted Tailnet Control-Plane Profile

`local_tailnet_cluster_control_plane` is now the explicit local-first
Mac-to-4080 roundtrip profile for:

- Tailnet-backed 4080 smoke runs that return a retained bundle
- decision-grade 4080 runs that need the full Mac -> 4080 -> Mac evidence path
- confirmation reruns on the same admitted controller/worker workflow

It is grounded in the already-shipped surfaces:

- `fixtures/swarm/first_swarm_trusted_lan_topology_contract_v1.json`
- `fixtures/swarm/runs/tailrun-home-admitted-20260327e/operator_manifest.json`
- `fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json`
- `docs/audits/2026-03-27-tailrun-admitted-home-tailnet-run-audit.md`
- `scripts/run-first-swarm-tailnet-admitted-live.sh`
- `scripts/check-first-swarm-trusted-lan-real-run.sh`
- `crates/psionic-train/src/swarm_first_live_runtime.rs`
- `crates/psionic-train/src/swarm_trusted_lan.rs`

The responsibility split is now explicit:

- controller responsibilities:
  materialize the operator manifest and bundle root, publish the explicit
  Tailnet endpoints and ports, and retain the final bundle under
  `fixtures/swarm/runs/<run_id>/`
- worker responsibilities:
  execute only the bounded contributor role, return staged artifacts to the
  controller-owned bundle path, and avoid any independent validator, publish,
  or promotion claim

The artifact return rule is now explicit too:

- remote scratch is staging space only
- the Mac controller-owned run root is the canonical counting artifact home
- partial remote outputs do not count until the retained bundle exists locally

## Honest Boundary

This catalog still does not claim:

- the 4080 worker has independent checkpoint-writer authority
- the Mac profile alone closes remote launch
- dense cross-device training closure
- public-worker or external promotion authority

The current catalog closes the first three local profiles so later EPIC 1
issues can add the frozen eval packs, baseline truth, and variance thresholds
without inventing new profile vocabulary.

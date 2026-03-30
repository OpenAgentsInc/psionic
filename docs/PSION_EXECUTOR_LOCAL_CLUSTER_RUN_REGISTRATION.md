# Psion Executor Local Cluster Run Registration

> Status: canonical `PSION-0401` / `#734` record, updated 2026-03-30 after
> landing the first canonical local-cluster run-registration schema for the
> admitted executor lane.

This document records the first machine-readable registration packet that the
local executor cluster uses before ledger, dashboard, and roundtrip closure
stack on top of it.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_local_cluster_run_registration_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_local_cluster_run_registration_fixtures
```

## What Landed

`psionic-train` now owns one typed local-cluster run-registration packet for
the executor lane.

That packet keeps one canonical registration row for:

- the retained MLX decision-grade Mac run
- the retained 4080 decision-grade Tailnet run

Both rows now register the same required fields:

- admitted machine profile
- compute-source identity
- run id plus search run ids
- model id
- candidate status
- active mixture version id
- frozen eval-pack ids
- wallclock budget
- observed duration
- stop condition
- batch geometry
- memory-headroom posture
- expected throughput
- checkpoint family

This means the first Mac and 4080 decision-grade executor runs no longer depend
on separate ad hoc prose for admission facts. Missing required fields now fail
validation at the schema boundary instead of staying review-time guesswork.

## Current Retained Truth

- packet digest:
  `dfad1972f358be079ddd80ac73f5ec85200c16e1e5a708fb11a18bc765cec229`
- MLX registration digest:
  `d326ad89e5ec8984b6794ccc77e129814bf596e8853cc0fbd83b31af72524a21`
- 4080 registration digest:
  `2e023b59b65e8efec98a41eb093127ec414671e0a3b4b310eb4c79b69ec898ae`
- model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- active mixture version:
  `psion_executor_canonical_mixture_v0`
- MLX run id:
  `same-node-wallclock-retained-mlx`
- MLX profile:
  `local_mac_mlx_aarch64`
- MLX compute source:
  `local_mlx_mac_workstation`
- MLX expected steps per second:
  `162.53061053630358`
- 4080 run id:
  `tailrun-home-admitted-20260328k`
- 4080 supporting search run ids:
  `same-node-wallclock-retained-cuda`,
  `tailrun-admitted-device-matrix-20260327b`
- 4080 worker profile:
  `local_4080_cuda_tailnet_x86_64`
- 4080 control-plane profile:
  `local_tailnet_cluster_control_plane`
- 4080 compute source:
  `local_rtx4080_workstation`
- 4080 expected steps per second:
  `82.40252049829174`

## Admission Rule

The registration packet now acts as the first hard admission grammar for local
executor runs:

- missing profile ids fail validation
- missing mixture version ids fail validation
- missing eval-pack ids fail validation
- zero budget or zero observed duration fails validation
- missing batch geometry fails validation
- missing memory-headroom posture fails validation
- missing expected-throughput posture fails validation

That is the exact closure the issue required: the schema itself now blocks
incomplete run registration.

## Claim Boundary

This packet closes only the canonical registration vocabulary.

It does **not** yet claim:

- searchable ledger closure
- shared dashboard closure
- promotion auto-block closure
- roundtrip-green cluster closure

Those remain the next EPIC 4 issues.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_run_registration_fixtures`
- `cargo test -q -p psionic-train builtin_executor_local_cluster_run_registration_packet_is_valid -- --exact --nocapture`
- `cargo test -q -p psionic-train executor_local_cluster_run_registration_fixture_matches_committed_truth -- --exact --nocapture`
- `cargo test -q -p psionic-train missing_profile_id_blocks_admission -- --exact --nocapture`

# Psion Executor Local Profile Reference

> Status: canonical `PSION-0101` / `#706` record, updated 2026-03-30 after
> landing the first typed executor admitted-profile catalog in
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

The first landed profile is:

- `local_mac_mlx_aarch64`

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
- `fixtures/training/compute_sources/local_mlx_mac_workstation_v1.json`
- `scripts/check-swarm-mac-mlx-bringup.sh`
- `crates/psionic-train/src/swarm_mlx_bringup.rs`
- `crates/psionic-train/src/bin/swarm_mac_mlx_bringup.rs`

## Honest Boundary

This issue does not claim:

- the 4080 Tailnet worker is admitted yet
- the Mac profile alone closes remote launch
- shared checkpoint-writer authority on the Mac
- cross-device training closure

This issue closes only the first local profile so later EPIC 1 issues can add
the 4080 worker and the Mac-to-4080 control plane without inventing new
profile vocabulary.

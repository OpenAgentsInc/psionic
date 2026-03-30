# Psion Executor Phase-Two Pre-Flight Checklist

> Status: canonical `PSION-0601` / `#747` record, updated 2026-03-30 after
> landing the first phase-two pre-flight checklist packet for the admitted
> executor lane.

This document records the first retained phase-two pre-flight checklist packet
for executor long-run launch discipline.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_phase_two_preflight_checklist_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_phase_two_preflight_checklist_fixtures
```

## What Landed

`psionic-train` now owns one typed phase-two pre-flight checklist packet that
binds:

- the canonical local-cluster run-registration packet
- the admitted MLX decision-grade run type
- the admitted 4080 decision-grade run type
- seven explicit red launch blockers

The packet makes one operating rule durable:

- any red pre-flight item blocks launch for the admitted phase-two run type

## Current Retained Truth

- packet digest:
  `5c8f4c8448b396b5bc1b7def7e6216cf6e03f4e2ad61e363c0d83b68faec7068`
- registration packet digest:
  `dfad1972f358be079ddd80ac73f5ec85200c16e1e5a708fb11a18bc765cec229`
- blocking rule:
  `any_red_item_blocks_launch`
- checklist item count:
  `7`
- run type row count:
  `2`
- admitted run types:
  `mlx_decision_grade`, `cuda_4080_decision_grade`
- admitted profiles:
  `local_mac_mlx_aarch64`, `local_4080_cuda_tailnet_x86_64`
- frozen blocker categories:
  `device_health`, `paths`, `dry_batch`, `eval_attach`,
  `resume_rehearsal`, `ledger_fields`, `export_plan`

## Honest Current Meaning

This packet does not claim long runs are already routine.

It does make launch discipline non-optional:

- both admitted phase-two run types now point at the same blocker categories
- both run types are hard-blocked if any checklist item is still red
- the checklist is now anchored to the same registration truth the rest of the
  local-cluster lane already uses

That is the point of this issue. Launch readiness is no longer an operator note
or a remembered ritual.

The follow-on retained long-run rehearsal packet that now proves the admitted
4080 decision-grade lane actually entered the run under this contract lives at:

- `docs/PSION_EXECUTOR_LONG_RUN_REHEARSAL.md`
- `fixtures/psion/executor/psion_executor_long_run_rehearsal_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_phase_two_preflight_checklist_fixtures`
- `cargo test -q -p psionic-train psion_executor_phase_two_preflight_checklist -- --nocapture`

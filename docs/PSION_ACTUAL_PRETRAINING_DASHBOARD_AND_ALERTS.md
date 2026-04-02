# Psion Actual Pretraining Dashboard And Alerts

> Status: canonical retained dashboard and active-alert feed for the actual
> `Psion` pretraining lane, written 2026-04-02 after landing one operator-owned
> visibility packet above the existing status, checkpoint, and preflight
> surfaces.

This document records the retained observability surface that now exists for
`psion_actual_pretraining_v1`.

It does not introduce a streaming dashboard service. It binds the current
actual-lane operator state to one machine-readable dashboard packet plus one
machine-readable active-alert feed under the same retained evidence family the
launcher already owns.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_dashboard.rs` owns the
  typed dashboard and alert-feed contracts.
- `crates/psionic-train/examples/psion_actual_pretraining_dashboard_fixtures.rs`
  regenerates the committed dashboard fixtures and example run roots.
- `crates/psionic-train/examples/psion_actual_pretraining_operator.rs` writes
  the retained dashboard and alert feed during `start`, `resume`,
  `record-checkpoint`, and `backup`.
- `scripts/psion-actual-pretraining-dashboard.sh` is the canonical operator
  reader for the retained dashboard surface.
- `fixtures/psion/pretrain/psion_actual_pretraining_dashboard_v1.json` is the
  canonical dashboard fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_alert_feed_v1.json` is the
  canonical active-alert feed fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_dashboard_example/`
  contains success and alerted example run roots.

Stable schema versions:

- `psion.actual_pretraining_dashboard.v1`
- `psion.actual_pretraining_alert_feed.v1`

## Retained Paths

Every actual-lane run root now reserves:

- `dashboard/current_dashboard.json`
- `alerts/active_alerts.json`

These sit alongside the existing retained status files:

- `status/current_run_status.json`
- `status/retained_summary.json`

and the existing retained alert receipt:

- `alerts/latest_redacted_alert.json`

The redacted alert remains the per-trigger receipt. The active-alert feed is
the aggregate operator surface that the dashboard points at.

## Dashboard Packet

The retained dashboard packet now carries:

- current phase
- selected git ref and exact git commit SHA
- dirty-tree admission posture
- throughput health against the trusted-cluster anchor
- explicit loss visibility state
- gradient visibility state from the frozen distributed qualification
- latest checkpoint label, backup posture, and checkpoint-eval posture
- hardware health summary from the retained hardware qualification
- active-alert summary pointing at one response runbook

The current actual-lane dashboard keeps two absences explicit:

- loss remains `not_emitted_by_actual_lane_operator` once training has started
- gradient health is `qualified_reference_only`, not a live per-step stream

That keeps the operator surface honest instead of inventing missing telemetry.

## Active Alert Kinds

The aggregate active-alert feed currently supports:

- `checkpoint_eval_retry_required`
- `checkpoint_backup_refused`
- `hardware_health_refused`
- `hardware_worker_unhealthy`
- `throughput_degraded`

The feed only retains declared source paths, redacted source names, and short
operator summaries. It does not copy raw credentials, raw SSH targets, or
service-account payloads into retained artifacts.

## Operator Command

```bash
./TRAIN --lane actual_pretraining dashboard --run-root <path>
```

This is a thin wrapper over `scripts/psion-actual-pretraining-dashboard.sh`.
It prints the current phase, git provenance, throughput posture, checkpoint
posture, hardware health, and any active alerts from the retained dashboard
packet and alert feed.

## Response Runbook

Use the retained dashboard as the first operator stop:

- If throughput is `degraded`, inspect
  `preflight/run_shape_qualification.json` before continuing a long run.
- If hardware health is `degraded` or `refused`, inspect
  `preflight/hardware_qualification.json` before launch or resume.
- If checkpoint backup is refused, inspect
  `checkpoints/latest_accepted_checkpoint_backup_receipt.json` and the failure
  drill under `checkpoints/failures/`.
- If checkpoint eval needs retry, inspect
  `evals/latest_checkpoint_eval_failure.json` and
  `alerts/latest_redacted_alert.json`.

## Claim Boundary

This surface now proves:

- the actual lane has one retained operator dashboard packet
- the actual lane has one retained aggregate active-alert feed
- the launcher updates that surface from real retained status, checkpoint,
  preflight, backup, and checkpoint-eval artifacts

It does not yet prove:

- external alert delivery or paging
- a cluster-connected streaming dashboard
- live loss or live gradient streams beyond the retained current contracts

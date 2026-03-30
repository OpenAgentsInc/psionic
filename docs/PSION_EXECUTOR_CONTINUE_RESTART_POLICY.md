# Psion Executor Continue-Vs-Restart Incident Policy

> Status: canonical `PSION-0602` / `#748` record, updated 2026-03-30 after
> landing the first continue-vs-restart incident policy packet for the
> admitted executor lane.

This document records the first retained incident-policy packet for executor
long-run operating discipline.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_continue_restart_policy_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_continue_restart_policy_fixtures
```

## What Landed

`psionic-train` now owns one typed continue-vs-restart policy packet that
binds:

- the canonical weekly local-cluster review workflow
- the named executor ownership surface
- six canonical incident classes
- one required review-logging rule for all six classes

The packet makes one operating rule durable:

- incidents are no longer handled ad hoc; every retained incident class now has
  a default action, owner role, required evidence set, and weekly-review
  logging requirement

## Current Retained Truth

- packet digest:
  `40ef44b2c92651ab7e5ae3c7c039388bf472ca05467fc679054ffe4fb8413186`
- review workflow digest:
  `c11b48bb9cb4381ccba810b5c154ffad6014c3b130c539a32b43dff4298078bf`
- ownership ref:
  `docs/PSION_EXECUTOR_OWNERSHIP.md`
- incident class count:
  `6`
- launch drift:
  `restart_after_preflight_recheck`
- transient interruption:
  `continue_from_last_green_checkpoint`
- missing facts:
  `hold_until_facts_repaired`
- throughput degradation:
  `continue_under_review_no_promotion`
- non-finite loss:
  `restart_from_last_green_checkpoint`
- export failure:
  `stop_and_repair_before_review`

## Honest Current Meaning

This packet does not claim the lane is already stable under every incident.

It does make the handling discipline explicit:

- launch drift is forced back through the pre-flight checklist instead of being
  hand-waved
- transient interruption is allowed to continue only from the last green
  checkpoint
- missing facts and export failure are hard holds, not best-effort notes
- throughput degradation can continue, but only under review and without a
  promotion claim

The review surface is now explicit too. Every incident class in the packet
requires logging in the weekly local-cluster review path.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_continue_restart_policy_fixtures`
- `cargo test -q -p psionic-train psion_executor_continue_restart_policy -- --nocapture`

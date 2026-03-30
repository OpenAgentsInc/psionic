# Psion Executor 4080 Frequent Eval Attachment

> Status: canonical `PSION-0303` / `#727` record, updated 2026-03-30 after
> landing the first retained frequent-pack checkpoint-eval attachment packet for
> the admitted Mac -> 4080 Tailnet executor lane.

This document records the first retained packet that attaches checkpoint-time
frequent-pack review to the admitted 4080 lane instead of letting later smoke
or decision-grade review depend on missing eval facts.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_4080_frequent_eval_attachment_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_4080_frequent_eval_attachment_fixtures
```

## What Landed

`psionic-train` now owns one typed frequent-eval attachment packet that binds:

- the prerequisite durable-checkpoint packet
- the frozen executor eval-pack catalog
- the retained admitted rerun bundle
- one per-checkpoint ledger row keyed to the retained pointer digest
- an explicit promotion block whenever frequent-pack coverage is missing or
  unscored

That means the admitted 4080 lane now has one explicit packet for:

- automatic frequent-pack ledger attachment at checkpoint time
- suite-by-suite visibility into what is green versus what is still unscored
- the rule that missing or unscored frequent-pack surfaces block later
  promotion instead of silently disappearing

## Current Retained Truth

- packet digest:
  `b61245ffc124b3a8fd9ec3e15bee20782fe4538f30a4595829e735bdbc360fb7`
- prerequisite durable-checkpoint packet SHA256:
  `c1a983ae9eeb6bfdcd1fbd9f01cc13a3e2976c363e5d9035cc4268f84255c1f6`
- eval-pack catalog SHA256:
  `b6b4683128af9eb7d2eb901bfa4a3cba10247c21858d8f096a01fc593c679d5f`
- eval-pack catalog digest:
  `d0f7e608b7be0e39fc319e12033a230dbb6953ccd550978c93193e895508ad2d`
- frequent-pack digest:
  `aa61897f9b4731421865a9cb38f2633ccbbd2fc60dcb140fb567ae21d14c73e8`
- retained run bundle SHA256:
  `35c4cdaba64e5b4b235e3d1f77cc506cc74ec6bcf9a8f8eb9309bf9441e0bc83`
- run id:
  `tailrun-home-admitted-20260328k`
- ledger row id:
  `psion.executor.4080.frequent_eval_row:tailrun-home-admitted-20260328k:dd1aa85c355eb43934a20afe7e4204b3ed82bb85f4fe392dfde45229f4e434f8`
- checkpoint family:
  `swarm.local.open_adapter.policy:tailrun-home-admitted-20260328k`
- checkpoint pointer digest:
  `dd1aa85c355eb43934a20afe7e4204b3ed82bb85f4fe392dfde45229f4e434f8`
- checkpoint ref:
  `checkpoint://swarm/first-swarm-live-plan/policy`
- checkpoint step:
  `12`
- ledger row digest:
  `7b9b1159894d59c216bf4e8c069036a9c2820b9f949a7298d13ba93d4a52eaa4`
- missing eval blocks promotion:
  `true`
- promotion blocker ids:
  `frequent_exactness_cases_v0`,
  `frequent_held_out_exclusions_v0`,
  `frequent_throughput_blockers_v0`

## Retained Suite Status

- `frequent_exactness_cases_v0`: `blocked_missing_executor_outputs`
- `frequent_held_out_exclusions_v0`: `blocked_missing_executor_outputs`
- `frequent_operator_review_cases_v0`: `green`
- `frequent_throughput_blockers_v0`: `blocked_missing_executor_metrics`

## Retained Operator-Review Case Status

- `artifact_packet_complete`: `green`
- `checkpoint_restore_rehearsal_green`: `green`
- `export_smoke_green`: `green`
- `local_cluster_roundtrip_green`: `green`

## Claim Boundary

This packet counts as the first admitted **automatic frequent-pack attachment**
for the 4080 lane.

It does **not** claim:

- full executor exactness scoring on the retained open-adapter infrastructure run
- held-out exclusion scoring on executor outputs that do not yet exist here
- admitted executor throughput-blocker scoring on this retained rerun
- decision-grade readiness by itself

Instead it makes the missing or unscored suites explicit and keeps them as hard
promotion blockers.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_4080_frequent_eval_attachment_fixtures`
- `cargo test -q -p psionic-train psion_executor_4080_frequent_eval_attachment -- --nocapture`

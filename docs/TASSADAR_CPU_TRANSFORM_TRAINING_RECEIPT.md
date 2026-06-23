# Tassadar Percepta CPU-Transform Training Receipt

> Status: Psionic-side execution-evidence lane for the
> `models.tassadar_percepta_executor.v1` product promise (EPIC DE-5, openagents
> issue #5528). Added the first dereferenceable CPU-transform training receipt.

This document records the Psionic-owned execution-evidence boundary for the
Tassadar Percepta CPU-transform training receipt gate. It is the
"verified work, not a claim" half of the promise. The openagents public status
route
`/api/public/models/tassadar-percepta-executor/cpu-transform-training-receipts`
reports the same gate as missing-in-production until a real Pylon-dispatched
receipt with settlement exists; this lane produces the local, deterministic
proof shape that route is waiting for.

## Canonical Fixture

- `fixtures/tassadar/operator/tassadar_percepta_cpu_transform_training_receipt_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train \
  --example tassadar_percepta_cpu_transform_training_receipt_fixtures
```

## What Landed

`psionic-train` now owns one typed, dereferenceable CPU-transform training
receipt (`TassadarCpuTransformTrainingReceipt`,
`tassadar_cpu_transform_training_receipt.rs`). The generator:

1. Runs a real, deterministic, CPU-only Tassadar executor transformer training
   rehearsal (`train_tassadar_executor_transformer`,
   `TassadarExecutorTrainingConfig::reference()`) over the frozen SudokuV0
   verified-trace manifest.
2. Replays the selected checkpoint against CPU-reference truth, producing a
   per-case exact-trace / final-output / halt verdict by first divergence
   (verification class `exact_trace_replay`).
3. Emits a public-safe receipt that binds the training-manifest digest, trained
   model descriptor + weight digests (the trained-artifact digest), the
   training-report digest, and the exact-replay verifier verdict, with explicit
   gate state.

The receipt schema version is
`openagents.models.tassadar_percepta_executor.cpu_transform_training_receipt.v1`
and the receipt reference is
`receipt.models.tassadar_percepta_executor.cpu_transform_training.local_cpu_transform_rehearsal_v1`,
which slots into the openagents projection's expected pattern
`receipt.models.tassadar_percepta_executor.cpu_transform_training.{assignmentRef}`.

## Gate State (honest boundary)

Locally provable (satisfied):

- `cpu_transform_training_completed`
- `exact_replay_verifier_verdict_local`
- `trained_artifact_digest_present`

Compute / owner gated (must stay unsatisfied; a validation invariant forbids a
local run from flipping any of these):

- `pylon_assignment_receipt` — no real Pylon assignment record
- `accepted_work_receipt` — no accepted-work closeout
- `real_settlement_receipt` — no money moved
- `green_promise_transition` — no green transition

## Honest Meaning

The bounded `reference()` baseline is a weak 1-epoch run; the exact-replay
verdict honestly records that the model does NOT replay the validation traces
exactly (`exact_trace_case_count` is 0 of 2 in the committed fixture). The value
of this receipt is the **verified work and dereferenceable proof shape** — a
real training run plus a real exact-replay verdict and a trained-artifact digest
— not a passing model.

This receipt does NOT claim:

- a trained Tassadar product model
- accepted Pylon work or verifier-accepted paid work
- real settlement where money moved
- model promotion, hosted inference, CPU replacement, or CPU outperformance
- a green transition for `models.tassadar_percepta_executor.v1`

The promise stays `planned`. The promise informs but does not clear
`blocker.product_promises.pylon_v03_cpu_transform_training_receipts_missing`.

## The Single Remaining Compute / Owner Gate

Dispatch this CPU-transform training as a real Pylon assignment (owner arms
compute / spend), let it produce accepted-work + exact-replay verdict + real
settlement receipts where money moves, then take the receipt-first green upgrade
under `proof.claim_upgrade_receipts.v1` with owner sign-off. No local code change
is needed to reach green other than swapping the local rehearsal assignment
reference for the real dispatched-assignment receipt set.

## Validation

- `cargo run -q -p psionic-train --example tassadar_percepta_cpu_transform_training_receipt_fixtures`
- `cargo test -q -p psionic-train tassadar_cpu_transform_training_receipt`

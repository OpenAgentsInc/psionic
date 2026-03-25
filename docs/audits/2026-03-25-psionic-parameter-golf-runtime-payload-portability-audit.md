# Parameter Golf Runtime Payload Portability Audit

Date: 2026-03-25

## Scope

This audit records the current shipped Linux replay-runtime posture for the
Parameter Golf exported folder and the real RunPod evidence that forced the
portable fixture refresh.

This is narrower than distributed-trainer closure. It only answers one
question:

- does the committed `runtime/parameter_golf_submission_runtime` payload run on
  the real RunPod Ubuntu image used by the current `8xH100` lane?

## Problem

The first official real `8xH100` launcher run on
`157.66.254.12:19362` failed before the distributed bring-up boundary because
the committed Linux replay payload required `GLIBC_2.39`.

That failure happened in the retained run root:

- `/workspace/parameter-golf-runpod/parameter-golf-runpod-real-8xh100-20260325T015234Z`

This was not a distributed-trainer failure. It was a shipped-runtime
portability failure.

## Observed Real-Hardware Facts

The same exported folder was rerun on the same real `8xH100` pod with a
portable stripped Linux binary built on the target host.

Retained real-hardware probe:

- `/workspace/parameter-golf-runpod/parameter-golf-runpod-debugstrip-runtime-check-20260325T023809Z`

Observed execution boundary from that probe:

- the launcher reached the shipped Rust runtime
- the runtime admitted all `8` H100s:
  `matching_h100_device_count=8`
- the runtime preserved the exact distributed bring-up report under the
  exported folder
- the runtime then refused for the correct remaining reason: the real
  distributed `WORLD_SIZE=8` trainer payload still does not exist

That means the portable runtime fixture fixes the real host-compatibility bug
without pretending that `#473` is closed.

## Current Shipped Posture

The committed Linux replay payload is now the portable stripped binary shipped
at:

- `fixtures/parameter_golf/runtime/parameter_golf_submission_runtime.x86_64-unknown-linux-gnu`

Current fixture identity:

- SHA-256:
  `925973482624f88a72c0d2ff11ab06780be4907da82c59642b52745d84cfe9fd`
- size: about `3.5 MiB`

This payload is honest for the current non-record exported-folder lane because:

- it is prebuilt and shipped inside the folder
- its bytes still count toward the exported artifact accounting surface
- it has already been exercised on the real RunPod Ubuntu image used by the
  live `8xH100` lane

## Release-Build Boundary

The pod-local debug build admitted the real H100 inventory correctly. The
pod-local release build on the same source tree still showed the earlier
zero-match H100 admission behavior during investigation.

That means the repo should currently treat the portable stripped payload as the
known-correct shipped Linux replay runtime and should not silently replace it
with a release-profile artifact until the release-only discrepancy is explained
and revalidated on real hardware.

## What This Fixes

- official exported-folder execution on the real RunPod `8xH100` image no
  longer depends on a manual binary swap outside the repo to get past the libc
  boundary
- the operator lane can now preserve the intended explicit distributed-trainer
  refusal instead of failing earlier on a host-libc mismatch
- the canonical PGOLF reports can now bind the portable runtime bytes and
  digest directly

## What It Does Not Fix

- it does not add the real distributed `WORLD_SIZE=8` trainer payload
- it does not produce measured distributed training receipts
- it does not close `#473`
- it does not help the single-H100 validation runtime bottleneck

## Required Follow-On

1. Keep the portable replay-runtime fixture as the shipped Linux default until
   a release-profile replacement is proven equivalent on the real RunPod image.
2. Refresh every canonical PGOLF report that binds the replay-runtime digest or
   counted code bytes in the same change as the fixture update.
3. Continue `#473` on the actual missing boundary:
   the real Rust-native distributed trainer and its measured receipts.

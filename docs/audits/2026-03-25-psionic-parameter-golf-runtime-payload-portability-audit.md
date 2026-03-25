# Parameter Golf Runtime Payload Portability Audit

Date: 2026-03-25

## Scope

This audit records the current shipped Linux replay-runtime posture for the
Parameter Golf exported folder and the real RunPod evidence that forced the
portable fixture refreshes.

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

That was not a distributed-trainer failure. It was a shipped-runtime
portability failure.

After the first fixture refresh, a later real launcher run on
`157.66.254.12:16223` reached the exported-folder runtime but still failed at
execution because the shipped fixture binary had drifted behind current source.

That retained run root is:

- `/workspace/parameter-golf-runpod/parameter-golf-runpod-scoreproof-20260325T081200Z`

Observed failure boundary from that run:

- remote preflight passed
- pre-training built and staged the exported folder successfully
- execution still printed the older bring-up-era refusal:
  `the real distributed trainer payload still has not landed`

That message no longer matches current source. The exported folder was still
copying the older committed fixture binary instead of the later source-level
distributed runtime path. This was a shipped-fixture freshness failure.

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

That proved the host-compatibility fix without pretending that `#473` was
closed.

The fixture was then rebuilt again from current `main` on the real RunPod pod
after the distributed source widened beyond bring-up only.

## Current Shipped Posture

The committed Linux replay payload is now the portable stripped binary shipped
at:

- `fixtures/parameter_golf/runtime/parameter_golf_submission_runtime.x86_64-unknown-linux-gnu`

Current fixture identity:

- SHA-256:
  `75c0a72c000e76f639f352d8c83025f0b5b43987ffca0fc6e9de9ebb3795048d`
- size: `6,214,696` bytes

Current fixture provenance:

- rebuilt on the real RunPod `8xH100` host at `157.66.254.12:16223`
- source revision: `8a5638d1`
- source binary: `target/debug/parameter_golf_submission_runtime`
- stripped target copied back into the committed fixture path
- the canonical local review-host reports were then regenerated with
  `LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12` so the same shipped bytes
  remained runnable on the maintainer workstation without changing the fixture
  again

This payload is honest for the current non-record exported-folder lane because:

- it is prebuilt and shipped inside the folder
- its bytes still count toward the exported artifact accounting surface
- it has already been exercised on the real RunPod Ubuntu image used by the
  live `8xH100` lane
- it no longer lags behind the current distributed-runtime source tree
- the committed evidence and PR-bundle reports now bind the same RunPod-built
  runtime digest and size across the exported-folder, replay-verification, and
  record-track contract surfaces

## Release-Build Boundary

The pod-local debug build admitted the real H100 inventory correctly. The
pod-local release build on the same source tree still showed the earlier
zero-match H100 admission behavior during investigation.

That means the repo should currently treat the portable stripped debug payload
as the known-correct shipped Linux runtime for the exported folder and should
not silently replace it with a release-profile artifact until the release-only
discrepancy is explained and revalidated on real hardware.

## What This Fixes

- official exported-folder execution on the real RunPod `8xH100` image no
  longer depends on a manual binary swap outside the repo to get past the libc
  boundary
- the exported folder now ships the current distributed-runtime source instead
  of an older bring-up-era fixture binary
- the operator lane can now preserve the intended explicit distributed-trainer
  execution boundary instead of failing earlier on a host-libc mismatch or
  stale fixture drift
- the canonical PGOLF reports can now bind the portable runtime bytes and
  digest directly

## What It Does Not Fix

- it does not implement the real `600` second multi-step distributed challenge
  loop
- it does not by itself produce a measured end-to-end distributed score
- it does not close `#473`
- it does not help the single-H100 validation runtime bottleneck

## Required Follow-On

1. Keep the portable replay-runtime fixture as the shipped Linux default until
   a release-profile replacement is proven equivalent on the real RunPod image.
2. Re-run the real `8xH100` launcher immediately after each runtime fixture
   refresh so the repo proves the exported folder is actually executing the
   current shipped binary.
3. Continue `#473` and `#543` on the actual missing boundaries:
   the real Rust-native distributed training loop, measured receipts, and a
   challenge-speed score.

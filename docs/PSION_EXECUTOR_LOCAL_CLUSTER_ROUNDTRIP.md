# Psion Executor Local Cluster Roundtrip

> Status: canonical `PSION-0406` / `#739` record, updated 2026-03-30 after
> proving the first full Mac -> 4080 -> Mac local-cluster roundtrip for the
> admitted executor lane.

This document records the first retained roundtrip closeout packet for the
executor local-cluster lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_local_cluster_roundtrip_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_local_cluster_roundtrip_fixtures
```

## What Landed

`psionic-train` now owns one typed roundtrip-closeout packet that binds:

- the canonical local-cluster run-registration packet
- the retained Mac -> 4080 Tailnet launch packet
- the retained 4080 durable-checkpoint packet
- the retained 4080 decision-grade run packet
- the retained 4080 recovery packet
- the retained Mac export-inspection and CPU validation packet

The packet keeps six required green steps explicit:

- launch on Mac
- train on 4080
- checkpoint on 4080
- recover on 4080
- export back to Mac
- validate back on Mac

That means local-cluster closure no longer depends on reading five older
packets and inferring that the loop finished. The closeout is now one retained
artifact with one machine-readable answer.

## Current Retained Truth

- packet digest:
  `820e605be48dfd4acdef6e1de3e5cd59972c0c7de0894b83f20343a9860f8299`
- registration packet digest:
  `dfad1972f358be079ddd80ac73f5ec85200c16e1e5a708fb11a18bc765cec229`
- remote-launch packet digest:
  `19aef8ffcf62006272d40206793d22f031a064a63dbc1254ad69ccd1351f4158`
- durable-checkpoint packet digest:
  `f8b326e0eb2ae45a3b7bafc553dfa4119074fd9e7e4c0d318c5bf52f2d03f0b5`
- decision-grade packet digest:
  `3e406cd85a6bf918fe0e219dc2e77f4801750798bc16aa28ca355457e44ae8f5`
- recovery packet digest:
  `d07f14dd64ce0f66d8827a9de1c6353dd5f1d001a9c81bc74669bc12e2def2c6`
- Mac validation digest:
  `c5460eb2187b91ee61f048e7a021a27984a3c65c47bc84cdf1fd6c8571797140`
- current-best registration id:
  `psion_executor_local_cluster_registration_4080_v1`
- current-best run id:
  `tailrun-home-admitted-20260328k`
- shared run-search key:
  `tailrun-admitted-device-matrix-20260327b`
- export bundle ref:
  `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/archlinux_cuda/portable_bundle.safetensors`
- export bundle sha256:
  `5e3b708051e6ccb86d9efe5e13f4ab425b3ec1bb74c28f6cc8f2b44a49129f7f`
- Mac validation machine class:
  `host_cpu_aarch64`
- reference anchor metric:
  `tassadar.reference_linear_steps_per_second`
- fast-route metric:
  `tassadar.hull_cache_steps_per_second`
- minimum `hull_cache` speedup over `reference_linear`:
  `1.690977509006`
- maximum remaining CPU gap:
  `2.683604159673`
- phase exit green:
  `true`
- cluster closure status:
  `green`

## Six-Step Closure

All six retained steps are now green:

- `launch_on_mac_green`
- `train_on_4080_green`
- `checkpoint_on_4080_green`
- `recover_on_4080_green`
- `export_back_to_mac_green`
- `validate_back_on_mac_green`

## Honest Current Meaning

This closes EPIC 4 cluster closure.

It does **not** claim promotion readiness.

What is now true:

- the Mac -> 4080 -> Mac loop is retained and green
- the current-best accelerator row can now be marked export-green in the
  ledger and dashboard
- phase exit no longer needs a review-memory exception

What is still true:

- promotion remains blocked until the admitted frequent-pack eval truth turns
  green
- `reference_linear` still remains the measured truth anchor outside the
  admitted fast-route envelope

## Follow-On Surfaces

The roundtrip packet now feeds directly into:

- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_LEDGER.md`
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_DASHBOARD.md`
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_AUTOBLOCKS.md`
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_REVIEW_WORKFLOW.md`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_roundtrip_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_roundtrip -- --nocapture`

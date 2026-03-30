# Psion Executor Mac Export Inspection

> Status: canonical `PSION-0205` / `#724` record, updated 2026-03-30 after
> landing the first retained Mac-local export inspection and CPU-route
> validation packet.

This document records the first Mac-local packet that validates export
inspection, CPU-route posture, and the `reference_linear` versus `hull_cache`
claim boundary on the admitted executor lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_mac_export_inspection_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_mac_export_inspection_fixtures
```

## What Landed

`psionic-train` now owns one typed Mac export-inspection packet that binds:

- the retained MLX-local decision-grade packet
- the retained MLX portable bundle under
  `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/m5_mlx/portable_bundle.safetensors`
- the canonical model-IO deferred import plan
- the emitted torch-style compatibility artifact and roundtrip import
- the retained CPU reproducibility report
- the retained fast-route implementation report
- the retained fast-route throughput-floor report
- the retained hull-cache closure report
- the retained replacement publication

That means the admitted Mac profile now has one reviewable local packet for:

- import planning on the retained MLX bundle
- portable-bundle import and state-dict roundtrip
- CPU `host_cpu_aarch64` route validation
- local recheck of the `reference_linear` truth anchor
- local recheck of the admitted `hull_cache` fast-route floor
- replacement-publication continuity before broader promotion claims

## Current Retained Truth

- packet digest:
  `9d6a39d78400f4a0c6c86398b677b9880080e8351653b3f68ccadb6e4a06aa8a`
- deferred import-plan digest:
  `8932fc2305dd3cf091902e163bb70a74c0e380f707af56a3c53718ad43cb2ccd`
- compatibility-contract digest:
  `fba40c5402d57ca5efbaad21f8b93b663328377e1d6c5bad5fef2449376ee6d1`
- torch-style compatibility artifact digest:
  `88006c8f787eb0e031a9e413778659e701689a7f217b669b6525d129ec45f6b1`
- CPU reproducibility report digest:
  `7d1122039677a89f2d7e1079e78cd895212105f3e12b0af8a27ab6921a09663f`
- fast-route implementation report digest:
  `c6303163853b87aae3c00109afdbf93dfa69af46af2d133cb74fb9b3b9a7d8ad`
- fast-route throughput-floor report digest:
  `b500d330f5146399b4b49f054e8ebd45aa584707f8f66b0a3433424ccbfa086d`
- hull-cache closure report digest:
  `582b36210e020462e1d52844d3de28aff0c7beed5b1b73eb856af1a110631c9b`
- replacement-publication digest:
  `75119558c8435d0e6681c3192acff385943e15b1451eb296bdba478e528b5ea1`
- local CPU machine class:
  `host_cpu_aarch64`
- minimum retained `hull_cache` speedup over `reference_linear`:
  `1.690977509006`
- maximum retained `hull_cache` remaining gap versus CPU reference:
  `2.683604159673`
- retained checklist row count:
  `6`

## Retained Checklist Rows

- `portable_bundle_import_plan_green`
- `portable_bundle_roundtrip_green`
- `cpu_aarch64_route_green`
- `reference_linear_anchor_green`
- `hull_cache_fast_route_green`
- `replacement_publication_green`

## Claim Boundary

This packet counts as the **local Mac export inspection and CPU-route
validation receipt** for the admitted Mac profile.

It does **not** claim:

- broader promotion closure by itself
- Mac -> 4080 -> Mac roundtrip truth
- independent replacement authority without the promotion packet
- that the `hull_cache` fast path can erase the `reference_linear` truth anchor

The retained claim boundary stays explicit:

- `reference_linear` remains the measured baseline truth anchor
- `hull_cache` remains the admitted fast-route target on the executor workload
  family
- throughput wins do not override exactness, held-out, runtime, or serving
  blockers

## Validation

- `cargo run -q -p psionic-train --example psion_executor_mac_export_inspection_fixtures`
- `cargo test -q -p psionic-train psion_executor_mac_export_inspection -- --nocapture`

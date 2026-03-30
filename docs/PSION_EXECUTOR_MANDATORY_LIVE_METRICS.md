# Psion Executor Mandatory Live Metrics

> Status: canonical `PSION-0603` / `#749` record, updated 2026-03-30 after
> landing the first mandatory live-metrics packet for the admitted executor
> lane.

This document records the first retained live-metrics packet for executor
long-run observability.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_mandatory_live_metrics_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_mandatory_live_metrics_fixtures
```

## What Landed

`psionic-train` now owns one typed mandatory live-metrics packet built
directly on top of:

- the canonical local-cluster ledger
- the canonical local-cluster dashboard
- the canonical local-cluster registration packet
- the retained source-family contribution report
- the retained interruption-recovery packet

The packet freezes one required metric set for the retained MLX candidate row
and the retained 4080 current-best row.

## Current Retained Truth

- packet digest:
  `ed90d86a315b4c37427dcbc4353f6113cacc89862ffdd3ceefa8a7161c2d04c6`
- ledger digest:
  `618605effd540810a884fb6797bee683327033cdaae3e79fa5ab0fec51b7b63c`
- dashboard digest:
  `026da39b01fff5eb4e93025f0a39ad5356c4d8368e603b34b3690e16b140ee28`
- required metric count:
  `11`
- MLX row:
  `psion_executor_local_cluster_ledger_row_mlx_v1`
- MLX tokens per second:
  `4649676`
- MLX step latency ms:
  `6`
- MLX checkpoint latency ms:
  `384`
- MLX recovery latency ms:
  `1`
- 4080 row:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- 4080 tokens per second:
  `2357371`
- 4080 step latency ms:
  `12`
- 4080 checkpoint latency ms:
  `768`
- 4080 recovery latency ms:
  `5000`
- exactness delta bps:
  `0`
- held-out delta bps:
  `0`
- gradient posture:
  `retained_nominal_band`, `not_triggered`
- device and thermal posture:
  `green_nominal`, `green_no_throttle`

## Honest Current Meaning

This packet does not claim the lane has full production telemetry yet.

It does make the minimum operator metric set explicit and durable:

- long-run review no longer stops at loss and raw throughput
- both retained rows now carry latency, headroom, recovery, and health posture
  in one surface
- the packet keeps exactness and held-out deltas visible even when the current
  retained value is flat

That is the point of this issue. The lane now has one explicit answer to what
counts as mandatory live metrics instead of leaving the metric set implicit.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_mandatory_live_metrics_fixtures`
- `cargo test -q -p psionic-train psion_executor_mandatory_live_metrics -- --nocapture`

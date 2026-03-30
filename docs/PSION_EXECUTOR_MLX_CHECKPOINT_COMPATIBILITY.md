# Psion Executor MLX Checkpoint Compatibility

> Status: canonical `PSION-0202` / `#721` retained MLX checkpoint
> compatibility packet for the executor lane, updated 2026-03-30.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_mlx_checkpoint_compatibility_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_mlx_checkpoint_compatibility_fixtures
```

## What Landed

The executor lane now has one typed MLX checkpoint compatibility packet built
from the retained same-node MLX training/export lane:

- retained report:
  `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/m5_mlx/report.json`
- retained bundle:
  `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/m5_mlx/portable_bundle.safetensors`
- prerequisite parity packet:
  `fixtures/psion/executor/psion_executor_mlx_forward_load_parity_v1.json`

The packet keeps three truths explicit:

- MLX same-node runs write a checkpoint family that survives into the retained
  portable bundle
- the portable bundle imports through the shipped deferred plan and eager
  restore paths instead of ad hoc glue
- model-IO compatibility metadata is ledger-visible rather than buried inside
  one binary artifact

## Retained Checkpoint Truth

The committed packet keeps the first executor-lane MLX checkpoint facts visible:

- admitted profile id: `local_mac_mlx_aarch64`
- backend label: `open_adapter_backend.mlx.metal.gpt_oss_lm_head`
- checkpoint family: `benchmark.open_adapter.same_node.mlx.retained`
- retained run id: `same-node-wallclock-retained-mlx`
- retained completed steps: `93184`
- restored state-dict digest:
  `8e4bdfd3cd6c7a99cc574725a53b3e4e30a7b43e9813139be06e0b516b3065e8`

## Compatibility Boundary

Phase-one checkpoint compatibility is intentionally narrow and explicit.

The packet proves:

- deferred import-plan metadata is stable enough for ledger packets
- eager restore reconstructs canonical training groups
- the bundle carries an explicit compatibility contract instead of opaque
  checkpoint claims

It does not claim blanket executor-model checkpoint parity across every future
MLX training recipe.

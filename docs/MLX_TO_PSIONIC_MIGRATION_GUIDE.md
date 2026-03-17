# Psionic MLX To Psionic Migration Guide

> Status: canonical `PMLX-608` / `#3873` reference record, updated 2026-03-17
> after landing the bounded MLX naming facade, C ABI, migration examples, and
> adoption guide in `psionic-mlx-compat`, `psionic-mlx-capi`, and
> `psionic-compat`.

This guide closes the bounded MLX adoption story for Psionic.

It does not claim MLX-identical Python behavior, full upstream package parity,
or a second execution stack hidden behind wrappers. It explains how to move
bounded MLX-shaped code and workflows onto the Rust-native Psionic substrate
without losing receipt, refusal, or replay truth.

Read this together with:

- `docs/MLX_COMPATIBILITY_SCOPE.md`
- `docs/MLX_COMPATIBILITY_MATRIX.md`
- `docs/MLX_ACCEPTANCE_MATRIX.md`
- `docs/ARCHITECTURE.md`

## Choose One Migration Path

| Starting point | Use this first | Drop to native Psionic when you need | Notes |
| --- | --- | --- | --- |
| MLX-shaped array, transform, `nn`, optimizer, `.mlxfn`, or distributed code | `psionic-mlx-compat` | `psionic-array`, `psionic-ir`, `psionic-compiler`, `psionic-nn`, `psionic-function-io`, `psionic-distributed` | Same execution path underneath; the facade is only a naming and module-layout shell. |
| C, Swift, or Python-ctypes caller that needs a small interop lane | `psionic-mlx-capi` | native Rust crates when the caller can move into Rust | Current C ABI is JSON-driven and intentionally thin. |
| Native Rust migration from MLX concepts into Psionic-owned crates | native Psionic crates directly | n/a | This is the long-term preferred path for reusable Psionic code. |

## Current Adoption Status

The canonical machine-readable truth still lives in
`docs/MLX_COMPATIBILITY_MATRIX.md`. The grouped tables below are the human
entrypoint.

### Supported Today

| Surface | What it means in practice | Primary entrypoint |
| --- | --- | --- |
| `governance_contracts` | version window, acceptance matrix, parity harness, and compatibility matrix are all runnable | `psionic-compat` |
| `seeded_transform_compile_export_parity_anchors` | seeded transform, compile, and export evidence exists | `psionic-ir`, `psionic-compiler`, `psionic-function-io` |
| `graph_first_function_export_bridge` | native `.psifn` export/import is real | `psionic-function-io` |
| `module_state_tree_bridge` | bounded module tree plus `save_weights` / `load_weights` is real | `psionic-nn` |
| `public_mlx_array_api` | bounded lazy-array surface with CPU plus explicit Metal/CUDA slices is real | `psionic-array` or `psionic-mlx-compat::core` |
| `public_mlx_transform_api` | bounded `grad`, `vmap`, `custom_vjp`, and compile-as-transform surface is real | `psionic-ir`, `psionic-compiler`, or `psionic-mlx-compat::transforms` |
| `public_mlx_nn_optimizer_api` | bounded `nn`, optimizer, scheduler, and quantized eval surface is real | `psionic-nn` or `psionic-mlx-compat::{nn, optimizers}` |
| `mlxfn_interop` | bounded `.mlxfn` import/export shell exists above native `.psifn` | `psionic-function-io` or `psionic-mlx-compat::io` |
| `mlx_naming_facade_and_bindings` | bounded MLX-like naming shell and one C ABI binding layer exist | `psionic-mlx-compat`, `psionic-mlx-capi` |
| `public_mlx_distributed_api` | bounded distributed group, collective, launch, tensor-parallel, and FSDP helper surface is real | `psionic-distributed` or `psionic-mlx-compat::distributed` |

### Convertible Today

| Surface | Current truth | Primary entrypoint |
| --- | --- | --- |
| `portable_model_io_bridge` | safetensors manifests and GGUF import can carry bounded migration paths, but this is not native MLX artifact parity | `psionic-array-io`, `psionic-function-io`, `psionic-models` |
| `mlx_package_ecosystem` | `psionic-mlx-lm`, `psionic-mlx-catalog`, `psionic-mlx-serve`, `psionic-mlx-vlm`, and `psionic-mlx-audio` now provide bounded local text-runtime, catalog, OpenAI-compatible text-serving, multimodal request-planning, and audio synthesis/codec workflows for GGUF paths, Ollama model ids, local Hugging Face cache snapshots, image/audio/video inputs, and CPU-reference speech requests, but recipe layers, synthetic/publish workflows, and benchmark packages remain later work | `psionic-mlx-lm`, `psionic-mlx-catalog`, `psionic-mlx-serve`, `psionic-mlx-vlm`, `psionic-mlx-audio`, `docs/MLX_LM_PACKAGE.md`, `docs/MLX_MODEL_CATALOG.md`, `docs/MLX_TEXT_SERVE.md`, `docs/MLX_VLM_PACKAGE.md`, `docs/MLX_AUDIO_PACKAGE.md` |

## Common Translation Map

| MLX-shaped concept | Bounded compatibility shell | Native Psionic owner |
| --- | --- | --- |
| array context, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `eval` | `psionic_mlx_compat::core::Context` | `psionic_array::ArrayContext` |
| `grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`, `custom_vjp` | `psionic_mlx_compat::transforms` | `psionic-ir` |
| compile-as-transform | `psionic_mlx_compat::transforms::compile_transform` | `psionic-compiler` |
| module tree, weights, quantized eval modules | `psionic_mlx_compat::nn` | `psionic-nn` |
| optimizers and schedulers | `psionic_mlx_compat::optimizers` | `psionic-nn` |
| `.mlxfn` import/export | `psionic_mlx_compat::io` | `psionic-function-io` |
| distributed helpers | `psionic_mlx_compat::distributed` | `psionic-distributed` |
| compatibility reports | `psionic_mlx_compat::reports` | `psionic-compat` |
| C/Swift/Python-ctypes interop | `psionic-mlx-capi` | optional shell only; runtime stays native |

## Example Suite

Run the migration examples from the repo root:

```bash
cargo run -p psionic-mlx-compat --example mlx_array_facade_walkthrough
cargo run -p psionic-mlx-compat --example mlx_native_drop_down
cargo run -p psionic-mlx-capi --example mlx_capi_eval_request
```

What each example demonstrates:

- `mlx_array_facade_walkthrough`
  - start from the bounded MLX-like facade
  - build and evaluate one deterministic array graph
  - inspect the compatibility matrix through the facade crate
- `mlx_native_drop_down`
  - start in `psionic-mlx-compat`
  - drop to the underlying `psionic-array::ArrayContext`
  - keep the same runtime truth and array values without switching execution paths
- `mlx_capi_eval_request`
  - send one JSON eval request through the bounded C ABI
  - receive owned JSON back with explicit status and typed refusal/error posture

The existing report runners remain the machine-readable companions:

```bash
cargo run -p psionic-compat --example mlx_compatibility_scope_report
cargo run -p psionic-compat --example mlx_compatibility_matrix_report
```

## Migration Pattern

1. Start with `psionic-mlx-compat` if you need a bounded MLX-like module layout.
2. Keep new reusable Rust code in the native crates once you touch a subsystem in earnest.
3. Use `psionic-mlx-capi` only when a non-Rust caller genuinely needs a narrow bridge.
4. Treat the compatibility matrix as the truth source for supported versus convertible versus unsupported claims.
5. Refuse or document unsupported surfaces explicitly instead of inventing shim behavior.

## What Not To Claim

- Do not call the facade or C ABI MLX-identical.
- Do not imply Python package parity from the current C ABI.
- Do not treat portable weight IO as proof of native MLX module compatibility.
- Do not imply the package ecosystem is done because `PMLX-701` through `PMLX-705` landed; `PMLX-706` through `PMLX-709` still bound the remaining ecosystem work.
- Do not hide typed refusal or bounded backend support behind optimistic fallback prose.

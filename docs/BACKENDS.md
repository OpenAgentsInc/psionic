# Psionic Backends

Psionic backends are explicit runtime implementations, not hidden environment
switches.

## Backend Inventory

- `psionic-backend-cpu`
  Reference backend for correctness, deterministic tests, and early smoke flows.
- `psionic-backend-metal`
  Apple GPU backend with a first model-backed embeddings product path.
- `psionic-backend-amd-kfd`
  AMD discovery/readiness backend aligned with the standard amdgpu/KFD posture.
- `psionic-backend-amd-userspace`
  AMD discovery/readiness backend for the higher-risk userspace driver posture.

## Backend Contract

Each backend is expected to report:

- backend identifier
- discovered devices
- supported dtypes
- runtime health/readiness
- supported execution modes and limitations

Shipped backend claims are bounded by
[HARDWARE_VALIDATION_MATRIX.md](./HARDWARE_VALIDATION_MATRIX.md). Do not claim a
backend/product lane as shipped unless it has a row in that matrix and the
referenced tests are green.

## Shared Conformance Harness

Psionic now keeps one shared backend conformance harness in
`crates/psionic-backend-tests`.

That harness currently covers:

- execution-backend discovery and selection truth
- a canonical `matmul + add` graph execution case
- one documented unsupported boundary per backend so the refusal posture stays
  explicit even when backend support sets differ
- array-context conformance for MLX-facing and array-facing contexts

The harness is wired into:

- `psionic-backend-cpu`
- `psionic-backend-cuda`
- `psionic-backend-metal`
- `psionic-mlx-compat`

Backend-specific tests still matter for kernel families, quantization modes,
and platform edge cases. The shared harness exists to stop the basic discovery,
selection, execution, and refusal story from drifting across backends.

## AMD Policy

AMD KFD and AMD userspace remain separate backends in both code and provider
capability reporting. They should not be collapsed into a single generic `amd`
mode because they have different operational and trust assumptions.

See [AMD.md](./AMD.md) for the operator runbook, readiness states, and current
phase boundary.

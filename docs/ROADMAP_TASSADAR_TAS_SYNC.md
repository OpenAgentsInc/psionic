# Psionic Tassadar TAS Sync

This file maps the public `TAS-*` GitHub issues in `OpenAgentsInc/psionic` to
the concrete repo-owned implementation and artifact surfaces that close them.
It exists so issue closure stays tied to landed code, tests, and committed
artifacts instead of comment-only summaries.

Ownership note: the canonical executor-trace schema is runtime-owned in
`psionic-runtime`, not `psionic-ir`, because it is an execution ABI and proof
surface rather than a generic graph/program IR contract.

## Implemented

| Issue | Status | Repo evidence |
| --- | --- | --- |
| `TAS-001` / `#65` | implemented | `psionic-runtime` now carries canonical typed trace-step, trace-artifact, trace-proof, and trace-diff surfaces, plus deterministic replay and JSON round-trip tests; `psionic-provider` now exposes provider-facing trace-artifact and trace-diff receipts above those runtime-owned artifacts. |
| `TAS-002` / `#66` | implemented | The CPU-reference executor harness already spans `psionic-runtime`, `psionic-eval`, and `psionic-environments`; the validation corpus now also includes an explicit bounded shortest-path fixture beside arithmetic, memory, branch, Sudoku, Hungarian, and broader Wasm-like workloads, so the public issue scope is covered by committed golden cases and parity helpers. |
| `TAS-003` / `#67` | implemented | `psionic-data` now carries a public `benchmark-package-set` contract for Tassadar families and reporting axes; `psionic-environments` now binds benchmark-package-set metadata into each benchmark surface; `psionic-eval` now materializes a repo-facing summary artifact at `fixtures/tassadar/reports/tassadar_benchmark_package_set_summary.json` plus a public runner at `cargo run -p psionic-eval --example tassadar_benchmark_package_set_summary`, and the validation benchmark package now explicitly covers the seeded CLRS shortest-path fixture instead of silently folding it into the older microprogram set. |
| `TAS-004` / `#68` | implemented | `psionic-models` now owns a public `TassadarWorkloadCapabilityMatrix` schema with explicit workload classes, support posture, benchmark gates, and typed refusal reasons, and the current fixture plus learned research lanes materialize that matrix directly from their model descriptors. `psionic-serve` now exposes a benchmark-gated `TassadarExecutorCapabilityPublication` above the explicit executor-trace surface and refuses publication for unbenchmarked lanes such as `core_i32_v2`, while `psionic-provider` now wraps the same served publication in a provider-facing `TassadarCapabilityEnvelope` instead of inventing a separate capability story. The repo also now has targeted matrix-generation, refusal-coverage, research-only, and benchmark-gated publication tests across the three crates. |

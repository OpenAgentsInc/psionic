# Legal Benchmark Sweeps

> Status: implemented_early.

The sweep runner contract lives in
`crates/psionic-eval/src/legal_benchmark_sweeps.rs`. It plans task/model-config
jobs, applies resume state, enforces budgets, continues through individual
failures, and emits a machine-readable manifest for Autopilot4 import.

## Scope

`LegalBenchmarkSweepScope` supports:

- exact task ids
- practice area
- workflow
- named task slices
- all tasks

The first implementation plans jobs from the configured task ids and
run-config hashes. The executor boundary is trait-based so live agent/evaluator
execution can attach without changing the manifest schema.

## Budgets And Resume

`LegalBenchmarkSweepBudget` can cap:

- cost
- wall time
- tokens
- failures

Completed jobs from `LegalBenchmarkSweepResumeState` become `resumed`.
Failures become `failed` and do not stop unrelated jobs until the failure
budget is exhausted. Budget exhaustion marks new jobs `budget_exhausted`
instead of dropping them from the manifest.

## Command

The checked example command runs a deterministic mock sweep:

```bash
cargo run -p psionic-eval --example legal_benchmark_sweep -- \
  fixtures/legal_benchmark/sweep_smoke_config.json \
  /tmp/legal-benchmark-sweep-manifest.json
```

It requires no live provider credentials. A live runner should implement
`LegalBenchmarkSweepExecutor` by invoking the agent runner, evaluator, and
static report generator for each job.

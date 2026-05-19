# Legal Benchmark Reports

> Status: implemented_early.

The static report generator lives in
`crates/psionic-eval/src/legal_benchmark_reports.rs`. It converts score
reports into human-readable Markdown and stable machine-readable JSON for
Autopilot4 import.

## Outputs

The generator produces:

- Markdown run report
- `LegalBenchmarkAutopilotReportExport`
- per-task summaries
- per-model-config summaries
- pairwise comparison reports
- failure cluster exports

Reports include all-pass state, missed criteria, judge reasoning summaries,
artifact/run hashes, cost, latency, token usage, document coverage, extraction
receipt refs, and failure diagnostics.

## Failure Clusters

Failure clusters group missed criteria by family:

- `deterministic_precheck`
- `judge_fail`
- `judge_ambiguous`
- `not_evaluated`

Each cluster records task ids, criterion ids, failure count, score delta,
affected modules, diagnostics, and a repro command. Autopilot4 can import the
JSON directly for dashboards, Work Orders, and issue generation.

## Command

The checked example command writes static artifacts from one score report:

```bash
cargo run -p psionic-eval --example legal_benchmark_report -- \
  fixtures/legal_benchmark/evaluator_mock_score_report.json \
  /tmp/legal-benchmark-report
```

It writes:

- `/tmp/legal-benchmark-report/report.md`
- `/tmp/legal-benchmark-report/autopilot_report.json`
- `/tmp/legal-benchmark-report/failure_clusters.json`

The command does not require live provider credentials.

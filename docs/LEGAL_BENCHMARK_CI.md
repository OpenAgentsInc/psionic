# Legal Benchmark CI Checks

> Status: implemented_early.

The repo-native legal benchmark check target is:

```bash
scripts/check-legal-benchmark-ci.sh
```

It requires no live model provider credentials.

## Coverage

The check script runs:

- Harvey corpus metadata fixture checks
- minimal normalized task snapshot checks
- coverage tracker integrity and hill-climb policy checks
- document helper tool checks
- sandbox traversal and symlink escape checks
- mock report generation checks
- mock sweep manifest checks
- mock sweep matrix fixture validation
- product regression guardrail fixture and blocked-promotion simulation checks
- public/synthetic legal signature-routing fixture checks for Probe+Codex
- JSON validation for the legal benchmark fixtures used by the runner,
  evaluator, report, sweep, product-regression, and signature-routing paths

The audited Harvey corpus fixture pins:

- commit `5aa41694`
- 1,251 tasks
- 24 practice areas
- 74,990 criteria
- 9,537 source documents

If upstream corpus counts or normalization output intentionally change, update
the fixture in the same PR as the compatibility change.

# Legal Benchmark Evaluator

> Status: implemented_early.

The criterion-scoped evaluator lives in
`crates/psionic-eval/src/legal_benchmark_evaluator.rs`. It consumes completed
Rust agent-run artifacts and produces `ScoreReport` records with deterministic
prechecks, criterion-level judge provenance, and diagnostics.

## Inputs

`LegalBenchmarkEvaluationInput` binds:

- normalized `BenchmarkTaskSpec`
- completed `RunRecord`
- output `ArtifactManifest`
- output root containing generated deliverables

The evaluator recomputes the output manifest hash and verifies each manifest
artifact before trusting the run for scoring.

## Deterministic Prechecks

Before calling a judge, the evaluator checks:

- output manifest hash matches the run record
- required deliverables exist
- required deliverables are readable
- deliverable extension matches the declared deliverable kind
- output artifact hashes, byte sizes, and media types match the manifest

Failures become report diagnostics. Criteria tied to missing or invalid
deliverables fail through `deterministic_precheck` without spending judge
tokens.

## Judge Interface

`LegalBenchmarkJudgeAdapter` is provider-neutral. A judge receives
`LegalBenchmarkJudgeRequest` with:

- criterion spec
- output text bundle
- evidence refs
- prompt template id
- prompt template hash
- judge model id

The response records verdict, reasoning, optional confidence, raw response,
latency, and cost. `MockLegalBenchmarkJudge` supports deterministic CI without
live provider credentials.

## Score Report

`ScoreReport` now carries:

- all-pass status
- criterion pass rate
- run metrics with judge latency and cost included
- document coverage in basis points
- failure diagnostics
- extraction receipt refs
- criterion result judge model, prompt hash, raw response hash, confidence,
  latency, and cost

The score report hash is computed with the existing
`score_report_digest()` helper and can feed static reports, sweep comparisons,
and Autopilot4 imports.

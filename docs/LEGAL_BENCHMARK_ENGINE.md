# Legal Benchmark Engine

> Status: implemented_early for the Rust schema contract and Harvey
> compatibility scanner.

Psionic owns the Rust execution and evaluation substrate for legal-agent
benchmark runs. The first landed contract is the schema foundation in
`crates/psionic-eval/src/legal_benchmark.rs`.

## Current Contract

The `psionic-eval` legal benchmark module defines:

- `BenchmarkTaskSpec`
- `ArtifactManifest`
- `SourceArtifact`
- `DeliverableSpec`
- `CriterionSpec`
- `JudgePolicy`
- `ToolPolicy`
- `RunConfig`
- `RunRecord`
- `TranscriptEvent`
- `ToolCallRecord`
- `RunMetrics`
- `CoverageSnapshot`
- `CriterionResult`
- `ScoreReport`
- `ComparisonReport`

Every top-level contract has an explicit schema version. The run contract
requires task identity, task version, input artifact manifest hash, run config
hash, and output artifact manifest hash, so later runner work cannot produce a
score without the immutable execution identity Autopilot needs.

## Artifact Manifests And Hashing

The module exposes SHA-256 helpers over canonical serde JSON encodings:

- `task_spec_digest`
- `artifact_manifest_digest`
- `run_config_digest`
- `run_record_digest`
- `transcript_digest`
- `score_report_digest`
- `comparison_report_digest`

These helpers are namespaced by artifact family. Downstream code should store
the digest and the JSON artifact together rather than recomputing identity from
partial database rows.

The module also provides reusable manifest helpers:

- `artifact_from_file`
- `build_input_artifact_manifest`
- `build_output_artifact_manifest`
- `build_derived_artifact_manifest`
- `compare_artifact_manifests`

Input manifests are built from normalized task source artifacts. Output and
derived manifests take generated artifacts from the runner or extractor layer.
All builders sort artifacts deterministically before hashing, so unchanged
inputs produce stable manifest hashes and changed source bytes produce manifest
drift that can be compared before a run is trusted.

## Fixture

The minimal checked fixture lives at:

- `fixtures/legal_benchmark/minimal_task_bundle.json`

It covers one Harvey-compatible legal task, input and output manifests, a run
config, a run record, a score report, and a comparison report. The fixture is
used by unit tests to prove the schema round-trips and the digest helpers are
deterministic.

## Harvey Compatibility Loader

The module `crates/psionic-eval/src/legal_benchmark_harvey.rs` scans a Harvey
`tasks` directory into owned `BenchmarkTaskSpec` values and a
`HarveyCorpusSummary`.

The loader:

- discovers nested `task.json` files under a caller-supplied tasks root
- parses Harvey fields `title`, `work_type`, `tags`, `instructions`,
  `deliverables`, and `criteria`
- reads sibling `documents/` files as source artifacts
- preserves the upstream commit and task path under `SourceCompatibility`
- validates missing documents directories, empty criteria, empty deliverables,
  and criteria that reference unknown deliverables
- reports task, practice-area, criterion, source-document, deliverable,
  work-type, and extension counts

Run the summary scanner with:

```bash
cargo run -p psionic-eval --example legal_benchmark_harvey_scan -- \
  /Users/christopherdavid/work/competition/repos/harvey-labs/tasks \
  5aa41694
```

For the audited Harvey checkout, the expected summary is:

- 1,251 tasks
- 24 practice areas
- 74,990 criteria
- 9,537 source documents

## Sandbox Boundary

The local sandbox boundary for extraction and tool execution is documented in
`docs/PODMAN_SANDBOX_BACKEND.md` and implemented in
`crates/psionic-sandbox/src/podman.rs`.

For legal benchmark runs, the default Podman config disables network access,
mounts source documents read-only at `/workspace/inputs`, and exposes writable
scratch and output paths at `/workspace/scratch` and `/workspace/output`.
Path validation canonicalizes host roots and rejects traversal or symlink
escapes before a container command is built.

## Document Extraction

The extraction contract lives in
`crates/psionic-eval/src/legal_benchmark_extraction.rs`.

It defines:

- `ArtifactExtractor`
- `ArtifactExtractionPolicy`
- `ArtifactExtractionResult`
- `ExtractionReceipt`
- `ExtractionCoverage`
- `ExtractionFailureKind`
- `ArtifactExtractorRegistry`

Day-one native extraction handles text, Markdown, JSON, CSV/TSV, XML/HTML,
YAML, TOML, and log-like UTF-8 inputs without external tools. The registry also
declares pinned sandboxed external adapter specs for DOCX, PDF, PPTX, XLSX,
and EML. Until a live sandbox command executor is attached, those external
adapters return structured `external_tool_unavailable` or policy-denied
receipts instead of panicking or crashing a sweep.

Run records and score reports now retain `extraction_receipt_refs` so operator
surfaces can distinguish extraction failure, missing content, and bad
reasoning.

## Tool Surface

The closed Rust tool set is documented in `docs/LEGAL_BENCHMARK_TOOLS.md` and
implemented in `crates/psionic-eval/src/legal_benchmark_tools.rs`.

It covers shell, read, write, edit, glob, and grep with typed inputs, typed
outputs, structured errors, byte metrics, touched paths, transcript events, and
tool-call records. Read can prefer extracted text from the extraction layer;
write and edit are restricted to workspace/output roots; shell remains
sandbox-owned and routes through the Podman backend only when the full-feature
caller attaches one.

High-score document helpers now extend the same receipt-backed surface with
inventory, EML summary, spreadsheet summary, page-targeted PDF search, evidence
table construction, and deliverable validation. These tools improve source
coverage, evidence traceability, and pre-judge output checks without exposing
hidden criteria.

## Provider Adapter Layer

The provider-neutral model contract is documented in
`docs/LEGAL_BENCHMARK_PROVIDERS.md` and implemented in
`crates/psionic-eval/src/legal_benchmark_provider.rs`.

It defines model requests, messages, tool specs, tool calls, tool-result
messages, usage accounting, structured provider failures, retry policy,
OpenAI-compatible and Anthropic protocol adapters, and deterministic CI mocks.
Routes record provider family, model id, model config hash, elapsed time,
retry count, raw response hash, and secret reference id without writing raw
credentials into run artifacts.

## Agent Runner

The Rust agent loop is documented in `docs/LEGAL_BENCHMARK_RUNNER.md` and
implemented in `crates/psionic-eval/src/legal_benchmark_agent.rs`.

It builds policy/task prompts, drives provider turns, executes tool calls,
requires explicit JSON submit/finalize semantics, classifies terminal states,
and writes `config.json`, `transcript.jsonl`, `metrics.json`,
`output_artifact_manifest.json`, `extraction_receipts.json`,
`tool_receipts.json`, `run_record.json`, and `run_receipt.json`.

Integrity mode is the default prompt policy. The runner does not render hidden
criteria into model-visible messages; hill-climb runs may render only
policy-approved derived checklist items from run config metadata.

## Coverage Tracker

Criterion-adjacent coverage tracking is documented in
`docs/LEGAL_BENCHMARK_COVERAGE.md` and implemented in
`crates/psionic-eval/src/legal_benchmark_coverage.rs`.

Run records persist `CoverageSnapshot` values covering discovered/read
documents, extracted facts, evidence refs, drafted deliverable sections,
validations, self-checks, and the policy mode used. Score reports copy the
snapshot and add post-judge failure comparisons that classify missed criteria
as coverage, extraction, drafting, or reasoning gaps for Autopilot4 import.

## Evaluator And Judge Interface

The criterion-scoped evaluator is documented in
`docs/LEGAL_BENCHMARK_EVALUATOR.md` and implemented in
`crates/psionic-eval/src/legal_benchmark_evaluator.rs`.

It loads completed task/run/output-manifest artifacts, runs deterministic
manifest and deliverable prechecks, extracts output text per criterion, calls a
provider-neutral judge adapter, and emits `ScoreReport` values with all-pass
status, criterion pass rate, judge provenance, confidence, latency, cost,
document coverage, failure diagnostics, extraction receipt refs, coverage
snapshots, and missed-criterion failure classifications.

## Static Reports

Static reporting is documented in `docs/LEGAL_BENCHMARK_REPORTS.md` and
implemented in `crates/psionic-eval/src/legal_benchmark_reports.rs`.

It generates Markdown reports for humans plus stable Autopilot4 import JSON
with global, per-task, per-model-config, comparison, and failure-cluster
summaries. The example command
`cargo run -p psionic-eval --example legal_benchmark_report -- <score.json> <out-dir>`
writes `report.md`, `autopilot_report.json`, and `failure_clusters.json`
without live provider credentials.

## Sweep Runner

Sweep planning and manifests are documented in `docs/LEGAL_BENCHMARK_SWEEPS.md`
and implemented in `crates/psionic-eval/src/legal_benchmark_sweeps.rs`.

The sweep layer plans task/config jobs, expands provider/reasoning/context/
extraction/tool-policy matrices, applies resume state, enforces cost,
wall-time, token, and failure budgets, keeps going through individual
task/model failures, and emits a manifest with skipped, resumed, succeeded,
failed, blocked, and budget-exhausted job states for Autopilot4 import.

Matrix exports summarize every recorded config hash by all-pass score,
criterion pass rate, document coverage, reliability, cost, and latency, then
mark Pareto-front configs for promotion-gate review.

## CI And Golden Fixtures

Repo-native compatibility checks are documented in
`docs/LEGAL_BENCHMARK_CI.md` and run with
`scripts/check-legal-benchmark-ci.sh`.

The check target pins the audited Harvey corpus metadata, verifies the minimal
normalization snapshot, covers sandbox traversal and symlink escape behavior,
and exercises mock report and sweep fixtures without live provider credentials.

## Next Work

The next implementation issue is production regression guardrails before
benchmark-optimized modules can affect Autopilot agents.

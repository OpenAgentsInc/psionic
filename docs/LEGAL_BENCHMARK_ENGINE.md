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

## Next Work

The next implementation issue is document extraction receipts. It should attach
versioned extractor identity, input hashes, output hashes, warnings, and
coverage metadata to derived artifacts before runner and evaluator work depends
on extracted text.

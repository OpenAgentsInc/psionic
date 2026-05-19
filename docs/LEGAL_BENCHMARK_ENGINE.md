# Legal Benchmark Engine

> Status: implemented_early for the Rust schema contract.

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

## Hashing

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

## Fixture

The minimal checked fixture lives at:

- `fixtures/legal_benchmark/minimal_task_bundle.json`

It covers one Harvey-compatible legal task, input and output manifests, a run
config, a run record, a score report, and a comparison report. The fixture is
used by unit tests to prove the schema round-trips and the digest helpers are
deterministic.

## Next Work

The next implementation issue is the Harvey compatibility loader. It should
parse `competition/repos/harvey-labs/tasks/**/task.json` into
`BenchmarkTaskSpec`, preserve upstream provenance under `SourceCompatibility`,
and reproduce the audited corpus counts before any runner or evaluator code
depends on the data.

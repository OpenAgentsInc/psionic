# Legal Benchmark Training Records

Status: `implemented_early`

Psionic now has a canonical `legal_benchmark_training_record.v1` data contract
for Harvey-compatible legal fine-tuning work. The record is the bridge between
legal benchmark execution and Qwen-family adapter SFT.

## Ownership

- `psionic-data` owns the canonical record and bundle schema.
- `psionic-eval` owns the exporter from legal task, run, and score records.
- `psionic-train` consumes exported bundles for Qwen legal adapter smoke work.

The contract deliberately keeps model-visible training examples separate from
judge-only scoring data. Hidden rubric or criterion material must not leak into
fine-tuning examples.

## Canonical Types

Primary Rust surface:

- `psionic_data::LegalBenchmarkTrainingRecord`
- `psionic_data::LegalBenchmarkTrainingRecordBundle`
- `psionic_data::LegalBenchmarkTrainingExample`
- `psionic_eval::LegalBenchmarkTrainingRecordExportInput`
- `psionic_eval::export_legal_benchmark_training_records`

The stable schema versions are:

- `psionic.legal_benchmark.training_record.v1`
- `psionic.legal_benchmark.training_record_bundle.v1`

## Record Contents

Each record carries:

- suite id, task id, task version, practice area, and work type;
- source artifact manifest digest;
- tool policy digest;
- ordered tool invocation rows;
- evidence refs with source refs, locators, and span hashes;
- deliverable refs;
- coverage snapshot digest;
- score report ref and digest;
- failure-family labels;
- judge provenance rows;
- hidden-criterion policy;
- split assignment;
- derived examples.

Derived examples are tagged as one of:

- `model_visible`
- `judge_only`
- `excluded_from_training`

If the source coverage snapshot says hidden criteria were visible to the model,
the exporter emits no model-visible examples for that record. Those examples are
retained only as excluded provenance with an explicit exclusion reason.

## Initial Export Policy

The first Qwen smoke should use:

```text
split_policy = retained_smoke
suite_id = harvey_labs
target_model = Qwen3.5-4B
```

This is a dataset/eval artifact, not a score-lift claim. Public score claims
still require retained score reports from the benchmark runner.

## Validation

Focused tests cover:

- deterministic record and bundle digests;
- duplicate id rejection;
- split counting;
- export determinism;
- hidden-criterion exclusion from model-visible examples.

Run the focused checks with:

```bash
cargo test -p psionic-data legal_benchmark_training_record
cargo test -p psionic-eval legal_benchmark_training_records
```

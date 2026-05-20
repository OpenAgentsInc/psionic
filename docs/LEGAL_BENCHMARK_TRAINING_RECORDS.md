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

## Synthetic Workflow Task Generator

`psionic-data` now owns a Rust-only generator for internal synthetic legal
workflow tasks:

```bash
cargo run -p psionic-data --example legal_benchmark_generate_synthetic_tasks -- \
  --count 100 \
  --out tasks/synthetic/legal-workflow-v1
```

The committed 2026-05-20 generation produced:

- `100` synthetic tasks
- `50` deterministic base-policy success runs
- `50` deterministic base-policy failed runs
- `250` SFT examples
- `1,408` sampled DPO preference pairs
- SFT dataset hash:
  `02f31b3c8c481bdd9cb14ac150c80ff01b9e3bbe0953b986b18cece77b578719`
- DPO dataset hash:
  `8023f7c4c0e80ed71268eb000fe2c978b809fe55d333178aed1b15763ebd1ab3`
- manifest hash:
  `037ba69d95751849f7a5f92184c15f315b2b51d861269791776623f59392c1e8`

The task families are:

- contract clause extraction
- employment agreement summary
- NDA risk list
- lease obligation extraction
- litigation memo source summarization
- statute-to-facts application with provided text
- privilege log classification from provided docs
- answer-file workflow-only tasks

The generated task files are Harvey-shaped, but they are clearly tagged
`synthetic`. Source documents are generated text. Expected answers and scoring
points live under the separate `rubrics/` tree and are marked
`judge_only_not_model_visible`.

The generator also writes deterministic base-policy runs under `runs/`, then
uses the existing SFT and DPO dataset builders to create the training files
under `training/`. Every success and failure is used for SFT. DPO uses a
bounded representative sample so the generated corpus stays small enough to
review and commit.

Plain boundary: these tasks are useful for teaching source use, answer-file
discipline, and legal answer shape. They are not Harvey benchmark tasks, and
their outcomes must never be reported as Harvey scores.

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
cargo test -p psionic-data generator_builds_synthetic_tasks_and_training_data --lib
cargo test -p psionic-data generator_refuses_empty_count --lib
```

# Legal Benchmark Engine

> Status: implemented_early for the Rust schema contract and Harvey
> compatibility scanner.

Psionic owns the Rust execution and evaluation substrate for legal-agent
benchmark runs. The first landed contract is the schema foundation in
`crates/psionic-eval/src/legal_benchmark.rs`.

Operating boundary: the upstream Harvey Python harness is reference/backfill
only. Owned Harvey-compatible execution runs through Psionic's Rust legal
benchmark engine, while Autopilot Blueprint/Program policy selects the
upgradable prompts/modules, provider adapter, judge policy, release gates, and
promotion rules. Provider names such as Gemini, OpenAI-compatible local
servers, or Qwen fine-tunes are adapter metadata, not benchmark authority.

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

The first fine-tuning export path is now split across:

- `crates/psionic-data/src/legal_benchmark_training_record.rs`
- `crates/psionic-eval/src/legal_benchmark_training_records.rs`
- `docs/LEGAL_BENCHMARK_TRAINING_RECORDS.md`

That path exports legal task, run, coverage, transcript, and score artifacts
into `legal_benchmark_training_record.v1` bundles for Qwen-family adapter
smoke work. The exporter keeps judge-only scoring data separate from
model-visible examples and excludes model-visible examples when a run exposed
hidden criteria.

Every top-level contract has an explicit schema version. The run contract
requires task identity, task version, input artifact manifest hash, run config
hash, and output artifact manifest hash, so later runner work cannot produce a
score without the immutable execution identity Autopilot needs.

The Rust agent runner must not add answer text to model output. Run 015 did
that through an output-scaffold path and therefore does not count as a
benchmark score. Keep it only as a diagnostic example of a bad runner design:
the scorer could be fooled when the runner inserted the words the scorer was
looking for.

The current supported no-cheat runner metadata is limited to controls that do
not rewrite the deliverable:

- `max_output_tokens`: per-run override for the model request output budget.
- `force_write_until_required_deliverables`: keep prompting until the model
  itself writes the required deliverable or the run budget is exhausted.
- `force_validate_after_write`: require a model-authored validation step after
  a model-authored write.
- `plain_text_tool_protocol`: ask a weak local model to emit plain JSON tool
  requests in text, then execute only the JSON tool call the model wrote.

Those controls affect request shape, turn order, and tool-call parsing. They
do not add legal analysis, coverage markers, citations, headings, or scoring
phrases to output files.

The runner now emits an `answer_integrity` report in both run receipts and run
metadata. The report records every required or declared answer file, its
pre-score and post-score hash, size, mtime, writer tool-call id, and actor
classification. A scored answer file is valid only when a model-authored
`write` or `edit` tool call produced the final bytes. Files created by the
harness, files changed during scoring, missing required files, and files whose
hash no longer matches the model write receipt are marked invalid.

The evaluator reads the same integrity report before and after scoring. If any
answer file changes during scoring, the score report is forced invalid and the
diagnostic is retained in `failure_diagnostics`. This keeps invalid runs
auditable without allowing them into promotion metrics.

`crates/psionic-eval/src/legal_benchmark_schema.rs` adds the canonical v1
schema layer used for long-lived legal benchmark receipts and training data.
It wraps today's `RunRecord`, output manifest, score report, and
`answer_integrity` report into a `LegalRunReceipt` with:

- benchmark id and visibility
- base model, adapter, tokenizer, prompt template, thinking mode, and tools
- source document or input-manifest hashes
- transcript action hashes and tool-call hashes
- answer file hashes and actor information
- scorer version, score hash, wall-clock timings, git commit, dirty-tree flag,
  worker id, hardware summary, replay command, and artifact refs

The schema module also defines the first Rust shapes for legal training
examples, bad-run examples, preference pairs, reward traces, dataset
manifests, adapter manifests, model candidates, promotion decisions, Pylon
training jobs, Pylon worker receipts, Psionic training configs, and Psionic
training receipts. These are plain canonical JSON contracts for now; moving
them into a shared crate is deferred until `psionic-train` needs to consume
the same stable structs directly.

Validation rejects receipts that omit benchmark visibility, answer file
content hashes, scorer version, replay command, or required artifact hashes.
Use:

```bash
cargo test -p psionic-eval legal_benchmark_schema
cargo run -p psionic-eval --example legal_benchmark_validate_run_receipt -- <path>
cargo run -p psionic-eval --example legal_benchmark_print_run_summary -- <path>
```

`crates/psionic-eval/src/legal_benchmark_failed_trajectory.rs` captures failed
legal runs as complete `LegalBadRunExample` artifacts. A bad-run example keeps
the full prompt, full model response, tool-call transcript, attempted writes,
required file status, answer content and hashes when present, action sequence,
stop reason, score, scorer feedback, integrity status, failure class,
suggested correction, and training eligibility. Bad examples are never marked
SFT-eligible directly; a later dataset builder has to convert them into a
positive example or preference pair.

Failure capture currently classifies missing files, wrong output paths, empty
or badly sized answers, missing source use, hallucinated citations, malformed
or invalid JSON tool calls, harness integrity failures, scorer outages,
timeouts, refusals, missing submissions, and uncategorized failures. Hidden and
private benchmark failures stay audit-only unless an explicit private-training
override is supplied and no hidden labels or scorer secrets are present.

Use:

```bash
cargo test -p psionic-eval failed_trajectory_capture
cargo run -p psionic-eval --example legal_benchmark_inspect_failures -- <run-dir>
```

`crates/psionic-data/src/legal_benchmark_sft_dataset.rs` builds canonical
`legal_sft_v1` JSONL from honest successful receipts and training-eligible
bad-run examples. It refuses hidden/private-by-default receipts, unknown
visibility, answer-integrity failures, non-model answer mutation, hidden
scoring labels, scorer-only target labels, and known harness-injected marker
runs. Successful runs become golden workflow, source-grounded answer,
tool-discipline, and minimal-answer examples when answer file content is
available. Failed runs stay excluded from raw SFT, but the builder may convert
training-eligible failures into correction examples.

Use:

```bash
cargo run -p psionic-data --example legal_benchmark_build_sft_dataset -- \
  --runs ./runs \
  --out ./datasets/legal-sft-v1.jsonl \
  --manifest ./datasets/legal-sft-v1.manifest.json
```

`crates/psionic-data/src/legal_benchmark_dpo_dataset.rs` builds canonical
`legal_dpo_v1` JSONL preference pairs from honest good runs and
training-eligible bad-run examples. Each pair has the Rust/Psionic-native DPO
shape:

- `prompt`: system message plus the original task prompt and one explicit
  training focus
- `chosen`: the model-written legal answer or correct tool trajectory to
  imitate
- `rejected`: the bad model response or broken trajectory to avoid
- `reason`: normalized failure class, such as `DidNotWriteRequiredFile`
- `source_run_ids`, `visibility`, and `exclusion_flags`

The builder emits file-discipline, correct-path, source-grounding,
conciseness, submission, and integrity-safe pair families. Its manifest records
total pair count, excluded input count, pair counts by failure class, pair
counts by family, source receipt refs, excluded input reasons, and a dataset
hash.

The DPO builder uses the same safety boundary as the SFT builder: hidden and
private benchmark labels are rejected, unknown visibility is rejected,
integrity-invalid successful runs are rejected, harness-authored answer files
are rejected, hidden/scorer-only markers are rejected, and integrity-invalid or
harness-assisted bad runs are excluded instead of being used as negative
examples. The checked local fixture currently yields 22 public-training pairs
from one good run and one `DidNotWriteRequiredFile` bad run.

Use:

```bash
cargo run -p psionic-data --example legal_benchmark_build_dpo_dataset -- \
  --runs ./runs \
  --out ./datasets/legal-dpo-v1.jsonl
```

The optional `--manifest ./datasets/legal-dpo-v1.manifest.json` flag overrides
the default manifest path derived from the JSONL output path. The loader
function `load_legal_dpo_dataset` is the handoff point for the Psionic DPO
trainer.

Qwen3.6 prompt handling is Rust-native. `crates/psionic-models/src/qwen36.rs`
renders Qwen3.6 chat prompts in explicit `Thinking`, `DirectAnswer`, and
`MixedExplicit` modes without using `/think` or `/nothink` soft-switch tokens.
The renderer supports tokenizer JSON loading through the Rust `tokenizers`
crate, tool-response transcript rendering, optional empty think-block emission
for direct-answer generation prompts, deterministic prompt hashes, and a small
`Qwen36PromptReceipt` that can be embedded in later run receipts.
`crates/psionic-transformer/src/qwen36_loss_masks.rs` owns the matching loss
mask contract: system, user, and tool spans are ignored under assistant-only
loss, and empty think blocks can be ignored explicitly.
The SFT dataset example schema now also carries `reasoning_mode`, so legal
examples can declare `direct_answer` or a later thinking-mode value without
using hidden prompt switches.

Use:

```bash
cargo test -p psionic-models qwen36_template
cargo test -p psionic-transformer qwen36_loss_masks
```

The first Rust-only Qwen3.6 SFT command is now:

```bash
cargo run -p psionic-train -- sft \
  --config configs/legal/qwen36_sft_smoke.json
```

That smoke config uses `Qwen/Qwen3.6-27B` metadata, QLoRA-style adapter
settings, assistant-only loss flags, and the `all-linear` Qwen3.6 target-module
declaration. The current executable training surface is intentionally smaller:
it trains an adapter-only LM-head LoRA update over tiny legal hidden-state
samples, writes `adapter.safetensors`, `loss_curve.json`,
`checkpoint_summary.json`, and `training_receipt.json`, and records that no
Python process or Python-generated trainer artifact was used. It proves the
Rust config, training, receipt, and export loop. Full dense Qwen3.6 causal-LM
target coverage remains the next model-path expansion work.

The first Rust-only Qwen3.6 DPO command is now:

```bash
cargo run -p psionic-train -- dpo \
  --config configs/legal/qwen36_dpo_smoke.json
```

That smoke config loads the parent SFT LoRA adapter, bootstraps it from
`configs/legal/qwen36_sft_smoke.json` when the local target artifact is
missing, loads `legal_dpo_v1` prompt/chosen/rejected pairs, renders prompts
through the Qwen3.6 direct-answer template, and runs adapter-only weighted
chosen/rejected updates with `beta = 0.25`. It writes the same core artifact
family as SFT: `adapter.safetensors`, `loss_curve.json`,
`checkpoint_summary.json`, and `training_receipt.json`.

The checked smoke DPO dataset lives at
`fixtures/legal_benchmark/dpo_smoke/legal-dpo-v1.jsonl`. The 2026-05-20 local
command run completed 6 Rust-only steps over 22 pairs, moved the synthetic
preference accuracy from `0.59090906` to `0.95454544`, and moved the average
chosen-minus-rejected logprob margin from `0.3191057` to `4.2714095`. This is
evidence that the adapter-only DPO path can train toward file-writing
preference behavior on the synthetic smoke surface. It is not a hidden Harvey
score claim.

The same DPO adapter path was accepted by the deterministic replay eval suite:
`harvey_public_three_deterministic_replay_v1` reported base `3333` bps,
adapter `10000` bps, and delta `6667` bps with report hash
`bd01ce5a8653414a2189d935c80c835c774f55f195ed6809021c135a352faa66`. That is
replay-harness compatibility evidence, not retained benchmark proof.

The current honest Harvey MFN local result is run 016: the actual local Qwen
LoRA adapter 005 submitted through the Rust tool loop, wrote its own output,
and scored `4 / 18` on a rubric-free legal work-product proxy. Broad suite
runs 019 and 025 compare model-only and scaffold-assisted prompts across three
public Harvey tasks with runner output mutation disabled. Adapter 020 is a
real local MLX LoRA fine-tune over clean no-cheat supervised trajectories, but
it did not yet improve the broad suite score.

## Deterministic Replay Eval

`crates/psionic-eval/src/legal_benchmark_eval_suite.rs` adds the first local
replay harness for comparing a base model binding and an adapter binding
against the exact same suite. The harness freezes:

- suite id and eval mode
- fixed task order
- source document hashes
- prompt template hash
- scorer version
- inference settings
- base model id and adapter id or adapter artifact hash

The checked smoke suite is:

- `suites/harvey_public_three.json`
- `fixtures/legal_benchmark/eval_suite_public_three/*`

Run it with:

```bash
cargo run -p psionic-eval --example legal_benchmark_eval_suite -- \
  --suite suites/harvey_public_three.json \
  --model Qwen/Qwen3.6-27B \
  --adapter target/legal/qwen36_sft_smoke/adapter.safetensors \
  --out runs/harvey-public-three-smoke
```

The output directory contains:

- `eval_report.json`
- `promotion_gate_input.json`
- `replay_receipt.json`
- per-task base and adapter run records and score reports
- base and adapter static Markdown summaries

The report separates answer-file success rate, legal score, integrity
failures, tool failures, timeout failures, and failure-class counts. It also
writes a promotion-gate JSON object so later adapter registry work can make a
single promote/hold/reject decision without re-parsing scorer internals.

Plain boundary: this harness replays declared local outputs through the Rust
scorer. It is useful for proving that the evaluator, receipts, ordering, and
promotion inputs are stable. It is not proof that a model improved on hidden
Harvey tasks. Hidden audit suites are rejected if marked training-allowed.

## Adapter Registry And Promotion Gates

`crates/psionic-train/src/qwen_legal_adapter_registry.rs` adds the first local
Qwen legal adapter registry. Each registry entry records the adapter id, base
model id and hash, training dataset id and hash, training config id and hash,
Psionic version, git commit, training workers, training receipt hash, eval
suite id and hash, eval result hash, parent adapter id, promotion status, and
the eval summary used by hard gates.

Register adapters with:

```bash
cargo run -p psionic-train --example qwen_legal_register_adapter -- \
  fixtures/legal_benchmark/adapter_registry/qwen_legal_champion_adapter_manifest.json

cargo run -p psionic-train --example qwen_legal_register_adapter -- \
  fixtures/legal_benchmark/adapter_registry/qwen_legal_candidate_adapter_manifest.json
```

Promote a candidate with:

```bash
cargo run -p psionic-train --example qwen_legal_promote_adapter -- \
  --candidate qwen36-legal-public-three-candidate-001 \
  --suite harvey_public_three_deterministic_replay_v1
```

Set `PSIONIC_LEGAL_ADAPTER_REGISTRY` or pass `--registry <path>` to use a
non-default local registry path. The default path is
`target/legal/qwen_adapter_registry/registry.json`.

Registration rejects missing training receipts, missing eval receipts, empty
worker sets, excluded training data, invalid hashes, hidden benchmark leakage,
harness-modified answer text, integrity failures, and adapters that were not
produced by an allowed Psionic/Pylon path. Promotion rejects lower scores,
different suite hashes, answer-file write-rate regressions, required-workflow
regressions, hidden leakage, incomplete receipts, and non-Psionic production
paths. A promoted candidate supersedes the previous champion for that suite and
writes a promotion receipt next to the registry.

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

Native extraction handles text, Markdown, JSON, CSV/TSV, XML/HTML, YAML, TOML,
log-like UTF-8 inputs, DOCX, PPTX, XLSX, and EML without external tools. The
Office path reads the Open XML ZIP container and preserves text from supported
document, slide, workbook, shared-string, header, and footer XML parts. The EML
path preserves common headers and readable body text. Both native paths are
lossy by design and emit extraction warnings because layout, attachments,
tracked changes, formulas, comments, MIME encodings, and hidden workbook
semantics still require a higher-fidelity sandboxed extractor.

The registry still declares pinned sandboxed external adapter specs for PDF and
future high-fidelity Office/EML extraction. Until a live sandbox command
executor is attached, those external adapters return structured
`external_tool_unavailable` or policy-denied receipts instead of panicking or
crashing a sweep.

The 2026-05-20 local run against the audited Harvey checkout proves the native
path across the full corpus:

```bash
cargo run -q -p psionic-eval --no-default-features \
  --example legal_benchmark_extract_slice -- \
  /Users/christopherdavid/work/competition/repos/harvey-labs/tasks \
  5aa41694 1251
```

Result: 1,251 tasks scanned, 9,537 source artifacts extracted, and zero
structured extraction failures. The prior 25-task pre-change slice extracted
1 of 198 artifacts and returned 197 `ExternalToolUnavailable` failures.

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
hidden criteria. Normalized Harvey tasks enable this full document-tool set by
default so live runs do not fall back to the older read/write/grep-only surface.

## Provider Adapter Layer

The provider-neutral model contract is documented in
`docs/LEGAL_BENCHMARK_PROVIDERS.md` and implemented in
`crates/psionic-eval/src/legal_benchmark_provider.rs`.

It defines model requests, messages, tool specs, tool calls, tool-result
messages, usage accounting, structured provider failures, retry policy, a
Google Vertex Gemini `generateContent` adapter for the active
`gemini-3-flash-preview` benchmark lane, OpenAI-compatible and Anthropic
protocol adapters for fallback/parity lanes, and deterministic CI mocks. Routes
record provider family, model id, model config hash, elapsed time, retry count,
raw response hash, and secret reference id without writing raw credentials into
run artifacts.

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

## Product Regression Guardrails

Product regression guardrails are documented in
`docs/LEGAL_BENCHMARK_REGRESSION_GUARDRAILS.md` and implemented in
`crates/psionic-eval/src/legal_benchmark_regression.rs`.

The guardrail suite uses synthetic fixtures for chat, Coder, Work Orders,
GitHub provider, CRM, memory, and provider/tool routing. Gate reports include
both benchmark target scores and product regression scores, export Autopilot4
release-gate import JSON, create blocking Work Orders for failed product
regressions, and disallow live user data or Harvey hidden criteria in the
regression fixture suite.

## CI And Golden Fixtures

Repo-native compatibility checks are documented in
`docs/LEGAL_BENCHMARK_CI.md` and run with
`scripts/check-legal-benchmark-ci.sh`.

The check target pins the audited Harvey corpus metadata, verifies the minimal
normalization snapshot, covers sandbox traversal and symlink escape behavior,
and exercises mock report, sweep, and product-regression fixtures without live
provider credentials.

## Next Work

The next implementation issue is the Autopilot4-side release-gate import and
operator surface for these Psionic reports.

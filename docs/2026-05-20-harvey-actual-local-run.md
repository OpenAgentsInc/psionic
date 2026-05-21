# Harvey Actual Local Run

Date: 2026-05-20

This records the actual local execution work for the Harvey-compatible legal
benchmark lane. It distinguishes real corpus execution from planning gates.

## Inputs

- Harvey checkout: `/Users/christopherdavid/work/competition/repos/harvey-labs`
- Harvey tasks root:
  `/Users/christopherdavid/work/competition/repos/harvey-labs/tasks`
- Upstream commit recorded in Psionic evidence: `5aa41694`
- Psionic runner: `psionic-eval`

## Upstream Harness Attempt

The upstream Harvey Python harness was invoked only as a reference/backfill
check against
`trusts-estates-private-client/compare-trust-documents-against-client-instructions`.
Owned benchmark execution should use the Psionic Rust agent loop launched under
Autopilot Blueprint/Program policy. The reference harness attempt failed before
model execution because this host did not have `podman` on `PATH`.

Failure class:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'podman'
```

This is a local runtime dependency blocker, not a model failure and not a
Harvey score.

A follow-up `scripts/setup.sh` run installed `pandoc` and `podman`, initialized
the Podman machine, and then failed while starting the VM:

```text
Error: vfkit exited unexpectedly with exit code 1
```

The manual `podman machine start` retry then hung while the Podman SSH socket
refused connections. The current blocker is therefore a non-working local
Podman Machine VM, not a missing binary.

## Corpus Scan

Command:

```bash
cargo run -q -p psionic-eval --no-default-features \
  --example legal_benchmark_harvey_scan -- \
  /Users/christopherdavid/work/competition/repos/harvey-labs/tasks \
  5aa41694 > /tmp/harvey_scan_actual_2026-05-20.json
```

Result:

- 1,251 tasks
- 24 practice areas
- 74,990 criteria
- 9,537 source documents
- 1,655 deliverables
- source extensions: 7,489 DOCX, 1,018 EML, 969 XLSX, 50 PPTX, 6 JSON, 5 TXT

## Extraction Baseline Before Native Office/EML

Command:

```bash
cargo run -q -p psionic-eval --no-default-features \
  --example legal_benchmark_extract_slice -- \
  /Users/christopherdavid/work/competition/repos/harvey-labs/tasks \
  5aa41694 25 > /tmp/harvey_extract_actual_2026-05-20.json
```

Result:

- 25 tasks scanned
- 198 source artifacts
- 1 extracted artifact
- 197 structured failures
- all 197 failures were `ExternalToolUnavailable`

## Native Office/EML Improvement

`ArtifactExtractorRegistry` now tries native text extraction, native Office XML
extraction, and native EML extraction before falling through to sandboxed
external extractor specs.

Implemented native paths:

- DOCX: `word/document.xml`, headers, and footers
- PPTX: slide XML text
- XLSX: shared strings and worksheet text/value cells
- EML: common headers and readable body text

The Office and EML paths are lossy. They preserve text for benchmark access and
emit warnings for semantics that still require a sandboxed high-fidelity
extractor.

## Extraction Slice After Native Office/EML

Command:

```bash
cargo run -q -p psionic-eval --no-default-features \
  --example legal_benchmark_extract_slice -- \
  /Users/christopherdavid/work/competition/repos/harvey-labs/tasks \
  5aa41694 25 > /tmp/harvey_extract_native_office_actual_2026-05-20.json
```

Result:

- 25 tasks scanned
- 198 source artifacts
- 198 extracted artifacts
- 0 structured failures

## Full-Corpus Extraction After Native Office/EML

Command:

```bash
cargo run -q -p psionic-eval --no-default-features \
  --example legal_benchmark_extract_slice -- \
  /Users/christopherdavid/work/competition/repos/harvey-labs/tasks \
  5aa41694 1251 > /tmp/harvey_extract_full_native_office_actual_2026-05-20.json
```

Result:

- 1,251 tasks scanned
- 9,537 source artifacts
- 9,537 extracted artifacts
- 0 structured failures

This is the first full-corpus local execution improvement in this lane. It
does not score model answers, but it removes the document-access failure that
would otherwise dominate any live benchmark run.

## Sweep And Training/RL Substrate Checks

Deterministic mock sweep:

```bash
cargo run -q -p psionic-eval --example legal_benchmark_sweep -- \
  fixtures/legal_benchmark/sweep_smoke_config.json \
  /tmp/psionic_legal_sweep_actual_2026-05-20.json
```

Result: 4 jobs, 4 succeeded, 0 failed, 0 blocked. This is a deterministic mock
sweep, not a score on private Harvey tasks.

Focused validation:

- `cargo test -p psionic-eval --no-default-features --lib legal_benchmark_extraction -- --nocapture`: 7 passed
- `cargo test -p psionic-eval --no-default-features --lib legal_benchmark_harvey -- --nocapture`: 5 passed
- `cargo test -p psionic-train qwen_legal`: 14 passed
- `cargo test -p psionic-train live_rl_update`: 2 passed

## Next Actual Run

Repair or route around the local Podman Machine blocker, then run a retained
live agent slice that uses the now-extracted document text:

- start with the pinned 20-task slice in Autopilot4;
- require extraction receipts for every source document;
- run one model/prompt/blueprint configuration at a time;
- import immutable Psionic score reports into Autopilot4;
- compare failures by document coverage, citation evidence, legal reasoning,
  spreadsheet reasoning, missing facts, and pre-submit self-check.

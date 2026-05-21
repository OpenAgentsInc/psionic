# Legal Benchmark Coverage Tracker

> Status: implemented_early.

The coverage tracker lives in
`crates/psionic-eval/src/legal_benchmark_coverage.rs`. It records
criterion-adjacent state during a run without exposing private Harvey
`match_criteria` text to the model in integrity mode.

## Modes

- `integrity`: default mode. The runner does not render raw criteria or
  derived checklist hints into model-visible task prompts.
- `hill_climb`: training mode. The runner may render policy-approved
  `derived_checklist_items` from `RunConfig.metadata`, but still does not
  render hidden criteria text.

The user prompt now includes task instructions and deliverables. Criteria are
used only by the evaluator and post-judge failure classifier.

## Snapshot Contents

`CoverageSnapshot` records:

- discovered and read source documents
- whether extracted text was used
- facts and evidence spans captured by read/grep, PDF search, summary, and
  evidence-table tools
- drafted deliverable sections
- output-manifest and explicit deliverable validations
- agent self-check declarations
- policy-approved derived checklist items
- whether hidden criteria appeared in model-visible transcript text

`RunRecord` persists the snapshot for replay. `ScoreReport` copies the snapshot
and adds post-judge `failure_comparisons`.

## Failure Classes

After judging, missed criteria are compared to the snapshot and classified as:

- `coverage_gap`: required source material was not read
- `extraction_gap`: source material was read but no matching evidence was
  captured
- `drafting_gap`: evidence was available but the required output was missing or
  failed validation
- `reasoning_gap`: coverage and drafting were present, so the miss is likely
  legal reasoning or application quality

Reports export these comparisons for Autopilot4 failure clustering and
improvement planning.

## Document Coverage Semantics

Returned `grep` matches now count the matched source document as read. The
tool has inspected source bytes and returned a quoted line to the model, so a
criterion tied to that source should not be classified as a coverage gap when
the matching evidence span is present. Unmatched glob or inventory discovery
still counts only as discovery, not reading.

## Pre-Submit Guard

`run_legal_benchmark_agent` now checks the coverage-adjacent protocol before it
accepts a source-backed final submission. The guard requires complete source
inventory, per-source inspection through the best enabled document helper,
evidence rows, required deliverable validation, and a final self-check. Early
submissions are bounced back to the model with concrete missing steps while
turns remain, which makes hill-climb runs spend remaining budget fixing
coverage and citation gaps before judge scoring.

## Validation

Run:

```bash
cargo test -p psionic-eval --no-default-features --lib legal_benchmark
```

The coverage tests prove integrity mode hides derived hints, hill-climb mode
allows approved checklist items, hidden criterion leaks are detectable, and a
known miss is classified as a coverage gap.

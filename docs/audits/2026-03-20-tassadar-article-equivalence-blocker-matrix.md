# Tassadar Article-Equivalence Blocker Matrix

Date: March 20, 2026

## Result

The repo now has one canonical machine-readable blocker matrix for the
article-equivalence program.

This issue does not close article equivalence. It freezes the bar.

The committed blocker-matrix report is:

- `fixtures/tassadar/reports/tassadar_article_equivalence_blocker_matrix_report.json`

The companion summary is:

- `fixtures/tassadar/reports/tassadar_article_equivalence_blocker_matrix_summary.json`

On the current committed truth:

- the blocker-matrix contract is structurally green
- article equivalence itself remains red by design

That is the intended posture for `TAS-157`.

## Commands Run

- `cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report`
- `cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary`
- `cargo test -p psionic-eval article_equivalence_blocker_matrix -- --nocapture`
- `cargo test -p psionic-research article_equivalence_blocker_matrix_summary -- --nocapture`
- `./scripts/check-tassadar-article-equivalence-blocker-matrix.sh`

## Decisive Outcomes

### 1. The blocker set is now frozen in one machine-readable surface

The committed matrix now freezes seven blocker categories:

- frontend scope
- interpreter breadth
- Transformer-stack reality
- fast-route scope
- benchmark scope
- single-run scope
- weights/ownership scope

Each blocker row now carries:

- one stable blocker id
- one repo-status verdict using the canonical status vocabulary
- one current-gap summary
- one current-public-truth summary
- one explicit closure-requirements list
- one or more exact article line references

That keeps later closure work tied to one declared bar instead of letting the
bar drift issue by issue.

### 2. The article line provenance is now explicit and current

The blocker matrix does not reuse the older line numbering that predated the
2026-03-19 implementation-status note added to the article working copy.

Instead, the report freezes current line provenance against the current reviewed
article text, including the strongest claims for:

- arbitrary C ingress
- WebAssembly interpreter breadth
- transformer-weight execution
- logarithmic-time fast decoding
- Hungarian throughput
- hard-Sudoku benchmark closure
- million-step single-run posture
- interpreter behavior encoded in weights

That matters because stale line citations would undermine the whole point of a
frozen blocker contract.

### 3. The later issue wave is now tied back to blocker ids

The matrix now carries explicit issue coverage rows for:

- prerequisite `TAS-156A`
- every later article-gap issue from `TAS-158` through `TAS-186`
- optional follow-on `TAS-R1`

On the committed truth:

- `prerequisite_transformer_boundary_green = true`
- `all_later_issues_covered = true`
- `all_issue_refs_point_to_known_blockers = true`
- `all_blockers_covered_by_issue_map = true`

That closes the main contract gap this issue was supposed to solve:
the later issue wave can no longer float free of the blocker set.

### 4. The contract is green while the claim stays red

The committed matrix records:

- `matrix_contract_green = true`
- `article_equivalence_green = false`
- `open_blocker_count = blocker_count`

This is the right behavior.

`TAS-157` is not a capability-widening issue. It is a bar-freezing issue.

The contract must therefore succeed as a machine-readable artifact while still
keeping the final article-equivalence verdict red until later issues land.

## Status Judgment

What is now closed:

- one canonical blocker matrix for the article-equivalence program
- one exact article-line provenance surface for each blocker
- one explicit issue-to-blocker coverage map across the full later issue wave
- one checker that proves the matrix stays structurally complete while the
  final claim remains red

What remains outside this issue:

- any positive article-equivalence closure
- any widening of the current public served posture
- any claim that the current route is already a canonical owned
  Transformer-backed article path
- any claim that the article fast path, hard-Sudoku benchmark, single-run
  no-spill posture, or clean-room weight causality are already closed

## Final Judgment

The right current statement is:

`Psionic now has one canonical machine-readable article-equivalence blocker matrix with current article-line provenance and complete later-issue coverage, while article equivalence itself remains explicitly red until the blocker rows actually close.`

That is now true on the committed artifact set, and it is the necessary first
freeze-point for the rest of the article-gap closure wave.

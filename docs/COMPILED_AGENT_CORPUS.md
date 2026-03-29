# Compiled Agent Corpus

> Status: expanded retained compiled-agent corpus, updated 2026-03-29.

## What This Covers

The compiled-agent learning loop now has a materially broader retained narrow
task corpus instead of the original 4-receipt proof bundle.

The current admitted task families stay intentionally narrow:

- provider readiness
- wallet balance and recent earnings
- unsupported refusal

## Current Retained Counts

- 18 retained source receipts
- 12 training receipts
- 6 held-out receipts
- 24 replay samples
- 12 route replay samples
- 12 grounded-answer replay samples
- 6 correction receipts

## Training vs Held-Out Split

Training receipts are replay-eligible and feed the learned artifacts.

Held-out receipts are retained for validator scoring only and are explicitly
excluded from the replay bundle.

This keeps the first loop honest:

- training can move the candidate
- held-out rows check whether the candidate generalized

## Failure Classes Now Retained

- `grounded_answer_mismatch`
- `negated_route_false_positive`
- `tool_argument_mismatch`
- `unexpected_tool_exposure`
- `unsafe_final_outcome`

## Coverage Themes

The expanded source receipts now cover:

- provider-readiness phrasing variation
- provider blocked state
- wallet balance phrasing variation
- recent-earnings phrasing variation
- unsupported requests
- route ambiguity
- negated false positives
- grounded synthesis drift

## Canonical Fixture Set

Source receipts:

- `fixtures/compiled_agent/source/`

Normalized learning ledger:

- `fixtures/compiled_agent/compiled_agent_learning_receipts_v1.json`

Replay bundle:

- `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`

## Entry Point

Regenerate the source receipts, learning ledger, and replay bundle:

```bash
cargo run -q -p psionic-train --bin compiled_agent_receipt_to_replay
```

## Honest Boundary

This is now broad enough to make the first route-model result operationally
more credible than the original toy proof.

It is still a narrow product lane, not a claim of broad autonomous coverage.

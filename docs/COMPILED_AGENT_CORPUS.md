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

- 30 retained source receipts
- 18 training receipts
- 12 held-out receipts
- 36 replay samples
- 18 route replay samples
- 18 grounded-answer replay samples
- 15 replay correction samples

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
- unsupported wallet and provider requests
- explicit negation and exclusion phrasing
- route ambiguity
- grounded synthesis drift
- confidence-edge prompts that try to force an answer without supported facts

## Latest Honest Validator Result

The widened held-out split did what it was supposed to do:

- the route candidate still improves replay matches `13 -> 18`
- the route candidate no longer promotes because the held-out comparison row
  `openagents_wallet_provider_compare_heldout_receipt_v1` exposed a real
  ambiguity regression
- the grounded-answer candidate still promotes with replay matches `12 -> 18`
  and held-out matches `7 -> 10`

This is the right phase-three outcome. Corpus growth made the bounded loop more
credible by surfacing a route hold instead of letting the smaller proof bundle
overclaim generalization.

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

This is now broad enough to make the first bounded learning loop more credible
than the original toy proof.

It is still a narrow product lane, not a claim of broad autonomous coverage.

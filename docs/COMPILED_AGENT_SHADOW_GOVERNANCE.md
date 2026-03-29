# Compiled Agent Shadow Governance

> Status: retained confidence-policy and disagreement-harvesting surface for
> the bounded compiled-agent loop, updated 2026-03-29.

## Why This Exists

The bounded learning loop is not operationally credible if promoted-versus-
candidate disagreements only show up during occasional training reviews.

This layer makes shadow evaluation part of the normal loop by retaining:

- explicit confidence bands and thresholds for the learned modules
- every harvested promoted-versus-candidate disagreement
- machine-readable human-review triggers
- codified rollback thresholds for promoted regressions

## Canonical Fixtures

- `fixtures/compiled_agent/compiled_agent_confidence_policy_v1.json`
- `fixtures/compiled_agent/compiled_agent_shadow_disagreement_receipts_v1.json`

Regenerate them with the normal bounded loop:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

## Current Policy Surface

Route policy:

- promoted artifact: `compiled_agent.baseline.rule_v1.route`
- shadow candidate: `compiled_agent.route.multinomial_nb_v1`
- candidate label: `psionic_candidate`
- confidence bands:
  - `high >= 0.80`
  - `watch >= 0.60 and < 0.80`
  - `review < 0.60`

Grounded-answer policy:

- promoted artifact: `compiled_agent.grounded_answer.multinomial_nb_v1`
- rollback candidate: `compiled_agent.baseline.rule_v1.grounded_answer`
- candidate label: `last_known_good`
- confidence bands:
  - `high >= 0.85`
  - `watch >= 0.65 and < 0.85`
  - `review < 0.65`

Shared rollback discipline:

- any held-out promoted regression count at or above `1` is rollback-eligible
- any admitted-family runtime regression count at or above `1` is rollback-eligible
- ambiguous regressions and low-confidence disagreements queue human review

## Latest Retained Outcome

- confidence policy digest:
  `4a1e25a25f6bcc1e314e516fdadd0411181582844df9c31b68597b4558ce15b8`
- disagreement receipts digest:
  `7636190e5a1901e909874a670f4d71191b9cacf85214d6675042819dc2454cad`
- promoted artifact contract digest:
  `5f4ed2e440803e71b54fc1a97da9c96d7c8b5bc152187a4a5a916af6805994fa`
- XTRAIN receipt digest:
  `5bcaf4f72761ba90693bed44e926cf7c1e5ca418b0d58bc43dfc7e33076042e6`

Current disagreement counts:

- 23 promoted-versus-candidate disagreements retained
- 7 human-review triggers retained
- 2 low-confidence disagreements retained
- 0 rollback-ready regressions retained
- runtime disagreements retained explicitly:
  - 1 route runtime disagreement
  - 2 grounded-answer runtime disagreements

## What Counts As Human Review

Human review is retained instead of silently training through the disagreement
when:

- the promoted and candidate outputs both miss the expected outcome
- a disagreement came from an admitted-family runtime shadow trace and the
  promoted authority was wrong
- a disagreement falls into the `review` confidence band

The current retained runtime human-review rows are:

- `disagreement.compiled_agent.route.openagents_runtime_shadow_compare_receipt_v1`
- `disagreement.compiled_agent.grounded_answer.openagents_runtime_shadow_compare_receipt_v1`

## Honest Boundary

This does not widen the task family and it does not claim broad autonomy.

It makes the current bounded loop stricter and more inspectable:

- shadow disagreement harvesting is now routine
- confidence policy is explicit instead of implied
- human-review triggers are retained artifacts
- rollback thresholds are codified before the loop is widened

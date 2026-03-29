# Compiled Agent XTRAIN

> Status: first validator-gated compiled-agent XTRAIN cycle, updated 2026-03-28.

## Why This Exists

The compiled-agent learning loop is only real if replay samples can produce
candidate module revisions that are then gated by independent validators before
promotion.

This first cycle stays deliberately narrow:

- `route`
- `grounded_answer`

## Candidate Revisions

Route candidate:

- `compiled_agent.route.rule_v2.negation_guard`

Grounded-answer candidate:

- `compiled_agent.grounded_answer.rule_v2.recent_earnings`

## Validator Inputs

- independent module eval surface from `crates/psionic-eval/src/compiled_agent_module_eval.rs`
- replay bundle from `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`

## Canonical Fixtures

- `fixtures/compiled_agent/compiled_agent_route_candidate_module_eval_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_grounded_candidate_module_eval_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_xtrain_cycle_receipt_v1.json`

## Current Truth

- route candidate clears the retained negated-route false-positive case
- grounded-answer candidate keeps the module eval surface non-regressing
- grounded-answer candidate improves replay fidelity on the wallet answer by
  including recent earnings from the receipt-backed facts

## Entry Point

Regenerate the canonical validator outputs:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

This is the first real bounded XTRAIN loop for the compiled-agent path. It does
not try to mutate the whole system at once, and it does not make Tassadar a
blocker.

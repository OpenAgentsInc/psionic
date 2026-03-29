# Compiled Agent XTRAIN

> Status: validator-gated compiled-agent XTRAIN cycle with learned route and grounded-answer artifacts, updated 2026-03-29.

## Why This Exists

The compiled-agent learning loop is only real if replay samples can produce
candidate module revisions that are then gated by independent validators before
promotion.

This first cycle stays deliberately narrow:

- `route`
- `grounded_answer`

## Candidate Revisions

Route candidate:

- `compiled_agent.route.multinomial_nb_v1`

Grounded-answer candidate:

- `compiled_agent.grounded_answer.multinomial_nb_v1`

## Validator Inputs

- independent module eval surface from `crates/psionic-eval/src/compiled_agent_module_eval.rs`
- replay bundle from `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`
- held-out learning-receipt rows from
  `fixtures/compiled_agent/compiled_agent_learning_receipts_v1.json`

## Evidence Boundary

The bounded loop now retains explicit `evidence_class` values across:

- learning receipts
- replay samples and replay bundles
- independent module-eval reports
- the XTRAIN cycle receipt
- the promoted-artifact contract
- decentralized-role receipts

The current phase-three stack is still `learned_lane` only.

Validator and contract generation now refuse silent mixing between
`learned_lane` and `stronger_evidence_lane` rows. That keeps the current loop
Tassadar-ready without letting later exact-execution evidence rewrite what the
learned compiled-agent slice actually proved.

## Canonical Fixtures

- `fixtures/compiled_agent/compiled_agent_route_model_v1.json`
- `fixtures/compiled_agent/compiled_agent_grounded_answer_model_v1.json`
- `fixtures/compiled_agent/compiled_agent_route_candidate_module_eval_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_grounded_candidate_module_eval_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_xtrain_cycle_receipt_v1.json`
- `fixtures/compiled_agent/compiled_agent_promoted_artifact_contract_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_roles_contract_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_role_receipts_v1.json`

## Current Truth

- route candidate is now a replay-trained route model artifact, not a hand-authored keyword guard
- route candidate clears the retained negated-route false-positive case
- route candidate now improves both replay and held-out route matches with no held-out regressions
- grounded-answer candidate is now a replay-trained fact-only model artifact, not a rule revision
- grounded-answer candidate improves replay fidelity on the wallet answer by
  learning the retained recent-earnings answer program from receipt-backed facts
- grounded-answer candidate now also improves held-out wallet phrasing rows
- grounded-answer candidate now passes the independent insufficient-facts and
  conflicting-facts fallback rows
- the promoted-artifact contract now hydrates the retained grounded-answer
  artifact fixture instead of rebuilding a numerically drifted copy
- promoted and candidate module authority is now exported through a retained
  runtime-consumable artifact contract instead of only being implied by docs
- the first pre-network decentralized improvement roles are now typed retained
  contracts and receipts rather than roadmap-only nouns

## Latest Retained Outcome

- route decision: `promote`
- grounded-answer decision: `promote`
- replay bundle digest:
  `ba512235bf016ff3c30283b8b6052bfc13cbc9c07fa8f4d5a3a6d0b60174e8c9`
- source ledger digest:
  `d6d9066e1cfef58d199aa5ea3e97a7e211623366779d2250d47fe2c0a7eacad3`
- grounded-answer model digest:
  `250dfa2deff1b02216195a3c40a4ff7865cf3b45e2f339d8c1f9679b47a0388b`

## Scope Boundary

This makes the training loop more real, but it does not silently claim runtime
promotion in `openagents`.

- `psionic` now trains and retains a route-model artifact from the replay bundle
- validator-gated XTRAIN evaluates that artifact as the route candidate
- `psionic` now also publishes a retained promoted-artifact contract for runtime
  consumers
- runtime adoption into the compiled-agent authority lane remains a separate
  `openagents` promotion step

## Entry Point

Regenerate the canonical validator outputs:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

This is the first real bounded XTRAIN loop for the compiled-agent path. It does
not try to mutate the whole system at once, and it does not make Tassadar a
blocker.

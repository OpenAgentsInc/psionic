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

- route candidate is a replay-trained route model artifact, not a hand-authored
  keyword guard
- route candidate clears the retained negated-route false-positive case and
  improves replay matches `13 -> 18`
- route candidate no longer promotes on the widened held-out split because the
  comparison row
  `receipt.compiled_agent.learning.openagents_wallet_provider_compare_heldout_receipt_v1`
  exposed a real ambiguity regression; held-out matches stay `7 -> 7`
- grounded-answer candidate is a replay-trained fact-only model artifact, not a
  rule revision
- grounded-answer candidate improves replay fidelity `12 -> 18`
- grounded-answer candidate also improves held-out fidelity `7 -> 10`
- grounded-answer candidate continues to pass the independent
  insufficient-facts and conflicting-facts fallback rows
- learned route and grounded artifacts are now normalized through JSON
  roundtrip before their digest is stamped, so the retained artifact digests,
  persisted fixtures, and runtime-loaded payloads all agree
- the promoted-artifact contract now hydrates the retained grounded-answer
  artifact fixture and keeps route authority on the baseline revision when the
  validator says `hold`
- promoted and candidate module authority is exported through a retained
  runtime-consumable artifact contract instead of only being implied by docs
- the first pre-network decentralized improvement roles are typed retained
  contracts and receipts rather than roadmap-only nouns

## Latest Retained Outcome

- route decision: `hold`
- grounded-answer decision: `promote`
- replay bundle digest:
  `a046cd4a216178f44d92e720fcc18c4caf7dd4de1e4826931dd9179f97fee4f8`
- source ledger digest:
  `71f84bc71752ba77c8201249537fdd94c7b9d459c91a5a5f35da3ac623941165`
- route model digest:
  `935ffdec233c08e05b7e9377e9176ac6504518ef2512741e7efb5c90e0b1d1ce`
- grounded-answer model digest:
  `fbce731ad2b680a4cd1983e6127ebd140c8da40a7d1018cbc53b18b17a86ec01`
- XTRAIN cycle receipt digest:
  `e1de89d31db7c4bc03d98c06892c69a938aa13bb24db4bb199b4732fe80d89c4`
- promoted-artifact contract digest:
  `aa77ac9a5f0a342bff489c114927c11c5c54c368a2845b852d7be22600b7418a`

## Scope Boundary

This makes the training loop more real, but it does not silently claim runtime
promotion in `openagents`.

- `psionic` now trains and retains a route-model artifact from the replay bundle
- validator-gated XTRAIN evaluates that artifact as the route candidate and can
  honestly keep it in shadow-only `hold` state when widened evidence says not
  to promote
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

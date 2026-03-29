# Compiled Agent XTRAIN

> Status: validator-gated compiled-agent XTRAIN cycle with promoted learned route and grounded-answer artifacts after the Tailnet-first M5 plus RTX 4080 pilot, updated 2026-03-29.

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

Stronger bounded families under comparison:

- `compiled_agent.route.tfidf_centroid_v1`
- `compiled_agent.grounded_answer.tfidf_centroid_v1`

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
- `fixtures/compiled_agent/compiled_agent_route_tfidf_centroid_model_v1.json`
- `fixtures/compiled_agent/compiled_agent_grounded_answer_tfidf_centroid_model_v1.json`
- `fixtures/compiled_agent/compiled_agent_route_tfidf_centroid_candidate_module_eval_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_grounded_tfidf_centroid_candidate_module_eval_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_stronger_candidate_family_report_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_benchmark_kit_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_benchmark_run_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_runtime_receipt_submission_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_replay_proposal_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_submission_staging_ledger_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_quarantine_report_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_worker_beta_contract_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_worker_receipts_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_worker_dry_run_v1.json`
- `fixtures/compiled_agent/tailnet/compiled_agent_tailnet_m5_node_bundle_v1.json`
- `fixtures/compiled_agent/tailnet/compiled_agent_tailnet_archlinux_node_bundle_v1.json`
- `fixtures/compiled_agent/tailnet/compiled_agent_tailnet_submission_staging_ledger_v1.json`
- `fixtures/compiled_agent/tailnet/compiled_agent_tailnet_quarantine_report_v1.json`
- `fixtures/compiled_agent/tailnet/compiled_agent_tailnet_governed_run_v1.json`
- `fixtures/compiled_agent/compiled_agent_xtrain_cycle_receipt_v1.json`
- `fixtures/compiled_agent/compiled_agent_promoted_artifact_contract_v1.json`
- `fixtures/compiled_agent/compiled_agent_confidence_policy_v1.json`
- `fixtures/compiled_agent/compiled_agent_shadow_disagreement_receipts_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_roles_contract_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_role_receipts_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_role_dry_run_v1.json`

## Current Truth

- the first Tailnet-first governed contributor run is now real:
  - local M5 bundle digest `8a1755d9ffaa3af2aff425129dde232b318c8bf839f707324409e4af390b0c3a`
  - remote RTX 4080 bundle digest `c47df518233bd687bd88b442002840f8fedc45960db8298795db8562a097022d`
  - shared retained external benchmark contract digest
    `9a2a53cc95fdb1a674a0da0612dda1a013718a5756fa7764bde305b49b4174f4`
  - governed run digest
    `dc9ab99b00fa05ae990693b5e758cc728d7d06dcef36bb51b86bf769c7f18b37`
- route candidate is a replay-trained route model artifact, not a hand-authored
  keyword guard
- the route path now promotes honestly under unchanged validator thresholds
- route validation now clears with:
  - eval `3 -> 4`
  - replay `19 -> 25`
  - held-out `10 -> 12`
  - no replay or held-out regressions
- grounded-answer candidate is a replay-trained fact-only model artifact, not a
  rule revision
- grounded-answer candidate now improves replay fidelity `19 -> 25`
- grounded-answer candidate also improves held-out fidelity `9 -> 13`
- grounded-answer candidate continues to pass the independent
  insufficient-facts and conflicting-facts fallback rows
- the promoted-artifact contract now keeps the learned route artifact promoted
  and retains the baseline route as the rollback candidate with label
  `last_known_good`
- learned route and grounded artifacts are now normalized through JSON
  roundtrip before their digest is stamped, so the retained artifact digests,
  persisted fixtures, and runtime-loaded payloads all agree
- sanitized runtime receipts now feed the same learning ledger and replay
  bundle instead of living in a separate runtime-only path
- the promoted-artifact contract now hydrates the retained grounded-answer
  artifact fixture and keeps route authority on the baseline revision when the
  validator says `hold`
- confidence bands, human-review triggers, and rollback thresholds are now
  retained as machine-readable artifacts instead of living only in issue text
- promoted-versus-candidate disagreement harvesting is now part of the normal
  xtrain loop, including admitted-family runtime shadow traces
- promoted and candidate module authority is exported through a retained
  runtime-consumable artifact contract instead of only being implied by docs
- the first pre-network decentralized improvement roles are typed retained
  contracts and receipts rather than roadmap-only nouns
- those same decentralized roles now rerun as a retained boring dry run over
  the stricter bounded corpus and runtime-ingested receipts without weakening
  validator or rollback discipline
- the first outside-compatible benchmark pack now emits external contributor
  receipts in the same bounded ledger shape, with one retained review-required
  negated-wallet row to keep the current route weakness visible
- external benchmark runs, runtime disagreement receipts, and replay proposals
  can now enter a retained staging ledger and quarantine report without
  weakening the evidence-versus-authority boundary
- outside workers can now execute the four admitted early roles on the same
  narrow family and submit governed receipts that are accepted, rejected, or
  routed for review without granting promotion or runtime authority
- stronger bounded candidate families can now be evaluated against the same
  route and grounded-answer contracts without changing the runtime interface
- the retained stronger-family report now keeps the incumbent NB candidates on
  both modules because the TF-IDF centroid family only tied the widened eval,
  replay, and held-out surfaces

## Latest Retained Outcome

- route decision: `promote`
- grounded-answer decision: `promote`
- replay bundle digest:
  `0c818359040ad2ccd3cba75a86e8ac72d8a1a67f3544106fc772c9198db2692c`
- source ledger digest:
  `6dd2c757e2a4534210899a667edb82b1c819d64592d3b4aca07a2a6cf1864812`
- route model digest:
  `cd4d7d6703de508a30a6172baa753ece9b1da1e49b54c541cedb847427d8a2ac`
- grounded-answer model digest:
  `5a7bd1af3709dc6c24fb75627c898c37993ee3946f402ce0761fc5adc211a708`
- stronger route model digest:
  `f6d653e5b16fc7f99420f7ec1f99764da05981a761d53af70de4374136f512c3`
- stronger grounded-answer model digest:
  `3adc205ed4b2855a8cad36fe61a1d9c40d9ffac2733ed464e5b73409b403e413`
- stronger candidate family report digest:
  `04112b458376e968b518c0c1653ed939c2748dd471592c408e2574029bbc8479`
- XTRAIN cycle receipt digest:
  `4f7655b1b65931c538c3fbea643452a8a16e1ad7738ae4a9e12896ef722cef45`
- promoted-artifact contract digest:
  `80b130858c414d13f2351a2ff3a2b4e7597ad7b20cd467922883c5ce90981720`
- confidence policy digest:
  `51c1182a1e0c699ab878182c72e7ee7ba0eb83c42f41203edce080bde3595fb1`
- shadow disagreement receipts digest:
  `0548cec6c9a09dfc07e6aea3623467763adccf4d308ac5c7ddd369bdfa60e0b8`
- decentralized roles contract digest:
  `303bf8445d4afbae4a329b2cb9d3b9be8619f77aa1113d9b96fb0b62b2ef81fc`
- decentralized role receipts digest:
  `a9bcc9c0f4042c5d690b3ee99ff51bb20d6a29372dc592ba0465d7fecc634dc7`
- decentralized role dry-run digest:
  `1895a9d50c00d49261e8e00ccf3cdbca4fa38b098407b29aca8ea5ed3810192a`
- external worker beta contract digest:
  `0faec27692dca082fbd58837b7722ba20bcda22ac3f3db0b0f97309abb23539a`
- external worker receipts digest:
  `e9b42272444d0b0781c07ffc141a8c3642e66093dda65b17f0c4d287d971ac13`
- external worker dry-run digest:
  `e576a14307d2149312d2dcd3dca92b35b39e4fc1750e0c52d85cdb82336a8fce`

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

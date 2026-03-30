# Compiled Agent XTRAIN

> Status: validator-gated compiled-agent XTRAIN cycle with promoted learned route and grounded-answer artifacts after the retained phase-six rerun, updated 2026-03-29.

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
- `fixtures/compiled_agent/compiled_agent_phase_six_operational_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_promoted_artifact_contract_v1.json`
- `fixtures/compiled_agent/compiled_agent_confidence_policy_v1.json`
- `fixtures/compiled_agent/compiled_agent_shadow_disagreement_receipts_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_roles_contract_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_role_receipts_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_role_dry_run_v1.json`

## Current Truth

- the Tailnet-first loop is now retained as a repeatable phase-six run instead
  of a one-off demo:
  - local M5 bundle digest `5a8f3fa13bfd67c1609251f4f9a4ad9c514e688d892c3f6f16e155a46ed9dda9`
  - remote RTX 4080 bundle digest `f2682730f546c68715e66bd0bdb2144a792c1eacb44838feefa4ab59b4b90ec4`
  - governed run digest
    `a6f3caf208fb510793d048fa44e8bab8f2761282f0a1c6d42d0845fe8208f2dc`
  - phase-six operational report digest
    `b0d4061de01e35a21b83ff6c2f57fb5737905420cfb110d031c0116c3fabad86`
- route candidate is still a replay-trained route model artifact, not a
  hand-authored keyword guard
- the route path still promotes under unchanged validator thresholds
- route validation now clears with:
  - eval `3 -> 4`
  - replay `22 -> 28`
  - held-out `13 -> 14`
  - no replay or held-out regressions
- grounded-answer candidate is still a replay-trained fact-only model artifact,
  not a rule revision
- grounded-answer candidate now clears:
  - replay `21 -> 28`
  - held-out `12 -> 16`
  - no replay or held-out regressions
- the promoted-artifact contract now reflects the latest retained route and
  grounded artifacts after the widened phase-six rerun
- confidence bands, human-review triggers, rollback thresholds, and
  promoted-versus-candidate disagreement harvesting are still retained as
  machine-readable artifacts
- route now also retains permanent regression-trap held-out rows for compare,
  exclusion, negation, and provider-vs-wallet ambiguity instead of only the
  earlier phase-five failures
- external intake now adds anomaly flags and contributor trust posture while
  keeping the same staging -> quarantine -> replay boundary
- the retained stronger-family report still keeps the incumbent NB candidates
  on both modules because the TF-IDF centroid family does not beat the widened
  eval, replay, and held-out surfaces honestly

## Latest Retained Outcome

- route decision: `promote`
- grounded-answer decision: `promote`
- replay bundle digest:
  `85d108ecebb9ae71bb005709c8b643c1b7eca86058786c01d7501faa40ca9036`
- source ledger digest:
  `22f790143a9da07527548333d4b9d53e811fbecc74b191722a3f4e94fb1d4b9e`
- route model digest:
  `2b66abacb8647f719f9b9a46a8cef007a5026b18c27af998cf2351e7a7a4560c`
- grounded-answer model digest:
  `869217d751e61e52f32f1dfdd0f5dc18d3e9c0a1d15dce0f86066356075b2782`
- stronger route model digest:
  `4230c12f354af0cbde18f0d0f7fe128758dcb085750086f01f361b981cac5336`
- stronger grounded-answer model digest:
  `47acc85795914944fec15dd706f5c553a210e8a52484b77b4ce964eafe349b1c`
- stronger candidate family report digest:
  `fc1c32a58dbc93aeddab9fb5fcc913b59809d129c44bc8b995fd1036e2375298`
- XTRAIN cycle receipt digest:
  `b432ca5f00bffae428592411712bcae980262038bd684aa7ad2f6f39b8d49073`
- promoted-artifact contract digest:
  `5835f484b83deb27ac7a7a96ae909011dd02a0c74582109b11ae04b3dfbeada4`
- confidence policy digest:
  `a38c5a52ba39e60d3b9f7d6396f4dacb9e7c641b34b1cdb9fa5d2b03490b317e`
- shadow disagreement receipts digest:
  `be6c101f629f8c70afa4f293c690fc50af1468f5fa43d3ccb9093ccc1095fe5d`
- phase-six operational report digest:
  `b0d4061de01e35a21b83ff6c2f57fb5737905420cfb110d031c0116c3fabad86`

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

This is still the bounded XTRAIN loop for the compiled-agent path. Phase six
does not widen the family or relax the gates. It keeps the loop repeatable and
harder to game while the same narrow evidence family grows.

# Compiled Agent Decentralized Roles

> Status: first governed pre-network decentralized compiled-agent role set,
> updated 2026-03-29.

## Why This Exists

The compiled-agent learning loop is now real enough that the first
decentralized roles can stop being abstract roadmap language.

These roles are deliberately narrow:

- replay generation
- ranking and labeling
- validator scoring
- bounded module training

They are governed interfaces over retained compiled-agent artifacts. They are
not claims about broad autonomous workers, open incentives, or validator-free
promotion.

## Canonical Fixtures

- `fixtures/compiled_agent/compiled_agent_decentralized_roles_contract_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_role_receipts_v1.json`

## Current Authority Inputs

The role contract binds directly back to the retained compiled-agent loop:

- `fixtures/compiled_agent/compiled_agent_learning_receipts_v1.json`
- `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`
- `fixtures/compiled_agent/compiled_agent_default_row_v1.json`
- `fixtures/compiled_agent/compiled_agent_route_model_v1.json`
- `fixtures/compiled_agent/compiled_agent_grounded_answer_model_v1.json`
- `fixtures/compiled_agent/compiled_agent_route_candidate_module_eval_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_grounded_candidate_module_eval_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_xtrain_cycle_receipt_v1.json`
- `fixtures/compiled_agent/compiled_agent_promoted_artifact_contract_v1.json`

Every retained authority input in this first phase-three role contract now
preserves an explicit `evidence_class` when that artifact is part of the
learned compiled-agent loop. The current contract and retained role receipts
stay in `learned_lane`. They do not silently absorb stronger-evidence or exact
execution rows.

## What Each Role Does

`replay_generation`

- starts from the retained learning ledger
- emits replay rows for route and grounded-answer training
- still requires human review before new rows enter training

`ranking_labeling`

- curates labels and corpus placement over retained receipts and replay rows
- keeps accepted labels explicit instead of hiding them in ad hoc notebooks
- still requires human review before labels become training or held-out truth

`validator_scoring`

- scores bounded candidates against independent module eval, replay, and held-out rows
- emits the bounded XTRAIN receipt and promoted-artifact contract
- is still gate-kept by validator rules and rollback authority

`bounded_module_training`

- trains narrow candidate modules on the locked default learned row
- currently targets route and grounded-answer only
- still queues outputs for validator scoring before promotion

## Local Reference Runner

Regenerate the canonical role contract and retained role receipts:

```bash
cargo run -q -p psionic-train --bin compiled_agent_decentralized_roles
```

Inspect one role and its retained local receipt:

```bash
cargo run -q -p psionic-train --bin compiled_agent_decentralized_roles -- --role validator_scoring
```

Supported role selectors:

- `replay_generation`
- `ranking_labeling`
- `validator_scoring`
- `bounded_module_training`

## Honest Boundary

This is the first governed decentralized improvement surface for the
compiled-agent path.

It does prove:

- each early decentralized role has a machine-readable contract
- each role has retained local receipts
- each role links back into the real compiled-agent learning loop
- each role now preserves the learned-lane evidence boundary explicitly so
  future stronger-evidence lanes cannot be mixed in by accident

It does not prove:

- open public incentives
- autonomous swarm behavior
- validator-free promotion
- broad decentralized agent autonomy

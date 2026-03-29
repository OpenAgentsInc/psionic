# Compiled Agent Learning Receipts

> Status: canonical `AGENT-PLATFORM` receipt-to-replay substrate, updated 2026-03-29.

## Why This Exists

The compiled-agent vertical slice is not useful for XTRAIN unless real runtime
outputs can be normalized into Psionic-owned learning receipts and replay
bundles.

This layer is the first narrow handoff:

- raw compiled-agent run receipts from the app slice
- normalized learning receipts with expected behavior attached
- replay samples focused on the first two bounded training surfaces:
  route and grounded answer

## Canonical Fixtures

Source receipts captured from the existing `openagents` harness:

- `fixtures/compiled_agent/source/openagents_provider_ready_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_provider_blocked_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_provider_readiness_variant_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_recent_earnings_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_recent_earnings_phrase_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_balance_variant_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_unsupported_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_unsupported_restart_rig_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_unsupported_calendar_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_ambiguous_provider_wallet_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_negated_wallet_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_provider_account_ready_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_balance_phrase_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_ambiguous_provider_wallet_heldout_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_negated_provider_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_earnings_phrase_heldout_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_unsupported_schedule_meeting_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_exclusion_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_provider_exclusion_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_address_unsupported_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_best_provider_unsupported_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_recent_earnings_short_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_go_online_guess_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_address_heldout_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_best_provider_heldout_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_negated_wallet_provider_heldout_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_negated_provider_wallet_heldout_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_recent_earnings_short_heldout_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_provider_compare_heldout_receipt_v1.json`

Sanitized runtime receipts admitted into the same governed schema and ledger:

- `fixtures/compiled_agent/runtime/openagents_runtime_shadow_compare_receipt_v1.json`
- `fixtures/compiled_agent/runtime/openagents_runtime_wallet_recent_earnings_receipt_v1.json`

Normalized Psionic-owned artifacts:

- `fixtures/compiled_agent/compiled_agent_learning_receipts_v1.json`
- `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`

## What The Ledger Keeps Explicit

- source receipt lineage and digest
- source fixture family, including tracked `source/` and `runtime/` provenance
- expected route and expected public response
- observed route, tool calls, tool results, and public response
- per-module correctness flags
- failure classes
- phase confidence map
- primary authority manifests and shadow candidate manifests
- corpus split between training and held-out
- operator note for why the row matters

## What The Replay Bundle Keeps Explicit

- route replay samples
- grounded-answer replay samples
- behavioral-clone rows for already-correct behavior
- failure-correction rows for retained mistakes
- exclusion of held-out receipts from the training replay bundle

The first correction row is deliberate:

- `openagents_negated_wallet_receipt_v1.json`

That raw source receipt exposed `wallet_status` for a negated unsupported
request. The replay bundle preserves that drift as a correction target instead
of hiding it.

## Runtime Ingestion Contract

Real admitted-family runtime evidence now enters the exact same governed receipt
shape as the retained benchmark-style rows.

The runtime path keeps explicit:

- promoted artifact manifest lineage
- shadow candidate lineage when a compare run happened
- confidence values for the recorded phases
- correction-required rows when operator review overruled the promoted answer
- the same learning-ledger and replay-bundle governance used by synthetic rows

This means runtime evidence does not bypass the validator contract. It is
normalized into the same ledger, the same replay bundle, and the same held-out
split discipline before it can influence XTRAIN.

## Entry Point

Regenerate both canonical fixtures from the repo root:

```bash
cargo run -q -p psionic-train --bin compiled_agent_receipt_to_replay
```

## Current Truth

- 30 retained benchmark-style source receipts under `source/`
- 2 retained sanitized runtime receipts under `runtime/`
- 32 governed learning receipts total
- 19 training receipts
- 13 held-out receipts
- 19 correction receipts
- 38 replay samples total
- 19 route replay samples
- 19 grounded-answer replay samples
- 17 replay correction samples
- learning ledger digest:
  `48ebcfa41ae8f52a80745eb803be332e04596d63a293b965df260382fde07f83`
- replay bundle digest:
  `da0e79fdfdea3b751fd90e84178b219693d5e3a348c675ebad8d4eeda25c600a`

The current failure-class ledger retains:

- `grounded_answer_mismatch`
- `negated_route_false_positive`
- `tool_argument_mismatch`
- `unexpected_tool_exposure`
- `unsafe_final_outcome`

The runtime rows currently prove two important things:

- real admitted-family traffic can enter the governed training split without a
  second receipt format
- promoted-versus-candidate disagreement can be retained as a first-class
  correction row instead of disappearing into runtime logs

This is enough to make the first bounded XTRAIN loop real without pretending
the whole serving stack or Tassadar lane must already be finished.

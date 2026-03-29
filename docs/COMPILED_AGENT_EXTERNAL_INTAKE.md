# Compiled Agent External Intake

> Status: governed external evidence staging and quarantine for the admitted
> compiled-agent family, updated 2026-03-29.

## Why This Exists

The external benchmark kit is only useful if outside-produced evidence can
enter the bounded learning loop without weakening the evidence-versus-authority
boundary.

This intake layer is deliberately narrow:

- one admitted compiled-agent family
- one external benchmark and receipt contract
- one staging ledger for outside submissions
- one quarantine report before replay or training admission

It does not grant outside promotion authority, live runtime authority, or raw
log admission into replay.

## Canonical Fixtures

- `fixtures/compiled_agent/external/compiled_agent_external_runtime_receipt_submission_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_replay_proposal_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_submission_staging_ledger_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_quarantine_report_v1.json`

## What Enters Intake

The first retained intake path accepts only three outside contribution shapes:

- benchmark runs from the bounded external benchmark kit
- runtime disagreement receipts on the same admitted family
- replay proposals derived from quarantined external receipts

Every submission must preserve:

- contributor identity
- source machine identity
- machine and environment class
- accepted contract version
- evidence class
- payload digest
- validator status
- explicit failure classes where relevant

## Staging Rules

The staging ledger separates accepted, rejected, and review-required submissions.

Rejected rows fail before quarantine when they drift on:

- schema
- digest integrity
- contract linkage
- benchmark consistency
- required environment metadata

Accepted and review-required rows still stay outside authority. They are
retained only as quarantined evidence until the existing validator and review
logic says otherwise.

## Quarantine Rules

The quarantine report keeps outside evidence reviewable instead of silently
admitted.

It retains:

- accepted submission ids
- rejected submission ids
- review-required submission ids
- replay-candidate receipt ids
- proposed replay sample ids
- promoted-versus-candidate shadow assessments

That means the first external runtime disagreement can be compared against the
promoted route authority and the shadow learned route candidate without letting
either one rewrite live truth.

## Retained Shape

The first retained intake path keeps one important external disagreement live:

- the negated wallet regression row is still visible as an external runtime
  disagreement receipt
- that disagreement is shadow-scored against promoted-versus-candidate route
  and grounded-answer authority
- the derived replay proposal is structurally valid but still review-required

This is the exact boundary the bounded learning loop needs: outside evidence is
admitted as evidence, not mistaken for authority.

## Local Runner

Generate or refresh the retained external intake fixtures:

```bash
cargo run -q -p psionic-train --bin compiled_agent_external_intake
```

Verify the retained intake fixtures against the canonical generator:

```bash
cargo test -q -p psionic-train compiled_agent_external_intake -- --nocapture
```

## Honest Boundary

This intake layer proves that external benchmark rows, runtime disagreement
receipts, and replay proposals can flow into the same governed contract shape
the internal loop already uses.

It does not yet prove:

- external worker-role execution
- outside-facing product onboarding
- contributor trust or accounting surfaces
- broad external traffic ingestion

The next retained boundary after intake is now described in
`docs/COMPILED_AGENT_EXTERNAL_WORKERS.md`.

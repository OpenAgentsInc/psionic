# Compiled Agent External Benchmark Kit

> Status: first outside-compatible benchmark and receipt package for the
> admitted compiled-agent family, updated 2026-03-29.

## Why This Exists

The bounded compiled-agent loop now has enough retained structure that external
systems can contribute benchmark evidence without changing the governance
model.

This kit is deliberately narrow:

- one admitted compiled-agent family
- the existing learned-lane evidence boundary
- the existing receipt schema and replay contract
- the existing validator and promotion discipline

It does not claim broad agent evaluation, open promotion authority, or raw-log
admission into training.

## Canonical Fixtures

- `fixtures/compiled_agent/external/compiled_agent_external_benchmark_kit_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_benchmark_run_v1.json`

Current retained truth:

- contract digest:
  `dbfad2b850ad1ff71e6528be9e4a3b0f6f38cc273df7c6f79f75c5b45e02ccb8`
- run digest:
  `0d26ba6d9805902e8d0284883d1c5b79ef3299cfa14d7f8b247d40bca7af6283`
- accepted rows: `5`
- review-required rows: `1`

## Contract Inputs

The external benchmark kit binds back to the same retained bounded-loop
artifacts the internal path already uses:

- `fixtures/compiled_agent/compiled_agent_learning_receipts_v1.json`
- `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`
- `fixtures/compiled_agent/compiled_agent_decentralized_roles_contract_v1.json`
- `fixtures/compiled_agent/compiled_agent_promoted_artifact_contract_v1.json`

The generated run normalizes its outputs into the same source-receipt and
learning-receipt shape:

- route decision
- tool exposure and tool calls
- grounded-answer output
- verify outcome
- phase confidences
- artifact identity and artifact digest
- failure taxonomy
- contributor identity and environment metadata

## Admitted Rows

The first retained external package stays on six narrow rows:

- canonical provider-ready
- provider-blocked
- wallet balance
- wallet wording variant
- unsupported refusal
- negated wallet regression row

The negated wallet row is intentionally retained as `review_required` because
the current promoted route authority is still the baseline rule path. That row
keeps the current weakness visible instead of hiding it behind synthetic
success.

## External Source Contract

Every external run is marked as externally sourced and carries:

- contributor identity
- source machine identity
- machine class
- environment class
- declared capabilities
- accepted contract version
- evidence class
- validator outcome per row

This keeps external evidence inside the same hard rule the learning loop
already uses: evidence is not authority.

## Local Runner

Generate or refresh the retained external benchmark contract and run fixtures:

```bash
cargo run -q -p psionic-train --bin compiled_agent_external_benchmark_kit
```

Print the canonical external benchmark run without rewriting fixtures:

```bash
cargo run -q -p psionic-train --bin compiled_agent_external_benchmark_kit -- --run
```

Verify the retained fixtures against the canonical generator:

```bash
cargo test -q -p psionic-train compiled_agent_external_benchmark -- --nocapture
```

## Honest Boundary

This package proves that an outside system can produce:

- a canonical bounded benchmark contract
- governed external receipts
- normalized route and grounded-answer outcomes
- explicit review-required disagreement rows

It does not yet prove:

- external evidence ingestion
- quarantine and acceptance logic
- external worker-role execution
- contributor-facing product surfaces

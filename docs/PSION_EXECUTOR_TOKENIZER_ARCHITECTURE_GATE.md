# Psion Executor Tokenizer / Architecture Gate

> Status: canonical `PSION-0806` / `#781` record, updated 2026-03-30 after
> landing the first executor evidence gate for tokenizer and architecture work.

This document records the first retained gate packet that decides when
tokenizer work or architecture work is actually allowed to open.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_tokenizer_architecture_gate_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_tokenizer_architecture_gate_fixtures
```

## What Landed

`psionic-train` now owns one typed evidence-gate packet that binds:

- the retained optimizer ablation packet
- the retained scheduler ablation packet
- the retained batch / accumulation ablation packet
- the retained trace-family weighting ablation packet
- the retained supervision-density ablation packet

The retained packet makes two rules durable:

- tokenizer work stays blocked until compression or fit limits are real
- architecture work only becomes eligible once the full five-run same-baseline
  ablation tranche exists

## Current Retained Truth

- packet digest:
  `PENDING_PACKET_DIGEST`
- required successful ablation runs:
  `5`
- successful same-baseline run count:
  `5`
- tokenizer issue open allowed:
  `false`
- architecture issue open allowed:
  `true`
- compression-limit evidence present:
  `false`
- fit-limit evidence present:
  `false`
- review decision:
  `keep_tokenizer_blocked_allow_architecture_only_after_ablation_tranche`

## Honest Current Meaning

This packet does not claim that tokenizer work is obsolete forever, and it does
not claim that architecture work should start immediately.

It does claim the gating truth the roadmap asked for:

- the executor lane now has enough same-baseline ablation evidence to open an
  architecture issue if the current candidate path runs out of room
- the executor lane still does not have honest compression or fit-limit
  evidence, so tokenizer work remains blocked

## Validation

- `cargo run -q -p psionic-train --example psion_executor_tokenizer_architecture_gate_fixtures`
- `cargo test -q -p psionic-train builtin_executor_tokenizer_architecture_gate_packet_is_valid -- --exact --nocapture`
- `cargo test -q -p psionic-train tokenizer_architecture_gate_fixture_matches_committed_truth -- --exact --nocapture`

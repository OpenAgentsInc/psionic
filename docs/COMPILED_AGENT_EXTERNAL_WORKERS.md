# Compiled Agent External Workers

> Status: governed external worker beta for the admitted compiled-agent family,
> updated 2026-03-29.

## Why This Exists

The bounded learning loop now has enough governance that outside operators can
do useful work without weakening the evidence boundary.

This beta stays deliberately narrow:

- replay generation
- ranking and labeling
- validator scoring
- bounded module training

Outside workers still do not get:

- promotion authority
- live runtime authority
- task-family widening authority
- validator bypass

## Canonical Fixtures

- `fixtures/compiled_agent/external/compiled_agent_external_worker_beta_contract_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_worker_receipts_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_worker_dry_run_v1.json`

Current retained truth:

- worker beta contract digest:
  `0faec27692dca082fbd58837b7722ba20bcda22ac3f3db0b0f97309abb23539a`
- worker receipts digest:
  `e9b42272444d0b0781c07ffc141a8c3642e66093dda65b17f0c4d287d971ac13`
- worker dry-run digest:
  `e576a14307d2149312d2dcd3dca92b35b39e4fc1750e0c52d85cdb82336a8fce`

## What The Contract Binds To

The external worker beta stands on top of the same governed retained loop:

- the internal decentralized-role contract, receipts, and dry run
- the external benchmark kit and benchmark run
- the external runtime disagreement receipt
- the external replay proposal
- the external staging ledger and quarantine report
- the promoted-artifact contract
- the XTRAIN receipt
- the stronger-family report
- the confidence policy and shadow disagreement receipts

That means outside worker output is still judged by the same validator and
rollback logic that already governs the internal bounded loop.

## Role Boundaries

`replay_generation`

- can normalize admitted external evidence into a replay proposal
- can feed the same bounded training queue
- cannot skip replay admission review

`ranking_labeling`

- can curate review-required external rows
- can send those rows back into staging and quarantine
- cannot turn curation directly into authority

`validator_scoring`

- can score bounded candidates against retained validator surfaces
- can emit a governed review packet
- cannot move promotion or runtime authority

`bounded_module_training`

- can train bounded route and grounded candidate artifacts
- can queue those artifacts for validator scoring
- cannot promote them directly

## Governance Shape

Every retained worker submission keeps:

- contributor identity
- machine and environment class
- accepted contract version
- input refs and output refs
- source submission ids
- source receipt ids
- validator status
- quarantine status
- review state

That is the important part: outside work now enters the same governed contract
shape as internal work. It can be accepted, rejected, or routed for review
without inventing a second control plane.

The retained beta currently keeps:

- accepted outside submissions: `2`
- rejected outside submissions: `1`
- review-required outside submissions: `1`

## Local Runner

Write or refresh the external worker beta contract, receipts, and dry run:

```bash
cargo run -q -p psionic-train --bin compiled_agent_external_workers
```

Print the retained dry run:

```bash
cargo run -q -p psionic-train --bin compiled_agent_external_workers -- --dry-run
```

Inspect one role contract and its retained submission receipt:

```bash
cargo run -q -p psionic-train --bin compiled_agent_external_workers -- --role replay_generation
```

Supported role selectors:

- `replay_generation`
- `ranking_labeling`
- `validator_scoring`
- `bounded_module_training`

## Honest Boundary

This proves that outside operators can execute bounded roles and produce
governed outputs on the admitted compiled-agent family.

It does not yet prove:

- a public market
- open incentives
- broad runtime traffic from contributors
- external promotion authority

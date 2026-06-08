# Psionic Qwen Legal Pylon Boundary v0.2.0

Date: 2026-05-22

## Boundary Identity

- Psionic boundary version: `0.2.0`
- Boundary gate: `scripts/check-v0.2-pylon-release.sh`
- Retained boundary-readiness run:
  `reports/qwen-legal-v02-pylon-release-readiness-20260522.md`
- Primary crate surface: `psionic-train`
- Pylon worker entrypoint:
  `cargo run -p psionic-train --bin qwen_legal_pylon_worker_server -- <bind-addr> <worker-id>`
- Scheduler dispatch modes: `LocalOnly`, `Loopback`, `Tailnet`, `Production`

This document is not the OpenAgents public Pylon v0.2 release record. It is the
Psionic-side Qwen legal Pylon worker and scheduler boundary that later
OpenAgents Pylon releases can consume.

## What The Boundary Supports

- Signed Qwen legal Pylon job envelopes.
- Local and loopback Pylon worker execution.
- Tailnet and production TCP worker dispatch using signed job envelopes.
- Worker-side scheduler signature verification before job execution.
- Scheduler-side worker receipt signature verification before payment marking.
- Per-job transport telemetry: request bytes, response bytes, response digest,
  worker receipt digest, and signature verification flags.
- Per-worker receipt telemetry: input/output counts, required output count,
  output bytes, runtime, shard coordinates, budget, and success state.
- Payable, withheld, duplicate-shard, deferred, failed-payment, and
  live-small-value operator-approved settlement proof paths.
- Public-network, reward-ledger, settlement-publication, bootstrap, explorer,
  and open/incentivized decentralized-run contract checkers.

## Payment Boundary

Psionic v0.2.0 proves the payment decision, Treasury handoff, settlement proof
validation, and promotion-gate logic. It does not custody wallet secrets and it
does not execute wallet sends directly. Live payment execution remains owned by
Treasury or Nexus, which must return a proof containing a settlement time plus
a payment hash or transaction proof. Psionic rejects duplicate proofs, unknown
authorizations, amount mismatches, bad proof digests, and secret-looking proof
fields.

## Required Boundary Validation

Run from the repository root:

```bash
scripts/check-v0.2-pylon-release.sh
```

The gate covers:

- provider-neutral evidence bundle stability
- cross-provider run graph stability
- decentralized network and public registry contracts
- public work assignment, dataset authority, miner protocol, validator scoring,
  consensus, fraud/slashing, reward ledger, settlement publication, bootstrap,
  explorer, readiness, curated/open/incentivized public-run contracts
- `qwen_legal_pylon_worker_server` compile check
- Qwen legal Pylon dispatch tests, including signed remote TCP dispatch
- Qwen legal Pylon payment and settlement tests
- Qwen legal Pylon network SFT fixture regeneration

## Operator Notes

- For Tailnet, bind the worker server to a Tailnet-reachable `host:port` and
  register the node with `tcp://host:port` or `tailnet://host:port`.
- For production, put the TCP worker behind the admitted private network or
  gateway policy. The envelope and receipt signatures are repo-level integrity
  checks; network admission, firewalling, and wallet execution remain operator
  responsibilities.
- Do not store wallet secrets in Psionic settlement proofs.

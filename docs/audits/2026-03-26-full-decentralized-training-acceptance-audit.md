# 2026-03-26 Full Decentralized Training Acceptance Audit

## Bottom Line

Psionic now has a coherent decentralized-training contract stack from public
identity and discovery through WAN runtime, public work assignment, validator
governance, fraud policy, reward accounting, settlement publication, operator
packages, public explorer visibility, staged testnet admission, curated runs,
open public participation, and one incentivized rewarded closeout.

## What Psionic Now Proves

- typed public-network authority and signed node identity
- public registry, discovery, elastic mesh, route policy, catch-up, and
  quantized outer-sync surfaces
- deterministic public work assignment, dataset truth, artifact exchange, and
  public miner/validator protocols
- multi-validator consensus, fraud policy, reward accounting, and signed-ledger
  settlement publication
- reproducible operator packages, public explorer visibility, staged public
  readiness, one curated run, one open public run, and one incentivized closeout

## Comparison To `prime-diloco`

- Psionic now has Psionic-native equivalents for the core elastic internet
  runtime layers: join and leave truth, catch-up, outer sync, and retained
  fault evidence.
- Psionic remains more contract-heavy and proof-oriented than `prime-diloco`.
  That is a strength for retained acceptance evidence, but it also means the
  current implementation still favors explicit typed closure over raw runtime
  throughput claims.

## Comparison To `templar`

- Psionic now has Psionic-native equivalents for public miner and validator
  surfaces, fraud discipline, reward accounting, staged onboarding, and public
  visibility.
- Psionic still does not claim Templar's exact chain-native incentive design or
  its exact network incentives. It instead proves a signed-ledger settlement
  route with explicit optional chain refusal.

## Fresh Validation State

On 2026-03-27 the full decentralized checker sweep was rerun successfully on
current `main`, from `check-decentralized-network-contract.sh` through
`check-incentivized-decentralized-run-contract.sh`.

The stale failure story from the first draft of this acceptance audit is no
longer current. `psionic-backend-cuda` no longer stops the lane on a missing
`PlatformSubmission::allocate`, `elastic_device_mesh_contract.rs` now keeps
relay leases above the stale-peer timeout, `live_checkpoint_catchup_contract.rs`
now binds completed restore assignments against the serving record's
`source_id`, and the pure canonical builders used by later decentralized
generators are memoized so the fresh checker path no longer degenerates into
recursive recomputation.

## Remaining Honest Limits

- the first incentivized run still pays the current reward-eligible set rather
  than the outside canary participants
- optional chain publication remains refused rather than shipped

## Acceptance

Within the claim boundaries above, Psionic can now be described as having full
decentralized training runs under Psionic-native contracts and retained public
evidence.

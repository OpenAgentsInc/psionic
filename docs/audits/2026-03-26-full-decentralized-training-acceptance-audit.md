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

## Remaining Honest Limits

- the current cargo-backed generators and tests are still blocked by the
  unrelated `psionic-backend-cuda` compile failure on
  `PlatformSubmission::allocate` in `crates/psionic-backend-cuda/src/lib.rs:876`
- the first incentivized run still pays the current reward-eligible set rather
  than the outside canary participants
- optional chain publication remains refused rather than shipped

## Acceptance

Within the claim boundaries above, Psionic can now be described as having full
decentralized training runs under Psionic-native contracts and retained public
evidence.

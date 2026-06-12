# Psionic Collective Failure Semantics

> Status: canonical `psionic#1126` record, 2026-06-12. Pluralis roadmap P2.1:
> chunked transfers, ban-for-round, partial-result preservation.
>
> Authored by Fable (claude-fable-5) for psionic#1126.

This document records the collective-op failure semantics landed in
`crates/psionic-collectives/src/failure_semantics.rs` under schema version
`psionic.collectives.failure_semantics.v1`.

## Canonical Runner

Run the contract checker from the repo root:

```bash
scripts/check-collective-failure-semantics.sh
```

The checker regenerates the committed fixture with
`cargo run -p psionic-collectives --bin collective_failure_semantics_contract`
and verifies `fixtures/training/collective_failure_semantics_v1.json` against
it plus the invariants below.

## Upstream Reference

The semantics are ported from the Pluralis agora fault-tolerance rules in
`docs/agora-system/fault-tolerance.md` inside the read-only reference lane
`projects/pluralis/repos/agora`:

- Tensors are divided into smaller chunks with a strict per-chunk time limit
  on the send path, and a reciprocal timeout protects the return path where
  reducers send reduced chunks back.
- A failing sender is banned for the round, not retried. Failures are logged
  and the round continues.
- Chunking preserves partial all-reduce results; valid work from healthy peers
  is never discarded because one peer failed.
- A full-round timeout protects against additional stalls; the round retains
  its partial reduce result.
- Contributors that consistently fail are removed from the swarm.
- A peer failure is fatal only if an entire pipeline stage empties.

## Ported Semantics

The crate models the round as a deterministic state machine:

- `plan_tensor_chunks` divides a tensor byte-length into chunks with explicit
  `chunk_bytes`, `per_chunk_timeout_ms`, `reciprocal_per_chunk_timeout_ms`,
  and `round_timeout_ms` budgets from `CollectiveRoundConfig`. There are no
  built-in magic constants; every budget is caller-supplied config.
- `CollectiveFailureRound` accepts `ChunkTransferOutcome` observations
  (`delivered`, `timed_out`, `disconnected`) on both `send` and
  `reciprocal_return` paths. A delivered chunk whose elapsed time exceeds the
  path budget counts as a timeout.
- A member's first failed chunk emits a typed, digest-carrying
  `BanForRoundEvent` and bans the member for the rest of the round. Later
  outcomes from a banned member are ignored, never retried.
- Chunks a member delivered before its ban stay in the round's reduce
  accounting. The receipt carries explicit `included_contributions` and
  `excluded_contributions` lists with typed exclusion reasons
  (`failed_transfer`, `excluded_after_ban`, `not_delivered`) and a
  `preserved_fraction_bps` over expected chunk deliveries.
- `finalize` applies the full-round timeout: members still owing chunks when
  the round budget elapses are banned with `full_round_timeout_stall`, and the
  round keeps its partial result.
- The round verdict is `Complete`, `Partial { preserved_fraction_bps,
  banned_node_ids, promoted_standby_node_id }`, or `Aborted { reason }` with
  `no_standby_promotable` or `round_timeout`. A round aborts only when the
  stage would empty (every member banned) and the typed
  `StandbyPromotionDecision` says no warm standby is promotable. With a
  promotable standby the round stays `Partial` and records the promotion.
- `CollectiveBanLedger` accumulates per-round bans under a
  `BanEscalationPolicy` (`removal_ban_threshold` bans within the trailing
  `window_rounds`) and emits a receipt-compatible
  `SwarmRemovalRecommendation` once per persistent failer.

Every record is serde-serializable, carries the schema version, a stable
SHA-256 digest, refs (plan digest, ban-event digests), and caller-supplied
timestamps. Volunteer churn is a priced, receipted event rather than an
incident.

## Monorepo Seam

The standby-promotion dispatcher lives monorepo-side (openagents). This crate
owns only the collective-layer `StandbyPromotionDecision` record the
dispatcher consumes: whether the stage would empty, which warm standby is
deterministically selected, and the typed reason. Lease management, actual
node promotion, swarm-membership enforcement, and ban/lease bookkeeping
against removal recommendations are monorepo-side responsibilities; the
`SwarmRemovalRecommendation` receipt is the hand-off artifact.

## Honest Current Meaning

These are simulation-level semantics. The round state machine is
deterministic and consumes caller-supplied chunk outcomes and timestamps; no
live WAN transport, real chunked tensor transfer, or wall-clock timeout
enforcement is exercised here. Actual collective execution still delegates to
`psionic-distributed`, and live WAN transport exercise is R2-gated.

## Landed Surface

- `crates/psionic-collectives/src/failure_semantics.rs`
  - `CollectiveRoundConfig`, `TensorChunkPlan`, `plan_tensor_chunks`
  - `CollectiveFailureRound`, `ChunkTransferOutcome`,
    `ChunkOutcomeDisposition`
  - `BanForRoundEvent`, `BanForRoundReason`
  - `IncludedContribution`, `ExcludedContribution`, `ChunkExclusionReason`
  - `StandbyPromotionDecision`, `StandbyPromotionReason`
  - `CollectiveRoundVerdict`, `CollectiveRoundAbortReason`,
    `CollectiveRoundReceipt`
  - `BanEscalationPolicy`, `CollectiveBanLedger`,
    `SwarmRemovalRecommendation`
  - `canonical_collective_failure_semantics_contract`,
    `write_collective_failure_semantics_contract`
- `crates/psionic-collectives/src/bin/collective_failure_semantics_contract.rs`
- `fixtures/training/collective_failure_semantics_v1.json`
- `scripts/check-collective-failure-semantics.sh`

## Validation

```bash
cargo test -p psionic-collectives
scripts/check-collective-failure-semantics.sh
```

Tests cover: the happy round; a send-path failure at chunk 3 of 8 with the
first two chunks preserved; a reciprocal-path timeout with identical ban
semantics; full-round timeout retaining the partial result; stage-empty abort
without a standby; standby promotion preventing the abort; round-timeout
abort when nothing was delivered and no standby exists; ban escalation to a
removal recommendation; window expiry of old bans; config-driven chunk plans;
and deterministic, serde round-trippable contract generation.

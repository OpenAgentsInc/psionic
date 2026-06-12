# Psion Shadow-Window Join Ramp

> Status: canonical `psionic#1125` record, 2026-06-12, after landing the first
> shadow-window join-ramp contract (Pluralis roadmap item P1.1).

Authored by Fable (claude-fable-5) for psionic#1125.

## What This Closes

A joining device does not enter a paid merged window on day one. Psionic now
owns a typed join ramp for the statistical training lane:

- Phase 1 (`shadow_replay`): the joiner binds to a durable checkpoint by
  digest (`checkpoint_object_digest` + `manifest_digest`) and replays a sealed
  window locally. Receipts are verified but never merged.
- Phase 2 (`live_shadow`): the joiner runs live windows whose outputs are
  verified and receipted but excluded from the merge — our `weight=0`
  analogue — warming scheduler trust.
- Then `active`: windows are merge-eligible and paid at the full class rate.

The contract surface lives in:

- `crates/psionic-train/src/shadow_window_ramp.rs`
- `crates/psionic-train/src/bin/shadow_window_ramp_contract.rs`
- `fixtures/training/shadow_window_ramp_v1.json`
- `scripts/check-shadow-window-ramp.sh`

## Contract Shape

The canonical contract (`psionic.train.shadow_window_ramp.v1`) freezes:

- `ShadowWindowRampConfig`: ramp lengths are operator config, never
  constants. The type has no `Default` and refuses zero-length phases.
  Pluralis ships 400 + 100 steps; that is their tuning, not ours.
- `ShadowJoinCheckpointBinding`: digest binding is required before phase 1.
  A joiner that has not verified the durable checkpoint cannot begin shadow
  replay.
- `ShadowJoinRampState`: the per-joiner phase machine. Phase transitions are
  typed errors when illegal — no phase skipping (`CannotSkipPhase`), no
  advance before the configured window count is satisfied
  (`PhaseCountNotSatisfied`), no progress past `active` (`AlreadyActive`),
  and wrong-phase or unverified receipts are refused.
- `ShadowWindowExecutionReceipt`: wraps the existing
  `psionic.train.window_execution.v1` and
  `psionic.train.contribution_receipt.v1` lane schemas with an explicit phase
  and merge-eligibility marker. Shadow-phase receipts cannot claim merge
  eligibility; active-phase receipts cannot claim shadow status.
- Merge-eligibility gating, structurally enforced:
  `MergeEligibleReceipt` is a proof token whose only constructor is
  `MergeEligibleReceipt::admit`, which refuses shadow and unverified
  receipts. `ShadowAwareMergeSetBuilder::push` accepts only that token, so a
  `merge_eligibility: shadow` receipt cannot become a merge-set member at the
  type level. This builder is the integration point for any later merge-set
  construction in the window lifecycle.
- `ShadowRampDivergenceComparison`: the measurement hook comparing a
  ramped-join cohort against a bootstrap-and-merge cohort, with per-window
  divergence samples and an explicit `measurement_status` of
  `synthetic_fixture` or `measured`.

Contract validation replays the committed receipts through the phase machine
and rebuilds the merge set through the type-level gate, so the recorded final
state and membership/exclusion lists cannot drift from what the code would
actually produce.

## Pluralis Source

Adapted from the Pluralis agora startup sequence
(`docs/agora-system/startup-sequence.md` in the read-only reference lane
`projects/pluralis/repos/agora`):

- Sync Phase 1 (weight synchronisation): the node receives averaged
  parameters with `weight=0`; trainers skip it; no batches are processed.
- Sync Phase 2 (optimizer warm-up): the node processes real batches but
  still contributes `weight=0` to averaging; its samples do not count toward
  the progress target.
- Active: full participation.

The Psion adaptation moves these semantics into the window vocabulary:
phase 1 becomes sealed-window replay against a digest-verified durable
checkpoint, phase 2 becomes live windows that are verified and receipted but
merge-excluded, and the `weight=0` rule becomes structural merge exclusion
rather than an averaging coefficient.

## OpenAgents Dispatcher Seam

Psionic owns the window mechanics: shadow execution, merge exclusion, and the
divergence-comparison contract. The openagents monorepo owns dispatcher
policy above this seam:

- openagents#4850 — joiners bootstrap from the last durable seal only
  (snapshot-lags-live rule). The `ShadowJoinCheckpointBinding` digests are
  the Psionic-side anchor for those bootstrap grants.
- openagents#4851 — dispatcher join-blocking window around merge/seal
  operations. The join barrier decides when a ramp may begin; this contract
  decides what the joiner must do once admitted.

Presence-tier payment for shadow work is monorepo policy (openagents roadmap
items P0.1/P2.3), not a Psionic claim.

## Divergence Comparison and Falsifier

The committed fixture demonstrates the comparison shape only: both cohorts
carry clearly-labeled synthetic divergence values
(`measurement_status: synthetic_fixture`), and the contract refuses any
synthetic comparison that does not say so in its detail and claim boundary.

The pre-registered falsifier from the roadmap stands: if R1 measurement shows
the phase-2 analogue does not reduce post-join divergence or dispute rates
versus bootstrap-and-merge, the ramp collapses to phase 1 only and the
negative result is recorded.

## Honest Current Meaning

- No live ramp has run. Every receipt, digest, and divergence value in the
  fixture is synthetic and exists to freeze the contract shape.
- Measuring the ramp length that actually reduces post-join divergence is the
  hardware-gated R1 deliverable and is not claimed here.
- These are statistical-lane contracts: `verified` records that the lane's
  verification policy accepted the output, not bit-exact replay.
- Ramp lengths in the fixture (3 + 2 windows) are fixture-scale walkthrough
  inputs, not recommended values.

## Validation

- `cargo test -p psionic-train` covers legal/illegal phase transitions,
  config-driven ramp lengths, the digest-binding requirement, type-level
  merge exclusion, and fixture round-trip against the committed artifact.
- `./scripts/check-shadow-window-ramp.sh` regenerates the contract and
  verifies the committed fixture has not drifted, shadow receipts never enter
  the merge set, and the synthetic measurement boundary stays explicit.

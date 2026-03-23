# PSION Plugin Route And Refusal Hardening

> Status: canonical `PSION_PLUGIN-31` route/refusal hardening contract for the
> bounded plugin-conditioned learned lanes, written 2026-03-22 after landing
> the first host-native and mixed single-node proofs.

This document freezes the follow-on hardening tranche for plugin-conditioned
route selection, refusal behavior, and no-implicit-execution discipline.

It sits above the individual plugin benchmark packages and above the host-native
and mixed learned-lane publications.

The job here is not to add a new benchmark family. The job is to retain one
machine-readable hardening bundle that:

- freezes the current route/refusal regression rows
- freezes an explicit overdelegation budget
- tests execution-implication failures explicitly
- cites committed route/refusal receipts instead of narrative-only confidence

## Canonical Artifacts

- `docs/PSION_PLUGIN_ROUTE_REFUSAL_HARDENING.md` is the canonical human-readable
  contract.
- `crates/psionic-train/src/psion_plugin_route_refusal_hardening.rs` owns the
  typed bundle, validation rules, and digest logic.
- `crates/psionic-train/examples/psion_plugin_route_refusal_hardening.rs`
  writes the canonical bundle.
- `fixtures/psion/plugins/hardening/psion_plugin_route_refusal_hardening_v1/`
  carries the committed bundle.

The hardening bundle reuses the earlier committed tranche artifacts directly:

- host-native reference run bundle
- host-native capability matrix and served posture
- mixed reference run bundle
- mixed capability matrix and served posture
- refusal/request-for-structure benchmark bundle
- result-interpretation benchmark bundle
- guest capability-boundary benchmark bundle
- the first host-native and mixed Google operator audits

## Regression Suite

The route/refusal regression suite freezes the current learned-lane scores for:

- discovery and selection
- argument construction
- refusal and request-for-structure
- result interpretation
- mixed-lane sequencing where the mixed lane newly closes the old zero-eligible
  gap

Each regression row keeps explicit:

- which lane the row belongs to
- which benchmark family it covers
- which prior reference label it compares against
- the eligible and out-of-scope counts
- the frozen current score
- a zero-bps regression budget

This means later plugin-conditioned widening does not get to quietly drop route
or refusal quality while still citing the old proof tranche.

The first committed bundle freezes:

- `9` regression rows
- `2` overdelegation budgets
- `5` explicit execution-implication cases

## Overdelegation Budget

The hardening bundle freezes a dedicated overdelegation budget from the shared
refusal/request-for-structure benchmark package.

The current tranche keeps:

- the exact overdelegation-negative item ids
- the reference overdelegation-rejection accuracy
- the observed overdelegation failure budget in basis points
- the maximum allowed overdelegation failure budget

The first frozen posture is intentionally strict:

- max allowed overdelegation failure budget: `0`

That keeps direct-answer negatives and unsupported-capability negatives from
drifting into silent delegation.

## Execution-Implication Cases

The hardening bundle also keeps explicit execution-implication negatives.

The first committed cases cover:

- host-native overdelegation answer-in-language without hidden execution
- host-native unsupported-capability refusal without hidden fallback execution
- host-native receipt-bound result interpretation
- mixed guest unsupported-digest-load refusal without hidden guest execution
- mixed receipt-bound result interpretation

Each case keeps:

- the source benchmark item
- the expected route when a route is part of the case
- the claim surfaces that must stay supported
- the claim surfaces that must stay blocked
- the refusal reasons that must stay explicit when relevant
- whether explicit runtime receipts are required
- whether unseen execution claims are forbidden
- the policy phrase that must remain present in the served posture

This keeps no-implicit-execution discipline tied to committed benchmark items
and committed served postures rather than operator memory.

## Claim Boundary

This tranche does not claim:

- generic plugin safety closure
- publication widening
- public plugin universality
- arbitrary software capability
- that route or refusal is solved generically outside the bounded learned lanes

It only freezes the current host-native and mixed plugin-conditioned claim
boundary so later operator or cluster decisions have one stable hardening bundle
to cite.

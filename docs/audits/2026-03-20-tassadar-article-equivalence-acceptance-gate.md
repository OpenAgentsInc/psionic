# TAS-158 Article-Equivalence Acceptance Gate Skeleton

`TAS-158` adds the single final machine-readable acceptance gate for the
post-`TAS-156A` article-equivalence closure wave.

This issue does not close article equivalence. It freezes the end-state gate in
red form so later issues can only turn it green by satisfying the already
declared rows.

## What Landed

- one committed eval artifact at
  `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`
- one checker script at
  `scripts/check-tassadar-article-equivalence-acceptance-gate.sh`
- one provider-facing receipt
  `TassadarArticleEquivalenceAcceptanceGateReceipt`

## Gate Shape

The acceptance gate now requires all of the following to be green:

- the blocker-matrix contract must stay structurally green
- the owned `psionic-transformer` route boundary must stay explicit
- the blocker matrix itself must turn `article_equivalence_green=true`
- every required article-gap issue row from `TAS-158` through `TAS-186`
  must be closed

The optional follow-on `TAS-R1` remains visible inside the gate but does not
block green.

## Current Status

The committed gate remains intentionally blocked.

Current bounded truth:

- `TAS-158` itself is now closed inside the blocker matrix
- the blocker-matrix contract remains green
- the owned Transformer boundary remains green
- the blocker rows themselves remain open
- the remaining required TAS issue tranches remain open
- `public_claim_allowed` remains `false` by design

This means the repo now has one frozen final acceptance surface, but not final
article-equivalent closure.

## Validation

- `cargo test -p psionic-eval article_equivalence_acceptance_gate -- --nocapture`
- `cargo test -p psionic-provider article_equivalence_acceptance_gate_receipt -- --nocapture`
- `./scripts/check-tassadar-article-equivalence-acceptance-gate.sh`

## Claim Boundary

This gate is necessary for article-equivalent closure, but it is not sufficient
by itself to widen public capability claims. It still does not imply arbitrary
C ingress, arbitrary Wasm ingress, or generic interpreter-in-weights closure
outside the declared article envelope.

## Audit Statement

`Psionic now has one canonical final article-equivalence acceptance gate that freezes the blocker-matrix contract, the owned Transformer boundary, the blocker-closure row, and every required TAS tranche from TAS-158 through TAS-186, while keeping the current verdict explicitly blocked until those rows actually close.`

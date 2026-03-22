# Psion Sampling Policy

> Status: canonical `PSION-10` / `#366` sampling-policy contract, written
> 2026-03-22 after landing the replay-safe tokenized corpus contract.

This document freezes the first train-time sampling and weighting artifact for
the `Psion` lane.

It makes mixture policy a first-class Psionic contract instead of a notebook or
launcher flag.

## Canonical Artifacts

- `crates/psionic-train/src/psion_sampling_policy.rs` owns the typed
  sampling-policy contract and comparison receipt.
- `fixtures/psion/sampling/psion_sampling_policy_baseline_v1.json` is the
  canonical baseline policy fixture.
- `fixtures/psion/sampling/psion_sampling_policy_manifest_v1.json` is the
  canonical candidate policy fixture.
- `fixtures/psion/sampling/psion_sampling_policy_comparison_receipt_v1.json`
  is the canonical comparison receipt fixture.

The stable schema versions are:

- `psion.sampling_policy.v1`
- `psion.sampling_policy_comparison.v1`

## What The Policy Freezes

The policy artifact now carries:

- stable dataset identity and packing-policy version
- source-family weights
- per-source contribution caps
- repetitive-region down-weighting controls
- token-share reporting for prose, spec text, and code
- an explicit maximum code-token ratio
- regression thresholds for explanation quality, spec interpretation, tradeoff
  reasoning, invariant articulation, and coding fluency

That keeps tokenized-corpus lineage separate from training-time mixture policy
while still binding the policy back to the same replay-safe dataset identity.

## Code-Dominance Controls

The contract does not assume code should dominate because the raw corpus gets
larger.

Instead it makes the following machine-legible:

- family-level token-share caps
- per-source contribution caps
- region-level down-weighting for repetitive sections
- a maximum code-token ratio that is checked against the observed token-share
  report

That keeps the lane aimed at technical explanation and systems reasoning rather
than drifting into a generic coding-assistant mixture.

## Regression Comparison

Mixture changes are not allowed to hide behind LM-loss movement alone.

The comparison receipt records:

- token-share deltas across prose, spec text, and code
- reasoning and coding score deltas
- the measured regression against baseline per tracked dimension

When coding fluency improves, the receipt validation rejects the change if
explanation quality, spec interpretation, tradeoff reasoning, or invariant
articulation regress.

## Mechanical Enforcement

`psionic-train` now validates that:

- family weights cover exactly the train-visible source families from the
  tokenized corpus
- per-source caps cover exactly the train-visible source ids
- repetitive-region controls resolve to real raw-source documents and section
  anchors
- the token-share report covers prose, spec text, and code exactly once
- observed code share stays below the declared ceiling
- comparison receipts recompute token-share deltas and regression deltas from
  the compared policies
- coding-fluency gains cannot mask reasoning regressions

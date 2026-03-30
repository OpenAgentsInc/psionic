# Psion Served Output Claims

> Status: canonical `PSION-28` / `#384` served-output claim-discipline
> contract, written 2026-03-22 after landing the first claim-posture schema on
> top of `PSION-27`.

This document freezes the first served-output claim-posture contract for the
`Psion` learned-model lane.

It sits one layer above `docs/PSION_SERVED_EVIDENCE.md`.

The evidence bundle says what artifacts support the output.

The claim-posture contract says what the served path is actually allowed to
claim, what assumptions it is making, what route or refusal behavior is being
shown, and what published context or latency envelope is in force.

## Canonical Artifacts

- `docs/PSION_SERVED_OUTPUT_CLAIMS.md` is the canonical human-readable
  contract.
- `crates/psionic-serve/src/psion_served_output_claim_posture.rs` owns the
  typed schema, validation rules, and digest logic.
- `crates/psionic-serve/examples/psion_served_output_claim_fixtures.rs` writes
  the canonical example postures.
- `fixtures/psion/serve/psion_served_output_claim_direct_v1.json`
  demonstrates a direct learned output with explicit assumptions and visible
  bounded claims.
- `fixtures/psion/serve/psion_served_output_claim_executor_backed_v1.json`
  demonstrates an exact-executor handoff with visible executor-backed claim
  posture but no verification widening.
- `fixtures/psion/serve/psion_served_output_claim_refusal_v1.json`
  demonstrates a typed refusal posture with zero unsupported claim flags.
- `docs/PSION_GENERIC_LOAD_AND_GENERATE.md` records the first retained generic
  learned-lane runtime packet that reuses this claim-posture contract on a real
  artifact-backed load-and-generate seam.
- `docs/PSION_FAMILY_SERVE_VOCABULARY.md` records the family-level language for
  how generic learned, plugin-conditioned, and executor-capable `Psion` lanes
  reuse or project this claim discipline without collapsing into one evidence
  class.

The stable schema version is `psion.served_output_claim_posture.v1`.

## Claim Posture Shape

Each claim posture carries:

- one capability-matrix id and version
- one evidence-bundle id and digest
- one visible-claims block
- one ordered assumption list
- one visible route-or-refusal block
- one context envelope surface
- one latency envelope surface
- one stable posture digest

Validation binds the posture back to both the capability matrix and the
attached evidence bundle.

## Visible Claims

The visible-claims block now makes five things explicit:

- whether the output is surfaced as learned judgment
- whether the output is surfaced as source-grounded
- whether the output is surfaced as executor-backed
- whether the output is surfaced as benchmark-backed
- whether the output implies verification

The first four may only be visible when the attached `PSION-27` evidence bundle
has the corresponding evidence labels.

The fifth is currently rejected outright for the learned lane. `Psion` does not
get to imply formal verification here.

## Assumptions

Served outputs now have one explicit assumption lane.

The first schema supports:

- `input_constraint`
- `interpretation_boundary`
- `missing_structured_input`
- `environment_boundary`

This keeps bounded answers from pretending to be context-free or world-state-
free when they are not.

## Visible Behavior

The claim posture now surfaces one visible behavior block:

- either a route with route kind plus route class
- or a typed refusal with capability region and refusal reason

This block must match the attached route or refusal evidence from
`PSION-27`. The served output cannot talk like one thing while the provenance
artifacts say another.

## Envelope Reflection

The claim posture now carries the published:

- context envelope
- latency envelope

alongside the observed prompt-token count for the served request.

That keeps the supported envelope visible at the point of service instead of
burying it only in the capability matrix doc.

## Reuse Contract

The claim posture now lives in:

- `psionic-serve::GenerationProvenance`
- OpenAI-compatible `psionic_claim_posture` response fields
- `psionic-provider::TextGenerationReceipt`

without a provider-only or HTTP-only wrapper.

This is the discipline bar for the served learned lane:

- no execution implication without executor-backed evidence
- no benchmark-proof implication without benchmark-backed evidence
- no source-grounded implication without cited source evidence
- no verification implication at all on the current learned lane
- explicit assumptions and explicit route or refusal behavior in the served
  output contract

`PSION-29` now adds the matching downgrade history in
`docs/PSION_CAPABILITY_WITHDRAWAL.md`. When rights, contamination, replay,
route, or refusal posture regress, the served claim surface is now required to
show an explicit depublish, narrowing, or behavior-change record instead of
quietly drifting while the old claim posture remains on paper.

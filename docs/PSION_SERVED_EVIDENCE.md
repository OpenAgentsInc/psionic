# Psion Served Evidence

> Status: canonical `PSION-27` / `#383` served-evidence and provenance
> contract, written 2026-03-22 after landing the first shared serve/provider
> schema for `Psion`.

This document freezes the first shared served-evidence bundle for the `Psion`
learned-model lane.

It is the schema that later serving and provider surfaces must reuse when they
need to say what kind of support a served output actually has.

The point is to keep four things separate:

- learned judgment
- source-grounded statement
- executor-backed result
- benchmark-backed capability claim

without implying any of them when the attached artifacts are absent.

## Canonical Artifacts

- `docs/PSION_SERVED_EVIDENCE.md` is the canonical human-readable contract.
- `crates/psionic-serve/src/psion_served_evidence.rs` owns the shared typed
  schema, validation rules, and digest logic.
- `crates/psionic-serve/examples/psion_served_evidence_fixtures.rs` writes the
  canonical example bundles.
- `fixtures/psion/serve/psion_served_evidence_direct_grounded_v1.json`
  demonstrates a direct learned answer with source grounding and
  benchmark-backed capability evidence.
- `fixtures/psion/serve/psion_served_evidence_executor_backed_v1.json`
  demonstrates an exact-executor handoff with explicit executor artifact
  binding.
- `fixtures/psion/serve/psion_served_evidence_refusal_v1.json` demonstrates a
  typed refusal with refusal-calibration binding and explicit no-implicit-
  execution posture.

The stable schema version is `psion.served_evidence.v1`.

## Bundle Shape

Each served-evidence bundle carries:

- one capability-matrix id and version
- either one route receipt or one refusal receipt
- one explicit no-implicit-execution posture
- zero or more typed evidence labels
- one stable bundle digest

The bundle validates against the published capability matrix so route and
refusal evidence cannot drift away from the current public capability surface.

## Route And Refusal Binding

The route side now records:

- coarse route kind
- fine route class
- capability region id
- route boundary ref
- route-calibration receipt id
- route-class evaluation receipt id and digest

The refusal side now records:

- capability region id
- typed refusal reason
- refusal boundary ref
- refusal-calibration receipt id and digest

This keeps per-output route and refusal provenance tied to the canonical
`PSION-25` and `PSION-26` artifacts rather than informal strings.

## No-Implicit-Execution

Every bundle now carries an explicit no-implicit-execution posture:

- execution claims are only allowed through an explicit published executor
  surface
- direct learned answers keep `executor_surface_invoked = false`
- executor-backed results must attach an explicit executor artifact ref
- refusals keep executor invocation false instead of hinting at hidden tools

That means language outputs cannot silently smuggle in execution claims.

## Evidence Labels

The shared schema now distinguishes four label classes directly:

- `learned_judgment`
- `source_grounded_statement`
- `executor_backed_result`
- `benchmark_backed_capability_claim`

Each label class has its own required fields.

Examples:

- source-grounded statements require cited source ids, digests, and stable
  anchors
- executor-backed results require executor surface identity plus an explicit
  executor artifact
- benchmark-backed claims require a capability region, promotion decision id,
  benchmark receipt id, and benchmark artifact id/digest

## Reuse Contract

The shared bundle now lives in `psionic-serve::GenerationProvenance` and
projects into `psionic-provider::TextGenerationReceipt` without a provider-only
wrapper.

That is the main point of this issue:

- serve and provider surfaces can carry the same schema directly
- later claim-discipline work can refine behavior without replacing the
  evidence bundle type
- the capability matrix, route receipts, refusal receipts, and served output
  evidence now have one common typed join point

`PSION-28` now adds the next layer on top of this at
`docs/PSION_SERVED_OUTPUT_CLAIMS.md`: explicit assumptions, visible route or
refusal behavior, surfaced claim flags, and published envelope limits.

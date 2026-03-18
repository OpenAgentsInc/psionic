# Psionic Parameter Golf Research Harness

> Status: canonical `PGOLF-402` / `#173` research-harness contract, updated
> 2026-03-18 after landing the committed report builder in
> `crates/psionic-research/src/parameter_golf_research_harness.rs`.

This document records the first controlled research harness for post-parity
Parameter Golf variants.

It exists to keep future architecture and compression experiments comparable to
the baseline lane instead of letting each candidate drift onto a new oracle or
new accounting story.

## What Landed

`psionic-research` now exposes:

- `ParameterGolfResearchHarnessReport`
- `ParameterGolfResearchComparisonSurface`
- `ParameterGolfResearchVariantReport`
- `build_parameter_golf_research_harness_report()`
- `write_parameter_golf_research_harness_report(...)`

The committed machine-readable report now lives at:

- `fixtures/parameter_golf/reports/parameter_golf_research_harness_report.json`

## Baseline Surface

The harness freezes one measured control:

- the Psionic local-reference baseline lane
- the current challenge review benchmark reference
- the fixed training and validation digests from the local reference fixture
- the same `parameter_golf.validation_bits_per_byte` submission metric
- the same counted-byte vocabulary from the landed non-record submission package

That means future research candidates cannot quietly swap out the oracle, the
metric, or the accounting story.

## Candidate Families

The first harness stages three research families explicitly:

- shared-depth or recurrent decoder candidates
- stronger parameter-tying candidates
- compression or quantization candidates

Each candidate now carries:

- changed surfaces
- oracle guardrails
- accounting guardrails
- an explicit boundary note

## Current Honest Boundary

This harness is a comparison contract, not a promotion claim.

Today it gives the repo:

- one measured baseline control
- one committed report binding future candidates to the same oracle
- one stable counted-byte surface that future variants must preserve

It does not yet give the repo:

- improved research results
- promoted new architectures
- any stronger challenge claim than the current non-record package posture

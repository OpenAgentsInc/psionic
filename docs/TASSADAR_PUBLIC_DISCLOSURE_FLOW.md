# Tassadar Public Disclosure Flow

## Purpose

This checklist exists to move Tassadar material from private or alpha-only
contexts into public `psionic` surfaces without leaking private framing,
unannounced capability posture, or over-broad claims.

## Review Triggers

Run the disclosure review before landing:

- new public roadmap entries
- new public issue bodies
- new public benchmark or report summaries
- new public docs that were informed by `~/code/alpha`
- wording changes that widen capability posture, route posture, trust posture,
  or product framing

## Required Decomposition Pass

For every candidate public change:

1. Replace private names or private shorthand with public repo-facing names.
2. Remove internal product framing, internal market framing, and unannounced
   roadmap nouns.
3. Reduce claims to the strongest honest public claim surface supported by the
   committed evidence.
4. Preserve dependency markers when the public repo does not own the full
   behavior.
5. Add explicit refusal language when a capability is still bounded, challenge
   gated, or blocked on a dependency surface.

## Checklist Fields

The machine-readable review artifact must record:

- review scope
- private source refs consulted
- public surface refs being changed
- whether private naming was removed
- whether private product framing was removed
- whether benchmark claims stayed bounded
- whether dependency markers stayed explicit
- whether private-only language was refused when necessary
- red-team findings, if any
- final status: `approved` or `refused`

## Approval Rules

`approved` is allowed only when all of the following are true:

- private naming is removed
- private product framing is removed
- benchmark and capability claims stay bounded
- dependency markers remain explicit
- red-team findings are empty

## Refusal Rules

Use `refused` when any of the following remain:

- private-only roadmap nouns
- internal product or market framing
- language that implies broader capability than the committed public evidence
- language that collapses dependency markers into implied local ownership

When `refused`, the artifact must include a non-empty
`blocked_publication_reason`.

## Checker

Validate the machine-readable review artifact with:

```bash
scripts/check-tassadar-public-disclosure.sh
```

Or point it at an explicit artifact:

```bash
scripts/check-tassadar-public-disclosure.sh path/to/review.json
```

## Current Canonical Artifact

The canonical committed review artifact for this workflow is:

- `fixtures/tassadar/reports/tassadar_public_disclosure_release_review.json`

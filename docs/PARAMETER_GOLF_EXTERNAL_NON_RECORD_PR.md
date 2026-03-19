# Psionic Parameter Golf External Non-Record PR

> Status: canonical `PGOLF-607` / `#251` historical external non-record PR
> record, updated 2026-03-19 after
> `https://github.com/openai/parameter-golf/pull/119` was closed on
> 2026-03-19.

This document records the first real external Parameter Golf submission loop
for Psionic's non-record lane as historical evidence only.

It exists so the exact folder, fork branch, PR URL, verifier evidence, and
the earlier counted-runtime question do not live only in GitHub comments or
terminal history.

It is not a live workflow. The PR is closed, and Psionic now treats outbound
PRs, issues, and maintainer-facing comments as forbidden unless the user
explicitly directs them.

## What Landed

Psionic opened one real upstream non-record PR, and that PR is now closed:

- upstream PR: `https://github.com/openai/parameter-golf/pull/119`
- PR state: `closed`
- closed at: `2026-03-19T19:46:39Z`
- upstream repo: `openai/parameter-golf`
- fork branch: `OpenAgentsInc:psionic-non-record-runtime-replay-v2`
- submitted folder:
  `records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2`

The machine-readable receipt now lives at:

- `fixtures/parameter_golf/reports/parameter_golf_external_non_record_pr.json`

## Preserved Evidence

The receipt freezes:

- the exact fork commit and record-folder tree hash used for the PR
- the canonical Psionic final-PR-bundle report digest
- the canonical Psionic local-clone dry-run report digest
- the staged-folder compatibility-verifier digest
- the staged-folder replay-verifier digest
- the PR URL, closed state, and close timestamp

That means later external submissions can start from one committed reference
instead of reconstructing the first submission path from memory.

## Current Accounting Posture

The first external submission also carried one counted-runtime question, but
that question is now only a historical artifact from a closed PR.

Preserved historical link:

- `https://github.com/openai/parameter-golf/pull/119#issuecomment-4092766684`

Current posture:

- no maintainer answer exists
- the PR that carried the question is closed
- Psionic no longer tracks a live maintainer-facing clarification lane inside
  this repo because outbound contributions now require explicit user direction
- counted-runtime and build-dependency posture therefore remain explicitly
  unresolved and locally blocked rather than “awaiting” a reply that Psionic is
  not currently allowed to pursue

## Current Honest Boundary

This preserves the first real external non-record PR loop as historical
evidence only.

It does not claim:

- an active upstream review path
- maintainer acceptance
- record-track readiness
- counted-runtime rule clarity
- real `8xH100` success

Any future external PR or maintainer-facing accounting question now requires
explicit user direction first. Until that happens, the repo should preserve the
current ambiguity and blocked posture locally instead of pretending there is a
live upstream path.

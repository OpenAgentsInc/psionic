# Psion Executor Artifact Naming Policy

> Status: canonical `PSION-0003` / `#702` phase-one executor artifact naming
> policy, updated 2026-03-30.

## Why This Doc Exists

Phase one should not spend critical-path time renaming live executor artifacts.

The executor lane already has real `tassadar-...` model ids, route ids,
reports, fixtures, and acceptance surfaces. This doc freezes the rule that
keeps those identifiers valid while still describing them honestly as
executor-capable `Psion` artifacts.

## Phase-One Naming Rule

For phase one:

- keep the current executor artifact ids valid
- keep the current executor route ids valid
- keep the current executor report ids and fixture families valid
- describe those artifacts as executor-capable `Psion` artifacts in roadmap and
  review prose
- do not force an artifact-id migration just to make the umbrella family name
  cleaner

## Current Live Families Covered By This Policy

The phase-one no-forced-migration rule covers current executor families such
as:

- `tassadar-...` model artifact ids
- `tassadar.article_route...` route ids
- `fixtures/tassadar/...` artifact and report families
- repo-local `Tassadar` issue and report vocabulary tied to the bounded
  executor lane

This keeps the repo honest about what is already live instead of pretending the
lane has been renamed before the lane itself is stable.

## What New Phase-One Work Should Do

- continue using the current executor naming families for executor-lane
  promotion, export, replacement, and report work
- make the umbrella relationship explicit in prose instead of renaming the ids
- keep generic compact-decoder `Psion` naming separate from executor-lane
  naming where needed

## What Phase One Must Not Do

- no forced rename of the current executor artifact family
- no forced route-id migration
- no issue-body or report wording that implies the existing `tassadar-...`
  ids are invalid before a later migration decision is made

## Later Migration Decision

Artifact-id migration is allowed only as later explicit work.

It is not part of phase-one critical-path closure.

Any future migration decision must:

- evaluate impact on live routes and retained reports
- keep route replacement and promotion truth intact
- land as a separate deliberate decision instead of as incidental cleanup

The current roadmap tracks that later decision under the secondary umbrella
integrity work in `PSION-0903`.

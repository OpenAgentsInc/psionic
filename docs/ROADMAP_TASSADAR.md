# Psionic Tassadar Roadmap Bridge

> Status: repo-local bridge refreshed 2026-03-19 after reviewing the live
> external Tassadar roadmap, `README.md`, `docs/ARCHITECTURE.md`,
> `docs/ROADMAP_TASSADAR_INDEX.md`, `docs/ROADMAP_TASSADAR_TAS_SYNC.md`,
> `docs/TASSADAR_WASM_RUNBOOK.md`, and the open `TAS-*` GitHub issue queue.

This file stays in the repo because a large set of GitHub issues and repo docs
already link to `docs/ROADMAP_TASSADAR.md`.

It is no longer the full living Tassadar roadmap.

## Canonical Live Roadmap

The active Tassadar roadmap now lives outside this repo at:

- `/Users/christopherdavid/code/alpha/tassadar/tassadar-llm-as-computer-roadmap.md`

Use that file for:

- tranche definitions
- issue sequencing after `TAS-102`
- terminal-contract language
- the current widening path from bounded named profiles to `TCM.v1`

This repo-local file remains a stable bridge so existing issue links keep
landing on something truthful instead of a stale copied roadmap.

## Current Repo-Local Summary

As of 2026-03-19, the live external roadmap and the current GitHub `tassadar`
issue queue are aligned.

Current honest posture:

- benchmarked, bounded internal computation under named profiles with explicit
  refusal surfaces
- the current served profile remains
  `tassadar.internal_compute.article_closeout.v1`
- broader profiles above that remain benchmarked, suppressed, refused, or
  unpromoted unless their explicit gates are green

Still unclaimed:

- arbitrary Wasm execution
- full frozen core-Wasm closure
- broad practical internal computation as a generally served capability
- Turing-complete support

Current dependency spine:

1. `TAS-103` to `TAS-106`: real program execution
2. `TAS-107` to `TAS-110`: resumable and effect-safe execution
3. `TAS-111` to `TAS-112`: portable and publishable execution
4. `TAS-113` to `TAS-114`: frozen core-Wasm closure
5. `TAS-115` to `TAS-124`: numeric and proposal-family widening
6. `TAS-125` to `TAS-136`: process and internal-platform closure
7. `TAS-137` to `TAS-140`: learned and hybrid broad-compute comparison
8. `TAS-141` to `TAS-150`: public-claim, economic, and governance closure
9. `TAS-151` to `TAS-156`: universal-substrate and Turing-completeness
   closeout

Issue-state note:

- `TAS-103` through `TAS-109` are already implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- the current open GitHub backlog begins at `TAS-110` and runs through
  `TAS-156`

## Repo-Local Companion Docs

Use the repo-local docs for the questions they actually answer:

- `README.md` for the current Tassadar inventory and claim surface inside this
  repo
- `docs/ARCHITECTURE.md` for canonical Psionic-wide status vocabulary and
  machine-facing contract boundaries
- `docs/ROADMAP_TASSADAR_INDEX.md` for landed phase-to-artifact mapping
- `docs/ROADMAP_TASSADAR_TAS_SYNC.md` for issue-to-implementation closure
  notes
- `docs/TASSADAR_WASM_RUNBOOK.md` for the current bounded Wasm operator path
- `fixtures/tassadar/reports/tassadar_acceptance_report.json` and
  `scripts/check-tassadar-acceptance.sh` for the current machine-readable claim
  gate

If you only have the public `psionic` checkout and not the external Tassadar
workspace, use this bridge plus the current `TAS-*` issue bodies, the TAS sync
doc, and the artifact index together instead of relying on the old copied
roadmap text.

## Maintenance Rule

- update the external roadmap when tranche order, issue sequencing, or
  terminal-contract language changes
- update this repo-local bridge when the pointer, current truthful posture, or
  repo-visible issue alignment changes
- do not reintroduce a copied issue-by-issue backlog here unless the canonical
  roadmap moves back into this repo

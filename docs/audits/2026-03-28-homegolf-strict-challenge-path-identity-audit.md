# HOMEGOLF Strict Challenge Path Identity Audit

Date: 2026-03-28

## Scope

This audit records the next integration fix after
`docs/audits/2026-03-28-homegolf-local-wallclock-projection-refusal-audit.md`.

The strict HOMEGOLF preflight was still too weak.

It claimed `ready_to_execute` for any existing directory named
`fineweb10B_sp1024` and any existing file named `fineweb_1024_bpe.model`,
even when those paths were just temp placeholders and not the canonical
challenge inputs under `~/code/parameter-golf/...`.

That was an honest bug in the HOMEGOLF integration surface.

## What Changed

`crates/psionic-train/src/parameter_golf_homegolf_strict_challenge.rs` now
expands the retained `~/code/parameter-golf/...` shell paths into concrete
expected paths and compares supplied inputs against those exact paths.

The strict challenge lane now:

- refuses temp or relocated lookalike paths with the right basename
- keeps `wrong_path_identity` instead of `present_exact_named_path` for those
  fake inputs
- only emits `ready_to_execute` when the actual expected challenge dataset
  root and tokenizer path are present

The same file now also keeps a regression test that covers the fake-basename
false positive directly.

## Reproduced Bug Before The Fix

One local reproduction used a temp directory and temp file only:

- dataset path:
  `/tmp/.../fineweb10B_sp1024`
- tokenizer path:
  `/tmp/.../fineweb_1024_bpe.model`

Before this fix, that produced:

- `disposition=ReadyToExecute`
- `dataset_root_status=present_exact_named_path`
- `tokenizer_path_status=present_exact_named_path`

That output was wrong because those inputs were not the canonical HOMEGOLF
challenge inputs.

## Post-Fix Validation

The same fake-path reproduction now returns:

- `disposition=RefusedMissingChallengeInputs`
- `dataset_root_status=wrong_path_identity`
- `tokenizer_path_status=wrong_path_identity`
- refusal subject:
  `parameter_golf_homegolf_strict_challenge_inputs`

Retained CLI output after the fix:

- `wrote .../report.json disposition=RefusedMissingChallengeInputs profile_id=parameter_golf_challenge_sp1024_v0 contest_bpb_required=true artifact_cap_required=true`
- `refusal subject=Some("parameter_golf_homegolf_strict_challenge_inputs") detail=dataset_root must be supplied as the exact FineWeb SP1024 lane '~/code/parameter-golf/data/datasets/fineweb10B_sp1024'; tokenizer_path must be supplied as the exact SP1024 tokenizer '~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model'; local-reference fallback is denied for the strict HOMEGOLF lane`

Code validation:

- `rustfmt crates/psionic-train/src/parameter_golf_homegolf_strict_challenge.rs`
- one post-fix local rerun of
  `parameter_golf_homegolf_strict_challenge_lane` against fake temp inputs

## Why This Matters

The strict challenge lane is the canonical HOMEGOLF contest-style preflight
surface.

If that preflight can be satisfied by placeholder temp paths, later operators
get a false green light before they ever reach the real dataset or tokenizer.

After this fix:

- the strict lane no longer overstates challenge readiness
- the refusal happens at preflight time instead of later during execution
- the retained HOMEGOLF docs can honestly say that exact path identity is part
  of the strict contract, not just basename matching

## Honest Boundary After This Audit

What is true:

- the strict HOMEGOLF preflight is now stricter and more honest
- the local exact HOMEGOLF lane still fast-refuses impossible local wallclock
  postures
- the retained actual full-validation PGOLF score is still
  `6.306931747817168`

What is not true:

- this did not produce a new PGOLF score
- this did not make the local `RTX 4080` strict score path viable
- this did not create a live dense mixed-device HOMEGOLF score operator

## Improvement Over The Previous Audit

Compared with the wallclock-refusal audit, this change improves a different
layer of the HOMEGOLF stack:

- the trainer was already refusing impossible local score attempts honestly
- now the strict preflight also refuses fake challenge-input identity instead
  of calling it ready

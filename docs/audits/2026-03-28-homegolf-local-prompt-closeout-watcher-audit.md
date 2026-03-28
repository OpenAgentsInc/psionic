# HOMEGOLF Local Prompt Closeout Watcher Audit

Date: 2026-03-28

## Scope

This audit records the next operator improvement after the scratch-first local
CUDA runner.

The reachable `archlinux` HOMEGOLF lane already had:

- one repo-owned clean-main launch wrapper
- real local artifact persistence before long score closeout completed
- one separate artifact-only prompt utility

But those pieces were still disconnected operationally.

The operator still had to come back manually after the long score run finished
and re-bind the exported artifact to a prompt proof.

## What Changed

Added:

- `scripts/wait-parameter-golf-homegolf-prompt-closeout.sh`

This watcher:

- waits for one HOMEGOLF training report path to appear and parse cleanly
- runs `parameter_golf_homegolf_prompt` from that same retained report
- writes a prompt report JSON
- writes one combined closeout summary JSON containing:
  - run id
  - machine profile
  - executed steps
  - artifact path
  - compressed bytes
  - final roundtrip BPB if present
  - generated text

## Why This Matters

The user asked for the system to produce actual text and to keep the proof tied
to the real submission-style artifact family.

This change moves that from an operator memory step into one repo-owned
closeout path.

It does not improve model quality directly.

It improves the integrity of the HOMEGOLF iteration loop by ensuring that the
same retained artifact can be scored and prompted without a second hand-built
shell sequence.

## Honest Boundary After This Audit

What is true:

- the reachable local HOMEGOLF lane now has:
  - one clean-main scratch-first run wrapper
  - one post-closeout prompt watcher
- once a long local score run lands its report, the text-generation proof can
  now be emitted automatically from the same retained artifact family

What is not true:

- this did not itself finish the currently running score loop
- this did not itself improve the retained PGOLF score
- this did not unblock the live H100 provider path

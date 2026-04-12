# Psion Local-First Reference Runbook

Status: canonical bounded smoke/reference runbook for the local-first Psion
reference-pilot lane, written 2026-03-30 and updated 2026-04-02 after the
actual broader-pretraining lane became the default meaning of `./TRAIN`.

## What This Runbook Is For

This runbook exists so the older bounded reference-pilot lane remains usable as
an explicit smoke/reference path without pretending to be the main operator
path.

The command is:

```bash
./TRAIN --lane reference_pilot
```

From the Psionic repo root, that now means:

- prefer the canonical accelerator-backed bounded reference pilot
- stage the current committed git revision to the admitted Tailnet CUDA host
- run `psion_accelerated_reference_pilot` there
- copy the retained artifacts back locally
- write one local operator manifest and one local operator summary

`./TRAIN` without `--lane reference_pilot` now means the actual broader-
pretraining lane. The actual lane, recipe, and evidence family are frozen
separately in:

- `docs/PSION_ACTUAL_PRETRAINING_LANE.md`
- `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SCALING_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_DATA_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SYSTEMS_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`
- `docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md`
- `docs/PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF.md`

It does **not** mean:

- HOMEGOLF
- Parameter Golf
- first swarm open-adapter training
- arbitrary lane auto-selection across the whole repo

## Canonical Lane

The lane this runbook prioritizes is the public-safe bounded Psion training
target already named in the Psion docs:

- trainer example:
  `crates/psionic-train/examples/psion_accelerated_reference_pilot.rs`
- lane id:
  `psion_accelerated_reference_pilot`

The bounded CPU reference lane still exists:

- trainer example:
  `crates/psionic-train/examples/psion_reference_pilot.rs`

But that is now the explicit fallback or smoke path, not the default meaning of
`./TRAIN`.

## Default Operator Posture

The default run is:

- local Mac as control plane
- `archlinux` as the admitted Tailnet CUDA training host
- live Tailnet status as the source of truth for the remote host's current IPv4
  instead of relying on one static SSH alias
- staged committed git revision, not dirty working-copy state
- copied-back retained artifacts under a local run root

## First Command

From the Psionic repo root:

```bash
./TRAIN --lane reference_pilot
```

The launcher now resolves logical hosts like `archlinux` through live
`tailscale status` output before opening SSH. You can still override that
explicitly with `--remote-host christopherdavid@<tailnet-ip>` when you want to
pin one exact target.

Reference-lane result:

- mode: `accelerated_reference`
- remote host: `archlinux`
- local run root:
  `~/scratch/psion_reference_pilot_runs/<run_id>`

The accelerated claim is still narrow:

- control plane: the local host that launched `./TRAIN --lane reference_pilot`
- worker count: `1`
- worker host: the admitted remote CUDA host
- execution classification:
  `local_control_plane_single_remote_worker`

That means a successful accelerated run is a real remote single-worker CUDA
pilot, not mixed-device Mac + CUDA training and not a broader cluster proof.

## Useful Options

### Dry run

```bash
./TRAIN --lane reference_pilot --dry-run
```

This writes the operator manifest and prints the selected plan without launching
training.

### Explicit accelerated run

```bash
./TRAIN --lane reference_pilot --mode accelerated_reference
```

### Explicit bounded local fallback

```bash
./TRAIN --lane reference_pilot --mode local_reference
```

### Auto mode with explicit fallback

```bash
./TRAIN --lane reference_pilot --allow-local-reference-fallback
```

This still prefers the accelerated lane. It only falls back to the CPU
reference lane when the remote accelerated lane is unavailable.

## Output Layout

Every bounded reference-pilot run writes:

- `reference_pilot_operator_manifest.json`
- `reference_pilot_operator_summary.json`
- `reference_pilot_train.log`

`reference_pilot_operator_summary.json` is the canonical operator-side topology
and cost surface for the bounded reference-pilot lane. It now records at
least:

- `control_plane_host`
- `worker_host`
- `worker_count`
- `execution_location`
- `execution_topology_classification`
- `delivered_backend`
- `total_cost_microusd`
- `truth_surface_kind`
- `actual_lane_relation`

Every completed bounded reference-pilot run also writes:

- `reference_pilot_artifacts/`

For accelerated runs, `reference_pilot_artifacts/` should contain:

- `psion_accelerated_reference_pilot_stage_receipt.json`
- `psion_accelerated_reference_pilot_observability_receipt.json`
- `psion_accelerated_reference_pilot_checkpoint_manifest.json`
- the related checkpoint and visualization artifacts emitted by the example

For local reference runs, `reference_pilot_artifacts/` should contain:

- `psion_reference_pilot_stage_receipt.json`
- `psion_reference_pilot_observability_receipt.json`
- `psion_reference_pilot_checkpoint_manifest.json`
- the related checkpoint artifacts emitted by the example

## Checkpoint Restore Verification

After a bounded local reference run completes, verify that the retained
checkpoint can be restored through the live resume-probe surface:

```bash
cargo run -q -p psionic-train --example psion_reference_pilot_resume_probe -- \
  ~/scratch/psion_reference_pilot_runs/<run_id>/reference_pilot_artifacts \
  /tmp/psion_reference_pilot_resume_probe_<run_id>
```

That command writes:

- `/tmp/psion_reference_pilot_resume_probe_<run_id>/psion_reference_pilot_resume_probe.json`

The resume-probe receipt is the retained proof that the saved checkpoint can be
reloaded and advanced through one resumed optimizer step without inventing a
second runtime path.

## Refusal Behavior

`./TRAIN --lane reference_pilot` now refuses explicitly when:

- the remote Tailnet host is unreachable
- `cargo` or `nvidia-smi` is missing on the remote host
- the remote GPU name cannot be resolved
- the remote GPU already has resident compute processes

That is deliberate. The command should fail loudly instead of pretending that a
different lane counted as the same thing, and it should not read as though the
bounded reference pilot were the actual broader-pretraining lane.

If staging or launch fails, `train.log` is also the canonical failure trace. It
records the selected staging strategy and the last completed launcher step so a
wrapper failure can be retained honestly instead of reconstructed from memory.

## Staging Behavior

In accelerated mode, `./TRAIN --lane reference_pilot` now prefers the fastest
honest staging path:

- use a remote detached git worktree when the admitted remote seed clone
  already contains the requested committed ref
- fall back to a copied tar archive when the committed ref exists only on the
  local machine

Both paths preserve the same claim boundary. The first is faster. The second is
the escape hatch for a committed local ref that has not been published yet.

## Claim Boundary

`./TRAIN --lane reference_pilot` proves one of two things:

### Accelerated mode

One real operator invocation of the bounded accelerator-backed Psion reference
lane on the admitted Tailnet CUDA host.

### Local reference mode

One real operator invocation of the bounded CPU reference Psion lane on the
local host.

It does not prove:

- broader cluster closure
- mixed-backend dense training
- plugin-conditioned Psion closure
- HOMEGOLF or Parameter Golf progress
- public promotion or serve readiness beyond the receipts the lane already owns

## Why This Exists

Before this runbook and entrypoint landed, the repo had multiple real training
surfaces but no single Psion-first operator command. The actual-lane closure
changed that. This runbook now preserves the older pilot as a clearly labeled
bounded reference surface.

This runbook keeps the older public-safe reference lane available without
letting it stay cognitively equal to the actual lane.

## Actual-Lane Default

The actual broader-pretraining lane is now the default operator path:

```bash
./TRAIN --dry-run
```

Resume and status use the same lane selector:

```bash
./TRAIN resume --run-root <path>
./TRAIN rehearse-base-lane --run-id <id>
./TRAIN status --run-root <path>
```

Those commands are the primary operator path. The bounded local-first
reference runbook remains the smoke/reference escape hatch. The default path
materializes the actual-lane retained evidence family and enforces the
dirty-tree/ref provenance contract described in
`docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md`.

# Psion Local-First Train Runbook

Status: canonical local-first operator runbook for the bounded Psion
reference-pilot lane, written 2026-03-30 and updated 2026-04-02 after
separating bounded reference-pilot truth from the actual broader-pretraining
lane.

## What This Runbook Is For

This runbook exists so one operator command can launch the **current bounded
reference-pilot Psion lane** without guessing between unrelated training
systems.

The command is:

```bash
./TRAIN
```

From the Psionic repo root, that now means:

- prefer the canonical accelerator-backed bounded reference pilot
- stage the current committed git revision to the admitted Tailnet CUDA host
- run `psion_accelerated_reference_pilot` there
- copy the retained artifacts back locally
- write one local operator manifest and one local operator summary

It does not launch `psion_actual_pretraining_v1`. The actual broader-
pretraining lane, recipe, and evidence family are frozen separately in:

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
`TRAIN`.

## Default Operator Posture

The default run is:

- local Mac as control plane
- `archlinux` as the admitted Tailnet CUDA training host
- staged committed git revision, not dirty working-copy state
- copied-back retained artifacts under a local run root

## First Command

From the Psionic repo root:

```bash
./TRAIN
```

Default result:

- mode: `accelerated_reference`
- remote host: `archlinux`
- local run root:
  `~/scratch/psion_reference_pilot_runs/<run_id>`

The accelerated claim is still narrow:

- control plane: the local host that launched `./TRAIN`
- worker count: `1`
- worker host: the admitted remote CUDA host
- execution classification:
  `local_control_plane_single_remote_worker`

That means a successful accelerated run is a real remote single-worker CUDA
pilot, not mixed-device Mac + CUDA training and not a broader cluster proof.

## Useful Options

### Dry run

```bash
./TRAIN --dry-run
```

This writes the operator manifest and prints the selected plan without launching
training.

### Explicit accelerated run

```bash
./TRAIN --mode accelerated_reference
```

### Explicit bounded local fallback

```bash
./TRAIN --mode local_reference
```

### Auto mode with explicit fallback

```bash
./TRAIN --allow-local-reference-fallback
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

## Refusal Behavior

`./TRAIN` now refuses explicitly when:

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

In accelerated mode, `./TRAIN` now prefers the fastest honest staging path:

- use a remote detached git worktree when the admitted remote seed clone
  already contains the requested committed ref
- fall back to a copied tar archive when the committed ref exists only on the
  local machine

Both paths preserve the same claim boundary. The first is faster. The second is
the escape hatch for a committed local ref that has not been published yet.

## Claim Boundary

`./TRAIN` proves one of two things:

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
surfaces but no single Psion-first operator command. That made `TRAIN`
ambiguous.

This runbook closes that ambiguity by binding `TRAIN` to the public Psion lane
that matters most for user-facing Psion progress.

## Actual-Lane Escape Hatch

The actual broader-pretraining lane now has its own explicit operator path:

```bash
./TRAIN --lane actual_pretraining start --dry-run
```

Resume and status use the same lane selector:

```bash
./TRAIN --lane actual_pretraining resume --run-root <path>
./TRAIN --lane actual_pretraining rehearse-base-lane --run-id <id>
./TRAIN --lane actual_pretraining status --run-root <path>
```

Those commands do not replace the bounded local-first reference runbook. They
materialize the actual-lane retained evidence family and enforce the
dirty-tree/ref provenance contract described in
`docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md`.

# Psion Actual Pretraining Runbook

Status: canonical operator runbook for the explicit actual `Psion`
pretraining lane, written 2026-04-02 when the repo landed the first real
start, dry-run, resume, and status command for `psion_actual_pretraining_v1`.

## What This Runbook Is For

This runbook exists so the repo has one explicit operator path for the actual
broader-pretraining lane without changing the default meaning of `./TRAIN`.

The commands are:

```bash
./TRAIN --lane actual_pretraining start [options]
./TRAIN --lane actual_pretraining resume --run-root <path> [options]
./TRAIN --lane actual_pretraining status --run-root <path>
```

`./TRAIN` without `--lane actual_pretraining` still means the bounded
reference-pilot lane.

## Current Claim Boundary

The actual-lane command now does these things for real:

- loads the frozen actual-lane spec, recipe bundle, scaling bundle, data
  bundle, baseline-tools bundle, systems bundle, topology/storage bundle,
  evidence contract, and status surface contract directly from committed repo
  artifacts
- refuses dirty launches by default unless `--allow-dirty-tree` is supplied
- resolves and retains the selected git ref plus exact commit SHA
- writes the canonical launch or resume manifest under the retained evidence
  family
- writes the canonical current-status and retained-summary files
- writes the canonical latest-checkpoint pointer file
- repeats provenance into the provisional closeout bundle
- exposes the canonical status command

It does not yet claim:

- hardware admission gating
- durable checkpoint backup
- automatic checkpoint evals
- dashboards or alert routing
- completed distributed cluster execution

Those come later in the roadmap. This launcher is the operator contract, not
the full hardening pass.

## Start Command

Dry-run materialization:

```bash
./TRAIN --lane actual_pretraining start --dry-run
```

Start with an explicit run id and output root:

```bash
./TRAIN --lane actual_pretraining start \
  --run-id run-psion-actual-20260402t120000z \
  --output-root ~/scratch/psion_actual_pretraining_runs/run-psion-actual-20260402t120000z
```

Default output root:

- `~/scratch/psion_actual_pretraining_runs/<run_id>`

The start path writes:

- `manifests/launch_manifest.json`
- `status/current_run_status.json`
- `status/retained_summary.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `closeout/closeout_bundle.json`
- `logs/launcher.log`

Before the first accepted checkpoint exists, the retained state is explicit:

- phase: `dry_run_planned` or `launch_staged`
- latest checkpoint label: `pending_first_checkpoint`
- last completed step: `0`

## Resume Command

Canonical resume:

```bash
./TRAIN --lane actual_pretraining resume --run-root <path>
```

Resume reads exactly:

- `<run-root>/checkpoints/latest_accepted_checkpoint_pointer.json`

Resume refuses when that pointer is missing or still in
`pending_first_checkpoint` state. When it succeeds, it writes:

- `manifests/resume_manifest.json`
- refreshed status and retained-summary files
- `continuation/accepted_checkpoint_handoff.json`
- refreshed provisional closeout bundle
- appended `logs/launcher.log`

The continuation handoff binds the accepted checkpoint to the frozen
`general_sft -> agentic_sft` target and preserves the bounded plugin
benchmark-pack bindings already attached to that target. It does not claim that
the continuation stage has already run.

## Status Command

```bash
./TRAIN --lane actual_pretraining status --run-root <path>
```

This is a thin wrapper over `scripts/psion-actual-pretraining-status.sh`. It
prints:

- run id
- phase
- last completed step
- latest checkpoint label
- selected git ref
- git commit SHA
- dirty-tree admission posture
- status surface id

## Dirty Trees And Provenance

Dirty working trees are refused by default.

If an operator deliberately overrides that rule:

```bash
./TRAIN --lane actual_pretraining start --allow-dirty-tree --dry-run
```

the launcher records:

- `dirty_tree_admission = allowed_by_operator_override`
- `workspace_status_sha256`

It still retains:

- `selected_git_ref`
- `git_commit_sha`

Those fields appear in the launch or resume manifest and repeat in the
provisional closeout bundle.

## Related Docs

- `docs/PSION_ACTUAL_PRETRAINING_LANE.md`
- `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SCALING_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_BASELINE_TOOLS_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_DATA_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`
- `docs/PSION_ACTUAL_PRETRAINING_STATUS_SURFACE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SYSTEMS_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF.md`

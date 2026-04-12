# 2026-04-12 M5 Local Reference Training Run Log

This log records one single-device Psion training run executed on the local
Apple M5 Max host by following the canonical bounded local-first runbook:

- `docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md`

## Goal

Run one real M5-only training job, keep the run inside the existing
`reference_pilot` path, fix anything that breaks, and record the exact commands
and retained outputs.

## Host

- machine: `MacBook Pro`
- chip: `Apple M5 Max`
- memory: `128 GB`
- GPU cores: `40`

## Commands Run

### 1. Confirm clean repo state

```bash
git status --short --branch
```

Result:

- `## main...origin/main`

### 2. Read the canonical local runbook

```bash
sed -n '1,260p' docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md
```

### 3. Run a bounded single-device training job on this M5

```bash
./TRAIN --lane reference_pilot --mode local_reference --run-id m5-local-reference-20260412t114300z
```

Result:

- `status=completed`
- `mode=local_reference`
- run root:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/m5-local-reference-20260412t114300z`

## Retained Outputs

Run root:

- `/Users/christopherdavid/scratch/psion_reference_pilot_runs/m5-local-reference-20260412t114300z`

Key files:

- operator manifest:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/m5-local-reference-20260412t114300z/reference_pilot_operator_manifest.json`
- operator summary:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/m5-local-reference-20260412t114300z/reference_pilot_operator_summary.json`
- train log:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/m5-local-reference-20260412t114300z/reference_pilot_train.log`
- artifact directory:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/m5-local-reference-20260412t114300z/reference_pilot_artifacts`

Artifact files present:

- `psion_reference_pilot_checkpoint.safetensors`
- `psion_reference_pilot_checkpoint_manifest.json`
- `psion_reference_pilot_observability_receipt.json`
- `psion_reference_pilot_optimizer_state.json`
- `psion_reference_pilot_stage_config.json`
- `psion_reference_pilot_stage_receipt.json`
- `psion_reference_pilot_summary.json`

## Observed Training Outcome

From the retained operator summary and train log:

- selected mode: `local_reference`
- execution location: `local`
- worker host: `Christophers-MacBook-Pro-2`
- total cost: `4000` microusd
- checkpoint ref: `psion-reference-pilot-step-16`
- checkpoint parameter-state digest:
  `0138e9580636d7b0b294a6a7eaf2d12401e0989a14d499dd5800edac7a476812`
- stage receipt digest:
  `051d82e02fe00fcfe0cc752ab7c0a62b8df7bdbda5bbeacde86126429b635afb`
- observability digest:
  `131eb981fabc3786e5b2b755b501d5ca8497efd193ca3b4640dea1bb857cd600`

The launcher ended with:

```text
psion reference pilot completed: stage=psion-reference-pretrain-stage checkpoint=psion-reference-pilot-step-16 output=/Users/christopherdavid/scratch/psion_reference_pilot_runs/m5-local-reference-20260412t114300z/reference_pilot_artifacts
held-out loss milli: initial=5380 final=5390
```

This is a real bounded training run. It is not a dry run and not an
actual-pretraining claim.

## Checkpoint Restore Verification

The older `psion_reference_pilot_generate` example referenced in a March audit
is not a live example target in the current repo. I did not treat that as a
runbook regression because it is not part of the canonical operator workflow.

I used the live retained restore path that does exist:

```bash
cargo run -q -p psionic-train --example psion_reference_pilot_resume_probe -- \
  /Users/christopherdavid/scratch/psion_reference_pilot_runs/m5-local-reference-20260412t114300z/reference_pilot_artifacts \
  /tmp/m5-local-reference-resume-probe-20260412t114300z
```

Result:

- `psion reference pilot resume probe completed`
- output:
  `/tmp/m5-local-reference-resume-probe-20260412t114300z/psion_reference_pilot_resume_probe.json`

Important retained fields from that probe:

- recovery mode: `resume_from_last_stable_checkpoint`
- checkpoint ref: `psion-reference-pilot-step-16`
- checkpoint lineage digest:
  `df239381fd46a8bbb0529f4eeb7124a6b0cc9cc1b6e64ca35b1851ba38acbe00`
- resumed step receipt id:
  `psion-reference-pilot-run-resume-step-1`
- resumed step loss: `5.327144`
- probe digest:
  `7e323409c3471604223d5c7be81a8e1bfed3a32fd601e44a674ebdb4f90ba55e`

That confirms the retained checkpoint can be restored and advanced through one
resumed optimizer step on this same M5 host.

## Breakages Encountered

No code changes were required to complete this single-device M5 reference run.

The only dead end during validation was trying to call a historical example
target that no longer exists:

```bash
cargo run -q -p psionic-train --example psion_reference_pilot_generate -- ...
```

That command failed because `psion_reference_pilot_generate` is not a current
example target in `psionic-train`. The current bounded verification path is the
live `psion_reference_pilot_resume_probe` example, and this log plus the
runbook now point to that path explicitly.

## Final Status

- local M5 training run: `completed`
- retained checkpoint written: `yes`
- retained checkpoint restore probe: `completed`
- code fix required for this run: `no`

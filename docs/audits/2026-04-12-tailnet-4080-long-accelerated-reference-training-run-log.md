# 2026-04-12 Tailnet 4080 Long Accelerated Reference Training Run Log

This log records the longer retained `reference_pilot` run executed from the
local M5 Mac against the admitted Tailnet RTX 4080 host `archlinux`.

## Goal

Run a materially larger accelerated reference-pilot job than the earlier
4-step multidevice smoke run while staying on the same shipped reference lane
and the same operator surface:

- local M5 Mac as control plane
- `archlinux` over Tailnet as the single remote CUDA worker
- copied-back retained artifacts stored under the local run root
- bounded budget overrides passed through `./TRAIN --lane reference_pilot`

## Remote preflight

Confirmed Tailnet reachability and remote GPU visibility:

```bash
tailscale status
ssh christopherdavid@archlinux \
  'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader'
```

Confirmed no admitted CUDA compute process was still occupying the 4080 before
launch:

```bash
ssh christopherdavid@archlinux \
  'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true'
```

As a defensive measure, any remaining compute-app PIDs were then force-closed
before launch.

## Long accelerated run

Command:

```bash
./TRAIN --lane reference_pilot --mode accelerated_reference --cleanup-remote \
  --run-id tailnet-4080-reference-long-20260412t173800Z \
  --max-steps 24 \
  --steps-per-window 6 \
  --windows-per-cadence 2
```

Result:

- status: `completed`
- selected mode: `accelerated_reference`
- truth surface kind: `bounded_reference_pilot`
- actual lane relation: `not_actual_pretraining_lane`
- control plane host: `Christophers-MacBook-Pro-2`
- worker host: `archlinux`
- worker Tailnet IP: `100.108.56.85`
- delivered backend: `cuda`
- remote GPU: `NVIDIA GeForce RTX 4080`
- remote stage strategy: `remote_git_worktree`
- local run root:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-reference-long-20260412t173800Z`
- local artifact dir:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-reference-long-20260412t173800Z/reference_pilot_artifacts`
- checkpoint ref: `psion-reference-pilot-step-24`
- checkpoint parameter digest:
  `96c12a66723fe678ed6c7312fb2ab9fb9d66780d1154035936914cd5108184eb`
- operator-summary digest:
  `043ffafba9702d6261a9a54906e03c1a3319bc09c6ea99b54e8cc4580a0290b3`
- estimated total cost: `14800` microusd

The retained operator summary recorded the requested budget override as:

- `max_steps=24`
- `steps_per_window=6`
- `windows_per_cadence=2`
- `step_duration_ms=null`

## Observability results

From
`reference_pilot_artifacts/psion_accelerated_reference_pilot_observability_receipt.json`:

- train tokens processed: `2274168`
- validation tokens processed: `530`
- held-out tokens scored: `161`
- optimizer steps completed: `24`
- wall clock: `24000 ms`
- mean tokens per second: `94785`
- peak tokens per second: `94849`
- mean step latency: `1000 ms`
- total artifact bytes: `387768`

The retained stage receipt also records:

- real CUDA-backed optimizer steps on the canonical single-node trainer path
- durable checkpoint promotion at step `24`
- replay receipt with `exact_replay_observed=true`
- source-family loss reports across train, validation, and held-out splits

## Resume verification on the copied-back accelerated artifact bundle

Command:

```bash
cargo run -q -p psionic-train --example psion_reference_pilot_resume_probe -- \
  /Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-reference-long-20260412t173800Z/reference_pilot_artifacts \
  /tmp/tailnet-4080-reference-long-resume-probe-20260412t173800Z
```

Result:

- completed successfully
- restored checkpoint ref: `psion-reference-pilot-step-24`
- recovery mode: `resume_from_last_stable_checkpoint`
- output:
  `/tmp/tailnet-4080-reference-long-resume-probe-20260412t173800Z/psion_reference_pilot_resume_probe.json`

This confirms the copied-back accelerated artifacts can be restored through the
same checkpoint-pointer path used by the bounded reference lane, not a one-off
debug path.

## Remote cleanup verification

After the successful run:

```bash
ssh christopherdavid@archlinux \
  'test -d $HOME/code/psion-reference-pilot/tailnet-4080-reference-long-20260412t173800Z && echo REMOTE_DIR_PRESENT || echo REMOTE_DIR_REMOVED'
```

Result:

- `REMOTE_DIR_REMOVED`

## Interpretation

This is a stronger retained proof than the earlier 4-step accelerated smoke
run:

- it stays on the same shipped `reference_pilot` lane
- it uses the admitted Tailnet RTX 4080 worker over the normal operator path
- it runs a larger bounded budget with real CUDA-backed optimizer steps
- it retains copied-back artifacts that pass the standard resume probe
- it leaves the remote staging directory cleaned up afterward

It is still intentionally scoped as a bounded reference-pilot proof. It does
not upgrade the claim surface to the broader actual-pretraining lane.

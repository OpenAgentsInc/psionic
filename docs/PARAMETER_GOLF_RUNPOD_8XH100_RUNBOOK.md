# Parameter Golf RunPod 8xH100 Runbook

> Status: canonical bounded RunPod operator runbook for the Parameter Golf
> `8xH100` lane, updated 2026-03-24 after the repo-owned launch profile,
> operator preflight, SSH-capable launcher, and finalizer contract landed.

This runbook defines the current honest RunPod posture for the Parameter Golf
distributed `8xH100` lane.

It is narrower than a completed real `8xH100` run and narrower than a
record-track claim.

## Canonical Artifacts

- launch profile authority:
  `fixtures/parameter_golf/runpod/parameter_golf_runpod_8xh100_launch_profiles_v1.json`
- operator preflight policy:
  `fixtures/parameter_golf/runpod/parameter_golf_runpod_8xh100_operator_preflight_policy_v1.json`
- cost and runtime guardrail:
  `fixtures/parameter_golf/runpod/parameter_golf_runpod_8xh100_cost_guardrails_v1.json`
- operator preflight wrapper:
  `scripts/parameter-golf-runpod-operator-preflight.sh`
- SSH-capable launcher with manifest-only rehearsal mode:
  `scripts/parameter-golf-runpod-launch-8xh100.sh`
- distributed-evidence finalizer:
  `scripts/parameter-golf-runpod-finalize-8xh100.sh`
- distributed receipt builder from a real RunPod run root:
  `scripts/parameter-golf-runpod-build-8xh100-receipt.sh`
- app-facing visualization contract:
  `docs/REMOTE_TRAINING_VISUALIZATION.md`
- local rehearsal:
  `scripts/check-parameter-golf-runpod-8xh100-lane.sh`
- committed local rehearsal report:
  `fixtures/parameter_golf/reports/parameter_golf_runpod_8xh100_operator_rehearsal.json`

## Profile Contract

The committed RunPod lane is:

- profile id: `runpod_8xh100_parameter_golf`
- trainer lane id: `parameter_golf_distributed_8xh100`
- provider: `runpod`
- pod shape: one single pod with exactly `8` non-MIG `H100` devices
- expected execution backend: `cuda`
- declared world size: `8`
- declared grad accumulation steps: `1`
- workspace root: `/workspace`
- declared run cost ceiling: `500.0` USD
- declared runtime ceiling: `4.0` hours

The profile also keeps the operator assumptions explicit:

- the pod is user-provided rather than API-created by this repo
- the workspace is a persistent mount rooted at `/workspace`
- the runtime image is a user-provided CUDA-capable Ubuntu image rather than a
  repo-owned cloud image family
- the immutable PGOLF input-package descriptor is reused from the committed
  Google bucket materialization

## Execution Posture

The committed launcher is explicit about four separate operator phases:

- remote preflight:
  prove SSH reachability, preserve the launch manifest in the remote run root,
  verify the required command inventory, and snapshot `nvidia-smi`

- pre-training:
  validate the workspace contract, bind the immutable PGOLF input descriptor,
  and stage the exported submission folder directly into the retained
  `records/track_non_record_16mb/<submission_id>` root that later execution
  and finalization consume
- execution entrypoint:
  run the exported folder under the public `WORLD_SIZE=8` posture while
  forcing the explicit exported-folder execution mode
  `PSIONIC_PARAMETER_GOLF_EXECUTION_MODE=distributed_8xh100_train`
- finalization:
  generate the exported-folder submission run evidence under the RunPod
  `8xH100` posture, mirror the retained distributed challenge receipt into the
  run root, capture `nvidia-smi` inventory and `nvidia-smi topo -m`, then seal
  the provider-neutral remote-training visualization bundle and run index under
  one machine-readable finalizer report

The launcher now also preserves the remote operator surface itself under one
machine-readable launch receipt plus per-phase logs:

- `parameter_golf_runpod_8xh100_launch_manifest.json`
- `parameter_golf_runpod_8xh100_launch_receipt.json`
- `remote_preflight.log`
- `pre_training.log`
- `execution.log`
- `finalizer.log`

This means later real `8xH100` runs can retain not only the exported-folder
evidence and finalizer outputs, but also the exact remote phase boundary,
phase commands, and per-phase exit results from the launcher that drove the
pod.

This explicit execution-mode requirement is intentional. The current exported
folder only ships the default bounded local-reference replay payload plus the
real single-H100 trainer payload. It does not yet ship a real distributed
`8xH100` trainer payload. The RunPod launcher therefore requests the reserved
distributed mode explicitly so the execution phase fails closed instead of
silently taking the local-reference replay path under `WORLD_SIZE=8`.

The finalizer now also resolves the exported submission root explicitly. The
canonical path is still the retained `records/track_non_record_16mb/<submission_id>`
folder, but the finalizer will fail closed only after checking both the
requested path and the retained `${run_root}/exported_submission` root for
`submission.json`. This keeps the operator surface explicit while surviving the
older pre-fix layout during rehearsal.

When a real distributed receipt already exists in the run root, the finalizer
now passes that exact receipt into the exported-folder evidence generator rather
than regenerating a synthetic `MeasurementsMissing` refusal from posture alone.

When a real distributed receipt does not exist yet but the run root contains
`parameter_golf_distributed_8xh100_measurements.json`, the finalizer now asks
Psionic to build the typed receipt directly from:

- `nvidia_smi_inventory.txt`
- the canonical RunPod `tensor_collective_mesh` capability posture
- the operator-collected distributed timing or memory measurements JSON

This removes the old manual `devices.json` / capability JSON / config JSON
assembly step from the first real RunPod `8xH100` evidence path.

When the measurements JSON is also absent but the run root still preserves
`execution.log`, the finalizer now first asks Psionic to derive
`parameter_golf_distributed_8xh100_measurements.json` from that retained log,
then lifts the resulting measurements plus `nvidia_smi_inventory.txt` into the
typed distributed receipt.

This is intentional. `#460` closes the RunPod lane definition and rehearsal,
not the later real `8xH100` evidence bundle.

## Local Rehearsal

The repo-owned local rehearsal is:

```bash
bash scripts/check-parameter-golf-runpod-8xh100-lane.sh \
  --report /tmp/parameter_golf_runpod_8xh100_operator_rehearsal.json
```

That rehearsal verifies:

- the committed launch profile, cost guardrail, and operator preflight policy
  are internally consistent
- the preflight accepts the RunPod `8xH100` profile locally
- the manifest-only launcher emits a launch manifest that preserves the public
  `WORLD_SIZE=8` posture, the immutable input descriptor, the launch-receipt
  paths, and the finalizer contract

## Honest Boundary

This runbook does not claim:

- a successful real RunPod `8xH100` execution
- final challenge metrics from exported-folder `8xH100` hardware
- challenge-speed closure
- record-track readiness
- that the current exported folder already contains a real distributed `8xH100`
  trainer payload

It closes one narrower but important thing:

- the repo now owns one explicit RunPod `8xH100` operator lane with a
  committed profile, preflight, SSH-capable launcher with manifest-only
  rehearsal mode, launch receipt, finalizer contract, and app-facing
  visualization bundle family that later real hardware runs can reuse without
  redefining the operator surface from scratch

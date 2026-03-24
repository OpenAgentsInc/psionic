# Parameter Golf RunPod 8xH100 Runbook

> Status: canonical bounded RunPod operator runbook for the Parameter Golf
> `8xH100` lane, written on 2026-03-23 after the repo-owned launch profile,
> operator preflight, manifest-only launcher, and finalizer contract landed.

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
- manifest-only launcher:
  `scripts/parameter-golf-runpod-launch-8xh100.sh`
- distributed-evidence finalizer:
  `scripts/parameter-golf-runpod-finalize-8xh100.sh`
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

The committed manifest is explicit about three separate operator phases:

- pre-training:
  validate the workspace contract, bind the immutable PGOLF input descriptor,
  and stage the exported submission folder
- execution entrypoint:
  run the exported folder under the public `WORLD_SIZE=8` posture
- finalization:
  capture `nvidia-smi` inventory, `nvidia-smi topo -m`, exported-folder
  digests, and the later distributed evidence paths under one machine-readable
  finalizer report

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
  `WORLD_SIZE=8` posture, the immutable input descriptor, and the finalizer
  contract

## Honest Boundary

This runbook does not claim:

- a successful real RunPod `8xH100` execution
- final challenge metrics from exported-folder `8xH100` hardware
- challenge-speed closure
- record-track readiness

It closes one narrower but important thing:

- the repo now owns one explicit RunPod `8xH100` operator lane with a
  committed profile, preflight, manifest-only launcher, and finalizer contract
  that later real hardware runs can reuse without redefining the operator
  surface from scratch

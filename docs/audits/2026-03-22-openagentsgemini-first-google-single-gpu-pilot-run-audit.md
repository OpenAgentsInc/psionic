# 2026-03-22 OpenAgentsGemini First Google Single-GPU Pilot Run Audit

This audit records the first actual Google-hosted `Psion` pilot run on
`openagentsgemini` after the same-day readiness program closed.

Bottom line:

- one bounded Google single-node run completed as `bounded_success`
- the run proved launch, bootstrap, evidence retention, checkpoint archive, and
  cold-restore truth on real Google infra
- the current reference pilot bundle still did not exercise GPU compute on the
  L4 host, so this is not yet proof of effective GPU-backed pretraining

## Claim Boundary

This audit proves only one bounded single-project, single-region, single-node
Google pilot with retained evidence.

It does not claim:

- trusted-cluster readiness
- multi-node or cross-region training readiness
- invoice-grade cost truth inside the repo
- broader `Psion` pretraining completion
- effective GPU utilization for the current reference pilot command

## Successful Run

- final successful run id:
  `psion-g2-l4-google-pilot-20260322t184426z`
- project: `openagentsgemini`
- zone: `us-central1-a`
- profile: `g2_l4_single_node`
- machine: `g2-standard-8`
- accelerator: `nvidia-l4 x1`
- base image:
  `deeplearning-platform-release/common-cu128-ubuntu-2204-nvidia-570-v20260320`
- git revision:
  `11decde48ef69c401acd16090bd3f34a7956586b`
- declared run cost ceiling: `15 USD`
- training command:
  `cargo run -p psionic-train --example psion_reference_pilot_bundle -- "$PSION_OUTPUT_DIR"`

## Attempt History

The first truthful Google run did not go green on the first paid attempt. The
attempt history matters because it shows what was actually broken in the live
lane and what had to be fixed before the final bounded-success run.

1. `psion-g2-l4-google-pilot-20260322t174150z`
   - result: `bootstrap_failure`
   - preserved cause:
     startup script exited on `HOME: unbound variable`
   - fix:
     `cb5be11` `Handle missing HOME in Google startup`
2. `psion-g2-l4-google-pilot-20260322t180136z`
   - result: `artifact_upload_failure`
   - preserved cause:
     checkpoint archive helper assumed the current working directory was a git
     repo
   - fix:
     `fe1eb3d` `Resolve Google checkpoint helpers from script path`
3. `psion-g2-l4-google-pilot-20260322t181648z`
   - result: `artifact_upload_failure`
   - preserved cause:
     `git` rejected the repo overlay as a dubious-ownership checkout on the VM
   - fix:
     `57b4b56` `Trust Google run checkout for archive helpers`
4. `psion-g2-l4-google-pilot-20260322t182953z`
   - result: `checkpoint_restore_failure`
   - preserved cause:
     cold restore ran from `/` and failed to find `Cargo.toml`
   - fix:
     `11decde` `Run Google cold restore from repo root`
5. `psion-g2-l4-google-pilot-20260322t184426z`
   - result: `bounded_success`

## Timeline

Final successful run timeline from the retained final manifest:

- launch manifest created: `2026-03-22T18:44:39Z`
- bootstrap started: `2026-03-22T18:45:27Z`
- bootstrap finished: `2026-03-22T18:46:26Z`
- training started: `2026-03-22T18:46:26Z`
- training finished: `2026-03-22T18:54:07Z`
- checkpoint archive finished: `2026-03-22T18:54:27Z`
- cold restore finished: `2026-03-22T18:54:44Z`
- teardown finished: `2026-03-22T18:54:48Z`

Interpretation:

- the host spent most of its wall time compiling the repo-owned bundle on the
  fresh VM
- the pilot program itself recorded `16` optimizer steps and `640 ms`
  pilot-runtime wall clock inside the observability receipt

## Retained Evidence

Key retained Google evidence:

- final manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-google-pilot-20260322t184426z/final/psion_google_run_final_manifest.json`
- manifest of manifests:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-google-pilot-20260322t184426z/final/psion_google_run_manifest_of_manifests.json`
- launch manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-google-pilot-20260322t184426z/launch/psion_google_single_node_launch_manifest.json`
- checkpoint archive manifest:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/reference_pilot/psion-reference-pilot-run/psion-reference-pilot-step-16/archive/psion_google_reference_checkpoint_archive_manifest.json`
- cold-restore manifest:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/reference_pilot/psion-reference-pilot-run/psion-reference-pilot-step-16/cold_restore/psion-google-cold-restore-20260322t185439z/psion_google_reference_checkpoint_cold_restore_manifest.json`

The final manifest retained `34` objects with per-object SHA-256 digests,
including:

- launch manifest, startup-script snapshot, and quota-preflight receipt
- host facts, runtime snapshot, GPU samples, GPU summary, run timeline, and run
  outcome
- structured event log plus training stdout and stderr
- stage config, stage receipt, observability receipt, optimizer state, replay
  receipt, and checkpoint lineage surfaces
- architecture, normative-spec, held-out, route, and refusal benchmark outputs
- checkpoint archive and cold-restore manifests
- one manifest-of-manifests checksum surface for the full retained bundle

## Pilot Outcome

What the run actually proved:

- the Google launch profile allocated cleanly in `us-central1-a`
- the repo-owned startup path bootstrapped the VM successfully
- the immutable input package materialized and overlaid correctly
- the reference pilot bundle completed with exit code `0`
- the host uploaded a durable checkpoint archive
- the repo-owned cold-restore probe replayed
  `resume_from_last_stable_checkpoint`
- the final Google-host evidence bundle was uploaded before teardown

Reference-pilot receipts from the bounded-success run:

- stage receipt:
  `psion-reference-pretrain-stage` for run
  `psion-reference-pilot-run`
- replay receipt:
  `psion-reference-replay`
- checkpoint lineage:
  `psion-reference-pilot-run-checkpoint-lineage`
- observability receipt:
  `psion-reference-pilot-run-observability`

Pilot metrics from retained receipts:

- optimizer steps completed: `16`
- train tokens processed: `32768`
- validation tokens processed: `530`
- held-out tokens scored: `161`
- mean tokens per second: `52279`
- checkpoint label:
  `psion-reference-pilot-step-16`
- checkpoint artifact bytes: `387256`

Benchmark and policy receipts from the run:

- architecture benchmark pass rate: `10000 bps`
- normative-spec benchmark pass rate: `10000 bps`
- held-out benchmark pass rate: `10000 bps`
- route-class selection accuracy: `10000 bps`
- route false-positive delegation: `0 bps`
- route false-negative delegation: `0 bps`
- unsupported-request refusal accuracy: `10000 bps`
- refusal reason-code match: `10000 bps`
- supported-control overrefusal: `0 bps`

## GPU Reality

This run used a real `nvidia-l4` host, but the retained GPU summary recorded:

- sample count: `99`
- average GPU utilization: `0%`
- max GPU utilization: `0%`
- max GPU memory used: `0 MiB`

That is not a telemetry bug. The current
`psion_reference_pilot_bundle` lane is still CPU-bound even when it runs on a
GPU VM, so this audit proves Google-host execution truth, not accelerator
throughput truth.

## Cost Truth

The launch manifest enforced a declared `15 USD` run ceiling for the bounded
pilot.

The retained observability receipt reports an internal estimated pilot cost of
`4000 microusd` (`0.004 USD`), but that receipt is still the reference-pilot's
internal estimate, not the project's invoice-grade Google billing record.

So the run now has:

- explicit cost ceiling truth
- bounded runtime truth
- retained evidence truth

It still does not have repo-local invoice-grade Cloud Billing truth.

## Cleanup

After the final manifest landed, the VM was deleted with the repo-owned teardown
script:

- deleted instance:
  `psion-g2-l4-google-pilot-20260322t184426z`

## Go Or No-Go

Go:

- yes for the next bounded Google-hosted `Psion` pilot if the question is
  whether the repo can launch, retain evidence, archive checkpoints, and prove
  cold restore on real Google infra

No-go:

- no for claiming effective GPU-backed pretraining throughput
- no for claiming broader Google training readiness beyond this bounded
  single-node reference lane

Recommended next step:

- keep the same launch, evidence, archive, and restore envelope
- replace the current CPU-bound reference pilot command with the first real
  accelerator-using `Psion` training lane before making any stronger Google
  pretraining claim

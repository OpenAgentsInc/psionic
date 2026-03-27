# First Swarm Trusted-LAN Runbook

> Status: canonical bounded runbook for the first local mixed-hardware swarm
> lane, added 2026-03-24 after landing the trusted-LAN topology contract,
> launch bundle script, failure-drill bundle, and exact two-node runbook path.

This runbook is the exact operator guide for the first local swarm lane:

- one Mac Apple Silicon host running the MLX Metal contributor plus validator
  and aggregation roles
- one Linux desktop host with an RTX 4080 running the CUDA contributor role
- one trusted-LAN cluster namespace with no internet discovery posture
- one shared `gpt_oss.decoder_lm_head_lora` open-adapter contract

This is intentionally narrower than the broader cluster bring-up runbooks. Use
this runbook when the goal is the exact first local Mac-plus-4080 lane and not
a general cluster rehearsal.

## What This Runbook Freezes

- topology contract:
  `fixtures/swarm/first_swarm_trusted_lan_topology_contract_v1.json`
- shared binder projection:
  `fixtures/training/runpod_local_training_binder_projection_v1.json`
- failure-drill bundle:
  `fixtures/swarm/reports/first_swarm_trusted_lan_failure_drills_v1.json`
- rehearsal report:
  `fixtures/swarm/reports/first_swarm_trusted_lan_rehearsal_v1.json`
- live-attempt evidence bundle:
  `fixtures/swarm/reports/first_swarm_trusted_lan_evidence_bundle_v1.json`
- closeout report:
  `fixtures/swarm/reports/first_swarm_trusted_lan_closeout_v1.json`
- after-action audit:
  `docs/audits/2026-03-24-first-swarm-closeout-after-action-audit.md`
- retained real-run bundle:
  `fixtures/swarm/runs/first-swarm-live-20260327-real-2/first_swarm_real_run_bundle.json`
- retained coordinator runtime report:
  `fixtures/swarm/runs/first-swarm-live-20260327-real-2/coordinator_runtime_report.json`
- retained contributor runtime report:
  `fixtures/swarm/runs/first-swarm-live-20260327-real-2/contributor_runtime_report.json`
- admitted Tailnet operator:
  `scripts/run-first-swarm-tailnet-admitted-live.sh`
- retained admitted Tailnet run bundle:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/first_swarm_real_run_bundle.json`
- retained admitted Tailnet per-device summary:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json`
- retained admitted Tailnet audit:
  `docs/audits/2026-03-27-tailrun-admitted-home-tailnet-run-audit.md`
- retained real-run after-action audit:
  `docs/audits/2026-03-27-first-swarm-trusted-lan-real-run-audit.md`
- retained local snapshot publication report:
  `fixtures/swarm/publications/first_swarm_local_snapshot_publication_v1.json`
- retained local snapshot publication root:
  `fixtures/swarm/publications/local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot`
- retained local snapshot publication audit:
  `docs/audits/2026-03-27-first-swarm-local-snapshot-publication-proof.md`
- first swarm workflow plan:
  `fixtures/swarm/first_swarm_live_workflow_plan_v1.json`
- Mac bring-up report:
  `fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json`
- Linux bring-up report:
  `fixtures/swarm/reports/swarm_linux_rtx4080_bringup_v1.json`
- bundle-materializing launcher:
  `scripts/first-swarm-launch-trusted-lan.sh`
- end-to-end checker:
  `scripts/check-first-swarm-trusted-lan.sh`
- rehearsal checker:
  `scripts/check-first-swarm-trusted-lan-rehearsal.sh`
- live-attempt bundle checker:
  `scripts/check-first-swarm-trusted-lan-evidence-bundle.sh`
- closeout checker:
  `scripts/check-first-swarm-trusted-lan-closeout.sh`
- real-run operator:
  `scripts/run-first-swarm-trusted-lan-live.sh`
- real-run bundle checker:
  `scripts/check-first-swarm-trusted-lan-real-run.sh`
- local snapshot publication checker:
  `scripts/check-first-swarm-local-snapshot-publication.sh`
- shared binder reference:
  `docs/RUNPOD_LOCAL_TRAINING_BINDER_REFERENCE.md`

## What This Runbook Does Not Claim

- no claim that this repo already ships a general mixed-backend trainer
- no claim that internet-wide discovery, elastic world-size changes, or
  configured-peer rollout are part of this lane
- no claim that bundle materialization is the same thing as a live successful
  two-node training run
- no claim that the retained real run automatically published or promoted a
  served model

## Frozen Host Posture

Mac coordinator host:

- node id: `swarm-mac-a`
- host alias: `swarm-mac-a.local`
- backend label: `open_adapter_backend.mlx.metal.gpt_oss_lm_head`
- logical device label: `metal:0`
- cluster endpoint: `swarm-mac-a.local:34100`
- repo dir: `~/code/psionic`
- run root: `~/swarm-runs/<run_id>/mac`

Linux contributor host:

- node id: `swarm-linux-4080-a`
- host alias: `swarm-linux-4080-a.local`
- backend label: `open_adapter_backend.cuda.gpt_oss_lm_head`
- logical device label: `cuda:0`
- cluster endpoint: `swarm-linux-4080-a.local:34101`
- repo dir: `~/code/psionic`
- run root: `~/swarm-runs/<run_id>/linux`

Shared cluster posture:

- namespace: `cluster.swarm.local.trusted_lan`
- admission posture: `trusted_lan.shared_secret`
- admission-token env var: `PSIONIC_SWARM_ADMISSION_TOKEN`
- heartbeat interval: `1000 ms`
- stale-worker threshold: `5000 ms`
- contributor-loss grace: `7500 ms`
- max worker skew: `15000 ms`

## First Command

Validate the exact lane contract and the bundle-materialization launcher:

```bash
scripts/check-first-swarm-trusted-lan.sh
```

If this fails, do not describe the lane as frozen or operator-repeatable.

## Bundle Materialization

Materialize one operator bundle for the exact lane:

```bash
scripts/first-swarm-launch-trusted-lan.sh \
  --run-id first-swarm-local-$(date -u +%Y%m%dT%H%M%SZ) \
  --bundle-dir /tmp/first-swarm-local-bundle \
  --manifest-only
```

This writes:

- `first_swarm_trusted_lan_topology_contract_v1.json`
- `reports/first_swarm_trusted_lan_failure_drills_v1.json`
- `first_swarm_live_workflow_plan_v1.json`
- `reports/swarm_mac_mlx_bringup_v1.json`
- `reports/swarm_linux_rtx4080_bringup_v1.json`
- `first_swarm_trusted_lan_launch_manifest.json`
- `first_swarm_trusted_lan_launch_receipt.json`

The launcher stops after local bundle materialization. It does not contact
either host and it does not claim a live run.

## Current Rehearsal Verdict

The canonical rehearsal report now lives at:

- `fixtures/swarm/reports/first_swarm_trusted_lan_rehearsal_v1.json`

Regenerate and validate it with:

```bash
scripts/check-first-swarm-trusted-lan-rehearsal.sh
```

Current verdict:

- recommendation: `no_go`
- why:
  the exact trusted-LAN topology, launch bundle, and failure drills are real,
  but contributor execution, upload staging, validator timing, and aggregation
  timing are still partly simulated and not yet backed by a live two-node
  contribution receipt set

## Current Live-Attempt Bundle

The canonical first live-attempt evidence bundle now lives at:

- `fixtures/swarm/reports/first_swarm_trusted_lan_evidence_bundle_v1.json`

Regenerate and validate it with:

```bash
scripts/check-first-swarm-trusted-lan-evidence-bundle.sh
```

Current live-attempt outcome:

- disposition: `refused`
- promotion: `no_promotion`
- why:
  the bundle preserves the exact contributor plan, launch status, and no-go
  gate, but refuses to fabricate contributor execution, validator, aggregation,
  or publication receipts that do not exist yet

That bundle remains historically useful, but it is no longer the newest
truthful retained outcome for the lane.

## Current Retained Real Run

The canonical retained real run now lives at:

- `fixtures/swarm/runs/first-swarm-live-20260327-real-2/first_swarm_real_run_bundle.json`

Validate it with:

```bash
scripts/check-first-swarm-trusted-lan-real-run.sh \
  --bundle fixtures/swarm/runs/first-swarm-live-20260327-real-2/first_swarm_real_run_bundle.json
```

Current retained real-run outcome:

- result classification: `bounded_success`
- merge: `merged`
- publish: `refused`
- promotion: `held`
- why:
  the live run earned two accepted contributor submissions, two replay-checked
  contributions, one shared validator summary, and one aggregated bounded
  result across the Mac MLX coordinator and Linux RTX 4080 contributor, but it
  still stopped short of a promoted published snapshot

## Current Admitted Tailnet Proof

The canonical admitted-device Tailnet proof now lives at:

- `fixtures/swarm/runs/tailrun-home-admitted-20260327e/first_swarm_real_run_bundle.json`
- `fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json`

Validate the retained bundle with:

```bash
scripts/check-first-swarm-trusted-lan-real-run.sh \
  --bundle fixtures/swarm/runs/tailrun-home-admitted-20260327e/first_swarm_real_run_bundle.json
```

Rerun the admitted-device Tailnet operator path with fresh ports:

```bash
RUN_ID="tailrun-home-admitted-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/run-first-swarm-tailnet-admitted-live.sh \
  --run-id "${RUN_ID}" \
  --bundle-dir "fixtures/swarm/runs/${RUN_ID}" \
  --coordinator-port 35200 \
  --contributor-port 35201
```

Current retained admitted-Tailnet outcome:

- result classification: `bounded_success`
- merge: `merged`
- publish: `refused`
- promotion: `held`
- admitted device set:
  local M5 MLX coordinator plus `archlinux` RTX 4080 CUDA contributor
- contribution split:
  two accepted contributions and two replay-checked contributions, one from
  each admitted device

The matching operator audit now lives at:

- `docs/audits/2026-03-27-tailrun-admitted-home-tailnet-run-audit.md`

## Current Publication Proof

The canonical retained local snapshot publication proof now lives at:

- `fixtures/swarm/publications/first_swarm_local_snapshot_publication_v1.json`
- `fixtures/swarm/publications/local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot/publish_manifest.json`

Validate it with:

```bash
scripts/check-first-swarm-local-snapshot-publication.sh
```

Regenerate it with:

```bash
cargo run -q -p psionic-mlx-workflows --bin first_swarm_local_snapshot_publication -- \
  fixtures/swarm/publications
```

Current retained publication-proof outcome:

- publish target: `hugging_face_snapshot`
- publish id: `first-swarm-local-snapshot`
- snapshot root:
  `local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot`
- why:
  the repo now retains one truthful local snapshot directory for the frozen
  first-swarm publish target, but this proof is separate from the retained live
  run and does not change that run's `publish=refused` and `promotion=held`
  outcome

The matching publication audit now lives at:

- `docs/audits/2026-03-27-first-swarm-local-snapshot-publication-proof.md`

## Current Closeout Outcome

The canonical first swarm closeout report now lives at:

- `fixtures/swarm/reports/first_swarm_trusted_lan_closeout_v1.json`

Regenerate and validate it with:

```bash
scripts/check-first-swarm-trusted-lan-closeout.sh
```

Current closeout verdict:

- merge: `no_merge`
- publish: `refused`
- expected publish path if a later run earns promotion:
  `local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot`
- why:
  the lane still has no accepted contributor receipt set, no replay receipts,
  no aggregation result, and no promoted local snapshot, so the closeout keeps
  the existing MLX publish surface explicit without pretending a snapshot was
  actually published

The matching after-action audit now lives at:

- `docs/audits/2026-03-24-first-swarm-closeout-after-action-audit.md`

Treat that closeout as the historical pre-success refusal record. The current
completion record for `SWARM-0` is the retained real run plus:

- `docs/audits/2026-03-27-first-swarm-trusted-lan-real-run-audit.md`

## Exact Per-Host Commands

Mac coordinator:

```bash
cd ~/code/psionic
scripts/check-swarm-mac-mlx-bringup.sh \
  --report ~/swarm-runs/<run_id>/mac/reports/swarm_mac_mlx_bringup_v1.json
```

Linux contributor:

```bash
cd ~/code/psionic
scripts/check-swarm-linux-4080-bringup.sh \
  --report ~/swarm-runs/<run_id>/linux/reports/swarm_linux_rtx4080_bringup_v1.json
```

Coordinator workflow freeze:

```bash
cd ~/code/psionic
cargo run -q -p psionic-mlx-workflows --bin first_swarm_live_workflow_plan -- \
  /tmp/first-swarm-local-bundle/first_swarm_live_workflow_plan_v1.json
```

Coordinator topology and drill freeze:

```bash
cd ~/code/psionic
cargo run -q -p psionic-train --bin first_swarm_trusted_lan_topology_contract -- \
  /tmp/first-swarm-local-bundle/first_swarm_trusted_lan_topology_contract_v1.json

cargo run -q -p psionic-train --bin first_swarm_trusted_lan_failure_drills -- \
  /tmp/first-swarm-local-bundle/reports/first_swarm_trusted_lan_failure_drills_v1.json
```

Live retained run:

```bash
cd ~/code/psionic
scripts/run-first-swarm-trusted-lan-live.sh \
  --run-id first-swarm-live-$(date -u +%Y%m%dT%H%M%SZ)
```

## Required Failure Drills

The lane is not ready to describe as operator-repeatable unless these four
drills stay frozen:

- stale worker:
  Linux contributor heartbeat exceeds `5000 ms`; validator posture stays
  `replay_required`
- upload disagreement:
  contributor upload manifest digest mismatches the workflow plan; validator
  posture stays `rejected`
- contributor loss:
  Linux contributor departs during the active window; aggregation stays blocked
- uneven worker speed:
  contributor skew exceeds `15000 ms`; operator waits briefly, then replays

The canonical machine-legible drill bundle is:

```bash
cargo run -q -p psionic-train --bin first_swarm_trusted_lan_failure_drills
```

## Stop Conditions

Stop the attempt immediately if any of the following happens:

- the Mac bring-up report no longer emits the MLX Metal contributor receipt
- the Linux bring-up report no longer emits the CUDA contributor receipt
- the workflow-plan digest or membership receipt digest drifts from the
  topology contract
- the upload manifest observed on either host diverges from the workflow plan
- a contributor disappears and the operator cannot replay under the same
  contract

## Claim Boundary

This runbook proves that the first swarm lane now has one exact trusted-LAN
topology contract, one exact bundle-materializing launcher, one exact per-host
preflight path, one exact failure-drill bundle, and one exact rehearsal-grade
bottleneck report plus one explicit refused live-attempt evidence bundle and
one explicit no-merge/no-publish closeout report, one retained real mixed-
hardware run, and one separate retained local snapshot publication proof. It
does not by itself prove automatic publication from the real run, full-model
mixed-backend dense training, or internet-facing swarm operation.

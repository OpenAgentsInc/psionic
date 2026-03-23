# Psion Google Single-GPU Runbook

> Status: canonical `PSION-44` / `#410` runbook, written on March 22, 2026
> after the Google launch bundle, immutable input package, checkpoint archive,
> host observability finalizer, and validation folders landed on `main`, then
> extended on March 23, 2026 after the first truthful accelerator-backed
> single-node follow-up audit landed.

This runbook is the operator entrypoint for the bounded Google-hosted `Psion`
pilot lane.

It is intentionally narrow:

- one project: `openagentsgemini`
- one region family: `us-central1`
- one node at a time
- one Google Compute Engine VM, not a trusted cluster
- one bounded single-node lane with retained evidence

It does not claim broader pretraining completion, trusted-cluster readiness,
cross-region orchestration, or production serving readiness for trained
checkpoints.

## CPU-Reference Boundary

The historical bounded Google lane in this runbook is a GPU-hosted operator
proof, not an accelerator-backed training proof.

Explicitly:

- `psion_reference_pilot_bundle` is a CPU-reference lane
- the current host-native plugin-conditioned reference lane is a bounded
  reference or evidence lane, not a GPU proof target
- the current mixed plugin-conditioned reference lane is a bounded reference or
  evidence lane, not a GPU proof target

These lanes remain valid for:

- operator bootstrap proof
- retained evidence proof
- archive and restore proof
- bounded route, refusal, and capability-boundary fixtures

They are not valid as the primary proof target for Google GPU training claims.

Every future GPU-training audit must name:

- the exact trainer command
- the delivered execution backend
- whether the run is only GPU-hosted or truly accelerator-backed

The default Google single-node profile is now the bounded accelerated lane:

- profile: `g2_l4_single_node_accelerated`
- trainer lane: `psion_accelerated_reference_pilot`
- expected execution backend: `cuda`

The first successful retained run on that lane is:

- Google run id: `psion-g2-l4-accelerated-20260323t074419z`
- result classification: `bounded_success`
- audit:
  `docs/audits/2026-03-23-openagentsgemini-first-google-accelerator-backed-single-node-psion-training-audit.md`

## Accelerator-Backed Pass Criteria

The accelerated Google lane is no longer allowed to count a run as successful
only because the VM booted and the training command exited zero.

For any launch profile whose manifest declares `expected_execution_backend =
cuda`, the Google host lane now requires:

- a stage receipt whose `delivered_execution.runtime_backend` is `cuda`
- a stage receipt whose `accelerator_execution.accelerator_backed` flag is
  `true`
- non-zero post-warmup GPU utilization in the retained `nvidia-smi` sample log
- non-zero post-warmup GPU memory residency in the retained `nvidia-smi`
  sample log
- a retained accelerator-validation receipt that links backend truth, GPU
  sample truth, and throughput truth

If those conditions are not met, the run is classified as
`training_runtime_failure` rather than `bounded_success`.

The committed accelerator gate policy lives in:

- `fixtures/psion/google/psion_google_host_observability_policy_v1.json`

The committed retained accelerator evidence now includes:

- `psion_google_gpu_samples.csv`
- `psion_google_gpu_summary.json`
- `psion_google_accelerator_validation_receipt.json`
- the training stage receipt
- the training observability receipt
- the final manifest that links all of the above by digest and remote URI

## Canonical Artifacts

- launch authority:
  `fixtures/psion/google/psion_google_single_node_launch_profiles_v1.json`
- local operator preflight:
  `fixtures/psion/google/psion_google_operator_preflight_policy_v1.json`
- host observability and final-manifest authority:
  `fixtures/psion/google/psion_google_host_observability_policy_v1.json`
- local operator preflight command:
  `scripts/psion-google-operator-preflight.sh`
- quota and cost preflight:
  `scripts/psion-google-quota-preflight.sh`
- launch command:
  `scripts/psion-google-launch-single-node.sh`
- host startup:
  `scripts/psion-google-single-node-startup.sh`
- host finalizer:
  `scripts/psion-google-finalize-run.sh`
- checkpoint archive:
  `scripts/psion-google-archive-reference-pilot-checkpoint.sh`
- checkpoint cold restore:
  `scripts/psion-google-cold-restore-reference-pilot.sh`
- teardown:
  `scripts/psion-google-delete-single-node.sh`
- readiness audit:
  `docs/audits/2026-03-22-openagentsgemini-gpu-training-pilot-audit.md`
- first real run audit:
  `docs/audits/2026-03-22-openagentsgemini-first-google-single-gpu-pilot-run-audit.md`
- first real accelerator-backed single-node run audit:
  `docs/audits/2026-03-23-openagentsgemini-first-google-accelerator-backed-single-node-psion-training-audit.md`
- first real host-native plugin-conditioned run audit:
  `docs/audits/2026-03-22-openagentsgemini-first-google-host-native-plugin-conditioned-run-audit.md`
- first real mixed plugin-conditioned run audit:
  `docs/audits/2026-03-22-openagentsgemini-first-google-mixed-plugin-conditioned-run-audit.md`

## Local Requirements

Required local commands:

- `gcloud`
- `bq`
- `jq`
- `git`

Minimum local CLI versions:

- `gcloud >= 470.0.0`
- `bq >= 2.1.0`

Current validated local versions at the time this runbook landed:

- `gcloud 556.0.0`
- `bq 2.1.28`

Required auth posture:

- active `gcloud` account must be able to mint an access token
- active `gcloud` project must be `openagentsgemini`
- the operator must be able to read
  `gs://openagentsgemini-psion-train-us-central1`
- the operator must be able to read
  `openagentsgemini:psion_training_finops`
- the operator must be able to describe
  `psion-train-single-node@openagentsgemini.iam.gserviceaccount.com`

Environment variables:

- no override is required for the default lane
- optional overrides:
  `PROJECT_ID`, `PROFILE_ID`, `ZONE`, `RUN_ID`, `INSTANCE_NAME`
- this runbook only authorizes `PROJECT_ID=openagentsgemini`

Secret posture:

- prefer attached-service-account access on the VM
- do not inject runtime secrets unless strictly required by a later issue
- if a future run depends on one secret, record the dependency in the launch or
  follow-up audit without recording the secret value

## Claim Boundary

This runbook covers one bounded Google single-node pilot only.

It still refuses to claim:

- trusted-cluster training
- cross-region or multi-node training
- arbitrary machine-family support beyond the committed launch profiles
- broad cost optimality across Google GPU families
- broader pretraining completion for `Psion`

## Bounded Lane Overrides

The launcher now also supports bounded lane overrides on top of the committed
single-node profile:

- `--input-package-descriptor-uri`
- `--training-command`
- `--post-training-archive-command`
- `--post-training-restore-command`

Use these only when the alternate lane still preserves:

- one committed input-package descriptor
- one explicit training command in the launch manifest
- one explicit archive or restore posture in the launch manifest
- the same operator-internal, single-node, no-publication claim boundary

To disable the default post-training restore path for a bounded lane that
retains archive evidence but does not prove the reference-pilot cold-restore
surface, pass:

```bash
--post-training-restore-command __none__
```

The first committed example of this bounded-override posture is the host-native
plugin-conditioned Google audit above, followed by the mixed plugin-conditioned
Google audit.

## 1. Local Preflight

Run the repo-owned operator gate before any paid launch:

```bash
bash scripts/psion-google-operator-preflight.sh --profile g2_l4_single_node_accelerated
```

Optional explicit zone override:

```bash
bash scripts/psion-google-operator-preflight.sh \
  --profile g2_l4_single_node_accelerated \
  --zone us-central1-a
```

This command rejects:

- missing local CLI dependencies
- stale `gcloud` or `bq`
- missing or mismatched `gcloud` auth
- missing bucket access
- missing BigQuery FinOps access
- missing training service-account visibility
- quota-unready launch profiles

Do not continue if the result is not `ready`.

## 2. Optional Manifest-Only Rehearsal

Before the paid launch, you may upload only the launch manifest, startup-script
snapshot, and quota-preflight receipt:

```bash
RUN_ID="psion-g2-l4-accelerated-$(date -u +%Y%m%dt%H%M%Sz | tr '[:upper:]' '[:lower:]')"

bash scripts/psion-google-launch-single-node.sh \
  --profile g2_l4_single_node_accelerated \
  --manifest-only \
  --run-id "${RUN_ID}" \
  --instance-name "${RUN_ID}"
```

Expected launch artifacts:

- `gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/launch/psion_google_single_node_launch_manifest.json`
- `gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/launch/psion-google-single-node-startup.sh`
- `gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/launch/psion_google_quota_preflight.json`

## 3. Paid Launch

Primary accelerated lane:

- profile: `g2_l4_single_node_accelerated`
- trainer lane:
  `cargo run -p psionic-train --example psion_accelerated_reference_pilot -- "$PSION_OUTPUT_DIR"`
- expected backend: `cuda`
- fallback order:
  `us-central1-a`, `us-central1-b`, `us-central1-c`

Launch:

```bash
RUN_ID="psion-g2-l4-accelerated-$(date -u +%Y%m%dt%H%M%Sz | tr '[:upper:]' '[:lower:]')"

bash scripts/psion-google-launch-single-node.sh \
  --profile g2_l4_single_node_accelerated \
  --run-id "${RUN_ID}" \
  --instance-name "${RUN_ID}"
```

The launch manifest for this profile now records:

- `trainer_lane_id = psion_accelerated_reference_pilot`
- `expected_execution_backend = cuda`
- the exact `cargo run ... psion_accelerated_reference_pilot` training command

Historical CPU-reference lane:

- profile: `g2_l4_single_node`
- fallback order:
  `us-central1-a`, `us-central1-b`, `us-central1-c`

Launch:

```bash
RUN_ID="psion-g2-l4-$(date -u +%Y%m%dt%H%M%Sz | tr '[:upper:]' '[:lower:]')"

bash scripts/psion-google-launch-single-node.sh \
  --profile g2_l4_single_node \
  --run-id "${RUN_ID}" \
  --instance-name "${RUN_ID}"
```

What happens:

- the launcher resolves the pinned GPU image from
  `deeplearning-platform-release`
- one launch manifest is uploaded before the VM request
- the VM boots without an external IP on `oa-lightning`
- the startup script owns bootstrap, repo checkout, input materialization,
  training, checkpoint archive, cold restore, and final evidence upload

For the accelerated profile, the same host lifecycle stays intact, but the VM
now executes the canonical CUDA trainer instead of the CPU reference bundle.

This historical lane remains acceptable for bounded operator rehearsals and
CPU-reference evidence retention. It is no longer an acceptable primary target
for accelerator-backed GPU validation.

If `gcloud compute instances create` fails before the VM exists, the launch
folder still retains:

- a typed `launch_capacity_failure` final manifest
- the launch-failure log
- the launch manifest, startup snapshot, and quota-preflight receipt

## 4. Monitoring

Describe the instance:

```bash
gcloud compute instances describe "${RUN_ID}" \
  --project=openagentsgemini \
  --zone=us-central1-a \
  --format='value(status)'
```

IAP SSH if you need interactive inspection:

```bash
gcloud compute ssh "${RUN_ID}" \
  --project=openagentsgemini \
  --zone=us-central1-a \
  --tunnel-through-iap
```

Watch the run folder:

```bash
gcloud storage ls \
  "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/**"
```

Expected host-emitted artifacts:

- `host/psion_google_host_facts.json`
- `host/psion_google_runtime_snapshot.json`
- `host/psion_google_gpu_summary.json`
- `host/psion_google_run_timeline.json`
- `host/psion_google_run_outcome.json`
- `logs/psion_google_run_events.jsonl`
- `logs/training.stdout.log`
- `logs/training.stderr.log`
- `logs/psion_google_gpu_samples.csv`
- `receipts/...`
- `artifacts/psion_reference_pilot_checkpoint.safetensors`
- `final/psion_google_run_manifest_of_manifests.json`
- `final/psion_google_run_final_manifest.json`

## 5. Checkpoint Archive And Restore

The host startup now runs these automatically after a green reference-pilot
bundle:

- `scripts/psion-google-archive-reference-pilot-checkpoint.sh`
- `scripts/psion-google-cold-restore-reference-pilot.sh`

Operator verification after the run:

```bash
gcloud storage cat \
  "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/final/psion_google_run_final_manifest.json" \
  | jq '.checkpoint_recovery'
```

Manual restore drill against a known archive manifest remains available:

```bash
bash scripts/psion-google-cold-restore-reference-pilot.sh \
  "gs://openagentsgemini-psion-train-us-central1/checkpoints/reference_pilot/psion-reference-pilot-run/psion-reference-pilot-step-16/archive/psion_google_reference_checkpoint_archive_manifest.json"
```

Do not treat local VM disks as durable checkpoint authority. The archive
manifest and cold-restore manifest are the durable recovery surfaces.

## 6. Evidence Bundle Verification

The single source of truth after the VM is gone is:

```bash
gcloud storage cat \
  "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/final/psion_google_run_final_manifest.json" \
  | jq '.result_classification, .timeline, .manifest_of_manifests, .retained_objects | length'
```

What the final manifest must preserve:

- launch topology and machine profile
- image identity
- input package bindings and digests
- host facts and runtime snapshot
- GPU summary and raw GPU samples
- structured event log plus stdout and stderr
- pilot receipts and benchmark outputs
- checkpoint archive and cold-restore references
- one manifest-of-manifests checksum
- one retained-object list with per-object digests

Deletion is refused by default if the final manifest is missing.

## 7. Teardown

After the final manifest exists:

```bash
bash scripts/psion-google-delete-single-node.sh --run-id "${RUN_ID}"
```

Only bypass the retention guard if you are intentionally preserving a failed
partial launch folder and the VM is already absent:

```bash
bash scripts/psion-google-delete-single-node.sh --run-id "${RUN_ID}" --force
```

## 8. Known Refusals After The First Google Run

Even after one successful bounded Google run, this repo still refuses to claim:

- multi-node data parallel or model parallel Google training
- cross-region or failover-ready Google orchestration
- trusted-cluster promotion
- broader `Psion` pretraining completion
- production serving readiness for the resulting checkpoint

Those claims stay blocked on later issues and later audits.

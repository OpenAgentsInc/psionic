# 2026-03-22 OpenAgentsGemini Single-GPU Training Pilot Readiness Audit

This audit checks whether the current Google Cloud project can honestly launch
one bounded Psion GPU training pilot and retain enough infra, artifact, and
cost evidence to make the run useful.

I did not allocate a GPU VM in this audit. This is a control-plane and quota
readiness pass over the current `gcloud`-visible project state.

## Non-goals Of This Audit

This audit does not claim:

- live zonal GPU stock proof
- benchmark success for the training run itself
- trusted-cluster readiness
- cross-region or multi-node training readiness
- production serving readiness for trained checkpoints
- broad cost optimality across machine families

It is strictly a single-project, single-region, single-node readiness audit for
one bounded Psion GPU pilot with retained evidence.

## Scope

- inspected local `gcloud` auth, active config, and project identity
- checked billing attachment and enabled services
- checked regional quota and zone accelerator catalogs in `us-central1`
- inventoried existing Compute Engine, network, storage, logging, and BigQuery
  surfaces
- checked project IAM for the active user
- did not verify the exact promotional-credit balance because the queried CLI
  surfaces did not expose it directly
- did not verify live GPU stock at allocation time because no launch was
  attempted

## Project Facts

- active account: `chris@openagents.com`
- active project: `openagentsgemini`
- active default region and zone: `us-central1` and `us-central1-a`
- active user project role: `roles/owner`
- billing is enabled on
  `billingAccounts/01D15C-64524A-1062EA`
- enabled services already include:
  `compute.googleapis.com`, `storage.googleapis.com`,
  `logging.googleapis.com`, `monitoring.googleapis.com`,
  `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`,
  `container.googleapis.com`, and `aiplatform.googleapis.com`

## Training-Relevant Capacity

Regional quota in `us-central1`:

- `CPUS`: limit `200`, usage `14`
- `INSTANCES`: limit `720`, usage `7`
- `SSD_TOTAL_GB`: limit `20480`, usage `3612`
- `NVIDIA_T4_GPUS`: limit `4`, usage `0`
- `PREEMPTIBLE_NVIDIA_T4_GPUS`: limit `4`, usage `0`
- `NVIDIA_L4_GPUS`: limit `8`, usage `0`
- `PREEMPTIBLE_NVIDIA_L4_GPUS`: limit `8`, usage `0`
- `NVIDIA_A100_GPUS`: limit `1`, usage `0`
- `PREEMPTIBLE_NVIDIA_A100_GPUS`: limit `16`, usage `0`
- `A2_CPUS`: limit `12`, usage `0`
- `NVIDIA_A100_80GB_GPUS`: limit `0`, usage `0`
- `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS`: limit `0`, usage `0`

Zone accelerator catalog across `us-central1-a`, `us-central1-b`,
`us-central1-c`, and `us-central1-f`:

- `nvidia-tesla-t4` is visible in `a`, `b`, `c`, and `f`
- `nvidia-l4` is visible in `a`, `b`, and `c`
- `nvidia-tesla-a100` is visible in `a`, `b`, `c`, and `f`
- `nvidia-a100-80gb` is visible in `a` and `c`, but the project quota for
  `NVIDIA_A100_80GB_GPUS` is `0`
- higher-end catalog entries such as `nvidia-h100-80gb`, `nvidia-h200-141gb`,
  `nvidia-gb200`, and `nvidia-b200` are visible in some zones, but this audit
  did not find matching non-zero project quota for them, so they should be
  treated as unavailable for launch planning

Machine families seen in the same region:

- `g2-standard-4` and `g2-standard-8` in `a`, `b`, and `c`
- `a2-highgpu-1g` in `a`, `b`, `c`, and `f`

Practical reading:

- one standard A100 pilot is available on paper today
- at least one L4-backed G2 pilot is available on paper today
- T4 remains available for a cheaper smoke pass
- A100 80GB is not available today because quota is zero even though the zone
  catalog exposes the accelerator type

## Existing Infra

Current Compute Engine footprint:

- five running non-GPU VMs:
  `nexus-mainnet-1`, `nexus-staging-1`, `oa-bitcoind`, `oa-lnd`, and
  `symphony-mainnet-1`
- all current project hosts are in `us-central1-a`
- current instances are operational infrastructure, not training hosts

Current disk footprint:

- persistent disks already consume `3612 GB` of the regional SSD quota
- existing disks are attached to the current operational hosts, not a training
  lane

Current networks:

- `default`
- `oa-lightning`

Current subnets:

- `oa-lightning-us-central1` on `10.42.0.0/24`
- the standard auto-mode `default` network subnets across many regions

Current firewall posture:

- `default` still has the broad default SSH rule from `0.0.0.0/0`
- `oa-lightning` already has IAP SSH rules, but only for existing target tags
  such as `nexus-host`
- a new training VM on `oa-lightning` would need its own target tag and SSH
  rule, or the launch should use the `default` network instead

Current storage and recordkeeping:

- the only bucket listed is `gs://openagentsgemini_cloudbuild`
- there is no dedicated training artifact bucket
- there are no BigQuery datasets listed, so there is no obvious billing-export
  dataset or training-metrics dataset already in place
- logging only has the default `_Required` and `_Default` sinks

Current service-account posture:

- the project has the default Compute Engine service account plus a small set of
  app-specific service accounts
- there is no dedicated Psion training service account yet

Current governance hygiene:

- the project has no resource-manager tag bindings
- the `gcloud` config warning about a missing `environment` tag is accurate
- this is not a launch blocker, but it is a small hygiene gap

## Status Call

The GCP project is close, but not fully pilot-ready.

What is green:

- billing is enabled
- the active user has owner rights
- Compute Engine, storage, logging, monitoring, and Vertex APIs are already on
- `us-central1` quota is sufficient for a bounded single-GPU pilot
- the zone catalog shows realistic T4, L4, and A100 launch targets

What is still only `partial`:

- no dedicated training artifact bucket exists
- no dedicated training service account exists
- no billing-export dataset or budget surface was found from the queried CLI
  views
- no explicit training-host network and SSH posture exists yet
- no runbook exists yet for preserving the full infra evidence bundle around a
  GPU launch

Blunt conclusion:

- yes, this project appears capable of launching a bounded single-GPU Psion
  pilot, subject to live zonal stock and a short setup pass
- no, it is not yet cleanly prepared to do that and preserve all relevant run
  evidence without a short setup pass first

## Recommended First GPU Pilot

Use a single L4-backed G2 launch first unless the run truly needs A100 memory
or throughput.

Why:

- L4 quota is wider than standard A100 quota
- G2 shapes are visible in multiple `us-central1` zones
- burning the only standard A100 slot is unnecessary for a first truthful pilot
- A100 80GB is not actually launchable today under the current quota

Recommended order:

1. first truthful pilot: one L4-backed G2 instance in `us-central1-a`,
   `us-central1-b`, or `us-central1-c`
2. second step if memory pressure requires it: one `a2-highgpu-1g`
3. leave Spot or preemptible usage for later once checkpoint persistence is
   proven, even though spot-like quota exists

Preferred zonal fallback order for the first L4 pilot:

1. `us-central1-a`
2. `us-central1-b`
3. `us-central1-c`

Initial zone fallback order for the first launch:

- L4 plus `g2`: `us-central1-a`, then `us-central1-b`, then `us-central1-c`
- A100 plus `a2-highgpu-1g`: `us-central1-b`, then `us-central1-c`, then
  `us-central1-a`, then `us-central1-f`

These are only operator defaults. Actual launch truth still depends on live
zonal stock at allocation time.

If all preferred L4 zones fail for live capacity reasons, do not silently
upgrade to A100. Record the capacity miss and require an explicit operator
decision before changing machine profile.

## Minimum Setup Before Launch

- create a dedicated regional bucket for training artifacts, checkpoints, and
  receipts instead of reusing the Cloud Build bucket
- create a dedicated service account for the pilot instead of reusing the
  existing app runtime accounts
- choose the network posture explicitly:
  `default` for the fastest bring-up, or `oa-lightning` plus a new IAP SSH tag
  and rule for a cleaner private-network story
- create either a Cloud Billing budget or a billing export dataset so credits
  and cost drift are not invisible during the run
- define one launch wrapper that snapshots quota, instance metadata, machine
  details, and training outputs into the bucket

## Launch Decision Rule

A first real Google Psion pilot may be launched only when:

- Issues 1 through 9 are closed
- the preferred machine profile still passes quota preflight
- the launch manifest has a declared cost ceiling
- the immutable input package has been materialized
- cold-restore procedure has already passed on archived checkpoints or the run
  is explicitly classified as a non-resumable smoke pass

## Remaining Operational Risks

This audit does not yet close the following launch-critical risks:

- live zonal GPU stock may still block allocation even when quota and catalog
  look green
- bootstrap reproducibility may drift if the first run relies on floating base
  images or first-boot NVIDIA driver installation behavior
- local boot-disk pressure could invalidate the run if dataset staging,
  checkpoints, or logs exceed the undeclared working-disk budget
- artifact durability is still partial until object digests, checkpoint
  manifests, and cold-restore proof exist
- cost truth remains partial until billing export or an equivalent
  machine-queryable surface is wired into the run bundle
- operator-local tooling drift can still break launch or evidence collection if
  `gcloud`, `bq`, or auth posture remain implicit
- the audit still does not prove Google-host execution readiness until one real
  run completes with preserved evidence

## Required Issue Program

The missing work splits into infra setup issues and repo-owned execution issues.

These issues are the minimum program I would use before saying Google infra is
ready for the first real Psion pretraining run.

The intended claim boundary stays narrow:

- single project: `openagentsgemini`
- single region: `us-central1`
- single node first, not trusted-cluster scale-up
- on-demand first, with Spot or preemptible only after checkpoint restore is
  proven
- one bounded Psion pretraining run with full evidence retention, not a broad
  pretraining-completion claim

### Issue 1: Provision A Dedicated Psion Training Artifact Bucket

Description:
The project only has `gs://openagentsgemini_cloudbuild`. That bucket is not a
clean training artifact authority, and the rented-cluster runbook explicitly
requires checkpoints to survive ephemeral hosts.

Required details:

- create a dedicated bucket in the training region or a compatible low-latency
  location for the first run
- use a stable naming scheme such as
  `gs://openagentsgemini-psion-train-us-central1`
- enable uniform bucket-level access
- enable object versioning
- define one layout at minimum:
  `runs/`, `checkpoints/`, `receipts/`, `logs/`, and `manifests/`
- set retention and lifecycle policy intentionally instead of leaving retention
  behavior implicit
- keep checkpoint artifacts and final run bundles more durable than transient
  logs

Acceptance details:

- the eventual training host can upload and read back checkpoint and receipt
  objects
- deleting the VM does not delete the durable run artifacts
- the bucket path is recorded in the launch manifest and final run bundle

Dependencies:
none

### Issue 2: Create A Dedicated Training Service Account And Project Hygiene Baseline

Description:
There is no dedicated Psion training service account, and the project currently
has no `environment` tag bindings. The first real run should not borrow broad
app runtime identities or leave metadata hygiene implicit.

Required details:

- create a dedicated service account for the training VM
- grant only the roles needed for bucket writes, log writes, metric writes, and
  any required read access for launch assets
- keep operator launch rights with the human owner account instead of teaching
  the VM service account to mutate unrelated project resources
- add explicit labels or tags for training hosts, buckets, and run folders
- add the missing project `environment` tag or an equivalent bounded project
  hygiene marker

Acceptance details:

- the training VM can write its artifacts and telemetry without using the
  default compute service account
- the training service account cannot administer unrelated project infra
- training resources are machine-filterable by labels or tags

Dependencies:
Issue 1

### Issue 3: Define The Training Network And Operator Access Posture

Description:
The project has `default` plus `oa-lightning`, but neither is already shaped as
an explicit training network. `default` still carries broad SSH posture, while
`oa-lightning` only has IAP SSH for existing app target tags.

Required details:

- choose the first-run network posture explicitly instead of inheriting one by
  accident
- if `default` is used for speed, state that it is a temporary bootstrap choice
  and not the long-term training network posture
- if `oa-lightning` is used, add a dedicated training tag and IAP SSH rule
  instead of reusing the existing operational node tags
- document whether the first training VM should have an external IP or rely on
  IAP plus private egress
- preserve OS Login and IAP-based access truth if the host stays on the custom
  network
- keep the training host separate from the existing operational node tag set

Acceptance details:

- an operator can reach the training VM through the documented path
- the training host does not silently inherit unrelated app firewall posture
- the network and tag choice is captured in the run manifest

Dependencies:
Issue 2

### Issue 4: Add Billing Export, Budget Guardrails, And Quota Preflight

Description:
The project has billing enabled, but this audit did not find a visible billing
export dataset, budget artifact, or training-specific quota preflight wrapper.
That leaves credit burn and overrun detection too implicit for a rented run.

Required details:

- create a Cloud Billing budget with alert thresholds
- export billing data to BigQuery or another machine-queryable cost sink
- define one quota-preflight command that checks region, zone, accelerator, CPU,
  and disk headroom before launch
- define a bounded launch-attempt policy covering stock failure, zone fallback,
  and the point where the operator must stop retrying and record a
  `launch_capacity_failure`
- define an estimated max-cost envelope for the first run and bind it to the
  launch manifest
- align stop conditions with the rented-cluster runbook cost posture rather
  than inventing a separate cloud-only policy

Acceptance details:

- launch fails early when quota or estimated cost is outside the declared
  bounds
- run cost can be reconstructed after the fact without reading the Cloud
  Console manually
- the final run bundle carries enough cost facts to support a
  `PSION-13`-class observability receipt

Dependencies:
none

### Issue 5: Land A Repo-Owned Google Single-Node Launch Bundle

Description:
There is no repo-owned GCP launch script, manifest writer, startup script, or
machine-profile authority yet. Today the repo can run the Psion reference pilot
locally, but not through one committed Google-infra operator surface.

Required details:

- add one repo-owned launch entrypoint for the first Google single-node pilot
- bind the chosen machine profile into a manifest instead of making the
  operator remember the shape manually
- carry at least these machine profiles explicitly:
  `g2-standard-*` plus `1x L4`, and `a2-highgpu-1g` plus `1x A100`
- pin the base image family, source image project, and exact image identifier
  recorded in the manifest instead of relying on a floating implicit image
- prefer a prevalidated GPU-ready image after the first successful run instead
  of treating first-boot driver installation as the permanent operator story
- capture zone, machine type, accelerator type, disk type, disk size, bucket
  path, git revision, and training command in one launch manifest
- define a minimum boot-disk size and local scratch policy covering dataset
  cache, logs, and temporary checkpoint materialization
- define low-disk watermark and fail-fast behavior before the run silently
  collapses on local storage
- carry the initial zone fallback order in the launch logic instead of leaving
  zone choice to operator improvisation
- use a startup script or equivalent reproducible bootstrap path that can:
  install or verify NVIDIA driver state, fetch the repo at the selected
  revision, materialize run directories, and start the bounded training command
- add a cleanup or teardown companion so abandoned VMs do not become backlog

Acceptance details:

- one command launches the declared training VM shape
- one manifest object records the exact infra and code inputs
- the manifest records the exact image identity, boot-disk policy, and zone
  chosen after any fallback
- one command tears down the host after artifacts are safely retained

Dependencies:
Issues 1 through 4

### Issue 6: Package And Bind Immutable Training Inputs For Remote Execution

Description:
A real cloud run needs more than a machine launch. It also needs a stable input
package so the remote host is provably running the intended corpus, tokenizer,
stage config, benchmark package set, and code revision.

Required details:

- define the exact remote input package for the first run
- include digests for:
  source or tokenized corpus inputs, tokenizer artifacts, stage config,
  benchmark fixtures, and the exact code revision
- define how the first run obtains training data:
  staged to local disk, streamed from GCS, or cached with an explicit bounded
  local working-set policy
- teach the launch path to fetch those exact artifacts, not floating local
  state
- write one manifest that ties all input digests to the run id
- require checksum validation for the input package, benchmark fixtures, and
  any uploaded manifests before training begins
- define the benchmark execution posture explicitly:
  minimal held-out and pilot-critical probes on-host, with heavier benchmark
  passes allowed post hoc from the archived checkpoint unless the pilot contract
  requires them during the run
- keep the Google operator lane aligned with the canonical Psion stage and pilot
  docs instead of inventing a second cloud-specific training schema

Acceptance details:

- rerunning the same manifest on a fresh VM reproduces the same input digests
- the stage receipt and final run bundle point back to the immutable input
  package
- the run can prove whether inputs were staged, streamed, or cached locally
- there is no hidden dependency on the operator’s unstated local workspace

Dependencies:
Issues 1 and 5

### Issue 7: Add GCS-Backed Checkpoint Archive And Cold-Restore Validation

Description:
The rented-cluster runbook explicitly rejects ephemeral checkpoint truth. A real
Google run therefore needs GCS-backed checkpoint persistence plus a restore path
validated on a fresh host.

Required details:

- upload stable checkpoints and checkpoint manifests to the training bucket
- keep local VM disks as cache or working state, not the only durable copy
- implement or script cold restore onto a new VM from the archived checkpoint
- record restore success or failure with the same checkpoint-lineage discipline
  used by the existing Psion training artifacts
- require object digests for archived checkpoint artifacts and the checkpoint
  manifest itself before a restore is accepted as valid
- explicitly test the required rented-cluster downgrade path:
  `resume_from_last_stable_checkpoint`

Acceptance details:

- a checkpoint archived during the run can be restored on a fresh host
- resumed training can prove which checkpoint was resumed and why
- VM deletion does not destroy the only recoverable training state

Dependencies:
Issues 1, 5, and 6

### Issue 8: Add Google-Host Observability And Evidence Export

Description:
The repo already has the canonical `PSION-13` observability receipt and the
pilot bundle contract, but there is no Google-host collector that captures the
hardware and infra facts needed to populate those receipts honestly on a real
cloud run.

Required details:

- capture instance metadata:
  project, zone, machine type, accelerator type, disk shape, and network mode
- record the exact base image identity and the UTC launch, bootstrap, training
  start, checkpoint, and teardown timestamps
- capture runtime metadata:
  NVIDIA driver, CUDA runtime, `nvidia-smi`, CPU, memory, disk, and uptime
- capture GPU utilization and memory-usage summaries so a "green" run cannot
  hide obviously bad GPU occupancy or throughput
- retain stdout, stderr, and structured training logs
- emit or assemble the Psion stage receipt, observability receipt, replay
  receipt, checkpoint lineage, benchmark receipts, and pilot bundle from the
  actual remote run
- require typed failure classification for non-green runs, with at least:
  `launch_capacity_failure`, `bootstrap_failure`, `driver_runtime_failure`,
  `artifact_upload_failure`, `training_divergence`,
  `checkpoint_restore_failure`, `cost_guardrail_abort`, and `operator_abort`
- require per-object checksums in the final uploaded manifest, including a
  manifest-of-manifests checksum for the full evidence folder
- upload one final manifest listing every retained object for the run

Acceptance details:

- every Google training run produces one bucket folder that is self-describing
- the observability receipt can be regenerated from retained machine facts
- a failed run still produces typed timestamps, failure code, and object digests
- the final run folder is enough to audit cost, topology, checkpoint identity,
  and run outcome after the VM is gone

Dependencies:
Issues 1, 5, 6, and 7

### Issue 9: Publish A Google Single-GPU Psion Operator Runbook

Description:
There is no repo-owned operator runbook for a Google single-node Psion training
run yet. The first truthful run should be executable from docs and committed
scripts rather than from remembered terminal history.

Required details:

- add one dedicated runbook under `docs/`
- document preflight, launch, monitoring, checkpoint archive, restore drill,
  evidence upload, and teardown
- document required local operator tooling versions or minimum versions for
  `gcloud` and `bq`, required auth posture, and required environment variables
- add one local preflight that fails if the operator toolchain or auth posture
  is incompatible with the Google pilot lane
- document the secret posture explicitly:
  prefer attached-service-account access, avoid runtime secret injection unless
  strictly required, and record secret dependencies without recording secret
  values
- keep the claim boundary explicit:
  single-region, single-node, Google Compute Engine first
- document refusal boundaries too:
  no trusted-cluster claim, no cross-region cluster claim, no broader
  pretraining completion claim
- point the runbook at the exact scripts, manifests, and evidence bundle
  expected by the repo’s existing training docs

Acceptance details:

- one operator can run the full bounded procedure from the repo without hidden
  tribal knowledge
- every command and artifact path in the runbook is still executable
- local tooling or auth mismatch is rejected before any paid launch begins
- the runbook states what remains refused after the first Google run

Dependencies:
Issues 1 through 8

### Issue 10: Execute The First Real Google Single-GPU Psion Pretraining Run

Description:
After the setup issues are closed, the repo still needs one actual bounded run
on Google infra. Until that exists, the project only has inferred launch
readiness, not proved execution readiness.

Result classes for the follow-up audit:

- `launch_capacity_failure`
- `bootstrap_failure`
- `input_materialization_failure`
- `training_runtime_failure`
- `checkpoint_archive_failure`
- `checkpoint_restore_failure`
- `cost_guardrail_abort`
- `bounded_success`

Required details:

- use the preferred first-run shape:
  one L4-backed `g2` instance unless memory pressure forces `a2-highgpu-1g`
- follow the declared zone fallback order and stop retrying after the bounded
  launch-attempt policy is exhausted
- define when the run should:
  abort immediately, checkpoint then abort, retry in another zone, or downgrade
  from an A100 request to an L4 request
- keep the run bounded by explicit cost, time, step, or token ceilings
- abort if the run fails to upload a durable checkpoint by the declared
  checkpoint deadline
- abort if the evidence-upload path is broken and the run can no longer produce
  a truthful retained bundle
- archive the full evidence bundle into the dedicated training bucket
- validate the emitted stage receipt, observability receipt, replay facts,
  checkpoint lineage, and pilot bundle from the real run
- write a follow-up audit that states whether the run stayed green or exposed
  blockers for the next attempt, and record a typed result classification for
  the final outcome

Acceptance details:

- one actual Google-hosted run completes or fails with explicit preserved cause
- the final evidence bundle is enough to support a truthful go or no-go call on
  the next pretraining step
- the repo has a committed audit of the result, not just a console anecdote
- the final result can be classified cleanly as success, bounded refusal, or
  typed failure

Dependencies:
Issues 1 through 9

## Evidence Bundle The Pilot Should Save

At minimum, the first run should upload one bucket path containing:

- the instance description JSON
- the image self-link or equivalent exact image identity
- the launch command or startup script
- the launch manifest with cost ceiling, machine profile, zone choice, and
  input-package digests
- `nvidia-smi`, disk, memory, and CPU snapshots
- GPU utilization summaries and step-throughput snapshots
- the git revision and exact cargo or binary command used
- the Psion stage config, observability receipt, replay receipt, and checkpoint
  lineage
- stdout and stderr logs
- checkpoint manifests and checkpoint artifacts
- per-object SHA-256 digests for every retained object
- UTC timestamps for launch, training start, checkpoint writes, final upload,
  and teardown
- a typed failure classification if the run does not finish green
- a final upload manifest listing every saved object plus a manifest-of-
  manifests checksum

Without that bundle, the project may still run a GPU job, but it will not meet
the standard of a truthful training pilot with preserved evidence.

## Commands Used

The audit relied on these command families:

```bash
gcloud auth list
gcloud config list
gcloud projects list
gcloud beta billing projects describe openagentsgemini
gcloud beta billing accounts list
gcloud services list --enabled --project openagentsgemini
gcloud compute project-info describe --project openagentsgemini
gcloud compute regions describe us-central1 --project openagentsgemini
gcloud compute accelerator-types list --project openagentsgemini
gcloud compute machine-types list --project openagentsgemini
gcloud compute instances list --project openagentsgemini
gcloud compute disks list --project openagentsgemini
gcloud compute networks list --project openagentsgemini
gcloud compute networks subnets list --project openagentsgemini
gcloud compute firewall-rules list --project openagentsgemini
gcloud iam service-accounts list --project openagentsgemini
gcloud projects get-iam-policy openagentsgemini
gcloud storage buckets list --project openagentsgemini
bq ls --project_id=openagentsgemini
gcloud logging sinks list --project openagentsgemini
gcloud resource-manager tags bindings list --parent=//cloudresourcemanager.googleapis.com/projects/157437760789
```

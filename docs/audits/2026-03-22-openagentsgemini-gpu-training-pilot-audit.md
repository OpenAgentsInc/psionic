# 2026-03-22 OpenAgentsGemini Single-GPU Training Pilot Readiness Audit

This audit checks whether the current Google Cloud project can honestly launch
one bounded Psion GPU training pilot and retain enough infra, artifact, and
cost evidence to make the run useful.

Follow-up status:

- the first real bounded Google-hosted run was executed later on
  `2026-03-22`
- follow-up audit:
  `docs/audits/2026-03-22-openagentsgemini-first-google-single-gpu-pilot-run-audit.md`

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
- `oa-lightning` now has the dedicated
  `oa-allow-psion-train-iap-ssh` firewall rule scoped to
  `35.235.240.0/20 -> tcp:22` on target tag `psion-train-host`
- `oa-lightning` still also carries broader existing IAP SSH rules for other
  operational target tags, so the training lane should keep using its explicit
  dedicated tag rather than piggybacking on those rules

Current storage and recordkeeping:

- the project now has a dedicated training bucket:
  `gs://openagentsgemini-psion-train-us-central1`
- that bucket now carries uniform access, public-access prevention, object
  versioning, a 7-day retention policy, a 14-day soft-delete window, and the
  committed top-level layout for `runs/`, `checkpoints/`, `receipts/`, `logs/`,
  and `manifests/`
- the project now has the BigQuery dataset
  `openagentsgemini:psion_training_finops` for the repo-owned training FinOps
  lane
- logging only has the default `_Required` and `_Default` sinks

Current billing guardrails:

- the project now has the monthly Cloud Billing budget
  `Psion Google Single-Node Pilot` under
  `billingAccounts/01D15C-64524A-1062EA`
- the budget amount is `150 USD` with thresholds at `50%`, `75%`, `90%`, and
  `100% forecasted spend`
- budget notifications now publish to
  `projects/openagentsgemini/topics/psion-google-budget-alerts`
- the project now has the BigQuery dataset
  `openagentsgemini:psion_training_finops`
- the repo-owned cost sink table
  `openagentsgemini.psion_training_finops.single_node_price_profiles_v1` now
  holds the live us-central1 price envelope for the first `g2 + L4` and
  `a2 + A100` single-node profiles
- the repo now has the quota preflight command
  `scripts/psion-google-quota-preflight.sh` for those two first-launch profiles

Current launch bundle:

- the repo now has the launch entrypoint
  `scripts/psion-google-launch-single-node.sh`
- the repo now has the startup path
  `scripts/psion-google-single-node-startup.sh`
- the repo now has the host finalizer
  `scripts/psion-google-finalize-run.sh`
- the repo now has the teardown companion
  `scripts/psion-google-delete-single-node.sh`
- the committed launch-profile authority now lives at
  `fixtures/psion/google/psion_google_single_node_launch_profiles_v1.json`
- the committed host-observability authority now lives at
  `fixtures/psion/google/psion_google_host_observability_policy_v1.json`
- the launch bundle now resolves and records the exact GPU image from
  `deeplearning-platform-release` family
  `common-cu128-ubuntu-2204-nvidia-570`
- a manifest-only run now uploads a full launch manifest, startup-script
  snapshot, and quota-preflight receipt into the training bucket without yet
  allocating a GPU host
- a failed `gcloud compute instances create` attempt now also uploads a typed
  final manifest plus a retained launch-failure log instead of only returning
  terminal stderr

Current observability and evidence path:

- startup now owns the full bounded run lifecycle instead of detaching
  immediately, so bootstrap, training, archive, cold-restore, and teardown
  timestamps remain explicit
- the host path now records structured run events, host facts, runtime
  snapshots, GPU utilization samples, GPU summary stats, stdout, stderr, and
  the exact pilot output files into one run folder
- the host finalizer now uploads one manifest-of-manifests plus one final
  manifest listing retained objects, object digests, topology, input package
  bindings, checkpoint-recovery references, and the typed result classification
- the repo-owned checkpoint archive and cold-restore helpers are now machine-
  readable enough for the startup path to bind their manifest URIs into the
  final run folder
- validation now exists for both:
  one bounded success-style evidence folder at
  `runs/psion-g2-l4-observability-validation-20260322t172627z/`
  and one pre-VM typed failure folder at
  `runs/psion-g2-l4-launch-failure-validation-20260322t173300z/`

Current service-account posture:

- the project now has the dedicated training service account
  `psion-train-single-node@openagentsgemini.iam.gserviceaccount.com`
- that service account has the bounded writer roles needed for logs, metrics,
  and artifact writes into the dedicated training bucket
- the project still has the default Compute Engine service account plus a small
  set of app-specific service accounts, but the Psion lane no longer needs to
  rely on them

Current governance hygiene:

- the project has no resource-manager tag bindings
- the `gcloud` config warning about a missing `environment` tag is accurate
- the project and bucket now carry bounded labels for the training lane even
  though org-level tags remain unavailable from the active account
- this is not a launch blocker, but it remains a small hygiene gap

## Status Call

At the time of this readiness pass, the GCP project was close, but not fully
pilot-ready. The follow-up run later closed the bounded single-node proof.

What is green:

- billing is enabled
- the active user has owner rights
- Compute Engine, storage, logging, monitoring, and Vertex APIs are already on
- `us-central1` quota is sufficient for a bounded single-GPU pilot
- the zone catalog shows realistic T4, L4, and A100 launch targets
- a dedicated training artifact bucket now exists with committed retention and
  lifecycle policy
- a dedicated single-node Psion training service account now exists with
  bounded bucket, logging, and metrics write rights
- the first training network posture is now explicit: private `oa-lightning`
  host, Cloud NAT egress, and IAP-only SSH through `psion-train-host`
- a monthly billing budget, Pub/Sub budget-notification topic, BigQuery FinOps
  dataset, and repo-owned quota preflight now exist for the first bounded run
- the repo now has one committed single-node Google launch surface with pinned
  image-family resolution, machine-profile authority, low-disk policy, and a
  teardown companion
- the repo now has one committed immutable reference input package path with a
  stable descriptor URI, archive digests, benchmark posture, and launch-manifest
  binding for remote execution
- the repo now has one committed checkpoint archive path with GCS-backed weights,
  manifest, optimizer-state retention, and a repo-owned cold-restore drill for
  `resume_from_last_stable_checkpoint`
- the repo now has one committed Google-host evidence collector and finalizer
  path that retains machine facts, GPU summaries, structured run events,
  training receipts, checkpoint-recovery references, and one integrity-linked
  final manifest for both bounded-success and pre-VM failure cases
- the repo now has one committed Google single-node operator runbook plus one
  local operator preflight command that rejects stale tooling, auth drift, and
  quota-unready launch posture before any paid launch begins
- the first real bounded Google-hosted pilot now exists and is recorded in
  `docs/audits/2026-03-22-openagentsgemini-first-google-single-gpu-pilot-run-audit.md`

What is still only `partial`:

- exact Cloud Billing export to BigQuery is still console-managed; the current
  repo-owned machine-queryable cost sink is the live price-catalog table plus
  budget notifications rather than invoice-grade billing-export rows
- the current `psion_reference_pilot_bundle` lane is still CPU-bound even on an
  L4 host, so Google launch truth is now proved but GPU-backed pretraining
  throughput truth is not

Blunt conclusion:

- yes, this project has now proved one bounded Google-hosted single-GPU Psion
  pilot with retained evidence, checkpoint archive, and cold-restore proof
- no, this still does not prove effective GPU-utilized pretraining because the
  current reference pilot command remained CPU-bound on the L4 host

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

- define the final evidence collector before the first paid run
- decide whether the first bounded pilot needs console-managed Cloud Billing
  export in addition to the repo-owned budget-plus-price-catalog cost sink

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
- artifact durability is now proved for the bounded reference lane, but broader
  checkpoint promotion and later train variants remain unproved
- cost truth remains partial until billing export or an equivalent
  machine-queryable surface is wired into the run bundle
- operator-local tooling drift can still break launch or evidence collection if
  `gcloud`, `bq`, or auth posture remain implicit
- the current bounded-success run still does not prove accelerator utilization
  truth because the reference pilot lane remained CPU-bound on the GPU host

## Required Issue Program

The missing work splits into infra setup issues and repo-owned execution issues.

These issues are the minimum program I would use before saying Google infra is
ready for the first real Psion pretraining run.

Progress note:

- Issues 1 through 10 are now closed for the bounded Google single-node
  reference lane
- broader Google pretraining readiness still remains narrower than full
  accelerator-backed training readiness because the current reference pilot
  command is still CPU-bound

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
Closed. The repo now has the canonical Google-host collector and final-manifest
path through:

- `fixtures/psion/google/psion_google_host_observability_policy_v1.json`
- `scripts/psion-google-single-node-startup.sh`
- `scripts/psion-google-finalize-run.sh`
- `scripts/psion-google-launch-single-node.sh`

Validation folder proofs now exist at:

- `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-observability-validation-20260322t172627z/final/psion_google_run_final_manifest.json`
- `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-launch-failure-validation-20260322t173300z/final/psion_google_run_final_manifest.json`

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
Closed. The canonical operator runbook now lives at:

- `docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md`
- `scripts/psion-google-operator-preflight.sh`
- `fixtures/psion/google/psion_google_operator_preflight_policy_v1.json`

The runbook now binds the exact local preflight, launch, monitoring, evidence,
checkpoint-recovery, and teardown commands for the bounded Google single-node
lane, and it keeps the refusal boundary explicit.

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

Status:

- closed on `2026-03-22` as `bounded_success`
- follow-up audit:
  `docs/audits/2026-03-22-openagentsgemini-first-google-single-gpu-pilot-run-audit.md`

Outcome summary:

- successful run id:
  `psion-g2-l4-google-pilot-20260322t184426z`
- final topology:
  `g2-standard-8` plus `nvidia-l4` in `us-central1-a`
- the run completed bootstrap, training, checkpoint archive, cold restore, and
  evidence finalization on real Google infra
- the final manifest retained `34` objects with per-object digests
- the run preserved a full typed attempt history before the bounded-success
  pass:
  `bootstrap_failure`, two `artifact_upload_failure` cases, and one
  `checkpoint_restore_failure`
- the fixes that unblocked the successful pass were committed as:
  `cb5be11`, `fe1eb3d`, `57b4b56`, and `11decde`
- the current claim boundary remains narrow:
  this proved Google-host launch and evidence truth, but not effective
  accelerator utilization, because the current reference pilot lane stayed
  CPU-bound on the L4 host

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

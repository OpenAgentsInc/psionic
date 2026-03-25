# Google Two-Node Swarm Compute Audit

> Status: written 2026-03-24 after reviewing the current Google operator
> surface, the current cluster and swarm contracts, and the current GitHub
> issue queue in `psionic`.

## Question

Can `psionic` test a swarm compute run across two Google Cloud machines now,
while simulating a wider and less reliable network than a local trusted LAN?

Yes, but not by reusing the current Google single-node scripts unchanged and
not by pretending that the existing trusted-cluster full-model lane already
covers this case.

The first honest Google test is:

- two Google Compute Engine nodes
- authenticated configured-peer cluster posture
- bounded adapter-delta swarm execution on the existing adapter-cluster
  substrate
- explicit network impairment injection on the cluster ports
- retained evidence that proves membership, reachability, delay, contributor
  receipts, validator posture, aggregation posture, and failure handling

The first honest Google test is not:

- a trusted-LAN run
- a wider-network discovery rollout
- a full-model trusted-cluster training claim
- a cross-region or public-internet all-reduce claim

## Executive Verdict

The repo is close enough to run a truthful two-node Google swarm rehearsal, but
it is not close enough to claim that the current Google lane already supports
multi-node swarm compute.

What is already real:

- the Google project, bucket, network, service-account, quota, launch, archive,
  and observability surfaces exist for one bounded single-node lane
- the cluster substrate already supports authenticated configured-peer posture,
  operator manifests, multi-subnet dial health, coordinator lease truth,
  failover fencing, and widened non-LAN posture diagnostics
- the training substrate already has a cluster-backed adapter coordinator and a
  bounded decentralized adapter contribution lane

What is still missing:

- a repo-owned Google dual-node launch bundle
- a Google dual-node operator preflight
- a Google dual-node network and identity manifest
- a Google-specific impairment policy and cluster-port shaping helper
- a two-node Google swarm evidence bundle and checker

That means the answer is:

- possible soon: yes
- supported today without new work: no
- best first target: two CUDA-backed GCE nodes running one bounded
  configured-peer adapter swarm rehearsal with explicit netem-backed network
  impairment

## Sources Reviewed

Canonical docs:

- `docs/TRAIN_SYSTEM.md`
- `docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md`
- `docs/CLUSTER_VALIDATION_RUNBOOK.md`
- `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md`
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md`
- `docs/FIRST_SWARM_TRUSTED_LAN_RUNBOOK.md`
- `docs/ROADMAP_CLUSTER.md`
- `docs/ARCHITECTURE_EXPLAINER_CLUSTER_BRINGUP_RUNBOOK.md`

Google fixtures and scripts:

- `fixtures/psion/google/psion_google_single_node_launch_profiles_v1.json`
- `fixtures/psion/google/psion_google_network_posture_v1.json`
- `fixtures/psion/google/psion_google_training_identity_profile_v1.json`
- `fixtures/psion/google/psion_google_training_storage_profile_v1.json`
- `scripts/psion-google-operator-preflight.sh`
- `scripts/psion-google-quota-preflight.sh`
- `scripts/psion-google-ensure-training-network.sh`

Training and swarm substrate:

- `crates/psionic-train/src/adapter_cluster.rs`
- `crates/psionic-train/src/open_adapter.rs`
- `crates/psionic-train/src/swarm_open_adapter.rs`
- `crates/psionic-train/src/psion_trusted_cluster_run.rs`

GitHub issues:

- closed Google lineage: `#402`, `#404`, `#406`, `#410`, `#411`
- closed cluster lineage: `#372`, `#373`
- open swarm master issue: `#484`

There is no current `psionic` GitHub issue dedicated to a Google two-node
configured-peer swarm run.

## Current Repo Truth

### Google Infra Truth

The current Google operator lane is intentionally single-node.

`docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md` freezes:

- one project: `openagentsgemini`
- one region family: `us-central1`
- one node at a time
- one Google Compute Engine VM, not a trusted cluster

The committed launch authority confirms the same boundary.

`fixtures/psion/google/psion_google_single_node_launch_profiles_v1.json`
defines only single-node profiles. The current service-account and storage
fixtures do the same. The identity profile is explicitly labeled
`single-node`, and the current operator preflight and quota preflight both
validate one profile in one zone for one node.

This is strong enough for the current bounded Google lane. It is not a dual-node
operator surface.

### Cluster Truth

The cluster substrate is ahead of the Google operator tooling.

`docs/CLUSTER_VALIDATION_RUNBOOK.md` and `docs/ROADMAP_CLUSTER.md` show that
`psionic-cluster` already has:

- authenticated configured-peer discovery
- refusal of unknown peers under configured-peer posture
- persisted operator manifest boot
- explicit non-LAN discovery posture diagnostics
- explicit dial health and degraded reachability truth
- coordinator lease freshness and stale-leader expiry
- fenced failover validation

This matters because a two-node Google rehearsal should use that configured-peer
posture instead of pretending that the Google lane is still a local trusted LAN.

### Training Truth

The right training substrate for the first Google swarm test already exists, but
it is not the trusted-cluster full-model lane.

`docs/PSION_TRUSTED_CLUSTER_RUN.md` freezes a much narrower and different
contract:

- homogeneous four-node CUDA H100 topology
- trusted-cluster posture
- explicit refusal of mixed-backend, cross-region, shared, and elastic modes

That lane should not be widened to explain a two-node Google rehearsal on a
simulated degraded network.

The better fit is the existing bounded adapter-cluster path:

- `crates/psionic-train/src/adapter_cluster.rs`
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md`

That substrate already binds live cluster membership and telemetry into
contributor eligibility, window planning, artifact staging, and validator or
aggregation posture without requiring a claim that public or trusted-cluster
full-model all-reduce is already solved for this topology.

## What This Means

If the goal is "prove we can run truthful swarm compute across two Google
machines under non-LAN-like conditions," the first run should be:

- bounded
- configured-peer
- adapter-delta
- operator-managed
- evidence-heavy

If the goal is "prove full distributed training across two arbitrary Google
machines," the repo is not there yet and the current docs already refuse that
claim.

## Recommended First Google Test

### Goal

Prove one operator-repeatable two-node Google swarm compute rehearsal with:

- authenticated configured-peer admission
- explicit contributor and coordinator roles
- one bounded adapter window
- explicit contributor upload and validator posture
- explicit aggregation or refusal posture
- explicit network impairment evidence
- retained cluster and host evidence in GCS

### Topology

Use two GCE VMs in `openagentsgemini`:

- node A:
  coordinator, validator, aggregator, and contributor
- node B:
  contributor

Use the same execution backend on both nodes for the first run:

- `open_adapter_backend.cuda.gpt_oss_lm_head`

Do not mix hardware classes in the first Google rehearsal. The main unknown for
this audit is network posture plus operator rollout, not cross-backend math
parity.

Recommended first machine family:

- two `g2` plus `1x L4` nodes

Reason:

- the repo already has a truthful Google `g2` plus `L4` operator lane
- this keeps cost and bring-up lower than introducing a new multi-node A100
  story first
- it isolates the network and cluster question from the machine-family
  question

Recommended placement:

- same project: `openagentsgemini`
- same region family: `us-central1`
- different zones inside that region family
- different dedicated training subnetworks inside the same VPC
- no external IPs
- IAP plus OS Login access only

Reason:

- different zones create a real non-LAN path without immediately widening to a
  cross-region claim
- separate subnetworks make the configured-peer story explicit
- keeping both nodes inside one repo-owned project avoids inventing a multi-VPC
  or public internet story before the operator tooling exists

### Admission And Discovery Posture

Use:

- authenticated configured-peer posture
- persisted operator manifest rollout
- explicit refusal of wider-network discovery

Do not use:

- trusted-LAN seed-peer posture
- public discovery
- elastic membership claims

The first Google test should exercise the cluster posture that the current
cluster runbook already treats as the truthful non-LAN baseline.

## How To Simulate Disparate Networks Honestly

The first simulation should stay inside GCE but impair the cluster links
explicitly.

Use Linux traffic control on the cluster ports only:

- `tc netem` for delay, jitter, loss, duplication, and reordering
- optional `tbf` or rate controls if bandwidth pressure needs to be made
  explicit
- targeted qdisc filters for the cluster peer ports so GCS, IAP SSH, and system
  package traffic remain mostly unaffected

The right failure model is not "make the whole VM slow."

The right failure model is:

- impair the cluster transport
- retain the impairment policy as an artifact
- keep the impairment aligned with cluster evidence and failure drills

Recommended impairment phases:

1. Clean baseline
   - no shaping
   - proves the configured-peer rollout and adapter window function when the
     network is healthy
2. Mild WAN simulation
   - symmetric extra latency
   - small jitter
   - very small packet loss
   - proves healthy but non-LAN contributor scheduling and window sealing
3. Asymmetric degraded path
   - higher delay on one direction or one node
   - moderate jitter
   - bounded packet loss
   - proves explicit dial health, backoff, stale-worker handling, and validator
     posture under skew
4. Short partition or blackhole drill
   - temporary 100 percent loss or explicit firewall deny on cluster ports
   - proves unreachable-peer diagnostics, late join recovery, lease freshness,
     and replay-safe refusal or rejoin behavior

The first truthful claim should stay at:

- simulated disparate networks inside one operator-managed Google environment

It should not widen to:

- cross-region production readiness
- arbitrary internet peer discovery
- public cluster trust

## Exact Validation Sequence

The operator flow should look like this.

### 1. Local preflight

Add a dual-node preflight that extends the current single-node checks to:

- two launch profiles
- two zones
- bucket access for both node roles
- identity and manifest visibility for both node roles
- explicit cluster namespace and configured-peer policy inputs

### 2. Repo cluster validation

Before any Google launch, the current cluster validation drills should stay
green:

- authenticated membership drill
- discovery posture drill
- multi-subnet dial health drill
- coordinator lease drill
- coordinator failover drill

This is already defined in `docs/CLUSTER_VALIDATION_RUNBOOK.md`.

### 3. Google launch and manifest freeze

Launch two nodes and write one machine-legible cluster manifest that binds:

- run id
- cluster id
- node ids
- zones
- subnetworks
- internal IP endpoints
- cluster namespace
- admission posture
- chosen impairment profile
- selected training command
- bucket prefixes for per-node and cluster-wide artifacts

### 4. Host bring-up

Each node should emit a per-node bring-up report that records:

- machine identity
- accelerator inventory
- driver and CUDA posture
- cluster endpoint
- selected role
- local scratch posture
- repo revision
- health of the exact training command prerequisites

### 5. Controlled swarm run

Run one bounded adapter-delta window with:

- fixed dataset slice identity
- fixed contributor set
- fixed validator policy
- fixed aggregation policy
- fixed checkpoint or base-model identity
- explicit contributor upload receipts
- explicit validator or refusal receipts
- explicit aggregation or no-aggregation result

### 6. Impaired reruns

Repeat the same bounded window under the committed impairment profiles and
retain:

- impairment policy receipt
- host timeline and GPU samples
- cluster health and lease timeline
- contributor heartbeats and stale-worker transitions
- validator timing and refusal posture
- aggregation timing or refusal posture

### 7. Finalizer and audit

Seal one cluster-wide evidence bundle and write a follow-up audit classifying
the outcome as:

- configured_peer_launch_failure
- cluster_membership_failure
- network_impairment_gate_failure
- contributor_execution_failure
- validator_refusal
- aggregation_failure
- bounded_success

## Repo-Owned Work Still Needed

The missing work is concrete.

### Google operator artifacts

Add:

- a dual-node launch authority fixture
- a dual-node identity profile or role-aware identity manifest
- a dual-node network posture fixture
- a dual-node operator preflight script
- a dual-node launch script
- a dual-node teardown script

The current single-node files should not be overloaded silently. Their scope is
already explicit.

### Cluster-specific Google artifacts

Add:

- one Google swarm cluster manifest schema
- one configured-peer endpoint manifest per node
- one impairment policy fixture
- one host-side impairment helper script
- one Google swarm finalizer that uploads both per-node and cluster-wide
  evidence

### Evidence and validation

Add:

- one Google two-node swarm evidence bundle schema
- one checker script for that bundle
- one operator runbook for the exact Google two-node configured-peer swarm lane
- one follow-up audit for the first real run

## Issue Readiness

The current issue history is enough to support opening this work cleanly.

What already exists:

- `#402`, `#404`, `#406`, `#410`, `#411` closed the bounded Google single-node
  lane
- `#372` and `#373` closed the rented-cluster and trusted-cluster cluster
  contracts
- `#484` is the current local mixed-hardware swarm master issue

What does not exist:

- a GitHub issue for a Google two-node configured-peer swarm rehearsal
- a GitHub issue for Google network impairment simulation
- a GitHub issue for a Google dual-node operator surface

The clean follow-on is a small issue stack that mirrors the single-node Google
lineage:

- Google two-node network and identity posture
- Google two-node launch bundle
- Google two-node configured-peer manifest and impairment policy
- Google two-node swarm runbook
- first real Google two-node swarm rehearsal and audit

## Recommended Claim Boundary

If this work lands, the truthful public statement is:

"Psionic can run one bounded two-node Google configured-peer swarm rehearsal
for adapter-delta contribution under explicit simulated non-LAN impairment, with
retained membership, network, contributor, validator, and aggregation evidence."

Do not say:

- "Google multi-node training is done"
- "trusted-cluster Google training is done"
- "cross-region swarm compute is done"
- "internet discovery is done"

## Bottom Line

Testing swarm compute across two Google Cloud machines is possible with the
current repo direction.

The shortest honest path is:

- stay inside `openagentsgemini`
- use two `g2` plus `L4` nodes
- use authenticated configured-peer cluster posture
- use the existing adapter-cluster swarm substrate
- inject network impairment on the cluster ports
- retain a cluster-wide evidence bundle

The current Google lane does not do this yet. The cluster and training
substrates are ready enough that the missing work is now mainly operator
surface, rollout artifacts, impairment tooling, and evidence packaging.

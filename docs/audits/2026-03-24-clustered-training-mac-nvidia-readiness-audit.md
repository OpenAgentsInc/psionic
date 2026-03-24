# 2026-03-24 Clustered Training Across Mac and NVIDIA Readiness Audit

This audit answers one concrete question:

- how ready `psionic` is, today, to do truthful clustered training across
  multiple devices
- how much of that readiness already applies to mixed Mac and NVIDIA hardware
- what the fastest honest path is to get to a real clustered training result
  across Mac and NVIDIA without overclaiming what the repo can do now

## Executive Verdict

`psionic` is meaningfully ready for cluster control, cluster truth, and bounded
homogeneous distributed-training evidence.

`psionic` is not yet ready for a truthful live clustered training lane where Mac
and NVIDIA devices participate as one coherent mixed-backend training job.

The current repo state is:

- cluster membership, topology, scheduling, artifact staging, and receipt
  surfaces are real
- a bounded homogeneous CUDA trusted-cluster training lane is real as evidence
- single-host Apple adapter training is real
- cluster-backed decentralized adapter planning, worker protocol, validator
  receipts, and promotion receipts are real
- the first mixed Apple-plus-NVIDIA path is real only as a rehearsal path for
  cluster truth, artifact flow, receipt flow, and validator flow
- live multi-node Apple training execution is not yet claimed
- live mixed Apple-plus-NVIDIA gradient exchange for one coherent job is not
  yet claimed

The fastest honest path to "Mac plus NVIDIA clustered training ASAP" is not a
full mixed all-reduce trainer.

The fastest honest path is:

1. use the existing decentralized adapter window substrate
2. keep Apple as coordinator, export host, and runtime-validation authority
3. let NVIDIA participate first through the bounded open-adapter execution lane
4. prove mixed-hardware contribution, staging, validation, replay, and
   promotion truth
5. only then widen into true mixed-backend execution if the product still needs
   one coherent gradient-sharing job

If the requirement is stricter than that, and you mean one live synchronized
mixed-backend training graph across Apple and NVIDIA devices, the repo is not
close enough to claim that as an ASAP deliverable. The missing work is still in
runtime semantics, backend contracts, collectives, checkpoint interchange, and
mixed-backend acceptance coverage.

## Sources Consulted

Canonical docs:

- `docs/ARCHITECTURE.md`
- `docs/TRAIN_SYSTEM.md`
- `docs/ROADMAP_CLUSTER.md`
- `docs/ROADMAP_METAL.md`
- `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- `docs/CLUSTER_VALIDATION_RUNBOOK.md`
- `docs/ARCHITECTURE_EXPLAINER_CLUSTER_BRINGUP_RUNBOOK.md`
- `docs/DISTRIBUTED_OPTIMIZER_REFERENCE.md`
- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md`
- `docs/PSION_PLUGIN_CLUSTER_SCALE_DECISION.md`

Relevant code:

- `crates/psionic-cluster/src/scheduler.rs`
- `crates/psionic-cluster/src/layer_sharded.rs`
- `crates/psionic-cluster/src/tensor_sharded.rs`
- `crates/psionic-cluster/tests/support/mod.rs`
- `crates/psionic-runtime/src/local_multi_device.rs`
- `crates/psionic-distributed/src/lib.rs`
- `crates/psionic-collectives/src/lib.rs`
- `crates/psionic-train/src/adapter_cluster.rs`
- `crates/psionic-train/src/open_adapter.rs`
- `crates/psionic-train/src/psion_trusted_cluster_run.rs`

Live GitHub state checked on 2026-03-24 with `gh`:

- `OpenAgentsInc/openagents#3286`: `CLOSED`
- `OpenAgentsInc/openagents#3285`: `CLOSED`
- `OpenAgentsInc/openagents#3269`: `CLOSED`
- `OpenAgentsInc/openagents#3262`: `CLOSED`
- `OpenAgentsInc/openagents#3661`: `CLOSED`
- `OpenAgentsInc/openagents#3662`: `CLOSED`

That live GitHub state matters because several repo docs and test fixtures still
point at the old Metal queue as the cluster block.

## Readiness Matrix

| Surface | Status | Current Truth |
| --- | --- | --- |
| Cluster identity, membership, transport, scheduling, and evidence | `implemented_early` | `psionic-cluster` has real trusted-LAN and configured-peer cluster substrate, planner refusals, replay-safe state, and validation runbooks |
| Homogeneous CUDA trusted-cluster training evidence | `implemented_early` | `docs/PSION_TRUSTED_CLUSTER_RUN.md` freezes one bounded four-node CUDA H100 trusted-cluster training lane with replay and checkpoint-restart receipts |
| Single-host Apple adapter training | `implemented_early` | `docs/TRAIN_SYSTEM.md` says the Apple adapter SFT and export lane is real on one host |
| Cluster-backed decentralized adapter control plane | `implemented_early` | `psionic-train` already owns contributor selection, window planning, worker heartbeats, upload receipts, validator dispositions, aggregation, and promotion receipts |
| Mixed Apple-plus-NVIDIA rehearsal path | `partial` | the runbook defines a truthful experimental mixed path, but only for cluster, artifact, replay, and validator truth, not one coherent Apple-valid training job |
| Multi-node Apple training execution | `partial` | the reusable substrate exists, but `docs/TRAIN_SYSTEM.md` explicitly says no real multi-node Apple training run is currently claimed |
| Distributed optimizer runtime | `partial` | the typed contract is real, but the reference doc explicitly says broad multi-device execution kernels and real transport-backed ZeRO or FSDP exchange do not exist yet |
| Heterogeneous same-job Mac-plus-NVIDIA execution | `planned` | no current repo claim covers one coherent mixed-backend training job with live gradient exchange across Apple and NVIDIA devices |
| Cross-backend local sharding | `planned` | `LocalShardingContract` still requires one backend family and only exposes caller-managed future collective posture plus metrics-only evidence |
| Cross-format Apple plus CUDA checkpoint or adapter interchange | `planned` | inferred from current code: Apple `.fmadapter` and CUDA open-adapter `safetensors` lanes exist, but there is no current shared optimizer-state or package-merge contract |

## What Is Actually Ready Now

### Cluster control and proof surfaces are real

The cluster substrate is no longer hypothetical.

`docs/CLUSTER_VALIDATION_RUNBOOK.md` treats these as current truthful scope:

- trusted-LAN connectivity
- configured-peer multi-subnet posture
- replay-safe ordered membership and recovery
- remote scheduling
- replicated serving
- layer-sharded and tensor-sharded evidence surfaces
- capability-profile publication and planner-refusal truth

That is real cluster machinery. It is not just a design memo.

`docs/ROADMAP_CLUSTER.md` also shows a large cluster issue queue already landed
on `main`. The cluster layer is one of the more mature control-plane parts of
the repo.

### The repo already has one real homogeneous distributed training proof

`docs/PSION_TRUSTED_CLUSTER_RUN.md` freezes one bounded four-node CUDA H100
trusted-cluster training lane.

That bundle binds:

- topology
- distributed-group truth
- distributed-optimizer truth
- replay truth
- checkpoint-restart truth

That matters because it proves the repo can already carry a truthful clustered
training story when the topology is narrow and homogeneous.

### Apple training is real, but only on one host

`docs/TRAIN_SYSTEM.md` is clear:

- single-host Apple adapter training is real
- the higher-level Apple SFT and export lane is real
- the reusable distributed substrate exists below it

The same doc is equally clear that the shipped Apple lane does not yet use the
broader distributed-training substrate as a live execution path.

That means Apple training is implemented.

It does not mean Apple clustered execution is implemented.

### Decentralized adapter cluster control is already strong

This is the strongest near-term lever for mixed hardware.

`docs/TRAIN_SYSTEM.md` marks these rows as `implemented_early`:

- decentralized adapter window contracts
- decentralized adapter cluster selection
- decentralized adapter worker protocol
- decentralized adapter artifact staging
- decentralized adapter validator and window scoring
- decentralized adapter aggregation and promotion

`crates/psionic-train/src/adapter_cluster.rs` shows that the coordinator already
mirrors cluster membership and telemetry into contributor eligibility,
selection, and churn-safe window replanning.

This means the repo already knows how to coordinate distributed contribution
work across a cluster.

## What Is Not Ready Now

### Multi-node Apple execution is still not a shipped truth claim

`docs/TRAIN_SYSTEM.md` explicitly says:

- no real `psionic-cluster` multi-node Apple training run is claimed
- no collective-backed gradient exchange or sharded optimizer execution is used
  by the shipped Apple backend
- no production multi-device training kernel or memory-sharded Apple trainer is
  claimed
- no broader cluster scheduler is yet dispatching Apple training windows across
  multiple machines

That is the core answer for the Mac side.

The control plane exists. The live multi-node Apple training execution path does
not.

### The current mixed Apple-plus-NVIDIA path is a rehearsal, not a live mixed trainer

`docs/ARCHITECTURE_EXPLAINER_CLUSTER_BRINGUP_RUNBOOK.md` is explicit that the
mixed Apple-plus-NVIDIA path is useful today for:

- cluster membership truth
- datastream staging
- replay identity
- validator receipts
- failure handling on mixed hardware

The same runbook is explicit that the mixed path is not yet good for:

- claiming NVIDIA is already contributing Apple-valid adapter gradients into the
  final Apple package
- claiming one Apple-plus-NVIDIA cluster can already train one Apple-valid
  adapter end to end

That is the honest current line.

### The distributed optimizer layer is still a contract layer, not a finished runtime

`docs/DISTRIBUTED_OPTIMIZER_REFERENCE.md` says the typed distributed optimizer
exists, but it does not yet implement:

- broad multi-device execution kernels
- real ZeRO or FSDP transport and partition exchange
- model-format import or export
- full replay guarantees

That is a major blocker for any claim of real mixed-device clustered execution.

### Local multi-device execution still assumes one backend family

`crates/psionic-runtime/src/local_multi_device.rs` hard-codes
`requires_same_backend_family: true` in `LocalShardingContract`.

The same contract also exposes:

- `LocalCollectivePosture::CallerManagedFuture`
- `LocalExecutionOutcomePosture::MetricsOnlyEvidence`

That is not enough for a truthful cross-backend execution story.

It means the current local sharding contract still assumes:

- one effective backend family for the participating devices
- collectives are still future or caller-managed
- the local runner only guarantees metrics or evidence, not a finished coherent
  multi-device execution result

You cannot stretch that into mixed Apple-plus-CUDA execution by flipping one
boolean.

### The only current non-Apple participant is still narrow

`crates/psionic-train/src/open_adapter.rs` names
`open_adapter_backend.cuda.gpt_oss_lm_head` as the first concrete NVIDIA/CUDA
participant in the mixed Apple-plus-NVIDIA experiment.

That lane is real, but it is narrow:

- it is one bounded open-adapter backend
- it targets LM-head LoRA `safetensors`
- it currently only admits `OpenAdapterPrecisionPolicy::F32Reference`
- `Bf16Mixed` is reserved and still rejected

So the current NVIDIA side of the heterogeneous story is present, but it is not
yet the wide high-throughput H100 training lane you would want for a serious
clustered mixed-backend trainer.

## The Metal Cluster Reality Is In An Inconsistent State

There is one important discrepancy in the repo right now.

Live GitHub says the old Metal queue `#3286 -> #3285 -> #3269 -> #3262` is
closed.

The repo still carries the old queue as the reason Metal remains blocked in
cluster fixtures and tests:

- `crates/psionic-cluster/tests/support/mod.rs`
- `crates/psionic-cluster/src/scheduler.rs`
- `crates/psionic-cluster/src/layer_sharded.rs`
- `crates/psionic-cluster/src/tensor_sharded.rs`

The practical meaning is:

- there is still no positive Metal cluster capability profile in the cluster
  test substrate
- the only repo-owned Metal cluster capability profile is the blocked one
- the current cluster acceptance story is still CUDA-first and homogeneous

So the old GitHub gate being closed does not mean mixed Mac-plus-NVIDIA cluster
training is suddenly ready.

It means the repo now has code and doc drift that must be resolved before any
truthful widening claim.

Until that drift is fixed, the safe answer is to trust the stricter repo-owned
execution and test posture, not the historical issue text.

## What This Means For "Mac Plus NVIDIA ASAP"

There are two different goals hidden inside that sentence.

### Goal A: mixed-hardware clustered training work is visibly happening

This goal is achievable sooner.

The repo is already close to this with the decentralized adapter substrate.

A truthful near-term result would be:

- one Apple coordinator host
- one or more Apple hosts for Apple-lane work when available
- one NVIDIA host for the bounded open-adapter lane
- shared cluster membership, staging, replay, validator, and promotion truth
- mixed-hardware receipts proving both hardware families participated in one
  operator-managed clustered training program

That is still real clustered training work across Mac and NVIDIA.

It is not one coherent mixed-backend collective.

### Goal B: one coherent synchronized mixed-backend training job

This goal is much farther away.

That version requires:

- Apple-side live multi-node execution
- mixed-backend mesh semantics
- collective or bridge semantics across Apple and CUDA
- optimizer-state and checkpoint interchange across backend families
- validation proving the mixed job is numerically and operationally sound

That is a different project from "get mixed hardware into the training program
ASAP."

## Fastest Honest Path

The fastest honest path is:

1. Treat decentralized adapter windows as the first real mixed-hardware
   clustered training product.
2. Keep Apple as the coordinator, `.fmadapter` export host, and runtime
   validation authority.
3. Use the NVIDIA open-adapter backend as the first non-Apple executor.
4. Make the operator flow, receipts, and validator surfaces first-class for
   mixed hardware.
5. Add one explicit mixed-hardware acceptance matrix for cluster planning,
   upload, replay, validator, and promotion truth.
6. Only after that, decide whether the product truly needs one coherent mixed
   Apple-plus-NVIDIA training graph.

This path reuses what the repo already has.

It does not wait for a perfect mixed-backend collective runtime before shipping
a real clustered training story across Mac and NVIDIA.

## What Must Land Before A True Mixed-Backend Training Job

### 1. A new mixed-backend training contract

The repo needs one explicit contract for what crosses the Apple/CUDA boundary.

That contract has to answer:

- are Apple and CUDA workers contributing gradients into one optimizer state
- are they contributing adapter deltas into one windowed aggregation step
- are they exchanging activations, gradients, or only validated artifacts
- which host owns final checkpoint authority
- which host owns final runtime validation for Apple-targeted artifacts
- what dtype and layout conversions are legal at the backend boundary
- what failure or replay semantics apply when only one backend family falls
  behind

Without this, "mixed training" remains a slogan instead of a machine-legible
runtime shape.

### 2. Positive Metal cluster capability profiles and acceptance tests

Right now the cluster substrate has CUDA profiles and blocked Metal profiles.

The repo needs explicit positive capability profiles for the Metal lanes it
actually supports in a cluster.

That also requires new tests and topology-acceptance coverage that prove:

- Metal cluster admission is explicit
- mixed Apple-plus-CUDA membership is explicit
- the selected execution lanes are explicit
- refused mixed combinations remain explicit

### 3. Multi-node Apple training execution

The Apple lane needs to become a real cluster execution participant, not just
the authority host for export and runtime validation.

That means:

- real multi-machine Apple execution for at least one bounded training lane
- explicit worker-side Apple execution receipts
- real checkpointing and replay across Apple executors
- no ambiguity about which Apple hosts contributed learning work

Without this, the Mac half of the mixed trainer is still mostly authority and
validation, not scale-out training compute.

### 4. Mixed-backend mesh and collective semantics

The current local sharding contract and distributed optimizer contract are not
enough for this.

The repo needs:

- backend-partition-aware mesh semantics instead of one effective backend family
- explicit bridge edges between Apple and CUDA partitions
- legal transport, buffer, dtype, and residency transitions across those edges
- real collective or bridge execution rather than caller-managed future work

This is where the current `requires_same_backend_family` assumption must be
replaced with something stronger and more explicit, not merely relaxed.

### 5. Cross-backend checkpoint and artifact interchange

This is an inferred gap from current code, but it is real.

Today the Apple lane and the first NVIDIA lane do not share one obvious final
artifact format:

- Apple lane: `.fmadapter`
- open NVIDIA lane: LM-head LoRA `safetensors`

The repo needs one explicit contract for:

- shared optimizer-state lineage
- shard layout lineage across backend families
- promotion semantics when multiple artifact families contribute
- final package or checkpoint authority for the mixed run

### 6. Mixed-hardware validation and regression gates

Current validation commands prove cluster truth and distributed-contract truth,
but not mixed Apple-plus-CUDA training truth.

The repo needs one runnable acceptance layer for:

- mixed-hardware cluster membership and admission
- mixed-hardware contributor selection
- mixed-hardware upload and replay
- mixed-hardware checkpoint restore
- mixed-hardware validator dispositions
- mixed-hardware promotion or explicit no-promotion outcomes
- refusal coverage for unsupported mixed-execution combinations

## Recommended Dependency Order

If the goal is "Mac plus NVIDIA clustered training ASAP," this is the right
implementation order.

### Phase 1: ship the truthful mixed-hardware clustered contribution lane

1. Freeze a mixed-hardware decentralized adapter contract.
2. Add a dedicated mixed Apple-plus-NVIDIA acceptance matrix for clustered
   adapter windows.
3. Make target-specific contributor capability policies explicit so mixed
   windows do not rely on the Apple default in `adapter_cluster.rs`.
4. Productize the first real Apple coordinator plus NVIDIA contributor runbook
   and fixtures.
5. Add operator-facing receipts that make mixed-hardware contribution state easy
   to inspect.

This phase gets you a real mixed-hardware clustered training program fastest.

### Phase 2: make Apple a real cluster executor

1. add positive Metal cluster capability profiles
2. add bounded multi-node Apple training execution
3. add Apple cluster checkpoint and replay receipts
4. extend topology acceptance and train validation around Apple clustered
   execution

This phase turns Mac from authority-only into real clustered training compute.

### Phase 3: build true coherent mixed-backend execution

1. define the mixed-backend mesh contract
2. replace same-backend-family sharding assumptions with explicit mixed-backend
   partition and bridge contracts
3. add real transport-backed cross-backend collective or bridge execution
4. define cross-backend optimizer and checkpoint authority
5. validate one bounded mixed-backend training lane end to end

This phase is the real closure step for one coherent Apple-plus-NVIDIA trainer.

## Proposed First Swarm Issue Spine

The issue spine for the first swarm run should copy the PGOLF discipline, not
the PGOLF workload.

That means:

- start with bring-up contracts and refusal-proof hardware reports
- freeze one narrow lane before asking for a real run
- add same-node parity and rehearsal evidence before the first real mixed run
- retain one full evidence bundle and one after-action audit after the first
  real run

It should not copy the full-model H100 benchmark shape.

The current codebase does not justify that for the Mac-plus-4080 target.

The repo facts that matter are:

- `#3647` already generalized decentralized adapter execution to one open
  adapter backend, but the only concrete backend label today is
  `open_adapter_backend.cuda.gpt_oss_lm_head`
- `#3661` and `#3662` already froze the truthful mixed Apple-plus-NVIDIA
  cluster posture as a mixed-role rehearsal path, not a symmetric live trainer
- `PMLX-706` / `#19` and `PMLX-707` / `#20` already give us MLX recipe and
  workflow planning, but `docs/MLX_RECIPE_PACKAGE.md` and
  `docs/MLX_WORKFLOW_PACKAGE.md` are explicit that these packages are not a
  second training runtime
- `PMLX-602` / `#3867` and `PMLX-603` / `#3868` only close bounded MLX Metal
  and CUDA backend slices; `docs/MLX_ACCEPTANCE_MATRIX.md` still keeps
  `backend-closure` at `planned`
- `docs/MLX_COMPATIBILITY_MATRIX.md` is explicit that the public MLX
  distributed surface is bounded and reference-emulated, not proof of a live
  vendor-transport-backed trainer
- local repo evidence already shows the Linux desktop RTX 4080 is a real useful
  CUDA validation and profiling node, even though it is not challenge-matching
  H100 hardware

That means the first honest swarm lane is:

- one trusted-LAN two-node clustered run
- one Mac node through a new MLX-backed open-adapter execution backend
- one Linux desktop node through the existing CUDA open-adapter lane
- one shared `gpt_oss.decoder_lm_head_lora` adapter family
- one shared decentralized adapter window and contribution flow
- not one full-model all-reduce or FSDP claim

These issue titles are proposed backlog items. They are not filed on GitHub
yet.

| Proposed issue | Why it exists | Depends on |
| --- | --- | --- |
| `SWARM-0: Ship the first local mixed-hardware swarm training run on one Mac MLX node plus one Linux RTX 4080 node` | Master task for the whole local swarm lane. Done only when one real two-node run, one retained evidence bundle, and one audit exist. | none |
| `SWARM-1: Freeze the first swarm run contract as one decentralized open-adapter delta lane` | The first swarm run needs one honest lane definition before any execution work. It should freeze dataset identity, tokenizer digest, adapter family `gpt_oss.decoder_lm_head_lora`, cluster namespace, acceptance gate, replay posture, and explicit non-goals. | `SWARM-0` |
| `SWARM-2: Add a Mac MLX swarm bring-up report and refusal-proof machine contract` | This should be the Mac equivalent of the PGOLF bring-up report: what exact MLX-capable Metal hardware exists, what backend slice is actually admitted, what run geometry is safe, and where the current lane still refuses. | `SWARM-1` |
| `SWARM-3: Add a Linux RTX 4080 swarm bring-up report and same-node parity harness` | The 4080 needs a swarm-specific report, not only PGOLF-specific receipts. This should freeze the exact CUDA inventory, admitted precision posture, local batch limits, and current same-node execution evidence for the chosen open-adapter lane. | `SWARM-1` |
| `SWARM-4: Implement the first MLX-backed open-adapter execution backend on Metal` | Today the MLX lane has planning and bounded backend execution, but no live train backend for the open-adapter family. This issue should add a concrete backend label such as `open_adapter_backend.mlx.metal.gpt_oss_lm_head` and make the Mac node a real contributor instead of only a planner. | `SWARM-2` |
| `SWARM-5: Make open-adapter manifests, receipts, and precision policy comparable across MLX/Metal and CUDA` | The swarm run needs one shared artifact and receipt language across both nodes. This should freeze comparable backend-tagged receipts, supported dtypes, quantization posture, hidden-state shape assumptions, and replay identity across the Mac MLX and Linux CUDA backends. | `SWARM-3`, `SWARM-4` |
| `SWARM-6: Bind MLX recipe and workflow plans into live adapter-cluster contributor selection and window planning` | The current MLX packages stop at planning. The first swarm run needs those plans wired into `adapter_cluster.rs`, contributor capability policy, assignment seeds, dataset-slice plans, and upload expectations without inventing a parallel trainer. | `SWARM-1`, `SWARM-4`, `SWARM-5` |
| `SWARM-7: Add the first trusted-LAN Mac-plus-4080 swarm topology contract, launch scripts, and failure drills` | This is the cluster bring-up issue for the exact local hardware pair. It should freeze node roles, network assumptions, artifact staging, replay identity, stale-worker handling, and the exact launch sequence for the two-node swarm. | `SWARM-3`, `SWARM-6` |
| `SWARM-8: Run the first simulated mixed-hardware swarm rehearsal and bottleneck report` | We already know from `#3660` that a rehearsal stage is useful. This issue should produce a measured dry run that separates cluster-control truth from optimistic speedup stories and surfaces the remaining serial phases before the first live attempt. | `SWARM-7` |
| `SWARM-9: Run and retain the first real two-node Mac-MLX plus RTX-4080 swarm training evidence bundle` | This is the first real run. It should retain contributor receipts, upload manifests, validator dispositions, promotion or no-promotion truth, timing, replay receipts, and cluster topology evidence in one committed bundle. | `SWARM-8` |
| `SWARM-10: Merge the accepted swarm artifacts, publish the local snapshot, and write the after-action audit` | The first swarm run is not done at “the processes exited.” This issue should verify merge or no-merge truth, export the resulting local snapshot through the existing MLX workflow or model-IO surface where valid, and publish one audit that says exactly what the run proved and what it still did not prove. | `SWARM-9` |

The right execution order is:

1. `SWARM-1`
2. `SWARM-2`
3. `SWARM-3`
4. `SWARM-4`
5. `SWARM-5`
6. `SWARM-6`
7. `SWARM-7`
8. `SWARM-8`
9. `SWARM-9`
10. `SWARM-10`

This keeps the first swarm run honest in three ways:

- it uses the existing decentralized adapter cluster substrate instead of
  overclaiming the current MLX distributed helpers
- it makes the Mac node a real MLX-backed contributor only after a backend
  exists for that lane
- it treats the Linux RTX 4080 as a truthful local CUDA worker with its own
  explicit contract, not as a fake H100 stand-in

## Current Validation That Already Exists

The repo already has useful gates for the parts that are real today:

- `cargo test -p psionic-cluster --test local_cluster_transport`
- `cargo test -p psionic-cluster --test cluster_validation_matrix`
- `scripts/release/check-psionic-topology-acceptance-matrix.sh --cluster-only`
- `scripts/release/check-psionic-distributed-optimizer-contracts.sh`

These are valuable, but they are not enough to claim mixed Apple-plus-NVIDIA
clustered training.

They need mixed-hardware train-specific follow-ons.

## Bottom Line

`psionic` is already strong enough to support an honest clustered training push.

It is not yet strong enough to honestly claim live mixed Mac-plus-NVIDIA
clustered execution for one coherent training job.

The shortest route to real progress is to use the decentralized adapter cluster
substrate as the first mixed-hardware training product and make that path
operator-solid first.

If you want true mixed Apple-plus-CUDA collective-backed training after that,
the repo still needs explicit work in:

- Apple multi-node execution
- mixed-backend mesh semantics
- cross-backend optimizer and checkpoint contracts
- mixed-hardware validation coverage

That is the honest readiness answer on 2026-03-24.

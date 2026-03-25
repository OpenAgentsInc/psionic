# 2026-03-25 Cross-Provider Pretraining System Readiness Audit

This audit answers one concrete question:

- how the new Google two-node swarm lane compares to the rest of the `psionic`
  training stack
- how close `psionic` is to one training system that works across any compute
  source
- what still has to land before Google, RunPod, local NVIDIA, local Apple, and
  later providers can all contribute truthfully to one real pretraining program

## Executive Verdict

`psionic` is now strong on training control-plane truth.

It is not yet strong enough to claim one provider-neutral pretraining system
where arbitrary compute from Google and elsewhere can all participate in one
real full-model pretraining run under the same train-step semantics.

The current state is:

- cluster identity, topology, configured-peer manifests, evidence bundles, and
  failure drills are `implemented_early`
- train run graph, orchestrator state, worker protocol, replay truth, and
  checkpoint lineage are `implemented_early`
- single-node accelerated training is real on Google
- bounded multi-node swarm contribution is real on Google
- bounded mixed-hardware swarm contribution is partially real on Mac plus RTX
  4080 hardware, but the master issue still stays open because the final
  accepted live result bar is not yet met
- the strongest distributed full-model lane, Parameter Golf `8xH100`, still
  stops at a real Rust-native bootstrap boundary and does not yet own the full
  distributed train-step, validation, export, and evidence closure
- mixed-backend dense training across Apple and CUDA is still `planned`

The fastest honest path to "many compute sources contribute to one real
pretraining program" is not one immediate mixed-backend synchronous trainer.

The fastest honest path is:

1. finish one real homogeneous distributed CUDA training runtime and reuse it
   across providers
2. unify provider launch, artifact, and telemetry contracts around the same run
   graph and runtime surfaces
3. let heterogeneous hardware participate under the same pretraining program
   through explicit contribution classes that match its actual capability
4. attempt mixed-backend dense training only after the homogeneous distributed
   trainer, model IO, checkpoint, and conformance layers are already stable

## Sources Consulted

Canonical docs:

- `docs/TRAIN_SYSTEM.md`
- `docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md`
- `docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md`
- `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md`
- `docs/PSION_PILOT_PRETRAINING_RUN.md`
- `docs/PSION_PRETRAIN_STAGE.md`
- `docs/PARAMETER_GOLF_RUNPOD_8XH100_RUNBOOK.md`
- `docs/PARAMETER_GOLF_DISTRIBUTED_8XH100.md`
- `docs/CLUSTER_VALIDATION_RUNBOOK.md`
- `docs/ROADMAP_CLUSTER.md`
- `docs/TRAIN_RUN_GRAPH_REFERENCE.md`
- `docs/TRAIN_ORCHESTRATOR_REFERENCE.md`
- `docs/DISTRIBUTED_OPTIMIZER_REFERENCE.md`
- `docs/DISTRIBUTED_DATA_FEED_SEMANTICS.md`
- `docs/TRAIN_CHECKPOINT_RECOVERY_REFERENCE.md`
- `docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md`
- `docs/MODEL_IO_REFERENCE.md`
- `docs/REMOTE_TRAINING_VISUALIZATION.md`
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md`

Relevant audits:

- `docs/audits/2026-03-22-psion-training-system-full-state-audit.md`
- `docs/audits/2026-03-24-clustered-training-mac-nvidia-readiness-audit.md`
- `docs/audits/2026-03-24-google-two-node-swarm-compute-audit.md`
- `docs/audits/2026-03-24-psionic-parameter-golf-validation-runtime-audit.md`
- `docs/audits/2026-03-25-google-two-node-swarm-first-real-run-audit.md`
- `docs/audits/2026-03-25-psionic-burn-port-audit.md`

Relevant code:

- `crates/psionic-train/src/run_graph.rs`
- `crates/psionic-train/src/orchestrator.rs`
- `crates/psionic-train/src/adapter_cluster.rs`
- `crates/psionic-train/src/open_adapter.rs`
- `crates/psionic-train/src/swarm_open_adapter.rs`
- `crates/psionic-train/src/psion_google_two_node_swarm_runtime.rs`
- `crates/psionic-train/src/psion_trusted_cluster_run.rs`
- `crates/psionic-train/src/distributed_optimizer.rs`
- `crates/psionic-train/src/model_io.rs`
- `crates/psionic-train/src/parameter_golf_distributed_8xh100_runtime_bootstrap.rs`
- `crates/psionic-train/src/parameter_golf_submission_runtime.rs`
- `crates/psionic-train/src/remote_training_visualization.rs`

Live GitHub issue state checked with `gh` on 2026-03-25:

- `#473` is open for real Rust-native distributed Parameter Golf execution
- `#466` is open for the remaining `train_gpt.py` parity gaps
- `#479` is open because the remote-visualization substrate is still not fully
  live on the RunPod distributed lane
- `#484` is open because the mixed Mac-plus-RTX 4080 swarm master bar still
  requires one truthful accepted live result
- `#514`, `#515`, and `#516` are open for backend conformance, async
  checkpoint writeback, and shared local metric sinks
- the Google two-node swarm issue stack `#501` through `#508` is closed

## Comparative State

| Lane or Subsystem | Status | What Is Real | Current Limit |
| --- | --- | --- | --- |
| Psion single-node accelerated Google run | `implemented_early` | one real accelerator-backed single-node training lane with retained Google evidence | single-node only |
| Google two-node swarm | `implemented_early` | one real two-node configured-peer Google adapter-delta lane with real `bounded_success` runs and retained evidence | not full-model training, not elastic, not public swarm |
| Local Mac plus RTX 4080 swarm | `partial` | one truthful mixed-hardware decentralized adapter stack with comparable receipts, bring-up, rehearsal, evidence, and closeout surfaces | master issue still open because the lane has not yet earned the final accepted live result bar |
| Trusted-cluster training | `implemented_early` | one bounded homogeneous CUDA trusted-cluster training claim with topology and distributed-group receipts | narrow homogeneous cluster claim only |
| Parameter Golf single-H100 | `implemented_early` | one real remote CUDA training lane with provider-neutral visualization support | single host |
| Parameter Golf RunPod `8xH100` | `partial` | one real Rust-native distributed bootstrap and explicit operator/finalizer lane | full distributed train-step, validation, export, and evidence closure still open |
| Train run graph and orchestrator | `implemented_early` | typed run identity, contributor revisions, window state, assignment posture, and trainer-batch control | not yet the universal runtime entrypoint for all training lanes |
| Distributed optimizer contract | `partial` | typed optimizer, memory, and precision contracts with bounded public helper surfaces | no broad real transport-backed distributed trainer runtime yet |
| Model IO portability | `implemented_early` | typed portable import/export, selective import, remap, and deferred materialization | not yet full distributed checkpoint interchange and shard recovery closure |
| Remote training visualization | `implemented_early` | one provider-neutral app-facing bundle and run-index family | distributed RunPod lane still lacks the live writer required to close `#479` |

## What The Google Two-Node Swarm Lane Actually Adds

The new Google two-node swarm lane is important because it proves that
`psionic` can already do all of the following on real cloud hardware:

- own the network, identity, quota, launch, startup, impairment, finalizer, and
  evidence surfaces for a multi-node lane
- keep the lane provider-managed without moving training truth into provider
  logs or ad hoc operator notes
- reuse the existing generic adapter-cluster, worker-protocol, validation, and
  aggregation substrate instead of inventing a second Google-specific training
  control plane
- survive real cloud timing skew and mild network impairment with explicit
  receipts

That closes one important infrastructure question.

It does not close the main math question.

The Google swarm lane is one bounded adapter-delta contribution program. It
proves that multi-node cross-host contribution control is real on Google. It
does not prove that `psionic` can already run one full-model multi-rank
pretraining job across those nodes.

## How It Relates To The Other Training Lanes

### Google single-node Psion

This is the strongest proof that `psionic` can run one real accelerator-backed
full-model training lane under a provider-managed operator surface.

It gives the system:

- real single-node training math
- real checkpoint and evidence retention
- real Google operator scripts

It does not give the system:

- multi-node train-step semantics
- multi-rank validation
- provider-neutral multi-host launch semantics

### Google two-node swarm

This is the strongest proof that `psionic` can run one real cloud multi-node
training control path with launch, runtime, impairment, and finalization truth.

It gives the system:

- real configured-peer cloud cluster bring-up
- real multi-node runtime coordination
- real provider-owned evidence bundle retention

It does not give the system:

- full-model distributed optimizer execution
- full-model distributed validation
- a generic provider-neutral launcher shared with RunPod or local lanes

### Local mixed-hardware swarm

This is the strongest proof that `psionic` already understands that different
compute sources need different admitted execution roles.

It gives the system:

- one shared decentralized adapter contract across MLX/Metal and CUDA
- comparable backend-tagged contributor receipts
- a real mixed-hardware bring-up surface

It does not give the system:

- accepted same-job dense training across MLX and CUDA
- one truthful mixed-backend all-reduce path

### Trusted-cluster training

This is the strongest proof that the repo already has a truthful homogeneous
cluster-training claim with explicit distributed-group and checkpoint-restart
receipts.

It gives the system:

- a bounded homogeneous distributed-training proof
- a clear contract for trusted topology and replay-safe recovery

It does not give the system:

- cross-provider launch portability
- heterogeneous hardware participation
- a broad elastic training mesh

### Parameter Golf

Parameter Golf is the strongest forcing function for the actual missing runtime.

It gives the system:

- the strongest current remote CUDA operator surfaces
- the sharpest parity bar against a public distributed baseline
- a real Rust-native distributed bootstrap on `8xH100`

It does not yet give the system:

- the real distributed train-step closure
- distributed validation sharding and aggregation closure
- final export and evidence closure from real distributed execution

This matters because the universal training system should reuse the same
distributed runtime that Parameter Golf is already forcing into existence.

## What Is Already Universal Across Compute Sources

Several important layers are already shaped correctly for a compute-source-
agnostic system.

### Control-plane truth is ahead of runtime math

`run_graph.rs`, `orchestrator.rs`, `adapter_cluster.rs`, and the worker
protocol surfaces already make these things explicit:

- stable run identity
- stable participant identity
- contributor selection and window planning
- assignment posture
- heartbeats, stale-worker handling, and departure state
- validator and aggregation transitions

That is the right shared control-plane shape for local, Google, RunPod, and
later providers.

### Machine-legible evidence is already a first-class requirement

The Google single-node, Google swarm, trusted-cluster, Parameter Golf, and
mixed-hardware swarm lanes all keep receipts, manifests, evidence bundles,
finalizers, and refusal posture explicit.

That matters because a cross-provider training system fails if each provider
needs its own narrative-only success story.

### Model-state portability is finally getting real

`model_io.rs` and `docs/MODEL_IO_REFERENCE.md` now give `psionic`:

- portable state dict ownership
- adapter merge and unmerge
- selective import
- key remap
- deferred materialization

That is necessary for a system that moves state between providers, runtimes,
and machines with different memory budgets.

### Remote visualization already has the right authority split

`remote_training_visualization.rs` keeps Psionic responsible for machine-facing
truth while app surfaces consume one provider-neutral bundle family.

That same pattern should apply to the broader provider-neutral training system:
Psionic should emit one training truth family; providers should only satisfy
resource and transport contracts.

## What Is Still Missing For One Real Cross-Provider Pretraining System

### 1. One universal dense distributed training runtime

This is the largest missing piece.

`docs/DISTRIBUTED_OPTIMIZER_REFERENCE.md` is explicit that the distributed
optimizer layer is still a typed contract, not a completed broad multi-device
runtime. Parameter Golf `8xH100` now has a real Rust-native bootstrap path, but
`#473` is still open because the real distributed train-step still is not done.

Until this lands, `psionic` does not have one reusable full-model distributed
trainer that Google, RunPod, and later providers can all share.

### 2. One provider-neutral compute-source contract

The current provider lanes are still too script-shaped and provider-specific.

Google single-node, Google swarm, and RunPod `8xH100` each have good launch and
finalizer surfaces. They do not yet fold into one shared provider-neutral
contract that answers:

- what a compute source must report before admission
- how a compute source publishes network, storage, accelerator, and cost truth
- how a training program binds artifact roots, launch receipts, and teardown
- how the same run graph maps to local, Google, RunPod, or later providers

The cluster layer already has much of the topology and identity substrate.
What is still missing is the training-facing compute-source contract on top of
that substrate.

### 3. One explicit role taxonomy for heterogeneous contributors

If the goal is "any compute source can contribute," the system needs to stop
pretending every node will be a dense synchronous rank.

The current repo already hints at the correct split:

- homogeneous CUDA nodes can eventually be dense full-model ranks
- Apple MLX and weaker NVIDIA nodes can already participate honestly in bounded
  adapter windows
- validators, checkpoint writers, data builders, and eval workers can also be
  first-class run roles

That role split is not yet frozen as one universal pretraining-program
contract.

Without that contract, the system keeps oscillating between two bad options:

- overclaiming mixed-backend dense training that does not exist
- underusing heterogeneous compute that could still contribute honestly

### 4. Distributed data semantics that survive real multi-provider runs

`docs/DISTRIBUTED_DATA_FEED_SEMANTICS.md` is still bounded to fixed-world-size
seeded distributed feeds. That is not enough for:

- cross-provider contributor churn
- mixed dense-rank plus contributor-window programs
- long-running pretraining programs that reassign work across machine classes

The data plane needs a stronger contract for:

- shard ownership across provider boundaries
- deterministic re-assignment after node loss
- dense-rank versus contributor-window sampling semantics
- stable data receipts that remain comparable across execution classes

### 5. Distributed checkpoint and recovery closure

`docs/TRAIN_CHECKPOINT_RECOVERY_REFERENCE.md` still does not claim distributed
optimizer recovery or parameter-shard semantics. `#515` is still open for async
checkpoint writeback. `docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md` also still does
not claim a real external blob-store client or remote placement strategy.

That blocks a serious cross-provider pretraining program because those runs need
all of the following:

- frequent checkpoint flush without stalling the train loop
- resumable sharded checkpoint uploads
- explicit durability and restore semantics across providers
- the ability to recover a dense rank mesh after preemption or provider loss

### 6. Shared backend conformance and portability discipline

`#514` is still open. That matters directly.

The training system cannot "work the same way across any compute source" unless
the backend family beneath it can prove:

- what is supported
- what is unsupported
- what is correct
- what only works on one backend today

The Mac-plus-CUDA swarm lane already uses comparable receipts. That is not yet
the same thing as one shared correctness harness across CPU, CUDA, Metal, and
MLX-backed lanes.

### 7. Shared train-loop metric and live telemetry plumbing

`#516` is still open. `#479` is still open.

The app-facing remote visualization contract is good. The remaining gap is that
not every runtime writes a consistent local metric stream and not every remote
lane can yet keep the live bundle fresh every second.

That becomes more important, not less, when one pretraining program spans many
providers and many role classes.

## Estimated Distance To The Goal

These percentages are audit estimates, not machine-derived metrics.

| Goal | Estimated Readiness | Why |
| --- | --- | --- |
| Shared control plane, receipts, and evidence model across providers | `65%` | run graph, orchestrator, cluster manifests, evidence bundles, model IO, and visualization contracts are real; provider launch and dense-runtime entrypoints are still fragmented |
| Real homogeneous CUDA full-model training across multiple providers | `35%` | single-node Google and distributed bootstrap work are real, but the actual distributed train-step and validation closure remain open in `#473`, `#510`, `#511`, and `#512` |
| One pretraining program with multiple admitted contribution classes across mixed hardware | `45%` | decentralized adapter windows and mixed-hardware receipts are real, but the role taxonomy and shared program contract are not yet frozen end to end |
| True mixed-backend dense training across Apple and CUDA | `10%` | the repo has strong bring-up, comparable receipts, and bounded mixed-hardware rehearsals, but no truthful mixed dense train-step runtime exists |
| One system that "works the same way across any compute source" | `25%` | the foundations are good, but the universal runtime, role model, distributed data, and checkpoint semantics are still missing |

## What The Full Implementation Should Look Like

The full implementation should not be "some scripts for Google, some scripts
for RunPod, and a separate mixed-hardware experiment."

It should be one training system with these stable layers.

### Layer 1: Provider-neutral training program contract

The root object should bind:

- run identity
- stage identity
- checkpoint family
- dataset family
- admitted execution classes
- artifact roots
- cost and time budgets
- finalizer and evidence policy

This should sit above provider launch details and below app UX.

### Layer 2: Provider-neutral compute-source contract

Every compute source should publish the same training-facing identity:

- provider and region or locality
- accelerator family and count
- backend family
- precision support
- network posture
- storage posture
- cost ceiling and billing unit
- checkpoint and artifact bandwidth expectations

This should let the same planner admit:

- a Google `g2` plus `L4` node
- a RunPod `8xH100` pod
- a local RTX 4080 desktop
- a local MLX-capable Mac

without inventing a separate training truth surface for each one.

### Layer 3: Explicit execution classes

The training program should admit several role kinds and keep them separate:

- `dense_full_model_rank`
- `validated_contributor_window`
- `validator`
- `checkpoint_writer`
- `eval_worker`
- `data_builder`

That is how "many compute sources contribute" becomes truthful.

The dense full-model ranks should run the same distributed runtime. The weaker
or incompatible nodes should still contribute to the same pretraining program
through admitted validated work units, not fake dense-rank membership.

### Layer 4: One real distributed CUDA runtime reused everywhere

This is the central runtime target:

- same train-step semantics
- same validation semantics
- same checkpoint semantics
- same metric semantics
- same evidence semantics

The provider should change. The runtime contract should not.

This is why Parameter Golf matters so much. The work in `#473` and its
dependencies should become the canonical dense distributed runtime for the
broader train system, not a contest-only side lane.

### Layer 5: Shared checkpoint and model-state mobility

Checkpoint and model IO need to support:

- sharded distributed checkpoints
- resumable remote writeback
- restore across provider boundaries
- partial and remapped import for heterogeneous follow-on roles
- explicit optimizer-state portability boundaries

The new model-IO layer is a real start. It is not the whole answer.

### Layer 6: Shared telemetry, finalization, and app surfaces

Every lane should emit:

- one local metric stream
- one remote visualization bundle family
- one final evidence bundle family
- one refusal classification vocabulary

The app should not care whether the run came from Google, RunPod, or local
cluster hardware.

## Recommended Expansion Sequence

### Phase 1: Finish the real dense distributed runtime

Do this first.

Primary existing issue stack:

- `#473`
- `#466`
- `#510`
- `#511`
- `#512`

Reason:

- without this, the system still lacks the main reusable multi-rank train-step
  engine

### Phase 2: Promote that runtime from PGOLF lane to train-system substrate

After the distributed CUDA runtime is real, bind it into the broader pretrain
system instead of leaving it isolated behind Parameter Golf wrappers.

This phase should freeze:

- one provider-neutral dense-rank runtime entrypoint
- one shared finalizer and evidence family
- one shared step, validation, and checkpoint metric vocabulary

This work is only partially covered by the current issue stack. It likely needs
new issues.

### Phase 3: Freeze the compute-source and execution-class contract

Land one canonical training-facing compute-source contract and one canonical
role taxonomy.

This should cover:

- local versus rented versus cloud-managed sources
- dense ranks versus contributor windows
- validator and auxiliary roles

This also likely needs new issues.

### Phase 4: Unify Google, RunPod, and local launch paths around that contract

Keep provider-specific resource creation where it belongs. Move training truth
into shared launch manifests, startup contracts, runtime env contracts, and
finalizer contracts.

The Google swarm work already shows the right pattern. The RunPod and
single-node Google paths should converge toward it.

### Phase 5: Close checkpoint, storage, and metric gaps

Primary existing issues:

- `#515`
- `#516`
- `#479`
- `#514`

Additional work still needs to be defined for:

- distributed checkpoint shard manifests
- provider-neutral remote artifact placement and restore strategy
- dense-rank recovery after provider or node loss

### Phase 6: Expand heterogeneous contribution under the same program

Keep this separate from dense mixed-backend math.

Primary existing issue:

- `#484`

Broader follow-on should:

- generalize the mixed-hardware swarm role model
- bind the contributor-window path to a named pretraining-program contract
- keep accepted contribution, replay, and promotion rules explicit

### Phase 7: Attempt mixed-backend dense training only if still needed

Do this last.

The repo does not need mixed Apple-plus-CUDA dense all-reduce to make many
compute sources contribute to one pretraining program. It only needs that if
product or research goals require the exact same dense train-step graph to span
those backends.

## Comprehensive Phase-By-Phase Issue Program

The phase summary above is not enough to drive the work. The repo needs one
explicit issue spine that covers the whole implementation surface.

Some of that spine already exists as open GitHub issues. The rest is still
missing from the backlog. This section treats both the same way: each item has
one name, one short description, and one filing-ready issue body.

The `XTRAIN-*` labels below are proposed new issues that do not exist in GitHub
yet.

Unless an item says otherwise, the canonical inputs are:

- this audit
- `docs/TRAIN_SYSTEM.md`
- the subsystem reference docs cited in the relevant phase
- the runtime modules and runbooks named earlier in this audit

### Phase 0: Program Contract

#### `XTRAIN-0: Ship The Provider-Neutral Cross-Provider Pretraining System`

Description: master issue for the full training-system program.

Issue body:

- Summary: land one Psionic-native training system that can admit local,
  Google, RunPod, and later providers under one run graph, one checkpoint
  family, one evidence model, and one execution-class contract.
- Why: the repo has strong bounded lanes but no single master issue whose done
  bar is the full cross-provider system.
- Depends On: every issue in phases 0 through 7.
- Scope: define the master acceptance bar, link the whole child stack, and
  keep the final proof runs and final audit explicit.
- Acceptance Criteria: every child issue is closed, one real multi-provider
  dense CUDA pretraining run exists with retained evidence, one real
  mixed-execution-class pretraining program exists, and if same-job MLX plus
  CUDA dense training remains required the mixed-backend proof issue is also
  closed.

#### `XTRAIN-1: Freeze The Cross-Provider Training Program Manifest And Run Authority`

Description: canonical root contract for one cross-provider pretraining program.

Issue body:

- Summary: add one typed training-program manifest that binds run id, stage,
  checkpoint family, dataset family, artifact roots, cost budget, and admitted
  execution classes.
- Why: the current system has strong run graph and stage contracts, but there
  is still no single top-level manifest that spans dense ranks, contributors,
  validators, and provider-managed launch surfaces.
- Depends On: none.
- Scope: add the manifest type, fixture, checker, and documentation; bind it to
  `TrainingRunState`, `PSION_PRETRAIN_STAGE`, and final evidence surfaces.
- Acceptance Criteria: every multi-provider lane can point at one canonical
  program manifest instead of provider-specific launch manifests as the root
  training authority.

#### `XTRAIN-2: Freeze The Compute-Source And Execution-Class Admission Contract`

Description: canonical training-facing identity for machines and roles.

Issue body:

- Summary: add one typed compute-source contract that reports provider,
  locality, accelerator inventory, backend family, network posture, storage
  posture, cost posture, and admitted execution classes.
- Why: the current lanes each publish some of this, but they do it through
  different Google, RunPod, swarm, and bring-up artifacts.
- Depends On: `XTRAIN-1`.
- Scope: define the machine contract for local, Google, RunPod, and later
  providers; freeze the execution classes `dense_full_model_rank`,
  `validated_contributor_window`, `validator`, `checkpoint_writer`,
  `eval_worker`, and `data_builder`.
- Acceptance Criteria: the same planner can admit a Google node, a RunPod pod,
  a local RTX host, and a local Mac through one machine-legible contract.

#### `XTRAIN-3: Add The Provider-Neutral Launch, Startup, Runtime-Env, And Finalizer Contract`

Description: one shared runtime envelope above provider-specific resource
creation.

Issue body:

- Summary: define one launch contract that every provider binder must project
  into concrete startup scripts, runtime environment variables, artifact roots,
  and finalizer hooks.
- Why: the current operator paths are good but fragmented across Google
  single-node, Google swarm, RunPod, and local launchers.
- Depends On: `XTRAIN-1`, `XTRAIN-2`.
- Scope: freeze one shared manifest for runtime env, cluster ports, checkpoint
  roots, metric roots, evidence roots, and finalizer expectations.
- Acceptance Criteria: provider-specific scripts only create and bind resources;
  they no longer define training truth or lane-specific runtime semantics.

### Phase 1: Dense Distributed Runtime

#### `#466 PGOLF_PARITY-0: Close The Remaining train_gpt.py Parity Gaps In Psionic`

Description: existing umbrella issue for the remaining public-baseline parity
stack.

Issue body:

- Summary: keep one explicit Rust-versus-`train_gpt.py` parity stack so the
  distributed runtime work stays measurable.
- Why: the repo still has several lower-level gaps that block an honest parity
  claim even after the operator and bootstrap surfaces landed.
- Depends On: `#473`, `#510`, `#511`, `#512`, and the rest of the current PGOLF
  parity stack.
- Scope: preserve one measurable parity backlog instead of letting the missing
  work disappear into broader training issues.
- Acceptance Criteria: the final readiness audit cites no remaining
  `train_gpt.py` parity blocker for the distributed baseline lane.

#### `#473 PGOLF_PARITY-7: Replace The Analytic 8xH100 Lane With Real Rust-Native Distributed Execution`

Description: existing master runtime issue for real multi-rank Parameter Golf
execution.

Issue body:

- Summary: replace the current analytic `8xH100` lane with one real Rust-native
  distributed runtime under the public `WORLD_SIZE=8` posture.
- Why: without a real runtime, the strongest distributed lane in the repo still
  stops before the actual train step and final metric closure.
- Depends On: the current closed bootstrap prerequisites plus `#510`, `#511`,
  and `#512`.
- Scope: make the `8xH100` lane execution-backed for train-step, validation,
  receipt generation, and final evidence.
- Acceptance Criteria: the distributed lane no longer relies on analytic or
  measurements-missing closure for steady-state success.

#### `#511 PGOLF_PARITY-14: Implement Real Rust-Native Distributed PGOLF Train-Step Execution`

Description: existing issue for the actual multi-rank train step.

Issue body:

- Summary: implement real per-rank batches, forward or backward passes,
  gradient synchronization, and optimizer updates under the exact distributed
  PGOLF geometry.
- Why: bootstrap is not enough; the main missing boundary is the actual
  multi-rank train loop.
- Depends On: the current bootstrap and train-step prerequisites already named
  on the live issue.
- Scope: land real train-step execution with measured step, communication, and
  memory observations.
- Acceptance Criteria: the distributed receipt and runtime logs are fed by real
  multi-rank execution rather than placeholders.

#### `#510 PGOLF_PARITY-15: Implement Real Distributed Validation And Metric Aggregation For PGOLF 8xH100`

Description: existing issue for distributed validation closure.

Issue body:

- Summary: shard validation sequences across ranks and emit one aggregated final
  metric result for the whole run.
- Why: a real distributed training system cannot close on single-rank
  validation or narrative-only aggregation.
- Depends On: the bootstrap, train-step, and validation prerequisites already
  named on the live issue.
- Scope: add rank-local validation facts, aggregation, and one execution-backed
  final validation result.
- Acceptance Criteria: the distributed receipt, finalizer, and exported-folder
  evidence all preserve the same aggregated distributed validation result.

#### `#512 PGOLF_PARITY-16: Close The Exported-Folder Distributed Runtime Outcome And Receipt Path`

Description: existing issue for turning the distributed mode into a real success
path instead of a default refusal path.

Issue body:

- Summary: replace the current refusal-only `distributed_8xh100_train` outcome
  with one real completion path that emits final receipts and final artifacts.
- Why: even after bootstrap and train-step work exist, the shipped runtime and
  finalizer still need to stop treating distributed success as unsupported.
- Depends On: `#510` and `#511`.
- Scope: switch the exported-folder runtime, finalizer, and submission evidence
  path from refusal semantics to real success semantics when the lane succeeds.
- Acceptance Criteria: the shipped runtime can finish the distributed lane with
  real final receipts and no silent fallback to local replay.

#### `XTRAIN-4: Promote The PGOLF Distributed Runtime Into One Generic Dense-Rank Engine In psionic-train`

Description: turn the PGOLF-forced runtime into shared train-system substrate.

Issue body:

- Summary: extract the real distributed CUDA train-step, validation, and
  receipt surfaces from the PGOLF-specific lane into one reusable dense-rank
  engine inside `psionic-train`.
- Why: the broader training system should not depend on a contest-only runtime
  if the goal is one provider-neutral pretraining substrate.
- Depends On: `#473`, `#510`, `#511`, `#512`.
- Scope: move common dense-rank runtime types and hooks into generic
  train-system modules; keep PGOLF as one consumer instead of the only owner.
- Acceptance Criteria: a non-PGOLF pretraining lane can call the same generic
  dense-rank runtime without inheriting contest-specific wrappers.

### Phase 2: Data, Checkpoint, And Model-State Mobility

#### `XTRAIN-5: Extend Distributed Data-Feed Semantics To Topology-Revisable Cross-Provider Pretraining`

Description: move the data plane past fixed-world-size seeded feeds.

Issue body:

- Summary: extend `psionic-data` from fixed-world-size distributed feeds to a
  contract that can survive admitted topology revision, dense-rank replacement,
  and cross-provider host loss.
- Why: the current data-feed contract explicitly refuses elastic or rebalance-
  aware partitioning.
- Depends On: `XTRAIN-1`, `XTRAIN-2`, `XTRAIN-4`.
- Scope: define topology-aware shard ownership, re-assignment, replay-safe
  ordering, and data receipts for dense-rank runs.
- Acceptance Criteria: a dense-rank run can lose or replace a node and still
  retain explicit data-ordering and replay truth.

#### `XTRAIN-6: Add A Hybrid Dense-Rank Plus Validated-Contributor Data Planner`

Description: let one pretraining program assign data honestly to different
execution classes.

Issue body:

- Summary: add a planner that can split one pretraining program into dense-rank
  batches, contributor windows, validator work, and eval slices without
  blurring those classes together.
- Why: many compute sources can contribute only if the program can assign
  heterogeneous work honestly.
- Depends On: `XTRAIN-2`, `XTRAIN-5`, `#484`.
- Scope: bind dense-rank batch planning, contributor-window planning, and eval
  planning to the same dataset and checkpoint family.
- Acceptance Criteria: one program manifest can emit admitted work plans for
  dense ranks and contributor windows in the same run.

#### `#515 PLIB-320C: Add Asynchronous Checkpoint Writeback For psionic-train`

Description: existing issue for non-blocking checkpoint writeback.

Issue body:

- Summary: add bounded asynchronous checkpoint writeback so long-running train
  loops do not stall on checkpoint serialization.
- Why: a serious multi-provider pretraining run cannot rely only on synchronous
  checkpoint writes.
- Depends On: none.
- Scope: immutable checkpoint handoff, bounded writer queues, atomic
  finalization, backpressure, and restore-equivalent semantics.
- Acceptance Criteria: async writeback reduces train-loop stall without
  weakening checkpoint lineage or durability truth.

#### `XTRAIN-7: Add Sharded Distributed Checkpoint Manifests, Upload Receipts, And Restore Planning`

Description: close the distributed checkpoint gap that current checkpoint docs
still leave open.

Issue body:

- Summary: extend checkpoint recovery from dense single-manifest closure to
  true sharded distributed checkpoint manifests, shard placement receipts,
  restore planning, and optimizer-state recovery.
- Why: the current checkpoint-recovery doc explicitly does not claim
  distributed optimizer recovery or parameter-shard semantics.
- Depends On: `XTRAIN-4`, `#515`.
- Scope: define distributed checkpoint shard manifests, shard uploader
  assignments, optimizer-state shard bindings, restore receipts, and refusal
  posture for partial or incompatible shard sets.
- Acceptance Criteria: a dense-rank multi-provider run can checkpoint and
  restore distributed state without inventing provider-specific restore logic.

#### `XTRAIN-8: Add Provider-Neutral Remote Artifact Backends And Placement Policy For Train Artifacts`

Description: close the gap between local artifact lifecycle logic and real
remote storage behavior.

Issue body:

- Summary: add blob-store-backed artifact and checkpoint backends with explicit
  placement policy, restore policy, and cost posture across providers.
- Why: the current artifact-storage layer explicitly stops before real remote
  storage and placement optimization.
- Depends On: `XTRAIN-3`, `XTRAIN-7`.
- Scope: add remote artifact backends, byte-accounted placement policy, restore
  policy, and retention rules for checkpoints, logs, metrics, and evidence.
- Acceptance Criteria: Google, RunPod, and later providers all persist train
  artifacts through one typed storage backend contract.

### Phase 3: Telemetry, Conformance, And Evidence

#### `#514 PLIB-320B: Add A Shared Backend Conformance Harness Across CPU, CUDA, Metal, And MLX-Backed Lanes`

Description: existing issue for one shared backend truth surface.

Issue body:

- Summary: run one canonical correctness harness across all admitted backend
  lanes with explicit `pass`, `fail`, and `unsupported` semantics.
- Why: one training system cannot work the same way across compute sources
  without one shared backend-truth discipline.
- Depends On: none.
- Scope: operator coverage, dtype coverage, deterministic seed behavior, and
  explicit refusal coverage for unsupported features.
- Acceptance Criteria: backend promotion no longer depends on scattered
  backend-local correctness scaffolding.

#### `#516 PLIB-320D: Add A Local Metric-Sink Layer For Train-Loop Telemetry`

Description: existing issue for stable train-loop metric emission.

Issue body:

- Summary: add one typed metric fanout layer that can drive structured logs,
  JSONL output, local progress, and pre-aggregation consumers.
- Why: the remote-visualization contract is not enough if every runtime still
  writes its own local telemetry format.
- Depends On: none.
- Scope: stable step binding, phase binding, deterministic flush, and a shared
  schema across train loops.
- Acceptance Criteria: training packages emit one local metric vocabulary
  instead of bespoke per-lane telemetry structures.

#### `#479 PSION_VIS-0: Ship The Psionic Live Remote-Training Visualization Substrate`

Description: existing issue for always-live provider-neutral visualization.

Issue body:

- Summary: close the remaining gap between the provider-neutral visualization
  bundle family and every active remote training lane.
- Why: the RunPod distributed lane still lacks the coordinator-owned live
  writer needed for the every-second requirement.
- Depends On: the remaining RunPod distributed runtime closure.
- Scope: one-second heartbeat and metric updates, live bundle refresh, and
  truthful stale-state behavior across all active remote lanes.
- Acceptance Criteria: active Google and RunPod training runs keep the typed
  visualization bundle fresh every second during execution.

#### `XTRAIN-9: Add One Provider-Neutral Final Evidence Bundle Family Across All Training Execution Classes`

Description: unify final machine-legible proof across single-node, dense-rank,
contributor-window, validator, and hybrid runs.

Issue body:

- Summary: define one final evidence bundle family that can seal single-node
  training, dense-rank distributed training, validated-contributor windows,
  hybrid runs, and after-action audit refs under one schema family.
- Why: the current evidence discipline is strong, but the bundle shapes are
  still lane-specific.
- Depends On: `XTRAIN-1`, `XTRAIN-3`, `XTRAIN-4`, `#516`.
- Scope: freeze one family for launch facts, runtime facts, checkpoints,
  metrics, validator results, visualization refs, and final disposition.
- Acceptance Criteria: finalizers stop inventing lane-specific proof JSON when
  the underlying run class differs.

### Phase 4: Provider Convergence, Launch Binding, And Planning

#### `XTRAIN-10: Add One Provider-Neutral Launcher And Runtime Binder Above Google, RunPod, And Local Lanes`

Description: shared binder between the training program contract and concrete
provider launchers.

Issue body:

- Summary: add one runtime binder that turns the program manifest and admitted
  compute sources into concrete launch manifests, runtime envs, startup plans,
  and finalizer plans.
- Why: provider-specific scripts should not remain the long-term training API.
- Depends On: `XTRAIN-2`, `XTRAIN-3`.
- Scope: freeze the binder interface, generated launch records, runtime env
  contract, and provider hooks.
- Acceptance Criteria: Google, RunPod, and local launchers become provider
  adapters over one shared binder instead of separate control planes.

#### `XTRAIN-11: Rebind The Google Single-Node And Google Swarm Lanes To The Shared Binder`

Description: make the Google lanes consumers of the shared launcher contract.

Issue body:

- Summary: migrate the existing Google single-node and Google swarm paths onto
  the provider-neutral binder without weakening their current evidence and
  preflight posture.
- Why: the Google surfaces are the strongest current operator paths and should
  become the first consumers of the shared contract.
- Depends On: `XTRAIN-10`.
- Scope: map Google launch profiles, quota preflight, startup, finalizer, and
  evidence generation into the shared binder contract.
- Acceptance Criteria: both Google lanes still run truthfully, but their
  training semantics come from shared contracts rather than Google-only wiring.

#### `XTRAIN-12: Rebind The RunPod And Local Fleet Lanes To The Shared Binder`

Description: converge non-Google operator surfaces onto the same launch model.

Issue body:

- Summary: migrate the RunPod distributed lane and the local workstation or
  trusted-LAN lanes onto the same provider-neutral binder used by Google.
- Why: a cross-provider system only closes when non-Google lanes stop being
  special operator programs.
- Depends On: `XTRAIN-10`.
- Scope: bind RunPod launch and finalization plus local launch and bring-up
  flows to the shared launch, runtime, and finalizer contracts.
- Acceptance Criteria: Google, RunPod, and local training lanes all project
  from the same binder and differ only in provider-specific resource steps.

#### `XTRAIN-13: Add A Cross-Provider Admission Planner With Cost, Network, And Trust Policy`

Description: one planner that decides which compute sources may join which run
in which role.

Issue body:

- Summary: add a planner that ranks and admits compute sources by backend,
  accelerator, network posture, storage posture, trust tier, and cost budget.
- Why: one universal system needs one explicit answer for where a machine fits
  before launch begins.
- Depends On: `XTRAIN-2`, `XTRAIN-10`.
- Scope: add admission policy, refusal reasons, role placement policy, and
  explicit trust-tier gating for local, cloud, and rented nodes.
- Acceptance Criteria: operator surfaces can explain why a machine was admitted
  as a dense rank, contributor, validator, or refused entirely.

### Phase 5: Heterogeneous Contribution Under One Pretraining Program

#### `#484 SWARM-0: Ship The First Local Mixed-Hardware Swarm Training Run On One Mac MLX Node Plus One Linux RTX 4080 Node`

Description: existing master issue for the first truthful mixed-hardware
contributor lane.

Issue body:

- Summary: finish the first real mixed-hardware swarm run as a bounded
  decentralized open-adapter lane.
- Why: it is the strongest current proof that heterogeneous hardware can
  contribute honestly without fake mixed dense training claims.
- Depends On: its closed child issue stack plus the remaining truthful live-run
  acceptance bar.
- Scope: retain one accepted live result with honest contributor, validator,
  replay, and closeout truth.
- Acceptance Criteria: the master issue closes with one real retained mixed-
  hardware swarm result and one explicit after-action audit.

#### `XTRAIN-14: Bind Validated Contributor Windows To Canonical Pretraining Checkpoint Families And Policy Revisions`

Description: stop treating contributor windows as a side system outside the main
pretraining program.

Issue body:

- Summary: bind contributor windows to the same checkpoint family, policy
  revision lineage, and dataset authority used by the dense pretraining
  program.
- Why: heterogeneous contributors only count toward one program if their work is
  attached to the same lineage and governance objects.
- Depends On: `XTRAIN-1`, `XTRAIN-6`, `#484`.
- Scope: freeze checkpoint, policy, dataset, and lineage bindings between
  dense-rank runs and contributor windows.
- Acceptance Criteria: contributor work can be traced directly into one
  canonical pretraining program lineage instead of a separate swarm namespace.

#### `XTRAIN-15: Unify Validator, Replay, And Promotion Contracts Across Providers And Execution Classes`

Description: one acceptance discipline for dense ranks, contributor windows, and
later hybrid programs.

Issue body:

- Summary: make validation, replay, quarantine, rejection, acceptance, and
  promotion semantics uniform across Google, RunPod, and local contributors and
  dense-rank lanes.
- Why: one cross-provider program fails if each execution class invents its own
  acceptance vocabulary.
- Depends On: `XTRAIN-9`, `XTRAIN-14`.
- Scope: freeze shared verdict classes, replay rules, promotion gates, and
  final disposition language.
- Acceptance Criteria: final evidence bundles and program audits use one shared
  validator and promotion vocabulary across all admitted execution classes.

#### `XTRAIN-16: Carry Dense Ranks, Contributor Windows, Validators, Checkpoint Writers, And Eval Workers In One Pretraining Run Graph`

Description: expand the run graph from one class of training participants to
one whole-program participant model.

Issue body:

- Summary: extend the run graph and orchestrator so one pretraining program can
  carry dense ranks, contributor windows, validators, checkpoint writers, and
  eval workers at the same time.
- Why: this is the actual system-level answer to “a lot of different compute
  can all contribute to one run.”
- Depends On: `XTRAIN-6`, `XTRAIN-13`, `XTRAIN-15`.
- Scope: add participant classes, assignment policy, state transitions, and
  finalizer bindings for the hybrid run graph.
- Acceptance Criteria: one run id can admit and track every execution class
  without splitting into separate program identities.

### Phase 6: Resilience, Elasticity, And Cross-Provider Proof Runs

#### `XTRAIN-17: Add Dense-Rank Recovery After Preemption, Node Loss, And Provider Loss`

Description: explicit failure recovery for real multi-provider dense training.

Issue body:

- Summary: add dense-rank recovery logic for preemption, host loss, provider
  loss, and controlled rejoin under one checkpoint and data-ordering contract.
- Why: multi-provider pretraining is not real if the first cloud failure forces
  manual operator recovery.
- Depends On: `XTRAIN-5`, `XTRAIN-7`, `XTRAIN-8`.
- Scope: define recovery paths, restore plans, refusal paths, and recovery
  receipts for dense-rank runs.
- Acceptance Criteria: a multi-provider dense run can recover from admitted node
  or provider loss with explicit receipts instead of ad hoc operator action.

#### `XTRAIN-18: Add Controlled Topology Revision And Elasticity For Cross-Provider Dense Clusters`

Description: let the dense cluster change under explicit rules instead of fixed
world-size only.

Issue body:

- Summary: add controlled topology revision for dense clusters, including
  admitted grow, shrink, and replace operations with replay-safe receipts.
- Why: the current dense data and checkpoint semantics are still fixed-world-
  size oriented.
- Depends On: `XTRAIN-5`, `XTRAIN-7`, `XTRAIN-17`.
- Scope: define allowed topology revisions, re-assignment rules, and refusal
  rules for unsupported changes.
- Acceptance Criteria: the dense cluster can revise topology under explicit
  policy without invalidating run truth or data-ordering receipts.

#### `XTRAIN-19: Execute The First Real Multi-Provider Dense CUDA Pretraining Run And Publish The After-Action Audit`

Description: the proof run that closes the core cross-provider dense system for
homogeneous CUDA hardware.

Issue body:

- Summary: run the first real dense CUDA pretraining job across at least Google
  and one non-Google provider under the shared contracts and retain the full
  evidence bundle.
- Why: until this run exists, the cross-provider dense system is still only
  architectural closure.
- Depends On: `XTRAIN-10` through `XTRAIN-18`.
- Scope: execute one real bounded multi-provider dense CUDA run, retain the
  evidence bundle, and write one after-action audit.
- Acceptance Criteria: the repo can cite one truthful multi-provider dense CUDA
  pretraining run as implemented rather than planned.

### Phase 7: Mixed-Backend Dense Training, If Same-Step MLX Plus CUDA Remains Required

#### `XTRAIN-20: Implement MLX-Backed Dense-Rank Training Runtime Parity On Metal`

Description: move the Apple lane from contributor-only truth toward dense-rank
runtime truth.

Issue body:

- Summary: add one MLX-backed dense-rank train-step runtime on Metal with
  checkpoint, metric, and receipt semantics comparable to the dense CUDA path.
- Why: same-job MLX plus CUDA dense training is impossible without a real MLX
  dense-rank runtime.
- Depends On: `#514`, `XTRAIN-4`.
- Scope: implement dense train-step, validation, metric, and checkpoint hooks
  for MLX-backed training.
- Acceptance Criteria: the Apple lane can run as a truthful dense rank instead
  of only a validated contributor window.

#### `XTRAIN-21: Add A Cross-Backend Collective, Precision, And Optimizer Contract For CUDA Plus MLX Dense Meshes`

Description: define the math and transport boundary for same-job mixed-backend
dense training.

Issue body:

- Summary: define the collective, precision, optimizer, and master-weight
  contract required for a dense mesh that spans CUDA and MLX backends.
- Why: mixed-backend dense training fails without one explicit answer for how
  synchronization, precision, and optimizer ownership work.
- Depends On: `XTRAIN-20`.
- Scope: freeze cross-backend collective semantics, precision policy,
  optimizer-state ownership, and refusal posture for unsupported operations.
- Acceptance Criteria: a mixed CUDA-plus-MLX dense mesh has one typed math
  contract instead of ad hoc backend glue.

#### `XTRAIN-22: Add Mixed-Backend Checkpoint, Restore, And Optimizer-State Parity Across CUDA Plus MLX`

Description: make mixed-backend dense runs restartable and portable.

Issue body:

- Summary: extend the distributed checkpoint and restore layer so mixed CUDA
  plus MLX dense runs can checkpoint, restore, and migrate state truthfully.
- Why: same-job mixed-backend training is not real if it cannot checkpoint or
  restore its mixed optimizer and parameter state.
- Depends On: `XTRAIN-7`, `XTRAIN-20`, `XTRAIN-21`.
- Scope: add mixed-backend checkpoint manifests, optimizer-state parity,
  restore receipts, and refusal posture for unsupported migrations.
- Acceptance Criteria: a mixed CUDA-plus-MLX dense run can checkpoint and
  restore without backend-specific manual conversion.

#### `XTRAIN-23: Execute The First Real Same-Job MLX-Plus-CUDA Dense Pretraining Run And Publish The Acceptance Audit`

Description: final proof issue only if the product requirement remains same-job
mixed dense training.

Issue body:

- Summary: execute the first real same-job dense pretraining run that spans at
  least one MLX-backed Mac node and one CUDA-backed NVIDIA node under the
  shared contracts.
- Why: until this run exists, the system still does not truthfully claim
  same-step mixed-backend dense training.
- Depends On: `XTRAIN-20`, `XTRAIN-21`, `XTRAIN-22`.
- Scope: run one bounded mixed-backend dense job, retain the evidence bundle,
  and write one acceptance audit that states exactly what was proved.
- Acceptance Criteria: the repo can truthfully claim one real same-job
  MLX-plus-CUDA dense pretraining run, or else keep the capability marked
  partial.

## What Not To Do

- Do not build separate Google, RunPod, and local training control planes.
- Do not describe adapter-delta swarm lanes as if they already solve full-model
  pretraining.
- Do not try to close the universal system by mixing MLX and CUDA into one
  dense synchronous trainer before the homogeneous CUDA distributed runtime is
  real.
- Do not let provider-specific scripts become the long-term training API.

## Conclusion

`psionic` is meaningfully closer to a real cross-provider training system than
it was even a few days ago.

The Google two-node swarm lane proves that real cloud multi-node training
control is no longer theoretical. The model-IO work proves that state
portability is starting to become explicit. The cluster layer, run graph,
orchestrator, and evidence discipline are already strong.

The missing center is still the reusable distributed training runtime.

Until `psionic` finishes one real provider-neutral dense distributed runtime and
binds it to one compute-source contract plus one heterogeneous role taxonomy,
the truthful claim is:

- `psionic` has the right training control-plane direction
- `psionic` has several real bounded training lanes
- `psionic` does not yet have one training system that works the same way
  across any compute source for one real broad pretraining run

The shortest honest implementation path is:

1. finish homogeneous distributed CUDA execution
2. make that runtime the shared train-system substrate
3. standardize compute-source and role contracts
4. expand heterogeneous contributors under the same pretraining program
5. leave true mixed-backend dense training for later

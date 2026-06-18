# 2026-06-18 cuTile Rust GPU Concurrency Adaptation Audit

Status: public-safe research audit
Scope date: 2026-06-18
Primary source: Melih Elibol, Jared Roesch, Isaac Gelado, Eric Buehler, and
Michael Garland, "Fearless Concurrency on the GPU",
arXiv:2606.15991v1, 2026-06-14,
https://arxiv.org/html/2606.15991v1

This audit records what Psionic should learn from the cuTile Rust paper. It is
not a claim that Psionic already has cuTile Rust, Grout, safe CUDA graph
capture, a Tile IR backend, or the reported throughput. It does not change the
Psionic acceptance matrix or any OpenAgents product promise.

## Short Answer

Yes. The paper is useful for Psionic.

The paper's concrete system is CUDA and NVIDIA-centered. It uses NVIDIA GPU
hardware, CUDA graph replay, and Tile IR/PTX-oriented assumptions. Direct
implementation lessons therefore start in `psionic-backend-cuda`.

The durable Psionic lesson is broader than CUDA. The useful abstraction is a
backend-independent device execution contract: partitioned mutable tensors,
shared read-only tensors, typed launch boundaries, replayable execution plans,
and explicit unsafe proof boundaries. CUDA is the first paper-backed target,
not the limit of the design.

This is mainly useful as a design reference for a future device-kernel contract
inside `psionic-array`, `psionic-compiler`, `psionic-ir`, and
`psionic-runtime`, with backend-specific implementations in
`psionic-backend-cuda`, `psionic-backend-metal`, AMD backends, CPU task-graph
execution, and eventually distributed collectives.

It should not be adopted directly into OpenAgents product surfaces. OpenAgents
should consume this work only after Psionic has implementation evidence:
capability reports, receipts, parity fixtures, benchmark records, and explicit
refusal boundaries. Until then, OpenAgents docs and UI should say no more than
that a Rust-native safe GPU-kernel lane is a Psionic research direction.

Psionic should not vendor the paper's system by default. Treat cuTile Rust and
Grout as a reference. If public code becomes available and the license,
maintenance posture, CUDA-version support, and dependency shape are acceptable,
study it in a reference lane first. Port the useful contracts into Psionic's
own runtime and receipt model instead of making OpenAgents depend on an
unproven external kernel DSL.

## Paper Summary

The paper presents cuTile Rust, a tile-based Rust system for writing GPU
kernels while preserving a Rust-like ownership discipline across host code,
kernel launch, and device code.

The central idea is simple and strong:

- immutable tensors can be shared as read-only device inputs
- mutable output tensors are partitioned before launch
- each device tile program receives a disjoint mutable sub-tensor
- branded partition indices and bounded iterators carry enough information to
  prove common partition accesses are bounded and disjoint
- Tile IR token ordering preserves the sequencing needed for mutable tensor
  operations
- unsupported low-level patterns use explicit unchecked escape hatches

The host execution model is also relevant. GPU work is represented as lazy,
typed device operations. The same typed launch can run synchronously, run
through async Rust, or be captured as a CUDA graph for repeated low-latency
replay. The generated host launch code prevents host access while GPU work is
in flight and returns ownership only after the relevant synchronization or
graph boundary.

That exact replay mechanism is CUDA-specific in the paper. The shape of the
contract is not. Metal command buffers, HIP graphs, Vulkan/WebGPU command
buffers, CPU task graphs, and distributed collective schedules can all expose
similar lifecycle states: built, submitted, in flight, synchronized, replayed,
or refused.

The evaluation matters because it argues that safety did not obviously destroy
performance. The paper reports B200 elementwise throughput around 7 TB/s and
GEMM around 2 PFlop/s, about 96% of cuBLAS for the measured case. It also
describes Grout, a cuTile-Rust-based Qwen3 inference engine. The abstract
reports batch-1 decode at 171 generated tokens/s for Qwen3-4B on RTX 5090 and
82 generated tokens/s for Qwen3-32B on B200. The body also reports a plotted
sweep with 154.7 tokens/s and 80.1 tokens/s respectively. That is still enough
to treat the result as serious evidence that a Rust GPU authoring stack can
compete on a specialized LLM inference path.

The limitations are important. The safe surface does not eliminate the need for
unchecked code in performance-critical kernels. The paper calls out attention,
fused normalization, low-level SIMT control, GEMM gaps at some sizes, and a
young tensor API as remaining constraints. Grout is also a specialized batch-1
engine, not proof that the same system covers general batched serving,
multi-tenant scheduling, speculative decode, prefix sharing, tool latency, or
OpenAI-compatible production behavior.

## Fit With Psionic

Psionic already has the right repo boundary for this idea.

`docs/ARCHITECTURE.md` defines Psionic as the Rust-native execution substrate
for OpenAgents. Psionic owns runtime execution, backend capability,
execution-planning truth, artifact identity, proof posture, and receipts. That
is exactly where a typed GPU-kernel launch contract belongs.

The current public array and CUDA surfaces are more bounded than the paper's
target. `psionic-array` already exposes graph-backed lazy arrays, deterministic
graph snapshots, accelerator-labeled evaluation, debug capture posture, and a
bounded CUDA eval surface. `psionic-backend-cuda` exists, but Psionic should not
pretend that a cuTile-class safe kernel authoring model has landed there.

The paper is therefore a good direction-of-travel reference, not an immediate
drop-in. Psionic should use it to define a backend-neutral contract first, then
map that contract onto CUDA, Metal, AMD, CPU, and distributed backends only
where each backend can support the claim honestly.

## Generalized Psionic Contract

The generalized contract should be backend-neutral.

The shared concepts are:

- partitioned mutable tensors: one writer owns one disjoint output region
- shared read-only tensors: many tasks or kernels can read the same buffer
- typed launch boundaries: host code records which buffers are borrowed, which
  partitions are mutable, and when ownership returns
- execution lifecycle states: build, submit, in flight, synchronize, replay,
  recover, invalidate, or refuse
- unsafe escape hatches: hot paths can bypass the safe surface only with
  reason-coded proof-boundary labels
- receipts: Psionic records partition geometry, backend identity, replay
  identity, unsafe labels, and parity evidence

The backend mappings can differ:

- CUDA maps replay to CUDA graphs and stream-ordered execution
- Metal maps replay to command buffers, encoders, heaps, and pipeline-state
  reuse where the semantics fit
- AMD maps replay to HIP graphs or ROCm/HSA queue semantics where available
- CPU maps the same ownership contract to Rayon or Tokio task graphs
- distributed execution maps partitions to rank-owned tensor shards,
  collectives, peer copies, checkpoint barriers, and per-rank receipt
  fragments

This keeps the safety claim above any one vendor API. A backend can implement a
subset, report that subset, and refuse unsupported modes without weakening the
common Psionic contract.

## What Psionic Should Learn

### 1. Treat host-to-backend launch as a typed contract

Psionic should model device launch as a machine-legible runtime contract, not
only as a backend implementation detail.

The contract should name:

- input tensor identities
- mutable output tensor identities
- partition geometry
- disjointness proof posture
- stream, queue, command-buffer, or task-graph identity
- graph-capture identity when applicable
- synchronization or replay boundary
- ownership recovery point
- unsafe escape hatch labels
- backend and architecture constraints

That contract should be eligible for inclusion in Psionic receipts. The right
eventual surface is not a prose claim that a kernel was safe. The useful surface
is a typed launch record that validators can inspect and replay against bounded
fixtures.

### 2. Add partitioned mutable tensors before broad custom device kernels

Psionic should not rush into a pile of custom backend kernels with ordinary raw
pointer launch wrappers.

The first useful substrate work is a partition API for mutable device tensors:

- deterministic partition shapes
- bounded partition indices
- non-overlap checks
- shape-aware view recovery
- compile-fail or refusal behavior for aliasing patterns
- debug receipts that show which partition contract was used

This can begin above the existing graph-backed array surface and does not need
to support every attention kernel on day one. Elementwise operations, tiled
copy, reductions, RMSNorm-like kernels, and small matmul cases are enough to
prove whether the model fits Psionic's type and receipt structure.

### 3. Preserve explicit unsafe escape hatches

The paper's escape-hatch discipline is better than pretending every hot kernel
can be safe immediately.

Psionic should make unsafe backend paths explicit and inspectable. A CUDA
kernel that bypasses partition proof should carry a reason-coded label such as:

- `unchecked_raw_pointer_kernel`
- `unchecked_attention_layout`
- `unchecked_shared_memory_protocol`
- `unchecked_vendor_blas_call`
- `unchecked_graph_replay_pointer_stability`

Those labels should flow into capability reports and receipts. A fast path can
be valid while still saying which proof boundary it skipped.

### 4. Make replay a typed execution mode

The paper's CUDA graph treatment maps well to Psionic's existing graph and
receipt vocabulary. Psionic should generalize the contract instead of making
the concept CUDA-only.

Psionic should treat captured replay as a runtime execution mode with its own
contract:

- capture input shape contract
- non-allocating or allocation-stable node restriction
- pointer-stability requirement
- stream, queue, command-buffer, or task-graph dependency record
- replay count
- graph digest
- invalidation reasons
- fallback/refusal reasons

CUDA graphs are one implementation. Metal command-buffer reuse, HIP graphs,
Vulkan/WebGPU command-buffer replay, CPU task-graph replay, and distributed
collective schedules can implement smaller or different subsets of the same
contract.

Captured replay is especially relevant for batch-1 decode and other repeated
low-latency paths. It is also relevant for fixed-shape training inner loops.
The important rule is that replay must remain a typed mode with receipts, not
an opaque optimization hidden behind a backend flag.

### 5. Use async host/device orchestration only where it pays

The paper's appendix is useful because it does not treat async as a universal
win. Async helps when there is meaningful host work, many streams, I/O, control
tasks, or heterogeneous scheduling. It does less for tiny single-model decode
loops where the host has little useful work to overlap.

That maps directly to Psionic:

- use async for stream queues, VAD, ASR, tool/control-plane work, ingest,
  datastream, and multi-session serving
- do not add async indirection to a single hot decode loop unless a benchmark
  proves it helps
- record CPU-footprint and stream-count effects when testing async CUDA paths

### 6. Extend the partition idea to distributed execution cautiously

The paper notes that the same partitioning idea could extend across multiple
devices, where each GPU owns a partition and collectives or peer-to-peer
operations participate in the ownership model.

This fits Psionic's distributed and collectives work, but it should stay
research-scoped until there is a concrete model. The useful future contract is:

- rank-owned tensor partitions
- collective preconditions
- peer-copy alias rules
- checkpoint and recovery ownership transfer
- per-rank receipt fragments
- whole-run aggregation into a proof bundle

This should be developed inside Psionic's distributed substrate. OpenAgents
should see only the resulting capability envelope and proof posture.

### 7. Keep Grout in the right box

Grout is strong evidence that Rust GPU authoring can power a serious
model-specialized inference engine. It is not evidence that Psionic should
replace broad serving work with a single batch-1 Qwen path.

Psionic should study Grout for:

- CUDA graph replay in one-token decode
- device-side greedy token selection
- minimal scheduler/cache overhead
- QK-norm, RoPE, KV-cache write, GQA attention, and split-K merge fusion
- clear roofline sanity checks

Psionic should not treat Grout as covering:

- general batched serving
- mixed tenant scheduling
- speculative decode
- arbitrary model families
- OpenAI-compatible server behavior
- production admission, routing, or settlement

## Recommended Adoption Path

### Immediate

Keep this as an audit only. Do not change user-facing OpenAgents claims.

Record one planned Psionic work item: a backend-neutral partitioned-tensor
launch contract, plus a first CUDA tile-safety experiment that proves
partitioned mutable tensors, typed launch records, and unsafe escape labels on
a tiny kernel set.

Good first kernels:

- elementwise add or scale
- tiled copy
- sum reduction
- RMSNorm-like normalization
- one small matmul case if it can be compared against cuBLAS without claiming
  full GEMM coverage

Acceptance should require CPU parity, CUDA parity for the first backend slice,
deterministic receipt generation, and explicit refusal for unsupported aliasing
or shape patterns. The contract should be written so Metal, AMD, CPU, and
distributed backends can later implement honest subsets without rewriting the
receipt vocabulary.

### Short Term

Add a design doc or issue program for a Psionic device launch contract.

The design should cover:

- partitioned mutable tensors
- read-only tensor sharing
- typed stream, queue, command-buffer, and task-graph execution
- typed capture and replay, with CUDA graph capture as the first concrete lane
- unsafe escape-hatch labeling
- launch receipts
- compile-fail tests where possible
- runtime refusal tests where compile-time checking is not yet available

This belongs in Psionic docs first. Implementation should stay scoped to
`psionic-array`, `psionic-ir`, `psionic-compiler`, `psionic-runtime`, and
backend crates, starting with `psionic-backend-cuda`.

### Medium Term

Build one benchmarked captured-replay lane for a fixed-shape decode or train
inner loop. CUDA graph replay is the best first target because the paper gives
direct evidence for that path.

The target should not be "beat vLLM" on the first pass. The target should be:

- exact parity with a reference path
- stable graph digest
- replay determinism under fixed shapes
- measured replay overhead
- clear refusal when shape, allocation, pointer-stability, or backend
  constraints are violated

After that exists, Psionic can compare against vLLM, SGLang, cuBLAS, and any
public cuTile/Grout implementation if one is available and license-safe to
study.

### OpenAgents Integration

OpenAgents should adopt only the outputs of this work:

- capability reports
- benchmark receipts
- public-safe claim boundaries
- worker/provider eligibility labels
- refusal reasons
- proof-bundle links

OpenAgents should not adopt the paper as product copy. It should not imply
that safe Rust GPU kernels are already part of the production substrate until
Psionic produces retained evidence.

### Product Surfaces That Benefit First

The first OpenAgents benefit is better evidence, not new marketing copy.

The surfaces that would benefit first are:

1. `/api/training/runs/...` and the public run pages. These can show a stronger
   backend execution posture per run window: backend subset, partition
   geometry, replay mode, unsafe labels, parity evidence, and refusal reasons.
   The immediate value is clearer run truth for operators, contributors, and
   public auditors.
2. `/tassadar`, `/api/public/tassadar-run-summary`, and proof-replay bundles.
   Tassadar already depends on digest-pinned exact replay. A Psionic
   launch-contract lane would let the public Tassadar surface distinguish CPU
   reference replay, CUDA replay, Metal replay, unsafe-kernel replay, and
   refused accelerator replay without collapsing them into one green check.
3. Pylon training commands and capability reporting. `pylon training status`,
   `pylon training claim`, and provider capability envelopes can advertise
   executor or training capacity with backend-specific proof posture instead of
   vague GPU availability.
4. Training registry and promise evidence. Records such as
   `training.verification_classes.v1`,
   `training.device_capability_dataset.v1`,
   `training.public_gradient_windows.v1`,
   `pylon.first_real_model_training_run.v1`, and
   `compute.tassadar_executor_poc.v1` would gain better evidence fields. The
   state of those promises should not move until real accepted work, verifier
   receipts, and settlement evidence exist.
5. Proof replay and social replay surfaces. The replay UI can visualize the
   device-execution contract as part of the proof story: what ran, where it
   ran, what replayed, which unsafe boundaries were used, and which backend
   modes refused.
6. Admin and Autopilot review surfaces. Operator views can use the same
   receipts to decide whether a worker, run window, or provider claim is
   admissible. This is later than run-page and Pylon status value because it
   depends on the same Psionic evidence being available first.

The product ordering should be:

1. Psionic emits launch receipts and backend-subset capability reports.
2. OpenAgents run APIs expose those reports without changing promise state.
3. Pylon CLI and provider envelopes display the evidence and refusal posture.
4. Tassadar proof pages and replay surfaces render the evidence.
5. Product promises move only after accepted work and settlement receipts prove
   the broader claim.

### Value For Tassadar And Psion Runs

This approach is valuable for Tassadar first, but it must preserve Tassadar's
exactness boundary.

Tassadar's current value is exact replay: digest-pinned compiled workloads,
independent replay, verifier verdicts, and settlement receipts. A generalized
device launch contract would strengthen the next accelerator-backed Tassadar
steps by recording exactly how a compiled dense module, linked module, or
runtime trace was executed on a backend.

For Tassadar, the useful additions are:

- exact CPU/reference parity before any accelerator receipt is admitted
- backend-specific replay labels such as CPU reference, CUDA captured replay,
  Metal command-buffer replay, or refused accelerator replay
- partition geometry and ownership records for dense modules and linked
  modules
- unsafe-boundary labels when a hot kernel bypasses the safe partition surface
- deterministic trace digest binding across worker and validator devices
- public-safe replay metadata for `/tassadar` and proof-replay bundles

The risk is also clear. Tassadar must not let an approximate GPU fast path
weaken exact replay. If a CUDA, Metal, AMD, or CPU task-graph backend cannot
produce deterministic parity with the reference trace, the Tassadar lane should
refuse the accelerated mode or label it as non-authoritative. Accelerator
throughput is useful only after exactness survives.

For generic Psion training runs, the value is different. Psion is not the
exactness-claiming lane; it needs honest training and run evidence:

- rank-owned tensor partition receipts for distributed windows
- gradient/update ownership and checkpoint recovery records
- backend-subset reports for CUDA, Metal, AMD, CPU, and mixed-device runs
- captured-replay receipts for fixed-shape inner loops
- unsafe-kernel labels for fused optimizer, attention, normalization, or matmul
  paths
- device capability dataset rows that distinguish real accelerator support from
  dark or refused capacity

That would help the training program's verification classes, device-capability
dataset, public run pages, and eventually `pylon.first_real_model_training_run`
evidence. It does not by itself prove public decentralized gradient training.
Accepted public gradient windows still need contributor devices, verified
updates, checkpoint/promotion policy, economics gates, and settlement receipts.

## Validation Requirements

The first implementation slice should include tests and fixtures that check the
contract, not only throughput.

Required validation:

- CPU reference parity for every admitted kernel
- CUDA parity for every admitted kernel
- negative tests for overlapping mutable partitions
- negative tests for out-of-bounds partition maps
- receipt snapshots that include partition geometry and unsafe labels
- graph replay determinism for fixed-shape replay
- refusal snapshots for unsupported shapes, dynamic allocation in graph replay,
  unavailable CUDA devices, and unsupported architecture features
- benchmark reports that separate launch overhead, kernel time, graph replay
  time, and end-to-end throughput
- backend-subset reports that distinguish CUDA, Metal, AMD, CPU, and
  distributed support instead of collapsing them into one generic "safe GPU"
  claim

Useful formal or model notes:

- a bounded partition-disjointness model
- a launch-state model for borrowed, in-flight, synchronized, and recovered
  tensor ownership
- a graph-capture model that distinguishes recorded, launched, replayed, and
  invalidated graph state

Tests cannot prove a CUDA system data-race-free in the full sense. They can
prevent Psionic from advertising a proof posture that the implementation does
not actually support.

## Risks And Kill Conditions

The main risks are concrete:

- the paper's system is CUDA-specific and NVIDIA-centered
- the results may depend on B200 or RTX 5090 behavior that does not transfer
  to Psionic's other backends
- backend-independent wording could overstate what non-CUDA backends support
  if each backend does not emit precise subset reports
- the safe surface does not yet cover all performance-critical LLM kernels
- unchecked code remains necessary for some attention and fused paths
- Grout's batch-1 Qwen specialization does not prove broad serving maturity
- dependency, license, or maintenance constraints may block direct reuse
- adding a parallel kernel DSL could duplicate Psionic's existing graph,
  compiler, runtime, and receipt abstractions

Kill direct adoption if any of the following are true:

- no license-safe implementation is available to study or depend on
- the implementation requires an external runtime shape that bypasses Psionic
  receipts
- unsafe kernels dominate before Psionic has unsafe-boundary labels
- CUDA graph replay cannot produce deterministic replay receipts
- the work would create product claims before retained Psionic evidence exists

## Decision

Adopt the ideas, not the product claim.

Psionic should use the paper to guide a Rust-native device launch-contract
lane: partitioned mutable tensors, typed host-to-backend launch, explicit
unsafe escape hatches, captured replay as a typed mode, and receipt-backed
validation. CUDA should be the first concrete backend target because the paper
is CUDA-centered. The contract should be general enough for Metal, AMD, CPU,
and distributed backends to implement honest subsets.

OpenAgents should wait. It should expose this only after Psionic lands retained
evidence and emits bounded capability reports. The near-term OpenAgents action
is to avoid overstating the paper. The near-term Psionic action is to turn the
paper into a small, testable CUDA-kernel safety experiment.

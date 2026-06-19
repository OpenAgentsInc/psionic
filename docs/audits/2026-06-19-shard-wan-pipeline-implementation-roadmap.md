# 2026-06-19 Shard WAN Pipeline Implementation Roadmap

> Status: planned audit.
>
> This is a point-in-time implementation audit after reading the external
> reference repo at `../../../projects/repos/shard`. It is not a canonical
> current-state Psionic spec. The canonical cluster and inference surfaces
> remain `../ROADMAP_CLUSTER.md`, `../INFERENCE_ENGINE.md`,
> `../INFERENCE_MESH_OWNERSHIP.md`, `../OWNERSHIP.md`, and the crate APIs under
> `../../crates/psionic-*`.

## Scope

External Shard source material read for this audit:

- `../../../projects/repos/shard/README.md`
- `../../../projects/repos/shard/docs/ARCHITECTURE.md`
- `../../../projects/repos/shard/docs/PROOF.md`
- `../../../projects/repos/shard/docs/ROADMAP.md`
- `../../../projects/repos/shard/docs/research/wan-speculative-decoding.md`
- `../../../projects/repos/shard/docs/research/glm-5.2-on-consumer-blackwell.md`
- `../../../projects/repos/shard/docs/receipts/glm52-nvfp4-wan-20260618.json`
- `../../../projects/repos/shard/docs/receipts/gpt-oss-120b-wan-20260619.json`
- `../../../projects/repos/shard/shard/*.py`
- `../../../projects/repos/shard/phase0/*.py`

Psionic surfaces inspected for fit:

- `../../README.md`
- `../ROADMAP_CLUSTER.md`
- `../../Cargo.toml`
- `../../crates/psionic-cluster/src/lib.rs`
- `../../crates/psionic-cluster/src/pipeline_sharded.rs`
- `../../crates/psionic-cluster/src/layer_sharded.rs`
- `../../crates/psionic-cluster/src/tensor_sharded.rs`
- `../../crates/psionic-cluster/src/benchmark_receipts.rs`
- `../../crates/psionic-distributed/src/lib.rs`
- `../../crates/psionic-net/src/lib.rs`

## Executive Summary

Shard proves a useful serving shape for Psionic: run one large model by splitting
contiguous transformer layer blocks across separate machines, stream activations
between stages, and hide WAN round trips with speculative verification,
direct-return routing, asynchronous in-flight chunks, topology-aware placement,
and backend-specific static-cache graph execution.

Psionic should implement this as a Rust-native cluster inference lane, not as a
Python port and not as a wrapper around Shard. The right Psionic product is:

- `psionic-cluster` owns planning, topology selection, stage assignment,
  sharded-session receipts, and scheduler policy
- `psionic-net` owns authenticated activation transport, edge health,
  rendezvous or relay posture, and direct-return sessions
- `psionic-runtime`, `psionic-models`, and backend crates own model-family stage
  execution, KV lifecycle, quantization eligibility, and exact greedy parity
- `psionic-serve` owns coordinator serving, streaming, OpenAI-compatible request
  mapping, and user-visible response provenance
- provider, identity, pricing, payout, and marketplace authority remain outside
  Psionic unless an explicit product task moves only evidence hooks into this
  repo

The first milestone should be a trusted two-stage and then N-stage local or LAN
split on a small model with exact greedy parity and typed receipts. The next
milestone should be WAN topology measurement and plain pipeline serving. Only
after that should Psionic add speculative decoding, direct return, async
pipelining, and backend-specific static-KV fast paths. Large-model claims such
as gpt-oss-120B MXFP4 or GLM-5.2 NVFP4 should stay blocked until the backend
stage executor, quantized layer-block loader, and receipts have real hardware
evidence.

## What Shard Actually Proves

Shard has two different maturity levels:

- the top-level `shard/` package contains mostly interface stubs, except for a
  real topology solver
- the `phase0/` directory contains the meaningful prototypes: sealed tensor
  framing, two-node split inference, N-stage pipeline inference, speculative
  decoding, tree speculative decoding, WAN mesh ordering, CUDA graph fast
  verification, and receipt generation

The central Shard architecture is:

| Area | Shard behavior | Psionic interpretation |
| --- | --- | --- |
| Layer split | Assign contiguous transformer layer ranges to stages. No worker holds the whole model. | Implement `PipelineSharded` as a real execution lane over `ShardedModelManifest` and explicit layer-block assignments. |
| Coordinator | Holds prompt handling, token selection, draft model, and sometimes embed/head or tail-return logic. | Put request authority in `psionic-serve`; keep execution topology and receipt authority in `psionic-cluster`. |
| Transport | Prototype TCP with JSON header plus raw tensor bytes, ChaCha20-Poly1305 under `SHARD_PSK`, and no pickle. QUIC/NAT remains roadmap. | Build a typed Rust activation frame in `psionic-net` using cluster identity, session keys, replay protection, deadlines, and no ad hoc untyped tensor deserialization. |
| Topology | Measure RTTs and solve a minimum-latency ring/path with an exact Held-Karp solver for small node counts. | Add route optimization from measured `ClusterLink` facts and carry the selected route into `ExecutionTopologyPlan`. |
| Plain decode | Stream activations stage by stage for every token. | First Psionic correctness gate. Throughput can be low if the receipt is honest. |
| Speculative decode | Draft K tokens locally, verify `[current, draft...]` through the large target in one traversal, accept greedy prefix, correct on divergence, crop KV. | Add after plain split correctness. Exact greedy parity is the non-negotiable acceptance gate. |
| Direct return | Tail stage can send logits or accepted verification result directly to coordinator instead of traversing the ring backward. | Add explicit return channels and receipt the realized data path. |
| Async pipelining | Keep multiple verification chunks in flight, discard stale results after divergence. | Add only after spec decode and direct return are deterministic under tests. |
| Fast verify | Static KV cache plus CUDA graph capture for fixed query shapes; graph removes launch overhead after WAN is hidden. | Backend-specific optimization. Do not make this a generic cluster guarantee. |
| Receipts | Record nodes, public IP or region, GPU UUID, RTT edges, model, quant, prompt/output hash, throughput, and reference tokens. | Extend Psionic benchmark and evidence receipts with run-level sharded inference receipts. |
| Privacy | Shard is clear that sealed wire does not hide activations from participating nodes. | Psionic must expose activation-visible privacy posture and trusted-routing policy. |

Shard's reported results are useful as target evidence, not as Psionic claims:

- Qwen2.5-14B bf16 two-node prototype: contiguous split and sealed activation
  wire proved feasibility
- gpt-oss-120B MXFP4 WAN receipt: three stage nodes plus coordinator, around
  39.8 warm tok/s, exact tokens against same-engine reference
- GLM-5.2 744B NVFP4 WAN receipt: six stage nodes plus coordinator, around
  30.03 warm tok/s, model-family features such as MLA and native MTP make WAN
  serving more favorable

## Current Psionic Fit

Psionic already has several pieces that line up with Shard:

| Psionic surface | Current fit | Status |
| --- | --- | --- |
| `psionic-cluster::pipeline_sharded` | Public-network pipeline-sharded planning policy, stage counts, layer counts, activation and KV byte estimates, staged manifest validation, and topology evidence. | `implemented_early` |
| `psionic-cluster::layer_sharded` | Layer-sharded planning surface for explicit layer distribution. | `implemented_early` |
| `psionic-cluster::tensor_sharded` | Tensor-sharded planning surface and manifest compatibility checks. | `implemented_early` |
| `psionic-cluster::benchmark_receipts` | Typed cluster benchmark receipts for planner gates. | `implemented_early` |
| `psionic-net` | Cluster identity, admission tokens, signed evidence bundles, hello/ping substrate, and trust posture types. | `implemented_early` |
| `psionic-distributed` | Public distributed semantics, groups, topology profiles, backend capabilities, and collective-style abstractions. | `implemented_early` |
| Runtime stage execution | Psionic has local inference and backend lanes, but a production activation-driven layer-block runtime is not yet visible as a complete public contract. | `partial` |
| WAN activation data plane | Existing networking is cluster/session-oriented; Shard-style tensor activation streaming, direct return, and per-edge decode deadlines are not yet the main data plane. | `planned` |
| Speculative verification | No inspected Psionic surface provides Shard-style multi-token target verification with KV crop/gather semantics. | `planned` |
| Large quantized stage execution | gpt-oss MXFP4 and GLM NVFP4 stage loading require model-family and backend-specific evidence. | `planned` |
| Privacy disclosure for activation-visible workers | Cluster trust posture exists, but Shard-style activation visibility needs a first-class serving policy. | `partial` |

The strongest existing Psionic foundation is the planning and evidence shape.
The largest missing piece is the actual inference data path: an authenticated
activation stream plus backend stage runtimes with KV lifecycle control.

## Implementation Boundary

Psionic should copy concepts, not code structure.

Use Shard as a reference for:

- contiguous layer-block decomposition
- no-whole-model worker residency
- measured topology selection
- exact greedy reference checks
- speculative target verification
- direct-return and async in-flight verification
- receipt fields that rule out common fake demos

Do not import Shard's Python runtime as a production dependency. Shard gets
performance by reusing SGLang, vLLM, Transformers, CUDA graphs, and hand-run
servers. Psionic's stated scope is the machine-facing execution substrate, so
the production path needs typed Rust contracts and backend-specific execution
truth inside `crates/psionic-*`.

Keep these out of Psionic's core implementation:

- c0mpute marketplace UX
- pricing and payout authority
- dashboard state
- user workroom or app-level orchestration
- broad claims of trustless verification for arbitrary WAN workers

Psionic can emit evidence that those outer systems consume.

## Proposed Architecture

### 1. Sharded Session Plan

Add a first-class sharded inference session plan in `psionic-cluster`, backed by
existing `ExecutionTopologyPlan`, `ClusterShardHandoff`, and
`ShardedModelManifest` concepts.

The plan should include:

- `session_id`
- `served_artifact_digest`
- tokenizer and model-family digests
- requested backend and quantization
- coordinator node
- optional draft node or draft-local coordinator declaration
- ordered stage assignments
- per-stage layer ranges
- stage roles: `head`, `middle`, `tail`, `draft`, `coordinator`
- edge transport endpoints
- direct-return edge, when used
- RTT and bandwidth facts used for selection
- cache policy
- privacy policy
- receipt policy
- topology and policy digests

This plan is the authority for the run. Stage workers should refuse activation
frames that do not match an admitted session plan.

### 2. Activation Transport

Add an activation data-plane module in `psionic-net`.

The minimum frame header should carry:

- schema version
- session id
- request id
- stage id
- sequence number
- epoch or branch id for speculative decode
- frame kind: `prefill`, `decode`, `verify`, `verify_result`, `tail_logits`,
  `kv_crop`, `kv_gather`, `abort`, `heartbeat`
- tensor dtype
- tensor shape
- tensor layout
- uncompressed payload bytes
- codec
- payload digest
- deadline

The payload should be raw tensor bytes or a typed codec payload, authenticated
and encrypted with session keys derived from Psionic cluster identity. The
Python-specific "pickle-free" lesson maps to this Rust rule: no unbounded,
untyped, code-executing deserialization for activation payloads.

Transport acceptance gates:

- malformed frames are rejected with stable errors
- replayed sequence numbers are rejected
- wrong session ids are rejected
- wrong stage ids are rejected
- payload digest mismatch is rejected
- missing or expired deadlines are rejected by policy
- every rejection can be carried into a receipt without printing secrets

QUIC should be the public-network target, but the first correctness gate can
use a deterministic TCP implementation if it preserves the same frame contract.

### 3. Stage Runtime Contract

Add a backend-neutral stage runtime trait in `psionic-runtime` or the model
runtime layer, then implement it per backend and model family.

The contract should cover:

- load only the assigned layer block plus required boundary modules
- refuse missing quantization or unsupported layer families
- prefill into stage-local KV
- run one decode step from incoming hidden states
- run multi-token verify from incoming hidden states or token ids
- crop KV to a committed prefix length
- gather or remap KV after tree/speculative acceptance when supported
- reset or abort a session cleanly
- report actual memory residency and warm state

Sketch:

```rust
pub trait PipelineStageRuntime {
    type Error;

    fn admit_session(&mut self, plan_digest: &str) -> Result<(), Self::Error>;
    fn prefill(&mut self, input: StageInput) -> Result<StageOutput, Self::Error>;
    fn decode(&mut self, input: StageInput) -> Result<StageOutput, Self::Error>;
    fn verify(&mut self, input: VerifyInput) -> Result<VerifyOutput, Self::Error>;
    fn crop_kv(&mut self, committed_len: u64) -> Result<(), Self::Error>;
    fn gather_kv(&mut self, selection: KvGatherSelection) -> Result<(), Self::Error>;
    fn finish_session(&mut self) -> Result<StageReceipt, Self::Error>;
}
```

The exact API should follow existing Psionic runtime conventions. The required
semantics are more important than this shape.

### 4. Coordinator Algorithms

`psionic-serve` should expose one coordinator execution mode with increasingly
capable algorithms behind explicit capabilities.

Plain pipeline decode:

1. Tokenize prompt.
2. Send prefill through ordered stages.
3. For each token, send decode activation through stages.
4. Tail produces logits.
5. Coordinator samples or greedy-selects token.
6. Receipt records route, timing, and output hash.

Speculative decode:

1. Coordinator draft model proposes K tokens.
2. Target pipeline verifies the current token plus proposed tokens in one
   traversal.
3. Coordinator accepts the longest greedy prefix.
4. On divergence, coordinator emits the correction token.
5. Coordinator instructs stages to crop or gather KV to the committed prefix.
6. Receipt records K, accepted tokens per traversal, correction count, and exact
   reference parity.

Async pipelined verify:

1. Coordinator keeps up to `depth` verify chunks in flight.
2. Every chunk carries an epoch and branch id.
3. Divergence advances the committed epoch.
4. Stale results are discarded and receipted.
5. Exact output parity remains the acceptance gate.

Direct return:

1. Forward activation path remains ordered through stages.
2. Tail returns logits or verification result directly to coordinator.
3. The direct-return edge is in the session plan and receipt.
4. If the direct edge fails, policy either aborts or falls back explicitly.

### 5. Topology Selection

Port Shard's topology lesson into `psionic-cluster`:

- measure RTTs at the application edge, not only from static geography
- preserve asymmetric RTT facts when available
- solve exact best route for small N
- use a heuristic only when the exact search would be too expensive
- score coordinator placement and stage order together
- include direct-return edge cost in the scoring objective
- attach the selected route and all measured facts to the receipt

For up to 16 candidates, a Held-Karp-style dynamic program is acceptable. Above
that, use an explicit heuristic with a receipt field naming the heuristic and
its input digest.

### 6. Receipts And Proof Posture

Shard's proof document is useful because it names what receipts can and cannot
prove. Psionic should keep the same honesty.

A Psionic sharded inference receipt should prove:

- the run used more than one admitted node
- the session plan selected those nodes and layer ranges
- the model artifact and quantization were the declared ones
- the measured topology facts were present before scheduling
- stage workers accepted the same plan digest
- the output hash matches the emitted text or tokens
- exact greedy parity was checked when a local or same-engine reference is
  available
- throughput windows are defined and reproducible
- tamper or missing-evidence failures are machine-checkable

It should not claim:

- trustless verification of every worker's internal computation
- privacy from a participating stage worker
- bit-exact parity across unrelated engines or quantization implementations
- general WAN performance from a single hand-picked route

Recommended schema names:

- `psionic.cluster.pipeline_sharded_session_plan.v1`
- `psionic.net.activation_frame.v1`
- `psionic.net.activation_edge_health.v1`
- `psionic.cluster.topology_measurement.v1`
- `psionic.serve.spec_decode_round_receipt.v1`
- `psionic.serve.pipeline_sharded_run_receipt.v1`
- `psionic.serve.activation_privacy_policy.v1`

### 7. Privacy And Trust

Shard's privacy warning should become a Psionic policy surface. Sealed wire
protects data in transit. It does not protect activations from the worker that
must compute on them.

Add explicit serving privacy postures:

- `trusted_stage_activation_visible`
- `trusted_boundary_pinned`
- `untrusted_activation_visible_refused`
- `activation_obfuscation_research_only`

Boundary pinning means sensitive prompt, final logits, or user-visible decoding
authority stays on a trusted coordinator or trusted boundary stage. It does not
make middle-stage activations private.

For public claims, the safe language is:

- encrypted transport
- trusted worker routing
- activation-visible worker disclosure
- no trustless privacy guarantee

## Detailed Roadmap

### Phase 0: Freeze The Reference Contract

Status: `planned`

Owner crates: docs only, then `psionic-cluster` fixtures.

Tasks:

- preserve this audit as the Shard-to-Psionic interpretation
- add one fixture receipt modeled on Shard's fields with fake test data
- add schema comments for the future sharded run receipt
- record explicit non-goals: no marketplace authority, no trustless privacy
  claim, no Python runtime dependency

Acceptance:

- docs state the implementation boundary
- fixture receipt can be parsed by a test without opening network sockets
- final status stays `planned` until real execution exists

### Phase 1: Trusted Local Two-Stage Split

Status: `planned`

Owner crates: `psionic-cluster`, `psionic-runtime`, `psionic-models`, backend
crate selected for the first small model.

Tasks:

- define `PipelineStageRuntime` semantics
- add a two-stage session plan over one small supported model
- split layers into contiguous `[0..mid)` and `[mid..n)` ranges
- run prefill and decode through the same activation contract in-process or
  over loopback
- compare exact greedy tokens to the non-sharded same-engine path
- emit a sharded run receipt

Recommended model target:

- start with a small Qwen or Gemma family that already has the strongest local
  Psionic execution truth
- do not start with GLM-5.2 or gpt-oss-120B

Acceptance:

- 20/20 deterministic prompts match the same-engine reference
- receipt records two distinct logical stages and layer ranges
- unsupported models refuse with a typed reason
- no fallback silently runs the whole model on one stage

### Phase 2: Rust Activation Transport

Status: `planned`

Owner crate: `psionic-net`.

Tasks:

- implement `ActivationFrame` and frame codecs
- implement session-key authenticated encryption using Psionic cluster identity
  material or an explicit test-only key path
- add replay windows and sequence checks
- add decode deadlines and timeout errors
- add frame-level edge health
- add tamper, replay, wrong-stage, and malformed-frame tests

Acceptance:

- local TCP transport can carry tensor frames without untyped deserialization
- every rejected frame has a stable error code
- no secret key material appears in debug output or receipts
- the same session plan digest gates both sender and receiver

### Phase 3: N-Stage Plain Pipeline Serving

Status: `planned`

Owner crates: `psionic-cluster`, `psionic-net`, `psionic-serve`,
`psionic-runtime`.

Tasks:

- generalize two-stage execution to N ordered stages
- wire `PipelineShardedExecutionRequest` into the actual coordinator path
- add head, middle, and tail stage roles
- support tail logits return through the ordered route
- preserve stage-local KV across decode steps
- abort the full run on any stage timeout or plan mismatch
- stream tokens through `psionic-serve`

Acceptance:

- N-stage local or LAN run matches the same-engine reference
- receipt records all stages, layer ranges, stage-local cache posture, and edge
  timings
- changing the stage order changes the topology digest
- losing a stage fails closed instead of falling back silently

### Phase 4: WAN Topology Measurement And Route Selection

Status: `planned`

Owner crates: `psionic-cluster`, `psionic-net`.

Tasks:

- add active RTT probes over the same transport class used for activations
- store asymmetric link facts when measured
- implement exact route search for small candidate sets
- implement an explicit heuristic for larger candidate sets
- choose coordinator placement, stage ordering, and optional direct-return edge
  from measured facts
- add topology-selection receipts

Acceptance:

- route selection is deterministic from the same measured input
- tests cover asymmetric RTTs
- tests cover exact-vs-heuristic boundary
- receipts distinguish measured facts from inferred or configured facts

### Phase 5: Plain WAN Serving Gate

Status: `planned`

Owner crates: `psionic-serve`, `psionic-net`, `psionic-cluster`.

Tasks:

- run the N-stage plain pipeline over separate machines on a trusted network
- record public-network or tailnet topology facts honestly
- add warm and cold throughput windows
- add output hash and optional exact reference token hash
- publish a blocked receipt if hardware or networking cannot clear the gate

Acceptance:

- at least one real multi-host run emits a verifiable receipt
- receipt distinguishes LAN, tailnet, and public-WAN conditions
- output tokens match same-engine reference when reference execution is feasible
- performance claims name prompt length, generated tokens, warmup, and decode
  mode

### Phase 6: Speculative Decode Coordinator

Status: `planned`

Owner crates: `psionic-serve`, `psionic-runtime`, `psionic-models`.

Tasks:

- add draft model admission and draft capability receipts
- implement fixed-K verification through the target pipeline
- implement greedy prefix acceptance
- implement correction token emission on divergence
- implement KV crop on all stages
- record accepted tokens per traversal and correction count
- add exact parity tests against plain greedy target decode

Acceptance:

- spec decode produces identical tokens to target greedy decode
- K can be changed without changing correctness
- cache crop failures abort the session
- acceptance metrics are in the receipt

### Phase 7: Direct Return

Status: `planned`

Owner crates: `psionic-net`, `psionic-cluster`, `psionic-serve`.

Tasks:

- add a planned direct-return edge from tail to coordinator
- add direct-return frame kinds for logits and verify results
- include direct-return health in route selection
- add explicit fallback policy: abort or ordered-route fallback
- record the realized return path per run

Acceptance:

- direct return can be enabled and disabled under test
- direct-return output matches ordered-return output
- receipt records direct-return use and edge timing
- failed direct-return edge follows the declared fallback policy

### Phase 8: Async Pipelined Verification

Status: `planned`

Owner crate: `psionic-serve`.

Tasks:

- add in-flight verification depth
- add branch ids and epochs to verify requests
- discard stale verify results after divergence
- bound memory and outstanding activation bytes
- add pressure controls to avoid unbounded coordinator queues
- receipt in-flight depth, stale chunks, and acceptance

Acceptance:

- output remains exact against target greedy decode
- tests force divergence while multiple chunks are in flight
- stale chunks are discarded deterministically
- throughput receipt separates plain, spec, direct-return, and async-pipelined
  modes

### Phase 9: Backend Fast Verify

Status: `planned`

Owner crates: backend-specific runtime crates plus `psionic-runtime`.

Tasks:

- add static KV cache capability declaration
- add fixed-shape verify operation for supported backend/model pairs
- add graph capture or equivalent only where the backend can prove exactness
- add rollback by overwrite or explicit crop
- add graph-on and graph-off parity tests
- refuse unsupported dynamic shapes with a typed reason

Acceptance:

- fast verify is a capability, not a default promise
- graph-on and graph-off tokens match
- receipt records backend, graph mode, static cache shape, and fallback/refusal
- unsupported backends do not advertise fast verify

### Phase 10: Quantized Large-Model Stage Execution

Status: `planned`

Owner crates: `psionic-models`, backend crates, `psionic-runtime`,
`psionic-serve`.

Tasks:

- define model-family adapters for the first large target
- add layer-block artifact manifests for quantized weights
- add per-stage memory admission checks
- support required attention and MoE primitives for the target model
- add reference comparison strategy for same-engine reproducibility
- emit blocked receipts for unsupported quantization or kernels

Recommended sequence:

1. Qwen-family or Gemma-family small split for correctness.
2. gpt-oss-120B style layer-block serving when MXFP4 and model-family runtime
   evidence exists.
3. GLM-5.2 style serving only after MLA, MoE, NVFP4/fp8, and draft/MTP
   behavior have explicit backend support.

Acceptance:

- no large-model README claim lands without a hardware-backed receipt
- stage memory fits are validated before run admission
- quantization support is named exactly
- same-engine reproducibility is tested where feasible

### Phase 11: Self-Managing Trusted Swarm

Status: `planned`

Owner crates: `psionic-cluster`, `psionic-net`, `psionic-serve`.

Tasks:

- add join and leave handling for trusted stage workers
- add hot-spare planning
- add route refit on stage loss
- add session abort and restart receipts
- add health-driven degraded posture
- keep active decode correctness above throughput

Acceptance:

- a worker loss during a run produces either a typed abort or a typed restart
- a hot spare can be selected from the same policy input
- receipts record the original route and replacement route
- no route refit changes output without restarting from a verified state

### Phase 12: Permissionless Envelope

Status: `planned`

Owner crates: evidence hooks in Psionic; product and payment surfaces outside
Psionic.

Tasks:

- expose node capability and attestation requirements
- expose provider evidence bundles for outer systems
- support one-command worker admission only after trusted cluster posture is
  stable
- keep pricing, payout, and marketplace settlement outside Psionic
- add public disclosure for activation-visible workers

Acceptance:

- Psionic emits evidence, not payment authority
- unknown or untrusted workers are refused for activation-visible sessions
- public capability manifests distinguish trusted from permissionless posture
- trustless verification remains `planned` or `research` until solved

### Phase 13: Privacy And Hardening

Status: `planned`

Owner crates: `psionic-net`, `psionic-cluster`, `psionic-serve`.

Tasks:

- add boundary pinning policy
- add trusted-only routing policy
- add prompt/output redaction controls for receipts
- evaluate activation quantization or obfuscation only as research
- document that worker-visible activations can leak information
- add security tests for session confusion, replay, downgrade, and wrong-route
  activation injection

Acceptance:

- every activation-visible run has a privacy posture
- untrusted activation-visible routing refuses by default
- receipts avoid printing prompt text unless policy permits it
- security tests cover malicious frame inputs

## Proposed Work Item Queue

| ID | Status | Title | Primary crate(s) | Exit gate |
| --- | --- | --- | --- | --- |
| `SHARD-001` | `planned` | Add pipeline-sharded run receipt schema and fixture | `psionic-cluster` | Fixture verifies and documents proof limits |
| `SHARD-002` | `planned` | Add `PipelineStageRuntime` semantics | `psionic-runtime` | Stage contract covers prefill, decode, verify, KV crop, and session abort |
| `SHARD-003` | `planned` | Build trusted local two-stage split | runtime/backend crates | 20/20 prompts match same-engine greedy reference |
| `SHARD-004` | `planned` | Add sealed activation frames | `psionic-net` | Tamper, replay, wrong-stage, and malformed-frame tests pass |
| `SHARD-005` | `planned` | Generalize to N-stage plain pipeline | `psionic-cluster`, `psionic-serve` | N-stage run emits topology and output receipt |
| `SHARD-006` | `planned` | Add measured topology route optimizer | `psionic-cluster`, `psionic-net` | Deterministic route choice from asymmetric RTT matrix |
| `SHARD-007` | `planned` | Produce first real trusted multi-host receipt | all cluster path crates | Receipt distinguishes LAN/tailnet/public WAN and reference parity |
| `SHARD-008` | `planned` | Add fixed-K speculative verification | `psionic-serve`, runtime crates | Tokens match target greedy decode |
| `SHARD-009` | `planned` | Add direct-return tail channel | `psionic-net`, `psionic-serve` | Direct-return and ordered-return outputs match |
| `SHARD-010` | `planned` | Add async pipelined verification | `psionic-serve` | Divergence with in-flight chunks remains exact |
| `SHARD-011` | `planned` | Add backend fast-verify capability | backend crates | Graph/static-cache path matches eager path |
| `SHARD-012` | `planned` | Add first large quantized stage target | model/backend crates | Hardware-backed large-model receipt or typed refusal |
| `SHARD-013` | `planned` | Add trusted-swarm refit and restart receipts | `psionic-cluster`, `psionic-net` | Worker loss produces typed abort or restart |
| `SHARD-014` | `planned` | Add activation-visible privacy policy | `psionic-serve`, `psionic-cluster` | Every sharded run declares privacy posture |

## Test Strategy

Unit tests:

- topology solver exact route on small matrices
- asymmetric RTT handling
- activation frame encode/decode
- tamper and replay rejection
- stage-plan digest mismatch rejection
- layer-range validation
- unsupported quantization refusal
- receipt digest stability

Integration tests:

- two-stage local split equals same-engine reference
- N-stage local split equals same-engine reference
- direct-return equals ordered-return
- fixed-K spec decode equals greedy target
- async spec decode equals greedy target under forced divergence
- frame timeout aborts the full session
- dropped stage produces typed abort

Hardware or lab receipts:

- trusted LAN plain N-stage run
- trusted WAN or tailnet plain N-stage run
- trusted WAN spec decode run
- direct-return comparison run
- backend fast-verify graph-on/off parity run
- large-model blocked or passed receipt

Simulation tests:

- high RTT and low RTT route selection
- stale in-flight speculative chunks
- degraded edge fallback
- hot-spare route refit
- activation byte budget pressure

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Trustless verification is unsolved for arbitrary workers | Permissionless compute claims can overstate security | Keep receipts honest: same-engine reproducibility and node evidence, not proof of every internal FLOP |
| Stage workers see activations | User privacy can be misrepresented | Add explicit activation-visible privacy posture and refuse untrusted routing by default |
| Fast verify is backend-specific | A graph optimization can silently change semantics | Require graph-on/off parity and advertise capability per backend/model pair |
| Speculative pipelining can corrupt KV after divergence | Output can differ from greedy target | Use epochs, branch ids, crop/gather receipts, and exact parity tests |
| Large quantized models have model-family-specific kernels | GLM/gpt-oss claims can outrun runtime support | Add model-family admission gates and blocked receipts |
| WAN topology is unstable | Performance claims can be non-reproducible | Receipt measured RTT matrix, route, time window, warmup, and fallback behavior |
| NAT and relay are not solved by the prototype | One-command public worker admission can fail | Treat rendezvous/relay as a later `psionic-net` deliverable |
| Whole-model fallback can fake success | Sharded demos become untrustworthy | Stage workers must refuse layers outside their range; receipts must record residency |

## Definition Of Done For Psionic Parity

Psionic reaches Shard-style parity only when all of the following are true:

- a real multi-host sharded inference run executes through Psionic-owned crates
- no worker holds the entire target model unless the plan declares a replicated
  or fallback lane
- activation transport is authenticated, encrypted, replay-protected, and typed
- measured topology facts drive route selection
- exact greedy parity is checked for correctness gates
- speculative decode remains token-identical to target greedy decode
- direct return and async pipelining have separate receipts
- backend fast paths declare graph/static-cache capability per model/backend
  pair
- large-model claims have hardware-backed receipts or remain typed refusals
- privacy posture states that trusted stage workers can see activations
- outer marketplace and payout systems consume Psionic evidence instead of
  becoming Psionic core authority

Until then, the honest status is `planned` for full Shard-style WAN pipeline
serving and `implemented_early` only for the existing planning and evidence
substrate.

# 2026-04-09 Psionic CommitLLM Adaptation Audit

Status: repo analysis  
Scope date: 2026-04-09

Open-weight inference has an execution-integrity problem that is already large
enough to matter for product design. A buyer, validator, or downstream system
can often see the returned text, the provider identity, and some serving
metadata. That is not enough to establish which model actually ran, which
quantization or adapter state was active, which decode policy produced the
answer, whether the route or cluster placement matched the claim, or whether
the provider rewrote the answer after decode. Those gaps matter in normal
commercial settings and matter even more in decentralized or semi-trusted
compute markets.

`Psionic` already has stronger execution truth surfaces than most inference
systems. It owns the runtime, scheduler, route, and clustered execution path in
one codebase. It already publishes provenance and delivery-proof material
through `GenerationProvenance`, `ExecutionDeliveryProof`,
`SettlementLinkageInput`, `ClusterEvidenceBundlePayload`, `x-psionic-*`
headers, and `psionic_cluster_execution` payloads. That means this repo is not
starting from a blank "add some receipts" position. It already has a concrete
language for execution truth. The open problem is how far those surfaces can be
upgraded from useful accountability metadata into auditable execution claims.

The newly synced `CommitLLM` reference is relevant because it is a serious
attempt to close that gap without requiring a heavy proof object for every
response. It uses a commit-and-audit design: the provider stays on a normal GPU
serving path, emits a compact receipt, and later answers verifier challenges by
opening selected trace regions. The verifier checks those openings on CPU
against committed model identity, deployment semantics, decode policy, and
delivered output behavior. That puts `CommitLLM` in the practical middle ground
between weak fingerprinting and expensive proof systems.

`Psionic` is not in the same implementation position as `CommitLLM`. That
project has to retrofit verification discipline onto an existing serving stack.
`Psionic` owns its own runtime in Rust. It can define auditable execution
classes, trace boundaries, route and cluster receipts, and verifier-facing data
structures inside the engine rather than layering them on from the outside.
That ownership creates real opportunities, but it does not remove the hard
parts. GPU attention exactness, prefix reuse, speculative decode, sparse or
MoE routing, audit freshness, and hostile-network economics still need explicit
protocol design.

This audit exists to answer one concrete repo question: which parts of
`CommitLLM` should `Psionic` adapt, which parts should it avoid copying
directly, and which opportunities become available specifically because
`Psionic` owns the runtime, scheduler, and cluster path itself. The goal is not
to produce a generic survey of verifiable inference. The goal is to identify a
credible `Psionic` adaptation path that fits the repo's actual architecture and
claim surfaces.

## Scope

This note follows the broader 2026-04-09 inference-verifiability work and
narrows the scope to `Psionic` itself:

> given the newly synced `lambdaclass/CommitLLM` reference, and given that
> `Psionic` owns its own inference runtime in Rust rather than wrapping an
> external serving engine, what should `Psionic` actually adapt from
> `CommitLLM`, what becomes easier because of that ownership, and what still
> remains hard?

The relevant local reference surfaces are:

- the synced `CommitLLM` reference clone
- `crates/psionic-serve`
- `crates/psionic-runtime`
- `docs/INFERENCE_ENGINE.md`

This is not a claim that `Psionic` should copy `CommitLLM` wholesale.
It is an audit of:

- the verifier ideas worth porting
- the places where `Psionic` has an architectural advantage
- the parts of the problem that remain difficult even with full runtime control

## Short Answer

Yes, full ownership of the inference engine materially improves the situation.

It does **not** make verifiable inference easy in the absolute sense, and it
does **not** remove the hard problems around attention nondeterminism, prefix
state, speculative decode, or hostile-node economics.

But it does create a real advantage over a retrofit design like `CommitLLM` in
five ways:

- `Psionic` can make receipt and verifier semantics native runtime concepts
  instead of sidecar or hook artifacts
- `Psionic` can define an explicit fail-closed audited execution class at the
  scheduler and kernel boundary
- `Psionic` can bind clustered execution truth directly into receipts because
  it already owns route, shard, and handoff semantics
- `Psionic` can keep runtime capture, receipt generation, and verifier replay
  in one Rust type system instead of splitting them across serving and audit
  stacks
- `Psionic` can introduce verification-aware kernels, deterministic submodes,
  and auditable fast-path restrictions without waiting for an upstream engine

`Psionic` should not use `CommitLLM` as-is.

`Psionic` should:

- borrow `CommitLLM`'s commit-and-audit discipline
- borrow its guarantee vocabulary
- borrow its threat model and red-team habits
- but implement the resulting audited lane as a first-class `Psionic`
  execution mode rather than a serving overlay

## What CommitLLM Gets Right

`CommitLLM` is worth taking seriously because it solves a real gap between:

- weak signed receipts or fingerprint heuristics
- and expensive proof systems that are not practical for production LLM serving

Its best ideas are conceptual, not incidental.

### 1. It binds the whole inference surface

`CommitLLM`'s main contribution is full-surface binding.
The protocol insists that verification must bind the whole answer-producing
surface:

- input preprocessing
- tokenizer and chat-template semantics
- model and quantization identity
- adapter identity
- decode policy
- randomness discipline
- output cleanup and formatting

That is exactly the level of truth an inference receipt needs if we care about:

- model substitution
- silent requantization
- hidden prompt drift
- decode-policy drift
- post-generation answer rewriting

### 2. It separates exact, approximate, and statistical guarantees

`CommitLLM` is unusually honest about its guarantee boundary.

It says, in effect:

- some parts are exact
- some parts are bounded but approximate
- some parts are statistical unless deeper audit is used
- unsupported semantics fail closed

That vocabulary is extremely valuable for `Psionic`.
It is better than generic "verified" language because it forces the product and
runtime to describe what is actually being claimed.

### 3. It treats audit tiers as a first-class part of the design

Routine audit versus deep audit is not a docs flourish.
It is a systems design decision:

- cheap frequent checks most of the time
- expensive stronger openings when randomly sampled, disputed, or high-value

That posture fits marketplace reality much better than "everything is either
fully verified or not."

### 4. It thinks adversarially, not only structurally

The red-team attack matrix in `CommitLLM` matters almost as much as the
protocol design itself.

It explicitly treats the following as normal things a cheating provider might
try:

- receipt tampering
- trace forgery
- cross-request splicing
- decode-output tampering
- model substitution
- replay or freshness abuse

That is the correct verification mindset for `Psionic`.

## Where Psionic Already Has Better Starting Position

`Psionic`'s advantage comes from runtime ownership.
Rust matters because the same repo already owns execution truth rather than
only observing someone else's execution path.

### 1. Psionic already has native provenance types

This is not hypothetical future infrastructure.

`Psionic` already has:

- `GenerationProvenance` in `psionic-serve`
- `ExecutionDeliveryProof` in `psionic-runtime`
- `SettlementLinkageInput` in `psionic-runtime`
- `ClusterEvidenceBundlePayload` in `psionic-runtime`
- `x-psionic-*` publication headers and `psionic_cluster_execution` payloads in
  the OpenAI-compatible server

Those surfaces already encode:

- served artifact identity
- execution-plan digest
- runtime backend
- load state
- delivery-proof counts
- cache observations
- structured-output mode
- cluster execution context
- route worker and route strategy
- scheduling class
- prefix-cache posture

That means `Psionic` is not starting from zero. It already has a receipt
language. The current gap is that the language is richer than the proof
discipline behind it.

### 2. Psionic owns the scheduler and route truth

`CommitLLM` has to recover execution truth from a kept serving path.

`Psionic` directly owns:

- route selection
- worker identity
- batch posture
- scheduling class
- prefix-cache state
- cluster topology
- selected nodes
- pipeline stages
- shard handoffs

That matters because many important distributed lies include more than wrong
weights:

- wrong worker
- wrong route locality
- wrong fallback path
- wrong cluster placement
- fake shard handoff
- fake cache reuse claim

`Psionic` can bind those directly because they are native engine facts, not
external observations.

### 3. Psionic owns clustered execution as a first-class runtime concern

This is the biggest opportunity that `CommitLLM` does not obviously solve as
well.

`Psionic` already covers more than a single-host serving wrapper.
Its admitted runtime story includes:

- remote whole-request execution
- replicated execution
- pipeline sharding
- layer sharding
- tensor sharding
- sparse expert topologies

Because of that, `Psionic` can make verification cluster-native from the
beginning.

That means receipts can eventually bind not only:

- which model ran

but also:

- which workers held which shard artifacts
- which stage produced which handoff
- which placement digest was active
- which expert route was admitted
- which nodes signed which segment of the trace

That is a real strategic advantage over a verifier design that starts from
"one provider, one GPU path, one audit transcript."

### 4. Psionic can keep execution and verification in one type system

Owning the runtime in Rust creates a practical engineering advantage:

- one schema for execution receipts
- one schema for cluster evidence
- one verifier library reusing the same data structures
- one canonical serialization story
- one refusal and capability vocabulary

That reduces a major class of failure:

- runtime says one thing
- verifier replays another thing
- product docs describe a third thing

Retrofit systems drift here easily. `Psionic` has a chance not to.

### 5. Psionic can create explicit audited modes instead of auditing every mode

This is the most important near-term product opportunity.

Because `Psionic` owns the runtime, it does not need to pretend that every
serving feature is equally auditable.

It can define one or more explicit audited execution classes such as:

- `audited_dense_greedy`
- `audited_dense_seeded`
- `audited_cluster_pipeline`

and require those classes to obey stricter constraints than the fast path:

- fixed kernel family
- fixed quantization semantics
- fixed or explicit seed discipline
- refusal on unsupported speculative modes
- refusal on unsupported cross-request cache reuse
- refusal on unsupported sampler features

That is much cleaner than trying to retrofit proof language across every
serving optimization at once.

## What Owning Rust Runtime Makes Easier Specifically

The phrase "written in Rust" matters less than the execution consequences of
full ownership.

### 1. Engine-native trace ABI

`Psionic` can define an internal trace ABI for audited execution:

- per-step commitment payloads
- hidden-state or bridge-boundary digests
- KV transcript digests
- stage-handoff digests
- sampler inputs and outcomes

Those can be emitted directly by the runtime rather than scraped by hooks.

This should improve:

- correctness
- performance predictability
- forward-compatibility
- fail-closed behavior

### 2. Verification-aware kernel boundaries

`Psionic` can choose where the exact claim boundary begins and ends.

For audited modes, it can standardize specific boundary states such as:

- pre-attention bridge state
- post-attention output state
- pre-final-norm residual
- sampler input logits

That opens the door to a cleaner audited-mode contract than "whatever the
underlying engine happened to expose."

### 3. Shared verifier library

`Psionic` can create a native verifier crate rather than treating verification
as an external tool.

That verifier can consume:

- `GenerationProvenance`
- `ExecutionDeliveryProof`
- `ClusterEvidenceBundlePayload`
- future audited-trace objects

and verify:

- receipt integrity
- claim completeness
- challenge openings
- route and cluster signatures
- policy compliance for the audited lane

This is a major product advantage because the verifier can be shipped in many
forms:

- CLI
- embedded service
- marketplace auditor
- buyer-side library
- eventually WASM or mobile-capable verifier paths where feasible

### 4. Canonical audited scheduler lane

Because `Psionic` owns scheduling, it can make auditable execution a scheduler
decision rather than post-hoc verification metadata.

That means:

- explicit queueing and admission for audited jobs
- explicit refusal when the runtime cannot satisfy audit requirements
- explicit cost and latency publication for audited versus unaudited lanes

This makes the economics and product story much clearer.

### 5. Multi-node signed segment receipts

This is likely the best net-new opportunity.

For clustered inference, `Psionic` can require each stage or node to sign its
segment claim:

- input digest received
- shard artifact digest used
- output digest produced
- handoff digest forwarded
- local kernel or cache counters
- local timestamp window

The final clustered receipt can then aggregate these into one
request-level bundle.

That gives `Psionic` a plausible near-term route to stronger distributed
accountability than a single-provider receipt ever could.

## What Psionic Should Adapt Directly

### 1. Adopt CommitLLM's threat model explicitly

`Psionic` should import the concrete attack categories as well as the protocol
style.

At minimum, audited inference should explicitly defend against:

- model substitution
- quantization/config drift
- hidden prompt or preprocessing drift
- decode-policy drift
- output rewriting
- cross-request splicing
- freshness replay
- selective audit denial
- fake cluster placement or fake handoff claims

### 2. Extend provenance into an auditable inference manifest

The current `GenerationProvenance` surfaces are strong, but they are not yet
the full answer-producing manifest that `CommitLLM` pushes toward.

`Psionic` should add one explicit auditable manifest object that binds:

- input semantics
- model identity
- quantization identity
- adapter identity
- decode policy
- output policy
- route and cluster posture
- audited execution class
- verifier policy version

This should become the canonical receipt payload for audited inference.

### 3. Introduce routine and deep audit tiers

`Psionic` should not talk about one generic "verified inference" mode.

It should define at least:

- receipt-only accountability
- routine audit
- deep audit
- dispute-escalation audit

Those tiers should differ in:

- required retained state
- verifier workload
- bandwidth/storage cost
- coverage guarantees

### 4. Add fail-closed support matrix for audited modes

This should become part of the product contract, not buried implementation
detail.

For each audited execution class, `Psionic` should publish:

- supported decode features
- unsupported decode features
- supported cache behavior
- unsupported cache behavior
- supported topology classes
- unsupported topology classes
- exact versus approximate versus statistical guarantee regions

### 5. Build a red-team audit suite

`CommitLLM` is correct to treat adversarial testing as core, not optional.

`Psionic` should add explicit attack campaigns for:

- stale receipt replay
- output tampering after decode
- route-worker substitution
- placement digest forgery
- fake shard handoff
- prefix/KV injection
- decode-downgrade attacks
- unsupported-fast-path misuse under audited labels

## What Psionic Should Not Copy Blindly

### 1. Do not inherit vLLM-specific capture assumptions

`CommitLLM` necessarily reflects the serving stack it sits beside.
`Psionic` should borrow the proof posture, not the implementation accidents of
that stack.

In particular, `Psionic` should avoid reintroducing a hook-fragile architecture
when it can define native capture boundaries itself.

### 2. Do not pretend one verifier story covers every runtime feature

`Psionic` should resist the temptation to market every path as equally
auditable.

The better move is:

- fewer audited modes
- stronger claims
- explicit refusal outside those modes

### 3. Do not over-index on single-node assumptions

`CommitLLM` is strongest as a single-provider open-weight verifier reference.
`Psionic` should take advantage of its cluster-native architecture instead of
treating distributed execution as an afterthought.

### 4. Do not confuse Rust ownership with solved numerics

Rust helps with control, schemas, verifier implementation, and receipt hygiene.

It does not by itself solve:

- GPU floating-point nondeterminism
- approximate attention replay
- prefix-state trust
- speculative decode verification
- MoE routing verification

Those still need real protocol design.

## What Still Remains Hard Even With Full Control

This is the honest boundary.

### 1. Attention exactness is still hard

If audited `Psionic` still relies on vendor FP16/BF16 attention kernels, then
the exactness gap remains.

Full runtime ownership makes it easier to:

- isolate the gap
- define the exact claim boundary
- create deterministic submodes later

but not to wish the problem away.

### 2. Prefix caching remains a major verification seam

Cross-request prefix reuse is economically important and verification-hostile.

`Psionic` can handle this more honestly than many systems by:

- refusing it in audited modes at first
- or promoting prefix cache entries into first-class committed objects later

But it is still real work.

### 3. Speculative decoding needs its own protocol

If `Psionic` wants audited speculative or multi-token execution, that needs
draft-model, accept-mask, and transcript semantics designed on purpose.

This should not be treated as a free extension of basic audited decoding.

### 4. Sparse/MoE verification is a separate frontier

`Psionic` is already interested in sparse topologies.
That makes routing verification, expert selection, and expert-shard identity
first-class verification problems.

That is a major opportunity, but also a major protocol surface.

### 5. Hostile-network economics do not disappear

Even with strong receipts, the network still needs:

- audit policy
- challenge policy
- freshness windows
- denial-of-audit consequences
- fraud labels
- reputation or throttling policy

Runtime ownership helps generate better evidence.
It does not replace governance.

## Recommended Psionic Adaptation Plan

### Phase 0: clean guarantee language

Before new protocol work, `Psionic` should standardize one guarantee vocabulary
for all auditable lanes:

- exact
- approximate
- statistical
- fail-closed

### Phase 1: auditable manifest and signed execution receipt

Build first:

- auditable inference manifest
- signed execution receipt digest
- explicit audited execution classes
- refusal on unsupported semantics

This should use existing provenance surfaces rather than inventing a parallel
receipt universe.

### Phase 2: native trace ABI and verifier crate

Build next:

- audited trace schema
- retained-state policy
- Rust verifier crate
- challenge/open flow for routine and deep audits

### Phase 3: cluster-native segment receipts

Build after the single-node audited lane is stable:

- per-stage signatures
- handoff digests
- placement digest binding
- aggregated clustered execution receipt

This is where `Psionic` can surpass retrofit verifier systems.

### Phase 4: deterministic or stronger audited submodes

Only after the above:

- deterministic audited kernels where worth it
- stronger prefix-state commitments
- speculative transcript support
- MoE or sparse expert audited lanes

## Bottom Line

`CommitLLM` is the best current external reference available in the broader
workspace for practical open-weight inference verification.

But `Psionic` has an architectural advantage that `CommitLLM` does not:

- it owns the runtime
- it owns the scheduler
- it owns the cluster path
- it already owns receipt and provenance types

That means `Psionic` should not merely imitate `CommitLLM`.

It should adapt the core ideas:

- bind the whole inference surface
- use explicit audit tiers
- fail closed outside supported audited modes
- red-team the verifier path

and then push further in the places where full runtime ownership matters most:

- native trace ABI
- shared verifier library
- audited scheduler classes
- cluster-native segment receipts

Rust does not make the hard math problems disappear.
It does make the system much easier to shape honestly.

That is enough to create a real opportunity:

`Psionic` can plausibly build a cleaner, more integrated, and more
cluster-capable audited inference lane than a retrofit verifier stack can.

## Primary Files Reviewed

- `../competition/repos/CommitLLM/README.md`
- `../competition/repos/CommitLLM/article/commitllm.md`
- `../competition/repos/CommitLLM/roadmap.md`
- `../competition/repos/CommitLLM/redteam/attack_matrix.md`
- `../competition/repos/CommitLLM/sidecar/verilm/server.py`
- `../competition/repos/CommitLLM/scripts/modal/serve.py`
- `docs/INFERENCE_ENGINE.md`
- `crates/psionic-serve/src/lib.rs`
- `crates/psionic-serve/src/openai_http.rs`
- `crates/psionic-runtime/src/lib.rs`

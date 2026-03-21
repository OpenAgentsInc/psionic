# Tassadar Plugin-System And Turing-Completeness Audit

Date: 2026-03-20

## Purpose

This audit reviews the draft plugin-system spec at
`~/code/alpha/tassadar/plugin-system.md` against the current public Psionic
repo state and the current post-`TAS-186` Turing-completeness plan.

It answers three concrete questions:

1. What does the plugin-system spec actually require?
2. Which parts of that system already exist in Psionic/Tassadar, and which do
   not?
3. If the long-term goal is for the Turing-complete Tassadar path to implement
   that plugin system, does that change the shape of the post-`TAS-186`
   Turing-completeness push?

Companion audit:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`

## Source Set

Primary external design inputs:

- `~/code/alpha/tassadar/plugin-system.md`
- `~/code/alpha/tassadar/vm.md`
- `~/code/alpha/tassadar/tassadar-gap-analysis.md`

Primary repo surfaces reviewed:

- `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- `docs/audits/2026-03-18-tassadar-post-tas-102-final-audit.md`
- `docs/audits/2026-03-19-tassadar-general-internal-compute-red-team-audit.md`
- `docs/audits/2026-03-19-tassadar-pre-closeout-universality-audit.md`
- `docs/audits/2026-03-19-tassadar-turing-completeness-closeout-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`

Primary machine-readable repo surfaces reviewed:

- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_audit_report.json`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`
- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_manager_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_route_policy_report.json`
- `fixtures/tassadar/reports/tassadar_internal_component_abi_report.json`
- `fixtures/tassadar/reports/tassadar_session_process_profile_report.json`
- `fixtures/tassadar/reports/tassadar_spill_tape_store_report.json`
- `fixtures/tassadar/reports/tassadar_virtual_fs_mount_profile_report.json`
- `fixtures/tassadar/reports/tassadar_simulator_effect_profile_report.json`
- `fixtures/tassadar/reports/tassadar_async_lifecycle_profile_report.json`
- `fixtures/tassadar/reports/tassadar_effectful_replay_audit_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_portability_report.json`

Relevant public issue waves reviewed:

- `TAS-058` through `TAS-068`
- `TAS-101` through `TAS-111`
- `TAS-126`
- `TAS-129` through `TAS-136`
- `TAS-141`
- `TAS-150` through `TAS-156`
- `TAS-157` through `TAS-186`

## Executive Verdict

Current audit verdict:

- software-module, mount, effect, replay, and publication substrate below the
  plugin system for bounded operator/internal posture: `implemented`
- plugin-specific Wasm manifest, packet ABI, runtime API, receipt family, and
  hot-swap contract: `partial`
- weighted controller that selects and sequences plugins while preserving
  owned-route control logic: `planned`
- plugin-system-complete route built on the canonical post-`TAS-156A` owned
  Transformer path: `planned`
- broad served/public plugin platform: `partial_outside_psionic`

The short answer to the user's question is:

- no, the plugin system should **not** become a prerequisite for the core
  post-`TAS-186` Turing-completeness rebase
- yes, it **does** change the shape of that push if the goal is not just
  bounded universality, but a productizable weighted software platform

The right move is:

1. finish `TAS-182` through `TAS-186`
2. land the post-article Turing-completeness rebase from the companion audit
3. make that rebase plugin-aware at the boundary level
4. then land a separate plugin-system tranche on top of it

That sequencing keeps the Turing-completeness claim honest while still steering
the system toward the plugin model.

## Computability, Programmability, And Productization

The cleanest way to keep the plugin story honest is to distinguish three
different things the repo may eventually prove:

- computability:
  the post-`TAS-186` Turing-completeness rebase would prove that the canonical
  owned route carries bounded universal computation under declared continuation
  semantics
- programmability:
  the plugin tranche would prove that the same route can sit above a bounded
  software-capability layer with stable artifacts, manifests, envelopes,
  receipts, and model-visible control surfaces
- productization:
  later promotion, authority, publication, and governance gates would decide
  whether that programmable layer is usable beyond bounded operator/internal
  posture

Those are different claim classes. The Turing-completeness rebase buys the
first. The plugin tranche buys the second. The governance and publication
machinery decides the third.

## Cross-Tranche Invariants

Both this audit and the companion Turing-completeness audit should be read
under the same invariants.

- Plane Separation:
  data plane, control plane, and capability plane remain explicit with no
  hidden cross-plane leakage
- State Ownership:
  durable workflow truth must live only in explicit weights-owned, ephemeral,
  resumed, or host-backed state classes
- Control Ownership:
  host may execute declared mechanics, but host may not decide workflow
- Control-Plane Provenance:
  every decision-bearing trace must bind branch, retry, stop, and later
  capability-choice outcomes back to model outputs, canonical route identity,
  and replayable control evidence
- Semantic Preservation:
  adapters, continuation mechanics, marshalling, and reinjection must preserve
  declared meaning or fail closed
- Carrier Separation:
  direct article-equivalent, bounded resumable universality, and later
  plugin-capability claims remain distinct carriers
- Choice-Set Integrity:
  admissible choices may not be hidden, pre-ranked, filtered, or rewritten
  off-trace
- Resource Transparency:
  latency, cost, quota, availability, and pool pressure that affect branching
  must be model-visible or fixed by contract
- Scheduling Ownership:
  ordering, concurrency, and result-visibility timing must be model-decided or
  fixed as a declared runtime contract
- Determinism Classes:
  decision traces must declare whether they are `strict_deterministic`,
  `seeded_stochastic`, or
  `bounded_nondeterministic_with_equivalence_class`
- Failure Semantics:
  refusal, policy failure, semantic failure, capability failure, resource
  exhaustion, and determinism violation must remain distinct typed classes
- Time Semantics:
  logical time, wall time, time observability, and time-driven branching must
  be declared explicitly
- Information Boundary:
  every model-visible signal that can shape control flow must be declared
- Training-Inference Boundary:
  model updates, runtime adaptation, telemetry, caches, and logs must stay in
  explicit classes
- Canonical Machine Identity:
  one named machine identity tuple must bind model id, weight digest, route
  digest, continuation contract, and carrier class for every proof, witness,
  benchmark, and audit
- Execution-Semantics Equivalence:
  proof-bearing route changes must preserve declared transition semantics or an
  explicit proof abstraction boundary, not merely output parity
- Downward Non-Influence:
  capability or served layers may not silently rewrite lower-plane compute,
  continuation, or proof assumptions
- Computational Model Statement:
  every claim-bearing artifact must name the computational model,
  continuation semantics, and effect boundaries it relies on
- Proof Class Distinction:
  mechanistic proof artifacts and observational audits must remain separate
  artifact classes
- Hidden State Channels:
  all state that can influence control flow must live in weights, declared
  continuation state, or explicit receipt-backed host state
- Machine Minimality:
  minimality must be defined over machine components, semantics, and preserved
  transforms rather than implied loosely from route shape
- Observer Model:
  verifier roles, trust assumptions, and acceptance conditions must be
  machine-readable
- Machine Closure Bundle:
  no terminal universality, plugin, or platform claim is valid unless it
  references one canonical closure bundle by digest
- Served Conformance:
  served posture may deviate from operator truth only inside an explicit
  conformance envelope and otherwise must fail closed

## Three-Plane Contract

The cleanest way to keep the plugin architecture honest is to freeze three
separate planes:

- data plane:
  `TCM.v1` and the canonical owned compute route carry pure compute evolution
- control plane:
  the weighted decision trace carries plugin selection, branching, retry, and
  stop logic
- capability plane:
  bounded plugins execute declared capabilities under envelopes and receipts

No plane may silently absorb another plane's responsibilities.

That means:

- the capability plane may not quietly become the planner
- the control plane may not be hidden inside packet schemas or runtime
  admissibility logic
- the data plane may not be widened into arbitrary capability execution by
  implication
- higher planes may not push downward requirements that silently redefine the
  lower-plane machine model, continuation semantics, or proof assumptions

## Adversarial Host Model

The audits should assume a host that is allowed to try to cheat unless a rule
forbids it.

The host may attempt to:

- reorder execution
- cache and substitute results
- inject ranking or routing heuristics
- hide candidates or refusal modes
- exploit latency, cost, quota, or pool asymmetries
- adapt scheduling and concurrency

The system must therefore either:

- surface these behaviors explicitly in the model-visible trace and receipts
- freeze them as non-adaptive runtime contract
- or refuse them outright

## Stability Risks That Need Explicit Artifacts

Reading the plugin and Turing-completeness audits together exposes a second
class of risk:

- Dual Truth Carrier Drift:
  proofs, witnesses, route claims, and plugin execution claims can drift back
  onto different underlying machines unless one canonical machine identity lock
  binds them
- Proof Versus Execution Semantics Split:
  output parity can stay green while batching, caching, continuation, or fast
  routes silently change the semantics the proof relied on
- Continuation Inflation:
  resumable execution can quietly become a second machine unless continuation
  is constrained to extend execution without adding expressivity
- Fast-Route Legitimacy Drift:
  production can migrate to a fast route that is benchmark-green but no longer
  the same proof-bearing machine
- Plugin Backpressure Into Core Semantics:
  plugin ergonomics can pressure the bridge into changing continuation,
  carrier, or proof assumptions from above
- Choice-Set Neutrality Drift:
  surfacing "all choices" is not enough if ordering, latency, cost, or
  soft-failure effects still bias effectively equivalent options
- Control-Plane Proof Gap:
  the repo can forbid host control in prose while still lacking one
  machine-readable proof that decisions remain weight-owned across the stack
- Operator Versus Served Drift:
  served/plugin posture can drift away from operator truth unless a conformance
  envelope freezes what is allowed to differ
- Computational Model Ambiguity:
  compute, continuation, and capability artifacts can all sound correct while
  referring to different machines

Those risks are why the plugin lane needs explicit anti-drift artifacts, not
only a manifest and runtime API.

## Primary Claim Dependency

The plugin system is only valid if weighted control ownership is proven on the
canonical route.

All other plugin work is subordinate to that dependency:

- manifest identity
- runtime API
- receipts
- conformance harnesses
- promotion and publication machinery

Without weighted control ownership, those surfaces may still describe useful
bounded software components, but they do not yet add up to the claimed weighted
plugin system.

## The Missing Proof Surface: Control-Plane Ownership And Decision Provenance

Across the full stack, the repo now has:

- compute proof:
  `TCM.v1`, the universal-machine construction, and the witness suite
- route and execution proof:
  the canonical owned-route article lane plus semantics and continuation audits
- capability substrate:
  manifests, packets, receipts, runtime contracts, and plugin envelopes

What it still needs as its own object before a plugin controller claim is:

- control-plane ownership and decision provenance proof:
  one machine-readable proof that workflow decisions remain weight-owned on the
  canonical route

That proof surface must positively show:

- decision provenance from model output to canonical route identity
- choice-set completeness and neutrality inside the declared equivalent-choice
  class
- absence or explicit surfacing of hidden control channels
- replayability of branch, retry, and stop decisions before later plugin
  sequencing is added

The later weighted plugin-controller trace should extend that proof surface,
not substitute for it.

To become a real proof surface rather than a bundle of policy prose, that
artifact must also freeze a concrete formal-object set:

- a determinism class contract
- an equivalent-choice relation with admissible divergence bounds
- a failure-semantics lattice
- a time-semantics contract for logical versus wall time
- a model-information boundary
- a training-versus-inference boundary
- a proof-versus-audit distinction
- a machine-minimality definition
- a hidden-state-channel closure rule
- and one observer or acceptance model

## The Missing Composition Layer: Canonical Machine Closure Bundle

Even if the bridge, control proof, plugin substrate, controller trace, and
anti-drift contracts all land cleanly, the repo can still fail to close as one
machine if those artifacts remain only loosely related.

What is still needed is one canonical machine closure bundle that binds:

- machine identity tuple
- computational model statement
- determinism contract
- continuation contract
- control-plane provenance proof
- execution-semantics equivalence class
- carrier split contract
- state ownership and hidden-state closure
- failure lattice
- observer and acceptance model
- portability and minimality bindings
- proof-carrying artifacts by digest rather than by doc title
- proof-versus-audit classification
- invalidation conditions for when the closure stops being valid

The important hard rule is:

no terminal universality, plugin, or platform claim should be treated as valid
unless it references this exact closure bundle by digest.

That is the difference between:

- a system that has the right proofs, audits, receipts, and controller traces
- and a closed machine whose claims all bind to the same object

## What The Plugin-System Spec Actually Requires

The draft plugin-system spec is not asking for "arbitrary tools" in a vague
sense. It is asking for one very specific architecture:

- weighted control logic in the model
- Wasm capability components as hot-swappable software artifacts
- host-owned bounded execution with explicit policy and receipts
- explicit separation between reasoning ownership and capability ownership

The key design law in the spec is:

- model owns decision logic
- runtime owns execution contract
- plugin owns black-box capability implementation

That means the plugin system is not merely:

- more broad internal compute
- more module packaging
- more effect profiles
- or more Turing-completeness evidence

It is a software-platform layer above those substrates, with one hard extra
requirement:

the host must not quietly become the owner of orchestration logic.

That requirement is exactly why the plugin system intersects with the
post-`TAS-186` ownership and route-minimality work.

## What Already Exists In Psionic

Large parts of the plugin-system substrate are already present in bounded
operator/internal form.

### 1. Module And Package Substrate: `implemented`

These old TAS lanes already give Psionic most of the software-component
substrate a plugin system would need:

- `TAS-058`:
  deterministic module linker and dependency-graph semantics
- `TAS-059`:
  installed-module evidence bundles and decompilation receipts
- `TAS-060`:
  module catalog and reusable primitive library surfaces
- `TAS-061`:
  overlap resolution and mount-aware import selection
- `TAS-134`:
  bounded internal-compute package manager and routeable named packages
- `TAS-135`:
  cross-profile link compatibility and downgrade/refusal behavior
- `TAS-136`:
  installed-process lifecycle with migration and rollback receipts

This means the repo already understands:

- digest-bound component identity
- dependency graphs
- package resolution
- compatibility and downgrade planning
- lifecycle, rollback, and evidence posture

Those are real plugin-adjacent building blocks.

### 2. Capability And Policy Substrate: `implemented`

The plugin spec's world-mount and capability model also maps strongly onto
already-landed Psionic surfaces:

- `TAS-062`:
  typed host-call and import policy matrix
- `TAS-063`:
  trust-tier isolation and privilege-escalation refusal
- `TAS-064`:
  promotion lifecycle, quarantine, and revocation posture
- `TAS-066`:
  self-installation policy and rollback gate
- `TAS-068`:
  world-mount compatibility and mount-time negotiation
- `TAS-101` and `TAS-102`:
  portability envelopes, publication posture, and route policy
- `TAS-110` and `TAS-111`:
  subset promotion and backend/toolchain portability matrices

This means the repo already has real machinery for:

- explicit mounted capability posture
- trust and publication boundaries
- route policy
- portability gating
- profile-specific suppression
- quarantine and revocation

That is much closer to the plugin spec than a greenfield system would be.

### 3. Effect, Session, And Replay Substrate: `implemented`

The plugin spec requires:

- bounded external interaction
- deterministic or declared replay classes
- explicit cancellation and timeout posture
- durable state only in host-backed stores

Psionic already has direct analogs:

- `TAS-126`:
  bounded session-process profile with explicit refusal on open-ended external
  streams
- `TAS-129`:
  deterministic virtual filesystem and artifact-mount effect profile
- `TAS-130`:
  deterministic clock, randomness, and simulator-backed network profiles
- `TAS-131`:
  async call, interrupt, retry, and cancellation semantics
- `TAS-132`:
  effectful replay, challenge, and audit receipts
- `TAS-127`, `TAS-136`, and the `TCM.v1` closeout:
  process objects, spill/tape continuation, and persistent lifecycle receipts

This means the repo already understands:

- mounted artifact state
- seeded external effect simulation
- bounded async lifecycle
- replay classes and challenge receipts
- durable host-backed continuation state

Again, those are real plugin-system prerequisites.

### 4. Component ABI Substrate: `implemented`

`TAS-133` already lands a bounded internal component-model ABI lane.

That matters because the plugin spec wants:

- versioned Wasm modules or linked Wasm packages
- packet-style entrypoints
- declared exports
- explicit refusal on unsupported interface shapes

The current component ABI is not yet the plugin ABI, but it proves the repo
already has one bounded interface-contract lane for modular software-like
computation.

## What Does Not Yet Exist

Even with all that substrate, the actual plugin system described in
`plugin-system.md` is not yet implemented.

The missing pieces are specific.

### 1. Canonical Plugin Manifest Contract: `planned`

The repo does not yet have one dedicated plugin manifest that freezes:

- `plugin_id`
- `plugin_version`
- `artifact_digest`
- declared export list
- packet ABI version
- input/output schema ids
- per-plugin limits
- capability requirements
- replay class
- trust tier
- evidence settings
- hot-swap compatibility posture

Current module and package reports are close, but they are not yet this exact
plugin contract.

### 2. Packet ABI And Rust-First Plugin PDK: `planned`

The spec explicitly wants:

- one narrow packet ABI
- one input packet
- one output packet or typed refusal
- one explicit error channel
- one Rust-first guest authoring path

The repo has component and module ABI surfaces, but not one dedicated
plugin-packet ABI plus one Rust-first guest PDK for bounded Wasm plugins.

### 3. Host-Owned Plugin Runtime API: `planned`

The spec wants one plugin runtime with:

- engine abstraction
- artifact loading
- export invocation
- capability mounting
- packet marshalling
- timeout and memory ceilings
- per-plugin pools
- cancellation handles
- usage and evidence capture

Psionic has Wasm runtimes, sandboxes, and effect profiles, but not one unified
plugin runtime API that exposes plugins as named hot-swappable Wasm capability
components.

### 4. Plugin-Specific Receipt Family: `planned`

The plugin spec requires every invocation receipt to carry:

- plugin id
- plugin version
- artifact digest
- export name
- packet ABI version
- input/output digests
- mount-envelope identity
- capability-envelope identity
- backend id
- refusal/failure class

Current receipt families cover execution truth, effects, replay, modules, and
processes. They do not yet freeze this exact plugin invocation identity.

### 5. Weighted Plugin Control Trace: `planned`

This is the biggest missing piece.

The spec requires the model to own:

- plugin selection
- export selection
- argument generation
- multi-step sequencing
- retries
- refusal
- stop conditions

The repo does **not** yet have one canonical weighted plugin control trace on
the post-`TAS-156A` owned Transformer route.

That is a deeper requirement than ordinary route policy or module resolution.
It is a weights-ownership requirement.

### 6. Refusal-Aware Return Path Into The Model Loop: `planned`

The plugin system is incomplete without a canonical path from:

- model emits plugin-use intent
- runtime checks admissibility and runs bounded plugin
- runtime emits receipt
- result packet returns to the model loop
- weights decide continue/retry/stop

The repo has route and process profiles, but not this exact weighted
continuation loop for plugins.

### 7. Hot-Swap Compatibility Contract: `planned`

The plugin spec explicitly wants hot-swappable Wasm components with stable
identity and compatibility checks.

Current package, module, and lifecycle systems provide parts of this, but the
repo does not yet have one plugin-specific hot-swap contract over:

- stable plugin id
- version compatibility
- ABI compatibility
- schema compatibility
- trust posture continuity

### 8. Plugin-Specific Benchmarks And Conformance Harness: `planned`

The plugin spec requires:

- packet roundtrip tests
- malformed-packet refusal tests
- capability denial tests
- timeout tests
- memory-limit tests
- digest mismatch refusal tests
- replay posture checks
- receipt integrity checks
- mount-envelope compatibility checks
- multi-plugin workflow integrity checks
- refusal propagation across chained plugins
- envelope intersection checks across multi-step workflows
- replay under partial cancellation
- cold/warm/pool/cancel overhead benchmarks

That conformance surface does not yet exist as one plugin-system bar.

### 9. Plugin Result-Binding And Schema-Stability Contract: `planned`

The plugin system also needs one explicit contract for getting plugin outputs
back into the model loop without semantic drift.

That contract must freeze:

- output schema evolution rules
- backward-compatibility posture
- refusal normalization rules
- digest binding from packet output to the next model-visible state
- typed failure when result reinjection would change declared task meaning

The repo does not yet have that result-binding contract.

### 10. Plugin Authority And Governance Contract: `planned`

The plugin system also needs governance identity, not only technical identity.

That contract must freeze:

- who may declare a plugin canonical
- who may widen a plugin capability envelope
- which receipts are required before a plugin moves from private or
  operator-only posture toward broader use
- who may change trust, promotion, quarantine, or publication state

The repo already has adjacent trust and publication substrate, but not one
plugin-specific authority model that answers those questions.

### 11. Planner-Indistinguishability And Runtime-Steering Guardrails: `planned`

The plugin system also needs explicit guardrails against host steering that
still looks surface-compliant.

Those guardrails must cover:

- choice-set integrity
- resource transparency
- scheduling ownership
- explicit refusal when hidden filtering, ranking, rewriting, or adaptive
  runtime steering would alter workflow

Without those guardrails, the host can still become the planner indirectly.

### 12. Cross-Plugin Isolation And Composition Integrity: `planned`

The plugin system also needs explicit rules for composition boundaries.

Those rules must cover:

- multi-plugin workflow integrity
- refusal propagation across chains
- envelope intersection across composed workflows
- semantic closure under multi-step chaining
- non-lossy schema transitions across chained plugin outputs
- fail-closed behavior when composition introduces ambiguity
- replay under partial cancellation
- explicit prohibition on implicit shared-memory, cache, or timing channels

Without those rules, plugins can influence each other outside the declared
trace.

### 13. Control-Trace Replay And Determinism Contract: `planned`

The plugin system also needs replay posture for the weighted control trace
itself, not only for individual plugin invocations.

That contract must freeze:

- control-trace determinism class
- model sampling policy
- temperature and randomness controls
- any external signals that may influence branching

Without that, the repo can audit calls without being able to audit workflows.

### 14. Model-Plugin Compatibility Contract: `planned`

The plugin system also needs one explicit rule for model version versus plugin
schema expectations.

That contract must freeze:

- which plugin schemas a given model was trained or validated against
- when explicit compatibility adapters are allowed
- when version skew must fail closed

Without that, training/runtime mismatch drift can break semantic preservation.

### 15. No Externalized Learning Guardrail: `planned`

The host runtime may not adapt plugin selection, ranking, or sequencing across
executions except through:

- explicit model updates
- or declared, auditable policy changes

Without that rule, the repo can grow a shadow planner via telemetry or cache
history while still sounding weighted.

### 16. Plugin Language Boundary And Closed-World Discovery Contract: `planned`

The plugin system also needs one explicit statement of what plugins are allowed
to be and how they are discovered.

That contract must freeze:

- plugins as bounded effectful procedures over declared packet and effect
  envelopes, not a silent second universal compute substrate
- first-version discovery as closed-world and operator-curated
- enumeration guarantees for the admissible plugin set
- explicit refusal on open-world dynamic discovery unless separately audited

Without that contract, plugins can reintroduce hidden general compute or break
choice-set integrity by discovery drift.

### 17. Anti-Interpreter-Smuggling Rule: `planned`

The plugin system also needs one explicit refusal boundary against plugins that
internalize workflow logic and substitute for the weighted controller.

That rule must freeze:

- no plugin may become an undeclared interpreter for the workflow itself
- no plugin may absorb branching, retry, or stop logic that is supposed to
  remain weighted and model-owned
- plugin internals may not be used to reintroduce hidden Turing-completeness
  outside the declared compute substrate

Without that rule, the model can degrade into a thin wrapper around a host-run
interpreter.

### 18. Failure-Domain Isolation Contract: `planned`

The plugin system also needs explicit failure-domain isolation.

That contract must freeze:

- per-plugin failure domains
- per-step failure domains
- per-workflow failure domains
- refusal behavior when failures would corrupt unrelated state or alter the
  admissible choice set

Without that contract, failures in one plugin can silently mutate global
workflow truth.

## The Most Important Boundary Question

The central architectural question is not "can Tassadar host Wasm modules?"

It already can, in bounded and policy-mediated ways.

The real question is:

Can the canonical owned Transformer route become the owner of plugin
orchestration logic without the host quietly becoming the real planner?

That is why the plugin system is not just another module or package issue.
It intersects directly with:

- `TAS-184`: clean-room weight causality and interpreter ownership
- `TAS-184A`: cache and activation-state discipline
- `TAS-185A`: route minimality and publication verdict
- `TAS-186`: final article-equivalence checker and audit

Until those are closed, any weighted plugin-controller claim would be weak,
because the host runtime could still be doing too much orchestration work.

## Non-Negotiable Platform Laws

The plugin tranche should freeze these laws early, because they are the
easiest place for the repo to drift into host-orchestrated tool behavior while
still using weighted-plugin language.

### 1. State Ownership Law

Plugin-related state should stay in explicit classes:

- packet-local state:
  request and response state carried in the invocation itself
- instance-local ephemeral state:
  warmed tables, memoized helpers, and caches that may improve performance but
  may never become canonical workflow truth
- host-backed durable state:
  explicit mounted stores, artifact stores, checkpoints, queues, and worklists
  under receipts and replay posture
- weights-owned control state:
  the model's own plugin-selection, branching, retry, and stop logic

The hard refusal is:

no durable workflow truth may be smuggled into plugin instance memory, hidden
runtime state, or cache-like host layers and still be described as weighted
software behavior.

### 2. Control Ownership Law

The plugin system also needs one blunt rule:

host may execute capability, but host may not decide workflow.

That means the host may:

- resolve manifests and verify digests
- mount capabilities
- enforce bounds
- run Wasm exports
- emit receipts and typed failure/refusal outcomes

But the host may not:

- choose which plugin to call
- choose which export to call
- decide retries
- decide stop conditions
- or branch multi-step workflow logic on behalf of the model

Without this law, the repo would drift into a tool runtime that only sounds
weighted.

### 3. Semantic Preservation Law

Packet marshalling, schema evolution, mount adaptation, and result reinjection
must preserve declared task meaning or fail closed with typed refusal.

That means:

- schema conversion may not silently rewrite the meaning of the task
- capability mounting may not widen or alter meaning under the guise of
  compatibility
- result reinjection may not silently coerce plugin outputs into a different
  model-visible contract
- adapter layers must fail closed when semantic preservation cannot be shown

Without this law, the plugin system can become adapter-led while still sounding
weighted.

### 4. Authority And Governance Law

The plugin system also needs one explicit authority model.

It must answer:

- who may declare a plugin canonical
- who may widen a plugin capability envelope
- what receipts are required before promotion or publication state may change
- who may move a plugin between private, operator-only, and broader posture

Without governance identity, the platform can have technical identity while
still lacking a trustworthy authority boundary.

### 5. Choice-Set Integrity Law

The host may not restrict, rank, pre-filter, or rewrite the set of admissible
plugins, exports, or arguments in a way that is not explicitly visible to and
attributable within the model-controlled trace.

That means:

- the candidate set must be fully enumerated or explicitly bounded
- any filtering or transformation must be receipt-visible
- refusal and failure classes that affect branching may not be hidden
- hidden heuristics may not narrow the effective choice set
- precomputed candidate outputs may not be surfaced as if they were neutral
  options
- the repo must define an equivalent-choice class or constrained admissibility
  model rather than treating mere enumeration as neutrality

Without this law, host filtering can decide workflow while preserving surface
compliance.

### 6. Resource Transparency Law

Any resource constraint that can affect plugin selection, sequencing, retry,
or termination must be surfaced as model-visible state or frozen as a declared
contract.

That includes:

- latency
- cost
- quotas
- availability
- pool pressure
- throttling

Equivalent choices must not be biased by hidden cost-model or latency-shaping
drift.

Either:

- cost and latency are fixed constants for the contract
- or they are fully surfaced and stable across equivalent choices

Soft failures, queueing asymmetries, and ordering effects may not become
hidden tie-breakers inside the equivalent-choice class.

Without this law, economics become hidden orchestration.

### 7. Scheduling Ownership Law

Ordering, concurrency, and result-visibility timing must be either:

- explicitly decided by the model
- or explicitly frozen as a non-adaptive runtime contract

The host may not adapt scheduling opportunistically and still describe itself
as execution-only.

### 8. Plugin Isolation Law

Plugins must not communicate or influence each other except through:

- explicit packet exchange visible in the model-controlled trace
- or declared shared host-backed stores under receipts

Implicit shared memory, shared cache coordination, or timing-based signaling
must be treated as out of contract.

Outputs must also not carry undeclared covert channels through latent shared
representations or schema fields whose semantic completeness is not audited.

When outputs compose across steps, that composition must remain semantically
closed and non-lossy or fail closed with typed refusal.

### 9. Control Trace Replay Law

The weighted plugin control trace must be replayable under a declared
determinism class.

That means the repo must freeze:

- model sampling policy
- temperature or randomness controls
- any external signals influencing branching

Without this law, the repo can replay calls without being able to replay
workflow decisions.

### 10. Model-Plugin Compatibility Law

A model must only operate against plugin schemas it was trained or validated
against, unless:

- explicit compatibility adapters are declared, versioned, and audited
- or invocation fails closed

Without this law, schema drift can silently change task meaning.

### 11. No Externalized Learning Law

The host runtime may not adapt plugin selection, ranking, or sequencing across
executions except through:

- explicit model updates
- or declared, auditable policy changes

Without this law, a shadow planner can emerge outside the model.

### 12. Anti-Interpreter-Smuggling Law

Plugins may not internalize workflow logic that substitutes for the weighted
controller.

That means:

- plugins may not become undeclared workflow interpreters
- packet schemas may not become hidden programs that relocate control logic
- capability execution may not reintroduce hidden general compute outside the
  declared substrate

Without this law, the plugin system can pass surface conformance while moving
real control out of the model.

### 13. Downward Non-Influence Law

Plugin requirements may not redefine:

- continuation semantics
- core compute-substrate rules
- proof assumptions
- or carrier identity

The plugin layer may sit above the rebased carrier and use reserved hooks, but
it may not push downward changes that alter the underlying machine truth.

### 14. Canonical Machine Inheritance Law

The plugin layer must inherit one canonical machine identity lock and one
computational model statement from the rebased carrier.

That means:

- plugin manifests, receipts, and controller traces must bind back to the same
  underlying machine identity tuple
- plugin capability may extend bounded execution surfaces without forking the
  underlying compute claim
- plugin wording may not imply a different machine than the one the bridge
  already names

### 15. Served Conformance Envelope Law

Served or broader plugin posture must publish one explicit conformance envelope
that says:

- which deviations from operator truth are allowed
- which properties must remain identical
- and which cases must fail closed

Without this law, the served plugin system can drift away from the operator
truth carrier while still sounding like the same platform.

### 16. Control-Plane Provenance Proof Dependency

Before the repo can claim weighted plugin control, it needs one inherited,
machine-readable control-plane proof from the rebased carrier.

That proof must show:

- branch, retry, and stop decisions already trace back to model outputs, the
  canonical route digest, and the machine identity tuple
- admissible choice sets are complete and neutral inside the declared
  equivalent-choice class
- latency, cost, queueing, scheduling, cache hits, helper substitution, and
  fast-route fallback do not become hidden control channels
- the pre-plugin control trace is replayable under the declared determinism
  posture

That inherited proof also needs explicit formal objects for:

- determinism classes:
  `strict_deterministic`, `seeded_stochastic`, and
  `bounded_nondeterministic_with_equivalence_class`
- equivalent-choice relation and admissible divergence bounds
- failure-semantics lattice across semantic failure, capability failure,
  policy refusal, resource exhaustion, and determinism violation
- logical-time versus wall-time semantics and whether time is model-observable
- model information boundary for cost, queue, retry, and other
  control-shaping signals
- training-versus-inference boundary for runtime adaptation, telemetry, and
  cache behavior
- hidden-state-channel closure for every stateful input to control flow
- proof-carrying versus observational artifact distinction
- observer and acceptance model for who verifies replay, receipts, and gates
- machine-level minimality relation and preserved transforms

The plugin-specific controller trace should extend this proof to plugin
selection and sequencing, not invent control ownership from scratch.

### 17. Machine Closure Bundle Dependency

The plugin lane must also inherit one canonical closure object from the
rebased machine rather than recomposing claims from scattered green artifacts.

That bundle must bind:

- machine identity tuple
- computational model statement
- determinism contract
- continuation contract
- control-plane provenance proof reference
- execution-semantics equivalence class
- carrier split declaration
- allowed state classes and hidden-state closure
- failure lattice
- observer and acceptance model
- portability and minimality binding
- proof artifacts by digest
- proof-versus-audit classification
- invalidation conditions

Invalidation conditions should include at least:

- route drift
- determinism mismatch
- hidden state-channel detection
- fast-route substitution outside the declared carrier
- downward influence from later layers
- observer or acceptance mismatch
- portability or minimality failure
- and proof-bundle digest mismatch

The plugin controller, receipts, and publication claims should point at that
bundle by digest rather than silently recomposing the machine from separate
artifacts.

## Does The Plugin System Change The Shape Of The Turing-Completeness Push?

Yes, but only in a constrained way.

### What Should Not Change

The plugin system should **not** change the minimal post-`TAS-186`
Turing-completeness objective from the companion audit:

- rebase the old `TCM.v1` closeout onto the canonical owned route
- prove continuation ownership on that route
- publish a control-plane ownership and decision-provenance proof on that route
- rebind universal-machine proofs and witness suites to that route
- issue a new canonical-route universal-substrate gate
- refresh portability and verdict split on that route

That is still the right minimal push.

If the repo skips that and jumps straight to plugins, it will blur:

- pure compute universality
- bounded operator continuation
- weighted control ownership
- capability-mediated software execution

That would weaken the claim discipline.

### What Should Change

If the long-term target includes the plugin system, the post-`TAS-186`
Turing-completeness push should become plugin-aware in six concrete ways.

#### 1. Do Not Freeze A Pure-Compute-Only Terminal Shape

The rebased canonical-route universality contract should not be written as if
the only future consumer is:

- witness programs
- pure interpreters
- and resumable kernels

It should leave room for:

- packet-mediated capability calls
- mount-envelope identities
- plugin-specific receipts
- explicit software-like capability components
- explicit control-plane versus capability-plane separation

That does **not** mean making plugins a prerequisite for Turing completeness.
It means not hard-coding a terminal contract that would need to be rewritten as
soon as plugins arrive.

#### 2. The Bridge Contract Should Reserve A Plugin Boundary

The post-`TAS-186` bridge tranche should explicitly classify plugin execution
as:

- outside the model's pure compute core
- inside the runtime's bounded execution contract
- admissible only under explicit capability envelopes

That lets later plugin work attach cleanly without mutating `TCM.v1` into a
vague "everything" substrate.

The right pattern is:

- keep `TCM.v1` as the bounded compute substrate
- add a separate named plugin execution contract above it
- bind the two through policy and receipts, not through hand-waving

That boundary should also carry a semantic-preservation rule for packet
marshalling, schema adaptation, and result reinjection.

It should also reserve:

- choice-set integrity
- resource transparency
- scheduling ownership
- closed-world enumeration guarantees for the first plugin tranche
- downward non-influence from later plugin ergonomics into bridge semantics
- forward-compatible packet and receipt hooks rather than ad hoc later
  extensions

#### 3. Continuation Ownership Must Become State-Class-Aware

The plugin spec makes a strong distinction between:

- packet-local state
- instance-local ephemeral state
- host-backed durable state

The canonical-route continuation-ownership audit from the companion
post-`TAS-186` plan should be expanded to preserve exactly that split.

Otherwise the repo will later struggle to say:

- what canonical workflow truth may live in host-backed stores
- what may only be cache-like ephemeral plugin state
- what the weighted controller is allowed to rely on across resumes

That state split is useful even before plugins exist.

#### 4. Freeze The Workflow-Ownership Rule

The post-`TAS-186` bridge should carry one blunt rule forward:

host may execute capability, but host may not decide workflow.

That rule belongs in the bridge before the plugin tranche starts, because later
plugin work will otherwise be forced to infer it from indirect article or
continuation artifacts.

#### 5. Publish A Pre-Plugin Control-Plane Proof Surface

The post-`TAS-186` bridge should also publish one machine-readable proof that:

- branch, retry, and stop decisions are attributable to model outputs on the
  canonical route
- admissible choice sets are complete and neutral inside the declared
  equivalent-choice class
- hidden control channels are surfaced or refused
- workflow decisions are replayable before later plugin sequencing is added

That proof should also freeze the formal-object bundle for:

- determinism classes
- equivalent-choice relation
- failure-semantics lattice
- time semantics
- model-information boundary
- training-versus-inference boundary
- hidden-state-channel closure
- proof-versus-audit distinction
- machine minimality
- and the observer or acceptance model

Without that artifact, the later plugin controller claim would inherit laws but
not yet inherit a positive proof that the model is the thing deciding what
happens next.

#### 6. Reserve The Canonical Machine Closure Bundle

The post-`TAS-186` bridge should also reserve one final composition step:

- publish one canonical machine closure bundle by digest
- bind the bridge, control proof, carrier split, proof artifacts,
  portability/minimality, and invalidation conditions into that bundle
- require later plugin-controller and publication claims to reference the
  bundle rather than reassembling the machine implicitly

Without that step, the repo can have all the right pieces and still fail to
close as a single machine.

#### 7. The Verdict Split Should Keep Plugin Capability Separate

The final rebased theory/operator/served split should not silently imply:

- operator universality means weighted plugin capability
- plugin capability means served plugin platform
- or served plugin platform means broad public universality

The plugin system adds one more policy-bearing capability layer. It should be
kept separate in the same way the repo already keeps:

- theory vs operator vs served
- profile promotion vs publication
- runtime success vs accepted outcomes

## What Should Be In The Immediate Post-`TAS-186` Push

The immediate post-`TAS-186` push should contain only the plugin-aware changes
that protect the architecture from rework.

Those are:

- preserve a clean separation between pure compute substrate and capability
  execution layer
- preserve a clean separation between data plane, control plane, and capability
  plane
- preserve one canonical machine identity lock and computational model
  statement across the future plugin-facing surfaces
- make the bridge contract reserve plugin-boundary identity fields
- make the continuation-ownership audit explicit about packet-local,
  ephemeral-instance, and durable-host state
- publish the control-plane ownership and decision-provenance proof before any
  weighted plugin-controller claim is made
- freeze the determinism, equivalence, failure, time, information, training,
  hidden-state, proof-vs-audit, minimality, and observer contracts through
  that pre-plugin control-plane proof
- reserve the canonical machine closure bundle as the final claim-bearing
  object rather than treating audits alone as the machine
- freeze a semantic-preservation rule for packet-mediated adapters and result
  reinjection before any weighted plugin-controller claim is made
- reserve choice-set integrity, resource transparency, and scheduling
  ownership before any adaptive plugin layer exists
- reserve downward non-influence so plugin ergonomics cannot rewrite lower
  compute or continuation truth
- reserve forward-compatible packet boundary hooks, capability invocation
  slots, receipt extensibility fields, and schema-version negotiation hooks
- freeze the workflow-ownership rule before any plugin-controller claim is made
- keep the rebased verdict split from over-reading future plugin capability

That is enough for now.

It is **not** the right moment to make the immediate push also close:

- full plugin manifest schema
- full packet ABI
- plugin PDK
- full plugin runtime API
- pool/cancel plugin runtime benchmarks
- weighted plugin control trace
- plugin-specific publication and promotion gates

Those belong in the next tranche.

## Proposed GitHub Issue Roadmap

Suggested numbering below assumes the post-`TAS-186` Turing-completeness
bridge consumes `TAS-187` through `TAS-196`, including `TAS-188A` for the
control-plane proof. If the tracker advances first, preserve the dependency
order and titles, not the exact numerals.

### Suggested `TAS-197`: Freeze Plugin Charter, Authority Boundary, And Platform Laws

Suggested GitHub title:

`Tassadar: freeze weighted plugin charter, authority boundary, and platform laws`

Summary:

Freeze the plugin system as a bounded software-capability layer above the
rebased compute substrate, with explicit state ownership, control ownership,
semantic-preservation, and governance rules.

Description:

- define plugin non-goals and internal-only/publication posture
- bind the plugin system explicitly to the post-`TAS-186` owned-route truth
  without mutating `TCM.v1`
- inherit one canonical machine identity lock and computational model
  statement from the rebased carrier
- inherit the pre-plugin control-plane ownership and decision-provenance proof
  as a hard dependency rather than treating plugin control as a fresh claim
- freeze the proof-versus-audit distinction and observer model for the plugin
  tranche instead of letting publication language stand in for proof
- freeze the three-plane contract across data plane, control plane, and
  capability plane
- freeze the adversarial host model the later conformance harness must defend
  against
- freeze the state-class split across packet-local, instance-local ephemeral,
  host-backed durable, and weights-owned control state
- freeze the rule that host may execute capability but may not decide workflow
- freeze the semantic-preservation rule for marshalling, schema adaptation, and
  result reinjection
- freeze choice-set integrity, resource transparency, and scheduling ownership
- freeze the no-externalized-learning rule for runtime behavior across
  executions
- freeze the plugin language boundary as bounded effectful procedures rather
  than an undeclared second compute substrate
- freeze the first plugin tranche as closed-world and operator-curated
- freeze the anti-interpreter-smuggling rule
- freeze the downward non-influence rule so plugin ergonomics cannot redefine
  continuation semantics, proof assumptions, or carrier identity
- define who may declare a plugin canonical, who may widen capability
  envelopes, and which receipts are required before posture changes

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`
- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`

### Suggested `TAS-198`: Canonical Plugin Manifest, Identity, And Hot-Swap Contract

Suggested GitHub title:

`Tassadar: add canonical plugin manifest and hot-swap identity contract`

Summary:

Land the canonical plugin contract so plugins become real named software
artifacts rather than loosely packaged modules.

Description:

- define `plugin_id`, `plugin_version`, `artifact_digest`, declared exports,
  packet ABI version, schema ids, limits, trust tier, replay class, and
  evidence settings
- define canonical invocation identity fields
- define compatibility and hot-swap rules across versions, ABI shapes, and
  trust posture
- keep multi-module packaging explicit when one plugin is a linked bundle

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`
- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_manager_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_route_policy_report.json`

### Suggested `TAS-199`: Packet ABI And Rust-First Plugin PDK

Suggested GitHub title:

`Tassadar: add packet ABI and Rust-first plugin PDK`

Summary:

Define the narrow invocation ABI and guest authoring surface that every bounded
plugin must use.

Description:

- define one input packet, one output packet or typed refusal, and one explicit
  error channel
- define schema ids, codec ids, payload bytes, and metadata envelope shape
- define the Rust-first guest refusal type and narrow host import surface
- keep the ABI narrow enough that conformance is easy to test and audit

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_internal_component_abi_report.json`
- `docs/audits/2026-03-18-tassadar-post-tas-102-final-audit.md`

### Suggested `TAS-200`: Host-Owned Plugin Runtime API And Engine Abstraction

Suggested GitHub title:

`Tassadar: add host-owned plugin runtime API and engine abstraction`

Summary:

Build the bounded runtime substrate that loads, mounts, executes, and cancels
plugin invocations without leaking backend-specific behavior into the top-level
contract.

Description:

- define artifact loading and digest verification
- define the engine abstraction for instantiate, invoke, mount, cancel, and
  usage collection
- define pool, queue, timeout, memory, concurrency, and optional fuel bounds
- freeze which resource signals are model-visible and which are fixed runtime
  contract
- define logical-time versus wall-time semantics and whether time is
  model-observable or control-neutral
- define the model-information boundary for queue depth, retries, cost, and
  similar runtime signals
- freeze cost-model invariance requirements across equivalent choices
- freeze concurrency and scheduling semantics so the host cannot adapt them
  opportunistically
- define failure-domain isolation across per-plugin, per-step, and per-workflow
  failures
- keep backend-specific details below the host-owned runtime API

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`
- `fixtures/tassadar/reports/tassadar_async_lifecycle_profile_report.json`
- `fixtures/tassadar/reports/tassadar_simulator_effect_profile_report.json`

### Suggested `TAS-201`: Plugin Invocation Receipts And Replay Classes

Suggested GitHub title:

`Tassadar: add plugin invocation receipts and replay classes`

Summary:

Freeze the receipt identity and replay posture needed to make plugin execution
auditable and challengeable.

Description:

- define invocation id, plugin id, version, artifact digest, export name,
  packet ABI version, digests, envelope ids, backend id, refusal/failure
  class, and resource summary
- define the failure-semantics lattice and keep replay, retry, and propagation
  behavior typed across those classes
- define deterministic and snapshot-based replay classes
- connect plugin receipts to route-integrated evidence and challenge receipts

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_effectful_replay_audit_report.json`
- `fixtures/tassadar/reports/tassadar_installed_module_evidence_report.json`
- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`

### Suggested `TAS-202`: World-Mount Envelope Compiler And Plugin Admissibility Contract

Suggested GitHub title:

`Tassadar: compile world-mount envelopes for plugins`

Summary:

Translate route and mount policy into explicit runtime-admissible plugin
envelopes so capability mediation is no longer ad hoc.

Description:

- define plugin admissibility checks
- make candidate sets fully enumerated or explicitly bounded
- make any filtering, ranking, or transformation receipt-visible
- define the equivalent-choice class or constrained admissibility model that
  makes neutral choice claims auditable
- define failure conditions when equivalent-choice neutrality is violated
- freeze the closed-world discovery assumption and explicit enumeration
  guarantees for the admissible set
- compile capability namespace grants, network rules, and mount posture into
  runtime envelopes
- bind route policy to plugin version constraints and trust posture
- keep explicit denial behavior for unsupported or disallowed invocations

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json`

### Suggested `TAS-203`: Plugin Conformance Sandbox And Benchmark Harness

Suggested GitHub title:

`Tassadar: add plugin conformance sandbox and benchmark harness`

Summary:

Prove that plugins are real software components with clean identity, bounded
execution, and refusal behavior before the repo claims model-owned plugin
planning.

Description:

- run static conformance harnesses and host-scripted traces only, not
  model-owned sequencing
- cover roundtrip success, malformed packet refusal, capability denial,
  timeout, memory-limit, packet-size, digest-mismatch, replay, and hot-swap
  compatibility rows
- cover multi-plugin workflow integrity, refusal propagation, envelope
  intersection, hot-swap inside composed workflows, and replay under partial
  cancellation
- cover per-plugin, per-step, and per-workflow failure-domain isolation
- cover shared-cache, shared-store, and timing-channel isolation negatives
- cover covert-channel negatives through latent shared representations or
  semantically incomplete schemas
- benchmark cold, warm, pooled, queued, and cancelled execution paths
- keep receipt integrity and envelope compatibility explicit

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_async_lifecycle_profile_report.json`
- `fixtures/tassadar/reports/tassadar_effectful_replay_audit_report.json`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`

### Suggested `TAS-203A`: Plugin Result-Binding, Schema-Stability, And Composition Contract

Suggested GitHub title:

`Tassadar: add plugin result-binding, schema-stability, and composition contract`

Summary:

Prove that plugin outputs can be bound back into the model loop in a stable,
typed, replayable, composition-safe way before the repo claims model-owned
plugin sequencing.

Description:

- define output schema evolution and backward-compatibility rules
- define refusal normalization and typed failure preservation
- bind output digests to the next model-visible state explicitly
- define model-version versus plugin-schema compatibility and fail-closed
  behavior on version skew
- keep proof-carrying result guarantees distinct from observational result
  audits
- require semantic closure under multi-step chaining
- require non-lossy schema transitions across chained plugin outputs
- refuse reinjection when schema repair or coercion would alter declared task
  meaning
- fail closed when composition across steps introduces ambiguity
- make the return path stable enough that weighted planning is not built on an
  adapter-defined contract

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_effectful_replay_audit_report.json`
- `fixtures/tassadar/reports/tassadar_internal_component_abi_report.json`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`

### Suggested `TAS-204`: Weighted Plugin Controller Trace And Refusal-Aware Model Loop

Suggested GitHub title:

`Tassadar: add weighted plugin controller trace and refusal-aware model loop`

Summary:

Only after the substrate is proven should the repo claim that the model owns
plugin selection, sequencing, retries, and stop conditions.

Description:

- extend the earlier pre-plugin control-plane proof through plugin selection
  and sequencing rather than inventing control ownership from scratch
- define the structured weighted plugin control trace
- define the control-trace determinism class, sampling policy, and randomness
  controls
- inherit and extend the equivalent-choice relation, failure lattice, time
  semantics, and information boundary instead of redefining them inside the
  controller
- encode packet arguments from model outputs under the canonical packet ABI
- return result packets and typed refusals back into the model loop
- prove that the host validates and executes but does not become the planner
- add negative rows for hidden host-side sequencing or retry logic
- add negative rows for host auto-retry, fallback export selection, heuristic
  plugin ranking, schema auto-repair, cached result substitution, candidate
  precomputation, and hidden top-k filtering
- add negative rows for adversarial host behaviors from the earlier threat
  model, not just obvious planner substitutions

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

### Suggested `TAS-205`: Plugin Authority, Promotion, Publication, And Trust-Tier Gate

Suggested GitHub title:

`Tassadar: add plugin authority, promotion, and publication gate`

Summary:

Add the policy and trust machinery needed to promote bounded plugins without
accidentally widening them into a public arbitrary-software claim.

Description:

- implement the earlier authority model on canonical declaration, capability
  envelope widening, and posture changes
- define plugin benchmark bars and trust-tier gates
- define operator-only versus served/public posture
- define the served conformance envelope for any broader plugin posture
- define which observers are allowed to accept, challenge, or publish plugin
  conformance results
- bind validator and accepted-outcome hooks where required
- keep promotion, quarantine, revocation, publication refusal, and required
  posture-change receipts explicit

Supporting material:

- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json`

### Suggested `TAS-206`: Publish The Bounded Weighted Plugin Platform Closeout Audit

Suggested GitHub title:

`Tassadar: publish bounded weighted plugin platform closeout audit`

Summary:

Publish the first honest plugin-platform closeout only after manifest, ABI,
runtime, receipts, mount compiler, conformance sandbox, weighted control, and
promotion/publication bars are all green.

Description:

- state clearly what the plugin system does and does not claim
- preserve the separation from the bounded Turing-completeness closeout
- state that weighted plugin control depends on the inherited control-plane
  proof plus the plugin-specific controller extension
- keep the final claim-bearing machine closure bundle as a separate object
  rather than treating the audit itself as the bundle
- keep operator and served/plugin publication posture explicit
- refuse any implication that broader served posture already shares operator
  conformance when the envelope is not published
- refuse any implication of arbitrary public Wasm or arbitrary public tool use

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `~/code/alpha/tassadar/plugin-system.md`

## Suggested `TAS-207` Through `TAS-215`: Anti-Drift Stability Tranche

If the tracker wants the second-order stability constraints called out as
first-class artifacts rather than only absorbed into `TAS-187` through
`TAS-206`, the cleanest follow-on tranche is:

### Suggested `TAS-207`: Freeze Canonical Machine Identity Lock

Suggested GitHub title:

`Tassadar: freeze canonical machine identity lock`

Summary:

Bind every proof, witness, benchmark, route, plugin receipt, and closeout to
one globally named machine identity tuple so the stack cannot drift back into
multiple underlying machines.

Description:

- define the canonical tuple over model id, weight digest, route digest,
  continuation contract, and carrier class
- require plugin-facing receipts and controller traces to inherit that tuple
  explicitly
- refuse mixed-carrier evidence bundles that silently bind different tuples

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

### Suggested `TAS-208`: Publish Canonical Computational Model Statement

Suggested GitHub title:

`Tassadar: publish canonical computational model statement`

Summary:

State exactly what machine the plugin layer sits on top of, under which
continuation semantics and effect boundaries.

Description:

- name the computational model the repo is claiming
- state the continuation, effect, and refusal boundaries that belong to that
  model
- keep plugin capability and public-serving implications explicitly out of
  scope unless separately green

Supporting material:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`

### Suggested `TAS-209`: Audit Execution-Semantics Equivalence And Proof Transport

Suggested GitHub title:

`Tassadar: audit execution-semantics equivalence and proof transport`

Summary:

Prove that route rebinding and later plugin-facing execution still sit on the
same proof-bearing semantics rather than merely preserving outputs.

Description:

- define the transition-level equivalence class or explicit proof abstraction
  boundary the rebinding relies on
- audit cache, batching, helper, continuation, and route-family effects
- refuse plugin-facing wording that assumes a stronger machine than the proof
  actually binds

Supporting material:

- `fixtures/tassadar/reports/tassadar_universal_machine_proof_report.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

### Suggested `TAS-210`: Freeze Continuation Non-Computationality Contract

Suggested GitHub title:

`Tassadar: freeze continuation non-computationality contract`

Summary:

Prove that continuation extends execution without becoming a second machine.

Description:

- audit checkpoint, spill, tape, and process-object structures for hidden
  workflow logic
- prove continuation does not add computational expressivity beyond the base
  route
- refuse plugin or controller designs that lean on resume semantics as hidden
  compute

Supporting material:

- `fixtures/tassadar/reports/tassadar_spill_tape_store_report.json`
- `fixtures/tassadar/reports/tassadar_session_process_profile_report.json`
- `fixtures/tassadar/reports/tassadar_installed_process_lifecycle_report.json`

### Suggested `TAS-211`: Freeze Fast-Route Legitimacy And Carrier Binding

Suggested GitHub title:

`Tassadar: freeze fast-route legitimacy and carrier binding`

Summary:

Make every fast route either provably part of the underlying carrier or
explicitly outside it before plugin or served wording leans on it.

Description:

- classify `ReferenceLinear`, `HullCache`, and any resumable family relative to
  the proof-bearing carrier
- require semantics-equivalence evidence before a fast route may inherit proof
  or universality status
- refuse served or plugin wording that treats an unproven fast route as the
  machine underneath the platform

Supporting material:

- `fixtures/tassadar/reports/tassadar_article_fast_route_architecture_selection_report.json`
- `fixtures/tassadar/reports/tassadar_article_fast_route_implementation_report.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

### Suggested `TAS-212`: Freeze Equivalent-Choice Neutrality And Admissibility Contract

Suggested GitHub title:

`Tassadar: freeze equivalent-choice neutrality and admissibility contract`

Summary:

Define when two admissible plugin choices are equivalent and when ordering,
cost, latency, or soft-failure effects are allowed to distinguish them.

Description:

- define an auditable equivalent-choice class or constrained admissibility
  model
- keep ordering, latency, cost, and soft-failure effects from becoming hidden
  control channels inside that class
- require receipt-visible justification when equivalent choices are narrowed or
  ordered differently

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`

### Suggested `TAS-213`: Freeze Downward Non-Influence And Served Conformance Envelope

Suggested GitHub title:

`Tassadar: freeze downward non-influence and served conformance envelope`

Summary:

Prove that plugin or served layers cannot rewrite lower-plane truth and that
broader posture stays inside a declared conformance envelope.

Description:

- forbid plugin ergonomics from redefining continuation semantics,
  compute-substrate rules, proof assumptions, or carrier identity
- define which served deviations from operator truth are allowed
- require fail-closed behavior when served posture would escape the declared
  conformance envelope

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`

### Suggested `TAS-214`: Publish Anti-Drift Stability Closeout

Suggested GitHub title:

`Tassadar: publish anti-drift stability closeout`

Summary:

Publish one explicit closeout that says machine identity, semantics,
continuation, fast-route, choice-neutrality, capability-boundary, and
served-envelope constraints are actually locked.

Description:

- summarize the final machine identity lock and computational model statement
- summarize the inherited control-plane provenance proof and the plugin-layer
  control extension verdict
- summarize determinism classes, equivalent-choice relation, failure lattice,
  time semantics, information boundary, and training-versus-inference
  boundary status
- summarize execution-semantics, continuation, and fast-route legitimacy
  verdicts
- summarize proof-versus-audit classification, machine minimality,
  hidden-state closure, observer model, downward non-influence, and
  served-envelope verdicts
- refuse any stronger plugin-platform claim when one of those locks is still
  open

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `~/code/alpha/tassadar/plugin-system.md`

### Suggested `TAS-215`: Publish Canonical Machine Closure Bundle

Suggested GitHub title:

`Tassadar: publish canonical machine closure bundle`

Summary:

Freeze one machine-readable closure object that binds proof, execution,
control, continuation, invalidation, and plugin inheritance semantics into a
single claim-bearing machine identity.

Description:

- publish one canonical closure bundle by digest
- bind the machine identity tuple, computational model statement,
  determinism contract, continuation contract, control-plane provenance proof,
  execution-semantics equivalence class, carrier split, state classes,
  hidden-state closure, failure lattice, observer model, portability binding,
  minimality binding, and proof-carrying artifact digests into that object
- classify which entries are proofs and which are audits
- centralize invalidation conditions for route drift, determinism mismatch,
  hidden state channels, fast-route substitution, downward influence,
  observer mismatch, portability failure, minimality failure, and proof-bundle
  digest mismatch
- require any terminal universality, plugin, or platform claim to reference
  this bundle by digest
- require plugin controller, receipt, and publication claims to inherit this
  bundle rather than silently recomposing the machine

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `fixtures/tassadar/reports/tassadar_universal_machine_proof_report.json`
- `fixtures/tassadar/reports/tassadar_universality_witness_suite_report.json`
- `fixtures/tassadar/reports/tassadar_universality_verdict_split_report.json`

## Recommended Dependency Order

The dependency order should be:

1. finish `TAS-182` through `TAS-186`
2. land the post-article Turing-completeness bridge tranche from the companion
   audit, including the pre-plugin control-plane proof and its formal-object
   bundle
3. land plugin charter, manifest, packet ABI, runtime API, receipts, and
   replay classes
4. land the world-mount envelope compiler and plugin admissibility contract
5. land the plugin conformance sandbox with static harnesses and scripted
   traces, not model-owned sequencing
6. land the plugin result-binding and schema-stability contract
7. only then land weighted plugin-controller integration
8. land plugin authority/promotion/publication policy
9. if those anti-drift constraints are not already absorbed into the earlier
   issues, land the `TAS-207` through `TAS-214` stability tranche
10. publish the canonical machine closure bundle
11. only then consider any stronger "weighted software platform" closeout

That order matters because it keeps the repo from using a not-yet-audited
weighted controller as evidence for a plugin system whose core control-ownership
question is still unresolved and keeps runtime steering, side channels, and
adapter drift out of the claim surface.

## What The Repo Can Honestly Say Today

The strongest honest statement today is:

Psionic/Tassadar already has most of the bounded operator/internal
software-platform substrate a plugin system would need:

- module and package identity
- module evidence and promotion lifecycle
- import and capability policy
- world-mount negotiation
- bounded effect profiles
- replay and challenge receipts
- session-process and persistent process objects
- profile publication and route policy

But the repo does **not** yet have:

- a canonical plugin manifest
- a packet ABI and Rust plugin PDK
- a host-owned plugin runtime API
- plugin-specific invocation receipts
- a stable result-binding and schema-stability contract
- a plugin-specific authority and governance model
- a formal three-plane contract across data, control, and capability planes
- a unified adversarial host model
- one canonical machine identity lock inherited across plugin-facing artifacts
- one canonical computational model statement for the underlying carrier
- one inherited control-plane ownership and decision-provenance proof
- one canonical machine closure bundle by digest that plugin claims must
  inherit
- a formal determinism class contract for workflow and controller traces
- a frozen plugin language boundary plus closed-world discovery contract
- planner-indistinguishability guardrails over choice sets, resources, and
  scheduling
- a formal equivalent-choice neutrality contract for admissible choices
- a formal failure-semantics lattice with replay and propagation rules
- a time-semantics contract and explicit model-information boundary
- a formal training-versus-inference boundary for adaptation, telemetry, and
  caches
- explicit failure-domain and covert-channel constraints
- a replayable control-trace determinism contract
- a model-plugin compatibility contract
- an explicit no-externalized-learning guardrail
- a proof-versus-audit distinction for plugin evidence
- a machine-level minimality definition for the inherited carrier
- a hidden-state-channel closure rule
- an observer and acceptance model for replay, challenge, and publication
- an explicit downward non-influence rule from plugin ergonomics into core
  compute truth
- a served conformance envelope for broader plugin posture
- or a weighted controller that owns plugin selection and sequencing on the
  canonical owned Transformer route

So the plugin system is not blocked on basic substrate. It is blocked on
unifying that substrate under one plugin contract, proving semantic
preservation through adapters and result reinjection, freezing governance
identity, closing host-steering attack surfaces, and then proving weighted
control ownership while keeping machine identity, computational-model truth,
choice neutrality, served conformance, control-plane provenance, and the
determinism/equivalence/failure/time/information/proof-class contracts from
drifting and then binding them into one non-fragmentable closure bundle.

## Final Judgment

The plugin-system spec does change the shape of the Turing-completeness push,
but not by replacing it.

It changes the shape by forcing one architectural discipline:

the post-`TAS-186` universality rebase must be written so the canonical owned
Transformer route can later become the weighted controller for a bounded
software-like Wasm capability platform, without turning `TCM.v1` into a vague
everything-machine, without letting the host quietly steal orchestration, and
without letting plugin ergonomics rewrite lower-plane machine truth.

So the right answer is:

- finish the core post-`TAS-186` Turing-completeness rebase first
- make that rebase publish the standalone control-plane ownership and
  decision-provenance proof
- make that rebase freeze the formal determinism, equivalence, failure, time,
  information, training, proof-class, minimality, hidden-state, and observer
  contracts
- make that rebase plugin-aware at the boundary level
- then build the plugin system as the next explicit tranche on top of it
- then publish the canonical machine closure bundle that all terminal claims
  reference by digest

That is the cleanest way to get both:

- one honest bounded Turing-completeness story
- and one honest weighted plugin-system story

without collapsing them into one over-broad claim.

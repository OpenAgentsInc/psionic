# Mesh-llm Harvest And Integration Audit For Psionic And Project Product Surfaces

Date: 2026-04-02

Scope: audit of the local synced open-source `competition/repos/mesh-llm`
checkout at `65ac783`, compared against the current `psionic`, `openagents`,
and `probe` docs in this workspace. This document answers four questions:

1. what `mesh-llm` actually is today
2. where it overlaps directly with `psionic`
3. where it overlaps with product surfaces in this workspace
4. what we should harvest, and what we should explicitly not adopt

## Why This Matters

The useful question here is not whether `mesh-llm` is a "competitor."

It is open source. We can read it, port from it, adapt it, and absorb the
parts that are actually good into our own codebase.

The real question is what it unlocks if we combine its best distributed-
inference product ideas with the stronger ownership split we already have
across `psionic`, `openagents`, and `probe`.

Absorbing the right parts would unlock:

- distributed inference bring-up
- much faster multi-machine join and bootstrap UX
- a cleaner operator view of mesh topology, model availability, and host state
- thin-client and standby-node roles that feel productized instead of ad hoc
- agent-adjacent coordination on top of a pooled GPU mesh
- a path to turn pooled inference into a first-class OpenAgents compute surface
  inside the broader compute market rather than a detached sidecar product

The end state should not be "run `mesh-llm` next to us forever."

The end state should be:

- absorb the useful mesh and operator patterns into `psionic`
- combine them with `psionic` runtime truth, evidence, and refusal posture
- connect that substrate to the larger OpenAgents Compute Market, wallet,
  settlement, and kernel authority
- reuse the agent-facing pieces where they fit `probe` instead of inventing a
  separate agent stack beside it

## Integration Thesis

`mesh-llm` is a high-value open-source harvest repo for one narrower band:

- easy multi-node inference packaging
- instant join-and-serve behavior
- lightweight management API plus web console
- consumer-grade install and service mode
- agent-facing coordination patterns such as blackboard and launcher shortcuts

It is not the right long-term owner for the broader things this workspace is
already building:

- native runtime ownership
- backend truth and capability publication
- artifact lineage, receipts, and refusal posture
- training, eval, and executor-class substrate
- economic kernel, wallet, payout, and market authority
- durable coding-runtime state, approvals, transcripts, and supervision

The right reading is:

- `mesh-llm` is a high-value harvest repo for the inference-mesh product layer
- it is not the right substrate to replace `psionic`
- it is not the right product authority to replace `openagents`
- it is not the right coding-runtime core to replace `probe`

## Source Set

This audit is grounded in the local copies of:

- `competition/repos/mesh-llm/README.md`
- `competition/repos/mesh-llm/AGENTS.md`
- `competition/repos/mesh-llm/ROADMAP.md`
- `competition/repos/mesh-llm/PLUGINS.md`
- `competition/repos/mesh-llm/mesh-llm/docs/DESIGN.md`
- `psionic/AGENTS.md`
- `psionic/README.md`
- `psionic/docs/ARCHITECTURE.md`
- `psionic/docs/INFERENCE_ENGINE.md`
- `psionic/docs/TRAIN_SYSTEM.md`
- `openagents/AGENTS.md`
- `openagents/README.md`
- `openagents/docs/MVP.md`
- `openagents/docs/OWNERSHIP.md`
- `openagents/docs/kernel/README.md`
- `probe/AGENTS.md`
- `probe/README.md`
- the root `AGENTS.md`
- the root `README.md`

## What Mesh-llm Actually Is

`mesh-llm` is a Rust product wrapper around `llama.cpp` inference, not a
general ML substrate.

Its own design doc defines it as:

- a Rust sidecar that turns `llama.cpp` RPC into a peer-to-peer mesh
- QUIC-connected nodes over `iroh`
- local `rpc-server` and `llama-server` processes behind a local API proxy
- a management API on port `3131`
- an OpenAI-compatible inference API on port `9337`

The current product package is already unusually complete for this narrow lane:

- install script and release bundles
- background service mode
- `--auto` discovery and join
- named or private meshes
- Nostr-based mesh discovery and publication
- bootstrap proxy so a new joiner gets API access before local model load
- web console over `/api/status` and `/api/events`
- multi-model routing by the `model` field
- dense-model pipeline parallelism
- MoE expert sharding with session-sticky routing
- demand-aware rebalancing and host election
- agent blackboard, plugin process model, MCP exposure, and launcher roadmap

This makes `mesh-llm` more than "distributed llama.cpp."

It is already an opinionated inference product with:

- operator UX
- node discovery
- mesh membership semantics
- local plugin/runtime extension seams
- agent-facing coordination features

That matters because the closest comparison inside this workspace is not one
crate. It is the combination of:

- `psionic` inference and cluster substrate
- `openagents` product/operator surfaces
- `probe` agent-runtime surfaces

## Direct Comparison To Psionic

## 1. Where Mesh-llm Is Ahead Today

### 1.1 Distributed inference is much more productized

`mesh-llm` already turns distributed inference into a user-facing workflow:

- install
- run `mesh-llm --auto`
- join a mesh or start one
- get an OpenAI-compatible API immediately
- inspect state in a web console
- invite another node with a token

`psionic` already owns much deeper inference truth, but its docs still describe
the stack primarily as a reusable execution substrate and explicit bounded
runtime lanes. That is the correct architecture, but it is not the same thing
as a polished inference-mesh product.

### 1.2 Join and bootstrap UX is materially better

The bootstrap proxy in `mesh-llm` is a real product insight.

A joining node can expose API service before its own model finishes loading.
That removes a visible dead zone from the user experience.

`psionic` should study this directly for:

- remote attach lanes
- cluster client roles
- operator-visible warmup transitions
- later consumer or thin-client modes

### 1.3 Mesh state is surfaced cleanly

`mesh-llm` has a coherent operator API:

- `/api/status`
- `/api/events`
- `/api/discover`
- `/api/join`
- embedded dashboard on the same management port

This is stronger than most pure runtime repos because the node state is already
legible as a product surface, not just log output.

### 1.4 Demand-aware distributed serving is simpler and more usable

`mesh-llm` already carries a simple demand gossip and promotion model:

- request-rate tracking by model
- standby node promotion
- per-model host election
- latency-aware tensor-split selection

That is not a replacement for `psionic` routing and capability truth. It is a
good reference for the user-visible control loop around distributed serving.

## 2. Where Psionic Is Ahead And Should Stay Ahead

### 2.1 Psionic owns native runtime truth

`mesh-llm` relies on spawned `llama-server` and `rpc-server` processes and
intentionally keeps `llama.cpp` largely unmodified at the serving boundary.

`psionic` is the opposite architectural bet:

- native runtime ownership
- backend crates
- explicit execution modes
- explicit capability publication
- explicit fallback and refusal posture
- runtime evidence and proof bundles

That difference is fundamental.

`mesh-llm` optimizes operator experience around an existing engine.
`psionic` is building the engine family that owns the execution truth itself.

### 2.2 Psionic is broader than inference

`mesh-llm` is narrowly inference-led.

`psionic` also owns:

- training substrate
- optimizer substrate
- cluster and distributed execution contracts
- artifact staging and lineage
- eval substrate
- executor-class work such as Tassadar
- hardware validation and bounded publication discipline

`mesh-llm` has no equivalent training or executor-class depth.

### 2.3 Psionic is much stronger on truth and honesty contracts

`psionic/docs/INFERENCE_ENGINE.md` is explicit about:

- supported versus unsupported regions
- fallback-required behavior
- runtime-side telemetry
- model-specific publication
- backend identity
- decode-path truth

`mesh-llm` is much lighter on this class of contract. Its user story is:

- join the mesh
- route requests
- surface status

That simplicity is useful. It is not a substitute for `psionic`'s stronger
evidence and refusal model.

### 2.4 Psionic is not tied to one serving engine family

`mesh-llm` is tightly shaped around `llama.cpp`.

That is one reason it ships quickly.
It is also one reason it should not become the hidden future of `psionic`.

If we copied the whole architecture, we would inherit:

- process-spawned engine dependency
- weight-format and runtime constraints driven by `llama.cpp`
- product decisions coupled to one external inference engine family

That is the wrong long-term substrate posture for `psionic`.

## 3. The Correct Psionic Harvest

The useful harvest from `mesh-llm` into `psionic` is not kernel code. It is
the inference-mesh product layer:

- bootstrap proxy semantics
- thin management API shape
- web-console status model
- passive client and standby node role split
- per-model host-election semantics
- demand-gossip ideas
- simple invite/join flows
- operator-grade model/peer/topology presentation

The wrong harvest would be:

- replacing native Psionic execution with a `llama.cpp` sidecar product
- dropping artifact or capability publication discipline
- treating "OpenAI-compatible and easy to join" as enough execution truth

## Comparison To Project Product Surfaces

## 1. OpenAgents / Autopilot

### 1.1 Where mesh-llm overlaps

The overlap with `openagents` is real but narrow:

- local compute-provider onboarding
- "your machine is now online" operator feel
- consumer-friendly local model serving
- a visible control panel for live node state
- eventual remote or mobile consumption of local or pooled inference

The strongest product lesson for `openagents` is that compute capacity becomes
much more legible when the product can show:

- who is in the mesh
- what models are available
- what each node is serving
- how to join another node immediately

That part is stronger in `mesh-llm` than in the current `openagents`
compute-facing user experience.

### 1.2 Where mesh-llm does not overlap

`openagents` owns economic and product authority that `mesh-llm` does not even
attempt:

- wallet and payout UX
- Lightning settlement
- compute-market product identity
- data-market UX and targeted request flows
- kernel contracts, receipts, verification, liability, and settlement
- multi-market product framing across compute, data, labor, liquidity, and risk
- buyer/provider economic workflows

`mesh-llm` offers a pooled OpenAI-compatible API and mesh coordination.

That is useful for the compute substrate and local operator experience.
It is not a substitute for the OpenAgents product shell or economy kernel.

### 1.3 Nostr means something different in each system

This distinction matters.

In `mesh-llm`, Nostr is used for mesh discovery and publication.

In `openagents`, Nostr is part of the protocol and coordination fabric around:

- NIP-90 demand
- marketplace signaling
- data-vending request/result paths
- relay-native product flows

Those are not interchangeable uses.

`mesh-llm`'s Nostr layer is a discovery tool.
`openagents` uses Nostr as part of market and product coordination.

## 2. Probe

### 2.1 Where mesh-llm overlaps

`mesh-llm` clearly wants to be agent-friendly:

- blackboard for shared status/findings/questions
- plugin process model
- MCP exposure
- launcher roadmap for Goose, `pi`, and `opencode`
- a local OpenAI-compatible endpoint for tool-using agents

That makes it relevant to `probe` as a backend environment and coordination
adjunct.

### 2.2 Where Probe is much stronger

`probe` owns the actual coding runtime:

- session lifecycle
- transcripts
- approvals
- tool execution
- daemon/server surfaces
- durable session storage
- TUI and CLI contracts

`mesh-llm` is not trying to own this runtime layer.

Its launcher story is:

- point an existing agent at the mesh backend
- give that agent a blackboard or plugin extension surface

That is additive to `probe`.
It is not a replacement for `probe`.

### 2.3 The correct relationship

The right relationship is:

- `probe` should be able to use `mesh-llm`-like backends
- `probe` can borrow blackboard-style coordination ideas
- `probe` should not inherit `mesh-llm`'s plugin/runtime model as its core

`probe` needs stronger session and policy semantics than `mesh-llm` is trying
to provide.

## Capability Matrix

| Capability | mesh-llm | Psionic | OpenAgents / Probe |
| --- | --- | --- | --- |
| Multi-node inference bring-up | strong | partial to strong, but less productized | consumed from substrate |
| Native execution ownership | weak | strong | not the owner |
| OpenAI-compatible local serving | strong | strong | consumed surface |
| Install-to-serving UX | very strong | weaker today | product-dependent |
| Web console for live node state | strong | weaker today | partial, repo-specific |
| Demand-aware model routing | strong | partial | product-specific consumers |
| Artifact lineage and refusal posture | weak | strong | consume higher-level truth |
| Training substrate | absent to weak | strong | not the owner except consumption |
| Economic kernel / settlement / wallet | absent | absent | strong in `openagents` |
| Coding runtime / approvals / transcripts | absent | absent | strong in `probe` |

## What We Should Port

## 1. Into Psionic

- a small management API and event stream for inference-mesh status
- explicit node roles such as host, worker, standby, and thin client
- bootstrap proxy semantics so client access is not blocked on local warmup
- demand-aware model gossip and promotion logic
- a first-class inference-mesh operator surface above native runtime truth

## 2. Into OpenAgents

- compute-provider onboarding that feels immediate instead of infrastructural
- clearer local-runtime status and model-availability views
- friendlier multi-machine join/invite flows where that fits the product
- a productized "what is my machine contributing right now?" control-panel view
- a first-class pooled-inference surface inside the broader Compute Market
  instead of a disconnected mesh-only experience

## 3. Into Probe

- blackboard-style shared coordination as an optional adjunct surface
- easier attach flows to pooled inference backends
- possibly a simple mesh-backed profile for narrow distributed inference use

## GitHub Issue Roadmap

If we want to absorb the full useful `mesh-llm` surface into our own codebase
without keeping `mesh-llm` as a parallel product, the work should be staged as
one explicit GitHub issue sequence.

I re-checked the current code before expanding this sequence. The relevant
anchors are:

- `competition/repos/mesh-llm/mesh-llm/src/api.rs`
- `competition/repos/mesh-llm/mesh-llm/src/mesh.rs`
- `competition/repos/mesh-llm/mesh-llm/src/proxy.rs`
- `competition/repos/mesh-llm/mesh-llm/src/plugin.rs`
- `competition/repos/mesh-llm/mesh-llm/src/plugin_mcp.rs`
- `crates/psionic-net/src/lib.rs`
- `crates/psionic-cluster/src/ordered_state.rs`
- `crates/psionic-cluster/src/replicated_serving.rs`
- `crates/psionic-router/src/lib.rs`
- `crates/psionic-serve/src/openai_http.rs`
- `crates/psionic-serve/src/bin/psionic-gpt-oss-server.rs`
- `../openagents/apps/autopilot-desktop/src/desktop_control.rs`
- `../openagents/apps/autopilot-desktop/src/bin/autopilotctl.rs`
- `../openagents/crates/openagents-provider-substrate/src/lib.rs`
- `../probe/crates/probe-protocol/src/backend.rs`
- `../probe/crates/probe-core/src/backend_profiles.rs`
- `../probe/crates/probe-server/src/server.rs`
- `../probe/docs/11-server-attach-and-launch.md`

The sequence should be:

1. `psionic`: "Freeze mesh-integration target and owner split"
   Current anchors already show the natural boundaries. `psionic-serve` owns
   the serving API, `psionic-net` owns identity, admission, discovery, and
   tunnels, `psionic-cluster` owns ordered topology and warm-replica truth,
   `openagents` owns the operator and market surfaces, and `probe` owns backend
   attach semantics. This issue should turn that into one short canonical spec
   under `psionic/docs/`.
   Scope: define the owned terms for `mesh`, `join`, `invite`, `bootstrap
   proxy`, `thin client`, `standby`, and `pooled inference`. State explicitly
   that market product IDs, receipts, payout, and provider UI remain outside
   `psionic`.
   Acceptance: later issues may add new crates or types, but they may not move
   wallet or settlement logic into `psionic`, and they may not move transport
   or topology truth into `openagents` or `probe`.
   Depends on: none.

2. `psionic`: "Add mesh node-role contract"
   `mesh-llm` has a serving-facing `NodeRole` of `worker`, `host`, and
   `client`. `psionic-net` already has a transport-facing `NodeRole` of
   `CoordinatorOnly`, `ExecutorOnly`, and `Mixed`. Those are not the same
   contract. This issue should add a second typed role layer instead of
   overloading the existing transport enum.
   Scope: introduce a served-mesh role contract that can express `host`,
   `worker`, `standby`, and `thin_client`, plus machine-readable transition or
   refusal reasons such as `warming`, `artifact_missing`, `admission_refused`,
   `draining`, and `remote_only`. Publish both the transport role and the
   served-mesh role in management state.
   Likely landing points: `psionic-net` for identity publication shape,
   `psionic-cluster` for role transitions, and `psionic-serve` for operator
   serialization.
   Acceptance: one node can honestly report "mixed transport role, standby
   serving role" without inventing ambiguous hybrid labels.
   Depends on: 1.

3. `psionic`: "Add mesh identity, invite, and join contract"
   `psionic-net` already has most of the low-level pieces: `ClusterAdmissionConfig`,
   `LocalClusterConfig`, `ClusterDiscoveryCandidate`,
   `ClusterIntroductionPolicy`, `SignedClusterIntroductionEnvelope`,
   `ClusterCandidateAdmissionDecision`, and file-backed network-state
   persistence. `mesh-llm` adds the operator experience around those facts.
   Scope: define one durable join bundle that carries a mesh label, namespace,
   admission material or signed introduction envelope, advertised control-plane
   addresses, and the minimum trust-policy metadata required for honest import.
   Persist last-joined mesh preference and last-imported join bundle separately
   from transient transport state.
   Non-goal: do not copy `mesh-llm`'s invite token format byte-for-byte. The
   contract should be built on top of Psionic admission and introduction types.
   Acceptance: a node can export a join bundle, another node can import it into
   file-backed network state, and the resulting admission decision is visible
   as typed refusal reasons or acceptance state.
   Depends on: 1, 2.

4. `psionic`: "Add inference-mesh management API"
   `mesh-llm/src/api.rs` exposes `/api/status`, `/api/runtime`,
   `/api/runtime/processes`, `/api/events`, `/api/discover`, and local
   model-control routes. `psionic-serve/src/openai_http.rs` currently exposes
   `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/responses`, and
   `/v1/embeddings`. `openagents` already expects a separate operator API shape
   through `desktop_control.rs` with `/v1/snapshot`, `/v1/events`, and
   `/v1/action`.
   Scope: add a separate Psionic management namespace rather than bloating the
   OpenAI-compatible paths. The minimum useful surface is status, event stream,
   discovery state, join state, loaded model state, replica warm state, and
   per-node route inventory.
   Likely response sources: `ClusterMembershipRecord`,
   `ClusterCandidateAdmissionDecision`, `ClusterReplicaSnapshot`,
   `ClusterReplicaLifecyclePolicy`, `FleetRouter::inventory`, and explicit
   local backend truth from `openai_http.rs`.
   Acceptance: a single management request can answer "what nodes exist, what
   role each node is in, what models are warm, what join state is pending, and
   what routes are available" without scraping logs.
   Depends on: 1, 2, 3.

5. `psionic`: "Add bootstrap proxy and thin-client mode"
   `mesh-llm/src/proxy.rs` already proves the useful behavior: buffer one HTTP
   request, inspect the requested `model`, and forward early while the local
   node is cold. `psionic-serve` already has `RoutingRequest`, `RouteSelection`,
   `RoutedWarmState`, and worker inventory. The missing piece is the join-time
   experience.
   Scope: allow a thin client or joining worker to bind the normal
   OpenAI-compatible API locally while routing to an existing warm peer until
   local artifact load completes. Support `/v1/chat/completions`,
   `/v1/responses`, and `/v1/embeddings` where the selected remote worker can
   honestly satisfy the route.
   Important constraint: proxied service must still publish explicit route
   provenance, warm-state reason, and fallback posture. A cold node cannot
   pretend it executed locally.
   Acceptance: a node in `thin_client` or `warming` role can answer requests
   immediately, and the operator API shows that the traffic was proxied rather
   than locally served.
   Depends on: 2, 4.

6. `psionic`: "Add QUIC mesh transport for remote inference lanes"
   `mesh-llm/src/mesh.rs` uses iroh QUIC with multiplexed control and tunnel
   streams. `psionic-net` already has trusted-LAN admission, signed identity,
   tunnel policy, and file-backed wider-network trust state, but it is still
   shaped around the current local cluster seam. This issue should add the
   remote transport seam without breaking the existing evidence model.
   Scope: add one remote-capable transport class for gossip, join control,
   management subscription, and request forwarding. Reuse Psionic admission,
   trust, and attestation checks. Preserve explicit tunnel policy and bounded
   service exposure rather than creating an untyped "mesh connection".
   Design rule: transport choice is an implementation detail. The public truth
   is membership, trust, tunnel policy, and route availability.
   Acceptance: two nodes can form a mesh across non-LAN boundaries, preserve
   node identity and admission truth, and survive disconnect or reconnect with
   deterministic membership transitions.
   Depends on: 3, 4, 5.

7. `psionic`: "Add per-model host election and standby promotion"
   `psionic-cluster/src/ordered_state.rs` already contains the right building
   blocks: `ClusterTerm`, `ClusterLeaseTick`, `ClusterEventIndex`, and typed
   membership state. `psionic-cluster/src/replicated_serving.rs` already has
   `ClusterReplicaSnapshot` and `ClusterReplicaLifecyclePolicy`. What is still
   missing is the operator-facing host-election loop from `mesh-llm`.
   Scope: add one ordered per-model lease or election record that decides which
   node is the active host, which nodes are standby, and when promotion or
   demotion occurs. Keep the election state in ordered cluster truth rather
   than in transient serve-layer memory.
   Acceptance: failover from active host to standby produces one explicit term
   change, one explicit reason, and one updated management snapshot. No two
   nodes may both claim to be the active host for the same model lane under the
   same term.
   Depends on: 2, 4, 6.

8. `psionic`: "Add demand gossip and hot-model rebalance"
   `mesh-llm/src/mesh.rs` has a simple `ModelDemand` map with TTL, request
   counts, requested-model declarations, and an RTT gate. `psionic-router`
   already has `RoutingRequest`, `RouteSelectionMetrics`, and warm-aware or
   cache-aware policies. `psionic-cluster` already has replica lifecycle truth.
   Scope: add a Psionic-native demand snapshot keyed by product ID, model ID,
   and maybe route alias. Feed that snapshot into `ClusterReplicaLifecyclePolicy`
   so target warm-replica counts and standby promotions respond to real use.
   Keep the policy legible. This should emit state like "promoted because
   demand_count rose above threshold" rather than opaque heuristics.
   Acceptance: stale demand expires predictably, hot demand increases target
   warm capacity, and management state shows the policy reason for every
   promotion or unload.
   Depends on: 4, 6, 7.

9. `psionic`: "Add multi-model mesh router"
   `psionic-router` is already close. `RoutingEndpoint`, `RoutingTarget`,
   `RoutedModelInventory`, `RoutedWorkerInventory`, `RouteSelection`, and
   `FleetRouter` already capture most of the routing truth. Today that
   inventory is still mostly local-serving oriented.
   Scope: make the router inventory mesh-native. One worker inventory entry
   should be able to represent a remote peer, its served models, its endpoint
   support, its warm state, and its execution truth. Route selection should be
   able to choose between local and remote workers without changing the public
   request schema.
   Important detail: `/v1/models` must return the routed union of mesh-visible
   models, and route selection should preserve capability filters for tool
   calling, structured outputs, and response-state support.
   Acceptance: one mesh can serve several model families at once, and route
   selection stays typed and explainable instead of falling back to ad hoc
   proxy logic.
   Depends on: 4, 5, 6, 8.

10. `psionic`: "Add operator console surface for inference mesh"
    `mesh-llm`'s strongest product move is the embedded dashboard. `psionic`
    does not need to copy the exact UI, but it does need a first-party operator
    surface that makes the mesh legible before `openagents` layers product UX
    on top.
    Scope: add a thin console that consumes only the Psionic management API and
    shows peers, roles, join state, demand, per-model host election, warm
    replicas, routed endpoints, and refusal reasons. Keep mutation narrow at
    first: join, leave, load, unload, and maybe standby or drain controls.
    Likely landing point: `psionic-serve` web assets plus the new management
    routes, not `openagents`.
    Acceptance: an operator can diagnose "why is this model not serving" and
    "which node is hot standby right now" from the console without opening
    logs or source.
    Depends on: 4, 7, 8, 9.

11. `psionic`: "Add published install and service mode for mesh lanes"
    `psionic-gpt-oss-server` currently exposes a low-level binary with explicit
    flags for model path, backend, host, port, context length, GPU layers, and
    reasoning budget. That is useful for development. It is not the same as a
    stable operator install story.
    Scope: add one supported mesh-service entrypoint with a durable runtime
    layout for identity, network state, model cache, logs, and config. Provide
    one service-mode story for macOS and Linux, plus a documented upgrade path
    that preserves node identity and join state.
    Design rule: the service wrapper should compose existing `psionic-serve`,
    `psionic-net`, and management API contracts instead of hiding them behind
    opaque shell scripts.
    Acceptance: a new machine can install the mesh lane, restart it after a
    reboot, and come back with the same node identity and trusted mesh
    membership.
    Depends on: 3, 4, 6, 10.

12. `psionic`: "Port MoE mesh orchestration into native model-family contracts"
    `mesh-llm` has useful control-plane ideas around expert placement and
    low-cross-node-traffic execution. `psionic-cluster` already has native
    distributed execution surfaces in `tensor_sharded.rs`, `pipeline_sharded.rs`,
    and `replicated_serving.rs`. The port should land there, not in a
    `llama.cpp`-style RPC wrapper.
    Scope: add model-family contracts for expert placement, expert-host
    inventory, assignment digests, and execution topology truth. Reuse
    `ExecutionTopologyPlan` and the existing sharded-lane evidence patterns.
    Non-goal: do not import `mesh-llm`'s backend assumptions, GGUF-specific
    process topology, or direct llama.cpp RPC orchestration.
    Acceptance: Psionic can describe an MoE lane with native topology truth,
    explicit artifact identity, and honest refusal when the required expert
    placement cannot be satisfied.
    Depends on: 6, 7, 8, 9.

13. `psionic`: "Port blackboard-class shared coordination surface"
    `mesh-llm` exposes `/api/blackboard/feed`, `/api/blackboard/search`, and
    `/api/blackboard/post`, backed by the built-in `blackboard` plugin and MCP
    bridge. The useful part is the cross-node coordination seam. The risky part
    is letting it become hidden runtime state.
    Scope: add an optional coordination surface with typed post, feed, search,
    TTL, provenance, visibility, and redaction semantics. Keep it outside the
    critical execution path. A dedicated crate is likely cleaner than pushing
    this into `psionic-serve` or `psionic-router` directly.
    Important boundary: this is shared mesh coordination, not task truth,
    approval truth, or transcript truth.
    Acceptance: a mesh can share short-lived findings or status across nodes,
    and operators can disable the feature entirely without affecting inference
    correctness.
    Depends on: 4, 6.

14. `probe`: "Add mesh-backed backend profile and attach flow"
    `probe-protocol/src/backend.rs` currently has `OpenAiChatCompletions`,
    `OpenAiCodexSubscription`, and `AppleFmBridge`. `probe-core` has named
    local profiles like `psionic-qwen35-2b-q8-registry`. `probe/docs/11-server-attach-and-launch.md`
    already defines the attach-versus-launch boundary. The new mesh lane should
    fit that model cleanly.
    Scope: add one Probe-owned backend profile that targets a Psionic mesh
    control plane and one attach flow that resolves the effective OpenAI base
    URL plus available model inventory from the management API. Preserve the
    rule that Probe does not own backend startup semantics for the mesh.
    Design choice: either add a new `BackendKind` such as `PsionicMesh`, or
    extend the OpenAI-compatible profile with an explicit control-plane URL and
    discovery mode. The issue should settle that shape deliberately.
    Acceptance: Probe can attach to a mesh, list targetable models, surface
    degraded or proxied-mode truth, and continue to use typed session metadata.
    Depends on: 4, 5, 9.

15. `probe`: "Integrate shared blackboard or mesh coordination adjunct"
    `probe-server` already owns detached sessions, runtime events, and durable
    transcript state. The mesh coordination surface should plug in as an
    optional adjunct, not as a replacement for session history.
    Scope: add optional read or post tools, or a small session-side panel, that
    consume the coordination API from issue 13. Keep usage outside transcript
    invariants and do not treat coordination messages as authoritative task
    state.
    Acceptance: a Probe session can query or post shared mesh coordination
    state while preserving its current transcript, approval, and replay model.
    Depends on: 13, 14.

16. `openagents`: "Add pooled inference surface to compute-provider product"
    `openagents/apps/autopilot-desktop/src/desktop_control.rs` already exposes
    `DesktopControlSnapshot` with `provider`, `local_runtime`, `gpt_oss`,
    `tailnet`, `tunnels`, and `cluster` sections, plus `/v1/snapshot`,
    `/v1/events`, and `/v1/action`. The shape is already there. It is still
    missing pooled inference truth.
    Scope: extend the desktop-control snapshot and `autopilotctl` surfaces to
    show mesh membership, served-mesh role, local versus proxied serving state,
    warm replicas, standby posture, routed model inventory, and demand-driven
    contribution state. Add new pane or subcommand surfaces rather than burying
    the mesh inside generic GPT-OSS status.
    Acceptance: an operator can answer "what is this machine contributing right
    now" and "am I serving locally, standing by, or proxying into the pool"
    from Autopilot without using Psionic-only tooling.
    Depends on: 4, 7, 8, 9, 10.

17. `openagents`: "Bind pooled inference to compute-market product identity"
    `openagents-provider-substrate` currently has `ProviderComputeProduct`
    values such as `psionic.local.inference.gpt_oss.single_node`,
    `psionic.local.embeddings.gpt_oss.single_node`, and the cluster-attached
    adapter-training product. The current enum has no first-class pooled
    inference product family.
    Scope: add cluster-capable inference products and map them to the existing
    compute truth vocabulary: `ComputeTopologyKind`, `ComputeProvisioningKind`,
    backend family, capability summary, and provider health. Expected new
    product IDs should follow the current naming pattern, for example
    `psionic.cluster.inference.gpt_oss.remote_whole_request`,
    `psionic.cluster.inference.gpt_oss.replicated`, and later
    `psionic.cluster.inference.gpt_oss.tensor_sharded` if the lane is actually
    marketable.
    Acceptance: OpenAgents can quote and advertise pooled inference with typed
    topology and provisioning truth instead of pretending every lane is a
    single-node product.
    Depends on: 9, 16.

18. `openagents`: "Add multi-machine join and invite UX above Psionic mesh"
    `desktop_control.rs` already has `tailnet`, `tunnels`, and `cluster`
    sections, and `autopilotctl` already exposes operator subcommands for those
    surfaces. The missing layer is a human-oriented join flow above Psionic's
    join contract.
    Scope: add actions and UI for exporting join packages, importing them on a
    second machine, showing trust or refusal reasons, and confirming membership
    state after the join. Tailnet status can help operators discover devices,
    but Psionic remains the source of truth for actual mesh admission.
    Acceptance: a user can bring a second machine into the pool from the
    existing Autopilot operator flow without dropping into raw Psionic config
    files or manually editing admission state.
    Depends on: 3, 4, 10, 16, 17.

19. `openagents`: "Connect pooled inference to wallet, earnings, and market receipts"
    This is where the port stops being only an infra improvement. `openagents`
    already has provider, buyer-procurement, proof, challenge, and wallet
    surfaces in `desktop_control.rs`, plus Compute Market authority types in
    the kernel and provider substrate. Pooled inference needs to feed that
    truth.
    Scope: add contribution accounting, route or window receipts, and provider
    earnings semantics for pooled inference. Settlement should understand the
    difference between single-node serving, remote whole-request serving,
    replicated standby capacity, and any later sharded lane that is actually
    sellable.
    Acceptance: pooled inference produces market-facing provider truth instead
    of staying a private operator convenience feature, and the product can
    explain why one contribution earned or did not earn revenue.
    Depends on: 16, 17.

20. `psionic` plus `openagents`: "Retire standalone mesh sidecar posture"
    The last issue is cleanup, not invention. By this point the owned stack
    should already have native join flows, management API, routing, operator
    UI, provider products, and Probe attach support.
    Scope: remove any remaining sidecar-specific docs, compatibility shims, or
    operator instructions that imply `mesh-llm` remains a required parallel
    runtime. Keep the upstream repo only as a read-only harvest reference under
    `competition/`.
    Acceptance: the supported path for pooled inference runs entirely through
    Psionic, OpenAgents, and Probe contracts, and there is no leftover
    ambiguity about whether the mesh lane belongs to an external sidecar.
    Depends on: 11, 14, 15, 18, 19.

## What We Should Not Port

## 1. Do not make Psionic a hidden llama.cpp wrapper

That would weaken:

- runtime ownership
- backend flexibility
- artifact truth
- refusal and capability publication

## 2. Do not collapse OpenAgents product surfaces into mesh chat

`openagents` is building:

- Autopilot
- wallet and payout flows
- compute and data market UX
- kernel authority and settlement surfaces

`mesh-llm` does not replace that product stack.

## 3. Do not treat blackboard or plugins as a substitute for Probe

`probe` needs:

- approvals
- task state
- durable transcripts
- runtime protocol guarantees

`mesh-llm`'s plugin system is useful, but it is not the same class of runtime.

## 4. Do not import the public-mesh trust model blindly

`mesh-llm` is comfortable with:

- public mesh discovery
- invite tokens
- mesh publication through Nostr

Some of that is useful.
It is not automatically the correct trust model for:

- kernel-bearing `openagents`
- evidence-first `psionic`

## Final Thesis

`mesh-llm` is one of the clearest open-source harvest targets in the
distributed-inference product lane.

It currently shows one thing this workspace still needs to absorb:

- how quickly a user can turn a pile of machines into one visible shared
  inference surface

This workspace already has the stronger long-term ownership split where the
system needs to be deeper than that:

- `psionic` for execution truth, training, and evidence
- `openagents` for product and economic authority
- `probe` for coding-runtime truth

The right move is not a wholesale embed of `mesh-llm` as a parallel stack.

The right move is to absorb its best distributed-inference UX, mesh operator
patterns, and selected implementation ideas into our own boundaries, then
combine them with `psionic` runtime truth and the larger OpenAgents Compute
Market rather than treating pooled inference as a separate product universe.

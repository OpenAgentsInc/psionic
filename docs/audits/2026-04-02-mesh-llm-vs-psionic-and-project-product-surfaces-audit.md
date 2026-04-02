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

The sequence should be:

1. `psionic`: "Freeze mesh-integration target and owner split"
   This issue should lock the porting boundary before code spreads. It should
   state directly that `psionic` owns runtime, transport, topology, and mesh
   truth, `openagents` owns compute-market and provider productization above
   that truth, and `probe` owns agent-runtime integration above the inference
   backend.

2. `psionic`: "Add mesh node-role contract"
   Port the explicit `host`, `worker`, `standby`, and `thin_client` role model
   into a typed Psionic contract with machine-readable status publication,
   refusal states, and upgrade or downgrade reasons.

3. `psionic`: "Add mesh identity, invite, and join contract"
   Port the mesh identity and join semantics into Psionic-owned manifests and
   receipts. This should cover private mesh identity, named mesh identity,
   invite-token or join-package format, and last-joined mesh preference
   tracking without inheriting `mesh-llm`'s exact external contract blindly.

4. `psionic`: "Add inference-mesh management API"
   Port the small operator API shape into Psionic. The minimum useful surface
   is a Psionic-owned equivalent of status, event stream, discoverable meshes,
   and join status, with runtime truth and capability publication folded into
   the same surface instead of sitting beside it.

5. `psionic`: "Add bootstrap proxy and thin-client mode"
   Port the join-before-load experience. A thin client or joining worker should
   be able to attach to an existing mesh and serve requests through a bootstrap
   proxy before local warmup is complete, while keeping runtime identity and
   fallback posture explicit.

6. `psionic`: "Add QUIC mesh transport for remote inference lanes"
   Port the tunnel and peer-management class of behavior into Psionic-owned
   transport instead of a llama.cpp sidecar. This issue should cover stream
   families, admission model, peer lifecycle, reconnect or death handling, and
   direct worker-to-worker transfer rules for native Psionic execution lanes.

7. `psionic`: "Add per-model host election and standby promotion"
   Port the simple operational control loop that makes the mesh usable:
   per-model election groups, standby promotion, and demotion or reassignment
   when topology or model availability changes.

8. `psionic`: "Add demand gossip and hot-model rebalance"
   Port request-rate and demand propagation into native Psionic mesh status so
   large or idle nodes can promote the right model family without operator
   guesswork. This issue should stay tied to explicit receipts and not become
   hidden heuristics.

9. `psionic`: "Add multi-model mesh router"
   Port model-aware routing above the native Psionic serving lanes so a mesh
   can host several models at once and route by request `model` field while
   keeping per-model capability envelopes explicit.

10. `psionic`: "Add operator console surface for inference mesh"
    Port the best part of the `mesh-llm` user story: a live topology and model
    view that makes the system legible. This can start as a thin web console or
    equivalent operator surface backed entirely by the Psionic management API.

11. `psionic`: "Add published install and service mode for mesh lanes"
    Port the operational packaging layer: install script, background service
    posture, cold-start rules, and upgrade-safe runtime layout. This matters
    because the current gap is not only runtime logic. It is operator
    repeatability.

12. `psionic`: "Port MoE mesh orchestration into native model-family contracts"
    Port the useful orchestration ideas from `mesh-llm`'s expert-sharding lane
    into Psionic-owned model-family logic. Do not import llama.cpp-specific
    assumptions. Keep the port focused on the control plane, assignment logic,
    and zero-cross-node-traffic design where it remains valid.

13. `psionic`: "Port blackboard-class shared coordination surface"
    Port the blackboard idea into a Psionic-adjacent coordination primitive
    only if it is defined as a clear optional surface and not as hidden runtime
    state. This issue should focus on message propagation, search, retention,
    privacy posture, and machine-readable transport semantics.

14. `probe`: "Add mesh-backed backend profile and attach flow"
    Once the Psionic mesh API exists, `probe` should get a first-class backend
    profile that can target a Psionic mesh directly, including attach, health,
    model selection, and explicit degraded-mode messaging.

15. `probe`: "Integrate shared blackboard or mesh coordination adjunct"
    If the blackboard-class surface lands, `probe` should expose it as an
    optional adjunct to sessions instead of building a second agent-runtime
    abstraction for the same problem.

16. `openagents`: "Add pooled inference surface to compute-provider product"
    Once Psionic mesh lanes are real, `openagents` should expose pooled
    inference as part of the Compute Market provider story, not as a detached
    engineering demo. This issue should cover provider inventory, operator
    visibility, and "what is my machine contributing right now?" product
    surfaces.

17. `openagents`: "Bind pooled inference to compute-market product identity"
    This issue should connect the new mesh substrate to actual Compute Market
    product identities, capability envelopes, receipts, and settlement-facing
    provider truth.

18. `openagents`: "Add multi-machine join and invite UX above Psionic mesh"
    Port the good human-facing join flow into Autopilot. The product layer
    should let a user add another trusted machine quickly, but the underlying
    truth must remain Psionic-owned.

19. `openagents`: "Connect pooled inference to wallet, earnings, and market receipts"
    This is where the absorbed mesh work stops being only infra. The pooled
    inference surface should feed actual provider, payout, and compute-market
    receipts rather than staying a local-serving convenience layer.

20. `psionic` plus `openagents`: "Retire standalone mesh sidecar posture"
    The final issue in the sequence should remove any residual dependency on a
    parallel `mesh-llm`-style sidecar product and declare the owned mesh lane
    complete inside our own runtime, product, and market boundaries.

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

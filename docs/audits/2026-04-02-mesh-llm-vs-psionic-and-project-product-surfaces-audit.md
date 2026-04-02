# Mesh-llm Versus Psionic And Project Product Surfaces Audit

Date: 2026-04-02

Scope: audit of the local synced `competition/repos/mesh-llm` checkout at
`65ac783`, compared against the current `psionic`, `openagents`, and `probe`
docs in this workspace. This document answers four questions:

1. what `mesh-llm` actually is today
2. where it overlaps directly with `psionic`
3. where it overlaps with product surfaces in this workspace
4. what we should harvest, and what we should explicitly not adopt

## Bottom Line

`mesh-llm` is not a whole-stack competitor to this workspace.

It is a strong competitor in one narrower band:

- distributed inference bring-up
- zero-friction node joining
- operator-friendly mesh status and routing UX
- agent-adjacent coordination on top of a pooled GPU mesh

It is strongest exactly where current `psionic` is still the least productized:

- easy multi-node inference packaging
- instant join-and-serve behavior
- lightweight management API plus web console
- consumer-grade install and service mode
- agent-facing coordination patterns such as blackboard and launcher shortcuts

It is weak exactly where `psionic` and the broader workspace are intentionally
stronger:

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

`mesh-llm` does not compete here.

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

## 3. Dataroom And Other Workspace Product Surfaces

There is effectively no direct overlap with `dataroom`.

`mesh-llm` does not address:

- investor access control
- allowlists
- private portal UX
- WorkOS flows

This repo is not a competitor in that lane.

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

## 3. Into Probe

- blackboard-style shared coordination as an optional adjunct surface
- easier attach flows to pooled inference backends
- possibly a simple mesh-backed profile for narrow distributed inference use

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

## Final Verdict

`mesh-llm` is a serious competitor in the distributed-inference product lane.

It currently beats this workspace on one concrete axis:

- how quickly a user can turn a pile of machines into one visible shared
  inference surface

This workspace still has the stronger long-term architecture where the system
needs to be deeper than that:

- `psionic` for execution truth, training, and evidence
- `openagents` for product and economic authority
- `probe` for coding-runtime truth

The correct response is not to copy `mesh-llm` whole.

The correct response is to harvest its best distributed-inference UX and mesh
operator patterns, then rebuild those ideas on top of the stronger ownership
split that this workspace already documents.

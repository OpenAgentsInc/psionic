# Inference Mesh Ownership

> Status: implemented on 2026-04-02 as the canonical owner split for the mesh
> integration program.

This document freezes the repo boundary for upcoming inference-mesh work.

The purpose is simple: later issues should add typed contracts for identity,
join, routing, and management without reopening the question of which repo owns
which layer.

## Owned Terms

### Inference Mesh

An inference mesh is a durable set of Psionic nodes that share node identity,
admission truth, topology truth, model inventory, route selection, and
management state for one pooled-inference lane.

It is not a product by itself. It is the execution substrate behind a product
surface.

### Join

A join is the Psionic admission flow that moves a machine from unknown candidate
to admitted mesh member. Join truth includes node identity, invite or admission
material, trust checks, role assignment, and ordered membership state.

### Invite

An invite is portable admission material for a future join. A product surface
may display or transmit an invite, but Psionic owns invite validation, invite
digest truth, and the resulting admission decision.

### Bootstrap Proxy

A bootstrap proxy is a Psionic-managed request path that lets a cold joiner or
thin client answer the normal inference API before local model residency is
ready. It is execution routing truth, not app-only glue.

### Thin Client

A thin client is an admitted mesh node that exposes the normal API and mesh
identity but does not need to host the selected model locally. It forwards work
through Psionic route selection instead of inventing a second control plane.

### Standby

A standby is an admitted node reserved for promotion into active serving when a
host fails, drains, or demand rises. Standby truth includes role, warm state,
promotion reason, and current election state.

### Pooled Inference

Pooled inference is the higher-level product claim that one request can be
served by the mesh rather than one fixed local host. Inside Psionic, the owned
truth is model inventory, node roles, route selection, execution receipts, and
refusal reasons that make that claim honest.

## Repo Split

### Psionic Owns

Psionic owns the machine-facing truth for the inference mesh:

- serving semantics
- node identity and signed mesh identity material
- invite validation and join admission
- topology, membership, and role truth
- standby, drain, and promotion state
- routed model inventory and capability publication
- route selection between local and remote workers
- bootstrap proxy and thin-client runtime semantics
- management API truth
- execution receipts, refusal reasons, and evidence

The current landing points are:

- `crates/psionic-net`
  - identity, admission, session claims, and transport substrate
- `crates/psionic-cluster/src/ordered_state.rs`
  - authoritative membership, role, election, and ordered state
- `crates/psionic-router`
  - routed worker and model inventory plus selection policy
- `crates/psionic-serve/src/openai_http.rs`
  - served API, model publication, management routes, and bootstrap behavior

### OpenAgents Owns

`openagents` owns the compute-market product layer above Psionic truth:

- pooled-inference product packaging
- operator UX above the management API
- provider and buyer workflow presentation
- wallet, payout, and settlement behavior
- market receipts and commercial state above execution receipts

`openagents` may consume mesh identity, model inventory, or admission status,
but it does not become the source of truth for those facts.

### Probe Owns

`probe` owns the coding-runtime seam above the inference backend:

- backend profiles that target a Psionic mesh
- attach semantics for an already-running mesh lane
- runtime integration that selects a model or base URL from Psionic-managed
  inventory

`probe` does not own mesh startup, node identity, admission, routing, standby
policy, or serving semantics.

## Explicit Non-Goals For Psionic

The inference-mesh program does not move these into `psionic`:

- wallet logic
- payout logic
- settlement authority
- provider marketplace UI
- buyer or provider procurement policy
- agent transcript truth
- agent task ownership
- desktop or mobile operator product UX

## Change Rule For Later Issues

Later mesh issues should extend typed Psionic contracts in this order:

1. identity, invite, and admission in `psionic-net`
2. membership, roles, and ordered state in `psionic-cluster`
3. routed worker and model inventory in `psionic-router`
4. served API and management surfaces in `psionic-serve`

Consumer repos should integrate above those contracts. They should not redefine
mesh identity, admission, topology, or routing in product code.

## Role Split

Mesh management state must publish two separate role surfaces:

- transport role
  - `coordinator_only`, `executor_only`, or `mixed`
  - owned by `psionic-net` transport and identity truth
- served-mesh role
  - `host`, `worker`, `standby`, or `thin_client`
  - owned by mesh serving and routing truth

The served-mesh role also carries posture and machine-readable reasons such as
`warming`, `artifact_missing`, `admission_refused`, `draining`, and
`remote_only`.

Those roles must not be collapsed back into one enum. A node can honestly be
`mixed` at the transport layer and `standby` at the served-mesh layer at the
same time.

## Per-Lane Host Election

Replicated serving now depends on one ordered host-election record per replica
lane.

That record lives in `psionic-cluster` ordered state and carries:

- election `term`
- `active_host_node_id`
- `standby_node_ids`
- `promoted_from_node_id`
- explicit promotion `reason`
- explicit lease state for the current host epoch

That means active-host truth is no longer an implicit byproduct of local serve
memory. It is replayable cluster truth with split-brain protection.

The execution contract above that ordered record is:

- same-lane same-term conflicting active hosts are refused
- one warm standby is promoted with one explicit next-term transition
- replicated routing only admits the elected active host for the lane
- management state can explain the current term, promotion reason, and standby
  set without inferring them from traffic

## Join Bundle Contract

Mesh setup now depends on one durable join bundle contract above the existing
admission and introduction types.

That bundle carries:

- mesh label
- namespace and cluster identity
- advertised control-plane addresses
- trust-policy metadata
- either shared admission material or a signed introduction envelope

Durable network state keeps the last imported join bundle separate from the
last joined mesh preference. That keeps setup intent and transient transport
state from collapsing into the same record.

## Management Namespace

Mesh management now lives in a dedicated served namespace instead of leaking
operator state into the OpenAI-compatible endpoints.

The first published paths are:

- `/psionic/management/status`
  - one typed snapshot for join posture, node roles, warm model state, and
    route inventory
- `/psionic/management/events`
  - one SSE stream that emits an initial topology snapshot and live
    route-selection events

Those responses must stay backed by typed router and network truth. They are
not allowed to depend on scraping logs or inferring topology from user traffic.

## Bootstrap Proxy Runtime

Bootstrap proxy mode is now configured directly on the Psionic OpenAI-compatible
server.

The current operator knobs are:

- `PSIONIC_BOOTSTRAP_PROXY_BASE_URL`
  - remote Psionic base URL used for bootstrap discovery and request proxying
- `PSIONIC_BOOTSTRAP_PROXY_MODE`
  - `thin_client` or `warming`

When bootstrap mode is enabled, the local server:

- discovers warm remote inventory from `/psionic/management/status`
- binds `/v1/chat/completions`, `/v1/responses`, and `/v1/embeddings` locally
- proxies only to warm remote workers that honestly match the local model plan
- publishes route execution provenance on both headers and management state

The execution proof surface is:

- response headers
  - `x-psionic-route-locality`
  - `x-psionic-route-provenance`
  - `x-psionic-route-warm-state-reason`
  - `x-psionic-route-fallback-posture`
- management status
  - `last_route_execution`

The local node truth differs by mode:

- `thin_client`
  - served-mesh role `thin_client`
  - reason `remote_only`
  - fallback posture `thin_client_remote_only`
- `warming`
  - served-mesh role `host`
  - posture `downgraded`
  - reason `warming`
  - fallback posture `warming_until_local_ready`

## Wider-Network Stream Transport

Explicit wider-network discovery is no longer a placeholder claim. It is now
gated by one typed remote-stream lane contract on each configured peer.

That lane must publish all four capabilities:

- `gossip`
- `join_control`
- `management_subscription`
- `request_forwarding`

When those capabilities are present, Psionic can honestly mark
`explicit_wider_network_requested` as eligible. When any capability is missing,
the refusal is machine-checkable and specific instead of falling back to one
generic "not implemented" posture.

The public cluster transport truth for those lanes is:

- `psionic-net`
  - direct, NAT-assisted, and relay-forwarded stream paths
- `psionic-cluster`
  - `wider_network_stream` transport class

The internal transport implementation remains replaceable. The public contract
is the wider-network stream lane and its admission, routing, and management
behavior.

## Session Generation And Ghost-Peer Fencing

Configured wider-network peers now carry explicit reconnect bookkeeping:

- peer `session_generation`
- peer `disconnect_count`
- peer `last_activity_ms`

If a configured remote lane stops answering long enough to be considered idle,
Psionic tears down the peer snapshot, closes any open tunnels with
`transport_unavailable`, and forces the next observation to land as a new
session generation.

That keeps membership transitions deterministic across restart or relay loss:

- stale peers are removed instead of lingering as ghost members
- restarts advance node epoch and peer session generation
- remote tunnels and health snapshots show one explicit disconnect before
  reconnect

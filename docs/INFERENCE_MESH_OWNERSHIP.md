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

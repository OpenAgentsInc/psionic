# 2026-03-22 Tassadar Full Plugin-System State Audit

This audit reviews the current plugin system in `psionic` after the starter
plugin user-authoring wave and compares it against the original local design
plans that informed the work.

The comparison inputs for this audit were:

- the current repo docs in `docs/`, especially
  `docs/ARCHITECTURE.md`,
  `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md`,
  `docs/TASSADAR_STARTER_PLUGIN_AUTHORING.md`,
  `docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md`,
  `docs/TASSADAR_STARTER_PLUGIN_CATALOG.md`,
  `docs/TASSADAR_STARTER_PLUGIN_WORKFLOW_CONTROLLER.md`,
  `docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`,
  `docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md`,
  `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md`, and
  `docs/TASSADAR_STARTER_PLUGIN_USER_AUTHORING_WAVE.md`
- the current committed plugin reports and run bundles under `fixtures/tassadar/`
- the original local alpha planning docs
  `~/code/alpha/tassadar/plugin-system.md` and
  `~/code/alpha/autopilot/autopilot-extism-plugin-porting-into-rust-runtime.md`

Those alpha docs are historical planning inputs for comparison here. They are
not the canonical current-state spec for this repo.

## Executive Summary

The repo now has a real bounded plugin system, not just a design sketch.

The lower platform layers are implemented and connected:

- plugin charter and identity
- manifest or artifact identity discipline
- packet ABI and Rust-first contract
- runtime API and engine abstraction
- invocation receipts and replay classes
- world-mount admissibility and capability mediation
- conformance harness and result-binding contract
- canonical weighted controller trace
- authority, promotion, publication, and trust-tier gate
- bounded plugin-platform closeout

Above that lower platform, the repo now has a real starter-plugin layer with
real runtime execution, real receipts, real controller lanes, and one real
user-authored extension path.

The current starter-plugin system is still intentionally narrow:

- operator-curated and operator-internal
- host-native Rust starter plugins
- explicit capability and publication boundaries
- no public marketplace
- no arbitrary external binary loading
- no arbitrary software-capability claim

That means the system is materially real and usable inside its boundary, but it
is not yet the broader plugin platform imagined by the earliest alpha planning
documents.

## What Is Implemented Now

### 1. Lower plugin platform

The repo already carries the lower plugin substrate described in
`docs/ARCHITECTURE.md`.

What is green at that layer:

- one bounded plugin identity and manifest contract
- one packet ABI and Rust-first plugin contract
- one host-owned runtime API and engine abstraction
- one receipt and replay contract
- one world-mount admissibility compiler
- one conformance and benchmark harness
- one result-binding and schema-stability contract
- one canonical weighted plugin controller trace
- one authority and trust-tier gate
- one bounded plugin-platform closeout

The bounded platform closeout report keeps the current high-level posture
explicit:

- `plugin_capability_claim_allowed = true`
- `plugin_publication_allowed = false`
- `served_public_universality_allowed = false`
- `arbitrary_software_capability_allowed = false`

The authority gate also stays explicit about posture instead of hiding it
behind green implementation detail:

- 4 trust-tier rows
- 5 publication-posture rows
- weighted plugin control allowed
- plugin publication still blocked

### 2. Starter-plugin runtime

The runtime-owned starter-plugin layer now has five real starter plugins:

- `plugin.text.url_extract`
- `plugin.text.stats`
- `plugin.http.fetch_text`
- `plugin.html.extract_readable`
- `plugin.feed.rss_atom_parse`

This is not a mock catalog. Each plugin has:

- runtime code
- typed packet schemas
- typed refusal classes
- a committed runtime bundle
- a checker script
- explicit negative claims

The central runtime registry is now the source of truth for:

- plugin id
- tool name
- input and output schema ids
- refusal schema ids
- replay class
- capability class
- capability namespaces
- origin class
- mount envelope id
- runtime bundle identity
- bridge exposure
- catalog exposure

That centralization matters because it removed the old risk that runtime,
bridge, catalog, and controller surfaces would drift through duplicated
metadata tables.

### 3. User-added starter-plugin path

The repo now has two real user-added starter-plugin proofs:

- `plugin.text.stats` for the capability-free local deterministic class
- `plugin.http.fetch_text` for the first manual `networked_read_only` class

Together they prove two different bounded authoring paths:

- one low-friction capability-free local deterministic path
- one narrower manual networked path with explicit mount and replay policy

The shared proof shape is:

- add one plugin registration row in the runtime registry
- define typed packet structs and invocation code
- emit a real receipt and runtime bundle
- expose it through the shared tool bridge
- publish it into the starter catalog
- run it in a deterministic controller flow
- admit it into the canonical weighted controller lane

This is the point where the repo proves that user-added plugins are not just
runnable in isolation, but can actually move through the same shared path used
by the existing starter-plugin system across both the capability-free and
bounded networked classes.

### 4. Shared bridge and controller surfaces

The shared starter-plugin bridge is now real and reusable across multiple
surfaces.

It projects five starter plugins into:

- deterministic workflow tools
- router `/v1/responses` function tools
- Apple FM tool definitions

The bridge also preserves:

- typed success versus refusal status
- structured output payloads
- plugin identity
- full plugin invocation receipts

Above that bridge, the repo now has four real controller or orchestration
surfaces:

- one deterministic workflow controller
- one router-owned `/v1/responses` tool loop
- one local Apple FM plugin session lane
- one canonical weighted plugin controller trace

The current weighted controller report keeps the strongest controller facts
machine-legible:

- 5 controller-case rows
- 40 control-trace rows
- 10 explicit host-negative rows
- `weighted_plugin_control_allowed = true`

The weighted controller lane also now includes two bounded admission rows:

- `plugin.text.stats`
- `plugin.http.fetch_text`

and it still carries explicit model-selected success traces instead of hidden
host-only admission logic.

### 5. Catalog and corpus surfaces

The starter catalog report currently freezes:

- 5 starter-plugin entries
- 5 capability rows
- 2 bounded composition rows
- 8 validation rows

The current starter catalog is still operator-only and explicitly not a public
plugin marketplace.

The multi-plugin trace corpus now freezes:

- 3 source bundles
- 6 normalized trace records
- 2 workflow parity rows
- 0 disagreement rows
- `bootstrap_ready = true`

That gives the repo one shared evidence surface across deterministic, router,
and Apple FM plugin-controller lanes instead of three disconnected local demos.

## Where The Current System Matches The Original Alpha Plans

The earliest alpha plans were broader, but the repo now matches their core
design law in several important ways.

### Model logic versus host execution

The original `plugin-system.md` said:

- the model owns plugin choice, arguments, sequencing, retries, and stop
  conditions
- the runtime owns bounded execution, capability mediation, and receipts

The current weighted controller lane matches that law. The repo now has one
canonical controller trace that keeps model-owned control explicit and host
planner attacks explicit as negative rows.

### Explicit contracts instead of loose tool calls

The alpha plan wanted:

- packet ABI
- typed refusal
- explicit capability envelope
- explicit receipt and replay posture

The current repo matches that shape. The plugin system is contract-heavy,
receipt-heavy, and fail-closed. It does not flatten plugin execution into
ambient stringly typed tool calls.

### Runtime substrate separate from plugin implementations

The Rust starter-plugin memo said the runtime should remain infrastructure and
the starter plugins should remain plugins above it rather than being recast as
runtime built-ins.

The current repo also matches that. The starter runtime is a shared substrate,
and the starter plugins remain individually named runtime entries with their
own schemas, refusals, bundles, and docs.

### Mount and policy discipline

The alpha plans called for mount-derived capability control rather than
ambient authority.

The current system keeps that explicit:

- capability-free local deterministic plugins stay capability-free
- networked plugins require explicit mount-envelope policy
- replay posture stays explicit
- publication posture stays explicit

### Publication and trust are gated

The alpha plans required promotion and publication policy, not automatic
admission.

The current repo matches that too. Weighted plugin control is green, but
publication is still suppressed and the trust-tier or authority gate remains
separate and explicit.

## Where The Current System Is Narrower Than The Original Alpha Plans

The differences are real and should be stated plainly.

### Host-native starter plugins, not guest Wasm starter plugins

The original `plugin-system.md` centered a weighted Wasm plugin system and the
starter-suite memo talked about Rust-native plugins compiled for the new
runtime.

The current implemented starter-plugin path is host-native Rust execution
inside the repo runtime layer. It does not yet give repo users a
digest-addressed guest-Wasm plugin artifact path.

That is not a hidden defect in the current docs. It is the actual current
boundary.

### Small starter suite, not full starter-suite closure

The original starter-suite memo targeted 18 canonical starter plugins across
four classes.

The current repo has:

- 5 implemented starter plugins
- 4 local deterministic plugins
- 1 read-only network plugin

The secret-backed network class and the stateful adapter class are not yet
implemented as a user-authoring or starter-suite closure path.

### User authoring is intentionally class-limited

The current first-class user path is:

- host-native
- capability-free
- local deterministic

The current docs do distinguish `networked_read_only` as a second class, but
that path is still intentionally manual and more restrictive. It is not yet
the same smooth end-to-end authoring experience as the capability-free path.

### Operator-internal only

The original plan left publication policy for later phases.

The current repo is still clearly inside that earlier posture:

- operator-curated
- operator-internal
- no public marketplace
- no general plugin publication
- no arbitrary external admission

### No broad plugin-platform claim widening

The alpha planning documents were careful about this, and the current repo is
still careful about it.

What the repo should not claim yet:

- public plugin-platform closure
- arbitrary binary loading
- arbitrary third-party plugin publication
- arbitrary software capability
- public served plugin universality

## Audit Verdict

The full plugin system is now real, internally coherent, and substantially
further along than the original alpha plans required for an initial usable
system.

The strongest honest statement today is:

- the lower bounded plugin platform is implemented
- the starter-plugin runtime and controller wave are implemented
- two bounded user-added plugin paths are implemented end to end
- canonical weighted plugin control is implemented for the bounded admitted set
- publication and broader platform widening remain intentionally blocked

That is a meaningful green state.

The strongest honest negative statement today is:

- the repo has not yet closed the broader starter-suite memo
- the repo has not yet closed secret-backed or stateful starter-plugin
  authoring
- the repo has not yet closed a public or general external plugin platform
- the repo has not yet reintroduced a guest-artifact plugin path if that still
  matters

## Recommended Next Steps

The next logical steps should stay ordered and explicit.

### 1. Add the first secret-backed starter plugin only after the secret mount contract is explicit

The alpha memo correctly treated secret-backed plugins as a separate class.

The next plugin wave should keep these invariants:

- no secrets inside user packets
- host-mediated header or token injection only
- typed refusal for missing or denied secret access
- explicit receipt posture for upstream secret-backed calls

One clean Class C plugin would do more to prove the platform than adding many
more capability-free or networked starter plugins.

### 2. Defer the first stateful adapter until durable truth is clearly host-backed

The alpha memo also correctly separated the stateful adapter class.

Do not blur this with ordinary network adapters. The first stateful plugin
should land only after the store-capability and durable-truth contract is
explicit enough that plugin-local cache cannot masquerade as canonical state.

### 3. Keep the networked class narrow even after the first proof

The first `networked_read_only` proof should not be mistaken for broad
automation or relaxed policy.

- no silent fallback from capability-free scaffolding into networked authoring
- no hidden widening of mount envelopes or replay posture
- no networked plugin admission without explicit refusal, receipt, and
  controller evidence

### 4. Expand the starter suite deliberately, not by one-off exceptions

The current five-plugin starter set is coherent. The next additions should
still come through the central registry and the shared derivation path instead
of reintroducing per-surface custom wiring.

The highest-value near-term additions are the ones that close missing class
coverage from the alpha memo rather than adding more variants of already-proven
local deterministic transforms.

### 5. Decide the artifact direction explicitly

The current repo now has a working host-native plugin system.

If that is the intended long-term plugin authoring product, the docs should
continue to say that clearly and the older Wasm-heavy planning docs should be
treated as historical design input.

If guest-artifact loading still matters, it should return as a separate narrow
issue wave with its own claim boundary:

- one digest-bound guest artifact
- one bounded runtime-loading path
- one receipt-equivalent invocation path
- no publication widening by implication

That should be a deliberate product decision, not an accidental leftover from
older planning language.

### 6. Grow the trace corpus only when new controller classes or plugin classes land

The current corpus is enough to prove the first connected orchestration wave,
but it is still small.

The best next expansion is not random volume. It is:

- first networked user-authored traces
- first secret-backed traces
- first stateful traces when that class exists
- more weighted-controller user-plugin rows

That keeps the corpus aligned with actual frontier movement.

### 7. Keep publication blocked until the class ladder is broader

The current authority and closeout docs are correct to keep publication off.

Before any widening of publication posture, the repo should first prove:

- more than the current two bounded user-added starter-plugin examples
- more than the current two authoring classes
- stable authority, trust-tier, and revocation posture across those classes
- no drift between runtime truth, catalog truth, controller truth, and receipt
  truth

## Bottom Line

Compared to the original alpha plans, the current plugin system is best
described as:

- deeper than the old plans at the bounded proof and receipt layer
- narrower than the old plans at the starter-suite breadth and artifact-loading
  layer
- materially real today for bounded internal use
- not yet the broad external plugin platform those older plans left open for
  later phases

That is the correct current reading of the system.

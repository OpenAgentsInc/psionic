# 2026-03-21 Multi-Plugin Real-Run Orchestration Audit

> Historical note: this is a point-in-time architecture audit written on
> 2026-03-21. Current canonical product and claim authority still lives in
> `docs/ARCHITECTURE.md`, `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`,
> `docs/ROADMAP_TASSADAR_TAS_SYNC.md`, `docs/ROADMAP_FM.md`, and the landed
> machine-readable artifacts they reference. This document is a planning and
> sequencing audit for real multi-plugin runs, not a statement that weighted
> plugin control is already closed.

## Intent

This audit answers one practical follow-up question:

> If Psionic now has a real bounded plugin charter, manifest contract, and
> packet ABI, what should the repo do next to run several plugins in one real
> workflow where a directive leads to multiple plugin calls in sequence?

The harder version of that question matters too:

> What should Psionic do if the current Tassadar Transformer cannot yet own
> those multi-plugin decisions by itself?

The wrong answer is:

- pretend the current canonical weights-only route already owns multi-plugin
  planning
- or collapse the problem into app-local glue and call that plugin closure

The right answer is:

- separate the plugin runtime from the controller that chooses plugins
- ship one honest operator-internal orchestration path quickly
- reuse existing Psionic tool-call surfaces where they already exist
- and treat weights-only multi-plugin control as a later proof-bearing lane
  built on top of those experiments, not as the first thing the repo must
  pretend is solved

## Scope

Primary plugin-system inputs reviewed:

- historical plugin-system design notes used only as background context
- starter-plugin porting notes used only as background context
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`

Primary current-state plugin surfaces reviewed:

- `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- `docs/audits/2026-03-21-tassadar-post-article-plugin-charter-authority-boundary.md`
- `docs/audits/2026-03-21-tassadar-post-article-plugin-manifest-identity-contract.md`
- `docs/audits/2026-03-21-tassadar-post-article-plugin-packet-abi-and-rust-pdk.md`
- `crates/psionic-runtime/src/tassadar_post_article_plugin_packet_abi_and_rust_pdk.rs`
- `crates/psionic-catalog/src/tassadar_post_article_plugin_manifest_identity_contract.rs`

Primary current-state multi-step controller surfaces reviewed:

- `docs/STRUCTURED_AGENT_WEATHER_PILOT.md`
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
- `crates/psionic-router/src/tool_loop.rs`
- `crates/psionic-router/src/response_state.rs`
- `crates/psionic-serve/src/lib.rs`

Primary Apple Foundation Models surfaces reviewed:

- `docs/ROADMAP_FM.md`
- `docs/FM_API_COVERAGE_MATRIX.md`
- `crates/psionic-apple-fm/src/lib.rs`
- `crates/psionic-apple-fm/src/tool.rs`
- `crates/psionic-apple-fm/src/client.rs`

Primary data and training bootstrap surfaces reviewed:

- `crates/psionic-data/src/apple_adapter.rs`
- `fixtures/apple_adapter/datasets/tool_calling_train.jsonl`
- `docs/APPLE_ADAPTER_DATASET_SPEC.md`

Starter-plugin issue wave reviewed:

- `TAS-216` through `TAS-220`

## Executive Verdict

Current audit verdict:

- plugin charter, manifest identity, and packet ABI for bounded plugins:
  `implemented`
- host-owned plugin runtime API, invocation receipts, world-mount envelope
  compiler, conformance sandbox, and result-binding contract: `partial`
- deterministic host-wired multi-plugin workflow lane above the plugin runtime:
  `planned`
- served-model multi-plugin controller using existing `/v1/responses` plus
  router-owned tool loops: `implemented_early`
- Apple FM multi-plugin controller using session-aware tool callbacks:
  `implemented_early`
- shared plugin-to-tool projection and receipt bridge across those controller
  lanes: `planned`
- canonical weights-only multi-plugin control on the rebased Tassadar route:
  `planned`

The short answer is:

- no, the repo should not wait for full weights-only plugin control before it
  runs real multi-plugin workflows
- no, the repo should not hide the gap by calling app-local glue "weighted
  plugin control"
- yes, the repo should build one shared plugin runtime and catalog, then drive
  it through three explicit controller lanes:
  1. deterministic host-wired workflows for known chains
  2. router-owned served-model tool loops for flexible experimentation
  3. Apple FM session tool loops for the macOS local lane
- and only after those lanes produce stable traces, receipts, and refusal truth
  should the repo try to close a true weights-only multi-plugin controller on
  the canonical Tassadar route

## What A Real Multi-Plugin Run Actually Needs

The repo should treat a real multi-plugin run as more than "the model emitted
two tool names."

One honest run needs all of the following:

1. A named plugin catalog
   - each plugin needs manifest identity, packet schemas, capability
     requirements, limits, replay class, and trust posture
2. A real invocation substrate
   - a host-owned runtime must encode packets, invoke plugins, enforce limits,
     and capture receipts
3. A controller
   - something must decide whether to call one plugin, several plugins, or stop
4. Explicit composition semantics
   - plugin outputs must bind back into the next controller-visible state
     without lossy schema repair or silent host rewriting
5. A bounded state model
   - prompt history, response state, tool results, and plugin receipts must stay
     explicit and challengeable
6. A failure lattice
   - policy refusal, plugin refusal, runtime failure, timeout, and controller
     stop conditions must stay typed
7. A trace and receipt story
   - a multi-step run must show which controller made each decision and which
     plugin invocation actually happened
8. Honest claim boundaries
   - host-wired control, served-model tool use, Apple FM sessions, and
     weights-only control are different claim classes and must stay separate

## What Exists Today

### 1. The first plugin-runtime contract surface is real

The plugin lane is no longer only a design memo.

`main` now has:

- `TAS-197` landed:
  plugin charter, authority boundary, operator/internal-only posture, and
  platform laws
- `TAS-198` landed:
  canonical plugin manifest, identity, and hot-swap contract
- `TAS-199` landed:
  canonical `packet.v1` ABI and Rust-first guest PDK

That means the repo now has enough contract truth to stop speaking vaguely
about "plugins" and start treating them as named bounded software artifacts.

### 2. The orchestration-critical plugin tranche is still open

The following issues remain open:

- `TAS-200`:
  host-owned plugin runtime API and engine abstraction
- `TAS-201`:
  plugin invocation receipts and replay classes
- `TAS-202`:
  world-mount envelope compiler and plugin admissibility contract
- `TAS-203`:
  conformance sandbox and benchmark harness
- `TAS-203A`:
  result-binding, schema-stability, and composition contract
- `TAS-204`:
  weighted plugin controller trace and refusal-aware model loop
- `TAS-205` and `TAS-206`:
  publication/trust gate and bounded platform closeout

So the repo does not yet have the full honest plugin runtime needed for real
operator runs, even though the manifest and packet contract are now real.

### 3. Psionic already has a real multi-step controller in the serving lane

`psionic-router` already owns a bounded multi-step tool loop in
`crates/psionic-router/src/tool_loop.rs`.

That lane already provides:

- step-bounded multi-turn execution
- explicit tool-call and tool-result messages
- typed tool gateway registration
- route selection per step
- prompt-history visibility controls
- tool-result visibility rules
- step receipts and final prompt-history output

The weather pilot in `docs/STRUCTURED_AGENT_WEATHER_PILOT.md` proves that the
repo can already run one real model -> tool -> continuation loop through
Psionic-owned runtime, router, and response-state code.

This is not the same as weighted plugin control.
It is, however, a real controller that can be reused.

### 4. Psionic already has a second real tool controller in the Apple FM lane

`psionic-apple-fm` already exposes:

- typed tool definitions
- session creation with tools
- session-aware tool callbacks
- one-shot completion with tools
- multi-tool bridge behavior and typed tool failure mapping

So on macOS the repo already has another real control surface that can execute
multiple tool calls across a session without flattening tools into prompt text.

Again, this is not the canonical weights-only Tassadar lane.
But it is a real orchestration substrate for bounded experimentation.

### 5. The repo already has one training-adjacent tool-call data lane

`psionic-data` already imports tool-calling Apple adapter datasets with:

- typed tool schemas
- explicit tool token accounting
- bounded conversation structure
- typed validation failures

That does not close a generic multi-plugin training stack, but it does prove
the repo already has one narrow corpus format that can carry tool-use
episodes.

## What The Current Transformer Can And Cannot Honestly Claim

The current Tassadar proof stack matters here because it limits what the repo
may say about weights-only multi-plugin control.

The current post-article lane now proves:

- canonical-route semantic preservation
- pre-plugin control-plane decision provenance for branch, retry, and stop
- plugin charter, manifest identity, and packet ABI

The current post-article lane does **not** yet prove:

- weighted plugin selection
- weighted export selection
- weighted multi-plugin sequencing
- weighted retry policy over plugin calls
- weighted result-binding over chained plugin outputs

And the canonical article route explicitly excludes planner-owned hybrid
orchestration from the article-equivalence claim path in
`docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`.

So the repo should not say:

- "the current Transformer can already plan several plugins in sequence"
- "the plugin system is effectively real because the model can already use
  tools elsewhere"
- or "router-owned or Apple FM tool loops are the same thing as the weighted
  Tassadar plugin controller"

The honest current statement is:

- the repo has one bounded plugin artifact contract
- it has two non-weights-only controller lanes that can already run multi-step
  tool workflows
- and it has not yet closed weights-only multi-plugin control on the canonical
  Tassadar route

## Controller Options

## 1. Deterministic host-wired plugin workflows

This is the simplest useful lane for starter plugins.

Shape:

- one explicit workflow definition
- one explicit allowed plugin set
- explicit branch rules over typed outputs
- no model-driven selection required

Example:

- directive classification says "web document fetch"
- call `plugin.text.url_extract`
- call `plugin.http.fetch_text`
- if content type is HTML, call `plugin.html.extract_readable`
- if content type is RSS or Atom, call `plugin.feed.rss_atom_parse`
- stop and return a typed result bundle

Why this lane matters:

- it gives the repo real multi-plugin runs before model planning is solved
- it exercises the runtime, envelope, receipts, and result-binding story
- it keeps control ownership explicit instead of pretending the model made the
  decision

What it must not claim:

- weights-only plugin planning
- open-ended agent intelligence
- general multi-plugin reasoning closure

Recommended posture:

- build this first, as the narrowest honest experimental lane above
  `TAS-200` through `TAS-203A`

## 2. Router-owned served-model plugin control

This is the best near-term flexible controller lane.

Shape:

- keep using the existing Psionic generic server and `/v1/responses`
- present plugins to the model as tools derived from plugin manifests
- let the router-owned `ToolLoopController` execute multi-step loops
- let the tool gateway invoke the plugin runtime instead of a hand-written
  native or MCP tool executor

Why this lane is strong:

- the tool-loop controller is already real
- response state and conversation continuation are already real
- route headers, route receipts, and response-state receipts already exist
- GPT-OSS / Harmony parser-backed tool use already exists in the served lane
- it can drive more flexible workflows than a deterministic chain without
  waiting for a weights-only Transformer proof

What is still missing:

- a shared plugin-to-tool projection layer
- a plugin runtime executor behind the tool gateway
- receipt binding from one served-model tool step to one plugin invocation
- controller-visible typed plugin outputs that do not degrade into loose text

Recommended posture:

- this should be the main experimentation lane for "use multiple plugins
  smartly in a row" before weights-only control is ready

## 3. Apple FM session-owned plugin control

This is the best near-term macOS local lane.

Shape:

- project the same plugin catalog into `AppleFmToolDefinition`s
- register those tools on one Apple FM session
- let the Apple FM bridge request tool calls through the existing callback
  runtime
- route each tool call into the same shared plugin runtime used elsewhere

Why this lane matters:

- it gives a real local macOS plugin-controller story without shipping a
  separate local model stack first
- Apple FM tool calling is already real, session-aware, and transcripted
- it gives the repo a second controller lane against the same plugin catalog,
  which is useful for parity and trace generation

What it must not claim:

- canonical Tassadar plugin control
- router-owned receipts unless those are added explicitly
- cross-platform serving closure

Recommended posture:

- treat this as the macOS local experimental lane, not as the canonical proof
  lane

## 4. Future weights-only plugin control

This remains the long-horizon control lane.

Shape:

- the model selects plugins, exports, arguments, retries, and stop conditions
- the plugin runtime only validates and executes
- multi-step traces and receipts bind back to the canonical machine identity

Why this should come later:

- the runtime API, receipt family, admissibility contract, conformance harness,
  and result-binding contract are not fully closed yet
- the repo does not yet have a real multi-plugin training corpus or pilot
  traces to bootstrap from
- it is cheaper to learn controller ergonomics and failure cases in the served
  and Apple FM lanes than inside the proof-bearing canonical route

Recommended posture:

- build this last, after the experimental controller lanes generate stable
  traces, schemas, refusal rows, and pilot evidence

## Recommended Architecture: One Plugin Catalog, Several Controllers

The repo should converge on one shared substrate:

- one plugin catalog
- one plugin manifest vocabulary
- one packet ABI
- one plugin runtime
- one envelope compiler
- one receipt family

And then allow several controllers above it:

1. `DeterministicPluginWorkflowController`
   - host-owned workflow graphs for known plugin chains
2. `RouterOwnedPluginToolController`
   - served-model `/v1/responses` tool loops via `psionic-router`
3. `AppleFmPluginToolController`
   - Apple FM sessions with callback-driven tools
4. `WeightedPluginController`
   - the later proof-bearing Tassadar lane

The key design rule is:

- the controller may vary
- the plugin artifact, invocation, envelope, and receipt substrate should not

That keeps the repo from inventing a separate plugin runtime for every model
lane.

## One Missing Bridge: Project Plugins Into Tool Schemas

The repo now needs one narrow adapter layer:

- plugin manifest and packet schema in
- tool schema and tool-call surface out

That layer should do all of the following:

- expose a tool name derived from canonical plugin identity
- render model-facing tool descriptions from manifest metadata
- project the input packet schema into the tool argument schema
- convert tool-call arguments back into packet bytes
- convert plugin output packets into model-visible structured results
- attach plugin invocation receipt refs to the tool-step receipt

This bridge is what lets the served router and Apple FM lanes reuse the same
plugin catalog honestly.

Without it, the repo will drift into two or three incompatible tool surfaces.

## Minimal Dependency Order

The shortest honest sequence is:

1. Finish the minimum runtime tranche needed for real operator-internal runs:
   - `TAS-200`
   - `TAS-201`
   - `TAS-202`
   - `TAS-203A`
2. Land the first starter-plugin catalog:
   - `TAS-216` through `TAS-220`
3. Add the shared plugin-to-tool projection layer
4. Add deterministic host-wired plugin workflows for the initial web-content
   chains
5. Adapt `psionic-router` tool loops to call the plugin runtime
6. Adapt Apple FM tools to call the same plugin runtime
7. Run the same starter workflows through all three experimental lanes and
   freeze the differences explicitly
8. Record multi-plugin traces and receipts into a training-ready corpus
9. Only then attempt the weights-only `TAS-204` controller closure

The important sequencing rule is:

- do not wait for weights-only control to start real runs
- and do not start broad plugin-family work before the runtime API, receipts,
  admissibility, and result-binding contract are real enough to keep the runs
  honest

## Training And Bootstrap Recommendation

The repo should assume the first weights-only controller will be trained or
distilled from earlier bounded controller traces.

The right bootstrap path is:

- deterministic workflow traces
- served-model router tool-loop traces
- Apple FM session tool traces

Those traces should then be converted into one bounded plugin-control corpus
that records:

- the user directive
- the admissible plugin set
- the controller-visible tool or plugin schema
- the selected plugin and arguments
- the plugin output packet or refusal
- the next-step decision
- the final stop condition

This is much more credible than trying to handwave a weights-only controller
into existence before the repo has even run a real multi-plugin workflow.

The current Apple adapter dataset lane is a useful narrow precedent here
because it already carries typed tool schemas and token accounting, but the
repo should not force all future plugin-control training through Apple-only
data shapes.

## Recommended Claim Language

Until the weighted controller closes, the repo should use this split:

- deterministic workflow:
  host-owned multi-plugin workflow above the bounded plugin runtime
- served-model plugin control:
  router-owned multi-step model-assisted plugin orchestration
- Apple FM plugin control:
  session-aware Apple FM plugin orchestration through the shared plugin runtime
- weighted plugin control:
  reserved for the later canonical Tassadar controller tranche only

That wording keeps experimentation real without laundering one controller lane
into another.

## Suggested Follow-On Issue Wave

After the current `TAS-216` through `TAS-220` starter-plugin set, the next
useful issue wave should be:

1. a wave umbrella issue that ties the orchestration tranche together
2. shared plugin-to-tool projection and schema bridge
3. deterministic starter-workflow controller and real-run pilot
4. router-owned plugin tool-loop integration on `/v1/responses`
5. Apple FM plugin tool integration and macOS local pilot
6. multi-plugin trace corpus, parity matrix, and training bootstrap contract
7. only then the later weighted-controller closure work already represented by
   `TAS-204`

Tentative issue slate for GitHub after review:

### `TAS-221`: Publish The First Real-Run Multi-Plugin Orchestration Wave

Draft issue description:

- Purpose:
  create one umbrella issue for the first real multi-plugin orchestration
  tranche above the plugin runtime and starter-plugin catalog
- Why:
  the repo already has starter plugins and partial plugin substrate truth, but
  it still lacks one tracked orchestration wave that ties deterministic
  workflows, served-model tool loops, Apple FM integration, and training
  bootstrap into one reviewable plan
- Scope:
  track `TAS-222` through `TAS-226` as one operator-internal experimental
  wave; keep the tranche explicitly below `TAS-204` weighted-controller claims
- Acceptance bar:
  the issue links the child issues, records the dependency order
  `TAS-200` -> `TAS-201` -> `TAS-202` -> `TAS-203A` -> orchestration wave, and
  freezes the claim language that this tranche is experimental controller work
  above the shared plugin runtime rather than proof-bearing weights-only plugin
  control

### `TAS-222`: Shared Plugin-To-Tool Projection And Receipt Bridge

Draft issue description:

- Purpose:
  add one shared adapter layer that projects plugin manifests and packet
  schemas into model-facing tool definitions and binds tool-step receipts back
  to plugin invocation receipts
- Why:
  without this bridge, the served router lane and Apple FM lane will drift into
  incompatible tool surfaces and duplicate plugin-schema logic
- Scope:
  derive stable tool names from canonical plugin identity; project packet input
  schema into tool arguments; project packet outputs into controller-visible
  structured results; attach plugin receipt references and replay class to the
  controller-visible step result; keep refusal kinds typed instead of turning
  them into free-form tool text
- Non-goals:
  this issue does not close weighted controller logic, plugin publication, or
  arbitrary third-party plugin registration
- Acceptance bar:
  one shared projection surface is reusable by deterministic workflows,
  `psionic-router` tool loops, and `psionic-apple-fm` tool sessions; the same
  plugin manifest yields stable tool schema and stable receipt binding in all
  controller lanes

### `TAS-223`: Deterministic Starter Workflow Controller And Web-Content Pilot

Draft issue description:

- Purpose:
  add the first host-owned multi-plugin workflow controller for the starter
  plugin catalog and prove one real web-content workflow end to end
- Why:
  the repo needs real multi-plugin runs before weights-only planning is ready,
  and deterministic workflows are the cleanest way to exercise runtime,
  admissibility, composition, and receipt truth without overclaiming model
  intelligence
- Scope:
  define one bounded workflow graph over the starter plugins, for example URL
  extraction followed by fetch-text and then readable HTML extraction or
  feed parsing based on typed content classification; capture final result
  bundle plus per-step plugin receipts, refusal rows, and stop condition
- Non-goals:
  this issue does not claim open-ended agent planning or generic plugin
  optimization
- Acceptance bar:
  one reproducible pilot run exists in repo-owned docs or fixtures, the
  workflow controller uses the shared plugin runtime rather than bespoke glue,
  and every branch or stop decision remains host-owned and explicit

### `TAS-224`: Router-Owned Plugin Tool-Loop Integration On `/v1/responses`

Draft issue description:

- Purpose:
  let the existing `psionic-router` tool loop execute plugins through the
  shared plugin runtime instead of only hand-wired tool executors
- Why:
  this is the strongest near-term lane for flexible "use multiple plugins
  smartly in a row" experiments because the router, response-state, and tool
  loop surfaces already exist on `main`
- Scope:
  project admissible plugins into tool definitions for one `/v1/responses`
  request, wire tool execution through the shared plugin runtime, preserve
  structured plugin outputs in tool results, and bind each tool-loop step to
  the underlying plugin invocation receipt
- Non-goals:
  this issue does not prove canonical Tassadar weighted control and does not
  widen article-equivalence claims
- Acceptance bar:
  one served-model pilot shows a real directive leading to several plugin calls
  in sequence through `ToolLoopController`, with bounded step count,
  response-state continuation, explicit route truth, and step receipts that
  name both the model tool step and the plugin invocation that satisfied it

### `TAS-225`: Apple FM Plugin Tool Integration And macOS Local Pilot

Draft issue description:

- Purpose:
  reuse the same plugin catalog and runtime in the Apple FM tool-calling lane
  and prove one local macOS multi-plugin pilot
- Why:
  Apple FM already has session-aware tool callbacks and is the most credible
  local experimental controller lane for bounded plugin workflows on macOS
- Scope:
  project admissible plugins into `AppleFmToolDefinition`s, route callbacks
  into the shared plugin runtime, preserve structured plugin results and typed
  refusals, and freeze one session-aware local pilot that exercises more than
  one plugin in sequence
- Non-goals:
  this issue does not claim cross-platform serving closure or canonical
  weighted plugin control
- Acceptance bar:
  one Apple FM session can call multiple projected plugins through the shared
  runtime, the pilot records transcript and plugin receipt truth, and the lane
  stays explicitly classified as local Apple FM plugin orchestration rather
  than Tassadar proof closure

### `TAS-226`: Multi-Plugin Trace Corpus, Parity Matrix, And Training Bootstrap Contract

Draft issue description:

- Purpose:
  define the first repo-owned corpus and evaluation contract for real
  multi-plugin orchestration traces captured from the deterministic, served,
  and Apple FM controller lanes
- Why:
  a future weighted controller should be bootstrapped from bounded real traces,
  not imagined from scratch; the repo also needs a machine-readable parity
  surface that shows where the three controller lanes agree or diverge
- Scope:
  freeze one training-ready record shape containing directive, admissible
  plugin set, projected tool schemas, controller decision, plugin arguments,
  plugin outputs or refusals, receipt references, and stop condition; freeze a
  parity matrix that compares deterministic, served-model, and Apple FM runs on
  the same starter workflows
- Non-goals:
  this issue does not itself train a weighted controller or publish one as a
  served claim
- Acceptance bar:
  committed fixture or report evidence exists for multi-plugin traces from all
  experimental controller lanes, the parity matrix keeps disagreement rows
  explicit, and the resulting corpus is positioned as the bootstrap input to
  later `TAS-204` controller work rather than proof that `TAS-204` is already
  solved

Those should stay clearly outside the article-equivalence claim route and
inside bounded operator-internal plugin experimentation until the later
platform and publication gates are green.

## Bottom Line

Psionic should not choose between "weights-only" and "hardwired or hybrid."

It should do all of the following in order:

- finish the plugin runtime tranche enough to run plugins honestly
- ship deterministic host-wired multi-plugin workflows first
- reuse the existing router-owned served tool loop as the main flexible
  controller lane
- reuse Apple FM tools as the macOS local controller lane
- collect traces and receipts from those real runs
- and only then try to close true weights-only multi-plugin control on the
  canonical Tassadar route

That is the fastest path to useful experiments and the cleanest path to later
proof-bearing weighted plugin control.

# Tassadar Post-Article Turing-Completeness Rebase Audit

Date: 2026-03-20

## Purpose

This audit reviews the older Tassadar universality and Turing-completeness
scaffold after the newer post-`TAS-156A` article-equivalence wave.

It answers two separate questions:

1. What from the older `TAS-141` and `TAS-150` through `TAS-156` tranche is
   still real and still honest?
2. What still has to land after `TAS-186` if the goal is not merely
   article-equivalent closure, but a truthful statement that the new canonical
   owned Transformer route is the route that carries the repo's
   Turing-completeness story?

Companion audit:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`

## Source Set

This audit is based on the current repo state plus the current public TAS issue
set.

Canonical repo surfaces reviewed:

- `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- `docs/audits/2026-03-19-tassadar-effective-unbounded-compute-claim-audit.md`
- `docs/audits/2026-03-19-tassadar-pre-closeout-universality-audit.md`
- `docs/audits/2026-03-19-tassadar-tcm-v1-substrate-audit.md`
- `docs/audits/2026-03-19-tassadar-turing-completeness-closeout-audit.md`
- `docs/audits/2026-03-19-tassadar-universality-verdict-split-audit.md`
- `docs/audits/2026-03-20-tassadar-article-equivalence-blocker-matrix.md`
- `docs/audits/2026-03-20-tassadar-article-equivalence-acceptance-gate.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`

Canonical machine-readable surfaces reviewed:

- `fixtures/tassadar/reports/tassadar_effective_unbounded_compute_claim_report.json`
- `fixtures/tassadar/reports/tassadar_pre_closeout_universality_claim_boundary_report.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `fixtures/tassadar/reports/tassadar_universal_machine_proof_report.json`
- `fixtures/tassadar/reports/tassadar_universality_witness_suite_report.json`
- `fixtures/tassadar/reports/tassadar_minimal_universal_substrate_acceptance_gate_report.json`
- `fixtures/tassadar/reports/tassadar_universality_verdict_split_report.json`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_audit_report.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_blocker_matrix_report.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

Public issue sets reviewed:

- `TAS-141`
- `TAS-150` through `TAS-156`
- `TAS-156A`
- `TAS-157` through `TAS-186`
- `TAS-R1`

## Executive Verdict

Current audit verdict:

- historical `TCM.v1` universality scaffold: `implemented`
- canonical post-`TAS-156A` article-equivalence route: `implemented` through
  `TAS-181`, `partial` through `TAS-186`
- Turing-completeness claim rebased onto the canonical owned Transformer route:
  `planned`
- post-`TAS-186` universality bridge written with an explicit future
  plugin-capability boundary: `planned`
- served/public universality on the new route: `partial_outside_psionic`

The important distinction is this:

The repo already has one coherent bounded Turing-completeness closeout under
declared `TCM.v1` semantics for theory and operator use. That closeout is still
internally consistent and still honest on its own terms.

What the repo does **not** yet have is a proof that the newer canonical
`psionic-transformer` article-equivalence route is itself the route that closes
that Turing-completeness story. The old closeout and the new article wave are
not the same lane, and they were not updated together.

## Computability, Programmability, And Productization

The cleanest way to keep these audits honest is to distinguish three different
things the repo may eventually prove:

- computability:
  the post-`TAS-186` Turing-completeness rebase would prove that the canonical
  owned route can carry bounded universal computation under declared
  continuation semantics
- programmability:
  a later plugin tranche would prove that the same route can sit above a
  bounded software-capability layer with stable artifacts, envelopes, and model
  visible control surfaces
- productization:
  later authority, promotion, publication, and governance gates would decide
  whether any of that is usable beyond bounded operator/internal posture

The first item is the point of the bridge tranche. The second is the point of
the plugin tranche. The third is a policy and release question above both.

## Cross-Tranche Invariants

Both this audit and the companion plugin audit should be read under the same
invariants.

- Plane Separation:
  data plane, control plane, and capability plane remain explicit with no
  hidden cross-plane leakage
- State Ownership:
  durable workflow truth must live only in explicit weights-owned, ephemeral,
  resumed, or host-backed state classes
- Control Ownership:
  host may execute declared mechanics, but host may not decide workflow
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
- Canonical Machine Identity:
  one named machine identity tuple must bind model id, weight digest, route
  digest, continuation contract, and carrier class for every proof, witness,
  benchmark, and audit
- Execution-Semantics Equivalence:
  proof-bearing route changes must preserve declared transition semantics or an
  explicit proof abstraction boundary, not merely output parity
- Downward Non-Influence:
  later capability or served layers may not silently rewrite lower-plane
  compute, continuation, or proof assumptions
- Computational Model Statement:
  every claim-bearing artifact must name the computational model,
  continuation semantics, and effect boundaries it relies on
- Served Conformance:
  served posture may deviate from operator truth only inside an explicit
  conformance envelope and otherwise must fail closed

## Three-Plane Contract

The bridge should preserve three separate planes:

- data plane:
  the canonical owned route and `TCM.v1` compute substrate carry pure compute
  evolution
- control plane:
  the weighted route and continuation truth carry branching and claim-bearing
  control semantics
- capability plane:
  any later plugin system sits above the bridge as a bounded execution layer

No plane may silently absorb another plane's responsibilities.

In particular:

- the data plane may not be widened into arbitrary capability execution by
  implication
- the control plane may not be hidden inside host continuation or admissibility
  logic
- the future capability plane may not silently rewrite compute or control
  claims
- higher planes may not push downward requirements that silently redefine the
  lower-plane machine model, continuation semantics, or proof assumptions

## Adversarial Host Model

The bridge should assume a host that may try to cheat unless a rule forbids
it.

The host may attempt to:

- reorder execution
- cache and substitute results
- inject heuristics
- hide candidates or refusal modes
- exploit latency, cost, quota, or pool asymmetries
- adapt scheduling or concurrency

The bridge must therefore either:

- surface these behaviors in route-visible truth and receipts
- freeze them as non-adaptive runtime contract
- or refuse them outright

## Stability Risks That Need Explicit Artifacts

Reading the Turing-completeness and plugin audits together exposes a second
class of risk:

- Dual Truth Carrier Drift:
  once `TCM.v1` and the canonical Transformer route are both historically real,
  proofs, witnesses, and benchmarks can drift apart again unless one canonical
  machine identity lock binds them
- Proof Versus Execution Semantics Split:
  output parity can stay green while caching, batching, continuation, or fast
  routes silently change the semantics the proof relied on
- Continuation Inflation:
  resumable execution can quietly become a second machine unless continuation
  is constrained to extend execution without adding expressivity
- Fast-Route Legitimacy Drift:
  production can migrate to a fast route that is benchmark-green but no longer
  the same proof-bearing machine
- Downward Plugin Backpressure:
  later plugin ergonomics can pressure the bridge into changing continuation,
  carrier, or proof assumptions from above
- Choice-Set Neutrality Drift:
  surfacing "all choices" is not enough if ordering, latency, or soft-failure
  effects still bias effectively equivalent options
- Operator Versus Served Drift:
  rebasing can reopen the earlier operator/served divergence if served routes
  are not held inside an explicit conformance envelope
- Computational Model Ambiguity:
  proofs, routes, continuation artifacts, and plugin layers can otherwise end
  up speaking about different machines without naming the difference

Those risks are why the bridge needs explicit anti-drift artifacts, not only
green witness rows.

## What Is Already Real And Still Valid

The older universality scaffold was not invalidated by the later article work.
The following facts remain `implemented`:

- `TAS-141`:
  one disclosure-safe effective-unbounded claim boundary exists, and it stays
  suppressed until broad portability, publication, and specialization-safety
  rows are green.
- `TAS-150`:
  one explicit pre-closeout universality boundary exists, and it correctly
  records that broadness-adjacent work did not itself equal a terminal
  Turing-completeness claim.
- `TAS-151`:
  `TCM.v1` exists as one declared universal substrate model with explicit
  control, memory, continuation, and effect-boundary rows.
- `TAS-152`:
  one explicit universal-machine witness construction exists over `TCM.v1`.
- `TAS-153`:
  one dedicated universality witness suite exists.
- `TAS-154`:
  one minimal universal-substrate acceptance gate exists.
- `TAS-155`:
  one explicit theory/operator/served verdict split exists.
- `TAS-156`:
  one bounded Turing-completeness closeout audit exists with
  `theory=green`, `operator=green`, and `served=suppressed`.

That older closeout is also disciplined in the right way:

- it does not imply arbitrary Wasm execution
- it does not imply broad served internal compute
- it does not imply public universality publication
- it does not imply settlement-qualified universality closure

Those refusal boundaries are still the right ones.

## What Changed After `TAS-156A`

The later article-equivalence wave changed the canonical truth carrier for the
article story.

The repo now has a separate post-`TAS-156A` canonical path:

- `TAS-160` freezes the owned multi-crate Transformer boundary with
  `psionic-transformer` as the architecture anchor.
- `TAS-163` through `TAS-175` build the real owned article model, trained
  artifact, reference-linear proof lane, fast-route selection, fast-route
  integration, exactness closure, and throughput-floor closure.
- `TAS-176` through `TAS-181` freeze the declared frontend/compiler envelope,
  broader frontend corpus, demo-source parity, declared interpreter breadth,
  breadth suite, Hungarian parity, and hard-Sudoku benchmark closure.
- `TAS-182` through `TAS-186` remain open and are aimed at final article
  closure, not at universality rebasing:
  demo/benchmark unification, no-spill single-run closure, clean-room weight
  ownership, cache and activation-state discipline, reproducibility,
  route-minimality, and the final article-equivalence audit.

That means the repo now has two different high-value truths:

- an older resumable operator universality contract over `TCM.v1`
- a newer canonical owned Transformer route for the article-equivalence story

Both are real. They are not yet the same route.

## Where The Old Turing-Completeness Scaffold Is Now Stale

The older closeout is still honest, but it is stale relative to the newer
canonical route in six concrete ways.

### 1. Substrate Anchoring Gap

`TCM.v1` still cites older named internal-compute profile anchors such as
`tassadar.internal_compute.article_closeout.v1`,
`tassadar.internal_compute.spill_tape_store.v1`, and
`tassadar.internal_compute.process_objects.v1`.

It does **not** yet cite:

- the canonical `psionic-transformer` route boundary from `TAS-160`
- the canonical article model artifact family from `TAS-168` through `TAS-169A`
- the reference-linear and fast-route proof surfaces from `TAS-171` through
  `TAS-175`
- the article-equivalence gate and blocker surfaces from `TAS-157` through
  `TAS-186`

So the terminal universality contract is still anchored to the pre-article-gap
operator lane, not to the new owned-route closeout program.

### 2. Ownership And Mechanism Gap

The older universality tranche proves operator-universality under resumable
envelopes. It does **not** prove:

- that the decisive computation lives in the owned Transformer weights
- that hidden runtime-controlled control flow is absent
- that helper modules or retrieval layers are not secretly carrying the work
- that KV cache or activation state are not the real interpreter substrate

Those are exactly the open concerns in `TAS-184` and `TAS-184A`.

### 3. Execution-Posture Gap

The older `TCM.v1` story is explicitly resumable and continuation-backed:

- bounded slice resume
- spill/tape extension
- process objects
- persistent process identity
- session-process profiles

The new article-equivalence wave is explicitly trying to close a different bar:

- no checkpoint restore
- no spill/tape extension
- one uninterrupted owned Transformer execution path
- no hidden re-entry
- no runtime-controlled loop unrolling outside the model

That is the point of `TAS-183`.

These are not contradictory, but they are different contracts. The repo has not
yet written down how they compose.

### 4. Frontend And Breadth Gap

The old Turing-completeness closeout leaned on:

- the broader internal-compute ladder
- VM-style module witnesses
- session-process kernels
- spill/tape continuation kernels

The new article-equivalence wave now has a much sharper declared public
envelope:

- one Rust-only frontend/compiler envelope
- one declared article interpreter breadth envelope
- one fixed generic article-program family suite

The older closeout was never updated to say whether the new canonical owned
route closes universality:

- inside that declared article envelope only
- across a wider operator-only envelope
- or across both, with an explicit split

### 5. Portability And Minimality Gap

The older closeout has bounded portability language, but it does not cite the
new canonical-route reproducibility and minimality surfaces because those do
not exist yet.

That is now the job of:

- `TAS-185`: cross-machine reproducibility matrix
- `TAS-185A`: route minimality audit and publication verdict

Until those land, any attempt to rebase Turing completeness onto the new
canonical route would be premature.

### 6. Final Claim-Surface Gap

The older closeout was allowed to be operator-green specifically because it
stayed separate from the article-equivalence program.

The new final article-equivalence gate at `TAS-186` will decide whether the
canonical owned Transformer route is article-equivalent.

It will **not** by itself decide:

- whether the canonical route is the universal substrate carrier
- whether persisted continuation on that route stays ownership-clean
- whether the old `TCM.v1` closeout should be reinterpreted in light of the
  newer canonical model and route identities

That still needs a second wave.

## What `TAS-186` Will And Will Not Mean

If `TAS-182` through `TAS-186` all turn green, the repo will have one strong
new fact:

The canonical owned `psionic-transformer` route will be article-equivalent
under the declared frontend, interpreter-breadth, benchmark, ownership, cache,
reproducibility, and minimality bars.

That still will **not** automatically mean:

- the canonical owned route is the route behind `TCM.v1`
- the canonical owned route is itself the final Turing-complete operator lane
- the old resumable universality contract has been updated to reference the new
  canonical route
- the canonical owned route is universally claimable without the old
  checkpoint, spill/tape, and process-object semantics

After `TAS-186`, the right statement would be:

Psionic has one canonical owned Transformer route that is article-equivalent
inside the declared article envelope, and Psionic separately still has one
older bounded `TCM.v1` operator-universality closeout under resumable
envelopes.

That is a stronger repo than either lane alone, but it is still not yet one
rebased unified story.

## Recommended Post-`TAS-186` Strategy

The recommended strategy is:

- keep the old `TCM.v1` closeout intact as historical machine-readable truth
- do **not** mutate the meaning of `TAS-156`
- do **not** pretend that article-equivalence alone already rebases the
  universality contract
- add one new post-`TAS-186` bridge tranche that explicitly ties the canonical
  owned Transformer route back into the existing `TCM.v1` theory/operator
  story
- write that bridge so it does **not** hard-code a pure-compute-only terminal
  shape that would have to be reopened for the later weighted plugin-system
  tranche

This is the least error-prone option because it preserves the old claim
discipline while letting the new route become the truth carrier for that older
claim.

## Why A Bridge Tranche Is Better Than Rewriting `TCM.v1`

The bridge approach is recommended for five reasons.

### 1. It preserves existing truthful artifacts

The old `TAS-151` through `TAS-156` artifacts remain correct on their own
terms. Rewriting them would create needless provenance churn.

### 2. It keeps the no-spill and resumable stories separate

Article-equivalence needs the stronger no-spill and no-helper audit.
Turing-completeness for real operator use still naturally lives inside bounded
slices with persisted continuation. Those are different claims and should stay
different.

### 3. It keeps claim scope honest

The bridge can say exactly which part of the new route is:

- direct single-run article-equivalent
- resumable operator-universality-capable
- still suppressed for served/public posture

That is much safer than flattening them into one sentence.

### 4. It avoids forcing the wrong terminal bar

The repo does not need to prove "one infinite uninterrupted no-spill run"
to make a truthful Turing-completeness statement. The older `TCM.v1` closeout
was already honest that universality lived under bounded slices and persisted
continuation.

### 5. It keeps the later plugin-system lane composable

The companion plugin audit shows that the repo already has much of the module,
mount, replay, and policy substrate needed for a weighted Wasm plugin system,
but not yet the plugin-specific manifest, packet ABI, receipt family, or
weighted controller.

That means the universality bridge should leave room for a later bounded
software-capability layer above `TCM.v1`, without implying that plugin
capability is already part of the Turing-completeness closeout.

## Non-Negotiable Bridge Laws

The bridge tranche should freeze these blunt laws early, because both the
rebased Turing-completeness story and the later plugin-system story can drift
if these are left implicit.

### 1. State Ownership Law

The bridge should classify state into explicit buckets:

- weights-owned compute state:
  the canonical owned route remains the truth carrier for the machine step and
  direct compute evolution
- cache and activation state:
  in-run ephemeral acceleration state only, never durable workflow truth
- resumed process-object and spill/checkpoint state:
  declared continuation state with preserved identity and explicit receipts
- host-backed durable state:
  only explicit checkpoint, artifact, queue, or worklist state under declared
  stores and receipts

The important refusal is:

no durable workflow truth may be smuggled into undeclared host helpers, hidden
runtime metadata, or cache-like state and still be described as belonging to
the canonical owned route.

### 2. Control Ownership Law

The bridge should also freeze one blunt rule:

host may execute continuation and capability mechanics, but host may not decide
workflow.

That means the host may:

- reload checkpoints
- advance declared spill/tape or process-object mechanics
- enforce bounds and policy
- execute later plugin capability calls once such a layer exists

But the host may not:

- invent branch choices
- choose retries
- choose stop conditions
- select hidden helper programs
- or quietly become the planner under the cover of resume logic

This is how the repo keeps operator universality under bounded continuation
from decaying into host-orchestrated behavior that only sounds weighted.

### 3. Semantic Preservation Law

Resume, spill, and process-object mechanics must preserve the semantics of the
canonical owned route, not merely preserve output parity on a few happy-path
cases.

That means:

- continuation mechanics may not silently rewrite the machine model that the
  route implements
- adapters may not change control meaning while still looking route-compatible
- process-object restore, checkpoint reload, and spill/tape mechanics must fail
  closed when semantic preservation cannot be shown

The important distinction is:

output equivalence alone is not enough if the proof-bearing structure or the
route's mechanistic assumptions have changed underneath it.

### 4. Carrier Split Law

The bridge must publish one explicit split between:

- direct article-equivalent carrier truths
- bounded resumable universality carrier truths

No later audit may collapse those carriers into one undifferentiated route
claim.

That split is the cleanest way to stop people from mentally merging:

- single-run article closure
- resumable operator universality
- later plugin-aware capability execution

### 5. Choice-Set Integrity Law

The bridge and any later capability layer above it must not restrict, rank,
filter, or rewrite the effective choice set in a way that is invisible to the
declared route truth.

That means:

- if later capability surfaces exist, admissible choices must be fully
  enumerated or explicitly bounded
- any filtering or transformation must be receipt-visible
- refusal and failure classes that affect branching may not be hidden
- hidden heuristics may not narrow the route's effective choice set

Otherwise the host can still decide workflow by constraining what the route is
allowed to see.

### 6. Resource Transparency Law

Any resource constraint that can affect branching, retries, continuation, or
later capability selection must be surfaced as part of the declared route truth
or fixed by contract.

That includes:

- latency
- cost
- quota
- availability
- pool pressure

Otherwise the host can steer workflow economically while still sounding
route-faithful.

### 7. Scheduling Ownership Law

Ordering, concurrency, and result-visibility timing must be either:

- explicitly decided by the route
- or explicitly frozen as a non-adaptive runtime contract

Otherwise scheduling becomes hidden planning.

### 8. Canonical Machine Identity Lock

The rebased universality lane needs one globally named machine identity tuple:

- canonical model id
- canonical weight digest
- canonical route digest
- canonical continuation contract
- carrier class

Every proof, witness, benchmark, and closeout must bind to that tuple or fail
closed.

Without this lock, the repo can re-open dual truth-carrier drift while still
sounding canonical.

### 9. Execution-Semantics Equivalence Law

Proof transport requires more than output equivalence.

The bridge must freeze either:

- transition-level equivalence for the rebased route family
- or one explicit abstraction boundary that the proof relies on

Caching, batching, continuation restore, fast routes, and helper mechanics may
not change the proof-bearing execution semantics off-trace.

### 10. Continuation Non-Computationality Law

Continuation may extend execution time, persistence, or resume posture, but it
may not add computational expressivity beyond the base route.

That means:

- checkpoint structure may not encode hidden workflow logic
- process-object restore may not introduce new control semantics
- spill or tape mechanics may not become an undeclared second machine

Without this law, the canonical route stops being the real compute carrier.

### 11. Fast-Route Legitimacy Law

Any fast route must be classified explicitly as either:

- inside the universality carrier because semantics equivalence is proven
- or outside the universality carrier as an operator or served acceleration

Benchmark parity alone is not enough.

Without this law, the repo can split the proof-bearing machine from the
production machine again.

### 12. Downward Non-Influence Law

Later capability or served layers may not redefine:

- continuation semantics
- core compute-substrate rules
- proof assumptions
- or carrier identity

The bridge may reserve hooks for later layers, but those hooks may not let a
higher plane rewrite lower-plane truth.

### 13. Served Conformance Envelope Law

Served posture must publish one explicit conformance envelope that says:

- which deviations from operator truth are allowed
- which properties must remain identical
- and which cases must fail closed

Without this law, served optimizations can drift away from operator truth while
still sounding aligned.

### 14. Computational Model Statement Law

The rebased lane must publish one explicit computational model statement that
answers:

- what machine the repo is claiming equivalence to
- under what continuation semantics
- with which declared effect boundaries
- and without what plugin or public-serving implications

Without this law, different artifacts can all sound correct while referring to
different machines.

## Necessary Work After `TAS-186`

The following follow-on work is `planned` if the goal is to make the new
canonical route fully Turing-complete in an honest and machine-readable way.

The bar here is not:

- one infinite uninterrupted no-spill run

The bar is:

- whether the canonical owned route remains the truth carrier under bounded
  resume, continuation, spill/tape, and process-identity semantics without
  losing state-ownership or control-ownership cleanliness

### A. Freeze The Bridge Contract

Land one dedicated bridge artifact that answers:

- which canonical model id from the article-equivalence wave is now the
  universality carrier?
- which canonical weight artifact digest is in scope?
- which route digest is in scope?
- which globally named machine identity tuple is in scope?
- which truths are carried by direct article-equivalent execution, and which
  are carried only by the bounded resumable universality carrier?
- what the canonical computational model statement is for that carrier
- is universality carried by `ReferenceLinear`, `HullCache`, a resumable route
  family above them, or an explicit split across those lanes?
- which old `TCM.v1` rows now bind to new owned-route evidence, and which still
  bind only to older operator surfaces?
- what execution-semantics equivalence class or proof abstraction boundary the
  bridge relies on
- where future packet-mediated software capability calls sit relative to the
  pure compute substrate, so the bridge does not later have to be redefined to
  accommodate plugins
- how choice-set integrity, resource transparency, and scheduling ownership are
  reserved for later capability layers above the bridge
- how downward non-influence from later capability layers is reserved
- which forward-compatible hooks remain reserved for packet boundaries,
  capability invocation slots, receipt extensibility, and schema negotiation

Without this contract, later universality rebasing will drift.

### B. Prove Continuation, Control, And State Ownership On The New Route

The canonical route needs one explicit audit for resumed execution ownership.

It must prove that checkpoint, spill/tape, and process-object semantics:

- wrap the canonical owned route instead of substituting for it
- preserve canonical model identity and route identity across resumes
- do not add helper-side control flow that would defeat `TAS-184`
- fail closed when resumed execution would leave the declared route or profile
- keep packet-local, instance-local ephemeral, and host-backed durable state
  classes explicit so later plugin execution does not blur continuation truth
- preserve route semantics rather than merely preserving surface outputs
- freeze the rule that host may execute continuation mechanics but may not
  decide workflow

This is the most important missing bridge between `TAS-183`/`TAS-184` and the
older operator-universality lane.

### C. Publish The Carrier Split Contract

The rebase should publish one dedicated carrier split statement that freezes:

- direct article-equivalent compute as one carrier class
- bounded resumable universality as a second carrier class
- the exact truths each carrier class is allowed to support
- the explicit refusal to merge them into one vague route claim

Without that split, later docs will keep over-reading article closure as if it
already closed operator universality.

### D. Rebind Universal-Machine Proofs To The Canonical Route

The older witness construction must be replayed against the new canonical
owned-route identities.

That means:

- exact proof receipts must cite the canonical post-`TAS-186` model id
- exact proof receipts must cite the canonical post-`TAS-186` weight artifact
- exact proof receipts must cite the canonical post-`TAS-186` route digest
- checkpoint/resume equivalence must remain explicit where resumable execution
  is still part of the operator claim
- rebinding must not be treated as a metadata swap
- one explicit proof-transport check must show that the new route preserves the
  mechanistic assumptions the proof actually relies on
- rebinding must fail closed if route changes alter the proof-bearing structure

Until this happens, the old universal-machine proof remains real but not
canonically attached to the new route.

### E. Rebuild The Witness Suite Around The Canonical Route

The universality witness suite should be reissued so the canonical owned route
is the route being exercised, not only the older operator lane.

At minimum that suite should still include:

- register-machine witnesses
- tape-machine witnesses
- VM-style interpreter kernels
- continuation-stress kernels
- explicit refusal-boundary rows

But it now also needs:

- route-identity binding to the post-`TAS-186` canonical model
- ownership-clean resumed execution rows
- negative rows for helper substitution, hidden cache-owned control flow, and
  resume-only cheating

### F. Add A Canonical-Route Universal-Substrate Gate

The old minimal gate should not be silently reused.

One new gate should require:

- `TAS-186` green
- canonical-route bridge contract green
- continuation/control/state ownership audit green
- carrier split contract green
- canonical-route universal-machine proof green
- canonical-route witness suite green
- portability and refusal rows green on the declared machine matrix

That gate is the first place where the repo can honestly ask whether the new
canonical route, not merely the older operator lane, is Turing-complete.

### G. Extend Portability And Minimality To The Universality Lane

`TAS-185` and `TAS-185A` are necessary, but not sufficient.

They need one follow-on extension over the universality carrier itself:

- cross-machine witness-suite replay
- cross-machine resumed-execution parity
- route-drift suppression across resume boundaries
- explicit refusal when portability or minimality fails

The older `TAS-156` closeout already carried portability and refusal language,
but it was not bound to the newer canonical route.

### H. Reissue The Final Verdict Split

Once the rebased gate is green, the repo should publish one new verdict split
for the canonical route:

- theory: whether the canonical route now carries the witness-backed substrate
  story
- operator: whether the canonical route now supports bounded-slice universal
  execution under explicit continuation envelopes
- served: whether any served/public universality widening is allowed

The expected served verdict remains `partial_outside_psionic` unless
`kernel-policy` and `nexus` ownership moves or the user explicitly asks for
that cross-repo work.

### I. Publish A New Final Closeout Audit

Only after the new bridge, proof, suite, gate, portability, and verdict-split
artifacts exist should the repo publish a second final closeout.

That new closeout should say:

- the older `TAS-156` closeout still stands historically
- the post-`TAS-186` canonical owned Transformer route is now the truth carrier
  for the bounded Turing-completeness statement
- served/public universality remains either suppressed or explicitly bounded

### J. Reserve A Separate Plugin-Capability Boundary

The rebased universality tranche should also publish one explicit boundary
statement that says:

- `TCM.v1` remains the bounded compute substrate
- later plugin execution is a separate software-capability layer above that
  substrate, not a silent expansion of it
- weighted plugin control is not implied by theory/operator universality
- plugin capability and served plugin publication will require their own gates
  and verdicts

This is not a request to build the plugin system inside the rebased
Turing-completeness tranche. It is a request to make sure that tranche does not
accidentally preclude the cleaner plugin architecture.

## Proposed GitHub Issue Roadmap

Suggested numbering below assumes the next dedicated post-`TAS-186` tranche
continues the current TAS sequence. If the tracker advances first, preserve the
dependency order and titles, not the exact numerals.

### Suggested `TAS-187`: Freeze Post-Article Universality Bridge Contract

Suggested GitHub title:

`Tassadar: freeze post-article universality bridge contract`

Summary:

Bind the old `TCM.v1` closeout to the canonical post-`TAS-186` owned-route
identities without rewriting the historical `TAS-151` through `TAS-156`
artifacts.

Description:

- declare the canonical model id, weight artifact digest, and route digest that
  now carry the rebased universality story
- freeze one globally named machine identity tuple for the rebased carrier
- publish the explicit split between direct article-equivalent carrier truths
  and bounded resumable universality carrier truths
- publish one canonical computational model statement for the rebased carrier
- freeze the three-plane contract across data plane, control plane, and later
  capability plane
- state whether the carrier is `ReferenceLinear`, `HullCache`, a resumable
  route family above them, or an explicit split across those lanes
- bind old `TCM.v1` rows to new owned-route evidence where appropriate
- keep the relation to later plugin capability calls explicit instead of
  leaving it implicit
- reserve choice-set integrity, resource transparency, and scheduling ownership
  invariants for later capability layers
- reserve forward-compatible packet boundary hooks, capability invocation
  slots, receipt extensibility fields, and schema-version negotiation hooks

Supporting material:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

### Suggested `TAS-188`: Audit Canonical-Route Ownership And Semantic Preservation

Suggested GitHub title:

`Tassadar: audit canonical-route ownership and semantic preservation`

Summary:

Prove that continuation mechanics preserve canonical-route control, state
ownership, and semantics rather than merely preserving a subset of outputs.

Description:

- preserve canonical model identity and route identity across resumes
- inspect actual continuation behavior before finalizing state classes
- freeze the rule that host may execute continuation mechanics but may not
  decide workflow
- prove continuation extends execution without adding expressivity beyond the
  base route
- prove resume, spill, and process-object mechanics preserve route semantics
  rather than merely surface outputs
- classify weights-owned, ephemeral, resumed, and durable state from that
  evidence and refuse undeclared workflow state outside those classes

Supporting material:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `fixtures/tassadar/reports/tassadar_spill_tape_store_report.json`
- `fixtures/tassadar/reports/tassadar_session_process_profile_report.json`
- `fixtures/tassadar/reports/tassadar_installed_process_lifecycle_report.json`

### Suggested `TAS-189`: Publish Carrier Split Between Direct And Resumable Carriers

Suggested GitHub title:

`Tassadar: publish carrier split between direct and resumable route truths`

Summary:

Freeze one machine-readable split between direct article-equivalent carrier
truths and bounded resumable universality carrier truths.

Description:

- define which truths are carried only by direct single-run execution
- define which truths are carried only by resumable continuation semantics
- bind each claim class to its carrier explicitly
- refuse any later audit or checker that collapses those carriers into one
  undifferentiated route claim

Supporting material:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_audit_report.json`

### Suggested `TAS-190`: Rebind The Universal-Machine Proof To The Canonical Route

Suggested GitHub title:

`Tassadar: rebind universal-machine proof to post-article canonical route`

Summary:

Replay the existing universal-machine proof against the canonical post-`TAS-186`
model, weight, and route identities.

Description:

- reissue proof receipts against the canonical owned route
- make resumed-execution equivalence explicit where the operator claim still
  depends on continuation
- require one proof-transport audit instead of treating rebinding as metadata
  relabeling
- define the transition-level equivalence class or explicit proof abstraction
  boundary the rebinding relies on
- prove the new route preserves the mechanistic assumptions the proof relies on
- keep helper substitution and route drift as explicit failure rows

Supporting material:

- `fixtures/tassadar/reports/tassadar_universal_machine_proof_report.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`
- `docs/audits/2026-03-19-tassadar-turing-completeness-closeout-audit.md`

### Suggested `TAS-191`: Reissue The Canonical-Route Universality Witness Suite

Suggested GitHub title:

`Tassadar: reissue universality witness suite on canonical route`

Summary:

Move the witness suite from the older operator lane onto the canonical owned
route while keeping refusal and cheating-detection rows explicit.

Description:

- replay register-machine, tape-machine, VM-style, and continuation-stress
  witnesses on the canonical route
- bind every witness receipt to the new model and route identities
- add negative rows for helper substitution, hidden cache-owned control flow,
  and resume-only cheating

Supporting material:

- `fixtures/tassadar/reports/tassadar_universality_witness_suite_report.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `docs/audits/2026-03-19-tassadar-universality-verdict-split-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`

### Suggested `TAS-192`: Add The Canonical-Route Universal-Substrate Gate

Suggested GitHub title:

`Tassadar: add canonical-route universal-substrate gate`

Summary:

Create one machine-readable gate that decides whether the canonical owned route
now carries the bounded universality story.

Description:

- require `TAS-186` closure
- require the bridge contract, ownership and semantic-preservation audit,
  carrier split contract, proof rebinding, and witness suite
- keep portability and refusal rows explicit
- refuse over-reading article equivalence as universality

Supporting material:

- `fixtures/tassadar/reports/tassadar_minimal_universal_substrate_acceptance_gate_report.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`

### Suggested `TAS-193`: Extend Universality Portability And Minimality Matrix

Suggested GitHub title:

`Tassadar: extend universality portability and minimality matrix to canonical route`

Summary:

Prove the rebased universality lane survives the declared machine matrix and
refuses route drift or minimality failure.

Description:

- replay witness workloads across the declared machine matrix
- require resumed-execution parity across the same matrix
- classify whether each fast route is inside the universality carrier or
  outside it as an acceleration-only surface
- make route-drift suppression and minimality failures explicit
- preserve the older served/public suppression boundary and define the served
  conformance envelope for any later widening

Supporting material:

- `fixtures/tassadar/reports/tassadar_broad_internal_compute_portability_report.json`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_audit_report.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`
- `docs/ROADMAP_TASSADAR_TAS_SYNC.md`

### Suggested `TAS-194`: Publish The Rebased Theory/Operator/Served Verdict Split

Suggested GitHub title:

`Tassadar: publish rebased theory/operator/served universality verdict split`

Summary:

Reissue the verdict split for the canonical route so theory, operator, and
served/public posture stay separate after the rebase.

Description:

- publish the theory verdict for the rebased witness-backed substrate story
- publish the operator verdict for bounded-slice universality under explicit
  continuation semantics
- keep served/public universality suppressed unless external dependencies move
  and freeze the served conformance envelope for any narrower served posture
- refuse any implication that plugin capability or publication has become green

Supporting material:

- `fixtures/tassadar/reports/tassadar_universality_verdict_split_report.json`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_summary.json`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`

### Suggested `TAS-195`: Reserve The Plugin-Capability Boundary Above `TCM.v1`

Suggested GitHub title:

`Tassadar: reserve plugin-capability boundary above TCM.v1`

Summary:

Make the rebase explicitly plugin-aware while keeping plugin capability outside
the bounded compute substrate and outside the universality verdict.

Description:

- declare that `TCM.v1` remains the bounded compute substrate
- declare the later plugin system as a distinct capability plane above the
  bridge rather than an implicit extension of compute truth
- declare that plugin execution is a separate software-capability layer above
  that substrate
- refuse downward pressure from plugin ergonomics into continuation semantics,
  carrier identity, or proof assumptions
- keep plugin state classes and receipt identity separate from the core compute
  substrate
- reserve choice-set integrity, resource transparency, and scheduling ownership
  as non-negotiable invariants for any later capability layer
- reserve the closed-world/operator-curated assumption for the first audited
  plugin tranche unless later discovery work is separately audited
- refuse any implication that theory/operator universality already grants
  weighted plugin control or plugin publication

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`

### Suggested `TAS-196`: Publish The Post-Article Turing-Completeness Closeout Audit

Suggested GitHub title:

`Tassadar: publish post-article turing-completeness closeout audit`

Summary:

Publish the final rebased closeout once the bridge, ownership, witness, gate,
portability, and verdict artifacts are all green.

Description:

- state that the historical `TAS-156` closeout still stands
- state that the canonical post-`TAS-186` route is now the truth carrier for
  the bounded Turing-completeness claim
- keep plugin capability and served/public plugin posture out of scope
- preserve explicit refusal and publication boundaries

Supporting material:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_audit_report.json`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_summary.json`

## Suggested `TAS-207` Through `TAS-214`: Anti-Drift Stability Tranche

If the tracker wants the second-order stability constraints called out as
first-class artifacts rather than only absorbed into `TAS-187` through
`TAS-206`, the cleanest follow-on tranche is:

### Suggested `TAS-207`: Freeze Canonical Machine Identity Lock

Suggested GitHub title:

`Tassadar: freeze canonical machine identity lock`

Summary:

Bind every proof, witness, benchmark, route, and closeout to one globally
named machine identity tuple so the bridge cannot drift back into dual truth
carriers.

Description:

- define the canonical tuple over model id, weight digest, route digest,
  continuation contract, and carrier class
- require all proof-bearing and benchmark-bearing artifacts to reference that
  tuple explicitly
- refuse mixed-carrier evidence bundles that silently bind different tuples

Supporting material:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

### Suggested `TAS-208`: Publish Canonical Computational Model Statement

Suggested GitHub title:

`Tassadar: publish canonical computational model statement`

Summary:

State exactly what machine the rebased carrier claims equivalence to, under
which continuation semantics and effect boundaries.

Description:

- name the computational model the repo is claiming
- state the continuation, effect, and refusal boundaries that belong to that
  model
- keep plugin capability and public-serving implications explicitly out of
  scope unless separately green

Supporting material:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`

### Suggested `TAS-209`: Audit Execution-Semantics Equivalence And Proof Transport

Suggested GitHub title:

`Tassadar: audit execution-semantics equivalence and proof transport`

Summary:

Prove that proof rebinding preserves execution semantics rather than merely
preserving outputs.

Description:

- define the transition-level equivalence class or explicit proof abstraction
  boundary the rebinding relies on
- audit cache, batching, helper, continuation, and route-family effects
- refuse proof transport when execution semantics drift outside the declared
  proof boundary

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
- refuse resume semantics that relocate real computation into host-managed
  continuation

Supporting material:

- `fixtures/tassadar/reports/tassadar_spill_tape_store_report.json`
- `fixtures/tassadar/reports/tassadar_session_process_profile_report.json`
- `fixtures/tassadar/reports/tassadar_installed_process_lifecycle_report.json`

### Suggested `TAS-211`: Freeze Fast-Route Legitimacy And Carrier Binding

Suggested GitHub title:

`Tassadar: freeze fast-route legitimacy and carrier binding`

Summary:

Make every fast route either provably part of the universality carrier or
explicitly outside it.

Description:

- classify `ReferenceLinear`, `HullCache`, and any resumable family relative to
  the universality carrier
- require semantics-equivalence evidence before a fast route may inherit proof
  or universality status
- refuse production or served wording that treats an unproven fast route as the
  proof-bearing machine

Supporting material:

- `fixtures/tassadar/reports/tassadar_article_fast_route_architecture_selection_report.json`
- `fixtures/tassadar/reports/tassadar_article_fast_route_implementation_report.json`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

### Suggested `TAS-212`: Freeze Equivalent-Choice Neutrality And Admissibility Contract

Suggested GitHub title:

`Tassadar: freeze equivalent-choice neutrality and admissibility contract`

Summary:

Define when two admissible choices are equivalent and when ordering, cost, or
soft-failure effects are allowed to distinguish them.

Description:

- define an auditable equivalent-choice class or constrained admissibility
  model
- keep ordering, latency, cost, and soft-failure effects from becoming hidden
  control channels inside that class
- require receipt-visible justification when equivalent choices are narrowed or
  ordered differently

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`

### Suggested `TAS-213`: Freeze Downward Non-Influence And Served Conformance Envelope

Suggested GitHub title:

`Tassadar: freeze downward non-influence and served conformance envelope`

Summary:

Prove that later capability and served layers cannot rewrite lower-plane truth
and that served posture stays inside a declared conformance envelope.

Description:

- forbid plugin or served ergonomics from redefining continuation semantics,
  compute-substrate rules, proof assumptions, or carrier identity
- define which served deviations from operator truth are allowed
- require fail-closed behavior when served posture would escape the declared
  conformance envelope

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_universality_verdict_split_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`

### Suggested `TAS-214`: Publish Anti-Drift Stability Closeout

Suggested GitHub title:

`Tassadar: publish anti-drift stability closeout`

Summary:

Publish one explicit closeout that says the rebased machine identity,
semantics, continuation, fast-route, capability-boundary, and served-envelope
constraints are actually locked.

Description:

- summarize the final machine identity lock and computational model statement
- summarize execution-semantics, continuation, and fast-route legitimacy
  verdicts
- summarize equivalent-choice, downward non-influence, and served-envelope
  verdicts
- refuse any stronger claim when one of those locks is still open

Supporting material:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_audit_report.json`

## Current Honest Statement

The strongest current statement this repo can make is:

Psionic/Tassadar has two separate truthful closure surfaces:

- one older bounded `TCM.v1` Turing-completeness closeout for theory and
  operator use under explicit resumable envelopes
- one newer canonical owned Transformer article-equivalence program that is
  implemented through `TAS-181` and remains `partial` on `TAS-182` through
  `TAS-186`

What the repo still cannot say is:

The new post-`TAS-156A` canonical owned Transformer route is already the route
that carries the repo's bounded Turing-completeness closeout.

It also still cannot say:

- that the rebased universality bridge is already written to host a weighted
  plugin-capability layer cleanly
- that one canonical machine identity lock already binds proofs, witnesses,
  benchmarks, and continuation semantics to the same route family
- that proof rebinding already carries an execution-semantics equivalence
  contract rather than output parity alone
- that continuation has already been proven non-computational beyond the base
  route
- that fast routes have already been classified as legitimate universality
  carriers or as acceleration-only surfaces
- that served posture already sits inside a frozen conformance envelope
- that theory/operator universality already implies weighted plugin control
- or that a future plugin platform would inherit publication rights from the
  Turing-completeness closeout

Those remain `planned`.

## Final Judgment

The older universality tranche should be treated as `implemented`, not as
obsolete.

The newer article-equivalence tranche should be treated as the new canonical
owned-route truth carrier for article claims, but not yet as the rebased
carrier for the repo's Turing-completeness claim.

The right next move after `TAS-186` is therefore not to rewrite history and not
to over-read article equivalence as universality. The right next move is one
explicit bridge tranche that binds `TCM.v1`, witness proofs, resumable
continuation, portability, and verdict splitting onto the canonical owned
Transformer route without weakening the existing refusal and publication
discipline.

That bridge should also be written so a later weighted plugin system can sit on
top of it cleanly, rather than forcing the repo to reopen the universality
contract just to explain packet-mediated capability execution or software-like
artifact identity, and it should lock machine identity, execution semantics,
continuation discipline, fast-route legitimacy, and served conformance before
the repo treats the rebase as stable.

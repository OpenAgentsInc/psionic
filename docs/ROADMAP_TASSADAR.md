# Psionic Tassadar Roadmap Bridge

> Status: repo-local bridge refreshed 2026-03-19 after reviewing the live
> external Tassadar roadmap, `README.md`, `docs/ARCHITECTURE.md`,
> `docs/ROADMAP_TASSADAR_INDEX.md`, `docs/ROADMAP_TASSADAR_TAS_SYNC.md`,
> `docs/TASSADAR_WASM_RUNBOOK.md`, and the open `TAS-*` GitHub issue queue.

This file stays in the repo because a large set of GitHub issues and repo docs
already link to `docs/ROADMAP_TASSADAR.md`.

It is no longer the full living Tassadar roadmap.

## Canonical Live Roadmap

The active Tassadar roadmap now lives outside this repo at:

- `/Users/christopherdavid/code/alpha/tassadar/tassadar-llm-as-computer-roadmap.md`

Use that file for:

- tranche definitions
- issue sequencing after `TAS-102`
- terminal-contract language
- the current widening path from bounded named profiles to `TCM.v1`

This repo-local file remains a stable bridge so existing issue links keep
landing on something truthful instead of a stale copied roadmap.

## Current Repo-Local Summary

As of 2026-03-19, the live external roadmap and the current GitHub `tassadar`
issue queue are aligned.

Current honest posture:

- benchmarked, bounded internal computation under named profiles with explicit
  refusal surfaces
- the current served profile remains
  `tassadar.internal_compute.article_closeout.v1`
- the public repo does have a bounded Rust-only article-closeout path for the
  committed canonical workloads with direct no-tool proof receipts, but that
  closeout is still route-bound and workload-bound rather than any broad
  arbitrary-Rust frontend scope or a generic public Wasm-interpreter claim
- the current audited article-closeout runtime remains the exact
  reference-linear CPU lane on those committed workloads; faster runtime
  families exist as separate runtime or research surfaces and should not be
  flattened into "the full article fast-attention headline is now the default
  public route"
- the frozen core-Wasm lane now has a declared semantic window plus a committed
  closure gate, plus a public acceptance gate and operator runbook-v2 summary;
  the current closure and public-acceptance verdicts remain suppressed with
  `served_publication_allowed = false`
- the repo now also has a bounded scalar-`f32` semantics matrix with canonical
  quiet-NaN normalization, ordered Wasm-style comparisons, CPU-reference-only
  execution posture, and explicit refusal on `f64`, NaN-payload preservation,
  and non-CPU fast-math regimes
- the repo now also has a staged mixed-numeric ladder over exact scalar-`f32`,
  exact mixed `i32`/`f32`, and bounded-approximate `f64 -> f32` conversion
  profiles, with malformed and out-of-envelope conversions kept on explicit
  typed refusal paths
- the repo now also has a numeric portability matrix over backend, toolchain,
  and machine-class envelopes for the bounded float and mixed-numeric lanes,
  keeping exact cpu-reference publication separate from suppressed non-CPU and
  bounded-approximate numeric regimes
- the repo now also has a float-profile acceptance gate plus route policy for
  exact numeric profiles, allowing bounded cpu-reference public named-profile
  posture without widening those profiles into the default served lane
- the repo now also has one bounded exceptions proposal profile over typed
  throw/catch/rethrow semantics with explicit trap-stack parity and malformed
  handler refusal truth, allowing named public profile posture only on the
  current-host cpu-reference envelope while keeping the default served
  exceptions lane empty
- the repo now also has one bounded `memory64` continuation profile over
  sparse single-memory checkpoints above the 4GiB boundary, with committed
  resume artifacts, typed datastream locators, and explicit backend-limit
  refusal truth on the current-host cpu-reference envelope
- the repo now also has one bounded multi-memory routing profile over two
  explicit topology families, with committed per-memory checkpoint artifacts
  and typed malformed-topology refusal truth on the current-host cpu-reference
  envelope
- the repo now also has one bounded component/linking proposal profile over
  two explicit component-pair topologies, with committed interface-type
  lineage artifacts and typed incompatible-interface refusal truth on the
  current-host cpu-reference envelope
- the repo now also has one bounded SIMD deterministic profile with committed
  backend rows for cpu-reference exactness, metal and cuda scalar fallback,
  and typed accelerator-specific refusal truth, carried through served
  publication without creating a default served SIMD lane
- the repo now also has one research-only relaxed-SIMD ladder with committed
  runtime, eval, and research artifacts over one exact cpu-reference anchor,
  bounded metal/cuda drift candidates, and typed refusal on unstable lane
  semantics plus cross-backend non-portability; it remains explicitly
  non-promoted and non-served by design
- the repo now also has one disclosure-safe general internal-compute red-team
  audit, with committed router, eval, and research artifacts over candidate-
  only broad internal-compute routes, operator-only proposal-family profiles,
  research-only threads publication, relaxed-SIMD non-promotion, and
  arbitrary-Wasm claim leakage; the current audit is clean and publication-safe
  as an audit surface, but it does not widen broad served internal compute,
  arbitrary Wasm, or Turing-complete support
- the repo now also has one disclosure-safe pre-closeout universality audit,
  with committed eval and research artifacts that freeze exactly which
  broadness-adjacent surfaces are already real before the terminal contract and
  exactly which terminal-contract artifacts are still missing; the current
  terminal claim still remains suppressed until the final closeout audit lands
- the repo now also has one declared terminal substrate model `TCM.v1`, with a
  committed IR model artifact and runtime contract report that bind universal-
  substrate language to explicit control, memory, continuation, and
  effect-boundary rows; this is the substrate declaration only, not yet the
  witness construction or final universality gate
- the repo now also has one explicit universal-machine witness construction,
  with committed compiler, runtime, eval, and research artifacts for a
  two-register machine and a single-tape machine over `TCM.v1`; this closes the
  witness-construction step only, not yet the final gate
- the repo now also has one dedicated universality witness benchmark suite,
  with committed data and environment contracts plus eval and research artifacts
  covering register-machine, tape-machine, vm-style interpreter, session-
  process, and spill/tape witness families while keeping VM parameter ABI and
  open-ended external event loops on explicit refusal boundaries; this still
  does not constitute the minimal universal-substrate gate, the verdict split,
  or served universality posture
- the repo now also has one minimal universal-substrate acceptance gate, with a
  committed runtime prerequisite report, eval gate artifact, provider receipt,
  and one-command checker script that turn green only when conditional control,
  mutable memory, spill/tape extension, persistent continuation, machine-step
  replay, witness coverage, portability envelopes, and refusal truth are all
  explicit; this still does not constitute the theory/operator/served verdict
  split, served universality posture, or Turing-complete closeout
- the repo now also has one explicit theory/operator/served universality
  verdict split, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_universality_verdict_split_report.json`,
  a served publication projection, a provider-facing receipt, and an audit note
  that keep `theory_green=true`, `operator_green=true`, and
  `served_green=false` separate instead of silently widening the final claim;
  the served lane remains suppressed because no named served universality
  profile is published and authority-bearing served closure still lives outside
  standalone `psionic`
- the repo now also has one final Turing-completeness closeout audit, with a
  committed eval report at
  `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_audit_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_summary.json`,
  a provider-facing receipt, and an audit note that freeze the exact bounded
  terminal statement: Turing-complete support is green for theory and operator
  use under declared `TCM.v1` semantics and explicit envelopes, while
  served/public universality remains suppressed
- the repo now also has one post-article universality bridge contract, with a
  committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that bind the
  historical `TCM.v1` closeout chain to one explicit post-`TAS-186` machine
  identity tuple on the canonical owned article route without rewriting the
  older artifacts; the bridge keeps the direct article-equivalent lane,
  resumable universality lane, and reserved later capability plane explicit,
  but it does not yet reissue the bounded Turing-completeness claim on the
  canonical route or imply served/public universality, weighted plugin
  control, or arbitrary software capability
- the repo now also has one post-article canonical-route semantic-preservation
  audit, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_route_semantic_preservation_audit_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_route_semantic_preservation_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that prove
  the declared continuation mechanics preserve canonical identity, declared
  state ownership, and declared semantics on the bridge machine identity
  instead of merely preserving a subset of outputs; this tranche still does
  not prove decision provenance, publish the final carrier split, reissue the
  bounded Turing-completeness claim on the canonical route, or imply
  served/public universality, weighted plugin control, or arbitrary software
  capability
- the repo now also has one post-article control-plane decision-provenance
  proof, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_control_plane_decision_provenance_proof_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_control_plane_decision_provenance_proof_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that bind
  branch, retry, and stop decisions to model outputs, canonical route
  identity, and the bridge machine identity while freezing determinism,
  equivalent-choice, failure semantics, time semantics, information
  boundaries, hidden-state closure, observer acceptance, and hidden-control
  channel exclusions; this tranche still does not publish the final carrier
  split, reissue the bounded Turing-completeness claim on the canonical route,
  imply served/public universality, weighted plugin control, or arbitrary
  software capability
- the repo now also has one post-article carrier-split contract, with a
  committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_carrier_split_contract_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_carrier_split_contract_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that bind
  direct article-equivalent truths and resumable bounded-universality truths
  to different carriers on the same bridge machine identity while explicitly
  blocking transfer by implication and keeping the reserved capability plane
  explicit; this tranche still does not rebind the historical
  universal-machine proof, reissue the witness suite, publish the rebased
  verdict split, imply served/public universality, weighted plugin control,
  or arbitrary software capability
- the repo now also has one post-article universal-machine proof-rebinding
  report, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_universal_machine_proof_rebinding_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_universal_machine_proof_rebinding_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that bind
  the historical universal-machine proof to the bridge machine identity,
  canonical model artifact, canonical weight artifact, and canonical route id
  through one explicit proof-transport boundary instead of metadata-only
  relabeling while keeping helper substitution, route-family drift,
  undeclared cache-owned control, undeclared batching semantics, and semantic
  drift blocked; this tranche still does not reissue the broader witness
  suite, enable the canonical-route universal-substrate gate, publish the
  rebased verdict split, imply served/public universality, weighted plugin
  control, or arbitrary software capability
- the repo now also has one post-article universality witness-suite reissue
  report, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_universality_witness_suite_reissue_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_universality_witness_suite_reissue_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that rebind
  the older seven-row witness suite onto the bridge machine identity,
  canonical model artifact, and canonical route id while keeping helper
  substitution, hidden cache-owned control flow, and resume-only cheating
  explicit as negative rows; this tranche still does not enable the
  canonical-route universal-substrate gate, publish the rebased verdict
  split, imply served/public universality, weighted plugin control, or
  arbitrary software capability
- the repo now also has one post-article canonical-route universal-substrate
  gate, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_route_universal_substrate_gate_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_route_universal_substrate_gate_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that join the
  historical minimal universal-substrate artifact with the bridge contract,
  semantic-preservation audit, control-plane proof, carrier split,
  proof-rebinding, and canonical-route witness-suite reissue on one declared
  machine identity while keeping portability, refusal truth, helper
  substitution, route drift, continuation abuse, semantic drift, and
  overclaim posture explicit; this tranche still does not publish the rebased
  verdict split, imply served/public universality, weighted plugin control,
  or arbitrary software capability
- the repo now also has one post-article universality portability/minimality
  matrix, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_universality_portability_minimality_matrix_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_universality_portability_minimality_matrix_summary.json`,
  a provider-facing receipt, a served conformance envelope at
  `fixtures/tassadar/reports/tassadar_post_article_universality_served_conformance_envelope.json`,
  a checker script, and an audit note that extend the rebased canonical-route
  universality lane across one declared CPU machine matrix, one explicit
  route-carrier classification, and one machine-level minimality contract
  while keeping out-of-envelope machines explicitly suppressed and preserving
  the narrower served article-closeout boundary; this tranche still does not
  publish the rebased theory/operator/served verdict split, imply served/public
  universality, weighted plugin control, or arbitrary software capability
- the repo now also has one post-article rebased theory/operator/served verdict
  split, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_split_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_split_summary.json`,
  a served publication at
  `fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_publication.json`,
  a provider-facing receipt, a checker script, and an audit note that rebind
  the older universality verdict split onto the canonical bridge machine,
  canonical route, and canonical portability/minimality envelope while keeping
  served/public universality suppressed and plugin capability explicitly out of
  scope; this tranche now allows the rebased theory/operator claim, but it
  still does not imply weighted plugin control, served/public universality, or
  arbitrary software capability
- the repo now also has one machine-readable article-equivalence blocker
  matrix, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_article_equivalence_blocker_matrix_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_article_equivalence_blocker_matrix_summary.json`,
  a checker script, and an audit note that freeze the separate post-`TAS-156`
  article-gap closure bar without widening the current public capability
  surface; the blocker contract is green and final bounded article
  equivalence now closes elsewhere on top of it
- the repo now also has one final article-equivalence acceptance gate skeleton,
  with a committed eval report at
  `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`,
  a checker script, a provider-facing receipt, and an audit note that freeze
  the final gate over the blocker-matrix contract, the owned
  `psionic-transformer` route boundary, blocker closure, and every required
  TAS tranche from `TAS-158` through `TAS-186`; the gate is now green and does
  not by itself widen public capability claims beyond the declared envelope
- the repo now also has one canonical existing-substrate inventory for the
  article-equivalence closure wave, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_existing_substrate_inventory_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_existing_substrate_inventory_summary.json`,
  and an audit note that freeze which current `psionic-core`, `psionic-array`,
  `psionic-nn`, `psionic-transformer`, `psionic-models`, and
  `psionic-runtime` surfaces are reusable as-is, reusable with extension,
  research-only, or still insufficient for canonical article closure; the
  inventory is explicitly tied to the final acceptance gate while the final
  article-equivalence verdict now closes elsewhere
- the repo now also has one canonical Transformer stack boundary for the
  article-equivalence closure wave, with a boundary spec at
  `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`, a committed eval
  report at
  `fixtures/tassadar/reports/tassadar_canonical_transformer_stack_boundary_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_canonical_transformer_stack_boundary_summary.json`,
  and an audit note that freeze one explicit multi-crate owned route boundary
  with `psionic-transformer` as the architecture anchor, `psionic-models` as
  the canonical article-model and artifact owner, and `psionic-runtime` as the
  replay and receipt owner; the boundary is explicitly tied to the final
  acceptance gate while the final article-equivalence verdict now closes
  elsewhere
- the repo now also has one canonical owned scaled dot-product attention and
  mask path for the article-equivalence closure wave, with reusable
  implementation in `crates/psionic-transformer/src/attention.rs`, a committed
  eval report at
  `fixtures/tassadar/reports/tassadar_attention_primitive_mask_closure_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_attention_primitive_mask_closure_summary.json`,
  and an audit note that freeze stable softmax, causal and padding masks,
  combined-mask support, deterministic forward behavior, and
  probability-trace export as owned `psionic-transformer` truth; this tranche
  is explicitly tied to the final acceptance gate while the final
  article-equivalence verdict now closes elsewhere
- the repo now also has one canonical reusable Transformer block layer for the
  article-equivalence closure wave, with reusable implementation in
  `crates/psionic-transformer/src/blocks.rs`, a committed eval report at
  `fixtures/tassadar/reports/tassadar_transformer_block_closure_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_transformer_block_closure_summary.json`,
  and an audit note that freeze token embedding plus positional binding,
  train-versus-eval dropout posture, multi-head projection and merge,
  feed-forward composition, residual plus layer norm, and reusable
  decoder-block execution as owned `psionic-transformer` truth while
  preserving `psionic-nn` as the lower-level layer substrate through the
  `psionic-nn-optimizers` split; this tranche is explicitly tied to the final
  acceptance gate while the overall article-equivalence verdict remains
  blocked
- the repo now also has one canonical paper-faithful article-Transformer
  model path for the article-equivalence closure wave, with reusable
  encoder-decoder stack implementation in
  `crates/psionic-transformer/src/encoder_decoder.rs`, one canonical article
  wrapper in `crates/psionic-models/src/tassadar_article_transformer.rs`, a
  committed eval report at
  `fixtures/tassadar/reports/tassadar_article_transformer_model_closure_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_article_transformer_model_closure_summary.json`,
  and an audit note that freeze the explicit `Attention Is All You Need`
  reference, encoder stack, masked decoder, cross-attention, logits
  projection, and embedding-sharing modes on one owned route while keeping
  article trace vocabulary, artifact-backed weights, replay receipts, and the
  final article-equivalence verdict open; this tranche is explicitly tied to
  the final acceptance gate while the final article-equivalence verdict now
  closes elsewhere
- the repo now also has one research-only shared-memory and threads profile
  with a deterministic scheduler envelope, explicit sandbox boundary truth,
  and typed refusal on host-nondeterministic scheduling plus relaxed
  memory-ordering regimes; it remains non-served and non-promoted by design
- the repo now also has one shared-state concurrency challenge lane with
  committed runtime, cluster, eval, and research artifacts over two
  operator-green single-host deterministic classes, one explicitly suppressed
  threads profile, and typed refusal on host-nondeterministic scheduling,
  relaxed memory ordering, and cross-worker shared heaps; it remains
  non-served and disclosure-bounded by design
- the repo now also has one durable process-object family
  `tassadar.internal_compute.process_objects.v1` over the committed
  checkpoint-backed workloads, with first-class snapshot, tape, and work-queue
  objects, typed datastream locators, provider receipts, and environment
  bindings while keeping stale snapshots, out-of-range tape cursors, and
  profile-mismatched queues on explicit refusal paths
- the repo now also has one bounded interactive session-process profile
  `tassadar.internal_compute.session_process.v1` over deterministic echo and
  stateful counter message-loop surfaces, with committed runtime, eval, and
  route-policy reports plus served-publication and provider-envelope bindings,
  while keeping open-ended external event streams on explicit refusal paths and
  leaving the default served session-process lane empty
- the repo now also has one bounded spill-aware continuation profile
  `tassadar.internal_compute.spill_tape_store.v1` over spill-backed memory
  segments and external tape-store artifacts, with committed runtime, eval,
  provider, and environment bindings on the current-host cpu-reference
  portability envelope while keeping oversize state, missing tape segments, and
  non-cpu portability widening on explicit refusal paths
- the repo now also has one bounded preemptive-job profile
  `tassadar.internal_compute.preemptive_jobs.v1` over deterministic
  round-robin and weighted-fair slice schedulers, with committed runtime,
  cluster, and eval reports plus served-publication and provider-envelope
  bindings that keep the lane operator-visible but non-served while refusing
  host-nondeterministic scheduling explicitly
- the repo now also has one bounded virtual-filesystem and artifact-mount
  profile `tassadar.effect_profile.virtual_fs_mounts.v1`, with committed
  runtime, sandbox, and eval artifacts plus provider receipts that keep the
  lane challengeable, replay-safe, and non-served while leaving canonical
  mount authority explicitly owned by `world-mounts` outside standalone psionic
- the repo now also has one bounded simulator-backed effect profile
  `tassadar.effect_profile.simulator_backed_io.v1`, with committed runtime,
  sandbox, and eval artifacts plus provider receipts and environment bindings
  that admit seeded clock, randomness, and loopback-network rows while keeping
  ambient clock, OS entropy, and socket I/O on explicit refusal paths and
  leaving the lane non-served by design
- the repo now also has one bounded async-lifecycle profile
  `tassadar.internal_compute.async_lifecycle.v1`, with committed runtime, eval,
  and router artifacts plus served-publication and provider-envelope bindings
  that admit deterministic interrupt, bounded retry, and safe-boundary
  cancellation rows while keeping open-ended callbacks, mid-effect
  cancellation, and unbounded retry on explicit refusal paths and leaving the
  default served async lane empty
- the repo now also has one bounded effectful replay-and-challenge lane
  `tassadar.effect_profile.replay_challenge_receipts.v1`, with committed
  runtime and eval artifacts plus provider-facing receipts that bind seeded
  simulator, virtual-filesystem, and async-safe-cancel replay rows to explicit
  effect receipts, replay digests, and challenge receipts while keeping
  missing-effect evidence, missing challenge evidence, and unsafe effect
  families on explicit refusal paths; named `kernel-policy` and `nexus`
  follow-ons remain explicit dependency markers outside standalone psionic and
  the lane stays non-served by design
- the repo now also has one bounded internal component-model ABI lane
  `tassadar.internal_compute.component_model_abi.v1`, with committed IR and
  compiler contracts, runtime and eval artifacts, and provider-facing receipts
  that bind session-checkpoint, artifact-reader-retry, and spill-resume
  component graphs to explicit interface manifests while keeping cross-profile
  handle mismatches and unsupported variant unions on typed refusal paths; the
  lane remains benchmark-only with `served_publication_allowed = false`
- the repo now also has one bounded internal-compute package-manager lane with
  committed compiler, router, eval, served-publication, and provider-facing
  surfaces that publish the named public packages
  `package.clrs_shortest_path_stack.v1`,
  `package.hungarian_matching_stack.v1`, and
  `package.verifier_search_stack.v1`, while keeping ambiguous solver,
  insufficient-evidence, and portability-mismatch requests on explicit refusal
  paths and leaving the default served package lane empty
- the repo now also has one bounded cross-profile link-compatibility lane with
  committed compiler, runtime, router, eval, and provider-facing surfaces that
  preserve one exact session-process -> spill-tape link, one explicit
  generalized-ABI -> component-model downgrade plan, and typed refusal on
  portability-envelope and effect-boundary mismatches while keeping
  `served_publication_allowed = false`
- broader profiles above that remain benchmarked, suppressed, refused, or
  unpromoted unless their explicit gates are green

Still unclaimed:

- arbitrary Wasm execution
- full frozen core-Wasm closure
- broad practical internal computation as a generally served capability
- served/public Turing-complete support
- the Percepta article headline in its broadest frontend/runtime reading

Current dependency spine:

1. `TAS-103` to `TAS-106`: real program execution
2. `TAS-107` to `TAS-110`: resumable and effect-safe execution
3. `TAS-111` to `TAS-112`: portable and publishable execution
4. `TAS-113` to `TAS-114`: frozen core-Wasm closure
5. `TAS-115` to `TAS-124`: numeric and proposal-family widening
6. `TAS-125` to `TAS-136`: process and internal-platform closure
7. `TAS-137` to `TAS-140`: learned and hybrid broad-compute comparison
8. `TAS-141` to `TAS-150`: public-claim, economic, and governance closure
9. `TAS-151` to `TAS-156`: universal-substrate and Turing-completeness
   closeout

Issue-state note:

- `TAS-103` through `TAS-124` are already implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-125` through `TAS-146` are now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-147` through `TAS-155` are now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-156` is now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-157` is now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- there is currently no open GitHub TAS backlog in this repo

## Repo-Local Companion Docs

Use the repo-local docs for the questions they actually answer:

- `README.md` for the current Tassadar inventory and claim surface inside this
  repo
- `docs/ARCHITECTURE.md` for canonical Psionic-wide status vocabulary and
  machine-facing contract boundaries
- `docs/ROADMAP_TASSADAR_INDEX.md` for landed phase-to-artifact mapping
- `docs/ROADMAP_TASSADAR_TAS_SYNC.md` for issue-to-implementation closure
  notes
- `docs/TASSADAR_WASM_RUNBOOK.md` for the current bounded Wasm operator path
- `fixtures/tassadar/reports/tassadar_acceptance_report.json` and
  `scripts/check-tassadar-acceptance.sh` for the current machine-readable claim
  gate

If you only have the public `psionic` checkout and not the external Tassadar
workspace, use this bridge plus the current `TAS-*` issue bodies, the TAS sync
doc, and the artifact index together instead of relying on the old copied
roadmap text.

## Maintenance Rule

- update the external roadmap when tranche order, issue sequencing, or
  terminal-contract language changes
- update this repo-local bridge when the pointer, current truthful posture, or
  repo-visible issue alignment changes
- do not reintroduce a copied issue-by-issue backlog here unless the canonical
  roadmap moves back into this repo

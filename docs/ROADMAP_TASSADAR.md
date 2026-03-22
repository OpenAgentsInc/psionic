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
- the repo now also has one post-article plugin-capability boundary contract,
  with a committed sandbox-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that make the
  rebased closeout explicitly plugin-aware without widening `TCM.v1`, the
  continuation contract, or the rebased verdict into weighted plugin control;
  this tranche keeps plugin execution on a separate software-capability layer,
  freezes plugin packet/state/receipt identity as separate from core compute,
  reserves choice-set integrity/resource transparency/scheduling ownership and
  the first closed-world operator-curated plugin tranche, and still does not
  imply plugin publication, served/public universality, or arbitrary software
  capability
- the repo now also has one post-article plugin charter and authority
  boundary contract, with a committed sandbox-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_charter_authority_boundary_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_charter_authority_boundary_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that bind
  the plugin lane to one canonical post-`TAS-186` machine identity and one
  computational-model statement, inherit the pre-plugin control-plane proof
  as a hard dependency, freeze explicit data/control/capability planes plus
  packet-local/instance-local/host-backed/weights-owned state classes, and
  freeze operator/internal-only publication posture, semantic-preservation,
  anti-interpreter-smuggling, and governance-receipt rules without widening
  the current claim surface; this tranche still does not imply weighted
  plugin control, plugin publication, served/public universality, or
  arbitrary software capability
- the repo now also has one post-article plugin manifest, identity, and
  hot-swap contract, with a committed catalog-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that bind
  canonical plugin identity to `plugin_id`, `plugin_version`,
  `artifact_digest`, declared exports, packet ABI version, schema ids,
  limits, trust tier, replay class, and evidence settings, define canonical
  invocation identity and typed hot-swap compatibility rules, and keep
  linked multi-module packaging explicit without widening the current claim
  surface beyond operator/internal plugin artifacts; this tranche still keeps
  weighted plugin control, plugin publication, served/public universality,
  and arbitrary software capability blocked
- the repo now also has one post-article plugin packet ABI and Rust-first
  PDK contract, with a committed sandbox-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_packet_abi_and_rust_pdk_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_packet_abi_and_rust_pdk_summary.json`,
  a committed runtime bundle, a provider-facing receipt, a checker script,
  and an audit note that freeze a single `packet.v1` invocation contract and
  a single Rust-first guest authoring surface, including one input packet,
  one output packet or typed refusal, one explicit host-error channel, one
  host receipt channel, packet-level schema and codec ids, bytes payloads,
  metadata envelopes, one `handle_packet` export, one typed refusal family,
  and one narrow packet-host import namespace; this tranche still keeps
  weighted plugin control, plugin publication, served/public universality,
  and arbitrary software capability blocked
- the repo now also has one post-article plugin runtime API and engine
  abstraction contract, with a committed sandbox-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary.json`,
  a committed runtime bundle, a served publication, a provider-facing
  receipt, a checker script, and an audit note that freeze one host-owned
  plugin runtime API and one backend-neutral engine layer above the packet
  ABI, including digest-verified loading, bounded instantiate/invoke/mount/
  cancel/usage operations, explicit timeout/memory/queue/pool/concurrency
  ceilings, fixed model-information boundaries for queue depth, retries,
  runtime cost, and time, fixed scheduling semantics, cost-model
  invariance, and explicit per-plugin/per-step/per-workflow failure
  isolation; this tranche still keeps weighted plugin control, plugin
  publication, served/public universality, and arbitrary software
  capability blocked
- the repo now also has one post-article plugin invocation-receipt and replay
  class contract, with a committed eval-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_summary.json`,
  a committed runtime bundle, a provider-facing receipt, a checker script, and
  an audit note that freeze canonical invocation receipt identity above the
  host-owned runtime API, including explicit receipt/install/plugin/artifact/
  export/packet/envelope/backend identity, required resource summaries, four
  replay classes, twelve typed refusal and failure classes, route-integrated
  evidence, and explicit challenge bindings for success and snapshot-replayable
  failure lanes; this tranche still keeps weighted plugin control, plugin
  publication, served/public universality, and arbitrary software capability
  blocked
- the repo now also has one post-article plugin world-mount envelope compiler
  and admissibility contract, with a committed sandbox-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_summary.json`,
  a committed runtime bundle, a provider-facing receipt, a checker script, and
  an audit note that freeze canonical closed-world plugin admissibility above
  the invocation-receipt layer, including explicit candidate-set enumeration,
  auditable equivalent-choice classes, route and mount binding, version
  constraints, trust and publication posture binding, compiled capability and
  network or artifact mount envelopes, receipt-visible filtering, and typed
  denied, suppressed, and quarantined outcomes; this tranche still keeps
  weighted plugin control, plugin publication, served/public universality, and
  arbitrary software capability blocked
- the repo now also has one post-article plugin conformance sandbox and
  benchmark harness, with a committed sandbox-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_report.json`,
  a committed eval-owned closure report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_eval_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_summary.json`,
  a committed runtime bundle, a provider-facing receipt, a checker script, and
  an audit note that freeze the first plugin conformance and benchmark harness
  above the admissibility layer, including static host-scripted conformance
  traces, roundtrip and typed refusal or limit behavior, explicit workflow
  integrity and envelope intersection, hot-swap compatibility, failure-domain
  isolation, side-channel and covert-channel negatives, and cold or warm or
  pooled or queued or cancelled benchmark evidence; this tranche clears the
  admissibility defer pointer to empty and later clears its own defer pointer
  to empty after `TAS-203A`, while still keeping weighted plugin sequencing,
  plugin publication, served/public universality, and arbitrary software
  capability blocked
- the repo now also has one post-article plugin result-binding,
  schema-stability, and composition contract, with a committed eval-owned
  report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_summary.json`,
  a committed runtime bundle, a provider-facing receipt, a checker script, and
  an audit note that freeze the transformer-owned result-binding contract plus
  runtime-owned schema-stability evidence above the conformance harness,
  including explicit output-to-state digest binding, backward-compatible
  schema evolution, typed refusal normalization, proof-versus-observational
  result boundaries, non-lossy multi-step semantic closure, and fail-closed
  posture on lossy coercion, schema auto-repair, ambiguous composition, and
  semantically incomplete reinjection; this tranche now clears its deferred
  frontier to empty after `TAS-204` and still keeps weighted plugin
  sequencing local to the dedicated controller artifact rather than this
  result-binding proof alone, plugin
  publication, served/public universality, and arbitrary software capability
  blocked
- the repo now also has one post-article weighted plugin controller trace and
  refusal-aware model loop, with a committed sandbox-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_report.json`,
  a committed eval-owned closure report at
  `fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_eval_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_summary.json`,
  a committed runtime bundle, a provider-facing receipt, a checker script, and
  an audit note that freeze the first canonical weighted plugin controller
  trace above the result-binding and pre-plugin control-plane proofs,
  including explicit model ownership over plugin selection, export selection,
  packet-argument construction, continuation, retry, refusal, and stop
  conditions, explicit determinism and external-signal boundaries, and
  explicit host-negative planner attacks; this tranche turns weighted plugin
  control green on the canonical route, clears the controller defer pointer to
  empty, with the later bridge reservation frontier now at `TAS-216`, and still
  keeps bounded plugin-platform closeout, plugin publication, served/public
  universality, and arbitrary software capability blocked
- the repo now also has one post-article plugin authority, promotion,
  publication, and trust-tier gate, with a committed catalog-owned report at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_summary.json`,
  a provider-facing receipt, a dedicated checker script, and an audit note
  that freeze explicit research-only, benchmark-gated-internal, and
  challenge-gated-install trust tiers, explicit promotion or quarantine or
  revocation or supersession receipts, explicit operator/internal-only
  posture, explicit profile-specific deterministic-import and runtime-support
  route hooks, and explicit broader public suppression or refusal above the
  already-closed weighted controller; this tranche turns the bounded plugin
  authority and posture gate green while now serving as one prerequisite
  beneath the separate bounded plugin-platform closeout audit and still does
  not by itself turn plugin publication, served/public universality, or
  arbitrary software capability green
- the repo now also has one post-article bounded weighted plugin-platform
  closeout audit, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_bounded_weighted_plugin_platform_closeout_audit_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_bounded_weighted_plugin_platform_closeout_summary.json`,
  a provider-facing receipt, a dedicated checker script, and an audit note
  that bind the post-article Turing closeout, plugin charter, manifest
  identity, packet ABI, runtime API, invocation receipts, admissibility
  compiler, conformance harness, result-binding contract, weighted controller
  trace, and authority or promotion or publication gate into one bounded
  operator/internal plugin-platform statement on the canonical rebased
  machine instead of letting adjacent green artifacts imply that platform by
  recomposition; this tranche turns `plugin_capability_claim_allowed=true`,
  keeps plugin publication suppressed, keeps served/public universality false,
  keeps arbitrary software capability false, and now requires the published
  canonical machine closure bundle from `TAS-215` by digest
- the repo now also has one post-article canonical machine identity lock, with
  a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_machine_identity_lock_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_machine_identity_lock_summary.json`,
  a transformer-owned anchor contract, a provider-facing receipt, a dedicated
  checker script, and an audit note that freeze one globally named machine
  tuple above the bridge, benchmark route, proof and witness chain, plugin
  receipts, controller traces, authority posture, and bounded platform
  closeout, explicitly rebind legacy partial-tuple artifacts by digest
  instead of leaving identity inheritance implicit, refuse mixed-carrier
  recomposition, and still keep plugin publication, served/public
  universality, arbitrary software capability, and the final claim-bearing
  canonical machine closure bundle out of scope
- the repo now also has one post-article canonical computational-model
  statement, with a committed runtime report at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_summary.json`,
  a provider-facing receipt, a dedicated checker script, and an audit note
  that publish one machine-readable statement over the already-locked canonical
  machine identity: direct article-equivalent compute is one owned Transformer
  route, resumable continuation semantics and effect boundaries attach only
  through the historical green `TCM.v1` runtime contract, and any plugin layer
  sits above that machine as a bounded software-capability overlay instead of
  redefining the substrate; this tranche now sits beneath the separate
  `TAS-209` proof-transport audit, keeps the final claim-bearing canonical
  machine closure bundle separate for `TAS-215`, and still keeps plugin
  publication, served/public universality, and arbitrary software capability
  blocked
- the repo now also has one post-article execution-semantics proof-transport
  audit, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_execution_semantics_proof_transport_audit_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_execution_semantics_proof_transport_audit_summary.json`,
  a transformer-owned anchor contract, a provider-facing receipt, a dedicated
  checker script, and an audit note that bind the historical universal-machine
  proof, the post-article proof-rebinding receipts, the published
  computational-model statement, the historical green `TCM.v1` continuation
  contract, and the current plugin runtime, conformance, and weighted-controller
  surfaces to one explicit proof-bearing execution boundary; this tranche
  closes proof transport, moves the next anti-drift frontier to `TAS-215`,
  keeps the final claim-bearing canonical machine closure bundle separate for
  `TAS-215`, and still keeps plugin publication, served/public universality,
  and arbitrary software capability blocked
- the repo now also has one post-article continuation non-computationality
  contract, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_summary.json`,
  a transformer-owned anchor contract, a provider-facing receipt, a dedicated
  checker script, and an audit note that freeze checkpoint, spill, tape,
  session, process-object, installed-process, and weighted-controller
  continuation surfaces as transport-only extensions of the same canonical
  machine instead of a second machine; this tranche closes continuation
  non-computationality, refreshes the dependent conformance, authority,
  bounded-platform-closeout, computational-model, proof-transport, and
  machine-lock artifacts onto the same canonical statement binding, moves the
  next anti-drift frontier to `TAS-215`, keeps the final claim-bearing
  canonical machine closure bundle separate for `TAS-215`, and still keeps
  plugin publication, served/public universality, and arbitrary software
  capability blocked
- the repo now also has one post-article fast-route legitimacy and
  carrier-binding contract, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_summary.json`,
  a transformer-owned anchor contract, a provider-facing receipt, a dedicated
  checker script, and an audit note that classify `reference_linear` as the
  historical proof baseline, `hull_cache` as the canonical direct carrier only
  while route selection, implementation, semantic-preservation, proof
  transport, and machine binding remain jointly green, `resumable_continuation_family`
  as a continuation-only carrier extension, and the current research-only fast
  families as outside the carrier until later explicit promotion; this tranche
  closes fast-route legitimacy and carrier binding, refuses served or plugin
  wording that treats an unbound fast route as the underlying machine, moves
  the next anti-drift frontier to `TAS-215`, keeps the final claim-bearing
  canonical machine closure bundle separate for `TAS-215`, and still keeps
  plugin publication, served/public universality, and arbitrary software
  capability blocked
- the repo now also has one post-article equivalent-choice neutrality and
  admissibility contract, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_summary.json`,
  a transformer-owned anchor contract, a provider-facing receipt, a dedicated
  checker script, and an audit note that freeze one auditable equivalent-choice
  class model on the canonical post-article machine above the already-closed
  admissibility compiler, runtime API report, control-plane decision-provenance
  proof, weighted-controller negative planner surface, fast-route carrier
  classification, and universality bridge instead of letting equivalent plugin
  choices drift by implication; this tranche requires receipt-visible
  narrowing, keeps hidden ordering or ranking plus latency or cost or
  soft-failure steering outside the admitted control surface, refuses served or
  plugin overread from equivalence posture alone, moves the next anti-drift
  frontier to `TAS-215`, keeps the final claim-bearing canonical machine
  closure bundle separate for `TAS-215`, and still keeps plugin publication,
  served/public universality, and arbitrary software capability blocked
- the repo now also has one post-article downward non-influence and served
  conformance contract, with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_downward_non_influence_and_served_conformance_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_downward_non_influence_and_served_conformance_summary.json`,
  a transformer-owned anchor contract, a provider-facing receipt, a dedicated
  checker script, and an audit note that bind the canonical computational-
  model statement, canonical machine lock, proof-transport boundary,
  continuation boundary, fast-route carrier boundary, equivalent-choice
  boundary, served conformance envelope, rebased verdict split, and historical
  served suppression posture into one explicit anti-rewrite contract instead
  of letting later plugin or served ergonomics redefine lower-plane truth by
  adjacency; this tranche closes downward non-influence and served
  conformance, keeps lower-plane truth rewrite refusal machine-checkable,
  keeps served posture explicitly narrower than operator truth inside one
  declared fail-closed envelope, moves the next anti-drift frontier to
  `TAS-215`, keeps the final claim-bearing canonical machine closure bundle
  separate for `TAS-215`, and still keeps plugin publication, served/public
  universality, and arbitrary software capability blocked
- the repo now also has one post-article anti-drift stability closeout audit,
  with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_anti_drift_stability_closeout_audit_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_anti_drift_stability_closeout_summary.json`,
  a transformer-owned anchor contract, a provider-facing receipt, a dedicated
  checker script, and an audit note that bind the published computational-
  model statement, canonical machine identity lock, control-plane
  decision-provenance proof, proof-transport boundary, continuation boundary,
  fast-route carrier boundary, equivalent-choice boundary, downward
  non-influence and served conformance boundary, rebased verdict split,
  portability/minimality matrix, plugin charter authority boundary, and
  bounded weighted plugin-platform closeout into one explicit anti-drift
  verdict over one canonical post-article machine; this tranche closes
  anti-drift stability, keeps sampled audits distinct from proof-carrying
  artifacts, and now sits beneath the published canonical machine closure
  bundle from `TAS-215`, which binds stronger terminal and stronger
  plugin-platform claims by digest while still not turning plugin publication,
  served/public universality, or arbitrary software capability green
- the repo now also has one post-article Turing-completeness closeout audit,
  with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_turing_completeness_closeout_audit_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_turing_completeness_closeout_summary.json`,
  a provider-facing receipt, a checker script, and an audit note that keep the
  historical `TAS-156` closeout standing while stating machine-readably that
  the canonical post-`TAS-186` route is now the truth carrier for the bounded
  Turing-completeness claim and that control-plane ownership plus
  decision-provenance proof are part of that truth carrier; this tranche now
  binds that stronger terminal claim to the separately published canonical
  machine closure bundle from `TAS-215` and still does not imply weighted
  plugin control, plugin publication, served/public universality, or
  arbitrary software capability
- the repo now also has one post-article canonical machine closure bundle,
  with a committed eval report at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_machine_closure_bundle_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_canonical_machine_closure_bundle_summary.json`,
  a transformer-owned anchor contract, a provider-facing receipt, a dedicated
  checker script, and an audit note that publish one digest-bound closure
  object binding machine identity, the computational-model statement,
  determinism and equivalent-choice posture, control-plane provenance,
  execution-semantics proof transport, continuation boundary, carrier split,
  hidden-state and observer model, portability or minimality posture, plugin
  inheritance posture, and proof-versus-audit classification into one
  indivisible canonical machine subject; downstream stronger terminal,
  controller, receipt, publication, and bounded plugin-platform claims now
  inherit that bundle by digest
- the repo now also has one post-article starter-plugin catalog, with a
  runtime-owned bundle at
  `fixtures/tassadar/runs/tassadar_post_article_starter_plugin_catalog_v1/tassadar_post_article_starter_plugin_catalog_bundle.json`,
  a catalog report at
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_report.json`,
  an eval report at
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_eval_report.json`,
  a disclosure-safe research summary at
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_summary.json`,
  a provider-facing receipt, a dedicated checker, and an audit note that bind
  four operator-curated starter plugins plus two bounded composition flows to
  the canonical machine closure bundle while keeping the entire tranche
  operator-only, runtime-builtin separate, and explicitly not a public plugin
  marketplace
- the repo now also has one real runtime-owned `plugin.text.url_extract`
  starter plugin, with reusable execution and bundle-writing code in
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`,
  a committed runtime bundle at
  `fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1/tassadar_post_article_plugin_text_url_extract_bundle.json`,
  a dedicated checker at
  `scripts/check-tassadar-post-article-plugin-text-url-extract.sh`,
  a starter-runtime doc at `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md`, and an
  audit note at
  `docs/audits/2026-03-22-tassadar-post-article-plugin-text-url-extract.md`;
  this keeps the first starter runtime capability-free, deterministic, and
  explicit about refusal classes and negative claims while moving the next open
  bridge frontier to `TAS-218`
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
- `TAS-157` through `TAS-205` are now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-206` through `TAS-213` are now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-215` is now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-216` is now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-217` is now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-218` through `TAS-226` remain open in GitHub and are tracked via the
  issue bodies plus `docs/ROADMAP_TASSADAR_TAS_SYNC.md`

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

# Psionic System Spec

> Status: updated 2026-03-16 after reviewing `docs/MVP.md`,
> `docs/OWNERSHIP.md`, `README.md`,
> `docs/TRAIN_SYSTEM.md`,
> `docs/audits/2026-03-14-covenant-code-lessons-for-psionic-train-audit.md`,
> `docs/INFERENCE_ENGINE.md`,
> `crates/psionic-array/src/lib.rs`,
> `crates/psionic-runtime/src/lib.rs`,
> `crates/psionic-cluster/src/lib.rs`,
> `crates/psionic-datastream/src/lib.rs`,
> `crates/psionic-sandbox/src/lib.rs`,
> `crates/psionic-collectives/src/lib.rs`,
> `crates/psionic-train/src/lib.rs`, and
> `crates/psionic-adapters/src/lib.rs`,
> `crates/psionic-distributed/src/lib.rs`, plus the current open and
> recently closed issue backlog through `#3868`.

## Why This Doc Exists

Psionic already has enough surface area that a short layering note is no longer
sufficient.

This document is the canonical system spec for Psionic as a whole. It answers:

- what Psionic is
- what Psionic owns and does not own
- what is implemented now
- how the subtree is layered
- what kinds of work Psionic runs
- what artifact and receipt families Psionic should emit
- how Psionic execution flows end to end
- how failures and security are handled at the substrate level

This doc should be read together with:

- `docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md`
  - framework-core completion bar for tensor, compiler, IO, replay, and local
    multi-device behavior, distinct from serving or train product acceptance,
    with a machine-readable runner artifact defined by
    `docs/framework_core_acceptance_report.schema.json`
- `docs/TRAIN_SYSTEM.md`
  - deep subsystem spec for training-class execution
- `docs/INFERENCE_ENGINE.md`
  - narrower completion criteria for inference-engine behavior
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
  - inference-completion plan and issue program

The Psionic Train system builds on Psionic runtime, cluster, datastream,
sandbox, and collective layers defined in this document.

## Doc Authority

- `README.md` is the entrypoint and map.
- `docs/ARCHITECTURE.md` is the canonical Psionic-wide system
  spec.
- `docs/TRAIN_SYSTEM.md` is the canonical training subsystem
  spec.
- research audits explain why the system should move in a given direction, but
  they are not the authoritative current-state spec.

## Status Vocabulary

The status labels in Psionic docs use these meanings:

| Label | Meaning |
| --- | --- |
| `implemented` | landed and materially usable as a current substrate |
| `implemented_early` | landed, real, and usable, but still clearly early or incomplete |
| `partial` | some of the subsystem exists, but major required pieces are still missing |
| `partial_outside_psionic` | the broader OpenAgents stack has the authority or control surface, but Psionic does not yet own the native runtime or execution layer |
| `planned` | still a design target rather than a landed subsystem |

## Short Definition

Psionic is the Rust-native execution substrate for compute workloads inside
OpenAgents.

Psionic owns reusable substrate for:

- runtime execution
- backend capability and execution planning
- clustered topology and ordered state
- artifact staging and resumable transport
- runtime and environment manifest binding
- session-bound execution identity for networked lanes
- sandbox execution
- serving contracts
- training-class recovery and collective planning
- execution evidence and proof bundles

Psionic does not own:

- app UX
- wallet or payout flows
- buyer or provider product orchestration
- kernel authority or final market settlement

## What Psionic Owns

Psionic owns the machine-facing execution truth for compute lanes.

In practical terms that means:

- what artifacts were bound to execution
- what runtime or environment manifest package was actually used
- what transport or session identity claims were attached to execution
- what backend or topology ran the work
- what staged data was transferred and verified
- what proof posture or evidence was available
- what recovery or reconfiguration happened
- what receipts and execution metadata the rest of the system can consume

## What Psionic Does Not Own

Psionic is not the whole OpenAgents stack.

It must not own:

- pane-facing or desktop UX
- payout and wallet behavior
- marketplace procurement or settlement authority
- final collateral, claim, or adjudication authority
- app-owned control flows that belong in `apps/autopilot-desktop`

That boundary is intentional. Psionic explains what happened at execution time.
It does not decide what the market counts or what the product UI should do.

## Non-Goals

Psionic is also not:

- final market or settlement authority
- a home for app workflows
- a claim that every compute lane is mature today
- a hidden Python control plane behind Rust wrappers

## Tassadar Executor-Class Lane

Psionic now has an implemented-early executor-class reference lane codenamed
`Tassadar`.

The current scope is:

- owner: `crates/psionic-*`
- first target: WebAssembly-first executor semantics
- landed Phase 1 bar: CPU reference fixture plus exact parity harness
- landed Phase 2 bar: digest-bound program artifacts plus explicit
  model/program compatibility contracts
- landed Phase 3 bar: typed environment bundle plus package-driven exactness
  benchmark suite with CPU and reference-linear baselines
- landed Phase 4 bar: emitted trace artifacts, runtime-manifest lineage, and
  proof-bundle integration for replay-stable executor evidence
- landed Phase 5 bar: explicit `HullCache` fast-path decode identity, exact
  CPU/reference-linear/hull-cache equivalence checks on the validated acyclic
  subset, typed refusal for backward-branch workloads outside that subset, and
  benchmark reporting for hull-cache throughput, linear-decode speedup, and
  remaining direct-CPU gap
- landed Phase 6 bar: machine-legible runtime capability reports plus
  direct/fallback/refused decode selection diagnostics covering hull-cache,
  approximate sparse-top-k fallback, unsupported ABI/profile requests, and
  model-effective decode mismatches
- landed Phase 7A bar: explicit served `psionic.executor_trace` product
  semantics in `psionic-serve`, with typed request/response contracts,
  pull-driven trace streaming, final output extraction helpers, typed refusal
  responses, and served evidence bundles that preserve decode selection, trace
  proof, and runtime-manifest lineage
- landed article-session follow-on: `psionic-serve` now also exposes the
  specialized `psionic.article_executor_session` surface, which resolves
  canonical article workloads by case id, preserves benchmark/workload plus
  proof identity across the serving boundary, and emits derived readable-log
  and symbolic token-trace session views without collapsing back into ordinary
  tool use; the committed acceptance artifact is
  `fixtures/tassadar/reports/tassadar_article_executor_session_artifact.json`
- landed Phase 7B bar: widened `core_i32_v2` Wasm profile, profile-aware
  runner construction, and article-class benchmark coverage for
  `MicroWasmKernel`, `BranchHeavyKernel`, `MemoryHeavyKernel`,
  `LongLoopKernel`, `SudokuClass`, and `HungarianMatching` with exact
  CPU/reference-linear/hull-cache parity plus published speedup and CPU-gap
  metrics in
  `fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json`;
  the same report keeps branch-heavy sparse fallback and long-loop
  hull/sparse fallback explicit instead of overstating fast-path closure
- landed Phase 7C bar: one explicit long-horizon trace ABI/versioning decision
  now exists in `psionic-runtime`, with the canonical report
  `fixtures/tassadar/reports/tassadar_trace_abi_decision_report.json` and the
  committed long-loop evidence bundle
  `fixtures/tassadar/runs/long_loop_kernel_trace_abi_v0/execution_evidence_bundle.json`;
  machine validators now have frozen ABI pointers across benchmark, compiled,
  and long-horizon artifacts, while readable logs remain explicitly
  non-authoritative views over the append-only trace artifact
- landed Phase 7D bar: one machine-readable Tassadar workload capability matrix
  now exists in `psionic-eval` at
  `fixtures/tassadar/reports/tassadar_workload_capability_matrix.json`,
  keeping runtime exact, runtime fallback-only, compiled exact, bounded
  learned, and partial learned-long-horizon posture separate per workload
  family instead of collapsing them into one green summary; `psionic-models`
  now also publishes the served-lane workload matrix schema plus typed refusal
  reasons, `psionic-serve` now exposes a benchmark-gated
  `TassadarExecutorCapabilityPublication` above that schema, and
  `psionic-provider` now wraps the same publication into a provider-facing
  `TassadarCapabilityEnvelope` instead of inventing a second capability story
- landed workload-hardness taxonomy and capability-frontier follow-on:
  `psionic-models` now publishes a public
  `TassadarWorkloadHardnessTaxonomyPublication` that names retrieval-like,
  parallelizable, search-heavy, memory-heavy, and control-heavy workload
  families plus explicit depth/width/recurrent/extra-trace-space budgets;
  `psionic-eval` now freezes the current frontier map at
  `fixtures/tassadar/reports/tassadar_workload_capability_frontier_report.json`
  by joining the workload matrix, module-scale Wasm suite, CLRS bridge,
  verifier-guided search, recurrent fast-path, and shared-depth artifacts
  instead of inventing a second benchmark universe; `psionic-research` now
  freezes the companion summary at
  `fixtures/tassadar/reports/tassadar_workload_capability_frontier_summary.json`;
  and `psionic-provider` now projects that research summary into a
  provider-facing `TassadarWorkloadCapabilityFrontierReceipt`. The frontier
  keeps preferred-lane recommendations, refusal-first regions, and under-mapped
  families explicit instead of widening any served capability claim
- landed finite-precision and attention-semantics robustness follow-on:
  `psionic-models` now publishes a public
  `TassadarPrecisionAttentionAuditPublication` over explicit `fp32`, `fp16`,
  `int8`, and noisy-`int8` numeric regimes plus hard-selection, sparse, and
  soft-proxy attention families; `psionic-runtime` now freezes the
  deterministic receipt-level audit at
  `fixtures/tassadar/reports/tassadar_precision_attention_runtime_audit.json`
  with explicit exact, approximate-bounded, and refused drift classes across
  the current six workload families; `psionic-eval` now joins that runtime
  audit to the efficient-attention baseline matrix in
  `fixtures/tassadar/reports/tassadar_precision_attention_robustness_audit.json`;
  and `psionic-research` now freezes the companion summary at
  `fixtures/tassadar/reports/tassadar_precision_attention_robustness_summary.json`.
  This lane keeps fragile workloads, refusal hotspots, and regime-specific
  degradation explicit instead of treating lower precision or proxy attention
  survival as proof of deployment-robust exactness
- landed quantization-truth-envelope follow-on: `psionic-runtime` now projects
  backend-specific deployment envelopes from the finite-precision audit lane in
  `fixtures/tassadar/reports/tassadar_quantization_truth_envelope_runtime_report.json`,
  keyed by backend family, numeric regime, and quantization setting with exact,
  constrained, and refused workload posture kept explicit; `psionic-models` now
  publishes the public `TassadarQuantizationTruthEnvelopePublication` above
  that runtime-owned truth; `psionic-serve` now carries the active backend
  family and deployment envelopes through the executor capability publication;
  `psionic-provider` now validates and projects the same surface as a provider
  deployment-truth envelope; and `psionic-eval` now freezes the joined summary
  at
  `fixtures/tassadar/reports/tassadar_quantization_truth_envelope_eval_report.json`.
  This lane keeps backend and quantization drift explicit instead of assuming
  one executor artifact preserves semantics across export, quantization, and
  served backend changes
- landed approximate-attention-closure follow-on: `psionic-runtime` now
  projects a same-workload closure report from the shared efficient-attention
  matrix in
  `fixtures/tassadar/reports/tassadar_approximate_attention_closure_runtime_report.json`,
  covering dense, sparse-top-k, linear recurrent, LSH-style proxy, hard-max
  proxy, HullCache, and hierarchical-hull families with explicit `direct`,
  `degraded_but_bounded`, and `refused` posture per workload row; `psionic-models`
  now publishes the public `TassadarApproximateAttentionClosurePublication`
  above that runtime truth; `psionic-eval` now freezes the machine-legible
  matrix at
  `fixtures/tassadar/reports/tassadar_approximate_attention_closure_matrix.json`;
  and `psionic-research` now freezes the promotion-boundary summary at
  `fixtures/tassadar/reports/tassadar_approximate_attention_closure_summary.json`.
  This lane keeps approximate-attention wins, degradations, and refusal
  hotspots explicit instead of treating one fast-path success as general
  executor closure
- landed state-design-study follow-on: `psionic-ir` now owns a public
  state-design contract over `full_append_only_trace`, `delta_trace`,
  `locality_scratchpad`, `recurrent_state`, and `working_memory_tier` with
  explicit exact-replay, reconstructable-replay, bounded-state-publication,
  and refusal posture; `psionic-data` now publishes the same-workload study
  contract for module-call trace, symbolic-locality, associative-recall,
  long-horizon control, and byte-memory loop workloads; `psionic-models` now
  publishes the research-only `TassadarStateDesignStudyPublication`;
  `psionic-runtime` now freezes the case matrix at
  `fixtures/tassadar/reports/tassadar_state_design_runtime_report.json`; and
  `psionic-eval` now freezes the joined workload-versus-family study report at
  `fixtures/tassadar/reports/tassadar_state_design_study_report.json`. This
  lane keeps replay floors, reconstructable deltas, bounded semantic-state
  wins, and explicit refusal thresholds separate instead of treating the
  append-only trace as the uniquely honest representation for every workload
  family
- landed error-regime-catalog follow-on: `psionic-runtime` now publishes
  bounded injected-error receipts over `uncorrected`, `checkpoint_only`,
  `verifier_only`, and `checkpoint_and_verifier` recovery surfaces for the
  seeded Sudoku-backtracking, search-kernel, long-horizon-control, and
  byte-memory-loop families with explicit `self_healing`, `slow_drift`, and
  `catastrophic_divergence` outcomes; `psionic-data` now publishes the public
  `TassadarErrorRegimeCatalogContract`; `psionic-train` now freezes the
  committed sweep artifact at
  `fixtures/tassadar/runs/tassadar_error_regime_catalog_v1/error_regime_sweep_report.json`;
  `psionic-eval` now freezes the workload-versus-recovery catalog at
  `fixtures/tassadar/reports/tassadar_error_regime_catalog.json`; and
  `psionic-research` now freezes the dominant-surface summary at
  `fixtures/tassadar/reports/tassadar_error_regime_summary.json`. This lane
  keeps self-healing, slow drift, catastrophic divergence, and explicit
  checkpoint-versus-verifier tradeoffs separate instead of treating the mere
  existence of a correction path as proof of exactness
- landed weak-supervision-executor follow-on: `psionic-data` now publishes a
  public `TassadarWeakSupervisionContract` over explicit `full_trace`,
  `mixed_weak`, and `io_only` supervision regimes plus machine-legible
  `io_targets`, `invariants`, `partial_state`, and `subroutine_labels` signal
  components for the seeded module-trace-v2, Hungarian-module,
  verifier-search-kernel, and module-state-control families; `psionic-models`
  now publishes the research-only `TassadarWeakSupervisionPublication` above
  that contract; `psionic-train` now freezes the committed evidence bundle at
  `fixtures/tassadar/runs/tassadar_weak_supervision_executor_v1/weak_supervision_evidence_bundle.json`
  with later-window exactness, final-output exactness, refusal calibration, and
  under-supervised failure counts kept explicit per workload/regime cell;
  `psionic-eval` now freezes the joined regime-versus-workload report at
  `fixtures/tassadar/reports/tassadar_weak_supervision_executor_report.json`;
  and `psionic-research` now freezes the companion summary at
  `fixtures/tassadar/reports/tassadar_weak_supervision_executor_summary.json`.
  This lane keeps mixed-supervision viability, io-only fragility, and
  full-trace-only outliers explicit instead of overstating weaker supervision as
  broad learned module-execution closure
- landed kernel-module-scaling follow-on: `psionic-data` now publishes a public
  `TassadarKernelModuleScalingContract` over explicit kernel-scale,
  bridge-scale, and module-scale families plus declared
  `call_graph_width`, `control_flow_depth`, `trace_length`,
  `memory_footprint`, and `import_complexity` pressure labels; `psionic-eval`
  now freezes the joined scaling report at
  `fixtures/tassadar/reports/tassadar_kernel_module_scaling_report.json` by
  reusing the committed compiled-kernel scaling artifact, CLRS bridge report,
  module-scale workload-suite report, and Wasm host-import refusal boundary
  instead of inventing a separate benchmark universe; `psionic-research` now
  freezes the companion summary at
  `fixtures/tassadar/reports/tassadar_kernel_module_scaling_summary.json`; and
  `psionic-provider` now projects that summary into a provider-facing
  `TassadarKernelModuleScalingReceipt`. This lane keeps exact ceilings,
  exact-but-cost-degraded regions, and refusal boundaries explicit instead of
  inferring module-scale closure from kernel-scale wins or smoothing distinct
  workload families into one generic scaling curve
- landed shared primitive transfer follow-on: `psionic-data` now publishes a
  public `TassadarSharedPrimitiveTransferContract` over explicit
  reachability-expand, relax-state, compare, select, merge, and
  bounded-backtrack primitives with declared sort/merge, shortest-path,
  CLRS-to-Wasm, Hungarian, Sudoku, and verifier-search algorithm families plus
  separate compiled and learned anchor refs; `psionic-models` now publishes the
  research-only `TassadarSharedPrimitiveTransferPublication` above that
  contract; `psionic-train` now freezes the held-out transfer evidence bundle at
  `fixtures/tassadar/runs/tassadar_shared_primitive_transfer_v1/shared_primitive_transfer_evidence_bundle.json`
  with explicit primitive-reuse versus final-task-exactness receipts and
  primitive ablations; `psionic-eval` now freezes the joined report at
  `fixtures/tassadar/reports/tassadar_shared_primitive_transfer_report.json`;
  and `psionic-research` now freezes the companion summary at
  `fixtures/tassadar/reports/tassadar_shared_primitive_transfer_summary.json`.
  The lane keeps foundational primitives, primitive-layer bottlenecks, and
  composition bottlenecks explicit instead of overstating shared primitive reuse
  as broad executor closure
- landed compiled-distillation follow-on: `psionic-data` now publishes a
  public `TassadarCompiledDistillationContract` over explicit `full_trace`,
  `io_only`, `partial_state`, `invariance_class`, and `mixed_distillation`
  supervision regimes plus bounded kernel, CLRS-to-Wasm, Hungarian, and Sudoku
  workload families; `psionic-runtime` now freezes the compiled/reference
  authority targets at
  `fixtures/tassadar/runs/tassadar_compiled_distillation_targets_v1/compiled_distillation_target_bundle.json`;
  `psionic-train` now freezes the lighter-supervision evidence bundle at
  `fixtures/tassadar/runs/tassadar_compiled_distillation_v1/compiled_distillation_training_evidence_bundle.json`
  with explicit later-window, held-out-family, invariance-ablation, and
  refusal facts; `psionic-eval` now freezes the joined comparison report at
  `fixtures/tassadar/reports/tassadar_compiled_distillation_report.json`; and
  `psionic-research` now freezes the companion summary at
  `fixtures/tassadar/reports/tassadar_compiled_distillation_summary.json`.
  This lane keeps weaker supervision, mixed-distillation rescue, and
  full-trace dependency explicit instead of overstating lighter supervision as
  broad learned executor closure
- landed working-memory-tier follow-on: `psionic-models` now publishes a
  public research-only `TassadarWorkingMemoryTierPublication` for one bounded
  Psionic-owned memory tier with explicit slot reads, writes, associative
  lookup, and state-publication semantics over copy-window, stable-sort,
  associative-recall, and long-carry accumulator kernels; `psionic-runtime`
  now freezes the bounded comparison report at
  `fixtures/tassadar/reports/tassadar_working_memory_tier_runtime_report.json`
  with pure-trace versus working-memory metrics, state-publication receipts,
  trace-shaping-only cases, and explicit refusal boundaries; `psionic-eval`
  now freezes the classifier report at
  `fixtures/tassadar/reports/tassadar_working_memory_tier_eval_report.json`;
  and `psionic-research` now freezes the companion summary at
  `fixtures/tassadar/reports/tassadar_working_memory_tier_summary.json`. This
  lane keeps bounded widening, trace-shaping-only behavior, and overflow
  refusal explicit instead of treating one memory tier as arbitrary-memory
  closure or a license to hide external tool semantics
- landed internal-module-library follow-on: `psionic-compiler` now publishes a
  bounded `TassadarInternalModuleLibrary` with versioned module artifacts, link
  manifests, compatibility digests, and explicit replacement or rollback
  plans; `psionic-models` now publishes the public
  `TassadarInternalModuleLibraryPublication`; `psionic-runtime` now freezes the
  benchmark-bound link report at
  `fixtures/tassadar/reports/tassadar_internal_module_library_report.json`
  with exact reuse, rollback, and refusal cases across CLRS, Hungarian, and
  verifier-search families; `psionic-serve` now publishes a benchmark-gated
  served module-library surface above that runtime report; `psionic-provider`
  now wraps that served surface in a provider-facing
  `TassadarInternalModuleLibraryReceipt`; and `psionic-research` now freezes
  the companion summary at
  `fixtures/tassadar/reports/tassadar_internal_module_library_summary.json`.
  This lane keeps reuse, rollback, and refusal explicit instead of treating a
  versioned module library as unrestricted self-extension or arbitrary install
  closure
- landed module-manifest follow-on: `psionic-ir` now owns a public
  `TassadarComputationalModuleManifest` ABI and manifest schema with typed
  imports, exports, state fields, claim class, trust posture, capability
  summary, benchmark lineage, and required evidence refs; `psionic-compiler`
  now validates those manifests and publishes deterministic compatibility
  receipts instead of letting module identity silently widen capability; and
  `psionic-provider` now projects the same manifest into a provider-facing
  `TassadarModuleManifestReceipt`. This lane stays bounded to manifest and
  compatibility truth only; later linker, catalog, and install flows consume
  this schema instead of inventing a second module story, and named
  `kernel-objects` follow-ons remain explicit dependency markers rather than
  backfilled behavior inside standalone `psionic`
- landed module-linker follow-on: `psionic-compiler` now resolves bounded
  module-link requests into deterministic dependency graphs with explicit
  version-conflict refusal, rollback selection, trust and claim-class
  compatibility checks, and internal-module import resolution; `psionic-runtime`
  now freezes the linked-program runtime witness at
  `fixtures/tassadar/reports/tassadar_module_link_runtime_report.json` with
  exact, rollback, and refused cases plus preserved dependency edges and parity
  facts; `psionic-eval` now freezes the joined summary at
  `fixtures/tassadar/reports/tassadar_module_link_eval_report.json`; and
  `psionic-provider` now projects that runtime truth into a provider-facing
  `TassadarModuleLinkReceipt`. This lane keeps dependency graphs, rollback
  paths, and refusal posture machine-legible instead of hiding module
  composition behind heuristic resolution or silent semantic drift
- landed installed-module-evidence follow-on: `psionic-runtime` now publishes a
  bounded `TassadarInstalledModuleEvidenceBundle` at
  `fixtures/tassadar/runs/tassadar_installed_module_evidence_v1/installed_module_evidence_bundle.json`
  that joins compile lineage, benchmark refs, audit or decompilation refs,
  refusal posture, revocation hooks, and reinstall parity under one receipt
  family; `psionic-eval` now freezes the joined summary at
  `fixtures/tassadar/reports/tassadar_installed_module_evidence_report.json`;
  and `psionic-provider` now projects the same runtime truth into a
  provider-facing `TassadarInstalledModuleEvidenceReceipt`. This lane keeps
  missing-evidence refusal, stale-evidence refusal, revocation readiness, and
  reinstall parity explicit instead of treating installation as implicit trust,
  while named `nexus` follow-ons remain explicit dependency markers outside
  standalone `psionic`
- landed module-catalog follow-on: `psionic-compiler` now freezes the bounded
  reusable primitive catalog at
  `fixtures/tassadar/reports/tassadar_module_catalog_report.json`, keyed by
  capability label, workload family, trust posture, benchmark lineage, reuse
  rate, and held-out-program lift; `psionic-serve` now publishes a
  benchmark-gated `TassadarModuleCatalogPublication` above that report;
  `psionic-router` now resolves bounded catalog lookups and keeps ambiguity plus
  insufficient-evidence refusal explicit; and `psionic-provider` now projects
  the same served truth into a provider-facing `TassadarModuleCatalogReceipt`.
  This lane keeps reusable primitive discovery measurable rather than anecdotal
  and does not widen served capability by implication from catalog membership
- landed overlap-resolution follow-on: `psionic-compiler` now publishes
  deterministic `TassadarModuleResolverPolicy` and overlap-candidate resolution
  semantics keyed by trust, evidence, compatibility, cost, and explicit
  preference ordering; `psionic-router` now freezes the mount-aware A/B report
  at `fixtures/tassadar/reports/tassadar_module_overlap_resolution_report.json`
  with default selection, mount-specific override, and ambiguity refusal kept
  explicit; and `psionic-provider` now projects that report into a
  provider-facing `TassadarModuleOverlapResolutionReceipt`. This lane keeps
  overlapping capability resolution honest instead of hiding selector drift
  behind defaults, while named `world-mounts` follow-ons remain explicit
  dependency markers outside standalone `psionic`
- landed import-policy follow-on: `psionic-sandbox` now publishes the typed
  host-call policy matrix at
  `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`,
  separating deterministic internal stubs, sandbox-only delegation, and refused
  side effects with explicit descriptor, challenge-receipt, and refusal paths;
  and `psionic-provider` now projects that report into a provider-facing
  `TassadarImportPolicyMatrixReceipt`. This lane keeps internal execution
  distinct from external delegation instead of letting import availability hide
  a boundary crossing, while named `kernel-policy` and `world-mounts`
  follow-ons remain explicit dependency markers outside standalone `psionic`
- landed trust-isolation follow-on: `psionic-runtime` now publishes the
  trust-tier isolation report at
  `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`,
  freezing research-contained, benchmark-internal, and challenge-gated module
  bundles plus explicit cross-tier, privilege-escalation, and mount-policy
  refusals; and `psionic-provider` now projects that report into a
  provider-facing `TassadarModuleTrustIsolationReceipt`. This lane keeps trust
  posture separate from benchmark count and peer composition instead of letting
  internal modules silently inherit more authority, while named `cluster-trust`,
  `kernel-policy`, and `world-mounts` follow-ons remain explicit dependency
  markers outside standalone `psionic`
- landed promotion-lifecycle follow-on: `psionic-eval` now publishes the
  promotion lifecycle report at
  `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`,
  freezing minimum evidence requirements plus challenge-open, quarantined,
  revoked, and superseded module states without rewriting install history; and
  `psionic-provider` now projects that report into a provider-facing
  `TassadarModulePromotionStateReceipt`. This lane keeps post-promotion
  challengeability explicit instead of treating one benchmark pass as permanent
  closure, while named `nexus` and `kernel-policy` follow-ons remain explicit
  dependency markers outside standalone `psionic`
- landed disclosure-flow follow-on: repo governance now owns a public
  decomposition checklist in `docs/TASSADAR_PUBLIC_DISCLOSURE_FLOW.md`, a
  machine-readable review artifact at
  `fixtures/tassadar/reports/tassadar_public_disclosure_release_review.json`,
  and a runnable checker at `scripts/check-tassadar-public-disclosure.sh`. This
  lane keeps private naming, private product framing, and over-broad public
  claims from leaking into `psionic`, and requires explicit refusal when
  private-only language survives review
- landed self-installation-gate follow-on: `psionic-router` now publishes the
  bounded self-installation gate report at
  `fixtures/tassadar/reports/tassadar_self_installation_gate_report.json`,
  freezing session-mount approval, worker-mount challenge windows, rollback
  under failed post-install benchmarks, and explicit policy denial for blocked
  proposals; and `psionic-provider` now projects that report into a
  provider-facing `TassadarSelfInstallationGateReceipt`. This lane keeps
  self-extension proposal review inspectable and policy-bound instead of
  implying unrestricted self-modification, while named `kernel-policy` and
  `nexus` follow-ons remain explicit dependency markers outside standalone
  `psionic`
- landed execution-unit-registration follow-on: `psionic-serve` now publishes
  the executor-family registration report at
  `fixtures/tassadar/reports/tassadar_execution_unit_registration_report.json`,
  freezing execution-unit identity, topology compatibility, capability profile,
  refusal taxonomy, benchmark lineage, and indicative pricing posture for the
  served executor family; and `psionic-provider` now projects that report into a
  provider-facing `TassadarExecutionUnitRegistrationReceipt`. This lane keeps
  execution-unit registration distinct from accepted-outcome truth or market
  eligibility instead of letting runtime success imply settlement posture
- landed world-mount-compatibility follow-on: `psionic-router` now publishes
  the world-mount compatibility report at
  `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`,
  freezing mount-time negotiation across trust posture, import posture,
  benchmark evidence, module dependencies, and validator bindings with explicit
  allow, deny, and unresolved outcomes; and `psionic-provider` now projects
  that report into a provider-facing `TassadarWorldMountCompatibilityReceipt`.
  This lane keeps mount compatibility distinct from accepted-outcome or
  settlement transitions instead of hiding it inside route heuristics
- landed accepted-outcome-binding follow-on: `psionic-provider` now publishes
  the accepted-outcome binding report at
  `fixtures/tassadar/reports/tassadar_accepted_outcome_binding_report.json`,
  freezing exact-compute accepted-outcome templates over explicit provider
  execution-unit receipts, world-mount compatibility receipts, evidence-routing
  receipts, validator policy bindings, challenge windows, and settlement
  dependencies with distinct candidate-outcome, accepted-outcome, and
  provider-receipt identities. This lane keeps runtime success separate from
  accepted-outcome closure or settlement-qualified authority instead of
  letting provider receipts imply either one
- landed composite-routing follow-on: `psionic-router` now publishes the
  heterogeneous composite-routing report at
  `fixtures/tassadar/reports/tassadar_composite_routing_report.json`, freezing
  seeded multi-lane route artifacts across planner, module, CPU-reference, GPU,
  validator, sandbox, and cluster steps with explicit fallback and
  challenge-path handling plus direct comparisons against single-lane
  baselines; and `psionic-provider` now projects that report into a
  provider-facing `TassadarCompositeRoutingReceipt`. This lane keeps composite
  route identity, evidence lift, and route economics explicit instead of
  treating `uses more than one lane` as success by itself
- landed exact-compute-market follow-on: `psionic-provider` now publishes the
  exact-compute market report at
  `fixtures/tassadar/reports/tassadar_exact_compute_market_report.json`,
  freezing a machine-legible exact-compute product family over explicit
  execution-unit identity, accepted-outcome binding dependencies, composite
  route posture, quote posture, pricing posture, and receipt-economic bridge
  objects with separate unsupported-envelope refusals. This lane keeps product
  identity, quote identity, receipt identity, and settlement posture explicit
  instead of flattening executor work into raw hardware or token-count pricing
- landed composite-accepted-outcome-template follow-on: `psionic-provider` now
  publishes the hybrid accepted-outcome template report at
  `fixtures/tassadar/reports/tassadar_composite_accepted_outcome_template_report.json`,
  freezing composite templates over planner, exact-compute, CPU-reference,
  validator, sandbox, GPU, and cluster lane obligations with explicit
  per-lane evidence refs, stitched provider receipts, candidate-versus-
  accepted outcome separation, missing-evidence refusals, missing-receipt-
  stitch refusals, and settlement-gated refusals. This lane keeps hybrid
  execution truth, accepted-outcome truth, and settlement posture bridged but
  distinct instead of collapsing them into one composite success bit
- landed counterfactual-route-quality follow-on: `psionic-router` now publishes
  the same-task counterfactual route-quality report at
  `fixtures/tassadar/reports/tassadar_counterfactual_route_quality_report.json`,
  comparing realized language-only, compiled-exact, internal-executor,
  external-tool, and hybrid-composite lane choices against the best available
  alternative on the same tasks with explicit accepted-outcome, evidence,
  cost, and latency deltas; `psionic-eval` now publishes the derived summary at
  `fixtures/tassadar/reports/tassadar_counterfactual_route_quality_eval_report.json`;
  and `psionic-provider` now projects that report into a provider-facing
  `TassadarCounterfactualRouteQualityReceipt`. This lane keeps realized-route
  success separate from best-available-route quality so planner training can
  consume honest overuse and underuse evidence without implying accepted-
  outcome or settlement authority
- landed cost-per-correct-job follow-on: `psionic-eval` now publishes the
  benchmark-bound economics report at
  `fixtures/tassadar/reports/tassadar_cost_per_correct_job_report.json`,
  freezing matched workload families across internal exact-compute, hybrid
  composite, and external delegation lanes with explicit refusal rates,
  evidence completeness, cost-per-correct-job values, and route-threshold
  crossings; and `psionic-provider` now projects that report into a
  provider-facing `TassadarCostPerCorrectJobReceipt`. This lane keeps
  benchmark economics explicit as refusal-bounded product research instead of
  treating them as market pricing truth by themselves
- landed wedge-taxonomy follow-on: `psionic-environments` now publishes the
  property-first taxonomy suite for deterministic structure, verifier cost,
  challenge cost, evidence value, and exact intermediate-compute benefit;
  `psionic-eval` now publishes the derived-suite report at
  `fixtures/tassadar/reports/tassadar_wedge_taxonomy_report.json`; and
  `psionic-provider` now projects that report into a provider-facing
  `TassadarWedgeTaxonomyReceipt`. This lane keeps domain selection grounded in
  workload properties, validator attachment, evidence completeness, and
  fallback-versus-internal comparisons instead of treating one domain demo as a
  full wedge claim
- landed latency-evidence-tradeoff follow-on: `psionic-router` now publishes
  the validator-heavy workload pack and threshold publication for compiled,
  learned, external, and hybrid lanes; and `psionic-eval` now publishes the
  benchmark-bound Pareto report at
  `fixtures/tassadar/reports/tassadar_latency_evidence_tradeoff_report.json`,
  freezing latency/correctness/evidence fronts, validator cost, challenge rate,
  and threshold-crossing cases without treating low latency as a win once
  evidence posture becomes invalid
- landed trace-state-ablation follow-on: `psionic-data` now publishes the fixed
  trace/state ablation canon across CLRS, arithmetic, Sudoku-search, and
  module-scale Wasm workloads; `psionic-models` now publishes the corresponding
  research-only ablation publication; `psionic-eval` now publishes the report
  at `fixtures/tassadar/reports/tassadar_trace_state_ablation_report.json`; and
  `psionic-research` now publishes the summary at
  `fixtures/tassadar/reports/tassadar_trace_state_ablation_summary.json`. This
  lane keeps representation-limited failures explicit instead of treating one
  state surface as privileged by default
- landed supervision-density follow-on: `psionic-data` now publishes the shared
  supervision-density canon, `psionic-train` now publishes the corresponding
  evidence bundle, `psionic-eval` now publishes the report at
  `fixtures/tassadar/reports/tassadar_supervision_density_report.json`, and
  `psionic-research` now publishes the summary at
  `fixtures/tassadar/reports/tassadar_supervision_density_summary.json`. This
  lane keeps supervision-limited and architecture-limited failures explicit
  across CLRS, arithmetic, search, and module-scale Wasm families
- landed locality-envelope follow-on: `psionic-runtime` now publishes the
  runtime report at
  `fixtures/tassadar/reports/tassadar_locality_envelope_runtime_report.json`
  with explicit max useful lookback, active memory footprint, branch
  locality, call-frame locality, exactness, cost, downgrade, and refusal
  posture across dense, sparse, linear, recurrent, and scratchpadized
  variants; `psionic-models` now publishes the corresponding
  `TassadarLocalityEnvelopePublication`; `psionic-eval` now publishes the
  report at `fixtures/tassadar/reports/tassadar_locality_envelope_report.json`;
  and `psionic-research` now publishes the summary at
  `fixtures/tassadar/reports/tassadar_locality_envelope_summary.json`. This
  lane keeps locality wins, downgrade boundaries, and refusal-first workload
  families explicit instead of treating one fast-path or state design as a
  generic long-context answer
- landed trap-exception-semantics follow-on: `psionic-compiler` now publishes
  the public `TassadarTrapExceptionContract` over success, bounds-fault,
  indirect-call-failure, malformed-import, and unsupported-profile cases;
  `psionic-runtime` now publishes the runtime report at
  `fixtures/tassadar/reports/tassadar_trap_exception_runtime_report.json`
  with explicit success, trap, and refusal parity; `psionic-eval` now
  publishes the joined audit report at
  `fixtures/tassadar/reports/tassadar_trap_exception_report.json`; and
  `psionic-provider` now projects that report into a provider-facing
  `TassadarTrapExceptionReceipt`. This lane keeps trap-state parity and
  refusal-state parity as first-class execution truth instead of letting
  successful exactness stand in for failure-path closure
- landed exception-profile follow-on: `psionic-ir` now publishes the public
  `TassadarExceptionProfileContract` naming the bounded
  `tassadar.proposal_profile.exceptions_try_catch_rethrow.v1` surface on the
  current-host cpu-reference envelope; `psionic-compiler` now publishes the
  companion `TassadarExceptionProfileCompilationContract` over typed
  throw/catch success, nested rethrow trap-stack parity, handler-tag mismatch
  trap parity, and malformed-handler refusal; `psionic-runtime` now publishes
  the runtime report at
  `fixtures/tassadar/reports/tassadar_exception_profile_runtime_report.json`;
  `psionic-eval` now publishes the joined report at
  `fixtures/tassadar/reports/tassadar_exception_profile_report.json`;
  `psionic-serve` now carries that report ref plus public/default-served and
  portability ids through `TassadarExecutorCapabilityPublication`; and
  `psionic-provider` now projects the same report into a
  `TassadarExceptionProfileReceipt`. This lane keeps typed throw/catch/rethrow
  parity, trap-stack parity, and the zero-default-served boundary explicit
  instead of silently widening exceptions into generic Wasm support
- landed architecture-bakeoff follow-on: `psionic-models` now publishes the
  public `TassadarArchitectureBakeoffPublication` over flat decoder trace,
  shared-depth recurrent refinement, linear or recurrentized attention,
  memory-augmented, pointer, and search-native executor families;
  `psionic-train` now freezes the same-task same-budget bundle at
  `fixtures/tassadar/runs/tassadar_architecture_bakeoff_v1/architecture_bakeoff_budget_bundle.json`;
  `psionic-eval` now publishes the matrix report at
  `fixtures/tassadar/reports/tassadar_architecture_bakeoff_report.json`; and
  `psionic-research` now publishes the summary at
  `fixtures/tassadar/reports/tassadar_architecture_bakeoff_summary.json`.
  This lane keeps workload-family ownership, cost, stability, and refusal
  posture explicit instead of treating whichever learned architecture landed
  first as the default frontier owner
- landed pointer-memory-scratchpad separation follow-on: `psionic-models` now
  publishes the public `TassadarPointerMemoryScratchpadPublication`;
  `psionic-train` now freezes the same-budget ablation bundle at
  `fixtures/tassadar/runs/tassadar_pointer_memory_scratchpad_study_v1/pointer_memory_scratchpad_ablation_bundle.json`;
  `psionic-runtime` now publishes the runtime report at
  `fixtures/tassadar/reports/tassadar_pointer_memory_scratchpad_runtime_report.json`;
  `psionic-eval` now publishes the joined report at
  `fixtures/tassadar/reports/tassadar_pointer_memory_scratchpad_report.json`;
  and `psionic-research` now publishes the summary at
  `fixtures/tassadar/reports/tassadar_pointer_memory_scratchpad_summary.json`.
  This lane keeps address-selection limits, mutable-memory limits, and
  representation limits explicit across graph, arithmetic, search, and
  module-scale Wasm workloads instead of flattening them into one blended
  learned-executor failure mode
- landed search-native executor follow-on: `psionic-data` now publishes the
  public `TassadarSearchNativeExecutorContract` over explicit `guess`,
  `verify`, `contradict`, `backtrack`, `branch_summary`, and `search_budget`
  event families plus seeded Sudoku, branch-heavy CLRS, search-kernel, and
  verifier-heavy workload rows; `psionic-models` now publishes the companion
  `TassadarSearchNativeExecutorPublication`; `psionic-train` now freezes the
  same-budget evidence bundle at
  `fixtures/tassadar/runs/tassadar_search_native_executor_v1/search_native_executor_evidence_bundle.json`;
  `psionic-runtime` now publishes the runtime report at
  `fixtures/tassadar/reports/tassadar_search_native_executor_runtime_report.json`;
  and `psionic-eval` now publishes the comparison report at
  `fixtures/tassadar/reports/tassadar_search_native_executor_report.json`.
  This lane keeps search-native wins, guess and recovery metrics, and
  search-budget refusal posture explicit instead of flattening combinatorial
  workloads into one straight-trace executor story
- landed module-installation-staging follow-on: `psionic-serve` now publishes a
  bounded staged-install surface with explicit session-mount versus worker-mount
  scope, challenge windows, activation, rollback, and refusal receipts over the
  current trusted module classes; `psionic-provider` now projects those served
  install receipts provider-side; `psionic-router` now owns policy-bound module
  install route negotiation with explicit challenge-ticket, scope, unsafe-class,
  and missing-benchmark refusal reasons; `psionic-eval` now freezes the staged
  drill report at
  `fixtures/tassadar/reports/tassadar_module_installation_staging_report.json`;
  and `psionic-research` now freezes the companion summary at
  `fixtures/tassadar/reports/tassadar_module_installation_staging_summary.json`.
  This lane keeps install, challenge, rollback, and refusal explicit instead of
  treating bounded installation as unrestricted self-modification
- landed route-contract follow-on: `psionic-router` now also owns a routeable
  Wasm capability matrix for `psionic.planner_executor_route`, with explicit
  module-class rows, opcode-family facts, import posture, module-class-specific
  direct-vs-fallback decode truth, and typed refusal reasons for unsupported
  module classes, opcode families, or import posture; `psionic-serve` now
  derives that matrix directly from the served workload capability publication,
  and `psionic-provider` now validates that served route descriptors keep the
  routeable Wasm rows benchmark-gated instead of collapsing everything into one
  coarse "supports Wasm" claim
- public Wasm claim discipline is therefore: "supports Wasm" means a named
  Tassadar profile or profile family inside a frozen WebAssembly spec window,
  with imports, effects, routing, and host capability policy kept separate
  from the core language claim; any future broader or full-Wasm closure should
  also be tied to the official test corpus and reference authority for that
  frozen window instead of an open-ended feature list
- the current declared frozen-window artifact is
  `fixtures/tassadar/reports/tassadar_frozen_core_wasm_window_report.json`,
  which freezes the text, binary decode, binary encode, validation, and
  reference-execution authorities for the current int-first core-Wasm target
- the current closure-verdict artifact is
  `fixtures/tassadar/reports/tassadar_frozen_core_wasm_closure_gate_report.json`,
  which currently remains `not_closed`; the declared window therefore stays a
  closure target rather than a closed full core-Wasm claim
- the current bounded float-widening artifact is
  `fixtures/tassadar/reports/tassadar_float_semantics_comparison_matrix_report.json`,
  which narrows float semantics to one scalar-`f32`, CPU-reference-only lane
  with canonical quiet-NaN normalization and ordered Wasm-style comparisons;
  `f64`, NaN-payload preservation, and non-CPU fast-math remain explicit refusals
- the current mixed-numeric widening artifact is
  `fixtures/tassadar/reports/tassadar_mixed_numeric_profile_ladder_report.json`,
  which stages exact scalar-`f32`, exact mixed `i32`/`f32`, and
  bounded-approximate `f64 -> f32` conversion profiles with typed refusal on
  malformed, out-of-range, and non-exact conversion regimes; this is not
  arbitrary Wasm numeric closure or full `f64` exactness
- the current numeric-portability artifact is
  `fixtures/tassadar/reports/tassadar_numeric_portability_report.json`, which
  freezes backend-, toolchain-, and machine-class envelopes for those bounded
  float and mixed-numeric profiles, publishing exact cpu-reference rows only
  for the exact `f32_only` and `mixed_i32_f32` profiles while keeping non-CPU
  backends and bounded-approximate `f64` narrowing explicit as suppressed
  states; this is not backend-invariant float exactness or broad numeric
  served publication
- the current float-profile publication artifacts are
  `fixtures/tassadar/reports/tassadar_float_profile_acceptance_gate_report.json`
  and `fixtures/tassadar/reports/tassadar_float_profile_route_policy_report.json`,
  which allow exact cpu-reference numeric profiles to influence public
  named-profile posture and profile-specific routing only while keeping
  bounded-approximate `f64` suppressed and keeping the default served lane
  unchanged; this is not arbitrary Wasm float closure or a new default served
  float capability
- the current memory64 proposal-profile artifacts are
  `fixtures/tassadar/reports/tassadar_memory64_profile_report.json` and
  `fixtures/tassadar/runs/tassadar_memory64_resume_v1/tassadar_memory64_resume_bundle.json`,
  which freeze one bounded single-memory large-address continuation profile
  with sparse-window checkpoint artifacts above the 4GiB boundary, typed
  datastream resume locators, and explicit backend-limit refusal truth on the
  current-host cpu-reference envelope; this is not arbitrary Wasm memory64
  closure, multi-memory support, generic large-heap portability, or broader
  served publication
- the current multi-memory proposal-profile artifacts are
  `fixtures/tassadar/reports/tassadar_multi_memory_profile_report.json` and
  `fixtures/tassadar/runs/tassadar_multi_memory_profile_v1/tassadar_multi_memory_runtime_bundle.json`,
  which freeze one bounded multi-memory routing profile over the explicit
  `rodata_heap_output_split` and `scratch_heap_checkpoint_split` topology
  families, with persisted per-memory checkpoint artifacts and explicit
  malformed-topology refusal truth on the current-host cpu-reference envelope;
  this is not arbitrary Wasm multi-memory closure, memory64 plus multi-memory
  mixing, generic allocator portability, or broader served publication
- the current component/linking proposal-profile artifacts are
  `fixtures/tassadar/reports/tassadar_component_linking_profile_report.json`
  and
  `fixtures/tassadar/runs/tassadar_component_linking_profile_v1/tassadar_component_linking_runtime_bundle.json`,
  which freeze one bounded component/linking profile over the explicit
  `utf8_decode_writer_component_pair` and `checkpoint_resume_component_pair`
  topologies, with persisted interface-type lineage artifacts and explicit
  incompatible-interface refusal truth on the current-host cpu-reference
  envelope; this is not arbitrary component-model closure, unrestricted
  interface-type lowering, arbitrary host-import composition, or broader
  served publication
- the current SIMD proposal-profile artifacts are
  `fixtures/tassadar/reports/tassadar_simd_profile_runtime_report.json` and
  `fixtures/tassadar/reports/tassadar_simd_profile_report.json`, which freeze
  one bounded SIMD deterministic profile with exact cpu-reference truth, metal
  and cuda scalar-fallback rows, and typed accelerator-specific refusal truth;
  served publication carries that named profile plus the backend matrix
  without creating a default served SIMD lane, and this still does not claim
  arbitrary SIMD closure, accelerator-invariant vector exactness, or broader
  served publication
- the current threads research-profile artifacts are
  `fixtures/tassadar/reports/tassadar_threads_research_profile_runtime_report.json`,
  `fixtures/tassadar/reports/tassadar_threads_scheduler_sandbox_boundary_report.json`,
  `fixtures/tassadar/reports/tassadar_threads_research_profile_report.json`,
  and `fixtures/tassadar/reports/tassadar_threads_research_profile_summary.json`,
  which freeze one research-only shared-memory and threads lane with a
  deterministic scheduler envelope, explicit sandbox-boundary truth, and typed
  refusal on host-nondeterministic scheduling plus relaxed memory-ordering
  regimes; this is not a served threads lane, arbitrary shared-memory
  portability, or general concurrency closure
- recommended closure order follows the spec and its test families: decode and
  validate first, then integer/control/direct-call core, then linear memory,
  then globals/tables/references/indirect calls, then instantiation/import
  boundaries, then floating-point widening, and only after that separate
  proposal-profile families such as exceptions, GC, memory64, multi-memory,
  linking, SIMD, and relaxed-SIMD
- landed Phase 8A bar: one widened HullCache closure report now exists in
  `psionic-eval` at
  `fixtures/tassadar/reports/tassadar_hull_cache_closure_report.json`,
  proving direct exact HullCache closure on the current micro, branch-heavy,
  memory-heavy, and bounded Hungarian families while keeping long-loop and
  Sudoku search workloads explicit as fallback-only under the current
  control-flow contract
- landed Phase 8B bar: one SparseTopK comparison report now exists in
  `psionic-eval` at
  `fixtures/tassadar/reports/tassadar_sparse_top_k_comparison_report.json`,
  comparing SparseTopK against reference-linear and HullCache on the shared
  article workload set while keeping branch-heavy, long-loop, and Sudoku
  search workloads explicit as fallback-only under the current validation
  contract
- landed Phase 8C bar: one decode-scaling report now exists in `psionic-eval`
  at `fixtures/tassadar/reports/tassadar_decode_scaling_report.json`,
  comparing requested reference-linear, HullCache, and SparseTopK execution on
  shared synthetic trace-length families while recording trace-artifact growth
  and direct-vs-fallback posture so the fast-path scaling story is tied to
  exactness and compatibility truth instead of a single throughput screenshot
- landed Phase 8D bar: one million-step decode benchmark bundle now exists in
  `psionic-runtime` at
  `fixtures/tassadar/runs/million_step_loop_benchmark_v0/benchmark_bundle.json`,
  proving one reproducible 1,048,575-step reference-linear CPU execution with
  compact trace-summary proof lineage, runtime-manifest identity, and explicit
  serialized trace-byte growth receipts, while keeping HullCache and
  SparseTopK explicit as fallback-only at that horizon
- landed Phase 8E bar: one geometric-variant comparison report now exists in
  `psionic-eval` at
  `fixtures/tassadar/reports/tassadar_geometric_variant_report.json`, keeping
  the promoted runtime HullCache surface separate from a research-only
  hierarchical-hull candidate; the candidate stays direct and exact on the
  long-loop and 4x4 Sudoku article workloads, but the widened class remains
  research-only until decode-mode identity and runtime closure bars are
  promoted explicitly
- landed memory ABI v2 bar: `psionic-runtime` now owns a public
  byte-addressed linear-memory contract with explicit i8/i16/i32 load-store
  widths, sign extension, `memory.size`, `memory.grow`, `memory.copy`,
  `memory.fill`, and delta-oriented memory tracing; `psionic-models` now
  publishes the same lane as an explicit repo-facing memory-ABI publication;
  `psionic-train` now materializes a training-facing supervision suite over
  width-parity, sign-extension, growth, memcpy-style trace-regression, and
  copy/fill exactness cases; `psionic-runtime` now also owns a bounded
  dynamic-memory pause-and-resume bundle over the same ABI; and
  `psionic-eval` now freezes the current evidence at
  `fixtures/tassadar/reports/tassadar_memory_abi_v2_report.json` instead of
  pretending the old fixed-slot memory ABI was already truthful enough for
  module-scale Wasm work, while the resumed-memory evidence lives at
  `fixtures/tassadar/reports/tassadar_dynamic_memory_resume_report.json`
- landed module-trace ABI v2 bar: `psionic-runtime` now owns explicit legacy
  v1 versus frame-aware delta-oriented v2 module-trace contracts plus v1/v2
  artifacts, deterministic replay from v2 back into the legacy snapshot-heavy
  execution trace, and shared lineage receipts; `psionic-models` now publishes
  the same lane as a repo-facing trace-ABI publication; `psionic-train` now
  materializes a training-facing supervision suite over global-state,
  call-indirect, and deterministic-import cases; and `psionic-eval` now
  freezes the current evidence at
  `fixtures/tassadar/reports/tassadar_module_trace_abi_v2_report.json`
- landed module-scale Wasm workload-suite bar: `psionic-data` now publishes a
  public `TassadarModuleScaleWorkloadSuiteContract` for deterministic memcpy,
  parsing, checksum, and VM-style module families plus explicit refusal cases;
  `psionic-environments` now binds the same suite into the repo-facing
  Tassadar environment bundle; and `psionic-eval` now freezes the current
  evidence at
  `fixtures/tassadar/reports/tassadar_module_scale_workload_suite_report.json`
  over committed source and compiled Wasm fixtures, keeping exactness,
  trace-length, deterministic CPU-reference cost, and typed refusal explicit
  instead of implying arbitrary module closure
- landed module-state learned-executor redesign bar: `psionic-models` now
  publishes a public `TassadarModuleStateExecutorPublication` over explicit
  call-frame, global-delta, memory-delta, and export-boundary channels plus
  staged curriculum anchors for module-scale workloads; `psionic-train` now
  materializes the corresponding curriculum/eval suite over deterministic
  memcpy, parsing, checksum, and vm-style families with held-out-family
  later-window, final-state, and gap metrics; and `psionic-research` now
  freezes the current evidence at
  `fixtures/tassadar/reports/tassadar_module_state_architecture_report.json`
  instead of treating flat-prefix token-trace prediction as the only learned
  architecture story for module execution
- landed structured-control closure bar: `psionic-compiler` now lowers one
  bounded zero-parameter i32-only Wasm subset with empty block types into
  validated executor-ready structured programs covering `block`, `loop`, `if`,
  `else`, `br`, `br_if`, and `br_table`; `psionic-runtime` now owns the exact
  nested-control executor and branch-trace surface for that lane; and
  `psionic-eval` now freezes exact and refused evidence at
  `fixtures/tassadar/reports/tassadar_structured_control_report.json`,
  including branch-table parity and malformed-label refusal, without claiming
  calls, memories, block-result closure, or arbitrary Wasm support
- landed call-frame execution bar: `psionic-runtime` now owns one bounded
  direct-call multi-function lane with function-local state, real frame-stack
  execution, replayable frame-stack trace snapshots, conditional in-frame
  control, exact bounded recursion below an explicit depth cap, and explicit
  recursion refusal at the cap; `psionic-models` now publishes the same lane
  as a repo-facing call-frame publication with explicit bounded-recursion and
  conditional-control posture; `psionic-train` now materializes a
  training-facing suite over direct-call parity, multi-function replay, exact
  bounded recursion, and recursion refusal; and `psionic-eval` now freezes the
  current evidence at `fixtures/tassadar/reports/tassadar_call_frame_report.json`
  instead of keeping multi-function execution trapped behind the old
  single-frame boundary
- landed numeric-opcode widening bar: `psionic-data` now publishes a public
  `TassadarNumericOpcodeLadderContract` that keeps i32 core arithmetic,
  comparisons, bit operations, i64 integer work, and floating-point work as
  separate explicit families; `psionic-compiler` and `psionic-runtime` now
  widen the bounded structured-control lane through exact i32 comparisons,
  `eqz`, and bitwise/shift operations while still refusing i64 and
  floating-point instructions explicitly; and `psionic-eval` now freezes the
  current evidence at
  `fixtures/tassadar/reports/tassadar_numeric_opcode_ladder_report.json`
  instead of pretending numeric closure already jumped from tiny i32 kernels
  to arbitrary Wasm
- landed module-boundary bar: `psionic-runtime` now owns a bounded
  module-execution contract with explicit i32 globals, one bounded linear
  memory, active data segments, funcref tables, active element-segment
  instantiation, zero-parameter start-function execution, zero-parameter
  direct calls, bounded `call_indirect`, `memory.size`, `memory.grow`,
  `memory.copy`, `memory.fill`, deterministic import stubs, and typed refusal
  for unsupported host calls or out-of-profile memory shapes; `psionic-ir`
  now owns the matching normalized Wasm-module structure for globals, tables,
  memories, data segments, start sections, and active element segments;
  `psionic-compiler` now lowers that admitted module slice into validated
  module-execution programs; `psionic-models` now publishes that boundary as a
  repo-facing module-capability publication; `psionic-serve` and
  `psionic-provider` now carry the same module-support and host-import refusal
  facts through the served capability path; and `psionic-sandbox` now exposes
  a sandbox-facing import-boundary contract instead of leaving host-import
  posture implicit
- landed exactness/refusal evidence bar: `psionic-runtime` now owns a shared
  `TassadarExactnessRefusalReport` contract that records exact, mismatch, and
  refused posture above current selection diagnostics and trace/output/halt
  parity facts; `psionic-provider` now projects the same report into a
  provider-facing receipt; and `psionic-eval` now commits the current exact
  direct, exact fallback, and explicit refusal examples at
  `fixtures/tassadar/reports/tassadar_exactness_refusal_report.json`
- landed trained-executor Phase 1 follow-on bar: a dedicated
  `tassadar.wasm.sudoku_v0_search.v1` profile now exists with a real 4x4
  backtracking Sudoku program representation on the CPU reference lane, while
  the validated hull/sparse fast paths still surface explicit fallback on that
  broader backward-branch search envelope
- landed trained-executor Phase 2 follow-on bar: the fake `SudokuClass`
  placeholder has been replaced by a real split-aware 4x4 Sudoku-v0 corpus
  with exact CPU-reference traces per puzzle and article-class benchmark
  reporting that stays honest about hull/sparse fallback on those search-heavy
  workloads
- landed trained-executor Phase 3 follow-on bar: the Sudoku-v0 corpus can now
  be materialized as deterministic program-plus-trace token sequences with a
  fixed executor vocabulary, reversible symbolic decode, versioned tokenized
  dataset manifests in `psionic-data`, CPU-reference dataset generation in
  `psionic-eval`, and frozen split packing plans in `psionic-train`
- landed trained-executor Phase 4 follow-on bar: `psionic-models` now carries a
  first real neural executor transformer family for the Sudoku-v0 lane, with
  explicit executor-specific descriptor/config surfaces, 2D lookup-head
  geometry claims, next-token logits over the fixed Tassadar vocabulary, and a
  claim boundary that stays honest about this being a trained sequence model
  rather than the already-exact handcrafted executor
- landed trained-executor Phase 5 follow-on bar: `psionic-train` now runs
  teacher-forced next-token optimization over the frozen Sudoku-v0 sequence
  manifest, while `psionic-eval` surfaces exact-trace, final-output, and halt
  correctness reports against the same CPU-reference sequences that generated
  the training corpus
- landed trained-executor Phase 6 follow-on bar: `psionic-eval` now benchmarks
  neural linear decode for the executor transformer against direct CPU
  reference execution on Sudoku-v0, with explicit decode-mode and KV-cache
  identity plus per-case exactness/fallback truth rather than only aggregate
  benchmark theater
- landed trained-executor Phase 7 follow-on bar: `psionic-train` now exposes a
  persisted first-run surface for the Sudoku-v0 neural executor lane, and the
  repo now carries one canonical run bundle at
  `fixtures/tassadar/runs/sudoku_v0_reference_run_v0` with the
  frozen training manifest, training report, linear benchmark report,
  checkpoint payload plus manifest, and trained-model artifact; the recorded
  run remains explicitly low-exactness (`0/2` validation exact-trace cases,
  `15` bps aggregate target exactness), so the claim stays at "first honest
  trained run exists" rather than "the trained executor already works"
- landed trained-executor Phase 8 follow-on bar: the same persisted run bundle
  now also carries machine-readable post-run telemetry and failure artifacts in
  `training_telemetry.json`, `exactness_curve.json`,
  `trace_divergence_report.json`, and `failure_samples.json`; those artifacts
  keep dataset/model/checkpoint identity explicit and show the current first
  run failing immediately on all 8 cases (first divergence at target token 0,
  case exactness between `9` and `16` bps), which is the correct baseline for
  later curriculum/model changes
- landed trained-executor Phase 9 follow-on bar: the same run bundle now also
  carries `postmortem.json` and `next_run_plan.json`, and the repo now has a
  human-readable first-run review in
  `docs/audits/2026-03-16-tassadar-first-run-postmortem.md`; the resulting
  plan explicitly keeps later claims tied to improved 4x4 boundary and
  short-trace exactness rather than letting scale claims outrun the evidence
- landed trained-executor Phase 10 follow-on bar: `psionic-models` now owns an
  explicit model-KV decode state plus machine-legible decode selection over
  `ReferenceLinear` and `HullCache`, `psionic-eval` now benchmarks the trained
  model’s explicit linear-scan KV path against a real hull-cache KV path and
  full direct CPU execution, and `psionic-train` now persists
  `neural_hull_benchmark_report.json` into the committed run bundle; the
  current committed run shows `8/8` hull-vs-linear prefix agreement with no
  fallback/refusal and about `1.93x` hull speedup (`42,172` vs `21,860`
  target tok/s over a `4,096`-token per-case window), while exactness remains
  `0/8`, so this phase closes the “real neural fast path exists” gap without
  pretending it closes the “trained executor works” gap
- landed trained-executor Phase 11 follow-on bar: `psionic-runtime` now owns a
  real `tassadar.wasm.sudoku_9x9_search.v1` profile plus a real split-aware
  9x9 Sudoku-class corpus, `psionic-eval` and `psionic-train` now freeze that
  workload into a tokenized sequence dataset plus training manifest,
  `psionic-models` now carries a matching 9x9 executor-transformer descriptor,
  and `psionic-train` now commits a machine-readable
  `fixtures/tassadar/runs/sudoku_9x9_scale_plan_v0/scale_plan.json`
  that keeps the promotion gate explicit: the real 9x9 workload is in-tree,
  but 4x4 first-target and short-trace exactness are still blocking honest 9x9
  promotion
- landed trained-executor Phase 12 follow-on bar: `psionic-eval` now emits
  first-target / first-8 / first-32 boundary exactness plus divergence and
  first-token-confusion reports, `psionic-train` now supports an explicit
  boundary curriculum with per-epoch validation and boundary-ranked checkpoint
  selection, and the committed follow-on run bundle at
  `fixtures/tassadar/runs/sudoku_v0_boundary_v1` records the
  first honest post-audit boundary improvement (`10000` bps first-target
  exactness, divergence moved to target index `1`) while still failing the
  later gates (`5000` bps first-32 exactness, `0/2` exact traces)
- landed trained-executor Phase 13 follow-on bar: the lookup-family executor
  now records a stable trainable surface in descriptors, manifests,
  checkpoints, and run bundles, `psionic-train` now supports controlled output
  head / embedding / small-mixer surfaces, and `psionic-research` now commits a
  same-corpus ablation root at
  `fixtures/tassadar/runs/sudoku_v0_trainable_surface_ablation_v1`
  where only `output_head_embeddings_and_small_learned_mixer` materially beats
  the preserved baseline (`3750` bps first-8 exactness, `5625` bps first-32
  exactness) while still leaving `0/2` exact traces
- landed trained-executor Phase 14 follow-on bar: `psionic-train` now owns a
  preserved red learned-lane promotion bundle at
  `fixtures/tassadar/runs/sudoku_v0_promotion_v1`, explicit
  `best_checkpoint_manifest.json` plus `promotion_gate_report.json` artifacts,
  a repo-owned `scripts/check-tassadar-4x4-promotion-gate.sh` checker for
  revalidating persisted gate reports, and live
  stage/epoch/batch/validation/checkpoint progress while long runs are
  executing; that original lookup-family promotion result remained explicitly
  below the bar at
  checkpoint `epoch_0006` (`10000` bps first-target, `7500` bps first-8,
  `6875` bps first-32, `0/2` exact validation traces), so that bundle closes
  the “promotion tooling exists” gap and remains preserved blocker evidence
- landed trained-executor Phase 14A follow-on bar: `psionic-train` now also
  preserves a separate teacher-forced continuation bundle at
  `fixtures/tassadar/runs/sudoku_v0_promotion_v2`; that run
  proves schedule-only churn on the current lookup family does not beat the
  canonical ceiling, because its selected checkpoint `epoch_0008` exactly
  reproduces the same gate result (`10000` bps first-target, `7500` bps
  first-8, `6875` bps first-32, `0/2` exact validation traces) before later
  32-token epochs regress again, so the next honest move is model/architecture
  change rather than more schedule tuning
- landed trained-executor Phase 14B closure: `psionic-research` now owns the
  canonical learned-lane promotion runner in
  `crates/psionic-research/examples/tassadar_executor_attention_promotion_run.rs`,
  the extracted-workspace checkpoint/bootstrap pathing is fixed, and the green
  learned bundle now lives at `fixtures/tassadar/runs/sudoku_v0_promotion_v3`
  with a bootstrap seed under `bootstrap_pc_boundary`; the selected checkpoint
  `epoch_0015` from `prompt_to_first_32_tokens` now clears the full gate
  (`10000` bps first-target, `10000` bps first-8, `10000` bps first-32,
  `2/2` exact validation traces), `exact_trace_samples.json` captures both
  validation cases, `failure_samples.json` is empty, and the repo-owned gate
  checker revalidates the stored report as passed; that clears the bounded
  benchmark gate, but the separate promotion-policy report still blocks served
  promotion until the learned lane also has stable refusal policy and
  route-contract compatibility
- landed trained-executor Phase 15 follow-on bar: `psionic-models` now carries
  a separate bounded `TassadarExecutorAttentionTransformer` family with layered
  full-prefix causal hard-max attention, fixed 2D head geometry, explicit
  per-layer semantics, and truthful hull fallback, while `psionic-eval` and
  `psionic-research` now persist a bounded same-corpus comparison root at
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v1`;
  the resulting report keeps the claim boundary explicit by showing the new
  family is architecturally closer to the article but still worse than the
  preserved lookup baseline on the bounded 4x4 window (`0` bps first-target /
  first-32 exactness and `1333` target tok/s, versus `10000` / `6563` bps and
  `32000` target tok/s for the lookup baseline), so this phase lands as a
  research candidate rather than a promotion result
- landed trained-executor Phase 15A follow-on bar: `psionic-research` now also
  owns a bounded attention-family training loop plus a preserved trained-family
  comparison root at
  `fixtures/tassadar/runs/sudoku_v0_attention_training_v1` and
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v2`;
  the resulting artifacts prove the executor-attention family is no longer just
  a seeded architectural candidate because it now trains off the `0`-bps floor
  to `6563` bps aggregate / first-32 exactness on the bounded window, but it
  still fails the first-token boundary (`0` bps first-target), still stays at
  `0` exact bounded traces, and still loses the preserved lookup baseline on
  the open 4x4 promotion metric, so the claim boundary remains
  `research_windowed_decode_only` rather than learned-lane success
- landed trained-executor Phase 16 follow-on bar: `psionic-train` now owns the
  canonical `crates/psionic-train/examples/tassadar_sudoku_9x9_reference_run.rs`
  replay path and the committed bundle
  `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0`; the learned lane
  now records an explicit `incremental_decode_window` teacher-forced strategy
  and `incremental_decode_window` long-trace family contract in the training
  manifest, persists `sequence_fit_report.json`, `postmortem.json`,
  `next_run_plan.json`, `later_window_exactness_report.json`,
  `suffix_window_failure_report.json`, `best_checkpoint_manifest.json`,
  `promotion_bundle.json`, and `promotion_gate_report.json`, and the
  repo-owned `scripts/check-tassadar-9x9-promotion-gate.sh` checker
  revalidates the stored gate as consistent; the selected checkpoint remains
  `epoch_0004` from `full_trace_supervision`, full 9x9 traces still exceed the
  current `524288`-token model context (`4891222` to `5335309` total tokens,
  overflow `4366934` to `4811021`), and the new gate keeps the learned failure
  boundary explicit: the early `512`-token prefix stays at `5938` bps
  first-32 exactness, the later fixed offset window at target token `262144`
  and the furthest fittable suffix window at target token `472240` both reach
  `8438` bps first-32 exactness, all three gate windows remain `0/1` exact
  windows, and full-trace exactness across the declared gate windows remains
  `0`, so later slices are now visible without pretending the full fit problem
  is solved
- landed explicit 9x9 long-trace family comparison: `psionic-train` now also
  materializes `fixtures/tassadar/runs/sudoku_9x9_v0_windowed_family_comparison_v1`,
  which keeps the learned claim bounded while making the family split explicit:
  the flat-prefix family stays
  `tassadar-executor-transformer-sudoku-9x9-v0`, the windowed family stays
  `tassadar-executor-transformer-sudoku-9x9-windowed-v0`, both remain at
  `5938` bps first-32 and `0/1` exact validation traces on the first `512`
  target tokens, but the declared live-state contract drops from
  `109715076` bytes on the flat-prefix family to `1459452` bytes on the
  windowed family under the same corpus and fit facts
- landed research-only sequential-vs-wavefront trace-family comparison:
  `psionic-data` now publishes a comparable public trace-family-set contract,
  `psionic-train` now materializes and reproducibly rechecks
  `fixtures/tassadar/runs/tassadar_trace_family_comparison_v1`, which keeps
  the sequential CPU trace as the only full execution authority while proving
  that alternate target families can preserve final outputs exactly on the
  same corpora; the anti-diagonal Sudoku family drops max total tokens from
  `5335309` to `52969` on 9x9 and the parallel Hungarian assignment frontier
  drops max total tokens from `11532454` to `22050` on 10x10, with all
  alternate families staying at `10000` bps final-output exactness under
  explicit `research_only` claim boundaries, and `psionic-research` now
  freezes the repo-facing variant summary at
  `fixtures/tassadar/reports/tassadar_trace_family_variant_report.json`
- landed research-only no-hint / self-supervised executor regime comparison:
  `psionic-train` now materializes public full-hint-trace, subroutine-hint,
  no-hint-output-only, and no-hint-self-supervised regime manifests plus
  reusable-signal proxies over the seeded sort / CLRS-shortest-path /
  sudoku-style corpus, and `psionic-research` now freezes the resulting
  architecture report at
  `fixtures/tassadar/reports/tassadar_no_hint_self_supervised_report.json`;
  the held-out CLRS proxy moves from `1666` bps reusable signal on full-hint
  traces to `5000` on output-only no-hint and `8000` on no-hint plus
  self-supervised regularizers, while reusable subroutine hints remain at
  `8333`, and the entire lane stays explicitly refused for served promotion
- landed scratchpad / controlled-position executor framework comparison:
  `psionic-ir` now owns bounded `flat_trace` and
  `delimited_chunk_scratchpad` formatting plus `absolute_monotonic`,
  `segment_reset`, and `trace_schema_buckets` controlled position-ID schemes,
  `psionic-models` now exposes public framework descriptors plus locality
  evidence inspection, and `psionic-train` now freezes the resulting
  arithmetic symbolic and algorithmic same-lane comparison at
  `fixtures/tassadar/reports/tassadar_scratchpad_framework_comparison_report.json`;
  the report keeps the claim boundary at `learned_bounded_success`, cuts
  arithmetic max output local position from `14` to `3`, cuts algorithmic max
  output local position from `11` to `3`, preserves final output tokens
  exactly, and keeps scratchpad overhead plus reset counts explicit
- landed efficient-attention baseline matrix:
  `psionic-eval` now freezes dense reference-linear, validated SparseTopK,
  artifact-backed recurrent runtime baseline, Reformer-style proxy, promoted
  HullCache, and research hierarchical-hull rows on the same article-class
  workload artifact at
  `fixtures/tassadar/reports/tassadar_efficient_attention_baseline_matrix.json`,
  and `psionic-research` now summarizes the resulting win/tie/lose/refuse
  posture at
  `fixtures/tassadar/reports/tassadar_efficient_attention_baseline_summary.json`;
  the committed matrix keeps the claim boundary explicitly research-only for
  the recurrent, Reformer-style, and hierarchical-hull rows, records promoted
  HullCache as fastest on `1` workload, records the research
  hierarchical-hull candidate as fastest on `2`, records the recurrent runtime
  baseline as fastest on `3`, and makes the Reformer-style proxy refuse the
  long-loop and Sudoku rows instead of letting specialized fast paths compare
  only to naive dense replay
- landed trained-executor Phase 15B follow-on bar: the same executor-attention
  family now also carries a bounded relative-target output-bias adapter in
  `psionic-models`, the preserved destructive boundary-first output-head
  attempt now lives at
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v1`, the
  improved adapter-backed run now lives at
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v2`, and
  the later projection-adapter follow-ons now live at
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v3` and
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v4`, the
  newer transition-adapter follow-on now lives at
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v5`, the
  later joint-adapter fine-tune now lives at
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v6`, the
  later trace-schema and per-position saturation runs now live at
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v7`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v8`, and
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v9`, and
  the current same-corpus comparison now lives at
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v11`;
  those artifacts keep improving the first executor-attention boundary surface
  over the lookup baseline, and the latest bounded pair now records
  `10000` bps first-target, `8750` bps first-8, and `7188` bps first-32
  versus lookup `10000` / `6250` / `6563`, but the learned gate is still not
  green because exact validation traces remain `0/2` and the sharper remaining
  blocker is still token `6`: the model predicts `<byte_00>` where the
  reference requires `<pc>`; the later joint transition+projection fine-tune
  reproduces that ceiling rather than beating it, and the later trace-schema /
  per-position saturation set proves the current bounded adapter family is
  saturated rather than merely under-tuned
- landed trained-executor Phase 16D follow-on bar: `psionic-models` now
  exposes a direct sparse-top-k lookup baseline beside the existing
  hull-specialized and windowed lookup families, `psionic-eval` now emits a
  four-family same-corpus learned baseline comparison, and `psionic-research`
  now persists the canonical root at
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v12`; the
  committed artifacts compare hull-specialized lookup, sparse lookup, hybrid
  attention, and recurrent/windowed lookup on the same bounded Sudoku-v0
  validation window while keeping the result honestly red and comparison-only:
  all four seeded trainable families stay at `0` bps first-target, first-8,
  and first-32 exactness, the recurrent family changes the long-trace
  contract from `flat_prefix_full_forward` to `incremental_decode_window`,
  and the hybrid attention family keeps its full-sequence fit cliff explicit
  with `0/2` shared cases fitting under the current `512`-token bound
- landed trained-executor Phase 17 follow-on bar: `psionic-models` now carries
  a bounded typed `TassadarCompiledProgramExecutor` surface with persisted
  compile-evidence bundles, `psionic-eval` now emits exactness and
  compatibility/refusal reports for the real Sudoku-v0 corpus under
  `tassadar.wasm.sudoku_v0_search.v1.compiled_executor`, and
  `psionic-research` now materializes the canonical bundle root at
  `fixtures/tassadar/runs/sudoku_v0_compiled_executor_v0`; the
  committed artifacts prove an exact bounded compiled/proof-backed lane on the
  matched corpus (`8/8` exact trace matches against CPU reference and `32/32`
  exact refusal matches on mismatched artifacts) while keeping the serving and
  claim boundary explicit (`eval_only`, not arbitrary-program closure, not
  learned-lane success, not article parity)
- landed trained-executor Phase 18 follow-on bar: `psionic-runtime` now
  carries a bounded real `tassadar.wasm.hungarian_v0_matching.v1` min-cost
  matching workload over 4x4 cost matrices, `psionic-models` exposes the
  matching compiled deployment fixture, `psionic-eval` now emits a real
  Hungarian-v0 benchmark package together with compiled exactness,
  compatibility/refusal, and learned-vs-compiled lane-status reports, and
  `psionic-research` now materializes the canonical bundle root at
  `fixtures/tassadar/runs/hungarian_v0_compiled_executor_v0`;
  the committed artifacts prove a bounded Hungarian-class workload contract
  plus an exact compiled/proof-backed lane on that matched corpus (`8/8`
  exact trace matches against CPU reference and `32/32` exact refusal
  matches) while keeping the serving and claim boundary explicit (`eval_only`,
  not a learned Hungarian lane, not arbitrary dimension/program closure, and
  not article parity)
- landed learned Hungarian-v0 follow-on bar: `psionic-models` now exposes a
  workload-specific learned Hungarian transformer family, `psionic-train`
  now emits the canonical bounded learned bundle at
  `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0`, and the
  persisted artifacts separate token exactness, dual-state exactness, and
  final-result exactness; the selected checkpoint remains research-only rather
  than promoted (`aggregate=6839`, `first_target=0`, `first_32=6875`,
  `exact_traces=0`, `final_outputs=0`, `workload_specific_state=7568`) even
  though full traces fit the current learned model window
- landed learned long-horizon refusal bar: `psionic-research` now emits
  `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`,
  which makes the learned long-horizon boundary machine-readable instead of
  implicit by freezing `unsupported_horizon` for million-step/article-class
  learned traces until an exact learned long-horizon benchmark bundle exists
- landed trained-executor Phase 19 follow-on bar: `psionic-runtime` and
  `psionic-models` now carry an exact compiled 9x9 Sudoku search deployment
  lane, `psionic-eval` now emits benchmark/environment, exactness, refusal,
  and throughput artifacts for
  `tassadar.wasm.sudoku_9x9_search.v1.compiled_executor`, and
  `psionic-research` now materializes the canonical bundle root at
  `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0`; the committed
  artifacts prove exact compiled/proof-backed 9x9 Sudoku closure on the
  matched corpus (`4/4` exact trace matches against CPU reference and `16/16`
  exact refusal matches on the full corpus) while keeping the serving and
  claim boundary explicit (`eval_only`, article-sized Sudoku closure only, not
  full compiled article parity)
- landed compiled article-sized matching follow-on bar: `psionic-runtime` and
  `psionic-models` now carry a dedicated
  `tassadar.wasm.hungarian_10x10_matching.v1` profile and exact compiled
  10x10 Hungarian deployment lane, `psionic-eval` now emits
  benchmark/environment, exactness, refusal, throughput, and claim-boundary
  artifacts for the committed 10x10 corpus, and `psionic-research` now
  materializes the canonical bundle root at
  `fixtures/tassadar/runs/hungarian_10x10_v0_compiled_executor_v0`; the
  committed artifacts prove exact compiled/proof-backed 10x10 Hungarian
  closure on the matched corpus while keeping the boundary explicit
  (`eval_only`, article-sized matching closure on the larger matching profile,
  not learned Hungarian execution, and not full compiled article parity)
- landed generic compiled kernel-suite follow-on bar: `psionic-eval` and
  `psionic-research` now materialize the canonical bundle root at
  `fixtures/tassadar/runs/compiled_kernel_suite_v0`, covering bounded
  arithmetic, memory-update, forward-branch, and backward-loop families under
  `tassadar.wasm.article_i32_compute.v1`; the committed artifacts now carry
  benchmark/environment, exactness, refusal, claim-boundary, exactness-vs-trace-length
  scaling, and proof-bearing per-regime deployment evidence for that suite,
  proving exact compiled/proof-backed kernel closure across those four
  families while keeping the boundary explicit (`eval_only`, generic kernel
  evidence only, not arbitrary-program closure, and not full compiled article
  parity)
- landed compiled article-closure checker: `psionic-research` now emits the
  machine-readable report
  `fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json`
  plus the repo-owned validator command
  `scripts/check-tassadar-compiled-article-closure.sh`, which only turns green
  once the Rust-only frontend canon is green, the article-sized compiled 9x9
  Sudoku bundle, the article-sized compiled 10x10 Hungarian bundle, and the
  generic compiled kernel suite all exist with proof and benchmark artifacts,
  and the older 4x4 proxies are no longer the strongest exact compiled
  evidence in-tree
- landed Phase 8A bar: typed `psionic-research` executor-variant family with
  benchmark/proof/lineage-backed bounded runs and machine-readable sweep
  records for reproducible same-contract candidate comparison
- landed Phase 8B bar: validated `SparseTopK` decode mode with explicit direct
  selection on the current subset, exact fallback on unsupported shapes, and
  benchmark reporting against CPU, reference-linear, and hull-cache baselines
- landed Phase 9A bar: planner-owned hybrid routing through
  `psionic.planner_executor_route`, with benchmark-gated route capability
  descriptors shared across `psionic-serve`, `psionic-router`, and
  `psionic-provider`, executor preflight, replay-stable routing decisions,
  typed completed/fallback/refused outcomes, and explicit planner-visible
  policy, budget, proof, selection, and refusal truth
- landed article-hybrid workflow follow-on: `psionic-serve` now also exposes
  the specialized `psionic.article_hybrid_workflow` surface, which binds
  canonical article workloads into planner-owned exact-compute workflows while
  preserving benchmark identity, routing receipts, proof identity on delegated
  success, and typed planner fallback/refusal truth; the committed acceptance
  artifact is
  `fixtures/tassadar/reports/tassadar_article_hybrid_workflow_artifact.json`
- landed Tassadar lab follow-on: `psionic-serve` now also exposes a
  renderer-neutral replay/live lab adapter that projects
  `psionic.article_executor_session`, `psionic.article_hybrid_workflow`, and
  canonical compiled or learned Tassadar artifacts into one stable
  snapshot/update surface for desktop panes without making OpenAgents parse raw
  report internals itself; the committed acceptance artifact is
  `fixtures/tassadar/reports/tassadar_lab_surface_artifact.json`
- landed Phase 9B bar: bounded small-model executor training in
  `psionic-train`, with package-backed Tassadar supervision, fixed-budget
  training receipts, proof-aware exactness comparison against the handcrafted
  reference lane, and explicit validation-corpus-only scope claims
- landed learned structural-supervision follow-on: `psionic-models` now
  classifies instruction-pointer, branch-outcome, stack-delta, memory-diff,
  and workload-specific target families from the executor trace ABI,
  `psionic-train` now persists structural-supervision profiles and split-level
  coverage inventory in the sequence training manifest and emits
  `structural_supervision_report.json` in learned run bundles, `psionic-eval`
  now scores family-level exactness on bounded validation decodes, and
  `psionic-research` now materializes the bounded comparison root at
  `fixtures/tassadar/runs/sudoku_v0_supervision_ablation_v1`; the committed
  artifacts prove richer learned targets moved the bounded lane materially
  without widening the claim boundary (`4570` to `7812` aggregate target-token
  exactness, `4375` to `6875` first-32 exactness, instruction-pointer `5000`
  to `7000` bps, stack-delta `2500` to `5833` bps, still `10000` bps
  first-target, and still bounded early-curriculum validation only)
- landed learned subroutine-library follow-on: `psionic-models` now carries a
  public reusable subroutine library for sort, CLRS shortest-path, and
  sudoku-style workloads, `psionic-train` now materializes the same seeded
  corpus under explicit `full_trace` versus `subroutine_library` supervision
  modes plus deterministic held-out-workload OOD reuse comparisons, and
  `psionic-research` now freezes the bounded label-reuse ablation at
  `fixtures/tassadar/reports/tassadar_subroutine_library_ablation_report.json`;
  the committed artifact keeps the claim boundary explicit by proving only
  supervision-target reuse deltas, not trained-model exactness
- landed full local AttnRes reference-run contract: `psionic-train` now exposes
  a non-`tiny` public AttnRes local-reference config/corpus/run surface on top
  of the existing CPU-reference training core, with a `320`-step local run
  budget plus stepwise model/config/corpus accessors and logical timing fields
  so the desktop lab can render the full local interactive run without
  quadratic replay
- landed Phase 9C bar: program-specialized compiled-weight deployments in
  `psionic-models` plus larger-2D-head family research outputs in
  `psionic-research`, with exact program-artifact binding, explicit
  runtime-contract truth, compile-time proof/runtime-manifest lineage,
  deterministic head-geometry and parameter-count declarations, first-class
  compiled-weight suite artifacts for `program_compiled` candidates, and the
  machine-readable direct-vs-compiled comparison report at
  `fixtures/tassadar/reports/tassadar_program_to_weights_benchmark_suite.json`
- landed module-aware compiled-specialization follow-on: `psionic-runtime` now
  owns a public module-specialization plan plus bounded exact export lowering
  over normalized Wasm module structure and call-graph reachability,
  `psionic-models` now publishes a research-only shared
  `TassadarCompiledModuleSpecialization` artifact with per-export compiled
  lineage and exactness facts, and `psionic-research` now freezes the current
  size-vs-dispatch-cost evidence at
  `fixtures/tassadar/reports/tassadar_module_specialization_benchmark.json`
  instead of treating today's per-export program-specialized lane as the only
  honest compile-to-weights story for multi-function modules
- landed bounded symbolic compiler-target follow-on: `psionic-ir` now owns a
  public straight-line `TassadarSymbolicProgram` contract with textual
  parsing, validation, direct symbolic evaluation, seeded
  arithmetic/parity/memory plus finite-state-counter and simple stack-machine
  examples, and explicit lowering-opcode requirements, `psionic-compiler` now
  lowers that bounded IR into concrete `TassadarProgram` instances with typed
  profile/opcode/local-budget refusal plus a first-class
  `TassadarSymbolicProgramArtifactBundle` contract for runnable runtime
  artifacts and expected execution manifests, `psionic-runtime` now publishes a
  dedicated `symbolic_program` source kind for that ingress path, `psionic-research`
  now freezes the current bounded compiler lane at
  `fixtures/tassadar/reports/tassadar_symbolic_program_artifact_suite.json`,
  and `psionic-models` now exposes the widened model-facing symbolic example
  suite under the explicit `compiled_bounded_exactness` claim class instead of
  leaving the "paper idea to runtime" gap implicit
- landed sparse-rule/minimality audit follow-on: `psionic-ir` now owns a
  public sparse-rule audit contract over the bounded symbolic lane, with
  statement-projected transition rules plus seeded kernel and scan-style audit
  cases, `psionic-compiler` now builds machine-legible sparse-rule compiler
  audits with duplicate-signature clustering, dead-rule detection, and
  final-state-versus-IO-only underconstraint facts, `psionic-eval` now
  summarizes compile-size and minimality behavior across those workload groups,
  and `psionic-research` now freezes the committed audit at
  `fixtures/tassadar/reports/tassadar_sparse_rule_compiler_audit_report.json`;
  the claim remains research-only compiled bounded exactness over the current
  symbolic subset and does not imply arbitrary Wasm or learnability closure
- landed shared-depth executor family follow-on: `psionic-models` now owns a
  public research-only `TassadarSharedDepthExecutorPublication` for shared
  recurrent refinement with explicit fixed-budget versus dynamic-halting
  posture over loop-heavy kernel and call-heavy module families, `psionic-train`
  now materializes the same lane as a deterministic curriculum suite comparing
  flat-prefix, shared-depth fixed-budget, and shared-depth dynamic-halting
  variants, `psionic-eval` now publishes an explicit halting-calibration report
  surface over budget exhaustion and later-window exactness, and
  `psionic-research` now freezes the committed architecture report at
  `fixtures/tassadar/reports/tassadar_shared_depth_architecture_report.json`;
  the lane remains research-only learned bounded architecture work and does not
  imply arbitrary long-horizon learned exactness, arbitrary Wasm closure, or
  served promotion
- landed conditional masking and address-selection follow-on:
  `psionic-runtime` now owns a bounded conditional-masking contract over
  local-slot, frame-window, and memory-region address families with explicit
  refusal kinds for wider or out-of-family access regimes, `psionic-models`
  now publishes a learned-bounded-success executor lane with explicit local,
  frame, and memory pointer heads above that runtime contract, `psionic-train`
  now materializes deterministic masked-versus-unmasked pointer/value and OOD
  locality comparisons, and `psionic-eval` now freezes the committed report at
  `fixtures/tassadar/reports/tassadar_conditional_masking_report.json`; the
  lane keeps masked access bounded and refuses broader address regimes instead
  of implying compiled exactness, arbitrary pointer arithmetic, or arbitrary
  Wasm closure
- landed CLRS-to-Wasm bridge follow-on: `psionic-data` now owns a public
  `TassadarClrsWasmBridgeContract` for a benchmark-bound CLRS shortest-path
  bridge with explicit sequential-versus-wavefront trajectory families and
  tiny-versus-small length buckets, `psionic-environments` now binds that
  bridge into the public Tassadar environment bundle as an optional
  `TassadarClrsWasmBridgeBinding`, `psionic-compiler` now publishes
  deterministic CLRS bridge case specs plus a repo-facing WAT -> Wasm ->
  bounded-artifact helper over the committed shortest-path fixtures, and
  `psionic-eval` now freezes the committed report at
  `fixtures/tassadar/reports/tassadar_clrs_wasm_bridge_report.json`; the lane
  is a literature-facing benchmark bridge over fixed shortest-path witnesses,
  not a claim of full CLRS coverage, arbitrary Wasm closure, or learned
  transfer
- landed verifier-guided search trace-family follow-on: `psionic-data` now
  owns a public `TassadarVerifierGuidedSearchTraceFamilyContract` with
  explicit guess, verify, contradiction, and backtrack events plus bounded
  search-budget metadata, `psionic-runtime` now publishes the corresponding
  machine-legible search trace artifacts over one real Sudoku-v0 case and one
  bounded search-kernel recovery case, `psionic-train` now materializes the
  committed run artifact at
  `fixtures/tassadar/runs/tassadar_verifier_guided_search_trace_family_v1/search_trace_family_report.json`,
  `psionic-eval` now freezes the committed report at
  `fixtures/tassadar/reports/tassadar_verifier_guided_search_report.json`,
  and `psionic-research` now freezes the companion architecture summary at
  `fixtures/tassadar/reports/tassadar_verifier_guided_search_architecture_report.json`;
  the lane is explicitly research-only verifier-guided search evidence rather
  than compiled correctness, general combinatorial closure, or served
  promotion
- landed locality-preserving scratchpad pass follow-on: `psionic-ir` now owns
  a public locality-preserving scratchpad pass contract over the seeded
  symbolic straight-line and module-trace-v2 families, `psionic-compiler` now
  materializes same-program baseline-vs-candidate scratchpad compilations over
  those families with explicit overhead-budget refusal, `psionic-runtime` now
  publishes replay receipts proving the candidate formatting preserves source
  token truth while measuring useful-lookback reduction and overhead,
  `psionic-models` now exposes the repo-facing
  `TassadarLocalityScratchpadPublication`, `psionic-train` now materializes
  the committed suite at
  `fixtures/tassadar/runs/tassadar_locality_scratchpad_suite_v1/locality_scratchpad_suite.json`,
  and `psionic-eval` now freezes the committed report at
  `fixtures/tassadar/reports/tassadar_locality_scratchpad_report.json`; the
  lane is explicitly replay-preserving trace formatting over bounded families,
  not a semantic program rewrite, arbitrary long-horizon learned exactness, or
  served promotion
- landed structured numeric encoding follow-on: `psionic-ir` now owns public
  structured numeric encodings for bounded immediate, offset, and address
  fields under legacy-token, binary-bit, and mixed-radix-hex schemes,
  `psionic-data` now publishes a public
  `TassadarStructuredNumericEncodingLaneContract` with explicit train-vs-held-out
  numeric splits across arithmetic-immediate, address-offset, and
  memory-address workload families, `psionic-models` now exposes the
  repo-facing `TassadarNumericEncodingPublication`, `psionic-train` now
  materializes the committed suite at
  `fixtures/tassadar/runs/tassadar_numeric_encoding_suite_v1/numeric_encoding_suite.json`,
  and `psionic-eval` now freezes the committed report at
  `fixtures/tassadar/reports/tassadar_numeric_encoding_report.json`; the lane
  is explicitly bounded numeric-representation research over held-out
  vocabulary coverage and exact roundtrip semantics, not arbitrary numeric
  closure, learned exactness, or served promotion
- landed decompilable learned-executor artifacts follow-on: `psionic-models`
  now publishes a research-only `TassadarDecompilableExecutorPublication`
  with two constrained learned families, seeded retrain artifacts, readable
  symbolic decompilation targets, and benchmark lineage back to compiled
  symbolic references, `psionic-eval` now freezes the committed fidelity
  report at
  `fixtures/tassadar/reports/tassadar_decompilation_fidelity_report.json`,
  `psionic-research` now freezes the companion artifact story at
  `fixtures/tassadar/reports/tassadar_decompilable_executor_artifacts_report.json`,
  and `psionic-provider` now projects those research summaries into a
  provider-facing `TassadarDecompilationReceipt`; the lane is explicitly
  research-only decompilation evidence for auditability and promotion
  discipline, not a claim of broad learned exactness, arbitrary Wasm closure,
  or served readiness
- landed learnability-gap analysis follow-on: `psionic-train` now materializes
  a committed `TassadarLearnabilityGapEvidenceBundle` at
  `fixtures/tassadar/runs/tassadar_learnability_gap_v1/learnability_gap_evidence_bundle.json`,
  tying seeded kernel, Sudoku, Hungarian, and CLRS-to-Wasm families to
  explicit baseline-vs-improved trace, position, supervision, and metric
  deltas plus plausible learnability-gap classes, `psionic-eval` now freezes
  the committed classifier report at
  `fixtures/tassadar/reports/tassadar_learnability_gap_report.json`, and
  `psionic-research` now freezes the companion action summary at
  `fixtures/tassadar/reports/tassadar_learnability_gap_analysis_report.json`;
  the lane is explicitly research-only analysis for gap diagnosis and next
  actions, not a promotion result or proof that any reported learnability gap
  is closed
- landed Phase 9D bar: typed learned-plus-compiled and learned-circuit
  research in `psionic-research`, with explicit research-line,
  instruction-set, execution-proxy, claim-boundary, and proof-expectation
  contracts, plus direct comparison against the handcrafted Wasm baseline and
  the bounded small-executor training lane on the validation corpus
- the canonical coarse Tassadar claim vocabulary is now
  `compiled_exact`, `compiled_article_class`, `learned_bounded`,
  `learned_article_class`, and `research_only`; the canonical current
  compiled, learned, and research bundles carry `claim_class`, while
  `claim_boundary`, `boundary_label`, and `serve_posture` keep the tighter
  executable and serving limits explicit
- landed planner-native language-versus-compute policy follow-on:
  `psionic-models` now publishes a research-only
  `TassadarPlannerLanguageComputePolicyPublication` over explicit
  `language_only`, `internal_exact_compute`, and `external_tool` route
  families with weighted correctness, cost, evidence-burden, refusal-risk,
  and workload-fit signals, `psionic-router` now freezes the deterministic
  benchmark report at
  `fixtures/tassadar/reports/tassadar_planner_language_compute_policy_report.json`
  with seeded hybrid cases that keep language-only wins, internal
  exact-compute wins, external-tool wins, and explicit `do_not_call_executor`
  refusals separate, and `psionic-provider` now projects that router report
  into a provider-facing `TassadarPlannerLanguageComputePolicyReceipt`; the
  lane remains benchmark-bound routing research and does not widen served
  capability, accepted-outcome closure, or market authority
- landed mixed language-and-compute trajectory follow-on: `psionic-ir` now
  owns a public `TassadarMixedTrajectory` schema with typed language spans,
  exact-compute spans, verifier spans, external-tool spans, explicit lane
  handoffs, and receipt boundaries, `psionic-data` now publishes a public
  `TassadarMixedTrajectoryContract` with seeded article-hybrid,
  verifier-attached-search, and external-tool-long-loop cases, `psionic-runtime`
  now projects those trajectories into bounded replay receipts with explicit
  lane-sequence, handoff-count, receipt-boundary-count, schema-roundtrip, and
  outcome-parity facts, `psionic-train` now freezes the committed suite
  artifact at
  `fixtures/tassadar/runs/tassadar_mixed_trajectory_suite_v1/mixed_trajectory_suite.json`,
  and `psionic-eval` now freezes the joined report at
  `fixtures/tassadar/reports/tassadar_mixed_trajectory_report.json`; the lane
  remains execution-truth substrate for hybrid replay and training and does
  not imply accepted-outcome closure or settlement authority
- landed evidence-calibrated routing follow-on: `psionic-router` now freezes a
  public mount-scoped `TassadarEvidenceCalibratedRoutingReport` at
  `fixtures/tassadar/reports/tassadar_evidence_calibrated_routing_report.json`
  over seeded hybrid cases that score `language_only`,
  `internal_exact_compute`, and `external_tool` routes against explicit
  evidence burden, trust tier, validator attachment, accepted-outcome
  requirements, route allowlists, and world-mount budgets instead of
  capability alone, and `psionic-provider` now projects that report into a
  provider-facing `TassadarEvidenceCalibratedRoutingReceipt`; the lane remains
  policy-bound routing research and does not imply accepted-outcome authority,
  settlement truth, or served capability widening
- landed negative-invocation planner-learning follow-on: `psionic-models` now
  publishes a public `TassadarNegativeInvocationPublication` over explicit
  unnecessary-internal-invocation, fallback-churn,
  evidence-quality-regression, and refusal-when-better-lane-exists penalties,
  `psionic-train` now freezes the committed evidence bundle at
  `fixtures/tassadar/runs/tassadar_negative_invocation_v1/negative_invocation_evidence_bundle.json`
  with matched baseline-versus-preferred route outcomes across seeded hybrid
  cases, `psionic-router` now freezes the before/after route audit at
  `fixtures/tassadar/reports/tassadar_negative_invocation_route_audit.json`,
  and `psionic-eval` now freezes the joined cost-latency-quality tradeoff
  report at
  `fixtures/tassadar/reports/tassadar_negative_invocation_report.json`; the
  lane remains research-only planner learning and does not imply served
  promotion, accepted-outcome authority, or settlement closure
- landed matched internal-vs-external delegation benchmark follow-on:
  `psionic-environments` now publishes a public
  `TassadarDelegationBenchmarkSuite` over matched arithmetic, memory-heavy,
  long-loop, Sudoku, branch-heavy, and CLRS bridge workload rows,
  `psionic-sandbox` now publishes the explicit external-tool foil as
  `TassadarExternalDelegationBaseline`, `psionic-router` now freezes the
  matched route matrix at
  `fixtures/tassadar/reports/tassadar_internal_external_delegation_route_matrix.json`
  with explicit `internal_exact_compute`, `cpu_reference`,
  `external_sandbox`, and `hybrid` winners, `psionic-eval` now freezes the
  joined report at
  `fixtures/tassadar/reports/tassadar_internal_external_delegation_benchmark_report.json`,
  and `psionic-provider` now projects the same matrix into a
  provider-facing `TassadarInternalExternalDelegationReceipt`; the lane
  remains a benchmark-bound comparison surface and does not imply product
  promotion, accepted-outcome authority, or settlement closure
- landed receipt-supervised planner-learning follow-on: `psionic-models` now
  publishes a public `TassadarReceiptSupervisionPublication` over explicit
  validator outcomes, accepted-outcome labels, and receipt-supervision refs,
  `psionic-provider` now publishes the seeded
  `TassadarReceiptSupervisionBundle` so receipt sources, heuristic routes,
  receipt-aware routes, validator outcomes, and accepted-outcome labels stay
  machine-legible and challengeable, and `psionic-train` now freezes the
  committed evidence bundle at
  `fixtures/tassadar/runs/tassadar_receipt_supervision_v1/receipt_supervision_evidence_bundle.json`
  plus the joined route-quality / refusal-quality / accepted-outcome report at
  `fixtures/tassadar/reports/tassadar_receipt_supervision_report.json`; the
  lane remains research-only planner learning and does not move authority
  closure or settlement truth into the planner
- the final article-parity closeout audit now exists at
  `docs/audits/2026-03-17-tassadar-article-parity-closeout-audit.md`; it is
  explicitly subordinate to `fixtures/tassadar/reports/tassadar_acceptance_report.json`
  and now records the green article-parity closeout at the committed
  benchmark-corpus scope
- landed crate surfaces:
  - `psionic-runtime::tassadar`
  - `psionic-models::TassadarExecutorFixture`
  - `psionic-models::TassadarCompiledProgramExecutor`
  - `psionic-models::TassadarCompiledProgramSuiteArtifact`
  - `psionic-environments::TassadarEnvironmentBundle`
  - `psionic-eval::run_tassadar_reference_fixture_benchmark`
  - `psionic-eval::run_tassadar_article_class_benchmark`
  - `psionic-runtime::build_tassadar_execution_evidence_bundle`
  - `psionic-serve::LocalTassadarExecutorService`
  - `psionic-serve::LocalTassadarPlannerRouter`
  - `psionic-train::train_tassadar_small_executor`
  - `psionic-research::ExperimentFamily::ExecutorVariants`
  - `psionic-research::ExperimentFamily::ExecutorCircuitResearch`
- `psionic-runtime::TassadarSparseTopKRunner`
- strategic value: giving larger reasoning systems inner exact-computation
  ability

The current non-goals are:

- not current MVP compute-market product scope
- not kernel or Nexus authority work
- not app-owned UX or orchestration work
- not a claim that native CPU execution is being replaced

Phase 0 through Phase 9D are now tracked in
[#3743](https://github.com/OpenAgentsInc/openagents/issues/3743),
[#3744](https://github.com/OpenAgentsInc/openagents/issues/3744), and
[#3745](https://github.com/OpenAgentsInc/openagents/issues/3745), and
[#3746](https://github.com/OpenAgentsInc/openagents/issues/3746), and
[#3747](https://github.com/OpenAgentsInc/openagents/issues/3747), and
[#3748](https://github.com/OpenAgentsInc/openagents/issues/3748), and
[#3749](https://github.com/OpenAgentsInc/openagents/issues/3749), and
[#3760](https://github.com/OpenAgentsInc/openagents/issues/3760), and
[#3761](https://github.com/OpenAgentsInc/openagents/issues/3761), and
[#3762](https://github.com/OpenAgentsInc/openagents/issues/3762), and
[#3763](https://github.com/OpenAgentsInc/openagents/issues/3763), and
[#3764](https://github.com/OpenAgentsInc/openagents/issues/3764), and
[#3765](https://github.com/OpenAgentsInc/openagents/issues/3765), and
[#3766](https://github.com/OpenAgentsInc/openagents/issues/3766), and
[#3767](https://github.com/OpenAgentsInc/openagents/issues/3767).

## System Status At A Glance

| Area | Current Status | Current Repo Truth |
| --- | --- | --- |
| Local inference substrate | `implemented_early` | runtime, backend, model, and serve crates exist with CPU and partial Metal lanes |
| Clustered serving substrate | `implemented_early` | `psionic-cluster` owns ordered state, placement, catch-up, and sharded serving topology truth |
| Datastream and artifact staging | `implemented_early` | resumable manifests, policy-weight broadcast refs, freshness enforcement, chunk transport, and delivery receipts exist in `psionic-datastream` |
| Data contracts | `implemented_early` | `psionic-data` now owns versioned dataset manifests, tokenizer digests, split declarations, streamed iteration, and long-context packing policies |
| Sandbox execution | `implemented_early` | bounded execution, runtime detection, background jobs, file transfer, warm reusable pools, staged loop inputs, and repeated agentic iteration receipts exist in `psionic-sandbox` |
| Execution proof bundles | `implemented_early` | canonical execution-proof bundles live in `psionic-runtime` |
| Framework-core autodiff | `implemented_early` | `psionic-ir` now owns autodiff-aware graph construction, a built-in operator registry with explicit schema, implementation-family, meta-execution, and fake-execution capability contracts, an explicit `detach` op, training/evaluation plus no-grad semantics, symbolic reverse-mode backward plans, a declared gradient-support matrix, dense reference materialization, broad current primitive-family gradient coverage, first public `grad` / `value_and_grad` / `vjp` / `jvp` / bounded `vmap` / `checkpoint` transform objects above `AutodiffGraph`, graph-scoped `custom_vjp` hook registration keyed by graph digest plus reverse-mode signature, and typed refusal over current cast/backend-extension transform barriers, plus a fixed-budget trainer integration proof |
| Public lazy-array facade | `implemented_early` | `psionic-array` now owns the first public lazy-array surface above `psionic-core` and `psionic-ir`, with public device and stream handles backed by runtime-owned device truth, honest unified-memory capability flags, explicit stream-dependency policy, graph-backed arithmetic, scalar and filled-array creation helpers, reshape/permute/transpose/slice/select/concat/broadcast view families, explicit runtime determinism contracts with device-scoped seeded random-uniform and random-normal creation, logical dtype casts, `arange` / `linspace` / `eye` helpers, explicit `eval` / deferred `async_eval` semantics over replay-stable graph snapshots, explicit host-owned typed buffer export, singleton `item()` extraction, deterministic tree flatten/map/unflatten utilities, bounded runtime resource reporting with active/peak/cache counters plus cache-limit and reset controls, fallible public Metal- and CUDA-context constructors with bounded actual dense `constant` / `add` / `matmul` execution against the selected runtime device while keeping accelerator numerics explicitly dense-`f32`, bounded backend-debug capture, bounded extension authoring and dispatch-resolution above the extensible operator registry, and an explicit-only implicit-materialization policy |
| Public array artifact IO | `implemented_early` | `psionic-array-io` now owns the first general array save/load surface above the lazy-array layer, with stable `ArrayArtifactReceipt` inventory, explicit dtype and quantization truth, bounded `npy` / `npz` / `safetensors` import-export, and a dense GGUF bridge that keeps export bounded to dense floating-point tensors while dequantizing GGUF block storage to logical `f32` on import instead of hiding that conversion inside model-local loaders |
| Public native function artifact IO | `implemented_early` | `psionic-function-io` now owns the first native `.psifn` function-artifact surface above `psionic-ir` and `psionic-compiler`, plus a bounded `.mlxfn` compatibility shell on top of that native substrate, with export-safe graph contracts, optional compiler artifacts, optional trace-family identity, optional deployment bundle binding, stable import/export receipts, compatibility receipts, and explicit validation that graph, compiler, trace, and deployment digests still describe the same replay-safe function boundary |
| Public distributed groups | `implemented_early` | `psionic-distributed` now owns the first public framework-distributed group layer above runtime mesh truth, with explicit mesh bootstrap from ordered member facts, reusable global-group initialization, honest singleton fallback when no reusable group exists, ordered member/rank snapshots, explicit-plan subgroup split semantics, and machine-readable backend-family capability snapshots over current topology profiles |
| Public distributed collectives | `implemented_early` | `psionic-distributed` now also owns the first bounded public collective-helper layer above `DistributedGroup`, with MLX-style singleton passthrough for `all_sum` / `all_gather` / `reduce_scatter`, explicit host-owned reference emulation for multi-rank `all_sum` / `all_gather` / `reduce_scatter` and `recv`, validation-only `send`, typed collective-support snapshots, and explicit `ring` / `mpi` / `nccl` mapping plus typed `jaccl` refusal instead of pretending backend transport execution is already public |
| Public distributed launch/config planning | `implemented_early` | `psionic-distributed` now also owns a bounded public launch/config planning shell above cluster, sandbox, and mesh truth, with hostfile parsing, honest single-rank-per-node validation, cluster membership/address/backend readiness checks, sandbox contract preflight, per-rank bootstrap payloads and sandbox job plans, distributed reserved-environment synthesis, cluster execution evidence, stable plan digests, and topology-profile-backed backend-family validation instead of a parallel compatibility-only launcher |
| Public distributed gradient helpers | `implemented_early` | `psionic-distributed` now also owns bounded tree-aware data-parallel gradient helpers above the public collective layer, with `grouped_all_sum` / `grouped_all_reduce` small-leaf packing over deterministic tree structure and floating-point `average_gradients` on top of the current reference-emulated all-reduce surface |
| Public distributed tensor-parallel helpers | `implemented_early` | `psionic-distributed` now also owns bounded MLX-style `AllToShardedLinear` and `ShardedToAllLinear` wrappers above the public distributed layer, with deterministic row/column sharding from bounded `psionic-nn::Linear`, inspectable shard-layout snapshots, local shard-input splitting, bias-slice versus bias-owner semantics, and reference-emulated multi-rank `ShardedToAllLinear` reconstruction that requires explicit rank wrappers and shard inputs instead of pretending backend transport is already public |
| Public distributed FSDP helpers | `implemented_early` | `psionic-distributed` now also owns a bounded MLX-style `fsdp_apply_gradients` helper above the public distributed and train-contract layer, with typed `zero_stage3` admission, mixed replicated/full-shard group handling, explicit remote-rank parameter-state and gradient-batch maps for reference emulation, optional global-norm clipping, shard-local optimizer updates with residency transitions, gathered full-parameter reconstruction, and stable apply receipts instead of inventing a trainer-private distributed update path |
| Collectives | `implemented_early` | elastic device-mesh observation, bandwidth-aware local/global sync planning, and benchmark-gated collective cadence receipts exist in `psionic-collectives` |
| Train recovery substrate | `implemented_early` | checkpoint, live-recovery, elastic-membership session truth, explicit checkpoint manifests or pointers, and restore receipts exist in `psionic-train` |
| Training run graph | `implemented_early` | `psionic-train` now owns typed training runs, stage-program identity, contributor-set revisions, topology revisions, participant lifecycle, and window transitions |
| Training orchestrator | `implemented_early` | `psionic-train` now owns typed window-control, assignment-posture, rollout-assignment refs, rollout-admission receipts, bounded off-policy freshness budgets, rollout-worker heartbeats, claims, upload receipts, curriculum receipts, instability verdicts, and trainer-batch assembly requests over the run graph |
| Adapter lineage | `implemented_early` | adapter identity, packaging, and hosted binding lineage exist in `psionic-adapters` |
| Eval runtime | `implemented_early` | `psionic-eval` now owns held-out eval runs, rubric-scored sample/runtime contracts, benchmark packages, repeat-run aggregation, and operator-local validator simulation, while kernel/Nexus now own canonical eval-run plus accepted-outcome authority truth |
| Environment package runtime | `implemented_early` | `psionic-environments` now owns the runtime ABI, typed workload/policy/difficulty/benchmark package shape, tool/rubric hooks, expected artifact contracts, deterministic reference sessions, digest-pinned package aliases, mixed-surface composition groups, and train/eval parity receipts, while kernel/Nexus now own environment, checkpoint-family, validator-policy, benchmark-package, and training-policy registry truth |
| Training core reference loop | `implemented_early` | `psionic-nn` now owns reusable module, parameter, buffer, deterministic state-tree/state-dict semantics, strict/non-strict keyed load behavior with explicit size-mismatch refusal, and a bounded eval-oriented quantized-module shell for supported weight families, while `psionic-train` owns the typed fixed-budget trainer-step path with parameter-group scaling semantics, scheduler bindings, optimizer state, residency transitions, checkpoint/model-IO state roundtrip, checkpoint restore lineage, and step telemetry; broader distributed trainer completion is still planned |
| AttnRes model-family pilot | `implemented_early` | `psionic-models`, `psionic-runtime`, `psionic-train`, and `psionic-eval` now own the bounded CPU-reference AttnRes family with config/weight identity, diagnostics snapshots, two-phase parity truth, tiny next-token training, checkpoint export/restore, and held-out loss-routing receipts, while `psionic-research` now persists a residual-vs-AttnRes comparison bundle and `psionic-serve` exposes the same model/runtime truth through a bounded local text-generation contract instead of a duplicate serving engine |
| Full synthetic-data or research loop | `partial_outside_psionic` | synthetic-data job and verification flows now exist in kernel/Nexus, but no Psionic-native generation runtime or research-loop crate family exists yet |
| Executor-class in-model compute lane | `implemented_early` | WebAssembly-first, CPU-reference-first `Tassadar` reference lane now exists in `psionic-runtime`, `psionic-models`, `psionic-environments`, `psionic-eval`, `psionic-serve`, `psionic-train`, and `psionic-research` with machine-legible `core_i32_v1`, widened `core_i32_v2`, dedicated article-shaped `tassadar.wasm.article_i32_compute.v1`, `tassadar.wasm.sudoku_v0_search.v1`, `tassadar.wasm.hungarian_v0_matching.v1`, `tassadar.wasm.hungarian_10x10_matching.v1`, and `tassadar.wasm.sudoku_9x9_search.v1` Wasm profiles, an append-only trace ABI, profile-aware CPU reference and fixture runners, a real 4x4 backtracking Sudoku search-program representation on the CPU reference lane, a real split-aware 4x4 Sudoku-v0 corpus with exact CPU-reference traces per puzzle, explicit `HullCache` fast path for the validated acyclic subset, a validated `SparseTopK` decode path on its own bounded subset, exact CPU/reference-linear/hull-cache/sparse-top-k equivalence harnesses, typed refusal surfaces including backward-branch and sparse-shape fallback truth, machine-legible runtime capability reports, machine-legible Wasm instruction/profile coverage reporting with typed unsupported-opcode examples, a Rust-only article frontend canon with committed Rust source fixtures plus stable source/toolchain/config/output lineage across the kernel, heap-input, long-loop, Hungarian, and Sudoku families, a public Rust-to-Wasm article profile completeness publication with explicit supported and refused module-shape, control-flow, table/global/indirect-call, numeric, and ABI rows projected into both the Tassadar environment bundle and served capability publication, a bounded Rust-only article ABI closure lane over direct scalar `i32` entrypoints plus pointer-length `i32` heap-input entrypoints with one direct scalar `i32` return on the committed canonical fixtures and the committed report `fixtures/tassadar/reports/tassadar_article_abi_closure_report.json`, one canonical Rust-only Hungarian-10x10 article reproducer root at `fixtures/tassadar/runs/hungarian_10x10_article_reproducer_v1` plus the committed closeout report `fixtures/tassadar/reports/tassadar_hungarian_10x10_article_reproducer_report.json`, one canonical Rust-only Sudoku-9x9 article closeout report at `fixtures/tassadar/reports/tassadar_sudoku_9x9_article_reproducer_report.json` bound to the exact compiled `sudoku_9x9_test_a` search deployment already persisted under `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0`, one Rust-only article runtime closeout bundle at `fixtures/tassadar/runs/article_runtime_closeout_v1/article_runtime_closeout_bundle.json` plus the committed eval and research reports `fixtures/tassadar/reports/tassadar_article_runtime_closeout_report.json` and `fixtures/tassadar/reports/tassadar_article_runtime_closeout_summary.json`, freezing exact direct-reference-linear CPU execution for the committed `rust.long_loop_kernel` and `rust.state_machine_kernel` families at the declared `million_step` and `two_million_step` horizons with explicit throughput-floor checks and explicit `HullCache` / `SparseTopK` fallback-only posture, plus a served runtime-closeout publication in `psionic-serve` that cites the summary without widening the generic served Wasm capability matrix, one canonical direct model-weight execution proof report at `fixtures/tassadar/reports/tassadar_direct_model_weight_execution_proof_report.json` over the committed article workloads `long_loop_kernel`, `sudoku_v0_test_a`, and `hungarian_matching`, with route-bound proof receipts that require direct-guaranteed routing, requested-versus-effective decode parity, zero external calls, and no CPU-substitution markers plus a provider-facing receipt projection instead of treating a completed session as sufficient proof for undeclared workloads or later routes, and one `psionic-serve` owned Rust-only article reproduction harness at `fixtures/tassadar/reports/tassadar_rust_only_article_reproduction_report.json` plus `scripts/check-tassadar-rust-only-article-reproduction.sh` and `docs/TASSADAR_RUST_ONLY_ARTICLE_RUNBOOK.md`, turning the current Rust-only article path into one executable operator procedure without widening semantics beyond the committed source canon, ABI closure, Hungarian, Sudoku, runtime, and direct-proof receipts, a historical `clang`-backed C-to-Wasm compile receipt path over the committed `tassadar_micro_wasm_kernel.c` fixture plus emitted Wasm binary with source/toolchain/config/output digests and canonical executor-artifact lineage, a real compile-pipeline matrix over committed Wasm-text multi-export arithmetic, memory-lookup, and parameter-ABI cases plus a typed missing-toolchain C-source refusal with compile-receipt digests and exact-vs-refused source-to-Wasm-to-Tassadar outcomes, a bounded normalized Wasm-module IR in `psionic-ir` with section-level round-trip truth plus compiler lowering of zero-parameter straight-line exported functions into digest-bound runtime artifacts and a committed ingress report over the canonical micro binary plus one seeded multi-function module, a real bounded module-execution conformance/differential harness against `wasmi` over curated and deterministically generated cases with exact success/trap parity and explicit unsupported-host boundary refusal, direct/fallback/refused decode selection diagnostics, digest-bound program artifacts, explicit model/program compatibility descriptors, typed environment bundles, package-driven exactness benchmark suites over both the validation corpus and the widened article-class corpus (`MicroWasmKernel`, `SudokuClass`, `HungarianMatching`) with CPU/reference-linear/hull-cache/sparse-top-k reporting and runtime capability/selection artifacts, emitted trace artifacts, runtime-manifest lineage, canonical proof-bundle integration, an explicit `psionic.executor_trace` served request/stream/terminal contract, a specialized `psionic.article_executor_session` served surface for canonical article workloads with benchmark/proof identity plus derived readable-log and symbolic token-trace views, a planner-owned `psionic.planner_executor_route` contract with preflight and replay-stable routing truth, a specialized `psionic.article_hybrid_workflow` contract for planner-owned article compute spans with preserved benchmark identity plus routing/proof receipts, a bounded small-model training lane with proof-aware exactness receipts over the validation corpus, a program-specialized compiled-weight deployment path with exact program binding and compile-time proof lineage, and typed research families that run benchmark/proof/lineage-backed executor variant sweeps plus learned-circuit research comparisons against the handcrafted and trained-small baselines while keeping claim boundaries explicit; it is still not current MVP product scope |

Recent issue closure changed one important reading of this table:

> environment packages, checkpoint-family policies, validator policies,
> benchmark packages, training policies, eval runs, training runs, accepted
> outcomes, and synthetic-data authority flows now exist in the broader
> OpenAgents stack, and Psionic now owns the first environment plus eval runtime
> crates, but broader generation loops still remain unfinished.

## Canonical Layer Model

Psionic should be understood as a layered subtree with clear dependency
direction.

### System Diagram

```text
Applications / Operators / Authority
        |
        v
  psionic-provider
        |
        v
 psionic-serve / psionic-models
        |
        v
 psionic-train / psionic-eval / psionic-data / psionic-collectives / psionic-adapters
        |
        v
 psionic-cluster / psionic-datastream / psionic-sandbox / psionic-net
        |
        v
 backend crates
        |
        v
 psionic-runtime / psionic-compiler / psionic-ir / psionic-core
```

### Layering By Crate

1. `psionic-core`
   - foundational tensor, dtype, shape, device, layout, view-semantics, and
     cross-library refusal-taxonomy types
2. `psionic-ir`
   - canonical graph, built-in plus extensible operator registry, custom-op
     schema and backend-dispatch registration contracts, transform-safety and
     functionalization contracts, dense plus non-dense meta-tensor family
     contracts, fake/meta execution and plan validation contracts,
     detach/no-grad/autodiff tracking, symbolic backward plans, and
     execution-plan representation
3. `psionic-compiler`
   - lowering, schedule-formation, fusion-policy, memory-plan, and
     plan-cache-identity boundaries over IR, plus the first public
     compile-transform surface with explicit purity, concrete-plan cache
     identity, bounded shapeless trace-family identity, trace capture, and
     plan-debug posture
   - public array-debug capture in `psionic-array` now reuses compiler
     trace/debug configuration instead of inventing a lane-local debug path
   - compiler replay fixtures now guard deterministic lowering, explicit
     schedule/fusion/memory/cache artifacts, and topology-bound program
     identity through
     `scripts/lint/psionic-compiler-replay-gate.sh`
4. `psionic-runtime`
   - runtime traits, runtime planning, execution-proof bundles, training-class
     runtime truth
   - backend-visible buffer storage identity and view-posture contracts
   - runtime-owned RNG, generator-state, checkpoint-restore, and
     deterministic-algorithm contracts
   - same-type local multi-device plan-runner contracts, explicit local
     sharding policy and refusal taxonomy, and local multi-device execution
     evidence kept distinct from clustered execution truth
5. `psionic-sandbox`
   - bounded execution profiles, runtime detection, execution receipts, and
     background-job lifecycle
6. `psionic-net`
   - peer identity, transport sessions, relay-backed rendezvous, trust and
     candidate state
7. `psionic-datastream`
   - resumable manifests, lightweight policy-weight broadcast refs, freshness
     control, chunk transfer, and delivery receipts for artifacts
8. `psionic-data`
   - versioned dataset manifests, tokenizer digests, split declarations,
     streamed iteration, and packing policy contracts
9. `psionic-eval`
   - held-out eval runs, rubric-scored runtime contracts, benchmark packages,
     repeat-run aggregation, and local validator simulation
10. `psionic-cluster`
   - ordered state, cluster admission, catch-up, scheduling, topology and
     placement truth
11. `psionic-collectives`
   - elastic device-mesh, local/global sync planning, transport-feedback
     replanning, and quantized collective policy
12. `psionic-train`
   - training-session truth for checkpointing, live recovery,
     elastic-membership posture, checkpoint pointers/manifests, restore
     receipts, and orchestrator control state
13. `psionic-adapters`
   - adapter identity, packaging, and hosted binding lineage
14. backend crates
   - backend-specific runtime implementations only
15. `psionic-models`
   - reusable model definitions and metadata
16. `psionic-serve`
   - request, response, and execution contracts for served products
17. `psionic-router`
   - reusable multi-model routing inventory, policy filters, and worker-path
     selection for served fleets
18. `psionic-provider`
   - provider-facing capability, readiness, and receipt types at the OpenAgents
     boundary

The crate list and layering are canonical for current ownership and dependency
direction, but they are not a guarantee that every planned subsystem will land
under exactly these final crate names.

### Dependency Direction

- lower crates must not depend on higher product-facing crates
- no crate in `crates/psionic-*` may path-depend on `apps/*`
- reusable engine crates must not own app workflows or market authority
- `psionic-provider` is the boundary adapter, not a place to hide app logic

## Canonical Psionic Work Classes

Psionic needs two different notions of work class:

- product-level execution classes
- low-level runtime scheduling classes

### Product-Level Work Classes

| Work Class | Meaning | Current Status |
| --- | --- | --- |
| Inference | generate model outputs for served requests | `implemented_early` |
| Embeddings | generate vectors or embedding outputs | `implemented_early` |
| Clustered serving | execute inference across replicas or sharded topology | `implemented_early` |
| Sandbox execution | run bounded remote or local sandbox jobs | `implemented_early` |
| Artifact staging | move datasets, checkpoints, served artifacts, and adapter bundles | `implemented_early` |
| Training-class coordination | coordinate checkpoints, recovery, collectives, and elastic membership | `implemented_early` |
| Full training | execute trainer-step and optimizer updates | `planned` |
| Eval | run shared held-out or online evaluation | `planned` |
| Synthetic-data generation | generate or score new data under the same substrate | `planned` |
| Adapter-backed serving | serve a base artifact plus attributed adapter lineage | `implemented_early` |

### Low-Level Runtime Work Classes

These are the scheduler-facing classes already encoded in
`psionic-runtime::RuntimeWorkClass`.

| Runtime Work Class | Meaning |
| --- | --- |
| `DecodeToken` | one latency-sensitive decode step |
| `PrefillBatch` | one prefill or preparation batch |
| `DatastreamChunk` | one chunk transfer over the data plane |
| `CollectiveStep` | one collective or synchronization step |
| `CheckpointFlush` | one checkpoint or persistence flush step |

The system-wide rule is:

> product work classes explain what Psionic is doing for the platform, while
> low-level runtime work classes explain how the runtime schedules the work.

## Canonical System Objects

Psionic needs a stable object vocabulary across serving, staging, sandbox, and
training subsystems.

| Object | Owner | Purpose | Current Status |
| --- | --- | --- | --- |
| `RuntimeWorkItem` | `psionic-runtime` | one low-level schedulable unit of work | `implemented` |
| `ExecutionProofBundle` | `psionic-runtime` | canonical execution evidence for runtime work | `implemented` |
| `LocalRuntimeObservability` + `BackendRuntimeResources` + `CompilePathEvidence` | `psionic-runtime` / `psionic-serve` | machine-legible local-runtime operator truth for execution posture, queue/scheduler posture, backend health, selected-device identity, and compile/cache state | `implemented` |
| `DatastreamManifest` | `psionic-datastream` | full resumable manifest for one artifact stream | `implemented` |
| `DatastreamManifestRef` | `psionic-datastream` | compact artifact reference embedded in other contracts, including explicit distributed KV spill/restore locators | `implemented` |
| `DatastreamPolicyWeightBroadcastManifest` | `psionic-datastream` | lightweight control-plane summary for a multi-shard policy-weight artifact | `implemented_early` |
| `DatasetManifest` | `psionic-data` | versioned dataset, tokenizer, split, and shard-lineage contract | `implemented_early` |
| `DatasetIterationContract` | `psionic-data` | resume-safe split iteration over datastream-backed shards | `implemented_early` |
| `DatasetPackingPolicy` | `psionic-data` | long-context sequence packing and token-budget batch planning contract | `implemented_early` |
| `DataIngressSemanticsReport` | `psionic-data` | machine-readable bounded local data-ingress capability report over dataset source, sampler, batch-sampler, and host-device staging contracts | `implemented` |
| `DistributedDataFeedSemanticsReport` | `psionic-data` | machine-readable bounded fixed-world-size distributed data-feed report over shard partitioning, worker coordination, and replay-safe per-rank ordering contracts | `implemented` |
| `PsionicRefusal` | `psionic-core` | canonical cross-library refusal taxonomy for unsupported op, gradient, layout, capability, serialization, sandbox-policy, and topology boundaries | `implemented_early` |
| `AdvancedDTypeSemanticsReport` | `psionic-core` | machine-readable bounded promotion, cast, and backend-capability matrix for complex, float8, wider integer, and higher-precision real dtype semantics above the compact runtime-core subset | `implemented` |
| `AutocastPolicyMatrixReport` | `psionic-core` | machine-readable bounded autocast-style precision-policy matrix over backend family, preferred low-precision dtype, operator family, numerics diagnostics, and typed refusal posture | `implemented` |
| `QuantizationCapabilitySemanticsReport` | `psionic-core` | machine-readable bounded PTQ, QAT, runtime-execution, compiler-lowering, and export-aware quantization capability matrix above raw file-format decode | `implemented` |
| `OperatorParityMatrixReport` | `psionic-ir` | machine-readable seeded operator parity cases and refusal proofs against the current PyTorch-derived oracle window | `implemented` |
| `AdvancedOperatorProgramMatrixReport` | `psionic-ir` | machine-readable bounded linalg, signal, and attention-family program matrix plus explicit refusal posture for distribution and special-function families | `implemented` |
| `ProgramTransformCapabilityMatrixReport` | `psionic-ir` | machine-readable bounded capability matrix for functionalization, symbolic-rewrite readiness, export-safe graphs, bounded public `checkpoint`/`vmap`/`jvp`, and explicit remaining higher-order transform refusal | `implemented` |
| `ExportableGraphContract` | `psionic-ir` | machine-readable export-safe graph envelope with named input/output bindings for downstream packaging and deployment | `implemented` |
| `ExtensionContractSemanticsReport` | `psionic-ir` | machine-readable bounded contract surface for custom ops, kernels, autograd, backend plugins, and quantizer plugins above the extensible registry | `implemented` |
| `TensorFamilyCapabilityMatrixReport` | `psionic-ir` | machine-readable capability and refusal matrix for dense, sparse, nested, masked, and storage-aware tensor-family semantics across meta, declared-output, alias-view, and runtime-materialization surfaces | `implemented` |
| `ArrayDevice` + `ArrayStream` + `ArrayContext` + `Array` + `EvaluatedArray` + `PendingAsyncEval` + `ArrayMemoryCounters` + `ArrayRuntimeResourceReport` + `ArrayCacheLimitControl` + `ArrayCacheResetReceipt` | `psionic-array` | first public lazy-array facade above `psionic-core` and `psionic-ir`, including runtime-backed device truth, unified-memory capability flags, explicit stream-dependency policy, context-owned graph construction, graph-backed arithmetic, scalar and filled-array creation helpers, reshape/permute/transpose/flatten/expand_dims/squeeze/slice/select/concat/broadcast view families, explicit runtime determinism contracts with seeded random creation, logical dtype casts, `arange` / `linspace` / `eye` helpers, axis-aware sum reduction, explicit `eval` / deferred `async_eval` semantics, bounded runtime resource reporting with active/peak/cache counters, explicit cache-limit and reset controls, fallible `ArrayContext::metal()` / `metal_seeded()` and `ArrayContext::cuda()` / `cuda_seeded()` constructors, bounded actual Metal and CUDA execution for dense `constant` / `add` / `matmul` graphs with dense-`f32` numerics disclosure, and explicit-only materialization boundaries over replay-stable graph snapshots | `implemented_early` |
| `ArrayArtifactReceipt` + `encode_*` / `decode_*` + `save_*_path` / `load_*_path` | `psionic-array-io` | public array artifact IO above `psionic-array`, including `npy`, `npz`, `safetensors`, and bounded dense GGUF import/export with explicit receipt inventory, dtype truth, and GGUF quantization-to-dense import disclosure | `implemented_early` |
| `FunctionArtifact` + `FunctionCompileBundle` + `FunctionArtifactReceipt` + `MlxfnCompatibilityReceipt` + `encode_function_artifact` / `decode_function_artifact` + `encode_mlxfn_function_artifact` / `decode_mlxfn_function_artifact` + `save_*_path` / `load_*_path` | `psionic-function-io` | public native `.psifn` function artifact IO above `psionic-ir` and `psionic-compiler`, plus a bounded `.mlxfn` compatibility shell on top of the native artifact, including export-safe graph contracts, optional compiler artifacts, optional trace-family identity, optional deployment bundle binding, stable artifact digests, import/export receipts, compatibility receipts, and explicit replay-safe validation with typed refusal outside the current `.mlxfn` subset | `implemented_early` |
| `CompilerHygieneParityMatrixReport` | `psionic-compiler` | machine-readable seeded symbolic-shape, fake-tensor, and compiler-hygiene parity cases including one bounded shapeless trace-family identity seed plus explicit symbolic-shape and reshape-formula refusal proofs for the current PyTorch-derived oracle window | `implemented` |
| `DeploymentArtifactContract` + `ExportDeploymentArtifactSemanticsReport` | `psionic-compiler` | machine-readable deployment bundle contract and bounded report for execution-plan and topology-aware graph-first artifacts | `implemented` |
| `SemanticsClaimReport` | `psionic-compat` | machine-readable claim vocabulary that separates seeded evidence from `PyTorch-credible` and `PyTorch-compatible later` posture across the current semantics program | `implemented` |
| `MlxCompatibilityScopeReport` | `psionic-compat` | machine-readable bounded upstream MLX version window and claim-language contract that keeps `MLX-class` distinct from later `MLX-compatible` facades | `implemented` |
| `MlxCpuReferenceCoverageReport` | `psionic-array` | machine-readable bounded CPU-reference coverage contract over imported MLX `array_core`, `ops_numeric`, and `device_eval_memory` families, with seeded supported cases and typed refusal posture above the public array surface | `implemented` |
| `MlxAcceptanceMatrixReport` | `psionic-compat` | machine-readable MLX-lane closure contract over array/runtime, transform/compile, `nn`/optimizer, export/tooling, distributed, and backend-closure categories | `implemented` |
| `MlxParityHarnessReport` | `psionic-compat` | machine-readable seeded upstream MLX test-family harness carrying bounded `pass`, `refusal`, and `unsupported` outcomes tied to repo-owned Psionic hooks | `implemented` |
| `MlxCompatibilityMatrixReport` | `psionic-compat` | machine-readable supported/convertible/unsupported adoption matrix that keeps current MLX claims bounded to governance support, explicit bridges, and intentionally unsupported public surfaces | `implemented` |
| `psionic-mlx-compat::core::Context` + `nn` + `optimizers` + `transforms` + `io` + `distributed` + `reports` | `psionic-mlx-compat` | optional bounded MLX naming and module-layout facade over supported Psionic-native array, transform, `nn`, optimizer, `.mlxfn`, distributed, and compatibility-report surfaces without changing execution ownership, plus runnable migration examples that show how to stay bounded or drop straight to the native crates | `implemented_early` |
| `psionic_mlx_capi_*` + `psionic_mlx_capi.h` + `compatibility_*_json_string` + `eval_json_string` | `psionic-mlx-capi` | optional bounded C ABI over `psionic-mlx-compat`, including owned JSON exports for compatibility reports and one JSON-driven dense-array eval bridge over the native facade for later Python or Swift consumers, plus safe Rust helpers that mirror the same JSON contract for examples and adapter code | `implemented_early` |
| `MlxLmTextRuntime` + `MlxLmTextRequest` + `MlxLmLoadReport` + `MlxLmPromptCacheArtifact` + `MlxLmBatchReport` + `psionic-mlx-lm` CLI | `psionic-mlx-lm` | bounded local MLX-lm-style text package over the native GGUF loader, prompt renderer, generation runtime, and scheduler, including load reports, chat rendering, generate/stream/batch workflows, explicit context-overflow and prefix-cache controls, and persisted prompt-cache artifacts without adding a second runtime | `implemented_early` |
| `MlxCatalogWorkspace` + `MlxCatalogResolutionReport` + `HuggingFaceCacheSnapshot` + `MlxArchitectureRegistry` + `MlxRemoteMetadataPolicy` + `MlxConversionEntryPoint` + `psionic-mlx-catalog` CLI | `psionic-mlx-catalog` | bounded MLX-style model-catalog package over `psionic-catalog`, `psionic-models`, and `psionic-mlx-lm`, including direct local GGUF resolution, local Ollama manifest resolution, local Hugging Face cache discovery, builtin architecture registration, conversion-entrypoint reporting, explicit processor/template metadata trust or refusal, and direct GGUF handoff into the text package without inventing a parallel loader stack | `implemented_early` |
| `MlxTextServeWorkspace` + `MlxTextServeConfig` + `MlxTextServeBootstrapReport` + `psionic-mlx-serve` CLI | `psionic-mlx-serve` | bounded MLX-style text-serving package over `psionic-mlx-catalog`, `psionic-router`, and `psionic-serve`, including MLX-style model-reference resolution into the shared OpenAI-compatible server, machine-readable bootstrap reports, explicit response-state storage selection, and honest current lifecycle truth (`loaded`, `warm/unload not implemented`) instead of a duplicate serving stack | `implemented_early` |
| `MlxVlmProcessorRegistry` + `MlxVlmInputPart` + `MlxVlmProjectionReport` + `MlxVlmServePlan` + `psionic-mlx-vlm` CLI | `psionic-mlx-vlm` | bounded MLX-style multimodal package over `psionic-models` and the shared text-serving lane, including builtin processor registries for `llava`, `qwen2_vl`, and `omni`-class families, OpenAI-compatible image/audio/video input parts, digest-bound attachment receipts, prompt projection into shared `PromptMessage`s, and served-request planning for `/v1/responses` and `/v1/chat/completions` without claiming a native multimodal encoder | `implemented_early` |
| `MlxAudioModelRegistry` + `MlxAudioClip` + `MlxTextToSpeechRequest` + `MlxSpeechToSpeechRequest` + `MlxAudioSpeechRequest` + `MlxAudioSynthesisReport` + `psionic-mlx-audio` CLI | `psionic-mlx-audio` | bounded MLX-style audio package with builtin `kokoro`/`xtts`/`encodec`-class family metadata, quantized-checkpoint descriptors, WAV IO, codec normalization helpers, CPU-reference text-to-speech and speech-to-speech lanes, explicit stream chunks, and one server-facing speech request/response contract without claiming a production neural speech runtime | `implemented_early` |
| `MlxRecipeWorkspace` + `MlxRecipeConfig` + `MlxRecipePlan` + `MlxRecipeMethodSummary` + `psionic-mlx-recipes` CLI | `psionic-mlx-recipes` | bounded MLX-style training-recipe package above `psionic-train`, including machine-readable method inventory, explicit stage mapping for SFT/preference/RL families, adapter-plan projection into the open adapter lane, rollout-validator posture for RL-style methods, and plan emission without creating a second trainer architecture | `implemented_early` |
| `MlxWorkflowWorkspace` + synthetic dataset specs/artifacts + supervision-helper plans + adapter-merge artifacts + publish manifests | `psionic-mlx-workflows` | bounded MLX-style workflow package above `psionic-data`, `psionic-mlx-recipes`, `psionic-train`, and portable model IO, including deterministic synthetic SFT/preference dataset bundles, reward/judge helper plans, adapter merge/export artifacts, and one local Hugging Face-style publish snapshot with explicit GGUF refusal | `implemented_early` |
| `MlxBenchWorkspace` + `MlxBenchmarkSuite` + `MlxTextBenchmarkProvider` + `MlxServedBenchmarkProvider` + `MlxBenchmarkRunReceipt` | `psionic-mlx-bench` | bounded MLX-style benchmark package above `psionic-eval`, `psionic-mlx-lm`, and `psionic-mlx-vlm`, including suite manifests, repeated benchmark receipts, local text and served provider adapters, and multimodal prompt projection without introducing a second eval runtime | `implemented_early` |
| `Module` | `psionic-nn` | reusable nested module tree with deterministic parameter, buffer, and submodule traversal, explicit trainable versus frozen posture, recursive freeze/unfreeze helpers, and bounded public `save_weights` / `load_weights` wrappers | `implemented` |
| `ModuleParityMatrixReport` | `psionic-nn` | machine-readable seeded module parity cases and refusal proofs for the current PyTorch-derived normalized module-tree and `state_dict` oracle window | `implemented` |
| `ModuleStateDict` | `psionic-nn` | deterministic keyed `state_dict` and saved-weights view with stable path order and persistent-vs-all-buffer selection | `implemented` |
| `ModuleStateTree` | `psionic-nn` | digest-bound flattened parameter or buffer view that downstream train, checkpoint, and compatibility code can consume | `implemented` |
| `ModuleStateLoadReport` | `psionic-nn` | explicit strict/non-strict load receipt returned by bounded public `load_weights` behavior, with loaded, missing, unexpected, and digest-transition facts | `implemented` |
| `NnTensor` | `psionic-nn` | bounded dense cpu-f32 layer input/output wrapper above `TensorSpec` plus `TensorData` | `implemented_early` |
| `Linear` + `Embedding` + `LayerNorm` + `RmsNorm` + `Activation` + `Dropout` + `Conv1d` + `Conv2d` + `Pool1d` + `Pool2d` | `psionic-nn` | bounded public CPU-reference core layer surface built above the shared module/state substrate | `implemented_early` |
| `LossReduction` + `mse_loss` + `l1_loss` + `binary_cross_entropy_loss` + `cross_entropy_loss` + `softmax_last_dim` + `log_softmax_last_dim` + `sigmoid` + `one_hot` + `InitKind` + `init_tensor` + `init_parameter` | `psionic-nn` | bounded public CPU-reference losses, initializers, and helper functions for tiny training loops above the shared module/state substrate | `implemented_early` |
| `OptimizerKind` + `OptimizerConfig` + `SchedulerKind` + `SchedulerConfig` + `SchedulerBinding` + `ParameterGroupSemantics` + `Optimizer` + `OptimizerStateSnapshot` + `OptimizerModuleStepReport` + `OptimizerGroup` + `MultiOptimizer` + `MultiOptimizerStepReport` | `psionic-nn` | bounded public optimizer-and-scheduler shell above `psionic-train` math with module-path keyed state, explicit frozen-parameter handling, parameter-group scaling, multi-optimizer composition, snapshot restore, and per-step receipts | `implemented_early` |
| `ModuleQuantizeConfig` + `Module::quantize` + `QuantizedModule` + `QuantizedLinear` + `QuantizedEmbedding` | `psionic-nn` | bounded eval-oriented quantized-module shell with explicit keep-dense versus strict posture, frozen quantized module reports, and dequantize-to-`f32` forward semantics for supported linear and embedding families | `implemented_early` |
| `OptimizerParityMatrixReport` | `psionic-train` | machine-readable seeded optimizer parity cases and refusal proofs for the current PyTorch-derived single-step optimizer oracle window | `implemented` |
| `GradientScalingSemanticsReport` | `psionic-train` | machine-readable bounded train-class mixed-precision report for fp16 dynamic loss scaling, overflow/underflow handling, bf16 no-scaling posture, and typed refusal boundaries | `implemented` |
| `ReproducibilitySemanticsReport` | `psionic-train` | machine-readable framework-wide replay seed, deterministic-mode, generator-derivation, and checkpoint-restore report across training replay and runtime determinism contracts | `implemented` |
| `BufferStorageContract` | `psionic-runtime` | backend-visible storage identity and logical view posture for one realized buffer | `implemented_early` |
| `RuntimeDeterminismContract` | `psionic-runtime` | runtime-owned RNG, generator-state, checkpoint-snapshot, and deterministic-algorithm contract for replayable execution | `implemented_early` |
| `RuntimeManifest` | `psionic-runtime` proof layer | digest-bound package for artifact, static-config, mutable-variable, and runtime lineage used at execution time | `implemented_early` |
| `DatastreamDeliveryReceipt` | `psionic-datastream` | verified proof of delivered bytes and chunk progress | `implemented` |
| `ClusterState` | `psionic-cluster` | authoritative cluster membership and ordered-state truth | `implemented` |
| `SessionClaimsBundle` | `psionic-net` / proof layer | session-scoped claims bound into the authenticated transport payload so peer identity carries runtime-manifest and proof posture in machine-legible form | `implemented_early` |
| `TrainingCheckpointReference` | `psionic-runtime` | stable identity for one training checkpoint | `implemented` |
| `TrainingRecoveryContext` | `psionic-runtime` | runtime-visible recovery posture for training-class execution | `implemented` |
| `TrainingDeviceMeshContext` | `psionic-runtime` | runtime-visible elastic device-mesh posture | `implemented` |
| `TrainingCollectiveContext` | `psionic-runtime` | runtime-visible collective posture and benchmark evidence | `implemented` |
| `ModelIoCompatibilityContract` | `psionic-train` | machine-readable boundary contract for supported and unsupported checkpoint/model portability surfaces | `implemented` |
| `AdapterArtifactIdentity` | `psionic-adapters` | stable identity for one adapter artifact | `implemented` |
| `AdapterPackageManifest` | `psionic-adapters` | package manifest for adapter bytes tied to datastream | `implemented` |
| `ProviderSandboxExecutionReceipt` | `psionic-sandbox` | receipt for one bounded sandbox run | `implemented` |
| `TrainingRun` | `psionic-train` | root identity, participant graph, and lifecycle state for one training program | `implemented_early` |
| `TrainingWindow` | `psionic-train` | one synchronized contribution or trainer interval with contributor-set and transition state | `implemented_early` |
| `TrainingSchedulerBinding` | `psionic-train` | typed scheduler config plus mutable per-group scheduler state for optimizer-step resolution | `implemented` |
| `TrainerBatchAssemblyRequest` | `psionic-train` | lightweight control-plane request for one trainer batch over rollout refs | `implemented_early` |
| `RolloutTaskClaim` | `psionic-train` | deterministic task-claim contract for one rollout assignment under one worker heartbeat | `implemented_early` |
| `RolloutAdmissionReceipt` | `psionic-train` | typed acceptance, quarantine, or discard receipt for one rollout artifact under bounded off-policy policy | `implemented_early` |
| `RolloutWorkerOutcomeReceipt` | `psionic-train` | typed claim-expiry, upload-policy, or orchestrator-wrapped outcome receipt for one rollout worker | `implemented_early` |
| `RolloutVerificationBundle` | `psionic-train` | validator-ready bundle for one rollout artifact, worker outcome, and optional benchmark evidence | `implemented_early` |
| `ValidatorVerdict` | `psionic-train` | typed validator outcome over one rollout bundle, including replay, duplicate, normalization, and benchmark checks | `implemented_early` |
| `CollectiveSyncCadenceReceipt` | `psionic-collectives` | typed cadence, transport-feedback, and replan-trace receipt for one sync step | `implemented_early` |
| `CheckpointPointer` | `psionic-train` | stable pointer to the latest accepted checkpoint for a run, stage, or window | `implemented_early` |
| `CheckpointManifest` | `psionic-train` | typed shard, digest, writer, and durability description for one checkpoint flush | `implemented_early` |
| `EnvironmentPackage` | `psionic-environments` | reusable task, rubric, tool, dataset, and artifact environment package | `implemented_early` |
| `EnvironmentBenchmarkProfile` | `psionic-environments` | validator- or operator-reusable benchmark profile bound into one environment package | `implemented_early` |
| `BenchmarkPackage` | `psionic-eval` | validator-owned packaged benchmark harness or reference evaluation profile with repeat-run aggregation | `implemented_early` |
| `EvalRun` | `psionic-eval` | one local evaluation execution over a declared environment and artifact set | `implemented_early` |

The important point is not that every object already exists. The important
point is that Psionic should converge on a typed object model rather than
passing loosely structured blobs between subsystems.

Psionic enforces capability envelopes at runtime, while higher-level compute
products define the admissible execution contract exposed to buyers, operators,
and authority layers.

## Glossary

| Term | Meaning |
| --- | --- |
| execution truth | what the Psionic runtime and cluster can honestly say happened at execution time |
| authority truth | what higher-level OpenAgents services accept as canonical outcome |
| artifact truth | what manifests, digests, package refs, and staged bytes were actually bound to execution |
| runtime identity | the verified execution origin responsible for a work item |
| session claims bundle | the signed session-scoped claim set that ties peer or session keys to runtime and artifact identity |
| training window | one bounded contributor or trainer interval with explicit control-plane state |
| checkpoint lineage | the chain of checkpoint identities, manifests, and durability transitions that define recoverable train state |
| checkpoint pointer | the stable reference to the latest accepted checkpoint for a run, stage, or window |
| checkpoint manifest | the typed shard, digest, writer, and durability description for one checkpoint flush |
| policy revision | the specific weight or policy version a worker, trainer, or eval run consumed |
| environment package | a versioned task, rubric, tool, and sandbox contract used by training or eval |
| benchmark package | a validator-owned packaged benchmark or reference evaluation profile reused for repeatable scoring |
| proof posture | the declared strength and availability of execution evidence |
| validator posture | the declared verification policy and adjudication expectations for a workload |
| manifest registry | a versioned allowlist or policy registry for manifests, proof profiles, or environment packages |
| receipt | the typed record of an accepted state transition or outcome |
| collective posture | the mesh, communication, quantization, and benchmark facts attached to one collective step |

## Artifact Model

Psionic is also an artifact system, not only an execution engine.

### Canonical Artifact Families

| Artifact | Current Carrier | Meaning |
| --- | --- | --- |
| Served artifact | `DatastreamSubjectKind::ServedArtifact` | model or sharded serving artifact used for inference |
| Checkpoint | `DatastreamSubjectKind::Checkpoint` plus `TrainingCheckpointReference` | recoverable training or optimizer state |
| Tokenized corpus | `DatastreamSubjectKind::TokenizedCorpus` | tokenized dataset shard delivered for training or eval |
| Eval bundle | `DatastreamSubjectKind::EvalBundle` | benchmark or evaluation harness artifact |
| Benchmark package | `psionic-eval` | validator-owned packaged benchmark harness or reference evaluation profile |
| Adapter package | `DatastreamSubjectKind::AdapterPackage` plus adapter manifests | adapter or LoRA artifact delivered with lineage |
| Proof artifact | execution-proof bundle or augmentation | evidence about what the runtime or cluster actually did |
| Sandbox artifact | sandbox input/output digest sets | staged inputs and outputs of bounded execution |
| Environment package | `psionic-environments` | versioned task, tool, rubric, dataset, and sandbox contract |

### Artifact Rules

- artifacts should be digest-bound
- artifacts should be referenceable through compact manifest refs where
  possible
- runtime and environment identity should distinguish digest-bound measured or
  static config from mutable runtime variables
- artifacts should carry enough lineage to explain what execution actually
  consumed
- policy-meaningful lanes should reference versioned manifest or profile
  registries rather than opaque free-form strings
- Psionic should not rely on unnamed side files for economically or
  operationally important artifacts

## Receipts And Truth Boundaries

Psionic is receipt-first, but it is not authority-first.

The tree should be understood through four truth domains.

| Truth Domain | Owned By | What It Says |
| --- | --- | --- |
| Runtime truth | `psionic-runtime` and lower execution crates | what device, work class, and proof posture actually ran |
| Artifact truth | `psionic-datastream`, `psionic-adapters`, `psionic-eval`, and `psionic-environments` | what bytes, manifests, packages, and digests were actually staged or referenced |
| Cluster and sandbox truth | `psionic-cluster`, `psionic-sandbox`, `psionic-collectives`, `psionic-train` | what topology, recovery posture, sandbox runtime, and collective decisions actually occurred |
| Authority truth | outside Psionic in kernel and control services | what the platform or market accepts as final outcome |

The key boundary is:

> Psionic determines execution truth. Higher-level OpenAgents services determine
> authority truth.

### Runtime Identity

Runtime identity means the verified execution origin responsible for a work
item, including provider node identity, sandbox instance identity, or cluster
member identity.

Runtime identity matters because it anchors:

- proof attribution
- validator checks
- receipt lineage

### Session Claims And Manifest Discipline

For proof-bearing networked execution, transport identity should carry a signed
session-claims bundle that references runtime, environment, and artifact
digests.

Psionic should also distinguish:

- digest-bound measured or static config
- mutable runtime variables
- higher-level policy profiles or manifest registries evaluated outside Psionic

That split keeps runtime truth honest without collapsing execution evidence and
policy authority into one crate.

### Canonical Receipt Families

| Receipt Family | Current Status | Producer |
| --- | --- | --- |
| runtime execution proof bundles | `implemented` | `psionic-runtime` |
| datastream delivery receipts | `implemented` | `psionic-datastream` |
| sandbox execution receipts | `implemented` | `psionic-sandbox` |
| clustered execution evidence | `implemented_early` | `psionic-cluster` |
| rollout admission receipts | `implemented_early` | `psionic-train` |
| rollout-worker outcome receipts | `implemented_early` | `psionic-train` |
| rollout validator verdicts | `implemented_early` | `psionic-train` |
| training run, trainer step, and eval receipts | `planned` | future `psionic-train` and `psionic-eval` layers |
| adapter package and hosted binding lineage | `implemented_early` | `psionic-adapters` |

## Canonical Execution Lifecycle

Every Psionic workload should fit the same high-level lifecycle even when the
details differ by lane.

1. Work is declared through typed contracts.
2. Artifact bindings and execution prerequisites are resolved.
3. Capability and topology are checked against the requested work.
4. Required artifacts are staged or resumed through datastream contracts.
5. Runtime or cluster planning produces executable work items and topology
   posture.
6. Execution occurs on the declared backend, sandbox, or cluster.
7. Evidence and receipts are emitted from the execution substrate.
8. Operator or authority surfaces consume the typed result rather than raw
   process logs.

## Time Semantics

Psionic execution participates in several time boundaries:

- artifact freshness windows
- checkpoint durability windows
- execution timeouts
- sandbox lifetime limits
- transport retry and resume windows

Training-class and clustered execution build additional timing contracts on top
of these substrate-level boundaries rather than inventing a separate execution
clock.

### Serving Variant

For serving lanes this typically means:

- served artifact resolution
- backend and capability gating
- queue admission, fairness, mixed prefill/decode work, and explicit TTFT/ITL
  plus prefill/decode handoff truth when the lane supports it
- hierarchical KV residency truth across host, device, and any explicit
  externalized tier contract the lane can actually surface
- structured outputs, tool or response-state semantics, and optional multi-model
  routing
- optional clustered placement and shard handoff
- response and proof emission

### Sandbox Variant

For sandbox lanes this typically means:

- profile realization
- bounded runtime selection
- input staging
- job execution
- output and receipt emission

### Training-Class Variant

For training-class lanes this should eventually mean:

- checkpoint and dataset staging
- participant topology formation
- mesh and collective planning
- trainer or rollout execution
- checkpoint flush and recovery handling
- train-specific receipt emission

The training variant is only partially implemented today.

## Control Plane And Observation Boundaries

Psionic exports typed state; it does not own the operator shell.

### App-Owned Control Plane

The desktop app and `autopilotctl` should consume Psionic truth for:

- capability and readiness
- runtime or cluster state
- manifest refs and session-claims posture
- artifact staging progress
- queue or admission posture, shard or cache placement, and sandbox pool health
- sandbox job state
- challenge or validator status when the lane uses one
- training and eval diagnostics, once those exist

### Authority Plane

Kernel and control services should consume Psionic truth for:

- receipts
- proof bundles
- staged artifact references
- cluster and recovery posture
- validator-facing evidence

### What Psionic Must Not Do Here

Psionic must not:

- own app workflows
- invent settlement authority
- collapse operator presentation and execution truth into one crate

## Failure Model

Psionic should handle failure explicitly and typefully.

| Failure | Expected Substrate Handling |
| --- | --- |
| backend unsupported or unavailable | fail capability checks early and expose truthful readiness posture |
| node loss during clustered execution | trigger catch-up, reconfiguration, or recovery according to cluster and train posture |
| network degradation | replan collective or transport decisions when observations degrade materially |
| datastream interruption | resume from cursor and committed bytes rather than restart whole transfer blindly |
| checkpoint flush failure | keep checkpoint non-durable and block any state transition that requires durability |
| sandbox crash | emit bounded execution failure receipt and apply retry or quarantine policy outside the sandbox engine |
| cluster membership mismatch | reject the state transition rather than silently rebasing to a different cluster |
| detached or invalid session claims | reject policy-meaningful networked execution rather than treating transport identity alone as sufficient |
| unapproved quantized collective request | reject planning rather than silently downgrade without record |
| stale artifact or policy revision | reject or quarantine the work item under explicit freshness rules |
| proof augmentation unavailable | emit explicit proof posture rather than pretending strong proof exists |

Psionic must surface failure as typed, reason-coded events rather than opaque
runtime exceptions.

Psionic should prefer:

- reason-coded failure
- replay-safe state transitions
- explicit degraded posture
- checked-in compiler replay fixtures for behavior-preserving lowering changes

It should avoid:

- silent fallback that changes truth without record
- opaque runtime-only failure behavior

## Security Model

Psionic is not the whole platform security model, but it does own several core
security surfaces.

| Threat | Mitigation Direction In Psionic |
| --- | --- |
| artifact tampering | manifest digests, chunk digests, object digests, provenance linkage |
| checkpoint tampering | checkpoint-family binding, writer identity, manifest verification, durable checkpoint posture |
| cluster spoofing or false membership | peer identity, admission policy, ordered-state truth, cluster mismatch rejection |
| detached transport identity or forged proof-bearing sessions | session-claims bundles bound to peer or session keys plus manifest refs and policy checks |
| sandbox escape or undeclared runtime behavior | bounded profiles, explicit runtime detection, execution receipts |
| proof opacity | explicit proof augmentation posture instead of hidden assumptions |
| manifest or policy-registry drift | versioned manifest registries and explicit profile identifiers carried through receipts and authority integrations |
| stale or mismatched policy artifacts | freshness windows and policy-revision binding in planned train layer |
| malicious rollout workers | planned validator sampling and train-layer admission control |
| transport degradation or relay ambiguity | explicit transport observations and candidate state in `psionic-net` and `psionic-cluster` |

The system-wide rule is:

> Psionic should always prefer explicit identity, digest binding, and typed
> degraded posture over implicit trust.

## Current And Planned Psionic Scope

Psionic already has real system scope across:

- runtime execution
- clustered serving
- sandbox execution
- artifact transport
- proof bundles
- training-class recovery substrate

Psionic is still growing into:

- full inference-engine maturity
- full Rust-native train core
- environment and eval runtime
- synthetic-data and research loops
- production-hardening around reproducibility, storage, and security

Those planned areas should still land inside the same system model described
here, not as a disconnected parallel stack.

## Companion Subsystem Specs

- `docs/TRAIN_SYSTEM.md`
  - deep specification for the training subsystem
- `docs/INFERENCE_ENGINE.md`
  - narrow inference completion criteria
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
  - detailed inference build-out and issue plan
- `docs/COMPILER_REPLAY_REFERENCE.md`
  - compiler replay-fixture policy and validation entrypoints

## Review Checklist

- Is this logic in the lowest Psionic crate that can honestly own it?
- Does the change keep execution truth separate from app or market authority?
- Are artifacts and receipts typed and inspectable?
- Is degraded or missing proof posture stated explicitly?
- Does the change preserve the boundary between reusable Psionic substrate and
  app-owned or authority-owned control flow?

## Bottom Line

Psionic is already more than an inference experiment. It is the reusable Rust
execution substrate for OpenAgents compute lanes.

Today it already owns:

- runtime execution truth
- clustered topology truth
- artifact staging
- sandbox execution
- proof bundles
- early training-class recovery and collective truth

What it still lacks is not a new architectural direction. It lacks completion
of the same direction:

- mature inference engine behavior
- full environment and eval layers
- broader distributed training completion
- production-grade receipt, security, and operating discipline across the whole
  subtree

That is the Psionic program.

# Psion Executor Program

> Status: canonical `PSION-0001` / `#700` executor-capable `Psion` naming and
> program-split contract, updated 2026-03-30.

## Why This Doc Exists

The repo already has real `Psion` docs, real `Tassadar` docs, and real
executor-lane artifacts.

What it did not have in one durable place was the naming rule that keeps those
lanes from being flattened into each other.

This doc exists to make three things explicit:

- `Psion` is the umbrella learned-model family inside `psionic`
- `Tassadar` names the executor-capable bounded `Psion` profile and route lane
- the generic compact-decoder `Psion` lane and the executor-capable `Psion`
  lane are different lanes with different acceptance truth

## Umbrella And Profile Rule

For the current program:

- `Psion` is the umbrella family name
- generic compact-decoder `Psion` remains a real learned lane with its own
  route, refusal, serving, and training contracts
- `Tassadar` is the executor-capable bounded `Psion` profile and route family
- the current executor-capable implementation remains the bounded
  article-transformer route and artifact family already shipped in this repo

This means phase one does not rename the live executor lane into some new
artifact family just to make the umbrella cleaner on paper.

## Current Program Split

The active `Psion` program is multi-lane:

### Generic compact-decoder `Psion`

This is the general learned lane documented through surfaces such as:

- `docs/PSION_PROGRAM_MAP.md`
- `docs/PSION_ACCEPTANCE_MATRIX.md`
- `docs/PSION_CAPABILITY_MATRIX.md`

This lane is about bounded learned route, refusal, serving, and training
truth. It is not implicitly executor-capable just because it shares the
umbrella family name.

### Executor-capable `Psion` / `Tassadar`

This is the bounded executor lane documented through surfaces such as:

- `docs/ROADMAP_TASSADAR.md`
- `docs/ROADMAP_TASSADAR_INDEX.md`
- the workspace-level umbrella roadmap in `docs/ROADMAP_PSION.md`

This lane is the current path toward bounded Percepta/Tassadar-style executor
closure, local-first training runs, export, replacement, and fast-route truth.

### Shared substrate

Both lanes still reuse the same `psionic` substrate for:

- training manifests and launch contracts
- runtime binder and admission
- validator and promotion contracts
- cluster and Tailnet execution surfaces
- receipts, lineage, and replay-safe evidence

## Hard Naming Rules

- Do not describe every `Psion` artifact as already executor-capable.
- Do not describe `Tassadar` as a separate top-level product family inside
  `psionic`.
- Do not flatten generic compact-decoder acceptance into executor acceptance.
- Do not hide the current executor lane behind generic learned-lane prose.

## What This Does Not Claim

This doc does not claim:

- that every `Psion` lane can already execute bounded programs
- that generic compact-decoder `Psion` is already executor-capable
- that the current executor lane is broader than the bounded workload family
  and route truth already retained in repo artifacts

## Canonical Follow-On Docs

- `docs/PSION_EXECUTOR_BASELINE.md` owns the frozen executor baseline record
- `docs/PSION_EXECUTOR_ARTIFACT_NAMING.md` owns the phase-one executor
  artifact naming policy
- `docs/PSION_EXECUTOR_ACCEPTANCE_PROFILE.md` owns the executor-capable
  acceptance profile
- `docs/PSION_EXECUTOR_EVAL_PACKS.md` owns the frozen frequent and promotion
  eval packs
- `docs/PSION_EXECUTOR_ARTICLE_CLOSEOUT_SET.md` owns the frozen bounded
  article-workload closeout set that later executor closeout, trace-native
  metrics, fast-route benchmark, and status-report work must cite
- `docs/PSION_EXECUTOR_TRACE_NATIVE_METRICS.md` owns the retained
  per-candidate and per-workload trace-native metrics packet that binds the
  frozen closeout set into the canonical local-cluster ledger surface
- `docs/PSION_EXECUTOR_HULL_CACHE_BENCHMARK.md` owns the retained
  `HullKVCache` versus `reference_linear` benchmark packet and the fast-route
  promotion-block rule when serving truth turns red on the frozen closeout set
- `docs/PSION_EXECUTOR_RESEARCH_BRANCH.md` owns the retained research-only
  2D-head / executor-style branch packet that keeps experimental fast-path
  work bound to the closeout trio, export truth, and carrier-binding truth
- `docs/PSION_EXECUTOR_PERCEPTA_CLOSEOUT_STATUS.md` owns the retained bounded
  Percepta / Tassadar-computation status packet that rolls workload truth,
  fast-path truth, route-replacement truth, and research-branch posture into
  one explicit `red | partial | green_bounded` verdict
- `docs/PSION_EXECUTOR_BASELINE_TRUTH.md` owns the frozen `trained-v0`
  baseline-truth packet for those packs
- `docs/PSION_EXECUTOR_FORMATTING_AUDIT.md` owns the suite-by-suite prompt,
  normalization, and post-processing audit for those packs
- `docs/PSION_EXECUTOR_DECISION_THRESHOLDS.md` owns the retained variance and
  minimum-meaningful-delta packet for promotion comparisons
- `docs/PSION_EXECUTOR_OWNERSHIP.md` owns named owners and review cadence
- `docs/PSION_EXECUTOR_LOCAL_PROFILE_REFERENCE.md` owns the admitted local
  executor profile catalog
- `docs/PSION_EXECUTOR_MLX_FORWARD_LOAD_PARITY.md` owns the retained MLX
  forward/load parity packet for the admitted Mac profile
- `docs/PSION_EXECUTOR_MLX_CHECKPOINT_COMPATIBILITY.md` owns the retained MLX
  checkpoint save/load compatibility packet for the admitted Mac profile
- `docs/PSION_EXECUTOR_MLX_SMOKE_RUN.md` owns the retained MLX smoke-run
  packet and admitted frequent-pack subset for the admitted Mac profile
- `docs/PSION_EXECUTOR_MLX_DECISION_GRADE_RUN.md` owns the retained MLX-local
  decision-grade packet plus the shared v2 dashboard visibility packet for the
  admitted Mac profile
- `docs/PSION_EXECUTOR_MAC_EXPORT_INSPECTION.md` owns the retained Mac-local
  export inspection and CPU-route validation packet for the admitted Mac
  profile
- `docs/PSION_EXECUTOR_4080_REMOTE_LAUNCH.md` owns the retained Mac -> 4080
  Tailnet launch packet for the admitted 4080 worker lane
- `docs/PSION_EXECUTOR_4080_DURABLE_CHECKPOINT.md` owns the retained 4080
  checkpoint-pointer, submission-anchor, and control-plane readback packet for
  the admitted 4080 worker lane
- `docs/PSION_EXECUTOR_4080_FREQUENT_EVAL_ATTACHMENT.md` owns the retained
  checkpoint-time frequent-pack ledger attachment packet and its hard
  promotion-block rule for the admitted 4080 worker lane
- `docs/PSION_EXECUTOR_4080_SMOKE_RUN.md` owns the retained accelerator-backed
  smoke-run packet for the admitted 4080 worker lane
- `docs/PSION_EXECUTOR_4080_INTERRUPTION_RECOVERY.md` owns the retained
  interruption-recovery packet and restore-evidence contract for the admitted
  4080 worker lane
- `docs/PSION_EXECUTOR_4080_DECISION_GRADE_RUN.md` owns the retained 4080
  decision-grade packet, run-registration row, weekly ablation review row, and
  shared v2 dashboard visibility packet for the admitted 4080 worker lane
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_RUN_REGISTRATION.md` owns the canonical
  Mac-and-4080 run-registration schema that later local-cluster ledger,
  dashboard, and roundtrip artifacts build on
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_LEDGER.md` owns the first searchable
  cumulative ledger for the retained MLX and 4080 executor rows
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_DASHBOARD.md` owns the first canonical
  baseline-vs-current-best-vs-candidate dashboard packet built directly on
  top of the retained local-cluster ledger
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_AUTOBLOCKS.md` owns the first canonical
  phase-exit and promotion auto-block report for missing eval, recovery,
  export, and `reference_linear` anchor facts on the retained local-cluster
  dashboard/ledger stack
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_REVIEW_WORKFLOW.md` owns the first
  canonical weekly baseline and ablation review workflow packet built directly
  on the retained local-cluster dashboard, ledger, and auto-block stack
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_ROUNDTRIP.md` owns the first retained
  Mac -> 4080 -> Mac roundtrip closeout packet and the explicit phase-exit
  cluster-closure proof for the admitted executor local-cluster lane
- `docs/PSION_EXECUTOR_CANONICAL_MIXTURE_V0.md` owns the first canonical
  executor-lane mixture manifest with explicit source-family weights, seed
  suite, held-out exclusions, and evaluation exclusions
- `docs/PSION_EXECUTOR_CURRICULUM_BOUNDARIES.md` owns the first canonical
  stagewise curriculum packet that binds the executor mixture to explicit
  boundary-anchor, frequent-pack, and promotion-pack transitions
- `docs/PSION_EXECUTOR_SOURCE_FAMILY_CONTRIBUTION.md` owns the first canonical
  mixture-review report that keeps per-family slice deltas separate from
  run-level throughput and stability regressions on the current-best 4080 row
- `docs/PSION_EXECUTOR_MIXTURE_SEARCH_CADENCE.md` owns the first canonical
  weekly mixture-search cadence packet that freezes the active mixture version
  into run registration truth and limits pre-lane-health parallel search
- `docs/PSION_EXECUTOR_MIXTURE_ROLLBACK_POLICY.md` owns the first canonical
  rollback policy packet for misleading mixture wins and the single-lever retry
  constraint that weekly review now retains directly
- `docs/PSION_EXECUTOR_OPTIMIZER_ABLATION.md` owns the first canonical
  same-budget optimizer ablation packet that repeats one promising 4080 result
  once, checks it against the frozen decision-threshold packet, and retains it
  only because the repeat still clears the frozen noise band without exactness,
  held-out, or adversarial regressions
- `docs/PSION_EXECUTOR_SCHEDULER_ABLATION.md` owns the first canonical
  same-budget scheduler and warmup ablation packet that keeps the retained
  4080 result directly comparable to the frozen baseline, logs it for review,
  and keeps it explicitly below the frozen promotion-noise threshold
- `docs/PSION_EXECUTOR_PHASE_TWO_PREFLIGHT_CHECKLIST.md` owns the first
  canonical phase-two pre-flight checklist packet that freezes launch blockers
  for the admitted MLX and 4080 decision-grade run types
- `docs/PSION_EXECUTOR_CONTINUE_RESTART_POLICY.md` owns the first canonical
  incident-policy packet that freezes continue-vs-restart handling for the
  admitted executor lane and binds it to the weekly review surface
- `docs/PSION_EXECUTOR_MANDATORY_LIVE_METRICS.md` owns the first canonical
  mandatory live-metrics packet that freezes the minimum long-run metric set on
  top of the retained local-cluster ledger and dashboard rows
- `docs/PSION_EXECUTOR_FAILURE_BUNDLE_TAXONOMY.md` owns the first canonical
  failure-bundle taxonomy packet that freezes long-run incident classes and
  emitted bundle posture on top of the retained live-metrics and weekly-review
  surfaces
- `docs/PSION_EXECUTOR_UNIFIED_THROUGHPUT_REPORTING.md` owns the first
  canonical throughput packet that keeps retained training throughput and
  admitted fast-route serving throughput together while freezing a replacement
  blocker for serving-throughput regression
- `docs/PSION_EXECUTOR_LONG_RUN_REHEARSAL.md` owns the first canonical
  long-run rehearsal packet that binds pre-flight admission, transient
  interruption recovery, export, replacement validation, and review logging
  into one retained closeout receipt
- `docs/PSION_PROGRAM_MAP.md` owns the generic learned `Psion` family map
- `docs/PSION_ACCEPTANCE_MATRIX.md` owns generic compact-decoder acceptance
- `docs/ROADMAP_TASSADAR.md` remains the repo-local executor-lane bridge
- `docs/ROADMAP_TASSADAR_INDEX.md` remains the executor artifact index

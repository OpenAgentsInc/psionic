# Psionic

Psionic is a Rust-native ML stack.

Its aim is to rebuild core ML infrastructure that is usually spread across
Python and C++ stacks such as PyTorch into a coherent Rust crate family:
tensor and graph contracts, compiler/runtime boundaries, backend truth,
artifact staging, cluster and sandbox execution, serving interfaces, adapter
packaging, evaluation, research, and the early training substrate.

OpenAgents uses Psionic as one downstream compute substrate, but the project is
intentionally broader than OpenAgents. It is meant to be useful to anyone who
wants a Rust-first foundation for local and decentralized inference, training,
clustered execution, and machine-legible execution truth.

## Doc Authority

- `README.md` is the Psionic entrypoint and map.
- `docs/ARCHITECTURE.md` is the canonical Psionic-wide system spec.
- `docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md` is the canonical framework-core
  completion bar for tensor, compiler, IO, replay, and local multi-device
  behavior.
- `docs/INFERENCE_ENGINE.md` is the canonical inference-engine completion doc.
- `docs/TOKIO_RUNTIME_TELEMETRY.md` is the canonical Tokio runtime telemetry
  and operator runbook for serving and transport surfaces.
- `docs/TRAIN_SYSTEM.md` is the canonical training subsystem spec.
- `docs/TASSADAR_MULTI_PLUGIN_ORCHESTRATION_WAVE.md` is the canonical umbrella
  closeout doc for the first real-run multi-plugin controller tranche above
  the shared starter-plugin runtime.
- `docs/TASSADAR_STARTER_PLUGIN_USER_AUTHORING_WAVE.md` is the canonical
  umbrella closeout doc for the bounded starter-plugin user-authoring wave.
- `docs/TASSADAR_STARTER_PLUGIN_AUTHORING.md` is the canonical bounded
  authoring template for adding a new capability-free starter plugin inside
  this repository.
- `docs/PSION_PROGRAM_MAP.md` is the canonical umbrella map for the `Psion`
  learned-model lane.
- `docs/PSION_CORPUS_ADMISSION.md` is the canonical first admission-policy doc
  for the `Psion` learned-model lane.
- `docs/PSION_SOURCE_LIFECYCLE.md` is the canonical source-lifecycle and
  downstream-lineage doc for the `Psion` learned-model lane.
- `docs/PSION_BENCHMARK_ISOLATION.md` is the canonical held-out and
  contamination-control doc for the `Psion` learned-model lane.
- `docs/PSION_RAW_SOURCE_INGESTION.md` is the canonical raw-source import and
  normalization doc for the `Psion` learned-model lane.
- `docs/PSION_TOKENIZER_TRAINING.md` is the canonical tokenizer manifest and
  artifact-bundle doc for the `Psion` learned-model lane.
- `docs/PSION_TOKENIZED_CORPUS.md` is the canonical tokenized dataset and
  replay-identity doc for the `Psion` learned-model lane.
- `docs/PSION_SAMPLING_POLICY.md` is the canonical sampling, weighting, and
  code-dominance-control doc for the `Psion` learned-model lane.
- `docs/PSION_COMPACT_DECODER.md` is the canonical compact decoder-family and
  checkpoint/export naming doc for the `Psion` learned-model lane.
- `docs/PSION_PRETRAIN_STAGE.md` is the canonical pretrain-stage and receipt
  doc for the `Psion` learned-model lane.
- `docs/PSION_RUN_OBSERVABILITY.md` is the canonical cost, throughput,
  checkpoint-size, and topology-observability doc for the `Psion`
  learned-model lane.
- `docs/PSION_PILOT_PRETRAINING_RUN.md` is the canonical first pilot-run and
  promotion-evidence doc for the `Psion` learned-model lane.
- `docs/PSION_REASONING_SFT.md` is the canonical bounded reasoning-SFT and
  style-control doc for the `Psion` learned-model lane.
- `docs/PSION_BENCHMARK_PACKAGES.md` is the canonical shared benchmark item,
  grader, and receipt doc for the `Psion` learned-model lane.
- `docs/PSION_BENCHMARK_LABEL_GENERATION.md` is the canonical benchmark-label,
  rubric-version, exact-truth, and derived-data-lineage doc for the `Psion`
  learned-model lane.
- `docs/PSION_ARCHITECTURE_REASONING_BENCHMARK.md` is the canonical
  architecture-reasoning benchmark-package and acceptance-binding doc for the
  `Psion` learned-model lane.
- `docs/PSION_NORMATIVE_SPEC_READING_BENCHMARK.md` is the canonical
  normative-reading benchmark-package and acceptance-binding doc for the
  `Psion` learned-model lane.
- `docs/PSION_ENGINEERING_SPEC_INTERPRETATION_BENCHMARK.md` is the canonical
  engineering-interpretation benchmark-package and acceptance-binding doc for
  the `Psion` learned-model lane.
- `docs/PSION_MEMORIZATION_VS_REASONING_PROBES.md` is the canonical
  memorization-versus-reasoning probe and acceptance-binding doc for the
  `Psion` learned-model lane.
- `docs/PSION_ROUTE_CLASS_EVALUATION.md` is the canonical route-class
  benchmark, receipt, and acceptance-binding doc for the `Psion` learned-model
  lane.
- `docs/PSION_REFUSAL_CALIBRATION.md` is the canonical unsupported-envelope
  refusal benchmark, receipt, and capability-binding doc for the `Psion`
  learned-model lane.
- `docs/PSION_SERVED_EVIDENCE.md` is the canonical served-evidence and
  provenance doc for the `Psion` learned-model lane.
- `docs/PSION_SERVED_OUTPUT_CLAIMS.md` is the canonical served-output claim
  discipline doc for the `Psion` learned-model lane.
- `docs/PSION_CHECKPOINT_RECOVERY.md` is the canonical dense-versus-sharded
  checkpoint restart, rollback, and invalidation doc for the `Psion`
  learned-model lane.
- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md` is the canonical rented-cluster
  runbook and failure-policy doc for the `Psion` learned-model lane.
- `docs/PSION_TRUSTED_CLUSTER_RUN.md` is the canonical trusted-cluster
  multi-host run doc for the `Psion` learned-model lane.
- `docs/PSION_ACCEPTANCE_MATRIX.md` is the canonical phase-gate and
  promotion-decision doc for the `Psion` learned-model lane.
- `docs/PSION_CAPABILITY_MATRIX.md` is the canonical served capability and
  publication doc for the `Psion` learned-model lane.
- `docs/PSION_CAPABILITY_WITHDRAWAL.md` is the canonical rollback and
  downgrade-history doc for the `Psion` learned-model lane.
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md` is the canonical bounded
  decentralized-contribution doc for the `Psion` learned-model lane.
- `docs/TASSADAR_RUST_ONLY_ARTICLE_RUNBOOK.md` is the canonical one-command
  operator guide for the Rust-only Tassadar article reproduction path.
- research audits explain direction and rationale, but they are not the
  authoritative current-state spec.

This repo was extracted from the larger `openagents` tree. Some deeper docs and
audits still reference historical parent-repo paths or app-owned surfaces that
do not ship here; treat those as external or historical context unless the file
exists in this repository.

## What Psionic is

- A reusable Rust-native ML infrastructure layer that OpenAgents uses as one
  downstream compute substrate.
- A Rust-native crate family for framework core, backends, transport,
  clustered execution, serving, adapters, data, eval, training, and research.
- The source of machine-legible execution truth: manifests, receipts, routing
  facts, cache facts, proof bundles, topology state, and training/eval
  lineage.
- The layer that can turn backend/runtime reality into truthful provider
  capabilities without owning desktop UX, market procurement, or settlement
  authority.

## What Psionic is not

- Not desktop UX, wallet or payout logic, or buyer/provider product
  orchestration.
- Not external authority for compute-market truth, settlement, or accepted
  outcomes.
- Not a shortcut around crate boundaries or the canonical specs in `docs/`.
- Not a claim that every backend, model family, serving topology, or
  training-class lane is fully productized.
- Not a hidden Python control plane disguised as Rust crates.

## Tassadar Executor Lane

Psionic now has an implemented-early executor-class reference lane codenamed
`Tassadar`.

Tassadar is based on Percepta's
[Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers).

Current posture:

- it lives under `crates/psionic-*`, not in app code and not in kernel or
  Nexus authority
- it is WebAssembly-first and CPU-reference-first
- it is intended to give larger reasoning systems inner exact-computation
  ability
- its Phase 1 reference substrate now exists in `psionic-runtime` and
  `psionic-models`
- its Phase 2 artifact/compatibility contract now exists as digest-bound
  program artifacts plus explicit executor compatibility descriptors
- its Phase 3 benchmark/environment package layer now exists in
  `psionic-data`, `psionic-environments`, and `psionic-eval`, including a
  public benchmark-package-set summary that separates exactness,
  length-generalization, and planner-usefulness across arithmetic,
  CLRS-seeded shortest path, Sudoku, Hungarian, and trace-length-stress
  families
- its Phase 4 proof/lineage layer now exists in `psionic-runtime`, with
  emitted trace artifacts, runtime-manifest lineage, and canonical proof-bundle
  integration
- its Phase 5 fast path now exists in `psionic-runtime` and `psionic-eval`,
  with explicit `HullCache` decode identity, exact CPU/linear/hull equivalence
  checks, typed refusal for backward-branch workloads outside the first
  validated subset, and benchmark reporting for hull-cache throughput, speedup
  over linear decode, and remaining gap vs direct CPU
- its Phase 6 runtime truth now exists in `psionic-runtime`, `psionic-models`,
  and `psionic-eval`, with a machine-legible capability report plus explicit
  direct/fallback/refused decode selection diagnostics for hull-cache,
  approximate sparse-top-k fallback, unsupported ABI/profile requests, and
  model-effective decode mismatches
- its Phase 7A served product surface now exists in `psionic-serve`, with an
  explicit `psionic.executor_trace` request/stream/terminal contract, typed
  refusal responses, trace-step streaming, final output extraction helpers, and
  served evidence bundles that preserve decode selection, trace proof, and
  runtime-manifest lineage
- its article-session serving follow-on now also exists in `psionic-serve`,
  with the specialized `psionic.article_executor_session` contract bound to the
  canonical article corpus, plus committed direct/fallback/refused session
  evidence at `fixtures/tassadar/reports/tassadar_article_executor_session_artifact.json`
- its replay/live Tassadar lab surface now also exists in `psionic-serve`,
  with one renderer-neutral snapshot/update contract that projects both live
  article-session or hybrid-workflow truth and replay truth over canonical
  compiled, learned, fit, and closure artifacts, plus committed evidence at
  `fixtures/tassadar/reports/tassadar_lab_surface_artifact.json`
- its Phase 7B widened executor envelope now exists in `psionic-runtime`,
  `psionic-models`, and `psionic-eval`, with the widened `core_i32_v2`
  profile, the dedicated article-shaped `tassadar.wasm.article_i32_compute.v1`
  profile, profile-aware runner construction, and article-class exact
  benchmark coverage for `MicroWasmKernel`, `BranchHeavyKernel`,
  `MemoryHeavyKernel`, `LongLoopKernel`, `SudokuClass`, and
  `HungarianMatching`, plus the committed report at
  `fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json`
  that keeps direct-vs-fallback posture explicit per workload family
- its Phase 7C long-horizon trace ABI posture now also exists in
  `psionic-runtime`, with the committed spec/report at
  `fixtures/tassadar/reports/tassadar_trace_abi_decision_report.json` and the
  committed long-loop evidence bundle at
  `fixtures/tassadar/runs/long_loop_kernel_trace_abi_v0/execution_evidence_bundle.json`;
  readable logs are now explicitly subordinate to the canonical trace
  artifact, and validator-facing ABI pointers are frozen across benchmark,
  compiled, and long-horizon fixture artifacts
- its Phase 7D workload capability matrix now also exists in `psionic-eval`,
  with the committed report at
  `fixtures/tassadar/reports/tassadar_workload_capability_matrix.json` that
  records runtime exact vs fallback-only posture per workload family and keeps
  compiled exact, bounded learned, and partial 9x9 learned evidence separate
- its Phase 8A widened `HullCache` closure report now also exists in
  `psionic-eval`, with the committed report at
  `fixtures/tassadar/reports/tassadar_hull_cache_closure_report.json` that
  freezes the current direct-exact HullCache class on
  `MicroWasmKernel`, `BranchHeavyKernel`, `MemoryHeavyKernel`, and bounded
  `HungarianMatching`, while keeping `LongLoopKernel` and `SudokuClass`
  explicitly fallback-only
- its Phase 8B `SparseTopK` comparison report now also exists in
  `psionic-eval`, with the committed report at
  `fixtures/tassadar/reports/tassadar_sparse_top_k_comparison_report.json`
  that compares SparseTopK against the same article workload set and keeps
  `BranchHeavyKernel`, `LongLoopKernel`, and `SudokuClass` explicitly
  fallback-only under the current validation contract
- its Phase 8C decode-scaling report now also exists in `psionic-eval`, with
  the committed report at
  `fixtures/tassadar/reports/tassadar_decode_scaling_report.json` that tracks
  trace-artifact growth, throughput, CPU-gap, and direct-vs-fallback posture
  across shared linear, branch-heavy, and long-loop synthetic families instead
  of relying on one headline fast-path number
- its Phase 8D million-step decode benchmark bundle now also exists in
  `psionic-runtime`, with the committed bundle at
  `fixtures/tassadar/runs/million_step_loop_benchmark_v0/benchmark_bundle.json`
  that proves one reproducible 1,048,575-step reference-linear execution under
  the Psionic-owned executor path, including exactness, proof lineage,
  runtime-manifest identity, and serialized trace-byte growth receipts while
  keeping HullCache and SparseTopK explicit as fallback-only at that horizon
- its Phase 8E geometric-variant comparison report now also exists in
  `psionic-eval`, with the committed report at
  `fixtures/tassadar/reports/tassadar_geometric_variant_report.json` that keeps
  the promoted runtime HullCache lane separate from a research-only
  hierarchical-hull candidate; the candidate stays direct and exact on
  long-loop and 4x4 Sudoku article workloads, but that widened class remains
  explicitly unpromoted until runtime closure bars are landed
- its byte-addressed linear-memory ABI v2 lane now also exists across
  `psionic-runtime`, `psionic-models`, `psionic-train`, and `psionic-eval`,
  with a public runtime-owned memory ABI contract, exact i8/i16/i32 width and
  sign-extension behavior, explicit `memory.size` / `memory.grow` execution,
  delta-oriented memory tracing, a training-facing supervision suite, and the
  committed report at
  `fixtures/tassadar/reports/tassadar_memory_abi_v2_report.json`
- its module-trace ABI v2 lane now also exists across `psionic-runtime`,
  `psionic-models`, `psionic-train`, and `psionic-eval`, with explicit
  legacy-v1 versus frame-aware delta-oriented v2 contracts, deterministic
  replay back into the snapshot-heavy module execution trace, a public
  training-facing supervision suite over global-state, call-indirect, and
  deterministic-import cases, and the committed report at
  `fixtures/tassadar/reports/tassadar_module_trace_abi_v2_report.json`
- its module-scale Wasm workload suite now also exists across `psionic-data`,
  `psionic-environments`, and `psionic-eval`, with a public deterministic
  workload-suite contract over memcpy, parsing, checksum, and VM-style module
  families, environment-bundle metadata that binds the same suite into the
  repo-facing Tassadar benchmark surface, committed source plus compiled Wasm
  fixtures, and the committed report at
  `fixtures/tassadar/reports/tassadar_module_scale_workload_suite_report.json`
  that keeps exactness, trace-length, deterministic CPU-reference cost, and
  typed refusal explicit per module case
- its module-state learned-executor redesign lane now also exists across
  `psionic-models`, `psionic-train`, and `psionic-research`, with a public
  research-only module-state executor publication over explicit call-frame,
  global-delta, memory-delta, and export-boundary channels, a training-facing
  module curriculum suite over the module-scale memcpy/parsing/checksum/vm-style
  families plus held-out family metrics, and the committed report at
  `fixtures/tassadar/reports/tassadar_module_state_architecture_report.json`
  that keeps later-window exactness, final-state accuracy, and trace-to-final-state
  gap deltas explicit against the flat-prefix baseline
- its structured-control closure lane now also exists across
  `psionic-compiler`, `psionic-runtime`, and `psionic-eval`, with compiler
  lowering from bounded zero-parameter Wasm functions into validated executor
  programs for `block`, `loop`, `if`, `else`, `br`, `br_if`, and `br_table`,
  runtime-owned exact execution and branch traces for that nested control
  surface, and the committed report at
  `fixtures/tassadar/reports/tassadar_structured_control_report.json`
- its bounded call-frame lane now also exists across `psionic-runtime`,
  `psionic-models`, `psionic-train`, and `psionic-eval`, with a real direct
  call-frame model, multi-function execution, replayable frame-stack traces,
  bounded-recursion refusal, and the committed report at
  `fixtures/tassadar/reports/tassadar_call_frame_report.json`
- its staged numeric-opcode widening ladder now also exists across
  `psionic-data`, `psionic-compiler`, `psionic-runtime`, and `psionic-eval`,
  with a public family-by-family contract for exact i32 arithmetic,
  comparisons, and bit operations versus explicit i64 and floating-point
  refusal, widened structured-control lowering and execution support for the
  implemented i32 families, and the committed report at
  `fixtures/tassadar/reports/tassadar_numeric_opcode_ladder_report.json`
- its bounded module-execution boundary now also exists across
  `psionic-runtime`, `psionic-models`, `psionic-serve`, `psionic-provider`,
  and `psionic-sandbox`, with explicit i32 global and funcref-table runtime
  models, bounded `call_indirect`, deterministic import stubs, typed refusal
  for unsupported host calls, a model-facing module-capability publication,
  served/provider capability-path projection, a sandbox-facing import
  boundary contract, and now a real Wasm conformance/differential harness
  against a reference authority over curated plus deterministically generated
  bounded module cases
- the repo now also carries a standardized exactness/refusal evidence surface
  across `psionic-runtime`, `psionic-provider`, and `psionic-eval`, with a
  shared runtime report schema for exact direct, exact fallback, mismatch, and
  refused posture, a provider-facing receipt projection, and the committed
  artifact at
  `fixtures/tassadar/reports/tassadar_exactness_refusal_report.json`
- the first trained-executor follow-on bar now also exists in
  `psionic-runtime` and `psionic-models`: a dedicated
  `tassadar.wasm.sudoku_v0_search.v1` profile plus a real 4x4 backtracking
  Sudoku search program representation that is exact on the CPU reference lane
  and explicitly outside the current hull/sparse validated fast-path subset
- the second trained-executor follow-on bar now also exists in
  `psionic-runtime` and `psionic-eval`: the fake `SudokuClass` placeholder has
  been replaced by a real multi-case 4x4 Sudoku-v0 corpus with stable
  train/validation/test split metadata, exact CPU-reference traces per puzzle,
  and truthful article-class benchmark reporting that surfaces hull/sparse
  fallback on those backtracking workloads instead of pretending they remain
  direct fast-path cases
- the third trained-executor follow-on bar now also exists in
  `psionic-data`, `psionic-models`, `psionic-eval`, and `psionic-train`: the
  Sudoku-v0 corpus can now be materialized as deterministic program-plus-trace
  token sequences with a fixed executor vocabulary, reversible symbolic decode,
  versioned dataset manifests, split-stable lineage metadata, and frozen
  packing plans for the first honest training run
- the fourth trained-executor follow-on bar now also exists in
  `psionic-models`: a first real neural executor transformer family now runs
  next-token forward passes over the Tassadar sequence vocabulary with explicit
  2D lookup-head geometry claims, linear decode state, and a descriptor that
  marks the lane as next-token-only rather than pretending the trained model is
  already an exact executor
- the fifth trained-executor follow-on bar now also exists in
  `psionic-train` and `psionic-eval`: the executor transformer can now be
  trained with teacher-forced next-token loss over the frozen Sudoku-v0
  sequence corpus, and validation reports now expose exact-trace,
  final-output, and halt-correctness metrics against the same CPU-reference
  sequences used to build the dataset
- the sixth trained-executor follow-on bar now also exists in `psionic-eval`
  and `psionic-train`: trained-model neural linear decode can now be benchmarked
  directly against CPU reference execution on the Sudoku-v0 corpus, with
  explicit decode-mode identity, explicit no-KV-cache prefix-recompute identity,
  and per-case exactness facts instead of only aggregate scores
- the seventh trained-executor follow-on bar now also exists in
  `psionic-train` and `fixtures/tassadar/runs/`: the first
  Psionic-only Sudoku-v0 reference run now persists a frozen training
  manifest, training report, linear benchmark report, checkpoint state plus
  checkpoint manifest, and a trained-model artifact bundle under
  `fixtures/tassadar/runs/sudoku_v0_reference_run_v0`; the
  current run is intentionally honest about still being weak
  (`validation_exact_trace_case_count = 0/2`, aggregate target exactness
  `15` bps), so this is a reproducible first-run artifact lane rather than a
  claim that Sudoku is already solved in-model
- the eighth trained-executor follow-on bar now also exists in
  `psionic-train` and that same run bundle: Phase 8 telemetry now persists
  `training_telemetry.json`, `exactness_curve.json`,
  `trace_divergence_report.json`, and `failure_samples.json`, and the current
  artifacts show that all 8 decoded cases diverge at target token 0 with case
  exactness only in the `9` to `16` bps range, which gives the next run a real
  failure-analysis baseline instead of an anecdotal “weak model” label
- the ninth trained-executor follow-on bar now also exists in
  `psionic-train`, the run bundle, and `docs/audits/`: the first run now has a
  machine-readable `postmortem.json` plus `next_run_plan.json`, and a
  human-readable review in `docs/audits/2026-03-16-tassadar-first-run-postmortem.md`;
  the resulting plan explicitly prioritizes a boundary curriculum, a larger
  optimization budget, conditional trainable-surface expansion, and truthful
  gating around what later phases do and do not prove
- the tenth trained-executor follow-on bar now also exists in
  `psionic-models`, `psionic-eval`, `psionic-train`, and that same run bundle:
  the trained executor model now exposes explicit model-KV decode selection,
  real hull-cache lookup over those KV points, and a persisted
  `neural_hull_benchmark_report.json`; on the committed Sudoku-v0 run, hull
  decode matches the explicit model-KV linear path on all `8/8` cases with no
  fallbacks or refusals and improves benchmarked decode throughput from
  `21,860` to `42,172` target tok/s over a `4,096`-token per-case window, but
  exactness remains `0/8`, so this is a real fast-path result rather than a
  claim that the model now solves Sudoku
- the eleventh trained-executor follow-on bar now also exists in
  `psionic-runtime`, `psionic-eval`, `psionic-models`, and `psionic-train`: a
  real `tassadar.wasm.sudoku_9x9_search.v1` profile, a real split-aware 9x9
  Sudoku-class corpus, a tokenized 9x9 sequence dataset plus frozen training
  manifest, a bounded 9x9 smoke-training config, and a committed
  `scale_plan.json` fixture under
  `fixtures/tassadar/runs/sudoku_9x9_scale_plan_v0`; that plan
  keeps Phase 11 honest by recording the current 4x4 gate as still closed
  (`0/2` validation first-target exact cases, `0/2` exact-trace cases) while
  still making the real 9x9 workload and curriculum plan explicit
- the twelfth trained-executor follow-on bar from the post-audit issue spine
  now also exists in `psionic-eval`, `psionic-train`, `docs/audits/`, and a
  new committed follow-on run bundle at
  `fixtures/tassadar/runs/sudoku_v0_boundary_v1`: the learned
  4x4 lane now emits first-target / first-8 / first-32 boundary metrics,
  divergence histograms, first-token confusion, and a checkpoint leaderboard,
  and the boundary-curriculum run clears the token-0 failure at the selected
  checkpoint (`10000` bps first-target exactness, no token-0 confusions,
  divergence moved to target index `1` on both validation cases); it still has
  `0/2` exact traces and only `5000` bps first-32 exactness, so this is
  honest boundary progress rather than an exact learned-executor claim
- the thirteenth trained-executor follow-on bar from the post-audit issue
  spine now also exists in `psionic-models`, `psionic-train`,
  `psionic-research`, `docs/audits/`, and a new same-corpus ablation root at
  `fixtures/tassadar/runs/sudoku_v0_trainable_surface_ablation_v1`:
  the lookup-family executor now records a stable trainable surface in model
  descriptors, training manifests, checkpoints, and run bundles, and
  `psionic-research` now persists a machine-readable
  `trainable_surface_ablation.json` across four controlled surfaces; the only
  surface that beats the preserved `output_head_only` baseline is
  `output_head_embeddings_and_small_learned_mixer`, which improves boundary
  exactness to `3750` bps over the first `8` target tokens and `5625` bps over
  the first `32`, but still leaves `0/2` exact validation traces and the first
  divergence bucket at target index `1`, so this is a truthful next-surface
  recommendation rather than a promotion claim
- the fourteenth trained-executor follow-on bar from the post-audit issue
  spine now also exists in `psionic-train`, `docs/audits/`, `scripts/`, and a
  preserved red promotion bundle at
  `fixtures/tassadar/runs/sudoku_v0_promotion_v1`: long Phase 14
  runs now emit live stage/epoch/batch/validation/checkpoint progress, the repo
  now persists `best_checkpoint_manifest.json` plus
  `promotion_gate_report.json`, the repo-owned
  `scripts/check-tassadar-4x4-promotion-gate.sh` checker revalidates persisted
  gate reports, and the original lookup-family promotion run recorded the
  first honest gate baseline at checkpoint `epoch_0006`
  (`10000` bps first-target, `7500` bps first-8, `6875` bps first-32,
  `0/2` exact validation traces); that bundle remains preserved blocker
  evidence rather than an exact learned-trace result
- the learned 4x4 promotion gate is now green in
  `psionic-research`, `psionic-train`, `docs/audits/`, and the canonical
  bundle `fixtures/tassadar/runs/sudoku_v0_promotion_v3`:
  `crates/psionic-research/examples/tassadar_executor_attention_promotion_run.rs`
  now replays the bootstrap-plus-promotion attention continuation in-repo,
  persists `best_checkpoint_manifest.json`, `exactness_curve.json`,
  `failure_samples.json`, `exact_trace_samples.json`, and
  `promotion_gate_report.json`, and the repo-owned
  `scripts/check-tassadar-4x4-promotion-gate.sh` checker revalidates that
  bundle as passed at checkpoint `epoch_0015`
  (`10000` bps first-target, `10000` bps first-8, `10000` bps first-32,
  `2/2` exact validation traces); that clears the bounded benchmark gate, but
  the separate promotion-policy report at
  `fixtures/tassadar/reports/tassadar_promotion_policy_report.json` still
  blocks served promotion until the learned lane also has stable refusal
  policy and route-contract compatibility, and the companion audit is
  `docs/audits/2026-03-16-tassadar-phase-14-promotion-green-audit.md`
- the fifteenth trained-executor follow-on bar from the post-audit issue spine
  now also exists in `psionic-models`, `psionic-eval`, `psionic-research`,
  `docs/audits/`, and a new bounded same-corpus comparison root at
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v1`:
  `psionic-models` now carries a separate layered causal-attention
  `TassadarExecutorAttentionTransformer` family with explicit 2D head geometry,
  per-layer semantics, and truthful hull fallback, while `psionic-research`
  now persists `architecture_comparison_report.json` plus per-family run bundles
  against the preserved Phase 13 lookup baseline; that report keeps the claim
  boundary honest by showing that the new family is architecturally closer to
  the article but still materially worse on the bounded 4x4 window (`0` bps
  first-target / first-32 exactness and `1333` target tok/s, versus the lookup
  baseline at `10000` / `6563` bps and `32000` target tok/s), so this is a
  research-family landing rather than a promotion or parity claim
- the post-Phase-15 trained-attention follow-on now also exists in
  `psionic-research`, `docs/audits/`, and two new bounded artifact roots at
  `fixtures/tassadar/runs/sudoku_v0_attention_training_v1` and
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v2`:
  the attention family now has a real output-head training loop plus a
  preserved same-corpus comparison against the lookup baseline; the trained
  attention checkpoint materially improves over the seeded Phase 15 candidate
  (`6563` bps aggregate and first-32 exactness instead of `0`), but it still
  fails the first-token boundary (`0` bps first-target), still yields `0/2`
  exact bounded traces, and still loses the lookup baseline on the specific
  gate that matters, so this is a truthful research follow-on rather than a
  learned-lane promotion result
- the Phase 16 first honest 9x9 run now also exists in `psionic-train`,
  `docs/audits/`, and the canonical bundle
  `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0`:
  `crates/psionic-train/examples/tassadar_sudoku_9x9_reference_run.rs`
  now replays the first bounded learned 9x9 run with the explicit
  `incremental_decode_window` teacher-forced strategy and
  `incremental_decode_window` long-trace family contract bound into the
  training manifest, persists `sequence_fit_report.json`, `postmortem.json`,
  `next_run_plan.json`, `later_window_exactness_report.json`,
  `suffix_window_failure_report.json`, `best_checkpoint_manifest.json`,
  `promotion_bundle.json`, and `promotion_gate_report.json`, while the
  repo-owned `scripts/check-tassadar-9x9-promotion-gate.sh` checker
  revalidates the stored learned 9x9 gate as internally consistent; the
  selected checkpoint remains `epoch_0004` from `full_trace_supervision`, the
  learned lane still cannot fit full honest 9x9 traces inside the current
  `524288`-token model context (`4891222` to `5335309` total tokens, overflow
  `4366934` to `4811021`), and the new gate keeps the failure shape explicit:
  early `512`-token first-32 exactness stays at `5938`, both the fixed later
  window at target offset `262144` and the furthest fittable suffix window
  improve to `8438`, all three gate windows remain `0/1` exact windows, and
  full-trace exactness across the declared windows remains `0`, so later
  slices are no longer hidden but the lane is still honestly partial; the
  companion audit is
  `docs/audits/2026-03-16-tassadar-phase-16-9x9-reference-run-audit.md`
- the first same-corpus 9x9 flat-prefix-vs-windowed learned comparison now
  also exists at
  `fixtures/tassadar/runs/sudoku_9x9_v0_windowed_family_comparison_v1`; it
  keeps the exactness claim bounded by showing both families still land at
  `5938` bps first-32 and `0/1` exact validation traces on the first `512`
  target tokens, while making the long-trace contract difference explicit by
  dropping estimated live bytes from `109715076` on the flat-prefix family to
  `1459452` on the windowed family under the same corpus and checkpoint stage
- the first same-corpus sequential-vs-wavefront trace-family comparison now
  also exists at
  `fixtures/tassadar/runs/tassadar_trace_family_comparison_v1`; it keeps the
  claim boundary at `research_only` for the alternate families while proving
  their exact final-output reconstruction on shared corpora: 9x9 Sudoku drops
  from `5335309` max total tokens on the sequential CPU trace to `52969` on
  the anti-diagonal wavefront family, and article-sized 10x10 Hungarian drops
  from `11532454` to `22050`, with all alternate families staying at
  `10000` bps final-output exactness and the sequential family remaining the
  only full CPU-trace authority; `psionic-data` now also publishes the
  comparable trace-family set contract for those sequence variants and
  `psionic-research` now freezes the repo-facing summary at
  `fixtures/tassadar/reports/tassadar_trace_family_variant_report.json`
- the first public no-hint / self-supervised executor regime report now also
  exists at
  `fixtures/tassadar/reports/tassadar_no_hint_self_supervised_report.json`;
  it keeps the whole lane explicitly `research_only_architecture` while
  freezing the seeded sort / CLRS-shortest-path / sudoku-style comparison:
  held-out CLRS reusable signal moves from `1666` bps on full-hint traces to
  `5000` on output-only no-hint and `8000` on no-hint plus self-supervised
  regularizers, while reusable subroutine hints stay the upper bound at `8333`
  and served promotion remains explicitly refused
- the first public scratchpad / controlled-position executor framework report
  now also exists at
  `fixtures/tassadar/reports/tassadar_scratchpad_framework_comparison_report.json`;
  `psionic-ir` now owns bounded `flat_trace` and
  `delimited_chunk_scratchpad` formatting plus `absolute_monotonic`,
  `segment_reset`, and `trace_schema_buckets` controlled position-ID schemes,
  `psionic-models` now exposes framework descriptors plus locality evidence,
  and `psionic-train` now freezes arithmetic symbolic and algorithmic same-lane
  comparisons under the explicit `learned_bounded_success` claim boundary; the
  arithmetic segment-reset variant cuts max output local position from `14` to
  `3`, the algorithmic trace-schema variant cuts it from `11` to `3`, and both
  preserve final output tokens exactly while surfacing scratchpad overhead and
  reset counts
- the first public efficient-attention baseline matrix now also exists at
  `fixtures/tassadar/reports/tassadar_efficient_attention_baseline_matrix.json`,
  with the companion research summary at
  `fixtures/tassadar/reports/tassadar_efficient_attention_baseline_summary.json`;
  `psionic-eval` now freezes dense reference-linear, validated SparseTopK,
  linear/recurrent proxy, Reformer-style proxy, promoted HullCache, and
  research hierarchical-hull rows on the same article-class workload artifact,
  and `psionic-research` now makes the win/tie/lose/refuse posture explicit
  across those same workloads; promoted HullCache is fastest on `2` workloads,
  the research hierarchical-hull candidate is fastest on `4`, and the
  Reformer-style proxy explicitly refuses the long-loop and Sudoku rows rather
  than hiding unsupported locality assumptions behind dense-only comparisons
- the post-Phase-15 boundary-adapter follow-on now also exists in
  `psionic-models`, `psionic-eval`, `psionic-research`, `docs/audits/`, and
  nine preserved bounded artifact roots at
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v1`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v2`, and
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v3`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v4`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v5`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v6`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v7`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v8`,
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v9`, and
  `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v11`:
  the executor-attention family now carries both a bounded relative-target
  output-bias adapter, a bounded hidden-state-conditioned relative-target
  output projection adapter, a bounded previous-token-conditioned transition
  adapter, and a bounded trace-schema-conditioned adapter; the preserved
  `boundary_v1` artifact records the
  destructive output-head-only attempt (`10000` bps first-target but only
  `313` bps first-32), the accepted `boundary_v2` artifact shows the first
  honest attention-family boundary improvement that keeps the suffix mostly
  intact (`10000` bps first-target, `7500` bps first-8, `6875` bps first-32),
  the follow-on `boundary_v3` / `boundary_v4` artifacts show that merely
  adding and scaling the hidden-state-conditioned adapter leaves the learned
  gate flat, the newer `boundary_v5` / `v7` pair proves the
  previous-token-conditioned transition surface moves the learned blocker
  deeper into the trace (`10000` bps first-target, `8750` bps first-8,
  `7188` bps first-32), the later `boundary_v6` / `v8` joint fine-tune
  preserves but does not beat that ceiling, and the later `boundary_v7` /
  `boundary_v8` / `boundary_v9` saturation set plus
  `architecture_comparison_v11` prove that adding trace-schema bias, then
  per-position bias, then aggressive per-position gain still left all
  `32/32` checkpoints on the exact same red validation signature before the
  separate green `promotion_v3` continuation cleared the gate
- the first four-family same-corpus learned baseline comparison now also
  exists at `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v12`:
  it compares hull-specialized lookup, direct sparse-top-k lookup, hybrid
  attention, and recurrent/windowed lookup on the same bounded Sudoku-v0
  validation window while keeping every family explicitly comparison-only; all
  four seeded trainable families remain red at `0` bps first-target,
  first-8, and first-32 exactness, the recurrent family changes the declared
  long-trace contract from `flat_prefix_full_forward` to
  `incremental_decode_window`, and the hybrid attention family keeps its fit
  cliff explicit with `0/2` full-sequence fit cases under the current
  `512`-token bound
- the separate post-audit Phase 17 bar now also exists in `psionic-models`,
  `psionic-eval`, `psionic-research`, `docs/audits/`, and a canonical bounded
  compiled-lane bundle at
  `fixtures/tassadar/runs/sudoku_v0_compiled_executor_v0`:
  `psionic-models` now exposes a typed `TassadarCompiledProgramExecutor`
  compile-evidence bundle, `psionic-eval` now emits machine-readable exactness
  and compatibility/refusal reports for the real Sudoku-v0 corpus, and
  `psionic-research` now persists per-case compiled deployment bundles plus a
  top-level `run_bundle.json`; the committed artifacts keep the claim boundary
  tight by proving only a bounded compiled/proof-backed lane on matched
  Sudoku-v0 programs (`8/8` exact trace matches against CPU reference,
  `32/32` exact refusal matches on mismatched artifacts, `eval_only`
  posture), not arbitrary-program closure, not learned-lane success, and not
  article parity
- the separate post-audit Phase 18 bar now also exists in `psionic-runtime`,
  `psionic-models`, `psionic-eval`, `psionic-research`, `docs/audits/`, and a
  canonical bounded benchmark-plus-compiled bundle at
  `fixtures/tassadar/runs/hungarian_v0_compiled_executor_v0`:
  `psionic-runtime` now carries a real bounded `tassadar.wasm.hungarian_v0_matching.v1`
  min-cost matching program family over 4x4 cost matrices, `psionic-eval` now
  emits a real Hungarian-v0 benchmark package plus machine-readable compiled
  exactness/refusal reports and learned-vs-compiled lane status, and
  `psionic-research` now persists the full run bundle plus eight per-case
  deployments; the committed artifacts keep the claim boundary tight by
  proving only a bounded Hungarian-class workload contract plus an exact
  compiled/proof-backed lane on that matched corpus (`8/8` exact trace
  matches, `32/32` exact refusal matches, `eval_only` posture), not learned
  Hungarian execution, not general Hungarian solver parity, and not article
  parity
- the separate learned Hungarian-v0 research lane now also exists in
  `psionic-models`, `psionic-train`, and the canonical bundle root
  `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0`: the learned lane
  now carries explicit dual-state supervision plus token/state/final-result
  receipts, and the selected checkpoint keeps those boundaries honest
  (`aggregate=6839`, `first_target=0`, `first_32=6875`, `exact_traces=0`,
  `final_outputs=0`, `workload_specific_state=7568`, full sequences fit the
  current model window), so this is a bounded research-only learned lane and
  does not change the compiled Hungarian closure claim
- the repo now also carries the explicit learned long-horizon refusal policy at
  `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`,
  which now anchors the learned article-class bar to the exact committed
  Hungarian-10x10 benchmark corpus while still keeping broader learned
  long-horizon widening explicit; this keeps the acceptance language honest
  instead of leaving the learned horizon limit implicit
- the separate post-audit Phase 19 bar now also exists in `psionic-runtime`,
  `psionic-models`, `psionic-eval`, `psionic-research`, and a canonical
  exact compiled 9x9 bundle at
  `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0`:
  `psionic-eval` now packages the real 9x9 Sudoku corpus with benchmark and
  environment contracts and emits compiled exactness, refusal, and throughput
  receipts, while `psionic-research` persists four per-case deployments with
  compile proof, runtime execution proof, readable-log, and compact
  token-trace artifacts; the committed bundle proves exact compiled/proof-backed
  9x9 Sudoku closure on the matched corpus (`4/4` exact trace matches against
  CPU reference, `16/16` exact refusal matches on the full corpus,
  `eval_only` posture), which is the article-sized Sudoku result but still not
  full compiled article parity by itself
- the separate compiled article-sized matching bar now also exists in
  `psionic-runtime`, `psionic-models`, `psionic-eval`, `psionic-research`,
  and a canonical exact compiled 10x10 bundle at
  `fixtures/tassadar/runs/hungarian_10x10_v0_compiled_executor_v0`:
  `psionic-runtime` now carries a dedicated
  `tassadar.wasm.hungarian_10x10_matching.v1` profile and exact
  branch-and-bound matching programs over the committed 10x10 corpus,
  `psionic-eval` now emits benchmark/environment, exactness, refusal,
  throughput, and claim-boundary artifacts for that lane, and
  `psionic-research` now persists proof-bearing per-case deployments plus the
  top-level article-class run bundle; the committed bundle proves exact
  compiled/proof-backed 10x10 Hungarian closure on the matched corpus while
  keeping the boundary explicit: this is article-sized matching closure on the
  larger matching profile, not learned Hungarian execution and not full
  compiled article parity by itself
- the generic compiled article-kernel suite now also exists in
  `psionic-eval`, `psionic-research`, and the canonical bundle root
  `fixtures/tassadar/runs/compiled_kernel_suite_v0`: `psionic-eval` now
  packages bounded arithmetic, memory-update, forward-branch, and
  backward-loop kernel families under the article-shaped i32 profile and emits
  exactness, refusal, claim-boundary, and exactness-vs-trace-length scaling
  reports, while `psionic-research` persists twelve proof-bearing compiled
  deployments with runtime execution proofs for those regimes; the committed
  bundle proves exact compiled/proof-backed kernel closure across all four
  families while keeping the boundary explicit: this widens compiled article
  evidence beyond Sudoku and Hungarian, but it is still not arbitrary-program
  closure or full compiled article parity by itself
- the canonical coarse Tassadar claim vocabulary is now
  `compiled_exact`, `compiled_article_class`, `learned_bounded`,
  `learned_article_class`, and `research_only`; the canonical current
  compiled, learned, and research bundles now carry a machine-readable
  `claim_class`, while `claim_boundary`,
  `boundary_label`, and `serve_posture` remain the finer-grained limits
- the repo now also carries one machine-readable Tassadar acceptance report at
  `fixtures/tassadar/reports/tassadar_acceptance_report.json` plus one
  checker command at `scripts/check-tassadar-acceptance.sh`; that report keeps
  current bounded compiled, bounded learned, research-only, bounded fast-path,
  and now-green learned article-class truth explicit in one place
- the repo now also carries the final article-parity closeout audit at
  `docs/audits/2026-03-17-tassadar-article-parity-closeout-audit.md`; it is
  explicitly subordinate to the acceptance report and now records the green
  article-parity closeout at the committed benchmark-corpus scope
- the repo now also carries one machine-readable Tassadar Wasm
  instruction-coverage report at
  `fixtures/tassadar/reports/tassadar_wasm_instruction_coverage_report.json`,
  emitted by `cargo run -p psionic-runtime --example tassadar_wasm_instruction_coverage_report`;
  it inventories the supported `tassadar.wasm.*` profiles, the current
  article-shaped opcode universe, explicit workload/case coverage, and typed
  refusal examples for unsupported opcodes
- the repo now also carries one machine-readable Rust-only Tassadar source
  canon report at
  `fixtures/tassadar/reports/tassadar_rust_source_canon_report.json`, emitted
  by `cargo run -p psionic-eval --example tassadar_rust_source_canon_report`;
  it binds the article-closure frontend path to committed Rust fixtures for
  the kernel, heap-input, long-loop, Hungarian, and Sudoku families with
  source/toolchain/config/output lineage and keeps the older C receipt out of
  the article-closure claim path
- the repo now also carries one machine-readable Rust-to-Wasm article profile
  completeness report at
  `fixtures/tassadar/reports/tassadar_rust_article_profile_completeness_report.json`,
  emitted by
  `cargo run -p psionic-eval --example tassadar_rust_article_profile_completeness_report`;
  it freezes the current Rust-only article family into supported and refused
  module-shape, control-flow, table/global/indirect-call, numeric, and ABI
  rows, and the same publication is now bound into the Tassadar environment
  bundle and served capability-publication surfaces
- the repo now also carries one bounded Rust-only article ABI closure report
  at `fixtures/tassadar/reports/tassadar_article_abi_closure_report.json`,
  emitted by
  `cargo run -p psionic-eval --example tassadar_article_abi_closure_report`;
  it closes direct scalar `i32` entrypoints plus pointer-length `i32`
  heap-input entrypoints with one direct scalar `i32` return on the committed
  `param_abi_fixture` and `heap_sum_article` Rust sources, while keeping
  floating-point params, multi-result returns, and general host ABI closure as
  explicit refusals instead of over-reading the generic Wasm-lowering path
- the repo now also carries one canonical Rust-only Hungarian-10x10 article
  reproducer root at
  `fixtures/tassadar/runs/hungarian_10x10_article_reproducer_v1` plus the
  machine-readable report
  `fixtures/tassadar/reports/tassadar_hungarian_10x10_article_reproducer_report.json`,
  both emitted by
  `cargo run -p psionic-research --example tassadar_hungarian_10x10_article_reproducer`;
  they bind the committed Rust source canon receipt to one exact compiled
  `hungarian_10x10_test_a` deployment with readable log, compact token trace,
  compile lineage, runtime proof lineage, and explicit direct/no-fallback
  posture without widening the claim to Sudoku, million-step, or arbitrary
  program closure
- the repo now also carries one canonical Rust-only Sudoku-9x9 article
  reproducer report at
  `fixtures/tassadar/reports/tassadar_sudoku_9x9_article_reproducer_report.json`,
  emitted by
  `cargo run -p psionic-research --example tassadar_sudoku_9x9_article_reproducer`;
  it binds the committed Rust source canon receipt to the exact compiled
  `sudoku_9x9_test_a` search deployment under
  `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0`, freezes the
  committed 9x9 corpus case set, and makes the direct/no-fallback/no-external-tool
  posture explicit without widening the claim to Hungarian, million-step, or
  arbitrary program closure
- the repo now also carries one Rust-only article runtime closeout bundle at
  `fixtures/tassadar/runs/article_runtime_closeout_v1/article_runtime_closeout_bundle.json`,
  one eval report at
  `fixtures/tassadar/reports/tassadar_article_runtime_closeout_report.json`,
  and one research summary at
  `fixtures/tassadar/reports/tassadar_article_runtime_closeout_summary.json`,
  emitted by
  `cargo run -p psionic-eval --example tassadar_article_runtime_closeout_report`
  and
  `cargo run -p psionic-research --example tassadar_article_runtime_closeout_summary`;
  together they freeze the current runtime-performance closeout on the direct
  reference-linear CPU path for exactly two committed Rust-owned long-horizon
  workload families, `rust.long_loop_kernel` and
  `rust.state_machine_kernel`, at the declared `million_step` and
  `two_million_step` horizons with explicit throughput-floor checks,
  exactness receipts, and explicit `HullCache` / `SparseTopK` fallback-only
  posture instead of over-reading those long-horizon kernels as generic
  fast-path closure
- the repo now also carries one canonical direct model-weight execution proof
  report at
  `fixtures/tassadar/reports/tassadar_direct_model_weight_execution_proof_report.json`,
  emitted by
  `cargo run -p psionic-serve --example tassadar_direct_model_weight_execution_proof_report`;
  it freezes three representative canonical article workloads,
  `long_loop_kernel`, `sudoku_v0_test_a`, and `hungarian_matching`, on one
  route-bound direct executor lane with explicit direct/no-fallback,
  zero-external-call, and no-CPU-substitution proof receipts, and it is the
  operator-facing artifact that closes the current "inside the model weights"
  claim only for those committed workloads rather than for undeclared routes or
  future workloads
- the repo now also carries one canonical Rust-only article reproduction
  harness report at
  `fixtures/tassadar/reports/tassadar_rust_only_article_reproduction_report.json`,
  emitted by
  `cargo run -p psionic-serve --example tassadar_rust_only_article_reproduction`
  and wrapped by
  `./scripts/check-tassadar-rust-only-article-reproduction.sh`; it turns the
  current Rust-only article path into one executable operator procedure over
  source canon, profile completeness, ABI closure, Hungarian, Sudoku,
  million-step runtime closeout, and direct model-weight proof surfaces without
  widening the claim beyond those committed workloads and receipts
- the repo now also carries one historical non-canonical Tassadar C-to-Wasm
  compile receipt at
  `fixtures/tassadar/reports/tassadar_c_to_wasm_compile_receipt.json`, emitted
  by `cargo run -p psionic-runtime --example tassadar_c_to_wasm_compile_receipt`;
  it binds one committed C source fixture plus the emitted Wasm binary to
  source/toolchain/config/output digests, projects that success into one
  canonical `TassadarProgramArtifact` lineage contract, and surfaces typed
  compile refusals instead of hiding toolchain failure behind ad hoc scripts;
  this receipt is retained as a historical frontend artifact and is not part
  of the current Rust-only article path
- the repo now also carries one real Tassadar compile-pipeline matrix at
  `fixtures/tassadar/reports/tassadar_compile_pipeline_matrix_report.json`,
  emitted by `cargo run -p psionic-eval --example tassadar_compile_pipeline_matrix_report`;
  it binds exact Wasm-text multi-export arithmetic and memory-lookup fixtures,
  an explicit Wasm-text parameter-ABI lowering refusal, and a typed
  missing-toolchain refusal on one optional historical C-source row to
  compile-receipt digests,
  Wasm-module digests, exact lowered export outputs, and typed refusal posture
  for the current bounded source-to-Wasm-to-Tassadar lane; direct parameter
  and pointer-length article entrypoints now close through the separate
  Rust-only article ABI lane rather than by pretending the generic Wasm
  lowering boundary already widened
- the repo now also carries one bounded Wasm-module ingress artifact at
  `fixtures/tassadar/reports/tassadar_wasm_module_ingress_report.json`, emitted
  by `cargo run -p psionic-eval --example tassadar_wasm_module_ingress_report`;
  it binds one real committed Wasm binary plus one canonical synthetic
  multi-function module to runtime-visible module summaries, normalized-module
  digests, section-level round-trip digests, exact lowered export outputs, and
  typed refusal when the current runtime boundary still blocks lowering
- the repo now also carries one bounded Tassadar Wasm conformance report at
  `fixtures/tassadar/reports/tassadar_wasm_conformance_report.json`, emitted
  by `cargo run -p psionic-eval --example tassadar_wasm_conformance_report`;
  it differentially checks the current bounded module-execution lane against
  `wasmi` over curated and deterministically generated module cases, keeping
  exact success, trap parity, and explicit unsupported-host boundary refusal
  separate instead of pretending the lane already closes arbitrary Wasm
- the canonical operator guide for the current bounded Wasm lane now lives at
  `docs/TASSADAR_WASM_RUNBOOK.md`, and the latest live status audit is
  `docs/audits/2026-03-18-tassadar-wasm-flow-status-audit.md`; together they
  separate optional local C-toolchain prerequisites from the repo-owned
  compile-pipeline, ingress, conformance, module-scale, and trap/refusal
  surfaces that should reproduce on a clean checkout
- public claim discipline for that lane is explicit in
  `docs/ARCHITECTURE.md`, the bridge notes in
  `docs/ROADMAP_TASSADAR.md`, and `docs/TASSADAR_WASM_RUNBOOK.md`:
  "supports Wasm" means a named Tassadar profile inside a frozen WebAssembly
  spec window, not an open-ended claim about the whole moving language
- its Phase 8A research family now exists in `psionic-research`, with a typed
  executor-variant family, benchmark/proof/lineage-backed bounded runs, and
  machine-readable sweep records for reproducible same-contract comparisons
- its Phase 8B sparse-top-k path now exists in `psionic-runtime`,
  `psionic-models`, and `psionic-eval`, with a validated direct decode mode,
  explicit fallback on unsupported shapes, and published sparse-top-k
  throughput/speedup/CPU-gap reporting alongside CPU, linear, and hull lanes
- its Phase 9A hybrid planner route now exists across `psionic-serve`,
  `psionic-router`, and `psionic-provider`, with an explicit
  `psionic.planner_executor_route` contract, benchmark-gated route capability
  descriptors, direct-vs-fallback route posture, replay-stable routing
  decisions, typed completed/fallback/refused outcomes, and planner-visible
  policy, budget, proof, selection, and refusal truth
- its article-hybrid workflow follow-on now also exists in `psionic-serve`,
  with the specialized `psionic.article_hybrid_workflow` contract bound to
  canonical article cases, preserved benchmark identity plus routing/proof
  receipts, and the committed artifact
  `fixtures/tassadar/reports/tassadar_article_hybrid_workflow_artifact.json`
- its Tassadar lab follow-on now also exists in `psionic-serve`, with the
  local replay/live adapter that projects `psionic.article_executor_session`,
  `psionic.article_hybrid_workflow`, and the canonical compiled or learned
  report bundles into one renderer-neutral snapshot/update surface consumed by
  desktop panes, with committed evidence at
  `fixtures/tassadar/reports/tassadar_lab_surface_artifact.json`
- its Phase 9A article-workload serving follow-on now also exists in
  `psionic-serve`, with a specialized `psionic.article_executor_session`
  surface that resolves canonical article workloads by case id, preserves
  benchmark and proof identity across the serving boundary, and emits derived
  readable-log plus symbolic token-trace views without pretending the session
  is ordinary tool use
- its Phase 9B bounded executor-training lane now exists in `psionic-train`,
  with a small-model Tassadar trainer over package-backed supervision, a
  fixed-budget train receipt, proof-aware exactness comparison against the
  handcrafted reference lane, and explicit validation-corpus-only scope claims
- its learned structural-supervision follow-on now also exists across
  `psionic-models`, `psionic-train`, `psionic-eval`, and `psionic-research`:
  the tokenizer now classifies instruction-pointer, branch-outcome,
  stack-delta, memory-diff, and workload-specific target families, the
  training manifest persists profile weights plus split-level coverage
  inventory, the run bundle persists a validation
  `structural_supervision_report.json`, and the bounded comparison root at
  `fixtures/tassadar/runs/sudoku_v0_supervision_ablation_v1` proves the richer
  targets changed the learned lane without widening the claim boundary
  (`4570` to `7812` aggregate target-token exactness, `4375` to `6875`
  first-32 exactness, `+2000` instruction-pointer bps, and `+3333`
  stack-delta bps versus the matched next-token-only baseline)
- its Phase 9C compiled-weight and larger-2D exploration now exists in
  `psionic-models` and `psionic-research`, with program-specialized compiled
  executor artifacts carrying exact program binding, runtime-contract truth,
  and compile-time proof/runtime-manifest lineage, plus the shared
  `fixtures/tassadar/reports/tassadar_program_to_weights_benchmark_suite.json`
  report that compares direct reference-linear execution against compiled-weight
  deployment on the same widened Wasm workloads, and explicit 2D-head family
  geometry and compiled-weight suite outputs in research runs
- its module-aware program-to-weights specialization follow-on now also exists
  across `psionic-runtime`, `psionic-models`, and `psionic-research`, with a
  public module-specialization plan over normalized Wasm module structure plus
  call-graph reachability, a research-only shared module-specialization
  artifact that preserves per-export compiled lineage and exactness facts, and
  the deterministic
  `fixtures/tassadar/reports/tassadar_module_specialization_benchmark.json`
  report that compares shared module-specialized artifact size plus modeled
  dispatch cost against today's per-export compiled lane while keeping
  import-boundary cases explicit refusal evidence
- its Phase 9D learned-plus-compiled and learned-circuit research program now
  exists in `psionic-research`, with a typed research-only family that
  benchmarks explicit proxy surfaces against the handcrafted Wasm baseline and
  the bounded small-executor training lane while keeping proof expectations and
  claim boundaries machine-legible
- it is not current MVP compute-market product scope
- it is not a claim that Psionic is replacing native CPU execution
- its landed Phase 0/1/2/3/4/5/6/7A/7B/8A/8B/9A/9B/9C/9D issue spine was first
  tracked in the original `openagents` backlog:
  [#3743](https://github.com/OpenAgentsInc/openagents/issues/3743) and
  [#3744](https://github.com/OpenAgentsInc/openagents/issues/3744) and
  [#3745](https://github.com/OpenAgentsInc/openagents/issues/3745) and
  [#3746](https://github.com/OpenAgentsInc/openagents/issues/3746) and
  [#3747](https://github.com/OpenAgentsInc/openagents/issues/3747) and
  [#3748](https://github.com/OpenAgentsInc/openagents/issues/3748) and
  [#3749](https://github.com/OpenAgentsInc/openagents/issues/3749) and
  [#3760](https://github.com/OpenAgentsInc/openagents/issues/3760) and
  [#3761](https://github.com/OpenAgentsInc/openagents/issues/3761) and
  [#3762](https://github.com/OpenAgentsInc/openagents/issues/3762) and
  [#3763](https://github.com/OpenAgentsInc/openagents/issues/3763) and
  [#3764](https://github.com/OpenAgentsInc/openagents/issues/3764) and
  [#3765](https://github.com/OpenAgentsInc/openagents/issues/3765) and
  [#3766](https://github.com/OpenAgentsInc/openagents/issues/3766) and
  [#3767](https://github.com/OpenAgentsInc/openagents/issues/3767)

## Crate Map

### Framework Core

- `psionic-core`: canonical tensor, shape, dtype, device, layout, and bounded
  advanced-dtype plus autocast-style precision-policy semantics contract.
- `psionic-ir`: graph, autodiff, `detach`, no-grad/training posture, and
  execution-plan types plus tensor-family capability matrices for dense,
  sparse, nested, masked, and storage-aware semantics, plus the first public
  `grad` / `value_and_grad` / `vjp` / `jvp` / `checkpoint` transform objects
  above `AutodiffGraph`, plus graph-scoped `custom_vjp` registration hooks.
- `psionic-array`: first public lazy-array facade above `psionic-core` and
  `psionic-ir`, including context-owned graph construction, public device and
  stream handles with honest unified-memory flags and dependency-policy truth,
  graph-backed arithmetic, scalar and filled-array creation helpers,
  reshape/permute/transpose/flatten/expand_dims/squeeze/slice/select/concat/
  broadcast view families, seeded or best-effort random uniform/normal
  creation, dtype casts, `arange`/`linspace`/`eye` helpers, axis-aware sum
  reduction, explicit `eval` / `async_eval` boundaries, explicit host-owned
  typed buffer export, singleton `item()` extraction, deterministic tree
  flatten/map/unflatten utilities, bounded runtime resource reporting with
  active/peak/cache counters plus cache-limit and reset controls, bounded
  backend debug snapshots/captures above `psionic-compiler` and
  `psionic-runtime`, a machine-readable MLX CPU-reference coverage report over
  imported `array_core`/`ops_numeric`/`device_eval_memory` families, fallible
  `ArrayContext::metal()` / `metal_seeded()` and `ArrayContext::cuda()` /
  `cuda_seeded()` constructors backed by the selected runtime Metal or CUDA
  device, bounded actual Metal execution for dense `constant`/`add`/`matmul`
  graphs plus widened CUDA execution for dense
  `constant`/`add`/`mul`/`matmul` graphs and backend-specialized `rms_norm`,
  with explicit refusal outside those slices and dense-`f32` numerics
  disclosure, bounded extension authoring and dispatch-resolution above
  `psionic-ir`'s extensible operator registry, and snapshot graph export for
  the current output set.
- `psionic-array-io`: public array artifact import/export companion above
  `psionic-array`, with stable receipts, explicit dtype and quantization truth,
  single-array `npy`, multi-array `npz`, multi-array `safetensors`, and a
  bounded dense GGUF import/export bridge that dequantizes GGUF block storage
  to logical `f32` on import instead of hiding storage changes.
- `psionic-function-io`: public function artifact companion above
  `psionic-ir` and `psionic-compiler`, with digest-bound native `.psifn`
  export-safe graphs, optional compiler artifacts, trace-family identity,
  optional deployment bundle binding, stable import/export receipts, and a
  bounded `.mlxfn` compatibility shell with explicit refusal outside the
  current subset.
- `psionic-distributed`: first public framework-distributed group, core
  collective-helper, and bounded launch/config planning surface above current
  runtime mesh truth, with explicit mesh bootstrap, reusable global group
  initialization, honest singleton fallback, ordered member/rank snapshots,
  explicit-plan subgroup split semantics, MLX-style singleton passthrough for
  `all_sum` / `all_gather` / `reduce_scatter`, explicit host-owned reference
  emulation for multi-rank `all_sum` / `all_gather` / `reduce_scatter` and
  `recv`, validation-only `send`, hostfile parsing, honest
  single-rank-per-node launch validation, cluster membership/address/backend
  readiness checks, sandbox contract preflight, per-rank bootstrap payloads
  and sandbox job plans, distributed reserved-environment synthesis, cluster
  execution evidence, stable plan digests, tree-aware `grouped_all_sum` /
  `grouped_all_reduce`, floating-point `average_gradients`, and bounded
  MLX-style `AllToShardedLinear` / `ShardedToAllLinear` wrappers with
  deterministic row/column sharding plus explicit reference-emulated
  multi-rank reconstruction, and a bounded MLX-style
  `fsdp_apply_gradients` helper above distributed optimizer contracts with
  typed `zero_stage3` admission, mixed replicated/full-shard handling,
  optional global-norm clipping, shard-local optimizer updates, gathered
  full-parameter reconstruction, and explicit backend-family capability
  mapping for `ring`, `mpi`, and `nccl`-class requests plus typed `jaccl`
  refusal over current topology profiles, while multi-rank backend transport
  execution remains later work.
- `psionic-compat`: machine-readable compatibility claim vocabulary, current
  PyTorch-facing semantics posture aggregation, the bounded MLX version-window
  or claim-language contract, the MLX acceptance-matrix report contract, and
  the seeded MLX parity-harness report plus the MLX compatibility matrix.
- `psionic-nn`: reusable public `Module` tree, parameter, buffer, explicit
  freeze posture, filtered recursive parameter discovery, deterministic
  state-dict/state-tree semantics, bounded public `save_weights` /
  `load_weights` wrappers, and a bounded CPU-reference core layer surface
  covering linear, embedding, norms, activations, dropout, conv, and pooling
  above `psionic-core`, plus bounded CPU-reference losses, initializers, and
  helper functions for tiny training loops, plus a bounded public optimizer and
  scheduler shell with module-path keyed state, parameter-group scaling,
  multi-optimizer composition, and snapshot restore, including strict and
  non-strict keyed load behavior, plus an eval-oriented quantized-module shell
  with `Module::quantize(...)`, explicit quantize reports, and
  `QuantizedLinear` / `QuantizedEmbedding` wrappers over `int8_symmetric`
  block storage and dequantize-to-`f32` forward semantics.
- `psionic-transformer`: reusable Transformer architecture primitives above
  `psionic-nn` and below `psionic-models`, including shared decoder configs and
  architecture-level `AttnRes` config/tensor/state surfaces that should not
  live in model-governance or artifact-loader code.
- `psionic-compiler`: lowering, scheduling, replay-stable program identity,
  compiler diagnostics, and the first public compile-transform surface with
  explicit purity, concrete-plan cache identity, bounded shapeless trace-family
  identity, trace capture, and debug posture.
- `psionic-runtime`: runtime traits, allocators, compiled-plan execution,
  local-multi-device truth, and canonical execution-proof bundles.
- `psionic-catalog`: local blob, artifact, and model-catalog substrate used by
  model and serving layers.
- `psionic-mlx-lm`: bounded local MLX-style text package and CLI above the
  native GGUF runtime.
- `psionic-mlx-catalog`: bounded MLX-style model-catalog and local Hugging
  Face cache workflow package above `psionic-catalog` and `psionic-mlx-lm`.
- `psionic-mlx-serve`: bounded MLX-style text-serving package that resolves
  MLX model references and boots the shared OpenAI-compatible Psionic server.
- `psionic-mlx-vlm`: bounded MLX-style multimodal package with processor
  registries, image/audio/video request shapes, and served-request planning
  over the shared text-serving lane.
- `psionic-mlx-audio`: bounded MLX-style audio package with CPU-reference
  synthesis, WAV IO, codec helpers, streaming chunk contracts, and
  server-facing speech request surfaces.
- `psionic-mlx-recipes`: bounded MLX-style training-recipe package and CLI
  above `psionic-train`, including method inventory plus plan emission for
  SFT, LoRA/DoRA/QLoRA, preference, and RL-family recipes.
- `psionic-mlx-workflows`: bounded MLX-style workflow package for synthetic
  datasets, reward/judge helper plans, adapter merge/export, and local publish
  snapshots above the shared data/train substrate.
- `psionic-mlx-bench`: bounded MLX-style benchmark package over
  `psionic-eval`, `psionic-mlx-lm`, and `psionic-mlx-vlm`, with suite
  manifests, local text and served provider adapters, multimodal projection,
  and repeated benchmark receipts.

### Backend And Platform Lanes

- `psionic-backend-cpu`: CPU backend and the current reference execution lane.
- `psionic-backend-metal`: Metal backend with the first embeddings and local
  Apple execution path.
- `psionic-backend-cuda`: CUDA backend architecture and truthful readiness
  surface.
- `psionic-backend-amd-kfd`: AMD KFD discovery/readiness substrate.
- `psionic-backend-amd-userspace`: AMD userspace discovery/readiness substrate.
- `psionic-apple-fm`: Apple Foundation Models bridge contracts and Rust client
  for the Swift sidecar.

### Network, Transport, And Execution Control

- `psionic-net`: peer identity, direct/NAT/relay sessions, rendezvous, trust
  state, and service-tunnel transport seams.
- `psionic-datastream`: resumable manifests, chunk transport, policy-weight
  broadcast refs, freshness windows, and delivery receipts.
- `psionic-cluster`: ordered-state, admission, catch-up, scheduling, and
  clustered topology truth over `psionic-net`.
- `psionic-sandbox`: bounded execution profiles, runtime detection,
  background-job lifecycle, file transfer, and repeated agentic iteration
  receipts.
- `psionic-collectives`: elastic device-mesh and benchmark-gated sync planning
  for training-class collectives.

### Serving And Adapter Surface

- `psionic-models`: model descriptors, artifact-loading seams, tokenizer
  hooks, and model-family wrappers above `psionic-transformer`.
- `psionic-serve`: served compute contracts for chat, responses, embeddings,
  scheduling, structured output, tool calling, adapter-backed execution, and
  the bounded local AttnRes text-generation surface.
- `psionic-router`: multi-model routing, worker inventory, policy filters,
  warm/cache-aware placement, and served-fleet reliability controls.
- `psionic-provider`: provider-facing capability, readiness, and receipt types
  derived from Psionic execution truth.
- `psionic-adapters`: adapter identity, packaging, Apple `.fmadapter`
  parsing/writing, lineage, and hosted binding semantics.

### Data, Eval, Training, And Research

- `psionic-data`: versioned dataset manifests, tokenizer digests, split
  declarations, streamed iteration, and packing contracts.
- `psionic-environments`: environment package ABI, workload/difficulty/policy
  contracts, tool/rubric hooks, deterministic runtime sessions, train/eval
  parity helpers, and the `Tassadar` exact-executor environment bundle.
- `psionic-eval`: held-out eval runs, rubric-scored samples, benchmark
  packages, repeat-run aggregation, local validator simulation, Apple
  adapter eval harnesses, and the `Tassadar` package-driven exactness
  benchmark suite with CPU/reference-linear/hull-cache/sparse-top-k baselines
  and exact-equivalence reporting plus runtime capability/selection artifacts.
- `psionic-train`: checkpoint/recovery truth, elastic membership, run graphs,
  rollout-worker protocol, orchestrator control, fixed-budget training core,
  parameter-group and scheduler semantics, replay-truth and reproducibility
  semantics, Apple training execution, Apple SFT/export, model-IO
  compatibility boundaries, optional Apple draft-model distillation, and the
  bounded `Tassadar` small-executor training lane.
- `psionic-research`: typed experiment specs, bounded run manifests, result
  summaries, promotion records, the AttnRes residual-vs-AttnRes comparison
  bundle, and the `Tassadar` executor-variant research family with
  machine-readable sweep records for hillclimb/research loops.

### Support Tree

- `docs/`: canonical specs, acceptance matrices, runbooks, and audits.
- `fixtures/`: repo-owned fixture corpora such as Apple adapter reference
  inputs.
- `scripts/`: Psionic-specific harnesses and validation helpers.

The crate list and layering are canonical for current ownership and dependency
direction, but they are not a guarantee that every planned subsystem will land
under exactly these final crate names.

## Design Principles

- Keep machine-facing execution truth in reusable crates and keep product truth
  above Psionic.
- Keep the compiler and runtime visible and inspectable.
- Keep crate ownership narrow and documented.
- Preserve a strict boundary between reusable engine crates and OpenAgents
  provider integration.
- Prefer explicit capability/refusal surfaces over vague "supported" claims.
- Make artifacts, manifests, and receipts first-class instead of hidden side
  effects.
- Model backend families explicitly; AMD KFD and AMD userspace are separate
  backends, not one hidden toggle.
- Keep inference, embeddings, adapters, eval, and training-class substrates
  first-class in architecture from the start.

## Current Phase

Psionic is in an implemented-substrate, not-yet-complete-engine phase.

That means the repo already has a real execution tree for local serving,
adapter hosting, bounded sandbox work, early eval/train/research lanes, and a
narrow Apple adapter training path, but it still does not claim complete
backend parity or fully generalized distributed training.

### Apple Foundation Models Status

The Apple Foundation Models lane in this standalone repo has two distinct
pieces:

- the Rust-side contract and client surface in `psionic-apple-fm`
- the repo-owned Apple adapter package, train, eval, and fixture lanes in
  `psionic-adapters`, `psionic-train`, `psionic-eval`, and
  `fixtures/apple_adapter/`

What ships here today:

- `psionic-apple-fm` defines the Rust client, request/response contracts,
  structured-output helpers, transcript/tool types, and error surface for an
  Apple FM bridge
- `psionic-adapters` can parse, validate, write, and bind Apple
  `.fmadapter` packages
- `psionic-train` owns a bounded adapter-only Apple training/export lane
- `psionic-eval` and `fixtures/apple_adapter/` own benchmark/eval fixtures and
  reference reports for that lane
- the standalone repo now includes a committed repo-local Apple overfit proof
  at `fixtures/apple_adapter/runs/psionic_architecture_explainer_reference_overfit_report.json`

What does not ship here:

- the Swift bridge implementation itself
- app-owned bridge supervision or packaging
- `autopilotctl` operator flows
- the old release harnesses that lived in the parent `openagents` repo

In concrete terms, this repo can train and export LoRA-style Apple adapter
packages and can evaluate them against repo-owned fixtures, but loading those
packages into a live Apple runtime still depends on external bridge and
operator tooling outside this repository.

The honest current scope is:

- frozen-base, adapter-only training over explicit low-rank parameter groups
- `f32` reference precision only
- graph-level checkpoint transforms exist in `psionic-ir`, but activation
  checkpointing remains disabled in the shipped Apple reference lane
- held-out eval and package/runtime contract validation are repo-owned here
- the weak Apple `overfit_non_zero` benchmark gate is proven in-repo by the
  reference overfit report, but the stronger standard usefulness bar is still
  a separate claim
- benchmark-useful quality remains a separate claim from package validity or
  export success

What this does not mean is "full distributed Apple training is done." The
current Apple lane reuses the repo's data, environment, eval, optimizer, and
autodiff substrate, but it does not yet execute through real
`psionic-cluster` multi-node training, collective-backed parameter exchange,
sharded optimizer state, or production multi-device training kernels. Those
cluster and distributed-training contracts already exist as Psionic substrate
and are intended to be reused later for broader training lanes, but the
current Apple adapter path is still a narrow single-host reference execution
lane.

Implemented now:

- `psionic-catalog` local blob and artifact-catalog substrate for model and
  runtime-facing assets.
- `psionic-mlx-lm` bounded local GGUF text package and CLI above that native
  substrate.
- `psionic-mlx-catalog` bounded model-id, Ollama, and local Hugging Face cache
  resolution/reporting layer above the same substrate.
- `psionic-mlx-serve` bounded MLX-style text-serving package over
  `psionic-mlx-catalog` and `psionic-serve`, with machine-readable bootstrap
  reports plus package-owned `plan`/`serve` CLIs for `/v1/chat/completions`
  and `/v1/responses`.
- `psionic-mlx-vlm` bounded MLX-style multimodal package with builtin
  processor registries for `llava`, `qwen2_vl`, and `omni`-class families,
  OpenAI-compatible image/audio/video request shapes, digest-bound attachment
  receipts, and text-serving request plans over the shared server.
- `psionic-mlx-audio` bounded MLX-style audio package with builtin
  `kokoro`, `xtts`, and `encodec`-class family metadata, quantized-checkpoint
  descriptors, WAV IO, text-to-speech and speech-to-speech request contracts,
  stream-chunk outputs, and a CPU-reference audio server contract.
- `psionic-mlx-recipes` bounded MLX-style training-recipe package over
  `psionic-train`, with machine-readable method inventory plus `plan` and
  `methods` CLIs for SFT, adapter, preference, and RL-style recipe families.
- `psionic-mlx-workflows` bounded MLX-style workflow package over
  `psionic-data`, `psionic-mlx-recipes`, and `psionic-train`, with synthetic
  SFT/preference dataset bundles, reward/judge helper plans, adapter merge
  artifacts, and a local Hugging Face-style publish snapshot.
- `psionic-mlx-bench` bounded MLX-style benchmark package over
  `psionic-eval`, `psionic-mlx-lm`, and `psionic-mlx-vlm`, with
  machine-readable suite manifests, local text and served provider adapters,
  multimodal prompt projection, and repeated benchmark receipts.
- CPU baseline plus a first Metal-backed `psionic.embeddings` lane.
- generic CPU GGUF decoder execution for GPT-OSS plus representative Llama,
  Qwen, and Mistral families through one Psionic-owned runtime surface.
- generic `psionic-openai-server` boot and model inventory for GPT-OSS plus
  non-GPT-OSS GGUF families on one `/v1/chat/completions` surface, plus
  safetensors-backed embeddings on `/v1/embeddings` and a first Psionic-owned
  `/v1/responses` surface, with per-model endpoint support reported explicitly.
- a first explicit non-GPT-OSS generic-server pilot for the Qwen family, with a
  dedicated end-to-end runbook and harness proving family inventory, scheduler
  headers, and scheduler receipts survive the same Psionic-owned runtime and
  server path as GPT-OSS.
- a first integrated structured-agent weather pilot, proving structured JSON
  output, response-state continuation, router-owned tool loops, and cache or
  route truth together in one Psionic-owned workload.
- explicit CPU-lane residency, fallback, and unsupported-control truth on that
  generic server surface instead of vague accelerator claims.
- explicit local-backend truth on the GPT-OSS server surface too, including
  native Metal, native CUDA, and explicit `llama.cpp` proxy posture with
  machine-checkable hybrid-offload labels instead of silent proxy or hybrid
  claims.
- Psionic-owned structured-output contracts on the generic server for choice,
  regex, grammar, `json_object`, `json_schema`, and tagged-structure cases via
  one shared request shape, explicit per-model capability reporting, response
  headers, and machine-readable structured values instead of hidden
  prompt-only conventions or string re-parsing.
- Psionic-owned tool-calling contracts on the generic server via `tools` plus
  `tool_choice`, with explicit `none` / `auto` / `required` / named modes,
  tagged tool envelopes, schema-backed argument validation, and
  machine-readable tool-call surfaces on both normal and streaming chat
  responses.
- a router-owned tool-loop boundary for those tool calls, with explicit
  multi-step model/tool receipts, provider descriptors, MCP-aware gateway
  seams, history-visibility controls, and refusal of hidden tool results
  instead of burying agent loops inside worker runtimes or app-local glue.
- Psionic-owned reasoning parser seams for reasoning-bearing families, starting
  with GPT-OSS / Harmony: typed parsed-response envelopes now separate final
  content, reasoning content, and side channels; `psionic_reasoning` request
  policy can explicitly separate or suppress reasoning; and both chat plus
  responses surfaces can return typed reasoning-aware response fields without
  falling back to raw-string scraping alone.
- Psionic-owned response-state and conversation contracts on `/v1/responses`,
  with router-owned pluggable in-memory or JSON-file backends, explicit
  response and conversation identifiers, truthful prompt-replay-only cache
  behavior, restart-safe local continuation on durable backends, per-model
  capability reporting, and explicit refusal for unsupported continuation
  modes instead of pushing multi-turn state emulation into callers.
- a first Psionic-owned router control plane for served fleets, with explicit
  worker/model inventory, capability filters, warm/cache-aware placement,
  bounded power-of-two least-loaded choice over warm or cache-matched pools,
  and generic-server route headers so model routing no longer lives as ad hoc
  alias logic inside `psionic-serve`.
- router-owned reliability controls for served fleets, with explicit queue
  depth, retry/refusal traces, rate-limit actions, circuit-breaker state, and
  health gating in `psionic-router` instead of app-specific failure handling.
- a first truthful adapter-serving lane for dense CPU GGUF decoder families,
  with LM-head LoRA import from safetensors, explicit attach/detach plus
  merge/unmerge residency modes, adapter compatibility/refusal surfaces, and
  real adapter-backed generation instead of metadata-only parsing or silent
  fallback to the base model.
- Apple Foundation Models bridge contracts plus live adapter inventory,
  load/unload, attach/detach, and request-level adapter binding through
  `psionic-apple-fm` and the Swift bridge sidecar.
- a first Psionic-owned continuous-batching scheduler for CPU text generation,
  with mixed prefill/decode admission, FIFO queue truth, per-request scheduling
  receipts, and generic-server execution headers instead of a hard-coded
  `single_request_only` posture on the shared local server lane.
- a real request-owned block/paged KV manager behind that scheduler, with page
  allocation, reclaim, eviction, session/request/shared-prefix owner bindings,
  and explicit KV ownership receipts across CPU and GPT-OSS execution paths.
- automatic shared prefix caching on top of that KV substrate, with explicit
  tenant/session and sampler boundaries, request-level auto/bypass/invalidate
  controls, refusal/invalidation receipts, and generic-server headers for
  prefix hit/miss/bypass truth.
- Psionic-owned prefill/decode capability contracts on top of that scheduler
  and KV substrate, with colocated and KV-transfer handoff seams, separate TTFT
  and ITL metrics, scheduler receipts, and generic-server headers that surface
  the realized prefill/decode mode instead of treating PD behavior as hidden
  runtime detail.
- hierarchical KV residency accounting across host, device, and explicit
  datastream-backed distributed tiers, with spill/prefetch/write-back movement
  truth, refusal surfaces, and cluster cache-capability reporting that only
  claims the tiers the lane can actually surface.
- one canonical serving-semantics model shared across local and clustered
  serving, with execution-profile, cache, and warm-route truth surfaced on
  whole-request, replica-routed, pipeline-sharded, layer-sharded, and
  tensor-sharded evidence paths.
- `psionic-net` direct, NAT, and relay session establishment.
- `psionic-cluster` ordered state, admission, catch-up, and clustered serving
  topology truth across replica, pipeline, layer-sharded, and tensor-sharded
  variants.
- sharded-model manifests, staged artifact residency, and clustered prefix or
  KV-cache compatibility truth.
- `psionic-datastream` resumable dataset and checkpoint delivery, now including
  explicit checkpoint-backed KV external locator contracts for distributed cache
  tiers.
- benchmark-backed quantization dispatch plus low-level batching and parking
  hooks used by serve and datastream layers.
- explicit policy-weight shard manifests, lightweight control-plane refs,
  freshness windows, mirror metadata, and assembled broadcast receipts on top
  of the resumable datastream plane.
- `psionic-sandbox` runtime detection, bounded execution, background jobs,
  file-transfer lifecycle, warm reusable pools, staged loop inputs, and
  repeated agentic iteration receipts.
- canonical execution-proof bundles and embeddings-first activation-fingerprint
  proof posture.
- early train substrate: checkpoint-backed recovery, elastic membership,
  bandwidth-aware local/global sync planning, typed fixed-budget trainer
  steps, explicit checkpoint pointers and checkpoint manifests, restore
  receipts with declared recovery modes, checkpoint-anchored restore, explicit
  run graphs, contributor-set revisions, stage-program identity across
  `general_sft` / `agentic_sft` / `rl`, typed SFT trace lineage, window
  lifecycle, first orchestrator state, rollout-admission receipts, bounded
  off-policy freshness budgets, worker heartbeats, claims, upload receipts,
  and adapter lineage.
- early RL substrate: checkpoint-aware policy revisions, proof-bearing rollout
  artifacts, deterministic trainer-batch assembly, explicit policy-lineage
  digests, quarantined-versus-discarded stale-rollout pruning, typed
  rollout-validation bundles or verdicts, and a first curriculum controller
  with difficulty- and advantage-aware sample filtering plus explicit
  halt/quarantine verdicts inside `psionic-train`.
- early data substrate: versioned dataset manifests, tokenizer digests, split
  declarations, resumable streamed-iteration contracts, and long-context
  packing policies in `psionic-data`, with environment packages now binding
  versioned dataset keys instead of free-form dataset refs.
- early environment substrate: a Psionic-native package ABI, tool interfaces,
  rubric hooks, expected artifact contracts, reference runtime sessions,
  digest-pinned package aliases, mixed-surface composition groups, and
  train/eval parity receipts in `psionic-environments`, keyed to the same
  `environment_ref@version` identity used by kernel authority.
- early eval substrate: held-out eval runs, rubric-scored sample/runtime
  contracts, benchmark packages with repeat-run aggregation, and operator-local
  validator simulation in `psionic-eval`, while kernel/Nexus still own
  canonical eval-run authority truth.
- a first repo-owned Apple training lane in `psionic-train`, including the
  Apple training execution backend, Apple adapter SFT/export, and optional
  Apple draft-model distillation.
- a first integrated `agentic_sft -> rl` reference program in `psionic-train`,
  proving environment packages, dataset and checkpoint lineage, datastream
  policy-weight delivery, sandbox reuse, rollout-worker protocol, validator
  verdicts, benchmark aggregation, and one fixed-budget trainer step together
  in one typed report instead of isolated subsystem tests.
- a first explicit distributed-optimizer contract in `psionic-train`, making
  parameter sharding, gradient accumulation, optimizer-state sharding,
  precision policy, activation checkpointing, long-run memory planning, and
  collective sync attachment machine-legible on top of the fixed-budget trainer
  core.
- a first typed model-IO portability layer in `psionic-train`, making
  state-dict traversal, training-group assignment, safetensors export/import,
  torch-style JSON state artifacts, GGUF import, tokenizer version binding,
  and adapter merge/unmerge explicit instead of ad hoc.
- a first deterministic replay-truth layer in `psionic-train`, making replay
  seeds, sample-selection rules, environment and tool pins, eval posture, and
  replay drift verification machine-legible instead of scattered across
  receipts.
- a first train-security posture layer in `psionic-train`, making environment
  verification, artifact trust roots, untrusted-worker admission, poisoning
  controls, and validator-bound security receipts explicit instead of
  hand-waved around the rollout validator.
- `psionic-research` experiment specs, bounded run manifests, and promotion
  records for hillclimb-style research loops.
- broader-stack authority flows for environment packages, checkpoint-family
  policies, validator policies, benchmark packages, training policies, eval
  runs, training runs, accepted outcomes, and synthetic-data jobs now exist
  outside Psionic in kernel or Nexus surfaces.
- a narrow broader-stack Apple adapter-hosting and Apple-training projection
  now exists above Psionic in provider-substrate, desktop-control, and
  compute-market docs, without implying a generalized training market.

Still planned:

- full inference-engine maturity across model families and broader serving
  surfaces.
- richer eval-policy productization and persistent environment publication or
  authority sync.
- broader distributed training completion, freshness or validator policy, and
  orchestrator layers.
- deeper benchmark or validator policy for training-class lanes.
- policy-meaningful runtime and environment manifests plus proof-bearing
  session-claims discipline for clustered and sandboxed execution.
- AMD execution support.

For canonical current-state detail, use `docs/ARCHITECTURE.md` and
`docs/TRAIN_SYSTEM.md` rather than treating this README as the full system spec.

## Docs

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — canonical Psionic-wide
  system spec covering layering, work classes, artifact and receipt model,
  execution lifecycle, failure, and security boundaries.
- **[docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md](docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md)** —
  canonical framework-core acceptance split for tensor, compiler, IO, replay,
  and local multi-device behavior.
- **[docs/OPERATOR_PARITY_MATRIX.md](docs/OPERATOR_PARITY_MATRIX.md)** —
  canonical seeded operator parity artifact for the current PyTorch-derived
  `OpInfo`-style coverage slice.
- **[docs/ADVANCED_OPERATOR_PROGRAM_MATRIX.md](docs/ADVANCED_OPERATOR_PROGRAM_MATRIX.md)** —
  canonical bounded advanced-operator program matrix for linalg, signal,
  attention, and explicit refusal posture for distribution and special-function
  families.
- **[docs/PROGRAM_TRANSFORM_CAPABILITY_MATRIX.md](docs/PROGRAM_TRANSFORM_CAPABILITY_MATRIX.md)** —
  canonical bounded capability matrix for functionalization, symbolic rewrites,
  export-safe graphs, bounded public `checkpoint` / `vmap` / `jvp`, and
  explicit remaining higher-order transform refusal.
- **[docs/EXPORT_DEPLOYMENT_ARTIFACT_CONTRACTS.md](docs/EXPORT_DEPLOYMENT_ARTIFACT_CONTRACTS.md)** —
  canonical bounded exportable-graph and deployment-artifact contract surface
  for graph-first packaging independent of raw checkpoints.
- **[docs/EXTENSION_CONTRACT_SEMANTICS.md](docs/EXTENSION_CONTRACT_SEMANTICS.md)** —
  canonical bounded contract surface for custom ops, kernels, autograd,
  backend plugins, and quantizer plugins.
- **[docs/DATA_INGRESS_SEMANTICS.md](docs/DATA_INGRESS_SEMANTICS.md)** —
  canonical bounded local data-ingress surface for dataset source, sampler,
  batch-sampler, and host-device staging contracts.
- **[docs/DISTRIBUTED_DATA_FEED_SEMANTICS.md](docs/DISTRIBUTED_DATA_FEED_SEMANTICS.md)** —
  canonical bounded fixed-world-size distributed data-feed surface for shard
  partitioning, worker coordination, and replay-safe input ordering.
- **[docs/TENSOR_FAMILY_CAPABILITY_MATRIX.md](docs/TENSOR_FAMILY_CAPABILITY_MATRIX.md)** —
  canonical capability and refusal matrix for dense, sparse, nested, masked,
  and storage-aware tensor-family contracts.
- **[docs/ADVANCED_DTYPE_SEMANTICS.md](docs/ADVANCED_DTYPE_SEMANTICS.md)** —
  canonical bounded promotion, cast, and backend-capability matrix for complex,
  low-precision, and wider integer dtype semantics above the compact runtime
  subset.
- **[docs/AUTOCAST_PRECISION_POLICY.md](docs/AUTOCAST_PRECISION_POLICY.md)** —
  canonical bounded autocast-style precision-policy matrix for backend-aware
  low-precision rules, numerics diagnostics, and typed refusal posture.
- **[docs/GRADIENT_SCALING_SEMANTICS.md](docs/GRADIENT_SCALING_SEMANTICS.md)** —
  canonical bounded train-class mixed-precision gradient-scaling surface for
  fp16 overflow/underflow handling and bf16 no-scaling posture.
- **[docs/QUANTIZATION_CAPABILITY_SEMANTICS.md](docs/QUANTIZATION_CAPABILITY_SEMANTICS.md)** —
  canonical bounded PTQ, QAT, quantized execution, compiler-lowering, and
  export-aware quantization capability surface above raw decode.
- **[docs/REPRODUCIBILITY_SEMANTICS.md](docs/REPRODUCIBILITY_SEMANTICS.md)** —
  canonical framework-wide replay seed, generator-derivation, and
  checkpoint-restore truth surface across runtime and training replay.
- **[docs/MODULE_PARITY_MATRIX.md](docs/MODULE_PARITY_MATRIX.md)** —
  canonical seeded module parity artifact for the current PyTorch-derived
  `module_db`-style state-tree and `state_dict` coverage slice.
- **[docs/OPTIMIZER_PARITY_MATRIX.md](docs/OPTIMIZER_PARITY_MATRIX.md)** —
  canonical seeded optimizer parity artifact for the current PyTorch-derived
  `optim_db`-style single-step optimizer coverage slice.
- **[docs/COMPILER_HYGIENE_PARITY_MATRIX.md](docs/COMPILER_HYGIENE_PARITY_MATRIX.md)** —
  canonical seeded symbolic-shape, fake-tensor, and compiler-hygiene parity
  artifact for the current PyTorch-derived compiler coverage slice.
- **[docs/SEMANTICS_CLAIM_REPORT.md](docs/SEMANTICS_CLAIM_REPORT.md)** —
  canonical machine-readable truth source for what Psionic currently treats as
  seeded evidence only versus `PyTorch-compatible later`.
- **[docs/MLX_COMPATIBILITY_SCOPE.md](docs/MLX_COMPATIBILITY_SCOPE.md)** —
  canonical bounded upstream MLX version window and claim-language contract
  for the Psionic MLX roadmap.
- **[docs/MLX_CPU_REFERENCE_COVERAGE.md](docs/MLX_CPU_REFERENCE_COVERAGE.md)** —
  canonical bounded CPU-reference oracle for imported MLX `array_core`,
  `ops_numeric`, and `device_eval_memory` families.
- **[docs/MLX_ACCEPTANCE_MATRIX.md](docs/MLX_ACCEPTANCE_MATRIX.md)** —
  canonical MLX-lane acceptance categories and machine-readable report
  contract.
- **[docs/MLX_PARITY_HARNESS.md](docs/MLX_PARITY_HARNESS.md)** —
  canonical seeded upstream MLX test families and parity-harness report
  contract.
- **[docs/MLX_COMPATIBILITY_MATRIX.md](docs/MLX_COMPATIBILITY_MATRIX.md)** —
  canonical supported/convertible/unsupported adoption matrix for the Psionic
  MLX roadmap.
- **[docs/MLX_TO_PSIONIC_MIGRATION_GUIDE.md](docs/MLX_TO_PSIONIC_MIGRATION_GUIDE.md)** —
  bounded MLX adoption guide covering runnable examples, native drop-down
  points, and current supported versus convertible versus unsupported posture.
- **[docs/MLX_LM_PACKAGE.md](docs/MLX_LM_PACKAGE.md)** —
  canonical first-package spec for the bounded local `psionic-mlx-lm` text
  package, CLI, and prompt-cache artifact contract.
- **[docs/MLX_MODEL_CATALOG.md](docs/MLX_MODEL_CATALOG.md)** —
  canonical bounded model-catalog spec for `psionic-mlx-catalog`, including
  local Ollama/Hugging Face cache resolution and remote-metadata trust policy.
- **[docs/MLX_TEXT_SERVE.md](docs/MLX_TEXT_SERVE.md)** —
  canonical bounded text-serving spec for `psionic-mlx-serve`, including
  MLX-style model-reference resolution, bootstrap reports, and response-state
  posture over the shared Psionic OpenAI-compatible server.
- **[docs/MLX_VLM_PACKAGE.md](docs/MLX_VLM_PACKAGE.md)** —
  canonical bounded multimodal package spec for `psionic-mlx-vlm`, including
  builtin processor registries, prompt projection, attachment receipts, and
  served-request planning for image/audio/video inputs.
- **[docs/MLX_AUDIO_PACKAGE.md](docs/MLX_AUDIO_PACKAGE.md)** —
  canonical bounded audio package spec for `psionic-mlx-audio`, including
  family/quantization metadata, WAV IO, text-to-speech and speech-to-speech
  requests, stream chunks, and the server-facing speech contract.
- **[docs/MLX_RECIPE_PACKAGE.md](docs/MLX_RECIPE_PACKAGE.md)** —
  canonical bounded training-recipe package spec for `psionic-mlx-recipes`,
  including method inventory, stage mapping, adapter posture, and recipe-plan
  emission above `psionic-train`.
- **[docs/MLX_WORKFLOW_PACKAGE.md](docs/MLX_WORKFLOW_PACKAGE.md)** —
  canonical bounded workflow package spec for `psionic-mlx-workflows`,
  including synthetic dataset bundles, reward/judge helper plans, adapter
  merge/export, and the local publish snapshot boundary.
- **[docs/MLX_BENCH_PACKAGE.md](docs/MLX_BENCH_PACKAGE.md)** —
  canonical bounded benchmark-package spec for `psionic-mlx-bench`,
  including suite manifests, text/served provider adapters, multimodal
  projection, and repeated benchmark receipts above `psionic-eval`.
- **[docs/MLX_ECOSYSTEM_GUIDE.md](docs/MLX_ECOSYSTEM_GUIDE.md)** —
  package-facing CLI and fixture guide for the bounded MLX ecosystem in this
  repo, covering text, multimodal, audio, serving, recipes, and evaluation.
- **[docs/INFERENCE_ENGINE.md](docs/INFERENCE_ENGINE.md)** — canonical
  inference-engine completion criteria and current boundaries.
- **[docs/TRAIN_SYSTEM.md](docs/TRAIN_SYSTEM.md)** — canonical training
  subsystem spec covering current substrate, planned architecture, object
  model, receipts, policy surfaces, and the issue-program path to a full
  Rust-native train stack, first tracked as GitHub issues `#3564` through
  `#3593` and later extended through `#3631`.
- **[docs/APPLE_ADAPTER_DATASET_SPEC.md](docs/APPLE_ADAPTER_DATASET_SPEC.md)** —
  canonical Apple adapter dataset contract and fixture baseline.
- **[docs/APPLE_FMADAPTER_PACKAGE_SPEC.md](docs/APPLE_FMADAPTER_PACKAGE_SPEC.md)** —
  canonical `.fmadapter` package inventory, metadata, and export contract.
- **[docs/APPLE_ADAPTER_LINEAGE_SPEC.md](docs/APPLE_ADAPTER_LINEAGE_SPEC.md)** —
  canonical Apple adapter lineage and authority-facing metadata contract.
- **[docs/TRAINING_CORE_FIXED_BUDGET_REFERENCE.md](docs/TRAINING_CORE_FIXED_BUDGET_REFERENCE.md)** —
  canonical reference loop, runbook, and acceptance criteria for the first
  real `psionic-train` fixed-budget training-core path.
- **[docs/ROLLOUT_ARTIFACT_POLICY_LINEAGE_REFERENCE.md](docs/ROLLOUT_ARTIFACT_POLICY_LINEAGE_REFERENCE.md)** —
  canonical rollout-artifact, trainer-batch, and policy-lineage runbook for
  the first reusable RL-facing contracts in `psionic-train`.
- **[docs/TRAIN_STAGE_PROGRAM_REFERENCE.md](docs/TRAIN_STAGE_PROGRAM_REFERENCE.md)** —
  canonical multi-stage `general_sft -> agentic_sft -> rl` runbook for
  `psionic-train`.
- **[docs/TRAIN_CURRICULUM_REFERENCE.md](docs/TRAIN_CURRICULUM_REFERENCE.md)** —
  canonical difficulty-aware curriculum, filtering, and non-zero-advantage
  runbook for `psionic-train`.
- **[docs/TRAIN_STABILITY_REFERENCE.md](docs/TRAIN_STABILITY_REFERENCE.md)** —
  canonical instability-telemetry, risky-optimization, and halt-policy runbook
  for `psionic-train`.
- **[docs/ENVIRONMENT_ABI_REFERENCE.md](docs/ENVIRONMENT_ABI_REFERENCE.md)** —
  canonical package ABI, runtime-session runbook, and acceptance criteria for
  the first Psionic-native environment contract.
- **[docs/ENVIRONMENT_PACKAGE_CONTRACT_REFERENCE.md](docs/ENVIRONMENT_PACKAGE_CONTRACT_REFERENCE.md)** —
  canonical package-shape runbook for workload classes, policy refs,
  difficulty metadata, and validator benchmark profiles in
  `psionic-environments`.
- **[docs/ENVIRONMENT_REGISTRY_REFERENCE.md](docs/ENVIRONMENT_REGISTRY_REFERENCE.md)** —
  canonical install, pinning, mixed-group composition, and train/eval parity
  runbook for `psionic-environments`.
- **[docs/SANDBOX_RL_THROUGHPUT_REFERENCE.md](docs/SANDBOX_RL_THROUGHPUT_REFERENCE.md)** —
  canonical warm-pool, staged-input, repeated-loop, and pool-reuse runbook for
  `psionic-sandbox`.
- **[docs/DATASET_TOKENIZER_PACKING_REFERENCE.md](docs/DATASET_TOKENIZER_PACKING_REFERENCE.md)** —
  canonical versioned-dataset, tokenizer-digest, streamed-iteration, and
  long-context packing runbook for the first Psionic-native data-contract
  layer.
- **[docs/EVAL_RUNTIME_REFERENCE.md](docs/EVAL_RUNTIME_REFERENCE.md)** —
  canonical held-out eval, benchmark-package, and local validator-simulation
  runbook for the first Psionic-native eval runtime.
- **[docs/TRAIN_RUN_GRAPH_REFERENCE.md](docs/TRAIN_RUN_GRAPH_REFERENCE.md)** —
  canonical run-graph, contributor-set, and window-lifecycle runbook for the
  first Psionic-native training run-state machine.
- **[docs/TRAIN_CHECKPOINT_RECOVERY_REFERENCE.md](docs/TRAIN_CHECKPOINT_RECOVERY_REFERENCE.md)** —
  canonical checkpoint-pointer, checkpoint-manifest, and restore-ladder
  runbook for the first explicit Psionic checkpoint-recovery receipt path.
- **[docs/COLLECTIVE_SYNC_POLICY_REFERENCE.md](docs/COLLECTIVE_SYNC_POLICY_REFERENCE.md)** —
  canonical local/global sync cadence, transport-feedback, and replanning
  runbook for the first explicit Psionic collective sync planner.
- **[docs/POLICY_WEIGHT_BROADCAST_REFERENCE.md](docs/POLICY_WEIGHT_BROADCAST_REFERENCE.md)** —
  canonical policy-weight shard, freshness, and heavy-artifact broadcast
  runbook for the first explicit Psionic datastream control-plane split.
- **[docs/TRAIN_ORCHESTRATOR_REFERENCE.md](docs/TRAIN_ORCHESTRATOR_REFERENCE.md)** —
  canonical window-control, assignment-posture, and trainer-batch assembly
  runbook for the first explicit Psionic train orchestrator.
- **[docs/AGENTIC_SFT_RL_REFERENCE_PROGRAM.md](docs/AGENTIC_SFT_RL_REFERENCE_PROGRAM.md)** —
  canonical end-to-end agentic-SFT-plus-RL pilot, including environment and
  dataset lineage, sandbox reuse, rollout-worker receipts, validator verdicts,
  online eval, benchmark aggregation, and operator-view pass criteria.
- **[docs/DISTRIBUTED_OPTIMIZER_REFERENCE.md](docs/DISTRIBUTED_OPTIMIZER_REFERENCE.md)** —
  canonical parameter-sharding, optimizer-state-sharding, precision,
  microbatch-accumulation, activation-checkpointing, and memory-budget runbook
  for the distributed optimizer layer in `psionic-train`.
- **[docs/MODEL_IO_REFERENCE.md](docs/MODEL_IO_REFERENCE.md)** —
  canonical state-dict traversal, tokenizer binding, safetensors export/import,
  GGUF import, and adapter merge/unmerge runbook for the portable model-IO
  layer in `psionic-train`.
- **[docs/TRAIN_REPLAY_TRUTH_REFERENCE.md](docs/TRAIN_REPLAY_TRUTH_REFERENCE.md)** —
  canonical replay-seed, sample-selection, environment-pin, eval-posture, and
  replay-verification runbook for `psionic-train`.
- **[docs/TRAIN_SECURITY_POSTURE_REFERENCE.md](docs/TRAIN_SECURITY_POSTURE_REFERENCE.md)** —
  canonical environment verification, artifact trust-root, untrusted-worker
  admission, and poisoning-control runbook for `psionic-train`.
- **[docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md](docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md)** —
  canonical retention-profile, deduplication, archival, garbage-collection,
  and cold-restore runbook for the train artifact-storage layer in
  `psionic-train`.
- **[docs/TRAIN_SCHEDULING_ACCOUNTING_REFERENCE.md](docs/TRAIN_SCHEDULING_ACCOUNTING_REFERENCE.md)** —
  canonical queue-class, budget-cap, preemption, and cost-attribution runbook
  for the train scheduling and accounting layer in `psionic-train`.
- **[docs/TRAIN_RELIABILITY_REFERENCE.md](docs/TRAIN_RELIABILITY_REFERENCE.md)** —
  canonical chaos-scenario, failure-injection, and recovery-suite runbook for
  the train reliability layer in `psionic-train`.
- **[docs/TRAIN_BENCHMARK_ACCEPTANCE_REFERENCE.md](docs/TRAIN_BENCHMARK_ACCEPTANCE_REFERENCE.md)** —
  canonical threshold profile, benchmark categories, and runnable acceptance
  harness for the quantitative train completion layer in `psionic-train`.
- **[docs/TRAIN_OFF_POLICY_BUDGET_REFERENCE.md](docs/TRAIN_OFF_POLICY_BUDGET_REFERENCE.md)** —
  canonical bounded stale-rollout admission, quarantine, and discard runbook
  for the first explicit Psionic off-policy control layer.
- **[docs/TRAIN_ROLLOUT_WORKER_PROTOCOL_REFERENCE.md](docs/TRAIN_ROLLOUT_WORKER_PROTOCOL_REFERENCE.md)** —
  canonical rollout-worker heartbeat, claim, upload, and worker-outcome
  runbook for the first trust-aware worker protocol in `psionic-train`.
- **[docs/TRAIN_ROLLOUT_VALIDATION_REFERENCE.md](docs/TRAIN_ROLLOUT_VALIDATION_REFERENCE.md)** —
  canonical rollout-verification bundle, sampled-adjudication, duplicate-
  detection, and validator-verdict runbook for the first validator-ready train
  integrity layer.
- **[docs/NETWORK_EXECUTION_IDENTITY_REFERENCE.md](docs/NETWORK_EXECUTION_IDENTITY_REFERENCE.md)** —
  canonical runtime-manifest, session-claims, required-vs-best-effort posture,
  and operator-surface runbook for proof-bearing networked execution identity.
- **[docs/RESEARCH_EXPERIMENT_REFERENCE.md](docs/RESEARCH_EXPERIMENT_REFERENCE.md)** —
  canonical experiment-spec, bounded result-manifest, score-contract, and
  promotion-record reference for Psionic hillclimb loops.
- **[docs/RESEARCH_RUNNER_REFERENCE.md](docs/RESEARCH_RUNNER_REFERENCE.md)** —
  canonical invocation, result-manifest, and failure-semantics reference for
  the compiled `psionic-research-runner` boundary.
- **[docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md](docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md)** —
  canonical source split, owner matrix, completion matrix, and issue-program
  authority for the current `PSI-232` through `PSI-258` inference backlog.
- **[docs/TOPOLOGY_ACCEPTANCE_MATRIX.md](docs/TOPOLOGY_ACCEPTANCE_MATRIX.md)** —
  canonical support matrix and runnable validation entrypoint for local and
  clustered serving topologies, including `DP`, `PP`, `TP`, `PD`, explicit
  refusal boundaries, and current expert-parallel non-support.
- **[docs/PRODUCT_CLASS_ACCEPTANCE_MATRICES.md](docs/PRODUCT_CLASS_ACCEPTANCE_MATRICES.md)** —
  canonical split between local portability, high-throughput serving, and
  structured-agent acceptance, plus the runnable category harness that keeps
  those product claims from collapsing into one benchmark headline.
- **[docs/NON_GPT_OSS_QWEN_PILOT.md](docs/NON_GPT_OSS_QWEN_PILOT.md)** —
  canonical first non-GPT-OSS generic-server pilot, including the Qwen runbook,
  pass criteria, expected signals, and current limitations.
- **[docs/STRUCTURED_AGENT_WEATHER_PILOT.md](docs/STRUCTURED_AGENT_WEATHER_PILOT.md)** —
  canonical integrated structured-agent workload pilot, including the weather
  runbook, pass criteria, expected signals, and bounded current scope.
- **[docs/FM_BRIDGE_CONSIDERATIONS.md](docs/FM_BRIDGE_CONSIDERATIONS.md)** — Apple Foundation Models bridge: architecture, binary discovery, build, run, test, shipping, and user requirements in full detail.
- **[docs/ACTIVATION_FINGERPRINT_PROOFS.md](docs/ACTIVATION_FINGERPRINT_PROOFS.md)** — activation-fingerprint proof posture, embeddings-first artifact generation, and benchmark semantics.
- **[docs/PARAMETER_GOLF_ACCOUNTING.md](docs/PARAMETER_GOLF_ACCOUNTING.md)** — canonical Parameter Golf claim-language and artifact-accounting contract for research, non-record, and record-track posture.
- **[docs/PARAMETER_GOLF_ACCEPTANCE_MATRIX.md](docs/PARAMETER_GOLF_ACCEPTANCE_MATRIX.md)** — canonical Parameter Golf acceptance matrix for oracle parity, trainer parity, throughput closure, packaging readiness, and record-track readiness.
- **[docs/PARAMETER_GOLF_SINGLE_H100_BRINGUP.md](docs/PARAMETER_GOLF_SINGLE_H100_BRINGUP.md)** — canonical Rust-native Parameter Golf single-H100 bring-up command, report seam, and current refusal boundary.
- **[docs/PARAMETER_GOLF_SINGLE_H100_TRAINER.md](docs/PARAMETER_GOLF_SINGLE_H100_TRAINER.md)** — canonical Rust-native Parameter Golf single-H100 trainer command, report seam, and bounded claim boundary.
- **[docs/PARAMETER_GOLF_RECORD_FOLDER_COMPATIBILITY.md](docs/PARAMETER_GOLF_RECORD_FOLDER_COMPATIBILITY.md)** — canonical Parameter Golf challenge-repo folder-compatibility contract and verifier gate for `records/...` exports.
- **[docs/PARAMETER_GOLF_EXPORTED_SUBMISSION_EVIDENCE.md](docs/PARAMETER_GOLF_EXPORTED_SUBMISSION_EVIDENCE.md)** — canonical Parameter Golf exported-folder evidence and replay-verification contract for metrics, wallclock, and counted bytes.
- **[docs/PARAMETER_GOLF_PR_SUBMISSION_FLOW.md](docs/PARAMETER_GOLF_PR_SUBMISSION_FLOW.md)** — canonical Parameter Golf promotion receipt, final PR-bundle, and local challenge-clone dry-run contract.
- **[docs/PARAMETER_GOLF_ARCHITECTURE_EXPERIMENT_QUEUE.md](docs/PARAMETER_GOLF_ARCHITECTURE_EXPERIMENT_QUEUE.md)** — canonical concrete post-parity architecture queue for shared-depth, stronger parameter-tying, and restricted-attention evidence.
- **[docs/PARAMETER_GOLF_RESTRICTED_ATTENTION_REPORT.md](docs/PARAMETER_GOLF_RESTRICTED_ATTENTION_REPORT.md)** — canonical restricted-attention evidence record on one committed `seq_len=1024` challenge-format validation slice, including the current negative-evidence result for a fixed `256`-token window.
- **[docs/PARAMETER_GOLF_EXTERNAL_NON_RECORD_PR.md](docs/PARAMETER_GOLF_EXTERNAL_NON_RECORD_PR.md)** — canonical historical record of the first external non-record PR, including the exact folder identity, verifier digests, closed-PR state, and the current local blocked posture for counted-runtime ambiguity.
- **[docs/PARAMETER_GOLF_AFTER_ACTION.md](docs/PARAMETER_GOLF_AFTER_ACTION.md)** — canonical stop record for the paused Parameter Golf lane, including the final honest posture, last H100 findings, and the closed-not-planned resolution of the remaining open PGOLF issues.
- **[docs/ROADMAP_FM.md](docs/ROADMAP_FM.md)** — Apple FM lane roadmap and API coverage.
- **[docs/ROADMAP_PARAMETERGOLF.md](docs/ROADMAP_PARAMETERGOLF.md)** — Parameter Golf lane roadmap for challenge-oracle parity, compact decoder training, artifact accounting, and submission packaging inside Psionic.
- `scripts/check-parameter-golf-acceptance.sh` and
  `fixtures/parameter_golf/reports/parameter_golf_acceptance_report.json` are
  the canonical checker and machine-readable acceptance artifact for current
  Parameter Golf claim truth.
- `scripts/check-parameter-golf-single-h100-bringup.sh` and
  `fixtures/parameter_golf/reports/parameter_golf_single_h100_bringup.json`
  are the canonical verifier and machine-readable artifact for current
  Rust-native Parameter Golf single-H100 bring-up posture.
- `scripts/check-parameter-golf-record-folder-compatibility.sh` and
  `fixtures/parameter_golf/reports/parameter_golf_record_folder_compatibility.json`
  are the canonical verifier and machine-readable artifact for current
  Parameter Golf challenge-repo folder compatibility.
- `fixtures/parameter_golf/reports/parameter_golf_after_action_report.json`
  is the canonical machine-readable stop record for the paused Parameter Golf
  lane.
- **[docs/ROADMAP_TASSADAR.md](docs/ROADMAP_TASSADAR.md)** — Tassadar lane roadmap from the current bounded executor substrate to article-grade WebAssembly in-model compute.
- **[docs/ROADMAP_TASSADAR_INDEX.md](docs/ROADMAP_TASSADAR_INDEX.md)** — compact Tassadar phase-to-artifact index for canonical bundle roots, audits, validators, and current claim boundaries.
- `scripts/check-tassadar-acceptance.sh` and
  `fixtures/tassadar/reports/tassadar_acceptance_report.json` are the canonical
  live checker and machine-readable acceptance artifact for current Tassadar
  claim truth.
- `scripts/check-tassadar-compiled-article-closure.sh` and
  `fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json`
  are the canonical compiled-lane closure checker and machine-readable report
  for the article-sized compiled/proof-backed Tassadar bar.
- `fixtures/tassadar/reports/tassadar_wasm_instruction_coverage_report.json`
  is the canonical machine-readable Tassadar Wasm profile/instruction coverage
  artifact.
- `fixtures/tassadar/reports/tassadar_rust_source_canon_report.json`,
  `fixtures/tassadar/sources/tassadar_micro_wasm_kernel.rs`,
  `fixtures/tassadar/sources/tassadar_heap_sum_kernel.rs`,
  `fixtures/tassadar/sources/tassadar_long_loop_kernel.rs`,
  `fixtures/tassadar/sources/tassadar_hungarian_10x10_article.rs`,
  `fixtures/tassadar/sources/tassadar_sudoku_9x9_article.rs`, and the
  corresponding committed `fixtures/tassadar/wasm/*.wasm` outputs are the
  canonical Rust-only frontend lineage artifacts for the Tassadar
  article-closure path.
- `fixtures/tassadar/reports/tassadar_rust_article_profile_completeness_report.json`
  is the canonical machine-readable profile boundary for the current Rust-only
  article family, and the same publication is bound into the Tassadar
  environment bundle and served capability publication.
- `fixtures/tassadar/reports/tassadar_c_to_wasm_compile_receipt.json`,
  `fixtures/tassadar/sources/tassadar_micro_wasm_kernel.c`, and
  `fixtures/tassadar/wasm/tassadar_micro_wasm_kernel.wasm` are the canonical
  Tassadar source/toolchain/output lineage artifacts for the historical
  repo-owned C-to-Wasm compile path, not the article-closure frontend anchor
  and not part of the current Rust operator flow.
- `fixtures/tassadar/reports/tassadar_compile_pipeline_matrix_report.json`,
  `fixtures/tassadar/sources/tassadar_multi_export_kernel.wat`,
  `fixtures/tassadar/sources/tassadar_memory_lookup_kernel.wat`,
  `fixtures/tassadar/sources/tassadar_param_abi_kernel.wat`,
  `fixtures/tassadar/sources/tassadar_micro_wasm_kernel.c`, and the
  corresponding `fixtures/tassadar/wasm/*.wasm` outputs are the canonical
  repo-owned real compile-pipeline matrix artifacts for the current bounded
  Wasm-text exact lane plus typed C-toolchain-refusal ingress check.
- `fixtures/tassadar/reports/tassadar_wasm_module_ingress_report.json` is the
  canonical machine-readable artifact for bounded normalized Wasm-module
  ingress, section-level round-trip truth, and current exact-vs-refused export
  lowering posture.
- `fixtures/tassadar/reports/tassadar_wasm_conformance_report.json` is the
  canonical machine-readable artifact for bounded module-execution
  differential checks against the current `wasmi` reference authority.
- Other planning and reference docs live under `docs/`.

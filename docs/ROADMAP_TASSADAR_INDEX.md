# Tassadar Roadmap-To-Artifact Index

This document is a compact artifact index for the currently landed,
artifact-bearing Tassadar phases.

It is subordinate to:

- `README.md`
- `docs/ROADMAP_TASSADAR.md`
- `docs/ROADMAP.md`
- `docs/ARCHITECTURE.md`

It does not replace the roadmap text. It exists so contributors can answer a
much narrower question quickly:

> for one landed Tassadar phase, which committed root, audit, validator, and
> claim boundary should I cite?

## Current Operator Entrypoints

- `docs/TASSADAR_WASM_RUNBOOK.md` is the canonical operator guide for the
  current bounded Wasm flow.
- `docs/audits/2026-03-18-tassadar-wasm-flow-status-audit.md` is the latest
  live status audit for that flow, including the exact commands that were run
  and the local toolchain caveat for the optional C-to-Wasm step.

## Current Artifact Index

| Phase | Canonical artifact root | Supporting audit | Primary validator | Current claim boundary |
| --- | --- | --- | --- | --- |
| `PTAS-003` acceptance gate | `fixtures/tassadar/reports/tassadar_acceptance_report.json` | none; machine-readable report is the current canonical review artifact | `scripts/check-tassadar-acceptance.sh` | one repo-owned claim gate: `research_only`, `compiled_exact`, and `learned_bounded` are green today; article-class and final article closure remain red |
| `PTAS-101` Wasm instruction coverage | `fixtures/tassadar/reports/tassadar_wasm_instruction_coverage_report.json` | none; machine-readable report is the current canonical review artifact | `cargo run -p psionic-runtime --example tassadar_wasm_instruction_coverage_report` | current article-shaped i32 coverage is explicit across the supported `tassadar.wasm.*` profiles, but unsupported opcodes still refuse by profile and this is not arbitrary Wasm closure |
| `TAS-084` Rust-only article source canon | `fixtures/tassadar/reports/tassadar_rust_source_canon_report.json` plus the committed Rust fixtures under `fixtures/tassadar/sources/*.rs` and `fixtures/tassadar/wasm/*.wasm` | none; machine-readable report plus committed source/output pairs are the canonical review artifacts | `cargo run -p psionic-eval --example tassadar_rust_source_canon_report` | the article-closure frontend path is now rooted in committed Rust fixtures only, with stable source/toolchain/config/output lineage; this is not arbitrary Rust frontend closure and not arbitrary Wasm lowering |
| `PTAS-102` C-to-Wasm compile receipt | `fixtures/tassadar/reports/tassadar_c_to_wasm_compile_receipt.json` plus `fixtures/tassadar/sources/tassadar_micro_wasm_kernel.c` and `fixtures/tassadar/wasm/tassadar_micro_wasm_kernel.wasm` | none; machine-readable receipt and committed source/output pair are the canonical review artifacts | `cargo run -p psionic-runtime --example tassadar_c_to_wasm_compile_receipt` | one canonical micro-kernel compile receipt plus source/toolchain/output/artifact lineage only; not general C/C++ frontend closure and not arbitrary Wasm lowering |
| `PTAS-103` article-class kernel-family benchmark widening | `fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json` | none; machine-readable benchmark report is the current canonical review artifact | `cargo run -p psionic-eval --example tassadar_article_class_benchmark_report` | exact article-class benchmark coverage now includes `branch_heavy_kernel`, `memory_heavy_kernel`, and `long_loop_kernel` alongside the prior mixed corpus; branch-heavy sparse fallback and long-loop hull/sparse fallback remain explicit, and the separate million-step decode closure is still open |
| `PTAS-104` long-horizon trace ABI decision | `fixtures/tassadar/reports/tassadar_trace_abi_decision_report.json` plus `fixtures/tassadar/runs/long_loop_kernel_trace_abi_v0/execution_evidence_bundle.json` | none; the machine-readable decision report and long-horizon evidence bundle are the canonical review artifacts | `cargo run -p psionic-runtime --example tassadar_trace_abi_decision_report` | `tassadar.trace.v1` is now explicitly frozen as the current article-class machine-truth trace ABI; readable logs are non-authoritative, validator-facing ABI pointers are recorded across benchmark/compiled/long-horizon artifacts, and the long-loop exemplar proves the ABI carries a 16k-step article-class trace without claiming million-step closure |
| `PTAS-105` workload capability matrix | `fixtures/tassadar/reports/tassadar_workload_capability_matrix.json` | none; the machine-readable matrix itself is the canonical review artifact | `cargo run -p psionic-eval --example tassadar_workload_capability_matrix` | current workload families now map cleanly to runtime exact vs fallback-only posture and to separate compiled exact, bounded learned, and partial learned-long-horizon evidence; the matrix remains artifact-backed rather than doc-only |
| `PTAS-201` widened HullCache closure | `fixtures/tassadar/reports/tassadar_hull_cache_closure_report.json` | none; the machine-readable HullCache closure report is the canonical review artifact | `cargo run -p psionic-eval --example tassadar_hull_cache_closure_report` | direct exact HullCache closure is now artifact-backed on the widened micro, branch-heavy, memory-heavy, and bounded Hungarian subset, while long-loop and Sudoku search workloads stay explicit as fallback-only under the current control-flow contract |
| `PTAS-202` SparseTopK comparison family | `fixtures/tassadar/reports/tassadar_sparse_top_k_comparison_report.json` | none; the machine-readable SparseTopK comparison report is the canonical review artifact | `cargo run -p psionic-eval --example tassadar_sparse_top_k_comparison_report` | SparseTopK now has an article-path comparison artifact over the same workload set as reference-linear and HullCache; direct exact support is explicit on the bounded subset, and branch-heavy, long-loop, and Sudoku search workloads remain explicit fallback-only families |
| `PTAS-203` decode scaling report | `fixtures/tassadar/reports/tassadar_decode_scaling_report.json` | none; the machine-readable scaling report is the canonical review artifact | `cargo run -p psionic-eval --example tassadar_decode_scaling_report` | the current decode modes now have one shared trace-length scaling artifact that records exactness, throughput, CPU-gap, trace-artifact growth, and direct-vs-fallback posture across bounded straight-line, branch-heavy, and long-loop synthetic families |
| `PTAS-204` million-step decode benchmark bundle | `fixtures/tassadar/runs/million_step_loop_benchmark_v0/benchmark_bundle.json` | none; the machine-readable benchmark bundle is the canonical review artifact | `cargo run -p psionic-runtime --example tassadar_million_step_decode_benchmark_bundle` | the repo now proves one reproducible 1,048,575-step reference-linear execution with exactness, proof lineage, runtime identity, and serialized trace-byte growth receipts at the million-step horizon; HullCache and SparseTopK remain explicit fallback-only requests on this workload family |
| `PTAS-205` geometric-variant comparison | `fixtures/tassadar/reports/tassadar_geometric_variant_report.json` | none; the machine-readable comparison report is the canonical review artifact | `cargo run -p psionic-eval --example tassadar_geometric_variant_report` | the promoted runtime HullCache surface now has an artifact-backed comparison against a research-only hierarchical-hull candidate; long-loop and 4x4 Sudoku stay fallback-only on the runtime surface while the candidate remains direct and exact there without widening the promoted claim boundary |
| trained-executor Phase 7 reference run | `fixtures/tassadar/runs/sudoku_v0_reference_run_v0` | `docs/audits/2026-03-16-tassadar-first-run-postmortem.md` | `cargo run -p psionic-train --example tassadar_reference_training_run` | first bounded learned reference run only; weak exactness is preserved as evidence rather than hidden |
| trained-executor Phase 8 telemetry | `fixtures/tassadar/runs/sudoku_v0_reference_run_v0` | `docs/audits/2026-03-16-tassadar-first-run-postmortem.md` | `cargo run -p psionic-train --example tassadar_reference_training_telemetry` | telemetry and failure analysis for the same bounded run; not a stronger claim than Phase 7 |
| trained-executor Phase 9 postmortem and next plan | `fixtures/tassadar/runs/sudoku_v0_reference_run_v0` | `docs/audits/2026-03-16-tassadar-first-run-postmortem.md` | `cargo run -p psionic-train --example tassadar_reference_training_postmortem` | artifact-backed diagnosis and next-step planning only; no claim widening |
| trained-executor Phase 10 hull benchmark | `fixtures/tassadar/runs/sudoku_v0_reference_run_v0` | `fixtures/tassadar/reports/tassadar_acceptance_report.json` | `cargo run -p psionic-train --example tassadar_reference_training_hull_benchmark` | bounded fast-path equivalence on declared Sudoku-v0 benchmark windows; not full-task exactness and not article parity |
| trained-executor Phase 11 9x9 scale plan | `fixtures/tassadar/runs/sudoku_9x9_scale_plan_v0` | `docs/audits/2026-03-16-tassadar-first-run-postmortem.md` | `cargo run -p psionic-train --example tassadar_sudoku_9x9_scale_plan` | real 9x9 workload planning and scale facts only; does not claim learned 9x9 success |
| trained-executor Phase 12 boundary run | `fixtures/tassadar/runs/sudoku_v0_boundary_v1` | `docs/audits/2026-03-16-tassadar-phase-12-boundary-audit.md` | `cargo run -p psionic-train --example tassadar_boundary_training_run` | first-token boundary is fixed on 4x4 validation, but promotion remains red and exact traces remain `0/2` |
| trained-executor Phase 13 trainable-surface ablation | `fixtures/tassadar/runs/sudoku_v0_trainable_surface_ablation_v1` | `docs/audits/2026-03-16-tassadar-phase-13-trainable-surface-audit.md` | `cargo run -p psionic-research --example tassadar_trainable_surface_ablation` | same-corpus surface comparison only; identifies the next surface without implying gate closure |
| trained-executor Phase 14 promotion gate | `fixtures/tassadar/runs/sudoku_v0_promotion_v3` | `docs/audits/2026-03-16-tassadar-phase-14-blocker-audit.md`, `docs/audits/2026-03-16-tassadar-promotion-v2-teacher-forced-audit.md`, `docs/audits/2026-03-16-tassadar-phase-14-promotion-green-audit.md` | `cargo run -p psionic-research --example tassadar_executor_attention_promotion_run`; `scripts/check-tassadar-4x4-promotion-gate.sh fixtures/tassadar/runs/sudoku_v0_promotion_v3` | green learned 4x4 promotion bundle only; bounded learned lane, not article-class learned execution |
| trained-executor Phase 15 architecture comparison | `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v1` | none | `cargo run -p psionic-research --example tassadar_executor_architecture_comparison` | bounded research comparison between lookup and attention families; does not close the promotion gate |
| trained-executor Phase 15A trained attention follow-on | `fixtures/tassadar/runs/sudoku_v0_attention_training_v1` and `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v2` | none | `cargo run -p psionic-research --example tassadar_executor_attention_training` | attention-family training improvement under the same bounded 4x4 contract; still below promotion |
| trained-executor Phase 15B attention boundary saturation | `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v9` and `fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v11` | none | `cargo run -p psionic-research --example tassadar_executor_architecture_comparison_boundary_attention` | research-only attention family with real first-token progress but still no exact validation traces; preserved as `research_only` |
| trained-executor Phase 16 honest 9x9 reference run | `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0` | `docs/audits/2026-03-16-tassadar-phase-16-9x9-reference-run-audit.md` | `cargo run -p psionic-train --example tassadar_sudoku_9x9_reference_run` | bounded learned 9x9 partial only; the committed fit report says full 9x9 traces do not fit the current learned context |
| trained-executor Phase 17 compiled Sudoku executor lane | `fixtures/tassadar/runs/sudoku_v0_compiled_executor_v0` | `fixtures/tassadar/reports/tassadar_acceptance_report.json` | `cargo run -p psionic-research --example tassadar_compiled_executor_bundle` | bounded compiled/proof-backed Sudoku-v0 exactness on the matched corpus with `eval_only` posture; not arbitrary-program or article-class closure |
| trained-executor Phase 18 compiled Hungarian executor lane | `fixtures/tassadar/runs/hungarian_v0_compiled_executor_v0` | `fixtures/tassadar/reports/tassadar_acceptance_report.json` | `cargo run -p psionic-research --example tassadar_hungarian_compiled_executor_bundle` | bounded compiled/proof-backed Hungarian-v0 exactness on the matched corpus with explicit learned-lane non-closure and `eval_only` posture |

## Preserved Negative-Evidence Roots

These roots should stay citeable as negative evidence rather than being erased:

| Root | Why it still matters | Primary checker |
| --- | --- | --- |
| `fixtures/tassadar/runs/sudoku_v0_promotion_v1` | first preserved red learned promotion attempt | `cargo run -p psionic-train --example tassadar_promotion_gate_check -- fixtures/tassadar/runs/sudoku_v0_promotion_v1` |
| `fixtures/tassadar/runs/sudoku_v0_promotion_v2` | second preserved red learned promotion attempt | `cargo run -p psionic-train --example tassadar_promotion_gate_check -- fixtures/tassadar/runs/sudoku_v0_promotion_v2` |

## Use Rules

- cite `docs/ROADMAP_TASSADAR.md` for dependency order, goals, and open work
- cite this index when the question is “which exact root, audit, and command
  should I use for the landed phase?”
- cite `fixtures/tassadar/reports/tassadar_acceptance_report.json` and
  `scripts/check-tassadar-acceptance.sh` whenever the question is about what
  current claim language is still allowed

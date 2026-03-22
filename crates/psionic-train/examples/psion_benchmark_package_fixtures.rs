use std::{error::Error, fs, path::PathBuf};

use psionic_data::{PsionExclusionManifest, PsionSourceLifecycleManifest};
use psionic_environments::EnvironmentPackageKey;
use psionic_eval::{
    BenchmarkAggregationKind, BenchmarkCase, BenchmarkPackage, BenchmarkPackageKey,
};
use psionic_train::{
    record_psion_benchmark_catalog, record_psion_benchmark_contamination_inputs,
    record_psion_benchmark_package, record_psion_benchmark_package_receipt,
    record_psion_benchmark_receipt_set, PsionBenchmarkExactLabelGrader,
    PsionBenchmarkExactRefusalGrader, PsionBenchmarkExactRouteGrader,
    PsionBenchmarkExpectedResponseFormat, PsionBenchmarkGraderInterface, PsionBenchmarkItem,
    PsionBenchmarkPackageFamily, PsionBenchmarkPromptEnvelope, PsionBenchmarkPromptFormat,
    PsionBenchmarkRubricDimension, PsionBenchmarkRubricGrader, PsionBenchmarkTaskContract,
    PsionMetricKind, PsionObservedMetric, PsionPhaseGate, PsionRouteKind,
};

fn main() -> Result<(), Box<dyn Error>> {
    let root = workspace_root()?;
    let fixtures_dir = root.join("fixtures/psion/benchmarks");
    fs::create_dir_all(&fixtures_dir)?;

    let lifecycle: PsionSourceLifecycleManifest = serde_json::from_str(&fs::read_to_string(
        root.join("fixtures/psion/lifecycle/psion_source_lifecycle_manifest_v1.json"),
    )?)?;
    let exclusion: PsionExclusionManifest = serde_json::from_str(&fs::read_to_string(
        root.join("fixtures/psion/isolation/psion_exclusion_manifest_v1.json"),
    )?)?;

    let packages = package_contracts()?;
    let catalog = record_psion_benchmark_catalog(
        "psion-benchmark-catalog-v1",
        &lifecycle,
        &exclusion,
        packages.clone(),
        "Canonical catalog proving the main Psion benchmark families can all build on one shared prompt, item, grader, contamination-input, and receipt contract.",
    )?;
    let receipts = vec![
        record_psion_benchmark_package_receipt(
            "psion-architecture-benchmark-receipt-v1",
            PsionPhaseGate::Pilot,
            &packages[0],
            vec![PsionObservedMetric {
                metric_kind: PsionMetricKind::PassRateBps,
                observed_bps: 8420,
                regression_from_baseline_bps: 0,
            }],
            "Architecture reasoning benchmark receipt on the shared contract.",
        )?,
        record_psion_benchmark_package_receipt(
            "psion-normative-benchmark-receipt-v1",
            PsionPhaseGate::Pilot,
            &packages[1],
            vec![PsionObservedMetric {
                metric_kind: PsionMetricKind::PassRateBps,
                observed_bps: 8910,
                regression_from_baseline_bps: 0,
            }],
            "Normative spec reading benchmark receipt on the shared contract.",
        )?,
        record_psion_benchmark_package_receipt(
            "psion-engineering-benchmark-receipt-v1",
            PsionPhaseGate::Pilot,
            &packages[2],
            vec![PsionObservedMetric {
                metric_kind: PsionMetricKind::PassRateBps,
                observed_bps: 8760,
                regression_from_baseline_bps: 0,
            }],
            "Engineering spec interpretation benchmark receipt on the shared contract.",
        )?,
        record_psion_benchmark_package_receipt(
            "psion-memorization-benchmark-receipt-v1",
            PsionPhaseGate::Pilot,
            &packages[3],
            vec![
                PsionObservedMetric {
                    metric_kind: PsionMetricKind::PassRateBps,
                    observed_bps: 8040,
                    regression_from_baseline_bps: 0,
                },
                PsionObservedMetric {
                    metric_kind: PsionMetricKind::ImprovementOverSeedBaselineBps,
                    observed_bps: 1260,
                    regression_from_baseline_bps: 0,
                },
            ],
            "Memorization-versus-reasoning benchmark receipt on the shared contract.",
        )?,
        record_psion_benchmark_package_receipt(
            "psion-route-benchmark-receipt-v1",
            PsionPhaseGate::Pilot,
            &packages[4],
            vec![PsionObservedMetric {
                metric_kind: PsionMetricKind::RouteSelectionAccuracyBps,
                observed_bps: 9730,
                regression_from_baseline_bps: 0,
            }],
            "Route benchmark receipt on the shared contract.",
        )?,
        record_psion_benchmark_package_receipt(
            "psion-refusal-benchmark-receipt-v1",
            PsionPhaseGate::Pilot,
            &packages[5],
            vec![
                PsionObservedMetric {
                    metric_kind: PsionMetricKind::UnsupportedRequestRefusalBps,
                    observed_bps: 9910,
                    regression_from_baseline_bps: 0,
                },
                PsionObservedMetric {
                    metric_kind: PsionMetricKind::OverrefusalBps,
                    observed_bps: 340,
                    regression_from_baseline_bps: 0,
                },
            ],
            "Refusal benchmark receipt on the shared contract.",
        )?,
    ];
    let receipt_set = record_psion_benchmark_receipt_set(
        "psion-benchmark-receipt-set-v1",
        &catalog,
        &lifecycle,
        &exclusion,
        receipts,
        "Canonical receipt set proving the shared Psion benchmark contract emits acceptance-ready package receipts across the main benchmark families.",
    )?;

    fs::write(
        fixtures_dir.join("psion_benchmark_catalog_v1.json"),
        serde_json::to_vec_pretty(&catalog)?,
    )?;
    fs::write(
        fixtures_dir.join("psion_benchmark_receipt_set_v1.json"),
        serde_json::to_vec_pretty(&receipt_set)?,
    )?;
    Ok(())
}

fn package_contracts() -> Result<Vec<psionic_train::PsionBenchmarkPackageContract>, Box<dyn Error>>
{
    Ok(vec![
        record_psion_benchmark_package(
            "psion_architecture_reasoning_benchmark_v1",
            PsionBenchmarkPackageFamily::ArchitectureReasoning,
            benchmark_package("psion_architecture_reasoning_benchmark_v1", &["arch-case-1"]),
            vec![explanation_prompt_format()],
            vec![rubric_grader()],
            contamination_inputs(&["spec_quiz_eval_pack_v1"])?,
            vec![PsionBenchmarkItem {
                item_id: String::from("arch-case-1"),
                family: PsionBenchmarkPackageFamily::ArchitectureReasoning,
                prompt_format_id: String::from("bounded_explanation_v1"),
                grader_id: String::from("rubric_reasoning_v1"),
                prompt_digest: String::from("arch-prompt-digest-1"),
                source_ids: vec![String::from("spec_quiz_eval_pack_v1")],
                task: PsionBenchmarkTaskContract::ArchitectureReasoning {
                    target_architecture: String::from("bounded_three_stage_pipeline"),
                    expected_focus: String::from("memory hierarchy tradeoff"),
                },
                detail: String::from("Architecture benchmark item checks bounded system reasoning."),
            }],
            "Architecture reasoning benchmark package uses the shared prompt, item, and rubric-grader contracts.",
        )?,
        record_psion_benchmark_package(
            "psion_normative_spec_benchmark_v1",
            PsionBenchmarkPackageFamily::NormativeSpecReading,
            benchmark_package("psion_normative_spec_benchmark_v1", &["spec-case-1"]),
            vec![explanation_prompt_format()],
            vec![exact_label_grader()],
            contamination_inputs(&["spec_quiz_eval_pack_v1", "wasm_core_spec_release_2"])?,
            vec![PsionBenchmarkItem {
                item_id: String::from("spec-case-1"),
                family: PsionBenchmarkPackageFamily::NormativeSpecReading,
                prompt_format_id: String::from("bounded_explanation_v1"),
                grader_id: String::from("exact_label_v1"),
                prompt_digest: String::from("spec-prompt-digest-1"),
                source_ids: vec![
                    String::from("spec_quiz_eval_pack_v1"),
                    String::from("wasm_core_spec_release_2"),
                ],
                task: PsionBenchmarkTaskContract::NormativeSpecReading {
                    normative_source_ref: String::from("wasm://core/validation"),
                    required_section_anchor: String::from("2.5.1"),
                },
                detail: String::from("Normative spec item checks section-anchored reading."),
            }],
            "Normative spec package uses the shared contract with an exact-label grader.",
        )?,
        record_psion_benchmark_package(
            "psion_engineering_spec_benchmark_v1",
            PsionBenchmarkPackageFamily::EngineeringSpecInterpretation,
            benchmark_package(
                "psion_engineering_spec_benchmark_v1",
                &["eng-case-1", "eng-case-2"],
            ),
            vec![explanation_prompt_format()],
            vec![rubric_grader(), exact_label_grader()],
            contamination_inputs(&["spec_quiz_eval_pack_v1", "wasm_core_spec_release_2"])?,
            vec![
                PsionBenchmarkItem {
                    item_id: String::from("eng-case-1"),
                    family: PsionBenchmarkPackageFamily::EngineeringSpecInterpretation,
                    prompt_format_id: String::from("bounded_explanation_v1"),
                    grader_id: String::from("rubric_reasoning_v1"),
                    prompt_digest: String::from("eng-prompt-digest-1"),
                    source_ids: vec![String::from("spec_quiz_eval_pack_v1")],
                    task: PsionBenchmarkTaskContract::EngineeringSpecInterpretation {
                        artifact_ref: String::from("artifact://psion/spec/queueing_model"),
                        expected_constraint: String::from("throughput ceiling"),
                    },
                    detail: String::from(
                        "Engineering spec interpretation item checks bounded implementation inference.",
                    ),
                },
                PsionBenchmarkItem {
                    item_id: String::from("eng-case-2"),
                    family: PsionBenchmarkPackageFamily::EngineeringSpecInterpretation,
                    prompt_format_id: String::from("bounded_explanation_v1"),
                    grader_id: String::from("exact_label_v1"),
                    prompt_digest: String::from("eng-prompt-digest-2"),
                    source_ids: vec![
                        String::from("spec_quiz_eval_pack_v1"),
                        String::from("wasm_core_spec_release_2"),
                    ],
                    task: PsionBenchmarkTaskContract::EngineeringSpecInterpretation {
                        artifact_ref: String::from("artifact://psion/spec/queue_depth_limit"),
                        expected_constraint: String::from("queue depth hard limit"),
                    },
                    detail: String::from(
                        "Engineering spec interpretation exact-label item checks deterministic constraint extraction against a canonical engineering boundary.",
                    ),
                },
            ],
            "Engineering spec package uses the shared contract with both rubric-backed and exact-label graders.",
        )?,
        record_psion_benchmark_package(
            "psion_memorization_reasoning_benchmark_v1",
            PsionBenchmarkPackageFamily::MemorizationVersusReasoning,
            benchmark_package("psion_memorization_reasoning_benchmark_v1", &["mem-case-1"]),
            vec![explanation_prompt_format()],
            vec![exact_label_grader()],
            contamination_inputs(&["spec_quiz_eval_pack_v1"])?,
            vec![PsionBenchmarkItem {
                item_id: String::from("mem-case-1"),
                family: PsionBenchmarkPackageFamily::MemorizationVersusReasoning,
                prompt_format_id: String::from("bounded_explanation_v1"),
                grader_id: String::from("exact_label_v1"),
                prompt_digest: String::from("mem-prompt-digest-1"),
                source_ids: vec![String::from("spec_quiz_eval_pack_v1")],
                task: PsionBenchmarkTaskContract::MemorizationVersusReasoning {
                    seed_fact_ref: String::from("seed://psion/memorization/1"),
                    perturbation_ref: String::from("perturbation://psion/memorization/1"),
                    reasoning_required: true,
                },
                detail: String::from(
                    "Memorization-versus-reasoning item checks that the package can separate recall from transfer.",
                ),
            }],
            "Memorization-versus-reasoning package uses the shared exact-label contract.",
        )?,
        record_psion_benchmark_package(
            "psion_route_benchmark_v1",
            PsionBenchmarkPackageFamily::RouteEvaluation,
            benchmark_package("psion_route_benchmark_v1", &["route-case-1"]),
            vec![route_prompt_format()],
            vec![exact_route_grader()],
            contamination_inputs(&["spec_quiz_eval_pack_v1"])?,
            vec![PsionBenchmarkItem {
                item_id: String::from("route-case-1"),
                family: PsionBenchmarkPackageFamily::RouteEvaluation,
                prompt_format_id: String::from("route_decision_v1"),
                grader_id: String::from("exact_route_v1"),
                prompt_digest: String::from("route-prompt-digest-1"),
                source_ids: vec![String::from("spec_quiz_eval_pack_v1")],
                task: PsionBenchmarkTaskContract::RouteEvaluation {
                    expected_route: PsionRouteKind::ExactExecutorHandoff,
                    route_boundary_ref: String::from("route://psion/exactness_boundary"),
                },
                detail: String::from("Route item checks direct vs handoff vs refusal decisions."),
            }],
            "Route package uses the shared structured route-prompt and exact-route grader contracts.",
        )?,
        record_psion_benchmark_package(
            "psion_refusal_benchmark_v1",
            PsionBenchmarkPackageFamily::RefusalEvaluation,
            benchmark_package("psion_refusal_benchmark_v1", &["refusal-case-1"]),
            vec![refusal_prompt_format()],
            vec![exact_refusal_grader()],
            contamination_inputs(&["spec_quiz_eval_pack_v1"])?,
            vec![PsionBenchmarkItem {
                item_id: String::from("refusal-case-1"),
                family: PsionBenchmarkPackageFamily::RefusalEvaluation,
                prompt_format_id: String::from("refusal_decision_v1"),
                grader_id: String::from("exact_refusal_v1"),
                prompt_digest: String::from("refusal-prompt-digest-1"),
                source_ids: vec![String::from("spec_quiz_eval_pack_v1")],
                task: PsionBenchmarkTaskContract::RefusalEvaluation {
                    expected_reason_code: String::from("unsupported_exactness_request"),
                    refusal_boundary_ref: String::from("route://psion/refusal_boundary"),
                },
                detail: String::from("Refusal item checks structured unsupported-request refusal."),
            }],
            "Refusal package uses the shared structured refusal-prompt and exact-refusal grader contracts.",
        )?,
    ])
}

fn benchmark_package(package_id: &str, case_ids: &[&str]) -> BenchmarkPackage {
    BenchmarkPackage::new(
        BenchmarkPackageKey::new(package_id, "v1"),
        format!("Display {package_id}"),
        EnvironmentPackageKey::new("env.psion.benchmark", "2026.03.22"),
        3,
        BenchmarkAggregationKind::MedianScore,
    )
    .with_cases(
        case_ids
            .iter()
            .map(|case_id| BenchmarkCase::new(*case_id))
            .collect(),
    )
}

fn contamination_inputs(
    source_ids: &[&str],
) -> Result<psionic_train::PsionBenchmarkContaminationInputs, Box<dyn Error>> {
    Ok(record_psion_benchmark_contamination_inputs(
        source_ids.iter().map(|source| String::from(*source)).collect(),
        vec![String::from("spec_quiz_eval_pack_v1")],
        vec![String::from("spec_quiz_eval_pack_v1")],
        "review://psion/benchmark/near-duplicate-v1",
        "Benchmark package preserves the held-out source ids, training-excluded ids, and near-duplicate review reference required by the isolation contract.",
    )?)
}

fn explanation_prompt_format() -> PsionBenchmarkPromptFormat {
    PsionBenchmarkPromptFormat {
        format_id: String::from("bounded_explanation_v1"),
        system_instruction_ref: String::from("prompt://psion/benchmark/system/bounded-explanation"),
        user_template_ref: String::from("prompt://psion/benchmark/user/bounded-explanation"),
        envelope: PsionBenchmarkPromptEnvelope::CitedSectionPrompt,
        expected_response_format: PsionBenchmarkExpectedResponseFormat::BoundedExplanationJson,
        preserve_source_boundaries: true,
        detail: String::from(
            "Explanation prompts preserve source boundaries and produce bounded explanation JSON.",
        ),
    }
}

fn route_prompt_format() -> PsionBenchmarkPromptFormat {
    PsionBenchmarkPromptFormat {
        format_id: String::from("route_decision_v1"),
        system_instruction_ref: String::from("prompt://psion/benchmark/system/route"),
        user_template_ref: String::from("prompt://psion/benchmark/user/route"),
        envelope: PsionBenchmarkPromptEnvelope::StructuredRouteDecisionJson,
        expected_response_format: PsionBenchmarkExpectedResponseFormat::RouteDecisionJson,
        preserve_source_boundaries: true,
        detail: String::from("Route prompts require a structured route-decision JSON response."),
    }
}

fn refusal_prompt_format() -> PsionBenchmarkPromptFormat {
    PsionBenchmarkPromptFormat {
        format_id: String::from("refusal_decision_v1"),
        system_instruction_ref: String::from("prompt://psion/benchmark/system/refusal"),
        user_template_ref: String::from("prompt://psion/benchmark/user/refusal"),
        envelope: PsionBenchmarkPromptEnvelope::StructuredRefusalDecisionJson,
        expected_response_format: PsionBenchmarkExpectedResponseFormat::RefusalDecisionJson,
        preserve_source_boundaries: true,
        detail: String::from(
            "Refusal prompts require a structured refusal-decision JSON response.",
        ),
    }
}

fn rubric_grader() -> PsionBenchmarkGraderInterface {
    PsionBenchmarkGraderInterface::RubricScore(PsionBenchmarkRubricGrader {
        grader_id: String::from("rubric_reasoning_v1"),
        rubric_ref: String::from("rubric://psion/benchmark/reasoning"),
        minimum_pass_bps: 7800,
        dimensions: vec![
            PsionBenchmarkRubricDimension {
                dimension_id: String::from("correctness"),
                weight_bps: 6000,
                detail: String::from("Checks the substantive answer."),
            },
            PsionBenchmarkRubricDimension {
                dimension_id: String::from("truth_boundary"),
                weight_bps: 4000,
                detail: String::from(
                    "Checks explicit assumptions, uncertainty, and normative-versus-inference separation.",
                ),
            },
        ],
        detail: String::from(
            "Rubric grader supports bounded reasoning labels without collapsing them into one exact string.",
        ),
    })
}

fn exact_label_grader() -> PsionBenchmarkGraderInterface {
    PsionBenchmarkGraderInterface::ExactLabel(PsionBenchmarkExactLabelGrader {
        grader_id: String::from("exact_label_v1"),
        label_namespace: String::from("psion.spec.reading"),
        accepted_labels: vec![String::from("pass"), String::from("boundary_clear")],
        detail: String::from("Exact-label grader supports deterministic label-based grading."),
    })
}

fn exact_route_grader() -> PsionBenchmarkGraderInterface {
    PsionBenchmarkGraderInterface::ExactRoute(PsionBenchmarkExactRouteGrader {
        grader_id: String::from("exact_route_v1"),
        expected_route: PsionRouteKind::ExactExecutorHandoff,
        detail: String::from("Route grader requires the declared route exactly."),
    })
}

fn exact_refusal_grader() -> PsionBenchmarkGraderInterface {
    PsionBenchmarkGraderInterface::ExactRefusal(PsionBenchmarkExactRefusalGrader {
        grader_id: String::from("exact_refusal_v1"),
        accepted_reason_codes: vec![String::from("unsupported_exactness_request")],
        detail: String::from("Refusal grader requires one of the admitted refusal codes exactly."),
    })
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .map(PathBuf::from)
        .ok_or_else(|| String::from("could not resolve workspace root").into())
}

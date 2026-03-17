use std::{collections::BTreeMap, fs, path::Path};

use psionic_eval::{EvalArtifact, TassadarSequenceEvalError, build_tassadar_sequence_dataset};
use psionic_models::{TassadarExecutorTrainableSurface, TassadarExecutorTransformer};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    TASSADAR_EXECUTOR_NEXT_RUN_PLAN_FILE, TASSADAR_EXECUTOR_POSTMORTEM_FILE,
    TASSADAR_EXECUTOR_TRACE_DIVERGENCE_FILE, TassadarExecutorReferenceRunBundle,
    TassadarExecutorRunError, TassadarExecutorTeacherForcedTrainingStrategy,
    TassadarExecutorTelemetryError, TassadarExecutorTraceDivergenceReport,
    TassadarExecutorTrainingConfig, TassadarExecutorTrainingReport,
    augment_tassadar_training_run_with_telemetry,
    execute_tassadar_training_run_without_benchmark,
};

const RUN_BUNDLE_FILE: &str = "run_bundle.json";
const TRAINING_REPORT_FILE: &str = "training_report.json";

/// Stable run identifier for the first honest 9x9 reference run.
pub const TASSADAR_EXECUTOR_SUDOKU_9X9_REFERENCE_RUN_ID: &str =
    "tassadar-executor-transformer-sudoku-9x9-reference-run-v0";
/// Canonical repo path for the first honest 9x9 reference run.
pub const TASSADAR_EXECUTOR_SUDOKU_9X9_REFERENCE_RUN_OUTPUT_DIR: &str =
    "fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0";
/// Canonical machine-readable fit report for the 9x9 reference run.
pub const TASSADAR_EXECUTOR_SEQUENCE_FIT_REPORT_FILE: &str = "sequence_fit_report.json";

/// One artifact file size captured for the persisted 9x9 run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorArtifactSizeReport {
    /// Relative artifact path inside the run directory.
    pub artifact_ref: String,
    /// Actual file size on disk in bytes.
    pub size_bytes: u64,
}

/// One stage-scoped context-fit fact for the 9x9 run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorContextStageFit {
    /// Stable stage identifier.
    pub stage_id: String,
    /// Target-token cap applied during the stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_token_cap: Option<u32>,
    /// Longest prompt-plus-window token count admitted by the stage.
    pub max_total_tokens: u32,
    /// Whether the longest stage sequence fits the model context.
    pub fits_model_context: bool,
    /// Remaining model-context headroom when the stage fits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_headroom_tokens: Option<u32>,
    /// Positive context overflow when the stage does not fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_overflow_tokens: Option<u32>,
}

/// Machine-readable fit analysis for the first honest 9x9 run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorSequenceFitReport {
    /// Stable run identifier.
    pub run_id: String,
    /// Dataset storage key used for the run.
    pub dataset_storage_key: String,
    /// Dataset digest used for the run.
    pub dataset_digest: String,
    /// Stable model identifier.
    pub model_id: String,
    /// Stable trained model descriptor digest.
    pub trained_model_descriptor_digest: String,
    /// Active trainable surface.
    pub trainable_surface: TassadarExecutorTrainableSurface,
    /// Explicit teacher-forced strategy used by the run.
    pub teacher_forced_training_strategy: TassadarExecutorTeacherForcedTrainingStrategy,
    /// Vocabulary size for the learned lane.
    pub vocab_size: u32,
    /// Hidden width for one decode step.
    pub hidden_width: u32,
    /// Maximum context tokens admitted by the model.
    pub model_max_sequence_tokens: u32,
    /// Minimum prompt length in the dataset.
    pub prompt_token_count_min: u32,
    /// Maximum prompt length in the dataset.
    pub prompt_token_count_max: u32,
    /// Minimum full target length in the dataset.
    pub target_token_count_min: u32,
    /// Maximum full target length in the dataset.
    pub target_token_count_max: u32,
    /// Minimum full sequence length in the dataset.
    pub total_token_count_min: u32,
    /// Maximum full sequence length in the dataset.
    pub total_token_count_max: u32,
    /// Whether any full 9x9 sequence fits the current model context.
    pub full_sequence_fits_model_context: bool,
    /// Smallest overflow over the model context on the full dataset.
    pub full_sequence_context_overflow_min: u32,
    /// Largest overflow over the model context on the full dataset.
    pub full_sequence_context_overflow_max: u32,
    /// Explicit stage-by-stage bounded fit facts.
    pub stage_fits: Vec<TassadarExecutorContextStageFit>,
    /// Explicit eval window used for training-time validation and telemetry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_target_token_cap: Option<u32>,
    /// Estimated float-buffer bytes for a naive full forward pass on the longest full sequence.
    pub estimated_full_forward_buffer_bytes_max: u64,
    /// Estimated float-buffer bytes for the largest bounded training stage.
    pub estimated_bounded_forward_buffer_bytes_max: u64,
    /// Estimated live decode-state bytes for the bounded incremental strategy.
    pub estimated_incremental_decode_live_bytes_max: u64,
    /// Actual artifact file sizes currently on disk.
    pub artifact_file_sizes: Vec<TassadarExecutorArtifactSizeReport>,
    /// Exact statement the audit must keep honest.
    pub outcome_statement: String,
    /// Plain blockers that keep the learned 9x9 lane partial.
    pub blocking_reasons: Vec<String>,
    /// Stable report digest.
    pub report_digest: String,
}

impl TassadarExecutorSequenceFitReport {
    fn new(
        output_dir: &Path,
        run_bundle: &TassadarExecutorReferenceRunBundle,
        config: &TassadarExecutorTrainingConfig,
    ) -> Result<Self, TassadarSudoku9x9ReferenceRunError> {
        let dataset_bundle =
            build_tassadar_sequence_dataset(config.workload, config.dataset_version.as_str())
                .map_err(TassadarSudoku9x9ReferenceRunError::SequenceEval)?;
        let model = TassadarExecutorTransformer::sudoku_9x9_with_surface(config.trainable_surface);
        let prompt_token_count_min = dataset_bundle
            .dataset
            .examples
            .iter()
            .map(|example| example.metadata.prompt_token_count)
            .min()
            .unwrap_or(0);
        let prompt_token_count_max = dataset_bundle
            .dataset
            .examples
            .iter()
            .map(|example| example.metadata.prompt_token_count)
            .max()
            .unwrap_or(0);
        let target_token_count_min = dataset_bundle
            .dataset
            .examples
            .iter()
            .map(|example| example.metadata.target_token_count)
            .min()
            .unwrap_or(0);
        let target_token_count_max = dataset_bundle
            .dataset
            .examples
            .iter()
            .map(|example| example.metadata.target_token_count)
            .max()
            .unwrap_or(0);
        let total_token_count_min = dataset_bundle
            .dataset
            .examples
            .iter()
            .map(|example| example.metadata.total_token_count)
            .min()
            .unwrap_or(0);
        let total_token_count_max = dataset_bundle
            .dataset
            .examples
            .iter()
            .map(|example| example.metadata.total_token_count)
            .max()
            .unwrap_or(0);
        let model_max_sequence_tokens = model.descriptor().config.max_sequence_tokens as u32;
        let full_sequence_fits_model_context = total_token_count_max <= model_max_sequence_tokens;
        let full_sequence_context_overflow_min =
            total_token_count_min.saturating_sub(model_max_sequence_tokens);
        let full_sequence_context_overflow_max =
            total_token_count_max.saturating_sub(model_max_sequence_tokens);
        let stage_fits = config
            .resolved_stages()
            .into_iter()
            .map(|stage| {
                let target_token_cap = stage
                    .max_train_target_tokens_per_example
                    .map(|value| value as u32)
                    .or(Some(target_token_count_max));
                let max_total_tokens = prompt_token_count_max.saturating_add(
                    stage
                        .max_train_target_tokens_per_example
                        .map(|value| value as u32)
                        .unwrap_or(target_token_count_max),
                );
                let fits_model_context = max_total_tokens <= model_max_sequence_tokens;
                TassadarExecutorContextStageFit {
                    stage_id: stage.stage_id,
                    target_token_cap,
                    max_total_tokens,
                    fits_model_context,
                    context_headroom_tokens: if fits_model_context {
                        Some(model_max_sequence_tokens - max_total_tokens)
                    } else {
                        None
                    },
                    context_overflow_tokens: if fits_model_context {
                        None
                    } else {
                        Some(max_total_tokens - model_max_sequence_tokens)
                    },
                }
            })
            .collect::<Vec<_>>();
        let bounded_stage_max_total_tokens = stage_fits
            .iter()
            .map(|stage| stage.max_total_tokens)
            .max()
            .unwrap_or(prompt_token_count_max);
        let estimated_full_forward_buffer_bytes_max = estimate_forward_buffer_bytes(
            total_token_count_max as usize,
            model.descriptor().config.hidden_width(),
            model.descriptor().config.vocab_size,
        );
        let estimated_bounded_forward_buffer_bytes_max = estimate_forward_buffer_bytes(
            bounded_stage_max_total_tokens as usize,
            model.descriptor().config.hidden_width(),
            model.descriptor().config.vocab_size,
        );
        let estimated_incremental_decode_live_bytes_max = estimate_incremental_decode_live_bytes(
            bounded_stage_max_total_tokens as usize,
            model.descriptor().config.hidden_width(),
            model.descriptor().config.vocab_size,
        );
        let artifact_file_sizes = collect_artifact_sizes(output_dir, run_bundle)?;
        let blocking_reasons = vec![
            format!(
                "full 9x9 sequences exceed the current model context by {} to {} tokens",
                full_sequence_context_overflow_min, full_sequence_context_overflow_max
            ),
            format!(
                "the run therefore only validates the first {} target tokens per case",
                config.max_eval_target_tokens_per_example.unwrap_or(0)
            ),
        ];

        let mut report = Self {
            run_id: run_bundle.run_id.clone(),
            dataset_storage_key: run_bundle.dataset_storage_key.clone(),
            dataset_digest: run_bundle.dataset_digest.clone(),
            model_id: model.descriptor().model.model_id.clone(),
            trained_model_descriptor_digest: run_bundle.trained_model_descriptor_digest.clone(),
            trainable_surface: config.trainable_surface,
            teacher_forced_training_strategy: config.teacher_forced_training_strategy,
            vocab_size: model.descriptor().config.vocab_size as u32,
            hidden_width: model.descriptor().config.hidden_width() as u32,
            model_max_sequence_tokens,
            prompt_token_count_min,
            prompt_token_count_max,
            target_token_count_min,
            target_token_count_max,
            total_token_count_min,
            total_token_count_max,
            full_sequence_fits_model_context,
            full_sequence_context_overflow_min,
            full_sequence_context_overflow_max,
            stage_fits,
            eval_target_token_cap: config.max_eval_target_tokens_per_example.map(|value| value as u32),
            estimated_full_forward_buffer_bytes_max,
            estimated_bounded_forward_buffer_bytes_max,
            estimated_incremental_decode_live_bytes_max,
            artifact_file_sizes,
            outcome_statement: String::from("9x9 only partially fit and remains blocked"),
            blocking_reasons,
            report_digest: String::new(),
        };
        report.report_digest =
            stable_digest(b"psionic_tassadar_executor_sequence_fit_report|", &report);
        Ok(report)
    }
}

/// Typed postmortem finding for the first honest 9x9 run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarSudoku9x9PostmortemFinding {
    /// Stable finding identifier.
    pub finding_id: String,
    /// Human-readable summary.
    pub summary: String,
    /// Supporting artifact refs.
    pub supporting_artifacts: Vec<String>,
}

/// Machine-readable review for the first honest 9x9 run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarSudoku9x9PostmortemReport {
    /// Stable run identifier.
    pub run_id: String,
    /// Stable dataset storage key used for the run.
    pub dataset_storage_key: String,
    /// Stable fit report digest backing the review.
    pub sequence_fit_report_digest: String,
    /// Stable divergence report digest backing the review.
    pub divergence_report_digest: String,
    /// Ordered findings.
    pub findings: Vec<TassadarSudoku9x9PostmortemFinding>,
    /// Stable report digest.
    pub report_digest: String,
}

impl TassadarSudoku9x9PostmortemReport {
    fn new(
        run_bundle: &TassadarExecutorReferenceRunBundle,
        fit_report: &TassadarExecutorSequenceFitReport,
        training_report: &TassadarExecutorTrainingReport,
        divergence: &TassadarExecutorTraceDivergenceReport,
    ) -> Self {
        let findings = vec![
            TassadarSudoku9x9PostmortemFinding {
                finding_id: String::from("full_sequence_exceeds_context"),
                summary: format!(
                    "The full 9x9 traces are too long for the current learned lane: the longest sequence is {} tokens against a {}-token model context.",
                    fit_report.total_token_count_max, fit_report.model_max_sequence_tokens
                ),
                supporting_artifacts: vec![String::from(TASSADAR_EXECUTOR_SEQUENCE_FIT_REPORT_FILE)],
            },
            TassadarSudoku9x9PostmortemFinding {
                finding_id: String::from("bounded_prefix_only"),
                summary: format!(
                    "This run is only an early-prefix truth run: all exactness and divergence artifacts are bounded to the first {} target tokens.",
                    fit_report.eval_target_token_cap.unwrap_or(0)
                ),
                supporting_artifacts: vec![
                    String::from(TASSADAR_EXECUTOR_SEQUENCE_FIT_REPORT_FILE),
                    String::from(TRAINING_REPORT_FILE),
                ],
            },
            TassadarSudoku9x9PostmortemFinding {
                finding_id: String::from("bounded_prefix_still_inexact"),
                summary: format!(
                    "Even on that bounded window, validation is still red: first-target exactness is {} bps, first-32 exactness is {} bps, and exact traces are {}/{}.",
                    training_report.evaluation.first_target_exactness_bps,
                    training_report.evaluation.first_32_token_exactness_bps,
                    training_report.evaluation.exact_trace_case_count,
                    training_report.evaluation.case_reports.len(),
                ),
                supporting_artifacts: vec![
                    String::from(TRAINING_REPORT_FILE),
                    String::from(TASSADAR_EXECUTOR_TRACE_DIVERGENCE_FILE),
                ],
            },
        ];

        let mut report = Self {
            run_id: run_bundle.run_id.clone(),
            dataset_storage_key: run_bundle.dataset_storage_key.clone(),
            sequence_fit_report_digest: fit_report.report_digest.clone(),
            divergence_report_digest: divergence.report_digest.clone(),
            findings,
            report_digest: String::new(),
        };
        report.report_digest =
            stable_digest(b"psionic_tassadar_sudoku_9x9_postmortem_report|", &report);
        report
    }
}

/// One explicit next-run action for the 9x9 lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarSudoku9x9NextRunAction {
    /// Stable action identifier.
    pub action_id: String,
    /// Human-readable summary.
    pub summary: String,
    /// Concrete success criteria.
    pub success_criteria: Vec<String>,
}

/// Machine-readable next-run plan for the 9x9 lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarSudoku9x9NextRunPlan {
    /// Stable run identifier.
    pub run_id: String,
    /// Stable postmortem digest backing the plan.
    pub basis_postmortem_digest: String,
    /// Ordered actions.
    pub actions: Vec<TassadarSudoku9x9NextRunAction>,
    /// Stable plan digest.
    pub plan_digest: String,
}

impl TassadarSudoku9x9NextRunPlan {
    fn new(run_id: &str, postmortem: &TassadarSudoku9x9PostmortemReport) -> Self {
        let actions = vec![
            TassadarSudoku9x9NextRunAction {
                action_id: String::from("later_window_truth"),
                summary: String::from(
                    "Add later-prefix or suffix-window eval artifacts instead of pretending the first bounded window says enough about the full 9x9 trace.",
                ),
                success_criteria: vec![
                    String::from("Persist at least one non-zero-offset target window report."),
                    String::from("Keep every window report explicit about which target slice it covers."),
                ],
            },
            TassadarSudoku9x9NextRunAction {
                action_id: String::from("context_or_model_change"),
                summary: String::from(
                    "Either increase the learned lane context budget or adopt an explicitly windowed long-trace training/eval regime beyond the first 512 tokens.",
                ),
                success_criteria: vec![
                    String::from("Full 9x9 prompt-plus-target fit becomes truthful, or later windows are covered by explicit machine-readable reports."),
                    String::from("No artifact implies full-trace 9x9 exactness before that fit exists."),
                ],
            },
            TassadarSudoku9x9NextRunAction {
                action_id: String::from("keep_phase_claim_partial"),
                summary: String::from(
                    "Keep the 9x9 learned lane in a partial state until bounded windows are exact and the full-trace fit problem is solved honestly.",
                ),
                success_criteria: vec![
                    String::from("Audit language continues to say `9x9 only partially fit and remains blocked` until the blocker is actually removed."),
                ],
            },
        ];

        let mut plan = Self {
            run_id: run_id.to_string(),
            basis_postmortem_digest: postmortem.report_digest.clone(),
            actions,
            plan_digest: String::new(),
        };
        plan.plan_digest = stable_digest(b"psionic_tassadar_sudoku_9x9_next_run_plan|", &plan);
        plan
    }
}

/// Errors while materializing the first honest 9x9 run bundle and review.
#[derive(Debug, Error)]
pub enum TassadarSudoku9x9ReferenceRunError {
    /// Training-run persistence failed.
    #[error(transparent)]
    Run(#[from] TassadarExecutorRunError),
    /// Telemetry augmentation failed.
    #[error(transparent)]
    Telemetry(#[from] TassadarExecutorTelemetryError),
    /// Dataset generation failed.
    #[error(transparent)]
    SequenceEval(#[from] TassadarSequenceEvalError),
    /// Reading one persisted artifact failed.
    #[error("failed to read `{path}`: {error}")]
    Read {
        /// File path.
        path: String,
        /// Source error.
        error: std::io::Error,
    },
    /// Decoding one persisted artifact failed.
    #[error("failed to decode `{artifact_kind}` from `{path}`: {error}")]
    Deserialize {
        /// Artifact kind.
        artifact_kind: String,
        /// File path.
        path: String,
        /// Source error.
        error: serde_json::Error,
    },
    /// Serializing one artifact failed.
    #[error("failed to serialize `{artifact_kind}`: {error}")]
    Serialize {
        /// Artifact kind.
        artifact_kind: String,
        /// Source error.
        error: serde_json::Error,
    },
    /// Writing one artifact failed.
    #[error("failed to write `{path}`: {error}")]
    Write {
        /// File path.
        path: String,
        /// Source error.
        error: std::io::Error,
    },
}

/// Returns the canonical config for the first honest 9x9 reference run.
#[must_use]
pub fn tassadar_executor_sudoku_9x9_reference_run_config() -> TassadarExecutorTrainingConfig {
    TassadarExecutorTrainingConfig {
        run_id: String::from(TASSADAR_EXECUTOR_SUDOKU_9X9_REFERENCE_RUN_ID),
        workload: psionic_eval::TassadarSequenceWorkload::Sudoku9x9,
        dataset_version: String::from("scale-v0"),
        epochs: 1,
        learning_rate: 0.05,
        max_train_target_tokens_per_example: Some(512),
        max_eval_target_tokens_per_example: Some(512),
        terminal_stage_learning_rate_scale: None,
        trainable_surface: TassadarExecutorTrainableSurface::OutputHeadOnly,
        teacher_forced_training_strategy:
            TassadarExecutorTeacherForcedTrainingStrategy::IncrementalDecodeWindow,
        structural_supervision: crate::TassadarExecutorStructuralSupervisionConfig::next_token_only(),
        curriculum_stages: vec![
            crate::TassadarExecutorCurriculumStage::new("prompt_to_first_token", Some(1), 1),
            crate::TassadarExecutorCurriculumStage::new("prompt_to_first_8_tokens", Some(8), 1),
            crate::TassadarExecutorCurriculumStage::new("prompt_to_first_32_tokens", Some(32), 1),
            crate::TassadarExecutorCurriculumStage::new(
                "prompt_to_first_128_tokens",
                Some(128),
                1,
            ),
        ],
        validate_every_epoch: true,
        select_best_checkpoint_by_boundary: true,
    }
}

/// Executes the first honest 9x9 reference run and augments it with fit/review artifacts.
pub fn execute_tassadar_sudoku_9x9_reference_run(
    output_dir: &Path,
) -> Result<TassadarExecutorReferenceRunBundle, TassadarSudoku9x9ReferenceRunError> {
    execute_tassadar_training_run_without_benchmark(
        output_dir,
        &tassadar_executor_sudoku_9x9_reference_run_config(),
    )?;
    augment_tassadar_training_run_with_telemetry(output_dir)?;
    augment_tassadar_sudoku_9x9_run_with_fit_and_review(output_dir)
}

/// Augments one persisted 9x9 run with the fit report, postmortem, and next-run plan.
pub fn augment_tassadar_sudoku_9x9_run_with_fit_and_review(
    output_dir: &Path,
) -> Result<TassadarExecutorReferenceRunBundle, TassadarSudoku9x9ReferenceRunError> {
    let run_bundle: TassadarExecutorReferenceRunBundle = read_json(
        output_dir.join(RUN_BUNDLE_FILE),
        "tassadar_reference_run_bundle",
    )?;
    let training_report: TassadarExecutorTrainingReport = read_json(
        output_dir.join(TRAINING_REPORT_FILE),
        "tassadar_training_report",
    )?;
    let config: TassadarExecutorTrainingConfig = training_report.config.clone();
    let divergence: TassadarExecutorTraceDivergenceReport = read_json(
        output_dir.join(TASSADAR_EXECUTOR_TRACE_DIVERGENCE_FILE),
        "tassadar_trace_divergence_report",
    )?;

    let fit_report = TassadarExecutorSequenceFitReport::new(output_dir, &run_bundle, &config)?;
    let postmortem = TassadarSudoku9x9PostmortemReport::new(
        &run_bundle,
        &fit_report,
        &training_report,
        &divergence,
    );
    let next_run_plan = TassadarSudoku9x9NextRunPlan::new(run_bundle.run_id.as_str(), &postmortem);

    let mut artifact_map = run_bundle
        .artifacts
        .iter()
        .cloned()
        .map(|artifact| (artifact.artifact_ref.clone(), artifact))
        .collect::<BTreeMap<_, _>>();
    for artifact in [
        write_json_artifact(
            output_dir,
            TASSADAR_EXECUTOR_SEQUENCE_FIT_REPORT_FILE,
            "tassadar_sequence_fit_report",
            &fit_report,
        )?,
        write_json_artifact(
            output_dir,
            TASSADAR_EXECUTOR_POSTMORTEM_FILE,
            "tassadar_sudoku_9x9_postmortem",
            &postmortem,
        )?,
        write_json_artifact(
            output_dir,
            TASSADAR_EXECUTOR_NEXT_RUN_PLAN_FILE,
            "tassadar_sudoku_9x9_next_run_plan",
            &next_run_plan,
        )?,
    ] {
        artifact_map.insert(artifact.artifact_ref.clone(), artifact);
    }

    let mut updated_bundle = run_bundle;
    updated_bundle.artifacts = artifact_map.into_values().collect();
    updated_bundle.sequence_fit_report_digest = Some(fit_report.report_digest.clone());
    updated_bundle.bundle_digest.clear();
    updated_bundle.bundle_digest =
        stable_digest(b"psionic_tassadar_executor_reference_run_bundle|", &updated_bundle);
    write_json(
        output_dir.join(RUN_BUNDLE_FILE),
        "tassadar_reference_run_bundle",
        &updated_bundle,
    )?;
    Ok(updated_bundle)
}

fn collect_artifact_sizes(
    output_dir: &Path,
    run_bundle: &TassadarExecutorReferenceRunBundle,
) -> Result<Vec<TassadarExecutorArtifactSizeReport>, TassadarSudoku9x9ReferenceRunError> {
    let mut reports = run_bundle
        .artifacts
        .iter()
        .map(|artifact| {
            let path = output_dir.join(&artifact.artifact_ref);
            let size_bytes = fs::metadata(&path)
                .map_err(|error| TassadarSudoku9x9ReferenceRunError::Read {
                    path: path.display().to_string(),
                    error,
                })?
                .len();
            Ok(TassadarExecutorArtifactSizeReport {
                artifact_ref: artifact.artifact_ref.clone(),
                size_bytes,
            })
        })
        .collect::<Result<Vec<_>, TassadarSudoku9x9ReferenceRunError>>()?;
    let run_bundle_path = output_dir.join(RUN_BUNDLE_FILE);
    reports.push(TassadarExecutorArtifactSizeReport {
        artifact_ref: String::from(RUN_BUNDLE_FILE),
        size_bytes: fs::metadata(&run_bundle_path)
            .map_err(|error| TassadarSudoku9x9ReferenceRunError::Read {
                path: run_bundle_path.display().to_string(),
                error,
            })?
            .len(),
    });
    reports.sort_by(|left, right| {
        right
            .size_bytes
            .cmp(&left.size_bytes)
            .then(left.artifact_ref.cmp(&right.artifact_ref))
    });
    Ok(reports)
}

fn estimate_forward_buffer_bytes(
    sequence_len: usize,
    hidden_width: usize,
    vocab_size: usize,
) -> u64 {
    let positions = sequence_len.saturating_sub(1) as u64;
    positions.saturating_mul(((hidden_width * 2 + vocab_size) * 4) as u64)
}

fn estimate_incremental_decode_live_bytes(
    sequence_len: usize,
    hidden_width: usize,
    vocab_size: usize,
) -> u64 {
    let kv_point_bytes = 24_u64;
    let prefix_token_bytes = 4_u64;
    let state_bytes =
        (sequence_len as u64).saturating_mul(kv_point_bytes.saturating_add(prefix_token_bytes));
    let scratch_bytes = ((hidden_width * 2 + vocab_size) * 4) as u64;
    state_bytes.saturating_add(scratch_bytes)
}

fn read_json<T>(
    path: impl AsRef<Path>,
    artifact_kind: &str,
) -> Result<T, TassadarSudoku9x9ReferenceRunError>
where
    T: DeserializeOwned,
{
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| TassadarSudoku9x9ReferenceRunError::Read {
        path: path.display().to_string(),
        error,
    })?;
    serde_json::from_slice(&bytes).map_err(|error| TassadarSudoku9x9ReferenceRunError::Deserialize {
        artifact_kind: artifact_kind.to_string(),
        path: path.display().to_string(),
        error,
    })
}

fn write_json_artifact<T>(
    output_dir: &Path,
    relative_path: &str,
    artifact_kind: &str,
    value: &T,
) -> Result<EvalArtifact, TassadarSudoku9x9ReferenceRunError>
where
    T: Serialize,
{
    let path = output_dir.join(relative_path);
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        TassadarSudoku9x9ReferenceRunError::Serialize {
            artifact_kind: artifact_kind.to_string(),
            error,
        }
    })?;
    fs::write(&path, &bytes).map_err(|error| TassadarSudoku9x9ReferenceRunError::Write {
        path: path.display().to_string(),
        error,
    })?;
    Ok(EvalArtifact::new(
        artifact_kind,
        relative_path,
        bytes.as_slice(),
    ))
}

fn write_json<T>(
    path: impl AsRef<Path>,
    artifact_kind: &str,
    value: &T,
) -> Result<(), TassadarSudoku9x9ReferenceRunError>
where
    T: Serialize,
{
    let path = path.as_ref();
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        TassadarSudoku9x9ReferenceRunError::Serialize {
            artifact_kind: artifact_kind.to_string(),
            error,
        }
    })?;
    fs::write(path, &bytes).map_err(|error| TassadarSudoku9x9ReferenceRunError::Write {
        path: path.display().to_string(),
        error,
    })
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded = serde_json::to_vec(value)
        .expect("Tassadar Sudoku-9x9 reference run value should serialize");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

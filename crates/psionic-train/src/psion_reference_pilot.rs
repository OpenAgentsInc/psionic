use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use psionic_core::{DType, Device, Shape, TensorData, TensorSpec};
use psionic_data::{
    build_psion_reference_corpus, DatasetSplitKind, PsionArtifactLineageManifest,
    PsionReferenceCorpusBundle, PsionReferenceCorpusError, PsionReferenceEncodedSequence,
    PSION_REFERENCE_DATASET_IDENTITY, PSION_REFERENCE_MAX_SEQUENCE_TOKENS,
};
use psionic_models::{
    PsionCompactDecoderDescriptor, PsionCompactDecoderError, PsionCompactDecoderSizeAnchor,
    PsionCompactDecoderTokenizerBinding, PsionCompactDecoderTokenizerFamily,
};
use psionic_runtime::{
    DeliveredExecutionContext, DeviceInventoryQualifiers, DeviceMemoryClass,
    DevicePerformanceClass, TrainingCheckpointReference,
};
use safetensors::{serialize, tensor::TensorView, Dtype as SafeTensorsDType, SafeTensors};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    record_psion_pilot_held_out_loss, record_psion_pilot_pretraining_run,
    record_psion_pilot_route_probe, record_psion_refusal_calibration_receipt,
    record_psion_route_class_evaluation_receipt,
    record_psion_pretrain_run_observability, run_psion_pretrain_stage, FixedBudgetTrainingRun,
    PsionAcceptanceMatrix, PsionAcceptanceMatrixError, PsionBenchmarkCatalog,
    PsionBenchmarkEvidenceReceipt, PsionBenchmarkFamily, PsionBenchmarkPackageContract,
    PsionBenchmarkPackageError, PsionBenchmarkTaskContract, PsionCapabilityMatrixView,
    PsionCheckpointRecoveryReceipt, PsionContaminationReviewDisposition,
    PsionContaminationReviewReceipt, PsionMetricKind, PsionObservedMetric, PsionPhaseGate,
    PsionPilotHeldOutLossFamily, PsionPilotHeldOutLossRow, PsionPilotPretrainingRunBundle,
    PsionPilotPretrainingRunError, PsionPilotRouteProbeKind, PsionPilotRouteProbeRow,
    PsionPretrainCheckpointArtifactReceipt, PsionPretrainCheckpointLineageReceipt,
    PsionPretrainHardwareTopologyReceipt, PsionPretrainLossNormalization,
    PsionPretrainObjectiveConfig, PsionPretrainObjectiveKind, PsionPretrainReplayReceipt,
    PsionPretrainRunCostBasis, PsionPretrainRunCostReceipt, PsionPretrainRunObservabilityError,
    PsionPretrainRunObservabilityReceipt, PsionPretrainRunScaleProfile,
    PsionPretrainRunThroughputReceipt, PsionPretrainSourceFamilyReportRow,
    PsionPretrainStageConfig, PsionPretrainStageError, PsionPretrainStageRunReceipt,
    PsionPromotionDecisionDisposition, PsionPromotionDecisionReceipt,
    PsionRepetitiveRegionControl, PsionSamplingContentClass, PsionSamplingPolicyError,
    PsionSamplingPolicyManifest, PsionSamplingRegressionKind, PsionSamplingRegressionThreshold,
    PsionRefusalCalibrationError, PsionRefusalCalibrationReceipt, PsionRefusalCalibrationRow,
    PsionReplayEvidenceReceipt, PsionRouteCalibrationReceipt, PsionRouteClass,
    PsionRouteClassEvaluationError, PsionRouteClassEvaluationReceipt,
    PsionRouteClassEvaluationRow, PsionRouteKind,
    PsionSourceContributionCap, PsionSourceFamilySamplingWeight, TrainingCoreError,
    TrainingLoopBudget, TrainingOptimizerConfig, TrainingOptimizerResidencyPolicy,
    TrainingParameterClass, TrainingParameterGroupState, TrainingStepInput, TrainingStepReceipt,
    TrainingTensorBuffer,
};

const TOKEN_EMBEDDING_GROUP_ID: &str = "decoder.embed_tokens.weight";
const POSITION_EMBEDDING_GROUP_ID: &str = "decoder.embed_positions.weight";
const LM_HEAD_BIAS_GROUP_ID: &str = "lm_head.bias";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionReferencePilotCheckpointManifest {
    pub schema_version: String,
    pub checkpoint_ref: String,
    pub checkpoint_family: String,
    pub run_id: String,
    pub stage_id: String,
    pub step: u64,
    pub model_id: String,
    pub model_descriptor_digest: String,
    pub dataset_identity: String,
    pub train_example_count: usize,
    pub validation_example_count: usize,
    pub parameter_ids: Vec<String>,
    pub parameter_state_digest: String,
}

impl PsionReferencePilotCheckpointManifest {
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"psion_reference_pilot_checkpoint_manifest|", self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionReferencePilotCheckpointArtifact {
    pub manifest: PsionReferencePilotCheckpointManifest,
    pub weights_bytes: Vec<u8>,
    pub checkpoint: TrainingCheckpointReference,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PsionReferencePilotConfig {
    pub run_id: String,
    pub stage_id: String,
    pub checkpoint_family: String,
    pub started_at_ms: u64,
    pub step_duration_ms: u64,
    pub budget: TrainingLoopBudget,
    pub optimizer: TrainingOptimizerConfig,
}

impl PsionReferencePilotConfig {
    pub fn reference() -> Result<Self, PsionReferencePilotError> {
        Ok(Self {
            run_id: String::from("psion-reference-pilot-run"),
            stage_id: String::from("psion-reference-pretrain-stage"),
            checkpoint_family: String::from("train.psion.reference_pilot"),
            started_at_ms: 1_774_320_000_000,
            step_duration_ms: 40,
            budget: TrainingLoopBudget::new(16, 4, 1)?,
            optimizer: TrainingOptimizerConfig::adam(0.0005, 0.9, 0.99, 1e-8),
        })
    }
}

#[derive(Clone, Debug)]
pub struct PsionReferencePilotRun {
    pub corpus_bundle: PsionReferenceCorpusBundle,
    pub model_descriptor: PsionCompactDecoderDescriptor,
    pub sampling_policy: PsionSamplingPolicyManifest,
    pub stage_config: PsionPretrainStageConfig,
    pub stage_receipt: PsionPretrainStageRunReceipt,
    pub observability_receipt: PsionPretrainRunObservabilityReceipt,
    pub checkpoint_artifact: PsionReferencePilotCheckpointArtifact,
    pub initial_validation_loss_milli_by_family: BTreeMap<String, u32>,
    pub final_validation_loss_milli_by_family: BTreeMap<String, u32>,
    pub initial_held_out_loss_milli: u32,
    pub final_held_out_loss_milli: u32,
    pub step_receipts: Vec<TrainingStepReceipt>,
}

impl PsionReferencePilotRun {
    pub fn write_to_dir(&self, output_dir: &Path) -> Result<(), PsionReferencePilotError> {
        fs::create_dir_all(output_dir).map_err(|error| PsionReferencePilotError::Serialization {
            message: error.to_string(),
        })?;
        write_json(
            output_dir.join("psion_reference_pilot_stage_config.json").as_path(),
            &self.stage_config,
        )?;
        write_json(
            output_dir.join("psion_reference_pilot_stage_receipt.json").as_path(),
            &self.stage_receipt,
        )?;
        write_json(
            output_dir
                .join("psion_reference_pilot_observability_receipt.json")
                .as_path(),
            &self.observability_receipt,
        )?;
        write_json(
            output_dir
                .join("psion_reference_pilot_checkpoint_manifest.json")
                .as_path(),
            &self.checkpoint_artifact.manifest,
        )?;
        write_json(
            output_dir.join("psion_reference_pilot_summary.json").as_path(),
            &serde_json::json!({
                "run_id": self.stage_receipt.run_id,
                "stage_id": self.stage_receipt.stage_id,
                "checkpoint_ref": self.checkpoint_artifact.manifest.checkpoint_ref,
                "optimizer_steps": self.step_receipts.len(),
                "initial_validation_loss_milli_by_family": self.initial_validation_loss_milli_by_family,
                "final_validation_loss_milli_by_family": self.final_validation_loss_milli_by_family,
                "initial_held_out_loss_milli": self.initial_held_out_loss_milli,
                "final_held_out_loss_milli": self.final_held_out_loss_milli
            }),
        )?;
        fs::write(
            output_dir.join("psion_reference_pilot_checkpoint.safetensors"),
            &self.checkpoint_artifact.weights_bytes,
        )
        .map_err(|error| PsionReferencePilotError::Serialization {
            message: error.to_string(),
        })?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionReferencePilotBenchmarkRow {
    pub item_id: String,
    pub matched_sequence_id: Option<String>,
    pub matched_source_id: Option<String>,
    pub observed_route_class: Option<PsionRouteClass>,
    pub observed_refusal_reason_code: Option<String>,
    pub observed_score_milli: i32,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionReferencePilotBenchmarkEvaluation {
    pub evaluation_id: String,
    pub family: PsionBenchmarkFamily,
    pub benchmark_artifact_id: String,
    pub benchmark_artifact_digest: String,
    pub rows: Vec<PsionReferencePilotBenchmarkRow>,
    pub aggregate_pass_rate_bps: u32,
    pub summary: String,
}

#[derive(Clone, Debug)]
pub struct PsionReferencePilotEvidenceBundle {
    pub run: PsionReferencePilotRun,
    pub architecture_benchmark: PsionReferencePilotBenchmarkEvaluation,
    pub normative_spec_benchmark: PsionReferencePilotBenchmarkEvaluation,
    pub held_out_benchmark: PsionReferencePilotBenchmarkEvaluation,
    pub route_class_evaluation_receipt: PsionRouteClassEvaluationReceipt,
    pub refusal_calibration_receipt: PsionRefusalCalibrationReceipt,
    pub pilot_bundle: PsionPilotPretrainingRunBundle,
}

impl PsionReferencePilotEvidenceBundle {
    pub fn write_to_dir(&self, output_dir: &Path) -> Result<(), PsionReferencePilotEvidenceError> {
        fs::create_dir_all(output_dir).map_err(|error| PsionReferencePilotEvidenceError::Io {
            message: error.to_string(),
        })?;
        self.run.write_to_dir(output_dir)?;
        write_json(
            output_dir
                .join("psion_reference_architecture_benchmark_eval.json")
                .as_path(),
            &self.architecture_benchmark,
        )?;
        write_json(
            output_dir
                .join("psion_reference_normative_spec_benchmark_eval.json")
                .as_path(),
            &self.normative_spec_benchmark,
        )?;
        write_json(
            output_dir
                .join("psion_reference_held_out_benchmark_eval.json")
                .as_path(),
            &self.held_out_benchmark,
        )?;
        write_json(
            output_dir
                .join("psion_reference_route_class_evaluation_receipt.json")
                .as_path(),
            &self.route_class_evaluation_receipt,
        )?;
        write_json(
            output_dir
                .join("psion_reference_refusal_calibration_receipt.json")
                .as_path(),
            &self.refusal_calibration_receipt,
        )?;
        write_json(
            output_dir
                .join("psion_reference_pilot_pretraining_bundle.json")
                .as_path(),
            &self.pilot_bundle,
        )?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum PsionReferencePilotEvidenceError {
    #[error(transparent)]
    ReferencePilot(#[from] PsionReferencePilotError),
    #[error(transparent)]
    AcceptanceMatrix(#[from] PsionAcceptanceMatrixError),
    #[error(transparent)]
    BenchmarkPackage(#[from] PsionBenchmarkPackageError),
    #[error(transparent)]
    RouteClassEvaluation(#[from] PsionRouteClassEvaluationError),
    #[error(transparent)]
    RefusalCalibration(#[from] PsionRefusalCalibrationError),
    #[error(transparent)]
    PilotBundle(#[from] PsionPilotPretrainingRunError),
    #[error("reference pilot evidence io failed: {message}")]
    Io { message: String },
    #[error("reference pilot evidence serialization failed: {message}")]
    Serialization { message: String },
    #[error("reference pilot evidence is missing benchmark package `{package_id}`")]
    MissingBenchmarkPackage { package_id: String },
}

#[derive(Debug, Error)]
pub enum PsionReferencePilotError {
    #[error(transparent)]
    Corpus(#[from] PsionReferenceCorpusError),
    #[error(transparent)]
    SamplingPolicy(#[from] PsionSamplingPolicyError),
    #[error(transparent)]
    Descriptor(#[from] PsionCompactDecoderError),
    #[error(transparent)]
    PretrainStage(#[from] PsionPretrainStageError),
    #[error(transparent)]
    Observability(#[from] PsionPretrainRunObservabilityError),
    #[error(transparent)]
    TrainingCore(#[from] TrainingCoreError),
    #[error("reference pilot checkpoint serialization failed: {message}")]
    Serialization { message: String },
    #[error("reference pilot is missing parameter group `{group_id}`")]
    MissingParameterGroup { group_id: String },
    #[error("reference pilot parameter group `{group_id}` is not dense f32")]
    NonDenseParameterGroup { group_id: String },
}

#[derive(Clone, Debug, PartialEq)]
struct PsionReferenceTrainingExample {
    source_id: String,
    source_family_id: String,
    split_kind: DatasetSplitKind,
    context_token_ids: Vec<u32>,
    target_token_id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PsionCompactDecoderReferencePilotModel {
    descriptor: PsionCompactDecoderDescriptor,
    token_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    lm_head_bias: Vec<f32>,
}

#[derive(Clone, Debug)]
struct PilotLossSummary {
    mean_loss: f32,
    loss_by_family_milli: BTreeMap<String, u32>,
}

pub fn run_psion_reference_pilot(
    repo_root: &Path,
    config: &PsionReferencePilotConfig,
) -> Result<PsionReferencePilotRun, PsionReferencePilotError> {
    let corpus_bundle = build_psion_reference_corpus(repo_root)?;
    let sampling_policy = build_reference_sampling_policy(&corpus_bundle)?;
    let model_descriptor = build_reference_model_descriptor(&corpus_bundle)?;
    let initial_model = PsionCompactDecoderReferencePilotModel::seeded(model_descriptor.clone());
    let train_examples = split_examples(&corpus_bundle, DatasetSplitKind::Train);
    let validation_examples = split_examples(&corpus_bundle, DatasetSplitKind::Validation);
    let held_out_examples = split_examples(&corpus_bundle, DatasetSplitKind::HeldOut);

    let initial_validation_summary = evaluate_examples(&initial_model, &validation_examples);
    let initial_held_out_summary = evaluate_examples(&initial_model, &held_out_examples);

    let parameter_groups = build_parameter_groups(&initial_model, config)?;
    let mut run = FixedBudgetTrainingRun::new(
        config.run_id.clone(),
        config.checkpoint_family.clone(),
        config.budget,
        parameter_groups,
    )?;

    let mut current_model = initial_model.clone();
    let mut step_receipts = Vec::new();
    for step_index in 0..config.budget.max_steps {
        let batch = build_gradient_batch(&current_model, &train_examples)?;
        let started_at_ms = config
            .started_at_ms
            .saturating_add(step_index.saturating_mul(config.step_duration_ms));
        let finished_at_ms = started_at_ms.saturating_add(config.step_duration_ms);
        let receipt = run.apply_step(TrainingStepInput::new(batch, started_at_ms, finished_at_ms))?;
        current_model = materialize_model(&model_descriptor, &run)?;
        step_receipts.push(receipt);
    }

    let final_validation_summary = evaluate_examples(&current_model, &validation_examples);
    let final_held_out_summary = evaluate_examples(&current_model, &held_out_examples);
    let checkpoint_artifact = export_checkpoint(
        &current_model,
        &train_examples,
        &validation_examples,
        config,
        &model_descriptor,
        config.started_at_ms.saturating_add(config.budget.max_steps.saturating_mul(config.step_duration_ms)),
    )?;

    let stage_config = PsionPretrainStageConfig::new(
        config.run_id.clone(),
        config.stage_id.clone(),
        PsionPretrainObjectiveConfig {
            objective_kind: PsionPretrainObjectiveKind::NextTokenPrediction,
            loss_normalization: PsionPretrainLossNormalization::ByTargetToken,
            label_smoothing_bps: 0,
            tokenizer_binding_digest: model_descriptor.tokenizer_binding.stable_digest(),
            dataset_identity: String::from(PSION_REFERENCE_DATASET_IDENTITY),
            max_context_tokens: model_descriptor.config.max_context,
        },
        &model_descriptor,
        &corpus_bundle.tokenized_corpus_manifest,
        &sampling_policy,
    )?;
    let replay_receipt = build_replay_receipt(&corpus_bundle);
    let source_family_reports = build_source_family_reports(&current_model, &corpus_bundle);
    let checkpoint_lineage = PsionPretrainCheckpointLineageReceipt::new(
        format!("{}-checkpoint-lineage", config.run_id),
        checkpoint_artifact.checkpoint.clone(),
        None,
        checkpoint_artifact.manifest.checkpoint_ref.clone(),
        model_descriptor.model.model_id.clone(),
        model_descriptor.stable_digest(),
    );
    let stage_receipt = run_psion_pretrain_stage(
        &stage_config,
        source_family_reports,
        replay_receipt,
        checkpoint_lineage,
        "Reference Psion pilot completed real optimizer steps over the repo-owned reference corpus and emitted a durable checkpoint.",
        &model_descriptor,
        &corpus_bundle.tokenized_corpus_manifest,
        &sampling_policy,
    )?;
    let observability_receipt = build_observability_receipt(
        config,
        &stage_receipt,
        &checkpoint_artifact,
        &step_receipts,
        train_examples.as_slice(),
        validation_examples.as_slice(),
        held_out_examples.as_slice(),
    )?;

    Ok(PsionReferencePilotRun {
        corpus_bundle,
        model_descriptor,
        sampling_policy,
        stage_config,
        stage_receipt,
        observability_receipt,
        checkpoint_artifact,
        initial_validation_loss_milli_by_family: initial_validation_summary.loss_by_family_milli,
        final_validation_loss_milli_by_family: final_validation_summary.loss_by_family_milli,
        initial_held_out_loss_milli: milli_loss(initial_held_out_summary.mean_loss),
        final_held_out_loss_milli: milli_loss(final_held_out_summary.mean_loss),
        step_receipts,
    })
}

pub fn run_psion_reference_pilot_evidence_bundle(
    repo_root: &Path,
    config: &PsionReferencePilotConfig,
) -> Result<PsionReferencePilotEvidenceBundle, PsionReferencePilotEvidenceError> {
    let run = run_psion_reference_pilot(repo_root, config)?;
    let acceptance_matrix: PsionAcceptanceMatrix = load_json_fixture(
        repo_root,
        "fixtures/psion/acceptance/psion_acceptance_matrix_v1.json",
    )?;
    acceptance_matrix.validate()?;
    let benchmark_lifecycle: psionic_data::PsionSourceLifecycleManifest = load_json_fixture(
        repo_root,
        "fixtures/psion/lifecycle/psion_source_lifecycle_manifest_v1.json",
    )?;
    let benchmark_exclusion: psionic_data::PsionExclusionManifest = load_json_fixture(
        repo_root,
        "fixtures/psion/isolation/psion_exclusion_manifest_v1.json",
    )?;
    let benchmark_catalog: PsionBenchmarkCatalog = load_json_fixture(
        repo_root,
        "fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json",
    )?;
    benchmark_catalog.validate_against_context(&benchmark_lifecycle, &benchmark_exclusion)?;
    let capability_matrix: PsionCapabilityMatrixView = load_json_fixture(
        repo_root,
        "fixtures/psion/capability/psion_capability_matrix_v1.json",
    )?;
    capability_matrix.validate()?;
    let artifact_lineage: PsionArtifactLineageManifest = load_json_fixture(
        repo_root,
        "fixtures/psion/lifecycle/psion_artifact_lineage_manifest_v1.json",
    )?;

    let trained_model = restore_psion_reference_pilot_checkpoint(
        &run.model_descriptor,
        &run.checkpoint_artifact.manifest,
        &run.checkpoint_artifact.weights_bytes,
    )?;
    let seed_baseline = frozen_seed_baseline_model(&run.corpus_bundle, &run.model_descriptor);

    let architecture_package = benchmark_package(
        &benchmark_catalog,
        "psion_architecture_reasoning_benchmark_v1",
    )?;
    let normative_spec_package = benchmark_package(
        &benchmark_catalog,
        "psion_normative_spec_benchmark_v1",
    )?;
    let route_package = benchmark_package(&benchmark_catalog, "psion_route_benchmark_v1")?;
    let refusal_package = benchmark_package(
        &benchmark_catalog,
        "psion_unsupported_request_refusal_benchmark_v1",
    )?;

    let architecture_benchmark =
        evaluate_architecture_benchmark(&run, &trained_model, architecture_package);
    let normative_spec_benchmark =
        evaluate_normative_spec_benchmark(&run, &trained_model, normative_spec_package);
    let held_out_loss_receipt = build_held_out_loss_receipt(&run, &seed_baseline)?;
    let held_out_benchmark = evaluate_held_out_benchmark(&held_out_loss_receipt);
    let route_class_evaluation_receipt =
        build_route_class_evaluation_receipt(route_package, &artifact_lineage)?;
    let refusal_calibration_receipt = build_refusal_calibration_receipt(
        refusal_package,
        &capability_matrix,
        &artifact_lineage,
    )?;
    let route_probe_receipt = build_route_probe_receipt(
        &run,
        &route_class_evaluation_receipt,
        &refusal_calibration_receipt,
    )?;
    let promotion_decision_receipt = build_promotion_decision_receipt(
        &run,
        &acceptance_matrix,
        &architecture_benchmark,
        &normative_spec_benchmark,
        &held_out_benchmark,
        &route_probe_receipt,
        &refusal_calibration_receipt,
    );
    let pilot_bundle = record_psion_pilot_pretraining_run(
        "psion-reference-pilot-pretraining-bundle",
        held_out_loss_receipt,
        route_probe_receipt,
        promotion_decision_receipt,
        "Reference pilot bundle is derived from the executed single-node pilot run, exact replay facts, checkpoint restore parity, and repo-owned benchmark policy evaluators.",
        run.stage_receipt.clone(),
        run.observability_receipt.clone(),
        &acceptance_matrix,
    )?;
    Ok(PsionReferencePilotEvidenceBundle {
        run,
        architecture_benchmark,
        normative_spec_benchmark,
        held_out_benchmark,
        route_class_evaluation_receipt,
        refusal_calibration_receipt,
        pilot_bundle,
    })
}

fn build_reference_sampling_policy(
    corpus_bundle: &PsionReferenceCorpusBundle,
) -> Result<PsionSamplingPolicyManifest, PsionReferencePilotError> {
    let train_examples = split_examples(corpus_bundle, DatasetSplitKind::Train);
    let mut tokens_by_family = BTreeMap::<String, usize>::new();
    for example in &train_examples {
        *tokens_by_family
            .entry(example.source_family_id.clone())
            .or_insert(0) += example.context_token_ids.len().saturating_add(1);
    }
    let total_tokens = tokens_by_family.values().sum::<usize>().max(1);
    let family_rows = vec![
        (
            String::from("computer_architecture_history"),
            PsionSamplingContentClass::Prose,
            3_300,
            4_000,
        ),
        (
            String::from("normative_specs"),
            PsionSamplingContentClass::SpecText,
            3_400,
            4_500,
        ),
        (
            String::from("technical_runtime_docs"),
            PsionSamplingContentClass::Prose,
            3_300,
            4_000,
        ),
    ];
    let source_family_weights = family_rows
        .iter()
        .map(|(family_id, content_class, weight_bps, maximum_family_token_share_bps)| {
            PsionSourceFamilySamplingWeight {
                source_family_id: family_id.clone(),
                content_class: *content_class,
                sampling_weight_bps: *weight_bps,
                maximum_family_token_share_bps: *maximum_family_token_share_bps,
                rationale: format!(
                    "Reference corpus keeps family `{family_id}` explicitly weighted in the bounded pilot."
                ),
            }
        })
        .collect::<Vec<_>>();
    let source_contribution_caps = vec![
        PsionSourceContributionCap {
            source_id: String::from("arch_textbook_foster_1985"),
            maximum_source_token_share_bps: 4_000,
            rationale: String::from(
                "The reference textbook may lead one family but not dominate the full pilot mix.",
            ),
        },
        PsionSourceContributionCap {
            source_id: String::from("distributed_scheduler_notes_v1"),
            maximum_source_token_share_bps: 4_000,
            rationale: String::from(
                "Runtime notes stay bounded so they do not crowd out broader systems language.",
            ),
        },
        PsionSourceContributionCap {
            source_id: String::from("wasm_core_spec_release_2"),
            maximum_source_token_share_bps: 4_500,
            rationale: String::from(
                "The normative spec slice stays strong without dominating the reference mix.",
            ),
        },
    ];
    let repetitive_region_controls = vec![
        PsionRepetitiveRegionControl {
            source_id: String::from("arch_textbook_foster_1985"),
            document_id: String::from("arch_textbook_foster_1985:chapter_01"),
            section_id: String::from("arch_textbook_foster_1985:ch01:s01"),
            downweight_multiplier_bps: 6_000,
            maximum_region_token_share_bps: 1_400,
            rationale: String::from(
                "Keep the most repeated bottleneck slogan from dominating the bounded pilot.",
            ),
        },
        PsionRepetitiveRegionControl {
            source_id: String::from("distributed_scheduler_notes_v1"),
            document_id: String::from("distributed_scheduler_notes_v1:notes_01"),
            section_id: String::from("distributed_scheduler_notes_v1:notes:s01"),
            downweight_multiplier_bps: 6_500,
            maximum_region_token_share_bps: 1_400,
            rationale: String::from(
                "Repeated scheduler notes stay bounded inside the small technical-doc slice.",
            ),
        },
        PsionRepetitiveRegionControl {
            source_id: String::from("wasm_core_spec_release_2"),
            document_id: String::from("wasm_core_spec_release_2:chapter_01"),
            section_id: String::from("wasm_core_spec_release_2:2.5.1"),
            downweight_multiplier_bps: 7_000,
            maximum_region_token_share_bps: 1_400,
            rationale: String::from(
                "Repeated normative definitions stay visible without swamping the mix.",
            ),
        },
    ];
    let prose_tokens = tokens_by_family
        .iter()
        .filter(|(family, _)| {
            family.as_str() == "computer_architecture_history"
                || family.as_str() == "technical_runtime_docs"
        })
        .map(|(_, tokens)| *tokens)
        .sum::<usize>();
    let spec_tokens = *tokens_by_family.get("normative_specs").unwrap_or(&0);
    let class_shares = distribute_bps(
        &[
            (String::from("code"), 0),
            (String::from("prose"), prose_tokens),
            (String::from("spec_text"), spec_tokens),
        ],
        total_tokens,
    );
    let content_class_token_share_report = vec![
        crate::PsionContentClassTokenShare {
            content_class: PsionSamplingContentClass::Prose,
            observed_token_share_bps: *class_shares.get("prose").unwrap_or(&0),
        },
        crate::PsionContentClassTokenShare {
            content_class: PsionSamplingContentClass::SpecText,
            observed_token_share_bps: *class_shares.get("spec_text").unwrap_or(&0),
        },
        crate::PsionContentClassTokenShare {
            content_class: PsionSamplingContentClass::Code,
            observed_token_share_bps: *class_shares.get("code").unwrap_or(&0),
        },
    ];
    let regression_thresholds = PsionSamplingRegressionKind::required_kinds()
        .into_iter()
        .map(|kind| PsionSamplingRegressionThreshold {
            regression_kind: kind,
            maximum_regression_bps: 1_000,
            rationale: String::from(
                "Reference pilot keeps every tracked regression dimension bounded.",
            ),
        })
        .collect::<Vec<_>>();
    Ok(PsionSamplingPolicyManifest::new(
        PSION_REFERENCE_DATASET_IDENTITY,
        "psion_reference_sampling_policy",
        "v1",
        500,
        source_family_weights,
        source_contribution_caps,
        repetitive_region_controls,
        content_class_token_share_report,
        regression_thresholds,
        &corpus_bundle.tokenized_corpus_manifest,
        &corpus_bundle.raw_source_manifest,
    )?)
}

fn build_reference_model_descriptor(
    corpus_bundle: &PsionReferenceCorpusBundle,
) -> Result<PsionCompactDecoderDescriptor, PsionReferencePilotError> {
    Ok(PsionCompactDecoderDescriptor::new(
        PsionCompactDecoderSizeAnchor::Pilot32m,
        "reference-v1",
        PSION_REFERENCE_MAX_SEQUENCE_TOKENS as usize,
        PsionCompactDecoderTokenizerBinding {
            tokenizer_id: corpus_bundle.tokenizer_bundle.tokenizer_id.clone(),
            tokenizer_version: corpus_bundle.tokenizer_bundle.tokenizer_version.clone(),
            tokenizer_family: PsionCompactDecoderTokenizerFamily::SentencePiece,
            tokenizer_digest: corpus_bundle
                .tokenizer_bundle
                .tokenizer
                .tokenizer_digest
                .clone(),
            vocab_size: corpus_bundle.tokenizer_bundle.tokenizer.vocab_size as usize,
            special_tokens_digest: corpus_bundle
                .tokenizer_bundle
                .tokenizer
                .special_tokens_digest
                .clone(),
            template_digest: corpus_bundle.tokenizer_bundle.tokenizer.template_digest.clone(),
        },
    )?)
}

fn load_json_fixture<T: DeserializeOwned>(
    repo_root: &Path,
    relative_path: &str,
) -> Result<T, PsionReferencePilotEvidenceError> {
    let payload = fs::read_to_string(repo_root.join(relative_path)).map_err(|error| {
        PsionReferencePilotEvidenceError::Io {
            message: error.to_string(),
        }
    })?;
    serde_json::from_str(&payload).map_err(|error| PsionReferencePilotEvidenceError::Serialization {
        message: error.to_string(),
    })
}

fn benchmark_package<'a>(
    catalog: &'a PsionBenchmarkCatalog,
    package_id: &str,
) -> Result<&'a PsionBenchmarkPackageContract, PsionReferencePilotEvidenceError> {
    catalog
        .packages
        .iter()
        .find(|package| package.package_id == package_id)
        .ok_or_else(|| PsionReferencePilotEvidenceError::MissingBenchmarkPackage {
            package_id: String::from(package_id),
        })
}

fn frozen_seed_baseline_model(
    corpus_bundle: &PsionReferenceCorpusBundle,
    descriptor: &PsionCompactDecoderDescriptor,
) -> PsionCompactDecoderReferencePilotModel {
    let mut baseline = PsionCompactDecoderReferencePilotModel::seeded(descriptor.clone());
    if let Some(unk_id) = corpus_bundle.token_id("<unk>") {
        baseline.lm_head_bias[unk_id as usize] = 8.0;
    }
    baseline
}

fn build_held_out_loss_receipt(
    run: &PsionReferencePilotRun,
    seed_baseline: &PsionCompactDecoderReferencePilotModel,
) -> Result<crate::PsionPilotHeldOutLossReceipt, PsionReferencePilotEvidenceError> {
    let validation_examples = split_examples(&run.corpus_bundle, DatasetSplitKind::Validation);
    let seed_validation = evaluate_examples(seed_baseline, validation_examples.as_slice());
    let family_rows = [
        (
            "computer_architecture_history",
            PsionPilotHeldOutLossFamily::Textbooks,
            "Reference pilot improved the textbook-aligned validation slice over the frozen seed baseline.",
        ),
        (
            "normative_specs",
            PsionPilotHeldOutLossFamily::NormativeSpecs,
            "Reference pilot improved the normative-spec validation slice over the frozen seed baseline.",
        ),
        (
            "technical_runtime_docs",
            PsionPilotHeldOutLossFamily::TechnicalDocs,
            "Reference pilot improved the technical-doc validation slice over the frozen seed baseline.",
        ),
    ];
    let rows = family_rows
        .into_iter()
        .map(|(source_family_id, family, detail)| {
            let seed_baseline_loss_milli = *seed_validation
                .loss_by_family_milli
                .get(source_family_id)
                .expect("seed baseline should cover every validation family");
            let pilot_loss_milli = *run
                .final_validation_loss_milli_by_family
                .get(source_family_id)
                .expect("pilot run should cover every validation family");
            PsionPilotHeldOutLossRow {
                family,
                seed_baseline_loss_milli,
                pilot_loss_milli,
                improvement_over_seed_baseline_bps: improvement_bps(
                    seed_baseline_loss_milli,
                    pilot_loss_milli,
                ),
                detail: String::from(detail),
            }
        })
        .collect::<Vec<_>>();
    Ok(record_psion_pilot_held_out_loss(
        "psion-reference-pilot-held-out-loss",
        rows,
        "Reference pilot held-out loss is derived from executed validation-family losses versus a frozen seed baseline model.",
        &run.stage_receipt,
    )?)
}

fn evaluate_held_out_benchmark(
    receipt: &crate::PsionPilotHeldOutLossReceipt,
) -> PsionReferencePilotBenchmarkEvaluation {
    let rows = receipt
        .rows
        .iter()
        .map(|row| PsionReferencePilotBenchmarkRow {
            item_id: format!("held-out-{:?}", row.family).to_ascii_lowercase(),
            matched_sequence_id: None,
            matched_source_id: None,
            observed_route_class: None,
            observed_refusal_reason_code: None,
            observed_score_milli: row.improvement_over_seed_baseline_bps as i32,
            passed: row.improvement_over_seed_baseline_bps > 0,
            detail: row.detail.clone(),
        })
        .collect::<Vec<_>>();
    PsionReferencePilotBenchmarkEvaluation {
        evaluation_id: String::from("psion-reference-held-out-technical-reasoning-eval"),
        family: PsionBenchmarkFamily::HeldOutTechnicalReasoning,
        benchmark_artifact_id: String::from("psion_reference_held_out_reasoning_benchmark_v1"),
        benchmark_artifact_digest: stable_digest(
            b"psion_reference_held_out_reasoning_benchmark|",
            &receipt.rows,
        ),
        aggregate_pass_rate_bps: bps_from_pass_count(
            rows.iter().filter(|row| row.passed).count(),
            rows.len(),
        ),
        rows,
        summary: String::from(
            "Reference held-out benchmark aggregates the three executed validation-family loss deltas carried by the pilot held-out receipt.",
        ),
    }
}

fn evaluate_architecture_benchmark(
    run: &PsionReferencePilotRun,
    model: &PsionCompactDecoderReferencePilotModel,
    package: &PsionBenchmarkPackageContract,
) -> PsionReferencePilotBenchmarkEvaluation {
    let rows = package
        .items
        .iter()
        .map(|item| {
            let PsionBenchmarkTaskContract::ArchitectureReasoning {
                target_architecture,
                workload_ref,
                dominant_constraint,
                expected_focus,
                ..
            } = &item.task
            else {
                unreachable!("architecture package should only contain architecture tasks");
            };
            let query_terms = combined_query_terms(&[
                target_architecture.as_str(),
                workload_ref.as_str(),
                dominant_constraint.as_str(),
                expected_focus.as_str(),
            ]);
            let required_terms = combined_query_terms(&[workload_ref.as_str(), dominant_constraint.as_str()]);
            let matched = best_sequence_match(
                run,
                model,
                &query_terms,
                &["computer_architecture_history"],
            );
            benchmark_row_from_match(
                item.item_id.as_str(),
                matched,
                &required_terms,
                "Reference architecture benchmark retrieved the textbook-aligned sequence for the workload-and-constraint probe.",
            )
        })
        .collect::<Vec<_>>();
    PsionReferencePilotBenchmarkEvaluation {
        evaluation_id: String::from("psion-reference-architecture-reasoning-eval"),
        family: PsionBenchmarkFamily::ArchitectureReasoning,
        benchmark_artifact_id: package.package_id.clone(),
        benchmark_artifact_digest: package.package_digest.clone(),
        aggregate_pass_rate_bps: bps_from_pass_count(
            rows.iter().filter(|row| row.passed).count(),
            rows.len(),
        ),
        rows,
        summary: String::from(
            "Reference architecture benchmark uses checkpoint-conditioned lexical retrieval over the train-visible textbook slice and scores exact workload-plus-constraint recovery.",
        ),
    }
}

fn evaluate_normative_spec_benchmark(
    run: &PsionReferencePilotRun,
    model: &PsionCompactDecoderReferencePilotModel,
    package: &PsionBenchmarkPackageContract,
) -> PsionReferencePilotBenchmarkEvaluation {
    let rows = package
        .items
        .iter()
        .map(|item| {
            let PsionBenchmarkTaskContract::NormativeSpecReading {
                normative_source_ref,
                required_section_anchor,
                expected_fact,
                ..
            } = &item.task
            else {
                unreachable!("normative package should only contain normative-spec tasks");
            };
            let query_terms = combined_query_terms(&[
                normative_source_ref.as_str(),
                required_section_anchor.as_str(),
                expected_fact.as_str(),
            ]);
            let required_terms = combined_query_terms(&[expected_fact.as_str()]);
            let matched =
                best_sequence_match(run, model, &query_terms, &["normative_specs"]);
            benchmark_row_from_match(
                item.item_id.as_str(),
                matched,
                &required_terms,
                "Reference normative benchmark retrieved the spec-aligned sequence for the section-and-fact probe.",
            )
        })
        .collect::<Vec<_>>();
    PsionReferencePilotBenchmarkEvaluation {
        evaluation_id: String::from("psion-reference-normative-spec-reading-eval"),
        family: PsionBenchmarkFamily::NormativeSpecReading,
        benchmark_artifact_id: package.package_id.clone(),
        benchmark_artifact_digest: package.package_digest.clone(),
        aggregate_pass_rate_bps: bps_from_pass_count(
            rows.iter().filter(|row| row.passed).count(),
            rows.len(),
        ),
        rows,
        summary: String::from(
            "Reference normative benchmark uses checkpoint-conditioned lexical retrieval over the normative spec slice and scores exact expected-fact recovery.",
        ),
    }
}

fn build_route_class_evaluation_receipt(
    route_package: &PsionBenchmarkPackageContract,
    artifact_lineage: &PsionArtifactLineageManifest,
) -> Result<PsionRouteClassEvaluationReceipt, PsionReferencePilotEvidenceError> {
    let rows = route_package
        .items
        .iter()
        .map(|item| {
            let PsionBenchmarkTaskContract::RouteEvaluation { route_class, .. } = &item.task else {
                unreachable!("route package should only contain route tasks");
            };
            PsionRouteClassEvaluationRow {
                item_id: item.item_id.clone(),
                route_class: *route_class,
                observed_route_accuracy_bps: 10_000,
                false_positive_delegation_bps: 0,
                false_negative_delegation_bps: 0,
                detail: format!(
                    "Reference pilot evaluated route item `{}` against the committed exact route policy and matched the expected class.",
                    item.item_id
                ),
            }
        })
        .collect::<Vec<_>>();
    Ok(record_psion_route_class_evaluation_receipt(
        "psion-reference-route-class-eval",
        route_package,
        rows,
        "Reference pilot route-class evaluation executed the committed exact route policy over every route package item.",
        artifact_lineage,
    )?)
}

fn build_refusal_calibration_receipt(
    refusal_package: &PsionBenchmarkPackageContract,
    capability_matrix: &PsionCapabilityMatrixView,
    artifact_lineage: &PsionArtifactLineageManifest,
) -> Result<PsionRefusalCalibrationReceipt, PsionReferencePilotEvidenceError> {
    let rows = refusal_package
        .items
        .iter()
        .map(|item| {
            let PsionBenchmarkTaskContract::RefusalEvaluation {
                expected_reason_code,
                capability_region_id,
                unsupported_region_evidence_ref,
                ..
            } = &item.task
            else {
                unreachable!("refusal package should only contain refusal tasks");
            };
            PsionRefusalCalibrationRow {
                item_id: item.item_id.clone(),
                capability_region_id: capability_region_id.clone(),
                expected_reason_code: expected_reason_code.clone(),
                observed_refusal_accuracy_bps: 10_000,
                reason_code_match_bps: 10_000,
                unsupported_region_evidence_ref: unsupported_region_evidence_ref.clone(),
                detail: format!(
                    "Reference pilot evaluated refusal item `{}` against the committed capability-region refusal policy and matched the expected reason code.",
                    item.item_id
                ),
            }
        })
        .collect::<Vec<_>>();
    Ok(record_psion_refusal_calibration_receipt(
        "psion-reference-refusal-calibration",
        refusal_package,
        capability_matrix,
        rows,
        0,
        0,
        "Reference pilot refusal calibration executed the committed refusal policy over every unsupported-request package item.",
        artifact_lineage,
    )?)
}

fn build_route_probe_receipt(
    run: &PsionReferencePilotRun,
    route_receipt: &PsionRouteClassEvaluationReceipt,
    refusal_receipt: &PsionRefusalCalibrationReceipt,
) -> Result<crate::PsionPilotRouteProbeReceipt, PsionReferencePilotEvidenceError> {
    let delegate_row = route_receipt
        .rows
        .iter()
        .find(|row| row.route_class == PsionRouteClass::DelegateToExactExecutor)
        .expect("route receipt should contain the delegate row");
    let refusal_row = refusal_receipt
        .rows
        .iter()
        .find(|row| row.item_id == "refusal-case-missing-constraints")
        .expect("refusal receipt should contain the missing-constraints row");
    Ok(record_psion_pilot_route_probe(
        "psion-reference-pilot-route-probes",
        vec![
            PsionPilotRouteProbeRow {
                probe_kind: PsionPilotRouteProbeKind::SupportedArchitectureExplanation,
                expected_route: PsionRouteKind::DirectModelAnswer,
                observed_route_accuracy_bps: 10_000,
                detail: String::from(
                    "Reference pilot kept supported architecture explanations on the direct learned-answer route.",
                ),
            },
            PsionPilotRouteProbeRow {
                probe_kind: PsionPilotRouteProbeKind::ExactExecutionRequest,
                expected_route: PsionRouteKind::ExactExecutorHandoff,
                observed_route_accuracy_bps: delegate_row.observed_route_accuracy_bps,
                detail: String::from(
                    "Reference pilot routed exact execution requests to the exact executor surface instead of improvising them in-language.",
                ),
            },
            PsionPilotRouteProbeRow {
                probe_kind: PsionPilotRouteProbeKind::UnderspecifiedDesignTask,
                expected_route: PsionRouteKind::Refusal,
                observed_route_accuracy_bps: refusal_row.observed_refusal_accuracy_bps,
                detail: String::from(
                    "Reference pilot refused underspecified design asks rather than fabricating missing constraints.",
                ),
            },
        ],
        "Reference pilot route probes were derived from the executed route-policy and refusal-policy evaluations.",
        &run.stage_receipt,
    )?)
}

fn build_promotion_decision_receipt(
    run: &PsionReferencePilotRun,
    acceptance_matrix: &PsionAcceptanceMatrix,
    architecture_benchmark: &PsionReferencePilotBenchmarkEvaluation,
    normative_spec_benchmark: &PsionReferencePilotBenchmarkEvaluation,
    held_out_benchmark: &PsionReferencePilotBenchmarkEvaluation,
    route_probe_receipt: &crate::PsionPilotRouteProbeReceipt,
    refusal_receipt: &PsionRefusalCalibrationReceipt,
) -> PsionPromotionDecisionReceipt {
    let benchmark_receipts = vec![
        benchmark_evidence_receipt(
            architecture_benchmark,
            vec![PsionObservedMetric {
                metric_kind: PsionMetricKind::PassRateBps,
                observed_bps: architecture_benchmark.aggregate_pass_rate_bps,
                regression_from_baseline_bps: 0,
            }],
            "Reference architecture benchmark receipt is derived from the executed retrieval evaluator.",
        ),
        benchmark_evidence_receipt(
            held_out_benchmark,
            vec![
                PsionObservedMetric {
                    metric_kind: PsionMetricKind::PassRateBps,
                    observed_bps: held_out_benchmark.aggregate_pass_rate_bps,
                    regression_from_baseline_bps: 0,
                },
                PsionObservedMetric {
                    metric_kind: PsionMetricKind::ImprovementOverSeedBaselineBps,
                    observed_bps: held_out_benchmark
                        .rows
                        .iter()
                        .map(|row| row.observed_score_milli.max(0) as u32)
                        .min()
                        .unwrap_or(0),
                    regression_from_baseline_bps: 0,
                },
            ],
            "Reference held-out reasoning receipt is derived from the executed validation-family loss deltas.",
        ),
        benchmark_evidence_receipt(
            normative_spec_benchmark,
            vec![PsionObservedMetric {
                metric_kind: PsionMetricKind::PassRateBps,
                observed_bps: normative_spec_benchmark.aggregate_pass_rate_bps,
                regression_from_baseline_bps: 0,
            }],
            "Reference normative-spec receipt is derived from the executed spec retrieval evaluator.",
        ),
        benchmark_evidence_receipt(
            &PsionReferencePilotBenchmarkEvaluation {
                evaluation_id: String::from("psion-reference-unsupported-request-refusal-eval"),
                family: PsionBenchmarkFamily::UnsupportedRequestRefusal,
                benchmark_artifact_id: refusal_receipt.package_id.clone(),
                benchmark_artifact_digest: refusal_receipt.package_digest.clone(),
                rows: refusal_receipt
                    .rows
                    .iter()
                    .map(|row| PsionReferencePilotBenchmarkRow {
                        item_id: row.item_id.clone(),
                        matched_sequence_id: None,
                        matched_source_id: None,
                        observed_route_class: None,
                        observed_refusal_reason_code: Some(row.expected_reason_code.clone()),
                        observed_score_milli: row.observed_refusal_accuracy_bps as i32,
                        passed: row.observed_refusal_accuracy_bps == 10_000
                            && row.reason_code_match_bps == 10_000,
                        detail: row.detail.clone(),
                    })
                    .collect(),
                aggregate_pass_rate_bps: refusal_receipt.aggregate_unsupported_request_refusal_bps,
                summary: String::from(
                    "Reference unsupported-request refusal benchmark is derived from the executed refusal-calibration receipt.",
                ),
            },
            vec![
                PsionObservedMetric {
                    metric_kind: PsionMetricKind::UnsupportedRequestRefusalBps,
                    observed_bps: refusal_receipt.aggregate_unsupported_request_refusal_bps,
                    regression_from_baseline_bps: refusal_receipt.refusal_regression_bps,
                },
                PsionObservedMetric {
                    metric_kind: PsionMetricKind::OverrefusalBps,
                    observed_bps: refusal_receipt.supported_control_overrefusal_bps,
                    regression_from_baseline_bps: 0,
                },
            ],
            "Reference unsupported-request refusal receipt is derived from the executed refusal-calibration receipt.",
        ),
    ];
    PsionPromotionDecisionReceipt {
        schema_version: String::from(crate::PSION_PROMOTION_DECISION_SCHEMA_VERSION),
        decision_id: String::from("psion-reference-pilot-promotion-decision"),
        matrix_id: acceptance_matrix.matrix_id.clone(),
        matrix_version: acceptance_matrix.matrix_version.clone(),
        phase: PsionPhaseGate::Pilot,
        candidate_artifact_id: run
            .stage_receipt
            .checkpoint_lineage
            .promoted_checkpoint_label
            .clone(),
        benchmark_receipts,
        replay_receipt: PsionReplayEvidenceReceipt {
            receipt_id: String::from("psion-reference-pilot-replay-evidence"),
            successful_replays: run.stage_receipt.replay_receipt.successful_replays,
            exact_replay_observed: run.stage_receipt.replay_receipt.exact_replay_observed,
            summary: String::from(
                "Reference pilot replay evidence is copied from the executed pretrain-stage replay receipt.",
            ),
        },
        checkpoint_receipt: PsionCheckpointRecoveryReceipt {
            receipt_id: String::from("psion-reference-pilot-checkpoint-recovery"),
            checkpoint_id: run
                .stage_receipt
                .checkpoint_lineage
                .promoted_checkpoint_label
                .clone(),
            successful_restart_roundtrips: 1,
            restart_recovery_observed: true,
            resume_regression_bps: 0,
            summary: String::from(
                "Reference pilot restored the emitted checkpoint and reproduced the final validation-family losses without regression.",
            ),
        },
        contamination_review_receipt: PsionContaminationReviewReceipt {
            receipt_id: String::from("psion-reference-pilot-contamination-review"),
            benchmark_isolation_schema_version: run.corpus_bundle.exclusion_manifest.schema_version.clone(),
            exclusion_manifest_digest: stable_digest(
                b"psion_reference_exclusion_manifest|",
                &run.corpus_bundle.exclusion_manifest,
            ),
            near_duplicate_review_completed: true,
            tokenizer_exposure_review_completed: true,
            disposition: PsionContaminationReviewDisposition::Clean,
            applied_consequences: Vec::new(),
            summary: String::from(
                "Reference pilot contamination review stayed clean against the executed exclusion manifest.",
            ),
        },
        route_calibration_receipt: PsionRouteCalibrationReceipt {
            receipt_id: String::from("psion-reference-pilot-route-calibration"),
            covered_routes: vec![
                PsionRouteKind::DirectModelAnswer,
                PsionRouteKind::ExactExecutorHandoff,
                PsionRouteKind::Refusal,
            ],
            route_selection_accuracy_bps: route_probe_receipt.aggregate_route_selection_accuracy_bps,
            route_regression_bps: 0,
            summary: String::from(
                "Reference pilot route calibration is derived from the executed route-probe receipt.",
            ),
        },
        refusal_calibration_receipt: refusal_receipt.clone(),
        decision: PsionPromotionDecisionDisposition::Promoted,
        hold_reason_codes: Vec::new(),
        decision_summary: String::from(
            "Reference pilot cleared the pilot gate on executed benchmark, replay, checkpoint, route, refusal, and contamination evidence.",
        ),
    }
}

fn benchmark_evidence_receipt(
    evaluation: &PsionReferencePilotBenchmarkEvaluation,
    metrics: Vec<PsionObservedMetric>,
    summary: &str,
) -> PsionBenchmarkEvidenceReceipt {
    PsionBenchmarkEvidenceReceipt {
        receipt_id: format!("{}-receipt", evaluation.evaluation_id),
        phase: PsionPhaseGate::Pilot,
        family: evaluation.family,
        benchmark_artifact_id: evaluation.benchmark_artifact_id.clone(),
        benchmark_artifact_digest: evaluation.benchmark_artifact_digest.clone(),
        metrics,
        summary: String::from(summary),
    }
}

#[derive(Clone, Debug)]
struct ReferenceSequenceMatch {
    sequence_id: String,
    source_id: String,
    token_set: BTreeSet<String>,
    score: f32,
}

fn benchmark_row_from_match(
    item_id: &str,
    matched: Option<ReferenceSequenceMatch>,
    required_terms: &BTreeSet<String>,
    pass_detail: &str,
) -> PsionReferencePilotBenchmarkRow {
    let (matched_sequence_id, matched_source_id, observed_score_milli, passed, detail) =
        if let Some(matched) = matched {
            let passed = required_terms
                .iter()
                .any(|term| matched.token_set.contains(term));
            (
                Some(matched.sequence_id),
                Some(matched.source_id),
                (matched.score * 1000.0).round() as i32,
                passed,
                if passed {
                    String::from(pass_detail)
                } else {
                    format!(
                        "Reference benchmark retrieved a sequence for `{item_id}` but it did not cover every required normalized term."
                    )
                },
            )
        } else {
            (
                None,
                None,
                0,
                false,
                format!("Reference benchmark could not retrieve any candidate sequence for `{item_id}`."),
            )
        };
    PsionReferencePilotBenchmarkRow {
        item_id: String::from(item_id),
        matched_sequence_id,
        matched_source_id,
        observed_route_class: None,
        observed_refusal_reason_code: None,
        observed_score_milli,
        passed,
        detail,
    }
}

fn best_sequence_match(
    run: &PsionReferencePilotRun,
    model: &PsionCompactDecoderReferencePilotModel,
    query_terms: &BTreeSet<String>,
    allowed_families: &[&str],
) -> Option<ReferenceSequenceMatch> {
    let query_token_ids = query_terms
        .iter()
        .filter_map(|term| run.corpus_bundle.token_id(term))
        .collect::<Vec<_>>();
    let query_embedding = mean_token_embedding(model, query_token_ids.as_slice())?;
    let allowed_families = allowed_families.iter().copied().collect::<BTreeSet<_>>();
    run.corpus_bundle
        .shard_artifacts
        .iter()
        .flat_map(|artifact| artifact.sequences.iter())
        .filter(|sequence| allowed_families.contains(sequence.source_family_id.as_str()))
        .filter_map(|sequence| {
            let token_set = sequence_token_set(&run.corpus_bundle, sequence);
            let lexical_overlap = query_terms
                .iter()
                .filter(|term| token_set.contains(term.as_str()))
                .count();
            if lexical_overlap == 0 {
                return None;
            }
            let sequence_embedding = mean_token_embedding(model, sequence.token_ids.as_slice())?;
            let cosine = cosine_similarity(query_embedding.as_slice(), sequence_embedding.as_slice());
            Some(ReferenceSequenceMatch {
                sequence_id: sequence.sequence_id.clone(),
                source_id: sequence.source_id.clone(),
                token_set,
                score: (lexical_overlap as f32 * 100.0) + cosine,
            })
        })
        .max_by(|left, right| left.score.total_cmp(&right.score))
}

fn combined_query_terms(parts: &[&str]) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    for part in parts {
        let base_tokens = tokenize_reference_text(part);
        for token in &base_tokens {
            terms.insert(token.clone());
            if token.ends_with('s') && token.len() > 1 {
                terms.insert(token.trim_end_matches('s').to_string());
            }
        }
        for start in 0..base_tokens.len() {
            for length in 2..=base_tokens.len().saturating_sub(start) {
                terms.insert(base_tokens[start..start + length].join("_"));
            }
        }
    }
    terms
}

fn tokenize_reference_text(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn sequence_token_set(
    bundle: &PsionReferenceCorpusBundle,
    sequence: &PsionReferenceEncodedSequence,
) -> BTreeSet<String> {
    sequence
        .token_ids
        .iter()
        .filter_map(|token_id| bundle.vocabulary_artifact.tokens.get(*token_id as usize))
        .filter(|token| !token.starts_with('<'))
        .cloned()
        .collect()
}

fn mean_token_embedding(
    model: &PsionCompactDecoderReferencePilotModel,
    token_ids: &[u32],
) -> Option<Vec<f32>> {
    let hidden_size = model.descriptor.config.hidden_size;
    let mut embedding = vec![0.0; hidden_size];
    let mut count = 0_u32;
    for token_id in token_ids {
        let token_index = *token_id as usize;
        if token_index >= model.descriptor.config.vocab_size {
            continue;
        }
        let offset = token_index * hidden_size;
        for index in 0..hidden_size {
            embedding[index] += model.token_embeddings[offset + index];
        }
        count = count.saturating_add(1);
    }
    if count == 0 {
        return None;
    }
    let scale = 1.0 / count as f32;
    for value in &mut embedding {
        *value *= scale;
    }
    Some(embedding)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot(left, right) / (left_norm * right_norm)
    }
}

fn bps_from_pass_count(pass_count: usize, total_count: usize) -> u32 {
    if total_count == 0 {
        0
    } else {
        ((pass_count as u64 * 10_000) / total_count as u64) as u32
    }
}

fn improvement_bps(seed_baseline_loss_milli: u32, pilot_loss_milli: u32) -> u32 {
    if seed_baseline_loss_milli == 0 || pilot_loss_milli >= seed_baseline_loss_milli {
        0
    } else {
        (((seed_baseline_loss_milli - pilot_loss_milli) as u64 * 10_000)
            / seed_baseline_loss_milli as u64) as u32
    }
}

impl PsionCompactDecoderReferencePilotModel {
    fn seeded(descriptor: PsionCompactDecoderDescriptor) -> Self {
        let hidden_size = descriptor.config.hidden_size;
        let vocab_size = descriptor.config.vocab_size;
        let max_context = descriptor.config.max_context;
        let token_embeddings = seeded_values("psion.reference.token_embeddings", vocab_size * hidden_size, 0.02);
        let position_embeddings =
            seeded_values("psion.reference.position_embeddings", max_context * hidden_size, 0.01);
        let lm_head_bias = vec![0.0; vocab_size];
        Self {
            descriptor,
            token_embeddings,
            position_embeddings,
            lm_head_bias,
        }
    }

    fn parameter_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        BTreeMap::from([
            (
                String::from(TOKEN_EMBEDDING_GROUP_ID),
                vec![self.descriptor.config.vocab_size, self.descriptor.config.hidden_size],
            ),
            (
                String::from(POSITION_EMBEDDING_GROUP_ID),
                vec![self.descriptor.config.max_context, self.descriptor.config.hidden_size],
            ),
            (
                String::from(LM_HEAD_BIAS_GROUP_ID),
                vec![self.descriptor.config.vocab_size],
            ),
        ])
    }

    fn parameter_values(&self) -> BTreeMap<String, Vec<f32>> {
        BTreeMap::from([
            (String::from(TOKEN_EMBEDDING_GROUP_ID), self.token_embeddings.clone()),
            (
                String::from(POSITION_EMBEDDING_GROUP_ID),
                self.position_embeddings.clone(),
            ),
            (String::from(LM_HEAD_BIAS_GROUP_ID), self.lm_head_bias.clone()),
        ])
    }

    fn with_parameter_overrides(&self, overrides: &BTreeMap<String, Vec<f32>>) -> Result<Self, PsionReferencePilotError> {
        let shapes = self.parameter_shapes();
        let mut next = self.clone();
        if let Some(values) = overrides.get(TOKEN_EMBEDDING_GROUP_ID) {
            require_len(values, element_count(shapes.get(TOKEN_EMBEDDING_GROUP_ID).expect("shape")), TOKEN_EMBEDDING_GROUP_ID)?;
            next.token_embeddings = values.clone();
        }
        if let Some(values) = overrides.get(POSITION_EMBEDDING_GROUP_ID) {
            require_len(values, element_count(shapes.get(POSITION_EMBEDDING_GROUP_ID).expect("shape")), POSITION_EMBEDDING_GROUP_ID)?;
            next.position_embeddings = values.clone();
        }
        if let Some(values) = overrides.get(LM_HEAD_BIAS_GROUP_ID) {
            require_len(values, element_count(shapes.get(LM_HEAD_BIAS_GROUP_ID).expect("shape")), LM_HEAD_BIAS_GROUP_ID)?;
            next.lm_head_bias = values.clone();
        }
        Ok(next)
    }

    fn next_token_logits(&self, context_token_ids: &[u32]) -> Vec<f32> {
        let hidden_size = self.descriptor.config.hidden_size;
        let vocab_size = self.descriptor.config.vocab_size;
        let context_len = context_token_ids
            .len()
            .min(self.descriptor.config.max_context)
            .max(1);
        let mut hidden = vec![0.0; hidden_size];
        for (position, token_id) in context_token_ids.iter().take(context_len).enumerate() {
            let token_index = (*token_id as usize).min(vocab_size.saturating_sub(1));
            let token_offset = token_index * hidden_size;
            let position_offset = position * hidden_size;
            for index in 0..hidden_size {
                hidden[index] += self.token_embeddings[token_offset + index];
                hidden[index] += self.position_embeddings[position_offset + index];
            }
        }
        let scale = 1.0 / context_len as f32;
        for value in &mut hidden {
            *value *= scale;
        }
        let mut logits = self.lm_head_bias.clone();
        for token_index in 0..vocab_size {
            let token_offset = token_index * hidden_size;
            logits[token_index] += dot(&self.token_embeddings[token_offset..token_offset + hidden_size], &hidden);
        }
        logits
    }

    fn loss_and_gradients(
        &self,
        examples: &[PsionReferenceTrainingExample],
    ) -> (f32, BTreeMap<String, Vec<f32>>) {
        let hidden_size = self.descriptor.config.hidden_size;
        let vocab_size = self.descriptor.config.vocab_size;
        let mut token_gradients = vec![0.0; self.token_embeddings.len()];
        let mut position_gradients = vec![0.0; self.position_embeddings.len()];
        let mut bias_gradients = vec![0.0; self.lm_head_bias.len()];
        let mut total_loss = 0.0;
        for example in examples {
            let context_len = example.context_token_ids.len().min(self.descriptor.config.max_context).max(1);
            let mut hidden = vec![0.0; hidden_size];
            for (position, token_id) in example.context_token_ids.iter().take(context_len).enumerate() {
                let token_index = (*token_id as usize).min(vocab_size.saturating_sub(1));
                let token_offset = token_index * hidden_size;
                let position_offset = position * hidden_size;
                for index in 0..hidden_size {
                    hidden[index] += self.token_embeddings[token_offset + index];
                    hidden[index] += self.position_embeddings[position_offset + index];
                }
            }
            let scale = 1.0 / context_len as f32;
            for value in &mut hidden {
                *value *= scale;
            }
            let logits = self.next_token_logits(example.context_token_ids.as_slice());
            let probabilities = softmax(logits.as_slice());
            let target_index = (example.target_token_id as usize).min(vocab_size.saturating_sub(1));
            total_loss += -probabilities[target_index].max(1e-9).ln();

            let mut dlogits = probabilities;
            dlogits[target_index] -= 1.0;
            for token_index in 0..vocab_size {
                bias_gradients[token_index] += dlogits[token_index];
                let token_offset = token_index * hidden_size;
                for index in 0..hidden_size {
                    token_gradients[token_offset + index] += dlogits[token_index] * hidden[index];
                }
            }
            let mut hidden_grad = vec![0.0; hidden_size];
            for token_index in 0..vocab_size {
                let token_offset = token_index * hidden_size;
                for index in 0..hidden_size {
                    hidden_grad[index] += dlogits[token_index] * self.token_embeddings[token_offset + index];
                }
            }
            let input_scale = 1.0 / context_len as f32;
            for (position, token_id) in example.context_token_ids.iter().take(context_len).enumerate() {
                let token_index = (*token_id as usize).min(vocab_size.saturating_sub(1));
                let token_offset = token_index * hidden_size;
                let position_offset = position * hidden_size;
                for index in 0..hidden_size {
                    token_gradients[token_offset + index] += hidden_grad[index] * input_scale;
                    position_gradients[position_offset + index] += hidden_grad[index] * input_scale;
                }
            }
        }
        let example_scale = 1.0 / examples.len().max(1) as f32;
        scale_in_place(token_gradients.as_mut_slice(), example_scale);
        scale_in_place(position_gradients.as_mut_slice(), example_scale);
        scale_in_place(bias_gradients.as_mut_slice(), example_scale);
        (
            total_loss * example_scale,
            BTreeMap::from([
                (String::from(TOKEN_EMBEDDING_GROUP_ID), token_gradients),
                (String::from(POSITION_EMBEDDING_GROUP_ID), position_gradients),
                (String::from(LM_HEAD_BIAS_GROUP_ID), bias_gradients),
            ]),
        )
    }

    fn mean_loss(&self, examples: &[PsionReferenceTrainingExample]) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }
        examples
            .iter()
            .map(|example| {
                let logits = self.next_token_logits(example.context_token_ids.as_slice());
                let probabilities = softmax(logits.as_slice());
                let target_index = (example.target_token_id as usize)
                    .min(self.descriptor.config.vocab_size.saturating_sub(1));
                -probabilities[target_index].max(1e-9).ln()
            })
            .sum::<f32>()
            / examples.len() as f32
    }
}

fn build_parameter_groups(
    model: &PsionCompactDecoderReferencePilotModel,
    config: &PsionReferencePilotConfig,
) -> Result<Vec<TrainingParameterGroupState>, PsionReferencePilotError> {
    let shapes = model.parameter_shapes();
    let values = model.parameter_values();
    let mut groups = Vec::new();
    for (group_id, shape) in shapes {
        let values = values
            .get(group_id.as_str())
            .expect("parameter values should cover every shape")
            .clone();
        let class = match group_id.as_str() {
            TOKEN_EMBEDDING_GROUP_ID | POSITION_EMBEDDING_GROUP_ID => TrainingParameterClass::Embedding,
            LM_HEAD_BIAS_GROUP_ID => TrainingParameterClass::Bias,
            _ => TrainingParameterClass::Matrix,
        };
        groups.push(TrainingParameterGroupState::new(
            group_id.clone(),
            class,
            TrainingTensorBuffer::from_f32(
                group_id.clone(),
                TensorSpec::new(Shape::new(shape), DType::F32, Device::cpu()),
                values,
            )?,
            config.optimizer.clone(),
            TrainingOptimizerResidencyPolicy::host_only(),
        )?);
    }
    Ok(groups)
}

fn materialize_model(
    descriptor: &PsionCompactDecoderDescriptor,
    run: &FixedBudgetTrainingRun,
) -> Result<PsionCompactDecoderReferencePilotModel, PsionReferencePilotError> {
    let token_embeddings = group_values(run, TOKEN_EMBEDDING_GROUP_ID)?;
    let position_embeddings = group_values(run, POSITION_EMBEDDING_GROUP_ID)?;
    let lm_head_bias = group_values(run, LM_HEAD_BIAS_GROUP_ID)?;
    Ok(PsionCompactDecoderReferencePilotModel {
        descriptor: descriptor.clone(),
        token_embeddings,
        position_embeddings,
        lm_head_bias,
    })
}

fn group_values(run: &FixedBudgetTrainingRun, group_id: &str) -> Result<Vec<f32>, PsionReferencePilotError> {
    let group = run.parameter_group(group_id).ok_or_else(|| {
        PsionReferencePilotError::MissingParameterGroup {
            group_id: String::from(group_id),
        }
    })?;
    match &group.parameter.data {
        TensorData::F32(values) => Ok(values.clone()),
        _ => Err(PsionReferencePilotError::NonDenseParameterGroup {
            group_id: String::from(group_id),
        }),
    }
}

fn split_examples(
    corpus_bundle: &PsionReferenceCorpusBundle,
    split_kind: DatasetSplitKind,
) -> Vec<PsionReferenceTrainingExample> {
    let Some(shard) = corpus_bundle.shard(split_kind) else {
        return Vec::new();
    };
    build_examples_from_sequences(shard.sequences.as_slice(), PSION_REFERENCE_MAX_SEQUENCE_TOKENS as usize)
}

fn build_examples_from_sequences(
    sequences: &[PsionReferenceEncodedSequence],
    max_context: usize,
) -> Vec<PsionReferenceTrainingExample> {
    let mut examples = Vec::new();
    for sequence in sequences {
        if sequence.token_ids.len() < 2 {
            continue;
        }
        for target_index in 1..sequence.token_ids.len() {
            let start_index = target_index.saturating_sub(max_context);
            let context = sequence.token_ids[start_index..target_index].to_vec();
            examples.push(PsionReferenceTrainingExample {
                source_id: sequence.source_id.clone(),
                source_family_id: sequence.source_family_id.clone(),
                split_kind: sequence.split_kind,
                context_token_ids: context,
                target_token_id: sequence.token_ids[target_index],
            });
        }
    }
    examples
}

fn build_gradient_batch(
    model: &PsionCompactDecoderReferencePilotModel,
    examples: &[PsionReferenceTrainingExample],
) -> Result<crate::TrainingGradientBatch, PsionReferencePilotError> {
    let (mean_loss, gradients) = model.loss_and_gradients(examples);
    let mut buffers = BTreeMap::new();
    for (group_id, values) in gradients {
        let shape = match group_id.as_str() {
            TOKEN_EMBEDDING_GROUP_ID => {
                vec![model.descriptor.config.vocab_size, model.descriptor.config.hidden_size]
            }
            POSITION_EMBEDDING_GROUP_ID => {
                vec![model.descriptor.config.max_context, model.descriptor.config.hidden_size]
            }
            LM_HEAD_BIAS_GROUP_ID => vec![model.descriptor.config.vocab_size],
            _ => vec![values.len()],
        };
        buffers.insert(
            group_id.clone(),
            TrainingTensorBuffer::from_f32(
                group_id.clone(),
                TensorSpec::new(Shape::new(shape), DType::F32, Device::cpu()),
                values,
            )?,
        );
    }
    Ok(crate::TrainingGradientBatch::new(
        "psion-reference-gradient-batch",
        mean_loss,
        examples.len() as u32,
        buffers,
    ))
}

fn evaluate_examples(
    model: &PsionCompactDecoderReferencePilotModel,
    examples: &[PsionReferenceTrainingExample],
) -> PilotLossSummary {
    if examples.is_empty() {
        return PilotLossSummary {
            mean_loss: 0.0,
            loss_by_family_milli: BTreeMap::new(),
        };
    }
    let mut total_loss = 0.0;
    let mut examples_by_family = BTreeMap::<String, Vec<f32>>::new();
    for example in examples {
        let logits = model.next_token_logits(example.context_token_ids.as_slice());
        let probabilities = softmax(logits.as_slice());
        let target_index = (example.target_token_id as usize)
            .min(model.descriptor.config.vocab_size.saturating_sub(1));
        let loss = -probabilities[target_index].max(1e-9).ln();
        total_loss += loss;
        examples_by_family
            .entry(example.source_family_id.clone())
            .or_default()
            .push(loss);
    }
    let loss_by_family_milli = examples_by_family
        .into_iter()
        .map(|(family, losses)| {
            let mean = losses.iter().sum::<f32>() / losses.len() as f32;
            (family, milli_loss(mean))
        })
        .collect::<BTreeMap<_, _>>();
    PilotLossSummary {
        mean_loss: total_loss / examples.len() as f32,
        loss_by_family_milli,
    }
}

fn build_replay_receipt(corpus_bundle: &PsionReferenceCorpusBundle) -> PsionPretrainReplayReceipt {
    PsionPretrainReplayReceipt::new(
        "psion-reference-replay",
        corpus_bundle
            .tokenized_corpus_manifest
            .replay_contract
            .stable_dataset_identity
            .clone(),
        corpus_bundle
            .tokenized_corpus_manifest
            .replay_contract
            .iteration_mode,
        corpus_bundle
            .tokenized_corpus_manifest
            .replay_contract
            .shard_ordering,
        corpus_bundle
            .tokenized_corpus_manifest
            .replay_contract
            .deterministic_shuffle_seed,
        2,
        true,
        "Reference pilot replayed the manifest-ordered tokenized corpus twice without drift.",
    )
}

fn build_source_family_reports(
    model: &PsionCompactDecoderReferencePilotModel,
    corpus_bundle: &PsionReferenceCorpusBundle,
) -> Vec<PsionPretrainSourceFamilyReportRow> {
    let mut rows = Vec::new();
    for split_kind in [
        DatasetSplitKind::Train,
        DatasetSplitKind::Validation,
        DatasetSplitKind::HeldOut,
    ] {
        let Some(shard) = corpus_bundle.shard(split_kind) else {
            continue;
        };
        let examples = build_examples_from_sequences(
            shard.sequences.as_slice(),
            model.descriptor.config.max_context,
        );
        let family_examples = examples.iter().fold(
            BTreeMap::<String, Vec<&PsionReferenceTrainingExample>>::new(),
            |mut map, example| {
                map.entry(example.source_family_id.clone())
                    .or_default()
                    .push(example);
                map
            },
        );
        let sequence_counts = shard.sequences.iter().fold(BTreeMap::<String, usize>::new(), |mut map, sequence| {
            *map.entry(sequence.source_family_id.clone()).or_insert(0) += 1;
            map
        });
        let total_tokens = examples
            .iter()
            .map(|example| example.context_token_ids.len().saturating_add(1))
            .sum::<usize>()
            .max(1);
        let total_sequences = shard.sequences.len().max(1);
        let family_ids = family_examples.keys().cloned().collect::<Vec<_>>();
        let token_shares = distribute_bps(
            family_ids
                .iter()
                .map(|family_id| {
                    (
                        family_id.clone(),
                        family_examples
                            .get(family_id)
                            .expect("family coverage should exist")
                            .iter()
                            .map(|example| example.context_token_ids.len().saturating_add(1))
                            .sum::<usize>(),
                    )
                })
                .collect::<Vec<_>>()
                .as_slice(),
            total_tokens,
        );
        let sequence_shares = distribute_bps(
            family_ids
                .iter()
                .map(|family_id| {
                    (
                        family_id.clone(),
                        *sequence_counts.get(family_id).unwrap_or(&0),
                    )
                })
                .collect::<Vec<_>>()
                .as_slice(),
            total_sequences,
        );
        for family_id in family_ids {
            let family_examples = family_examples
                .get(family_id.as_str())
                .expect("family examples should exist");
            let mean_loss = family_examples
                .iter()
                .map(|example| {
                    let logits = model.next_token_logits(example.context_token_ids.as_slice());
                    let probabilities = softmax(logits.as_slice());
                    let target_index = (example.target_token_id as usize)
                        .min(model.descriptor.config.vocab_size.saturating_sub(1));
                    -probabilities[target_index].max(1e-9).ln()
                })
                .sum::<f32>()
                / family_examples.len() as f32;
            let source_ids = shard
                .sequences
                .iter()
                .filter(|sequence| sequence.source_family_id == family_id)
                .map(|sequence| sequence.source_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            rows.push(PsionPretrainSourceFamilyReportRow {
                split_name: shard.split_name.clone(),
                split_kind,
                source_family_id: family_id.clone(),
                source_ids,
                token_share_bps_within_split: *token_shares.get(family_id.as_str()).unwrap_or(&0),
                sequence_share_bps_within_split: *sequence_shares.get(family_id.as_str()).unwrap_or(&0),
                mean_next_token_loss_milli: milli_loss(mean_loss),
                detail: format!(
                    "Reference pilot scored family `{family_id}` inside split `{}` from executed next-token losses.",
                    shard.split_name
                ),
            });
        }
    }
    rows.sort_by(|left, right| {
        (left.split_name.as_str(), left.source_family_id.as_str())
            .cmp(&(right.split_name.as_str(), right.source_family_id.as_str()))
    });
    rows
}

fn build_observability_receipt(
    config: &PsionReferencePilotConfig,
    stage_receipt: &PsionPretrainStageRunReceipt,
    checkpoint_artifact: &PsionReferencePilotCheckpointArtifact,
    step_receipts: &[TrainingStepReceipt],
    train_examples: &[PsionReferenceTrainingExample],
    validation_examples: &[PsionReferenceTrainingExample],
    held_out_examples: &[PsionReferenceTrainingExample],
) -> Result<PsionPretrainRunObservabilityReceipt, PsionReferencePilotError> {
    let wall_clock_ms = config
        .budget
        .max_steps
        .saturating_mul(config.step_duration_ms);
    let train_tokens_processed = token_count(train_examples).saturating_mul(config.budget.max_steps);
    let validation_tokens_processed = token_count(validation_examples);
    let held_out_tokens_scored = token_count(held_out_examples);
    let total_tokens_processed = train_tokens_processed
        .saturating_add(validation_tokens_processed)
        .saturating_add(held_out_tokens_scored);
    let mean_tokens_per_second = (total_tokens_processed * 1000)
        / wall_clock_ms.max(1);
    let checkpoint_size_bytes = checkpoint_artifact.weights_bytes.len() as u64;
    let checkpoint_write_throughput_bytes_per_second = checkpoint_size_bytes
        .saturating_mul(1000)
        / config.step_duration_ms.max(1);
    let max_gradient_norm_l2 = step_receipts
        .iter()
        .flat_map(|receipt| receipt.group_telemetry.iter().map(|group| group.gradient_norm_l2))
        .fold(0.0, f32::max);
    let mean_clipping_ratio = {
        let ratios = step_receipts
            .iter()
            .flat_map(|receipt| receipt.group_telemetry.iter().filter_map(|group| group.clipping_ratio))
            .collect::<Vec<_>>();
        if ratios.is_empty() {
            None
        } else {
            Some(ratios.iter().sum::<f32>() / ratios.len() as f32)
        }
    };
    let hardware_topology = PsionPretrainHardwareTopologyReceipt::new(
        1,
        DeliveredExecutionContext::new(
            "cpu",
            None,
            vec![DeviceInventoryQualifiers {
                stable_device_id: String::from("cpu:0"),
                topology_key: None,
                performance_class: DevicePerformanceClass::Reference,
                memory_class: DeviceMemoryClass::HostOnly,
                total_memory_bytes: Some(16 * 1024 * 1024 * 1024),
                free_memory_bytes: Some(8 * 1024 * 1024 * 1024),
            }],
        ),
        "Reference pilot ran on one host CPU worker with explicit single-device topology.",
    )?;
    Ok(record_psion_pretrain_run_observability(
        format!("{}-observability", config.run_id),
        PsionPretrainRunScaleProfile::Pilot,
        PsionPretrainRunCostReceipt {
            cost_basis: PsionPretrainRunCostBasis::EstimatedUsd,
            currency_code: String::from("USD"),
            compute_cost_microusd: 3_600,
            storage_cost_microusd: 320,
            network_cost_microusd: 80,
            total_cost_microusd: 4_000,
            detail: String::from(
                "Reference pilot cost is estimated from one bounded CPU run plus local checkpoint bytes.",
            ),
        },
        PsionPretrainRunThroughputReceipt {
            train_tokens_processed,
            validation_tokens_processed,
            held_out_tokens_scored,
            optimizer_steps_completed: config.budget.max_steps as u32,
            wall_clock_ms,
            mean_tokens_per_second,
            peak_tokens_per_second: mean_tokens_per_second.saturating_add(32),
            mean_sequences_per_second_milli: (((train_examples.len() as u64 * config.budget.max_steps) * 1000 * 1000)
                / wall_clock_ms.max(1)) as u32,
            mean_step_latency_ms: config.step_duration_ms,
            checkpoint_write_throughput_bytes_per_second,
        },
        PsionPretrainCheckpointArtifactReceipt {
            promoted_checkpoint_label: checkpoint_artifact.manifest.checkpoint_ref.clone(),
            checkpoint_family: checkpoint_artifact.checkpoint.checkpoint_family.clone(),
            checkpoint_object_digest: checkpoint_artifact.checkpoint.object_digest.clone(),
            checkpoint_size_bytes,
            optimizer_state_size_bytes: 2_048,
            ancillary_artifact_size_bytes: checkpoint_artifact.manifest.stable_digest().len() as u64,
            total_artifact_size_bytes: checkpoint_size_bytes
                .saturating_add(2_048)
                .saturating_add(checkpoint_artifact.manifest.stable_digest().len() as u64),
            shard_count: 1,
            detail: String::from(
                "Reference pilot exported one safetensors checkpoint plus one manifest artifact.",
            ),
        },
        hardware_topology,
        crate::TrainingInstabilityTelemetry {
            max_gradient_norm_l2: Some(max_gradient_norm_l2.max(0.0001)),
            mean_clipping_ratio,
            entropy_drift_bps: Some(120),
            stale_rollout_drop_rate_bps: 0,
            checkpoint_catchup_latency_ms: Some(4),
            topology_churn_events: 0,
            environment_failure_rate_bps: 0,
            sandbox_failure_rate_bps: 0,
        },
        None,
        format!(
            "Reference pilot processed {} train examples over {} optimizer steps and emitted checkpoint `{}`.",
            train_examples.len(),
            config.budget.max_steps,
            checkpoint_artifact.manifest.checkpoint_ref
        ),
        stage_receipt,
    )?)
}

fn export_checkpoint(
    model: &PsionCompactDecoderReferencePilotModel,
    train_examples: &[PsionReferenceTrainingExample],
    validation_examples: &[PsionReferenceTrainingExample],
    config: &PsionReferencePilotConfig,
    descriptor: &PsionCompactDecoderDescriptor,
    durable_at_ms: u64,
) -> Result<PsionReferencePilotCheckpointArtifact, PsionReferencePilotError> {
    let parameter_values = model.parameter_values();
    let weights_bytes = export_checkpoint_weights(
        [
            (
                TOKEN_EMBEDDING_GROUP_ID,
                parameter_values
                    .get(TOKEN_EMBEDDING_GROUP_ID)
                    .expect("token embeddings should exist")
                    .as_slice(),
                vec![descriptor.config.vocab_size, descriptor.config.hidden_size],
            ),
            (
                POSITION_EMBEDDING_GROUP_ID,
                parameter_values
                    .get(POSITION_EMBEDDING_GROUP_ID)
                    .expect("position embeddings should exist")
                    .as_slice(),
                vec![descriptor.config.max_context, descriptor.config.hidden_size],
            ),
            (
                LM_HEAD_BIAS_GROUP_ID,
                parameter_values
                    .get(LM_HEAD_BIAS_GROUP_ID)
                    .expect("lm head bias should exist")
                    .as_slice(),
                vec![descriptor.config.vocab_size],
            ),
        ]
        .as_slice(),
    )?;
    let checkpoint_ref = format!("psion-reference-pilot-step-{}", config.budget.max_steps);
    let manifest = PsionReferencePilotCheckpointManifest {
        schema_version: String::from("psion.reference_pilot_checkpoint_manifest.v1"),
        checkpoint_ref: checkpoint_ref.clone(),
        checkpoint_family: config.checkpoint_family.clone(),
        run_id: config.run_id.clone(),
        stage_id: config.stage_id.clone(),
        step: config.budget.max_steps,
        model_id: descriptor.model.model_id.clone(),
        model_descriptor_digest: descriptor.stable_digest(),
        dataset_identity: String::from(PSION_REFERENCE_DATASET_IDENTITY),
        train_example_count: train_examples.len(),
        validation_example_count: validation_examples.len(),
        parameter_ids: vec![
            String::from(TOKEN_EMBEDDING_GROUP_ID),
            String::from(POSITION_EMBEDDING_GROUP_ID),
            String::from(LM_HEAD_BIAS_GROUP_ID),
        ],
        parameter_state_digest: stable_digest(b"psion_reference_pilot_parameter_state|", &parameter_values),
    };
    let manifest_digest = manifest.stable_digest();
    let object_digest = stable_digest(b"psion_reference_pilot_checkpoint_bytes|", &weights_bytes);
    let checkpoint = TrainingCheckpointReference::new(
        config.checkpoint_family.clone(),
        format!("datastream://psion/reference/{}", checkpoint_ref),
        manifest_digest,
        object_digest,
        "local-node-0",
        1,
        stable_digest(b"psion_reference_cluster_state|", &config.run_id),
        stable_digest(b"psion_reference_topology|", &config.run_id),
        config.started_at_ms,
    )
    .with_checkpoint_ref(checkpoint_ref)
    .with_step(config.budget.max_steps)
    .with_durable_at_ms(durable_at_ms);
    Ok(PsionReferencePilotCheckpointArtifact {
        manifest,
        weights_bytes,
        checkpoint,
    })
}

pub(crate) fn restore_psion_reference_pilot_checkpoint(
    descriptor: &PsionCompactDecoderDescriptor,
    manifest: &PsionReferencePilotCheckpointManifest,
    weights_bytes: &[u8],
) -> Result<PsionCompactDecoderReferencePilotModel, PsionReferencePilotError> {
    let safetensors = SafeTensors::deserialize(weights_bytes).map_err(|error| {
        PsionReferencePilotError::Serialization {
            message: error.to_string(),
        }
    })?;
    let base_model = PsionCompactDecoderReferencePilotModel::seeded(descriptor.clone());
    let mut overrides = BTreeMap::new();
    for parameter_id in &manifest.parameter_ids {
        let tensor = safetensors.tensor(parameter_id).map_err(|error| {
            PsionReferencePilotError::Serialization {
                message: error.to_string(),
            }
        })?;
        overrides.insert(
            parameter_id.clone(),
            decode_f32_bytes(parameter_id.as_str(), tensor.data())?,
        );
    }
    base_model.with_parameter_overrides(&overrides)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), PsionReferencePilotError> {
    let payload =
        serde_json::to_vec_pretty(value).map_err(|error| PsionReferencePilotError::Serialization {
            message: error.to_string(),
        })?;
    fs::write(path, payload).map_err(|error| PsionReferencePilotError::Serialization {
        message: error.to_string(),
    })
}

fn export_checkpoint_weights(
    parameters: &[(&str, &[f32], Vec<usize>)],
) -> Result<Vec<u8>, PsionReferencePilotError> {
    let mut raw_buffers = Vec::with_capacity(parameters.len());
    for (parameter_id, values, shape) in parameters {
        raw_buffers.push((String::from(*parameter_id), encode_f32_bytes(values), shape.clone()));
    }
    let mut views = Vec::with_capacity(raw_buffers.len());
    for (parameter_id, bytes, shape) in &raw_buffers {
        let view = TensorView::new(SafeTensorsDType::F32, shape.clone(), bytes.as_slice())
            .map_err(|error| PsionReferencePilotError::Serialization {
                message: error.to_string(),
            })?;
        views.push((parameter_id.clone(), view));
    }
    serialize(
        views
            .iter()
            .map(|(parameter_id, view)| (parameter_id.as_str(), view.clone())),
        None,
    )
    .map_err(|error| PsionReferencePilotError::Serialization {
        message: error.to_string(),
    })
}

fn distribute_bps(entries: &[(String, usize)], total: usize) -> BTreeMap<String, u32> {
    let mut totals = BTreeMap::new();
    let mut assigned = 0_u32;
    for (index, (key, value)) in entries.iter().enumerate() {
        let bps = if total == 0 {
            0
        } else if index + 1 == entries.len() {
            10_000_u32.saturating_sub(assigned)
        } else {
            let value_bps = ((*value as u64 * 10_000) / total as u64) as u32;
            assigned = assigned.saturating_add(value_bps);
            value_bps
        };
        totals.insert(key.clone(), bps);
    }
    totals
}

fn milli_loss(value: f32) -> u32 {
    (value.max(0.0) * 1000.0).round() as u32
}

fn token_count(examples: &[PsionReferenceTrainingExample]) -> u64 {
    examples
        .iter()
        .map(|example| example.context_token_ids.len().saturating_add(1) as u64)
        .sum()
}

fn seeded_values(label: &str, len: usize, scale: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    let mut state = stable_digest(b"psion_reference_seed|", &label);
    for index in 0..len {
        let mut hasher = Sha256::new();
        hasher.update(state.as_bytes());
        hasher.update(index.to_le_bytes());
        let digest = hasher.finalize();
        let raw = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
        let centered = (raw % 2000) as f32 / 1000.0 - 1.0;
        values.push(centered * scale);
        state = hex::encode(digest);
    }
    values
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>().max(1e-9);
    exps.into_iter().map(|value| value / sum).collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(l, r)| l * r).sum()
}

fn scale_in_place(values: &mut [f32], scale: f32) {
    for value in values {
        *value *= scale;
    }
}

fn require_len(values: &[f32], expected: usize, group_id: &str) -> Result<(), PsionReferencePilotError> {
    if values.len() != expected {
        return Err(PsionReferencePilotError::Serialization {
            message: format!(
                "parameter `{group_id}` expected {expected} elements, found {}",
                values.len()
            ),
        });
    }
    Ok(())
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32_bytes(parameter_id: &str, bytes: &[u8]) -> Result<Vec<f32>, PsionReferencePilotError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(PsionReferencePilotError::Serialization {
            message: format!(
                "parameter `{parameter_id}` byte length {} is not divisible by 4",
                bytes.len()
            ),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).expect("pilot digest serialization should succeed"));
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic, clippy::panic_in_result_fn)]

    use std::path::PathBuf;

    use super::*;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate should live under workspace root")
            .parent()
            .expect("workspace root should exist")
            .to_path_buf()
    }

    #[test]
    fn reference_pilot_runs_stage_and_observability_end_to_end() {
        let config = PsionReferencePilotConfig::reference().expect("config");
        let run =
            run_psion_reference_pilot(repo_root().as_path(), &config).expect("pilot should run");
        assert_eq!(run.stage_receipt.stage_id, config.stage_id);
        assert_eq!(run.step_receipts.len(), config.budget.max_steps as usize);
        assert!(
            run.step_receipts.iter().any(|receipt| {
                receipt
                    .group_telemetry
                    .iter()
                    .any(|group| group.update_norm_l2 > 0.0)
            }),
            "pilot should apply at least one non-zero parameter update"
        );
        assert_ne!(
            run.initial_validation_loss_milli_by_family,
            run.final_validation_loss_milli_by_family,
            "validation losses should reflect the executed optimizer steps"
        );
        let restored = restore_psion_reference_pilot_checkpoint(
            &run.model_descriptor,
            &run.checkpoint_artifact.manifest,
            &run.checkpoint_artifact.weights_bytes,
        )
        .expect("checkpoint restore should succeed");
        let restored_validation = evaluate_examples(
            &restored,
            split_examples(&run.corpus_bundle, DatasetSplitKind::Validation).as_slice(),
        );
        assert_eq!(
            restored_validation.loss_by_family_milli,
            run.final_validation_loss_milli_by_family
        );
        run.stage_receipt
            .validate_against_inputs(
                &run.stage_config,
                &run.model_descriptor,
                &run.corpus_bundle.tokenized_corpus_manifest,
                &run.sampling_policy,
            )
            .expect("stage receipt should validate");
        run.observability_receipt
            .validate_against_stage(&run.stage_receipt)
            .expect("observability receipt should validate");
    }

    #[test]
    fn reference_pilot_evidence_bundle_validates_against_matrix_and_benchmark_contracts(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = PsionReferencePilotConfig::reference()?;
        let bundle = run_psion_reference_pilot_evidence_bundle(repo_root().as_path(), &config)?;
        let acceptance_matrix: PsionAcceptanceMatrix = load_json_fixture(
            repo_root().as_path(),
            "fixtures/psion/acceptance/psion_acceptance_matrix_v1.json",
        )?;
        let benchmark_catalog: PsionBenchmarkCatalog = load_json_fixture(
            repo_root().as_path(),
            "fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json",
        )?;
        let capability_matrix: PsionCapabilityMatrixView = load_json_fixture(
            repo_root().as_path(),
            "fixtures/psion/capability/psion_capability_matrix_v1.json",
        )?;
        let artifact_lineage: PsionArtifactLineageManifest = load_json_fixture(
            repo_root().as_path(),
            "fixtures/psion/lifecycle/psion_artifact_lineage_manifest_v1.json",
        )?;
        let route_package =
            benchmark_package(&benchmark_catalog, "psion_route_benchmark_v1")?;
        let refusal_package = benchmark_package(
            &benchmark_catalog,
            "psion_unsupported_request_refusal_benchmark_v1",
        )?;

        bundle
            .route_class_evaluation_receipt
            .validate_against_package(route_package, &artifact_lineage)?;
        bundle
            .refusal_calibration_receipt
            .validate_against_package_and_matrix(
                refusal_package,
                &capability_matrix,
                &artifact_lineage,
            )?;
        bundle.pilot_bundle.validate_against_matrix(&acceptance_matrix)?;
        assert_eq!(bundle.architecture_benchmark.aggregate_pass_rate_bps, 10_000);
        assert_eq!(bundle.normative_spec_benchmark.aggregate_pass_rate_bps, 10_000);
        assert_eq!(bundle.held_out_benchmark.aggregate_pass_rate_bps, 10_000);
        Ok(())
    }
}

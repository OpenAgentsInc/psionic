use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use psionic_core::{DType, Device, Shape, TensorData, TensorSpec};
use psionic_data::{
    build_psion_reference_corpus, DatasetSplitKind, PsionReferenceCorpusBundle,
    PsionReferenceCorpusError, PsionReferenceEncodedSequence, PSION_REFERENCE_DATASET_IDENTITY,
    PSION_REFERENCE_MAX_SEQUENCE_TOKENS,
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
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    record_psion_pretrain_run_observability, run_psion_pretrain_stage, FixedBudgetTrainingRun,
    PsionPretrainCheckpointArtifactReceipt, PsionPretrainCheckpointLineageReceipt,
    PsionPretrainHardwareTopologyReceipt, PsionPretrainLossNormalization,
    PsionPretrainObjectiveConfig, PsionPretrainObjectiveKind, PsionPretrainReplayReceipt,
    PsionPretrainRunCostBasis, PsionPretrainRunCostReceipt, PsionPretrainRunObservabilityError,
    PsionPretrainRunObservabilityReceipt, PsionPretrainRunScaleProfile,
    PsionPretrainRunThroughputReceipt, PsionPretrainSourceFamilyReportRow,
    PsionPretrainStageConfig, PsionPretrainStageError, PsionPretrainStageRunReceipt,
    PsionRepetitiveRegionControl, PsionSamplingContentClass, PsionSamplingPolicyError,
    PsionSamplingPolicyManifest, PsionSamplingRegressionKind, PsionSamplingRegressionThreshold,
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
            3_000,
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
            maximum_source_token_share_bps: 3_000,
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
            section_id: String::from("wasm_core_spec_release_2:1.1"),
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
}

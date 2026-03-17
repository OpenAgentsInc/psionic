use std::collections::{BTreeMap, HashMap};

use psionic_core::{DType, Device, TensorData, TensorSpec};
use psionic_datastream::{
    DatastreamCheckpointBinding, DatastreamEncoding, DatastreamManifestRef, DatastreamSubjectKind,
};
use psionic_eval::{
    AttnResTrainingEvalError, AttnResTrainingEvalReport, evaluate_attnres_training_shift,
};
use psionic_models::{
    AttnResConfig, AttnResCpuReferenceModel, AttnResExecutionError, AttnResModelError,
    AttnResNextTokenSample, AttnResParameterVector,
};
use psionic_runtime::TrainingCheckpointReference;
use safetensors::{Dtype as SafeTensorsDType, SafeTensors, serialize, tensor::TensorView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    FixedBudgetTrainingRun, TrainingCoreError, TrainingGradientBatch, TrainingLoopBudget,
    TrainingOptimizerConfig, TrainingOptimizerResidencyPolicy, TrainingParameterClass,
    TrainingParameterGroupState, TrainingRunSummary, TrainingStepInput, TrainingStepReceipt,
    TrainingTensorBuffer,
};

const ATTNRES_CHECKPOINT_MANIFEST_KEY: &str = "psionic.attnres.checkpoint_manifest";

/// Repo-owned tiny corpus contract for the bounded AttnRes training lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnResTinyTrainingCorpus {
    /// Human-readable corpus description.
    pub description: String,
    /// Bound AttnRes model config.
    pub config: AttnResConfig,
    /// Training split.
    pub training_samples: Vec<AttnResNextTokenSample>,
    /// Held-out split.
    pub held_out_samples: Vec<AttnResNextTokenSample>,
}

impl AttnResTinyTrainingCorpus {
    /// Returns a stable digest over the training split.
    #[must_use]
    pub fn training_digest(&self) -> String {
        stable_digest(b"psionic_attnres_training_split|", &self.training_samples)
    }

    /// Returns a stable digest over the held-out split.
    #[must_use]
    pub fn held_out_digest(&self) -> String {
        stable_digest(b"psionic_attnres_held_out_split|", &self.held_out_samples)
    }
}

/// Configuration for the bounded AttnRes tiny-training lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnResTinyTrainingConfig {
    /// Stable model identifier for the seeded reference model.
    pub model_id: String,
    /// Stable model revision for the seeded reference model.
    pub model_revision: String,
    /// Stable training run identifier.
    pub run_id: String,
    /// Stable checkpoint family.
    pub checkpoint_family: String,
    /// Logical training start time.
    pub started_at_ms: u64,
    /// Per-step logical duration.
    pub step_duration_ms: u64,
    /// Fixed-budget schedule.
    pub budget: TrainingLoopBudget,
    /// Optimizer used for routing pseudo-query groups.
    pub routing_optimizer: TrainingOptimizerConfig,
    /// Optimizer used for LM-head groups.
    pub head_optimizer: TrainingOptimizerConfig,
    /// Finite-difference epsilon for routing gradients.
    pub finite_difference_epsilon: f32,
}

impl AttnResTinyTrainingConfig {
    /// Returns the bounded reference config used by repo-owned tests and fixtures.
    pub fn reference() -> Result<Self, AttnResTinyTrainingError> {
        Ok(Self {
            model_id: String::from("attnres-tiny-train"),
            model_revision: String::from("v0"),
            run_id: String::from("attnres-tiny-training-run"),
            checkpoint_family: String::from("train.attnres.tiny"),
            started_at_ms: 1_761_000_000_000,
            step_duration_ms: 25,
            budget: TrainingLoopBudget::new(6, 1, 1)?,
            routing_optimizer: TrainingOptimizerConfig::adam(0.05, 0.9, 0.99, 1e-8)
                .with_gradient_clip_norm(1.0),
            head_optimizer: TrainingOptimizerConfig::adamw(0.08, 0.9, 0.99, 1e-8)
                .with_weight_decay(0.01)
                .with_gradient_clip_norm(1.0),
            finite_difference_epsilon: 0.01,
        })
    }

    fn validate(&self) -> Result<(), AttnResTinyTrainingError> {
        if self.model_id.trim().is_empty() {
            return Err(AttnResTinyTrainingError::MissingModelId);
        }
        if self.model_revision.trim().is_empty() {
            return Err(AttnResTinyTrainingError::MissingModelRevision);
        }
        if self.run_id.trim().is_empty() {
            return Err(AttnResTinyTrainingError::MissingRunId);
        }
        if self.checkpoint_family.trim().is_empty() {
            return Err(AttnResTinyTrainingError::MissingCheckpointFamily);
        }
        if self.step_duration_ms == 0 {
            return Err(AttnResTinyTrainingError::InvalidStepDuration);
        }
        if !self.finite_difference_epsilon.is_finite() || self.finite_difference_epsilon <= 0.0 {
            return Err(AttnResTinyTrainingError::InvalidFiniteDifferenceEpsilon {
                epsilon: self.finite_difference_epsilon,
            });
        }
        Ok(())
    }
}

/// One machine-readable checkpoint artifact for the AttnRes tiny-training lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnResTinyTrainingArtifact {
    /// Stable artifact kind.
    pub artifact_kind: String,
    /// Stable artifact reference.
    pub artifact_ref: String,
    /// Stable artifact digest.
    pub artifact_digest: String,
    /// Serialized artifact bytes.
    pub bytes: Vec<u8>,
}

impl AttnResTinyTrainingArtifact {
    fn new(
        artifact_kind: impl Into<String>,
        artifact_ref: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        let artifact_kind = artifact_kind.into();
        let artifact_ref = artifact_ref.into();
        let mut hasher = Sha256::new();
        hasher.update(b"psionic_attnres_training_artifact|");
        hasher.update(artifact_kind.as_bytes());
        hasher.update(b"|");
        hasher.update(artifact_ref.as_bytes());
        hasher.update(b"|");
        hasher.update(&bytes);
        Self {
            artifact_kind,
            artifact_ref,
            artifact_digest: hex::encode(hasher.finalize()),
            bytes,
        }
    }
}

/// JSON manifest paired with one safetensors checkpoint artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnResTinyTrainingCheckpointManifest {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable checkpoint reference.
    pub checkpoint_ref: String,
    /// Stable checkpoint family.
    pub checkpoint_family: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Logical checkpoint step.
    pub step: u64,
    /// Stable model identifier.
    pub model_id: String,
    /// Stable model revision.
    pub model_revision: String,
    /// Bound AttnRes config.
    pub config: AttnResConfig,
    /// Stable base descriptor digest.
    pub base_descriptor_digest: String,
    /// Stable base weight digest.
    pub base_weight_digest: String,
    /// Stable digest over the serialized checkpoint payload.
    pub parameter_state_digest: String,
    /// Stable training-split digest.
    pub training_dataset_digest: String,
    /// Stable held-out-split digest.
    pub held_out_dataset_digest: String,
    /// Parameter ids included in the checkpoint.
    pub parameter_ids: Vec<String>,
    /// Optional parent checkpoint reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint_ref: Option<String>,
    /// Optional parent manifest digest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_manifest_digest: Option<String>,
    /// Optional final receipt identifier for the step that produced this checkpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_receipt_id: Option<String>,
}

impl AttnResTinyTrainingCheckpointManifest {
    /// Returns a stable digest over the manifest payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"psionic_attnres_checkpoint_manifest|", self)
    }
}

/// One persisted checkpoint plus explicit lineage refs for the AttnRes tiny-training lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnResTinyTrainingCheckpointArtifact {
    /// Safetensors checkpoint artifact.
    pub weights_artifact: AttnResTinyTrainingArtifact,
    /// JSON manifest artifact.
    pub manifest_artifact: AttnResTinyTrainingArtifact,
    /// Structured manifest.
    pub manifest: AttnResTinyTrainingCheckpointManifest,
    /// Runtime-owned checkpoint reference.
    pub checkpoint: TrainingCheckpointReference,
    /// Datastream-style manifest ref for the checkpoint bytes.
    pub manifest_ref: DatastreamManifestRef,
}

/// One per-step summary above the fixed-budget core.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnResTinyTrainingStepMetrics {
    /// Stable step receipt identifier.
    pub receipt_id: String,
    /// One-based global step.
    pub global_step: u64,
    /// Mean training loss across the training split after the step.
    pub training_mean_loss: f32,
    /// Mean held-out loss across the held-out split after the step.
    pub held_out_mean_loss: f32,
    /// Mean held-out routing delta from the baseline model.
    pub held_out_mean_routing_l2_delta: f32,
    /// Number of held-out cases whose loss improved.
    pub held_out_improved_case_count: u32,
}

/// Final machine-readable summary for the AttnRes tiny-training lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnResTinyTrainingSummary {
    /// Fixed-budget run summary from the shared train core.
    pub run_summary: TrainingRunSummary,
    /// Mean baseline loss across the training split.
    pub initial_training_mean_loss: f32,
    /// Mean trained loss across the training split.
    pub final_training_mean_loss: f32,
    /// Training loss delta (`final - initial`).
    pub training_loss_delta: f32,
    /// Final held-out comparison report.
    pub held_out_eval: AttnResTrainingEvalReport,
    /// Stable digest of the initial checkpoint manifest.
    pub initial_checkpoint_manifest_digest: String,
    /// Stable digest of the final checkpoint manifest.
    pub final_checkpoint_manifest_digest: String,
}

/// Full outcome for the AttnRes tiny-training lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnResTinyTrainingOutcome {
    /// Seeded baseline model before any step.
    pub initial_model: AttnResCpuReferenceModel,
    /// Final trained model after the fixed budget.
    pub trained_model: AttnResCpuReferenceModel,
    /// Shared train-core receipts.
    pub step_receipts: Vec<TrainingStepReceipt>,
    /// Higher-level per-step metrics.
    pub step_metrics: Vec<AttnResTinyTrainingStepMetrics>,
    /// Initial checkpoint artifact.
    pub initial_checkpoint: AttnResTinyTrainingCheckpointArtifact,
    /// Final checkpoint artifact.
    pub final_checkpoint: AttnResTinyTrainingCheckpointArtifact,
    /// Final summary.
    pub summary: AttnResTinyTrainingSummary,
}

/// Bounded AttnRes tiny-training failure.
#[derive(Debug, Error, PartialEq)]
pub enum AttnResTinyTrainingError {
    #[error("attnres tiny training requires a non-empty model id")]
    MissingModelId,
    #[error("attnres tiny training requires a non-empty model revision")]
    MissingModelRevision,
    #[error("attnres tiny training requires a non-empty run id")]
    MissingRunId,
    #[error("attnres tiny training requires a non-empty checkpoint family")]
    MissingCheckpointFamily,
    #[error("attnres tiny training requires a non-zero step duration")]
    InvalidStepDuration,
    #[error("attnres tiny training requires a positive finite-difference epsilon, got {epsilon}")]
    InvalidFiniteDifferenceEpsilon { epsilon: f32 },
    #[error("attnres tiny training requires at least one training sample")]
    EmptyTrainingSamples,
    #[error("attnres tiny training requires at least one held-out sample")]
    EmptyHeldOutSamples,
    #[error("attnres tiny training sample `{sample_id}` has an empty prefix")]
    EmptySamplePrefix { sample_id: String },
    #[error(
        "attnres tiny training sample `{sample_id}` target token {target_token} exceeds vocab size {vocab_size}"
    )]
    TargetOutOfRange {
        sample_id: String,
        target_token: u32,
        vocab_size: usize,
    },
    #[error("attnres tiny training run is missing parameter group `{group_id}`")]
    MissingParameterGroup { group_id: String },
    #[error("attnres tiny training group `{group_id}` is not dense f32")]
    NonDenseGroup { group_id: String },
    #[error("{context}: {message}")]
    Serialization {
        context: &'static str,
        message: String,
    },
    #[error(transparent)]
    TrainCore(#[from] TrainingCoreError),
    #[error(transparent)]
    Model(#[from] AttnResModelError),
    #[error(transparent)]
    Execution(#[from] AttnResExecutionError),
    #[error(transparent)]
    Eval(#[from] AttnResTrainingEvalError),
}

/// Runs the bounded AttnRes tiny-training lane end to end.
pub fn train_attnres_tiny_next_token(
    corpus: &AttnResTinyTrainingCorpus,
    config: &AttnResTinyTrainingConfig,
) -> Result<AttnResTinyTrainingOutcome, AttnResTinyTrainingError> {
    config.validate()?;
    validate_corpus(corpus)?;

    let initial_model = AttnResCpuReferenceModel::seeded(
        config.model_id.clone(),
        config.model_revision.clone(),
        corpus.config.clone(),
    )?;
    let trainable_parameters = initial_model
        .weights()
        .parameter_vectors()
        .into_iter()
        .filter(|parameter| {
            parameter.parameter_id.ends_with(".pseudo_query")
                || parameter.parameter_id == "lm_head.weight"
                || parameter.parameter_id == "lm_head.bias"
        })
        .collect::<Vec<_>>();
    let mut run = FixedBudgetTrainingRun::new(
        config.run_id.clone(),
        config.checkpoint_family.clone(),
        config.budget,
        build_training_groups(
            trainable_parameters.as_slice(),
            &config.routing_optimizer,
            &config.head_optimizer,
        )?,
    )?;

    let initial_checkpoint = export_checkpoint(
        &initial_model,
        &run,
        &trainable_parameters,
        corpus,
        config,
        0,
        None,
        None,
    )?;
    let initial_training_mean_loss = mean_loss(&initial_model, &corpus.training_samples)?;
    let mut step_receipts = Vec::new();
    let mut step_metrics = Vec::new();

    for step_idx in 0..config.budget.max_steps {
        let sample = &corpus.training_samples[step_idx as usize % corpus.training_samples.len()];
        let current_model = materialize_model(&initial_model, &run)?;
        let batch = build_gradient_batch(
            &initial_model,
            &run,
            &current_model,
            sample,
            config.finite_difference_epsilon,
        )?;
        let started_at_ms = config.started_at_ms + step_idx * config.step_duration_ms;
        let finished_at_ms = started_at_ms + config.step_duration_ms;
        let receipt =
            run.apply_step(TrainingStepInput::new(batch, started_at_ms, finished_at_ms))?;
        let stepped_model = materialize_model(&initial_model, &run)?;
        let held_out_eval = evaluate_attnres_training_shift(
            &initial_model,
            &stepped_model,
            &corpus.held_out_samples,
        )?;
        step_metrics.push(AttnResTinyTrainingStepMetrics {
            receipt_id: receipt.receipt_id.clone(),
            global_step: receipt.schedule.global_step,
            training_mean_loss: mean_loss(&stepped_model, &corpus.training_samples)?,
            held_out_mean_loss: held_out_eval.trained_mean_loss,
            held_out_mean_routing_l2_delta: held_out_eval.mean_routing_l2_delta,
            held_out_improved_case_count: held_out_eval.improved_case_count,
        });
        step_receipts.push(receipt);
    }

    let trained_model = materialize_model(&initial_model, &run)?;
    let held_out_eval =
        evaluate_attnres_training_shift(&initial_model, &trained_model, &corpus.held_out_samples)?;
    let final_checkpoint = export_checkpoint(
        &initial_model,
        &run,
        &trainable_parameters,
        corpus,
        config,
        run.completed_steps(),
        Some(&initial_checkpoint),
        step_receipts.last(),
    )?;
    let final_training_mean_loss = mean_loss(&trained_model, &corpus.training_samples)?;
    let summary = AttnResTinyTrainingSummary {
        run_summary: run.summary(),
        initial_training_mean_loss,
        final_training_mean_loss,
        training_loss_delta: final_training_mean_loss - initial_training_mean_loss,
        held_out_eval,
        initial_checkpoint_manifest_digest: initial_checkpoint.manifest.stable_digest(),
        final_checkpoint_manifest_digest: final_checkpoint.manifest.stable_digest(),
    };

    Ok(AttnResTinyTrainingOutcome {
        initial_model,
        trained_model,
        step_receipts,
        step_metrics,
        initial_checkpoint,
        final_checkpoint,
        summary,
    })
}

/// Restores one AttnRes model from a persisted tiny-training checkpoint.
pub fn restore_attnres_tiny_checkpoint(
    manifest: &AttnResTinyTrainingCheckpointManifest,
    weights_bytes: &[u8],
) -> Result<AttnResCpuReferenceModel, AttnResTinyTrainingError> {
    let base_model = AttnResCpuReferenceModel::seeded(
        manifest.model_id.clone(),
        manifest.model_revision.clone(),
        manifest.config.clone(),
    )?;
    let safetensors = SafeTensors::deserialize(weights_bytes)
        .map_err(|error| serialization_error("checkpoint restore", error))?;
    let mut overrides = BTreeMap::new();
    for parameter_id in &manifest.parameter_ids {
        let tensor = safetensors
            .tensor(parameter_id)
            .map_err(|error| serialization_error("checkpoint restore", error))?;
        let values = decode_f32_bytes(parameter_id.as_str(), tensor.data())?;
        overrides.insert(parameter_id.clone(), values);
    }
    let weights = base_model
        .weights()
        .with_parameter_overrides(&manifest.config, &overrides)?;
    AttnResCpuReferenceModel::with_weights(
        base_model.descriptor().model.clone(),
        manifest.config.clone(),
        weights,
    )
    .map_err(Into::into)
}

fn validate_corpus(corpus: &AttnResTinyTrainingCorpus) -> Result<(), AttnResTinyTrainingError> {
    if corpus.training_samples.is_empty() {
        return Err(AttnResTinyTrainingError::EmptyTrainingSamples);
    }
    if corpus.held_out_samples.is_empty() {
        return Err(AttnResTinyTrainingError::EmptyHeldOutSamples);
    }
    for sample in corpus
        .training_samples
        .iter()
        .chain(corpus.held_out_samples.iter())
    {
        if sample.input_tokens.is_empty() {
            return Err(AttnResTinyTrainingError::EmptySamplePrefix {
                sample_id: sample.sample_id.clone(),
            });
        }
        if sample.target_token.as_u32() as usize >= corpus.config.vocab_size {
            return Err(AttnResTinyTrainingError::TargetOutOfRange {
                sample_id: sample.sample_id.clone(),
                target_token: sample.target_token.as_u32(),
                vocab_size: corpus.config.vocab_size,
            });
        }
    }
    Ok(())
}

fn build_training_groups(
    parameters: &[AttnResParameterVector],
    routing_optimizer: &TrainingOptimizerConfig,
    head_optimizer: &TrainingOptimizerConfig,
) -> Result<Vec<TrainingParameterGroupState>, AttnResTinyTrainingError> {
    let mut groups = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        let class = if parameter.parameter_id.ends_with(".pseudo_query") {
            TrainingParameterClass::Scalar
        } else if parameter.parameter_id.ends_with(".bias") {
            TrainingParameterClass::Bias
        } else {
            TrainingParameterClass::Head
        };
        let optimizer = if parameter.parameter_id.ends_with(".pseudo_query") {
            routing_optimizer.clone()
        } else {
            head_optimizer.clone()
        };
        groups.push(TrainingParameterGroupState::new(
            parameter.parameter_id.clone(),
            class,
            TrainingTensorBuffer::from_f32(
                parameter.parameter_id.clone(),
                TensorSpec::new(parameter.shape.clone(), DType::F32, Device::cpu()),
                parameter.values.clone(),
            )?,
            optimizer,
            TrainingOptimizerResidencyPolicy::host_only(),
        )?);
    }
    Ok(groups)
}

fn build_gradient_batch(
    initial_model: &AttnResCpuReferenceModel,
    run: &FixedBudgetTrainingRun,
    current_model: &AttnResCpuReferenceModel,
    sample: &AttnResNextTokenSample,
    epsilon: f32,
) -> Result<TrainingGradientBatch, AttnResTinyTrainingError> {
    let (last_hidden, last_logits, loss) = sample_forward(current_model, sample)?;
    let probs = softmax(last_logits.as_slice());
    let target_index = sample.target_token.as_u32() as usize;
    let mut logits_gradient = probs.clone();
    logits_gradient[target_index] -= 1.0;

    let weight_group = required_group(run, "lm_head.weight")?;
    let bias_group = required_group(run, "lm_head.bias")?;
    let mut gradients = BTreeMap::new();
    gradients.insert(
        String::from("lm_head.weight"),
        TrainingTensorBuffer::from_f32(
            String::from("lm_head.weight"),
            weight_group.parameter.spec.clone(),
            lm_head_weight_gradient(last_hidden.as_slice(), logits_gradient.as_slice()),
        )?,
    );
    gradients.insert(
        String::from("lm_head.bias"),
        TrainingTensorBuffer::from_f32(
            String::from("lm_head.bias"),
            bias_group.parameter.spec.clone(),
            logits_gradient.clone(),
        )?,
    );

    let base_overrides = collect_overrides(run)?;
    for group_id in run
        .summary()
        .final_parameter_norms_l2
        .keys()
        .filter(|group_id| group_id.ends_with(".pseudo_query"))
    {
        let group = required_group(run, group_id)?;
        let mut gradient = vec![0.0f32; dense_values(group, group_id.as_str())?.len()];
        for index in 0..gradient.len() {
            let mut plus = base_overrides.clone();
            let mut minus = base_overrides.clone();
            plus.get_mut(group_id).expect("routing group override")[index] += epsilon;
            minus.get_mut(group_id).expect("routing group override")[index] -= epsilon;
            let plus_model = materialize_model_with_overrides(initial_model, &plus)?;
            let minus_model = materialize_model_with_overrides(initial_model, &minus)?;
            let plus_loss = sample_loss(&plus_model, sample)?;
            let minus_loss = sample_loss(&minus_model, sample)?;
            gradient[index] = (plus_loss - minus_loss) / (2.0 * epsilon);
        }
        gradients.insert(
            group_id.clone(),
            TrainingTensorBuffer::from_f32(
                group_id.clone(),
                group.parameter.spec.clone(),
                gradient,
            )?,
        );
    }

    Ok(TrainingGradientBatch::new(
        format!("{}-gradient", sample.sample_id),
        loss,
        1,
        gradients,
    ))
}

fn sample_forward(
    model: &AttnResCpuReferenceModel,
    sample: &AttnResNextTokenSample,
) -> Result<(Vec<f32>, Vec<f32>, f32), AttnResTinyTrainingError> {
    let batch = [sample.input_tokens.clone()];
    let hidden = model.forward_hidden(&batch)?;
    let logits = model.forward(&batch)?;
    let last_hidden = last_position_slice(&hidden);
    let last_logits = last_position_slice(&logits);
    let probabilities = softmax(last_logits.as_slice());
    let target_probability = probabilities[sample.target_token.as_u32() as usize].max(f32::EPSILON);
    Ok((last_hidden, last_logits, -target_probability.ln()))
}

fn sample_loss(
    model: &AttnResCpuReferenceModel,
    sample: &AttnResNextTokenSample,
) -> Result<f32, AttnResTinyTrainingError> {
    sample_forward(model, sample).map(|(_, _, loss)| loss)
}

fn mean_loss(
    model: &AttnResCpuReferenceModel,
    samples: &[AttnResNextTokenSample],
) -> Result<f32, AttnResTinyTrainingError> {
    let mut total = 0.0f32;
    for sample in samples {
        total += sample_loss(model, sample)?;
    }
    Ok(total / samples.len() as f32)
}

fn lm_head_weight_gradient(hidden: &[f32], logits_gradient: &[f32]) -> Vec<f32> {
    let mut gradient = vec![0.0f32; hidden.len() * logits_gradient.len()];
    for (input_index, hidden_value) in hidden.iter().enumerate() {
        for (output_index, logit_grad) in logits_gradient.iter().enumerate() {
            gradient[input_index * logits_gradient.len() + output_index] =
                hidden_value * logit_grad;
        }
    }
    gradient
}

fn collect_overrides(
    run: &FixedBudgetTrainingRun,
) -> Result<BTreeMap<String, Vec<f32>>, AttnResTinyTrainingError> {
    let mut overrides = BTreeMap::new();
    for group_id in run.summary().final_parameter_norms_l2.keys() {
        let group = required_group(run, group_id.as_str())?;
        overrides.insert(
            group_id.clone(),
            dense_values(group, group_id.as_str())?.to_vec(),
        );
    }
    Ok(overrides)
}

fn materialize_model(
    initial_model: &AttnResCpuReferenceModel,
    run: &FixedBudgetTrainingRun,
) -> Result<AttnResCpuReferenceModel, AttnResTinyTrainingError> {
    materialize_model_with_overrides(initial_model, &collect_overrides(run)?)
}

fn materialize_model_with_overrides(
    initial_model: &AttnResCpuReferenceModel,
    overrides: &BTreeMap<String, Vec<f32>>,
) -> Result<AttnResCpuReferenceModel, AttnResTinyTrainingError> {
    let weights = initial_model
        .weights()
        .with_parameter_overrides(initial_model.config(), overrides)?;
    AttnResCpuReferenceModel::with_weights(
        initial_model.descriptor().model.clone(),
        initial_model.config().clone(),
        weights,
    )
    .map_err(Into::into)
}

fn required_group<'a>(
    run: &'a FixedBudgetTrainingRun,
    group_id: &str,
) -> Result<&'a TrainingParameterGroupState, AttnResTinyTrainingError> {
    run.parameter_group(group_id)
        .ok_or_else(|| AttnResTinyTrainingError::MissingParameterGroup {
            group_id: String::from(group_id),
        })
}

fn dense_values<'a>(
    group: &'a TrainingParameterGroupState,
    group_id: &str,
) -> Result<&'a [f32], AttnResTinyTrainingError> {
    match &group.parameter.data {
        TensorData::F32(values) => Ok(values.as_slice()),
        TensorData::QuantizedBlocks(_) => Err(AttnResTinyTrainingError::NonDenseGroup {
            group_id: String::from(group_id),
        }),
    }
}

fn last_position_slice(tensor: &psionic_models::AttnResTensor3) -> Vec<f32> {
    let width = tensor.width();
    let last_position = tensor.sequence_length() - 1;
    let offset = last_position * width;
    tensor.values()[offset..offset + width].to_vec()
}

fn export_checkpoint(
    initial_model: &AttnResCpuReferenceModel,
    run: &FixedBudgetTrainingRun,
    trainable_parameters: &[AttnResParameterVector],
    corpus: &AttnResTinyTrainingCorpus,
    config: &AttnResTinyTrainingConfig,
    step: u64,
    parent: Option<&AttnResTinyTrainingCheckpointArtifact>,
    receipt: Option<&TrainingStepReceipt>,
) -> Result<AttnResTinyTrainingCheckpointArtifact, AttnResTinyTrainingError> {
    let checkpoint_ref = format!("{}:step:{}", config.run_id, step);
    let parameter_ids = trainable_parameters
        .iter()
        .map(|parameter| parameter.parameter_id.clone())
        .collect::<Vec<_>>();
    let weights_bytes = export_checkpoint_weights(run, parameter_ids.as_slice())?;
    let manifest = AttnResTinyTrainingCheckpointManifest {
        schema_version: 1,
        checkpoint_ref: checkpoint_ref.clone(),
        checkpoint_family: config.checkpoint_family.clone(),
        run_id: config.run_id.clone(),
        step,
        model_id: config.model_id.clone(),
        model_revision: config.model_revision.clone(),
        config: corpus.config.clone(),
        base_descriptor_digest: initial_model.descriptor().stable_digest(),
        base_weight_digest: initial_model.descriptor().weights.digest.clone(),
        parameter_state_digest: stable_bytes_digest(
            b"psionic_attnres_checkpoint_weights|",
            &weights_bytes,
        ),
        training_dataset_digest: corpus.training_digest(),
        held_out_dataset_digest: corpus.held_out_digest(),
        parameter_ids: parameter_ids.clone(),
        parent_checkpoint_ref: parent.map(|checkpoint| checkpoint.manifest.checkpoint_ref.clone()),
        parent_manifest_digest: parent.map(|checkpoint| checkpoint.manifest.stable_digest()),
        step_receipt_id: receipt.map(|receipt| receipt.receipt_id.clone()),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| {
        AttnResTinyTrainingError::Serialization {
            context: "attnres checkpoint manifest export",
            message: error.to_string(),
        }
    })?;
    let weights_artifact = AttnResTinyTrainingArtifact::new(
        "attnres_checkpoint_weights",
        format!(
            "artifact://attnres/{}/checkpoint/{step}/weights.safetensors",
            config.run_id
        ),
        weights_bytes.clone(),
    );
    let manifest_artifact = AttnResTinyTrainingArtifact::new(
        "attnres_checkpoint_manifest",
        format!(
            "artifact://attnres/{}/checkpoint/{step}/manifest.json",
            config.run_id
        ),
        manifest_bytes,
    );
    let stream_id = format!("attnres.checkpoint.{}.{}", config.run_id, step);
    let manifest_ref = DatastreamManifestRef {
        stream_id: stream_id.clone(),
        manifest_digest: manifest_artifact.artifact_digest.clone(),
        subject: DatastreamSubjectKind::Checkpoint,
        object_digest: weights_artifact.artifact_digest.clone(),
        total_bytes: weights_artifact.bytes.len() as u64,
        chunk_count: 1,
        chunk_bytes: weights_artifact.bytes.len(),
        encoding: DatastreamEncoding::Safetensors,
        compression: None,
        provenance_digest: Some(initial_model.descriptor().stable_digest()),
        dataset_binding: None,
        checkpoint_binding: Some(
            DatastreamCheckpointBinding::new(config.checkpoint_family.clone())
                .with_checkpoint_ref(checkpoint_ref.clone())
                .with_step(step),
        ),
        policy_weight_binding: None,
        mirrors: Vec::new(),
    };
    let checkpoint = TrainingCheckpointReference::new(
        config.checkpoint_family.clone(),
        stream_id,
        manifest_ref.manifest_digest.clone(),
        manifest_ref.object_digest.clone(),
        "psionic.local.cpu_reference",
        0,
        "cluster.local.cpu_reference",
        "topology.cpu_reference",
        config.started_at_ms + step.saturating_mul(config.step_duration_ms),
    )
    .with_checkpoint_ref(checkpoint_ref)
    .with_step(step)
    .with_durable_at_ms(config.started_at_ms + step.saturating_mul(config.step_duration_ms));
    Ok(AttnResTinyTrainingCheckpointArtifact {
        weights_artifact,
        manifest_artifact,
        manifest,
        checkpoint,
        manifest_ref,
    })
}

fn export_checkpoint_weights(
    run: &FixedBudgetTrainingRun,
    parameter_ids: &[String],
) -> Result<Vec<u8>, AttnResTinyTrainingError> {
    let manifest_json = serde_json::to_string(parameter_ids).map_err(|error| {
        AttnResTinyTrainingError::Serialization {
            context: "attnres checkpoint metadata export",
            message: error.to_string(),
        }
    })?;
    let mut metadata = HashMap::new();
    metadata.insert(String::from(ATTNRES_CHECKPOINT_MANIFEST_KEY), manifest_json);

    let mut raw_buffers = Vec::with_capacity(parameter_ids.len());
    for parameter_id in parameter_ids {
        let group = required_group(run, parameter_id.as_str())?;
        raw_buffers.push((
            parameter_id.clone(),
            encode_f32_bytes(dense_values(group, parameter_id.as_str())?),
            group.parameter.spec.shape().dims().to_vec(),
        ));
    }

    let mut views = Vec::with_capacity(raw_buffers.len());
    for (parameter_id, raw_bytes, shape) in &raw_buffers {
        let view = TensorView::new(SafeTensorsDType::F32, shape.clone(), raw_bytes.as_slice())
            .map_err(|error| serialization_error("attnres checkpoint safetensors export", error))?;
        views.push((parameter_id.clone(), view));
    }
    serialize(
        views
            .iter()
            .map(|(parameter_id, view)| (parameter_id.as_str(), view.clone())),
        Some(metadata),
    )
    .map_err(|error| serialization_error("attnres checkpoint safetensors export", error))
}

fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32_bytes(
    parameter_id: &str,
    bytes: &[u8],
) -> Result<Vec<f32>, AttnResTinyTrainingError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(AttnResTinyTrainingError::Serialization {
            context: "checkpoint restore",
            message: format!(
                "tensor `{parameter_id}` byte length {} is not divisible by 4",
                bytes.len()
            ),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = values
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>().max(f32::EPSILON);
    exp.into_iter().map(|value| value / sum).collect()
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded = match serde_json::to_vec(value) {
        Ok(encoded) => encoded,
        Err(error) => error.to_string().into_bytes(),
    };
    stable_bytes_digest(prefix, &encoded)
}

fn stable_bytes_digest(prefix: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn serialization_error(context: &'static str, error: impl ToString) -> AttnResTinyTrainingError {
    AttnResTinyTrainingError::Serialization {
        context,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use serde::Deserialize;

    use super::{
        AttnResTinyTrainingConfig, AttnResTinyTrainingCorpus, restore_attnres_tiny_checkpoint,
        train_attnres_tiny_next_token,
    };

    #[derive(Debug, Deserialize)]
    struct TinyTrainingFixture {
        description: String,
        config: psionic_models::AttnResConfig,
        training_samples: Vec<psionic_models::AttnResNextTokenSample>,
        held_out_samples: Vec<psionic_models::AttnResNextTokenSample>,
    }

    #[test]
    fn attnres_tiny_training_runs_end_to_end_and_restores_checkpoint() -> Result<(), Box<dyn Error>>
    {
        let fixture: TinyTrainingFixture = serde_json::from_str(include_str!(
            "../../../fixtures/attnres/tiny_training_cases.json"
        ))?;
        let corpus = AttnResTinyTrainingCorpus {
            description: fixture.description,
            config: fixture.config,
            training_samples: fixture.training_samples,
            held_out_samples: fixture.held_out_samples,
        };
        let config = AttnResTinyTrainingConfig::reference()?;
        let outcome = train_attnres_tiny_next_token(&corpus, &config)?;
        assert!(
            outcome.summary.final_training_mean_loss < outcome.summary.initial_training_mean_loss
        );
        assert!(outcome.summary.held_out_eval.mean_routing_l2_delta > 0.0);
        let restored = restore_attnres_tiny_checkpoint(
            &outcome.final_checkpoint.manifest,
            &outcome.final_checkpoint.weights_artifact.bytes,
        )?;
        assert_eq!(restored.descriptor(), outcome.trained_model.descriptor());
        let restored_eval = psionic_eval::evaluate_attnres_training_shift(
            &outcome.initial_model,
            &restored,
            &corpus.held_out_samples,
        )?;
        assert_eq!(restored_eval, outcome.summary.held_out_eval);
        Ok(())
    }
}

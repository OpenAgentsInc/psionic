use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    io::{Read, Write},
    path::Path,
};

use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use half::f16;
use psionic_data::{
    ParameterGolfDataError, ParameterGolfSentencePieceByteLuts,
    ParameterGolfSentencePieceTokenEntry,
};
use psionic_eval::{
    evaluate_parameter_golf_validation, ParameterGolfValidationEvalError,
    ParameterGolfValidationEvalReport,
};
use psionic_models::{
    ParameterGolfBankedWeights, ParameterGolfExecutionError, ParameterGolfModelError,
    ParameterGolfParameterVector, ParameterGolfReferenceModel,
};
use psionic_runtime::TrainingCheckpointReference;
use safetensors::{serialize, tensor::TensorView, Dtype as SafeTensorsDType, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    apply_parameter_golf_muon_step, parameter_golf_optimizer_plan, AsyncCheckpointWritebackError,
    AsyncCheckpointWritebackFile, AsyncCheckpointWritebackOptions, AsyncCheckpointWritebackPayload,
    AsyncCheckpointWritebackReceipt, AsyncCheckpointWritebackTicket,
    AsyncCheckpointWritebackWorker, LocalTrainMetricEvent, LocalTrainMetricFanout,
    LocalTrainMetricPhase, LocalTrainMetricSinkError, LocalTrainMetricValue,
    ParameterGolfMuonConfig, ParameterGolfMuonState, ParameterGolfOptimizerExecution,
    ParameterGolfTrainError, ParameterGolfTrainingHyperparameters, TrainingOptimizerConfig,
    TrainingOptimizerError, TrainingOptimizerState, PARAMETER_GOLF_CONTROL_TENSOR_NAME_PATTERNS,
};

const PARAMETER_GOLF_CHECKPOINT_MANIFEST_KEY: &str = "psionic.parameter_golf.checkpoint_manifest";
const PARAMETER_GOLF_WEIGHT_SURFACE_KEY: &str = "psionic.parameter_golf.weight_surface";
const PARAMETER_GOLF_SPLIT_WEIGHT_SURFACE: &str = "split_f32_v1";
const PARAMETER_GOLF_BANKED_WEIGHT_SURFACE: &str = "banked_f32_v1";
const PARAMETER_GOLF_INT8_ZLIB_FORMAT: &str = "int8_clean_per_row_v1";
const PARAMETER_GOLF_INT8_KEEP_FLOAT_MAX_NUMEL: usize = 65_536;
const PARAMETER_GOLF_INT8_CLIP_Q: f32 = 0.999_998_4;

/// Stable single-device batch geometry for the Parameter Golf lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfBatchGeometry {
    /// Distributed world size encoded by the upstream challenge script.
    pub world_size: usize,
    /// Global train-token budget per step.
    pub train_batch_tokens: usize,
    /// Global validation-token budget per eval batch.
    pub validation_batch_tokens: usize,
    /// Sequence length for both training and validation.
    pub train_sequence_length: usize,
    /// Gradient-accumulation steps per optimizer step.
    pub grad_accum_steps: usize,
}

impl ParameterGolfBatchGeometry {
    /// Returns the current public single-device challenge defaults from `train_gpt.py`.
    #[must_use]
    pub const fn challenge_single_device_defaults() -> Self {
        Self {
            world_size: 1,
            train_batch_tokens: 524_288,
            validation_batch_tokens: 524_288,
            train_sequence_length: 1024,
            grad_accum_steps: 8,
        }
    }

    /// Returns the bounded local-reference defaults used by repo-owned tests.
    #[must_use]
    pub const fn local_reference_defaults() -> Self {
        Self {
            world_size: 1,
            train_batch_tokens: 32,
            validation_batch_tokens: 32,
            train_sequence_length: 4,
            grad_accum_steps: 8,
        }
    }

    /// Returns the current public `8xH100` challenge defaults from `train_gpt.py`.
    #[must_use]
    pub const fn challenge_distributed_8xh100_defaults() -> Self {
        Self {
            world_size: 8,
            train_batch_tokens: 524_288,
            validation_batch_tokens: 524_288,
            train_sequence_length: 1024,
            grad_accum_steps: 1,
        }
    }

    /// Returns the per-rank, per-microbatch train-token count.
    #[must_use]
    pub const fn local_train_batch_tokens(&self) -> usize {
        self.train_batch_tokens / (self.world_size * self.grad_accum_steps)
    }

    /// Returns the per-rank, per-microbatch validation-token count.
    #[must_use]
    pub const fn local_validation_batch_tokens(&self) -> usize {
        self.validation_batch_tokens / (self.world_size * self.grad_accum_steps)
    }

    /// Returns the number of local train sequences per microbatch.
    #[must_use]
    pub const fn local_train_batch_sequences(&self) -> usize {
        self.local_train_batch_tokens() / self.train_sequence_length
    }

    /// Returns the number of local validation sequences per batch.
    #[must_use]
    pub const fn local_validation_batch_sequences(&self) -> usize {
        self.local_validation_batch_tokens() / self.train_sequence_length
    }

    fn validate(&self) -> Result<(), ParameterGolfReferenceTrainingError> {
        if self.world_size != 1 {
            return Err(ParameterGolfReferenceTrainingError::InvalidBatchGeometry {
                message: format!(
                    "parameter golf local reference trainer only supports world_size=1, found {}",
                    self.world_size
                ),
            });
        }
        if self.train_sequence_length == 0 {
            return Err(ParameterGolfReferenceTrainingError::InvalidBatchGeometry {
                message: String::from("train_sequence_length must be positive"),
            });
        }
        if self.grad_accum_steps == 0 {
            return Err(ParameterGolfReferenceTrainingError::InvalidBatchGeometry {
                message: String::from("grad_accum_steps must be positive"),
            });
        }
        let denom = self.world_size.saturating_mul(self.grad_accum_steps);
        if self.train_batch_tokens == 0
            || self.train_batch_tokens % denom != 0
            || self.local_train_batch_tokens() < self.train_sequence_length
            || self.local_train_batch_tokens() % self.train_sequence_length != 0
        {
            return Err(ParameterGolfReferenceTrainingError::InvalidBatchGeometry {
                message: format!(
                    "train_batch_tokens={} must divide cleanly across world_size={} grad_accum_steps={} and admit at least one full sequence of length {}",
                    self.train_batch_tokens,
                    self.world_size,
                    self.grad_accum_steps,
                    self.train_sequence_length
                ),
            });
        }
        if self.validation_batch_tokens == 0
            || self.validation_batch_tokens % denom != 0
            || self.local_validation_batch_tokens() < self.train_sequence_length
            || self.local_validation_batch_tokens() % self.train_sequence_length != 0
        {
            return Err(ParameterGolfReferenceTrainingError::InvalidBatchGeometry {
                message: format!(
                    "validation_batch_tokens={} must divide cleanly across world_size={} grad_accum_steps={} and admit at least one full sequence of length {}",
                    self.validation_batch_tokens,
                    self.world_size,
                    self.grad_accum_steps,
                    self.train_sequence_length
                ),
            });
        }
        Ok(())
    }
}

/// Repo-owned bounded fixture for the local Parameter Golf reference trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfLocalReferenceFixture {
    /// Human-readable fixture description.
    pub description: String,
    /// Tokenizer vocabulary size.
    pub tokenizer_vocab_size: usize,
    /// Tokenizer entries used by the eval byte-accounting LUTs.
    pub sentencepiece_entries: Vec<ParameterGolfSentencePieceTokenEntry>,
    /// Flat training-token stream.
    pub training_tokens: Vec<u16>,
    /// Flat validation-token stream.
    pub validation_tokens: Vec<u16>,
}

impl ParameterGolfLocalReferenceFixture {
    /// Returns the canonical repo-owned local-reference fixture.
    pub fn reference() -> Result<Self, ParameterGolfReferenceTrainingError> {
        let fixture: Self = serde_json::from_str(include_str!(
            "../../../fixtures/parameter_golf/train/parameter_golf_local_reference_fixture.json"
        ))
        .map_err(|error| ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf local reference fixture load",
            message: error.to_string(),
        })?;
        fixture.validate()?;
        Ok(fixture)
    }

    /// Builds the SentencePiece byte-accounting LUTs for the fixture tokenizer.
    pub fn byte_luts(
        &self,
    ) -> Result<ParameterGolfSentencePieceByteLuts, ParameterGolfReferenceTrainingError> {
        Ok(ParameterGolfSentencePieceByteLuts::build(
            self.tokenizer_vocab_size,
            self.sentencepiece_entries.as_slice(),
        )?)
    }

    /// Returns a stable digest over the training token stream.
    #[must_use]
    pub fn training_digest(&self) -> String {
        stable_digest(
            b"psionic_parameter_golf_training_tokens|",
            &self.training_tokens,
        )
    }

    /// Returns a stable digest over the validation token stream.
    #[must_use]
    pub fn validation_digest(&self) -> String {
        stable_digest(
            b"psionic_parameter_golf_validation_tokens|",
            &self.validation_tokens,
        )
    }

    fn validate(&self) -> Result<(), ParameterGolfReferenceTrainingError> {
        if self.tokenizer_vocab_size == 0 {
            return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
                message: String::from("tokenizer_vocab_size must be positive"),
            });
        }
        if self.training_tokens.len() < 2 {
            return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
                message: String::from("training_tokens must contain at least two tokens"),
            });
        }
        if self.validation_tokens.len() < 2 {
            return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
                message: String::from("validation_tokens must contain at least two tokens"),
            });
        }
        for &token_id in self
            .training_tokens
            .iter()
            .chain(self.validation_tokens.iter())
        {
            if token_id as usize >= self.tokenizer_vocab_size {
                return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
                    message: format!(
                        "token id {} exceeds tokenizer_vocab_size {}",
                        token_id, self.tokenizer_vocab_size
                    ),
                });
            }
        }
        Ok(())
    }
}

/// One selected trainable coordinate for the bounded local-reference trainer.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ParameterGolfTrainableCoordinate {
    /// Stable tensor identifier.
    pub parameter_id: String,
    /// Flat row-major index inside the tensor.
    pub flat_index: usize,
}

/// Config for the bounded Parameter Golf local-reference trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfReferenceTrainingConfig {
    /// Stable training run identifier.
    pub run_id: String,
    /// Stable checkpoint family.
    pub checkpoint_family: String,
    /// Logical training start time.
    pub started_at_ms: u64,
    /// Logical per-step duration.
    pub step_duration_ms: u64,
    /// Fixed maximum step count.
    pub max_steps: u64,
    /// Batch geometry contract for the run.
    pub geometry: ParameterGolfBatchGeometry,
    /// Baseline optimizer and schedule hyperparameters.
    pub hyperparameters: ParameterGolfTrainingHyperparameters,
    /// Finite-difference epsilon used for selected coordinates.
    pub finite_difference_epsilon: f32,
    /// Selected trainable coordinates.
    pub selected_coordinates: Vec<ParameterGolfTrainableCoordinate>,
}

impl ParameterGolfReferenceTrainingConfig {
    /// Returns the canonical bounded local-reference config used by repo-owned tests.
    #[must_use]
    pub fn local_reference() -> Self {
        let geometry = ParameterGolfBatchGeometry::local_reference_defaults();
        let model_dim = 512;
        Self {
            run_id: String::from("parameter-golf-local-reference-run"),
            checkpoint_family: String::from("train.parameter_golf.local_reference"),
            started_at_ms: 1_774_320_000_000,
            step_duration_ms: 50,
            max_steps: 2,
            geometry,
            hyperparameters: ParameterGolfTrainingHyperparameters::baseline_defaults(),
            finite_difference_epsilon: 0.01,
            selected_coordinates: vec![
                ParameterGolfTrainableCoordinate {
                    parameter_id: String::from("tok_emb.weight"),
                    flat_index: model_dim,
                },
                ParameterGolfTrainableCoordinate {
                    parameter_id: String::from("blocks.0.attn.c_q.weight"),
                    flat_index: 0,
                },
                ParameterGolfTrainableCoordinate {
                    parameter_id: String::from("skip_weights"),
                    flat_index: 0,
                },
            ],
        }
    }

    fn validate(&self) -> Result<(), ParameterGolfReferenceTrainingError> {
        if self.run_id.trim().is_empty() {
            return Err(ParameterGolfReferenceTrainingError::MissingRunId);
        }
        if self.checkpoint_family.trim().is_empty() {
            return Err(ParameterGolfReferenceTrainingError::MissingCheckpointFamily);
        }
        if self.step_duration_ms == 0 {
            return Err(ParameterGolfReferenceTrainingError::InvalidStepDuration);
        }
        if self.max_steps == 0 {
            return Err(ParameterGolfReferenceTrainingError::InvalidMaxSteps);
        }
        if !self.finite_difference_epsilon.is_finite() || self.finite_difference_epsilon <= 0.0 {
            return Err(
                ParameterGolfReferenceTrainingError::InvalidFiniteDifferenceEpsilon {
                    epsilon: self.finite_difference_epsilon,
                },
            );
        }
        if self.selected_coordinates.is_empty() {
            return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
                message: String::from("selected_coordinates must not be empty"),
            });
        }
        self.geometry.validate()?;
        Ok(())
    }
}

/// One serialized artifact emitted by the local-reference trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfTrainingArtifact {
    /// Stable artifact kind.
    pub artifact_kind: String,
    /// Stable artifact reference.
    pub artifact_ref: String,
    /// Stable artifact digest.
    pub artifact_digest: String,
    /// Serialized artifact bytes.
    pub bytes: Vec<u8>,
}

impl ParameterGolfTrainingArtifact {
    pub(crate) fn new(
        artifact_kind: impl Into<String>,
        artifact_ref: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        let artifact_kind = artifact_kind.into();
        let artifact_ref = artifact_ref.into();
        let mut hasher = Sha256::new();
        hasher.update(b"psionic_parameter_golf_training_artifact|");
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

/// Serializable optimizer-state payload for one trainable tensor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum ParameterGolfReferenceOptimizerState {
    /// Adam-family state over selected coordinates.
    Adam {
        /// Optimizer state for the selected coordinate vector.
        state: TrainingOptimizerState,
    },
    /// Muon state over a dense matrix tensor.
    Muon {
        /// Muon momentum buffer.
        state: ParameterGolfMuonState,
    },
}

/// Serializable checkpoint view for one trainable tensor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfCheckpointTrainableTensor {
    /// Stable tensor identifier.
    pub parameter_id: String,
    /// Logical tensor shape.
    pub shape: Vec<usize>,
    /// Selected flat coordinates used for finite-difference gradients.
    pub selected_indices: Vec<usize>,
    /// Reconstructed optimizer execution for the tensor.
    pub execution: ParameterGolfOptimizerExecution,
    /// Mutable optimizer state for the tensor.
    pub optimizer_state: ParameterGolfReferenceOptimizerState,
}

/// JSON manifest paired with one raw checkpoint export.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfCheckpointManifest {
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
    /// Logical training start time.
    pub started_at_ms: u64,
    /// Logical per-step duration.
    pub step_duration_ms: u64,
    /// Fixed-budget step count.
    pub max_steps: u64,
    /// Single-device batch geometry.
    pub geometry: ParameterGolfBatchGeometry,
    /// Baseline optimizer and schedule hyperparameters.
    pub hyperparameters: ParameterGolfTrainingHyperparameters,
    /// Finite-difference epsilon for the run.
    pub finite_difference_epsilon: f32,
    /// Stable descriptor digest for the seeded baseline model.
    pub base_descriptor_digest: String,
    /// Stable descriptor digest for the current checkpointed model.
    pub current_descriptor_digest: String,
    /// Stable digest over the optimizer split.
    pub optimizer_plan_digest: String,
    /// Stable digest over the training split.
    pub training_dataset_digest: String,
    /// Stable digest over the validation split.
    pub validation_dataset_digest: String,
    /// Current validation-eval digest.
    pub validation_eval_digest: String,
    /// Step metrics accumulated so far.
    pub step_metrics: Vec<ParameterGolfReferenceTrainingStepMetrics>,
    /// Trainable tensor runtime state.
    pub trainable_tensors: Vec<ParameterGolfCheckpointTrainableTensor>,
    /// Optional parent checkpoint reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint_ref: Option<String>,
    /// Optional parent manifest digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_manifest_digest: Option<String>,
}

impl ParameterGolfCheckpointManifest {
    /// Returns a stable digest over the manifest payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"psionic_parameter_golf_checkpoint_manifest|", self)
    }
}

/// One persisted checkpoint plus lineage refs for the local-reference trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfCheckpointArtifact {
    /// Raw-model weights artifact.
    pub weights_artifact: ParameterGolfTrainingArtifact,
    /// JSON manifest artifact.
    pub manifest_artifact: ParameterGolfTrainingArtifact,
    /// Structured manifest.
    pub manifest: ParameterGolfCheckpointManifest,
    /// Runtime-visible checkpoint reference.
    pub checkpoint: TrainingCheckpointReference,
}

/// Higher-level per-step metrics for the local-reference trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfReferenceTrainingStepMetrics {
    /// One-based global step.
    pub global_step: u64,
    /// Mean microbatch loss across the gradient-accumulation window.
    pub mean_microbatch_loss: f32,
    /// Validation mean loss after the step.
    pub validation_mean_loss: f64,
    /// Validation bits per byte after the step.
    pub validation_bits_per_byte: f64,
    /// Effective LR multiplier at the step.
    pub learning_rate_multiplier: f32,
    /// Effective Muon momentum at the step.
    pub muon_momentum: f32,
}

/// Final machine-readable summary for one bounded local-reference run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfReferenceTrainingSummary {
    /// Initial validation mean loss.
    pub initial_validation_mean_loss: f64,
    /// Final validation mean loss.
    pub final_validation_mean_loss: f64,
    /// Final validation bits per byte.
    pub final_validation_bits_per_byte: f64,
    /// Raw roundtrip validation mean loss.
    pub raw_roundtrip_validation_mean_loss: f64,
    /// Int8+zlib roundtrip validation mean loss.
    pub int8_zlib_roundtrip_validation_mean_loss: f64,
    /// Stable digest of the initial checkpoint manifest.
    pub initial_checkpoint_manifest_digest: String,
    /// Stable digest of the final checkpoint manifest.
    pub final_checkpoint_manifest_digest: String,
    /// Stable digest of the raw model artifact.
    pub raw_model_artifact_digest: String,
    /// Stable digest of the int8+zlib model artifact.
    pub int8_zlib_model_artifact_digest: String,
}

/// Full bounded local-reference training outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfReferenceTrainingOutcome {
    /// Seeded baseline model before any step.
    pub initial_model: ParameterGolfReferenceModel,
    /// Final stepped model after the bounded run.
    pub trained_model: ParameterGolfReferenceModel,
    /// Accumulated per-step metrics.
    pub step_metrics: Vec<ParameterGolfReferenceTrainingStepMetrics>,
    /// Initial checkpoint artifact.
    pub initial_checkpoint: ParameterGolfCheckpointArtifact,
    /// Final checkpoint artifact.
    pub final_checkpoint: ParameterGolfCheckpointArtifact,
    /// Final raw full-precision model artifact.
    pub raw_model_artifact: ParameterGolfTrainingArtifact,
    /// Final int8+zlib model artifact.
    pub int8_zlib_model_artifact: ParameterGolfTrainingArtifact,
    /// Initial validation report.
    pub initial_validation_eval: ParameterGolfValidationEvalReport,
    /// Final validation report.
    pub final_validation_eval: ParameterGolfValidationEvalReport,
    /// Validation report after raw restore.
    pub raw_roundtrip_validation_eval: ParameterGolfValidationEvalReport,
    /// Validation report after int8+zlib restore.
    pub int8_zlib_roundtrip_validation_eval: ParameterGolfValidationEvalReport,
    /// Durable checkpoint writeback receipts when the run used async writeback.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoint_writeback_receipts: Vec<AsyncCheckpointWritebackReceipt>,
    /// Final summary.
    pub summary: ParameterGolfReferenceTrainingSummary,
}

/// Failure for the bounded Parameter Golf local-reference trainer.
#[derive(Debug, Error)]
pub enum ParameterGolfReferenceTrainingError {
    #[error("parameter golf local reference training requires a non-empty run id")]
    MissingRunId,
    #[error("parameter golf local reference training requires a non-empty checkpoint family")]
    MissingCheckpointFamily,
    #[error("parameter golf local reference training requires a non-zero step duration")]
    InvalidStepDuration,
    #[error("parameter golf local reference training requires max_steps > 0")]
    InvalidMaxSteps,
    #[error(
        "parameter golf local reference training requires a positive finite-difference epsilon, got {epsilon}"
    )]
    InvalidFiniteDifferenceEpsilon { epsilon: f32 },
    #[error("parameter golf local reference batch geometry is invalid: {message}")]
    InvalidBatchGeometry { message: String },
    #[error("parameter golf local reference fixture is invalid: {message}")]
    InvalidFixture { message: String },
    #[error(
        "parameter golf local reference fixture vocab size {fixture_vocab_size} does not match model vocab size {model_vocab_size}"
    )]
    FixtureVocabMismatch {
        fixture_vocab_size: usize,
        model_vocab_size: usize,
    },
    #[error("parameter golf coordinate `{parameter_id}`[{flat_index}] is duplicated")]
    DuplicateCoordinate {
        parameter_id: String,
        flat_index: usize,
    },
    #[error(
        "parameter golf coordinate `{parameter_id}`[{flat_index}] exceeds tensor length {parameter_len}"
    )]
    CoordinateOutOfRange {
        parameter_id: String,
        flat_index: usize,
        parameter_len: usize,
    },
    #[error("parameter golf optimizer split did not classify tensor `{parameter_id}`")]
    MissingOptimizerGroup { parameter_id: String },
    #[error("parameter golf local reference runner `{run_id}` already completed")]
    AlreadyCompleted { run_id: String },
    #[error(
        "parameter golf local reference runner ended early: completed {completed_steps} of {max_steps} steps"
    )]
    IncompleteRun {
        completed_steps: u64,
        max_steps: u64,
    },
    #[error("parameter golf artifact is missing tensor `{parameter_id}`")]
    MissingArtifactTensor { parameter_id: String },
    #[error(
        "parameter golf artifact tensor `{parameter_id}` had shape {actual:?}; expected {expected:?}"
    )]
    ArtifactTensorShape {
        parameter_id: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("{context}: {message}")]
    Serialization {
        context: &'static str,
        message: String,
    },
    #[error(transparent)]
    Data(#[from] ParameterGolfDataError),
    #[error(transparent)]
    Model(#[from] ParameterGolfModelError),
    #[error(transparent)]
    Execution(#[from] ParameterGolfExecutionError),
    #[error(transparent)]
    Eval(#[from] ParameterGolfValidationEvalError),
    #[error(transparent)]
    Train(#[from] ParameterGolfTrainError),
    #[error(transparent)]
    Optimizer(#[from] TrainingOptimizerError),
    #[error(transparent)]
    AsyncCheckpointWriteback(#[from] AsyncCheckpointWritebackError),
    #[error(transparent)]
    LocalMetricSink(#[from] LocalTrainMetricSinkError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Clone, Debug, PartialEq)]
enum TrainableTensorRuntime {
    AdamSparse {
        parameter_id: String,
        shape: Vec<usize>,
        selected_indices: Vec<usize>,
        selected_values: Vec<f32>,
        optimizer: TrainingOptimizerConfig,
        optimizer_state: TrainingOptimizerState,
    },
    MuonDense {
        parameter_id: String,
        shape: Vec<usize>,
        selected_indices: Vec<usize>,
        values: Vec<f32>,
        optimizer: ParameterGolfMuonConfig,
        optimizer_state: ParameterGolfMuonState,
    },
}

impl TrainableTensorRuntime {
    fn parameter_id(&self) -> &str {
        match self {
            Self::AdamSparse { parameter_id, .. } | Self::MuonDense { parameter_id, .. } => {
                parameter_id.as_str()
            }
        }
    }

    fn full_values(
        &self,
        baseline_vectors: &BTreeMap<String, ParameterGolfParameterVector>,
    ) -> Vec<f32> {
        match self {
            Self::AdamSparse {
                parameter_id,
                selected_indices,
                selected_values,
                ..
            } => {
                let mut values = baseline_vectors
                    .get(parameter_id)
                    .expect("baseline vector should exist")
                    .values
                    .clone();
                for (flat_index, value) in selected_indices.iter().zip(selected_values.iter()) {
                    values[*flat_index] = *value;
                }
                values
            }
            Self::MuonDense { values, .. } => values.clone(),
        }
    }

    fn checkpoint_tensor(&self) -> ParameterGolfCheckpointTrainableTensor {
        match self {
            Self::AdamSparse {
                parameter_id,
                shape,
                selected_indices,
                optimizer,
                optimizer_state,
                ..
            } => ParameterGolfCheckpointTrainableTensor {
                parameter_id: parameter_id.clone(),
                shape: shape.clone(),
                selected_indices: selected_indices.clone(),
                execution: ParameterGolfOptimizerExecution::Adam {
                    optimizer: optimizer.clone(),
                },
                optimizer_state: ParameterGolfReferenceOptimizerState::Adam {
                    state: optimizer_state.clone(),
                },
            },
            Self::MuonDense {
                parameter_id,
                shape,
                selected_indices,
                optimizer,
                optimizer_state,
                ..
            } => ParameterGolfCheckpointTrainableTensor {
                parameter_id: parameter_id.clone(),
                shape: shape.clone(),
                selected_indices: selected_indices.clone(),
                execution: ParameterGolfOptimizerExecution::Muon {
                    optimizer: optimizer.clone(),
                },
                optimizer_state: ParameterGolfReferenceOptimizerState::Muon {
                    state: optimizer_state.clone(),
                },
            },
        }
    }

    fn apply_gradients(
        &mut self,
        gradients: &[f32],
        learning_rate_multiplier: f32,
        muon_momentum: f32,
        step_number: u64,
    ) -> Result<(), ParameterGolfReferenceTrainingError> {
        match self {
            Self::AdamSparse {
                selected_values,
                optimizer,
                optimizer_state,
                ..
            } => {
                let mut effective_optimizer = optimizer.clone();
                effective_optimizer.learning_rate *= learning_rate_multiplier;
                effective_optimizer.apply_step(
                    selected_values.as_mut_slice(),
                    gradients,
                    optimizer_state,
                    step_number,
                )?;
                Ok(())
            }
            Self::MuonDense {
                shape,
                values,
                optimizer,
                optimizer_state,
                ..
            } => {
                let mut effective_optimizer = optimizer.clone();
                effective_optimizer.learning_rate *= learning_rate_multiplier;
                effective_optimizer.momentum = muon_momentum;
                apply_parameter_golf_muon_step(
                    values.as_mut_slice(),
                    shape.as_slice(),
                    gradients,
                    &effective_optimizer,
                    optimizer_state,
                )?;
                Ok(())
            }
        }
    }
}

/// Stepwise in-memory runner for the bounded Parameter Golf local-reference lane.
#[derive(Debug)]
pub struct ParameterGolfReferenceTrainingRunner {
    fixture: ParameterGolfLocalReferenceFixture,
    config: ParameterGolfReferenceTrainingConfig,
    initial_model: ParameterGolfReferenceModel,
    current_model: ParameterGolfReferenceModel,
    byte_luts: ParameterGolfSentencePieceByteLuts,
    baseline_vectors: BTreeMap<String, ParameterGolfParameterVector>,
    trainable_tensors: Vec<TrainableTensorRuntime>,
    optimizer_plan_digest: String,
    completed_steps: u64,
    step_metrics: Vec<ParameterGolfReferenceTrainingStepMetrics>,
    initial_checkpoint: ParameterGolfCheckpointArtifact,
    latest_checkpoint: ParameterGolfCheckpointArtifact,
    initial_validation_eval: ParameterGolfValidationEvalReport,
    current_validation_eval: ParameterGolfValidationEvalReport,
}

impl ParameterGolfReferenceTrainingRunner {
    /// Seeds one bounded local-reference runner.
    pub fn new(
        fixture: &ParameterGolfLocalReferenceFixture,
        config: &ParameterGolfReferenceTrainingConfig,
    ) -> Result<Self, ParameterGolfReferenceTrainingError> {
        config.validate()?;
        fixture.validate()?;

        let initial_model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
        if fixture.tokenizer_vocab_size != initial_model.descriptor().config.vocab_size {
            return Err(ParameterGolfReferenceTrainingError::FixtureVocabMismatch {
                fixture_vocab_size: fixture.tokenizer_vocab_size,
                model_vocab_size: initial_model.descriptor().config.vocab_size,
            });
        }
        if fixture.training_tokens.len() < config.geometry.local_train_batch_tokens() + 1 {
            return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
                message: format!(
                    "training_tokens must contain at least {} tokens",
                    config.geometry.local_train_batch_tokens() + 1
                ),
            });
        }
        if fixture.validation_tokens.len() <= config.geometry.train_sequence_length {
            return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
                message: format!(
                    "validation_tokens must contain at least {} tokens",
                    config.geometry.train_sequence_length + 1
                ),
            });
        }

        let optimizer_plan =
            parameter_golf_optimizer_plan(initial_model.descriptor(), &config.hyperparameters)?;
        let optimizer_plan_digest =
            stable_digest(b"psionic_parameter_golf_optimizer_plan|", &optimizer_plan);
        let coordinate_map = coordinate_map(config.selected_coordinates.as_slice())?;
        let mut baseline_vectors = BTreeMap::new();
        let mut trainable_tensors = Vec::with_capacity(coordinate_map.len());
        for (parameter_id, selected_indices) in coordinate_map {
            let parameter = initial_model
                .weights()
                .parameter_vector(&initial_model.descriptor().config, parameter_id.as_str())
                .ok_or_else(
                    || ParameterGolfReferenceTrainingError::MissingOptimizerGroup {
                        parameter_id: parameter_id.clone(),
                    },
                )?;
            validate_selected_indices(&parameter_id, &selected_indices, parameter.values.len())?;
            let group = optimizer_plan
                .groups
                .iter()
                .find(|group| group.tensor_names.iter().any(|name| name == &parameter_id))
                .ok_or_else(
                    || ParameterGolfReferenceTrainingError::MissingOptimizerGroup {
                        parameter_id: parameter_id.clone(),
                    },
                )?;
            baseline_vectors.insert(parameter_id.clone(), parameter.clone());
            match &group.execution {
                ParameterGolfOptimizerExecution::Adam { optimizer } => {
                    let selected_len = selected_indices.len();
                    let selected_values = selected_indices
                        .iter()
                        .map(|flat_index| parameter.values[*flat_index])
                        .collect::<Vec<_>>();
                    trainable_tensors.push(TrainableTensorRuntime::AdamSparse {
                        parameter_id,
                        shape: parameter.shape.dims().to_vec(),
                        selected_indices,
                        selected_values,
                        optimizer: optimizer.clone(),
                        optimizer_state: optimizer.initialize_state(selected_len),
                    });
                }
                ParameterGolfOptimizerExecution::Muon { optimizer } => {
                    let shape = parameter.shape.dims().to_vec();
                    let rows = shape.first().copied().unwrap_or(0);
                    let cols = shape.get(1).copied().unwrap_or(0);
                    trainable_tensors.push(TrainableTensorRuntime::MuonDense {
                        parameter_id,
                        shape,
                        selected_indices,
                        values: parameter.values,
                        optimizer: optimizer.clone(),
                        optimizer_state: ParameterGolfMuonState::zeros(rows, cols),
                    });
                }
            }
        }
        let byte_luts = fixture.byte_luts()?;
        let initial_validation_eval = evaluate_parameter_golf_validation(
            &initial_model,
            fixture.validation_tokens.as_slice(),
            config.geometry.train_sequence_length,
            config.geometry.local_validation_batch_tokens(),
            &byte_luts,
        )?;
        let initial_checkpoint = export_checkpoint(
            &initial_model,
            &initial_model,
            trainable_tensors.as_slice(),
            fixture,
            config,
            0,
            &initial_validation_eval,
            &[],
            optimizer_plan_digest.as_str(),
            None,
        )?;
        Ok(Self {
            fixture: fixture.clone(),
            config: config.clone(),
            initial_model: initial_model.clone(),
            current_model: initial_model,
            byte_luts,
            baseline_vectors,
            trainable_tensors,
            optimizer_plan_digest,
            completed_steps: 0,
            step_metrics: Vec::new(),
            initial_checkpoint: initial_checkpoint.clone(),
            latest_checkpoint: initial_checkpoint,
            initial_validation_eval: initial_validation_eval.clone(),
            current_validation_eval: initial_validation_eval,
        })
    }

    /// Returns whether the runner already exhausted its fixed budget.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completed_steps >= self.config.max_steps
    }

    /// Returns the current stepped model.
    #[must_use]
    pub fn current_model(&self) -> &ParameterGolfReferenceModel {
        &self.current_model
    }

    /// Returns the latest checkpoint artifact.
    #[must_use]
    pub fn latest_checkpoint(&self) -> &ParameterGolfCheckpointArtifact {
        &self.latest_checkpoint
    }

    /// Returns the accumulated per-step metrics.
    #[must_use]
    pub fn step_metrics(&self) -> &[ParameterGolfReferenceTrainingStepMetrics] {
        self.step_metrics.as_slice()
    }

    /// Advances the trainer by exactly one optimizer step.
    pub fn step(&mut self) -> Result<(), ParameterGolfReferenceTrainingError> {
        if self.is_complete() {
            return Err(ParameterGolfReferenceTrainingError::AlreadyCompleted {
                run_id: self.config.run_id.clone(),
            });
        }

        let step_index = self.completed_steps;
        let mut accumulated_gradients = self
            .trainable_tensors
            .iter()
            .map(|tensor| match tensor {
                TrainableTensorRuntime::AdamSparse {
                    selected_values, ..
                } => vec![0.0_f32; selected_values.len()],
                TrainableTensorRuntime::MuonDense { values, .. } => vec![0.0_f32; values.len()],
            })
            .collect::<Vec<_>>();
        let mut microbatch_loss_sum = 0.0_f32;

        for micro_step in 0..self.config.geometry.grad_accum_steps {
            let global_micro_step =
                (step_index as usize * self.config.geometry.grad_accum_steps) + micro_step;
            let (input_ids, target_ids) = take_training_microbatch(
                self.fixture.training_tokens.as_slice(),
                &self.config.geometry,
                global_micro_step,
            )?;
            let microbatch_loss = self
                .current_model
                .loss(input_ids.as_slice(), target_ids.as_slice())?;
            microbatch_loss_sum += microbatch_loss;
            for (tensor_index, tensor) in self.trainable_tensors.iter().enumerate() {
                let gradients = finite_difference_gradients(
                    &self.current_model,
                    tensor,
                    &self.baseline_vectors,
                    self.config.finite_difference_epsilon,
                    input_ids.as_slice(),
                    target_ids.as_slice(),
                )?;
                for (accumulated, gradient) in accumulated_gradients[tensor_index]
                    .iter_mut()
                    .zip(gradients.iter())
                {
                    *accumulated += *gradient / self.config.geometry.grad_accum_steps as f32;
                }
            }
        }

        clip_accumulated_gradients(
            accumulated_gradients.as_mut_slice(),
            self.config.hyperparameters.grad_clip_norm,
        );

        let elapsed_ms = step_index.saturating_mul(self.config.step_duration_ms) as f32;
        let learning_rate_multiplier = self
            .config
            .hyperparameters
            .learning_rate_multiplier(step_index, elapsed_ms);
        let muon_momentum = self
            .config
            .hyperparameters
            .muon_momentum_at_step(step_index);
        let step_number = step_index.saturating_add(1);
        for (tensor, gradients) in self
            .trainable_tensors
            .iter_mut()
            .zip(accumulated_gradients.iter())
        {
            tensor.apply_gradients(
                gradients.as_slice(),
                learning_rate_multiplier,
                muon_momentum,
                step_number,
            )?;
        }

        self.current_model = materialize_model(
            &self.initial_model,
            self.trainable_tensors.as_slice(),
            &self.baseline_vectors,
        )?;
        self.completed_steps = step_number;
        self.current_validation_eval = evaluate_parameter_golf_validation(
            &self.current_model,
            self.fixture.validation_tokens.as_slice(),
            self.config.geometry.train_sequence_length,
            self.config.geometry.local_validation_batch_tokens(),
            &self.byte_luts,
        )?;
        let step_metrics = ParameterGolfReferenceTrainingStepMetrics {
            global_step: step_number,
            mean_microbatch_loss: microbatch_loss_sum
                / self.config.geometry.grad_accum_steps as f32,
            validation_mean_loss: self.current_validation_eval.mean_loss,
            validation_bits_per_byte: self.current_validation_eval.bits_per_byte,
            learning_rate_multiplier,
            muon_momentum,
        };
        self.step_metrics.push(step_metrics);
        self.latest_checkpoint = export_checkpoint(
            &self.initial_model,
            &self.current_model,
            self.trainable_tensors.as_slice(),
            &self.fixture,
            &self.config,
            step_number,
            &self.current_validation_eval,
            self.step_metrics.as_slice(),
            self.optimizer_plan_digest.as_str(),
            Some(&self.latest_checkpoint),
        )?;
        Ok(())
    }

    /// Consumes a completed runner and returns the whole-run outcome.
    pub fn into_outcome(
        self,
    ) -> Result<ParameterGolfReferenceTrainingOutcome, ParameterGolfReferenceTrainingError> {
        if !self.is_complete() {
            return Err(ParameterGolfReferenceTrainingError::IncompleteRun {
                completed_steps: self.completed_steps,
                max_steps: self.config.max_steps,
            });
        }

        let raw_model_artifact = self.latest_checkpoint.weights_artifact.clone();
        let raw_roundtrip_model = restore_parameter_golf_model_from_safetensors(
            &self.initial_model,
            raw_model_artifact.bytes.as_slice(),
        )?;
        let raw_roundtrip_validation_eval = evaluate_parameter_golf_validation(
            &raw_roundtrip_model,
            self.fixture.validation_tokens.as_slice(),
            self.config.geometry.train_sequence_length,
            self.config.geometry.local_validation_batch_tokens(),
            &self.byte_luts,
        )?;
        let int8_zlib_model_artifact = export_int8_zlib_model_artifact(
            &self.current_model,
            self.config.run_id.as_str(),
            self.completed_steps,
        )?;
        let int8_zlib_roundtrip_model = restore_parameter_golf_model_from_int8_zlib(
            &self.initial_model,
            int8_zlib_model_artifact.bytes.as_slice(),
        )?;
        let int8_zlib_roundtrip_validation_eval = evaluate_parameter_golf_validation(
            &int8_zlib_roundtrip_model,
            self.fixture.validation_tokens.as_slice(),
            self.config.geometry.train_sequence_length,
            self.config.geometry.local_validation_batch_tokens(),
            &self.byte_luts,
        )?;
        let summary = ParameterGolfReferenceTrainingSummary {
            initial_validation_mean_loss: self.initial_validation_eval.mean_loss,
            final_validation_mean_loss: self.current_validation_eval.mean_loss,
            final_validation_bits_per_byte: self.current_validation_eval.bits_per_byte,
            raw_roundtrip_validation_mean_loss: raw_roundtrip_validation_eval.mean_loss,
            int8_zlib_roundtrip_validation_mean_loss: int8_zlib_roundtrip_validation_eval.mean_loss,
            initial_checkpoint_manifest_digest: self.initial_checkpoint.manifest.stable_digest(),
            final_checkpoint_manifest_digest: self.latest_checkpoint.manifest.stable_digest(),
            raw_model_artifact_digest: raw_model_artifact.artifact_digest.clone(),
            int8_zlib_model_artifact_digest: int8_zlib_model_artifact.artifact_digest.clone(),
        };
        Ok(ParameterGolfReferenceTrainingOutcome {
            initial_model: self.initial_model,
            trained_model: self.current_model,
            step_metrics: self.step_metrics,
            initial_checkpoint: self.initial_checkpoint,
            final_checkpoint: self.latest_checkpoint,
            raw_model_artifact,
            int8_zlib_model_artifact,
            initial_validation_eval: self.initial_validation_eval,
            final_validation_eval: self.current_validation_eval,
            raw_roundtrip_validation_eval,
            int8_zlib_roundtrip_validation_eval,
            checkpoint_writeback_receipts: Vec::new(),
            summary,
        })
    }
}

/// Runs the bounded local-reference trainer end to end.
pub fn train_parameter_golf_local_reference(
    fixture: &ParameterGolfLocalReferenceFixture,
    config: &ParameterGolfReferenceTrainingConfig,
) -> Result<ParameterGolfReferenceTrainingOutcome, ParameterGolfReferenceTrainingError> {
    let mut runner = ParameterGolfReferenceTrainingRunner::new(fixture, config)?;
    while !runner.is_complete() {
        runner.step()?;
    }
    runner.into_outcome()
}

/// Runs the bounded local-reference trainer end to end and fans typed local
/// telemetry into the provided metric sink surface.
pub fn train_parameter_golf_local_reference_with_metric_sink(
    fixture: &ParameterGolfLocalReferenceFixture,
    config: &ParameterGolfReferenceTrainingConfig,
    metric_sink: &mut LocalTrainMetricFanout,
) -> Result<ParameterGolfReferenceTrainingOutcome, ParameterGolfReferenceTrainingError> {
    let mut runner = ParameterGolfReferenceTrainingRunner::new(fixture, config)?;
    while !runner.is_complete() {
        runner.step()?;
        emit_parameter_golf_step_metrics(
            metric_sink,
            config.run_id.as_str(),
            runner.step_metrics(),
        )?;
    }
    metric_sink.flush()?;
    runner.into_outcome()
}

/// Runs the bounded local-reference trainer end to end and persists each emitted
/// checkpoint through the shared async writeback worker.
pub fn train_parameter_golf_local_reference_with_async_checkpoint_writeback(
    fixture: &ParameterGolfLocalReferenceFixture,
    config: &ParameterGolfReferenceTrainingConfig,
    checkpoint_output_root: &Path,
    writeback_options: AsyncCheckpointWritebackOptions,
) -> Result<ParameterGolfReferenceTrainingOutcome, ParameterGolfReferenceTrainingError> {
    let mut runner = ParameterGolfReferenceTrainingRunner::new(fixture, config)?;
    let max_in_flight_writes = writeback_options.max_in_flight_writes();
    let mut worker = AsyncCheckpointWritebackWorker::new(writeback_options)?;
    let mut pending_tickets = VecDeque::new();
    let mut receipts = Vec::new();

    submit_checkpoint_async_writeback(
        &worker,
        runner.latest_checkpoint(),
        checkpoint_output_root,
        max_in_flight_writes,
        &mut pending_tickets,
        &mut receipts,
    )?;
    while !runner.is_complete() {
        runner.step()?;
        submit_checkpoint_async_writeback(
            &worker,
            runner.latest_checkpoint(),
            checkpoint_output_root,
            max_in_flight_writes,
            &mut pending_tickets,
            &mut receipts,
        )?;
    }
    drain_checkpoint_async_writeback_tickets(&mut pending_tickets, &mut receipts)?;
    let shutdown_receipts = worker.shutdown_flush()?;

    let mut outcome = runner.into_outcome()?;
    outcome.checkpoint_writeback_receipts =
        merge_checkpoint_writeback_receipts(receipts, shutdown_receipts);
    Ok(outcome)
}

fn submit_checkpoint_async_writeback(
    worker: &AsyncCheckpointWritebackWorker,
    checkpoint: &ParameterGolfCheckpointArtifact,
    checkpoint_output_root: &Path,
    max_in_flight_writes: usize,
    pending_tickets: &mut VecDeque<AsyncCheckpointWritebackTicket>,
    receipts: &mut Vec<AsyncCheckpointWritebackReceipt>,
) -> Result<(), ParameterGolfReferenceTrainingError> {
    while pending_tickets.len() >= max_in_flight_writes {
        complete_oldest_checkpoint_async_writeback(pending_tickets, receipts)?;
    }

    let payload = checkpoint_async_writeback_payload(checkpoint, checkpoint_output_root)?;
    match worker.submit(payload) {
        Ok(ticket) => {
            pending_tickets.push_back(ticket);
            Ok(())
        }
        Err(AsyncCheckpointWritebackError::QueueFull { .. }) => {
            complete_oldest_checkpoint_async_writeback(pending_tickets, receipts)?;
            let payload = checkpoint_async_writeback_payload(checkpoint, checkpoint_output_root)?;
            let ticket = worker.submit(payload)?;
            pending_tickets.push_back(ticket);
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn drain_checkpoint_async_writeback_tickets(
    pending_tickets: &mut VecDeque<AsyncCheckpointWritebackTicket>,
    receipts: &mut Vec<AsyncCheckpointWritebackReceipt>,
) -> Result<(), ParameterGolfReferenceTrainingError> {
    while !pending_tickets.is_empty() {
        complete_oldest_checkpoint_async_writeback(pending_tickets, receipts)?;
    }
    Ok(())
}

fn complete_oldest_checkpoint_async_writeback(
    pending_tickets: &mut VecDeque<AsyncCheckpointWritebackTicket>,
    receipts: &mut Vec<AsyncCheckpointWritebackReceipt>,
) -> Result<(), ParameterGolfReferenceTrainingError> {
    let ticket = pending_tickets.pop_front().ok_or_else(|| {
        ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf async checkpoint writeback",
            message: String::from("bounded queue saturated without an in-flight ticket"),
        }
    })?;
    receipts.push(ticket.wait()?);
    Ok(())
}

fn checkpoint_async_writeback_payload(
    checkpoint: &ParameterGolfCheckpointArtifact,
    checkpoint_output_root: &Path,
) -> Result<AsyncCheckpointWritebackPayload, ParameterGolfReferenceTrainingError> {
    let step = checkpoint.manifest.step;
    let checkpoint_ref = checkpoint
        .checkpoint
        .checkpoint_ref
        .clone()
        .unwrap_or_else(|| checkpoint.manifest.checkpoint_ref.clone());
    AsyncCheckpointWritebackPayload::new(
        format!("{}-step-{step:05}", checkpoint.manifest.run_id),
        checkpoint_ref,
        checkpoint.checkpoint.checkpoint_family.clone(),
        checkpoint
            .checkpoint
            .stream_id
            .strip_suffix("/checkpoint_model.safetensors")
            .map_or_else(
                || {
                    checkpoint_output_root
                        .join(checkpoint.manifest.run_id.as_str())
                        .join(format!("step-{step:05}"))
                },
                |stream_prefix| checkpoint_output_root.join(stream_prefix),
            ),
        vec![
            AsyncCheckpointWritebackFile::new(
                "checkpoint_model.safetensors",
                checkpoint.weights_artifact.artifact_digest.clone(),
                checkpoint.weights_artifact.bytes.clone(),
            )?,
            AsyncCheckpointWritebackFile::new(
                "checkpoint_manifest.json",
                checkpoint.manifest_artifact.artifact_digest.clone(),
                checkpoint.manifest_artifact.bytes.clone(),
            )?,
        ],
    )
    .map_err(Into::into)
}

fn emit_parameter_golf_step_metrics(
    metric_sink: &mut LocalTrainMetricFanout,
    run_id: &str,
    step_metrics: &[ParameterGolfReferenceTrainingStepMetrics],
) -> Result<(), ParameterGolfReferenceTrainingError> {
    let Some(step_metrics) = step_metrics.last() else {
        return Ok(());
    };
    let step = step_metrics.global_step;
    metric_sink.record(LocalTrainMetricEvent::new(
        run_id,
        LocalTrainMetricPhase::Train,
        step,
        "mean_microbatch_loss",
        LocalTrainMetricValue::F32(step_metrics.mean_microbatch_loss),
    ))?;
    metric_sink.flush()?;
    metric_sink.record(LocalTrainMetricEvent::new(
        run_id,
        LocalTrainMetricPhase::Validation,
        step,
        "validation_mean_loss",
        LocalTrainMetricValue::F64(step_metrics.validation_mean_loss),
    ))?;
    metric_sink.record(LocalTrainMetricEvent::new(
        run_id,
        LocalTrainMetricPhase::Validation,
        step,
        "validation_bits_per_byte",
        LocalTrainMetricValue::F64(step_metrics.validation_bits_per_byte),
    ))?;
    metric_sink.flush()?;
    Ok(())
}

fn merge_checkpoint_writeback_receipts(
    completed: Vec<AsyncCheckpointWritebackReceipt>,
    shutdown: Vec<AsyncCheckpointWritebackReceipt>,
) -> Vec<AsyncCheckpointWritebackReceipt> {
    let mut seen = BTreeSet::new();
    completed
        .into_iter()
        .chain(shutdown)
        .filter(|receipt| seen.insert(receipt.write_id.clone()))
        .collect()
}

#[cfg(test)]
fn read_parameter_golf_checkpoint_from_directory(
    directory: &Path,
    checkpoint: &TrainingCheckpointReference,
) -> Result<ParameterGolfCheckpointArtifact, ParameterGolfReferenceTrainingError> {
    let weights_path = directory.join("checkpoint_model.safetensors");
    let manifest_path = directory.join("checkpoint_manifest.json");
    let weights_bytes = std::fs::read(weights_path.as_path())?;
    let manifest_bytes = std::fs::read(manifest_path.as_path())?;
    let manifest =
        serde_json::from_slice::<ParameterGolfCheckpointManifest>(manifest_bytes.as_slice())
            .map_err(|error| ParameterGolfReferenceTrainingError::Serialization {
                context: "parameter golf checkpoint manifest import",
                message: error.to_string(),
            })?;
    if manifest.checkpoint_family != checkpoint.checkpoint_family {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf checkpoint import",
            message: format!(
                "checkpoint family mismatch: manifest `{}` vs runtime `{}`",
                manifest.checkpoint_family, checkpoint.checkpoint_family
            ),
        });
    }
    if checkpoint.checkpoint_ref.as_ref() != Some(&manifest.checkpoint_ref) {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf checkpoint import",
            message: format!(
                "checkpoint ref mismatch: manifest `{}` vs runtime `{:?}`",
                manifest.checkpoint_ref, checkpoint.checkpoint_ref
            ),
        });
    }
    if checkpoint.step != Some(manifest.step) {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf checkpoint import",
            message: format!(
                "checkpoint step mismatch: manifest `{}` vs runtime `{:?}`",
                manifest.step, checkpoint.step
            ),
        });
    }
    let weights_artifact = ParameterGolfTrainingArtifact::new(
        "parameter_golf_model_safetensors",
        format!(
            "{}/step-{:05}/checkpoint_model.safetensors",
            manifest.run_id, manifest.step
        ),
        weights_bytes,
    );
    let manifest_artifact = ParameterGolfTrainingArtifact::new(
        "parameter_golf_checkpoint_manifest",
        format!(
            "{}/step-{:05}/checkpoint_manifest.json",
            manifest.run_id, manifest.step
        ),
        manifest_bytes,
    );
    Ok(ParameterGolfCheckpointArtifact {
        weights_artifact,
        manifest_artifact,
        manifest,
        checkpoint: checkpoint.clone(),
    })
}

/// Restores one local-reference runner from a persisted checkpoint.
pub fn restore_parameter_golf_local_reference_checkpoint(
    fixture: &ParameterGolfLocalReferenceFixture,
    checkpoint: &ParameterGolfCheckpointArtifact,
) -> Result<ParameterGolfReferenceTrainingRunner, ParameterGolfReferenceTrainingError> {
    fixture.validate()?;
    let manifest = &checkpoint.manifest;
    let config = ParameterGolfReferenceTrainingConfig {
        run_id: manifest.run_id.clone(),
        checkpoint_family: manifest.checkpoint_family.clone(),
        started_at_ms: manifest.started_at_ms,
        step_duration_ms: manifest.step_duration_ms,
        max_steps: manifest.max_steps,
        geometry: manifest.geometry.clone(),
        hyperparameters: manifest.hyperparameters.clone(),
        finite_difference_epsilon: manifest.finite_difference_epsilon,
        selected_coordinates: manifest
            .trainable_tensors
            .iter()
            .flat_map(|tensor| {
                tensor
                    .selected_indices
                    .iter()
                    .map(|flat_index| ParameterGolfTrainableCoordinate {
                        parameter_id: tensor.parameter_id.clone(),
                        flat_index: *flat_index,
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    };
    config.validate()?;
    if fixture.training_digest() != manifest.training_dataset_digest {
        return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
            message: String::from("training fixture digest does not match checkpoint manifest"),
        });
    }
    if fixture.validation_digest() != manifest.validation_dataset_digest {
        return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
            message: String::from("validation fixture digest does not match checkpoint manifest"),
        });
    }

    let initial_model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    if initial_model.descriptor().stable_digest() != manifest.base_descriptor_digest {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf checkpoint restore",
            message: String::from("baseline descriptor digest does not match checkpoint manifest"),
        });
    }
    let optimizer_plan =
        parameter_golf_optimizer_plan(initial_model.descriptor(), &manifest.hyperparameters)?;
    let optimizer_plan_digest =
        stable_digest(b"psionic_parameter_golf_optimizer_plan|", &optimizer_plan);
    if optimizer_plan_digest != manifest.optimizer_plan_digest {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf checkpoint restore",
            message: String::from("optimizer plan digest does not match checkpoint manifest"),
        });
    }

    let current_model = restore_parameter_golf_model_from_safetensors(
        &initial_model,
        checkpoint.weights_artifact.bytes.as_slice(),
    )?;
    if current_model.descriptor().stable_digest() != manifest.current_descriptor_digest {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf checkpoint restore",
            message: String::from("current descriptor digest does not match checkpoint manifest"),
        });
    }

    let mut baseline_vectors = BTreeMap::new();
    let mut trainable_tensors = Vec::with_capacity(manifest.trainable_tensors.len());
    for tensor in &manifest.trainable_tensors {
        let baseline_vector = initial_model
            .weights()
            .parameter_vector(
                &initial_model.descriptor().config,
                tensor.parameter_id.as_str(),
            )
            .ok_or_else(
                || ParameterGolfReferenceTrainingError::MissingOptimizerGroup {
                    parameter_id: tensor.parameter_id.clone(),
                },
            )?;
        validate_selected_indices(
            tensor.parameter_id.as_str(),
            tensor.selected_indices.as_slice(),
            baseline_vector.values.len(),
        )?;
        baseline_vectors.insert(tensor.parameter_id.clone(), baseline_vector);
        let current_vector = current_model
            .weights()
            .parameter_vector(
                &current_model.descriptor().config,
                tensor.parameter_id.as_str(),
            )
            .ok_or_else(
                || ParameterGolfReferenceTrainingError::MissingArtifactTensor {
                    parameter_id: tensor.parameter_id.clone(),
                },
            )?;
        match (&tensor.execution, &tensor.optimizer_state) {
            (
                ParameterGolfOptimizerExecution::Adam { optimizer },
                ParameterGolfReferenceOptimizerState::Adam { state },
            ) => {
                let selected_values = tensor
                    .selected_indices
                    .iter()
                    .map(|flat_index| current_vector.values[*flat_index])
                    .collect::<Vec<_>>();
                trainable_tensors.push(TrainableTensorRuntime::AdamSparse {
                    parameter_id: tensor.parameter_id.clone(),
                    shape: tensor.shape.clone(),
                    selected_indices: tensor.selected_indices.clone(),
                    selected_values,
                    optimizer: optimizer.clone(),
                    optimizer_state: state.clone(),
                });
            }
            (
                ParameterGolfOptimizerExecution::Muon { optimizer },
                ParameterGolfReferenceOptimizerState::Muon { state },
            ) => {
                trainable_tensors.push(TrainableTensorRuntime::MuonDense {
                    parameter_id: tensor.parameter_id.clone(),
                    shape: tensor.shape.clone(),
                    selected_indices: tensor.selected_indices.clone(),
                    values: current_vector.values,
                    optimizer: optimizer.clone(),
                    optimizer_state: state.clone(),
                });
            }
            _ => {
                return Err(ParameterGolfReferenceTrainingError::Serialization {
                    context: "parameter golf checkpoint restore",
                    message: format!(
                        "optimizer execution/state mismatch for tensor `{}`",
                        tensor.parameter_id
                    ),
                });
            }
        }
    }

    let byte_luts = fixture.byte_luts()?;
    let initial_validation_eval = evaluate_parameter_golf_validation(
        &initial_model,
        fixture.validation_tokens.as_slice(),
        config.geometry.train_sequence_length,
        config.geometry.local_validation_batch_tokens(),
        &byte_luts,
    )?;
    let current_validation_eval = evaluate_parameter_golf_validation(
        &current_model,
        fixture.validation_tokens.as_slice(),
        config.geometry.train_sequence_length,
        config.geometry.local_validation_batch_tokens(),
        &byte_luts,
    )?;
    if current_validation_eval.stable_digest() != manifest.validation_eval_digest {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf checkpoint restore",
            message: String::from("validation eval digest does not match checkpoint manifest"),
        });
    }
    let initial_checkpoint = export_checkpoint(
        &initial_model,
        &initial_model,
        trainable_tensors.as_slice(),
        fixture,
        &config,
        0,
        &initial_validation_eval,
        &[],
        optimizer_plan_digest.as_str(),
        None,
    )?;
    Ok(ParameterGolfReferenceTrainingRunner {
        fixture: fixture.clone(),
        config,
        initial_model,
        current_model,
        byte_luts,
        baseline_vectors,
        trainable_tensors,
        optimizer_plan_digest,
        completed_steps: manifest.step,
        step_metrics: manifest.step_metrics.clone(),
        initial_checkpoint,
        latest_checkpoint: checkpoint.clone(),
        initial_validation_eval,
        current_validation_eval,
    })
}

/// Restores one exact raw full-precision model export.
pub fn restore_parameter_golf_model_from_safetensors(
    baseline_model: &ParameterGolfReferenceModel,
    weights_bytes: &[u8],
) -> Result<ParameterGolfReferenceModel, ParameterGolfReferenceTrainingError> {
    let (_, metadata) = SafeTensors::read_metadata(weights_bytes).map_err(|error| {
        ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf raw safetensors restore",
            message: error.to_string(),
        }
    })?;
    let safetensors = SafeTensors::deserialize(weights_bytes).map_err(|error| {
        ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf raw safetensors restore",
            message: error.to_string(),
        }
    })?;
    let weight_surface = metadata
        .metadata()
        .as_ref()
        .and_then(|metadata| metadata.get(PARAMETER_GOLF_WEIGHT_SURFACE_KEY))
        .map(String::as_str)
        .unwrap_or(PARAMETER_GOLF_SPLIT_WEIGHT_SURFACE);
    match weight_surface {
        PARAMETER_GOLF_SPLIT_WEIGHT_SURFACE => {
            restore_parameter_golf_model_from_split_safetensors(baseline_model, &safetensors)
        }
        PARAMETER_GOLF_BANKED_WEIGHT_SURFACE => {
            restore_parameter_golf_model_from_banked_safetensors_impl(baseline_model, &safetensors)
        }
        other => Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf raw safetensors restore",
            message: format!("unsupported parameter golf weight surface `{other}`"),
        }),
    }
}

fn restore_parameter_golf_model_from_split_safetensors(
    baseline_model: &ParameterGolfReferenceModel,
    safetensors: &SafeTensors<'_>,
) -> Result<ParameterGolfReferenceModel, ParameterGolfReferenceTrainingError> {
    let mut overrides = BTreeMap::new();
    for parameter in baseline_model
        .weights()
        .parameter_vectors(&baseline_model.descriptor().config)
    {
        let tensor = safetensors
            .tensor(parameter.parameter_id.as_str())
            .map_err(
                |_| ParameterGolfReferenceTrainingError::MissingArtifactTensor {
                    parameter_id: parameter.parameter_id.clone(),
                },
            )?;
        validate_tensor_shape(
            parameter.parameter_id.as_str(),
            tensor.shape(),
            parameter.shape.dims(),
        )?;
        overrides.insert(
            parameter.parameter_id.clone(),
            decode_float_tensor(
                parameter.parameter_id.as_str(),
                tensor.dtype(),
                tensor.data(),
                tensor.shape(),
            )?,
        );
    }
    let weights = baseline_model
        .weights()
        .with_parameter_overrides(&baseline_model.descriptor().config, &overrides)?;
    Ok(ParameterGolfReferenceModel::new(
        baseline_model.descriptor().model.clone(),
        baseline_model.descriptor().config.clone(),
        weights,
    )?)
}

fn restore_parameter_golf_model_from_banked_safetensors_impl(
    baseline_model: &ParameterGolfReferenceModel,
    safetensors: &SafeTensors<'_>,
) -> Result<ParameterGolfReferenceModel, ParameterGolfReferenceTrainingError> {
    let config = &baseline_model.descriptor().config;
    let banked_weights = baseline_model.banked_weights()?;
    let mut overrides = BTreeMap::new();
    for parameter in banked_weights.parameter_vectors(config) {
        let tensor = safetensors
            .tensor(parameter.parameter_id.as_str())
            .map_err(
                |_| ParameterGolfReferenceTrainingError::MissingArtifactTensor {
                    parameter_id: parameter.parameter_id.clone(),
                },
            )?;
        validate_tensor_shape(
            parameter.parameter_id.as_str(),
            tensor.shape(),
            parameter.shape.dims(),
        )?;
        overrides.insert(
            parameter.parameter_id.clone(),
            decode_float_tensor(
                parameter.parameter_id.as_str(),
                tensor.dtype(),
                tensor.data(),
                tensor.shape(),
            )?,
        );
    }
    let weights = banked_weights
        .with_parameter_overrides(config, &overrides)?
        .to_split(config)?;
    Ok(ParameterGolfReferenceModel::new(
        baseline_model.descriptor().model.clone(),
        config.clone(),
        weights,
    )?)
}

/// Restores one int8+zlib model export back into the full-precision reference family.
pub fn restore_parameter_golf_model_from_int8_zlib(
    baseline_model: &ParameterGolfReferenceModel,
    artifact_bytes: &[u8],
) -> Result<ParameterGolfReferenceModel, ParameterGolfReferenceTrainingError> {
    let mut decoder = ZlibDecoder::new(artifact_bytes);
    let mut raw_bytes = Vec::new();
    decoder.read_to_end(&mut raw_bytes)?;
    let safetensors = SafeTensors::deserialize(raw_bytes.as_slice()).map_err(|error| {
        ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf int8+zlib restore",
            message: error.to_string(),
        }
    })?;
    let mut overrides = BTreeMap::new();
    for parameter in baseline_model
        .weights()
        .parameter_vectors(&baseline_model.descriptor().config)
    {
        let parameter_id = parameter.parameter_id.as_str();
        let restored = match safetensors.tensor(parameter_id) {
            Ok(tensor) => {
                validate_tensor_shape(parameter_id, tensor.shape(), parameter.shape.dims())?;
                decode_float_tensor(parameter_id, tensor.dtype(), tensor.data(), tensor.shape())?
            }
            Err(_) => {
                let quantized_name = format!("{parameter_id}.__q");
                let scale_name = format!("{parameter_id}.__scale");
                let quantized = safetensors.tensor(quantized_name.as_str()).map_err(|_| {
                    ParameterGolfReferenceTrainingError::MissingArtifactTensor {
                        parameter_id: parameter.parameter_id.clone(),
                    }
                })?;
                let scale = safetensors.tensor(scale_name.as_str()).map_err(|_| {
                    ParameterGolfReferenceTrainingError::MissingArtifactTensor {
                        parameter_id: scale_name.clone(),
                    }
                })?;
                validate_tensor_shape(parameter_id, quantized.shape(), parameter.shape.dims())?;
                dequantize_int8_tensor(
                    parameter_id,
                    parameter.shape.dims(),
                    quantized.data(),
                    scale.dtype(),
                    scale.data(),
                    scale.shape(),
                )?
            }
        };
        overrides.insert(parameter.parameter_id.clone(), restored);
    }
    let weights = baseline_model
        .weights()
        .with_parameter_overrides(&baseline_model.descriptor().config, &overrides)?;
    Ok(ParameterGolfReferenceModel::new(
        baseline_model.descriptor().model.clone(),
        baseline_model.descriptor().config.clone(),
        weights,
    )?)
}

fn coordinate_map(
    coordinates: &[ParameterGolfTrainableCoordinate],
) -> Result<BTreeMap<String, Vec<usize>>, ParameterGolfReferenceTrainingError> {
    let mut seen = BTreeSet::new();
    let mut grouped = BTreeMap::new();
    for coordinate in coordinates {
        let key = format!("{}:{}", coordinate.parameter_id, coordinate.flat_index);
        if !seen.insert(key) {
            return Err(ParameterGolfReferenceTrainingError::DuplicateCoordinate {
                parameter_id: coordinate.parameter_id.clone(),
                flat_index: coordinate.flat_index,
            });
        }
        grouped
            .entry(coordinate.parameter_id.clone())
            .or_insert_with(Vec::new)
            .push(coordinate.flat_index);
    }
    for values in grouped.values_mut() {
        values.sort_unstable();
    }
    Ok(grouped)
}

fn validate_selected_indices(
    parameter_id: &str,
    selected_indices: &[usize],
    parameter_len: usize,
) -> Result<(), ParameterGolfReferenceTrainingError> {
    for &flat_index in selected_indices {
        if flat_index >= parameter_len {
            return Err(ParameterGolfReferenceTrainingError::CoordinateOutOfRange {
                parameter_id: String::from(parameter_id),
                flat_index,
                parameter_len,
            });
        }
    }
    Ok(())
}

fn materialize_model(
    initial_model: &ParameterGolfReferenceModel,
    trainable_tensors: &[TrainableTensorRuntime],
    baseline_vectors: &BTreeMap<String, ParameterGolfParameterVector>,
) -> Result<ParameterGolfReferenceModel, ParameterGolfReferenceTrainingError> {
    let mut overrides = BTreeMap::new();
    for tensor in trainable_tensors {
        overrides.insert(
            String::from(tensor.parameter_id()),
            tensor.full_values(baseline_vectors),
        );
    }
    let weights = initial_model
        .weights()
        .with_parameter_overrides(&initial_model.descriptor().config, &overrides)?;
    Ok(ParameterGolfReferenceModel::new(
        initial_model.descriptor().model.clone(),
        initial_model.descriptor().config.clone(),
        weights,
    )?)
}

fn take_training_microbatch(
    training_tokens: &[u16],
    geometry: &ParameterGolfBatchGeometry,
    global_micro_step: usize,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>), ParameterGolfReferenceTrainingError> {
    let per_rank_span = geometry.local_train_batch_tokens() + 1;
    let start = (global_micro_step * per_rank_span) % training_tokens.len();
    let chunk = wraparound_slice(training_tokens, start, per_rank_span);
    microbatch_as_sequences(chunk.as_slice(), geometry.train_sequence_length)
}

fn wraparound_slice(tokens: &[u16], start: usize, len: usize) -> Vec<u16> {
    if start + len <= tokens.len() {
        return tokens[start..start + len].to_vec();
    }
    let mut output = Vec::with_capacity(len);
    output.extend_from_slice(&tokens[start..]);
    output.extend_from_slice(&tokens[..len - (tokens.len() - start)]);
    output
}

fn microbatch_as_sequences(
    tokens: &[u16],
    seq_len: usize,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>), ParameterGolfReferenceTrainingError> {
    if tokens.len() <= seq_len {
        return Err(ParameterGolfReferenceTrainingError::InvalidFixture {
            message: format!(
                "microbatch requires at least {} tokens, found {}",
                seq_len + 1,
                tokens.len()
            ),
        });
    }
    let input_ids = tokens[..tokens.len() - 1]
        .chunks(seq_len)
        .map(|row| {
            row.iter()
                .map(|token| u32::from(*token))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let target_ids = tokens[1..]
        .chunks(seq_len)
        .map(|row| {
            row.iter()
                .map(|token| u32::from(*token))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok((input_ids, target_ids))
}

fn finite_difference_gradients(
    current_model: &ParameterGolfReferenceModel,
    tensor: &TrainableTensorRuntime,
    baseline_vectors: &BTreeMap<String, ParameterGolfParameterVector>,
    epsilon: f32,
    input_ids: &[Vec<u32>],
    target_ids: &[Vec<u32>],
) -> Result<Vec<f32>, ParameterGolfReferenceTrainingError> {
    let parameter_id = tensor.parameter_id();
    let current_values = tensor.full_values(baseline_vectors);
    match tensor {
        TrainableTensorRuntime::AdamSparse {
            selected_indices, ..
        } => {
            let mut gradients = vec![0.0_f32; selected_indices.len()];
            for (gradient_index, flat_index) in selected_indices.iter().enumerate() {
                gradients[gradient_index] = symmetric_finite_difference(
                    current_model,
                    parameter_id,
                    current_values.as_slice(),
                    *flat_index,
                    epsilon,
                    input_ids,
                    target_ids,
                )?;
            }
            Ok(gradients)
        }
        TrainableTensorRuntime::MuonDense {
            selected_indices, ..
        } => {
            let mut gradients = vec![0.0_f32; current_values.len()];
            for flat_index in selected_indices {
                gradients[*flat_index] = symmetric_finite_difference(
                    current_model,
                    parameter_id,
                    current_values.as_slice(),
                    *flat_index,
                    epsilon,
                    input_ids,
                    target_ids,
                )?;
            }
            Ok(gradients)
        }
    }
}

fn symmetric_finite_difference(
    current_model: &ParameterGolfReferenceModel,
    parameter_id: &str,
    values: &[f32],
    flat_index: usize,
    epsilon: f32,
    input_ids: &[Vec<u32>],
    target_ids: &[Vec<u32>],
) -> Result<f32, ParameterGolfReferenceTrainingError> {
    let mut plus = values.to_vec();
    plus[flat_index] += epsilon;
    let plus_loss =
        loss_with_parameter_override(current_model, parameter_id, plus, input_ids, target_ids)?;
    let mut minus = values.to_vec();
    minus[flat_index] -= epsilon;
    let minus_loss =
        loss_with_parameter_override(current_model, parameter_id, minus, input_ids, target_ids)?;
    Ok((plus_loss - minus_loss) / (2.0 * epsilon))
}

fn loss_with_parameter_override(
    current_model: &ParameterGolfReferenceModel,
    parameter_id: &str,
    values: Vec<f32>,
    input_ids: &[Vec<u32>],
    target_ids: &[Vec<u32>],
) -> Result<f32, ParameterGolfReferenceTrainingError> {
    let mut overrides = BTreeMap::new();
    overrides.insert(String::from(parameter_id), values);
    let weights = current_model
        .weights()
        .with_parameter_overrides(&current_model.descriptor().config, &overrides)?;
    let perturbed = ParameterGolfReferenceModel::new(
        current_model.descriptor().model.clone(),
        current_model.descriptor().config.clone(),
        weights,
    )?;
    Ok(perturbed.loss(input_ids, target_ids)?)
}

fn clip_accumulated_gradients(accumulated_gradients: &mut [Vec<f32>], grad_clip_norm: f32) {
    if !(grad_clip_norm.is_finite() && grad_clip_norm > 0.0) {
        return;
    }
    let norm = accumulated_gradients
        .iter()
        .flat_map(|values| values.iter())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= grad_clip_norm || norm <= f32::EPSILON {
        return;
    }
    let scale = grad_clip_norm / norm;
    for values in accumulated_gradients {
        for value in values {
            *value *= scale;
        }
    }
}

fn export_checkpoint(
    initial_model: &ParameterGolfReferenceModel,
    current_model: &ParameterGolfReferenceModel,
    trainable_tensors: &[TrainableTensorRuntime],
    fixture: &ParameterGolfLocalReferenceFixture,
    config: &ParameterGolfReferenceTrainingConfig,
    step: u64,
    validation_eval: &ParameterGolfValidationEvalReport,
    step_metrics: &[ParameterGolfReferenceTrainingStepMetrics],
    optimizer_plan_digest: &str,
    previous: Option<&ParameterGolfCheckpointArtifact>,
) -> Result<ParameterGolfCheckpointArtifact, ParameterGolfReferenceTrainingError> {
    let weights_artifact = ParameterGolfTrainingArtifact::new(
        "parameter_golf_model_safetensors",
        format!(
            "{}/step-{step:05}/checkpoint_model.safetensors",
            config.run_id
        ),
        export_full_precision_model_bytes(current_model)?,
    );
    let checkpoint_ref = format!("{}:step-{step:05}", config.run_id);
    let manifest = ParameterGolfCheckpointManifest {
        schema_version: 1,
        checkpoint_ref: checkpoint_ref.clone(),
        checkpoint_family: config.checkpoint_family.clone(),
        run_id: config.run_id.clone(),
        step,
        started_at_ms: config.started_at_ms,
        step_duration_ms: config.step_duration_ms,
        max_steps: config.max_steps,
        geometry: config.geometry.clone(),
        hyperparameters: config.hyperparameters.clone(),
        finite_difference_epsilon: config.finite_difference_epsilon,
        base_descriptor_digest: initial_model.descriptor().stable_digest(),
        current_descriptor_digest: current_model.descriptor().stable_digest(),
        optimizer_plan_digest: String::from(optimizer_plan_digest),
        training_dataset_digest: fixture.training_digest(),
        validation_dataset_digest: fixture.validation_digest(),
        validation_eval_digest: validation_eval.stable_digest(),
        step_metrics: step_metrics.to_vec(),
        trainable_tensors: trainable_tensors
            .iter()
            .map(TrainableTensorRuntime::checkpoint_tensor)
            .collect(),
        parent_checkpoint_ref: previous
            .and_then(|artifact| artifact.checkpoint.checkpoint_ref.clone()),
        parent_manifest_digest: previous.map(|artifact| artifact.manifest.stable_digest()),
    };
    let manifest_artifact = ParameterGolfTrainingArtifact::new(
        "parameter_golf_checkpoint_manifest",
        format!("{}/step-{step:05}/checkpoint_manifest.json", config.run_id),
        serde_json::to_vec_pretty(&manifest).map_err(|error| {
            ParameterGolfReferenceTrainingError::Serialization {
                context: "parameter golf checkpoint manifest export",
                message: error.to_string(),
            }
        })?,
    );
    let started_at_ms = config
        .started_at_ms
        .saturating_add(step.saturating_mul(config.step_duration_ms));
    let cluster_state_digest = stable_digest(
        b"psionic_parameter_golf_local_reference_cluster|",
        &config.run_id,
    );
    let topology_digest = stable_digest(
        b"psionic_parameter_golf_local_reference_topology|",
        &config.geometry,
    );
    let checkpoint = TrainingCheckpointReference::new(
        config.checkpoint_family.clone(),
        weights_artifact.artifact_ref.clone(),
        manifest_artifact.artifact_digest.clone(),
        weights_artifact.artifact_digest.clone(),
        "local-reference",
        0,
        cluster_state_digest,
        topology_digest,
        started_at_ms,
    )
    .with_checkpoint_ref(checkpoint_ref)
    .with_step(step)
    .with_durable_at_ms(started_at_ms);
    Ok(ParameterGolfCheckpointArtifact {
        weights_artifact,
        manifest_artifact,
        manifest,
        checkpoint,
    })
}

fn export_full_precision_model_bytes(
    model: &ParameterGolfReferenceModel,
) -> Result<Vec<u8>, ParameterGolfReferenceTrainingError> {
    let mut metadata = HashMap::new();
    metadata.insert(
        String::from(PARAMETER_GOLF_CHECKPOINT_MANIFEST_KEY),
        model.descriptor().stable_digest(),
    );
    metadata.insert(
        String::from(PARAMETER_GOLF_WEIGHT_SURFACE_KEY),
        String::from(PARAMETER_GOLF_SPLIT_WEIGHT_SURFACE),
    );
    let raw_tensors = model
        .weights()
        .parameter_vectors(&model.descriptor().config)
        .into_iter()
        .map(|parameter| {
            (
                parameter.parameter_id,
                SafeTensorsDType::F32,
                parameter.shape.dims().to_vec(),
                encode_f32_bytes(parameter.values.as_slice()),
            )
        })
        .collect::<Vec<_>>();
    serialize_tensors(
        raw_tensors,
        Some(metadata),
        "parameter golf raw safetensors export",
    )
}

fn export_int8_zlib_model_artifact(
    model: &ParameterGolfReferenceModel,
    run_id: &str,
    step: u64,
) -> Result<ParameterGolfTrainingArtifact, ParameterGolfReferenceTrainingError> {
    let mut encoded_tensors = Vec::new();
    let mut metadata = HashMap::new();
    metadata.insert(
        String::from("psionic.parameter_golf.quant_format"),
        String::from(PARAMETER_GOLF_INT8_ZLIB_FORMAT),
    );
    for parameter in model
        .weights()
        .parameter_vectors(&model.descriptor().config)
    {
        let shape = parameter.shape.dims().to_vec();
        if keep_float_tensor(parameter.parameter_id.as_str(), parameter.values.len()) {
            if is_control_tensor_name(parameter.parameter_id.as_str()) {
                encoded_tensors.push((
                    parameter.parameter_id,
                    SafeTensorsDType::F32,
                    shape,
                    encode_f32_bytes(parameter.values.as_slice()),
                ));
            } else {
                encoded_tensors.push((
                    parameter.parameter_id,
                    SafeTensorsDType::F16,
                    shape,
                    encode_f16_bytes(parameter.values.as_slice()),
                ));
            }
            continue;
        }

        let quantized_name = format!("{}.__q", parameter.parameter_id);
        let scale_name = format!("{}.__scale", parameter.parameter_id);
        let (quantized_bytes, scale_bytes, scale_shape) =
            quantize_int8_tensor(parameter.values.as_slice(), shape.as_slice());
        encoded_tensors.push((quantized_name, SafeTensorsDType::I8, shape, quantized_bytes));
        encoded_tensors.push((scale_name, SafeTensorsDType::F16, scale_shape, scale_bytes));
    }
    let raw = serialize_tensors(
        encoded_tensors,
        Some(metadata),
        "parameter golf int8 safetensors export",
    )?;
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(raw.as_slice())?;
    let compressed = encoder.finish()?;
    Ok(ParameterGolfTrainingArtifact::new(
        "parameter_golf_model_int8_zlib",
        format!("{run_id}/step-{step:05}/final_model.int8.ptz"),
        compressed,
    ))
}

/// Exports one Parameter Golf model into the canonical int8-plus-zlib artifact
/// surface used by the bounded benchmark and submission lanes.
pub fn export_parameter_golf_int8_zlib_model_artifact(
    model: &ParameterGolfReferenceModel,
    run_id: &str,
    step: u64,
) -> Result<ParameterGolfTrainingArtifact, ParameterGolfReferenceTrainingError> {
    export_int8_zlib_model_artifact(model, run_id, step)
}

/// Exports one Parameter Golf model into the canonical raw safetensors bytes
/// surface used by checkpoint and distributed-runtime seams.
pub fn export_parameter_golf_full_precision_model_bytes(
    model: &ParameterGolfReferenceModel,
) -> Result<Vec<u8>, ParameterGolfReferenceTrainingError> {
    export_full_precision_model_bytes(model)
}

/// Exports one Parameter Golf model into the upstream-style banked full-precision
/// safetensors surface used by the distributed score path.
pub fn export_parameter_golf_banked_full_precision_model_bytes(
    model: &ParameterGolfReferenceModel,
) -> Result<Vec<u8>, ParameterGolfReferenceTrainingError> {
    let banked_weights = model.banked_weights()?;
    export_parameter_golf_banked_full_precision_weights_bytes(model, &banked_weights)
}

/// Exports one explicit banked runtime-weight surface into the canonical
/// safetensors bytes used by the distributed score path.
pub fn export_parameter_golf_banked_full_precision_weights_bytes(
    baseline_model: &ParameterGolfReferenceModel,
    banked_weights: &ParameterGolfBankedWeights,
) -> Result<Vec<u8>, ParameterGolfReferenceTrainingError> {
    let mut metadata = HashMap::new();
    metadata.insert(
        String::from(PARAMETER_GOLF_CHECKPOINT_MANIFEST_KEY),
        baseline_model.descriptor().stable_digest(),
    );
    metadata.insert(
        String::from(PARAMETER_GOLF_WEIGHT_SURFACE_KEY),
        String::from(PARAMETER_GOLF_BANKED_WEIGHT_SURFACE),
    );
    let raw_tensors = banked_weights
        .parameter_vectors(&baseline_model.descriptor().config)
        .into_iter()
        .map(|parameter| {
            (
                parameter.parameter_id,
                SafeTensorsDType::F32,
                parameter.shape.dims().to_vec(),
                encode_f32_bytes(parameter.values.as_slice()),
            )
        })
        .collect::<Vec<_>>();
    serialize_tensors(
        raw_tensors,
        Some(metadata),
        "parameter golf raw banked safetensors export",
    )
}

fn serialize_tensors(
    tensors: Vec<(String, SafeTensorsDType, Vec<usize>, Vec<u8>)>,
    metadata: Option<HashMap<String, String>>,
    context: &'static str,
) -> Result<Vec<u8>, ParameterGolfReferenceTrainingError> {
    let mut views = Vec::with_capacity(tensors.len());
    for (name, dtype, shape, bytes) in &tensors {
        let view = TensorView::new(*dtype, shape.clone(), bytes.as_slice()).map_err(|error| {
            ParameterGolfReferenceTrainingError::Serialization {
                context,
                message: error.to_string(),
            }
        })?;
        views.push((name.clone(), view));
    }
    serialize(
        views
            .iter()
            .map(|(name, view)| (name.as_str(), view.clone())),
        metadata,
    )
    .map_err(|error| ParameterGolfReferenceTrainingError::Serialization {
        context,
        message: error.to_string(),
    })
}

fn keep_float_tensor(parameter_id: &str, parameter_len: usize) -> bool {
    is_control_tensor_name(parameter_id)
        || parameter_len <= PARAMETER_GOLF_INT8_KEEP_FLOAT_MAX_NUMEL
}

fn is_control_tensor_name(parameter_id: &str) -> bool {
    PARAMETER_GOLF_CONTROL_TENSOR_NAME_PATTERNS
        .iter()
        .any(|pattern| parameter_id.contains(pattern))
}

fn quantize_int8_tensor(values: &[f32], shape: &[usize]) -> (Vec<u8>, Vec<u8>, Vec<usize>) {
    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        let mut quantized = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(rows);
        for row in 0..rows {
            let row_values = &values[row * cols..(row + 1) * cols];
            let clip_abs = quantile_abs(row_values, PARAMETER_GOLF_INT8_CLIP_Q);
            let scale = if clip_abs > 0.0 {
                clip_abs / 127.0
            } else {
                1.0
            };
            scales.push(scale);
            for value in row_values {
                let clipped = value.clamp(-clip_abs, clip_abs);
                let q = (clipped / scale).round().clamp(-127.0, 127.0) as i8;
                quantized.push(q as u8);
            }
        }
        return (quantized, encode_f16_bytes(scales.as_slice()), vec![rows]);
    }

    let clip_abs = quantile_abs(values, PARAMETER_GOLF_INT8_CLIP_Q);
    let scale = if clip_abs > 0.0 {
        clip_abs / 127.0
    } else {
        1.0
    };
    let quantized = values
        .iter()
        .map(|value| {
            let clipped = value.clamp(-clip_abs, clip_abs);
            (clipped / scale).round().clamp(-127.0, 127.0) as i8 as u8
        })
        .collect::<Vec<_>>();
    (quantized, encode_f16_bytes(&[scale]), vec![1])
}

fn dequantize_int8_tensor(
    parameter_id: &str,
    expected_shape: &[usize],
    quantized_bytes: &[u8],
    scale_dtype: SafeTensorsDType,
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> Result<Vec<f32>, ParameterGolfReferenceTrainingError> {
    let element_count = expected_shape.iter().product::<usize>();
    if quantized_bytes.len() != element_count {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf int8 restore",
            message: format!(
                "tensor `{parameter_id}` had {} int8 bytes for {} elements",
                quantized_bytes.len(),
                element_count
            ),
        });
    }
    let scales = decode_float_tensor(parameter_id, scale_dtype, scale_bytes, scale_shape)?;
    if expected_shape.len() == 2 {
        let rows = expected_shape[0];
        let cols = expected_shape[1];
        if scales.len() != rows {
            return Err(ParameterGolfReferenceTrainingError::Serialization {
                context: "parameter golf int8 restore",
                message: format!(
                    "tensor `{parameter_id}` expected {rows} row scales, found {}",
                    scales.len()
                ),
            });
        }
        let mut output = vec![0.0_f32; element_count];
        for row in 0..rows {
            let scale = scales[row];
            for col in 0..cols {
                let index = row * cols + col;
                output[index] = (quantized_bytes[index] as i8 as f32) * scale;
            }
        }
        return Ok(output);
    }
    let scale = scales.first().copied().unwrap_or(1.0);
    Ok(quantized_bytes
        .iter()
        .map(|value| (*value as i8 as f32) * scale)
        .collect())
}

fn validate_tensor_shape(
    parameter_id: &str,
    actual: &[usize],
    expected: &[usize],
) -> Result<(), ParameterGolfReferenceTrainingError> {
    if actual != expected {
        return Err(ParameterGolfReferenceTrainingError::ArtifactTensorShape {
            parameter_id: String::from(parameter_id),
            actual: actual.to_vec(),
            expected: expected.to_vec(),
        });
    }
    Ok(())
}

fn decode_float_tensor(
    parameter_id: &str,
    dtype: SafeTensorsDType,
    bytes: &[u8],
    shape: &[usize],
) -> Result<Vec<f32>, ParameterGolfReferenceTrainingError> {
    let expected_len = shape.iter().product::<usize>();
    match dtype {
        SafeTensorsDType::F32 => decode_f32_bytes(parameter_id, bytes, expected_len),
        SafeTensorsDType::F16 => decode_f16_bytes(parameter_id, bytes, expected_len),
        _ => Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf tensor decode",
            message: format!("tensor `{parameter_id}` had unsupported dtype `{dtype}`"),
        }),
    }
}

fn decode_f32_bytes(
    parameter_id: &str,
    bytes: &[u8],
    expected_len: usize,
) -> Result<Vec<f32>, ParameterGolfReferenceTrainingError> {
    if bytes.len() != expected_len.saturating_mul(4) {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf f32 decode",
            message: format!(
                "tensor `{parameter_id}` had {} bytes; expected {}",
                bytes.len(),
                expected_len.saturating_mul(4)
            ),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_f16_bytes(
    parameter_id: &str,
    bytes: &[u8],
    expected_len: usize,
) -> Result<Vec<f32>, ParameterGolfReferenceTrainingError> {
    if bytes.len() != expected_len.saturating_mul(2) {
        return Err(ParameterGolfReferenceTrainingError::Serialization {
            context: "parameter golf f16 decode",
            message: format!(
                "tensor `{parameter_id}` had {} bytes; expected {}",
                bytes.len(),
                expected_len.saturating_mul(2)
            ),
        });
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect())
}

fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&f16::from_f32(*value).to_le_bytes());
    }
    bytes
}

fn quantile_abs(values: &[f32], quantile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.iter().map(|value| value.abs()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let index = ((sorted.len().saturating_sub(1) as f32) * quantile)
        .round()
        .clamp(0.0, sorted.len().saturating_sub(1) as f32) as usize;
    sorted[index]
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded = match serde_json::to_vec(value) {
        Ok(encoded) => encoded,
        Err(error) => error.to_string().into_bytes(),
    };
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        io::{self, Write},
        sync::{Arc, Mutex},
        time::Instant,
    };

    use tempfile::tempdir;

    use crate::{
        async_checkpoint_writeback::write_checkpoint_payload_sync_with_options,
        AsyncCheckpointWritebackOptions, AsyncCheckpointWritebackWorker, LocalTrainMetricCollector,
        LocalTrainMetricFanout, LocalTrainMetricJsonlSink, LocalTrainMetricProgressSink,
        LocalTrainMetricStructuredLogSink, LocalTrainMetricValue,
    };

    use super::{
        checkpoint_async_writeback_payload, read_parameter_golf_checkpoint_from_directory,
        export_parameter_golf_banked_full_precision_model_bytes,
        export_parameter_golf_full_precision_model_bytes,
        restore_parameter_golf_local_reference_checkpoint,
        restore_parameter_golf_model_from_int8_zlib, restore_parameter_golf_model_from_safetensors,
        train_parameter_golf_local_reference,
        train_parameter_golf_local_reference_with_async_checkpoint_writeback,
        train_parameter_golf_local_reference_with_metric_sink, ParameterGolfLocalReferenceFixture,
        ParameterGolfReferenceTrainingConfig, ParameterGolfReferenceTrainingRunner,
    };
    use psionic_models::ParameterGolfReferenceModel;

    #[derive(Clone, Default)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl SharedWriter {
        fn contents(&self) -> String {
            String::from_utf8(
                self.0
                    .lock()
                    .expect("shared writer mutex should not be poisoned")
                    .clone(),
            )
            .expect("shared writer should only contain utf8")
        }
    }

    impl Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("shared writer mutex should not be poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn parameter_golf_local_reference_runner_restores_and_matches_continuous_run(
    ) -> Result<(), Box<dyn Error>> {
        let fixture = ParameterGolfLocalReferenceFixture::reference()?;
        let config = ParameterGolfReferenceTrainingConfig::local_reference();

        let mut continuous = ParameterGolfReferenceTrainingRunner::new(&fixture, &config)?;
        let mut restored_source = ParameterGolfReferenceTrainingRunner::new(&fixture, &config)?;

        continuous.step()?;
        continuous.step()?;

        restored_source.step()?;
        let checkpoint = restored_source.latest_checkpoint().clone();
        let mut restored =
            restore_parameter_golf_local_reference_checkpoint(&fixture, &checkpoint)?;
        restored.step()?;

        let continuous_outcome = continuous.into_outcome()?;
        let restored_outcome = restored.into_outcome()?;

        assert_eq!(
            continuous_outcome.trained_model,
            restored_outcome.trained_model
        );
        assert_eq!(
            continuous_outcome.final_validation_eval,
            restored_outcome.final_validation_eval
        );
        assert_eq!(
            continuous_outcome.step_metrics,
            restored_outcome.step_metrics
        );
        assert_eq!(
            continuous_outcome.final_checkpoint.manifest.stable_digest(),
            restored_outcome.final_checkpoint.manifest.stable_digest()
        );
        Ok(())
    }

    #[test]
    fn parameter_golf_local_reference_exports_raw_and_int8_roundtrips() -> Result<(), Box<dyn Error>>
    {
        let fixture = ParameterGolfLocalReferenceFixture::reference()?;
        let config = ParameterGolfReferenceTrainingConfig::local_reference();
        let outcome = train_parameter_golf_local_reference(&fixture, &config)?;

        let restored_raw = restore_parameter_golf_model_from_safetensors(
            &outcome.initial_model,
            outcome.raw_model_artifact.bytes.as_slice(),
        )?;
        assert_eq!(restored_raw, outcome.trained_model);
        assert_eq!(
            outcome.final_validation_eval,
            outcome.raw_roundtrip_validation_eval
        );

        let restored_int8 = restore_parameter_golf_model_from_int8_zlib(
            &outcome.initial_model,
            outcome.int8_zlib_model_artifact.bytes.as_slice(),
        )?;
        let int8_eval = psionic_eval::evaluate_parameter_golf_validation(
            &restored_int8,
            fixture.validation_tokens.as_slice(),
            config.geometry.train_sequence_length,
            config.geometry.local_validation_batch_tokens(),
            &fixture.byte_luts()?,
        )?;
        assert_eq!(int8_eval, outcome.int8_zlib_roundtrip_validation_eval);
        assert!(int8_eval.mean_loss.is_finite());
        assert!(int8_eval.bits_per_byte.is_finite());
        Ok(())
    }

    #[test]
    fn parameter_golf_local_reference_async_checkpoint_writeback_restores_sync_equivalently(
    ) -> Result<(), Box<dyn Error>> {
        let fixture = ParameterGolfLocalReferenceFixture::reference()?;
        let config = ParameterGolfReferenceTrainingConfig::local_reference();
        let sync_outcome = train_parameter_golf_local_reference(&fixture, &config)?;
        let checkpoint_root = tempdir()?;
        let async_outcome = train_parameter_golf_local_reference_with_async_checkpoint_writeback(
            &fixture,
            &config,
            checkpoint_root.path(),
            AsyncCheckpointWritebackOptions::bounded(1)?,
        )?;

        assert_eq!(async_outcome.trained_model, sync_outcome.trained_model);
        assert_eq!(
            async_outcome.final_validation_eval,
            sync_outcome.final_validation_eval
        );
        assert_eq!(async_outcome.step_metrics, sync_outcome.step_metrics);
        assert_eq!(
            async_outcome.final_checkpoint.manifest.stable_digest(),
            sync_outcome.final_checkpoint.manifest.stable_digest()
        );
        assert_eq!(
            async_outcome.checkpoint_writeback_receipts.len(),
            (config.max_steps + 1) as usize
        );

        let final_receipt = async_outcome
            .checkpoint_writeback_receipts
            .last()
            .expect("final checkpoint receipt should exist");
        let restored_checkpoint = read_parameter_golf_checkpoint_from_directory(
            final_receipt.final_directory.as_path(),
            &async_outcome.final_checkpoint.checkpoint,
        )?;
        assert_eq!(
            restored_checkpoint.manifest_artifact.bytes,
            async_outcome.final_checkpoint.manifest_artifact.bytes
        );
        assert_eq!(
            restored_checkpoint.weights_artifact.bytes,
            async_outcome.final_checkpoint.weights_artifact.bytes
        );
        assert_eq!(
            restored_checkpoint.manifest_artifact.artifact_digest,
            async_outcome
                .final_checkpoint
                .manifest_artifact
                .artifact_digest
        );
        assert_eq!(
            restored_checkpoint.weights_artifact.artifact_digest,
            async_outcome
                .final_checkpoint
                .weights_artifact
                .artifact_digest
        );

        let restored_runner =
            restore_parameter_golf_local_reference_checkpoint(&fixture, &restored_checkpoint)?;
        let restored_outcome = restored_runner.into_outcome()?;
        assert_eq!(restored_outcome.trained_model, sync_outcome.trained_model);
        assert_eq!(
            restored_outcome.final_validation_eval,
            sync_outcome.final_validation_eval
        );
        assert_eq!(
            restored_outcome.final_checkpoint.manifest.stable_digest(),
            sync_outcome.final_checkpoint.manifest.stable_digest()
        );
        Ok(())
    }

    #[test]
    fn parameter_golf_local_reference_async_checkpoint_handoff_beats_sync_stall(
    ) -> Result<(), Box<dyn Error>> {
        let fixture = ParameterGolfLocalReferenceFixture::reference()?;
        let config = ParameterGolfReferenceTrainingConfig::local_reference();
        let runner = ParameterGolfReferenceTrainingRunner::new(&fixture, &config)?;
        let sync_root = tempdir()?;
        let async_root = tempdir()?;
        let options = AsyncCheckpointWritebackOptions::bounded(1)?
            .with_test_injected_write_delay(std::time::Duration::from_millis(75));
        let sync_payload =
            checkpoint_async_writeback_payload(runner.latest_checkpoint(), sync_root.path())?;

        let sync_started = Instant::now();
        let _ = write_checkpoint_payload_sync_with_options(&sync_payload, &options)?;
        let sync_elapsed = sync_started.elapsed();

        let async_payload =
            checkpoint_async_writeback_payload(runner.latest_checkpoint(), async_root.path())?;
        let mut worker = AsyncCheckpointWritebackWorker::new(options.clone())?;
        let async_started = Instant::now();
        let ticket = worker.submit(async_payload)?;
        let async_submit_elapsed = async_started.elapsed();
        let receipt = ticket.wait()?;
        let _ = worker.shutdown_flush()?;

        assert!(receipt.final_directory.exists());
        assert!(sync_elapsed >= std::time::Duration::from_millis(60));
        assert!(
            async_submit_elapsed.as_millis() * 5 < sync_elapsed.as_millis(),
            "expected async handoff {:?} to be materially smaller than sync stall {:?}",
            async_submit_elapsed,
            sync_elapsed
        );
        Ok(())
    }

    #[test]
    fn parameter_golf_local_reference_metric_sink_fanout_stays_local_and_deterministic(
    ) -> Result<(), Box<dyn Error>> {
        let fixture = ParameterGolfLocalReferenceFixture::reference()?;
        let config = ParameterGolfReferenceTrainingConfig::local_reference();
        let baseline_outcome = train_parameter_golf_local_reference(&fixture, &config)?;
        let progress = SharedWriter::default();
        let structured = SharedWriter::default();
        let collector = LocalTrainMetricCollector::default();
        let jsonl_dir = tempdir()?;
        let jsonl_path = jsonl_dir.path().join("telemetry.jsonl");
        let mut sink = LocalTrainMetricFanout::new(config.run_id.clone());
        sink.add_sink(LocalTrainMetricProgressSink::new(progress.clone()));
        sink.add_sink(LocalTrainMetricStructuredLogSink::new(structured.clone()));
        sink.add_sink(LocalTrainMetricJsonlSink::create(jsonl_path.as_path())?);
        sink.add_sink(collector.clone());

        let sink_outcome =
            train_parameter_golf_local_reference_with_metric_sink(&fixture, &config, &mut sink)?;
        let collected = collector.events();
        let jsonl_lines = std::fs::read_to_string(jsonl_path.as_path())?
            .lines()
            .map(serde_json::from_str::<crate::LocalTrainMetricEvent>)
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(sink_outcome, baseline_outcome);
        assert_eq!(collected, jsonl_lines);
        assert_eq!(collected.len(), (config.max_steps as usize) * 3);
        assert!(collected
            .iter()
            .any(|event| event.metric_id == "mean_microbatch_loss"
                && matches!(event.value, LocalTrainMetricValue::F32(_))));
        assert!(progress.contents().contains("mean_microbatch_loss"));
        assert!(structured.contents().starts_with("metric_event {"));
        Ok(())
    }

    #[test]
    fn parameter_golf_split_full_precision_safetensors_roundtrip_restores_model(
    ) -> Result<(), Box<dyn Error>> {
        let model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
        let bytes = export_parameter_golf_full_precision_model_bytes(&model)?;
        let restored = restore_parameter_golf_model_from_safetensors(&model, bytes.as_slice())?;
        assert_eq!(restored, model);
        Ok(())
    }

    #[test]
    fn parameter_golf_banked_full_precision_safetensors_roundtrip_restores_model(
    ) -> Result<(), Box<dyn Error>> {
        let model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
        let bytes = export_parameter_golf_banked_full_precision_model_bytes(&model)?;
        let restored = restore_parameter_golf_model_from_safetensors(&model, bytes.as_slice())?;
        assert_eq!(restored, model);
        Ok(())
    }
}

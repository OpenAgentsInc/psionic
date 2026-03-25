use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use half::bf16;
use psionic_backend_cuda::{CudaBackend, CudaBuffer};
use psionic_core::{
    DType, PsionicRefusal, PsionicRefusalCode, PsionicRefusalScope, Shape, TensorData, TensorId,
};
use psionic_data::{
    load_parameter_golf_validation_tokens_from_paths, materialize_parameter_golf_token_window,
    parameter_golf_dataset_bundle_from_local_dir,
    parameter_golf_sentencepiece_byte_luts_from_tokenizer_path, DatasetIterationMode, DatasetKey,
    ParameterGolfDataError, ParameterGolfDatasetBundle, ParameterGolfSentencePieceByteLuts,
    ParameterGolfTokenStreamContract, ParameterGolfTokenStreamCursor,
    ParameterGolfTokenStreamWindow, PARAMETER_GOLF_TRAIN_SPLIT_NAME,
};
use psionic_ir::AutodiffBackwardResult;
use psionic_ir::{AutodiffError, GraphError};
use psionic_models::{
    ParameterGolfModelError, ParameterGolfReferenceModel, PARAMETER_GOLF_BASELINE_MODEL_ID,
    PARAMETER_GOLF_BASELINE_REVISION,
};
use psionic_runtime::{
    BufferHandle, DeliveredExecutionContext, DeviceDescriptor, RuntimeError, RuntimeHealth,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    apply_parameter_golf_cuda_bf16_master_weight_optimizer_step,
    apply_parameter_golf_cuda_muon_step, bind_parameter_golf_baseline_training_graph_inputs,
    build_parameter_golf_baseline_eval_graph, build_parameter_golf_baseline_training_graph,
    build_tokenizer_digest, builtin_parameter_golf_cuda_training_capability_report,
    device_matches_single_h100, export_parameter_golf_int8_zlib_model_artifact,
    inspect_local_single_h100_machine, materialize_parameter_golf_baseline_training_gradients,
    parameter_golf_optimizer_plan, restore_parameter_golf_model_from_int8_zlib,
    training_batch_from_window_tokens, ParameterGolfBaselineEvalGraph,
    ParameterGolfBaselineTrainingGraph, ParameterGolfBatchGeometry,
    ParameterGolfBf16MasterWeightStepReceipt, ParameterGolfOptimizerExecution,
    ParameterGolfOptimizerGroupKind, ParameterGolfOptimizerPlan,
    ParameterGolfReferenceTrainingError, ParameterGolfSingleH100BringupError,
    ParameterGolfSingleH100ChallengeThresholds, ParameterGolfSingleH100MachineObservation,
    ParameterGolfTrainError, ParameterGolfTrainingHyperparameters, TrainingOptimizerConfig,
    TrainingOptimizerState, TrainingPrecisionMode, PARAMETER_GOLF_SINGLE_H100_DATASET_REF,
    PARAMETER_GOLF_SINGLE_H100_DATASET_VERSION, PARAMETER_GOLF_SINGLE_H100_VARIANT,
};

/// Config for the bounded Rust-owned single-H100 Parameter Golf trainer lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100TrainingConfig {
    /// Stable run identifier.
    pub run_id: String,
    /// Local dataset root containing cached FineWeb/SP1024 shards.
    pub dataset_root: PathBuf,
    /// Local tokenizer artifact path.
    pub tokenizer_path: PathBuf,
    /// Stable dataset identity expected by the run.
    pub dataset_key: DatasetKey,
    /// Stable tokenizer or dataset variant label.
    pub variant: String,
    /// Public single-device challenge geometry.
    pub geometry: ParameterGolfBatchGeometry,
    /// Public baseline optimization contract.
    pub hyperparameters: ParameterGolfTrainingHyperparameters,
    /// Explicit optimizer-step cap for this run.
    pub max_steps: u64,
    /// Warmup steps used to prime the runtime before measured training starts.
    pub warmup_steps: u64,
    /// Validation cadence in optimizer steps. Zero disables periodic validation.
    pub validation_loss_every: u64,
    /// Train-log cadence in optimizer steps. Zero disables train-step summaries.
    pub train_log_every: u64,
    /// Explicit final-validation posture for the live-model and roundtrip passes.
    #[serde(default)]
    pub final_validation_mode: ParameterGolfSingleH100ValidationMode,
    /// Explicit evaluation semantics for live and roundtrip validation.
    #[serde(default)]
    pub validation_eval_mode: ParameterGolfValidationEvalMode,
}

impl ParameterGolfSingleH100TrainingConfig {
    /// Returns the canonical challenge-baseline single-H100 training defaults.
    #[must_use]
    pub fn challenge_defaults(
        dataset_root: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
    ) -> Self {
        let hyperparameters = ParameterGolfTrainingHyperparameters::baseline_defaults();
        Self {
            run_id: String::from("parameter-golf-single-h100-trainer"),
            dataset_root: dataset_root.into(),
            tokenizer_path: tokenizer_path.into(),
            dataset_key: DatasetKey::new(
                PARAMETER_GOLF_SINGLE_H100_DATASET_REF,
                PARAMETER_GOLF_SINGLE_H100_DATASET_VERSION,
            ),
            variant: String::from(PARAMETER_GOLF_SINGLE_H100_VARIANT),
            geometry: ParameterGolfBatchGeometry::challenge_single_device_defaults(),
            max_steps: hyperparameters.iterations,
            warmup_steps: 20,
            validation_loss_every: 1_000,
            train_log_every: 200,
            final_validation_mode: ParameterGolfSingleH100ValidationMode::Both,
            validation_eval_mode: ParameterGolfValidationEvalMode::NonOverlapping,
            hyperparameters,
        }
    }

    /// Returns the old explicit bounded proof posture for one short bring-up run.
    #[must_use]
    pub fn bounded_proof_defaults(
        dataset_root: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
        max_steps: u64,
    ) -> Self {
        let mut config = Self::challenge_defaults(dataset_root, tokenizer_path);
        config.max_steps = max_steps;
        config.warmup_steps = 0;
        config.validation_loss_every = 0;
        config.train_log_every = 1;
        config.final_validation_mode = ParameterGolfSingleH100ValidationMode::RoundtripOnly;
        config.validation_eval_mode = ParameterGolfValidationEvalMode::NonOverlapping;
        config.hyperparameters.max_wallclock_seconds = None;
        config
    }

    fn validate(&self) -> Result<(), ParameterGolfSingleH100TrainingError> {
        if self.run_id.trim().is_empty() {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("run_id must be non-empty"),
            });
        }
        if self.variant.trim().is_empty() {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("variant must be non-empty"),
            });
        }
        if self.max_steps == 0 {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("max_steps must be positive"),
            });
        }
        if self.hyperparameters.iterations == 0 {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("hyperparameter iterations must be positive"),
            });
        }
        if self.max_steps > self.hyperparameters.iterations {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: format!(
                    "max_steps={} must not exceed hyperparameter iterations={}",
                    self.max_steps, self.hyperparameters.iterations
                ),
            });
        }
        if self.geometry != ParameterGolfBatchGeometry::challenge_single_device_defaults() {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from(
                    "single-H100 training requires challenge_single_device_defaults geometry",
                ),
            });
        }
        if !self.dataset_root.is_dir() {
            return Err(ParameterGolfSingleH100TrainingError::MissingPath {
                path: self.dataset_root.display().to_string(),
                expected: String::from("dataset directory"),
            });
        }
        if !self.tokenizer_path.is_file() {
            return Err(ParameterGolfSingleH100TrainingError::MissingPath {
                path: self.tokenizer_path.display().to_string(),
                expected: String::from("tokenizer file"),
            });
        }
        self.validation_eval_mode
            .validate(self.geometry.train_sequence_length)?;
        Ok(())
    }
}

/// Observed current-state disposition for the bounded single-H100 trainer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfSingleH100TrainingDisposition {
    RefusedMachineContract,
    RefusedCudaBlockers,
    TrainingExecuted,
}

/// Explicit final-validation posture for the single-H100 trainer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfSingleH100ValidationMode {
    LiveOnly,
    RoundtripOnly,
    #[default]
    Both,
}

impl ParameterGolfSingleH100ValidationMode {
    /// Parses one CLI-visible validation mode label.
    pub fn parse(value: &str) -> Result<Self, ParameterGolfSingleH100TrainingError> {
        match value {
            "live_only" => Ok(Self::LiveOnly),
            "roundtrip_only" => Ok(Self::RoundtripOnly),
            "both" => Ok(Self::Both),
            actual => Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: format!(
                    "unsupported final validation mode `{actual}`, expected one of live_only, roundtrip_only, or both"
                ),
            }),
        }
    }

    /// Stable string label for logs, reports, and CLI wiring.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LiveOnly => "live_only",
            Self::RoundtripOnly => "roundtrip_only",
            Self::Both => "both",
        }
    }

    #[must_use]
    const fn runs_live_validation(self) -> bool {
        matches!(self, Self::LiveOnly | Self::Both)
    }

    #[must_use]
    const fn runs_roundtrip_validation(self) -> bool {
        matches!(self, Self::RoundtripOnly | Self::Both)
    }
}

/// Explicit evaluation semantics for Parameter Golf validation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ParameterGolfValidationEvalMode {
    #[default]
    NonOverlapping,
    SlidingWindow { stride: usize },
}

impl ParameterGolfValidationEvalMode {
    /// Parses one CLI-visible validation eval mode label.
    pub fn parse(value: &str) -> Result<Self, ParameterGolfSingleH100TrainingError> {
        if value == "non_overlapping" {
            return Ok(Self::NonOverlapping);
        }
        if let Some(stride) = value.strip_prefix("sliding_window:") {
            return Ok(Self::SlidingWindow {
                stride: stride.parse::<usize>().map_err(|error| {
                    ParameterGolfSingleH100TrainingError::InvalidConfig {
                        message: format!(
                            "invalid sliding_window stride `{stride}`: {error}"
                        ),
                    }
                })?,
            });
        }
        Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
            message: format!(
                "unsupported validation eval mode `{value}`, expected non_overlapping or sliding_window:<stride>"
            ),
        })
    }

    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NonOverlapping => "non_overlapping",
            Self::SlidingWindow { .. } => "sliding_window",
        }
    }

    fn validate(&self, sequence_length: usize) -> Result<(), ParameterGolfSingleH100TrainingError> {
        match self {
            Self::NonOverlapping => Ok(()),
            Self::SlidingWindow { stride } => {
                if *stride == 0 || *stride > sequence_length {
                    return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                        message: format!(
                            "sliding-window validation stride must be in 1..={sequence_length}, observed {stride}"
                        ),
                    });
                }
                Ok(())
            }
        }
    }
}

/// Machine-readable phase timings for one challenge-step aggregate.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100PhaseTimings {
    pub window_planning_ms: u64,
    pub token_materialization_ms: u64,
    pub embedding_gather_ms: u64,
    pub forward_loss_cuda_ms: u64,
    pub retained_forward_readback_ms: u64,
    pub backward_cuda_ms: u64,
    pub gradient_readback_ms: u64,
    pub host_gradient_materialization_ms: u64,
    pub optimizer_step_ms: u64,
    pub retained_binding_tensor_count: u32,
    pub retained_binding_f32_count: u64,
    pub gradient_tensor_count: u32,
    pub gradient_f32_count: u64,
}

impl ParameterGolfSingleH100PhaseTimings {
    fn accumulate(&mut self, other: &Self) {
        self.window_planning_ms = self
            .window_planning_ms
            .saturating_add(other.window_planning_ms);
        self.token_materialization_ms = self
            .token_materialization_ms
            .saturating_add(other.token_materialization_ms);
        self.embedding_gather_ms = self
            .embedding_gather_ms
            .saturating_add(other.embedding_gather_ms);
        self.forward_loss_cuda_ms = self
            .forward_loss_cuda_ms
            .saturating_add(other.forward_loss_cuda_ms);
        self.retained_forward_readback_ms = self
            .retained_forward_readback_ms
            .saturating_add(other.retained_forward_readback_ms);
        self.backward_cuda_ms = self.backward_cuda_ms.saturating_add(other.backward_cuda_ms);
        self.gradient_readback_ms = self
            .gradient_readback_ms
            .saturating_add(other.gradient_readback_ms);
        self.host_gradient_materialization_ms = self
            .host_gradient_materialization_ms
            .saturating_add(other.host_gradient_materialization_ms);
        self.optimizer_step_ms = self
            .optimizer_step_ms
            .saturating_add(other.optimizer_step_ms);
        self.retained_binding_tensor_count = self
            .retained_binding_tensor_count
            .saturating_add(other.retained_binding_tensor_count);
        self.retained_binding_f32_count = self
            .retained_binding_f32_count
            .saturating_add(other.retained_binding_f32_count);
        self.gradient_tensor_count = self
            .gradient_tensor_count
            .saturating_add(other.gradient_tensor_count);
        self.gradient_f32_count = self
            .gradient_f32_count
            .saturating_add(other.gradient_f32_count);
    }

    fn host_materialization_ms(&self) -> u64 {
        self.embedding_gather_ms
            .saturating_add(self.retained_forward_readback_ms)
            .saturating_add(self.gradient_readback_ms)
            .saturating_add(self.host_gradient_materialization_ms)
    }
}

/// Reusable one-window gradient result for Parameter Golf training.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ParameterGolfTrainingGradientBatchResult {
    pub window_id: String,
    pub loss: f32,
    pub phase_timings: ParameterGolfSingleH100PhaseTimings,
    pub parameter_gradients: BTreeMap<String, Vec<f32>>,
}

/// Per-step machine-readable training metrics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100TrainingStepMetrics {
    pub global_step: u64,
    pub train_window_ids: Vec<String>,
    pub mean_microbatch_loss: f32,
    pub learning_rate_multiplier: f32,
    pub muon_momentum: f32,
    pub observed_wallclock_ms: u64,
    pub phase_timings: ParameterGolfSingleH100PhaseTimings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_learning_rate: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_norm_after_clip: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_norm_after_step: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update_norm: Option<f32>,
    #[serde(default)]
    pub clip_applied: bool,
    #[serde(default)]
    pub non_finite_gradient_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples_per_second_milli: Option<u32>,
}

/// Machine-readable validation summary for the accelerated single-H100 lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100ValidationSummary {
    #[serde(default)]
    pub eval_mode: ParameterGolfValidationEvalMode,
    pub evaluated_sequence_count: usize,
    pub evaluated_token_count: u64,
    pub evaluated_byte_count: u64,
    pub mean_loss: f64,
    pub bits_per_byte: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_receipt: Option<ParameterGolfSingleH100ValidationRuntimeReceipt>,
}

/// Machine-readable runtime posture for one validation pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100ValidationRuntimeReceipt {
    pub path: String,
    pub graph_surface: String,
    #[serde(default)]
    pub eval_mode: ParameterGolfValidationEvalMode,
    pub session_count: usize,
    pub total_batches: usize,
    pub total_units: usize,
    pub persistent_parameter_buffer_count: usize,
    pub persistent_parameter_value_count: u64,
    pub resident_parameter_upload_us: u64,
    pub per_batch_stable_parameter_buffer_allocations: u64,
    pub reusable_input_token_buffer: bool,
    pub reusable_target_token_buffer: bool,
    pub total_input_token_write_us: u64,
    pub total_target_token_write_us: u64,
    pub byte_accounting_mode: String,
    pub total_byte_accounting_us: u64,
}

/// One preserved validation observation from the widened single-H100 control loop.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100ValidationCheckpoint {
    pub stage_label: String,
    pub trigger_step: u64,
    pub observed_training_time_ms: u64,
    pub observed_validation_ms: u64,
    pub summary: ParameterGolfSingleH100ValidationSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100RoundtripReceipt {
    pub metric_source: String,
    pub validation: ParameterGolfSingleH100ValidationSummary,
    pub observed_eval_ms: u64,
    pub compressed_model_bytes: u64,
    pub compressed_model_artifact_ref: String,
    pub compressed_model_artifact_digest: String,
}

/// Explicit stop reason for a measured single-H100 run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfSingleH100TrainingStopReason {
    StepBudgetReached,
    WallclockCapReached,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100GroupPrecisionReceipt {
    pub group_id: String,
    pub parameter_precision: TrainingPrecisionMode,
    pub gradient_precision: TrainingPrecisionMode,
    pub optimizer_state_precision: TrainingPrecisionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_weight_precision: Option<TrainingPrecisionMode>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100PrecisionReceipt {
    pub graph_parameter_upload_precision: TrainingPrecisionMode,
    pub graph_execution_precision: TrainingPrecisionMode,
    pub retained_activation_precision: TrainingPrecisionMode,
    pub group_receipts: Vec<ParameterGolfSingleH100GroupPrecisionReceipt>,
    pub notes: Vec<String>,
}

/// End-to-end machine-readable trainer report for the bounded single-H100 lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100TrainingReport {
    pub schema_version: u32,
    pub scope_window: String,
    pub run_id: String,
    pub dataset_root: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dataset_key: DatasetKey,
    pub variant: String,
    pub tokenizer_digest: psionic_data::TokenizerDigest,
    pub dataset_manifest_digest: String,
    pub train_shard_count: usize,
    pub validation_shard_count: usize,
    pub train_token_count: u64,
    pub validation_token_count: u64,
    pub geometry: ParameterGolfBatchGeometry,
    pub hyperparameters: ParameterGolfTrainingHyperparameters,
    pub max_steps: u64,
    pub warmup_steps: u64,
    pub completed_warmup_steps: u64,
    pub validation_loss_every: u64,
    pub train_log_every: u64,
    #[serde(default)]
    pub final_validation_mode: ParameterGolfSingleH100ValidationMode,
    #[serde(default)]
    pub validation_eval_mode: ParameterGolfValidationEvalMode,
    pub executed_steps: u64,
    pub stop_reason: Option<ParameterGolfSingleH100TrainingStopReason>,
    pub delivered_execution: DeliveredExecutionContext,
    pub machine_thresholds: ParameterGolfSingleH100ChallengeThresholds,
    pub observed_cuda_health: RuntimeHealth,
    pub cuda_discovery_error: Option<String>,
    pub observed_cuda_devices: Vec<DeviceDescriptor>,
    pub matching_h100_device_count: usize,
    pub machine_contract_satisfied: bool,
    pub baseline_model_id: String,
    pub baseline_model_revision: String,
    pub baseline_model_descriptor_digest: String,
    pub optimizer_plan_digest: String,
    pub precision_receipt: ParameterGolfSingleH100PrecisionReceipt,
    pub cuda_training_capability_report_digest: String,
    pub challenge_kernel_blockers: Vec<String>,
    pub validation_checkpoints: Vec<ParameterGolfSingleH100ValidationCheckpoint>,
    pub initial_validation: Option<ParameterGolfSingleH100ValidationSummary>,
    pub pre_export_final_validation: Option<ParameterGolfSingleH100ValidationSummary>,
    pub final_validation: Option<ParameterGolfSingleH100ValidationSummary>,
    pub warmup_observed_ms: u64,
    pub observed_training_time_ms: u64,
    pub pre_export_final_validation_observed_ms: Option<u64>,
    pub final_validation_observed_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_roundtrip_receipt: Option<ParameterGolfSingleH100RoundtripReceipt>,
    pub compressed_model_bytes: Option<u64>,
    pub compressed_model_artifact_ref: Option<String>,
    pub compressed_model_artifact_digest: Option<String>,
    pub step_metrics: Vec<ParameterGolfSingleH100TrainingStepMetrics>,
    pub aggregate_phase_timings: Option<ParameterGolfSingleH100PhaseTimings>,
    pub final_training_cursor: Option<ParameterGolfTokenStreamCursor>,
    pub started_at_ms: u64,
    pub finished_at_ms: u64,
    pub observed_wallclock_ms: u64,
    pub disposition: ParameterGolfSingleH100TrainingDisposition,
    pub refusal: Option<PsionicRefusal>,
    pub claim_boundary: String,
    pub summary: String,
    pub report_digest: String,
}

impl ParameterGolfSingleH100TrainingReport {
    /// Returns whether the report is an executed trainer proof.
    #[must_use]
    pub fn training_executed(&self) -> bool {
        self.disposition == ParameterGolfSingleH100TrainingDisposition::TrainingExecuted
    }
}

/// Config for a bounded same-node validation-runtime comparison.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100ValidationRuntimeComparisonConfig {
    pub run_id: String,
    pub dataset_root: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dataset_key: DatasetKey,
    pub variant: String,
    pub batch_sequences: usize,
    pub sequence_length: usize,
    pub batch_limit: usize,
}

impl ParameterGolfSingleH100ValidationRuntimeComparisonConfig {
    #[must_use]
    pub fn bounded_local_defaults(
        dataset_root: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            run_id: String::from("parameter-golf-validation-runtime-comparison"),
            dataset_root: dataset_root.into(),
            tokenizer_path: tokenizer_path.into(),
            dataset_key: DatasetKey::new(
                PARAMETER_GOLF_SINGLE_H100_DATASET_REF,
                PARAMETER_GOLF_SINGLE_H100_DATASET_VERSION,
            ),
            variant: String::from(PARAMETER_GOLF_SINGLE_H100_VARIANT),
            batch_sequences: 8,
            sequence_length: ParameterGolfBatchGeometry::challenge_single_device_defaults()
                .train_sequence_length,
            batch_limit: 2,
        }
    }

    fn validate(&self) -> Result<(), ParameterGolfSingleH100TrainingError> {
        if self.run_id.trim().is_empty() {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("run_id must be non-empty"),
            });
        }
        if self.batch_sequences == 0 {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("batch_sequences must be positive"),
            });
        }
        if self.sequence_length == 0 {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("sequence_length must be positive"),
            });
        }
        if self.batch_limit == 0 {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("batch_limit must be positive"),
            });
        }
        if !self.dataset_root.is_dir() {
            return Err(ParameterGolfSingleH100TrainingError::MissingPath {
                path: self.dataset_root.display().to_string(),
                expected: String::from("dataset directory"),
            });
        }
        if !self.tokenizer_path.is_file() {
            return Err(ParameterGolfSingleH100TrainingError::MissingPath {
                path: self.tokenizer_path.display().to_string(),
                expected: String::from("tokenizer file"),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100ValidationRuntimeComparisonLaneReceipt {
    pub mean_loss: f64,
    pub bits_per_byte: f64,
    pub observed_ms: u64,
    pub average_batch_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_receipt: Option<ParameterGolfSingleH100ValidationRuntimeReceipt>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100ValidationRuntimeComparisonReceipt {
    pub schema_version: u32,
    pub scope_window: String,
    pub run_id: String,
    pub dataset_root: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dataset_key: DatasetKey,
    pub variant: String,
    pub tokenizer_digest: psionic_data::TokenizerDigest,
    pub dataset_manifest_digest: String,
    pub batch_sequences: usize,
    pub sequence_length: usize,
    pub batch_limit: usize,
    pub observed_cuda_health: RuntimeHealth,
    pub observed_cuda_devices: Vec<DeviceDescriptor>,
    pub legacy: ParameterGolfSingleH100ValidationRuntimeComparisonLaneReceipt,
    pub device_resident: ParameterGolfSingleH100ValidationRuntimeComparisonLaneReceipt,
    pub summary: String,
    pub receipt_digest: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ParameterGolfSingleH100TrainerState {
    pub(crate) parameter_states: BTreeMap<String, ParameterGolfParameterState>,
}

#[derive(Clone, Debug)]
struct ParameterGolfValidationSequencePlan {
    valid_length: usize,
    score_start: usize,
}

#[derive(Clone, Debug)]
struct ParameterGolfValidationBatchPlan {
    batch_sequences: usize,
    evaluation_units: usize,
    input_tokens: Vec<i32>,
    target_tokens: Vec<i32>,
    sequence_plans: Vec<ParameterGolfValidationSequencePlan>,
    token_count: u64,
    byte_count: u64,
}

#[derive(Clone, Debug, Default)]
struct ParameterGolfValidationBatchRuntime {
    input_token_write_us: u64,
    target_token_write_us: u64,
}

#[derive(Clone, Debug)]
struct ParameterGolfCudaValidationSession {
    graph: ParameterGolfBaselineEvalGraph,
    static_inputs: BTreeMap<TensorId, CudaBuffer>,
    input_token_buffer: CudaBuffer,
    target_token_buffer: CudaBuffer,
    input_token_staging: Vec<i32>,
    target_token_staging: Vec<i32>,
    resident_parameter_upload_us: u64,
    persistent_parameter_buffer_count: usize,
    persistent_parameter_value_count: u64,
}

#[derive(Clone, Debug)]
pub(crate) enum ParameterGolfParameterState {
    AdamBf16Master {
        shape: Vec<usize>,
        train_visible_values: Vec<f32>,
        master_weight_values: Vec<f32>,
        optimizer: TrainingOptimizerConfig,
        optimizer_state: TrainingOptimizerState,
        last_step_receipt: Option<ParameterGolfBf16MasterWeightStepReceipt>,
    },
    AdamFp32 {
        shape: Vec<usize>,
        values: Vec<f32>,
        optimizer: TrainingOptimizerConfig,
        optimizer_state: TrainingOptimizerState,
    },
    MuonBf16 {
        shape: Vec<usize>,
        values: Vec<f32>,
        optimizer: crate::ParameterGolfMuonConfig,
        optimizer_state: crate::ParameterGolfMuonState,
    },
}

impl ParameterGolfParameterState {
    pub(crate) fn values(&self) -> &[f32] {
        match self {
            Self::AdamBf16Master {
                train_visible_values,
                ..
            } => train_visible_values.as_slice(),
            Self::AdamFp32 { values, .. } | Self::MuonBf16 { values, .. } => values.as_slice(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::AdamBf16Master { shape, .. }
            | Self::AdamFp32 { shape, .. }
            | Self::MuonBf16 { shape, .. } => shape.as_slice(),
        }
    }

    fn precision_receipt(&self, group_id: String) -> ParameterGolfSingleH100GroupPrecisionReceipt {
        match self {
            Self::AdamBf16Master { .. } => ParameterGolfSingleH100GroupPrecisionReceipt {
                group_id,
                parameter_precision: TrainingPrecisionMode::Bf16,
                gradient_precision: TrainingPrecisionMode::Bf16,
                optimizer_state_precision: TrainingPrecisionMode::Fp32,
                master_weight_precision: Some(TrainingPrecisionMode::Fp32),
            },
            Self::AdamFp32 { .. } => ParameterGolfSingleH100GroupPrecisionReceipt {
                group_id,
                parameter_precision: TrainingPrecisionMode::Fp32,
                gradient_precision: TrainingPrecisionMode::Fp32,
                optimizer_state_precision: TrainingPrecisionMode::Fp32,
                master_weight_precision: None,
            },
            Self::MuonBf16 { .. } => ParameterGolfSingleH100GroupPrecisionReceipt {
                group_id,
                parameter_precision: TrainingPrecisionMode::Bf16,
                gradient_precision: TrainingPrecisionMode::Fp32,
                optimizer_state_precision: TrainingPrecisionMode::Fp32,
                master_weight_precision: None,
            },
        }
    }

    fn apply_gradients(
        &mut self,
        gradients: &[f32],
        learning_rate_multiplier: f32,
        muon_momentum: f32,
        step_number: u64,
    ) -> Result<(), ParameterGolfSingleH100TrainingError> {
        match self {
            Self::AdamBf16Master {
                train_visible_values,
                master_weight_values,
                optimizer,
                optimizer_state,
                last_step_receipt,
                ..
            } => {
                let mut optimizer = optimizer.clone();
                optimizer.learning_rate *= learning_rate_multiplier;
                let receipt = apply_parameter_golf_cuda_bf16_master_weight_optimizer_step(
                    train_visible_values.as_mut_slice(),
                    master_weight_values.as_mut_slice(),
                    gradients,
                    &optimizer,
                    optimizer_state,
                    step_number,
                )?;
                *last_step_receipt = Some(receipt);
                Ok(())
            }
            Self::AdamFp32 {
                values,
                optimizer,
                optimizer_state,
                ..
            } => {
                let mut optimizer = optimizer.clone();
                optimizer.learning_rate *= learning_rate_multiplier;
                optimizer.apply_step(
                    values.as_mut_slice(),
                    gradients,
                    optimizer_state,
                    step_number,
                )?;
                Ok(())
            }
            Self::MuonBf16 {
                shape,
                values,
                optimizer,
                optimizer_state,
            } => {
                let mut optimizer = optimizer.clone();
                optimizer.learning_rate *= learning_rate_multiplier;
                optimizer.momentum = muon_momentum;
                apply_parameter_golf_cuda_muon_step(
                    values.as_mut_slice(),
                    shape.as_slice(),
                    gradients,
                    &optimizer,
                    optimizer_state,
                )?;
                round_values_to_bf16(values.as_mut_slice());
                Ok(())
            }
        }
    }
}

/// Failure while executing the bounded single-H100 trainer.
#[derive(Debug, Error)]
pub enum ParameterGolfSingleH100TrainingError {
    #[error("parameter golf single-H100 training config is invalid: {message}")]
    InvalidConfig { message: String },
    #[error("parameter golf single-H100 training expected {expected} at `{path}`")]
    MissingPath { path: String, expected: String },
    #[error("parameter golf single-H100 training serialization failed: {message}")]
    Serialization { message: String },
    #[error("parameter golf single-H100 training could not select a qualifying H100 CUDA device")]
    MissingSelectedH100,
    #[error("parameter golf single-H100 training graph is missing tensor `{tensor_id}`")]
    MissingGraphTensor { tensor_id: TensorId },
    #[error("parameter golf single-H100 training graph output `{tensor_id}` was not materialized")]
    MissingGraphOutput { tensor_id: TensorId },
    #[error("parameter golf single-H100 training is missing parameter state for `{parameter_id}`")]
    MissingParameterState { parameter_id: String },
    #[error(transparent)]
    Data(#[from] ParameterGolfDataError),
    #[error(transparent)]
    Model(#[from] ParameterGolfModelError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Autodiff(#[from] AutodiffError),
    #[error(transparent)]
    BaselineGraph(#[from] crate::ParameterGolfBaselineGraphError),
    #[error(transparent)]
    Bringup(#[from] ParameterGolfSingleH100BringupError),
    #[error(transparent)]
    ReferenceTraining(#[from] ParameterGolfReferenceTrainingError),
    #[error(transparent)]
    Train(#[from] ParameterGolfTrainError),
    #[error(transparent)]
    Optimizer(#[from] crate::TrainingOptimizerError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Visualization(#[from] crate::ParameterGolfSingleH100VisualizationError),
}

/// Writes the bounded single-H100 training report to disk.
pub fn write_parameter_golf_single_h100_training_report(
    output_path: &Path,
    config: &ParameterGolfSingleH100TrainingConfig,
) -> Result<ParameterGolfSingleH100TrainingReport, ParameterGolfSingleH100TrainingError> {
    let (report, mut live_visualization_writer) =
        build_parameter_golf_single_h100_training_report_inner(config, Some(output_path))?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded = serde_json::to_vec_pretty(&report).map_err(|error| {
        ParameterGolfSingleH100TrainingError::Serialization {
            message: error.to_string(),
        }
    })?;
    fs::write(output_path, encoded)?;
    if let Some(writer) = live_visualization_writer.as_mut() {
        writer.finish_with_report(&report)?;
    }
    Ok(report)
}

/// Writes one bounded same-node validation-runtime comparison receipt to disk.
pub fn write_parameter_golf_single_h100_validation_runtime_comparison_receipt(
    output_path: &Path,
    config: &ParameterGolfSingleH100ValidationRuntimeComparisonConfig,
) -> Result<
    ParameterGolfSingleH100ValidationRuntimeComparisonReceipt,
    ParameterGolfSingleH100TrainingError,
> {
    let receipt = build_parameter_golf_single_h100_validation_runtime_comparison_receipt(config)?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded = serde_json::to_vec_pretty(&receipt).map_err(|error| {
        ParameterGolfSingleH100TrainingError::Serialization {
            message: error.to_string(),
        }
    })?;
    fs::write(output_path, encoded)?;
    Ok(receipt)
}

/// Builds one bounded same-node validation-runtime comparison receipt.
pub fn build_parameter_golf_single_h100_validation_runtime_comparison_receipt(
    config: &ParameterGolfSingleH100ValidationRuntimeComparisonConfig,
) -> Result<
    ParameterGolfSingleH100ValidationRuntimeComparisonReceipt,
    ParameterGolfSingleH100TrainingError,
> {
    config.validate()?;
    let tokenizer_bytes = fs::read(&config.tokenizer_path)?;
    let tokenizer_digest = build_tokenizer_digest(tokenizer_bytes.as_slice());
    let bundle = parameter_golf_dataset_bundle_from_local_dir(
        config.dataset_key.clone(),
        &config.dataset_root,
        config.variant.clone(),
        tokenizer_digest.clone(),
        config.tokenizer_path.display().to_string(),
        None,
    )?;
    let validation_tokens = load_parameter_golf_validation_tokens_from_paths(
        &bundle
            .validation_shards
            .iter()
            .map(|receipt| PathBuf::from(&receipt.path))
            .collect::<Vec<_>>(),
        config.sequence_length,
    )?;
    let total_sequences = (validation_tokens.len() - 1) / config.sequence_length;
    let selected_sequences = total_sequences.min(
        config
            .batch_sequences
            .saturating_mul(config.batch_limit.max(1)),
    );
    if selected_sequences == 0 {
        return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
            message: String::from(
                "validation runtime comparison requires at least one complete validation sequence",
            ),
        });
    }
    let selected_token_end = selected_sequences
        .saturating_mul(config.sequence_length)
        .saturating_add(1);
    let selected_validation_tokens = &validation_tokens[..selected_token_end];
    let byte_luts =
        parameter_golf_sentencepiece_byte_luts_from_tokenizer_path(&config.tokenizer_path)?;
    let model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    let machine_observation = inspect_local_single_h100_machine();
    let mut legacy_graph_cache = BTreeMap::new();
    let mut device_resident_graph_cache = BTreeMap::new();
    let mut cuda_backend = CudaBackend::new();
    let Some(selected_device) = cuda_backend.selected_device().cloned() else {
        return Err(ParameterGolfSingleH100TrainingError::MissingSelectedH100);
    };

    let legacy_started = Instant::now();
    let legacy_summary = evaluate_validation_on_cuda_legacy(
        &mut cuda_backend,
        &selected_device.device,
        model.descriptor(),
        &model,
        selected_validation_tokens,
        &byte_luts,
        config.sequence_length,
        config.batch_sequences,
        &mut legacy_graph_cache,
        "legacy_validation_runtime_comparison",
        None,
    )?;
    let legacy_observed_ms = duration_ms(legacy_started);

    let device_resident_started = Instant::now();
    let device_resident_summary = evaluate_validation_on_cuda(
        &mut cuda_backend,
        &selected_device.device,
        model.descriptor(),
        &model,
        selected_validation_tokens,
        &byte_luts,
        config.sequence_length,
        config.batch_sequences,
        &ParameterGolfValidationEvalMode::NonOverlapping,
        &mut device_resident_graph_cache,
        "device_resident_validation_runtime_comparison",
        None,
    )?;
    let device_resident_observed_ms = duration_ms(device_resident_started);
    let selected_batch_count = selected_sequences.div_ceil(config.batch_sequences.max(1));

    let legacy = ParameterGolfSingleH100ValidationRuntimeComparisonLaneReceipt {
        mean_loss: legacy_summary.mean_loss,
        bits_per_byte: legacy_summary.bits_per_byte,
        observed_ms: legacy_observed_ms,
        average_batch_ms: legacy_observed_ms as f64 / selected_batch_count as f64,
        runtime_receipt: legacy_summary.runtime_receipt,
    };
    let device_resident = ParameterGolfSingleH100ValidationRuntimeComparisonLaneReceipt {
        mean_loss: device_resident_summary.mean_loss,
        bits_per_byte: device_resident_summary.bits_per_byte,
        observed_ms: device_resident_observed_ms,
        average_batch_ms: device_resident_observed_ms as f64 / selected_batch_count as f64,
        runtime_receipt: device_resident_summary.runtime_receipt,
    };
    let summary = format!(
        "Compared the legacy host-rebind validation path against the device-resident validation path on the local CUDA node over {} batch(es) of {} sequence(s); average batch time moved from {:.2}ms to {:.2}ms.",
        selected_batch_count,
        config.batch_sequences,
        legacy.average_batch_ms,
        device_resident.average_batch_ms,
    );
    let mut receipt = ParameterGolfSingleH100ValidationRuntimeComparisonReceipt {
        schema_version: 1,
        scope_window: String::from("parameter_golf_validation_runtime_comparison_v1"),
        run_id: config.run_id.clone(),
        dataset_root: config.dataset_root.clone(),
        tokenizer_path: config.tokenizer_path.clone(),
        dataset_key: config.dataset_key.clone(),
        variant: config.variant.clone(),
        tokenizer_digest,
        dataset_manifest_digest: bundle.manifest.stable_digest(),
        batch_sequences: config.batch_sequences,
        sequence_length: config.sequence_length,
        batch_limit: config.batch_limit,
        observed_cuda_health: machine_observation.observed_cuda_health,
        observed_cuda_devices: machine_observation.observed_cuda_devices,
        legacy,
        device_resident,
        summary,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = stable_digest(
        b"psionic_parameter_golf_validation_runtime_comparison_receipt|",
        &receipt_without_digest(&receipt),
    );
    Ok(receipt)
}

/// Builds the bounded single-H100 training report.
pub fn build_parameter_golf_single_h100_training_report(
    config: &ParameterGolfSingleH100TrainingConfig,
) -> Result<ParameterGolfSingleH100TrainingReport, ParameterGolfSingleH100TrainingError> {
    let (report, _) = build_parameter_golf_single_h100_training_report_inner(config, None)?;
    Ok(report)
}

fn build_parameter_golf_single_h100_training_report_inner(
    config: &ParameterGolfSingleH100TrainingConfig,
    output_path: Option<&Path>,
) -> Result<
    (
        ParameterGolfSingleH100TrainingReport,
        Option<crate::ParameterGolfSingleH100LiveVisualizationWriter>,
    ),
    ParameterGolfSingleH100TrainingError,
> {
    config.validate()?;
    let tokenizer_bytes = fs::read(&config.tokenizer_path)?;
    let tokenizer_digest = build_tokenizer_digest(tokenizer_bytes.as_slice());
    let bundle = parameter_golf_dataset_bundle_from_local_dir(
        config.dataset_key.clone(),
        &config.dataset_root,
        config.variant.clone(),
        tokenizer_digest.clone(),
        config.tokenizer_path.display().to_string(),
        None,
    )?;
    let machine_observation = inspect_local_single_h100_machine();
    let initial_model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    let optimizer_plan =
        parameter_golf_optimizer_plan(initial_model.descriptor(), &config.hyperparameters)?;
    let optimizer_plan_digest =
        stable_digest(b"psionic_parameter_golf_optimizer_plan|", &optimizer_plan);
    let precision_receipt = precision_receipt_from_optimizer_plan(&optimizer_plan);
    let capability_report = builtin_parameter_golf_cuda_training_capability_report()?;
    let started_at_ms = unix_time_ms();

    if !machine_observation.machine_contract_satisfied {
        return Ok((
            refusal_report(
                config,
                tokenizer_digest,
                &bundle,
                &machine_observation,
                &initial_model,
                optimizer_plan_digest,
                precision_receipt.clone(),
                capability_report.report_digest.clone(),
                capability_report.challenge_kernel_blockers().to_vec(),
                ParameterGolfSingleH100TrainingDisposition::RefusedMachineContract,
                machine_observation.refusal.clone(),
                started_at_ms,
                String::from(
                    "The Rust-owned single-H100 trainer path is now implemented, but this run still refused because the local machine contract does not satisfy the non-MIG H100 requirement.",
                ),
            ),
            None,
        ));
    }
    if !capability_report.challenge_kernel_blockers().is_empty() {
        return Ok((
            refusal_report(
                config,
                tokenizer_digest,
                &bundle,
                &machine_observation,
                &initial_model,
                optimizer_plan_digest,
                precision_receipt.clone(),
                capability_report.report_digest.clone(),
                capability_report.challenge_kernel_blockers().to_vec(),
                ParameterGolfSingleH100TrainingDisposition::RefusedCudaBlockers,
                Some(
                    PsionicRefusal::new(
                        PsionicRefusalCode::UnsupportedBackendCapability,
                        PsionicRefusalScope::Runtime,
                        format!(
                            "single-H100 trainer requires an empty Parameter Golf CUDA blocker list, found {:?}",
                            capability_report.challenge_kernel_blockers()
                        ),
                    )
                    .with_subject(String::from("parameter_golf_single_h100_cuda_blockers")),
                ),
                started_at_ms,
                String::from(
                    "The Rust-owned single-H100 trainer path refuses explicitly when the committed Parameter Golf CUDA capability report still carries challenge blockers.",
                ),
            ),
            None,
        ));
    }

    let mut cuda_backend = CudaBackend::new();
    let Some(selected_device) = cuda_backend.selected_device().cloned() else {
        return Err(ParameterGolfSingleH100TrainingError::MissingSelectedH100);
    };
    if !device_matches_single_h100(&selected_device, &machine_observation.thresholds) {
        return Err(ParameterGolfSingleH100TrainingError::MissingSelectedH100);
    }
    let delivered_execution =
        DeliveredExecutionContext::new("cuda", None, vec![selected_device.inventory_qualifiers()]);
    let mut live_visualization_writer = if let Some(output_path) = output_path {
        crate::ParameterGolfSingleH100LiveVisualizationWriter::try_start(
            config,
            output_path,
            started_at_ms,
        )?
    } else {
        None
    };

    emit_progress_line(format!(
        "single_h100_train_start run_id={} device={} max_steps={} iterations={} warmup_steps={} grad_accum_steps={} val_loss_every={} train_log_every={} final_validation_mode={} validation_eval_mode={} local_train_sequences={} local_validation_sequences={} max_wallclock_seconds={}",
        config.run_id,
        selected_device.device_name.as_deref().unwrap_or("unknown"),
        config.max_steps,
        config.hyperparameters.iterations,
        config.warmup_steps,
        config.geometry.grad_accum_steps,
        config.validation_loss_every,
        config.train_log_every,
        config.final_validation_mode.as_str(),
        config.validation_eval_mode.as_str(),
        config.geometry.local_train_batch_sequences(),
        config.geometry.local_validation_batch_sequences(),
        config.hyperparameters.max_wallclock_seconds.unwrap_or(0.0),
    ));
    if let Some(writer) = live_visualization_writer.as_mut() {
        writer.record_phase(
            "training",
            Some(String::from("warmup")),
            "The single-H100 trainer started and entered warmup or measured training.",
            vec![String::from("dataloader"), String::from("trainer_boot")],
            None,
            None,
            true,
        )?;
    }

    let byte_luts =
        parameter_golf_sentencepiece_byte_luts_from_tokenizer_path(&config.tokenizer_path)?;
    let validation_tokens = load_parameter_golf_validation_tokens_from_paths(
        &bundle
            .validation_shards
            .iter()
            .map(|receipt| PathBuf::from(&receipt.path))
            .collect::<Vec<_>>(),
        config.geometry.train_sequence_length,
    )?;
    let mut train_graph_cache = BTreeMap::new();
    let mut eval_graph_cache = BTreeMap::new();

    let mut trainer_state = seed_parameter_states(&initial_model, &optimizer_plan)?;
    let train_contract = ParameterGolfTokenStreamContract::new(
        bundle.manifest.key.clone(),
        PARAMETER_GOLF_TRAIN_SPLIT_NAME,
    )
    .with_mode(DatasetIterationMode::Repeat);
    let mut cursor = ParameterGolfTokenStreamCursor::new(PARAMETER_GOLF_TRAIN_SPLIT_NAME);
    let mut current_model = initial_model.clone();
    let mut step_metrics = Vec::new();
    let mut validation_checkpoints = Vec::new();
    let mut aggregate_phase_timings = ParameterGolfSingleH100PhaseTimings::default();
    let requested_train_tokens =
        config.geometry.local_train_batch_tokens().saturating_add(1) as u64;

    let mut warmup_observed_ms = 0_u64;
    if config.warmup_steps > 0 {
        let warmup_started = Instant::now();
        let trainer_state_checkpoint = trainer_state.clone();
        let cursor_checkpoint = cursor.clone();
        let current_model_checkpoint = current_model.clone();
        for warmup_step in 0..config.warmup_steps {
            let learning_rate_multiplier = 1.0;
            let muon_momentum = config.hyperparameters.muon_momentum_at_step(warmup_step);
            let effective_learning_rate = Some(
                config
                    .hyperparameters
                    .token_learning_rate(current_model.descriptor().config.tie_embeddings)
                    * learning_rate_multiplier,
            );
            execute_training_step(
                &mut cuda_backend,
                &selected_device.device,
                &bundle,
                &train_contract,
                &mut cursor,
                &initial_model,
                &mut current_model,
                &mut trainer_state,
                &mut train_graph_cache,
                &config.geometry,
                requested_train_tokens,
                warmup_step + 1,
                config.warmup_steps,
                config.hyperparameters.grad_clip_norm,
                learning_rate_multiplier,
                muon_momentum,
                effective_learning_rate,
                false,
                &mut live_visualization_writer,
            )?;
            if config.warmup_steps <= 20
                || (warmup_step + 1) % 10 == 0
                || warmup_step + 1 == config.warmup_steps
            {
                emit_progress_line(format!(
                    "warmup_step:{}/{}",
                    warmup_step + 1,
                    config.warmup_steps
                ));
            }
            if let Some(writer) = live_visualization_writer.as_mut() {
                writer.record_phase(
                    "training",
                    Some(String::from("warmup")),
                    format!(
                        "Warmup step {} of {} completed.",
                        warmup_step + 1,
                        config.warmup_steps
                    ),
                    vec![String::from("warmup"), String::from("dataloader")],
                    Some(warmup_step + 1),
                    Some(config.geometry.grad_accum_steps as u32),
                    false,
                )?;
            }
        }
        warmup_observed_ms = duration_ms(warmup_started);
        trainer_state = trainer_state_checkpoint;
        cursor = cursor_checkpoint;
        current_model = current_model_checkpoint;
        emit_progress_line(format!(
            "warmup_restore_complete steps={} elapsed_ms={}",
            config.warmup_steps, warmup_observed_ms
        ));
        if let Some(writer) = live_visualization_writer.as_mut() {
            writer.record_event(
                crate::RemoteTrainingEventSeverity::Info,
                "warmup_completed",
                format!(
                    "Warmup completed and the trainer restored the measured-state checkpoint after {} ms.",
                    warmup_observed_ms
                ),
            );
            writer.record_phase(
                "training",
                Some(String::from("optimizer_step")),
                "Warmup completed and the trainer resumed measured optimizer steps.",
                vec![String::from("dataloader"), String::from("optimizer")],
                None,
                None,
                true,
            )?;
        }
    }

    let max_wallclock_ms = config
        .hyperparameters
        .max_wallclock_seconds
        .filter(|seconds| *seconds > 0.0)
        .map(|seconds| (seconds * 1000.0) as u64);
    let mut training_time_ms = 0_u64;
    let mut step = 0_u64;
    let mut stop_reason = None;
    let mut initial_validation = None;
    let mut pre_export_final_validation = None;
    let mut pre_export_final_validation_observed_ms = None;

    loop {
        let last_step = step == config.max_steps || stop_reason.is_some();
        let should_validate = if last_step {
            config.final_validation_mode.runs_live_validation()
        } else {
            config.validation_loss_every > 0 && step % config.validation_loss_every == 0
        };
        if should_validate {
            let stage_label = if last_step {
                String::from("final_validation")
            } else if step == 0 {
                String::from("initial_validation")
            } else {
                format!("periodic_validation_step_{step}")
            };
            emit_progress_line(format!(
                "{}_start sequences={} batch_sequences={}",
                stage_label,
                (validation_tokens.len() - 1) / config.geometry.train_sequence_length,
                config.geometry.local_validation_batch_sequences(),
            ));
            if let Some(writer) = live_visualization_writer.as_mut() {
                writer.record_phase(
                    "training",
                    Some(String::from("validation")),
                    format!(
                        "Validation stage `{stage_label}` started for step {}.",
                        step
                    ),
                    vec![String::from("validation"), String::from("gpu_sampling")],
                    Some(step.max(1)),
                    Some(0),
                    true,
                )?;
                writer.record_event(
                    crate::RemoteTrainingEventSeverity::Info,
                    "validation_started",
                    format!("Validation stage `{stage_label}` started."),
                );
            }
            let validation_started = Instant::now();
            let validation_summary = evaluate_validation_on_cuda(
                &mut cuda_backend,
                &selected_device.device,
                current_model.descriptor(),
                &current_model,
                validation_tokens.as_slice(),
                &byte_luts,
                config.geometry.train_sequence_length,
                config.geometry.local_validation_batch_sequences(),
                &config.validation_eval_mode,
                &mut eval_graph_cache,
                &stage_label,
                live_visualization_writer.as_mut(),
            )?;
            let observed_validation_ms = duration_ms(validation_started);
            if last_step {
                pre_export_final_validation_observed_ms = Some(observed_validation_ms);
                pre_export_final_validation = Some(validation_summary.clone());
                if let Some(writer) = live_visualization_writer.as_mut() {
                    writer.record_validation_checkpoint(
                        ParameterGolfSingleH100ValidationCheckpoint {
                            stage_label: stage_label.clone(),
                            trigger_step: step,
                            observed_training_time_ms: training_time_ms,
                            observed_validation_ms,
                            summary: validation_summary,
                        },
                        crate::ValidationSlot::PreExportFinal,
                    )?;
                }
            } else {
                if step == 0 {
                    initial_validation = Some(validation_summary.clone());
                    if let Some(writer) = live_visualization_writer.as_mut() {
                        writer.record_validation_checkpoint(
                            ParameterGolfSingleH100ValidationCheckpoint {
                                stage_label: stage_label.clone(),
                                trigger_step: step,
                                observed_training_time_ms: training_time_ms,
                                observed_validation_ms,
                                summary: validation_summary.clone(),
                            },
                            crate::ValidationSlot::Initial,
                        )?;
                    }
                }
                validation_checkpoints.push(ParameterGolfSingleH100ValidationCheckpoint {
                    stage_label,
                    trigger_step: step,
                    observed_training_time_ms: training_time_ms,
                    observed_validation_ms,
                    summary: validation_summary,
                });
                if let Some(writer) = live_visualization_writer.as_mut() {
                    writer.record_validation_checkpoint(
                        validation_checkpoints
                            .last()
                            .expect("validation checkpoint should exist")
                            .clone(),
                        crate::ValidationSlot::Periodic,
                    )?;
                }
            }
        } else if last_step {
            emit_progress_line(format!(
                "final_validation_skipped mode={} reason=explicit_final_validation_mode",
                config.final_validation_mode.as_str(),
            ));
        }

        if last_step {
            if stop_reason == Some(ParameterGolfSingleH100TrainingStopReason::WallclockCapReached)
                && step < config.max_steps
            {
                emit_progress_line(format!(
                    "stopping_early: wallclock_cap train_time:{}ms step:{}/{}",
                    training_time_ms, step, config.max_steps
                ));
            }
            break;
        }

        let learning_rate_multiplier = config
            .hyperparameters
            .learning_rate_multiplier(step, training_time_ms as f32);
        let muon_momentum = config.hyperparameters.muon_momentum_at_step(step);
        let effective_learning_rate = Some(
            config
                .hyperparameters
                .token_learning_rate(current_model.descriptor().config.tie_embeddings)
                * learning_rate_multiplier,
        );
        let step_metrics_next = execute_training_step(
            &mut cuda_backend,
            &selected_device.device,
            &bundle,
            &train_contract,
            &mut cursor,
            &initial_model,
            &mut current_model,
            &mut trainer_state,
            &mut train_graph_cache,
            &config.geometry,
            requested_train_tokens,
            step + 1,
            config.max_steps,
            config.hyperparameters.grad_clip_norm,
            learning_rate_multiplier,
            muon_momentum,
            effective_learning_rate,
            true,
            &mut live_visualization_writer,
        )?;
        training_time_ms = training_time_ms.saturating_add(step_metrics_next.observed_wallclock_ms);
        aggregate_phase_timings.accumulate(&step_metrics_next.phase_timings);
        if let Some(writer) = live_visualization_writer.as_mut() {
            writer.record_step(step_metrics_next.clone())?;
        }
        step_metrics.push(step_metrics_next);
        step += 1;

        let should_log_train = config.train_log_every > 0
            && (step <= 10 || step % config.train_log_every == 0 || stop_reason.is_some());
        if should_log_train {
            let latest_step = step_metrics
                .last()
                .expect("step metrics should be present after a completed step");
            emit_progress_line(format!(
                "step:{}/{} train_loss:{:.4} train_time:{}ms step_avg:{:.2}ms",
                step,
                config.max_steps,
                latest_step.mean_microbatch_loss,
                training_time_ms,
                training_time_ms as f64 / step.max(1) as f64,
            ));
        }

        if stop_reason.is_none()
            && max_wallclock_ms.is_some_and(|wallclock_ms| training_time_ms >= wallclock_ms)
        {
            stop_reason = Some(ParameterGolfSingleH100TrainingStopReason::WallclockCapReached);
        }
    }

    let compressed_model_artifact =
        export_parameter_golf_int8_zlib_model_artifact(&current_model, &config.run_id, step)?;
    let mut final_validation = pre_export_final_validation.clone();
    let mut final_validation_observed_ms = pre_export_final_validation_observed_ms;
    let mut final_roundtrip_receipt = None;
    if config.final_validation_mode.runs_roundtrip_validation() {
        emit_progress_line(format!(
            "final_int8_zlib_roundtrip_start sequences={} batch_sequences={} compressed_model_bytes={} artifact_ref={} artifact_digest={}",
            (validation_tokens.len() - 1) / config.geometry.train_sequence_length,
            config.geometry.local_validation_batch_sequences(),
            compressed_model_artifact.bytes.len(),
            compressed_model_artifact.artifact_ref,
            compressed_model_artifact.artifact_digest,
        ));
        let roundtrip_model = restore_parameter_golf_model_from_int8_zlib(
            &initial_model,
            compressed_model_artifact.bytes.as_slice(),
        )?;
        let roundtrip_validation_started = Instant::now();
        let roundtrip_validation = evaluate_validation_on_cuda(
            &mut cuda_backend,
            &selected_device.device,
            roundtrip_model.descriptor(),
            &roundtrip_model,
            validation_tokens.as_slice(),
            &byte_luts,
            config.geometry.train_sequence_length,
            config.geometry.local_validation_batch_sequences(),
            &config.validation_eval_mode,
            &mut eval_graph_cache,
            "final_int8_zlib_roundtrip",
            live_visualization_writer.as_mut(),
        )?;
        let roundtrip_observed_ms = duration_ms(roundtrip_validation_started);
        emit_progress_line(format!(
            "final_int8_zlib_roundtrip val_loss:{:.4} val_bpb:{:.4} eval_time:{}ms compressed_model_bytes={} artifact_ref={} artifact_digest={}",
            roundtrip_validation.mean_loss,
            roundtrip_validation.bits_per_byte,
            roundtrip_observed_ms,
            compressed_model_artifact.bytes.len(),
            compressed_model_artifact.artifact_ref,
            compressed_model_artifact.artifact_digest,
        ));
        emit_progress_line(format!(
            "final_int8_zlib_roundtrip_exact val_loss:{:.8} val_bpb:{:.8}",
            roundtrip_validation.mean_loss, roundtrip_validation.bits_per_byte,
        ));
        final_validation_observed_ms = Some(roundtrip_observed_ms);
        final_validation = Some(roundtrip_validation.clone());
        final_roundtrip_receipt = Some(ParameterGolfSingleH100RoundtripReceipt {
            metric_source: String::from("int8_zlib_roundtrip"),
            validation: roundtrip_validation,
            observed_eval_ms: roundtrip_observed_ms,
            compressed_model_bytes: compressed_model_artifact.bytes.len() as u64,
            compressed_model_artifact_ref: compressed_model_artifact.artifact_ref.clone(),
            compressed_model_artifact_digest: compressed_model_artifact.artifact_digest.clone(),
        });
        if let Some(writer) = live_visualization_writer.as_mut() {
            writer.record_roundtrip_receipt(
                final_roundtrip_receipt
                    .clone()
                    .expect("roundtrip receipt should be present"),
            )?;
            writer.record_validation_checkpoint(
                ParameterGolfSingleH100ValidationCheckpoint {
                    stage_label: String::from("final_roundtrip"),
                    trigger_step: step,
                    observed_training_time_ms: training_time_ms,
                    observed_validation_ms: roundtrip_observed_ms,
                    summary: final_validation
                        .clone()
                        .expect("final validation should exist for roundtrip"),
                },
                crate::ValidationSlot::FinalRoundtrip,
            )?;
        }
    } else {
        emit_progress_line(format!(
            "final_int8_zlib_roundtrip_skipped mode={} reason=explicit_final_validation_mode compressed_model_bytes={} artifact_ref={} artifact_digest={}",
            config.final_validation_mode.as_str(),
            compressed_model_artifact.bytes.len(),
            compressed_model_artifact.artifact_ref,
            compressed_model_artifact.artifact_digest,
        ));
        if let Some(writer) = live_visualization_writer.as_mut() {
            writer.record_event(
                crate::RemoteTrainingEventSeverity::Info,
                "final_roundtrip_skipped",
                format!(
                    "The final int8+zlib roundtrip validation was skipped because final_validation_mode={}.",
                    config.final_validation_mode.as_str()
                ),
            );
        }
    }

    let finished_at_ms = unix_time_ms();
    let observed_wallclock_ms = finished_at_ms.saturating_sub(started_at_ms);
    let realized_stop_reason =
        stop_reason.unwrap_or(ParameterGolfSingleH100TrainingStopReason::StepBudgetReached);
    let final_metric_surface = match config.final_validation_mode {
        ParameterGolfSingleH100ValidationMode::LiveOnly => {
            "reported final live-model validation metrics without running the exported int8+zlib roundtrip validation"
        }
        ParameterGolfSingleH100ValidationMode::RoundtripOnly => {
            "reported canonical final contest metrics from the exported int8+zlib roundtrip artifact without also replaying the live-model final validation"
        }
        ParameterGolfSingleH100ValidationMode::Both => {
            "preserved the pre-export live-model validation separately and reported canonical final contest metrics from the exported int8+zlib roundtrip artifact"
        }
    };
    let summary = format!(
        "The Rust-owned single-H100 trainer executed {} optimizer step(s) with challenge single-device geometry on CUDA, used the widened train_gpt.py-style warmup, validation, and wallclock-stop control loop, ran with final_validation_mode={} and validation_eval_mode={}, {} before stopping via {:?}.",
        step,
        config.final_validation_mode.as_str(),
        config.validation_eval_mode.as_str(),
        final_metric_surface,
        realized_stop_reason
    );

    let mut report = ParameterGolfSingleH100TrainingReport {
        schema_version: 2,
        scope_window: String::from("parameter_golf_single_h100_training_v2"),
        run_id: config.run_id.clone(),
        dataset_root: config.dataset_root.clone(),
        tokenizer_path: config.tokenizer_path.clone(),
        dataset_key: config.dataset_key.clone(),
        variant: config.variant.clone(),
        tokenizer_digest,
        dataset_manifest_digest: bundle.manifest.stable_digest(),
        train_shard_count: bundle.train_shards.len(),
        validation_shard_count: bundle.validation_shards.len(),
        train_token_count: split_token_count(&bundle, PARAMETER_GOLF_TRAIN_SPLIT_NAME),
        validation_token_count: split_token_count(&bundle, "validation"),
        geometry: config.geometry.clone(),
        hyperparameters: config.hyperparameters.clone(),
        max_steps: config.max_steps,
        warmup_steps: config.warmup_steps,
        completed_warmup_steps: config.warmup_steps,
        validation_loss_every: config.validation_loss_every,
        train_log_every: config.train_log_every,
        final_validation_mode: config.final_validation_mode,
        validation_eval_mode: config.validation_eval_mode.clone(),
        executed_steps: step,
        stop_reason: Some(realized_stop_reason),
        delivered_execution,
        machine_thresholds: machine_observation.thresholds.clone(),
        observed_cuda_health: machine_observation.observed_cuda_health.clone(),
        cuda_discovery_error: machine_observation.cuda_discovery_error.clone(),
        observed_cuda_devices: machine_observation.observed_cuda_devices.clone(),
        matching_h100_device_count: machine_observation.matching_h100_device_count,
        machine_contract_satisfied: machine_observation.machine_contract_satisfied,
        baseline_model_id: String::from(PARAMETER_GOLF_BASELINE_MODEL_ID),
        baseline_model_revision: String::from(PARAMETER_GOLF_BASELINE_REVISION),
        baseline_model_descriptor_digest: current_model.descriptor().stable_digest(),
        optimizer_plan_digest,
        precision_receipt: precision_receipt_from_trainer_state(&trainer_state),
        cuda_training_capability_report_digest: capability_report.report_digest.clone(),
        challenge_kernel_blockers: capability_report.challenge_kernel_blockers().to_vec(),
        validation_checkpoints,
        initial_validation,
        pre_export_final_validation,
        final_validation,
        warmup_observed_ms,
        observed_training_time_ms: training_time_ms,
        pre_export_final_validation_observed_ms,
        final_validation_observed_ms,
        final_roundtrip_receipt,
        compressed_model_bytes: Some(compressed_model_artifact.bytes.len() as u64),
        compressed_model_artifact_ref: Some(compressed_model_artifact.artifact_ref.clone()),
        compressed_model_artifact_digest: Some(compressed_model_artifact.artifact_digest.clone()),
        step_metrics,
        aggregate_phase_timings: Some(aggregate_phase_timings),
        final_training_cursor: Some(cursor),
        started_at_ms,
        finished_at_ms,
        observed_wallclock_ms,
        disposition: ParameterGolfSingleH100TrainingDisposition::TrainingExecuted,
        refusal: None,
        claim_boundary: String::from(
            "bounded_single_h100_accelerated_trainer; challenge geometry and optimizer contract are real, but this report does not claim 8xH100, record-track, or challenge-speed closure",
        ),
        summary,
        report_digest: String::new(),
    };
    report.report_digest = stable_digest(
        b"psionic_parameter_golf_single_h100_training_report|",
        &report_without_digest(&report),
    );
    Ok((report, live_visualization_writer))
}

fn refusal_report(
    config: &ParameterGolfSingleH100TrainingConfig,
    tokenizer_digest: psionic_data::TokenizerDigest,
    bundle: &ParameterGolfDatasetBundle,
    machine_observation: &ParameterGolfSingleH100MachineObservation,
    initial_model: &ParameterGolfReferenceModel,
    optimizer_plan_digest: String,
    precision_receipt: ParameterGolfSingleH100PrecisionReceipt,
    capability_report_digest: String,
    challenge_kernel_blockers: Vec<String>,
    disposition: ParameterGolfSingleH100TrainingDisposition,
    refusal: Option<PsionicRefusal>,
    started_at_ms: u64,
    summary: String,
) -> ParameterGolfSingleH100TrainingReport {
    let finished_at_ms = unix_time_ms();
    let mut report = ParameterGolfSingleH100TrainingReport {
        schema_version: 2,
        scope_window: String::from("parameter_golf_single_h100_training_v2"),
        run_id: config.run_id.clone(),
        dataset_root: config.dataset_root.clone(),
        tokenizer_path: config.tokenizer_path.clone(),
        dataset_key: config.dataset_key.clone(),
        variant: config.variant.clone(),
        tokenizer_digest,
        dataset_manifest_digest: bundle.manifest.stable_digest(),
        train_shard_count: bundle.train_shards.len(),
        validation_shard_count: bundle.validation_shards.len(),
        train_token_count: split_token_count(bundle, PARAMETER_GOLF_TRAIN_SPLIT_NAME),
        validation_token_count: split_token_count(bundle, "validation"),
        geometry: config.geometry.clone(),
        hyperparameters: config.hyperparameters.clone(),
        max_steps: config.max_steps,
        warmup_steps: config.warmup_steps,
        completed_warmup_steps: 0,
        validation_loss_every: config.validation_loss_every,
        train_log_every: config.train_log_every,
        final_validation_mode: config.final_validation_mode,
        validation_eval_mode: config.validation_eval_mode.clone(),
        executed_steps: 0,
        stop_reason: None,
        delivered_execution: DeliveredExecutionContext::new("cuda", None, Vec::new()),
        machine_thresholds: machine_observation.thresholds.clone(),
        observed_cuda_health: machine_observation.observed_cuda_health.clone(),
        cuda_discovery_error: machine_observation.cuda_discovery_error.clone(),
        observed_cuda_devices: machine_observation.observed_cuda_devices.clone(),
        matching_h100_device_count: machine_observation.matching_h100_device_count,
        machine_contract_satisfied: machine_observation.machine_contract_satisfied,
        baseline_model_id: String::from(PARAMETER_GOLF_BASELINE_MODEL_ID),
        baseline_model_revision: String::from(PARAMETER_GOLF_BASELINE_REVISION),
        baseline_model_descriptor_digest: initial_model.descriptor().stable_digest(),
        optimizer_plan_digest,
        precision_receipt,
        cuda_training_capability_report_digest: capability_report_digest,
        challenge_kernel_blockers,
        validation_checkpoints: Vec::new(),
        initial_validation: None,
        pre_export_final_validation: None,
        final_validation: None,
        warmup_observed_ms: 0,
        observed_training_time_ms: 0,
        pre_export_final_validation_observed_ms: None,
        final_validation_observed_ms: None,
        final_roundtrip_receipt: None,
        compressed_model_bytes: None,
        compressed_model_artifact_ref: None,
        compressed_model_artifact_digest: None,
        step_metrics: Vec::new(),
        aggregate_phase_timings: None,
        final_training_cursor: None,
        started_at_ms,
        finished_at_ms,
        observed_wallclock_ms: finished_at_ms.saturating_sub(started_at_ms),
        disposition,
        refusal,
        claim_boundary: String::from(
            "single_h100_accelerated_trainer_refusal; dataset and model contracts are bound, but no training artifact is claimed when machine or CUDA blocker admission fails",
        ),
        summary,
        report_digest: String::new(),
    };
    report.report_digest = stable_digest(
        b"psionic_parameter_golf_single_h100_training_report|",
        &report_without_digest(&report),
    );
    report
}

pub(crate) fn seed_parameter_states(
    model: &ParameterGolfReferenceModel,
    optimizer_plan: &crate::ParameterGolfOptimizerPlan,
) -> Result<ParameterGolfSingleH100TrainerState, ParameterGolfSingleH100TrainingError> {
    let parameter_vectors = model
        .weights()
        .parameter_vectors(&model.descriptor().config)
        .into_iter()
        .map(|vector| (vector.parameter_id.clone(), vector))
        .collect::<BTreeMap<_, _>>();
    let mut parameter_states = BTreeMap::new();
    for group in &optimizer_plan.groups {
        for parameter_id in &group.tensor_names {
            let vector = parameter_vectors.get(parameter_id).ok_or_else(|| {
                ParameterGolfSingleH100TrainingError::MissingParameterState {
                    parameter_id: parameter_id.clone(),
                }
            })?;
            let state = match &group.execution {
                ParameterGolfOptimizerExecution::Adam { optimizer } => match group.kind {
                    ParameterGolfOptimizerGroupKind::TokenEmbeddingAdam
                    | ParameterGolfOptimizerGroupKind::UntiedLmHeadAdam => {
                        let mut train_visible_values = vector.values.clone();
                        round_values_to_bf16(train_visible_values.as_mut_slice());
                        ParameterGolfParameterState::AdamBf16Master {
                            shape: vector.shape.dims().to_vec(),
                            train_visible_values,
                            master_weight_values: vector.values.clone(),
                            optimizer: optimizer.clone(),
                            optimizer_state: optimizer.initialize_state(vector.values.len()),
                            last_step_receipt: None,
                        }
                    }
                    ParameterGolfOptimizerGroupKind::ScalarControlAdam => {
                        ParameterGolfParameterState::AdamFp32 {
                            shape: vector.shape.dims().to_vec(),
                            values: vector.values.clone(),
                            optimizer: optimizer.clone(),
                            optimizer_state: optimizer.initialize_state(vector.values.len()),
                        }
                    }
                    ParameterGolfOptimizerGroupKind::MatrixMuon => {
                        return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                            message: format!(
                                "optimizer group `{}` declared Adam execution for matrix Muon posture",
                                group.group_id
                            ),
                        });
                    }
                },
                ParameterGolfOptimizerExecution::Muon { optimizer } => {
                    let rows = vector.shape.dims().first().copied().unwrap_or(0);
                    let cols = vector.shape.dims().get(1).copied().unwrap_or(0);
                    let mut values = vector.values.clone();
                    round_values_to_bf16(values.as_mut_slice());
                    ParameterGolfParameterState::MuonBf16 {
                        shape: vector.shape.dims().to_vec(),
                        values,
                        optimizer: optimizer.clone(),
                        optimizer_state: crate::ParameterGolfMuonState::zeros(rows, cols),
                    }
                }
            };
            parameter_states.insert(parameter_id.clone(), state);
        }
    }
    Ok(ParameterGolfSingleH100TrainerState { parameter_states })
}

fn precision_receipt_from_optimizer_plan(
    optimizer_plan: &ParameterGolfOptimizerPlan,
) -> ParameterGolfSingleH100PrecisionReceipt {
    ParameterGolfSingleH100PrecisionReceipt {
        graph_parameter_upload_precision: TrainingPrecisionMode::Bf16,
        graph_execution_precision: TrainingPrecisionMode::Fp32,
        retained_activation_precision: TrainingPrecisionMode::Fp32,
        group_receipts: optimizer_plan
            .groups
            .iter()
            .map(|group| match group.kind {
                ParameterGolfOptimizerGroupKind::TokenEmbeddingAdam
                | ParameterGolfOptimizerGroupKind::UntiedLmHeadAdam => {
                    ParameterGolfSingleH100GroupPrecisionReceipt {
                        group_id: group.group_id.clone(),
                        parameter_precision: TrainingPrecisionMode::Bf16,
                        gradient_precision: TrainingPrecisionMode::Bf16,
                        optimizer_state_precision: TrainingPrecisionMode::Fp32,
                        master_weight_precision: Some(TrainingPrecisionMode::Fp32),
                    }
                }
                ParameterGolfOptimizerGroupKind::MatrixMuon => {
                    ParameterGolfSingleH100GroupPrecisionReceipt {
                        group_id: group.group_id.clone(),
                        parameter_precision: TrainingPrecisionMode::Bf16,
                        gradient_precision: TrainingPrecisionMode::Fp32,
                        optimizer_state_precision: TrainingPrecisionMode::Fp32,
                        master_weight_precision: None,
                    }
                }
                ParameterGolfOptimizerGroupKind::ScalarControlAdam => {
                    ParameterGolfSingleH100GroupPrecisionReceipt {
                        group_id: group.group_id.clone(),
                        parameter_precision: TrainingPrecisionMode::Fp32,
                        gradient_precision: TrainingPrecisionMode::Fp32,
                        optimizer_state_precision: TrainingPrecisionMode::Fp32,
                        master_weight_precision: None,
                    }
                }
            })
            .collect(),
        notes: vec![String::from(
            "train-visible Parameter Golf weights now upload through BF16 graph inputs on the token-embedding and linear hot path while scalar/control tensors and retained activations remain explicit F32 until wider BF16 execution kernels land",
        )],
    }
}

fn precision_receipt_from_trainer_state(
    state: &ParameterGolfSingleH100TrainerState,
) -> ParameterGolfSingleH100PrecisionReceipt {
    ParameterGolfSingleH100PrecisionReceipt {
        graph_parameter_upload_precision: TrainingPrecisionMode::Bf16,
        graph_execution_precision: TrainingPrecisionMode::Fp32,
        retained_activation_precision: TrainingPrecisionMode::Fp32,
        group_receipts: state
            .parameter_states
            .iter()
            .map(|(parameter_id, state)| state.precision_receipt(parameter_id.clone()))
            .collect(),
        notes: vec![String::from(
            "train-visible Parameter Golf weights now upload through BF16 graph inputs on the token-embedding and linear hot path while scalar/control tensors and retained activations remain explicit F32 until wider BF16 execution kernels land",
        )],
    }
}

fn round_values_to_bf16(values: &mut [f32]) {
    for value in values {
        *value = bf16::from_f32(*value).to_f32();
    }
}

pub(crate) fn zero_gradients(
    state: &ParameterGolfSingleH100TrainerState,
) -> Vec<(String, Vec<f32>)> {
    state
        .parameter_states
        .iter()
        .map(|(parameter_id, state)| (parameter_id.clone(), vec![0.0; state.values().len()]))
        .collect()
}

pub(crate) fn flatten_parameter_golf_optimizer_group_values(
    optimizer_plan: &ParameterGolfOptimizerPlan,
    values_by_parameter: &BTreeMap<String, Vec<f32>>,
) -> Result<BTreeMap<String, Vec<f32>>, ParameterGolfSingleH100TrainingError> {
    optimizer_plan
        .groups
        .iter()
        .map(|group| {
            let mut values = Vec::with_capacity(group.parameter_count);
            for parameter_id in &group.tensor_names {
                let parameter_values = values_by_parameter.get(parameter_id).ok_or_else(|| {
                    ParameterGolfSingleH100TrainingError::MissingParameterState {
                        parameter_id: parameter_id.clone(),
                    }
                })?;
                values.extend_from_slice(parameter_values);
            }
            if values.len() != group.parameter_count {
                return Err(ParameterGolfSingleH100TrainingError::Serialization {
                    message: format!(
                        "optimizer group `{}` expected {} flattened values but produced {}",
                        group.group_id,
                        group.parameter_count,
                        values.len()
                    ),
                });
            }
            Ok((group.group_id.clone(), values))
        })
        .collect()
}

pub(crate) fn flatten_parameter_golf_optimizer_group_buffers(
    optimizer_plan: &ParameterGolfOptimizerPlan,
    values_by_parameter: &BTreeMap<String, Vec<f32>>,
) -> Result<BTreeMap<String, crate::TrainingTensorBuffer>, ParameterGolfSingleH100TrainingError> {
    flatten_parameter_golf_optimizer_group_values(optimizer_plan, values_by_parameter)?
        .into_iter()
        .map(|(group_id, values)| {
            let value_len = values.len();
            Ok((
                group_id.clone(),
                crate::TrainingTensorBuffer::from_f32(
                    group_id.clone(),
                    psionic_core::TensorSpec::new(
                        Shape::new(vec![value_len]),
                        DType::F32,
                        psionic_core::Device::cpu(),
                    ),
                    values,
                )
                .map_err(|error| ParameterGolfSingleH100TrainingError::Serialization {
                    message: format!(
                        "failed to build flattened optimizer group buffer `{group_id}` with {value_len} values: {error}"
                    ),
                })?,
            ))
        })
        .collect()
}

pub(crate) fn accumulate_gradients(
    accumulated: &mut [(String, Vec<f32>)],
    trainer_state: &ParameterGolfSingleH100TrainerState,
    gradients: &BTreeMap<String, Vec<f32>>,
    divisor: f32,
) -> Result<(), ParameterGolfSingleH100TrainingError> {
    for (parameter_id, values) in accumulated {
        let gradient = gradients.get(parameter_id).ok_or_else(|| {
            ParameterGolfSingleH100TrainingError::MissingParameterState {
                parameter_id: parameter_id.clone(),
            }
        })?;
        let expected_len = trainer_state
            .parameter_states
            .get(parameter_id)
            .ok_or_else(
                || ParameterGolfSingleH100TrainingError::MissingParameterState {
                    parameter_id: parameter_id.clone(),
                },
            )?
            .values()
            .len();
        if gradient.len() != expected_len {
            return Err(ParameterGolfSingleH100TrainingError::Serialization {
                message: format!(
                    "gradient length mismatch for `{parameter_id}`: expected {expected_len}, found {}",
                    gradient.len()
                ),
            });
        }
        for (accumulated, gradient) in values.iter_mut().zip(gradient.iter()) {
            *accumulated += *gradient / divisor;
        }
    }
    Ok(())
}

pub(crate) fn apply_gradients_to_state(
    state: &mut ParameterGolfSingleH100TrainerState,
    gradients: &[(String, Vec<f32>)],
    learning_rate_multiplier: f32,
    muon_momentum: f32,
    step_number: u64,
) -> Result<(), ParameterGolfSingleH100TrainingError> {
    for (parameter_id, gradient_values) in gradients {
        let parameter_state = state
            .parameter_states
            .get_mut(parameter_id)
            .ok_or_else(
                || ParameterGolfSingleH100TrainingError::MissingParameterState {
                    parameter_id: parameter_id.clone(),
                },
            )?;
        parameter_state.apply_gradients(
            gradient_values.as_slice(),
            learning_rate_multiplier,
            muon_momentum,
            step_number,
        )?;
    }
    Ok(())
}

pub(crate) fn materialize_current_model(
    baseline: &ParameterGolfReferenceModel,
    state: &ParameterGolfSingleH100TrainerState,
) -> Result<ParameterGolfReferenceModel, ParameterGolfSingleH100TrainingError> {
    let overrides = state
        .parameter_states
        .iter()
        .map(|(parameter_id, state)| (parameter_id.clone(), state.values().to_vec()))
        .collect::<BTreeMap<_, _>>();
    let weights = baseline
        .weights()
        .with_parameter_overrides(&baseline.descriptor().config, &overrides)?;
    Ok(ParameterGolfReferenceModel::new(
        baseline.descriptor().model.clone(),
        baseline.descriptor().config.clone(),
        weights,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_parameter_golf_training_gradient_batch(
    cuda_backend: &mut CudaBackend,
    device: &psionic_core::Device,
    bundle: &ParameterGolfDatasetBundle,
    current_model: &ParameterGolfReferenceModel,
    graph_cache: &mut BTreeMap<usize, ParameterGolfBaselineTrainingGraph>,
    geometry: &ParameterGolfBatchGeometry,
    window: &ParameterGolfTokenStreamWindow,
) -> Result<ParameterGolfTrainingGradientBatchResult, ParameterGolfSingleH100TrainingError> {
    let mut phase_timings = ParameterGolfSingleH100PhaseTimings::default();

    let tokens_started = Instant::now();
    let tokens = materialize_parameter_golf_token_window(bundle, window)?;
    phase_timings.token_materialization_ms = duration_ms(tokens_started);
    let (input_ids, target_ids) = training_batch_from_window_tokens(tokens.as_slice(), geometry)?;

    let batch_size = input_ids.len();
    let graph = training_graph_for_batch(
        graph_cache,
        device.clone(),
        current_model.descriptor(),
        batch_size,
        geometry.train_sequence_length,
    )?;
    let inputs = bind_parameter_golf_baseline_training_graph_inputs(
        graph,
        current_model,
        input_ids.as_slice(),
        target_ids.as_slice(),
    )?;
    let backward_plan = graph.graph.backward_plan(graph.loss_tensor_id)?;
    let retained_graph = retained_forward_graph(graph, &backward_plan);

    let forward_started = Instant::now();
    let forward_outputs = execute_cuda_graph_outputs(cuda_backend, &retained_graph, &inputs)?;
    phase_timings.forward_loss_cuda_ms = duration_ms(forward_started);
    phase_timings.retained_binding_tensor_count = backward_plan.primal_bindings.len() as u32;
    phase_timings.retained_binding_f32_count =
        count_output_elements(&retained_graph, &backward_plan.primal_bindings)?;

    let loss = scalar_float_graph_output(&forward_outputs, graph.loss_tensor_id)?;

    let backward_started = Instant::now();
    let backward_outputs =
        execute_backward_plan(cuda_backend, &backward_plan, forward_outputs.as_slice())?;
    phase_timings.backward_cuda_ms = duration_ms(backward_started);
    phase_timings.gradient_tensor_count = backward_plan.gradient_targets.len() as u32;
    phase_timings.gradient_f32_count = count_gradient_elements(&backward_plan)?;

    let materialize_started = Instant::now();
    let backward_result =
        backward_result_from_outputs(&backward_plan, backward_outputs.as_slice())?;
    let gradients = materialize_parameter_golf_baseline_training_gradients(
        graph,
        &backward_result,
        &current_model.descriptor().config,
        input_ids.as_slice(),
    )?;
    phase_timings.host_gradient_materialization_ms = duration_ms(materialize_started);

    Ok(ParameterGolfTrainingGradientBatchResult {
        window_id: window.window_id.clone(),
        loss,
        phase_timings,
        parameter_gradients: gradients.parameter_gradients,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_training_step(
    cuda_backend: &mut CudaBackend,
    device: &psionic_core::Device,
    bundle: &ParameterGolfDatasetBundle,
    train_contract: &ParameterGolfTokenStreamContract,
    cursor: &mut ParameterGolfTokenStreamCursor,
    baseline_model: &ParameterGolfReferenceModel,
    current_model: &mut ParameterGolfReferenceModel,
    trainer_state: &mut ParameterGolfSingleH100TrainerState,
    graph_cache: &mut BTreeMap<usize, ParameterGolfBaselineTrainingGraph>,
    geometry: &ParameterGolfBatchGeometry,
    requested_train_tokens: u64,
    global_step: u64,
    max_steps: u64,
    grad_clip_norm: f32,
    learning_rate_multiplier: f32,
    muon_momentum: f32,
    effective_learning_rate: Option<f32>,
    emit_micro_step_logs: bool,
    live_visualization_writer: &mut Option<crate::ParameterGolfSingleH100LiveVisualizationWriter>,
) -> Result<ParameterGolfSingleH100TrainingStepMetrics, ParameterGolfSingleH100TrainingError> {
    let step_started = Instant::now();
    if emit_micro_step_logs {
        emit_progress_line(format!(
            "train_step_start step={}/{} grad_accum_steps={}",
            global_step, max_steps, geometry.grad_accum_steps,
        ));
    }
    if let Some(writer) = live_visualization_writer.as_mut() {
        writer.record_phase(
            "training",
            Some(String::from("optimizer_step")),
            format!("Optimizer step {} of {} started.", global_step, max_steps),
            vec![
                String::from("dataloader"),
                String::from("forward"),
                String::from("optimizer"),
            ],
            Some(global_step),
            Some(0),
            true,
        )?;
    }

    let mut accumulated_gradients = zero_gradients(trainer_state);
    let mut microbatch_loss_sum = 0.0_f32;
    let mut window_ids = Vec::new();
    let mut step_profile = ParameterGolfSingleH100PhaseTimings::default();
    for micro_step in 0..geometry.grad_accum_steps {
        let plan_started = Instant::now();
        let window = train_contract
            .plan_window(&bundle.manifest, cursor, requested_train_tokens)?
            .ok_or_else(|| ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("could not plan the next training token window"),
            })?;
        step_profile.window_planning_ms = step_profile
            .window_planning_ms
            .saturating_add(duration_ms(plan_started));
        *cursor = window.end_cursor.clone();
        let gradient_batch = execute_parameter_golf_training_gradient_batch(
            cuda_backend,
            device,
            bundle,
            current_model,
            graph_cache,
            geometry,
            &window,
        )?;
        window_ids.push(gradient_batch.window_id.clone());
        microbatch_loss_sum += gradient_batch.loss;
        step_profile.accumulate(&gradient_batch.phase_timings);
        accumulate_gradients(
            accumulated_gradients.as_mut_slice(),
            trainer_state,
            &gradient_batch.parameter_gradients,
            geometry.grad_accum_steps as f32,
        )?;
        if emit_micro_step_logs {
            emit_progress_line(format!(
                "micro_step_complete step={}/{} micro_step={}/{} window_id={} train_loss={:.8} forward_ms={} backward_ms={} host_materialization_ms={} retained_binding_f32={} gradient_f32={}",
                global_step,
                max_steps,
                micro_step + 1,
                geometry.grad_accum_steps,
                gradient_batch.window_id,
                gradient_batch.loss,
                step_profile.forward_loss_cuda_ms,
                step_profile.backward_cuda_ms,
                step_profile.host_materialization_ms(),
                step_profile.retained_binding_f32_count,
                step_profile.gradient_f32_count,
            ));
        }
        if let Some(writer) = live_visualization_writer.as_mut() {
            writer.record_phase(
                "training",
                Some(String::from("optimizer_step")),
                format!(
                    "Micro-step {} of {} completed for optimizer step {}.",
                    micro_step + 1,
                    geometry.grad_accum_steps,
                    global_step
                ),
                vec![
                    String::from("forward"),
                    String::from("backward"),
                    String::from("optimizer"),
                ],
                Some(global_step),
                Some((micro_step + 1) as u32),
                false,
            )?;
        }
    }

    let clip_observation = clip_gradients(accumulated_gradients.as_mut_slice(), grad_clip_norm);
    let optimizer_started = Instant::now();
    apply_gradients_to_state(
        trainer_state,
        accumulated_gradients.as_slice(),
        learning_rate_multiplier,
        muon_momentum,
        global_step,
    )?;
    *current_model = materialize_current_model(baseline_model, trainer_state)?;
    step_profile.optimizer_step_ms = duration_ms(optimizer_started);
    let observed_wallclock_ms = duration_ms(step_started);
    let step_metrics = ParameterGolfSingleH100TrainingStepMetrics {
        global_step,
        train_window_ids: window_ids,
        mean_microbatch_loss: microbatch_loss_sum / geometry.grad_accum_steps as f32,
        learning_rate_multiplier,
        muon_momentum,
        observed_wallclock_ms,
        phase_timings: step_profile,
        effective_learning_rate,
        gradient_norm_after_clip: clip_observation.gradient_norm_after_clip,
        parameter_norm_after_step: crate::state_parameter_norm(trainer_state),
        update_norm: None,
        clip_applied: clip_observation.clip_applied,
        non_finite_gradient_count: clip_observation.non_finite_count,
        tokens_per_second: crate::throughput_tokens_per_second(
            geometry.local_train_batch_tokens(),
            observed_wallclock_ms,
        ),
        samples_per_second_milli: crate::throughput_samples_per_second_milli(
            geometry.local_train_batch_sequences(),
            observed_wallclock_ms,
        ),
    };
    if emit_micro_step_logs {
        emit_progress_line(format!(
            "train_step_complete step={} mean_microbatch_loss={:.8} lr_mult={:.8} muon_momentum={:.8} host_materialization_ms={} optimizer_step_ms={}",
            step_metrics.global_step,
            step_metrics.mean_microbatch_loss,
            step_metrics.learning_rate_multiplier,
            step_metrics.muon_momentum,
            step_metrics.phase_timings.host_materialization_ms(),
            step_metrics.phase_timings.optimizer_step_ms,
        ));
    }
    Ok(step_metrics)
}

pub(crate) fn evaluate_validation_on_cuda(
    cuda_backend: &mut CudaBackend,
    device: &psionic_core::Device,
    descriptor: &psionic_models::ParameterGolfModelDescriptor,
    model: &ParameterGolfReferenceModel,
    validation_tokens: &[u16],
    byte_luts: &ParameterGolfSentencePieceByteLuts,
    sequence_length: usize,
    batch_sequences: usize,
    eval_mode: &ParameterGolfValidationEvalMode,
    graph_cache: &mut BTreeMap<usize, ParameterGolfBaselineEvalGraph>,
    stage_label: &str,
    live_visualization_writer: Option<&mut crate::ParameterGolfSingleH100LiveVisualizationWriter>,
) -> Result<ParameterGolfSingleH100ValidationSummary, ParameterGolfSingleH100TrainingError> {
    eval_mode.validate(sequence_length)?;
    if validation_tokens.len() <= sequence_length {
        return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
            message: String::from(
                "validation split is too short for the requested sequence length",
            ),
        });
    }
    let total_units = validation_unit_count(validation_tokens, sequence_length, eval_mode);
    let mut total_loss_sum = 0.0_f64;
    let mut total_token_count = 0_u64;
    let mut total_byte_count = 0_u64;
    let validation_batch_sequences = batch_sequences.max(1);
    let byte_accounting_started = Instant::now();
    let batch_plans = build_validation_batch_plans(
        validation_tokens,
        byte_luts,
        sequence_length,
        validation_batch_sequences,
        eval_mode,
    )?;
    let total_byte_accounting_us = duration_us(byte_accounting_started);
    let total_batches = batch_plans.len();
    let validation_started = Instant::now();
    let mut session_cache = BTreeMap::new();
    let mut live_visualization_writer = live_visualization_writer;
    let mut total_input_token_write_us = 0_u64;
    let mut total_target_token_write_us = 0_u64;
    let mut resident_parameter_upload_us = 0_u64;
    let mut persistent_parameter_buffer_count = 0_usize;
    let mut persistent_parameter_value_count = 0_u64;
    let mut processed_units = 0_usize;

    for (batch_index, batch_plan) in batch_plans.iter().enumerate() {
        emit_progress_line(format!(
            "validation_batch_start stage={} eval_mode={} batch={}/{} batch_sequences={} evaluated_tokens={} elapsed_ms={}",
            stage_label,
            eval_mode.as_str(),
            batch_index + 1,
            total_batches,
            batch_plan.batch_sequences,
            total_token_count,
            duration_ms(validation_started),
        ));
        let session = validation_session_for_batch(
            &mut session_cache,
            cuda_backend,
            graph_cache,
            device.clone(),
            descriptor,
            model,
            batch_plan.batch_sequences,
            sequence_length,
        )?;
        let (batch_token_losses, batch_runtime) = session.execute_batch(
            cuda_backend,
            batch_plan,
        )?;
        total_loss_sum += scored_token_loss_sum(
            batch_token_losses.as_slice(),
            batch_plan.sequence_plans.as_slice(),
            sequence_length,
        );
        total_token_count = total_token_count.saturating_add(batch_plan.token_count);
        total_byte_count = total_byte_count.saturating_add(batch_plan.byte_count);
        total_input_token_write_us =
            total_input_token_write_us.saturating_add(batch_runtime.input_token_write_us);
        total_target_token_write_us =
            total_target_token_write_us.saturating_add(batch_runtime.target_token_write_us);
        processed_units = processed_units.saturating_add(batch_plan.evaluation_units);
        if batch_index == 0 || (batch_index + 1) % 32 == 0 || batch_index + 1 == total_batches {
            emit_progress_line(format!(
                "validation_progress stage={} eval_mode={} batch={}/{} units={} tokens={} elapsed_ms={}",
                stage_label,
                eval_mode.as_str(),
                batch_index + 1,
                total_batches,
                processed_units,
                total_token_count,
                duration_ms(validation_started),
            ));
        }
        if let Some(writer) = live_visualization_writer.as_mut() {
            writer.record_phase(
                "training",
                Some(String::from("validation")),
                format!(
                    "Validation stage `{stage_label}` processed batch {} of {}.",
                    batch_index + 1,
                    total_batches
                ),
                vec![String::from("validation"), String::from("gpu_sampling")],
                None,
                Some(0),
                false,
            )?;
        }
    }

    for session in session_cache.values() {
        resident_parameter_upload_us =
            resident_parameter_upload_us.saturating_add(session.resident_parameter_upload_us);
        persistent_parameter_buffer_count = persistent_parameter_buffer_count
            .saturating_add(session.persistent_parameter_buffer_count);
        persistent_parameter_value_count = persistent_parameter_value_count
            .saturating_add(session.persistent_parameter_value_count);
    }

    let mean_loss = total_loss_sum / total_token_count.max(1) as f64;
    let bits_per_byte = (mean_loss / std::f64::consts::LN_2)
        * (total_token_count as f64 / total_byte_count.max(1) as f64);
    let runtime_receipt = ParameterGolfSingleH100ValidationRuntimeReceipt {
        path: String::from("device_resident_cuda_eval_graph_v1"),
        graph_surface: String::from("parameter_golf_baseline_eval_graph_v2"),
        eval_mode: eval_mode.clone(),
        session_count: session_cache.len(),
        total_batches,
        total_units,
        persistent_parameter_buffer_count,
        persistent_parameter_value_count,
        resident_parameter_upload_us,
        per_batch_stable_parameter_buffer_allocations: 0,
        reusable_input_token_buffer: true,
        reusable_target_token_buffer: true,
        total_input_token_write_us,
        total_target_token_write_us,
        byte_accounting_mode: String::from("precomputed_batch_target_bytes"),
        total_byte_accounting_us,
    };
    emit_progress_line(format!(
        "validation_runtime_receipt stage={} eval_mode={} path={} graph_surface={} sessions={} total_units={} stable_parameter_buffers={} stable_parameter_values={} resident_parameter_upload_us={} input_token_write_us={} target_token_write_us={} byte_accounting_us={}",
        stage_label,
        eval_mode.as_str(),
        runtime_receipt.path,
        runtime_receipt.graph_surface,
        runtime_receipt.session_count,
        runtime_receipt.total_units,
        runtime_receipt.persistent_parameter_buffer_count,
        runtime_receipt.persistent_parameter_value_count,
        runtime_receipt.resident_parameter_upload_us,
        runtime_receipt.total_input_token_write_us,
        runtime_receipt.total_target_token_write_us,
        runtime_receipt.total_byte_accounting_us,
    ));
    emit_progress_line(format!(
        "validation_complete stage={} eval_mode={} mean_loss={:.8} val_bpb={:.8} evaluated_tokens={} evaluated_bytes={} elapsed_ms={}",
        stage_label,
        eval_mode.as_str(),
        mean_loss,
        bits_per_byte,
        total_token_count,
        total_byte_count,
        duration_ms(validation_started),
    ));
    Ok(ParameterGolfSingleH100ValidationSummary {
        eval_mode: eval_mode.clone(),
        evaluated_sequence_count: total_units,
        evaluated_token_count: total_token_count,
        evaluated_byte_count: total_byte_count,
        mean_loss,
        bits_per_byte,
        runtime_receipt: Some(runtime_receipt),
    })
}

fn evaluate_validation_on_cuda_legacy(
    cuda_backend: &mut CudaBackend,
    device: &psionic_core::Device,
    descriptor: &psionic_models::ParameterGolfModelDescriptor,
    model: &ParameterGolfReferenceModel,
    validation_tokens: &[u16],
    byte_luts: &ParameterGolfSentencePieceByteLuts,
    sequence_length: usize,
    batch_sequences: usize,
    graph_cache: &mut BTreeMap<usize, ParameterGolfBaselineTrainingGraph>,
    stage_label: &str,
    live_visualization_writer: Option<&mut crate::ParameterGolfSingleH100LiveVisualizationWriter>,
) -> Result<ParameterGolfSingleH100ValidationSummary, ParameterGolfSingleH100TrainingError> {
    if validation_tokens.len() <= sequence_length {
        return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
            message: String::from(
                "validation split is too short for the requested sequence length",
            ),
        });
    }
    let total_sequences = (validation_tokens.len() - 1) / sequence_length;
    let mut total_loss_sum = 0.0_f64;
    let mut total_token_count = 0_u64;
    let mut total_byte_count = 0_u64;
    let validation_batch_sequences = batch_sequences.max(1);
    let total_batches = total_sequences.div_ceil(validation_batch_sequences);
    let validation_started = Instant::now();
    let mut live_visualization_writer = live_visualization_writer;

    for (batch_index, batch_start) in (0..total_sequences)
        .step_by(validation_batch_sequences)
        .enumerate()
    {
        let batch_end = (batch_start + validation_batch_sequences).min(total_sequences);
        let raw_start = batch_start * sequence_length;
        let raw_end = batch_end * sequence_length + 1;
        emit_progress_line(format!(
            "validation_batch_start stage={} batch={}/{} batch_sequences={} evaluated_tokens={} elapsed_ms={}",
            stage_label,
            batch_index + 1,
            total_batches,
            batch_end.saturating_sub(batch_start),
            total_token_count,
            duration_ms(validation_started),
        ));
        let local = &validation_tokens[raw_start..raw_end];
        let input_ids = local[..local.len() - 1]
            .chunks(sequence_length)
            .map(|row| {
                row.iter()
                    .map(|token| u32::from(*token))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let target_ids = local[1..]
            .chunks(sequence_length)
            .map(|row| {
                row.iter()
                    .map(|token| u32::from(*token))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let graph = training_graph_for_batch(
            graph_cache,
            device.clone(),
            descriptor,
            input_ids.len(),
            sequence_length,
        )?;
        let inputs = bind_parameter_golf_baseline_training_graph_inputs(
            graph,
            model,
            input_ids.as_slice(),
            target_ids.as_slice(),
        )?;
        let outputs = execute_cuda_graph_outputs(cuda_backend, graph.graph.graph(), &inputs)?;
        let batch_loss = scalar_float_graph_output(&outputs, graph.loss_tensor_id)?;
        let batch_token_count = target_ids.iter().map(Vec::len).sum::<usize>() as u64;
        total_loss_sum += f64::from(batch_loss) * batch_token_count as f64;
        total_token_count = total_token_count.saturating_add(batch_token_count);
        total_byte_count = total_byte_count.saturating_add(
            byte_luts.count_target_bytes(
                input_ids
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect::<Vec<_>>()
                    .as_slice(),
                target_ids
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?,
        );
        if batch_index == 0 || (batch_index + 1) % 32 == 0 || batch_index + 1 == total_batches {
            emit_progress_line(format!(
                "validation_progress stage={} batch={}/{} sequences={} tokens={} elapsed_ms={}",
                stage_label,
                batch_index + 1,
                total_batches,
                batch_end,
                total_token_count,
                duration_ms(validation_started),
            ));
        }
        if let Some(writer) = live_visualization_writer.as_mut() {
            writer.record_phase(
                "training",
                Some(String::from("validation")),
                format!(
                    "Validation stage `{stage_label}` processed batch {} of {}.",
                    batch_index + 1,
                    total_batches
                ),
                vec![String::from("validation"), String::from("gpu_sampling")],
                None,
                Some(0),
                false,
            )?;
        }
    }

    let mean_loss = total_loss_sum / total_token_count.max(1) as f64;
    let bits_per_byte = (mean_loss / std::f64::consts::LN_2)
        * (total_token_count as f64 / total_byte_count.max(1) as f64);
    emit_progress_line(format!(
        "validation_complete stage={} eval_mode={} mean_loss={:.8} val_bpb={:.8} evaluated_tokens={} evaluated_bytes={} elapsed_ms={}",
        stage_label,
        ParameterGolfValidationEvalMode::NonOverlapping.as_str(),
        mean_loss,
        bits_per_byte,
        total_token_count,
        total_byte_count,
        duration_ms(validation_started),
    ));
    Ok(ParameterGolfSingleH100ValidationSummary {
        eval_mode: ParameterGolfValidationEvalMode::NonOverlapping,
        evaluated_sequence_count: total_sequences,
        evaluated_token_count: total_token_count,
        evaluated_byte_count: total_byte_count,
        mean_loss,
        bits_per_byte,
        runtime_receipt: None,
    })
}

fn build_validation_batch_plans(
    validation_tokens: &[u16],
    byte_luts: &ParameterGolfSentencePieceByteLuts,
    sequence_length: usize,
    batch_sequences: usize,
    eval_mode: &ParameterGolfValidationEvalMode,
) -> Result<Vec<ParameterGolfValidationBatchPlan>, ParameterGolfSingleH100TrainingError> {
    match eval_mode {
        ParameterGolfValidationEvalMode::NonOverlapping => build_non_overlapping_validation_batch_plans(
            validation_tokens,
            byte_luts,
            sequence_length,
            batch_sequences,
        ),
        ParameterGolfValidationEvalMode::SlidingWindow { stride } => {
            build_sliding_window_validation_batch_plans(
                validation_tokens,
                byte_luts,
                sequence_length,
                batch_sequences,
                *stride,
            )
        }
    }
}

fn build_non_overlapping_validation_batch_plans(
    validation_tokens: &[u16],
    byte_luts: &ParameterGolfSentencePieceByteLuts,
    sequence_length: usize,
    batch_sequences: usize,
) -> Result<Vec<ParameterGolfValidationBatchPlan>, ParameterGolfSingleH100TrainingError> {
    let total_sequences = (validation_tokens.len() - 1) / sequence_length;
    let mut plans = Vec::new();
    for batch_start in (0..total_sequences).step_by(batch_sequences.max(1)) {
        let batch_end = (batch_start + batch_sequences.max(1)).min(total_sequences);
        let batch_sequence_count = batch_end - batch_start;
        let mut input_tokens = vec![0_i32; batch_sequence_count * sequence_length];
        let mut target_tokens = vec![0_i32; batch_sequence_count * sequence_length];
        let mut previous_tokens = Vec::with_capacity(batch_sequence_count * sequence_length);
        let mut scored_targets = Vec::with_capacity(batch_sequence_count * sequence_length);
        let mut sequence_plans = Vec::with_capacity(batch_sequence_count);
        for (local_index, sequence_index) in (batch_start..batch_end).enumerate() {
            let raw_start = sequence_index * sequence_length;
            let local = &validation_tokens[raw_start..raw_start + sequence_length + 1];
            let row_offset = local_index * sequence_length;
            for position in 0..sequence_length {
                let previous = local[position];
                let target = local[position + 1];
                input_tokens[row_offset + position] = i32::from(previous);
                target_tokens[row_offset + position] = i32::from(target);
                previous_tokens.push(u32::from(previous));
                scored_targets.push(u32::from(target));
            }
            sequence_plans.push(ParameterGolfValidationSequencePlan {
                valid_length: sequence_length,
                score_start: 0,
            });
        }
        plans.push(ParameterGolfValidationBatchPlan {
            batch_sequences: batch_sequence_count,
            evaluation_units: batch_sequence_count,
            input_tokens,
            target_tokens,
            sequence_plans,
            token_count: (batch_sequence_count * sequence_length) as u64,
            byte_count: byte_luts
                .count_target_bytes(previous_tokens.as_slice(), scored_targets.as_slice())?,
        });
    }
    Ok(plans)
}

fn build_sliding_window_validation_batch_plans(
    validation_tokens: &[u16],
    byte_luts: &ParameterGolfSentencePieceByteLuts,
    sequence_length: usize,
    batch_sequences: usize,
    stride: usize,
) -> Result<Vec<ParameterGolfValidationBatchPlan>, ParameterGolfSingleH100TrainingError> {
    let total_tokens = validation_tokens.len().saturating_sub(1);
    let window_starts = (0..total_tokens)
        .step_by(stride.max(1))
        .filter(|window_start| {
            total_tokens
                .saturating_sub(*window_start)
                .min(sequence_length)
                .saturating_sub(0)
                >= 1
        })
        .collect::<Vec<_>>();
    let mut plans = Vec::new();
    for batch_start in (0..window_starts.len()).step_by(batch_sequences.max(1)) {
        let batch_end = (batch_start + batch_sequences.max(1)).min(window_starts.len());
        let batch_sequence_count = batch_end - batch_start;
        let mut input_tokens = vec![0_i32; batch_sequence_count * sequence_length];
        let mut target_tokens = vec![0_i32; batch_sequence_count * sequence_length];
        let mut previous_tokens = Vec::new();
        let mut scored_targets = Vec::new();
        let mut sequence_plans = Vec::with_capacity(batch_sequence_count);
        let mut token_count = 0_u64;
        for (local_index, window_start) in window_starts[batch_start..batch_end].iter().enumerate() {
            let window_end = (*window_start + sequence_length).min(total_tokens);
            let valid_length = window_end.saturating_sub(*window_start);
            let chunk = &validation_tokens[*window_start..window_end + 1];
            let row_offset = local_index * sequence_length;
            for position in 0..valid_length {
                input_tokens[row_offset + position] = i32::from(chunk[position]);
                target_tokens[row_offset + position] = i32::from(chunk[position + 1]);
            }
            let score_start = if *window_start == 0 {
                0
            } else {
                valid_length.saturating_sub(stride)
            };
            for position in score_start..valid_length {
                previous_tokens.push(u32::from(chunk[position]));
                scored_targets.push(u32::from(chunk[position + 1]));
            }
            token_count = token_count.saturating_add(valid_length.saturating_sub(score_start) as u64);
            sequence_plans.push(ParameterGolfValidationSequencePlan {
                valid_length,
                score_start,
            });
        }
        plans.push(ParameterGolfValidationBatchPlan {
            batch_sequences: batch_sequence_count,
            evaluation_units: batch_sequence_count,
            input_tokens,
            target_tokens,
            sequence_plans,
            token_count,
            byte_count: byte_luts
                .count_target_bytes(previous_tokens.as_slice(), scored_targets.as_slice())?,
        });
    }
    Ok(plans)
}

fn validation_unit_count(
    validation_tokens: &[u16],
    sequence_length: usize,
    eval_mode: &ParameterGolfValidationEvalMode,
) -> usize {
    match eval_mode {
        ParameterGolfValidationEvalMode::NonOverlapping => {
            (validation_tokens.len().saturating_sub(1)) / sequence_length
        }
        ParameterGolfValidationEvalMode::SlidingWindow { stride } => (0
            ..validation_tokens.len().saturating_sub(1))
            .step_by((*stride).max(1))
            .filter(|window_start| {
                validation_tokens
                    .len()
                    .saturating_sub(1)
                    .saturating_sub(*window_start)
                    .min(sequence_length)
                    >= 1
            })
            .count(),
    }
}

fn scored_token_loss_sum(
    token_losses: &[f32],
    sequence_plans: &[ParameterGolfValidationSequencePlan],
    sequence_length: usize,
) -> f64 {
    sequence_plans
        .iter()
        .enumerate()
        .map(|(sequence_index, plan)| {
            let row_offset = sequence_index * sequence_length;
            token_losses[row_offset + plan.score_start..row_offset + plan.valid_length]
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
        })
        .sum()
}

fn validation_session_for_batch<'a>(
    cache: &'a mut BTreeMap<usize, ParameterGolfCudaValidationSession>,
    cuda_backend: &mut CudaBackend,
    graph_cache: &mut BTreeMap<usize, ParameterGolfBaselineEvalGraph>,
    device: psionic_core::Device,
    descriptor: &psionic_models::ParameterGolfModelDescriptor,
    model: &ParameterGolfReferenceModel,
    batch_sequences: usize,
    sequence_length: usize,
) -> Result<&'a mut ParameterGolfCudaValidationSession, ParameterGolfSingleH100TrainingError> {
    if !cache.contains_key(&batch_sequences) {
        let graph = eval_graph_for_batch(
            graph_cache,
            device,
            descriptor,
            batch_sequences,
            sequence_length,
        )?
        .clone();
        let session = ParameterGolfCudaValidationSession::new(
            cuda_backend,
            graph,
            model,
            batch_sequences,
            sequence_length,
        )?;
        cache.insert(batch_sequences, session);
    }
    cache.get_mut(&batch_sequences).ok_or_else(|| {
        ParameterGolfSingleH100TrainingError::Serialization {
            message: format!(
                "missing cached validation session for batch_sequences={batch_sequences}"
            ),
        }
    })
}

fn training_graph_for_batch<'a>(
    cache: &'a mut BTreeMap<usize, ParameterGolfBaselineTrainingGraph>,
    device: psionic_core::Device,
    descriptor: &psionic_models::ParameterGolfModelDescriptor,
    batch_size: usize,
    sequence_length: usize,
) -> Result<&'a ParameterGolfBaselineTrainingGraph, ParameterGolfSingleH100TrainingError> {
    if !cache.contains_key(&batch_size) {
        let graph = build_parameter_golf_baseline_training_graph(
            device,
            descriptor,
            batch_size,
            sequence_length,
        )?;
        cache.insert(batch_size, graph);
    }
    cache
        .get(&batch_size)
        .ok_or_else(|| ParameterGolfSingleH100TrainingError::Serialization {
            message: format!("missing cached training graph for batch_size={batch_size}"),
        })
}

fn eval_graph_for_batch<'a>(
    cache: &'a mut BTreeMap<usize, ParameterGolfBaselineEvalGraph>,
    device: psionic_core::Device,
    descriptor: &psionic_models::ParameterGolfModelDescriptor,
    batch_size: usize,
    sequence_length: usize,
) -> Result<&'a ParameterGolfBaselineEvalGraph, ParameterGolfSingleH100TrainingError> {
    if !cache.contains_key(&batch_size) {
        let graph = build_parameter_golf_baseline_eval_graph(
            device,
            descriptor,
            batch_size,
            sequence_length,
        )?;
        cache.insert(batch_size, graph);
    }
    cache
        .get(&batch_size)
        .ok_or_else(|| ParameterGolfSingleH100TrainingError::Serialization {
            message: format!("missing cached eval graph for batch_size={batch_size}"),
        })
}

impl ParameterGolfCudaValidationSession {
    fn new(
        cuda_backend: &mut CudaBackend,
        graph: ParameterGolfBaselineEvalGraph,
        model: &ParameterGolfReferenceModel,
        batch_sequences: usize,
        sequence_length: usize,
    ) -> Result<Self, ParameterGolfSingleH100TrainingError> {
        let config = &model.descriptor().config;
        let parameter_vectors = model
            .weights()
            .parameter_vectors(config)
            .into_iter()
            .map(|parameter| (parameter.parameter_id.clone(), parameter))
            .collect::<BTreeMap<_, _>>();
        let token_element_count = batch_sequences.saturating_mul(sequence_length);
        let token_shape = Shape::new(vec![batch_sequences, sequence_length]);
        let parameter_upload_started = Instant::now();
        let mut static_inputs = BTreeMap::new();
        let mut persistent_parameter_buffer_count = 0_usize;
        let mut persistent_parameter_value_count = 0_u64;
        for binding in &graph.parameter_bindings {
            let parameter = parameter_vectors
                .get(&binding.parameter_id)
                .ok_or_else(
                    || crate::ParameterGolfBaselineGraphError::MissingWeightVector {
                        parameter_id: binding.parameter_id.clone(),
                    },
                )?;
            let buffer = match binding.graph_input_dtype {
                DType::F32 => {
                    cuda_backend.input_buffer(binding.shape.clone(), parameter.values.clone())?
                }
                DType::BF16 => cuda_backend
                    .input_bf16_buffer(binding.shape.clone(), parameter.values.clone())?,
                actual => {
                    return Err(ParameterGolfSingleH100TrainingError::Serialization {
                        message: format!(
                            "validation session does not support graph input dtype {actual:?} for `{}`",
                            binding.parameter_id
                        ),
                    });
                }
            };
            persistent_parameter_buffer_count = persistent_parameter_buffer_count.saturating_add(1);
            persistent_parameter_value_count = persistent_parameter_value_count
                .saturating_add(buffer.spec().shape().element_count() as u64);
            static_inputs.insert(binding.graph_input_tensor_id, buffer);
        }
        let resident_parameter_upload_us = duration_us(parameter_upload_started);
        let input_token_buffer =
            cuda_backend.input_i32_buffer(token_shape.clone(), vec![0_i32; token_element_count])?;
        let target_token_buffer =
            cuda_backend.input_i32_buffer(token_shape, vec![0_i32; token_element_count])?;
        Ok(Self {
            graph,
            static_inputs,
            input_token_buffer,
            target_token_buffer,
            input_token_staging: vec![0_i32; token_element_count],
            target_token_staging: vec![0_i32; token_element_count],
            resident_parameter_upload_us,
            persistent_parameter_buffer_count,
            persistent_parameter_value_count,
        })
    }

    fn execute_batch(
        &mut self,
        cuda_backend: &mut CudaBackend,
        batch_plan: &ParameterGolfValidationBatchPlan,
    ) -> Result<(Vec<f32>, ParameterGolfValidationBatchRuntime), ParameterGolfSingleH100TrainingError>
    {
        if batch_plan.batch_sequences.saturating_mul(self.input_token_staging.len().max(1))
            < batch_plan.input_tokens.len()
        {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: format!(
                    "validation session staging capacity {} could not fit {} input tokens",
                    self.input_token_staging.len(),
                    batch_plan.input_tokens.len()
                ),
            });
        }
        if batch_plan.input_tokens.len() != self.input_token_staging.len()
            || batch_plan.target_tokens.len() != self.target_token_staging.len()
        {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: format!(
                    "validation session expected {} tokens per batch but observed {} input and {} target tokens",
                    self.input_token_staging.len(),
                    batch_plan.input_tokens.len(),
                    batch_plan.target_tokens.len(),
                ),
            });
        }

        let input_write_started = Instant::now();
        self.input_token_staging
            .copy_from_slice(batch_plan.input_tokens.as_slice());
        self.input_token_buffer
            .write_i32(self.input_token_staging.as_slice())?;
        let input_token_write_us = duration_us(input_write_started);

        let target_write_started = Instant::now();
        self.target_token_staging
            .copy_from_slice(batch_plan.target_tokens.as_slice());
        self.target_token_buffer
            .write_i32(self.target_token_staging.as_slice())?;
        let target_token_write_us = duration_us(target_write_started);

        let mut inputs = self.static_inputs.clone();
        inputs.insert(
            self.graph.input_token_ids_tensor_id,
            self.input_token_buffer.clone(),
        );
        inputs.insert(
            self.graph.target_ids_tensor_id,
            self.target_token_buffer.clone(),
        );
        let outputs =
            execute_cuda_graph_outputs_from_buffers(cuda_backend, &self.graph.graph, &inputs)?;
        let batch_token_losses = dense_f32_graph_output(&outputs, self.graph.token_losses_tensor_id)?;
        Ok((
            batch_token_losses,
            ParameterGolfValidationBatchRuntime {
                input_token_write_us,
                target_token_write_us,
            },
        ))
    }
}

fn retained_forward_graph(
    graph: &ParameterGolfBaselineTrainingGraph,
    backward_plan: &psionic_ir::AutodiffBackwardPlan,
) -> psionic_ir::Graph {
    let mut outputs = vec![graph.loss_tensor_id];
    outputs.extend(
        backward_plan
            .primal_bindings
            .iter()
            .map(|binding| binding.primal_tensor),
    );
    graph.graph.graph().with_outputs(unique_tensor_ids(outputs))
}

fn execute_backward_plan(
    cuda_backend: &mut CudaBackend,
    backward_plan: &psionic_ir::AutodiffBackwardPlan,
    forward_outputs: &[(TensorId, TensorData)],
) -> Result<Vec<(TensorId, TensorData)>, ParameterGolfSingleH100TrainingError> {
    let forward_map = forward_outputs
        .iter()
        .map(|(tensor_id, values)| (*tensor_id, values.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut inputs = Vec::new();
    for binding in &backward_plan.primal_bindings {
        let values = forward_map.get(&binding.primal_tensor).ok_or(
            ParameterGolfSingleH100TrainingError::MissingGraphOutput {
                tensor_id: binding.primal_tensor,
            },
        )?;
        let input_dtype = backward_plan
            .gradient_graph
            .node(binding.gradient_graph_input)
            .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphTensor {
                tensor_id: binding.gradient_graph_input,
            })?
            .tensor()
            .spec()
            .dtype();
        inputs.push((
            binding.gradient_graph_input,
            tensor_data_for_dtype(input_dtype, values.clone(), binding.gradient_graph_input)?,
        ));
    }
    let seed_dtype = backward_plan
        .gradient_graph
        .node(backward_plan.seed_input)
        .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphTensor {
            tensor_id: backward_plan.seed_input,
        })?
        .tensor()
        .spec()
        .dtype();
    inputs.push((
        backward_plan.seed_input,
        floating_tensor_data_for_dtype(seed_dtype, vec![1.0_f32])?,
    ));
    execute_cuda_graph_outputs(
        cuda_backend,
        &backward_plan.gradient_graph,
        &inputs.into_iter().collect(),
    )
}

fn floating_tensor_data_for_dtype(
    dtype: DType,
    values: Vec<f32>,
) -> Result<TensorData, ParameterGolfSingleH100TrainingError> {
    match dtype {
        DType::F32 => Ok(TensorData::F32(values)),
        DType::BF16 => Ok(TensorData::BF16(values)),
        actual => Err(ParameterGolfSingleH100TrainingError::Serialization {
            message: format!("unsupported floating tensor dtype {actual:?}"),
        }),
    }
}

fn tensor_data_for_dtype(
    dtype: DType,
    data: TensorData,
    tensor_id: TensorId,
) -> Result<TensorData, ParameterGolfSingleH100TrainingError> {
    match (dtype, data) {
        (DType::F32, TensorData::F32(values)) | (DType::F32, TensorData::BF16(values)) => {
            Ok(TensorData::F32(values))
        }
        (DType::BF16, TensorData::F32(values)) | (DType::BF16, TensorData::BF16(values)) => {
            Ok(TensorData::BF16(values))
        }
        (DType::I32, TensorData::I32(values)) => Ok(TensorData::I32(values)),
        (actual, observed) => Err(ParameterGolfSingleH100TrainingError::Serialization {
            message: format!(
                "tensor {tensor_id} expected {actual:?} data but observed {:?}",
                match observed {
                    TensorData::F32(_) => DType::F32,
                    TensorData::BF16(_) => DType::BF16,
                    TensorData::I32(_) => DType::I32,
                    TensorData::QuantizedBlocks(_) => DType::I8,
                }
            ),
        }),
    }
}

fn backward_result_from_outputs(
    backward_plan: &psionic_ir::AutodiffBackwardPlan,
    outputs: &[(TensorId, TensorData)],
) -> Result<AutodiffBackwardResult, ParameterGolfSingleH100TrainingError> {
    let gradient_targets = backward_plan
        .gradient_targets
        .iter()
        .map(|target| (target.gradient_tensor, target.primal_tensor))
        .collect::<BTreeMap<_, _>>();
    let gradients = outputs
        .iter()
        .filter_map(|(tensor_id, values)| {
            gradient_targets
                .get(tensor_id)
                .copied()
                .map(|primal_tensor| (*tensor_id, primal_tensor, values.clone()))
        })
        .map(
            |(gradient_tensor_id, primal_tensor, values)| -> Result<
                (TensorId, TensorData),
                ParameterGolfSingleH100TrainingError,
            > {
            let dtype = backward_plan
                .gradient_graph
                .node(gradient_tensor_id)
                .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphTensor {
                    tensor_id: gradient_tensor_id,
                })?
                .tensor()
                .spec()
                .dtype();
            Ok((
                primal_tensor,
                tensor_data_for_dtype(dtype, values, gradient_tensor_id)?,
            ))
        },
        )
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok(AutodiffBackwardResult {
        forward_values: BTreeMap::new(),
        plan: backward_plan.clone(),
        gradients,
    })
}

fn execute_cuda_graph_outputs(
    cuda_backend: &mut CudaBackend,
    graph: &psionic_ir::Graph,
    inputs: &BTreeMap<TensorId, TensorData>,
) -> Result<Vec<(TensorId, TensorData)>, ParameterGolfSingleH100TrainingError> {
    let mut buffers = BTreeMap::new();
    for (tensor_id, data) in inputs {
        let spec = graph
            .node(*tensor_id)
            .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphTensor {
                tensor_id: *tensor_id,
            })?
            .tensor()
            .clone();
        let buffer =
            match spec.spec().dtype() {
                psionic_core::DType::F32 => match data {
                    TensorData::F32(values) | TensorData::BF16(values) => {
                        cuda_backend.input_buffer(spec.spec().shape().clone(), values.clone())?
                    }
                    TensorData::I32(_) | TensorData::QuantizedBlocks(_) => {
                        return Err(ParameterGolfSingleH100TrainingError::Serialization {
                            message: format!(
                                "tensor {tensor_id} expected dense F32 host values for graph input"
                            ),
                        });
                    }
                },
                psionic_core::DType::BF16 => match data {
                    TensorData::F32(values) | TensorData::BF16(values) => cuda_backend
                        .input_bf16_buffer(spec.spec().shape().clone(), values.clone())?,
                    TensorData::I32(_) | TensorData::QuantizedBlocks(_) => {
                        return Err(ParameterGolfSingleH100TrainingError::Serialization {
                            message: format!(
                                "tensor {tensor_id} expected dense BF16 host values for graph input"
                            ),
                        });
                    }
                },
                psionic_core::DType::I32 => match data {
                    TensorData::I32(values) => cuda_backend
                        .input_i32_buffer(spec.spec().shape().clone(), values.clone())?,
                    TensorData::F32(_) | TensorData::BF16(_) | TensorData::QuantizedBlocks(_) => {
                        return Err(ParameterGolfSingleH100TrainingError::Serialization {
                            message: format!(
                                "tensor {tensor_id} expected dense I32 host values for graph input"
                            ),
                        });
                    }
                },
                actual => {
                    return Err(ParameterGolfSingleH100TrainingError::Serialization {
                        message: format!(
                            "unsupported graph input dtype {actual:?} for tensor {tensor_id}"
                        ),
                    });
                }
            };
        buffers.insert(*tensor_id, buffer);
    }
    execute_cuda_graph_outputs_from_buffers(cuda_backend, graph, &buffers)
}

fn execute_cuda_graph_outputs_from_buffers(
    cuda_backend: &mut CudaBackend,
    graph: &psionic_ir::Graph,
    inputs: &BTreeMap<TensorId, CudaBuffer>,
) -> Result<Vec<(TensorId, TensorData)>, ParameterGolfSingleH100TrainingError> {
    let result = cuda_backend.compile_and_execute(graph, inputs)?;
    graph
        .outputs()
        .iter()
        .map(|tensor_id| {
            let output = result.outputs.get(tensor_id).ok_or(
                ParameterGolfSingleH100TrainingError::MissingGraphOutput {
                    tensor_id: *tensor_id,
                },
            )?;
            let values = match output.spec().dtype() {
                psionic_core::DType::F32 => TensorData::F32(output.read_f32()?),
                psionic_core::DType::BF16 => TensorData::BF16(output.read_bf16_to_f32()?),
                psionic_core::DType::I32 => TensorData::I32(output.read_i32()?),
                actual => {
                    return Err(ParameterGolfSingleH100TrainingError::Serialization {
                        message: format!(
                            "unsupported graph output dtype {actual:?} for tensor {tensor_id}"
                        ),
                    });
                }
            };
            Ok((*tensor_id, values))
        })
        .collect()
}

fn scalar_float_graph_output(
    outputs: &[(TensorId, TensorData)],
    tensor_id: TensorId,
) -> Result<f32, ParameterGolfSingleH100TrainingError> {
    outputs
        .iter()
        .find(|(current, _)| *current == tensor_id)
        .and_then(|(_, values)| values.as_f32_slice())
        .and_then(|values| values.first().copied())
        .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphOutput { tensor_id })
}

fn dense_f32_graph_output(
    outputs: &[(TensorId, TensorData)],
    tensor_id: TensorId,
) -> Result<Vec<f32>, ParameterGolfSingleH100TrainingError> {
    outputs
        .iter()
        .find(|(current, _)| *current == tensor_id)
        .and_then(|(_, values)| values.as_f32_slice())
        .map(|values| values.to_vec())
        .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphOutput { tensor_id })
}

fn split_token_count(bundle: &ParameterGolfDatasetBundle, split_name: &str) -> u64 {
    bundle
        .manifest
        .split(split_name)
        .map(|split| split.shards.iter().map(|shard| shard.token_count).sum())
        .unwrap_or(0)
}

fn count_output_elements(
    retained_graph: &psionic_ir::Graph,
    bindings: &[psionic_ir::AutodiffPrimalBinding],
) -> Result<u64, ParameterGolfSingleH100TrainingError> {
    bindings.iter().try_fold(0_u64, |sum, binding| {
        let node = retained_graph.node(binding.primal_tensor).ok_or(
            ParameterGolfSingleH100TrainingError::MissingGraphTensor {
                tensor_id: binding.primal_tensor,
            },
        )?;
        Ok(sum.saturating_add(node.tensor().spec().shape().element_count() as u64))
    })
}

fn count_gradient_elements(
    backward_plan: &psionic_ir::AutodiffBackwardPlan,
) -> Result<u64, ParameterGolfSingleH100TrainingError> {
    backward_plan
        .gradient_targets
        .iter()
        .try_fold(0_u64, |sum, target| {
            let node = backward_plan
                .gradient_graph
                .node(target.gradient_tensor)
                .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphTensor {
                    tensor_id: target.gradient_tensor,
                })?;
            Ok(sum.saturating_add(node.tensor().spec().shape().element_count() as u64))
        })
}

pub(crate) fn clip_gradients(
    gradients: &mut [(String, Vec<f32>)],
    max_norm: f32,
) -> crate::GradientClipObservation {
    let non_finite_count = crate::non_finite_value_count(gradients);
    if !(max_norm.is_finite() && max_norm > 0.0) {
        return crate::GradientClipObservation {
            gradient_norm_after_clip: crate::finite_l2_norm(gradients),
            clip_applied: false,
            non_finite_count,
        };
    }
    let norm = gradients
        .iter()
        .flat_map(|(_, values)| values.iter())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let clip_applied = norm > max_norm && norm > f32::EPSILON;
    if clip_applied {
        let scale = max_norm / norm;
        for (_, values) in &mut *gradients {
            for value in values {
                *value *= scale;
            }
        }
    }
    crate::GradientClipObservation {
        gradient_norm_after_clip: crate::finite_l2_norm(gradients),
        clip_applied,
        non_finite_count,
    }
}

#[cfg(test)]
mod tests {
    use psionic_core::{DType, Device, Shape};
    use psionic_ir::{AutodiffContext, AutodiffGraphBuilder};
    use psionic_models::ParameterGolfReferenceModel;

    use super::*;

    #[test]
    fn backward_result_from_outputs_rekeys_gradient_outputs_to_primal_tensors(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut builder =
            AutodiffGraphBuilder::with_context(Device::cpu(), AutodiffContext::training());
        let input = builder.input("x", Shape::new(vec![1, 1]), DType::F32, true);
        let squared = builder.mul(&input, &input)?;
        let graph = builder.finish(vec![squared.clone()]);
        let backward_plan = graph.backward_plan(squared.id())?;
        let gradient_tensor_id = backward_plan
            .gradient_for(input.id())
            .ok_or("missing input gradient target")?;

        let backward_result = backward_result_from_outputs(
            &backward_plan,
            &[(gradient_tensor_id, TensorData::F32(vec![42.0_f32]))],
        )?;

        assert_eq!(
            backward_result
                .gradient(input.id())
                .and_then(TensorData::as_f32_slice),
            Some(&[42.0_f32][..])
        );
        Ok(())
    }

    #[test]
    fn backward_result_from_outputs_preserves_bf16_gradient_dtype(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut builder =
            AutodiffGraphBuilder::with_context(Device::cpu(), AutodiffContext::training());
        let input = builder.input("x", Shape::new(vec![1, 1]), DType::BF16, true);
        let squared = builder.mul(&input, &input)?;
        let graph = builder.finish(vec![squared.clone()]);
        let backward_plan = graph.backward_plan(squared.id())?;
        let gradient_tensor_id = backward_plan
            .gradient_for(input.id())
            .ok_or("missing input gradient target")?;

        let backward_result = backward_result_from_outputs(
            &backward_plan,
            &[(gradient_tensor_id, TensorData::BF16(vec![42.0_f32]))],
        )?;

        assert!(matches!(
            backward_result.gradient(input.id()),
            Some(TensorData::BF16(values)) if values == &vec![42.0_f32]
        ));
        Ok(())
    }

    #[test]
    fn tensor_data_for_dtype_preserves_i32_outputs() -> Result<(), Box<dyn std::error::Error>> {
        let tensor_id = TensorId(7);
        assert_eq!(
            tensor_data_for_dtype(DType::I32, TensorData::I32(vec![1, 2, 3]), tensor_id)?,
            TensorData::I32(vec![1, 2, 3])
        );
        Ok(())
    }

    #[test]
    fn challenge_defaults_use_widened_train_gpt_control_loop_defaults() {
        let config = ParameterGolfSingleH100TrainingConfig::challenge_defaults(
            "/tmp/dataset",
            "/tmp/tokenizer.model",
        );

        assert_eq!(config.max_steps, config.hyperparameters.iterations);
        assert_eq!(config.warmup_steps, 20);
        assert_eq!(config.validation_loss_every, 1_000);
        assert_eq!(config.train_log_every, 200);
        assert_eq!(
            config.final_validation_mode,
            ParameterGolfSingleH100ValidationMode::Both
        );
        assert_eq!(config.hyperparameters.max_wallclock_seconds, Some(600.0));
    }

    #[test]
    fn bounded_proof_defaults_disable_widened_loop_features() {
        let config = ParameterGolfSingleH100TrainingConfig::bounded_proof_defaults(
            "/tmp/dataset",
            "/tmp/tokenizer.model",
            3,
        );

        assert_eq!(config.max_steps, 3);
        assert_eq!(config.warmup_steps, 0);
        assert_eq!(config.validation_loss_every, 0);
        assert_eq!(config.train_log_every, 1);
        assert_eq!(
            config.final_validation_mode,
            ParameterGolfSingleH100ValidationMode::RoundtripOnly
        );
        assert_eq!(config.hyperparameters.max_wallclock_seconds, None);
    }

    #[test]
    fn validation_mode_parser_accepts_all_supported_labels(
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            ParameterGolfSingleH100ValidationMode::parse("live_only")?,
            ParameterGolfSingleH100ValidationMode::LiveOnly
        );
        assert_eq!(
            ParameterGolfSingleH100ValidationMode::parse("roundtrip_only")?,
            ParameterGolfSingleH100ValidationMode::RoundtripOnly
        );
        assert_eq!(
            ParameterGolfSingleH100ValidationMode::parse("both")?,
            ParameterGolfSingleH100ValidationMode::Both
        );
        Ok(())
    }

    #[test]
    fn validation_eval_mode_parser_accepts_supported_labels(
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            ParameterGolfValidationEvalMode::parse("non_overlapping")?,
            ParameterGolfValidationEvalMode::NonOverlapping
        );
        assert_eq!(
            ParameterGolfValidationEvalMode::parse("sliding_window:64")?,
            ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 }
        );
        Ok(())
    }

    #[test]
    fn sliding_window_validation_batch_plans_score_only_suffix_tokens(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let byte_luts = ParameterGolfSentencePieceByteLuts {
            base_bytes_lut: vec![1; 8],
            has_leading_space_lut: vec![false; 8],
            is_boundary_token_lut: vec![false; 8],
        };
        let validation_tokens = vec![0_u16, 1, 2, 3, 4, 5];
        let plans = build_validation_batch_plans(
            validation_tokens.as_slice(),
            &byte_luts,
            4,
            2,
            &ParameterGolfValidationEvalMode::SlidingWindow { stride: 2 },
        )?;

        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].evaluation_units, 2);
        assert_eq!(plans[0].token_count, 6);
        assert_eq!(plans[0].byte_count, 6);
        assert_eq!(
            plans[0]
                .sequence_plans
                .iter()
                .map(|plan| (plan.valid_length, plan.score_start))
                .collect::<Vec<_>>(),
            vec![(4, 0), (3, 1)]
        );
        assert_eq!(plans[1].evaluation_units, 1);
        assert_eq!(plans[1].token_count, 1);
        assert_eq!(plans[1].byte_count, 1);
        assert_eq!(
            plans[1]
                .sequence_plans
                .iter()
                .map(|plan| (plan.valid_length, plan.score_start))
                .collect::<Vec<_>>(),
            vec![(1, 0)]
        );
        Ok(())
    }

    #[test]
    fn seed_parameter_states_matches_upstream_precision_split(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
        let hyperparameters = ParameterGolfTrainingHyperparameters::baseline_defaults();
        let optimizer_plan = parameter_golf_optimizer_plan(model.descriptor(), &hyperparameters)?;
        let trainer_state = seed_parameter_states(&model, &optimizer_plan)?;

        let token_embedding = trainer_state
            .parameter_states
            .get("tok_emb.weight")
            .ok_or("missing tok_emb.weight state")?;
        let token_embedding_source = model
            .weights()
            .parameter_vector(&model.descriptor().config, "tok_emb.weight")
            .ok_or("missing tok_emb.weight source vector")?;
        match token_embedding {
            ParameterGolfParameterState::AdamBf16Master {
                train_visible_values,
                master_weight_values,
                ..
            } => {
                for (source, rounded) in token_embedding_source
                    .values
                    .iter()
                    .zip(train_visible_values.iter())
                {
                    assert_eq!(*rounded, bf16::from_f32(*source).to_f32());
                }
                assert_eq!(
                    master_weight_values.as_slice(),
                    token_embedding_source.values
                );
            }
            _ => return Err("expected AdamBf16Master token embedding state".into()),
        }

        let matrix = trainer_state
            .parameter_states
            .get("blocks.0.attn.c_q.weight")
            .ok_or("missing matrix state")?;
        assert!(matches!(
            matrix,
            ParameterGolfParameterState::MuonBf16 { .. }
        ));

        let control = trainer_state
            .parameter_states
            .get("blocks.0.attn_scale")
            .ok_or("missing control state")?;
        assert!(matches!(
            control,
            ParameterGolfParameterState::AdamFp32 { .. }
        ));
        let control_source = model
            .weights()
            .parameter_vector(&model.descriptor().config, "blocks.0.attn_scale")
            .ok_or("missing blocks.0.attn_scale source vector")?;
        assert_eq!(control.values(), control_source.values.as_slice());

        Ok(())
    }

    #[test]
    fn flatten_parameter_golf_optimizer_group_values_matches_group_parameter_counts(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
        let hyperparameters = ParameterGolfTrainingHyperparameters::baseline_defaults();
        let optimizer_plan = parameter_golf_optimizer_plan(model.descriptor(), &hyperparameters)?;
        let trainer_state = seed_parameter_states(&model, &optimizer_plan)?;
        let values_by_parameter = trainer_state
            .parameter_states
            .iter()
            .map(|(parameter_id, state)| (parameter_id.clone(), state.values().to_vec()))
            .collect::<BTreeMap<_, _>>();

        let flattened =
            flatten_parameter_golf_optimizer_group_values(&optimizer_plan, &values_by_parameter)?;
        for group in &optimizer_plan.groups {
            let values = flattened
                .get(group.group_id.as_str())
                .ok_or("missing flattened optimizer group")?;
            assert_eq!(values.len(), group.parameter_count);
        }
        Ok(())
    }

    #[test]
    fn flatten_parameter_golf_optimizer_group_buffers_emits_cpu_f32_tensor_buffers(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
        let hyperparameters = ParameterGolfTrainingHyperparameters::baseline_defaults();
        let optimizer_plan = parameter_golf_optimizer_plan(model.descriptor(), &hyperparameters)?;
        let trainer_state = seed_parameter_states(&model, &optimizer_plan)?;
        let values_by_parameter = trainer_state
            .parameter_states
            .iter()
            .map(|(parameter_id, state)| (parameter_id.clone(), state.values().to_vec()))
            .collect::<BTreeMap<_, _>>();

        let buffers =
            flatten_parameter_golf_optimizer_group_buffers(&optimizer_plan, &values_by_parameter)?;
        for group in &optimizer_plan.groups {
            let buffer = buffers
                .get(group.group_id.as_str())
                .ok_or("missing flattened optimizer buffer")?;
            assert_eq!(buffer.spec.dtype(), DType::F32);
            assert_eq!(buffer.spec.shape().element_count(), group.parameter_count);
            assert!(matches!(
                &buffer.data,
                TensorData::F32(values) if values.len() == group.parameter_count
            ));
        }
        Ok(())
    }
}

fn unique_tensor_ids(ids: Vec<TensorId>) -> Vec<TensorId> {
    let mut seen = std::collections::BTreeSet::new();
    ids.into_iter().filter(|id| seen.insert(*id)).collect()
}

fn duration_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

fn duration_us(started: Instant) -> u64 {
    started.elapsed().as_micros() as u64
}

fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_millis() as u64
}

fn emit_progress_line(message: String) {
    let mut stdout = io::stdout().lock();
    let _ = writeln!(stdout, "{message}");
    let _ = stdout.flush();
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).expect("digest serialization should succeed"));
    hex::encode(hasher.finalize())
}

fn report_without_digest(
    report: &ParameterGolfSingleH100TrainingReport,
) -> ParameterGolfSingleH100TrainingReport {
    let mut canonical = report.clone();
    canonical.report_digest.clear();
    canonical
}

fn receipt_without_digest(
    receipt: &ParameterGolfSingleH100ValidationRuntimeComparisonReceipt,
) -> ParameterGolfSingleH100ValidationRuntimeComparisonReceipt {
    let mut canonical = receipt.clone();
    canonical.receipt_digest.clear();
    canonical
}

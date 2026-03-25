use std::{
    collections::BTreeMap,
    env, fs,
    fs::OpenOptions,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use psionic_backend_cuda::CudaBackend;
use psionic_data::{
    load_parameter_golf_validation_tokens_from_paths, parameter_golf_dataset_bundle_from_local_dir,
    parameter_golf_sentencepiece_byte_luts_from_tokenizer_path, DatasetIterationMode, DatasetKey,
    ParameterGolfTokenStreamContract, ParameterGolfTokenStreamCursor,
    ParameterGolfTokenStreamWindow, PARAMETER_GOLF_TRAIN_SPLIT_NAME,
};
use psionic_eval::ParameterGolfDistributedThroughputReceipt;
use psionic_models::ParameterGolfReferenceModel;
use safetensors::{serialize, tensor::TensorView, Dtype as SafeTensorsDType, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    apply_gradients_to_state, benchmark_parameter_golf_distributed_8xh100,
    build_parameter_golf_validation_window_starts, build_tokenizer_digest,
    build_validation_observation_plan, clip_gradients, evaluate_validation_on_cuda,
    evaluate_validation_window_starts_on_cuda, execute_parameter_golf_training_gradient_batch,
    export_parameter_golf_full_precision_model_bytes,
    export_parameter_golf_int8_zlib_model_artifact, inspect_local_distributed_8xh100_machine,
    materialize_current_model, parameter_golf_optimizer_plan,
    parameter_golf_runpod_8xh100_capability_profile, restore_parameter_golf_model_from_safetensors,
    seed_parameter_states, zero_gradients, ParameterGolfBatchGeometry,
    ParameterGolfDistributed8xH100BringupReport,
    ParameterGolfDistributed8xH100RuntimeBootstrapReceipt,
    ParameterGolfDistributedLiveVisualizationWriter, ParameterGolfDistributedStepObservation,
    ParameterGolfDistributedValidationShardObservation, ParameterGolfDistributedVisualizationError,
    ParameterGolfRunPod8xH100Measurements, ParameterGolfSingleH100PhaseTimings,
    ParameterGolfSingleH100TrainerState, ParameterGolfSingleH100TrainingError,
    ParameterGolfTrainingHyperparameters, ParameterGolfValidationEvalMode,
    PARAMETER_GOLF_SINGLE_H100_VARIANT,
};

const CHILD_ENV_VAR: &str = "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_CHILD";
const CHILD_RANK_ENV_VAR: &str = "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_RANK";
const CHILD_LOCAL_RANK_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_LOCAL_RANK";
const CHILD_WORLD_SIZE_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_WORLD_SIZE";
const CHILD_RECEIPT_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_RECEIPT_PATH";
const CHILD_LOG_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_LOG_PATH";
const CHILD_WINDOW_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_WINDOW_PATH";
const CHILD_GRADIENT_ARTIFACT_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_GRADIENT_ARTIFACT_PATH";
const CHILD_MODEL_ARTIFACT_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_MODEL_ARTIFACT_PATH";
const VALIDATION_CHILD_ENV_VAR: &str = "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_CHILD";
const VALIDATION_CHILD_RANK_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_RANK";
const VALIDATION_CHILD_LOCAL_RANK_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_LOCAL_RANK";
const VALIDATION_CHILD_WORLD_SIZE_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_WORLD_SIZE";
const VALIDATION_CHILD_RECEIPT_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_RECEIPT_PATH";
const VALIDATION_CHILD_LOG_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_LOG_PATH";
const VALIDATION_CHILD_SHARD_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_SHARD_PATH";
const VALIDATION_CHILD_GRADIENT_ARTIFACT_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_GRADIENT_ARTIFACT_PATH";
const VALIDATION_CHILD_MODEL_ARTIFACT_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_MODEL_ARTIFACT_PATH";
const VALIDATION_BATCH_SEQUENCES_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_VALIDATION_BATCH_SEQUENCES";

const CHALLENGE_WORLD_SIZE: usize = 8;
const SCOREBOARD_SLIDING_WINDOW_BATCH_SEQUENCES: usize = 1024;

/// Aggregate runtime train-step receipt emitted by the shipped runtime payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfDistributed8xH100TrainStepReceipt {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable run identifier.
    pub run_id: String,
    /// Expected world size for the exact public posture.
    pub world_size: usize,
    /// Bring-up report path that gated the train-step attempt.
    pub bringup_report_path: String,
    /// Bring-up report digest that gated the train-step attempt.
    pub bringup_report_digest: String,
    /// Runtime bootstrap receipt path that gated the train-step attempt.
    pub runtime_bootstrap_receipt_path: String,
    /// Runtime bootstrap receipt digest that gated the train-step attempt.
    pub runtime_bootstrap_receipt_digest: String,
    /// Runtime payload path used for the child fanout.
    pub runtime_payload_path: String,
    /// Manifest path used for the child fanout.
    pub runtime_manifest_path: String,
    /// Retained measurements JSON path.
    pub measurements_path: String,
    /// Retained distributed challenge receipt path.
    pub distributed_receipt_path: String,
    /// Retained typed train-step receipt path.
    pub train_step_receipt_path: String,
    /// Root directory holding all step-scoped train-step artifacts.
    pub step_scope_root_dir: String,
    /// Number of completed train steps retained under `step_scope_root_dir`.
    #[serde(default)]
    pub executed_step_count: u64,
    /// Observed cumulative train-loop wallclock before final validation.
    #[serde(default)]
    pub observed_training_time_ms: u64,
    /// Ordered measured step observations across the retained train loop.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub step_observations: Vec<ParameterGolfDistributedStepObservation>,
    /// Honest stop reason for the retained repeated train loop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Mean train loss across all rank-local microbatches.
    pub mean_train_loss: f32,
    /// Global train tokens represented by the step.
    pub train_tokens: u64,
    /// Observed end-to-end wallclock for the step.
    pub observed_step_ms: u64,
    /// Observed gradient synchronization wallclock.
    pub gradient_sync_ms: u64,
    /// Observed optimizer-step wallclock on the parent.
    pub optimizer_step_ms: u64,
    /// Aggregated gradient norm after clipping.
    pub gradient_norm_after_clip: f32,
    /// Whether gradient clipping applied.
    pub clip_applied: bool,
    /// Number of non-finite gradient values observed before clipping.
    pub non_finite_gradient_count: u64,
    /// Ordered child launch outcomes.
    pub rank_launches: Vec<ParameterGolfDistributed8xH100TrainStepRankLaunch>,
    /// Retained aggregated gradient artifact path used to reconstruct the post-step model.
    pub aggregated_gradient_artifact_path: String,
    /// Stable SHA-256 over the aggregated gradient artifact.
    pub aggregated_gradient_artifact_sha256: String,
    /// Retained exact post-step full-precision model checkpoint path.
    pub current_model_artifact_path: String,
    /// Stable SHA-256 over the full-precision model checkpoint.
    pub current_model_artifact_sha256: String,
    /// Retained exact post-step int8+zlib model artifact path.
    pub current_model_int8_zlib_artifact_path: String,
    /// Stable SHA-256 over the post-step int8+zlib model artifact.
    pub current_model_int8_zlib_artifact_sha256: String,
    /// Size of the retained post-step int8+zlib model artifact.
    pub current_model_int8_zlib_artifact_size_bytes: u64,
    /// Ordered child launch outcomes for distributed validation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation_rank_launches: Vec<ParameterGolfDistributed8xH100ValidationRankLaunch>,
    /// One measured step observation lifted into the distributed receipt lane.
    pub step_observation: ParameterGolfDistributedStepObservation,
    /// Honest distributed validation wallclock as the slowest participating rank.
    pub validation_observed_ms: u64,
    /// Total validation sequence count covered by the distributed shard plan.
    pub validation_total_sequence_count: u64,
    /// Ordered rank-local validation shard observations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation_shard_observations: Vec<ParameterGolfDistributedValidationShardObservation>,
    /// Typed distributed receipt derived from the measured step.
    pub distributed_receipt: ParameterGolfDistributedThroughputReceipt,
    /// Honest claim boundary for the train-step receipt.
    pub claim_boundary: String,
    /// Stable digest over the aggregate receipt payload.
    pub receipt_digest: String,
}

impl ParameterGolfDistributed8xH100TrainStepReceipt {
    /// Returns a stable digest over the aggregate train-step receipt payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.receipt_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_distributed_8xh100_train_step_receipt|",
            &digestible,
        )
    }
}

/// Per-rank runtime train-step receipt emitted by one child.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfDistributed8xH100TrainStepRankReceipt {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable run identifier.
    pub run_id: String,
    /// Rank owned by this child.
    pub rank: usize,
    /// Local rank on the current pod.
    pub local_rank: usize,
    /// Declared world size.
    pub world_size: usize,
    /// Exact `CUDA_VISIBLE_DEVICES` contract observed by this rank.
    pub cuda_visible_devices: String,
    /// Selected CUDA device label.
    pub selected_device_label: String,
    /// Retained rank-local log path.
    pub log_path: String,
    /// Retained assigned window path.
    pub window_path: String,
    /// Stable window identifier executed by the child.
    pub window_id: String,
    /// Retained gradient artifact path.
    pub gradient_artifact_path: String,
    /// Stable SHA-256 over the gradient artifact.
    pub gradient_artifact_sha256: String,
    /// Exact full-precision model artifact used by this rank when one was provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_model_artifact_path: Option<String>,
    /// Stable SHA-256 over the full-precision model artifact used by this rank when one was provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_model_artifact_sha256: Option<String>,
    /// Rank-local train loss for the executed microbatch.
    pub loss: f32,
    /// Rank-local phase timings.
    pub phase_timings: ParameterGolfSingleH100PhaseTimings,
    /// Rank-local wallclock for the executed gradient batch.
    pub observed_wallclock_ms: u64,
    /// Honest claim boundary for the child receipt.
    pub claim_boundary: String,
    /// Stable digest over the child receipt payload.
    pub receipt_digest: String,
}

impl ParameterGolfDistributed8xH100TrainStepRankReceipt {
    /// Returns a stable digest over the child train-step receipt payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.receipt_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_distributed_8xh100_train_step_rank_receipt|",
            &digestible,
        )
    }
}

/// Parent-observed outcome for one spawned train-step child process.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfDistributed8xH100TrainStepRankLaunch {
    /// Rank that was launched.
    pub rank: usize,
    /// Local rank that was launched.
    pub local_rank: usize,
    /// Exact `CUDA_VISIBLE_DEVICES` assignment used for the child.
    pub cuda_visible_devices: String,
    /// Retained child window path.
    pub window_path: String,
    /// Retained child gradient artifact path.
    pub gradient_artifact_path: String,
    /// Retained child receipt path.
    pub receipt_path: String,
    /// Retained child log path.
    pub log_path: String,
    /// Child exit code when one was available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Machine-readable child receipt when one was preserved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<ParameterGolfDistributed8xH100TrainStepRankReceipt>,
}

/// Retained validation-shard plan for one rank.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfDistributed8xH100ValidationShardPlan {
    /// Zero-based rank identifier.
    pub rank: usize,
    /// Validation evaluation mode used by this rank.
    pub eval_mode: ParameterGolfValidationEvalMode,
    /// Zero-based validation sequence offset covered by this rank's scored tokens.
    pub sequence_start: u64,
    /// Number of validation sequences covered by this rank's scored tokens.
    pub sequence_count: u64,
    /// Zero-based validation evaluation-unit offset owned by this rank.
    pub evaluation_unit_start: u64,
    /// Number of validation evaluation units owned by this rank.
    pub evaluation_unit_count: u64,
    /// Zero-based scored-token offset owned by this rank.
    pub scored_token_start: u64,
    /// Number of scored tokens owned by this rank.
    pub scored_token_count: u64,
}

/// Per-rank runtime validation receipt emitted by one child.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfDistributed8xH100ValidationRankReceipt {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable run identifier.
    pub run_id: String,
    /// Rank owned by this child.
    pub rank: usize,
    /// Local rank on the current pod.
    pub local_rank: usize,
    /// Declared world size.
    pub world_size: usize,
    /// Exact `CUDA_VISIBLE_DEVICES` contract observed by this rank.
    pub cuda_visible_devices: String,
    /// Selected CUDA device label.
    pub selected_device_label: String,
    /// Retained rank-local log path.
    pub log_path: String,
    /// Retained shard-plan path.
    pub shard_path: String,
    /// Retained aggregated gradient artifact path.
    pub aggregated_gradient_artifact_path: String,
    /// Stable SHA-256 over the aggregated gradient artifact.
    pub aggregated_gradient_artifact_sha256: String,
    /// Exact post-step full-precision model artifact path when validation ran against one explicit checkpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_model_artifact_path: Option<String>,
    /// Stable SHA-256 over the explicit post-step full-precision model artifact when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_model_artifact_sha256: Option<String>,
    /// Validation evaluation mode used by this rank.
    pub eval_mode: ParameterGolfValidationEvalMode,
    /// Zero-based validation sequence offset covered by this rank's scored tokens.
    pub sequence_start: u64,
    /// Number of validation sequences covered by this rank's scored tokens.
    pub sequence_count: u64,
    /// Zero-based validation evaluation-unit offset owned by this rank.
    pub evaluation_unit_start: u64,
    /// Number of validation evaluation units owned by this rank.
    pub evaluation_unit_count: u64,
    /// Zero-based scored-token offset owned by this rank.
    pub scored_token_start: u64,
    /// Number of scored tokens owned by this rank.
    pub scored_token_count: u64,
    /// Expected local validation batch size in sequences or windows for this rank.
    pub local_batch_sequences: u64,
    /// Rank-local summed loss over the shard.
    pub loss_sum: f64,
    /// Rank-local evaluated token count.
    pub token_count: u64,
    /// Rank-local evaluated byte count.
    pub byte_count: u64,
    /// Rank-local mean loss over the shard.
    pub mean_loss: f64,
    /// Rank-local bits-per-byte over the shard.
    pub bits_per_byte: f64,
    /// Rank-local wallclock for the executed validation shard.
    pub observed_wallclock_ms: u64,
    /// Honest claim boundary for the child receipt.
    pub claim_boundary: String,
    /// Stable digest over the child receipt payload.
    pub receipt_digest: String,
}

impl ParameterGolfDistributed8xH100ValidationRankReceipt {
    /// Returns a stable digest over the child validation receipt payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.receipt_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_distributed_8xh100_validation_rank_receipt|",
            &digestible,
        )
    }
}

/// Parent-observed outcome for one spawned validation child process.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfDistributed8xH100ValidationRankLaunch {
    /// Rank that was launched.
    pub rank: usize,
    /// Local rank that was launched.
    pub local_rank: usize,
    /// Exact `CUDA_VISIBLE_DEVICES` assignment used for the child.
    pub cuda_visible_devices: String,
    /// Retained child shard-plan path.
    pub shard_path: String,
    /// Retained child receipt path.
    pub receipt_path: String,
    /// Retained child log path.
    pub log_path: String,
    /// Child exit code when one was available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Machine-readable child receipt when one was preserved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<ParameterGolfDistributed8xH100ValidationRankReceipt>,
}

/// Failure while executing the distributed `8xH100` train-step seam.
#[derive(Debug, Error)]
pub enum ParameterGolfDistributed8xH100TrainStepError {
    #[error("parameter golf distributed 8xH100 train-step failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("parameter golf distributed 8xH100 train-step failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("parameter golf distributed 8xH100 train-step missing environment variable `{key}`")]
    MissingEnv { key: &'static str },
    #[error("parameter golf distributed 8xH100 train-step invalid environment `{key}`=`{value}`")]
    InvalidEnv { key: &'static str, value: String },
    #[error(
        "parameter golf distributed 8xH100 train-step child spawn failed for rank {rank}: {error}"
    )]
    ChildSpawn { rank: usize, error: std::io::Error },
    #[error(
        "parameter golf distributed 8xH100 train-step child wait failed for rank {rank}: {error}"
    )]
    ChildWait { rank: usize, error: std::io::Error },
    #[error(
        "parameter golf distributed 8xH100 train-step child receipt decode failed at `{path}`: {error}"
    )]
    ChildDecode {
        path: String,
        error: serde_json::Error,
    },
    #[error("parameter golf distributed 8xH100 train-step child rank {rank} failed before writing one explicit receipt")]
    ChildMissingReceipt { rank: usize },
    #[error(
        "parameter golf distributed 8xH100 validation child spawn failed for rank {rank}: {error}"
    )]
    ValidationChildSpawn { rank: usize, error: std::io::Error },
    #[error(
        "parameter golf distributed 8xH100 validation child wait failed for rank {rank}: {error}"
    )]
    ValidationChildWait { rank: usize, error: std::io::Error },
    #[error(
        "parameter golf distributed 8xH100 validation child receipt decode failed at `{path}`: {error}"
    )]
    ValidationChildDecode {
        path: String,
        error: serde_json::Error,
    },
    #[error("parameter golf distributed 8xH100 validation child rank {rank} failed before writing one explicit receipt")]
    ValidationChildMissingReceipt { rank: usize },
    #[error("parameter golf distributed 8xH100 train-step aggregate failed: {message}")]
    Aggregate { message: String },
    #[error(transparent)]
    Visualization(#[from] ParameterGolfDistributedVisualizationError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    SingleH100Training(#[from] ParameterGolfSingleH100TrainingError),
    #[error(transparent)]
    DistributedLane(#[from] crate::ParameterGolfDistributedLaneError),
    #[error(transparent)]
    Data(#[from] psionic_data::ParameterGolfDataError),
    #[error(transparent)]
    Model(#[from] psionic_models::ParameterGolfModelError),
    #[error(transparent)]
    Train(#[from] crate::ParameterGolfTrainError),
    #[error(transparent)]
    ReferenceTraining(#[from] crate::ParameterGolfReferenceTrainingError),
}

/// Returns whether the current process is one internal distributed train-step child.
#[must_use]
pub fn parameter_golf_distributed_8xh100_train_step_child_enabled() -> bool {
    env::var_os(CHILD_ENV_VAR).is_some()
}

/// Returns whether the current process is one internal distributed validation child.
#[must_use]
pub fn parameter_golf_distributed_8xh100_validation_child_enabled() -> bool {
    env::var_os(VALIDATION_CHILD_ENV_VAR).is_some()
}

/// Derives the canonical aggregate train-step receipt path beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_train_step_receipt_path(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("parameter_golf_distributed_8xh100_train_step.json"),
        None => root.join("parameter_golf_distributed_8xh100_train_step.json"),
    }
}

/// Derives the canonical distributed measurements path beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_measurements_path(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("parameter_golf_distributed_8xh100_measurements.json"),
        None => root.join("parameter_golf_distributed_8xh100_measurements.json"),
    }
}

/// Derives the canonical typed distributed receipt path beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_receipt_path(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("parameter_golf_distributed_8xh100_receipt.json"),
        None => root.join("parameter_golf_distributed_8xh100_receipt.json"),
    }
}

/// Derives the canonical per-rank train-step receipt directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_train_step_rank_receipts_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_train_step_receipts"),
        None => root.join("runtime_train_step_receipts"),
    }
}

/// Derives the canonical per-rank train-step log directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_train_step_rank_logs_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_train_step_logs"),
        None => root.join("runtime_train_step_logs"),
    }
}

/// Derives the canonical per-rank train-step window directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_train_step_rank_windows_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_train_step_windows"),
        None => root.join("runtime_train_step_windows"),
    }
}

/// Derives the canonical per-rank train-step gradient directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_train_step_rank_gradients_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_train_step_gradients"),
        None => root.join("runtime_train_step_gradients"),
    }
}

/// Derives the canonical aggregate step-scope directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_step_scope_root_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_step_scopes"),
        None => root.join("runtime_step_scopes"),
    }
}

/// Derives the canonical runtime model-artifact directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_model_artifacts_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_model_artifacts"),
        None => root.join("runtime_model_artifacts"),
    }
}

/// Derives the canonical per-rank validation receipt directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_validation_rank_receipts_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_validation_receipts"),
        None => root.join("runtime_validation_receipts"),
    }
}

/// Derives the canonical per-rank validation log directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_validation_rank_logs_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_validation_logs"),
        None => root.join("runtime_validation_logs"),
    }
}

/// Derives the canonical per-rank validation shard-plan directory beside the bring-up report.
#[must_use]
pub fn parameter_golf_distributed_8xh100_validation_rank_shards_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_validation_shards"),
        None => root.join("runtime_validation_shards"),
    }
}

fn distributed_validation_batch_sequences(
    geometry: &ParameterGolfBatchGeometry,
    eval_mode: &ParameterGolfValidationEvalMode,
) -> usize {
    match eval_mode {
        ParameterGolfValidationEvalMode::NonOverlapping => {
            parse_validation_batch_sequences_override()
                .unwrap_or_else(|| geometry.local_validation_batch_sequences())
        }
        ParameterGolfValidationEvalMode::SlidingWindow { .. } => {
            parse_validation_batch_sequences_override()
                .unwrap_or(SCOREBOARD_SLIDING_WINDOW_BATCH_SEQUENCES)
        }
    }
}

fn parse_validation_batch_sequences_override() -> Option<usize> {
    env::var(VALIDATION_BATCH_SEQUENCES_ENV_VAR)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn step_scope_dir(step_scope_root_dir: &Path, step_index: u64) -> PathBuf {
    step_scope_root_dir.join(format!("step_{step_index:05}"))
}

fn emit_distributed_progress_line(message: impl AsRef<str>) {
    println!("{}", message.as_ref());
}

struct ParameterGolfDistributed8xH100StepExecution {
    mean_train_loss: f32,
    train_tokens: u64,
    observed_step_ms: u64,
    gradient_sync_ms: u64,
    optimizer_step_ms: u64,
    gradient_norm_after_clip: f32,
    clip_applied: bool,
    non_finite_gradient_count: u64,
    rank_launches: Vec<ParameterGolfDistributed8xH100TrainStepRankLaunch>,
    aggregated_gradient_artifact_path: PathBuf,
    aggregated_gradient_artifact_sha256: String,
    current_model_artifact_path: PathBuf,
    current_model_artifact_sha256: String,
    current_model: ParameterGolfReferenceModel,
    step_observation: ParameterGolfDistributedStepObservation,
}

#[allow(clippy::too_many_arguments)]
fn execute_parameter_golf_distributed_8xh100_step(
    root: &Path,
    manifest_path: &Path,
    dataset_root: &Path,
    tokenizer_path: &Path,
    bundle: &psionic_data::ParameterGolfDatasetBundle,
    hyperparameters: &ParameterGolfTrainingHyperparameters,
    initial_model: &ParameterGolfReferenceModel,
    trainer_state: &mut ParameterGolfSingleH100TrainerState,
    train_contract: &ParameterGolfTokenStreamContract,
    cursor: &mut ParameterGolfTokenStreamCursor,
    geometry: &ParameterGolfBatchGeometry,
    current_exe: &Path,
    step_scope_root_dir: &Path,
    requested_train_tokens: u64,
    completed_step_count: u64,
    observed_training_time_ms: u64,
    input_model_artifact_path: Option<&Path>,
) -> Result<ParameterGolfDistributed8xH100StepExecution, ParameterGolfDistributed8xH100TrainStepError>
{
    let step_index = completed_step_count + 1;
    let step_dir = step_scope_dir(step_scope_root_dir, step_index);
    let rank_receipts_dir = step_dir.join("runtime_train_step_receipts");
    let rank_logs_dir = step_dir.join("runtime_train_step_logs");
    let rank_windows_dir = step_dir.join("runtime_train_step_windows");
    let rank_gradients_dir = step_dir.join("runtime_train_step_gradients");
    for directory in [
        &step_dir,
        &rank_receipts_dir,
        &rank_logs_dir,
        &rank_windows_dir,
        &rank_gradients_dir,
    ] {
        fs::create_dir_all(directory).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: directory.display().to_string(),
                error,
            }
        })?;
    }
    for rank in 0..CHALLENGE_WORLD_SIZE {
        let window = train_contract
            .plan_window(&bundle.manifest, cursor, requested_train_tokens)?
            .ok_or_else(|| ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "failed to plan one train-step window for rank {rank} under the distributed posture"
                ),
            })?;
        *cursor = window.end_cursor.clone();
        let window_path = rank_windows_dir.join(format!("rank_{rank}.json"));
        fs::write(
            &window_path,
            format!("{}\n", serde_json::to_string_pretty(&window)?),
        )
        .map_err(
            |error| ParameterGolfDistributed8xH100TrainStepError::Write {
                path: window_path.display().to_string(),
                error,
            },
        )?;
    }
    let step_started = Instant::now();
    let mut children = Vec::new();
    for rank in 0..CHALLENGE_WORLD_SIZE {
        let window_path = rank_windows_dir.join(format!("rank_{rank}.json"));
        let gradient_artifact_path = rank_gradients_dir.join(format!("rank_{rank}.safetensors"));
        let receipt_path = rank_receipts_dir.join(format!("rank_{rank}.json"));
        let log_path = rank_logs_dir.join(format!("rank_{rank}.log"));
        let stdout = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&log_path)
            .map_err(
                |error| ParameterGolfDistributed8xH100TrainStepError::Write {
                    path: log_path.display().to_string(),
                    error,
                },
            )?;
        let stderr = stdout.try_clone().map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: log_path.display().to_string(),
                error,
            }
        })?;
        let mut command = Command::new(current_exe);
        command
            .arg(manifest_path)
            .current_dir(root)
            .env(
                crate::PARAMETER_GOLF_EXECUTION_MODE_ENV_VAR,
                crate::PARAMETER_GOLF_DISTRIBUTED_8XH100_EXECUTION_MODE,
            )
            .env(CHILD_ENV_VAR, "1")
            .env(CHILD_RANK_ENV_VAR, rank.to_string())
            .env(CHILD_LOCAL_RANK_ENV_VAR, rank.to_string())
            .env(CHILD_WORLD_SIZE_ENV_VAR, CHALLENGE_WORLD_SIZE.to_string())
            .env(CHILD_RECEIPT_PATH_ENV_VAR, &receipt_path)
            .env(CHILD_LOG_PATH_ENV_VAR, &log_path)
            .env(CHILD_WINDOW_PATH_ENV_VAR, &window_path)
            .env(
                CHILD_GRADIENT_ARTIFACT_PATH_ENV_VAR,
                &gradient_artifact_path,
            )
            .env(
                crate::PARAMETER_GOLF_SINGLE_H100_DATASET_ROOT_ENV_VAR,
                dataset_root,
            )
            .env(
                crate::PARAMETER_GOLF_SINGLE_H100_TOKENIZER_PATH_ENV_VAR,
                tokenizer_path,
            )
            .env("CUDA_VISIBLE_DEVICES", rank.to_string())
            .env("WORLD_SIZE", CHALLENGE_WORLD_SIZE.to_string())
            .env("PSIONIC_DISTRIBUTED_RANK", rank.to_string())
            .env("PSIONIC_DISTRIBUTED_LOCAL_RANK", rank.to_string())
            .env(
                "PSIONIC_DISTRIBUTED_WORLD_SIZE",
                CHALLENGE_WORLD_SIZE.to_string(),
            )
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        if let Some(model_artifact_path) = input_model_artifact_path {
            command.env(CHILD_MODEL_ARTIFACT_PATH_ENV_VAR, model_artifact_path);
        }
        let child = command.spawn().map_err(
            |error| ParameterGolfDistributed8xH100TrainStepError::ChildSpawn {
                rank,
                error,
            },
        )?;
        children.push((rank, window_path, gradient_artifact_path, receipt_path, log_path, child));
    }

    let mut rank_launches = Vec::with_capacity(CHALLENGE_WORLD_SIZE);
    for (rank, window_path, gradient_artifact_path, receipt_path, log_path, mut child) in children {
        let status = child.wait().map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::ChildWait { rank, error }
        })?;
        let receipt = if receipt_path.is_file() {
            let bytes = fs::read(&receipt_path).map_err(|error| {
                ParameterGolfDistributed8xH100TrainStepError::Read {
                    path: receipt_path.display().to_string(),
                    error,
                }
            })?;
            Some(
                serde_json::from_slice::<ParameterGolfDistributed8xH100TrainStepRankReceipt>(
                    &bytes,
                )
                .map_err(|error| {
                    ParameterGolfDistributed8xH100TrainStepError::ChildDecode {
                        path: receipt_path.display().to_string(),
                        error,
                    }
                })?,
            )
        } else {
            None
        };
        rank_launches.push(ParameterGolfDistributed8xH100TrainStepRankLaunch {
            rank,
            local_rank: rank,
            cuda_visible_devices: rank.to_string(),
            window_path: window_path.display().to_string(),
            gradient_artifact_path: gradient_artifact_path.display().to_string(),
            receipt_path: receipt_path.display().to_string(),
            log_path: log_path.display().to_string(),
            exit_code: status.code(),
            receipt,
        });
    }

    for launch in &rank_launches {
        if launch.exit_code != Some(0) {
            return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "distributed train-step child rank {} exited with {:?}; see {}",
                    launch.rank, launch.exit_code, launch.log_path
                ),
            });
        }
        if launch.receipt.is_none() {
            return Err(
                ParameterGolfDistributed8xH100TrainStepError::ChildMissingReceipt {
                    rank: launch.rank,
                },
            );
        }
    }

    let sync_started = Instant::now();
    let mut accumulated_gradients = zero_gradients(trainer_state);
    for launch in &rank_launches {
        let rank_receipt = launch
            .receipt
            .as_ref()
            .expect("rank receipt presence validated above");
        let gradients = load_gradient_artifact(Path::new(&rank_receipt.gradient_artifact_path))?;
        crate::accumulate_gradients(
            accumulated_gradients.as_mut_slice(),
            trainer_state,
            &gradients,
            CHALLENGE_WORLD_SIZE as f32,
        )?;
    }
    let gradient_sync_ms = duration_ms(sync_started);

    let clip_observation =
        clip_gradients(accumulated_gradients.as_mut_slice(), hyperparameters.grad_clip_norm);
    let aggregated_gradient_artifact_path = step_dir.join("aggregated_gradients.safetensors");
    let aggregated_gradient_artifact_sha256 = write_gradient_artifact(
        &aggregated_gradient_artifact_path,
        &accumulated_gradients
            .iter()
            .map(|(parameter_id, values)| (parameter_id.clone(), values.clone()))
            .collect::<BTreeMap<_, _>>(),
    )?;
    let optimizer_started = Instant::now();
    let learning_rate_multiplier = hyperparameters
        .learning_rate_multiplier(completed_step_count, observed_training_time_ms as f32);
    let muon_momentum = hyperparameters.muon_momentum_at_step(completed_step_count);
    apply_gradients_to_state(
        trainer_state,
        accumulated_gradients.as_slice(),
        learning_rate_multiplier,
        muon_momentum,
        step_index,
    )?;
    let current_model = materialize_current_model(initial_model, trainer_state)?;
    let current_model_artifact_path = step_dir.join("current_model.safetensors");
    let current_model_artifact_sha256 = write_bytes_artifact(
        &current_model_artifact_path,
        &export_parameter_golf_full_precision_model_bytes(&current_model)?,
    )?;
    let optimizer_step_ms = duration_ms(optimizer_started);
    let observed_step_ms = duration_ms(step_started);
    let mean_train_loss = rank_launches
        .iter()
        .map(|launch| {
            launch
                .receipt
                .as_ref()
                .expect("rank receipt presence validated above")
                .loss
        })
        .sum::<f32>()
        / CHALLENGE_WORLD_SIZE as f32;
    let train_tokens = geometry.train_batch_tokens as u64;
    let step_observation =
        ParameterGolfDistributedStepObservation::new(step_index, 0, observed_step_ms, train_tokens);

    Ok(ParameterGolfDistributed8xH100StepExecution {
        mean_train_loss,
        train_tokens,
        observed_step_ms,
        gradient_sync_ms,
        optimizer_step_ms,
        gradient_norm_after_clip: clip_observation.gradient_norm_after_clip.unwrap_or_default(),
        clip_applied: clip_observation.clip_applied,
        non_finite_gradient_count: u64::from(clip_observation.non_finite_count),
        rank_launches,
        aggregated_gradient_artifact_path,
        aggregated_gradient_artifact_sha256,
        current_model_artifact_path,
        current_model_artifact_sha256,
        current_model,
        step_observation,
    })
}

/// Executes one real multi-rank train step from the shipped runtime payload.
#[allow(clippy::too_many_arguments)]
pub fn execute_parameter_golf_distributed_8xh100_train_step(
    root: &Path,
    manifest_path: &Path,
    run_id: &str,
    bringup_report_path: &Path,
    bringup_report: &ParameterGolfDistributed8xH100BringupReport,
    bootstrap_receipt_path: &Path,
    bootstrap_receipt: &ParameterGolfDistributed8xH100RuntimeBootstrapReceipt,
    mut live_visualization_writer: Option<&mut ParameterGolfDistributedLiveVisualizationWriter>,
) -> Result<ParameterGolfDistributed8xH100TrainStepReceipt, ParameterGolfDistributed8xH100TrainStepError>
{
    if bootstrap_receipt.successful_rank_count != CHALLENGE_WORLD_SIZE {
        return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "runtime bootstrap admitted only {} of {} ranks",
                bootstrap_receipt.successful_rank_count, CHALLENGE_WORLD_SIZE
            ),
        });
    }

    let bringup_report_relpath = bringup_report_path
        .strip_prefix(root)
        .unwrap_or(bringup_report_path)
        .display()
        .to_string();
    let train_step_receipt_path =
        parameter_golf_distributed_8xh100_train_step_receipt_path(root, &bringup_report_relpath);
    let measurements_path =
        parameter_golf_distributed_8xh100_measurements_path(root, &bringup_report_relpath);
    let distributed_receipt_path =
        parameter_golf_distributed_8xh100_receipt_path(root, &bringup_report_relpath);
    let step_scope_root_dir =
        parameter_golf_distributed_8xh100_step_scope_root_dir(root, &bringup_report_relpath);
    let model_artifacts_dir =
        parameter_golf_distributed_8xh100_model_artifacts_dir(root, &bringup_report_relpath);
    let validation_rank_receipts_dir = parameter_golf_distributed_8xh100_validation_rank_receipts_dir(
        root,
        &bringup_report_relpath,
    );
    let validation_rank_logs_dir =
        parameter_golf_distributed_8xh100_validation_rank_logs_dir(root, &bringup_report_relpath);
    let validation_rank_shards_dir =
        parameter_golf_distributed_8xh100_validation_rank_shards_dir(root, &bringup_report_relpath);
    for directory in [
        &step_scope_root_dir,
        &model_artifacts_dir,
        &validation_rank_receipts_dir,
        &validation_rank_logs_dir,
        &validation_rank_shards_dir,
    ] {
        fs::create_dir_all(directory).map_err(
            |error| ParameterGolfDistributed8xH100TrainStepError::Write {
                path: directory.display().to_string(),
                error,
            },
        )?;
    }
    if let Some(parent) = train_step_receipt_path.parent() {
        fs::create_dir_all(parent).map_err(
            |error| ParameterGolfDistributed8xH100TrainStepError::Write {
                path: parent.display().to_string(),
                error,
            },
        )?;
    }

    let dataset_root = PathBuf::from(required_env(
        crate::PARAMETER_GOLF_SINGLE_H100_DATASET_ROOT_ENV_VAR,
    )?);
    let tokenizer_path = PathBuf::from(required_env(
        crate::PARAMETER_GOLF_SINGLE_H100_TOKENIZER_PATH_ENV_VAR,
    )?);
    let tokenizer_bytes = fs::read(&tokenizer_path).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Read {
            path: tokenizer_path.display().to_string(),
            error,
        }
    })?;
    let tokenizer_digest = build_tokenizer_digest(tokenizer_bytes.as_slice());
    let geometry = ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults();
    let bundle = parameter_golf_dataset_bundle_from_local_dir(
        DatasetKey::new(
            crate::PARAMETER_GOLF_SINGLE_H100_DATASET_REF,
            crate::PARAMETER_GOLF_SINGLE_H100_DATASET_VERSION,
        ),
        &dataset_root,
        String::from(PARAMETER_GOLF_SINGLE_H100_VARIANT),
        tokenizer_digest,
        tokenizer_path.display().to_string(),
        None,
    )?;
    let initial_model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    let hyperparameters = ParameterGolfTrainingHyperparameters::baseline_defaults();
    let optimizer_plan = parameter_golf_optimizer_plan(initial_model.descriptor(), &hyperparameters)?;
    let mut trainer_state = seed_parameter_states(&initial_model, &optimizer_plan)?;
    let mut cursor = ParameterGolfTokenStreamCursor::new(PARAMETER_GOLF_TRAIN_SPLIT_NAME);
    let train_contract = ParameterGolfTokenStreamContract::new(
        bundle.manifest.key.clone(),
        PARAMETER_GOLF_TRAIN_SPLIT_NAME,
    )
    .with_mode(DatasetIterationMode::Repeat);
    let requested_train_tokens = geometry.local_train_batch_tokens().saturating_add(1) as u64;
    let current_exe = env::current_exe().map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Read {
            path: String::from("current_exe"),
            error,
        }
    })?;
    let runtime_payload_path = current_exe.display().to_string();
    let manifest_path = manifest_path
        .canonicalize()
        .unwrap_or_else(|_| manifest_path.to_path_buf());
    let max_wallclock_ms = hyperparameters
        .max_wallclock_seconds
        .filter(|seconds| *seconds > 0.0)
        .map(|seconds| (seconds * 1000.0) as u64);
    let mut executed_step_count = 0_u64;
    let mut observed_training_time_ms = 0_u64;
    let mut stop_reason = None;
    let mut step_observations = Vec::new();
    let mut current_model_artifact_input_path = None;
    let mut final_step_execution = None;
    loop {
        if executed_step_count >= hyperparameters.iterations {
            stop_reason = Some(String::from("iteration_cap_reached"));
            break;
        }
        let step_execution = execute_parameter_golf_distributed_8xh100_step(
            root,
            &manifest_path,
            &dataset_root,
            &tokenizer_path,
            &bundle,
            &hyperparameters,
            &initial_model,
            &mut trainer_state,
            &train_contract,
            &mut cursor,
            &geometry,
            &current_exe,
            &step_scope_root_dir,
            requested_train_tokens,
            executed_step_count,
            observed_training_time_ms,
            current_model_artifact_input_path.as_deref(),
        )?;
        executed_step_count = step_execution.step_observation.global_step;
        observed_training_time_ms =
            observed_training_time_ms.saturating_add(step_execution.observed_step_ms);
        step_observations.push(step_execution.step_observation.clone());
        emit_distributed_progress_line(format!(
            "step:{}/{} train_loss:{:.4} train_time:{}ms step_avg:{:.2}ms",
            executed_step_count,
            hyperparameters.iterations,
            step_execution.mean_train_loss,
            observed_training_time_ms,
            observed_training_time_ms as f64 / executed_step_count.max(1) as f64,
        ));
        current_model_artifact_input_path =
            Some(step_execution.current_model_artifact_path.clone());
        final_step_execution = Some(step_execution);
        if max_wallclock_ms.is_some_and(|wallclock_ms| observed_training_time_ms >= wallclock_ms) {
            stop_reason = Some(String::from("wallclock_cap_reached"));
            emit_distributed_progress_line(format!(
                "stopping_early: wallclock_cap train_time:{}ms step:{}/{}",
                observed_training_time_ms, executed_step_count, hyperparameters.iterations,
            ));
            break;
        }
    }
    if stop_reason.is_none() {
        stop_reason = Some(String::from("iteration_cap_reached"));
    }
    let final_step_execution = final_step_execution.ok_or_else(|| {
        ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: String::from(
                "distributed 8xH100 train loop finished without one successful step",
            ),
        }
    })?;
    let current_model = final_step_execution.current_model.clone();
    let current_model_int8_zlib_artifact =
        export_parameter_golf_int8_zlib_model_artifact(&current_model, run_id, executed_step_count)?;
    let current_model_int8_zlib_artifact_path = model_artifacts_dir.join(format!(
        "post_step_{executed_step_count}_final_model.int8.ptz"
    ));
    let current_model_int8_zlib_artifact_sha256 = write_bytes_artifact(
        &current_model_int8_zlib_artifact_path,
        current_model_int8_zlib_artifact.bytes.as_slice(),
    )?;
    let current_model_int8_zlib_artifact_size_bytes =
        current_model_int8_zlib_artifact.bytes.len() as u64;
    let validation_eval_mode = ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 };
    let validation_batch_sequences =
        distributed_validation_batch_sequences(&geometry, &validation_eval_mode) as u64;
    let validation_tokens = load_parameter_golf_validation_tokens_from_paths(
        &bundle
            .validation_shards
            .iter()
            .map(|receipt| PathBuf::from(&receipt.path))
            .collect::<Vec<_>>(),
        geometry.train_sequence_length,
    )?;
    let validation_total_sequence_count =
        ((validation_tokens.len().saturating_sub(1)) / geometry.train_sequence_length) as u64;
    let validation_shard_plan = build_validation_observation_plan(
        validation_total_sequence_count,
        geometry.train_sequence_length,
        &validation_eval_mode,
        CHALLENGE_WORLD_SIZE,
    )?;
    let mut validation_children = Vec::with_capacity(CHALLENGE_WORLD_SIZE);
    if let Some(writer) = live_visualization_writer.as_deref_mut() {
        writer.record_phase(
            "evaluation",
            Some(String::from("distributed_validation")),
            format!(
                "Launching {} distributed validation ranks for optimizer step 1.",
                CHALLENGE_WORLD_SIZE
            ),
            vec![
                String::from("distributed_validation"),
                String::from("rank_fanout"),
                String::from("cuda_evaluation"),
            ],
            Some(1),
            true,
        )?;
    }
    for shard in &validation_shard_plan {
        let shard_path = validation_rank_shards_dir.join(format!("rank_{}.json", shard.rank));
        fs::write(
            &shard_path,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&ParameterGolfDistributed8xH100ValidationShardPlan {
                    rank: shard.rank,
                    eval_mode: validation_eval_mode.clone(),
                    sequence_start: shard.sequence_start,
                    sequence_count: shard.sequence_count,
                    evaluation_unit_start: shard.evaluation_unit_start,
                    evaluation_unit_count: shard.evaluation_unit_count,
                    scored_token_start: shard.scored_token_start,
                    scored_token_count: shard.scored_token_count,
                })?
            ),
        )
        .map_err(
            |error| ParameterGolfDistributed8xH100TrainStepError::Write {
                path: shard_path.display().to_string(),
                error,
            },
        )?;
        let receipt_path = validation_rank_receipts_dir.join(format!("rank_{}.json", shard.rank));
        let log_path = validation_rank_logs_dir.join(format!("rank_{}.log", shard.rank));
        let stdout = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&log_path)
            .map_err(
                |error| ParameterGolfDistributed8xH100TrainStepError::Write {
                    path: log_path.display().to_string(),
                    error,
                },
            )?;
        let stderr = stdout.try_clone().map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: log_path.display().to_string(),
                error,
            }
        })?;
        let child = Command::new(&current_exe)
            .arg(&manifest_path)
            .current_dir(root)
            .env(
                crate::PARAMETER_GOLF_EXECUTION_MODE_ENV_VAR,
                crate::PARAMETER_GOLF_DISTRIBUTED_8XH100_EXECUTION_MODE,
            )
            .env(VALIDATION_CHILD_ENV_VAR, "1")
            .env(VALIDATION_CHILD_RANK_ENV_VAR, shard.rank.to_string())
            .env(VALIDATION_CHILD_LOCAL_RANK_ENV_VAR, shard.rank.to_string())
            .env(
                VALIDATION_CHILD_WORLD_SIZE_ENV_VAR,
                CHALLENGE_WORLD_SIZE.to_string(),
            )
            .env(VALIDATION_CHILD_RECEIPT_PATH_ENV_VAR, &receipt_path)
            .env(VALIDATION_CHILD_LOG_PATH_ENV_VAR, &log_path)
            .env(VALIDATION_CHILD_SHARD_PATH_ENV_VAR, &shard_path)
            .env(
                VALIDATION_CHILD_GRADIENT_ARTIFACT_PATH_ENV_VAR,
                &final_step_execution.aggregated_gradient_artifact_path,
            )
            .env(
                VALIDATION_CHILD_MODEL_ARTIFACT_PATH_ENV_VAR,
                &final_step_execution.current_model_artifact_path,
            )
            .env(
                crate::PARAMETER_GOLF_SINGLE_H100_DATASET_ROOT_ENV_VAR,
                &dataset_root,
            )
            .env(
                crate::PARAMETER_GOLF_SINGLE_H100_TOKENIZER_PATH_ENV_VAR,
                &tokenizer_path,
            )
            .env("CUDA_VISIBLE_DEVICES", shard.rank.to_string())
            .env("WORLD_SIZE", CHALLENGE_WORLD_SIZE.to_string())
            .env("PSIONIC_DISTRIBUTED_RANK", shard.rank.to_string())
            .env("PSIONIC_DISTRIBUTED_LOCAL_RANK", shard.rank.to_string())
            .env(
                "PSIONIC_DISTRIBUTED_WORLD_SIZE",
                CHALLENGE_WORLD_SIZE.to_string(),
            )
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|error| {
                ParameterGolfDistributed8xH100TrainStepError::ValidationChildSpawn {
                    rank: shard.rank,
                    error,
                }
            })?;
        validation_children.push((shard.rank, shard_path, receipt_path, log_path, child));
    }
    let mut validation_rank_launches = Vec::with_capacity(CHALLENGE_WORLD_SIZE);
    let mut completed_validation_rank_count = 0_usize;
    for (rank, shard_path, receipt_path, log_path, mut child) in validation_children {
        let status = child.wait().map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::ValidationChildWait { rank, error }
        })?;
        let receipt = if receipt_path.is_file() {
            let bytes = fs::read(&receipt_path).map_err(|error| {
                ParameterGolfDistributed8xH100TrainStepError::Read {
                    path: receipt_path.display().to_string(),
                    error,
                }
            })?;
            Some(
                serde_json::from_slice::<ParameterGolfDistributed8xH100ValidationRankReceipt>(
                    &bytes,
                )
                .map_err(|error| {
                    ParameterGolfDistributed8xH100TrainStepError::ValidationChildDecode {
                        path: receipt_path.display().to_string(),
                        error,
                    }
                })?,
            )
        } else {
            None
        };
        validation_rank_launches.push(ParameterGolfDistributed8xH100ValidationRankLaunch {
            rank,
            local_rank: rank,
            cuda_visible_devices: rank.to_string(),
            shard_path: shard_path.display().to_string(),
            receipt_path: receipt_path.display().to_string(),
            log_path: log_path.display().to_string(),
            exit_code: status.code(),
            receipt,
        });
        completed_validation_rank_count = completed_validation_rank_count.saturating_add(1);
        if let Some(writer) = live_visualization_writer.as_deref_mut() {
            writer.record_phase(
                "evaluation",
                Some(String::from("distributed_validation")),
                format!(
                    "Distributed validation ranks completed: {completed_validation_rank_count}/{}.",
                    CHALLENGE_WORLD_SIZE
                ),
                vec![
                    String::from("distributed_validation"),
                    String::from("rank_wait"),
                    String::from("cuda_evaluation"),
                ],
                Some(1),
                false,
            )?;
        }
    }
    for launch in &validation_rank_launches {
        if launch.exit_code != Some(0) {
            return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "distributed validation child rank {} exited with {:?}; see {}",
                    launch.rank, launch.exit_code, launch.log_path
                ),
            });
        }
        if launch.receipt.is_none() {
            return Err(
                ParameterGolfDistributed8xH100TrainStepError::ValidationChildMissingReceipt {
                    rank: launch.rank,
                },
            );
        }
    }
    let mut validation_shard_observations = Vec::with_capacity(validation_rank_launches.len());
    let mut validation_observed_ms = 0_u64;
    for launch in &validation_rank_launches {
        let receipt = launch
            .receipt
            .as_ref()
            .expect("validation receipt presence validated above");
        validation_observed_ms = validation_observed_ms.max(receipt.observed_wallclock_ms);
        validation_shard_observations.push(ParameterGolfDistributedValidationShardObservation {
            rank: receipt.rank,
            sequence_start: receipt.sequence_start,
            sequence_count: receipt.sequence_count,
            evaluation_unit_start: receipt.evaluation_unit_start,
            evaluation_unit_count: receipt.evaluation_unit_count,
            scored_token_start: receipt.scored_token_start,
            scored_token_count: receipt.scored_token_count,
            loss_sum: receipt.loss_sum,
            token_count: receipt.token_count,
            byte_count: receipt.byte_count,
            observed_ms: receipt.observed_wallclock_ms,
        });
    }
    validation_shard_observations.sort_by_key(|observation| observation.rank);

    let mut measurements = ParameterGolfRunPod8xH100Measurements::challenge_defaults();
    measurements.run_id = Some(String::from(run_id));
    measurements.mesh_id = Some(String::from("mesh.parameter_golf.runpod_8xh100"));
    measurements.step_observations = step_observations.clone();
    measurements.validation_eval_mode = validation_eval_mode.clone();
    measurements.validation_batch_sequences = validation_batch_sequences;
    measurements.validation_observed_ms = validation_observed_ms;
    measurements.validation_total_sequence_count = validation_total_sequence_count;
    measurements.validation_shard_observations = validation_shard_observations.clone();
    fs::write(
        &measurements_path,
        format!("{}\n", serde_json::to_string_pretty(&measurements)?),
    )
    .map_err(
        |error| ParameterGolfDistributed8xH100TrainStepError::Write {
            path: measurements_path.display().to_string(),
            error,
        },
    )?;

    let machine = inspect_local_distributed_8xh100_machine();
    let mut distributed_config = crate::ParameterGolfDistributed8xH100Config::challenge_defaults();
    distributed_config.run_id = String::from(run_id);
    distributed_config.mesh_id = String::from("mesh.parameter_golf.runpod_8xh100");
    distributed_config.step_observations = step_observations.clone();
    distributed_config.validation_eval_mode = validation_eval_mode;
    distributed_config.validation_batch_sequences = validation_batch_sequences;
    distributed_config.validation_observed_ms = validation_observed_ms;
    distributed_config.validation_total_sequence_count = validation_total_sequence_count;
    distributed_config.validation_shard_observations = validation_shard_observations.clone();
    let distributed_receipt = benchmark_parameter_golf_distributed_8xh100(
        initial_model.descriptor(),
        &hyperparameters,
        machine.observed_cuda_devices.as_slice(),
        &parameter_golf_runpod_8xh100_capability_profile(),
        &distributed_config,
    )?;
    fs::write(
        &distributed_receipt_path,
        format!("{}\n", serde_json::to_string_pretty(&distributed_receipt)?),
    )
    .map_err(
        |error| ParameterGolfDistributed8xH100TrainStepError::Write {
            path: distributed_receipt_path.display().to_string(),
            error,
        },
    )?;

    let mut receipt = ParameterGolfDistributed8xH100TrainStepReceipt {
        schema_version: 1,
        run_id: String::from(run_id),
        world_size: CHALLENGE_WORLD_SIZE,
        bringup_report_path: bringup_report_path.display().to_string(),
        bringup_report_digest: bringup_report.report_digest.clone(),
        runtime_bootstrap_receipt_path: bootstrap_receipt_path.display().to_string(),
        runtime_bootstrap_receipt_digest: bootstrap_receipt.receipt_digest.clone(),
        runtime_payload_path,
        runtime_manifest_path: manifest_path.display().to_string(),
        measurements_path: measurements_path.display().to_string(),
        distributed_receipt_path: distributed_receipt_path.display().to_string(),
        train_step_receipt_path: train_step_receipt_path.display().to_string(),
        step_scope_root_dir: step_scope_root_dir.display().to_string(),
        executed_step_count,
        observed_training_time_ms,
        step_observations,
        stop_reason,
        mean_train_loss: final_step_execution.mean_train_loss,
        train_tokens: final_step_execution.train_tokens,
        observed_step_ms: final_step_execution.observed_step_ms,
        gradient_sync_ms: final_step_execution.gradient_sync_ms,
        optimizer_step_ms: final_step_execution.optimizer_step_ms,
        gradient_norm_after_clip: final_step_execution.gradient_norm_after_clip,
        clip_applied: final_step_execution.clip_applied,
        non_finite_gradient_count: final_step_execution.non_finite_gradient_count,
        rank_launches: final_step_execution.rank_launches,
        aggregated_gradient_artifact_path: final_step_execution
            .aggregated_gradient_artifact_path
            .display()
            .to_string(),
        aggregated_gradient_artifact_sha256: final_step_execution
            .aggregated_gradient_artifact_sha256,
        current_model_artifact_path: final_step_execution
            .current_model_artifact_path
            .display()
            .to_string(),
        current_model_artifact_sha256: final_step_execution.current_model_artifact_sha256,
        current_model_int8_zlib_artifact_path: current_model_int8_zlib_artifact_path
            .display()
            .to_string(),
        current_model_int8_zlib_artifact_sha256,
        current_model_int8_zlib_artifact_size_bytes,
        validation_rank_launches,
        step_observation: final_step_execution.step_observation,
        validation_observed_ms,
        validation_total_sequence_count,
        validation_shard_observations,
        distributed_receipt,
        claim_boundary: String::from(
            "This receipt proves the exported-folder distributed runtime executed one real wallclock-bounded repeated 8-rank Parameter Golf train loop, retained step-scoped train artifacts under the benchmark root, reconstructed the final post-step model on all ranks, and emitted one real distributed validation aggregation into the typed distributed receipt lane. It does not by itself claim record-track promotion.",
        ),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.stable_digest();
    fs::write(
        &train_step_receipt_path,
        format!("{}\n", serde_json::to_string_pretty(&receipt)?),
    )
    .map_err(
        |error| ParameterGolfDistributed8xH100TrainStepError::Write {
            path: train_step_receipt_path.display().to_string(),
            error,
        },
    )?;
    if let Some(writer) = live_visualization_writer.as_deref_mut() {
        writer.record_train_step_receipt(&receipt)?;
    }
    Ok(receipt)
}

/// Executes one train-step child rank inside the shipped distributed runtime.
pub fn execute_parameter_golf_distributed_8xh100_train_step_child(
    run_id: &str,
) -> Result<
    ParameterGolfDistributed8xH100TrainStepRankReceipt,
    ParameterGolfDistributed8xH100TrainStepError,
> {
    let rank = parse_env_usize(CHILD_RANK_ENV_VAR)?;
    let local_rank = parse_env_usize(CHILD_LOCAL_RANK_ENV_VAR)?;
    let world_size = parse_env_usize(CHILD_WORLD_SIZE_ENV_VAR)?;
    let receipt_path = PathBuf::from(required_env(CHILD_RECEIPT_PATH_ENV_VAR)?);
    let log_path = required_env(CHILD_LOG_PATH_ENV_VAR)?;
    let window_path = PathBuf::from(required_env(CHILD_WINDOW_PATH_ENV_VAR)?);
    let gradient_artifact_path = PathBuf::from(required_env(CHILD_GRADIENT_ARTIFACT_PATH_ENV_VAR)?);
    let input_model_artifact_path =
        env::var_os(CHILD_MODEL_ARTIFACT_PATH_ENV_VAR).map(PathBuf::from);
    let cuda_visible_devices = env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
    if world_size != CHALLENGE_WORLD_SIZE {
        return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "distributed train-step child requires world_size={} but observed {}",
                CHALLENGE_WORLD_SIZE, world_size
            ),
        });
    }
    if local_rank != rank {
        return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "distributed train-step child requires local_rank == rank on the single pod, found rank={rank} local_rank={local_rank}"
            ),
        });
    }
    let dataset_root = PathBuf::from(required_env(
        crate::PARAMETER_GOLF_SINGLE_H100_DATASET_ROOT_ENV_VAR,
    )?);
    let tokenizer_path = PathBuf::from(required_env(
        crate::PARAMETER_GOLF_SINGLE_H100_TOKENIZER_PATH_ENV_VAR,
    )?);
    let window_bytes = fs::read(&window_path).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Read {
            path: window_path.display().to_string(),
            error,
        }
    })?;
    let window: ParameterGolfTokenStreamWindow = serde_json::from_slice(&window_bytes)?;
    let tokenizer_bytes = fs::read(&tokenizer_path).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Read {
            path: tokenizer_path.display().to_string(),
            error,
        }
    })?;
    let tokenizer_digest = build_tokenizer_digest(tokenizer_bytes.as_slice());
    let bundle = parameter_golf_dataset_bundle_from_local_dir(
        DatasetKey::new(
            crate::PARAMETER_GOLF_SINGLE_H100_DATASET_REF,
            crate::PARAMETER_GOLF_SINGLE_H100_DATASET_VERSION,
        ),
        &dataset_root,
        String::from(PARAMETER_GOLF_SINGLE_H100_VARIANT),
        tokenizer_digest,
        tokenizer_path.display().to_string(),
        None,
    )?;
    let baseline_model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    let (model, input_model_artifact_sha256) = match input_model_artifact_path.as_ref() {
        Some(path) => {
            let bytes = fs::read(path).map_err(|error| {
                ParameterGolfDistributed8xH100TrainStepError::Read {
                    path: path.display().to_string(),
                    error,
                }
            })?;
            (
                restore_parameter_golf_model_from_safetensors(&baseline_model, bytes.as_slice())?,
                Some(sha256_hex(bytes.as_slice())),
            )
        }
        None => (baseline_model, None),
    };
    let geometry = ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults();
    let mut cuda_backend = CudaBackend::new();
    let selected_device = cuda_backend.selected_device().cloned().ok_or_else(|| {
        ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "distributed train-step child rank {rank} could not select one CUDA device"
            ),
        }
    })?;
    let selected_device_label = selected_device
        .device_name
        .clone()
        .unwrap_or_else(|| String::from("unknown"));
    let mut graph_cache = BTreeMap::new();

    let started = Instant::now();
    let gradient_batch = execute_parameter_golf_training_gradient_batch(
        &mut cuda_backend,
        &selected_device.device,
        &bundle,
        &model,
        &mut graph_cache,
        None,
        &geometry,
        &window,
    )?;
    let observed_wallclock_ms = duration_ms(started);
    let gradient_artifact_sha256 =
        write_gradient_artifact(&gradient_artifact_path, &gradient_batch.parameter_gradients)?;

    let mut receipt = ParameterGolfDistributed8xH100TrainStepRankReceipt {
        schema_version: 1,
        run_id: String::from(run_id),
        rank,
        local_rank,
        world_size,
        cuda_visible_devices,
        selected_device_label,
        log_path,
        window_path: window_path.display().to_string(),
        window_id: gradient_batch.window_id,
        gradient_artifact_path: gradient_artifact_path.display().to_string(),
        gradient_artifact_sha256,
        input_model_artifact_path: input_model_artifact_path
            .as_ref()
            .map(|path| path.display().to_string()),
        input_model_artifact_sha256,
        loss: gradient_batch.loss,
        phase_timings: gradient_batch.phase_timings,
        observed_wallclock_ms,
        claim_boundary: String::from(
            "This child receipt proves one rank-local distributed train-step gradient batch executed on one explicit H100 binding and exported one compact gradient artifact for later mesh aggregation. It does not yet claim later distributed validation or final artifact closure.",
        ),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.stable_digest();
    if let Some(parent) = receipt_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    fs::write(
        &receipt_path,
        format!("{}\n", serde_json::to_string_pretty(&receipt)?),
    )
    .map_err(
        |error| ParameterGolfDistributed8xH100TrainStepError::Write {
            path: receipt_path.display().to_string(),
            error,
        },
    )?;
    Ok(receipt)
}

/// Executes one distributed validation child rank inside the shipped runtime.
pub fn execute_parameter_golf_distributed_8xh100_validation_child(
    run_id: &str,
) -> Result<
    ParameterGolfDistributed8xH100ValidationRankReceipt,
    ParameterGolfDistributed8xH100TrainStepError,
> {
    let rank = parse_env_usize(VALIDATION_CHILD_RANK_ENV_VAR)?;
    let local_rank = parse_env_usize(VALIDATION_CHILD_LOCAL_RANK_ENV_VAR)?;
    let world_size = parse_env_usize(VALIDATION_CHILD_WORLD_SIZE_ENV_VAR)?;
    let receipt_path = PathBuf::from(required_env(VALIDATION_CHILD_RECEIPT_PATH_ENV_VAR)?);
    let log_path = required_env(VALIDATION_CHILD_LOG_PATH_ENV_VAR)?;
    let shard_path = PathBuf::from(required_env(VALIDATION_CHILD_SHARD_PATH_ENV_VAR)?);
    let aggregated_gradient_artifact_path = PathBuf::from(required_env(
        VALIDATION_CHILD_GRADIENT_ARTIFACT_PATH_ENV_VAR,
    )?);
    let current_model_artifact_path =
        env::var_os(VALIDATION_CHILD_MODEL_ARTIFACT_PATH_ENV_VAR).map(PathBuf::from);
    let cuda_visible_devices = env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
    if world_size != CHALLENGE_WORLD_SIZE {
        return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "distributed validation child requires world_size={} but observed {}",
                CHALLENGE_WORLD_SIZE, world_size
            ),
        });
    }
    if local_rank != rank {
        return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "distributed validation child requires local_rank == rank on the single pod, found rank={rank} local_rank={local_rank}"
            ),
        });
    }
    let dataset_root = PathBuf::from(required_env(
        crate::PARAMETER_GOLF_SINGLE_H100_DATASET_ROOT_ENV_VAR,
    )?);
    let tokenizer_path = PathBuf::from(required_env(
        crate::PARAMETER_GOLF_SINGLE_H100_TOKENIZER_PATH_ENV_VAR,
    )?);
    let shard_bytes = fs::read(&shard_path).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Read {
            path: shard_path.display().to_string(),
            error,
        }
    })?;
    let shard: ParameterGolfDistributed8xH100ValidationShardPlan =
        serde_json::from_slice(&shard_bytes)?;
    let tokenizer_bytes = fs::read(&tokenizer_path).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Read {
            path: tokenizer_path.display().to_string(),
            error,
        }
    })?;
    let tokenizer_digest = build_tokenizer_digest(tokenizer_bytes.as_slice());
    let bundle = parameter_golf_dataset_bundle_from_local_dir(
        DatasetKey::new(
            crate::PARAMETER_GOLF_SINGLE_H100_DATASET_REF,
            crate::PARAMETER_GOLF_SINGLE_H100_DATASET_VERSION,
        ),
        &dataset_root,
        String::from(PARAMETER_GOLF_SINGLE_H100_VARIANT),
        tokenizer_digest,
        tokenizer_path.display().to_string(),
        None,
    )?;
    let aggregated_gradients = load_gradient_artifact(&aggregated_gradient_artifact_path)?;
    let aggregated_gradient_artifact_sha256 = sha256_hex(
        &fs::read(&aggregated_gradient_artifact_path).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Read {
                path: aggregated_gradient_artifact_path.display().to_string(),
                error,
            }
        })?,
    );
    let baseline_model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    let (current_model, current_model_artifact_sha256) = match current_model_artifact_path.as_ref()
    {
        Some(path) => {
            let bytes = fs::read(path).map_err(|error| {
                ParameterGolfDistributed8xH100TrainStepError::Read {
                    path: path.display().to_string(),
                    error,
                }
            })?;
            (
                restore_parameter_golf_model_from_safetensors(&baseline_model, bytes.as_slice())?,
                Some(sha256_hex(bytes.as_slice())),
            )
        }
        None => {
            let hyperparameters = ParameterGolfTrainingHyperparameters::baseline_defaults();
            let optimizer_plan =
                parameter_golf_optimizer_plan(baseline_model.descriptor(), &hyperparameters)?;
            let mut trainer_state = seed_parameter_states(&baseline_model, &optimizer_plan)?;
            let learning_rate_multiplier = hyperparameters.learning_rate_multiplier(0, 0.0);
            let muon_momentum = hyperparameters.muon_momentum_at_step(0);
            apply_gradients_to_state(
                &mut trainer_state,
                &aggregated_gradients.into_iter().collect::<Vec<_>>(),
                learning_rate_multiplier,
                muon_momentum,
                1,
            )?;
            (
                materialize_current_model(&baseline_model, &trainer_state)?,
                None,
            )
        }
    };
    let geometry = ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults();
    let validation_tokens = load_parameter_golf_validation_tokens_from_paths(
        &bundle
            .validation_shards
            .iter()
            .map(|receipt| PathBuf::from(&receipt.path))
            .collect::<Vec<_>>(),
        geometry.train_sequence_length,
    )?;
    let total_sequence_count =
        (validation_tokens.len().saturating_sub(1) / geometry.train_sequence_length) as u64;
    let expected_shards = build_validation_observation_plan(
        total_sequence_count,
        geometry.train_sequence_length,
        &shard.eval_mode,
        CHALLENGE_WORLD_SIZE,
    )?;
    let Some(expected_shard) = expected_shards
        .iter()
        .find(|candidate| candidate.rank == shard.rank)
    else {
        return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "validation shard for rank {} is outside the distributed validation plan",
                shard.rank
            ),
        });
    };
    if shard.sequence_start != expected_shard.sequence_start
        || shard.sequence_count != expected_shard.sequence_count
        || shard.evaluation_unit_start != expected_shard.evaluation_unit_start
        || shard.evaluation_unit_count != expected_shard.evaluation_unit_count
        || shard.scored_token_start != expected_shard.scored_token_start
        || shard.scored_token_count != expected_shard.scored_token_count
    {
        return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "validation shard for rank {} did not match the expected {:?} distributed validation plan",
                shard.rank, shard.eval_mode
            ),
        });
    }
    let byte_luts = parameter_golf_sentencepiece_byte_luts_from_tokenizer_path(&tokenizer_path)?;
    let mut cuda_backend = CudaBackend::new();
    let selected_device = cuda_backend.selected_device().cloned().ok_or_else(|| {
        ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "distributed validation child rank {rank} could not select one CUDA device"
            ),
        }
    })?;
    let selected_device_label = selected_device
        .device_name
        .clone()
        .unwrap_or_else(|| String::from("unknown"));
    let validation_batch_sequences =
        distributed_validation_batch_sequences(&geometry, &shard.eval_mode);
    let started = Instant::now();
    let mut graph_cache = BTreeMap::new();
    let validation_summary = if shard.evaluation_unit_count == 0 {
        crate::ParameterGolfSingleH100ValidationSummary {
            eval_mode: shard.eval_mode.clone(),
            evaluated_sequence_count: 0,
            evaluated_token_count: 0,
            evaluated_byte_count: 0,
            mean_loss: 0.0,
            bits_per_byte: 0.0,
            runtime_receipt: None,
        }
    } else {
        match &shard.eval_mode {
            ParameterGolfValidationEvalMode::NonOverlapping => {
                let raw_start = shard.sequence_start as usize * geometry.train_sequence_length;
                let raw_end =
                    raw_start + shard.sequence_count as usize * geometry.train_sequence_length + 1;
                evaluate_validation_on_cuda(
                    &mut cuda_backend,
                    &selected_device.device,
                    current_model.descriptor(),
                    &current_model,
                    &validation_tokens[raw_start..raw_end],
                    &byte_luts,
                    geometry.train_sequence_length,
                    validation_batch_sequences,
                    &shard.eval_mode,
                    &mut graph_cache,
                    &format!("distributed_validation_rank_{rank}"),
                    None,
                )?
            }
            ParameterGolfValidationEvalMode::SlidingWindow { stride } => {
                let total_tokens = validation_tokens.len().saturating_sub(1);
                let window_starts = build_parameter_golf_validation_window_starts(
                    total_tokens,
                    geometry.train_sequence_length,
                    *stride,
                );
                let shard_window_starts = &window_starts[shard.evaluation_unit_start as usize
                    ..(shard.evaluation_unit_start + shard.evaluation_unit_count) as usize];
                evaluate_validation_window_starts_on_cuda(
                    &mut cuda_backend,
                    &selected_device.device,
                    current_model.descriptor(),
                    &current_model,
                    validation_tokens.as_slice(),
                    &byte_luts,
                    geometry.train_sequence_length,
                    validation_batch_sequences,
                    *stride,
                    shard_window_starts,
                    &mut graph_cache,
                    &format!("distributed_validation_rank_{rank}"),
                    None,
                )?
            }
        }
    };
    let observed_wallclock_ms = duration_ms(started);
    let loss_sum = validation_summary.mean_loss * validation_summary.evaluated_token_count as f64;
    let mut receipt = ParameterGolfDistributed8xH100ValidationRankReceipt {
        schema_version: 1,
        run_id: String::from(run_id),
        rank,
        local_rank,
        world_size,
        cuda_visible_devices,
        selected_device_label,
        log_path,
        shard_path: shard_path.display().to_string(),
        aggregated_gradient_artifact_path: aggregated_gradient_artifact_path
            .display()
            .to_string(),
        aggregated_gradient_artifact_sha256,
        current_model_artifact_path: current_model_artifact_path
            .as_ref()
            .map(|path| path.display().to_string()),
        current_model_artifact_sha256,
        eval_mode: shard.eval_mode,
        sequence_start: shard.sequence_start,
        sequence_count: shard.sequence_count,
        evaluation_unit_start: shard.evaluation_unit_start,
        evaluation_unit_count: shard.evaluation_unit_count,
        scored_token_start: shard.scored_token_start,
        scored_token_count: shard.scored_token_count,
        local_batch_sequences: validation_batch_sequences as u64,
        loss_sum,
        token_count: validation_summary.evaluated_token_count,
        byte_count: validation_summary.evaluated_byte_count,
        mean_loss: validation_summary.mean_loss,
        bits_per_byte: validation_summary.bits_per_byte,
        observed_wallclock_ms,
        claim_boundary: String::from(
            "This child receipt proves one rank-local distributed validation shard executed against the reconstructed post-step Parameter Golf model on one explicit H100 binding. It does not yet claim final artifact closure or full record-track completion.",
        ),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.stable_digest();
    if let Some(parent) = receipt_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    fs::write(
        &receipt_path,
        format!("{}\n", serde_json::to_string_pretty(&receipt)?),
    )
    .map_err(
        |error| ParameterGolfDistributed8xH100TrainStepError::Write {
            path: receipt_path.display().to_string(),
            error,
        },
    )?;
    Ok(receipt)
}

fn write_bytes_artifact(
    output_path: &Path,
    bytes: &[u8],
) -> Result<String, ParameterGolfDistributed8xH100TrainStepError> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    fs::write(output_path, bytes).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(sha256_hex(bytes))
}

fn write_gradient_artifact(
    output_path: &Path,
    gradients: &BTreeMap<String, Vec<f32>>,
) -> Result<String, ParameterGolfDistributed8xH100TrainStepError> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    let mut tensors = Vec::with_capacity(gradients.len());
    for (parameter_id, values) in gradients {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        tensors.push((
            parameter_id.clone(),
            SafeTensorsDType::F32,
            vec![values.len()],
            bytes,
        ));
    }
    let mut views = Vec::with_capacity(tensors.len());
    for (name, dtype, shape, bytes) in &tensors {
        let view = TensorView::new(*dtype, shape.clone(), bytes.as_slice()).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "failed to serialize gradient tensor `{name}` into safetensors: {error}"
                ),
            }
        })?;
        views.push((name.clone(), view));
    }
    let bytes = serialize(
        views
            .iter()
            .map(|(name, view)| (name.as_str(), view.clone())),
        None,
    )
    .map_err(
        |error| ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!("failed to serialize distributed gradient artifact: {error}"),
        },
    )?;
    fs::write(output_path, &bytes).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(sha256_hex(&bytes))
}

fn load_gradient_artifact(
    path: &Path,
) -> Result<BTreeMap<String, Vec<f32>>, ParameterGolfDistributed8xH100TrainStepError> {
    let bytes =
        fs::read(path).map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Read {
            path: path.display().to_string(),
            error,
        })?;
    let safetensors = SafeTensors::deserialize(bytes.as_slice()).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "failed to decode distributed gradient artifact `{}`: {error}",
                path.display()
            ),
        }
    })?;
    let mut gradients = BTreeMap::new();
    for name in safetensors.names() {
        let tensor = safetensors.tensor(name).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "failed to read gradient tensor `{name}` from `{}`: {error}",
                    path.display()
                ),
            }
        })?;
        if tensor.dtype() != SafeTensorsDType::F32 {
            return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "distributed gradient artifact `{}` tensor `{name}` expected f32 but found {:?}",
                    path.display(),
                    tensor.dtype()
                ),
            });
        }
        let values = tensor
            .data()
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("fixed 4-byte chunk should convert into one f32"),
                )
            })
            .collect::<Vec<_>>();
        if values.len() != tensor.shape().iter().product::<usize>() {
            return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "distributed gradient artifact `{}` tensor `{name}` length mismatch after f32 decode",
                    path.display()
                ),
            });
        }
        gradients.insert(String::from(name), values);
    }
    Ok(gradients)
}

fn parse_env_usize(
    key: &'static str,
) -> Result<usize, ParameterGolfDistributed8xH100TrainStepError> {
    let value = required_env(key)?;
    value
        .parse::<usize>()
        .map_err(|_| ParameterGolfDistributed8xH100TrainStepError::InvalidEnv { key, value })
}

fn required_env(key: &'static str) -> Result<String, ParameterGolfDistributed8xH100TrainStepError> {
    env::var(key).map_err(|_| ParameterGolfDistributed8xH100TrainStepError::MissingEnv { key })
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let encoded = match serde_json::to_vec(value) {
        Ok(encoded) => encoded,
        Err(error) => error.to_string().into_bytes(),
    };
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn duration_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, env};

    use super::{
        distributed_validation_batch_sequences, load_gradient_artifact, write_gradient_artifact,
        SCOREBOARD_SLIDING_WINDOW_BATCH_SEQUENCES, VALIDATION_BATCH_SEQUENCES_ENV_VAR,
    };
    use crate::{ParameterGolfBatchGeometry, ParameterGolfValidationEvalMode};

    #[test]
    fn distributed_train_step_gradient_artifact_roundtrips(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = tempfile::tempdir()?;
        let artifact_path = temp_dir.path().join("rank_0.safetensors");
        let gradients = BTreeMap::from([
            (String::from("a"), vec![1.0_f32, 2.0, 3.0]),
            (String::from("b"), vec![4.0_f32, 5.0]),
        ]);
        let _digest = write_gradient_artifact(&artifact_path, &gradients)?;
        let restored = load_gradient_artifact(&artifact_path)?;
        assert_eq!(restored, gradients);
        Ok(())
    }

    #[test]
    fn distributed_non_overlapping_validation_uses_training_geometry_by_default() {
        let geometry = ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults();
        assert_eq!(geometry.local_validation_batch_sequences(), 64);
        assert_eq!(
            distributed_validation_batch_sequences(
                &geometry,
                &ParameterGolfValidationEvalMode::NonOverlapping
            ),
            64
        );
    }

    #[test]
    fn distributed_sliding_window_validation_uses_scoreboard_eval_geometry_by_default() {
        let geometry = ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults();
        assert_eq!(
            distributed_validation_batch_sequences(
                &geometry,
                &ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 }
            ),
            SCOREBOARD_SLIDING_WINDOW_BATCH_SEQUENCES
        );
    }

    #[test]
    fn distributed_validation_batch_sequences_honors_explicit_override() {
        let geometry = ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults();
        unsafe {
            env::set_var(VALIDATION_BATCH_SEQUENCES_ENV_VAR, "256");
        }
        assert_eq!(
            distributed_validation_batch_sequences(
                &geometry,
                &ParameterGolfValidationEvalMode::NonOverlapping
            ),
            256
        );
        assert_eq!(
            distributed_validation_batch_sequences(
                &geometry,
                &ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 }
            ),
            256
        );
        unsafe {
            env::remove_var(VALIDATION_BATCH_SEQUENCES_ENV_VAR);
        }
    }
}

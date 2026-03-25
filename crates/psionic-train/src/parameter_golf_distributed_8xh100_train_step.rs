use std::{
    collections::BTreeMap,
    env, fs,
    fs::OpenOptions,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    thread,
    time::{Duration, Instant},
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
    export_parameter_golf_banked_full_precision_weights_bytes,
    export_parameter_golf_int8_zlib_model_artifact, inspect_local_distributed_8xh100_machine,
    materialize_current_banked_weights, materialize_current_model,
    parameter_golf_default_validation_batch_sequences, parameter_golf_optimizer_plan,
    parameter_golf_runpod_8xh100_capability_profile,
    parameter_golf_single_h100_training::{
        refresh_parameter_golf_cuda_training_sessions_from_state, ParameterGolfCudaTrainingSession,
        ParameterGolfSingleH100TrainingRuntimeReceipt,
    },
    restore_parameter_golf_banked_weights_from_safetensors,
    restore_parameter_golf_model_from_safetensors, seed_parameter_states, zero_gradients,
    ParameterGolfBaselineEvalGraph, ParameterGolfBaselineTrainingGraph, ParameterGolfBatchGeometry,
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
const WORKER_CHILD_ENV_VAR: &str = "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_WORKER_CHILD";
const WORKER_CHILD_RANK_ENV_VAR: &str = "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_WORKER_RANK";
const WORKER_CHILD_LOCAL_RANK_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_WORKER_LOCAL_RANK";
const WORKER_CHILD_WORLD_SIZE_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_WORKER_WORLD_SIZE";
const WORKER_CHILD_LOG_PATH_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_WORKER_LOG_PATH";
const WORKER_CHILD_COLLECTIVE_ADDR_ENV_VAR: &str =
    "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_WORKER_COLLECTIVE_ADDR";
const WORKER_CHILD_RUN_ID_ENV_VAR: &str = "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_WORKER_RUN_ID";
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
const WORKER_PROTOCOL_SCHEMA_VERSION: u32 = 1;
const WORKER_COLLECTIVE_CONNECT_RETRY_LIMIT: usize = 200;
const WORKER_COLLECTIVE_CONNECT_RETRY_DELAY_MS: u64 = 100;

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
    /// Explicit runtime surface stored in the retained full-precision artifact.
    pub current_model_artifact_surface: String,
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
    /// Current in-memory gradient synchronization transport.
    #[serde(default)]
    pub gradient_sync_transport: String,
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
    /// Explicit resident train-runtime receipt when the hot path used the device-resident surface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_receipt: Option<ParameterGolfSingleH100TrainingRuntimeReceipt>,
    /// Rank-local wallclock for the executed gradient batch.
    pub observed_wallclock_ms: u64,
    /// In-memory gradient synchronization wallclock.
    #[serde(default)]
    pub gradient_sync_ms: u64,
    /// Local optimizer-step wallclock after mesh synchronization.
    #[serde(default)]
    pub optimizer_step_ms: u64,
    /// Gradient norm after clipping on the synchronized gradient vector.
    #[serde(default)]
    pub gradient_norm_after_clip: f32,
    /// Whether clipping was applied on the synchronized gradient vector.
    #[serde(default)]
    pub clip_applied: bool,
    /// Number of non-finite gradient values seen before clipping.
    #[serde(default)]
    pub non_finite_gradient_count: u64,
    /// Worker PID that owned this resident rank runtime.
    #[serde(default)]
    pub worker_pid: u32,
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
    /// Expected local validation batch size in sequences or windows for this rank.
    #[serde(default)]
    pub local_batch_sequences: u64,
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
    /// Worker PID that owned this resident rank runtime.
    #[serde(default)]
    pub worker_pid: u32,
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
    #[error("parameter golf distributed 8xH100 persistent worker protocol failed for rank {rank}: {message}")]
    WorkerProtocol { rank: usize, message: String },
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

/// Returns whether the current process is one persistent distributed worker child.
#[must_use]
pub fn parameter_golf_distributed_8xh100_worker_child_enabled() -> bool {
    env::var_os(WORKER_CHILD_ENV_VAR).is_some()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ParameterGolfDistributed8xH100WorkerCommand {
    TrainStep {
        schema_version: u32,
        step_index: u64,
        window: ParameterGolfTokenStreamWindow,
        learning_rate_multiplier: f32,
        muon_momentum: f32,
    },
    Validation {
        schema_version: u32,
        shard: ParameterGolfDistributed8xH100ValidationShardPlan,
    },
    ExportArtifacts {
        schema_version: u32,
        executed_step_count: u64,
        current_model_artifact_path: String,
        current_model_int8_zlib_artifact_path: String,
    },
    Shutdown {
        schema_version: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ParameterGolfDistributed8xH100WorkerReadyReceipt {
    rank: usize,
    local_rank: usize,
    world_size: usize,
    selected_device_label: String,
    log_path: String,
    worker_pid: u32,
    collective_transport: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ParameterGolfDistributed8xH100WorkerArtifactExportReceipt {
    rank: usize,
    current_model_artifact_path: String,
    current_model_artifact_sha256: String,
    current_model_int8_zlib_artifact_path: String,
    current_model_int8_zlib_artifact_sha256: String,
    current_model_int8_zlib_artifact_size_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ParameterGolfDistributed8xH100WorkerResponse {
    Ready {
        schema_version: u32,
        receipt: ParameterGolfDistributed8xH100WorkerReadyReceipt,
    },
    TrainStepComplete {
        schema_version: u32,
        receipt: ParameterGolfDistributed8xH100TrainStepRankReceipt,
    },
    ValidationComplete {
        schema_version: u32,
        receipt: ParameterGolfDistributed8xH100ValidationRankReceipt,
    },
    ExportArtifactsComplete {
        schema_version: u32,
        receipt: ParameterGolfDistributed8xH100WorkerArtifactExportReceipt,
    },
    ShutdownComplete {
        schema_version: u32,
        rank: usize,
    },
    Error {
        schema_version: u32,
        rank: usize,
        message: String,
    },
}

struct ParameterGolfDistributed8xH100PersistentWorkerHandle {
    rank: usize,
    local_rank: usize,
    log_path: PathBuf,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    child: Child,
    _ready: ParameterGolfDistributed8xH100WorkerReadyReceipt,
}

enum ParameterGolfDistributed8xH100WorkerCollective {
    RankZero { peers: BTreeMap<usize, TcpStream> },
    Peer { stream: TcpStream },
}

struct ParameterGolfDistributed8xH100WorkerRuntime {
    run_id: String,
    rank: usize,
    local_rank: usize,
    world_size: usize,
    log_path: String,
    bundle: psionic_data::ParameterGolfDatasetBundle,
    baseline_model: ParameterGolfReferenceModel,
    current_model: ParameterGolfReferenceModel,
    current_model_stale: bool,
    trainer_state: ParameterGolfSingleH100TrainerState,
    geometry: ParameterGolfBatchGeometry,
    hyperparameters: ParameterGolfTrainingHyperparameters,
    byte_luts: psionic_data::ParameterGolfSentencePieceByteLuts,
    cuda_backend: CudaBackend,
    selected_device: psionic_core::Device,
    selected_device_label: String,
    training_graph_cache: BTreeMap<usize, ParameterGolfBaselineTrainingGraph>,
    training_session_cache: BTreeMap<usize, ParameterGolfCudaTrainingSession>,
    validation_graph_cache: BTreeMap<usize, ParameterGolfBaselineEvalGraph>,
    collective: ParameterGolfDistributed8xH100WorkerCollective,
}

fn parameter_golf_distributed_8xh100_worker_mesh_logs_dir(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("runtime_worker_mesh_logs"),
        None => root.join("runtime_worker_mesh_logs"),
    }
}

fn worker_response_from_reader(
    rank: usize,
    reader: &mut BufReader<ChildStdout>,
) -> Result<
    ParameterGolfDistributed8xH100WorkerResponse,
    ParameterGolfDistributed8xH100TrainStepError,
> {
    let mut line = String::new();
    let bytes_read = reader.read_line(&mut line).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to read one worker response line: {error}"),
        }
    })?;
    if bytes_read == 0 {
        return Err(
            ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                rank,
                message: String::from(
                    "worker stdout closed before one protocol response was written",
                ),
            },
        );
    }
    serde_json::from_str::<ParameterGolfDistributed8xH100WorkerResponse>(line.trim_end()).map_err(
        |error| ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to decode one worker response line: {error}"),
        },
    )
}

fn write_worker_command(
    rank: usize,
    writer: &mut BufWriter<ChildStdin>,
    command: &ParameterGolfDistributed8xH100WorkerCommand,
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    let encoded = serde_json::to_string(command)
        .map_err(ParameterGolfDistributed8xH100TrainStepError::Json)?;
    writer.write_all(encoded.as_bytes()).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to write worker command bytes: {error}"),
        }
    })?;
    writer.write_all(b"\n").map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to terminate worker command line: {error}"),
        }
    })?;
    writer.flush().map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to flush worker command bytes: {error}"),
        }
    })?;
    Ok(())
}

fn choose_loopback_collective_addr() -> Result<String, ParameterGolfDistributed8xH100TrainStepError>
{
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Write {
            path: String::from("127.0.0.1:0"),
            error,
        }
    })?;
    let address = listener.local_addr().map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Write {
            path: String::from("127.0.0.1:0"),
            error,
        }
    })?;
    Ok(address.to_string())
}

fn set_loopback_nodelay(
    stream: &TcpStream,
    rank: usize,
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    stream.set_nodelay(true).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to enable TCP_NODELAY: {error}"),
        }
    })
}

fn write_u64(
    stream: &mut TcpStream,
    value: u64,
    rank: usize,
    detail: &str,
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    stream.write_all(&value.to_le_bytes()).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to write {detail}: {error}"),
        }
    })
}

fn read_u64(
    stream: &mut TcpStream,
    rank: usize,
    detail: &str,
) -> Result<u64, ParameterGolfDistributed8xH100TrainStepError> {
    let mut bytes = [0_u8; 8];
    stream.read_exact(&mut bytes).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to read {detail}: {error}"),
        }
    })?;
    Ok(u64::from_le_bytes(bytes))
}

fn f32_values_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_values_from_bytes(
    bytes: &[u8],
    rank: usize,
) -> Result<Vec<f32>, ParameterGolfDistributed8xH100TrainStepError> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(
            ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                rank,
                message: format!(
                    "received {} collective bytes, which is not divisible by {}",
                    bytes.len(),
                    std::mem::size_of::<f32>()
                ),
            },
        );
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn write_collective_gradient_payload(
    stream: &mut TcpStream,
    step_index: u64,
    values: &[f32],
    rank: usize,
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    let bytes = f32_values_to_bytes(values);
    write_u64(stream, step_index, rank, "collective step_index")?;
    write_u64(
        stream,
        values.len() as u64,
        rank,
        "collective flattened gradient length",
    )?;
    write_u64(
        stream,
        bytes.len() as u64,
        rank,
        "collective flattened gradient byte length",
    )?;
    stream.write_all(bytes.as_slice()).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to write collective gradient payload: {error}"),
        }
    })?;
    stream.flush().map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to flush collective gradient payload: {error}"),
        }
    })?;
    Ok(())
}

fn read_collective_gradient_payload(
    stream: &mut TcpStream,
    rank: usize,
) -> Result<(u64, Vec<f32>), ParameterGolfDistributed8xH100TrainStepError> {
    let step_index = read_u64(stream, rank, "collective step_index")?;
    let value_count = read_u64(stream, rank, "collective flattened gradient length")? as usize;
    let byte_len = read_u64(stream, rank, "collective flattened gradient byte length")? as usize;
    let mut bytes = vec![0_u8; byte_len];
    stream.read_exact(bytes.as_mut_slice()).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!("failed to read collective gradient payload bytes: {error}"),
        }
    })?;
    let values = f32_values_from_bytes(bytes.as_slice(), rank)?;
    if values.len() != value_count {
        return Err(ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!(
                "collective payload length mismatch: header declared {value_count} f32 values but decoded {}",
                values.len()
            ),
        });
    }
    Ok((step_index, values))
}

fn flatten_gradients_for_worker_mesh(
    trainer_state: &ParameterGolfSingleH100TrainerState,
    gradients: &BTreeMap<String, Vec<f32>>,
) -> Result<Vec<f32>, ParameterGolfDistributed8xH100TrainStepError> {
    let mut flattened = Vec::with_capacity(
        trainer_state
            .parameter_states
            .values()
            .map(|state| state.values().len())
            .sum(),
    );
    for (parameter_id, state) in &trainer_state.parameter_states {
        let gradient = gradients.get(parameter_id).ok_or_else(|| {
            ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "missing flattened gradient values for parameter `{parameter_id}`"
                ),
            }
        })?;
        if gradient.len() != state.values().len() {
            return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "flattened gradient length mismatch for `{parameter_id}`: expected {}, found {}",
                    state.values().len(),
                    gradient.len()
                ),
            });
        }
        flattened.extend_from_slice(gradient);
    }
    Ok(flattened)
}

fn unflatten_gradients_from_worker_mesh(
    trainer_state: &ParameterGolfSingleH100TrainerState,
    flattened: &[f32],
) -> Result<Vec<(String, Vec<f32>)>, ParameterGolfDistributed8xH100TrainStepError> {
    let expected_len: usize = trainer_state
        .parameter_states
        .values()
        .map(|state| state.values().len())
        .sum();
    if flattened.len() != expected_len {
        return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!(
                "flattened worker-mesh gradient length mismatch: expected {expected_len}, found {}",
                flattened.len()
            ),
        });
    }
    let mut cursor = 0_usize;
    let mut gradients = Vec::with_capacity(trainer_state.parameter_states.len());
    for (parameter_id, state) in &trainer_state.parameter_states {
        let len = state.values().len();
        gradients.push((
            parameter_id.clone(),
            flattened[cursor..cursor + len].to_vec(),
        ));
        cursor = cursor.saturating_add(len);
    }
    Ok(gradients)
}

fn connect_worker_collective(
    rank: usize,
    collective_addr: &str,
) -> Result<
    ParameterGolfDistributed8xH100WorkerCollective,
    ParameterGolfDistributed8xH100TrainStepError,
> {
    if rank == 0 {
        let listener = TcpListener::bind(collective_addr).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                rank,
                message: format!(
                    "failed to bind rank-0 collective listener on `{collective_addr}`: {error}"
                ),
            }
        })?;
        let mut peers = BTreeMap::new();
        while peers.len() + 1 < CHALLENGE_WORLD_SIZE {
            let (mut stream, _) = listener.accept().map_err(|error| {
                ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                    rank,
                    message: format!("failed to accept one collective peer: {error}"),
                }
            })?;
            set_loopback_nodelay(&stream, rank)?;
            let peer_rank = read_u64(&mut stream, rank, "collective peer rank handshake")? as usize;
            peers.insert(peer_rank, stream);
        }
        Ok(ParameterGolfDistributed8xH100WorkerCollective::RankZero { peers })
    } else {
        for _attempt in 0..WORKER_COLLECTIVE_CONNECT_RETRY_LIMIT {
            match TcpStream::connect(collective_addr) {
                Ok(mut stream) => {
                    set_loopback_nodelay(&stream, rank)?;
                    write_u64(&mut stream, rank as u64, rank, "collective rank handshake")?;
                    return Ok(ParameterGolfDistributed8xH100WorkerCollective::Peer { stream });
                }
                Err(_) => thread::sleep(Duration::from_millis(
                    WORKER_COLLECTIVE_CONNECT_RETRY_DELAY_MS,
                )),
            }
        }
        Err(ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
            rank,
            message: format!(
                "failed to connect to rank-0 collective listener `{collective_addr}` after {} retries",
                WORKER_COLLECTIVE_CONNECT_RETRY_LIMIT
            ),
        })
    }
}

fn all_reduce_mean_flat_gradients(
    collective: &mut ParameterGolfDistributed8xH100WorkerCollective,
    rank: usize,
    step_index: u64,
    local_gradients: &[f32],
) -> Result<Vec<f32>, ParameterGolfDistributed8xH100TrainStepError> {
    match collective {
        ParameterGolfDistributed8xH100WorkerCollective::RankZero { peers } => {
            let mut reduced = local_gradients.to_vec();
            for (peer_rank, stream) in peers.iter_mut() {
                let (peer_step_index, peer_values) =
                    read_collective_gradient_payload(stream, rank)?;
                if peer_step_index != step_index {
                    return Err(
                        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                            rank,
                            message: format!(
                                "peer rank {} sent step_index {} while rank 0 expected {}",
                                peer_rank, peer_step_index, step_index
                            ),
                        },
                    );
                }
                if peer_values.len() != reduced.len() {
                    return Err(
                        ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                            rank,
                            message: format!(
                                "peer rank {} sent {} flattened gradients but rank 0 expected {}",
                                peer_rank,
                                peer_values.len(),
                                reduced.len()
                            ),
                        },
                    );
                }
                for (reduced_value, peer_value) in reduced.iter_mut().zip(peer_values.iter()) {
                    *reduced_value += *peer_value;
                }
            }
            for value in &mut reduced {
                *value /= CHALLENGE_WORLD_SIZE as f32;
            }
            for stream in peers.values_mut() {
                write_collective_gradient_payload(stream, step_index, reduced.as_slice(), rank)?;
            }
            Ok(reduced)
        }
        ParameterGolfDistributed8xH100WorkerCollective::Peer { stream } => {
            write_collective_gradient_payload(stream, step_index, local_gradients, rank)?;
            let (received_step_index, reduced) = read_collective_gradient_payload(stream, rank)?;
            if received_step_index != step_index {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                        rank,
                        message: format!(
                            "rank {} received averaged gradient step_index {} while expecting {}",
                            rank, received_step_index, step_index
                        ),
                    },
                );
            }
            Ok(reduced)
        }
    }
}

impl ParameterGolfDistributed8xH100WorkerRuntime {
    fn ensure_current_model_materialized(
        &mut self,
    ) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
        if self.current_model_stale {
            self.current_model =
                materialize_current_model(&self.baseline_model, &self.trainer_state)?;
            self.current_model_stale = false;
        }
        Ok(())
    }

    fn new(run_id: &str) -> Result<Self, ParameterGolfDistributed8xH100TrainStepError> {
        let rank = parse_env_usize(WORKER_CHILD_RANK_ENV_VAR)?;
        let local_rank = parse_env_usize(WORKER_CHILD_LOCAL_RANK_ENV_VAR)?;
        let world_size = parse_env_usize(WORKER_CHILD_WORLD_SIZE_ENV_VAR)?;
        let log_path = required_env(WORKER_CHILD_LOG_PATH_ENV_VAR)?;
        let collective_addr = required_env(WORKER_CHILD_COLLECTIVE_ADDR_ENV_VAR)?;
        if world_size != CHALLENGE_WORLD_SIZE {
            return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "persistent worker rank {rank} requires world_size={} but observed {}",
                    CHALLENGE_WORLD_SIZE, world_size
                ),
            });
        }
        if local_rank != rank {
            return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "persistent worker rank {rank} requires local_rank == rank on the single pod, found local_rank={local_rank}"
                ),
            });
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
        let hyperparameters = ParameterGolfTrainingHyperparameters::baseline_defaults();
        let optimizer_plan =
            parameter_golf_optimizer_plan(&baseline_model.banked_descriptor()?, &hyperparameters)?;
        let trainer_state = seed_parameter_states(&baseline_model, &optimizer_plan)?;
        let current_model = materialize_current_model(&baseline_model, &trainer_state)?;
        let byte_luts =
            parameter_golf_sentencepiece_byte_luts_from_tokenizer_path(&tokenizer_path)?;
        let cuda_backend = CudaBackend::new();
        let selected_device = cuda_backend.selected_device().cloned().ok_or_else(|| {
            ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!("persistent worker rank {rank} could not select one CUDA device"),
            }
        })?;
        let selected_device_label = selected_device
            .device_name
            .clone()
            .unwrap_or_else(|| String::from("unknown"));
        let collective = connect_worker_collective(rank, &collective_addr)?;
        Ok(Self {
            run_id: String::from(run_id),
            rank,
            local_rank,
            world_size,
            log_path,
            bundle,
            baseline_model,
            current_model,
            current_model_stale: false,
            trainer_state,
            geometry: ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults(),
            hyperparameters,
            byte_luts,
            cuda_backend,
            selected_device: selected_device.device,
            selected_device_label,
            training_graph_cache: BTreeMap::new(),
            training_session_cache: BTreeMap::new(),
            validation_graph_cache: BTreeMap::new(),
            collective,
        })
    }

    fn ready_receipt(&self) -> ParameterGolfDistributed8xH100WorkerReadyReceipt {
        ParameterGolfDistributed8xH100WorkerReadyReceipt {
            rank: self.rank,
            local_rank: self.local_rank,
            world_size: self.world_size,
            selected_device_label: self.selected_device_label.clone(),
            log_path: self.log_path.clone(),
            worker_pid: std::process::id(),
            collective_transport: String::from("rank0_loopback_tcp_mean_all_reduce_v1"),
        }
    }

    fn execute_train_step(
        &mut self,
        step_index: u64,
        window: &ParameterGolfTokenStreamWindow,
        learning_rate_multiplier: f32,
        muon_momentum: f32,
    ) -> Result<
        ParameterGolfDistributed8xH100TrainStepRankReceipt,
        ParameterGolfDistributed8xH100TrainStepError,
    > {
        let step_started = Instant::now();
        let current_banked_weights = if self.training_session_cache.is_empty() {
            self.ensure_current_model_materialized()?;
            Some(self.current_model.banked_weights()?)
        } else {
            None
        };
        let gradient_batch = execute_parameter_golf_training_gradient_batch(
            &mut self.cuda_backend,
            &self.selected_device,
            &self.bundle,
            &self.current_model,
            current_banked_weights.as_ref(),
            &mut self.training_graph_cache,
            Some(&mut self.training_session_cache),
            &self.geometry,
            window,
        )?;
        let flattened = flatten_gradients_for_worker_mesh(
            &self.trainer_state,
            &gradient_batch.parameter_gradients,
        )?;
        let gradient_sync_started = Instant::now();
        let averaged_flattened = all_reduce_mean_flat_gradients(
            &mut self.collective,
            self.rank,
            step_index,
            flattened.as_slice(),
        )?;
        let gradient_sync_ms = duration_ms(gradient_sync_started);
        let mut averaged_gradients = unflatten_gradients_from_worker_mesh(
            &self.trainer_state,
            averaged_flattened.as_slice(),
        )?;
        let clip_observation = clip_gradients(
            averaged_gradients.as_mut_slice(),
            self.hyperparameters.grad_clip_norm,
        );
        let optimizer_started = Instant::now();
        apply_gradients_to_state(
            &mut self.trainer_state,
            averaged_gradients.as_slice(),
            learning_rate_multiplier,
            muon_momentum,
            step_index,
        )?;
        refresh_parameter_golf_cuda_training_sessions_from_state(
            &mut self.training_session_cache,
            &self.trainer_state,
        )?;
        self.current_model_stale = true;
        let optimizer_step_ms = duration_ms(optimizer_started);
        let averaged_gradient_sha256 =
            sha256_hex(&f32_values_to_bytes(averaged_flattened.as_slice()));
        Ok(ParameterGolfDistributed8xH100TrainStepRankReceipt {
            schema_version: 1,
            run_id: self.run_id.clone(),
            rank: self.rank,
            local_rank: self.local_rank,
            world_size: self.world_size,
            cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
            selected_device_label: self.selected_device_label.clone(),
            log_path: self.log_path.clone(),
            window_path: String::from("inline://persistent_worker_mesh/train_step_window"),
            window_id: gradient_batch.window_id,
            gradient_artifact_path: format!(
                "in_memory://mesh.parameter_golf.runpod_8xh100/rank_{}/step_{step_index}/averaged_gradient",
                self.rank
            ),
            gradient_artifact_sha256: averaged_gradient_sha256,
            gradient_sync_transport: String::from("rank0_loopback_tcp_mean_all_reduce_v1"),
            input_model_artifact_path: None,
            input_model_artifact_sha256: None,
            loss: gradient_batch.loss,
            phase_timings: gradient_batch.phase_timings,
            runtime_receipt: gradient_batch.runtime.as_ref().map(|runtime| {
                ParameterGolfSingleH100TrainingRuntimeReceipt {
                    path: String::from("device_resident_cuda_training_graph_v1"),
                    graph_surface: String::from("parameter_golf_baseline_training_graph_v2"),
                    session_count: self.training_session_cache.len(),
                    persistent_parameter_buffer_count: runtime.persistent_parameter_buffer_count,
                    persistent_parameter_value_count: runtime.persistent_parameter_value_count,
                    resident_parameter_upload_us: runtime.resident_parameter_upload_us,
                    parameter_refresh_us: runtime.parameter_refresh_us,
                    reusable_input_token_buffer: true,
                    reusable_target_token_buffer: true,
                    total_input_token_write_us: runtime.input_token_write_us,
                    total_target_token_write_us: runtime.target_token_write_us,
                    resident_parameter_buffers_reused: runtime.resident_parameter_buffers_reused,
                }
            }),
            observed_wallclock_ms: duration_ms(step_started),
            gradient_sync_ms,
            optimizer_step_ms,
            gradient_norm_after_clip: clip_observation
                .gradient_norm_after_clip
                .unwrap_or_default(),
            clip_applied: clip_observation.clip_applied,
            non_finite_gradient_count: u64::from(clip_observation.non_finite_count),
            worker_pid: std::process::id(),
            claim_boundary: String::from(
                "This receipt proves one resident distributed worker executed one rank-local gradient batch, participated in one in-memory loopback mean all-reduce, and applied the synchronized optimizer step without per-step gradient file export.",
            ),
            receipt_digest: String::new(),
        }
        .with_digest())
    }

    fn execute_validation(
        &mut self,
        shard: &ParameterGolfDistributed8xH100ValidationShardPlan,
    ) -> Result<
        ParameterGolfDistributed8xH100ValidationRankReceipt,
        ParameterGolfDistributed8xH100TrainStepError,
    > {
        self.ensure_current_model_materialized()?;
        let validation_tokens = load_parameter_golf_validation_tokens_from_paths(
            &self
                .bundle
                .validation_shards
                .iter()
                .map(|receipt| PathBuf::from(&receipt.path))
                .collect::<Vec<_>>(),
            self.geometry.train_sequence_length,
        )?;
        let total_sequence_count = (validation_tokens.len().saturating_sub(1)
            / self.geometry.train_sequence_length) as u64;
        let expected_shards = build_validation_observation_plan(
            total_sequence_count,
            self.geometry.train_sequence_length,
            &shard.eval_mode,
            CHALLENGE_WORLD_SIZE,
        )?;
        let Some(expected_shard) = expected_shards
            .iter()
            .find(|candidate| candidate.rank == shard.rank)
        else {
            return Err(ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "validation shard for rank {} is outside the resident worker validation plan",
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
                    "resident validation shard for rank {} did not match the expected distributed validation plan",
                    shard.rank
                ),
            });
        }
        let validation_batch_sequences = distributed_validation_batch_sequences(
            shard.local_batch_sequences as usize,
            &self.geometry,
            &shard.eval_mode,
        );
        let started = Instant::now();
        let validation_summary = match &shard.eval_mode {
            ParameterGolfValidationEvalMode::NonOverlapping => {
                let raw_start = shard.sequence_start as usize * self.geometry.train_sequence_length;
                let raw_end = raw_start
                    + shard.sequence_count as usize * self.geometry.train_sequence_length
                    + 1;
                evaluate_validation_on_cuda(
                    &mut self.cuda_backend,
                    &self.selected_device,
                    self.current_model.descriptor(),
                    &self.current_model,
                    &validation_tokens[raw_start..raw_end],
                    &self.byte_luts,
                    self.geometry.train_sequence_length,
                    validation_batch_sequences,
                    &shard.eval_mode,
                    &mut self.validation_graph_cache,
                    &format!("distributed_validation_rank_{}", self.rank),
                    None,
                )?
            }
            ParameterGolfValidationEvalMode::SlidingWindow { stride } => {
                let total_tokens = validation_tokens.len().saturating_sub(1);
                let window_starts = build_parameter_golf_validation_window_starts(
                    total_tokens,
                    self.geometry.train_sequence_length,
                    *stride,
                );
                let shard_window_starts = &window_starts[shard.evaluation_unit_start as usize
                    ..(shard.evaluation_unit_start + shard.evaluation_unit_count) as usize];
                evaluate_validation_window_starts_on_cuda(
                    &mut self.cuda_backend,
                    &self.selected_device,
                    self.current_model.descriptor(),
                    &self.current_model,
                    validation_tokens.as_slice(),
                    &self.byte_luts,
                    self.geometry.train_sequence_length,
                    validation_batch_sequences,
                    *stride,
                    shard_window_starts,
                    &mut self.validation_graph_cache,
                    &format!("distributed_validation_rank_{}", self.rank),
                    None,
                )?
            }
        };
        let current_model_sha256 = sha256_hex(
            export_parameter_golf_banked_full_precision_weights_bytes(
                &self.baseline_model,
                &materialize_current_banked_weights(&self.baseline_model, &self.trainer_state)?,
            )?
            .as_slice(),
        );
        Ok(ParameterGolfDistributed8xH100ValidationRankReceipt {
            schema_version: 1,
            run_id: self.run_id.clone(),
            rank: self.rank,
            local_rank: self.local_rank,
            world_size: self.world_size,
            cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
            selected_device_label: self.selected_device_label.clone(),
            log_path: self.log_path.clone(),
            shard_path: String::from("inline://persistent_worker_mesh/validation_shard"),
            aggregated_gradient_artifact_path: format!(
                "in_memory://mesh.parameter_golf.runpod_8xh100/rank_{}/resident_state",
                self.rank
            ),
            aggregated_gradient_artifact_sha256: current_model_sha256.clone(),
            current_model_artifact_path: None,
            current_model_artifact_sha256: Some(current_model_sha256),
            eval_mode: shard.eval_mode.clone(),
            sequence_start: shard.sequence_start,
            sequence_count: shard.sequence_count,
            evaluation_unit_start: shard.evaluation_unit_start,
            evaluation_unit_count: shard.evaluation_unit_count,
            scored_token_start: shard.scored_token_start,
            scored_token_count: shard.scored_token_count,
            local_batch_sequences: validation_batch_sequences as u64,
            loss_sum: validation_summary.mean_loss * validation_summary.evaluated_token_count as f64,
            token_count: validation_summary.evaluated_token_count,
            byte_count: validation_summary.evaluated_byte_count,
            mean_loss: validation_summary.mean_loss,
            bits_per_byte: validation_summary.bits_per_byte,
            observed_wallclock_ms: duration_ms(started),
            worker_pid: std::process::id(),
            claim_boundary: String::from(
                "This receipt proves one resident distributed worker evaluated its assigned validation shard against the synchronized resident model state without respawning a new runtime.",
            ),
            receipt_digest: String::new(),
        }
        .with_digest())
    }

    fn export_artifacts(
        &mut self,
        executed_step_count: u64,
        current_model_artifact_path: &Path,
        current_model_int8_zlib_artifact_path: &Path,
    ) -> Result<
        ParameterGolfDistributed8xH100WorkerArtifactExportReceipt,
        ParameterGolfDistributed8xH100TrainStepError,
    > {
        self.ensure_current_model_materialized()?;
        let banked_weights =
            materialize_current_banked_weights(&self.baseline_model, &self.trainer_state)?;
        let current_model_sha256 = write_bytes_artifact(
            current_model_artifact_path,
            &export_parameter_golf_banked_full_precision_weights_bytes(
                &self.baseline_model,
                &banked_weights,
            )?,
        )?;
        let int8_artifact = export_parameter_golf_int8_zlib_model_artifact(
            &self.current_model,
            &self.run_id,
            executed_step_count,
        )?;
        let current_model_int8_zlib_artifact_sha256 = write_bytes_artifact(
            current_model_int8_zlib_artifact_path,
            int8_artifact.bytes.as_slice(),
        )?;
        Ok(ParameterGolfDistributed8xH100WorkerArtifactExportReceipt {
            rank: self.rank,
            current_model_artifact_path: current_model_artifact_path.display().to_string(),
            current_model_artifact_sha256: current_model_sha256,
            current_model_int8_zlib_artifact_path: current_model_int8_zlib_artifact_path
                .display()
                .to_string(),
            current_model_int8_zlib_artifact_sha256,
            current_model_int8_zlib_artifact_size_bytes: int8_artifact.bytes.len() as u64,
        })
    }
}

trait ParameterGolfDistributedWithDigest: Sized {
    fn with_digest(self) -> Self;
}

impl ParameterGolfDistributedWithDigest for ParameterGolfDistributed8xH100TrainStepRankReceipt {
    fn with_digest(mut self) -> Self {
        self.receipt_digest = self.stable_digest();
        self
    }
}

impl ParameterGolfDistributedWithDigest for ParameterGolfDistributed8xH100ValidationRankReceipt {
    fn with_digest(mut self) -> Self {
        self.receipt_digest = self.stable_digest();
        self
    }
}

fn write_worker_response_to_stdout(
    response: &ParameterGolfDistributed8xH100WorkerResponse,
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, response)
        .map_err(ParameterGolfDistributed8xH100TrainStepError::Json)?;
    stdout.write_all(b"\n").map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Write {
            path: String::from("stdout"),
            error,
        }
    })?;
    stdout.flush().map_err(
        |error| ParameterGolfDistributed8xH100TrainStepError::Write {
            path: String::from("stdout"),
            error,
        },
    )?;
    Ok(())
}

/// Executes one persistent distributed worker child owned by the shipped runtime.
pub fn execute_parameter_golf_distributed_8xh100_worker_child(
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    let run_id = required_env(WORKER_CHILD_RUN_ID_ENV_VAR)?;
    let mut runtime = ParameterGolfDistributed8xH100WorkerRuntime::new(&run_id)?;
    write_worker_response_to_stdout(&ParameterGolfDistributed8xH100WorkerResponse::Ready {
        schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
        receipt: runtime.ready_receipt(),
    })?;
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                rank: runtime.rank,
                message: format!("failed to read worker command line: {error}"),
            }
        })?;
        if bytes_read == 0 {
            return Ok(());
        }
        let command =
            serde_json::from_str::<ParameterGolfDistributed8xH100WorkerCommand>(line.trim_end())
                .map_err(ParameterGolfDistributed8xH100TrainStepError::Json)?;
        let response = match command {
            ParameterGolfDistributed8xH100WorkerCommand::TrainStep {
                step_index,
                window,
                learning_rate_multiplier,
                muon_momentum,
                ..
            } => ParameterGolfDistributed8xH100WorkerResponse::TrainStepComplete {
                schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
                receipt: runtime.execute_train_step(
                    step_index,
                    &window,
                    learning_rate_multiplier,
                    muon_momentum,
                )?,
            },
            ParameterGolfDistributed8xH100WorkerCommand::Validation { shard, .. } => {
                ParameterGolfDistributed8xH100WorkerResponse::ValidationComplete {
                    schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
                    receipt: runtime.execute_validation(&shard)?,
                }
            }
            ParameterGolfDistributed8xH100WorkerCommand::ExportArtifacts {
                executed_step_count,
                current_model_artifact_path,
                current_model_int8_zlib_artifact_path,
                ..
            } => ParameterGolfDistributed8xH100WorkerResponse::ExportArtifactsComplete {
                schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
                receipt: runtime.export_artifacts(
                    executed_step_count,
                    Path::new(&current_model_artifact_path),
                    Path::new(&current_model_int8_zlib_artifact_path),
                )?,
            },
            ParameterGolfDistributed8xH100WorkerCommand::Shutdown { .. } => {
                write_worker_response_to_stdout(
                    &ParameterGolfDistributed8xH100WorkerResponse::ShutdownComplete {
                        schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
                        rank: runtime.rank,
                    },
                )?;
                return Ok(());
            }
        };
        write_worker_response_to_stdout(&response)?;
    }
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
    configured_batch_sequences: usize,
    geometry: &ParameterGolfBatchGeometry,
    eval_mode: &ParameterGolfValidationEvalMode,
) -> usize {
    let configured_batch_sequences = if configured_batch_sequences == 0 {
        parameter_golf_default_validation_batch_sequences(geometry, eval_mode)
    } else {
        configured_batch_sequences
    };
    parse_validation_batch_sequences_override().unwrap_or(configured_batch_sequences)
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

fn remove_path_recursively(
    path: &Path,
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        fs::remove_dir_all(path).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: path.display().to_string(),
                error,
            }
        })?;
    } else {
        fs::remove_file(path).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: path.display().to_string(),
                error,
            }
        })?;
    }
    Ok(())
}

fn prune_step_scope_for_next_step(
    step_dir: &Path,
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    if !step_dir.is_dir() {
        return Ok(());
    }
    let retained_model_path = step_dir.join("current_model.runtime_surface.safetensors");
    for entry in fs::read_dir(step_dir).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Read {
            path: step_dir.display().to_string(),
            error,
        }
    })? {
        let entry = entry.map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Read {
            path: step_dir.display().to_string(),
            error,
        })?;
        let path = entry.path();
        if path == retained_model_path {
            continue;
        }
        remove_path_recursively(&path)?;
    }
    Ok(())
}

fn emit_distributed_progress_line(message: impl AsRef<str>) {
    println!("{}", message.as_ref());
}

fn spawn_parameter_golf_distributed_8xh100_persistent_worker_mesh(
    root: &Path,
    manifest_path: &Path,
    run_id: &str,
    current_exe: &Path,
    logs_dir: &Path,
    dataset_root: &Path,
    tokenizer_path: &Path,
) -> Result<
    Vec<ParameterGolfDistributed8xH100PersistentWorkerHandle>,
    ParameterGolfDistributed8xH100TrainStepError,
> {
    fs::create_dir_all(logs_dir).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Write {
            path: logs_dir.display().to_string(),
            error,
        }
    })?;
    let collective_addr = choose_loopback_collective_addr()?;
    let mut pending = Vec::with_capacity(CHALLENGE_WORLD_SIZE);
    for rank in 0..CHALLENGE_WORLD_SIZE {
        let log_path = logs_dir.join(format!("rank_{rank}.log"));
        let stderr = OpenOptions::new()
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
        let mut child = Command::new(current_exe)
            .arg(manifest_path)
            .current_dir(root)
            .env(
                crate::PARAMETER_GOLF_EXECUTION_MODE_ENV_VAR,
                crate::PARAMETER_GOLF_DISTRIBUTED_8XH100_EXECUTION_MODE,
            )
            .env(WORKER_CHILD_ENV_VAR, "1")
            .env(WORKER_CHILD_RANK_ENV_VAR, rank.to_string())
            .env(WORKER_CHILD_LOCAL_RANK_ENV_VAR, rank.to_string())
            .env(
                WORKER_CHILD_WORLD_SIZE_ENV_VAR,
                CHALLENGE_WORLD_SIZE.to_string(),
            )
            .env(WORKER_CHILD_LOG_PATH_ENV_VAR, &log_path)
            .env(WORKER_CHILD_COLLECTIVE_ADDR_ENV_VAR, &collective_addr)
            .env(WORKER_CHILD_RUN_ID_ENV_VAR, run_id)
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
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(
                |error| ParameterGolfDistributed8xH100TrainStepError::ChildSpawn { rank, error },
            )?;
        let stdin = child.stdin.take().ok_or_else(|| {
            ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                rank,
                message: String::from("persistent worker stdin pipe was not available"),
            }
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                rank,
                message: String::from("persistent worker stdout pipe was not available"),
            }
        })?;
        pending.push((
            rank,
            log_path,
            BufWriter::new(stdin),
            BufReader::new(stdout),
            child,
        ));
    }

    let mut workers = Vec::with_capacity(CHALLENGE_WORLD_SIZE);
    for (rank, log_path, stdin, mut stdout, child) in pending {
        let response = worker_response_from_reader(rank, &mut stdout)?;
        let ready = match response {
            ParameterGolfDistributed8xH100WorkerResponse::Ready { receipt, .. } => receipt,
            ParameterGolfDistributed8xH100WorkerResponse::Error { message, .. } => {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol { rank, message },
                );
            }
            other => {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                        rank,
                        message: format!("expected one ready response, found {other:?}"),
                    },
                );
            }
        };
        workers.push(ParameterGolfDistributed8xH100PersistentWorkerHandle {
            rank,
            local_rank: rank,
            log_path,
            stdin,
            stdout,
            child,
            _ready: ready,
        });
    }
    workers.sort_by_key(|worker| worker.rank);
    emit_distributed_progress_line(format!(
        "persistent_worker_mesh_ready run_id={} world_size={} transport=rank0_loopback_tcp_mean_all_reduce_v1",
        run_id,
        workers.len()
    ));
    Ok(workers)
}

fn shutdown_parameter_golf_distributed_8xh100_persistent_worker_mesh(
    workers: &mut [ParameterGolfDistributed8xH100PersistentWorkerHandle],
) -> Result<(), ParameterGolfDistributed8xH100TrainStepError> {
    for worker in workers.iter_mut() {
        write_worker_command(
            worker.rank,
            &mut worker.stdin,
            &ParameterGolfDistributed8xH100WorkerCommand::Shutdown {
                schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
            },
        )?;
    }
    for worker in workers.iter_mut() {
        match worker_response_from_reader(worker.rank, &mut worker.stdout)? {
            ParameterGolfDistributed8xH100WorkerResponse::ShutdownComplete { rank, .. }
                if rank == worker.rank => {}
            ParameterGolfDistributed8xH100WorkerResponse::Error { message, .. } => {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                        rank: worker.rank,
                        message,
                    },
                );
            }
            other => {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                        rank: worker.rank,
                        message: format!("expected shutdown response, found {other:?}"),
                    },
                );
            }
        }
        let status = worker.child.wait().map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::ChildWait {
                rank: worker.rank,
                error,
            }
        })?;
        if status.code() != Some(0) {
            return Err(
                ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                    rank: worker.rank,
                    message: format!(
                        "persistent worker exited with {:?}; see {}",
                        status.code(),
                        worker.log_path.display()
                    ),
                },
            );
        }
    }
    Ok(())
}

fn export_final_worker_artifacts(
    workers: &mut [ParameterGolfDistributed8xH100PersistentWorkerHandle],
    executed_step_count: u64,
    current_model_artifact_path: &Path,
    current_model_int8_zlib_artifact_path: &Path,
) -> Result<
    ParameterGolfDistributed8xH100WorkerArtifactExportReceipt,
    ParameterGolfDistributed8xH100TrainStepError,
> {
    let rank_zero = workers.first_mut().ok_or_else(|| {
        ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: String::from("persistent worker mesh was empty during artifact export"),
        }
    })?;
    write_worker_command(
        rank_zero.rank,
        &mut rank_zero.stdin,
        &ParameterGolfDistributed8xH100WorkerCommand::ExportArtifacts {
            schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
            executed_step_count,
            current_model_artifact_path: current_model_artifact_path.display().to_string(),
            current_model_int8_zlib_artifact_path: current_model_int8_zlib_artifact_path
                .display()
                .to_string(),
        },
    )?;
    match worker_response_from_reader(rank_zero.rank, &mut rank_zero.stdout)? {
        ParameterGolfDistributed8xH100WorkerResponse::ExportArtifactsComplete {
            receipt, ..
        } => Ok(receipt),
        ParameterGolfDistributed8xH100WorkerResponse::Error { message, .. } => Err(
            ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                rank: rank_zero.rank,
                message,
            },
        ),
        other => Err(
            ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                rank: rank_zero.rank,
                message: format!("expected export-artifacts response, found {other:?}"),
            },
        ),
    }
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
    current_model_artifact_surface: String,
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
        let child = command.spawn().map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::ChildSpawn { rank, error }
        })?;
        children.push((
            rank,
            window_path,
            gradient_artifact_path,
            receipt_path,
            log_path,
            child,
        ));
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
    for (_rank, gradients) in load_rank_gradient_artifacts_parallel(&rank_launches)? {
        crate::accumulate_gradients(
            accumulated_gradients.as_mut_slice(),
            trainer_state,
            &gradients,
            CHALLENGE_WORLD_SIZE as f32,
        )?;
    }
    let gradient_sync_ms = duration_ms(sync_started);

    let clip_observation = clip_gradients(
        accumulated_gradients.as_mut_slice(),
        hyperparameters.grad_clip_norm,
    );
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
    let current_banked_weights = materialize_current_banked_weights(initial_model, trainer_state)?;
    let current_model_artifact_path = step_dir.join("current_model.runtime_surface.safetensors");
    let current_model_artifact_sha256 = write_bytes_artifact(
        &current_model_artifact_path,
        &export_parameter_golf_banked_full_precision_weights_bytes(
            initial_model,
            &current_banked_weights,
        )?,
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
        gradient_norm_after_clip: clip_observation
            .gradient_norm_after_clip
            .unwrap_or_default(),
        clip_applied: clip_observation.clip_applied,
        non_finite_gradient_count: u64::from(clip_observation.non_finite_count),
        rank_launches,
        aggregated_gradient_artifact_path,
        aggregated_gradient_artifact_sha256,
        current_model_artifact_path,
        current_model_artifact_sha256,
        current_model_artifact_surface: String::from("banked_full_precision_v1"),
        step_observation,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_parameter_golf_distributed_8xh100_step_on_worker_mesh(
    bundle: &psionic_data::ParameterGolfDatasetBundle,
    hyperparameters: &ParameterGolfTrainingHyperparameters,
    train_contract: &ParameterGolfTokenStreamContract,
    cursor: &mut ParameterGolfTokenStreamCursor,
    geometry: &ParameterGolfBatchGeometry,
    step_scope_root_dir: &Path,
    requested_train_tokens: u64,
    completed_step_count: u64,
    observed_training_time_ms: u64,
    workers: &mut [ParameterGolfDistributed8xH100PersistentWorkerHandle],
) -> Result<ParameterGolfDistributed8xH100StepExecution, ParameterGolfDistributed8xH100TrainStepError>
{
    let step_index = completed_step_count + 1;
    let step_dir = step_scope_dir(step_scope_root_dir, step_index);
    let rank_receipts_dir = step_dir.join("runtime_train_step_receipts");
    for directory in [&step_dir, &rank_receipts_dir] {
        fs::create_dir_all(directory).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: directory.display().to_string(),
                error,
            }
        })?;
    }
    let learning_rate_multiplier =
        hyperparameters.learning_rate_multiplier(step_index - 1, observed_training_time_ms as f32);
    let muon_momentum = hyperparameters.muon_momentum_at_step(step_index - 1);
    let step_started = Instant::now();
    let mut planned_windows = Vec::with_capacity(CHALLENGE_WORLD_SIZE);
    for rank in 0..CHALLENGE_WORLD_SIZE {
        let window = train_contract
            .plan_window(&bundle.manifest, cursor, requested_train_tokens)?
            .ok_or_else(|| ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "failed to plan one persistent-worker train-step window for rank {rank}"
                ),
            })?;
        *cursor = window.end_cursor.clone();
        planned_windows.push(window);
    }
    for (worker, window) in workers.iter_mut().zip(planned_windows.iter()) {
        write_worker_command(
            worker.rank,
            &mut worker.stdin,
            &ParameterGolfDistributed8xH100WorkerCommand::TrainStep {
                schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
                step_index,
                window: window.clone(),
                learning_rate_multiplier,
                muon_momentum,
            },
        )?;
    }
    let mut rank_launches = Vec::with_capacity(CHALLENGE_WORLD_SIZE);
    let mut mean_train_loss = 0.0_f32;
    let mut gradient_sync_ms = 0_u64;
    let mut optimizer_step_ms = 0_u64;
    let mut gradient_norm_after_clip = 0.0_f32;
    let mut clip_applied = false;
    let mut non_finite_gradient_count = 0_u64;
    for worker in workers.iter_mut() {
        let response = worker_response_from_reader(worker.rank, &mut worker.stdout)?;
        let receipt = match response {
            ParameterGolfDistributed8xH100WorkerResponse::TrainStepComplete { receipt, .. } => {
                receipt
            }
            ParameterGolfDistributed8xH100WorkerResponse::Error { message, .. } => {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                        rank: worker.rank,
                        message,
                    },
                );
            }
            other => {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                        rank: worker.rank,
                        message: format!("expected train-step response, found {other:?}"),
                    },
                );
            }
        };
        let receipt_path = rank_receipts_dir.join(format!("rank_{}.json", worker.rank));
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
        mean_train_loss += receipt.loss / CHALLENGE_WORLD_SIZE as f32;
        gradient_sync_ms = gradient_sync_ms.max(receipt.gradient_sync_ms);
        optimizer_step_ms = optimizer_step_ms.max(receipt.optimizer_step_ms);
        gradient_norm_after_clip = gradient_norm_after_clip.max(receipt.gradient_norm_after_clip);
        clip_applied |= receipt.clip_applied;
        non_finite_gradient_count =
            non_finite_gradient_count.saturating_add(receipt.non_finite_gradient_count);
        rank_launches.push(ParameterGolfDistributed8xH100TrainStepRankLaunch {
            rank: worker.rank,
            local_rank: worker.local_rank,
            cuda_visible_devices: worker.rank.to_string(),
            window_path: String::from("inline://persistent_worker_mesh/train_step_window"),
            gradient_artifact_path: receipt.gradient_artifact_path.clone(),
            receipt_path: receipt_path.display().to_string(),
            log_path: worker.log_path.display().to_string(),
            exit_code: None,
            receipt: Some(receipt),
        });
    }
    let observed_step_ms = duration_ms(step_started);
    let step_observation = ParameterGolfDistributedStepObservation::new(
        step_index,
        0,
        observed_step_ms,
        geometry.train_batch_tokens as u64,
    );
    let aggregated_gradient_artifact_sha256 = rank_launches
        .first()
        .and_then(|launch| launch.receipt.as_ref())
        .map(|receipt| receipt.gradient_artifact_sha256.clone())
        .unwrap_or_default();
    Ok(ParameterGolfDistributed8xH100StepExecution {
        mean_train_loss,
        train_tokens: geometry.train_batch_tokens as u64,
        observed_step_ms,
        gradient_sync_ms,
        optimizer_step_ms,
        gradient_norm_after_clip,
        clip_applied,
        non_finite_gradient_count,
        rank_launches,
        aggregated_gradient_artifact_path: PathBuf::from(format!(
            "in_memory://mesh.parameter_golf.runpod_8xh100/step_{step_index}/averaged_gradient"
        )),
        aggregated_gradient_artifact_sha256,
        current_model_artifact_path: PathBuf::new(),
        current_model_artifact_sha256: String::new(),
        current_model_artifact_surface: String::from("banked_full_precision_v1"),
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
    validation_eval_mode: &ParameterGolfValidationEvalMode,
    validation_batch_sequences: u64,
    mut live_visualization_writer: Option<&mut ParameterGolfDistributedLiveVisualizationWriter>,
) -> Result<
    ParameterGolfDistributed8xH100TrainStepReceipt,
    ParameterGolfDistributed8xH100TrainStepError,
> {
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
    let validation_rank_receipts_dir =
        parameter_golf_distributed_8xh100_validation_rank_receipts_dir(
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
        fs::create_dir_all(directory).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: directory.display().to_string(),
                error,
            }
        })?;
    }
    if let Some(parent) = train_step_receipt_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ParameterGolfDistributed8xH100TrainStepError::Write {
                path: parent.display().to_string(),
                error,
            }
        })?;
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
    let mut cursor = ParameterGolfTokenStreamCursor::new(PARAMETER_GOLF_TRAIN_SPLIT_NAME);
    let train_contract = ParameterGolfTokenStreamContract::new(
        bundle.manifest.key.clone(),
        PARAMETER_GOLF_TRAIN_SPLIT_NAME,
    )
    .with_mode(DatasetIterationMode::Repeat);
    let requested_train_tokens = geometry.local_train_batch_tokens().saturating_add(1) as u64;
    let current_exe =
        env::current_exe().map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Read {
            path: String::from("current_exe"),
            error,
        })?;
    let runtime_payload_path = current_exe.display().to_string();
    let manifest_path = manifest_path
        .canonicalize()
        .unwrap_or_else(|_| manifest_path.to_path_buf());
    let worker_mesh_logs_dir =
        parameter_golf_distributed_8xh100_worker_mesh_logs_dir(root, &bringup_report_relpath);
    let max_wallclock_ms = hyperparameters
        .max_wallclock_seconds
        .filter(|seconds| *seconds > 0.0)
        .map(|seconds| (seconds * 1000.0) as u64);
    let mut executed_step_count = 0_u64;
    let mut observed_training_time_ms = 0_u64;
    let mut step_observations = Vec::new();
    let mut final_step_execution = None;
    let mut workers = spawn_parameter_golf_distributed_8xh100_persistent_worker_mesh(
        root,
        &manifest_path,
        run_id,
        &current_exe,
        &worker_mesh_logs_dir,
        &dataset_root,
        &tokenizer_path,
    )?;
    let stop_reason = loop {
        if executed_step_count >= hyperparameters.iterations {
            break String::from("iteration_cap_reached");
        }
        let step_execution = execute_parameter_golf_distributed_8xh100_step_on_worker_mesh(
            &bundle,
            &hyperparameters,
            &train_contract,
            &mut cursor,
            &geometry,
            &step_scope_root_dir,
            requested_train_tokens,
            executed_step_count,
            observed_training_time_ms,
            workers.as_mut_slice(),
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
        final_step_execution = Some(step_execution);
        if max_wallclock_ms.is_some_and(|wallclock_ms| observed_training_time_ms >= wallclock_ms) {
            emit_distributed_progress_line(format!(
                "stopping_early: wallclock_cap train_time:{}ms step:{}/{}",
                observed_training_time_ms, executed_step_count, hyperparameters.iterations,
            ));
            break String::from("wallclock_cap_reached");
        }
    };
    let final_step_execution = final_step_execution.ok_or_else(|| {
        ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: String::from(
                "distributed 8xH100 train loop finished without one successful step",
            ),
        }
    })?;
    let current_model_artifact_path = model_artifacts_dir.join(format!(
        "post_step_{executed_step_count}_current_model.runtime_surface.safetensors"
    ));
    let current_model_int8_zlib_artifact_path = model_artifacts_dir.join(format!(
        "post_step_{executed_step_count}_final_model.int8.ptz"
    ));
    let exported_artifacts = export_final_worker_artifacts(
        workers.as_mut_slice(),
        executed_step_count,
        &current_model_artifact_path,
        &current_model_int8_zlib_artifact_path,
    )?;
    let current_model_int8_zlib_artifact_sha256 = exported_artifacts
        .current_model_int8_zlib_artifact_sha256
        .clone();
    let current_model_int8_zlib_artifact_size_bytes =
        exported_artifacts.current_model_int8_zlib_artifact_size_bytes;
    let validation_batch_sequences = distributed_validation_batch_sequences(
        validation_batch_sequences as usize,
        &geometry,
        validation_eval_mode,
    ) as u64;
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
    if let Some(writer) = live_visualization_writer.as_deref_mut() {
        writer.record_phase(
            "evaluation",
            Some(String::from("distributed_validation")),
            format!(
                "Dispatching {} persistent distributed validation ranks for optimizer step 1.",
                CHALLENGE_WORLD_SIZE
            ),
            vec![
                String::from("distributed_validation"),
                String::from("persistent_worker_mesh"),
                String::from("cuda_evaluation"),
            ],
            Some(1),
            true,
        )?;
    }
    let mut validation_rank_launches = Vec::with_capacity(CHALLENGE_WORLD_SIZE);
    let mut completed_validation_rank_count = 0_usize;
    for shard in &validation_shard_plan {
        let shard_path = validation_rank_shards_dir.join(format!("rank_{}.json", shard.rank));
        fs::write(
            &shard_path,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&ParameterGolfDistributed8xH100ValidationShardPlan {
                    rank: shard.rank,
                    eval_mode: validation_eval_mode.clone(),
                    local_batch_sequences: validation_batch_sequences,
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
        let worker = workers.get_mut(shard.rank).ok_or_else(|| {
            ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "persistent validation worker rank {} was missing from the mesh",
                    shard.rank
                ),
            }
        })?;
        write_worker_command(
            worker.rank,
            &mut worker.stdin,
            &ParameterGolfDistributed8xH100WorkerCommand::Validation {
                schema_version: WORKER_PROTOCOL_SCHEMA_VERSION,
                shard: ParameterGolfDistributed8xH100ValidationShardPlan {
                    rank: shard.rank,
                    eval_mode: validation_eval_mode.clone(),
                    local_batch_sequences: validation_batch_sequences,
                    sequence_start: shard.sequence_start,
                    sequence_count: shard.sequence_count,
                    evaluation_unit_start: shard.evaluation_unit_start,
                    evaluation_unit_count: shard.evaluation_unit_count,
                    scored_token_start: shard.scored_token_start,
                    scored_token_count: shard.scored_token_count,
                },
            },
        )?;
        let receipt = match worker_response_from_reader(worker.rank, &mut worker.stdout)? {
            ParameterGolfDistributed8xH100WorkerResponse::ValidationComplete {
                receipt, ..
            } => receipt,
            ParameterGolfDistributed8xH100WorkerResponse::Error { message, .. } => {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                        rank: worker.rank,
                        message,
                    },
                );
            }
            other => {
                return Err(
                    ParameterGolfDistributed8xH100TrainStepError::WorkerProtocol {
                        rank: worker.rank,
                        message: format!("expected validation response, found {other:?}"),
                    },
                );
            }
        };
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
        validation_rank_launches.push(ParameterGolfDistributed8xH100ValidationRankLaunch {
            rank: worker.rank,
            local_rank: worker.local_rank,
            cuda_visible_devices: worker.rank.to_string(),
            shard_path: shard_path.display().to_string(),
            receipt_path: receipt_path.display().to_string(),
            log_path: worker.log_path.display().to_string(),
            exit_code: None,
            receipt: Some(receipt),
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
                    String::from("persistent_worker_mesh"),
                    String::from("cuda_evaluation"),
                ],
                Some(1),
                false,
            )?;
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
    distributed_config.validation_eval_mode = validation_eval_mode.clone();
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
        stop_reason: Some(stop_reason),
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
        current_model_artifact_path: exported_artifacts.current_model_artifact_path.clone(),
        current_model_artifact_sha256: exported_artifacts.current_model_artifact_sha256.clone(),
        current_model_artifact_surface: final_step_execution.current_model_artifact_surface,
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
    shutdown_parameter_golf_distributed_8xh100_persistent_worker_mesh(workers.as_mut_slice())?;
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
    let (model, explicit_banked_weights, input_model_artifact_sha256) =
        match input_model_artifact_path.as_ref() {
            Some(path) => {
                let bytes = fs::read(path).map_err(|error| {
                    ParameterGolfDistributed8xH100TrainStepError::Read {
                        path: path.display().to_string(),
                        error,
                    }
                })?;
                (
                    restore_parameter_golf_model_from_safetensors(
                        &baseline_model,
                        bytes.as_slice(),
                    )?,
                    restore_parameter_golf_banked_weights_from_safetensors(
                        &baseline_model,
                        bytes.as_slice(),
                    )?,
                    Some(sha256_hex(bytes.as_slice())),
                )
            }
            None => (baseline_model, None, None),
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
        explicit_banked_weights.as_ref(),
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
        gradient_sync_transport: String::from("file_artifact_parent_aggregation_v1"),
        input_model_artifact_path: input_model_artifact_path
            .as_ref()
            .map(|path| path.display().to_string()),
        input_model_artifact_sha256,
        loss: gradient_batch.loss,
        phase_timings: gradient_batch.phase_timings,
        runtime_receipt: gradient_batch.runtime.as_ref().map(|runtime| {
            ParameterGolfSingleH100TrainingRuntimeReceipt {
                path: String::from("device_resident_cuda_training_graph_v1"),
                graph_surface: String::from("parameter_golf_baseline_training_graph_v2"),
                session_count: 0,
                persistent_parameter_buffer_count: runtime.persistent_parameter_buffer_count,
                persistent_parameter_value_count: runtime.persistent_parameter_value_count,
                resident_parameter_upload_us: runtime.resident_parameter_upload_us,
                parameter_refresh_us: runtime.parameter_refresh_us,
                reusable_input_token_buffer: false,
                reusable_target_token_buffer: false,
                total_input_token_write_us: runtime.input_token_write_us,
                total_target_token_write_us: runtime.target_token_write_us,
                resident_parameter_buffers_reused: runtime.resident_parameter_buffers_reused,
            }
        }),
        observed_wallclock_ms,
        gradient_sync_ms: 0,
        optimizer_step_ms: 0,
        gradient_norm_after_clip: 0.0,
        clip_applied: false,
        non_finite_gradient_count: 0,
        worker_pid: std::process::id(),
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
            let aggregated_gradients = load_gradient_artifact(&aggregated_gradient_artifact_path)?;
            let hyperparameters = ParameterGolfTrainingHyperparameters::baseline_defaults();
            let runtime_descriptor = baseline_model.banked_descriptor()?;
            let optimizer_plan =
                parameter_golf_optimizer_plan(&runtime_descriptor, &hyperparameters)?;
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
    let validation_batch_sequences = distributed_validation_batch_sequences(
        shard.local_batch_sequences as usize,
        &geometry,
        &shard.eval_mode,
    );
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
            score_first_ttt_receipt: None,
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
        worker_pid: std::process::id(),
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

fn load_rank_gradient_artifacts_parallel(
    rank_launches: &[ParameterGolfDistributed8xH100TrainStepRankLaunch],
) -> Result<Vec<(usize, BTreeMap<String, Vec<f32>>)>, ParameterGolfDistributed8xH100TrainStepError>
{
    let mut loaded = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(rank_launches.len());
        for launch in rank_launches {
            let rank = launch.rank;
            let gradient_artifact_path = launch
                .receipt
                .as_ref()
                .expect("rank receipt presence validated above")
                .gradient_artifact_path
                .clone();
            handles.push(scope.spawn(move || {
                load_gradient_artifact(Path::new(&gradient_artifact_path))
                    .map(|gradients| (rank, gradients))
            }));
        }
        let mut loaded = Vec::with_capacity(handles.len());
        for handle in handles {
            let result = handle.join().map_err(|payload| {
                let message = if let Some(message) = payload.downcast_ref::<&'static str>() {
                    (*message).to_string()
                } else if let Some(message) = payload.downcast_ref::<String>() {
                    message.clone()
                } else {
                    String::from("unknown panic payload")
                };
                ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                    message: format!(
                        "parallel distributed gradient artifact load panicked: {message}"
                    ),
                }
            })?;
            loaded.push(result?);
        }
        Ok::<_, ParameterGolfDistributed8xH100TrainStepError>(loaded)
    })?;
    loaded.sort_by_key(|(rank, _)| *rank);
    Ok(loaded)
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
        distributed_validation_batch_sequences, load_gradient_artifact,
        prune_step_scope_for_next_step, write_gradient_artifact,
        VALIDATION_BATCH_SEQUENCES_ENV_VAR,
    };
    use crate::{
        parameter_golf_default_validation_batch_sequences, ParameterGolfBatchGeometry,
        ParameterGolfValidationEvalMode,
    };

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
                parameter_golf_default_validation_batch_sequences(
                    &geometry,
                    &ParameterGolfValidationEvalMode::NonOverlapping,
                ),
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
                parameter_golf_default_validation_batch_sequences(
                    &geometry,
                    &ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 },
                ),
                &geometry,
                &ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 }
            ),
            parameter_golf_default_validation_batch_sequences(
                &geometry,
                &ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 }
            )
        );
    }

    #[test]
    fn distributed_validation_batch_sequences_respects_explicit_configured_geometry() {
        let geometry = ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults();
        assert_eq!(
            distributed_validation_batch_sequences(
                256,
                &geometry,
                &ParameterGolfValidationEvalMode::NonOverlapping
            ),
            256
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
                parameter_golf_default_validation_batch_sequences(
                    &geometry,
                    &ParameterGolfValidationEvalMode::NonOverlapping,
                ),
                &geometry,
                &ParameterGolfValidationEvalMode::NonOverlapping
            ),
            256
        );
        assert_eq!(
            distributed_validation_batch_sequences(
                parameter_golf_default_validation_batch_sequences(
                    &geometry,
                    &ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 },
                ),
                &geometry,
                &ParameterGolfValidationEvalMode::SlidingWindow { stride: 64 }
            ),
            256
        );
        unsafe {
            env::remove_var(VALIDATION_BATCH_SEQUENCES_ENV_VAR);
        }
    }

    #[test]
    fn distributed_step_scope_prune_keeps_only_current_model(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = tempfile::tempdir()?;
        let step_dir = temp_dir.path().join("step_00001");
        std::fs::create_dir_all(step_dir.join("runtime_train_step_gradients"))?;
        std::fs::create_dir_all(step_dir.join("runtime_train_step_logs"))?;
        std::fs::write(
            step_dir.join("aggregated_gradients.safetensors"),
            b"gradients",
        )?;
        std::fs::write(
            step_dir.join("current_model.runtime_surface.safetensors"),
            b"model",
        )?;
        std::fs::write(
            step_dir
                .join("runtime_train_step_gradients")
                .join("rank_0.safetensors"),
            b"rank-gradients",
        )?;
        std::fs::write(
            step_dir.join("runtime_train_step_logs").join("rank_0.log"),
            b"log",
        )?;

        prune_step_scope_for_next_step(&step_dir)?;

        let retained = std::fs::read_dir(&step_dir)?
            .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            retained,
            vec![String::from("current_model.runtime_surface.safetensors")]
        );
        Ok(())
    }
}

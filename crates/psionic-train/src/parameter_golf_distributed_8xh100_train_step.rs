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
    parameter_golf_dataset_bundle_from_local_dir, DatasetIterationMode, DatasetKey,
    ParameterGolfTokenStreamContract, ParameterGolfTokenStreamCursor, ParameterGolfTokenStreamWindow,
    PARAMETER_GOLF_TRAIN_SPLIT_NAME,
};
use psionic_eval::ParameterGolfDistributedThroughputReceipt;
use psionic_models::ParameterGolfReferenceModel;
use safetensors::{serialize, tensor::TensorView, Dtype as SafeTensorsDType, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    apply_gradients_to_state, benchmark_parameter_golf_distributed_8xh100, build_tokenizer_digest,
    clip_gradients, execute_parameter_golf_training_gradient_batch,
    inspect_local_distributed_8xh100_machine, materialize_current_model,
    parameter_golf_optimizer_plan, parameter_golf_runpod_8xh100_capability_profile,
    seed_parameter_states, zero_gradients, ParameterGolfBatchGeometry,
    ParameterGolfDistributed8xH100BringupReport,
    ParameterGolfDistributed8xH100RuntimeBootstrapReceipt,
    ParameterGolfDistributedStepObservation, ParameterGolfRunPod8xH100Measurements,
    ParameterGolfSingleH100PhaseTimings, ParameterGolfSingleH100TrainingError,
    ParameterGolfTrainingHyperparameters, PARAMETER_GOLF_SINGLE_H100_VARIANT,
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

const CHALLENGE_WORLD_SIZE: usize = 8;

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
    /// One measured step observation lifted into the distributed receipt lane.
    pub step_observation: ParameterGolfDistributedStepObservation,
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

/// Failure while executing the distributed `8xH100` train-step seam.
#[derive(Debug, Error)]
pub enum ParameterGolfDistributed8xH100TrainStepError {
    #[error("parameter golf distributed 8xH100 train-step failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("parameter golf distributed 8xH100 train-step failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("parameter golf distributed 8xH100 train-step missing environment variable `{key}`")]
    MissingEnv { key: &'static str },
    #[error(
        "parameter golf distributed 8xH100 train-step invalid environment `{key}`=`{value}`"
    )]
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
    #[error("parameter golf distributed 8xH100 train-step aggregate failed: {message}")]
    Aggregate { message: String },
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
}

/// Returns whether the current process is one internal distributed train-step child.
#[must_use]
pub fn parameter_golf_distributed_8xh100_train_step_child_enabled() -> bool {
    env::var_os(CHILD_ENV_VAR).is_some()
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
    let rank_receipts_dir = parameter_golf_distributed_8xh100_train_step_rank_receipts_dir(
        root,
        &bringup_report_relpath,
    );
    let rank_logs_dir =
        parameter_golf_distributed_8xh100_train_step_rank_logs_dir(root, &bringup_report_relpath);
    let rank_windows_dir =
        parameter_golf_distributed_8xh100_train_step_rank_windows_dir(root, &bringup_report_relpath);
    let rank_gradients_dir = parameter_golf_distributed_8xh100_train_step_rank_gradients_dir(
        root,
        &bringup_report_relpath,
    );
    for directory in [
        &rank_receipts_dir,
        &rank_logs_dir,
        &rank_windows_dir,
        &rank_gradients_dir,
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
    for rank in 0..CHALLENGE_WORLD_SIZE {
        let window = train_contract
            .plan_window(&bundle.manifest, &cursor, requested_train_tokens)?
            .ok_or_else(|| ParameterGolfDistributed8xH100TrainStepError::Aggregate {
                message: format!(
                    "failed to plan one train-step window for rank {rank} under the distributed posture"
                ),
            })?;
        cursor = window.end_cursor.clone();
        let window_path = rank_windows_dir.join(format!("rank_{rank}.json"));
        fs::write(
            &window_path,
            format!("{}\n", serde_json::to_string_pretty(&window)?),
        )
        .map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Write {
            path: window_path.display().to_string(),
            error,
        })?;
    }

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
        let child = Command::new(&current_exe)
            .arg(&manifest_path)
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
            .env(CHILD_GRADIENT_ARTIFACT_PATH_ENV_VAR, &gradient_artifact_path)
            .env(
                crate::PARAMETER_GOLF_SINGLE_H100_DATASET_ROOT_ENV_VAR,
                &dataset_root,
            )
            .env(
                crate::PARAMETER_GOLF_SINGLE_H100_TOKENIZER_PATH_ENV_VAR,
                &tokenizer_path,
            )
            .env("CUDA_VISIBLE_DEVICES", rank.to_string())
            .env("WORLD_SIZE", CHALLENGE_WORLD_SIZE.to_string())
            .env("PSIONIC_DISTRIBUTED_RANK", rank.to_string())
            .env("PSIONIC_DISTRIBUTED_LOCAL_RANK", rank.to_string())
            .env("PSIONIC_DISTRIBUTED_WORLD_SIZE", CHALLENGE_WORLD_SIZE.to_string())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|error| ParameterGolfDistributed8xH100TrainStepError::ChildSpawn {
                rank,
                error,
            })?;
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
                .map_err(|error| ParameterGolfDistributed8xH100TrainStepError::ChildDecode {
                    path: receipt_path.display().to_string(),
                    error,
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
            return Err(ParameterGolfDistributed8xH100TrainStepError::ChildMissingReceipt {
                rank: launch.rank,
            });
        }
    }

    let sync_started = Instant::now();
    let mut accumulated_gradients = zero_gradients(&trainer_state);
    for launch in &rank_launches {
        let rank_receipt = launch
            .receipt
            .as_ref()
            .expect("rank receipt presence validated above");
        let gradients = load_gradient_artifact(Path::new(&rank_receipt.gradient_artifact_path))?;
        crate::accumulate_gradients(
            accumulated_gradients.as_mut_slice(),
            &trainer_state,
            &gradients,
            CHALLENGE_WORLD_SIZE as f32,
        )?;
    }
    let gradient_sync_ms = duration_ms(sync_started);

    let clip_observation =
        clip_gradients(accumulated_gradients.as_mut_slice(), hyperparameters.grad_clip_norm);
    let optimizer_started = Instant::now();
    let learning_rate_multiplier = hyperparameters.learning_rate_multiplier(0, 0.0);
    let muon_momentum = hyperparameters.muon_momentum_at_step(0);
    apply_gradients_to_state(
        &mut trainer_state,
        accumulated_gradients.as_slice(),
        learning_rate_multiplier,
        muon_momentum,
        1,
    )?;
    let _current_model = materialize_current_model(&initial_model, &trainer_state)?;
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
        ParameterGolfDistributedStepObservation::new(1, 0, observed_step_ms, train_tokens);

    let mut measurements = ParameterGolfRunPod8xH100Measurements::challenge_defaults();
    measurements.run_id = Some(String::from(run_id));
    measurements.mesh_id = Some(String::from("mesh.parameter_golf.runpod_8xh100"));
    measurements.step_observations = vec![step_observation.clone()];
    fs::write(
        &measurements_path,
        format!("{}\n", serde_json::to_string_pretty(&measurements)?),
    )
    .map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Write {
        path: measurements_path.display().to_string(),
        error,
    })?;

    let machine = inspect_local_distributed_8xh100_machine();
    let mut distributed_config = crate::ParameterGolfDistributed8xH100Config::challenge_defaults();
    distributed_config.run_id = String::from(run_id);
    distributed_config.mesh_id = String::from("mesh.parameter_golf.runpod_8xh100");
    distributed_config.step_observations = vec![step_observation.clone()];
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
    .map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Write {
        path: distributed_receipt_path.display().to_string(),
        error,
    })?;

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
        mean_train_loss,
        train_tokens,
        observed_step_ms,
        gradient_sync_ms,
        optimizer_step_ms,
        gradient_norm_after_clip: clip_observation.gradient_norm_after_clip.unwrap_or_default(),
        clip_applied: clip_observation.clip_applied,
        non_finite_gradient_count: u64::from(clip_observation.non_finite_count),
        rank_launches,
        step_observation,
        distributed_receipt,
        claim_boundary: String::from(
            "This receipt proves the exported-folder distributed runtime executed one real 8-rank Parameter Golf train step on explicit per-rank H100 bindings and emitted measured step observations into the typed distributed receipt lane. It does not yet claim distributed validation, final artifact closure, or full record-track completion.",
        ),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.stable_digest();
    fs::write(
        &train_step_receipt_path,
        format!("{}\n", serde_json::to_string_pretty(&receipt)?),
    )
    .map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Write {
        path: train_step_receipt_path.display().to_string(),
        error,
    })?;
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
    let model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    let geometry = ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults();
    let mut cuda_backend = CudaBackend::new();
    let selected_device = cuda_backend
        .selected_device()
        .cloned()
        .ok_or_else(|| ParameterGolfDistributed8xH100TrainStepError::Aggregate {
            message: format!("distributed train-step child rank {rank} could not select one CUDA device"),
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
        fs::create_dir_all(parent).map_err(
            |error| ParameterGolfDistributed8xH100TrainStepError::Write {
                path: parent.display().to_string(),
                error,
            },
        )?;
    }
    fs::write(
        &receipt_path,
        format!("{}\n", serde_json::to_string_pretty(&receipt)?),
    )
    .map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Write {
        path: receipt_path.display().to_string(),
        error,
    })?;
    Ok(receipt)
}

fn write_gradient_artifact(
    output_path: &Path,
    gradients: &BTreeMap<String, Vec<f32>>,
) -> Result<String, ParameterGolfDistributed8xH100TrainStepError> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(
            |error| ParameterGolfDistributed8xH100TrainStepError::Write {
                path: parent.display().to_string(),
                error,
            },
        )?;
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
        views.iter().map(|(name, view)| (name.as_str(), view.clone())),
        None,
    )
    .map_err(|error| ParameterGolfDistributed8xH100TrainStepError::Aggregate {
        message: format!("failed to serialize distributed gradient artifact: {error}"),
    })?;
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
    let bytes = fs::read(path).map_err(|error| {
        ParameterGolfDistributed8xH100TrainStepError::Read {
            path: path.display().to_string(),
            error,
        }
    })?;
    let safetensors =
        SafeTensors::deserialize(bytes.as_slice()).map_err(|error| {
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
    use std::collections::BTreeMap;

    use super::{load_gradient_artifact, write_gradient_artifact};

    #[test]
    fn distributed_train_step_gradient_artifact_roundtrips() -> Result<(), Box<dyn std::error::Error>>
    {
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
}

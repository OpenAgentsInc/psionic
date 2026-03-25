use std::{
    env, fs,
    path::{Path, PathBuf},
};

use psionic_eval::{evaluate_parameter_golf_validation, ParameterGolfValidationEvalReport};
use psionic_models::ParameterGolfReferenceModel;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    execute_parameter_golf_backed_dense_rank_runtime,
    execute_parameter_golf_distributed_8xh100_runtime_bootstrap_child,
    execute_parameter_golf_distributed_8xh100_train_step_child,
    execute_parameter_golf_distributed_8xh100_validation_child,
    parameter_golf_distributed_8xh100_runtime_bootstrap_child_enabled,
    parameter_golf_distributed_8xh100_train_step_child_enabled,
    parameter_golf_distributed_8xh100_validation_child_enabled,
    restore_parameter_golf_model_from_int8_zlib,
    write_parameter_golf_distributed_8xh100_bringup_report, DenseRankRuntimeError,
    DenseRankRuntimeExecutionReceipt, ParameterGolfDistributed8xH100BringupConfig,
    ParameterGolfDistributed8xH100BringupError, ParameterGolfDistributed8xH100BringupReport,
    ParameterGolfDistributed8xH100RuntimeBootstrapError,
    ParameterGolfDistributed8xH100RuntimeBootstrapRankReceipt,
    ParameterGolfDistributed8xH100RuntimeBootstrapReceipt,
    ParameterGolfDistributed8xH100TrainStepError,
    ParameterGolfDistributed8xH100TrainStepRankReceipt,
    ParameterGolfDistributed8xH100TrainStepReceipt, ParameterGolfLocalReferenceFixture,
    ParameterGolfNonRecordSubmissionManifest, ParameterGolfSubmissionAccountingReceipt,
    ParameterGolfSubmissionRealExecutionContract, PARAMETER_GOLF_DISTRIBUTED_8XH100_EXECUTION_MODE,
    PARAMETER_GOLF_EXECUTION_MODE_ENV_VAR,
    ParameterGolfDistributed8xH100ValidationRankReceipt,
};

/// Machine-readable runtime manifest shipped with the submission folder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSubmissionRuntimeManifest {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable package version.
    pub package_version: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Canonical benchmark reference.
    pub benchmark_ref: String,
    /// Top-level entrypoint path.
    pub entrypoint_path: String,
    /// Shipped runtime payload path.
    pub runtime_payload_path: String,
    /// Top-level `submission.json` path.
    pub submission_manifest_path: String,
    /// Shipped accounting receipt path.
    pub accounting_receipt_path: String,
    /// Shipped local-reference fixture path.
    pub fixture_path: String,
    /// Shipped counted model artifact path.
    pub model_artifact_path: String,
    /// Runtime receipt path written by the payload.
    pub runtime_receipt_path: String,
    /// Distributed bring-up report path written by the shipped payload in `distributed_8xh100_train` mode.
    pub distributed_bringup_report_path: String,
    /// Sequence length used by the bounded local-reference eval replay.
    pub sequence_length: usize,
    /// Validation batch tokens used by the bounded local-reference eval replay.
    pub validation_batch_tokens: usize,
    /// Expected final roundtrip validation loss from `submission.json`.
    pub expected_val_loss: f64,
    /// Expected final roundtrip validation bits-per-byte from `submission.json`.
    pub expected_val_bpb: f64,
    /// Default exported-folder execution mode.
    #[serde(default = "default_local_reference_execution_mode")]
    pub default_execution_mode: String,
    /// Additional explicit real execution contracts shipped with the folder.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub real_execution_contracts: Vec<ParameterGolfSubmissionRealExecutionContract>,
    /// Explicit runtime posture for the package.
    pub runtime_posture: String,
    /// Explicit claim boundary for the runtime.
    pub claim_boundary: String,
    /// Stable digest over the manifest.
    pub manifest_digest: String,
}

impl ParameterGolfSubmissionRuntimeManifest {
    /// Returns a stable digest over the manifest payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.manifest_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_submission_runtime_manifest|",
            &digestible,
        )
    }
}

/// Runtime result emitted by the shipped submission payload.
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterGolfSubmissionRuntimeOutcome {
    /// Bounded local-reference restore-and-eval replay.
    LocalReference(ParameterGolfSubmissionRuntimeReceipt),
    /// Rust-owned distributed `8xH100` bring-up report.
    Distributed8xH100Bringup {
        report_path: String,
        report: ParameterGolfDistributed8xH100BringupReport,
    },
    /// Rust-owned distributed `8xH100` runtime bootstrap receipt above the bring-up gate.
    Distributed8xH100Bootstrap {
        report_path: String,
        report: ParameterGolfDistributed8xH100BringupReport,
        receipt_path: String,
        receipt: ParameterGolfDistributed8xH100RuntimeBootstrapReceipt,
    },
    /// Rust-owned distributed `8xH100` train-step receipt above the bootstrap gate.
    Distributed8xH100TrainStep {
        report_path: String,
        report: ParameterGolfDistributed8xH100BringupReport,
        bootstrap_receipt_path: String,
        bootstrap_receipt: ParameterGolfDistributed8xH100RuntimeBootstrapReceipt,
        train_step_receipt_path: String,
        train_step_receipt: ParameterGolfDistributed8xH100TrainStepReceipt,
        dense_rank_execution_receipt_path: String,
        dense_rank_execution_receipt: DenseRankRuntimeExecutionReceipt,
    },
    /// Rust-owned distributed `8xH100` completion receipt above validation-backed execution.
    Distributed8xH100Completed {
        receipt_path: String,
        receipt: ParameterGolfDistributed8xH100CompletionReceipt,
    },
    /// Internal child-rank bootstrap receipt used by the shipped runtime fanout.
    Distributed8xH100BootstrapChild {
        receipt_path: String,
        receipt: ParameterGolfDistributed8xH100RuntimeBootstrapRankReceipt,
    },
    /// Internal child-rank train-step receipt used by the shipped runtime fanout.
    Distributed8xH100TrainStepChild {
        receipt_path: String,
        receipt: ParameterGolfDistributed8xH100TrainStepRankReceipt,
    },
    /// Internal child-rank validation receipt used by the shipped runtime fanout.
    Distributed8xH100ValidationChild {
        receipt_path: String,
        receipt: ParameterGolfDistributed8xH100ValidationRankReceipt,
    },
}

fn default_local_reference_execution_mode() -> String {
    String::from("local_reference_validation")
}

/// Completion receipt emitted when the shipped distributed runtime finishes the
/// current exported-folder execution contract without the old refusal posture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfDistributed8xH100CompletionReceipt {
    pub schema_version: u32,
    pub run_id: String,
    pub bringup_report_path: String,
    pub bringup_report_digest: String,
    pub runtime_bootstrap_receipt_path: String,
    pub runtime_bootstrap_receipt_digest: String,
    pub train_step_receipt_path: String,
    pub train_step_receipt_digest: String,
    pub distributed_receipt_path: String,
    pub distributed_receipt_digest: String,
    pub final_model_artifact_path: String,
    pub final_model_artifact_digest: String,
    pub final_model_artifact_size_bytes: u64,
    pub submission_manifest_path: String,
    pub submission_manifest_digest: String,
    pub distributed_validation_mean_loss: f64,
    pub distributed_validation_bits_per_byte: f64,
    pub distributed_validation_observed_ms: u64,
    pub claim_boundary: String,
    pub receipt_digest: String,
}

impl ParameterGolfDistributed8xH100CompletionReceipt {
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.receipt_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_distributed_8xh100_completion_receipt|",
            &digestible,
        )
    }
}

/// Runtime receipt emitted by the shipped submission payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSubmissionRuntimeReceipt {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable run identifier.
    pub run_id: String,
    /// Runtime posture executed by the payload.
    pub runtime_posture: String,
    /// Top-level entrypoint path.
    pub entrypoint_path: String,
    /// Runtime payload path.
    pub runtime_payload_path: String,
    /// Top-level `submission.json` path.
    pub submission_manifest_path: String,
    /// Accounting receipt path.
    pub accounting_receipt_path: String,
    /// Fixture path used for the eval replay.
    pub fixture_path: String,
    /// Counted model artifact path.
    pub model_artifact_path: String,
    /// Executed validation report from the shipped model artifact.
    pub executed_validation: ParameterGolfValidationEvalReport,
    /// Whether the executed validation loss matched `submission.json`.
    pub matches_submission_val_loss: bool,
    /// Whether the executed validation bits-per-byte matched `submission.json`.
    pub matches_submission_val_bpb: bool,
    /// Whether the counted-code bytes in `submission.json` matched the shipped accounting receipt.
    pub matches_accounting_code_bytes: bool,
    /// Whether the total counted bytes in `submission.json` matched the shipped accounting receipt.
    pub matches_accounting_total_bytes: bool,
    /// Whether the counted model size in `submission.json` matched the shipped artifact bytes.
    pub matches_submission_model_bytes: bool,
    /// Whether the counted model size in the accounting receipt matched the shipped artifact bytes.
    pub matches_accounting_model_bytes: bool,
    /// Stable digest over the receipt payload.
    pub receipt_digest: String,
}

impl ParameterGolfSubmissionRuntimeReceipt {
    fn new(
        manifest: &ParameterGolfSubmissionRuntimeManifest,
        executed_validation: ParameterGolfValidationEvalReport,
        submission_manifest: &ParameterGolfNonRecordSubmissionManifest,
        accounting_receipt: &ParameterGolfSubmissionAccountingReceipt,
        model_artifact_size_bytes: u64,
    ) -> Self {
        let mut receipt = Self {
            schema_version: 1,
            run_id: manifest.run_id.clone(),
            runtime_posture: manifest.runtime_posture.clone(),
            entrypoint_path: manifest.entrypoint_path.clone(),
            runtime_payload_path: manifest.runtime_payload_path.clone(),
            submission_manifest_path: manifest.submission_manifest_path.clone(),
            accounting_receipt_path: manifest.accounting_receipt_path.clone(),
            fixture_path: manifest.fixture_path.clone(),
            model_artifact_path: manifest.model_artifact_path.clone(),
            matches_submission_val_loss: metric_matches(
                executed_validation.mean_loss,
                submission_manifest.val_loss,
            ),
            matches_submission_val_bpb: metric_matches(
                executed_validation.bits_per_byte,
                submission_manifest.val_bpb,
            ),
            matches_accounting_code_bytes: submission_manifest.bytes_code
                == accounting_receipt.counted_code_bytes,
            matches_accounting_total_bytes: submission_manifest.bytes_total
                == accounting_receipt.total_counted_bytes,
            matches_submission_model_bytes: submission_manifest.bytes_model_int8_zlib
                == model_artifact_size_bytes,
            matches_accounting_model_bytes: accounting_receipt.compressed_model_bytes
                == model_artifact_size_bytes,
            executed_validation,
            receipt_digest: String::new(),
        };
        receipt.receipt_digest = receipt.stable_digest();
        receipt
    }

    /// Returns a stable digest over the receipt payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.receipt_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_submission_runtime_receipt|",
            &digestible,
        )
    }

    /// Returns whether the runtime result is consistent with the shipped manifest and accounting.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.matches_submission_val_loss
            && self.matches_submission_val_bpb
            && self.matches_accounting_code_bytes
            && self.matches_accounting_total_bytes
            && self.matches_submission_model_bytes
            && self.matches_accounting_model_bytes
    }
}

/// Failure while executing the shipped submission runtime.
#[derive(Debug, Error)]
pub enum ParameterGolfSubmissionRuntimeError {
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to decode `{artifact_kind}` from `{path}`: {error}")]
    Deserialize {
        artifact_kind: String,
        path: String,
        error: serde_json::Error,
    },
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("parameter golf submission runtime consistency error: {message}")]
    Consistency { message: String },
    #[error("parameter golf submission runtime execution-mode error: {message}")]
    ExecutionMode { message: String },
    #[error(transparent)]
    ReferenceTraining(#[from] crate::ParameterGolfReferenceTrainingError),
    #[error(transparent)]
    Eval(#[from] psionic_eval::ParameterGolfValidationEvalError),
    #[error(transparent)]
    Model(#[from] psionic_models::ParameterGolfModelError),
    #[error(transparent)]
    Execution(#[from] psionic_models::ParameterGolfExecutionError),
    #[error(transparent)]
    Data(#[from] psionic_data::ParameterGolfDataError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    DistributedBringup(#[from] ParameterGolfDistributed8xH100BringupError),
    #[error(transparent)]
    DistributedRuntimeBootstrap(#[from] ParameterGolfDistributed8xH100RuntimeBootstrapError),
    #[error(transparent)]
    DistributedTrainStep(#[from] ParameterGolfDistributed8xH100TrainStepError),
    #[error(transparent)]
    DenseRankRuntime(#[from] DenseRankRuntimeError),
}

/// Executes the shipped submission runtime manifest, writes the runtime receipt, and returns it.
pub fn execute_parameter_golf_submission_runtime_manifest(
    root: &Path,
    manifest_path: &Path,
) -> Result<ParameterGolfSubmissionRuntimeReceipt, ParameterGolfSubmissionRuntimeError> {
    let manifest = read_json::<ParameterGolfSubmissionRuntimeManifest>(
        manifest_path,
        "parameter_golf_submission_runtime_manifest",
    )?;
    let receipt = execute_parameter_golf_submission_runtime(root, &manifest)?;
    let receipt_path = root.join(&manifest.runtime_receipt_path);
    if let Some(parent) = receipt_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ParameterGolfSubmissionRuntimeError::CreateDir {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    let encoded = serde_json::to_string_pretty(&receipt)?;
    fs::write(&receipt_path, format!("{encoded}\n")).map_err(|error| {
        ParameterGolfSubmissionRuntimeError::Write {
            path: receipt_path.display().to_string(),
            error,
        }
    })?;
    Ok(receipt)
}

/// Executes the shipped submission runtime entrypoint for the current execution mode.
pub fn execute_parameter_golf_submission_runtime_entrypoint(
    root: &Path,
    manifest_path: &Path,
) -> Result<ParameterGolfSubmissionRuntimeOutcome, ParameterGolfSubmissionRuntimeError> {
    let manifest = read_json::<ParameterGolfSubmissionRuntimeManifest>(
        manifest_path,
        "parameter_golf_submission_runtime_manifest",
    )?;
    let execution_mode = env::var(PARAMETER_GOLF_EXECUTION_MODE_ENV_VAR)
        .unwrap_or(manifest.default_execution_mode.clone());
    match execution_mode.as_str() {
        "local_reference_validation" => Ok(ParameterGolfSubmissionRuntimeOutcome::LocalReference(
            execute_parameter_golf_submission_runtime_manifest(root, manifest_path)?,
        )),
        PARAMETER_GOLF_DISTRIBUTED_8XH100_EXECUTION_MODE => {
            if parameter_golf_distributed_8xh100_validation_child_enabled() {
                let receipt =
                    execute_parameter_golf_submission_distributed_8xh100_validation_child(
                        &manifest,
                    )?;
                return Ok(
                    ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100ValidationChild {
                        receipt_path: env::var(
                            "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_VALIDATION_RECEIPT_PATH",
                        )
                        .unwrap_or_default(),
                        receipt,
                    },
                );
            }
            if parameter_golf_distributed_8xh100_train_step_child_enabled() {
                let receipt =
                    execute_parameter_golf_submission_distributed_8xh100_train_step_child(
                        &manifest,
                    )?;
                return Ok(
                    ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100TrainStepChild {
                        receipt_path: env::var(
                            "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_TRAIN_STEP_RECEIPT_PATH",
                        )
                        .unwrap_or_default(),
                        receipt,
                    },
                );
            }
            if parameter_golf_distributed_8xh100_runtime_bootstrap_child_enabled() {
                let receipt = execute_parameter_golf_submission_distributed_8xh100_bootstrap_child(
                    &manifest,
                )?;
                return Ok(
                    ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100BootstrapChild {
                        receipt_path: env::var(
                            "PSIONIC_PARAMETER_GOLF_DISTRIBUTED_8XH100_BOOTSTRAP_RECEIPT_PATH",
                        )
                        .unwrap_or_default(),
                        receipt,
                    },
                );
            }
            execute_parameter_golf_submission_distributed_8xh100_bootstrap(
                root,
                manifest_path,
                &manifest,
            )
        }
        other => Err(ParameterGolfSubmissionRuntimeError::ExecutionMode {
            message: format!("unsupported execution mode `{other}`"),
        }),
    }
}

/// Executes the actual shipped restore-and-eval path described by the runtime manifest.
pub fn execute_parameter_golf_submission_runtime(
    root: &Path,
    manifest: &ParameterGolfSubmissionRuntimeManifest,
) -> Result<ParameterGolfSubmissionRuntimeReceipt, ParameterGolfSubmissionRuntimeError> {
    let submission_manifest = read_json::<ParameterGolfNonRecordSubmissionManifest>(
        root.join(&manifest.submission_manifest_path),
        "parameter_golf_submission_manifest",
    )?;
    let accounting_receipt = read_json::<ParameterGolfSubmissionAccountingReceipt>(
        root.join(&manifest.accounting_receipt_path),
        "parameter_golf_submission_accounting_receipt",
    )?;
    let fixture = read_json::<ParameterGolfLocalReferenceFixture>(
        root.join(&manifest.fixture_path),
        "parameter_golf_local_reference_fixture",
    )?;
    let model_artifact_path = root.join(&manifest.model_artifact_path);
    let model_artifact_bytes = fs::read(&model_artifact_path).map_err(|error| {
        ParameterGolfSubmissionRuntimeError::Read {
            path: model_artifact_path.display().to_string(),
            error,
        }
    })?;
    let baseline_model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    let restored_model = restore_parameter_golf_model_from_int8_zlib(
        &baseline_model,
        model_artifact_bytes.as_slice(),
    )?;
    let executed_validation = evaluate_parameter_golf_validation(
        &restored_model,
        fixture.validation_tokens.as_slice(),
        manifest.sequence_length,
        manifest.validation_batch_tokens,
        &fixture.byte_luts()?,
    )?;
    let receipt = ParameterGolfSubmissionRuntimeReceipt::new(
        manifest,
        executed_validation,
        &submission_manifest,
        &accounting_receipt,
        model_artifact_bytes.len() as u64,
    );
    if !receipt.is_consistent() {
        return Err(ParameterGolfSubmissionRuntimeError::Consistency {
            message: String::from(
                "executed runtime receipt did not match submission.json or the shipped accounting receipt",
            ),
        });
    }
    Ok(receipt)
}

fn execute_parameter_golf_submission_distributed_8xh100_bootstrap(
    root: &Path,
    manifest_path: &Path,
    manifest: &ParameterGolfSubmissionRuntimeManifest,
) -> Result<ParameterGolfSubmissionRuntimeOutcome, ParameterGolfSubmissionRuntimeError> {
    let output_path = root.join(&manifest.distributed_bringup_report_path);
    let config = ParameterGolfDistributed8xH100BringupConfig::challenge_defaults();
    let report =
        write_parameter_golf_distributed_8xh100_bringup_report(&output_path, &config, None)?;
    if !report.ready_to_attempt() {
        return Ok(
            ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100Bringup {
                report_path: output_path.display().to_string(),
                report,
            },
        );
    }
    let runtime = execute_parameter_golf_backed_dense_rank_runtime(
        root,
        manifest_path,
        &manifest.run_id,
        &output_path,
        &report,
    )?;
    let bootstrap_receipt_path = PathBuf::from(&runtime.bootstrap_receipt_path);
    let train_step_receipt_path = PathBuf::from(&runtime.train_step_receipt_path);
    if runtime
        .train_step_receipt
        .distributed_receipt
        .validation_aggregation
        .is_none()
    {
        return Ok(
            ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100TrainStep {
                report_path: output_path.display().to_string(),
                report,
                bootstrap_receipt_path: runtime.bootstrap_receipt_path,
                bootstrap_receipt: runtime.bootstrap_receipt,
                train_step_receipt_path: runtime.train_step_receipt_path,
                train_step_receipt: runtime.train_step_receipt,
                dense_rank_execution_receipt_path: runtime.dense_rank_execution_receipt_path,
                dense_rank_execution_receipt: runtime.dense_rank_execution_receipt,
            },
        );
    }
    let completion_receipt_path =
        parameter_golf_distributed_8xh100_completion_receipt_path(
            root,
            &manifest.distributed_bringup_report_path,
        );
    let completion_receipt = build_parameter_golf_distributed_8xh100_completion_receipt(
        root,
        manifest,
        &output_path,
        &report,
        &bootstrap_receipt_path,
        &runtime.bootstrap_receipt,
        &train_step_receipt_path,
        &runtime.train_step_receipt,
    )?;
    if let Some(parent) = completion_receipt_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ParameterGolfSubmissionRuntimeError::CreateDir {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    fs::write(
        &completion_receipt_path,
        format!("{}\n", serde_json::to_string_pretty(&completion_receipt)?),
    )
    .map_err(|error| ParameterGolfSubmissionRuntimeError::Write {
        path: completion_receipt_path.display().to_string(),
        error,
    })?;
    Ok(ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100Completed {
        receipt_path: completion_receipt_path.display().to_string(),
        receipt: completion_receipt,
    })
}

fn parameter_golf_distributed_8xh100_completion_receipt_path(
    root: &Path,
    bringup_report_path: &str,
) -> PathBuf {
    let resolved = root.join(bringup_report_path);
    match resolved.parent() {
        Some(parent) => parent.join("parameter_golf_distributed_8xh100_completion.json"),
        None => root.join("parameter_golf_distributed_8xh100_completion.json"),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_parameter_golf_distributed_8xh100_completion_receipt(
    root: &Path,
    manifest: &ParameterGolfSubmissionRuntimeManifest,
    bringup_report_path: &Path,
    bringup_report: &ParameterGolfDistributed8xH100BringupReport,
    bootstrap_receipt_path: &Path,
    bootstrap_receipt: &ParameterGolfDistributed8xH100RuntimeBootstrapReceipt,
    train_step_receipt_path: &Path,
    train_step_receipt: &ParameterGolfDistributed8xH100TrainStepReceipt,
) -> Result<ParameterGolfDistributed8xH100CompletionReceipt, ParameterGolfSubmissionRuntimeError> {
    let validation = train_step_receipt
        .distributed_receipt
        .validation_aggregation
        .as_ref()
        .ok_or_else(|| ParameterGolfSubmissionRuntimeError::Consistency {
            message: String::from(
                "distributed completion receipt requires one validation-backed distributed receipt",
            ),
        })?;
    let final_model_artifact_path = root.join(&manifest.model_artifact_path);
    let final_model_artifact_bytes = fs::read(&final_model_artifact_path).map_err(|error| {
        ParameterGolfSubmissionRuntimeError::Read {
            path: final_model_artifact_path.display().to_string(),
            error,
        }
    })?;
    let submission_manifest_path = root.join(&manifest.submission_manifest_path);
    let submission_manifest_bytes = fs::read(&submission_manifest_path).map_err(|error| {
        ParameterGolfSubmissionRuntimeError::Read {
            path: submission_manifest_path.display().to_string(),
            error,
        }
    })?;
    let mut receipt = ParameterGolfDistributed8xH100CompletionReceipt {
        schema_version: 1,
        run_id: manifest.run_id.clone(),
        bringup_report_path: bringup_report_path.display().to_string(),
        bringup_report_digest: bringup_report.report_digest.clone(),
        runtime_bootstrap_receipt_path: bootstrap_receipt_path.display().to_string(),
        runtime_bootstrap_receipt_digest: bootstrap_receipt.receipt_digest.clone(),
        train_step_receipt_path: train_step_receipt_path.display().to_string(),
        train_step_receipt_digest: train_step_receipt.receipt_digest.clone(),
        distributed_receipt_path: train_step_receipt.distributed_receipt_path.clone(),
        distributed_receipt_digest: train_step_receipt.distributed_receipt.receipt_digest.clone(),
        final_model_artifact_path: final_model_artifact_path.display().to_string(),
        final_model_artifact_digest: sha256_bytes(final_model_artifact_bytes.as_slice()),
        final_model_artifact_size_bytes: final_model_artifact_bytes.len() as u64,
        submission_manifest_path: submission_manifest_path.display().to_string(),
        submission_manifest_digest: sha256_bytes(submission_manifest_bytes.as_slice()),
        distributed_validation_mean_loss: validation.mean_loss,
        distributed_validation_bits_per_byte: validation.bits_per_byte,
        distributed_validation_observed_ms: validation.observed_ms,
        claim_boundary: String::from(
            "This completion receipt proves the exported-folder distributed runtime finished the current 8xH100 execution contract with one measured distributed validation aggregation and bound that execution to the exact shipped final artifact identity already present in the folder. It does not by itself claim record-track promotion.",
        ),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.stable_digest();
    Ok(receipt)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn execute_parameter_golf_submission_distributed_8xh100_bootstrap_child(
    manifest: &ParameterGolfSubmissionRuntimeManifest,
) -> Result<
    ParameterGolfDistributed8xH100RuntimeBootstrapRankReceipt,
    ParameterGolfSubmissionRuntimeError,
> {
    Ok(execute_parameter_golf_distributed_8xh100_runtime_bootstrap_child(&manifest.run_id)?)
}

fn execute_parameter_golf_submission_distributed_8xh100_train_step_child(
    manifest: &ParameterGolfSubmissionRuntimeManifest,
) -> Result<ParameterGolfDistributed8xH100TrainStepRankReceipt, ParameterGolfSubmissionRuntimeError>
{
    Ok(execute_parameter_golf_distributed_8xh100_train_step_child(
        &manifest.run_id,
    )?)
}

fn execute_parameter_golf_submission_distributed_8xh100_validation_child(
    manifest: &ParameterGolfSubmissionRuntimeManifest,
) -> Result<
    ParameterGolfDistributed8xH100ValidationRankReceipt,
    ParameterGolfSubmissionRuntimeError,
> {
    Ok(execute_parameter_golf_distributed_8xh100_validation_child(
        &manifest.run_id,
    )?)
}

fn metric_matches(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-9
}

fn read_json<T: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
    artifact_kind: &'static str,
) -> Result<T, ParameterGolfSubmissionRuntimeError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| ParameterGolfSubmissionRuntimeError::Read {
        path: path.display().to_string(),
        error,
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        ParameterGolfSubmissionRuntimeError::Deserialize {
            artifact_kind: String::from(artifact_kind),
            path: path.display().to_string(),
            error,
        }
    })
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

/// Returns the committed runtime payload path used by the current non-record package.
#[must_use]
pub fn parameter_golf_submission_runtime_payload_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("repo root should resolve from psionic-train crate dir")
        .join("fixtures/parameter_golf/runtime/parameter_golf_submission_runtime.x86_64-unknown-linux-gnu")
}

use std::{
    fmt::Write,
    fs,
    path::{Component, Path},
};

use psionic_eval::PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{ParameterGolfLocalReferenceBenchmarkBundle, ParameterGolfTrainingArtifact};

/// Stable version identifier for the first honest non-record submission package.
pub const PARAMETER_GOLF_NON_RECORD_SUBMISSION_VERSION: &str =
    "2026.03.18.non_record_submission.v1";

/// Public non-record track identifier used by Parameter Golf.
pub const PARAMETER_GOLF_NON_RECORD_TRACK_ID: &str = "non-record-unlimited-compute-16mb";

/// Public artifact cap in decimal bytes.
pub const PARAMETER_GOLF_SUBMISSION_ARTIFACT_CAP_BYTES: u64 = 16_000_000;

/// Explicit claim boundary for the first non-record submission package.
pub const PARAMETER_GOLF_NON_RECORD_SUBMISSION_CLAIM_BOUNDARY: &str =
    "first honest non-record submission package only; the shipped train_gpt.py is a review wrapper over preserved Psionic artifacts and explicit counted-byte accounting, not a record-track runtime claim";

const PARAMETER_GOLF_NON_RECORD_RECORDS_DIR: &str = "records/track_non_record_16mb";
const PARAMETER_GOLF_ACCOUNTING_COMPONENT_ENTRYPOINT: &str = "entrypoint_code_bytes";
const PARAMETER_GOLF_ACCOUNTING_COMPONENT_MODEL: &str = "compressed_model_bytes";
const PARAMETER_GOLF_ACCOUNTING_COMPONENT_RUNTIME: &str = "shipped_runtime_code_bytes";
const PARAMETER_GOLF_ACCOUNTING_COMPONENT_WRAPPER: &str = "shipped_wrapper_code_bytes";
const PARAMETER_GOLF_ACCOUNTING_COMPONENT_BUILD_DEPS: &str = "required_build_dependency_bytes";

/// Human-facing identity fields for one submission package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfSubmissionIdentity {
    /// Submission author label.
    pub author: String,
    /// Public GitHub identifier.
    pub github_id: String,
    /// Run name surfaced in `submission.json`.
    pub name: String,
    /// Human-readable one-paragraph summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blurb: Option<String>,
    /// ISO-8601 UTC date string.
    pub date: String,
}

impl ParameterGolfSubmissionIdentity {
    /// Returns a repo-owned default identity for the first Psionic package lane.
    #[must_use]
    pub fn psionic_local_reference_defaults() -> Self {
        Self {
            author: String::from("Psionic"),
            github_id: String::from("OpenAgentsInc"),
            name: String::from("Psionic Local-Reference Review"),
            blurb: None,
            date: String::from("2026-03-18T00:00:00Z"),
        }
    }
}

/// Config for one generated non-record submission package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfNonRecordSubmissionConfig {
    /// Stable submission folder identifier.
    pub submission_id: String,
    /// Human-facing identity information.
    pub identity: ParameterGolfSubmissionIdentity,
}

impl ParameterGolfNonRecordSubmissionConfig {
    /// Returns the canonical repo-owned config for the first package lane.
    #[must_use]
    pub fn local_reference_defaults() -> Self {
        Self {
            submission_id: String::from("2026-03-18_psionic_local_reference_review_v1"),
            identity: ParameterGolfSubmissionIdentity::psionic_local_reference_defaults(),
        }
    }

    fn validate(&self) -> Result<(), ParameterGolfSubmissionError> {
        if self.submission_id.trim().is_empty() {
            return Err(ParameterGolfSubmissionError::InvalidConfig {
                message: String::from("submission_id must not be empty"),
            });
        }
        validate_relative_path(self.submission_id.as_str())?;
        if self.identity.author.trim().is_empty() {
            return Err(ParameterGolfSubmissionError::InvalidConfig {
                message: String::from("identity.author must not be empty"),
            });
        }
        if self.identity.github_id.trim().is_empty() {
            return Err(ParameterGolfSubmissionError::InvalidConfig {
                message: String::from("identity.github_id must not be empty"),
            });
        }
        if self.identity.name.trim().is_empty() {
            return Err(ParameterGolfSubmissionError::InvalidConfig {
                message: String::from("identity.name must not be empty"),
            });
        }
        if self.identity.date.trim().is_empty() {
            return Err(ParameterGolfSubmissionError::InvalidConfig {
                message: String::from("identity.date must not be empty"),
            });
        }
        Ok(())
    }
}

/// One counted component in the submission artifact accounting receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfSubmissionAccountingComponent {
    /// Stable component identifier aligned to the accounting contract.
    pub component_id: String,
    /// Counted size in bytes.
    pub size_bytes: u64,
    /// Honest detail about what is or is not shipped for this component.
    pub detail: String,
}

/// Machine-readable counted-byte accounting for one submission package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfSubmissionAccountingReceipt {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable package version.
    pub package_version: String,
    /// Submission track identifier.
    pub track: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Canonical benchmark reference.
    pub benchmark_ref: String,
    /// Top-level entrypoint path.
    pub entrypoint_path: String,
    /// Ordered counted components.
    pub counted_components: Vec<ParameterGolfSubmissionAccountingComponent>,
    /// Total counted code bytes.
    pub counted_code_bytes: u64,
    /// Counted compressed-model bytes.
    pub compressed_model_bytes: u64,
    /// Total counted submission bytes.
    pub total_counted_bytes: u64,
    /// Whether the total stayed inside the public `16,000,000` byte cap.
    pub within_artifact_cap: bool,
    /// Explicit claim boundary for the package.
    pub claim_boundary: String,
    /// Stable digest over the accounting receipt.
    pub receipt_digest: String,
}

impl ParameterGolfSubmissionAccountingReceipt {
    fn new(
        run_id: impl Into<String>,
        entrypoint_path: impl Into<String>,
        counted_components: Vec<ParameterGolfSubmissionAccountingComponent>,
    ) -> Self {
        let run_id = run_id.into();
        let entrypoint_path = entrypoint_path.into();
        let compressed_model_bytes = counted_components
            .iter()
            .find(|component| component.component_id == PARAMETER_GOLF_ACCOUNTING_COMPONENT_MODEL)
            .map_or(0, |component| component.size_bytes);
        let counted_code_bytes = counted_components
            .iter()
            .filter(|component| component.component_id != PARAMETER_GOLF_ACCOUNTING_COMPONENT_MODEL)
            .map(|component| component.size_bytes)
            .sum();
        let total_counted_bytes = counted_components
            .iter()
            .map(|component| component.size_bytes)
            .sum();
        let mut receipt = Self {
            schema_version: 1,
            package_version: String::from(PARAMETER_GOLF_NON_RECORD_SUBMISSION_VERSION),
            track: String::from(PARAMETER_GOLF_NON_RECORD_TRACK_ID),
            run_id,
            benchmark_ref: String::from(PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF),
            entrypoint_path,
            counted_components,
            counted_code_bytes,
            compressed_model_bytes,
            total_counted_bytes,
            within_artifact_cap: total_counted_bytes <= PARAMETER_GOLF_SUBMISSION_ARTIFACT_CAP_BYTES,
            claim_boundary: String::from(PARAMETER_GOLF_NON_RECORD_SUBMISSION_CLAIM_BOUNDARY),
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
            b"psionic_parameter_golf_submission_accounting_receipt|",
            &digestible,
        )
    }
}

/// Submission-facing `submission.json` contract for the non-record package lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfNonRecordSubmissionManifest {
    /// Human-facing author label.
    pub author: String,
    /// Public GitHub identifier.
    pub github_id: String,
    /// Human-facing run name.
    pub name: String,
    /// Submission summary.
    pub blurb: String,
    /// ISO-8601 UTC date string.
    pub date: String,
    /// Public non-record track identifier.
    pub track: String,
    /// Final roundtrip validation loss.
    pub val_loss: f64,
    /// Final roundtrip validation bits-per-byte.
    pub val_bpb: f64,
    /// Pre-quantized validation loss.
    pub pre_quant_val_loss: f64,
    /// Pre-quantized validation bits-per-byte.
    pub pre_quant_val_bpb: f64,
    /// Final completed training step.
    pub step_stop: u64,
    /// Observed training wallclock in seconds.
    pub wallclock_seconds: f64,
    /// Total counted submission bytes.
    pub bytes_total: u64,
    /// Counted compressed-model bytes.
    pub bytes_model_int8_zlib: u64,
    /// Counted code bytes.
    pub bytes_code: u64,
    /// Stable run identifier.
    pub run_id: String,
    /// Canonical benchmark reference.
    pub benchmark_ref: String,
    /// Canonical claim posture for the package.
    pub claim_posture: String,
    /// Explicit claim boundary for the package.
    pub claim_boundary: String,
    /// Top-level wrapper entrypoint path.
    pub entrypoint: String,
    /// Accounting receipt artifact path.
    pub accounting_receipt_artifact_ref: String,
    /// Accounting receipt artifact digest.
    pub accounting_receipt_artifact_digest: String,
    /// Benchmark receipt artifact path.
    pub benchmark_receipt_artifact_ref: String,
    /// Benchmark receipt artifact digest.
    pub benchmark_receipt_artifact_digest: String,
}

/// File role inside the generated submission folder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfSubmissionFileRole {
    Readme,
    SubmissionManifest,
    TrainLog,
    Entrypoint,
    CompressedModel,
    BenchmarkPackage,
    ChallengeScoreReport,
    BenchmarkReceipt,
    AccountingReceipt,
    RunBundle,
}

/// One file in the generated submission folder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfSubmissionFileEntry {
    /// Relative path from the submission root.
    pub relative_path: String,
    /// Stable file role.
    pub role: ParameterGolfSubmissionFileRole,
    /// Artifact kind used inside Psionic.
    pub artifact_kind: String,
    /// Stable artifact digest.
    pub artifact_digest: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Whether this file counts toward the public artifact cap.
    pub counts_toward_artifact_cap: bool,
    /// Honest file detail.
    pub detail: String,
}

/// Top-level folder contract for the generated non-record submission package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfNonRecordSubmissionPackage {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable package version.
    pub package_version: String,
    /// Public record-folder root expected by the challenge repo.
    pub record_folder_relpath: String,
    /// Stable submission folder identifier.
    pub submission_id: String,
    /// Submission track identifier.
    pub track: String,
    /// Canonical claim posture for the package.
    pub claim_posture: String,
    /// Explicit claim boundary for the package.
    pub claim_boundary: String,
    /// Canonical benchmark reference.
    pub benchmark_ref: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Top-level entrypoint path.
    pub entrypoint_path: String,
    /// Top-level `submission.json` path.
    pub submission_manifest_path: String,
    /// Accounting receipt path.
    pub accounting_receipt_path: String,
    /// Benchmark receipt path.
    pub benchmark_receipt_path: String,
    /// Ordered submission-folder files.
    pub files: Vec<ParameterGolfSubmissionFileEntry>,
    /// Stable digest over the package contract.
    pub package_digest: String,
}

impl ParameterGolfNonRecordSubmissionPackage {
    fn new(
        submission_id: impl Into<String>,
        run_id: impl Into<String>,
        accounting_receipt_path: impl Into<String>,
        benchmark_receipt_path: impl Into<String>,
        files: Vec<ParameterGolfSubmissionFileEntry>,
    ) -> Self {
        let submission_id = submission_id.into();
        let run_id = run_id.into();
        let accounting_receipt_path = accounting_receipt_path.into();
        let benchmark_receipt_path = benchmark_receipt_path.into();
        let mut package = Self {
            schema_version: 1,
            package_version: String::from(PARAMETER_GOLF_NON_RECORD_SUBMISSION_VERSION),
            record_folder_relpath: format!(
                "{}/{}",
                PARAMETER_GOLF_NON_RECORD_RECORDS_DIR, submission_id
            ),
            submission_id,
            track: String::from(PARAMETER_GOLF_NON_RECORD_TRACK_ID),
            claim_posture: String::from("non_record_submission"),
            claim_boundary: String::from(PARAMETER_GOLF_NON_RECORD_SUBMISSION_CLAIM_BOUNDARY),
            benchmark_ref: String::from(PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF),
            run_id,
            entrypoint_path: String::from("train_gpt.py"),
            submission_manifest_path: String::from("submission.json"),
            accounting_receipt_path,
            benchmark_receipt_path,
            files,
            package_digest: String::new(),
        };
        package.package_digest = package.stable_digest();
        package
    }

    /// Returns a stable digest over the package payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.package_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_non_record_submission_package|",
            &digestible,
        )
    }
}

/// In-memory bundle for one generated non-record submission package.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfNonRecordSubmissionBundle {
    /// Typed folder contract.
    pub package: ParameterGolfNonRecordSubmissionPackage,
    /// Top-level `submission.json` payload.
    pub submission_manifest: ParameterGolfNonRecordSubmissionManifest,
    /// Machine-readable counted-byte receipt.
    pub accounting_receipt: ParameterGolfSubmissionAccountingReceipt,
    /// All file artifacts that should be materialized into the submission root.
    pub file_artifacts: Vec<ParameterGolfTrainingArtifact>,
}

impl ParameterGolfNonRecordSubmissionBundle {
    /// Returns one file artifact by its relative path.
    #[must_use]
    pub fn artifact(&self, relative_path: &str) -> Option<&ParameterGolfTrainingArtifact> {
        self.file_artifacts
            .iter()
            .find(|artifact| artifact.artifact_ref == relative_path)
    }
}

/// Submission-package construction failure.
#[derive(Debug, Error)]
pub enum ParameterGolfSubmissionError {
    #[error("parameter golf submission config is invalid: {message}")]
    InvalidConfig { message: String },
    #[error("parameter golf submission artifact path is invalid: `{path}`")]
    InvalidArtifactPath { path: String },
    #[error("{context}: {message}")]
    Serialization {
        context: &'static str,
        message: String,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Builds the first honest non-record submission package from a local-reference benchmark bundle.
pub fn build_parameter_golf_non_record_submission_bundle(
    benchmark_bundle: &ParameterGolfLocalReferenceBenchmarkBundle,
    config: &ParameterGolfNonRecordSubmissionConfig,
) -> Result<ParameterGolfNonRecordSubmissionBundle, ParameterGolfSubmissionError> {
    config.validate()?;

    let run_id = benchmark_bundle.run_bundle.run_id.clone();
    let model_artifact = benchmark_bundle
        .training_outcome
        .int8_zlib_model_artifact
        .clone();
    validate_relative_path(model_artifact.artifact_ref.as_str())?;
    validate_relative_path(benchmark_bundle.benchmark_package_artifact.artifact_ref.as_str())?;
    validate_relative_path(
        benchmark_bundle
            .challenge_score_report_artifact
            .artifact_ref
            .as_str(),
    )?;
    validate_relative_path(benchmark_bundle.benchmark_receipt_artifact.artifact_ref.as_str())?;
    validate_relative_path(benchmark_bundle.run_bundle_artifact.artifact_ref.as_str())?;

    let entrypoint_artifact = text_artifact(
        "parameter_golf_submission_entrypoint",
        String::from("train_gpt.py"),
        render_entrypoint_wrapper(
            benchmark_bundle.benchmark_receipt_artifact.artifact_ref.as_str(),
            format!("{run_id}/benchmark/parameter_golf_submission_accounting.json").as_str(),
            model_artifact.artifact_ref.as_str(),
        ),
    );

    let accounting_receipt = ParameterGolfSubmissionAccountingReceipt::new(
        run_id.as_str(),
        entrypoint_artifact.artifact_ref.as_str(),
        vec![
            ParameterGolfSubmissionAccountingComponent {
                component_id: String::from(PARAMETER_GOLF_ACCOUNTING_COMPONENT_ENTRYPOINT),
                size_bytes: entrypoint_artifact.bytes.len() as u64,
                detail: String::from(
                    "the package ships one Python-stdlib train_gpt.py review wrapper at the submission root",
                ),
            },
            ParameterGolfSubmissionAccountingComponent {
                component_id: String::from(PARAMETER_GOLF_ACCOUNTING_COMPONENT_MODEL),
                size_bytes: model_artifact.bytes.len() as u64,
                detail: String::from(
                    "the counted model artifact is the final int8+zlib roundtrip export emitted by the Psionic local-reference lane",
                ),
            },
            ParameterGolfSubmissionAccountingComponent {
                component_id: String::from(PARAMETER_GOLF_ACCOUNTING_COMPONENT_RUNTIME),
                size_bytes: 0,
                detail: String::from(
                    "no Rust binary, vendored workspace, or separate runtime payload is shipped in this first non-record package",
                ),
            },
            ParameterGolfSubmissionAccountingComponent {
                component_id: String::from(PARAMETER_GOLF_ACCOUNTING_COMPONENT_WRAPPER),
                size_bytes: 0,
                detail: String::from(
                    "no helper wrapper files beyond the top-level train_gpt.py entrypoint are shipped in this package",
                ),
            },
            ParameterGolfSubmissionAccountingComponent {
                component_id: String::from(PARAMETER_GOLF_ACCOUNTING_COMPONENT_BUILD_DEPS),
                size_bytes: 0,
                detail: String::from(
                    "the review wrapper uses only the Python standard library and requires no build step or vendored dependency tree",
                ),
            },
        ],
    );
    let accounting_receipt_artifact = json_artifact(
        "parameter_golf_submission_accounting_receipt",
        format!("{run_id}/benchmark/parameter_golf_submission_accounting.json"),
        &accounting_receipt,
    )?;

    let step_stop = benchmark_bundle
        .training_outcome
        .step_metrics
        .last()
        .map_or(0, |metrics| metrics.global_step);
    let wallclock_seconds =
        benchmark_bundle.benchmark_receipt.wallclock_receipt.training_observed_ms as f64 / 1_000.0;
    let blurb = config.identity.blurb.clone().unwrap_or_else(|| {
        format!(
            "Psionic non-record submission package for the bounded local-reference lane; it preserves benchmark and accounting receipts, ships a train_gpt.py-shaped review wrapper, and reports the final int8+zlib roundtrip score under the same oracle and counted-byte rules."
        )
    });
    let submission_manifest = ParameterGolfNonRecordSubmissionManifest {
        author: config.identity.author.clone(),
        github_id: config.identity.github_id.clone(),
        name: config.identity.name.clone(),
        blurb,
        date: config.identity.date.clone(),
        track: String::from(PARAMETER_GOLF_NON_RECORD_TRACK_ID),
        val_loss: benchmark_bundle
            .challenge_score_report
            .int8_zlib_roundtrip_validation
            .mean_loss,
        val_bpb: benchmark_bundle
            .challenge_score_report
            .int8_zlib_roundtrip_validation
            .bits_per_byte,
        pre_quant_val_loss: benchmark_bundle.challenge_score_report.trained_validation.mean_loss,
        pre_quant_val_bpb: benchmark_bundle.challenge_score_report.trained_validation.bits_per_byte,
        step_stop,
        wallclock_seconds,
        bytes_total: accounting_receipt.total_counted_bytes,
        bytes_model_int8_zlib: accounting_receipt.compressed_model_bytes,
        bytes_code: accounting_receipt.counted_code_bytes,
        run_id: run_id.clone(),
        benchmark_ref: String::from(PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF),
        claim_posture: String::from("non_record_submission"),
        claim_boundary: String::from(PARAMETER_GOLF_NON_RECORD_SUBMISSION_CLAIM_BOUNDARY),
        entrypoint: entrypoint_artifact.artifact_ref.clone(),
        accounting_receipt_artifact_ref: accounting_receipt_artifact.artifact_ref.clone(),
        accounting_receipt_artifact_digest: accounting_receipt_artifact.artifact_digest.clone(),
        benchmark_receipt_artifact_ref: benchmark_bundle.benchmark_receipt_artifact.artifact_ref.clone(),
        benchmark_receipt_artifact_digest: benchmark_bundle
            .benchmark_receipt_artifact
            .artifact_digest
            .clone(),
    };
    let submission_manifest_artifact = json_artifact(
        "parameter_golf_submission_manifest",
        String::from("submission.json"),
        &submission_manifest,
    )?;
    let train_log_artifact = text_artifact(
        "parameter_golf_submission_train_log",
        String::from("train.log"),
        render_train_log(benchmark_bundle, &submission_manifest),
    );
    let readme_artifact = text_artifact(
        "parameter_golf_submission_readme",
        String::from("README.md"),
        render_readme(
            benchmark_bundle,
            &submission_manifest,
            &accounting_receipt,
            config.submission_id.as_str(),
        ),
    );

    let file_artifacts = vec![
        readme_artifact,
        submission_manifest_artifact,
        train_log_artifact,
        entrypoint_artifact,
        model_artifact,
        benchmark_bundle.benchmark_package_artifact.clone(),
        benchmark_bundle.challenge_score_report_artifact.clone(),
        benchmark_bundle.benchmark_receipt_artifact.clone(),
        accounting_receipt_artifact,
        benchmark_bundle.run_bundle_artifact.clone(),
    ];
    for artifact in &file_artifacts {
        validate_relative_path(artifact.artifact_ref.as_str())?;
    }

    let files = file_artifacts
        .iter()
        .map(|artifact| {
            let (role, counts_toward_artifact_cap, detail) =
                submission_role_and_detail(artifact.artifact_ref.as_str());
            ParameterGolfSubmissionFileEntry {
                relative_path: artifact.artifact_ref.clone(),
                role,
                artifact_kind: artifact.artifact_kind.clone(),
                artifact_digest: artifact.artifact_digest.clone(),
                size_bytes: artifact.bytes.len() as u64,
                counts_toward_artifact_cap,
                detail,
            }
        })
        .collect::<Vec<_>>();

    let package = ParameterGolfNonRecordSubmissionPackage::new(
        config.submission_id.as_str(),
        run_id,
        format!(
            "{}/benchmark/parameter_golf_submission_accounting.json",
            benchmark_bundle.run_bundle.run_id
        ),
        benchmark_bundle.benchmark_receipt_artifact.artifact_ref.as_str(),
        files,
    );

    Ok(ParameterGolfNonRecordSubmissionBundle {
        package,
        submission_manifest,
        accounting_receipt,
        file_artifacts,
    })
}

/// Writes the generated submission package to one folder root.
pub fn write_parameter_golf_non_record_submission_bundle(
    bundle: &ParameterGolfNonRecordSubmissionBundle,
    output_dir: &Path,
) -> Result<(), ParameterGolfSubmissionError> {
    fs::create_dir_all(output_dir)?;
    for artifact in &bundle.file_artifacts {
        validate_relative_path(artifact.artifact_ref.as_str())?;
        let path = output_dir.join(artifact.artifact_ref.as_str());
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, artifact.bytes.as_slice())?;
    }
    Ok(())
}

fn submission_role_and_detail(
    relative_path: &str,
) -> (ParameterGolfSubmissionFileRole, bool, String) {
    match relative_path {
        "README.md" => (
            ParameterGolfSubmissionFileRole::Readme,
            false,
            String::from("human-readable non-record submission overview"),
        ),
        "submission.json" => (
            ParameterGolfSubmissionFileRole::SubmissionManifest,
            false,
            String::from("leaderboard-facing metadata plus explicit Psionic claim boundary"),
        ),
        "train.log" => (
            ParameterGolfSubmissionFileRole::TrainLog,
            false,
            String::from("preserved training-step log synthesized from the bounded Psionic run"),
        ),
        "train_gpt.py" => (
            ParameterGolfSubmissionFileRole::Entrypoint,
            true,
            String::from("top-level Python-stdlib review wrapper used as the counted entrypoint"),
        ),
        _ if relative_path.ends_with("final_model.int8.ptz") => (
            ParameterGolfSubmissionFileRole::CompressedModel,
            true,
            String::from("final int8+zlib model artifact counted toward the public byte cap"),
        ),
        _ if relative_path.ends_with("parameter_golf_benchmark_package.json") => (
            ParameterGolfSubmissionFileRole::BenchmarkPackage,
            false,
            String::from("preserved benchmark-package contract for offline review"),
        ),
        _ if relative_path.ends_with("parameter_golf_challenge_score_report.json") => (
            ParameterGolfSubmissionFileRole::ChallengeScoreReport,
            false,
            String::from("preserved final score report for the packaged run"),
        ),
        _ if relative_path.ends_with("parameter_golf_challenge_benchmark_receipt.json") => (
            ParameterGolfSubmissionFileRole::BenchmarkReceipt,
            false,
            String::from("preserved benchmark receipt for the packaged run"),
        ),
        _ if relative_path.ends_with("parameter_golf_submission_accounting.json") => (
            ParameterGolfSubmissionFileRole::AccountingReceipt,
            false,
            String::from("machine-readable counted-byte accounting receipt"),
        ),
        _ => (
            ParameterGolfSubmissionFileRole::RunBundle,
            false,
            String::from("preserved run-bundle artifact tying train and eval evidence together"),
        ),
    }
}

fn render_entrypoint_wrapper(
    benchmark_receipt_ref: &str,
    accounting_receipt_ref: &str,
    model_artifact_ref: &str,
) -> String {
    format!(
        r#"#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECEIPT_PATH = ROOT / "{benchmark_receipt_ref}"
ACCOUNTING_PATH = ROOT / "{accounting_receipt_ref}"
MODEL_PATH = ROOT / "{model_artifact_ref}"

submission = json.loads((ROOT / "submission.json").read_text())
receipt = json.loads(RECEIPT_PATH.read_text())
accounting = json.loads(ACCOUNTING_PATH.read_text())

if submission["bytes_total"] != accounting["total_counted_bytes"]:
    raise SystemExit("submission.json bytes_total does not match the accounting receipt")
if submission["bytes_code"] != accounting["counted_code_bytes"]:
    raise SystemExit("submission.json bytes_code does not match the accounting receipt")
if submission["val_bpb"] != receipt["score_report"]["int8_zlib_roundtrip_validation"]["bits_per_byte"]:
    raise SystemExit("submission.json val_bpb does not match the benchmark receipt")
if submission["val_loss"] != receipt["score_report"]["int8_zlib_roundtrip_validation"]["mean_loss"]:
    raise SystemExit("submission.json val_loss does not match the benchmark receipt")
if MODEL_PATH.stat().st_size != submission["bytes_model_int8_zlib"]:
    raise SystemExit("compressed model size does not match submission.json")

trained = receipt["score_report"]["trained_validation"]
roundtrip = receipt["score_report"]["int8_zlib_roundtrip_validation"]

print("psionic_non_record_submission_review track=" + submission["track"] + " run_id=" + submission["run_id"])
print(
    "trained_validation val_loss:{{:.8f}} val_bpb:{{:.8f}}".format(
        trained["mean_loss"], trained["bits_per_byte"]
    )
)
print(
    "final_int8_zlib_roundtrip_exact val_loss:{{:.8f}} val_bpb:{{:.8f}}".format(
        roundtrip["mean_loss"], roundtrip["bits_per_byte"]
    )
)
print(
    "artifact_accounting bytes_total={{}} bytes_code={{}} bytes_model_int8_zlib={{}}".format(
        submission["bytes_total"],
        submission["bytes_code"],
        submission["bytes_model_int8_zlib"],
    )
)
print("claim_boundary=" + submission["claim_boundary"])
"#,
        benchmark_receipt_ref = benchmark_receipt_ref,
        accounting_receipt_ref = accounting_receipt_ref,
        model_artifact_ref = model_artifact_ref,
    )
}

fn render_train_log(
    benchmark_bundle: &ParameterGolfLocalReferenceBenchmarkBundle,
    submission_manifest: &ParameterGolfNonRecordSubmissionManifest,
) -> String {
    let mut log = String::new();
    let _ = writeln!(
        log,
        "psionic_parameter_golf_submission run_id={} track={} benchmark_ref={}",
        submission_manifest.run_id, submission_manifest.track, submission_manifest.benchmark_ref
    );
    for step in &benchmark_bundle.training_outcome.step_metrics {
        let _ = writeln!(
            log,
            "step={} mean_microbatch_loss={:.8} validation_loss={:.8} validation_bpb={:.8} learning_rate_multiplier={:.8} muon_momentum={:.8}",
            step.global_step,
            step.mean_microbatch_loss,
            step.validation_mean_loss,
            step.validation_bits_per_byte,
            step.learning_rate_multiplier,
            step.muon_momentum,
        );
    }
    let _ = writeln!(
        log,
        "trained_validation_exact val_loss:{:.8} val_bpb:{:.8}",
        benchmark_bundle.challenge_score_report.trained_validation.mean_loss,
        benchmark_bundle.challenge_score_report.trained_validation.bits_per_byte,
    );
    let _ = writeln!(
        log,
        "final_int8_zlib_roundtrip_exact val_loss:{:.8} val_bpb:{:.8}",
        benchmark_bundle
            .challenge_score_report
            .int8_zlib_roundtrip_validation
            .mean_loss,
        benchmark_bundle
            .challenge_score_report
            .int8_zlib_roundtrip_validation
            .bits_per_byte,
    );
    log
}

fn render_readme(
    benchmark_bundle: &ParameterGolfLocalReferenceBenchmarkBundle,
    submission_manifest: &ParameterGolfNonRecordSubmissionManifest,
    accounting_receipt: &ParameterGolfSubmissionAccountingReceipt,
    submission_id: &str,
) -> String {
    let mut readme = String::new();
    let _ = writeln!(
        readme,
        "This record captures a Psionic-owned non-record submission package for the bounded local-reference Parameter Golf lane.\n"
    );
    let _ = writeln!(
        readme,
        "This package is intentionally **not** a record-track runtime claim. The shipped `train_gpt.py` is a review wrapper that validates the preserved Psionic benchmark and accounting artifacts, then prints the final metric lines from the included run.\n"
    );
    let _ = writeln!(readme, "Configuration:");
    let _ = writeln!(
        readme,
        "- Track: `{}`",
        submission_manifest.track
    );
    let _ = writeln!(
        readme,
        "- Run ID: `{}`",
        submission_manifest.run_id
    );
    let _ = writeln!(
        readme,
        "- Claim posture: `{}`",
        submission_manifest.claim_posture
    );
    let _ = writeln!(
        readme,
        "- Claim boundary: `{}`",
        submission_manifest.claim_boundary
    );
    let _ = writeln!(
        readme,
        "- Wrapper entrypoint: `{}`",
        submission_manifest.entrypoint
    );
    let _ = writeln!(
        readme,
        "- Validation oracle: `{}`\n",
        submission_manifest.benchmark_ref
    );
    let _ = writeln!(readme, "Key metrics:");
    let _ = writeln!(
        readme,
        "- Trained validation: `val_loss = {:.8}`, `val_bpb = {:.8}`",
        benchmark_bundle.challenge_score_report.trained_validation.mean_loss,
        benchmark_bundle.challenge_score_report.trained_validation.bits_per_byte,
    );
    let _ = writeln!(
        readme,
        "- Final int8+zlib roundtrip: `val_loss = {:.8}`, `val_bpb = {:.8}`",
        submission_manifest.val_loss, submission_manifest.val_bpb
    );
    let _ = writeln!(
        readme,
        "- Training wallclock: `{:.3}s`",
        submission_manifest.wallclock_seconds
    );
    let _ = writeln!(
        readme,
        "- Counted bytes: `{}` total = `{}` code + `{}` compressed model\n",
        submission_manifest.bytes_total,
        submission_manifest.bytes_code,
        submission_manifest.bytes_model_int8_zlib,
    );
    let _ = writeln!(readme, "Artifact accounting:");
    for component in &accounting_receipt.counted_components {
        let _ = writeln!(
            readme,
            "- `{}`: `{}` bytes; {}",
            component.component_id, component.size_bytes, component.detail
        );
    }
    let _ = writeln!(
        readme,
        "- Within 16,000,000-byte cap: `{}`\n",
        accounting_receipt.within_artifact_cap
    );
    let _ = writeln!(readme, "Included files:");
    let _ = writeln!(readme, "- `README.md`");
    let _ = writeln!(readme, "- `submission.json`");
    let _ = writeln!(readme, "- `train.log`");
    let _ = writeln!(readme, "- `train_gpt.py`");
    let _ = writeln!(
        readme,
        "- `{}`",
        benchmark_bundle
            .training_outcome
            .int8_zlib_model_artifact
            .artifact_ref
    );
    let _ = writeln!(
        readme,
        "- `{}`",
        benchmark_bundle.benchmark_package_artifact.artifact_ref
    );
    let _ = writeln!(
        readme,
        "- `{}`",
        benchmark_bundle.challenge_score_report_artifact.artifact_ref
    );
    let _ = writeln!(
        readme,
        "- `{}`",
        benchmark_bundle.benchmark_receipt_artifact.artifact_ref
    );
    let _ = writeln!(
        readme,
        "- `{}/benchmark/parameter_golf_submission_accounting.json`",
        submission_manifest.run_id
    );
    let _ = writeln!(
        readme,
        "- `{}`\n",
        benchmark_bundle.run_bundle_artifact.artifact_ref
    );
    let _ = writeln!(
        readme,
        "The package root for challenge-repo publication is `{}/{}`.",
        PARAMETER_GOLF_NON_RECORD_RECORDS_DIR,
        submission_id
    );
    readme
}

fn text_artifact(
    artifact_kind: &'static str,
    artifact_ref: String,
    text: String,
) -> ParameterGolfTrainingArtifact {
    ParameterGolfTrainingArtifact::new(artifact_kind, artifact_ref, text.into_bytes())
}

fn json_artifact<T: Serialize>(
    artifact_kind: &'static str,
    artifact_ref: String,
    value: &T,
) -> Result<ParameterGolfTrainingArtifact, ParameterGolfSubmissionError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        ParameterGolfSubmissionError::Serialization {
            context: "parameter golf submission artifact serialization",
            message: error.to_string(),
        }
    })?;
    Ok(ParameterGolfTrainingArtifact::new(
        artifact_kind,
        artifact_ref,
        bytes,
    ))
}

fn validate_relative_path(path: &str) -> Result<(), ParameterGolfSubmissionError> {
    if path.trim().is_empty() {
        return Err(ParameterGolfSubmissionError::InvalidArtifactPath {
            path: String::from(path),
        });
    }
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        return Err(ParameterGolfSubmissionError::InvalidArtifactPath {
            path: String::from(path),
        });
    }
    for component in candidate.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            _ => {
                return Err(ParameterGolfSubmissionError::InvalidArtifactPath {
                    path: String::from(path),
                });
            }
        }
    }
    Ok(())
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
    use std::{error::Error, fs};

    use super::{
        PARAMETER_GOLF_NON_RECORD_SUBMISSION_VERSION, PARAMETER_GOLF_NON_RECORD_TRACK_ID,
        build_parameter_golf_non_record_submission_bundle,
        write_parameter_golf_non_record_submission_bundle,
    };
    use crate::{
        ParameterGolfLocalReferenceFixture, ParameterGolfNonRecordSubmissionConfig,
        ParameterGolfReferenceTrainingConfig, benchmark_parameter_golf_local_reference,
    };

    #[test]
    fn parameter_golf_non_record_submission_bundle_is_machine_readable_and_writes_folder_contract()
    -> Result<(), Box<dyn Error>> {
        let fixture = ParameterGolfLocalReferenceFixture::reference()?;
        let config = ParameterGolfReferenceTrainingConfig::local_reference();
        let benchmark_bundle = benchmark_parameter_golf_local_reference(&fixture, &config)?;
        let submission_bundle = build_parameter_golf_non_record_submission_bundle(
            &benchmark_bundle,
            &ParameterGolfNonRecordSubmissionConfig::local_reference_defaults(),
        )?;

        assert_eq!(
            submission_bundle.package.package_version,
            PARAMETER_GOLF_NON_RECORD_SUBMISSION_VERSION
        );
        assert_eq!(submission_bundle.package.track, PARAMETER_GOLF_NON_RECORD_TRACK_ID);
        assert_eq!(submission_bundle.package.claim_posture, "non_record_submission");
        assert_eq!(submission_bundle.package.entrypoint_path, "train_gpt.py");
        assert!(submission_bundle.accounting_receipt.within_artifact_cap);
        assert!(submission_bundle
            .submission_manifest
            .bytes_total
            == submission_bundle.accounting_receipt.total_counted_bytes);
        assert!(submission_bundle
            .artifact("README.md")
            .is_some_and(|artifact| !artifact.bytes.is_empty()));
        assert!(submission_bundle
            .artifact("submission.json")
            .is_some_and(|artifact| !artifact.bytes.is_empty()));
        assert!(submission_bundle
            .artifact("train.log")
            .is_some_and(|artifact| !artifact.bytes.is_empty()));
        assert!(submission_bundle
            .artifact("train_gpt.py")
            .is_some_and(|artifact| !artifact.bytes.is_empty()));
        assert!(submission_bundle.package.files.iter().any(|file| {
            file.relative_path.ends_with("final_model.int8.ptz") && file.counts_toward_artifact_cap
        }));

        let temp_dir = tempfile::tempdir()?;
        write_parameter_golf_non_record_submission_bundle(
            &submission_bundle,
            temp_dir.path(),
        )?;
        assert!(temp_dir.path().join("README.md").is_file());
        assert!(temp_dir.path().join("submission.json").is_file());
        assert!(temp_dir.path().join("train.log").is_file());
        assert!(temp_dir.path().join("train_gpt.py").is_file());
        assert!(temp_dir
            .path()
            .join(format!(
                "{}/benchmark/parameter_golf_challenge_benchmark_receipt.json",
                benchmark_bundle.run_bundle.run_id
            ))
            .is_file());
        let submission_json = fs::read_to_string(temp_dir.path().join("submission.json"))?;
        assert!(submission_json.contains("\"track\": \"non-record-unlimited-compute-16mb\""));
        let readme = fs::read_to_string(temp_dir.path().join("README.md"))?;
        assert!(readme.contains("non-record submission package"));
        Ok(())
    }
}

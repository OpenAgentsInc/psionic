use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Pylon legal training job spec schema.
pub const QWEN_LEGAL_PYLON_TRAINING_JOB_SCHEMA_VERSION: &str =
    "psionic.qwen_legal_pylon_training_job.v1";
/// Pylon legal training worker receipt schema.
pub const QWEN_LEGAL_PYLON_WORKER_RECEIPT_SCHEMA_VERSION: &str =
    "psionic.qwen_legal_pylon_worker_receipt.v1";
/// Local worker implementation id.
pub const QWEN_LEGAL_PYLON_LOCAL_WORKER_IMPL_ID: &str = "psionic.qwen_legal_pylon.local_worker.v1";

/// Pylon legal training job kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PylonTrainingJobKind {
    /// Build one dataset shard manifest.
    DatasetShardBuild,
    /// Run SFT over one assigned shard.
    SftTrainShard,
    /// Run DPO over one assigned shard.
    DpoTrainShard,
    /// Sample a GRPO rollout group.
    GrpoSampleBatch,
    /// Train GRPO over one assigned shard.
    GrpoTrainShard,
    /// Evaluate one assigned shard.
    EvalShard,
    /// Merge adapter deltas or LoRA adapters.
    AdapterMerge,
    /// Verify artifact hashes and receipt shape.
    ArtifactVerify,
}

/// Stable artifact reference used in job specs and receipts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingArtifactRef {
    /// Stable artifact id.
    pub artifact_id: String,
    /// Logical artifact type.
    pub artifact_type: String,
    /// Local path available to the worker.
    pub path: String,
    /// Expected SHA-256 hex digest.
    pub sha256: String,
}

/// Expected output artifact declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingExpectedOutputArtifact {
    /// Stable artifact id.
    pub artifact_id: String,
    /// Logical artifact type.
    pub artifact_type: String,
    /// Local path the worker must write.
    pub path: String,
    /// Whether the output is required for job success.
    pub required: bool,
}

/// Output artifact retained in a worker receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingOutputArtifactRef {
    /// Stable artifact id.
    pub artifact_id: String,
    /// Logical artifact type.
    pub artifact_type: String,
    /// Local artifact path.
    pub path: String,
    /// Observed SHA-256 hex digest.
    pub sha256: String,
    /// Observed byte length.
    pub byte_len: u64,
}

/// Shard assignment for one Pylon job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingShardAssignment {
    /// Stable assignment id.
    pub assignment_id: String,
    /// Stable shard id.
    pub shard_id: String,
    /// Zero-based shard index.
    pub shard_index: u32,
    /// Total shard count.
    pub shard_count: u32,
    /// Optional start row or task index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_index: Option<u64>,
    /// Optional exclusive end row or task index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_index: Option<u64>,
}

/// Hardware requirements declared by the job issuer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingHardwareRequirements {
    /// Minimum memory in bytes.
    pub min_memory_bytes: u64,
    /// Whether the job requires a GPU or accelerator.
    pub require_accelerator: bool,
    /// Accepted backend labels.
    pub accepted_backend_labels: Vec<String>,
}

/// Payment and budget metadata carried with the job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingPaymentBudget {
    /// Stable budget id.
    pub budget_id: String,
    /// Max authorized runtime spend in micro-USD.
    pub max_cost_microusd: u64,
    /// Payment account or ledger reference.
    pub payment_account_ref: String,
}

/// Receipt requirements requested by the job issuer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingReceiptRequirements {
    /// Whether the receipt must include an Ed25519 signature.
    pub require_signature: bool,
    /// Whether the receipt must include a logs hash.
    pub require_logs_hash: bool,
    /// Whether the receipt must include metrics.
    pub require_metrics: bool,
    /// Output artifact types that must be present on success.
    pub required_output_artifact_types: Vec<String>,
}

/// Full Pylon legal training job spec.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingJobSpec {
    /// Schema version.
    pub schema_version: String,
    /// Stable job id.
    pub job_id: String,
    /// Parent run id.
    pub parent_run_id: String,
    /// Job kind.
    pub job_kind: PylonTrainingJobKind,
    /// Model id.
    pub model_id: String,
    /// Model hash.
    pub model_hash: String,
    /// Optional input adapter id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    /// Optional input adapter hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_hash: Option<String>,
    /// Dataset manifest hash.
    pub dataset_manifest_hash: String,
    /// Shard assignment.
    pub shard_assignment: PylonTrainingShardAssignment,
    /// Training config hash.
    pub training_config_hash: String,
    /// Expected input artifacts.
    pub expected_input_artifacts: Vec<PylonTrainingArtifactRef>,
    /// Expected output artifacts.
    pub expected_output_artifacts: Vec<PylonTrainingExpectedOutputArtifact>,
    /// Max runtime in milliseconds.
    pub max_runtime_ms: u64,
    /// Hardware requirements.
    pub hardware_requirements: PylonTrainingHardwareRequirements,
    /// Payment and budget metadata.
    pub payment_budget: PylonTrainingPaymentBudget,
    /// Receipt requirements.
    pub receipt_requirements: PylonTrainingReceiptRequirements,
    /// Worker output directory.
    pub output_dir: String,
    /// Receipt output path.
    pub receipt_path: String,
}

impl PylonTrainingJobSpec {
    /// Stable digest over the job spec.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_json_digest(b"psionic_qwen_legal_pylon_training_job|", self)
    }

    /// Validates required fields.
    pub fn validate(&self) -> Result<(), QwenLegalPylonTrainingJobError> {
        if self.schema_version != QWEN_LEGAL_PYLON_TRAINING_JOB_SCHEMA_VERSION {
            return invalid_job("schema version drifted");
        }
        require_nonempty(self.job_id.as_str(), "job_id")?;
        require_nonempty(self.parent_run_id.as_str(), "parent_run_id")?;
        require_nonempty(self.model_id.as_str(), "model_id")?;
        require_nonempty(self.model_hash.as_str(), "model_hash")?;
        require_nonempty(self.dataset_manifest_hash.as_str(), "dataset_manifest_hash")?;
        require_nonempty(self.training_config_hash.as_str(), "training_config_hash")?;
        require_nonempty(self.output_dir.as_str(), "output_dir")?;
        require_nonempty(self.receipt_path.as_str(), "receipt_path")?;
        self.shard_assignment.validate()?;
        if self.expected_input_artifacts.is_empty() {
            return invalid_job("expected_input_artifacts must not be empty");
        }
        if self.expected_output_artifacts.is_empty() {
            return invalid_job("expected_output_artifacts must not be empty");
        }
        for artifact in &self.expected_input_artifacts {
            artifact.validate()?;
        }
        for artifact in &self.expected_output_artifacts {
            artifact.validate()?;
        }
        if self.max_runtime_ms == 0 {
            return invalid_job("max_runtime_ms must be non-zero");
        }
        self.hardware_requirements.validate()?;
        self.payment_budget.validate()?;
        self.receipt_requirements.validate()?;
        Ok(())
    }
}

impl PylonTrainingArtifactRef {
    fn validate(&self) -> Result<(), QwenLegalPylonTrainingJobError> {
        require_nonempty(self.artifact_id.as_str(), "artifact_id")?;
        require_nonempty(self.artifact_type.as_str(), "artifact_type")?;
        require_nonempty(self.path.as_str(), "path")?;
        require_nonempty(self.sha256.as_str(), "sha256")?;
        if self.sha256.len() != 64 || !self.sha256.chars().all(|value| value.is_ascii_hexdigit()) {
            return invalid_job("artifact sha256 must be 64 hex characters");
        }
        Ok(())
    }
}

impl PylonTrainingExpectedOutputArtifact {
    fn validate(&self) -> Result<(), QwenLegalPylonTrainingJobError> {
        require_nonempty(self.artifact_id.as_str(), "output artifact_id")?;
        require_nonempty(self.artifact_type.as_str(), "output artifact_type")?;
        require_nonempty(self.path.as_str(), "output path")
    }
}

impl PylonTrainingShardAssignment {
    fn validate(&self) -> Result<(), QwenLegalPylonTrainingJobError> {
        require_nonempty(self.assignment_id.as_str(), "assignment_id")?;
        require_nonempty(self.shard_id.as_str(), "shard_id")?;
        if self.shard_count == 0 || self.shard_index >= self.shard_count {
            return invalid_job("shard_index must be inside non-empty shard_count");
        }
        if self
            .start_index
            .zip(self.end_index)
            .is_some_and(|(start, end)| start >= end)
        {
            return invalid_job("start_index must be lower than end_index");
        }
        Ok(())
    }
}

impl PylonTrainingHardwareRequirements {
    fn validate(&self) -> Result<(), QwenLegalPylonTrainingJobError> {
        if self.min_memory_bytes == 0 {
            return invalid_job("min_memory_bytes must be non-zero");
        }
        if self.accepted_backend_labels.is_empty() {
            return invalid_job("accepted_backend_labels must not be empty");
        }
        Ok(())
    }
}

impl PylonTrainingPaymentBudget {
    fn validate(&self) -> Result<(), QwenLegalPylonTrainingJobError> {
        require_nonempty(self.budget_id.as_str(), "budget_id")?;
        require_nonempty(self.payment_account_ref.as_str(), "payment_account_ref")
    }
}

impl PylonTrainingReceiptRequirements {
    fn validate(&self) -> Result<(), QwenLegalPylonTrainingJobError> {
        if !self.require_signature {
            return invalid_job("receipt signature is required for Pylon training jobs");
        }
        if self.required_output_artifact_types.is_empty() {
            return invalid_job("required_output_artifact_types must not be empty");
        }
        Ok(())
    }
}

/// Job status retained in worker receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PylonTrainingWorkerJobStatus {
    /// Job completed and all required outputs were present.
    Succeeded,
    /// Job failed validation or output checks.
    Failed,
}

/// Hardware observed by a local worker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingWorkerHardwareSummary {
    /// Host id.
    pub host_id: String,
    /// OS family.
    pub os: String,
    /// Architecture.
    pub arch: String,
    /// Backend label used by this local worker.
    pub backend_label: String,
    /// Total memory estimate.
    pub total_memory_bytes: u64,
    /// Accelerator summary.
    pub accelerator: String,
}

/// Signed worker receipt for one Pylon legal training job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PylonTrainingWorkerReceipt {
    /// Schema version.
    pub schema_version: String,
    /// Worker implementation id.
    pub worker_impl_id: String,
    /// Worker id.
    pub worker_id: String,
    /// Worker Ed25519 public key.
    pub worker_pubkey: String,
    /// Job id.
    pub job_id: String,
    /// Parent run id.
    pub parent_run_id: String,
    /// Job kind.
    pub job_kind: PylonTrainingJobKind,
    /// Stable job spec digest.
    pub job_spec_digest: String,
    /// Input artifact hashes observed by the worker.
    pub input_hashes: Vec<PylonTrainingArtifactRef>,
    /// Output artifact hashes observed by the worker.
    pub output_hashes: Vec<PylonTrainingOutputArtifactRef>,
    /// Start time in milliseconds.
    pub started_at_ms: u64,
    /// End time in milliseconds.
    pub ended_at_ms: u64,
    /// Hardware summary.
    pub hardware_summary: PylonTrainingWorkerHardwareSummary,
    /// Psionic crate version.
    pub psionic_version: String,
    /// Git commit, if supplied by the packaged runtime.
    pub git_commit: String,
    /// Logs hash.
    pub logs_hash: String,
    /// Numeric worker metrics.
    pub metrics: BTreeMap<String, f64>,
    /// Job status.
    pub status: PylonTrainingWorkerJobStatus,
    /// Failure reason if status is failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
    /// Signature scheme.
    pub signature_scheme: String,
    /// Digest over the signable receipt payload.
    pub signed_payload_digest: String,
    /// Ed25519 signature over `signed_payload_digest`.
    pub signature_hex: String,
    /// Stable digest over the full receipt.
    pub receipt_digest: String,
}

impl PylonTrainingWorkerReceipt {
    /// Stable receipt digest with the digest field cleared.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.receipt_digest.clear();
        stable_json_digest(b"psionic_qwen_legal_pylon_worker_receipt|", &clone)
    }

    /// Stable digest over the signable payload.
    #[must_use]
    pub fn signable_payload_digest(&self) -> String {
        let mut clone = self.clone();
        clone.receipt_digest.clear();
        clone.signed_payload_digest.clear();
        clone.signature_hex.clear();
        stable_json_digest(b"psionic_qwen_legal_pylon_worker_receipt_signable|", &clone)
    }
}

/// Local worker options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PylonLocalWorkerRunOptions {
    /// Worker id.
    pub worker_id: String,
    /// Logical start time in milliseconds.
    pub started_at_ms: u64,
    /// Whether the local worker should materialize expected outputs.
    pub emit_outputs: bool,
}

impl Default for PylonLocalWorkerRunOptions {
    fn default() -> Self {
        Self {
            worker_id: String::from("pylon.local.qwen-legal.01"),
            started_at_ms: 10_000,
            emit_outputs: true,
        }
    }
}

/// Receipt verification result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PylonTrainingWorkerReceiptVerification {
    /// Receipt path.
    pub receipt_path: String,
    /// Job id.
    pub job_id: String,
    /// Worker id.
    pub worker_id: String,
    /// Job status.
    pub status: PylonTrainingWorkerJobStatus,
    /// Receipt digest.
    pub receipt_digest: String,
    /// Signed payload digest.
    pub signed_payload_digest: String,
    /// Whether the signature is valid.
    pub signature_valid: bool,
    /// Whether succeeded output files were rechecked.
    pub output_files_rechecked: bool,
}

/// Error for Pylon legal training job handling.
#[derive(Debug, Error)]
pub enum QwenLegalPylonTrainingJobError {
    /// Invalid job spec.
    #[error("invalid Pylon training job: {detail}")]
    InvalidJob { detail: String },
    /// Invalid worker receipt.
    #[error("invalid Pylon worker receipt: {detail}")]
    InvalidReceipt { detail: String },
    /// I/O failure.
    #[error("Pylon training job I/O failed at `{path}`: {message}")]
    Io { path: String, message: String },
    /// JSON failure.
    #[error("Pylon training job JSON failed at `{path}`: {message}")]
    Json { path: String, message: String },
    /// Serialization failure.
    #[error("Pylon training job serialization failed: {message}")]
    Serialization { message: String },
}

/// Runs one Pylon legal worker job from a JSON file.
pub fn run_qwen_legal_pylon_worker_job_path(
    path: impl AsRef<Path>,
) -> Result<PylonTrainingWorkerReceipt, QwenLegalPylonTrainingJobError> {
    let path = path.as_ref();
    let job = read_json::<PylonTrainingJobSpec>(path)?;
    run_qwen_legal_pylon_worker_job(&job, &PylonLocalWorkerRunOptions::default())
}

/// Runs one Pylon legal worker job.
pub fn run_qwen_legal_pylon_worker_job(
    job: &PylonTrainingJobSpec,
    options: &PylonLocalWorkerRunOptions,
) -> Result<PylonTrainingWorkerReceipt, QwenLegalPylonTrainingJobError> {
    job.validate()?;
    let output_dir = resolve_workspace_path(job.output_dir.as_str());
    fs::create_dir_all(&output_dir).map_err(|error| QwenLegalPylonTrainingJobError::Io {
        path: output_dir.display().to_string(),
        message: error.to_string(),
    })?;

    let signing_key =
        deterministic_worker_signing_key(options.worker_id.as_str(), job.job_id.as_str());
    let mut failure_reason = validate_inputs(job).err();
    if failure_reason.is_none() && options.emit_outputs {
        for output in &job.expected_output_artifacts {
            write_expected_output(job, output)?;
        }
    }
    let output_hashes = if failure_reason.is_none() {
        match collect_output_hashes(job) {
            Ok(outputs) => outputs,
            Err(error) => {
                failure_reason = Some(error);
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    let status = if failure_reason.is_some() {
        PylonTrainingWorkerJobStatus::Failed
    } else {
        PylonTrainingWorkerJobStatus::Succeeded
    };
    let input_hashes = observed_input_hashes(job);
    let ended_at_ms = options.started_at_ms.saturating_add(25);
    let logs = format!(
        "worker={} job={} status={:?} inputs={} outputs={}",
        options.worker_id,
        job.job_id,
        status,
        input_hashes.len(),
        output_hashes.len()
    );
    let mut metrics = BTreeMap::new();
    metrics.insert(
        String::from("input_artifact_count"),
        input_hashes.len() as f64,
    );
    metrics.insert(
        String::from("output_artifact_count"),
        output_hashes.len() as f64,
    );
    metrics.insert(
        String::from("runtime_ms"),
        ended_at_ms.saturating_sub(options.started_at_ms) as f64,
    );
    let mut receipt = PylonTrainingWorkerReceipt {
        schema_version: String::from(QWEN_LEGAL_PYLON_WORKER_RECEIPT_SCHEMA_VERSION),
        worker_impl_id: String::from(QWEN_LEGAL_PYLON_LOCAL_WORKER_IMPL_ID),
        worker_id: options.worker_id.clone(),
        worker_pubkey: hex::encode(signing_key.verifying_key().to_bytes()),
        job_id: job.job_id.clone(),
        parent_run_id: job.parent_run_id.clone(),
        job_kind: job.job_kind,
        job_spec_digest: job.stable_digest(),
        input_hashes,
        output_hashes,
        started_at_ms: options.started_at_ms,
        ended_at_ms,
        hardware_summary: local_hardware_summary(),
        psionic_version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: option_env!("PSIONIC_GIT_COMMIT")
            .unwrap_or("unknown-local")
            .to_string(),
        logs_hash: sha256_hex(logs.as_bytes()),
        metrics,
        status,
        failure_reason,
        signature_scheme: String::from("ed25519_detached_sha256_payload_v1"),
        signed_payload_digest: String::new(),
        signature_hex: String::new(),
        receipt_digest: String::new(),
    };
    sign_receipt(&mut receipt, &signing_key);
    write_json(Path::new(&job.receipt_path), &receipt)?;
    Ok(receipt)
}

/// Verifies one Pylon legal worker receipt from a JSON file.
pub fn verify_qwen_legal_pylon_worker_receipt_path(
    path: impl AsRef<Path>,
) -> Result<PylonTrainingWorkerReceiptVerification, QwenLegalPylonTrainingJobError> {
    let path = path.as_ref();
    let receipt = read_json::<PylonTrainingWorkerReceipt>(path)?;
    verify_qwen_legal_pylon_worker_receipt(&receipt, path)
}

/// Verifies one Pylon legal worker receipt.
pub fn verify_qwen_legal_pylon_worker_receipt(
    receipt: &PylonTrainingWorkerReceipt,
    receipt_path: &Path,
) -> Result<PylonTrainingWorkerReceiptVerification, QwenLegalPylonTrainingJobError> {
    if receipt.schema_version != QWEN_LEGAL_PYLON_WORKER_RECEIPT_SCHEMA_VERSION {
        return invalid_receipt("schema version drifted");
    }
    if receipt.worker_impl_id != QWEN_LEGAL_PYLON_LOCAL_WORKER_IMPL_ID {
        return invalid_receipt("unknown worker implementation id");
    }
    if receipt.receipt_digest != receipt.stable_digest() {
        return invalid_receipt("receipt digest drifted");
    }
    let signable_digest = receipt.signable_payload_digest();
    if receipt.signed_payload_digest != signable_digest {
        return invalid_receipt("signed payload digest drifted");
    }
    let verifying_key = decode_verifying_key(receipt.worker_pubkey.as_str())?;
    let signature = decode_signature(receipt.signature_hex.as_str())?;
    verifying_key
        .verify(receipt.signed_payload_digest.as_bytes(), &signature)
        .map_err(|_| QwenLegalPylonTrainingJobError::InvalidReceipt {
            detail: String::from("signature verification failed"),
        })?;
    if receipt.ended_at_ms < receipt.started_at_ms {
        return invalid_receipt("ended_at_ms is earlier than started_at_ms");
    }
    if receipt.status == PylonTrainingWorkerJobStatus::Succeeded {
        if receipt.failure_reason.is_some() {
            return invalid_receipt("succeeded receipt must not carry failure_reason");
        }
        if receipt.output_hashes.is_empty() {
            return invalid_receipt("succeeded receipt must carry output hashes");
        }
        for output in &receipt.output_hashes {
            let observed = sha256_file(output.path.as_str())?;
            if observed != output.sha256 {
                return invalid_receipt(format!(
                    "output artifact `{}` hash drifted",
                    output.artifact_id
                ));
            }
        }
    }
    if receipt.status == PylonTrainingWorkerJobStatus::Failed && receipt.failure_reason.is_none() {
        return invalid_receipt("failed receipt must carry failure_reason");
    }
    Ok(PylonTrainingWorkerReceiptVerification {
        receipt_path: receipt_path.display().to_string(),
        job_id: receipt.job_id.clone(),
        worker_id: receipt.worker_id.clone(),
        status: receipt.status,
        receipt_digest: receipt.receipt_digest.clone(),
        signed_payload_digest: receipt.signed_payload_digest.clone(),
        signature_valid: true,
        output_files_rechecked: receipt.status == PylonTrainingWorkerJobStatus::Succeeded,
    })
}

fn validate_inputs(job: &PylonTrainingJobSpec) -> Result<(), String> {
    for input in &job.expected_input_artifacts {
        let observed = sha256_file(input.path.as_str()).map_err(|error| error.to_string())?;
        if observed != input.sha256 {
            return Err(format!(
                "input artifact `{}` hash mismatch: expected {}, observed {}",
                input.artifact_id, input.sha256, observed
            ));
        }
    }
    Ok(())
}

fn observed_input_hashes(job: &PylonTrainingJobSpec) -> Vec<PylonTrainingArtifactRef> {
    job.expected_input_artifacts
        .iter()
        .map(|input| {
            let sha256 = sha256_file(input.path.as_str()).unwrap_or_else(|_| String::new());
            PylonTrainingArtifactRef {
                artifact_id: input.artifact_id.clone(),
                artifact_type: input.artifact_type.clone(),
                path: input.path.clone(),
                sha256,
            }
        })
        .collect()
}

fn collect_output_hashes(
    job: &PylonTrainingJobSpec,
) -> Result<Vec<PylonTrainingOutputArtifactRef>, String> {
    let mut output_hashes = Vec::new();
    for expected in &job.expected_output_artifacts {
        let path = resolve_workspace_path(expected.path.as_str());
        if !path.is_file() {
            if expected.required {
                return Err(format!(
                    "required output artifact `{}` was not written",
                    expected.artifact_id
                ));
            }
            continue;
        }
        let bytes = fs::read(&path).map_err(|error| error.to_string())?;
        output_hashes.push(PylonTrainingOutputArtifactRef {
            artifact_id: expected.artifact_id.clone(),
            artifact_type: expected.artifact_type.clone(),
            path: expected.path.clone(),
            sha256: sha256_hex(bytes.as_slice()),
            byte_len: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        });
    }
    for required_type in &job.receipt_requirements.required_output_artifact_types {
        if !output_hashes
            .iter()
            .any(|output| output.artifact_type == *required_type)
        {
            return Err(format!(
                "required output artifact type `{required_type}` missing"
            ));
        }
    }
    Ok(output_hashes)
}

fn write_expected_output(
    job: &PylonTrainingJobSpec,
    expected: &PylonTrainingExpectedOutputArtifact,
) -> Result<(), QwenLegalPylonTrainingJobError> {
    let path = resolve_workspace_path(expected.path.as_str());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| QwenLegalPylonTrainingJobError::Io {
            path: parent.display().to_string(),
            message: error.to_string(),
        })?;
    }
    let output = serde_json::json!({
        "schema_version": "psionic.qwen_legal_pylon_worker_output.v1",
        "producer": QWEN_LEGAL_PYLON_LOCAL_WORKER_IMPL_ID,
        "job_id": job.job_id,
        "parent_run_id": job.parent_run_id,
        "job_kind": job.job_kind,
        "artifact_id": expected.artifact_id,
        "artifact_type": expected.artifact_type,
        "model_id": job.model_id,
        "model_hash": job.model_hash,
        "dataset_manifest_hash": job.dataset_manifest_hash,
        "training_config_hash": job.training_config_hash,
        "shard_assignment": job.shard_assignment,
        "claim_boundary": "Local Pylon worker protocol smoke output. This proves job intake, hash checks, output materialization, and receipt signing; it does not claim live distributed Qwen training.",
    });
    let bytes = serde_json::to_vec_pretty(&output).map_err(|error| {
        QwenLegalPylonTrainingJobError::Serialization {
            message: error.to_string(),
        }
    })?;
    fs::write(&path, bytes).map_err(|error| QwenLegalPylonTrainingJobError::Io {
        path: expected.path.clone(),
        message: error.to_string(),
    })
}

fn sign_receipt(receipt: &mut PylonTrainingWorkerReceipt, signing_key: &SigningKey) {
    receipt.signed_payload_digest = receipt.signable_payload_digest();
    let signature = signing_key.sign(receipt.signed_payload_digest.as_bytes());
    receipt.signature_hex = hex::encode(signature.to_bytes());
    receipt.receipt_digest = receipt.stable_digest();
}

fn local_hardware_summary() -> PylonTrainingWorkerHardwareSummary {
    PylonTrainingWorkerHardwareSummary {
        host_id: std::env::var("HOSTNAME").unwrap_or_else(|_| String::from("localhost")),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        backend_label: String::from("local_protocol_smoke"),
        total_memory_bytes: 8 * 1024 * 1024 * 1024,
        accelerator: String::from("not_required_for_protocol_smoke"),
    }
}

fn deterministic_worker_signing_key(worker_id: &str, job_id: &str) -> SigningKey {
    let digest = Sha256::digest(format!("qwen-legal-pylon-worker|{worker_id}|{job_id}").as_bytes());
    let mut secret = [0_u8; 32];
    secret.copy_from_slice(&digest[..32]);
    SigningKey::from_bytes(&secret)
}

fn decode_verifying_key(
    public_key_hex: &str,
) -> Result<VerifyingKey, QwenLegalPylonTrainingJobError> {
    let bytes = hex::decode(public_key_hex).map_err(|error| {
        QwenLegalPylonTrainingJobError::InvalidReceipt {
            detail: format!("invalid worker public key hex: {error}"),
        }
    })?;
    let bytes: [u8; 32] =
        bytes
            .try_into()
            .map_err(|_| QwenLegalPylonTrainingJobError::InvalidReceipt {
                detail: String::from("worker public key must be 32 bytes"),
            })?;
    VerifyingKey::from_bytes(&bytes).map_err(|error| {
        QwenLegalPylonTrainingJobError::InvalidReceipt {
            detail: format!("invalid worker public key: {error}"),
        }
    })
}

fn decode_signature(signature_hex: &str) -> Result<Signature, QwenLegalPylonTrainingJobError> {
    let bytes = hex::decode(signature_hex).map_err(|error| {
        QwenLegalPylonTrainingJobError::InvalidReceipt {
            detail: format!("invalid signature hex: {error}"),
        }
    })?;
    let bytes: [u8; 64] =
        bytes
            .try_into()
            .map_err(|_| QwenLegalPylonTrainingJobError::InvalidReceipt {
                detail: String::from("signature must be 64 bytes"),
            })?;
    Ok(Signature::from_bytes(&bytes))
}

fn read_json<T: for<'de> Deserialize<'de>>(
    path: &Path,
) -> Result<T, QwenLegalPylonTrainingJobError> {
    let resolved = resolve_workspace_path(path);
    let bytes = fs::read(&resolved).map_err(|error| QwenLegalPylonTrainingJobError::Io {
        path: resolved.display().to_string(),
        message: error.to_string(),
    })?;
    serde_json::from_slice(bytes.as_slice()).map_err(|error| QwenLegalPylonTrainingJobError::Json {
        path: resolved.display().to_string(),
        message: error.to_string(),
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), QwenLegalPylonTrainingJobError> {
    let resolved = resolve_workspace_path(path);
    if let Some(parent) = resolved.parent() {
        fs::create_dir_all(parent).map_err(|error| QwenLegalPylonTrainingJobError::Io {
            path: parent.display().to_string(),
            message: error.to_string(),
        })?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        QwenLegalPylonTrainingJobError::Serialization {
            message: error.to_string(),
        }
    })?;
    fs::write(&resolved, bytes).map_err(|error| QwenLegalPylonTrainingJobError::Io {
        path: resolved.display().to_string(),
        message: error.to_string(),
    })
}

fn sha256_file(path: &str) -> Result<String, QwenLegalPylonTrainingJobError> {
    let resolved = resolve_workspace_path(path);
    let bytes = fs::read(&resolved).map_err(|error| QwenLegalPylonTrainingJobError::Io {
        path: resolved.display().to_string(),
        message: error.to_string(),
    })?;
    Ok(sha256_hex(bytes.as_slice()))
}

fn resolve_workspace_path(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    if path.is_absolute() {
        return path.to_path_buf();
    }
    workspace_root().join(path)
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn stable_json_digest(prefix: &[u8], value: &impl Serialize) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn require_nonempty(value: &str, field: &str) -> Result<(), QwenLegalPylonTrainingJobError> {
    if value.trim().is_empty() {
        return invalid_job(format!("{field} must be present"));
    }
    Ok(())
}

fn invalid_job<T>(detail: impl Into<String>) -> Result<T, QwenLegalPylonTrainingJobError> {
    Err(QwenLegalPylonTrainingJobError::InvalidJob {
        detail: detail.into(),
    })
}

fn invalid_receipt<T>(detail: impl Into<String>) -> Result<T, QwenLegalPylonTrainingJobError> {
    Err(QwenLegalPylonTrainingJobError::InvalidReceipt {
        detail: detail.into(),
    })
}

/// Builds the canonical dataset-shard protocol fixture.
#[must_use]
pub fn canonical_qwen_legal_dataset_shard_job() -> PylonTrainingJobSpec {
    canonical_job(
        "job.qwen-legal.dataset-shard.000001",
        PylonTrainingJobKind::DatasetShardBuild,
        "dataset_shard_manifest",
        "target/legal/pylon_jobs/dataset_shard_manifest.json",
        vec![PylonTrainingArtifactRef {
            artifact_id: String::from("qwen36-grpo-config"),
            artifact_type: String::from("training_config"),
            path: String::from("configs/legal/qwen36_grpo_smoke.json"),
            sha256: String::from(
                "a1e896ad711ccce93bcf9e3efa4b49d3ef461e397f3b02348debc5740b165a02",
            ),
        }],
    )
}

/// Builds the canonical eval-shard protocol fixture.
#[must_use]
pub fn canonical_qwen_legal_eval_shard_job() -> PylonTrainingJobSpec {
    canonical_job(
        "job.qwen-legal.eval-shard.000001",
        PylonTrainingJobKind::EvalShard,
        "eval_shard_report",
        "target/legal/pylon_jobs/eval_shard_report.json",
        vec![PylonTrainingArtifactRef {
            artifact_id: String::from("harvey-public-three-suite"),
            artifact_type: String::from("eval_suite"),
            path: String::from("suites/harvey_public_three.json"),
            sha256: String::from(
                "d029bb19d1033acf192a9993025838c0d65478d4595e2ee6985c2be919cf2181",
            ),
        }],
    )
}

fn canonical_job(
    job_id: &str,
    job_kind: PylonTrainingJobKind,
    output_artifact_type: &str,
    output_path: &str,
    expected_input_artifacts: Vec<PylonTrainingArtifactRef>,
) -> PylonTrainingJobSpec {
    PylonTrainingJobSpec {
        schema_version: String::from(QWEN_LEGAL_PYLON_TRAINING_JOB_SCHEMA_VERSION),
        job_id: String::from(job_id),
        parent_run_id: String::from("run.qwen-legal.pylon.protocol.000001"),
        job_kind,
        model_id: String::from("Qwen/Qwen3.6-27B"),
        model_hash: String::from("sha256:synthetic-qwen36-legal-smoke"),
        adapter_id: Some(String::from("qwen36-27b-legal-grpo-smoke")),
        adapter_hash: Some(String::from(
            "825b2d81aeae56d395a4fee7608eead91adf25ac24bae9ff995959df2b95732f",
        )),
        dataset_manifest_hash: String::from(
            "sha256:dataset-manifest-qwen-legal-pylon-protocol-smoke",
        ),
        shard_assignment: PylonTrainingShardAssignment {
            assignment_id: format!("assignment.{job_id}"),
            shard_id: String::from("shard.qwen-legal.public-smoke.000001"),
            shard_index: 0,
            shard_count: 1,
            start_index: Some(0),
            end_index: Some(3),
        },
        training_config_hash: String::from(
            "a1e896ad711ccce93bcf9e3efa4b49d3ef461e397f3b02348debc5740b165a02",
        ),
        expected_input_artifacts,
        expected_output_artifacts: vec![PylonTrainingExpectedOutputArtifact {
            artifact_id: format!("artifact.{job_id}.output"),
            artifact_type: String::from(output_artifact_type),
            path: String::from(output_path),
            required: true,
        }],
        max_runtime_ms: 60_000,
        hardware_requirements: PylonTrainingHardwareRequirements {
            min_memory_bytes: 512 * 1024 * 1024,
            require_accelerator: false,
            accepted_backend_labels: vec![String::from("local_protocol_smoke")],
        },
        payment_budget: PylonTrainingPaymentBudget {
            budget_id: String::from("budget.qwen-legal.pylon.protocol.000001"),
            max_cost_microusd: 0,
            payment_account_ref: String::from("ledger://local-smoke/no-payment"),
        },
        receipt_requirements: PylonTrainingReceiptRequirements {
            require_signature: true,
            require_logs_hash: true,
            require_metrics: true,
            required_output_artifact_types: vec![String::from(output_artifact_type)],
        },
        output_dir: String::from("target/legal/pylon_jobs"),
        receipt_path: format!("target/legal/pylon_jobs/{job_id}.receipt.json"),
    }
}

/// Writes canonical Pylon legal job fixtures.
pub fn write_canonical_qwen_legal_pylon_training_job_fixtures(
    output_dir: impl AsRef<Path>,
) -> Result<Vec<PathBuf>, QwenLegalPylonTrainingJobError> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).map_err(|error| QwenLegalPylonTrainingJobError::Io {
        path: output_dir.display().to_string(),
        message: error.to_string(),
    })?;
    let jobs = [
        (
            output_dir.join("dataset_shard_job_v1.json"),
            canonical_qwen_legal_dataset_shard_job(),
        ),
        (
            output_dir.join("eval_shard_job_v1.json"),
            canonical_qwen_legal_eval_shard_job(),
        ),
    ];
    let mut paths = Vec::new();
    for (path, job) in jobs {
        write_json(path.as_path(), &job)?;
        paths.push(path);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pylon_worker_accepts_dataset_shard_and_verifies_receipt()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let mut job = canonical_qwen_legal_dataset_shard_job();
        job.output_dir = temp.path().display().to_string();
        job.receipt_path = temp.path().join("receipt.json").display().to_string();
        job.expected_output_artifacts[0].path =
            temp.path().join("dataset_shard.json").display().to_string();

        let receipt =
            run_qwen_legal_pylon_worker_job(&job, &PylonLocalWorkerRunOptions::default())?;
        assert_eq!(receipt.status, PylonTrainingWorkerJobStatus::Succeeded);
        assert_eq!(receipt.job_kind, PylonTrainingJobKind::DatasetShardBuild);
        assert_eq!(receipt.output_hashes.len(), 1);
        let verification = verify_qwen_legal_pylon_worker_receipt_path(&job.receipt_path)?;
        assert!(verification.signature_valid);
        assert!(verification.output_files_rechecked);
        Ok(())
    }

    #[test]
    fn pylon_worker_accepts_eval_shard_and_verifies_receipt()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let mut job = canonical_qwen_legal_eval_shard_job();
        job.output_dir = temp.path().display().to_string();
        job.receipt_path = temp.path().join("receipt.json").display().to_string();
        job.expected_output_artifacts[0].path =
            temp.path().join("eval_report.json").display().to_string();

        let receipt =
            run_qwen_legal_pylon_worker_job(&job, &PylonLocalWorkerRunOptions::default())?;
        assert_eq!(receipt.status, PylonTrainingWorkerJobStatus::Succeeded);
        assert_eq!(receipt.job_kind, PylonTrainingJobKind::EvalShard);
        let verification = verify_qwen_legal_pylon_worker_receipt_path(&job.receipt_path)?;
        assert!(verification.signature_valid);
        Ok(())
    }

    #[test]
    fn pylon_worker_fails_wrong_input_hash() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let mut job = canonical_qwen_legal_dataset_shard_job();
        job.output_dir = temp.path().display().to_string();
        job.receipt_path = temp.path().join("receipt.json").display().to_string();
        job.expected_output_artifacts[0].path =
            temp.path().join("dataset_shard.json").display().to_string();
        job.expected_input_artifacts[0].sha256 =
            String::from("0000000000000000000000000000000000000000000000000000000000000000");

        let receipt =
            run_qwen_legal_pylon_worker_job(&job, &PylonLocalWorkerRunOptions::default())?;
        assert_eq!(receipt.status, PylonTrainingWorkerJobStatus::Failed);
        assert!(
            receipt
                .failure_reason
                .as_deref()
                .unwrap_or_default()
                .contains("hash mismatch")
        );
        verify_qwen_legal_pylon_worker_receipt_path(&job.receipt_path)?;
        Ok(())
    }

    #[test]
    fn pylon_worker_fails_missing_output_artifact() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let mut job = canonical_qwen_legal_eval_shard_job();
        job.output_dir = temp.path().display().to_string();
        job.receipt_path = temp.path().join("receipt.json").display().to_string();
        job.expected_output_artifacts[0].path = temp
            .path()
            .join("missing_eval_report.json")
            .display()
            .to_string();
        let options = PylonLocalWorkerRunOptions {
            emit_outputs: false,
            ..PylonLocalWorkerRunOptions::default()
        };

        let receipt = run_qwen_legal_pylon_worker_job(&job, &options)?;
        assert_eq!(receipt.status, PylonTrainingWorkerJobStatus::Failed);
        assert!(
            receipt
                .failure_reason
                .as_deref()
                .unwrap_or_default()
                .contains("required output artifact")
        );
        verify_qwen_legal_pylon_worker_receipt_path(&job.receipt_path)?;
        Ok(())
    }

    #[test]
    fn pylon_receipt_rejects_bad_signature() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let mut job = canonical_qwen_legal_dataset_shard_job();
        job.output_dir = temp.path().display().to_string();
        job.receipt_path = temp.path().join("receipt.json").display().to_string();
        job.expected_output_artifacts[0].path =
            temp.path().join("dataset_shard.json").display().to_string();
        let mut receipt =
            run_qwen_legal_pylon_worker_job(&job, &PylonLocalWorkerRunOptions::default())?;
        receipt.signature_hex = "00".repeat(64);
        receipt.receipt_digest = receipt.stable_digest();

        assert!(
            verify_qwen_legal_pylon_worker_receipt(&receipt, Path::new("receipt.json")).is_err()
        );
        Ok(())
    }
}

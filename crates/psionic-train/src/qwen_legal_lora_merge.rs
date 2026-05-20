//! Deterministic Qwen legal LoRA adapter aggregation for Pylon workers.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use psionic_adapters::{
    AdapterArtifactFormat, AdapterArtifactIdentity, AdapterArtifactKind, AdapterTargetFamily,
    LmHeadLoraAdapterArtifact, LmHeadLoraLoadError,
};
use psionic_core::QuantizationMode;
use psionic_eval::{LegalBenchmarkEvalSuiteRunConfig, run_legal_benchmark_eval_suite};
use safetensors::{Dtype as SafeTensorsDType, serialize, tensor::TensorView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::QWEN_LEGAL_ADAPTER_LORA_ALPHA;

/// Manifest schema for deterministic legal LoRA merge jobs.
pub const QWEN_LEGAL_LORA_MERGE_MANIFEST_SCHEMA_VERSION: &str =
    "psionic.qwen_legal_lora_merge_manifest.v1";
/// Receipt schema emitted by `psionic-train merge-lora`.
pub const QWEN_LEGAL_LORA_MERGE_RECEIPT_SCHEMA_VERSION: &str =
    "psionic.qwen_legal_lora_merge_receipt.v1";
/// Deterministic delta-averaging rule for worker LoRA factors.
pub const QWEN_LEGAL_LORA_DELTA_AVERAGE_RULE: &str = "trusted_weighted_lora_factor_average_v1";
/// Deterministic sequential shard handoff rule.
pub const QWEN_LEGAL_LORA_SEQUENTIAL_RULE: &str = "trusted_sequential_lora_shard_handoff_v1";

/// Merge strategy for distributed worker adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLegalLoraMergeMode {
    /// Workers train from the same parent adapter and the aggregator averages deltas by token count.
    DeltaAveraging,
    /// Workers train one after another, with each worker continuing from the previous adapter.
    ShardSequentialTraining,
}

impl QwenLegalLoraMergeMode {
    fn aggregation_rule(self) -> &'static str {
        match self {
            Self::DeltaAveraging => QWEN_LEGAL_LORA_DELTA_AVERAGE_RULE,
            Self::ShardSequentialTraining => QWEN_LEGAL_LORA_SEQUENTIAL_RULE,
        }
    }
}

/// Base model binding shared by all worker adapters in a merge job.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraMergeBaseModel {
    /// Public model id the adapter targets.
    pub base_model_id: String,
    /// Base model revision the adapter targets.
    pub base_model_revision: String,
    /// Stable digest for the served base artifact.
    pub base_served_artifact_digest: String,
}

/// One worker-produced LoRA adapter input.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraWorkerAdapterInput {
    /// Stable worker id.
    pub worker_id: String,
    /// Stable adapter id written by the worker.
    pub adapter_id: String,
    /// Stable adapter revision written by the worker.
    pub adapter_revision: String,
    /// Adapter artifact path.
    pub path: String,
    /// Expected artifact SHA-256 digest.
    pub sha256: String,
    /// Dataset shard hash trained by this worker.
    pub dataset_shard_hash: String,
    /// Training-token count used as the merge weight.
    pub token_count: u64,
    /// Parent adapter hash observed by this worker before training.
    pub parent_adapter_sha256: String,
}

/// Output adapter artifact declaration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraMergeOutput {
    /// Stable aggregate adapter id.
    pub adapter_id: String,
    /// Stable aggregate adapter revision.
    pub adapter_revision: String,
    /// Output artifact path.
    pub path: String,
    /// Optional expected output artifact hash.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_sha256: Option<String>,
}

/// Optional local eval and promotion-gate declaration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraMergeValidation {
    /// Legal benchmark eval suite path.
    pub suite_path: String,
    /// Base model id used for the eval comparison.
    pub base_model: String,
    /// Eval output directory.
    pub output_dir: String,
    /// Current champion adapter id, if known.
    pub champion_adapter_id: String,
    /// Champion score in basis points on this same suite.
    pub champion_score_bps: u32,
}

/// Manifest consumed by `psionic-train merge-lora`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraMergeManifest {
    /// Schema version.
    pub schema_version: String,
    /// Stable merge id.
    pub merge_id: String,
    /// Merge strategy.
    pub mode: QwenLegalLoraMergeMode,
    /// Parent adapter hash all independent workers started from, or the first parent in sequential mode.
    pub parent_adapter_sha256: String,
    /// Shared base model binding.
    pub base_model: QwenLegalLoraMergeBaseModel,
    /// Output artifact declaration.
    pub output_adapter: QwenLegalLoraMergeOutput,
    /// Worker adapters to merge.
    pub worker_adapters: Vec<QwenLegalLoraWorkerAdapterInput>,
    /// Optional local eval gate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<QwenLegalLoraMergeValidation>,
}

/// One worker's verified contribution inside the merge receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraMergeWorkerReceipt {
    /// Stable worker id.
    pub worker_id: String,
    /// Adapter id.
    pub adapter_id: String,
    /// Adapter revision.
    pub adapter_revision: String,
    /// Adapter artifact path.
    pub adapter_path: String,
    /// Verified worker adapter hash.
    pub adapter_sha256: String,
    /// Dataset shard hash.
    pub dataset_shard_hash: String,
    /// Token count.
    pub token_count: u64,
    /// Merge weight, after token-count normalization.
    pub merge_weight: f64,
    /// Parent adapter hash declared by the worker.
    pub parent_adapter_sha256: String,
}

/// Local eval summary attached to a merge receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraMergeValidationReceipt {
    /// Eval suite id.
    pub suite_id: String,
    /// Eval suite hash.
    pub suite_hash: String,
    /// Eval output directory.
    pub output_dir: String,
    /// Base score in basis points.
    pub base_score_bps: u32,
    /// Candidate adapter score in basis points.
    pub adapter_score_bps: u32,
    /// Candidate score delta in basis points.
    pub score_delta_bps: i32,
    /// Candidate answer-file success rate.
    pub answer_file_success_rate_bps: u32,
    /// Candidate integrity failures.
    pub integrity_failure_count: u64,
    /// Candidate tool failures.
    pub tool_failure_count: u64,
    /// Candidate timeouts.
    pub timeout_failure_count: u64,
    /// Local eval report hash.
    pub report_hash: String,
}

/// Promotion gate decision for a merged adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLegalLoraMergePromotionDecision {
    /// Candidate beats champion and has no hard eval failures.
    Promote,
    /// Candidate is valid but does not beat champion.
    Hold,
    /// Candidate has hard eval failures.
    Reject,
}

/// Promotion-gate receipt. It does not mutate the adapter registry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraMergePromotionGateReceipt {
    /// Champion adapter id.
    pub champion_adapter_id: String,
    /// Candidate adapter id.
    pub candidate_adapter_id: String,
    /// Champion score in basis points.
    pub champion_score_bps: u32,
    /// Candidate score in basis points.
    pub candidate_score_bps: u32,
    /// Candidate minus champion score.
    pub score_delta_bps: i32,
    /// Promotion decision.
    pub decision: QwenLegalLoraMergePromotionDecision,
    /// Reasons for the decision.
    pub reasons: Vec<String>,
}

/// Receipt emitted by a completed merge job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLoraMergeReceipt {
    /// Schema version.
    pub schema_version: String,
    /// Stable merge id.
    pub merge_id: String,
    /// Merge mode.
    pub mode: QwenLegalLoraMergeMode,
    /// Aggregation rule.
    pub aggregation_rule: String,
    /// Manifest path.
    pub manifest_path: String,
    /// Manifest hash.
    pub manifest_hash: String,
    /// Parent adapter hash.
    pub parent_adapter_sha256: String,
    /// Worker receipts.
    pub workers: Vec<QwenLegalLoraMergeWorkerReceipt>,
    /// Total token count across accepted worker adapters.
    pub total_token_count: u64,
    /// Output adapter path.
    pub output_adapter_path: String,
    /// Output adapter SHA-256 digest.
    pub output_adapter_sha256: String,
    /// Output adapter identity digest.
    pub output_adapter_identity_digest: String,
    /// Local eval receipt, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<QwenLegalLoraMergeValidationReceipt>,
    /// Promotion gate receipt, when local eval was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub promotion_gate: Option<QwenLegalLoraMergePromotionGateReceipt>,
    /// Deterministic replay command.
    pub deterministic_merge_command: Vec<String>,
    /// Stable receipt hash.
    pub receipt_hash: String,
}

impl QwenLegalLoraMergeReceipt {
    fn stable_hash(&self) -> String {
        let mut clone = self.clone();
        clone.receipt_hash.clear();
        stable_json_digest("psionic.qwen_legal_lora_merge_receipt.v1", &clone)
    }
}

/// Error returned by deterministic Qwen legal LoRA merge jobs.
#[derive(Debug, Error)]
pub enum QwenLegalLoraMergeError {
    #[error("merge-lora usage: psionic-train merge-lora --manifest <manifest.json>")]
    Usage,
    #[error("invalid merge manifest: {0}")]
    InvalidManifest(String),
    #[error("input hash mismatch for `{path}`: expected {expected}, got {actual}")]
    InputHash {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("output hash mismatch for `{path}`: expected {expected}, got {actual}")]
    OutputHash {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors export failed: {0}")]
    Safetensors(String),
    #[error("adapter load failed: {0}")]
    AdapterLoad(#[from] LmHeadLoraLoadError),
    #[error("local eval failed: {0}")]
    Eval(#[from] psionic_eval::LegalBenchmarkEvalSuiteError),
}

/// Runs the `merge-lora` CLI path.
pub fn run_psionic_legal_merge_lora_cli(
    args: &[String],
) -> Result<QwenLegalLoraMergeReceipt, QwenLegalLoraMergeError> {
    let manifest_path = parse_manifest_path(args)?;
    run_qwen_legal_lora_merge_manifest(manifest_path)
}

/// Runs one manifest-driven deterministic LoRA merge.
pub fn run_qwen_legal_lora_merge_manifest(
    manifest_path: impl AsRef<Path>,
) -> Result<QwenLegalLoraMergeReceipt, QwenLegalLoraMergeError> {
    let manifest_path = manifest_path.as_ref();
    let manifest_bytes = read_file(manifest_path)?;
    let manifest = serde_json::from_slice::<QwenLegalLoraMergeManifest>(&manifest_bytes)?;
    validate_manifest(&manifest)?;
    let manifest_hash = stable_json_digest("psionic.qwen_legal_lora_merge_manifest.v1", &manifest);
    let loaded = load_worker_adapters(manifest_path, &manifest)?;
    let (output_bytes, output_identity, worker_receipts, total_token_count) =
        merge_loaded_adapters(&manifest, loaded.as_slice())?;
    let output_path = resolve_output_path(manifest.output_adapter.path.as_str());
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|source| QwenLegalLoraMergeError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(&output_path, output_bytes.as_slice()).map_err(|source| {
        QwenLegalLoraMergeError::Io {
            path: output_path.clone(),
            source,
        }
    })?;
    let output_adapter_sha256 = sha256_hex(output_bytes.as_slice());
    if let Some(expected) = &manifest.output_adapter.expected_sha256 {
        if expected != &output_adapter_sha256 {
            return Err(QwenLegalLoraMergeError::OutputHash {
                path: output_path.display().to_string(),
                expected: expected.clone(),
                actual: output_adapter_sha256,
            });
        }
    }
    let deterministic_merge_command = vec![
        String::from("cargo"),
        String::from("run"),
        String::from("-p"),
        String::from("psionic-train"),
        String::from("--"),
        String::from("merge-lora"),
        String::from("--manifest"),
        manifest_path.display().to_string(),
    ];
    let validation = run_validation(
        manifest.validation.as_ref(),
        output_path.as_path(),
        deterministic_merge_command.as_slice(),
    )?;
    let promotion_gate = validation.as_ref().and_then(|validation_receipt| {
        manifest.validation.as_ref().map(|validation_manifest| {
            promotion_gate_receipt(
                manifest.output_adapter.adapter_id.as_str(),
                validation_manifest,
                validation_receipt,
            )
        })
    });
    let mut receipt = QwenLegalLoraMergeReceipt {
        schema_version: String::from(QWEN_LEGAL_LORA_MERGE_RECEIPT_SCHEMA_VERSION),
        merge_id: manifest.merge_id.clone(),
        mode: manifest.mode,
        aggregation_rule: String::from(manifest.mode.aggregation_rule()),
        manifest_path: manifest_path.display().to_string(),
        manifest_hash,
        parent_adapter_sha256: manifest.parent_adapter_sha256.clone(),
        workers: worker_receipts,
        total_token_count,
        output_adapter_path: output_path.display().to_string(),
        output_adapter_sha256,
        output_adapter_identity_digest: output_identity.stable_digest(),
        validation,
        promotion_gate,
        deterministic_merge_command,
        receipt_hash: String::new(),
    };
    receipt.receipt_hash = receipt.stable_hash();
    let receipt_path = output_path.with_extension("merge-receipt.json");
    write_file_pretty(receipt_path.as_path(), &receipt)?;
    Ok(receipt)
}

#[derive(Clone)]
struct LoadedWorkerAdapter {
    input: QwenLegalLoraWorkerAdapterInput,
    bytes: Vec<u8>,
    adapter: LmHeadLoraAdapterArtifact,
}

fn parse_manifest_path(args: &[String]) -> Result<PathBuf, QwenLegalLoraMergeError> {
    args.windows(2)
        .find_map(|window| (window[0] == "--manifest").then(|| PathBuf::from(&window[1])))
        .ok_or(QwenLegalLoraMergeError::Usage)
}

fn validate_manifest(manifest: &QwenLegalLoraMergeManifest) -> Result<(), QwenLegalLoraMergeError> {
    if manifest.schema_version != QWEN_LEGAL_LORA_MERGE_MANIFEST_SCHEMA_VERSION {
        return Err(QwenLegalLoraMergeError::InvalidManifest(format!(
            "schema_version must be {QWEN_LEGAL_LORA_MERGE_MANIFEST_SCHEMA_VERSION}"
        )));
    }
    if manifest.merge_id.trim().is_empty() {
        return Err(QwenLegalLoraMergeError::InvalidManifest(String::from(
            "merge_id must not be empty",
        )));
    }
    if !is_complete_sha256(manifest.parent_adapter_sha256.as_str()) {
        return Err(QwenLegalLoraMergeError::InvalidManifest(String::from(
            "parent_adapter_sha256 must be a 64-character SHA-256 hex digest",
        )));
    }
    if manifest.worker_adapters.len() < 2 {
        return Err(QwenLegalLoraMergeError::InvalidManifest(String::from(
            "at least two worker adapters are required",
        )));
    }
    if manifest.output_adapter.adapter_id.trim().is_empty()
        || manifest.output_adapter.adapter_revision.trim().is_empty()
        || manifest.output_adapter.path.trim().is_empty()
    {
        return Err(QwenLegalLoraMergeError::InvalidManifest(String::from(
            "output adapter id, revision, and path must be present",
        )));
    }
    if let Some(expected) = &manifest.output_adapter.expected_sha256 {
        if !is_complete_sha256(expected) {
            return Err(QwenLegalLoraMergeError::InvalidManifest(String::from(
                "output expected_sha256 must be a 64-character SHA-256 hex digest",
            )));
        }
    }
    for worker in &manifest.worker_adapters {
        if worker.worker_id.trim().is_empty()
            || worker.adapter_id.trim().is_empty()
            || worker.adapter_revision.trim().is_empty()
            || worker.path.trim().is_empty()
            || worker.dataset_shard_hash.trim().is_empty()
        {
            return Err(QwenLegalLoraMergeError::InvalidManifest(String::from(
                "worker id, adapter id, revision, path, and dataset shard hash must be present",
            )));
        }
        if worker.token_count == 0 {
            return Err(QwenLegalLoraMergeError::InvalidManifest(format!(
                "worker `{}` token_count must be > 0",
                worker.worker_id
            )));
        }
        if !is_complete_sha256(worker.sha256.as_str()) {
            return Err(QwenLegalLoraMergeError::InvalidManifest(format!(
                "worker `{}` sha256 must be a 64-character SHA-256 hex digest",
                worker.worker_id
            )));
        }
        if !is_complete_sha256(worker.parent_adapter_sha256.as_str()) {
            return Err(QwenLegalLoraMergeError::InvalidManifest(format!(
                "worker `{}` parent_adapter_sha256 must be a 64-character SHA-256 hex digest",
                worker.worker_id
            )));
        }
    }
    match manifest.mode {
        QwenLegalLoraMergeMode::DeltaAveraging => {
            for worker in &manifest.worker_adapters {
                if worker.parent_adapter_sha256 != manifest.parent_adapter_sha256 {
                    return Err(QwenLegalLoraMergeError::InvalidManifest(format!(
                        "worker `{}` did not train from the merge parent adapter",
                        worker.worker_id
                    )));
                }
            }
        }
        QwenLegalLoraMergeMode::ShardSequentialTraining => {
            let mut expected_parent = manifest.parent_adapter_sha256.as_str();
            for worker in &manifest.worker_adapters {
                if worker.parent_adapter_sha256 != expected_parent {
                    return Err(QwenLegalLoraMergeError::InvalidManifest(format!(
                        "sequential worker `{}` parent hash mismatch",
                        worker.worker_id
                    )));
                }
                expected_parent = worker.sha256.as_str();
            }
        }
    }
    Ok(())
}

fn load_worker_adapters(
    manifest_path: &Path,
    manifest: &QwenLegalLoraMergeManifest,
) -> Result<Vec<LoadedWorkerAdapter>, QwenLegalLoraMergeError> {
    let mut workers = manifest.worker_adapters.clone();
    if manifest.mode == QwenLegalLoraMergeMode::DeltaAveraging {
        workers.sort_by(|left, right| {
            left.worker_id
                .cmp(&right.worker_id)
                .then(left.adapter_id.cmp(&right.adapter_id))
                .then(left.sha256.cmp(&right.sha256))
        });
    }
    workers
        .into_iter()
        .map(|worker| {
            let path = resolve_input_path(manifest_path, worker.path.as_str());
            let bytes = read_file(path.as_path())?;
            let actual = sha256_hex(bytes.as_slice());
            if actual != worker.sha256 {
                return Err(QwenLegalLoraMergeError::InputHash {
                    path: path.display().to_string(),
                    expected: worker.sha256,
                    actual,
                });
            }
            let identity = adapter_identity_for_hash(
                &manifest.base_model,
                worker.adapter_id.clone(),
                worker.adapter_revision.clone(),
                actual,
                0,
            );
            let adapter = LmHeadLoraAdapterArtifact::from_safetensors_bytes(
                bytes.as_slice(),
                identity,
                QWEN_LEGAL_ADAPTER_LORA_ALPHA,
            )?;
            Ok(LoadedWorkerAdapter {
                input: worker,
                bytes,
                adapter,
            })
        })
        .collect()
}

fn merge_loaded_adapters(
    manifest: &QwenLegalLoraMergeManifest,
    loaded: &[LoadedWorkerAdapter],
) -> Result<
    (
        Vec<u8>,
        AdapterArtifactIdentity,
        Vec<QwenLegalLoraMergeWorkerReceipt>,
        u64,
    ),
    QwenLegalLoraMergeError,
> {
    let total_token_count = loaded
        .iter()
        .map(|worker| worker.input.token_count)
        .sum::<u64>();
    let workers = loaded
        .iter()
        .map(|worker| QwenLegalLoraMergeWorkerReceipt {
            worker_id: worker.input.worker_id.clone(),
            adapter_id: worker.input.adapter_id.clone(),
            adapter_revision: worker.input.adapter_revision.clone(),
            adapter_path: worker.input.path.clone(),
            adapter_sha256: worker.input.sha256.clone(),
            dataset_shard_hash: worker.input.dataset_shard_hash.clone(),
            token_count: worker.input.token_count,
            merge_weight: worker.input.token_count as f64 / total_token_count as f64,
            parent_adapter_sha256: worker.input.parent_adapter_sha256.clone(),
        })
        .collect::<Vec<_>>();
    let output_bytes = match manifest.mode {
        QwenLegalLoraMergeMode::DeltaAveraging => merge_delta_average(loaded, total_token_count)?,
        QwenLegalLoraMergeMode::ShardSequentialTraining => loaded
            .last()
            .map(|worker| worker.bytes.clone())
            .ok_or_else(|| {
                QwenLegalLoraMergeError::InvalidManifest(String::from(
                    "sequential merge had no worker adapters",
                ))
            })?,
    };
    let output_hash = sha256_hex(output_bytes.as_slice());
    let first = &loaded[0].adapter;
    let parameter_count = u64::try_from(
        first
            .rank
            .saturating_mul(first.hidden_size.saturating_add(first.vocab_size)),
    )
    .unwrap_or(u64::MAX);
    let identity = adapter_identity_for_hash(
        &manifest.base_model,
        manifest.output_adapter.adapter_id.clone(),
        manifest.output_adapter.adapter_revision.clone(),
        output_hash,
        parameter_count,
    )
    .with_provenance_digest(stable_json_digest(
        "psionic.qwen_legal_lora_merge.provenance.v1",
        &workers,
    ))
    .with_governance_digest(stable_json_digest(
        "psionic.qwen_legal_lora_merge.governance.v1",
        &manifest.mode.aggregation_rule(),
    ));
    let reloaded = LmHeadLoraAdapterArtifact::from_safetensors_bytes(
        output_bytes.as_slice(),
        identity.clone(),
        QWEN_LEGAL_ADAPTER_LORA_ALPHA,
    )?;
    if reloaded.rank != first.rank
        || reloaded.hidden_size != first.hidden_size
        || reloaded.vocab_size != first.vocab_size
    {
        return Err(QwenLegalLoraMergeError::InvalidManifest(String::from(
            "output adapter shape changed after reload",
        )));
    }
    Ok((output_bytes, identity, workers, total_token_count))
}

fn merge_delta_average(
    loaded: &[LoadedWorkerAdapter],
    total_token_count: u64,
) -> Result<Vec<u8>, QwenLegalLoraMergeError> {
    let first = &loaded[0].adapter;
    let mut lora_a = vec![0.0_f32; first.lora_a().len()];
    let mut lora_b = vec![0.0_f32; first.lora_b().len()];
    for worker in loaded {
        let adapter = &worker.adapter;
        if adapter.rank != first.rank
            || adapter.hidden_size != first.hidden_size
            || adapter.vocab_size != first.vocab_size
        {
            return Err(QwenLegalLoraMergeError::InvalidManifest(String::from(
                "worker adapter shapes differ",
            )));
        }
        let weight = worker.input.token_count as f32;
        for (target, value) in lora_a.iter_mut().zip(adapter.lora_a().iter()) {
            *target += *value * weight;
        }
        for (target, value) in lora_b.iter_mut().zip(adapter.lora_b().iter()) {
            *target += *value * weight;
        }
    }
    let total_weight = total_token_count as f32;
    for value in &mut lora_a {
        *value /= total_weight;
    }
    for value in &mut lora_b {
        *value /= total_weight;
    }
    export_lora_safetensors(
        &lora_a,
        &lora_b,
        first.rank,
        first.hidden_size,
        first.vocab_size,
    )
}

fn export_lora_safetensors(
    lora_a: &[f32],
    lora_b: &[f32],
    rank: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Vec<u8>, QwenLegalLoraMergeError> {
    let raw_a = encode_f32_bytes(lora_a);
    let raw_b = encode_f32_bytes(lora_b);
    let view_a = TensorView::new(
        SafeTensorsDType::F32,
        vec![rank, hidden_size],
        raw_a.as_slice(),
    )
    .map_err(|error| QwenLegalLoraMergeError::Safetensors(error.to_string()))?;
    let view_b = TensorView::new(
        SafeTensorsDType::F32,
        vec![vocab_size, rank],
        raw_b.as_slice(),
    )
    .map_err(|error| QwenLegalLoraMergeError::Safetensors(error.to_string()))?;
    let mut metadata = HashMap::new();
    metadata.insert(
        String::from("openagents.aggregate_manifest"),
        String::from(QWEN_LEGAL_LORA_DELTA_AVERAGE_RULE),
    );
    serialize(
        [
            ("lm_head.lora_A.weight", view_a),
            ("lm_head.lora_B.weight", view_b),
        ],
        Some(metadata),
    )
    .map_err(|error| QwenLegalLoraMergeError::Safetensors(error.to_string()))
}

fn run_validation(
    validation: Option<&QwenLegalLoraMergeValidation>,
    output_adapter_path: &Path,
    deterministic_merge_command: &[String],
) -> Result<Option<QwenLegalLoraMergeValidationReceipt>, QwenLegalLoraMergeError> {
    let Some(validation) = validation else {
        return Ok(None);
    };
    let output_dir = resolve_output_path(validation.output_dir.as_str());
    let suite_path = resolve_output_path(validation.suite_path.as_str());
    let report = run_legal_benchmark_eval_suite(&LegalBenchmarkEvalSuiteRunConfig {
        suite_path,
        base_model: validation.base_model.clone(),
        adapter: output_adapter_path.display().to_string(),
        output_dir: output_dir.clone(),
        replay_command: deterministic_merge_command.to_vec(),
    })?;
    Ok(Some(QwenLegalLoraMergeValidationReceipt {
        suite_id: report.suite_id,
        suite_hash: report.suite_hash,
        output_dir: output_dir.display().to_string(),
        base_score_bps: report.base_model_result.legal_score_bps,
        adapter_score_bps: report.adapter_result.legal_score_bps,
        score_delta_bps: report.comparison.score_delta_bps,
        answer_file_success_rate_bps: report.adapter_result.answer_file_success_rate_bps,
        integrity_failure_count: report.adapter_result.integrity_failure_count,
        tool_failure_count: report.adapter_result.tool_failure_count,
        timeout_failure_count: report.adapter_result.timeout_failure_count,
        report_hash: report.replay_receipt.report_hash,
    }))
}

fn promotion_gate_receipt(
    candidate_adapter_id: &str,
    validation_manifest: &QwenLegalLoraMergeValidation,
    validation: &QwenLegalLoraMergeValidationReceipt,
) -> QwenLegalLoraMergePromotionGateReceipt {
    let score_delta_bps = i32::try_from(validation.adapter_score_bps).unwrap_or(i32::MAX)
        - i32::try_from(validation_manifest.champion_score_bps).unwrap_or(i32::MAX);
    let mut reasons = Vec::new();
    let hard_failures = validation.integrity_failure_count
        + validation.tool_failure_count
        + validation.timeout_failure_count;
    let decision = if hard_failures > 0 {
        reasons.push(String::from(
            "candidate has integrity, tool, or timeout failures",
        ));
        QwenLegalLoraMergePromotionDecision::Reject
    } else if validation.adapter_score_bps <= validation_manifest.champion_score_bps {
        reasons.push(String::from("candidate score does not beat champion"));
        QwenLegalLoraMergePromotionDecision::Hold
    } else {
        reasons.push(String::from(
            "candidate beats champion on the same local eval suite",
        ));
        QwenLegalLoraMergePromotionDecision::Promote
    };
    QwenLegalLoraMergePromotionGateReceipt {
        champion_adapter_id: validation_manifest.champion_adapter_id.clone(),
        candidate_adapter_id: candidate_adapter_id.to_owned(),
        champion_score_bps: validation_manifest.champion_score_bps,
        candidate_score_bps: validation.adapter_score_bps,
        score_delta_bps,
        decision,
        reasons,
    }
}

fn adapter_identity_for_hash(
    base_model: &QwenLegalLoraMergeBaseModel,
    adapter_id: String,
    adapter_revision: String,
    artifact_digest: String,
    parameter_count: u64,
) -> AdapterArtifactIdentity {
    AdapterArtifactIdentity::new(
        adapter_id,
        adapter_revision,
        AdapterArtifactKind::Lora,
        AdapterArtifactFormat::Safetensors,
        base_model.base_model_id.clone(),
        base_model.base_model_revision.clone(),
        base_model.base_served_artifact_digest.clone(),
        artifact_digest,
        QuantizationMode::None,
        AdapterTargetFamily::DecoderComposite,
        parameter_count,
    )
}

fn resolve_input_path(manifest_path: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() || path.exists() {
        path
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

fn resolve_output_path(path: &str) -> PathBuf {
    PathBuf::from(path)
}

fn read_file(path: &Path) -> Result<Vec<u8>, QwenLegalLoraMergeError> {
    fs::read(path).map_err(|source| QwenLegalLoraMergeError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_file_pretty<T>(path: &Path, value: &T) -> Result<(), QwenLegalLoraMergeError>
where
    T: Serialize,
{
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes).map_err(|source| QwenLegalLoraMergeError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn stable_json_digest<T>(namespace: &str, value: &T) -> String
where
    T: Serialize,
{
    let mut hasher = Sha256::new();
    hasher.update(namespace.as_bytes());
    hasher.update(b"|");
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}

fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn is_complete_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO_PARENT: &str = "821e206ee5e1b9bd158faf1ff8dc2faa0cbafa3cc7767eef77eaffd44d9a009d";
    const CUDA_SHA: &str = "5298c62de7f6b318f8889957433fe401ba693911997c1c606ce5099e73ca41cb";
    const METAL_SHA: &str = "6e8b6da6083606cbd36105a65d532e48e067094809f9db552ca9a9d13105af9b";
    const AGGREGATE_SHA: &str = "8e8dea3bc639ed2c147d6901f6ceda9b5f1a176034dc7bb65219daf7dd33116d";

    #[test]
    fn merge_lora_delta_averaging_is_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let manifest_path = temp.path().join("merge.json");
        let mut manifest = fixture_manifest(temp.path());
        manifest.output_adapter.path = temp
            .path()
            .join("aggregate.safetensors")
            .display()
            .to_string();
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        let first = run_qwen_legal_lora_merge_manifest(&manifest_path)?;
        let second = run_qwen_legal_lora_merge_manifest(&manifest_path)?;
        assert_eq!(first.output_adapter_sha256, AGGREGATE_SHA);
        assert_eq!(second.output_adapter_sha256, AGGREGATE_SHA);
        assert_eq!(first.output_adapter_sha256, second.output_adapter_sha256);
        assert_eq!(first.workers.len(), 2);
        assert_eq!(first.total_token_count, 272);
        Ok(())
    }

    #[test]
    fn merge_lora_rejects_worker_hash_mismatch() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let manifest_path = temp.path().join("merge.json");
        let mut manifest = fixture_manifest(temp.path());
        manifest.worker_adapters[0].sha256 =
            "0000000000000000000000000000000000000000000000000000000000000000".into();
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        let error = run_qwen_legal_lora_merge_manifest(&manifest_path)
            .expect_err("bad worker hash must fail");
        assert!(error.to_string().contains("input hash mismatch"));
        Ok(())
    }

    #[test]
    fn merge_lora_sequential_mode_uses_last_adapter_and_validates_chain()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let manifest_path = temp.path().join("merge.json");
        let mut manifest = fixture_manifest(temp.path());
        manifest.mode = QwenLegalLoraMergeMode::ShardSequentialTraining;
        manifest.output_adapter.path = temp.path().join("last.safetensors").display().to_string();
        manifest.output_adapter.expected_sha256 = Some(String::from(METAL_SHA));
        manifest.worker_adapters[1].parent_adapter_sha256 = String::from(CUDA_SHA);
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        let receipt = run_qwen_legal_lora_merge_manifest(&manifest_path)?;
        assert_eq!(receipt.aggregation_rule, QWEN_LEGAL_LORA_SEQUENTIAL_RULE);
        assert_eq!(receipt.output_adapter_sha256, METAL_SHA);
        Ok(())
    }

    #[test]
    fn promotion_gate_holds_non_improving_candidate() {
        let validation_manifest = QwenLegalLoraMergeValidation {
            suite_path: String::from("suite.json"),
            base_model: String::from("Qwen/Qwen3.6-27B"),
            output_dir: String::from("target/legal/eval"),
            champion_adapter_id: String::from("champion"),
            champion_score_bps: 9_000,
        };
        let validation = QwenLegalLoraMergeValidationReceipt {
            suite_id: String::from("suite"),
            suite_hash: String::from("hash"),
            output_dir: String::from("target/legal/eval"),
            base_score_bps: 3_000,
            adapter_score_bps: 9_000,
            score_delta_bps: 6_000,
            answer_file_success_rate_bps: 10_000,
            integrity_failure_count: 0,
            tool_failure_count: 0,
            timeout_failure_count: 0,
            report_hash: String::from("report"),
        };
        let receipt = promotion_gate_receipt("candidate", &validation_manifest, &validation);
        assert_eq!(receipt.decision, QwenLegalLoraMergePromotionDecision::Hold);
        assert!(
            receipt
                .reasons
                .iter()
                .any(|reason| reason.contains("does not beat"))
        );
    }

    fn fixture_manifest(temp: &Path) -> QwenLegalLoraMergeManifest {
        QwenLegalLoraMergeManifest {
            schema_version: String::from(QWEN_LEGAL_LORA_MERGE_MANIFEST_SCHEMA_VERSION),
            merge_id: String::from("legal-sft-round-001"),
            mode: QwenLegalLoraMergeMode::DeltaAveraging,
            parent_adapter_sha256: String::from(ZERO_PARENT),
            base_model: QwenLegalLoraMergeBaseModel {
                base_model_id: String::from("Qwen/Qwen3.5-4B"),
                base_model_revision: String::from("qwen3.5-4b-smoke-revision"),
                base_served_artifact_digest: String::from("sha256:synthetic-qwen35-4b-legal-smoke"),
            },
            output_adapter: QwenLegalLoraMergeOutput {
                adapter_id: String::from("qwen35-4b-legal-pylon-network-smoke"),
                adapter_revision: String::from("r1-trusted-aggregate"),
                path: temp.join("aggregate.safetensors").display().to_string(),
                expected_sha256: Some(String::from(AGGREGATE_SHA)),
            },
            worker_adapters: vec![
                QwenLegalLoraWorkerAdapterInput {
                    worker_id: String::from("worker.qwen-legal.cuda.01"),
                    adapter_id: String::from("qwen35-4b-legal-pylon-network-smoke"),
                    adapter_revision: String::from("r1-contributor-1"),
                    path: fixture_path("contributor-pylon-local-legal-cuda-01.safetensors"),
                    sha256: String::from(CUDA_SHA),
                    dataset_shard_hash: String::from("shard://harvey-legal-smoke/train/even"),
                    token_count: 140,
                    parent_adapter_sha256: String::from(ZERO_PARENT),
                },
                QwenLegalLoraWorkerAdapterInput {
                    worker_id: String::from("worker.qwen-legal.metal.01"),
                    adapter_id: String::from("qwen35-4b-legal-pylon-network-smoke"),
                    adapter_revision: String::from("r1-contributor-2"),
                    path: fixture_path("contributor-pylon-local-legal-metal-01.safetensors"),
                    sha256: String::from(METAL_SHA),
                    dataset_shard_hash: String::from("shard://harvey-legal-smoke/train/odd"),
                    token_count: 132,
                    parent_adapter_sha256: String::from(ZERO_PARENT),
                },
            ],
            validation: None,
        }
    }

    fn fixture_path(file: &str) -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen_legal/pylon_network_sft")
            .join(file)
            .display()
            .to_string()
    }
}

use std::{
    fs,
    path::{Path, PathBuf},
};

use psionic_data::{TokenizerDigest, TokenizerFamily};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    ModelIoArtifactReceipt, OPEN_ADAPTER_QWEN36_LEGAL_CUDA_BACKEND_LABEL,
    OpenAdapterAdmissibleModelFamily, OpenAdapterExecutionConfig, OpenAdapterHiddenStateSample,
    OpenAdapterLmHeadTarget, OpenAdapterPrecisionPolicy, OpenAdapterReferenceModel,
    OpenAdapterSftError, OpenAdapterSftRunRequest, OpenAdapterTrainingExecutionBackend,
    OpenAdapterTrainingExecutionError, TrainingCoreError, TrainingLoopBudget,
    TrainingOptimizerConfig, TrainingOptimizerResidencyPolicy, TrainingRunSummary,
    TrainingStepReceipt, run_open_adapter_sft_export,
};

/// Config schema accepted by `psionic-train sft --config`.
pub const PSIONIC_LEGAL_SFT_CONFIG_SCHEMA_VERSION: &str = "psionic.legal_sft_config.v1";
/// Receipt schema emitted by `psionic-train sft`.
pub const PSIONIC_TRAINING_RECEIPT_SCHEMA_VERSION: &str = "psionic.training_receipt.v1";
/// Checkpoint summary schema emitted by `psionic-train sft`.
pub const PSIONIC_LEGAL_SFT_CHECKPOINT_SCHEMA_VERSION: &str =
    "psionic.legal_sft_checkpoint_summary.v1";
/// Loss curve schema emitted by `psionic-train sft`.
pub const PSIONIC_LEGAL_SFT_LOSS_CURVE_SCHEMA_VERSION: &str = "psionic.legal_sft_loss_curve.v1";

const QWEN36_LEGAL_CHECKPOINT_FAMILY: &str = "psionic.qwen36.legal_adapter_sft";
const DEFAULT_QWEN36_BASE_DIGEST: &str = "sha256:synthetic-qwen36-legal-smoke";
const DEFAULT_QWEN36_TEMPLATE_DIGEST: &str = "sha256:qwen36-chat-template-v1-smoke";
const DEFAULT_QWEN36_TOKENIZER_DIGEST: &str = "sha256:qwen36-tokenizer-smoke";

/// Qwen3.6 dense LoRA target modules to use when config asks for `all-linear`.
pub const QWEN36_DENSE_LORA_TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

/// Config for one Rust-only legal SFT run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsionicLegalSftConfig {
    /// Schema version.
    pub schema_version: String,
    /// Stable run id.
    pub run_id: String,
    /// `lora` or `qlora`.
    pub train_type: String,
    /// Public base model id.
    pub base_model: String,
    /// Served model id used by later eval metadata.
    pub served_model_id: String,
    /// Base model revision or fixture revision.
    pub base_model_revision: String,
    /// Synthetic smoke or real artifact mode.
    pub base_artifact_mode: PsionicLegalSftBaseArtifactMode,
    /// Digest of the frozen base artifact or synthetic fixture.
    pub base_served_artifact_digest: String,
    /// Optional local safetensors files to admit for real-artifact runs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub base_safetensors_paths: Vec<String>,
    /// Optional local model config JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_config_path: Option<String>,
    /// Optional tokenizer JSON file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,
    /// Tokenizer digest used when no tokenizer file is bound.
    #[serde(default = "default_tokenizer_digest")]
    pub tokenizer_digest: String,
    /// Prompt template digest.
    #[serde(default = "default_template_digest")]
    pub prompt_template_digest: String,
    /// Synthetic fixture hidden width.
    pub hidden_size: usize,
    /// Synthetic fixture vocabulary width.
    pub vocab_size: usize,
    /// Adapter target used by the current smoke trainer.
    #[serde(default = "default_adapter_target_id")]
    pub adapter_target_id: String,
    /// Declared future Qwen dense target modules or `all-linear`.
    #[serde(default)]
    pub target_modules: Vec<String>,
    /// LoRA rank.
    pub lora_rank: usize,
    /// LoRA alpha.
    pub lora_alpha: f32,
    /// LoRA dropout. The smoke path records this but uses deterministic zero-dropout updates.
    #[serde(default)]
    pub lora_dropout: f32,
    /// Learning rate.
    pub learning_rate: f32,
    /// Epoch count recorded in the receipt.
    pub epochs: u32,
    /// Maximum sequence length recorded in the receipt.
    pub max_seq_len: u32,
    /// Gradient accumulation steps recorded in the receipt.
    pub gradient_accumulation_steps: u32,
    /// Fixed step count for the smoke trainer.
    pub max_steps: u64,
    /// Batch size.
    pub batch_size: usize,
    /// Assistant-only loss flag from Qwen legal SFT examples.
    pub assistant_only_loss: bool,
    /// Whether empty Qwen think blocks are masked out.
    pub ignore_empty_think_loss: bool,
    /// Optional gradient clipping norm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_clip_norm: Option<f32>,
    /// Logical start timestamp.
    pub started_at_ms: u64,
    /// Logical duration per step.
    pub step_duration_ms: u64,
    /// Dataset reference.
    pub dataset_ref: String,
    /// Validator policy reference.
    pub validator_policy_ref: String,
    /// Adapter id.
    pub adapter_id: String,
    /// Adapter revision.
    pub adapter_revision: String,
    /// Output directory.
    pub output_dir: String,
    /// Tiny legal workflow supervision samples for the smoke run.
    pub samples: Vec<PsionicLegalSftSample>,
}

impl PsionicLegalSftConfig {
    fn validate(&self) -> Result<(), PsionicLegalSftError> {
        if self.schema_version != PSIONIC_LEGAL_SFT_CONFIG_SCHEMA_VERSION {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: String::from("schema_version is not psionic.legal_sft_config.v1"),
            });
        }
        require_nonempty(self.run_id.as_str(), "run_id")?;
        require_nonempty(self.train_type.as_str(), "train_type")?;
        if !matches!(self.train_type.as_str(), "lora" | "qlora") {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: String::from("train_type must be `lora` or `qlora`"),
            });
        }
        require_nonempty(self.base_model.as_str(), "base_model")?;
        require_nonempty(self.served_model_id.as_str(), "served_model_id")?;
        require_nonempty(self.base_model_revision.as_str(), "base_model_revision")?;
        require_nonempty(
            self.base_served_artifact_digest.as_str(),
            "base_served_artifact_digest",
        )?;
        require_nonempty(self.tokenizer_digest.as_str(), "tokenizer_digest")?;
        require_nonempty(
            self.prompt_template_digest.as_str(),
            "prompt_template_digest",
        )?;
        require_nonempty(self.adapter_target_id.as_str(), "adapter_target_id")?;
        if self.hidden_size == 0 || self.vocab_size == 0 {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: String::from("hidden_size and vocab_size must be non-zero"),
            });
        }
        if self.lora_rank == 0 || !self.lora_alpha.is_finite() || self.lora_alpha <= 0.0 {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: String::from("lora_rank and lora_alpha must be positive"),
            });
        }
        if self.lora_dropout != 0.0 {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: String::from(
                    "the deterministic smoke trainer only admits lora_dropout = 0.0",
                ),
            });
        }
        if self.epochs == 0
            || self.max_seq_len == 0
            || self.gradient_accumulation_steps == 0
            || self.max_steps == 0
            || self.batch_size == 0
            || self.step_duration_ms == 0
        {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: String::from(
                    "epochs, max_seq_len, gradient_accumulation_steps, max_steps, batch_size, and step_duration_ms must be non-zero",
                ),
            });
        }
        require_nonempty(self.dataset_ref.as_str(), "dataset_ref")?;
        require_nonempty(self.validator_policy_ref.as_str(), "validator_policy_ref")?;
        require_nonempty(self.adapter_id.as_str(), "adapter_id")?;
        require_nonempty(self.adapter_revision.as_str(), "adapter_revision")?;
        require_nonempty(self.output_dir.as_str(), "output_dir")?;
        if self.samples.is_empty() {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: String::from("samples must not be empty"),
            });
        }
        for sample in &self.samples {
            sample.validate(self.hidden_size, self.vocab_size)?;
        }
        if self.base_artifact_mode == PsionicLegalSftBaseArtifactMode::RealArtifactRequired
            && self.base_safetensors_paths.is_empty()
        {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: String::from(
                    "real artifact mode requires at least one base_safetensors_paths entry",
                ),
            });
        }
        resolve_qwen36_target_modules(&self.target_modules)?;
        Ok(())
    }
}

/// Base artifact posture for one SFT run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicLegalSftBaseArtifactMode {
    /// Deterministic hidden-state fixture; no full base weights are claimed.
    SyntheticHiddenStateSmoke,
    /// Real Qwen artifacts must be materialized and hashed before training.
    RealArtifactRequired,
}

/// One tiny legal workflow sample consumed by the smoke trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsionicLegalSftSample {
    /// Stable sample id.
    pub sample_id: String,
    /// Legal training record id.
    pub legal_training_record_id: String,
    /// Frozen-base final hidden state.
    pub final_hidden_state: Vec<f32>,
    /// Target token id.
    pub target_token_id: u32,
    /// Approximate source-token count.
    pub source_token_count: u32,
}

impl PsionicLegalSftSample {
    fn validate(&self, hidden_size: usize, vocab_size: usize) -> Result<(), PsionicLegalSftError> {
        require_nonempty(self.sample_id.as_str(), "sample_id")?;
        require_nonempty(
            self.legal_training_record_id.as_str(),
            "legal_training_record_id",
        )?;
        if self.final_hidden_state.len() != hidden_size {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: format!(
                    "sample `{}` hidden size is {}, expected {hidden_size}",
                    self.sample_id,
                    self.final_hidden_state.len()
                ),
            });
        }
        if self.target_token_id as usize >= vocab_size {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: format!(
                    "sample `{}` target token {} is outside vocab size {vocab_size}",
                    self.sample_id, self.target_token_id
                ),
            });
        }
        if self.source_token_count == 0 {
            return Err(PsionicLegalSftError::InvalidConfig {
                detail: format!(
                    "sample `{}` source_token_count must be non-zero",
                    self.sample_id
                ),
            });
        }
        Ok(())
    }

    fn into_open_adapter_sample(
        self,
    ) -> Result<OpenAdapterHiddenStateSample, PsionicLegalSftError> {
        Ok(OpenAdapterHiddenStateSample::new(
            self.sample_id,
            self.final_hidden_state,
            self.target_token_id,
            self.source_token_count,
        )?)
    }
}

/// One loss-curve row emitted by the SFT command.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsionicLegalSftLossPoint {
    /// Global step.
    pub step: u64,
    /// Mean batch loss.
    pub loss: f32,
    /// Batch id.
    pub batch_id: String,
    /// Step receipt digest.
    pub receipt_digest: String,
}

/// Loss curve artifact emitted by the SFT command.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsionicLegalSftLossCurve {
    /// Schema version.
    pub schema_version: String,
    /// Run id.
    pub run_id: String,
    /// Loss points.
    pub points: Vec<PsionicLegalSftLossPoint>,
}

/// Deterministic checkpoint summary emitted by the SFT command.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsionicLegalSftCheckpointSummary {
    /// Schema version.
    pub schema_version: String,
    /// Run id.
    pub run_id: String,
    /// Checkpoint family.
    pub checkpoint_family: String,
    /// Final run summary.
    pub run_summary: TrainingRunSummary,
    /// Final bundle receipt.
    pub final_bundle_receipt: ModelIoArtifactReceipt,
    /// Step receipt digests.
    pub step_receipt_digests: Vec<String>,
    /// Stable checkpoint digest.
    pub checkpoint_digest: String,
}

impl PsionicLegalSftCheckpointSummary {
    fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.checkpoint_digest.clear();
        stable_json_digest(b"psionic_legal_sft_checkpoint|", &clone)
    }
}

/// Machine-readable receipt emitted by one `psionic-train sft` run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsionicTrainingReceipt {
    /// Schema version.
    pub schema_version: String,
    /// Stable run id.
    pub run_id: String,
    /// Trainer implementation id.
    pub trainer: String,
    /// Train type.
    pub train_type: String,
    /// Base model id.
    pub base_model: String,
    /// Served model id.
    pub served_model_id: String,
    /// Base revision.
    pub base_model_revision: String,
    /// Base artifact mode.
    pub base_artifact_mode: PsionicLegalSftBaseArtifactMode,
    /// Base artifact digest.
    pub base_served_artifact_digest: String,
    /// Optional loaded artifact hashes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loaded_artifacts: Vec<PsionicLoadedTrainingArtifact>,
    /// Tokenizer digest.
    pub tokenizer_digest: String,
    /// Prompt template digest.
    pub prompt_template_digest: String,
    /// Smoke active trainable target.
    pub active_trainable_target: String,
    /// Resolved declared target modules.
    pub resolved_target_modules: Vec<String>,
    /// LoRA rank.
    pub lora_rank: usize,
    /// LoRA alpha.
    pub lora_alpha: f32,
    /// Max sequence length.
    pub max_seq_len: u32,
    /// Gradient accumulation steps.
    pub gradient_accumulation_steps: u32,
    /// Assistant-only loss.
    pub assistant_only_loss: bool,
    /// Empty think-block loss ignored.
    pub ignore_empty_think_loss: bool,
    /// Completed steps.
    pub completed_steps: u64,
    /// First recorded loss.
    pub initial_loss: f32,
    /// Final recorded loss.
    pub final_loss: f32,
    /// Whether final loss is below first loss.
    pub loss_improved: bool,
    /// Adapter path.
    pub adapter_artifact_path: String,
    /// Adapter digest.
    pub adapter_artifact_digest: String,
    /// Adapter identity digest.
    pub adapter_identity_digest: String,
    /// Loss curve path.
    pub loss_curve_path: String,
    /// Checkpoint summary path.
    pub checkpoint_summary_path: String,
    /// No Python was invoked.
    pub python_invoked: bool,
    /// No Python trainer artifact is required.
    pub python_artifacts_required: bool,
    /// Plain limitation statement for the current smoke.
    pub claim_boundary: String,
    /// Receipt digest.
    pub receipt_digest: String,
}

impl PsionicTrainingReceipt {
    /// Stable receipt digest with the digest field cleared.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.receipt_digest.clear();
        stable_json_digest(b"psionic_training_receipt|", &clone)
    }
}

/// Hash of a local artifact loaded by the SFT command.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicLoadedTrainingArtifact {
    /// Artifact role.
    pub role: String,
    /// Local path.
    pub path: String,
    /// SHA-256 digest.
    pub sha256: String,
    /// Byte count.
    pub byte_len: u64,
}

/// Paths and receipt returned after running the SFT command.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsionicLegalSftRunArtifacts {
    /// Adapter path.
    pub adapter_artifact_path: String,
    /// Receipt path.
    pub receipt_path: String,
    /// Loss curve path.
    pub loss_curve_path: String,
    /// Checkpoint summary path.
    pub checkpoint_summary_path: String,
    /// Receipt payload.
    pub receipt: PsionicTrainingReceipt,
}

/// Error returned by the legal SFT command.
#[derive(Debug, Error)]
pub enum PsionicLegalSftError {
    #[error("invalid legal SFT config: {detail}")]
    InvalidConfig { detail: String },
    #[error("legal SFT I/O failed at `{path}`: {message}")]
    Io { path: String, message: String },
    #[error("legal SFT JSON failed at `{path}`: {message}")]
    Json { path: String, message: String },
    #[error("legal SFT serialization failed: {message}")]
    Serialization { message: String },
    #[error("legal SFT trainer failed: {0}")]
    Trainer(#[from] OpenAdapterTrainingExecutionError),
    #[error("legal SFT export failed: {0}")]
    OpenAdapter(#[from] OpenAdapterSftError),
    #[error("legal SFT core failed: {0}")]
    Core(#[from] TrainingCoreError),
}

/// Runs `psionic-train sft --config <path>` args and returns the receipt.
pub fn run_psionic_legal_sft_cli(
    args: &[String],
) -> Result<PsionicTrainingReceipt, PsionicLegalSftError> {
    let config_path = parse_config_path(args)?;
    let artifacts = run_psionic_legal_sft_config_path(config_path)?;
    Ok(artifacts.receipt)
}

/// Runs one SFT config from a JSON file.
pub fn run_psionic_legal_sft_config_path(
    path: impl AsRef<Path>,
) -> Result<PsionicLegalSftRunArtifacts, PsionicLegalSftError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| PsionicLegalSftError::Io {
        path: path.display().to_string(),
        message: error.to_string(),
    })?;
    let config: PsionicLegalSftConfig =
        serde_json::from_slice(bytes.as_slice()).map_err(|error| PsionicLegalSftError::Json {
            path: path.display().to_string(),
            message: error.to_string(),
        })?;
    run_psionic_legal_sft_config(&config)
}

/// Runs one parsed SFT config.
pub fn run_psionic_legal_sft_config(
    config: &PsionicLegalSftConfig,
) -> Result<PsionicLegalSftRunArtifacts, PsionicLegalSftError> {
    config.validate()?;
    let output_dir = PathBuf::from(&config.output_dir);
    fs::create_dir_all(&output_dir).map_err(|error| PsionicLegalSftError::Io {
        path: output_dir.display().to_string(),
        message: error.to_string(),
    })?;

    let loaded_artifacts = load_declared_artifacts(config)?;
    let tokenizer_digest = tokenizer_digest(config, loaded_artifacts.as_slice());
    let optimizer = optimizer_config(config);
    let backend = OpenAdapterTrainingExecutionBackend::new(
        OpenAdapterExecutionConfig {
            run_id: config.run_id.clone(),
            checkpoint_family: String::from(QWEN36_LEGAL_CHECKPOINT_FAMILY),
            execution_backend_label: String::from(OPEN_ADAPTER_QWEN36_LEGAL_CUDA_BACKEND_LABEL),
            admissible_model_family: OpenAdapterAdmissibleModelFamily::Qwen36LegalDecoderLmHeadLora,
            budget: TrainingLoopBudget::new(config.max_steps, 1, 1)?,
            batch_size: config.batch_size,
            precision_policy: OpenAdapterPrecisionPolicy::F32Reference,
            model: OpenAdapterReferenceModel {
                base_model_id: config.base_model.clone(),
                base_model_revision: config.base_model_revision.clone(),
                base_served_artifact_digest: config.base_served_artifact_digest.clone(),
                tokenizer: TokenizerDigest::new(
                    TokenizerFamily::BytePairEncoding,
                    tokenizer_digest.clone(),
                    config.vocab_size as u32,
                )
                .with_template_digest(config.prompt_template_digest.as_str()),
                hidden_size: config.hidden_size,
                vocab_size: config.vocab_size,
                target: OpenAdapterLmHeadTarget {
                    target_id: config.adapter_target_id.clone(),
                    lora_rank: config.lora_rank,
                    lora_alpha: config.lora_alpha,
                    optimizer,
                    optimizer_residency_policy: TrainingOptimizerResidencyPolicy::host_only(),
                },
            },
        },
        config
            .samples
            .clone()
            .into_iter()
            .map(PsionicLegalSftSample::into_open_adapter_sample)
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let outcome = run_open_adapter_sft_export(
        &backend,
        &OpenAdapterSftRunRequest {
            dataset_ref: config.dataset_ref.clone(),
            validator_policy_ref: config.validator_policy_ref.clone(),
            adapter_id: config.adapter_id.clone(),
            adapter_revision: config.adapter_revision.clone(),
            started_at_ms: config.started_at_ms,
            step_duration_ms: config.step_duration_ms,
        },
    )?;

    let adapter_path = output_dir.join("adapter.safetensors");
    outcome.write_artifact_to_path(&adapter_path)?;
    let loss_curve = loss_curve(config.run_id.as_str(), outcome.step_receipts.as_slice());
    let loss_curve_path = output_dir.join("loss_curve.json");
    write_json(&loss_curve_path, &loss_curve)?;
    let checkpoint = checkpoint_summary(&config.run_id, &outcome.summary.run_summary, &outcome)?;
    let checkpoint_path = output_dir.join("checkpoint_summary.json");
    write_json(&checkpoint_path, &checkpoint)?;

    let initial_loss = loss_curve
        .points
        .first()
        .map(|point| point.loss)
        .unwrap_or_default();
    let final_loss = loss_curve
        .points
        .last()
        .map(|point| point.loss)
        .unwrap_or_default();
    let mut receipt = PsionicTrainingReceipt {
        schema_version: String::from(PSIONIC_TRAINING_RECEIPT_SCHEMA_VERSION),
        run_id: config.run_id.clone(),
        trainer: String::from("psionic.open_adapter.qwen36_legal_lm_head_lora_sft.v1"),
        train_type: config.train_type.clone(),
        base_model: config.base_model.clone(),
        served_model_id: config.served_model_id.clone(),
        base_model_revision: config.base_model_revision.clone(),
        base_artifact_mode: config.base_artifact_mode,
        base_served_artifact_digest: config.base_served_artifact_digest.clone(),
        loaded_artifacts,
        tokenizer_digest,
        prompt_template_digest: config.prompt_template_digest.clone(),
        active_trainable_target: config.adapter_target_id.clone(),
        resolved_target_modules: resolve_qwen36_target_modules(&config.target_modules)?,
        lora_rank: config.lora_rank,
        lora_alpha: config.lora_alpha,
        max_seq_len: config.max_seq_len,
        gradient_accumulation_steps: config.gradient_accumulation_steps,
        assistant_only_loss: config.assistant_only_loss,
        ignore_empty_think_loss: config.ignore_empty_think_loss,
        completed_steps: outcome.summary.run_summary.completed_steps,
        initial_loss,
        final_loss,
        loss_improved: final_loss < initial_loss,
        adapter_artifact_path: adapter_path.display().to_string(),
        adapter_artifact_digest: outcome.summary.adapter_artifact_digest.clone(),
        adapter_identity_digest: outcome.summary.adapter_identity_digest.clone(),
        loss_curve_path: loss_curve_path.display().to_string(),
        checkpoint_summary_path: checkpoint_path.display().to_string(),
        python_invoked: false,
        python_artifacts_required: false,
        claim_boundary: String::from(
            "This smoke run is a real Rust-only adapter update over tiny legal hidden-state samples. It proves config loading, adapter-only training, deterministic export, loss receipts, and checkpoint receipts. It does not claim full Qwen3.6 dense-weight training or retained Harvey benchmark improvement.",
        ),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.stable_digest();
    let receipt_path = output_dir.join("training_receipt.json");
    write_json(&receipt_path, &receipt)?;

    Ok(PsionicLegalSftRunArtifacts {
        adapter_artifact_path: adapter_path.display().to_string(),
        receipt_path: receipt_path.display().to_string(),
        loss_curve_path: loss_curve_path.display().to_string(),
        checkpoint_summary_path: checkpoint_path.display().to_string(),
        receipt,
    })
}

fn parse_config_path(args: &[String]) -> Result<&Path, PsionicLegalSftError> {
    let index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--config" => {
                let Some(path) = args.get(index + 1) else {
                    return Err(PsionicLegalSftError::InvalidConfig {
                        detail: String::from("--config requires a path"),
                    });
                };
                return Ok(Path::new(path));
            }
            "--help" | "-h" => {
                return Err(PsionicLegalSftError::InvalidConfig {
                    detail: String::from("usage: psionic-train sft --config <path>"),
                });
            }
            other => {
                return Err(PsionicLegalSftError::InvalidConfig {
                    detail: format!("unsupported sft argument `{other}`"),
                });
            }
        }
    }
    Err(PsionicLegalSftError::InvalidConfig {
        detail: String::from("missing --config <path>"),
    })
}

fn optimizer_config(config: &PsionicLegalSftConfig) -> TrainingOptimizerConfig {
    let optimizer = TrainingOptimizerConfig::adamw(config.learning_rate, 0.9, 0.99, 1e-8);
    if let Some(gradient_clip_norm) = config.gradient_clip_norm {
        optimizer.with_gradient_clip_norm(gradient_clip_norm)
    } else {
        optimizer
    }
}

fn tokenizer_digest(
    config: &PsionicLegalSftConfig,
    artifacts: &[PsionicLoadedTrainingArtifact],
) -> String {
    artifacts
        .iter()
        .find(|artifact| artifact.role == "tokenizer")
        .map(|artifact| format!("sha256:{}", artifact.sha256))
        .unwrap_or_else(|| config.tokenizer_digest.clone())
}

fn load_declared_artifacts(
    config: &PsionicLegalSftConfig,
) -> Result<Vec<PsionicLoadedTrainingArtifact>, PsionicLegalSftError> {
    let mut artifacts = Vec::new();
    if let Some(path) = config.model_config_path.as_deref() {
        let loaded = load_artifact("model_config", path)?;
        let _: serde_json::Value = serde_json::from_slice(
            fs::read(path)
                .map_err(|error| PsionicLegalSftError::Io {
                    path: path.to_string(),
                    message: error.to_string(),
                })?
                .as_slice(),
        )
        .map_err(|error| PsionicLegalSftError::Json {
            path: path.to_string(),
            message: error.to_string(),
        })?;
        artifacts.push(loaded);
    }
    if let Some(path) = config.tokenizer_path.as_deref() {
        artifacts.push(load_artifact("tokenizer", path)?);
    }
    for path in &config.base_safetensors_paths {
        artifacts.push(load_artifact("base_safetensors", path)?);
    }
    Ok(artifacts)
}

fn load_artifact(
    role: &str,
    path: &str,
) -> Result<PsionicLoadedTrainingArtifact, PsionicLegalSftError> {
    let bytes = fs::read(path).map_err(|error| PsionicLegalSftError::Io {
        path: path.to_string(),
        message: error.to_string(),
    })?;
    Ok(PsionicLoadedTrainingArtifact {
        role: String::from(role),
        path: String::from(path),
        sha256: sha256_hex(bytes.as_slice()),
        byte_len: bytes.len() as u64,
    })
}

fn loss_curve(run_id: &str, receipts: &[TrainingStepReceipt]) -> PsionicLegalSftLossCurve {
    PsionicLegalSftLossCurve {
        schema_version: String::from(PSIONIC_LEGAL_SFT_LOSS_CURVE_SCHEMA_VERSION),
        run_id: String::from(run_id),
        points: receipts
            .iter()
            .map(|receipt| PsionicLegalSftLossPoint {
                step: receipt.schedule.global_step,
                loss: receipt.loss,
                batch_id: receipt.batch_id.clone(),
                receipt_digest: receipt.receipt_digest.clone(),
            })
            .collect(),
    }
}

fn checkpoint_summary(
    run_id: &str,
    run_summary: &TrainingRunSummary,
    outcome: &crate::OpenAdapterSftRunOutcome,
) -> Result<PsionicLegalSftCheckpointSummary, PsionicLegalSftError> {
    let mut checkpoint = PsionicLegalSftCheckpointSummary {
        schema_version: String::from(PSIONIC_LEGAL_SFT_CHECKPOINT_SCHEMA_VERSION),
        run_id: String::from(run_id),
        checkpoint_family: String::from(QWEN36_LEGAL_CHECKPOINT_FAMILY),
        run_summary: run_summary.clone(),
        final_bundle_receipt: outcome.final_bundle_receipt.clone(),
        step_receipt_digests: outcome
            .step_receipts
            .iter()
            .map(|receipt| receipt.receipt_digest.clone())
            .collect(),
        checkpoint_digest: String::new(),
    };
    checkpoint.checkpoint_digest = checkpoint.stable_digest();
    Ok(checkpoint)
}

fn resolve_qwen36_target_modules(
    target_modules: &[String],
) -> Result<Vec<String>, PsionicLegalSftError> {
    if target_modules.is_empty() || target_modules.iter().any(|value| value == "all-linear") {
        return Ok(QWEN36_DENSE_LORA_TARGET_MODULES
            .iter()
            .map(|value| String::from(*value))
            .collect());
    }
    let mut resolved = Vec::new();
    for module in target_modules {
        require_nonempty(module.as_str(), "target_module")?;
        resolved.push(module.clone());
    }
    Ok(resolved)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), PsionicLegalSftError> {
    let bytes =
        serde_json::to_vec_pretty(value).map_err(|error| PsionicLegalSftError::Serialization {
            message: error.to_string(),
        })?;
    fs::write(path, bytes).map_err(|error| PsionicLegalSftError::Io {
        path: path.display().to_string(),
        message: error.to_string(),
    })
}

fn require_nonempty(value: &str, field: &str) -> Result<(), PsionicLegalSftError> {
    if value.trim().is_empty() {
        return Err(PsionicLegalSftError::InvalidConfig {
            detail: format!("{field} must be present"),
        });
    }
    Ok(())
}

fn stable_json_digest(prefix: &[u8], value: &impl Serialize) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn default_tokenizer_digest() -> String {
    String::from(DEFAULT_QWEN36_TOKENIZER_DIGEST)
}

fn default_template_digest() -> String {
    String::from(DEFAULT_QWEN36_TEMPLATE_DIGEST)
}

fn default_adapter_target_id() -> String {
    String::from("lm_head")
}

/// Returns the default synthetic Qwen3.6 legal SFT smoke config.
#[must_use]
pub fn default_qwen36_legal_sft_smoke_config(
    output_dir: impl Into<String>,
) -> PsionicLegalSftConfig {
    PsionicLegalSftConfig {
        schema_version: String::from(PSIONIC_LEGAL_SFT_CONFIG_SCHEMA_VERSION),
        run_id: String::from("qwen36-legal-sft-smoke"),
        train_type: String::from("qlora"),
        base_model: String::from("Qwen/Qwen3.6-27B"),
        served_model_id: String::from("qwen3.6-27b"),
        base_model_revision: String::from("qwen3.6-27b-smoke-revision"),
        base_artifact_mode: PsionicLegalSftBaseArtifactMode::SyntheticHiddenStateSmoke,
        base_served_artifact_digest: String::from(DEFAULT_QWEN36_BASE_DIGEST),
        base_safetensors_paths: Vec::new(),
        model_config_path: None,
        tokenizer_path: None,
        tokenizer_digest: default_tokenizer_digest(),
        prompt_template_digest: default_template_digest(),
        hidden_size: 4,
        vocab_size: 256,
        adapter_target_id: default_adapter_target_id(),
        target_modules: vec![String::from("all-linear")],
        lora_rank: 16,
        lora_alpha: 32.0,
        lora_dropout: 0.0,
        learning_rate: 0.12,
        epochs: 1,
        max_seq_len: 8192,
        gradient_accumulation_steps: 8,
        max_steps: 8,
        batch_size: 2,
        assistant_only_loss: true,
        ignore_empty_think_loss: true,
        gradient_clip_norm: Some(1.0),
        started_at_ms: 1_000,
        step_duration_ms: 20,
        dataset_ref: String::from("dataset://openagents/legal-benchmark/harvey-smoke@v1"),
        validator_policy_ref: String::from("policy://validator/legal-benchmark/qwen36-smoke"),
        adapter_id: String::from("qwen36-27b-legal-smoke"),
        adapter_revision: String::from("r1"),
        output_dir: output_dir.into(),
        samples: vec![
            sample(
                "legal-smoke-a",
                "legal-record-a",
                [1.0, 0.0, 0.0, 0.0],
                12,
                41,
            ),
            sample(
                "legal-smoke-b",
                "legal-record-b",
                [0.0, 1.0, 0.0, 0.0],
                35,
                39,
            ),
            sample(
                "legal-smoke-c",
                "legal-record-c",
                [0.0, 0.0, 1.0, 0.0],
                62,
                47,
            ),
            sample(
                "legal-smoke-d",
                "legal-record-d",
                [0.0, 0.0, 0.0, 1.0],
                90,
                44,
            ),
            sample(
                "legal-smoke-e",
                "legal-record-e",
                [0.6, 0.4, 0.0, 0.0],
                118,
                52,
            ),
            sample(
                "legal-smoke-f",
                "legal-record-f",
                [0.0, 0.2, 0.5, 0.3],
                143,
                49,
            ),
        ],
    }
}

fn sample(
    sample_id: &str,
    legal_training_record_id: &str,
    hidden: [f32; 4],
    target_token_id: u32,
    source_token_count: u32,
) -> PsionicLegalSftSample {
    PsionicLegalSftSample {
        sample_id: String::from(sample_id),
        legal_training_record_id: String::from(legal_training_record_id),
        final_hidden_state: hidden.to_vec(),
        target_token_id,
        source_token_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_qwen36_sft_cli_smoke_writes_adapter_loss_curve_and_receipt()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let config =
            default_qwen36_legal_sft_smoke_config(temp.path().join("run").display().to_string());
        let artifacts = run_psionic_legal_sft_config(&config)?;

        assert!(Path::new(&artifacts.adapter_artifact_path).is_file());
        assert!(Path::new(&artifacts.receipt_path).is_file());
        assert!(Path::new(&artifacts.loss_curve_path).is_file());
        assert!(Path::new(&artifacts.checkpoint_summary_path).is_file());
        assert!(!artifacts.receipt.python_invoked);
        assert!(!artifacts.receipt.python_artifacts_required);
        assert_eq!(artifacts.receipt.completed_steps, config.max_steps);
        assert_eq!(
            artifacts.receipt.resolved_target_modules,
            QWEN36_DENSE_LORA_TARGET_MODULES
                .iter()
                .map(|value| String::from(*value))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            artifacts.receipt.receipt_digest,
            artifacts.receipt.stable_digest()
        );
        assert!(artifacts.receipt.final_loss.is_finite());
        assert!(artifacts.receipt.initial_loss.is_finite());
        Ok(())
    }

    #[test]
    fn legal_qwen36_sft_cli_rejects_python_style_dropout_randomness() {
        let mut config = default_qwen36_legal_sft_smoke_config("target/test-legal-sft");
        config.lora_dropout = 0.1;
        let error = run_psionic_legal_sft_config(&config).expect_err("dropout must be refused");
        assert!(error.to_string().contains("lora_dropout = 0.0"));
    }
}

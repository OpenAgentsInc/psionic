use std::path::Path;

use psionic_adapters::{AdapterArtifactIdentity, LmHeadLoraAdapterArtifact, LmHeadLoraLoadError};
use psionic_data::{
    LEGAL_BENCHMARK_TRAINING_RECORD_SCHEMA_VERSION, TokenizerDigest, TokenizerFamily,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    FixedBudgetTrainingRun, ModelAdapterDelta, ModelIoArtifactReceipt, ModelIoError,
    OPEN_ADAPTER_QWEN35_LEGAL_ADAPTER_FAMILY, OPEN_ADAPTER_QWEN35_LEGAL_CUDA_BACKEND_LABEL,
    OpenAdapterAdmissibleModelFamily, OpenAdapterArtifactExportRequest,
    OpenAdapterGradientBatchRecord, OpenAdapterHiddenStateSample, OpenAdapterLmHeadTarget,
    OpenAdapterPrecisionPolicy, OpenAdapterReferenceModel, OpenAdapterSftError,
    OpenAdapterTrainingExecutionBackend, OpenAdapterTrainingExecutionError, PortableModelBundle,
    PortableTokenizerAssetFormat, PortableTokenizerBinding, TrainingCoreError, TrainingLoopBudget,
    TrainingOptimizerConfig, TrainingOptimizerResidencyPolicy, TrainingRunSummary,
    TrainingStepReceipt,
};

/// Stable lane id for the first legal benchmark Qwen adapter smoke.
pub const QWEN_LEGAL_ADAPTER_SFT_LANE_ID: &str = "qwen_legal_adapter_sft_v1";
/// Public base-model id for the bounded smoke lane.
pub const QWEN35_4B_LEGAL_SMOKE_MODEL_ID: &str = "Qwen/Qwen3.5-4B";
/// Stable served-model id used by Psionic serving and eval metadata.
pub const QWEN35_4B_LEGAL_SMOKE_SERVED_MODEL_ID: &str = "qwen3.5-4b";
/// Retained target model for the Harvey legal hillclimb lane.
pub const QWEN36_35B_A3B_LEGAL_RETAINED_MODEL_ID: &str = "Qwen/Qwen3.6-35B-A3B";
/// Stable served-model id for the retained target lane.
pub const QWEN36_35B_A3B_LEGAL_RETAINED_SERVED_MODEL_ID: &str = "qwen3.6-35b-a3b";
/// Model-family acceptance label expected by Psionic qwen35 runtime gates.
pub const QWEN35_LEGAL_MODEL_FAMILY_ACCEPTANCE_LABEL: &str = "qwen35";
/// Stable checkpoint family for the first Qwen legal adapter lane.
pub const QWEN_LEGAL_ADAPTER_CHECKPOINT_FAMILY: &str = "psionic.qwen35_4b.legal_adapter_sft";
/// Stable schema version for typed Qwen legal adapter checkpoints.
pub const QWEN_LEGAL_ADAPTER_CHECKPOINT_SCHEMA_VERSION: &str =
    "psionic.qwen_legal_adapter_checkpoint.v1";
/// Stable schema version for Autopilot4-importable score metadata.
pub const QWEN_LEGAL_SCORE_IMPORT_BUNDLE_SCHEMA_VERSION: &str =
    "psionic.qwen_legal_score_import_bundle.v1";
/// Stable schema version for the legal RL hillclimb plan.
pub const QWEN_LEGAL_RL_HILLCLIMB_PLAN_SCHEMA_VERSION: &str =
    "psionic.qwen_legal_rl_hillclimb_plan.v1";
/// Stable plan id for the next Harvey legal hillclimb phase.
pub const QWEN_LEGAL_RL_HILLCLIMB_PLAN_ID: &str = "qwen_legal_rl_hillclimb_plan_v1";
/// Blueprint frontier consumed by the Psionic legal RL plan.
pub const QWEN_LEGAL_BLUEPRINT_OPTIMIZER_FRONTIER_REF: &str =
    "blueprint://harvey_legal_qwen_optimizer_frontier/optimizer_frontier_001";
/// Stable target-set id for the first narrow LM-head-only adapter.
pub const QWEN_LEGAL_ADAPTER_TARGET_SET_ID: &str = "qwen3.5-4b.legal.lm_head_lora.v1";
/// Stable adapter target id for the first smoke lane.
pub const QWEN_LEGAL_ADAPTER_TARGET_ID: &str = "lm_head";
/// Stable LoRA rank for the first Qwen legal adapter smoke.
pub const QWEN_LEGAL_ADAPTER_LORA_RANK: usize = 8;
/// Stable LoRA alpha for the first Qwen legal adapter smoke.
pub const QWEN_LEGAL_ADAPTER_LORA_ALPHA: f32 = 16.0;
/// Synthetic artifact digest admitted only for the deterministic unit smoke.
pub const QWEN_LEGAL_SYNTHETIC_SMOKE_BASE_ARTIFACT_DIGEST: &str =
    "sha256:synthetic-qwen35-4b-legal-smoke";

/// Whether a base model binding is a deterministic smoke fixture or a real artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLegalBaseArtifactMode {
    /// Deterministic hidden-state smoke without full Qwen weights.
    SyntheticHiddenStateSmoke,
    /// Real Qwen artifact must be materialized and explicitly bound.
    RealArtifactRequired,
}

/// Explicit first target set for the Qwen legal adapter smoke lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalAdapterTargetSet {
    /// Stable target-set identifier.
    pub target_set_id: String,
    /// Stable adapter target identifier.
    pub adapter_target_id: String,
    /// Stable LoRA rank.
    pub lora_rank: usize,
    /// Stable LoRA alpha.
    pub lora_alpha: f32,
    /// Evolution note for widening beyond LM-head LoRA.
    pub evolution_note: String,
}

impl QwenLegalAdapterTargetSet {
    fn validate(&self) -> Result<(), QwenLegalAdapterSftError> {
        if self.target_set_id != QWEN_LEGAL_ADAPTER_TARGET_SET_ID {
            return Err(QwenLegalAdapterSftError::InvalidTargetSet {
                detail: String::from("target-set id drifted from the bounded Qwen legal lane"),
            });
        }
        if self.adapter_target_id != QWEN_LEGAL_ADAPTER_TARGET_ID {
            return Err(QwenLegalAdapterSftError::InvalidTargetSet {
                detail: format!(
                    "Qwen legal smoke target surface must stay `{QWEN_LEGAL_ADAPTER_TARGET_ID}`"
                ),
            });
        }
        if self.lora_rank != QWEN_LEGAL_ADAPTER_LORA_RANK {
            return Err(QwenLegalAdapterSftError::InvalidTargetSet {
                detail: format!(
                    "Qwen legal target set must stay rank {QWEN_LEGAL_ADAPTER_LORA_RANK}"
                ),
            });
        }
        if (self.lora_alpha - QWEN_LEGAL_ADAPTER_LORA_ALPHA).abs() > f32::EPSILON {
            return Err(QwenLegalAdapterSftError::InvalidTargetSet {
                detail: format!(
                    "Qwen legal target set must stay alpha {QWEN_LEGAL_ADAPTER_LORA_ALPHA}"
                ),
            });
        }
        if self.evolution_note.trim().is_empty() {
            return Err(QwenLegalAdapterSftError::InvalidTargetSet {
                detail: String::from("target-set evolution note must be present"),
            });
        }
        Ok(())
    }

    /// Returns the stable trainable parameter count for this target set.
    #[must_use]
    pub fn parameter_count(&self, hidden_size: usize, vocab_size: usize) -> u64 {
        u64::try_from(
            self.lora_rank
                .saturating_mul(hidden_size.saturating_add(vocab_size)),
        )
        .unwrap_or(u64::MAX)
    }
}

/// Returns the canonical first Qwen legal adapter target set.
#[must_use]
pub fn canonical_qwen_legal_adapter_target_set() -> QwenLegalAdapterTargetSet {
    QwenLegalAdapterTargetSet {
        target_set_id: String::from(QWEN_LEGAL_ADAPTER_TARGET_SET_ID),
        adapter_target_id: String::from(QWEN_LEGAL_ADAPTER_TARGET_ID),
        lora_rank: QWEN_LEGAL_ADAPTER_LORA_RANK,
        lora_alpha: QWEN_LEGAL_ADAPTER_LORA_ALPHA,
        evolution_note: String::from(
            "The first smoke trains only an LM-head LoRA adapter from frozen hidden states. Attention and MLP LoRA targets are admitted only after the legal-record, checkpoint, export, and eval-import loop is green.",
        ),
    }
}

/// Compatibility binding between the trainer and one served Qwen base lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenLegalServedBaseModelBinding {
    /// Public model id, fixed to the first smoke target.
    pub public_model_id: String,
    /// Stable served-model id published by Psionic serving metadata.
    pub served_model_id: String,
    /// Psionic model-family acceptance label.
    pub model_family_acceptance_label: String,
    /// Stable base-model revision.
    pub base_model_revision: String,
    /// Stable served-artifact digest for the frozen base.
    pub base_served_artifact_digest: String,
    /// Optional materialized artifact path for real-model execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
    /// Whether this binding is synthetic smoke or real artifact execution.
    pub artifact_mode: QwenLegalBaseArtifactMode,
    /// Tokenizer identity the lane must preserve.
    pub tokenizer: TokenizerDigest,
    /// Prompt or chat-template digest the legal dataset was rendered against.
    pub prompt_template_digest: String,
    /// Hidden width surfaced by the served base or synthetic smoke fixture.
    pub hidden_size: usize,
    /// Maximum context window expected by the run.
    pub context_window_tokens: u32,
}

impl QwenLegalServedBaseModelBinding {
    /// Returns the served vocabulary width derived from the tokenizer.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        usize::try_from(self.tokenizer.vocab_size).unwrap_or(usize::MAX)
    }

    /// Returns `model@revision` for portable tokenizer/model binding.
    #[must_use]
    pub fn base_model_ref(&self) -> String {
        format!("{}@{}", self.public_model_id, self.base_model_revision)
    }

    fn validate(&self) -> Result<(), QwenLegalAdapterSftError> {
        if self.public_model_id != QWEN35_4B_LEGAL_SMOKE_MODEL_ID {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from("public model id drifted from Qwen3.5-4B smoke target"),
            });
        }
        if self.served_model_id != QWEN35_4B_LEGAL_SMOKE_SERVED_MODEL_ID {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from("served model id drifted from qwen3.5-4b"),
            });
        }
        if self.model_family_acceptance_label != QWEN35_LEGAL_MODEL_FAMILY_ACCEPTANCE_LABEL {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from("model family acceptance label must stay qwen35"),
            });
        }
        if self.base_model_revision.trim().is_empty() {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from("base model revision must be present"),
            });
        }
        if self.base_served_artifact_digest.trim().is_empty() {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from("base artifact digest must be present"),
            });
        }
        if self.tokenizer.tokenizer_digest.trim().is_empty() {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from("tokenizer digest must be present"),
            });
        }
        if self.tokenizer.family != TokenizerFamily::BytePairEncoding {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from("Qwen3.5 smoke expects a byte-pair tokenizer family"),
            });
        }
        match &self.tokenizer.template_digest {
            Some(template_digest) if template_digest == &self.prompt_template_digest => {}
            Some(_) => {
                return Err(QwenLegalAdapterSftError::Compatibility {
                    detail: String::from(
                        "served tokenizer template digest drifted from prompt template digest",
                    ),
                });
            }
            None => {
                return Err(QwenLegalAdapterSftError::Compatibility {
                    detail: String::from("tokenizer template digest must be bound"),
                });
            }
        }
        if self.hidden_size == 0 || self.context_window_tokens == 0 || self.vocab_size() == 0 {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from("served model shape and context window must be non-zero"),
            });
        }
        match self.artifact_mode {
            QwenLegalBaseArtifactMode::SyntheticHiddenStateSmoke => {
                if self.base_served_artifact_digest
                    != QWEN_LEGAL_SYNTHETIC_SMOKE_BASE_ARTIFACT_DIGEST
                {
                    return Err(QwenLegalAdapterSftError::Compatibility {
                        detail: String::from(
                            "synthetic smoke must use the explicit synthetic base artifact digest",
                        ),
                    });
                }
                if self.artifact_path.is_some() {
                    return Err(QwenLegalAdapterSftError::Compatibility {
                        detail: String::from(
                            "synthetic smoke must not pretend a real Qwen artifact path is bound",
                        ),
                    });
                }
            }
            QwenLegalBaseArtifactMode::RealArtifactRequired => {
                if self.base_served_artifact_digest
                    == QWEN_LEGAL_SYNTHETIC_SMOKE_BASE_ARTIFACT_DIGEST
                {
                    return Err(QwenLegalAdapterSftError::Compatibility {
                        detail: String::from("real artifact execution cannot use synthetic digest"),
                    });
                }
                let Some(path) = &self.artifact_path else {
                    return Err(QwenLegalAdapterSftError::Compatibility {
                        detail: String::from(
                            "real artifact execution requires an explicit artifact path",
                        ),
                    });
                };
                if path.trim().is_empty() {
                    return Err(QwenLegalAdapterSftError::Compatibility {
                        detail: String::from("real artifact path must be non-empty"),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validates that a real artifact path is materialized on local disk.
    pub fn validate_real_artifact_materialized(&self) -> Result<(), QwenLegalAdapterSftError> {
        self.validate()?;
        if self.artifact_mode != QwenLegalBaseArtifactMode::RealArtifactRequired {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: String::from(
                    "materialized artifact validation only applies to real artifact execution",
                ),
            });
        }
        let path = self.artifact_path.as_deref().ok_or_else(|| {
            QwenLegalAdapterSftError::Compatibility {
                detail: String::from("real artifact execution requires an explicit artifact path"),
            }
        })?;
        if !Path::new(path).is_file() {
            return Err(QwenLegalAdapterSftError::Compatibility {
                detail: format!("real artifact path is not materialized: {path}"),
            });
        }
        Ok(())
    }
}

/// Legal benchmark dataset binding consumed by the Qwen adapter smoke lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenLegalDatasetBinding {
    /// Stable dataset reference.
    pub dataset_ref: String,
    /// Stable digest for the exported legal training record bundle.
    pub dataset_digest: String,
    /// Schema version required from `LegalBenchmarkTrainingRecord`.
    pub training_record_schema_version: String,
    /// Training split reference.
    pub train_split_ref: String,
    /// Validation split reference.
    pub validation_split_ref: String,
    /// Hidden-criterion exclusion policy reference.
    pub hidden_criterion_policy_ref: String,
}

impl QwenLegalDatasetBinding {
    fn validate(&self) -> Result<(), QwenLegalAdapterSftError> {
        require_nonempty(self.dataset_ref.as_str(), "dataset_ref")?;
        require_nonempty(self.dataset_digest.as_str(), "dataset_digest")?;
        require_nonempty(self.train_split_ref.as_str(), "train_split_ref")?;
        require_nonempty(self.validation_split_ref.as_str(), "validation_split_ref")?;
        require_nonempty(
            self.hidden_criterion_policy_ref.as_str(),
            "hidden_criterion_policy_ref",
        )?;
        if self.training_record_schema_version != LEGAL_BENCHMARK_TRAINING_RECORD_SCHEMA_VERSION {
            return Err(QwenLegalAdapterSftError::DatasetDrift {
                detail: format!(
                    "expected legal training record schema `{LEGAL_BENCHMARK_TRAINING_RECORD_SCHEMA_VERSION}`"
                ),
            });
        }
        Ok(())
    }
}

/// Eval-pack binding emitted with the smoke output for Autopilot4 import.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenLegalEvalPackBinding {
    /// Stable eval-pack id.
    pub eval_pack_id: String,
    /// Stable eval-pack digest.
    pub eval_pack_digest: String,
    /// Legal benchmark suite id.
    pub benchmark_suite_id: String,
    /// Retained/public smoke slice id.
    pub retained_slice_id: String,
    /// Scorer version or image digest used by the evaluator.
    pub scorer_version: String,
    /// Downstream import target for score/history dashboards.
    pub import_target: String,
}

impl QwenLegalEvalPackBinding {
    fn validate(&self) -> Result<(), QwenLegalAdapterSftError> {
        require_nonempty(self.eval_pack_id.as_str(), "eval_pack_id")?;
        require_nonempty(self.eval_pack_digest.as_str(), "eval_pack_digest")?;
        require_nonempty(self.benchmark_suite_id.as_str(), "benchmark_suite_id")?;
        require_nonempty(self.retained_slice_id.as_str(), "retained_slice_id")?;
        require_nonempty(self.scorer_version.as_str(), "scorer_version")?;
        require_nonempty(self.import_target.as_str(), "import_target")?;
        Ok(())
    }
}

/// One bounded Qwen legal LM-head supervision sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalLmHeadSupervisionSample {
    /// Stable sample identifier.
    pub sample_id: String,
    /// Final hidden state emitted by the frozen base before LM-head projection.
    pub final_hidden_state: Vec<f32>,
    /// Target token the adapter should increase likelihood for.
    pub target_token_id: u32,
    /// Approximate source-token count preserved for telemetry.
    pub source_token_count: u32,
    /// Stable source legal training record id.
    pub legal_training_record_id: String,
}

impl QwenLegalLmHeadSupervisionSample {
    /// Creates one bounded Qwen legal LM-head supervision sample.
    #[must_use]
    pub fn new(
        sample_id: impl Into<String>,
        final_hidden_state: Vec<f32>,
        target_token_id: u32,
        source_token_count: u32,
        legal_training_record_id: impl Into<String>,
    ) -> Self {
        Self {
            sample_id: sample_id.into(),
            final_hidden_state,
            target_token_id,
            source_token_count,
            legal_training_record_id: legal_training_record_id.into(),
        }
    }

    fn into_open_adapter_sample(
        self,
    ) -> Result<OpenAdapterHiddenStateSample, OpenAdapterTrainingExecutionError> {
        if self.legal_training_record_id.trim().is_empty() {
            return Err(OpenAdapterTrainingExecutionError::MissingSampleId);
        }
        OpenAdapterHiddenStateSample::new(
            self.sample_id,
            self.final_hidden_state,
            self.target_token_id,
            self.source_token_count,
        )
    }
}

/// Execution config for the first Qwen legal adapter smoke trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalAdapterSftConfig {
    /// Stable run identifier.
    pub run_id: String,
    /// Fixed training-loop budget.
    pub budget: TrainingLoopBudget,
    /// Deterministic batch size.
    pub batch_size: usize,
    /// Optimizer config applied to the frozen target set.
    pub optimizer: TrainingOptimizerConfig,
    /// Optimizer residency policy.
    pub optimizer_residency_policy: TrainingOptimizerResidencyPolicy,
}

impl QwenLegalAdapterSftConfig {
    fn validate(&self) -> Result<(), QwenLegalAdapterSftError> {
        require_nonempty(self.run_id.as_str(), "run_id")?;
        if self.batch_size == 0 {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("batch_size must be greater than zero"),
            });
        }
        Ok(())
    }
}

/// Higher-level full-run request for the Qwen legal adapter smoke lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenLegalAdapterSftRunRequest {
    /// Stable dataset binding from legal benchmark training records.
    pub dataset_binding: QwenLegalDatasetBinding,
    /// Eval pack bound before the run starts.
    pub eval_pack_binding: QwenLegalEvalPackBinding,
    /// Stable validator policy reference.
    pub validator_policy_ref: String,
    /// Stable adapter identifier.
    pub adapter_id: String,
    /// Stable adapter revision.
    pub adapter_revision: String,
    /// Logical training start timestamp.
    pub started_at_ms: u64,
    /// Logical duration assigned to each trainer step.
    pub step_duration_ms: u64,
}

impl QwenLegalAdapterSftRunRequest {
    fn validate(&self) -> Result<(), QwenLegalAdapterSftError> {
        self.dataset_binding.validate()?;
        self.eval_pack_binding.validate()?;
        require_nonempty(self.validator_policy_ref.as_str(), "validator_policy_ref")?;
        require_nonempty(self.adapter_id.as_str(), "adapter_id")?;
        require_nonempty(self.adapter_revision.as_str(), "adapter_revision")?;
        if self.step_duration_ms == 0 {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("step_duration_ms must be greater than zero"),
            });
        }
        Ok(())
    }
}

/// Typed exported adapter artifact emitted by the Qwen legal adapter trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalAdapterExportedArtifact {
    /// Stable served compatibility digest.
    pub compatibility_digest: String,
    /// Stable dataset digest.
    pub dataset_digest: String,
    /// Stable eval-pack digest.
    pub eval_pack_digest: String,
    /// Stable adapter identity.
    pub adapter_identity: AdapterArtifactIdentity,
    /// Stable adapter identity digest.
    pub adapter_identity_digest: String,
    /// Stable adapter-artifact digest.
    pub adapter_artifact_digest: String,
    /// LoRA alpha needed to reload the artifact.
    pub adapter_alpha: f32,
    /// Raw `safetensors` artifact bytes.
    pub adapter_bytes: Vec<u8>,
}

impl QwenLegalAdapterExportedArtifact {
    /// Reloads the exported artifact through the shared LM-head LoRA parser.
    pub fn load_lm_head_lora_artifact(
        &self,
    ) -> Result<LmHeadLoraAdapterArtifact, QwenLegalAdapterSftError> {
        Ok(LmHeadLoraAdapterArtifact::from_safetensors_bytes(
            self.adapter_bytes.as_slice(),
            self.adapter_identity.clone(),
            self.adapter_alpha,
        )?)
    }
}

/// Typed checkpoint snapshot for the Qwen legal adapter smoke lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalAdapterCheckpoint {
    /// Stable checkpoint schema version.
    pub schema_version: String,
    /// Stable checkpoint identifier.
    pub checkpoint_id: String,
    /// Stable lane id.
    pub lane_id: String,
    /// Stable served compatibility digest.
    pub compatibility_digest: String,
    /// Stable target-set identifier.
    pub target_set_id: String,
    /// Stable base artifact digest.
    pub base_served_artifact_digest: String,
    /// Stable tokenizer contract digest.
    pub tokenizer_contract_digest: String,
    /// Stable prompt-template digest.
    pub prompt_template_digest: String,
    /// Stable dataset digest.
    pub dataset_digest: String,
    /// Stable eval-pack digest.
    pub eval_pack_digest: String,
    /// Logical checkpoint timestamp.
    pub saved_at_ms: u64,
    /// Exact serialized run state for continuation.
    pub run: FixedBudgetTrainingRun,
    /// Stable checkpoint digest.
    pub checkpoint_digest: String,
}

impl QwenLegalAdapterCheckpoint {
    /// Returns the stable digest over the checkpoint payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.checkpoint_digest.clear();
        stable_digest(b"psionic_qwen_legal_adapter_checkpoint|", &clone)
    }

    fn validate(&self) -> Result<(), QwenLegalAdapterSftError> {
        if self.schema_version != QWEN_LEGAL_ADAPTER_CHECKPOINT_SCHEMA_VERSION {
            return Err(QwenLegalAdapterSftError::InvalidCheckpoint {
                detail: String::from("checkpoint schema version drifted"),
            });
        }
        require_nonempty(self.checkpoint_id.as_str(), "checkpoint_id")?;
        if self.lane_id != QWEN_LEGAL_ADAPTER_SFT_LANE_ID {
            return Err(QwenLegalAdapterSftError::InvalidCheckpoint {
                detail: String::from("checkpoint lane id drifted"),
            });
        }
        if self.checkpoint_digest != self.stable_digest() {
            return Err(QwenLegalAdapterSftError::InvalidCheckpoint {
                detail: String::from("checkpoint digest drifted"),
            });
        }
        Ok(())
    }
}

/// Summary emitted by the higher-level Qwen legal adapter smoke lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalAdapterSftSummary {
    /// Final fixed-budget run summary.
    pub run_summary: TrainingRunSummary,
    /// Stable lane id.
    pub lane_id: String,
    /// Public base model id.
    pub public_model_id: String,
    /// Served base model id.
    pub served_model_id: String,
    /// Model-family acceptance label.
    pub model_family_acceptance_label: String,
    /// Stable served compatibility digest.
    pub compatibility_digest: String,
    /// Stable target-set identifier.
    pub target_set_id: String,
    /// Stable dataset reference.
    pub dataset_ref: String,
    /// Stable dataset digest.
    pub dataset_digest: String,
    /// Stable eval-pack id.
    pub eval_pack_id: String,
    /// Stable eval-pack digest.
    pub eval_pack_digest: String,
    /// Legal benchmark suite id.
    pub benchmark_suite_id: String,
    /// Retained/public smoke slice id.
    pub retained_slice_id: String,
    /// Stable validator policy reference.
    pub validator_policy_ref: String,
    /// Stable base artifact digest.
    pub base_served_artifact_digest: String,
    /// Stable tokenizer contract digest.
    pub tokenizer_contract_digest: String,
    /// Stable prompt-template digest.
    pub prompt_template_digest: String,
    /// Stable adapter-artifact digest.
    pub adapter_artifact_digest: String,
    /// Stable adapter-identity digest.
    pub adapter_identity_digest: String,
    /// Stable initial state-dict digest.
    pub initial_state_dict_digest: String,
    /// Stable final state-dict digest.
    pub final_state_dict_digest: String,
    /// Stable final checkpoint id.
    pub final_checkpoint_id: String,
}

/// Autopilot4-importable score/eval metadata emitted by the smoke lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenLegalScoreImportBundle {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable bundle id.
    pub bundle_id: String,
    /// Stable lane id.
    pub lane_id: String,
    /// Legal benchmark suite id.
    pub benchmark_suite_id: String,
    /// Retained/public smoke slice id.
    pub retained_slice_id: String,
    /// Stable dataset digest.
    pub dataset_digest: String,
    /// Stable eval-pack digest.
    pub eval_pack_digest: String,
    /// Stable base artifact digest.
    pub base_served_artifact_digest: String,
    /// Stable prompt-template digest.
    pub prompt_template_digest: String,
    /// Stable adapter-artifact digest.
    pub adapter_artifact_digest: String,
    /// Stable adapter-identity digest.
    pub adapter_identity_digest: String,
    /// Stable checkpoint reference.
    pub checkpoint_ref: String,
    /// Stable run summary digest.
    pub run_summary_digest: String,
    /// Downstream import target.
    pub import_target: String,
    /// Stable bundle digest.
    pub bundle_digest: String,
}

impl QwenLegalScoreImportBundle {
    /// Returns the stable digest over the bundle payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.bundle_digest.clear();
        stable_digest(b"psionic_qwen_legal_score_import_bundle|", &clone)
    }
}

/// Optimizer method used by one legal RL hillclimb target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLegalRlOptimizerMethod {
    /// GRPO-style policy optimization over accepted legal-agent rollouts.
    Grpo,
    /// GEPA-style trace reflection and candidate selection.
    GepaTraceSelection,
    /// MIPRO-style prompt/example search over Blueprint modules.
    MiproPromptSearch,
    /// Supervised adapter refresh from curated failure examples.
    SupervisedFineTuneRefresh,
    /// Manual legal-review correction before optimizer search.
    ManualReview,
}

/// One failure-family target for the legal RL hillclimb.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenLegalRlFailureFamilyTarget {
    /// Failure family shared with Blueprint and Autopilot4 reports.
    pub failure_family: String,
    /// Current miss rate or severity in basis points.
    pub baseline_miss_basis_points: u16,
    /// Target lift expected from this target in basis points.
    pub target_lift_basis_points: u16,
    /// Blueprint module slug receiving the resulting candidate.
    pub blueprint_module_slug: String,
    /// Optimizer method assigned to the target.
    pub optimizer_method: QwenLegalRlOptimizerMethod,
    /// Reward signal or scalarization used by rollout selection.
    pub reward_signal: String,
    /// Dataset request ref emitted for pylon/fine-tune work.
    pub dataset_request_ref: String,
}

/// Rollout policy for retained-slice legal RL collection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenLegalRlRolloutPolicy {
    /// Stable rollout window id.
    pub rollout_window_id: String,
    /// Minimum accepted rollouts before a candidate can be scored.
    pub min_accepted_rollouts: u16,
    /// Maximum quarantined rollouts tolerated in the window.
    pub max_quarantined_rollouts: u16,
    /// Validator policy applied before RL or fine-tune ingestion.
    pub validator_policy_ref: String,
    /// Maximum age for off-policy rollouts.
    pub off_policy_max_age_ms: u64,
    /// Whether retained Harvey slices are mandatory.
    pub retained_slice_required: bool,
    /// Reward model or reward program bundle reference.
    pub reward_program_ref: String,
    /// Sampling policy used by the rollout generator.
    pub sampling_policy_ref: String,
}

/// Typed bridge from the adapter-SFT lane to the legal RL hillclimb loop.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenLegalRlHillclimbPlan {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable plan id.
    pub plan_id: String,
    /// Blueprint objective id.
    pub objective_id: String,
    /// Base Psionic lane feeding adapter and dataset artifacts.
    pub base_lane_id: String,
    /// Current smoke served model id.
    pub smoke_served_model_id: String,
    /// Retained target model id.
    pub retained_target_model_id: String,
    /// Retained target served model id.
    pub retained_target_served_model_id: String,
    /// Dataset binding used by the current SFT smoke lane.
    pub dataset_binding: QwenLegalDatasetBinding,
    /// Eval pack bound before optimization starts.
    pub eval_pack_binding: QwenLegalEvalPackBinding,
    /// Retained rollout policy.
    pub rollout_policy: QwenLegalRlRolloutPolicy,
    /// Failure-family targets assigned to optimizer methods.
    pub targets: Vec<QwenLegalRlFailureFamilyTarget>,
    /// Blueprint optimizer frontier ref consumed by this plan.
    pub blueprint_optimizer_frontier_ref: String,
    /// Qwen smoke candidate ref being widened toward retained work.
    pub qwen_candidate_ref: String,
    /// Score import bundle emitted by the current smoke run.
    pub score_import_bundle_ref: String,
    /// Stable receipt id for plan materialization.
    pub receipt_id: String,
    /// Stable digest for the full plan payload.
    pub plan_digest: String,
}

impl QwenLegalRlHillclimbPlan {
    /// Returns the stable digest over the plan payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.plan_digest.clear();
        stable_digest(b"psionic_qwen_legal_rl_hillclimb_plan|", &clone)
    }

    fn validate(&self) -> Result<(), QwenLegalAdapterSftError> {
        if self.schema_version != QWEN_LEGAL_RL_HILLCLIMB_PLAN_SCHEMA_VERSION {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb plan schema version drifted"),
            });
        }
        if self.plan_id != QWEN_LEGAL_RL_HILLCLIMB_PLAN_ID {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb plan id drifted"),
            });
        }
        if self.objective_id != "harvey_legal_qwen_finetune_v1" {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb objective id drifted"),
            });
        }
        if self.base_lane_id != QWEN_LEGAL_ADAPTER_SFT_LANE_ID {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb base lane id drifted"),
            });
        }
        if self.smoke_served_model_id != QWEN35_4B_LEGAL_SMOKE_SERVED_MODEL_ID {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb smoke served model drifted"),
            });
        }
        if self.retained_target_model_id != QWEN36_35B_A3B_LEGAL_RETAINED_MODEL_ID
            || self.retained_target_served_model_id != QWEN36_35B_A3B_LEGAL_RETAINED_SERVED_MODEL_ID
        {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb retained target must be Qwen3.6-35B-A3B"),
            });
        }
        self.dataset_binding.validate()?;
        self.eval_pack_binding.validate()?;
        require_nonempty(
            self.rollout_policy.validator_policy_ref.as_str(),
            "rollout_policy.validator_policy_ref",
        )?;
        require_nonempty(
            self.rollout_policy.reward_program_ref.as_str(),
            "rollout_policy.reward_program_ref",
        )?;
        require_nonempty(
            self.rollout_policy.sampling_policy_ref.as_str(),
            "rollout_policy.sampling_policy_ref",
        )?;
        if !self.rollout_policy.retained_slice_required {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb requires retained Harvey slices"),
            });
        }
        if self.rollout_policy.min_accepted_rollouts <= self.rollout_policy.max_quarantined_rollouts
        {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("accepted legal rollouts must outnumber quarantined rollouts"),
            });
        }
        if self.targets.is_empty() {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb targets must be present"),
            });
        }
        for target in &self.targets {
            require_nonempty(target.failure_family.as_str(), "target.failure_family")?;
            require_nonempty(
                target.blueprint_module_slug.as_str(),
                "target.blueprint_module_slug",
            )?;
            require_nonempty(target.reward_signal.as_str(), "target.reward_signal")?;
            require_nonempty(
                target.dataset_request_ref.as_str(),
                "target.dataset_request_ref",
            )?;
            if target.baseline_miss_basis_points == 0 || target.target_lift_basis_points == 0 {
                return Err(QwenLegalAdapterSftError::InvalidConfig {
                    detail: String::from("legal RL targets require non-zero miss and lift values"),
                });
            }
        }
        require_nonempty(
            self.blueprint_optimizer_frontier_ref.as_str(),
            "blueprint_optimizer_frontier_ref",
        )?;
        require_nonempty(self.qwen_candidate_ref.as_str(), "qwen_candidate_ref")?;
        require_nonempty(
            self.score_import_bundle_ref.as_str(),
            "score_import_bundle_ref",
        )?;
        require_nonempty(self.receipt_id.as_str(), "receipt_id")?;
        if self.plan_digest != self.stable_digest() {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("legal RL hillclimb plan digest drifted"),
            });
        }
        Ok(())
    }
}

/// Builds the canonical next-phase legal RL hillclimb plan.
pub fn canonical_qwen_legal_rl_hillclimb_plan(
    dataset_binding: QwenLegalDatasetBinding,
    eval_pack_binding: QwenLegalEvalPackBinding,
    score_import_bundle_ref: impl Into<String>,
) -> Result<QwenLegalRlHillclimbPlan, QwenLegalAdapterSftError> {
    dataset_binding.validate()?;
    eval_pack_binding.validate()?;
    let mut plan = QwenLegalRlHillclimbPlan {
        schema_version: String::from(QWEN_LEGAL_RL_HILLCLIMB_PLAN_SCHEMA_VERSION),
        plan_id: String::from(QWEN_LEGAL_RL_HILLCLIMB_PLAN_ID),
        objective_id: String::from("harvey_legal_qwen_finetune_v1"),
        base_lane_id: String::from(QWEN_LEGAL_ADAPTER_SFT_LANE_ID),
        smoke_served_model_id: String::from(QWEN35_4B_LEGAL_SMOKE_SERVED_MODEL_ID),
        retained_target_model_id: String::from(QWEN36_35B_A3B_LEGAL_RETAINED_MODEL_ID),
        retained_target_served_model_id: String::from(
            QWEN36_35B_A3B_LEGAL_RETAINED_SERVED_MODEL_ID,
        ),
        dataset_binding,
        eval_pack_binding,
        rollout_policy: QwenLegalRlRolloutPolicy {
            rollout_window_id: String::from("harvey-legal-retained-20-task-window-001"),
            min_accepted_rollouts: 60,
            max_quarantined_rollouts: 12,
            validator_policy_ref: String::from("policy://validator/legal-benchmark/qwen-rl-v1"),
            off_policy_max_age_ms: 86_400_000,
            retained_slice_required: true,
            reward_program_ref: String::from(
                "blueprint://program.harvey_legal.final_self_check/reward_bundle_v1",
            ),
            sampling_policy_ref: String::from(
                "policy://psionic/legal-benchmark/retained-diverse-rollouts@v1",
            ),
        },
        targets: vec![
            QwenLegalRlFailureFamilyTarget {
                failure_family: String::from("document_coverage"),
                baseline_miss_basis_points: 4200,
                target_lift_basis_points: 900,
                blueprint_module_slug: String::from("harvey_legal.document_inventory"),
                optimizer_method: QwenLegalRlOptimizerMethod::MiproPromptSearch,
                reward_signal: String::from("document_manifest_recall_and_source_span_precision"),
                dataset_request_ref: String::from(
                    "dataset_request.harvey_legal.document_coverage.retained_001",
                ),
            },
            QwenLegalRlFailureFamilyTarget {
                failure_family: String::from("citation_evidence"),
                baseline_miss_basis_points: 3900,
                target_lift_basis_points: 860,
                blueprint_module_slug: String::from("harvey_legal.evidence_mapping"),
                optimizer_method: QwenLegalRlOptimizerMethod::GepaTraceSelection,
                reward_signal: String::from("citation_grounding_judge_agreement"),
                dataset_request_ref: String::from(
                    "dataset_request.harvey_legal.citation_evidence.retained_001",
                ),
            },
            QwenLegalRlFailureFamilyTarget {
                failure_family: String::from("legal_reasoning"),
                baseline_miss_basis_points: 3800,
                target_lift_basis_points: 820,
                blueprint_module_slug: String::from("harvey_legal.issue_fact_extraction"),
                optimizer_method: QwenLegalRlOptimizerMethod::Grpo,
                reward_signal: String::from("issue_rule_fact_conclusion_rubric_delta"),
                dataset_request_ref: String::from(
                    "dataset_request.harvey_legal.legal_reasoning.retained_001",
                ),
            },
            QwenLegalRlFailureFamilyTarget {
                failure_family: String::from("spreadsheet_reasoning"),
                baseline_miss_basis_points: 3300,
                target_lift_basis_points: 760,
                blueprint_module_slug: String::from("harvey_legal.evidence_mapping"),
                optimizer_method: QwenLegalRlOptimizerMethod::Grpo,
                reward_signal: String::from("spreadsheet_cell_lineage_and_arithmetic_exactness"),
                dataset_request_ref: String::from(
                    "dataset_request.harvey_legal.spreadsheet_reasoning.retained_001",
                ),
            },
            QwenLegalRlFailureFamilyTarget {
                failure_family: String::from("missing_fact"),
                baseline_miss_basis_points: 3100,
                target_lift_basis_points: 720,
                blueprint_module_slug: String::from("harvey_legal.issue_fact_extraction"),
                optimizer_method: QwenLegalRlOptimizerMethod::SupervisedFineTuneRefresh,
                reward_signal: String::from("missed_fact_recall_with_source_span_support"),
                dataset_request_ref: String::from(
                    "dataset_request.harvey_legal.missing_fact.retained_001",
                ),
            },
            QwenLegalRlFailureFamilyTarget {
                failure_family: String::from("pre_submit_self_check"),
                baseline_miss_basis_points: 2800,
                target_lift_basis_points: 560,
                blueprint_module_slug: String::from("harvey_legal.final_self_check"),
                optimizer_method: QwenLegalRlOptimizerMethod::GepaTraceSelection,
                reward_signal: String::from("self_check_rewrite_acceptance_before_submission"),
                dataset_request_ref: String::from(
                    "dataset_request.harvey_legal.pre_submit_self_check.retained_001",
                ),
            },
        ],
        blueprint_optimizer_frontier_ref: String::from(QWEN_LEGAL_BLUEPRINT_OPTIMIZER_FRONTIER_REF),
        qwen_candidate_ref: String::from("model_candidate.harvey_legal_qwen.qwen3_5_4b.smoke_001"),
        score_import_bundle_ref: score_import_bundle_ref.into(),
        receipt_id: String::from("receipt.psionic.qwen_legal_rl_hillclimb_plan_001"),
        plan_digest: String::new(),
    };
    plan.plan_digest = plan.stable_digest();
    plan.validate()?;
    Ok(plan)
}

/// Full higher-level Qwen legal adapter smoke outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenLegalAdapterSftRunOutcome {
    /// Step receipts emitted during the run.
    pub step_receipts: Vec<TrainingStepReceipt>,
    /// Gradient-production records emitted during the run.
    pub gradient_records: Vec<OpenAdapterGradientBatchRecord>,
    /// Summary and reproducibility metadata.
    pub summary: QwenLegalAdapterSftSummary,
    /// Initial adapter-only bundle receipt.
    pub initial_bundle_receipt: ModelIoArtifactReceipt,
    /// Final adapter-only bundle receipt.
    pub final_bundle_receipt: ModelIoArtifactReceipt,
    /// Typed adapter delta between the initial and final bundles.
    pub adapter_delta: ModelAdapterDelta,
    /// Typed exported artifact.
    pub exported_artifact: QwenLegalAdapterExportedArtifact,
    /// Final checkpoint snapshot.
    pub final_checkpoint: QwenLegalAdapterCheckpoint,
    /// Score/eval metadata for downstream import.
    pub score_import_bundle: QwenLegalScoreImportBundle,
    /// Next-phase RL hillclimb plan bound to this smoke run.
    pub rl_hillclimb_plan: QwenLegalRlHillclimbPlan,
}

/// First honest Qwen legal adapter-SFT smoke trainer.
#[derive(Clone, Debug)]
pub struct QwenLegalAdapterSftTrainer {
    target_set: QwenLegalAdapterTargetSet,
    base_binding: QwenLegalServedBaseModelBinding,
    compatibility_digest: String,
    backend: OpenAdapterTrainingExecutionBackend,
}

impl QwenLegalAdapterSftTrainer {
    /// Builds the first bounded Qwen legal adapter trainer.
    pub fn new(
        config: QwenLegalAdapterSftConfig,
        target_set: QwenLegalAdapterTargetSet,
        base_binding: QwenLegalServedBaseModelBinding,
        samples: Vec<QwenLegalLmHeadSupervisionSample>,
    ) -> Result<Self, QwenLegalAdapterSftError> {
        config.validate()?;
        target_set.validate()?;
        base_binding.validate()?;
        let compatibility_digest = stable_compatibility_digest(&target_set, &base_binding);
        let open_samples = samples
            .into_iter()
            .map(QwenLegalLmHeadSupervisionSample::into_open_adapter_sample)
            .collect::<Result<Vec<_>, _>>()?;
        let backend = OpenAdapterTrainingExecutionBackend::new(
            crate::OpenAdapterExecutionConfig {
                run_id: config.run_id,
                checkpoint_family: String::from(QWEN_LEGAL_ADAPTER_CHECKPOINT_FAMILY),
                execution_backend_label: String::from(OPEN_ADAPTER_QWEN35_LEGAL_CUDA_BACKEND_LABEL),
                admissible_model_family:
                    OpenAdapterAdmissibleModelFamily::Qwen35LegalDecoderLmHeadLora,
                budget: config.budget,
                batch_size: config.batch_size,
                precision_policy: OpenAdapterPrecisionPolicy::F32Reference,
                model: OpenAdapterReferenceModel {
                    base_model_id: base_binding.public_model_id.clone(),
                    base_model_revision: base_binding.base_model_revision.clone(),
                    base_served_artifact_digest: base_binding.base_served_artifact_digest.clone(),
                    tokenizer: base_binding.tokenizer.clone(),
                    hidden_size: base_binding.hidden_size,
                    vocab_size: base_binding.vocab_size(),
                    target: OpenAdapterLmHeadTarget {
                        target_id: target_set.adapter_target_id.clone(),
                        lora_rank: target_set.lora_rank,
                        lora_alpha: target_set.lora_alpha,
                        optimizer: config.optimizer,
                        optimizer_residency_policy: config.optimizer_residency_policy,
                    },
                },
            },
            open_samples,
        )?;
        Ok(Self {
            target_set,
            base_binding,
            compatibility_digest,
            backend,
        })
    }

    /// Returns the explicit target set.
    #[must_use]
    pub fn target_set(&self) -> &QwenLegalAdapterTargetSet {
        &self.target_set
    }

    /// Returns the served-base binding.
    #[must_use]
    pub fn base_binding(&self) -> &QwenLegalServedBaseModelBinding {
        &self.base_binding
    }

    /// Returns the stable served-compatibility digest.
    #[must_use]
    pub fn compatibility_digest(&self) -> &str {
        self.compatibility_digest.as_str()
    }

    /// Returns the underlying reusable open-adapter backend.
    #[must_use]
    pub fn backend(&self) -> &OpenAdapterTrainingExecutionBackend {
        &self.backend
    }

    /// Creates a fresh training run.
    pub fn initialize_run(&self) -> Result<FixedBudgetTrainingRun, QwenLegalAdapterSftError> {
        Ok(self.backend.initialize_run()?)
    }

    /// Advances one run for up to `step_limit` additional steps.
    pub fn advance_run(
        &self,
        run: &mut FixedBudgetTrainingRun,
        step_limit: Option<u64>,
        started_at_ms: u64,
        step_duration_ms: u64,
    ) -> Result<QwenLegalAdapterRunProgress, QwenLegalAdapterSftError> {
        if step_duration_ms == 0 {
            return Err(QwenLegalAdapterSftError::InvalidConfig {
                detail: String::from("step_duration_ms must be greater than zero"),
            });
        }
        let remaining_steps = run
            .summary()
            .budget
            .max_steps
            .saturating_sub(run.summary().completed_steps);
        let allowed_steps = step_limit.unwrap_or(remaining_steps).min(remaining_steps);
        let mut step_receipts = Vec::new();
        let mut gradient_records = Vec::new();
        for step_offset in 0..allowed_steps {
            let batch_index = run.completed_steps() as usize % self.backend.batches().len().max(1);
            let step_started_at_ms = started_at_ms + step_offset.saturating_mul(step_duration_ms);
            let step_finished_at_ms = step_started_at_ms + step_duration_ms;
            let (step_input, gradient_record) = self.backend.produce_step_input(
                run,
                batch_index,
                step_started_at_ms,
                step_finished_at_ms,
            )?;
            gradient_records.push(gradient_record);
            step_receipts.push(run.apply_step(step_input)?);
        }
        Ok(QwenLegalAdapterRunProgress {
            step_receipts,
            gradient_records,
        })
    }

    /// Saves one exact run-state checkpoint for later continuation.
    pub fn save_checkpoint(
        &self,
        checkpoint_id: impl Into<String>,
        run: &FixedBudgetTrainingRun,
        dataset_binding: &QwenLegalDatasetBinding,
        eval_pack_binding: &QwenLegalEvalPackBinding,
        saved_at_ms: u64,
    ) -> Result<QwenLegalAdapterCheckpoint, QwenLegalAdapterSftError> {
        let checkpoint_id = checkpoint_id.into();
        dataset_binding.validate()?;
        eval_pack_binding.validate()?;
        self.backend.snapshot_training_groups(run)?;
        let mut checkpoint = QwenLegalAdapterCheckpoint {
            schema_version: String::from(QWEN_LEGAL_ADAPTER_CHECKPOINT_SCHEMA_VERSION),
            checkpoint_id,
            lane_id: String::from(QWEN_LEGAL_ADAPTER_SFT_LANE_ID),
            compatibility_digest: self.compatibility_digest.clone(),
            target_set_id: self.target_set.target_set_id.clone(),
            base_served_artifact_digest: self.base_binding.base_served_artifact_digest.clone(),
            tokenizer_contract_digest: self.base_binding.tokenizer.stable_digest(),
            prompt_template_digest: self.base_binding.prompt_template_digest.clone(),
            dataset_digest: dataset_binding.dataset_digest.clone(),
            eval_pack_digest: eval_pack_binding.eval_pack_digest.clone(),
            saved_at_ms,
            run: run.clone(),
            checkpoint_digest: String::new(),
        };
        checkpoint.checkpoint_digest = checkpoint.stable_digest();
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Restores one previously saved exact run-state checkpoint.
    pub fn restore_run(
        &self,
        checkpoint: &QwenLegalAdapterCheckpoint,
    ) -> Result<FixedBudgetTrainingRun, QwenLegalAdapterSftError> {
        checkpoint.validate()?;
        if checkpoint.compatibility_digest != self.compatibility_digest
            || checkpoint.target_set_id != self.target_set.target_set_id
            || checkpoint.base_served_artifact_digest
                != self.base_binding.base_served_artifact_digest
            || checkpoint.tokenizer_contract_digest != self.base_binding.tokenizer.stable_digest()
            || checkpoint.prompt_template_digest != self.base_binding.prompt_template_digest
        {
            return Err(QwenLegalAdapterSftError::InvalidCheckpoint {
                detail: String::from("checkpoint compatibility drifted from active Qwen lane"),
            });
        }
        self.backend.snapshot_training_groups(&checkpoint.run)?;
        Ok(checkpoint.run.clone())
    }

    /// Runs the full cold-start Qwen legal adapter smoke lane.
    pub fn run_sft(
        &self,
        request: &QwenLegalAdapterSftRunRequest,
    ) -> Result<QwenLegalAdapterSftRunOutcome, QwenLegalAdapterSftError> {
        let run = self.initialize_run()?;
        self.run_from_existing_run(run, request, None)
    }

    /// Restores one checkpoint and continues until the fixed budget is reached.
    pub fn run_sft_from_checkpoint(
        &self,
        checkpoint: &QwenLegalAdapterCheckpoint,
        request: &QwenLegalAdapterSftRunRequest,
    ) -> Result<QwenLegalAdapterSftRunOutcome, QwenLegalAdapterSftError> {
        let run = self.restore_run(checkpoint)?;
        self.run_from_existing_run(run, request, Some(checkpoint.checkpoint_id.as_str()))
    }

    fn run_from_existing_run(
        &self,
        mut run: FixedBudgetTrainingRun,
        request: &QwenLegalAdapterSftRunRequest,
        resumed_from_checkpoint_id: Option<&str>,
    ) -> Result<QwenLegalAdapterSftRunOutcome, QwenLegalAdapterSftError> {
        request.validate()?;
        let initial_groups = self.backend.snapshot_training_groups(&run)?;
        let initial_bundle = self.bundle_from_groups(
            initial_groups.as_slice(),
            format!("checkpoint://{}/initial", self.backend.config().run_id),
        )?;
        let (_, initial_bundle_receipt) = initial_bundle.export_safetensors()?;
        let progress = self.advance_run(
            &mut run,
            None,
            request.started_at_ms,
            request.step_duration_ms,
        )?;
        let final_groups = self.backend.snapshot_training_groups(&run)?;
        let final_bundle = self.bundle_from_groups(
            final_groups.as_slice(),
            format!("checkpoint://{}/final", self.backend.config().run_id),
        )?;
        let (_, final_bundle_receipt) = final_bundle.export_safetensors()?;
        let adapter_delta = crate::PortableModelStateDict::derive_adapter_delta(
            &initial_bundle.state_dict,
            &final_bundle.state_dict,
            request.adapter_id.clone(),
        )?;
        let exported = self.backend.export_run_artifact(
            &run,
            &OpenAdapterArtifactExportRequest::new(
                request.dataset_binding.dataset_ref.clone(),
                request.validator_policy_ref.clone(),
                request.adapter_id.clone(),
                request.adapter_revision.clone(),
            ),
        )?;
        let exported_artifact = QwenLegalAdapterExportedArtifact {
            compatibility_digest: self.compatibility_digest.clone(),
            dataset_digest: request.dataset_binding.dataset_digest.clone(),
            eval_pack_digest: request.eval_pack_binding.eval_pack_digest.clone(),
            adapter_identity: exported.adapter_identity,
            adapter_identity_digest: exported.adapter_identity_digest,
            adapter_artifact_digest: exported.adapter_artifact_digest,
            adapter_alpha: self.target_set.lora_alpha,
            adapter_bytes: exported.adapter_bytes,
        };
        exported_artifact.load_lm_head_lora_artifact()?;
        let checkpoint_id = checkpoint_id_for(
            request.adapter_id.as_str(),
            request.adapter_revision.as_str(),
            resumed_from_checkpoint_id,
        );
        let final_checkpoint = self.save_checkpoint(
            checkpoint_id,
            &run,
            &request.dataset_binding,
            &request.eval_pack_binding,
            progress
                .step_receipts
                .last()
                .map(|receipt| receipt.timing.finished_at_ms)
                .unwrap_or(request.started_at_ms),
        )?;
        let run_summary = run.summary();
        let run_summary_digest = stable_digest(b"psionic_qwen_legal_run_summary|", &run_summary);
        let mut score_import_bundle = QwenLegalScoreImportBundle {
            schema_version: String::from(QWEN_LEGAL_SCORE_IMPORT_BUNDLE_SCHEMA_VERSION),
            bundle_id: format!(
                "{}-{}-score-import",
                request.adapter_id, request.adapter_revision
            ),
            lane_id: String::from(QWEN_LEGAL_ADAPTER_SFT_LANE_ID),
            benchmark_suite_id: request.eval_pack_binding.benchmark_suite_id.clone(),
            retained_slice_id: request.eval_pack_binding.retained_slice_id.clone(),
            dataset_digest: request.dataset_binding.dataset_digest.clone(),
            eval_pack_digest: request.eval_pack_binding.eval_pack_digest.clone(),
            base_served_artifact_digest: self.base_binding.base_served_artifact_digest.clone(),
            prompt_template_digest: self.base_binding.prompt_template_digest.clone(),
            adapter_artifact_digest: exported_artifact.adapter_artifact_digest.clone(),
            adapter_identity_digest: exported_artifact.adapter_identity_digest.clone(),
            checkpoint_ref: format!("checkpoint://{}", final_checkpoint.checkpoint_id),
            run_summary_digest,
            import_target: request.eval_pack_binding.import_target.clone(),
            bundle_digest: String::new(),
        };
        score_import_bundle.bundle_digest = score_import_bundle.stable_digest();
        let rl_hillclimb_plan = canonical_qwen_legal_rl_hillclimb_plan(
            request.dataset_binding.clone(),
            request.eval_pack_binding.clone(),
            score_import_bundle.bundle_id.clone(),
        )?;
        let summary = QwenLegalAdapterSftSummary {
            run_summary,
            lane_id: String::from(QWEN_LEGAL_ADAPTER_SFT_LANE_ID),
            public_model_id: self.base_binding.public_model_id.clone(),
            served_model_id: self.base_binding.served_model_id.clone(),
            model_family_acceptance_label: self.base_binding.model_family_acceptance_label.clone(),
            compatibility_digest: self.compatibility_digest.clone(),
            target_set_id: self.target_set.target_set_id.clone(),
            dataset_ref: request.dataset_binding.dataset_ref.clone(),
            dataset_digest: request.dataset_binding.dataset_digest.clone(),
            eval_pack_id: request.eval_pack_binding.eval_pack_id.clone(),
            eval_pack_digest: request.eval_pack_binding.eval_pack_digest.clone(),
            benchmark_suite_id: request.eval_pack_binding.benchmark_suite_id.clone(),
            retained_slice_id: request.eval_pack_binding.retained_slice_id.clone(),
            validator_policy_ref: request.validator_policy_ref.clone(),
            base_served_artifact_digest: self.base_binding.base_served_artifact_digest.clone(),
            tokenizer_contract_digest: self.base_binding.tokenizer.stable_digest(),
            prompt_template_digest: self.base_binding.prompt_template_digest.clone(),
            adapter_artifact_digest: exported_artifact.adapter_artifact_digest.clone(),
            adapter_identity_digest: exported_artifact.adapter_identity_digest.clone(),
            initial_state_dict_digest: initial_bundle_receipt.state_dict_digest.clone(),
            final_state_dict_digest: final_bundle_receipt.state_dict_digest.clone(),
            final_checkpoint_id: final_checkpoint.checkpoint_id.clone(),
        };
        Ok(QwenLegalAdapterSftRunOutcome {
            step_receipts: progress.step_receipts,
            gradient_records: progress.gradient_records,
            summary,
            initial_bundle_receipt,
            final_bundle_receipt,
            adapter_delta,
            exported_artifact,
            final_checkpoint,
            score_import_bundle,
            rl_hillclimb_plan,
        })
    }

    fn bundle_from_groups(
        &self,
        groups: &[crate::TrainingParameterGroupState],
        checkpoint_ref: String,
    ) -> Result<PortableModelBundle, QwenLegalAdapterSftError> {
        Ok(PortableModelBundle::from_training_groups(
            OPEN_ADAPTER_QWEN35_LEGAL_ADAPTER_FAMILY,
            self.base_binding.base_model_revision.clone(),
            QWEN_LEGAL_ADAPTER_CHECKPOINT_FAMILY,
            Some(checkpoint_ref),
            groups,
            PortableTokenizerBinding::new(
                self.base_binding.tokenizer.clone(),
                PortableTokenizerAssetFormat::PsionicDigest,
                self.base_binding.base_model_ref(),
            ),
            Some(self.base_binding.prompt_template_digest.clone()),
        )?)
    }
}

/// Progress emitted while advancing one Qwen legal run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalAdapterRunProgress {
    /// Step receipts emitted while advancing the run.
    pub step_receipts: Vec<TrainingStepReceipt>,
    /// Gradient-production records emitted while advancing the run.
    pub gradient_records: Vec<OpenAdapterGradientBatchRecord>,
}

/// Error surfaced by the higher-level Qwen legal adapter smoke lane.
#[derive(Debug, Error)]
pub enum QwenLegalAdapterSftError {
    #[error("Qwen legal adapter config is invalid: {detail}")]
    InvalidConfig { detail: String },
    #[error("Qwen legal served compatibility mismatch: {detail}")]
    Compatibility { detail: String },
    #[error("Qwen legal target set is invalid: {detail}")]
    InvalidTargetSet { detail: String },
    #[error("Qwen legal dataset binding is invalid: {detail}")]
    DatasetDrift { detail: String },
    #[error("Qwen legal checkpoint is invalid: {detail}")]
    InvalidCheckpoint { detail: String },
    #[error(transparent)]
    TrainingExecution(#[from] OpenAdapterTrainingExecutionError),
    #[error(transparent)]
    TrainingCore(#[from] TrainingCoreError),
    #[error(transparent)]
    OpenAdapterSft(#[from] OpenAdapterSftError),
    #[error(transparent)]
    ModelIo(#[from] ModelIoError),
    #[error(transparent)]
    AdapterLoad(#[from] LmHeadLoraLoadError),
}

fn checkpoint_id_for(
    adapter_id: &str,
    adapter_revision: &str,
    resumed_from_checkpoint_id: Option<&str>,
) -> String {
    match resumed_from_checkpoint_id {
        Some(previous) => format!("{adapter_id}-{adapter_revision}-continued-from-{previous}"),
        None => format!("{adapter_id}-{adapter_revision}-final"),
    }
}

fn stable_compatibility_digest(
    target_set: &QwenLegalAdapterTargetSet,
    base_binding: &QwenLegalServedBaseModelBinding,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_qwen_legal_adapter_compatibility|");
    hasher.update(target_set.target_set_id.as_bytes());
    hasher.update(b"|");
    hasher.update(target_set.adapter_target_id.as_bytes());
    hasher.update(b"|");
    hasher.update(target_set.lora_rank.to_string().as_bytes());
    hasher.update(b"|");
    hasher.update(target_set.lora_alpha.to_bits().to_le_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.public_model_id.as_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.served_model_id.as_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.model_family_acceptance_label.as_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.base_model_revision.as_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.base_served_artifact_digest.as_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.prompt_template_digest.as_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.hidden_size.to_string().as_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.context_window_tokens.to_string().as_bytes());
    hasher.update(b"|");
    hasher.update(base_binding.tokenizer.stable_digest().as_bytes());
    hex::encode(hasher.finalize())
}

fn stable_digest(prefix: &[u8], payload: &impl Serialize) -> String {
    let encoded = serde_json::to_vec(payload).expect("payload should serialize");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), QwenLegalAdapterSftError> {
    if value.trim().is_empty() {
        return Err(QwenLegalAdapterSftError::InvalidConfig {
            detail: format!("{field} must be present"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_base_binding() -> QwenLegalServedBaseModelBinding {
        let template_digest = "sha256:qwen35-legal-template-smoke";
        QwenLegalServedBaseModelBinding {
            public_model_id: String::from(QWEN35_4B_LEGAL_SMOKE_MODEL_ID),
            served_model_id: String::from(QWEN35_4B_LEGAL_SMOKE_SERVED_MODEL_ID),
            model_family_acceptance_label: String::from(QWEN35_LEGAL_MODEL_FAMILY_ACCEPTANCE_LABEL),
            base_model_revision: String::from("qwen3.5-4b-smoke-revision"),
            base_served_artifact_digest: String::from(
                QWEN_LEGAL_SYNTHETIC_SMOKE_BASE_ARTIFACT_DIGEST,
            ),
            artifact_path: None,
            artifact_mode: QwenLegalBaseArtifactMode::SyntheticHiddenStateSmoke,
            tokenizer: TokenizerDigest::new(
                TokenizerFamily::BytePairEncoding,
                "sha256:qwen35-legal-tokenizer-smoke",
                256,
            )
            .with_template_digest(template_digest),
            prompt_template_digest: String::from(template_digest),
            hidden_size: 4,
            context_window_tokens: 128,
        }
    }

    fn sample_config() -> QwenLegalAdapterSftConfig {
        QwenLegalAdapterSftConfig {
            run_id: String::from("qwen-legal-adapter-run"),
            budget: TrainingLoopBudget::new(4, 1, 1).expect("budget"),
            batch_size: 2,
            optimizer: TrainingOptimizerConfig::adamw(0.12, 0.9, 0.99, 1e-8)
                .with_gradient_clip_norm(1.0),
            optimizer_residency_policy: TrainingOptimizerResidencyPolicy::host_only(),
        }
    }

    fn sample_supervision() -> Vec<QwenLegalLmHeadSupervisionSample> {
        vec![
            QwenLegalLmHeadSupervisionSample::new(
                "legal-a",
                vec![1.0, 0.0, 0.0, 0.0],
                12,
                41,
                "legal-record-a",
            ),
            QwenLegalLmHeadSupervisionSample::new(
                "legal-b",
                vec![0.0, 1.0, 0.0, 0.0],
                35,
                39,
                "legal-record-b",
            ),
            QwenLegalLmHeadSupervisionSample::new(
                "legal-c",
                vec![0.0, 0.0, 1.0, 0.0],
                62,
                47,
                "legal-record-c",
            ),
            QwenLegalLmHeadSupervisionSample::new(
                "legal-d",
                vec![0.0, 0.0, 0.0, 1.0],
                90,
                44,
                "legal-record-d",
            ),
        ]
    }

    fn sample_dataset_binding() -> QwenLegalDatasetBinding {
        QwenLegalDatasetBinding {
            dataset_ref: String::from("dataset://openagents/legal-benchmark/harvey-smoke@v1"),
            dataset_digest: String::from("sha256:legal-training-record-bundle-smoke"),
            training_record_schema_version: String::from(
                LEGAL_BENCHMARK_TRAINING_RECORD_SCHEMA_VERSION,
            ),
            train_split_ref: String::from("split://legal-benchmark/harvey-smoke/train"),
            validation_split_ref: String::from("split://legal-benchmark/harvey-smoke/validation"),
            hidden_criterion_policy_ref: String::from(
                "policy://legal-benchmark/hidden-criteria/exclude-visible@v1",
            ),
        }
    }

    fn sample_eval_pack_binding() -> QwenLegalEvalPackBinding {
        QwenLegalEvalPackBinding {
            eval_pack_id: String::from("legal-benchmark-retained-smoke"),
            eval_pack_digest: String::from("sha256:legal-retained-smoke-eval-pack"),
            benchmark_suite_id: String::from("harvey-legal-benchmark"),
            retained_slice_id: String::from("retained-smoke"),
            scorer_version: String::from("psionic-legal-scorer.v1"),
            import_target: String::from("autopilot4://legal-benchmark/runs"),
        }
    }

    fn sample_request() -> QwenLegalAdapterSftRunRequest {
        QwenLegalAdapterSftRunRequest {
            dataset_binding: sample_dataset_binding(),
            eval_pack_binding: sample_eval_pack_binding(),
            validator_policy_ref: String::from("policy://validator/legal-benchmark/qwen-smoke"),
            adapter_id: String::from("qwen35-4b-legal-smoke"),
            adapter_revision: String::from("r1"),
            started_at_ms: 1_000,
            step_duration_ms: 20,
        }
    }

    fn sample_trainer() -> QwenLegalAdapterSftTrainer {
        QwenLegalAdapterSftTrainer::new(
            sample_config(),
            canonical_qwen_legal_adapter_target_set(),
            sample_base_binding(),
            sample_supervision(),
        )
        .expect("trainer")
    }

    #[test]
    fn qwen_legal_adapter_smoke_exports_artifact_checkpoint_and_import_bundle()
    -> Result<(), Box<dyn std::error::Error>> {
        let trainer = sample_trainer();
        let outcome = trainer.run_sft(&sample_request())?;
        assert_eq!(outcome.step_receipts.len(), 4);
        assert_eq!(
            outcome.summary.run_summary.checkpoint_family,
            QWEN_LEGAL_ADAPTER_CHECKPOINT_FAMILY
        );
        assert_eq!(
            outcome.exported_artifact.adapter_identity.base_model_id,
            QWEN35_4B_LEGAL_SMOKE_MODEL_ID
        );
        assert_eq!(
            trainer.backend().provenance().adapter_family,
            OPEN_ADAPTER_QWEN35_LEGAL_ADAPTER_FAMILY
        );
        assert_eq!(
            outcome.final_checkpoint.dataset_digest,
            sample_dataset_binding().dataset_digest
        );
        assert_eq!(
            outcome.score_import_bundle.import_target,
            "autopilot4://legal-benchmark/runs"
        );
        assert_eq!(
            outcome.score_import_bundle.bundle_digest,
            outcome.score_import_bundle.stable_digest()
        );
        assert_eq!(
            outcome.rl_hillclimb_plan.score_import_bundle_ref,
            outcome.score_import_bundle.bundle_id
        );
        assert_eq!(
            outcome.rl_hillclimb_plan.retained_target_model_id,
            QWEN36_35B_A3B_LEGAL_RETAINED_MODEL_ID
        );
        assert_eq!(
            outcome.rl_hillclimb_plan.plan_digest,
            outcome.rl_hillclimb_plan.stable_digest()
        );
        let loaded = outcome.exported_artifact.load_lm_head_lora_artifact()?;
        assert_eq!(loaded.hidden_size, 4);
        assert_eq!(loaded.rank, QWEN_LEGAL_ADAPTER_LORA_RANK);
        Ok(())
    }

    #[test]
    fn qwen_legal_rl_hillclimb_plan_covers_gepa_mipro_and_grpo_targets()
    -> Result<(), Box<dyn std::error::Error>> {
        let plan = canonical_qwen_legal_rl_hillclimb_plan(
            sample_dataset_binding(),
            sample_eval_pack_binding(),
            "qwen35-4b-legal-smoke-r1-score-import",
        )?;
        let methods = plan
            .targets
            .iter()
            .map(|target| target.optimizer_method)
            .collect::<Vec<_>>();
        let failure_families = plan
            .targets
            .iter()
            .map(|target| target.failure_family.as_str())
            .collect::<Vec<_>>();
        let expected_lift = plan
            .targets
            .iter()
            .map(|target| target.target_lift_basis_points)
            .sum::<u16>();

        assert_eq!(
            plan.schema_version,
            QWEN_LEGAL_RL_HILLCLIMB_PLAN_SCHEMA_VERSION
        );
        assert_eq!(plan.base_lane_id, QWEN_LEGAL_ADAPTER_SFT_LANE_ID);
        assert_eq!(
            plan.smoke_served_model_id,
            QWEN35_4B_LEGAL_SMOKE_SERVED_MODEL_ID
        );
        assert_eq!(
            plan.retained_target_served_model_id,
            QWEN36_35B_A3B_LEGAL_RETAINED_SERVED_MODEL_ID
        );
        assert!(plan.rollout_policy.retained_slice_required);
        assert!(
            plan.rollout_policy.min_accepted_rollouts
                > plan.rollout_policy.max_quarantined_rollouts
        );
        assert!(methods.contains(&QwenLegalRlOptimizerMethod::Grpo));
        assert!(methods.contains(&QwenLegalRlOptimizerMethod::GepaTraceSelection));
        assert!(methods.contains(&QwenLegalRlOptimizerMethod::MiproPromptSearch));
        assert!(methods.contains(&QwenLegalRlOptimizerMethod::SupervisedFineTuneRefresh));
        assert!(expected_lift >= 3_000);
        for family in [
            "document_coverage",
            "citation_evidence",
            "legal_reasoning",
            "spreadsheet_reasoning",
            "missing_fact",
            "pre_submit_self_check",
        ] {
            assert!(failure_families.contains(&family), "missing {family}");
        }
        assert!(
            plan.blueprint_optimizer_frontier_ref
                .contains("harvey_legal_qwen_optimizer_frontier")
        );
        assert_eq!(plan.plan_digest, plan.stable_digest());
        Ok(())
    }

    #[test]
    fn qwen_legal_adapter_refuses_template_and_dataset_schema_drift() {
        let mut base = sample_base_binding();
        base.prompt_template_digest = String::from("sha256:drifted-template");
        let template_error = QwenLegalAdapterSftTrainer::new(
            sample_config(),
            canonical_qwen_legal_adapter_target_set(),
            base,
            sample_supervision(),
        )
        .expect_err("template drift must refuse");
        assert!(template_error.to_string().contains("template digest"));

        let trainer = sample_trainer();
        let mut request = sample_request();
        request.dataset_binding.training_record_schema_version =
            String::from("psionic.legal_benchmark_training_record.v0");
        let dataset_error = trainer
            .run_sft(&request)
            .expect_err("dataset schema drift must refuse");
        assert!(
            dataset_error
                .to_string()
                .contains("legal training record schema")
        );
    }

    #[test]
    fn qwen_legal_adapter_gates_real_artifact_execution() {
        let mut real_binding = sample_base_binding();
        real_binding.artifact_mode = QwenLegalBaseArtifactMode::RealArtifactRequired;
        real_binding.base_served_artifact_digest = String::from("sha256:real-qwen35-4b-artifact");
        real_binding.artifact_path = None;
        let error = QwenLegalAdapterSftTrainer::new(
            sample_config(),
            canonical_qwen_legal_adapter_target_set(),
            real_binding,
            sample_supervision(),
        )
        .expect_err("real artifact mode must require a path");
        assert!(error.to_string().contains("artifact path"));
    }

    #[test]
    fn qwen_legal_adapter_checkpoint_restores_exact_run_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let trainer = sample_trainer();
        let mut run = trainer.initialize_run()?;
        let progress = trainer.advance_run(&mut run, Some(2), 1_000, 20)?;
        assert_eq!(progress.step_receipts.len(), 2);
        let checkpoint = trainer.save_checkpoint(
            "qwen-legal-midpoint",
            &run,
            &sample_dataset_binding(),
            &sample_eval_pack_binding(),
            1_040,
        )?;
        let restored = trainer.restore_run(&checkpoint)?;
        assert_eq!(restored.summary().completed_steps, 2);
        let resumed = trainer.run_sft_from_checkpoint(&checkpoint, &sample_request())?;
        assert_eq!(resumed.summary.run_summary.completed_steps, 4);
        assert_eq!(
            resumed.summary.run_summary.last_receipt_id.as_deref(),
            Some("qwen-legal-adapter-run-step-4")
        );
        Ok(())
    }
}

use std::collections::BTreeMap;

use psionic_adapters::{
    AdapterArtifactFormat, AdapterArtifactIdentity, AdapterArtifactKind, AdapterTargetFamily,
};
use psionic_core::QuantizationMode;
use psionic_models::{
    QWEN38_27B_MODEL_ID, QWEN38_27B_SERVED_MODEL_ID, QWEN38_27B_UPSTREAM_REVISION,
    canonical_qwen38_27b_artifact_facts,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::PsionicTrainCpuBudget;

pub const QWEN38_TRAINING_ADMISSION_SCHEMA_VERSION: &str =
    "psionic.qwen38.training_adapter_admission.v1";
pub const QWEN38_TRAINING_BASE_IDENTITY_SCHEMA_VERSION: &str =
    "psionic.qwen38.training_base_identity.v1";
pub const QWEN38_ADAPTER_IDENTITY_BINDING_SCHEMA_VERSION: &str =
    "psionic.qwen38.adapter_identity_binding.v1";
pub const QWEN38_LM_HEAD_BACKWARD_RECEIPT_SCHEMA_VERSION: &str =
    "psionic.qwen38.lm_head_lora_backward_receipt.v1";
pub const QWEN38_TRAINING_EVIDENCE_SCHEMA_VERSION: &str =
    "psionic.qwen38.training_adapter_evidence.v1";
pub const QWEN38_LM_HEAD_TARGET: &str = "lm_head.weight";
pub const QWEN38_LM_HEAD_BACKWARD_CONTRACT: &str = "qwen38_lm_head_lora_f32_reference_backward_v1";
pub const QWEN38_DEFERRED_ADAPTER_TARGETS: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
    "visual.merger.linear_fc2",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen38TrainingExecutionMode {
    TinyReferenceCpu,
    NativeCpu,
    NativeCuda,
    NativeMetal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen38TrainingAdmissionStatus {
    Admitted,
    Refused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen38TrainingRefusalCode {
    SchemaMismatch,
    UnsupportedModel,
    BaseRevisionMismatch,
    BaseArtifactIdentityMismatch,
    InheritedAdapterIdentity,
    UnsupportedTarget,
    UnsupportedExecutionMode,
    InvalidAdapterConfig,
    MissingLineage,
    InvalidResourceBudget,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen38TrainingRefusal {
    pub code: Qwen38TrainingRefusalCode,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen38TrainingBaseIdentity {
    pub schema_version: String,
    pub model_id: String,
    pub served_model_id: String,
    pub upstream_revision: String,
    pub artifact_facts_sha256: String,
    pub safetensors_index_sha256: String,
    pub base_artifact_identity_digest: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen38AdapterIdentityBinding {
    pub schema_version: String,
    pub adapter_id: String,
    pub adapter_revision: String,
    pub base_model_id: String,
    pub base_model_revision: String,
    pub base_artifact_identity_digest: String,
    pub target_modules: Vec<String>,
    pub binding_digest: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen38TrainingAdapterRequest {
    pub schema_version: String,
    pub request_id: String,
    pub model_id: String,
    pub upstream_revision: String,
    pub base_artifact_identity_digest: String,
    pub adapter_id: String,
    pub adapter_revision: String,
    pub target_modules: Vec<String>,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub execution_mode: Qwen38TrainingExecutionMode,
    pub corpus_manifest_sha256: String,
    pub evaluation_manifest_sha256: String,
    pub seed: u64,
    pub cpu_budget: PsionicTrainCpuBudget,
}

impl Default for Qwen38TrainingAdapterRequest {
    fn default() -> Self {
        Self {
            schema_version: String::from(QWEN38_TRAINING_ADMISSION_SCHEMA_VERSION),
            request_id: String::from("qwen38-lm-head-reference-001"),
            model_id: String::from(QWEN38_27B_MODEL_ID),
            upstream_revision: String::from(QWEN38_27B_UPSTREAM_REVISION),
            base_artifact_identity_digest: qwen38_training_base_identity()
                .base_artifact_identity_digest,
            adapter_id: String::from("qwen38-lm-head-reference-001"),
            adapter_revision: String::from("reference-step-001"),
            target_modules: vec![String::from(QWEN38_LM_HEAD_TARGET)],
            lora_rank: 2,
            lora_alpha: 4.0,
            execution_mode: Qwen38TrainingExecutionMode::TinyReferenceCpu,
            corpus_manifest_sha256: sha256_bytes(b"qwen38-reference-corpus-v1"),
            evaluation_manifest_sha256: sha256_bytes(b"qwen38-reference-eval-v1"),
            seed: 38,
            cpu_budget: PsionicTrainCpuBudget::BoundedSingleCore,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen38TrainingAdapterPlan {
    pub schema_version: String,
    pub request_id: String,
    pub status: Qwen38TrainingAdmissionStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Qwen38TrainingRefusal>,
    pub base_identity: Qwen38TrainingBaseIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_binding: Option<Qwen38AdapterIdentityBinding>,
    pub execution_mode: Qwen38TrainingExecutionMode,
    pub exact_backward_contract: Option<String>,
    pub deferred_target_modules: Vec<String>,
    pub corpus_manifest_sha256: String,
    pub evaluation_manifest_sha256: String,
    pub seed: u64,
    pub cpu_budget: PsionicTrainCpuBudget,
    pub claim_boundary: String,
    pub plan_digest: String,
}

impl Qwen38TrainingAdapterPlan {
    #[must_use]
    pub fn is_admitted(&self) -> bool {
        self.status == Qwen38TrainingAdmissionStatus::Admitted
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen38LmHeadLoraBackwardFixture {
    pub fixture_id: String,
    pub hidden: Vec<f32>,
    pub base_logits: Vec<f32>,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub lora_a: Vec<f32>,
    pub lora_b: Vec<f32>,
    pub target_token_id: usize,
    pub learning_rate: f32,
    pub finite_difference_epsilon: f32,
    pub gradient_tolerance: f32,
}

impl Default for Qwen38LmHeadLoraBackwardFixture {
    fn default() -> Self {
        Self {
            fixture_id: String::from("qwen38-lm-head-lora-tiny-v1"),
            hidden: vec![0.75, -0.5, 0.25, 1.0],
            base_logits: vec![0.2, -0.1, 0.05],
            lora_rank: 2,
            lora_alpha: 4.0,
            lora_a: vec![0.1, -0.2, 0.05, 0.3, -0.15, 0.25, 0.2, -0.1],
            lora_b: vec![0.2, -0.1, -0.15, 0.3, 0.1, 0.25],
            target_token_id: 2,
            learning_rate: 0.05,
            finite_difference_epsilon: 0.001,
            gradient_tolerance: 0.0002,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen38LmHeadLoraBackwardReceipt {
    pub schema_version: String,
    pub fixture_id: String,
    pub contract: String,
    pub hidden_size: usize,
    pub vocabulary_size: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub target_token_id: usize,
    pub learning_rate: f32,
    pub initial_loss: f32,
    pub updated_loss: f32,
    pub loss_improved: bool,
    pub initial_logits_sha256: String,
    pub updated_logits_sha256: String,
    pub lora_a_gradient_sha256: String,
    pub lora_b_gradient_sha256: String,
    pub updated_lora_a_sha256: String,
    pub updated_lora_b_sha256: String,
    pub finite_difference_epsilon: f32,
    pub gradient_max_abs_error: f32,
    pub gradient_tolerance: f32,
    pub gradient_check_passed: bool,
    pub base_weights_frozen: bool,
    pub deterministic_replay: bool,
    pub receipt_digest: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen38TrainingAdapterEvidenceReport {
    pub schema_version: String,
    pub phase: String,
    pub status: String,
    pub base_identity: Qwen38TrainingBaseIdentity,
    pub admitted_plan: Qwen38TrainingAdapterPlan,
    pub backward_receipt: Qwen38LmHeadLoraBackwardReceipt,
    pub refusals: BTreeMap<String, Qwen38TrainingAdapterPlan>,
    pub real_checkpoint_training_admitted: bool,
    pub native_backward_admitted: bool,
    pub adapter_artifact_written: bool,
    pub adapter_serving_admitted: bool,
    pub checkpoint_recovery_admitted: bool,
    pub promotion_admitted: bool,
    pub claim_boundary: String,
    pub report_digest: String,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum Qwen38LmHeadBackwardError {
    #[error("invalid Qwen3.8 LM-head backward fixture: {0}")]
    InvalidFixture(String),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum Qwen38AdapterIdentityError {
    #[error("Qwen3.8 adapter plan was not admitted")]
    PlanNotAdmitted,
    #[error("Qwen3.8 adapter artifact digest must be a lowercase sha256 hex digest")]
    InvalidArtifactDigest,
    #[error("adapter base model `{actual}` does not match Qwen3.8 `{expected}`")]
    BaseModelMismatch { expected: String, actual: String },
    #[error("adapter base revision `{actual}` does not match Qwen3.8 `{expected}`")]
    BaseRevisionMismatch { expected: String, actual: String },
    #[error("adapter base artifact digest `{actual}` does not match Qwen3.8 `{expected}`")]
    BaseArtifactMismatch { expected: String, actual: String },
    #[error("adapter kind or format is not an admitted Qwen3.8 LoRA safetensors artifact")]
    UnsupportedArtifactShape,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum Qwen38TrainingEvidenceError {
    #[error(transparent)]
    Backward(#[from] Qwen38LmHeadBackwardError),
}

#[must_use]
pub fn qwen38_training_base_identity() -> Qwen38TrainingBaseIdentity {
    let facts = canonical_qwen38_27b_artifact_facts();
    let artifact_facts_sha256 = facts.canonical_sha256();
    let mut identity = Qwen38TrainingBaseIdentity {
        schema_version: String::from(QWEN38_TRAINING_BASE_IDENTITY_SCHEMA_VERSION),
        model_id: String::from(QWEN38_27B_MODEL_ID),
        served_model_id: String::from(QWEN38_27B_SERVED_MODEL_ID),
        upstream_revision: String::from(QWEN38_27B_UPSTREAM_REVISION),
        artifact_facts_sha256,
        safetensors_index_sha256: facts.digests.safetensors_index_sha256,
        base_artifact_identity_digest: String::new(),
    };
    identity.base_artifact_identity_digest =
        stable_json_digest(b"qwen38_training_base_identity|", &identity);
    identity
}

#[must_use]
pub fn admit_qwen38_training_adapter(
    request: &Qwen38TrainingAdapterRequest,
) -> Qwen38TrainingAdapterPlan {
    let base_identity = qwen38_training_base_identity();
    if request.schema_version != QWEN38_TRAINING_ADMISSION_SCHEMA_VERSION {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::SchemaMismatch,
            format!(
                "expected schema `{QWEN38_TRAINING_ADMISSION_SCHEMA_VERSION}`, found `{}`",
                request.schema_version
            ),
        );
    }
    if request.model_id != QWEN38_27B_MODEL_ID {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::UnsupportedModel,
            format!(
                "Qwen3.8 training requires exact model id `{QWEN38_27B_MODEL_ID}`; `{}` is not inherited or normalized",
                request.model_id
            ),
        );
    }
    if request.upstream_revision != QWEN38_27B_UPSTREAM_REVISION {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::BaseRevisionMismatch,
            "Qwen3.8 training revision does not match the canonical upstream revision",
        );
    }
    if request.base_artifact_identity_digest != base_identity.base_artifact_identity_digest {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::BaseArtifactIdentityMismatch,
            "Qwen3.8 training base artifact identity digest does not match canonical facts",
        );
    }
    if names_inherited_qwen_adapter(request.adapter_id.as_str()) {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::InheritedAdapterIdentity,
            "Qwen3.5/Qwen3.6 adapter ids cannot be relabeled as Qwen3.8 adapters",
        );
    }
    if request.request_id.trim().is_empty()
        || request.adapter_id.trim().is_empty()
        || request.adapter_revision.trim().is_empty()
        || request.lora_rank == 0
        || request.lora_rank > 64
        || !request.lora_alpha.is_finite()
        || request.lora_alpha <= 0.0
    {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::InvalidAdapterConfig,
            "request/adapter ids, revision, rank in 1..=64, and positive finite alpha are required",
        );
    }
    if request.target_modules.as_slice() != [QWEN38_LM_HEAD_TARGET] {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::UnsupportedTarget,
            format!(
                "only `{QWEN38_LM_HEAD_TARGET}` has an exact Qwen3.8 reference backward contract; requested {:?}",
                request.target_modules
            ),
        );
    }
    if request.execution_mode != Qwen38TrainingExecutionMode::TinyReferenceCpu {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::UnsupportedExecutionMode,
            "native CPU, CUDA, and Metal Qwen3.8 backward execution is not implemented",
        );
    }
    if !is_sha256(&request.corpus_manifest_sha256)
        || !is_sha256(&request.evaluation_manifest_sha256)
    {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::MissingLineage,
            "corpus and evaluation manifests require lowercase sha256 digests",
        );
    }
    if let Err(error) = request.cpu_budget.validate() {
        return refused_plan(
            request,
            base_identity,
            Qwen38TrainingRefusalCode::InvalidResourceBudget,
            error.to_string(),
        );
    }

    let binding = adapter_binding(request, &base_identity);
    finalize_plan(Qwen38TrainingAdapterPlan {
        schema_version: String::from(QWEN38_TRAINING_ADMISSION_SCHEMA_VERSION),
        request_id: request.request_id.clone(),
        status: Qwen38TrainingAdmissionStatus::Admitted,
        refusal: None,
        base_identity,
        adapter_binding: Some(binding),
        execution_mode: request.execution_mode,
        exact_backward_contract: Some(String::from(QWEN38_LM_HEAD_BACKWARD_CONTRACT)),
        deferred_target_modules: deferred_targets(),
        corpus_manifest_sha256: request.corpus_manifest_sha256.clone(),
        evaluation_manifest_sha256: request.evaluation_manifest_sha256.clone(),
        seed: request.seed,
        cpu_budget: request.cpu_budget,
        claim_boundary: String::from(
            "This admits deterministic tiny-reference CPU math for an F32 Qwen3.8 LM-head LoRA adapter. It does not admit real-checkpoint training, native CPU/CUDA/Metal backward execution, decoder-layer or vision targets, adapter serving, checkpoint recovery, evaluation gains, or promotion.",
        ),
        plan_digest: String::new(),
    })
}

pub fn finalize_qwen38_lm_head_adapter_identity(
    plan: &Qwen38TrainingAdapterPlan,
    artifact_digest: &str,
    parameter_count: u64,
) -> Result<AdapterArtifactIdentity, Qwen38AdapterIdentityError> {
    if !plan.is_admitted() {
        return Err(Qwen38AdapterIdentityError::PlanNotAdmitted);
    }
    if !is_sha256(artifact_digest) {
        return Err(Qwen38AdapterIdentityError::InvalidArtifactDigest);
    }
    let binding = plan
        .adapter_binding
        .as_ref()
        .ok_or(Qwen38AdapterIdentityError::PlanNotAdmitted)?;
    Ok(AdapterArtifactIdentity::new(
        binding.adapter_id.clone(),
        binding.adapter_revision.clone(),
        AdapterArtifactKind::Lora,
        AdapterArtifactFormat::Safetensors,
        binding.base_model_id.clone(),
        binding.base_model_revision.clone(),
        binding.base_artifact_identity_digest.clone(),
        String::from(artifact_digest),
        QuantizationMode::None,
        AdapterTargetFamily::DecoderComposite,
        parameter_count,
    )
    .with_provenance_digest(plan.plan_digest.clone()))
}

pub fn validate_qwen38_adapter_identity(
    adapter: &AdapterArtifactIdentity,
) -> Result<(), Qwen38AdapterIdentityError> {
    let base = qwen38_training_base_identity();
    if adapter.base_model_id != QWEN38_27B_MODEL_ID {
        return Err(Qwen38AdapterIdentityError::BaseModelMismatch {
            expected: String::from(QWEN38_27B_MODEL_ID),
            actual: adapter.base_model_id.clone(),
        });
    }
    if adapter.base_model_revision != QWEN38_27B_UPSTREAM_REVISION {
        return Err(Qwen38AdapterIdentityError::BaseRevisionMismatch {
            expected: String::from(QWEN38_27B_UPSTREAM_REVISION),
            actual: adapter.base_model_revision.clone(),
        });
    }
    if adapter.base_served_artifact_digest != base.base_artifact_identity_digest {
        return Err(Qwen38AdapterIdentityError::BaseArtifactMismatch {
            expected: base.base_artifact_identity_digest,
            actual: adapter.base_served_artifact_digest.clone(),
        });
    }
    if adapter.kind != AdapterArtifactKind::Lora
        || adapter.format != AdapterArtifactFormat::Safetensors
        || adapter.quantization != QuantizationMode::None
        || adapter.target_family != AdapterTargetFamily::DecoderComposite
    {
        return Err(Qwen38AdapterIdentityError::UnsupportedArtifactShape);
    }
    if !is_sha256(&adapter.artifact_digest) {
        return Err(Qwen38AdapterIdentityError::InvalidArtifactDigest);
    }
    Ok(())
}

pub fn run_qwen38_lm_head_lora_backward_reference(
    fixture: &Qwen38LmHeadLoraBackwardFixture,
) -> Result<Qwen38LmHeadLoraBackwardReceipt, Qwen38LmHeadBackwardError> {
    validate_backward_fixture(fixture)?;
    let (initial_logits, initial_loss) = loss_and_logits(
        fixture,
        fixture.lora_a.as_slice(),
        fixture.lora_b.as_slice(),
    );
    let (gradient_a, gradient_b) = analytic_gradients(fixture, &initial_logits);
    let finite_difference_a = finite_difference_gradient(fixture, true);
    let finite_difference_b = finite_difference_gradient(fixture, false);
    let gradient_max_abs_error = gradient_a
        .iter()
        .zip(finite_difference_a.iter())
        .chain(gradient_b.iter().zip(finite_difference_b.iter()))
        .map(|(analytic, numeric)| (analytic - numeric).abs())
        .fold(0.0_f32, f32::max);
    let updated_a = fixture
        .lora_a
        .iter()
        .zip(gradient_a.iter())
        .map(|(weight, gradient)| weight - fixture.learning_rate * gradient)
        .collect::<Vec<_>>();
    let updated_b = fixture
        .lora_b
        .iter()
        .zip(gradient_b.iter())
        .map(|(weight, gradient)| weight - fixture.learning_rate * gradient)
        .collect::<Vec<_>>();
    let (updated_logits, updated_loss) =
        loss_and_logits(fixture, updated_a.as_slice(), updated_b.as_slice());
    let replay = loss_and_logits(fixture, updated_a.as_slice(), updated_b.as_slice());
    let mut receipt = Qwen38LmHeadLoraBackwardReceipt {
        schema_version: String::from(QWEN38_LM_HEAD_BACKWARD_RECEIPT_SCHEMA_VERSION),
        fixture_id: fixture.fixture_id.clone(),
        contract: String::from(QWEN38_LM_HEAD_BACKWARD_CONTRACT),
        hidden_size: fixture.hidden.len(),
        vocabulary_size: fixture.base_logits.len(),
        lora_rank: fixture.lora_rank,
        lora_alpha: fixture.lora_alpha,
        target_token_id: fixture.target_token_id,
        learning_rate: fixture.learning_rate,
        initial_loss,
        updated_loss,
        loss_improved: updated_loss < initial_loss,
        initial_logits_sha256: sha256_f32(&initial_logits),
        updated_logits_sha256: sha256_f32(&updated_logits),
        lora_a_gradient_sha256: sha256_f32(&gradient_a),
        lora_b_gradient_sha256: sha256_f32(&gradient_b),
        updated_lora_a_sha256: sha256_f32(&updated_a),
        updated_lora_b_sha256: sha256_f32(&updated_b),
        finite_difference_epsilon: fixture.finite_difference_epsilon,
        gradient_max_abs_error,
        gradient_tolerance: fixture.gradient_tolerance,
        gradient_check_passed: gradient_max_abs_error <= fixture.gradient_tolerance,
        base_weights_frozen: true,
        deterministic_replay: replay.0 == updated_logits && replay.1 == updated_loss,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = stable_json_digest(b"qwen38_lm_head_backward_receipt|", &receipt);
    Ok(receipt)
}

pub fn qwen38_training_adapter_evidence_report()
-> Result<Qwen38TrainingAdapterEvidenceReport, Qwen38TrainingEvidenceError> {
    let request = Qwen38TrainingAdapterRequest::default();
    let admitted_plan = admit_qwen38_training_adapter(&request);
    let backward_receipt =
        run_qwen38_lm_head_lora_backward_reference(&Qwen38LmHeadLoraBackwardFixture::default())?;
    let refusals = [
        (
            "inherited_model",
            Qwen38TrainingAdapterRequest {
                model_id: String::from("Qwen/Qwen3.6-27B"),
                ..request.clone()
            },
        ),
        (
            "inherited_adapter",
            Qwen38TrainingAdapterRequest {
                adapter_id: String::from("qwen36-legal-lora-001"),
                ..request.clone()
            },
        ),
        (
            "decoder_target",
            Qwen38TrainingAdapterRequest {
                target_modules: vec![String::from("q_proj")],
                ..request.clone()
            },
        ),
        (
            "native_cuda",
            Qwen38TrainingAdapterRequest {
                execution_mode: Qwen38TrainingExecutionMode::NativeCuda,
                ..request.clone()
            },
        ),
        (
            "artifact_drift",
            Qwen38TrainingAdapterRequest {
                base_artifact_identity_digest: sha256_bytes(b"different-base"),
                ..request.clone()
            },
        ),
        (
            "missing_lineage",
            Qwen38TrainingAdapterRequest {
                corpus_manifest_sha256: String::from("missing"),
                ..request
            },
        ),
    ]
    .into_iter()
    .map(|(name, request)| (String::from(name), admit_qwen38_training_adapter(&request)))
    .collect();
    let mut report = Qwen38TrainingAdapterEvidenceReport {
        schema_version: String::from(QWEN38_TRAINING_EVIDENCE_SCHEMA_VERSION),
        phase: String::from("R12"),
        status: String::from("implemented_early"),
        base_identity: qwen38_training_base_identity(),
        admitted_plan,
        backward_receipt,
        refusals,
        real_checkpoint_training_admitted: false,
        native_backward_admitted: false,
        adapter_artifact_written: false,
        adapter_serving_admitted: false,
        checkpoint_recovery_admitted: false,
        promotion_admitted: false,
        claim_boundary: String::from(
            "This report proves Qwen3.8-specific base/adapter admission and deterministic tiny-reference F32 LM-head LoRA gradients. It retains no trained adapter and makes no real-checkpoint training, native backward, serving, recovery, evaluation, or promotion claim.",
        ),
        report_digest: String::new(),
    };
    report.report_digest = stable_json_digest(b"qwen38_training_adapter_evidence|", &report);
    Ok(report)
}

fn refused_plan(
    request: &Qwen38TrainingAdapterRequest,
    base_identity: Qwen38TrainingBaseIdentity,
    code: Qwen38TrainingRefusalCode,
    detail: impl Into<String>,
) -> Qwen38TrainingAdapterPlan {
    finalize_plan(Qwen38TrainingAdapterPlan {
        schema_version: String::from(QWEN38_TRAINING_ADMISSION_SCHEMA_VERSION),
        request_id: request.request_id.clone(),
        status: Qwen38TrainingAdmissionStatus::Refused,
        refusal: Some(Qwen38TrainingRefusal {
            code,
            detail: detail.into(),
        }),
        base_identity,
        adapter_binding: None,
        execution_mode: request.execution_mode,
        exact_backward_contract: None,
        deferred_target_modules: deferred_targets(),
        corpus_manifest_sha256: request.corpus_manifest_sha256.clone(),
        evaluation_manifest_sha256: request.evaluation_manifest_sha256.clone(),
        seed: request.seed,
        cpu_budget: request.cpu_budget,
        claim_boundary: String::from(
            "The request was refused before training. No adapter, checkpoint, evaluation, serving binding, or promotion claim was produced.",
        ),
        plan_digest: String::new(),
    })
}

fn finalize_plan(mut plan: Qwen38TrainingAdapterPlan) -> Qwen38TrainingAdapterPlan {
    plan.plan_digest = stable_json_digest(b"qwen38_training_adapter_plan|", &plan);
    plan
}

fn adapter_binding(
    request: &Qwen38TrainingAdapterRequest,
    base: &Qwen38TrainingBaseIdentity,
) -> Qwen38AdapterIdentityBinding {
    let mut binding = Qwen38AdapterIdentityBinding {
        schema_version: String::from(QWEN38_ADAPTER_IDENTITY_BINDING_SCHEMA_VERSION),
        adapter_id: request.adapter_id.clone(),
        adapter_revision: request.adapter_revision.clone(),
        base_model_id: base.model_id.clone(),
        base_model_revision: base.upstream_revision.clone(),
        base_artifact_identity_digest: base.base_artifact_identity_digest.clone(),
        target_modules: request.target_modules.clone(),
        binding_digest: String::new(),
    };
    binding.binding_digest = stable_json_digest(b"qwen38_adapter_identity_binding|", &binding);
    binding
}

fn deferred_targets() -> Vec<String> {
    QWEN38_DEFERRED_ADAPTER_TARGETS
        .iter()
        .map(|target| String::from(*target))
        .collect()
}

fn names_inherited_qwen_adapter(adapter_id: &str) -> bool {
    let normalized = adapter_id.to_ascii_lowercase();
    ["qwen3.5", "qwen35", "qwen3.6", "qwen36"]
        .iter()
        .any(|marker| normalized.contains(marker))
}

fn validate_backward_fixture(
    fixture: &Qwen38LmHeadLoraBackwardFixture,
) -> Result<(), Qwen38LmHeadBackwardError> {
    let hidden_size = fixture.hidden.len();
    let vocabulary_size = fixture.base_logits.len();
    if fixture.fixture_id.trim().is_empty()
        || hidden_size == 0
        || vocabulary_size < 2
        || fixture.lora_rank == 0
        || fixture.lora_a.len() != fixture.lora_rank.saturating_mul(hidden_size)
        || fixture.lora_b.len() != vocabulary_size.saturating_mul(fixture.lora_rank)
        || fixture.target_token_id >= vocabulary_size
    {
        return Err(Qwen38LmHeadBackwardError::InvalidFixture(String::from(
            "ids, dimensions, rank, matrix shapes, and target token must be valid",
        )));
    }
    if !fixture.lora_alpha.is_finite()
        || fixture.lora_alpha <= 0.0
        || !fixture.learning_rate.is_finite()
        || fixture.learning_rate <= 0.0
        || !fixture.finite_difference_epsilon.is_finite()
        || fixture.finite_difference_epsilon <= 0.0
        || !fixture.gradient_tolerance.is_finite()
        || fixture.gradient_tolerance <= 0.0
        || fixture
            .hidden
            .iter()
            .chain(fixture.base_logits.iter())
            .chain(fixture.lora_a.iter())
            .chain(fixture.lora_b.iter())
            .any(|value| !value.is_finite())
    {
        return Err(Qwen38LmHeadBackwardError::InvalidFixture(String::from(
            "all values and positive hyperparameters must be finite",
        )));
    }
    Ok(())
}

fn loss_and_logits(
    fixture: &Qwen38LmHeadLoraBackwardFixture,
    lora_a: &[f32],
    lora_b: &[f32],
) -> (Vec<f32>, f32) {
    let hidden_size = fixture.hidden.len();
    let rank = fixture.lora_rank;
    let scale = fixture.lora_alpha / rank as f32;
    let intermediate = lora_a
        .chunks_exact(hidden_size)
        .map(|row| dot(row, fixture.hidden.as_slice()))
        .collect::<Vec<_>>();
    let logits = fixture
        .base_logits
        .iter()
        .zip(lora_b.chunks_exact(rank))
        .map(|(base, row)| base + scale * dot(row, intermediate.as_slice()))
        .collect::<Vec<_>>();
    let probabilities = softmax(logits.as_slice());
    (logits, -probabilities[fixture.target_token_id].ln())
}

fn analytic_gradients(
    fixture: &Qwen38LmHeadLoraBackwardFixture,
    logits: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let hidden_size = fixture.hidden.len();
    let rank = fixture.lora_rank;
    let scale = fixture.lora_alpha / rank as f32;
    let intermediate = fixture
        .lora_a
        .chunks_exact(hidden_size)
        .map(|row| dot(row, fixture.hidden.as_slice()))
        .collect::<Vec<_>>();
    let mut d_logits = softmax(logits);
    d_logits[fixture.target_token_id] -= 1.0;
    let mut gradient_b = vec![0.0; fixture.lora_b.len()];
    for (vocabulary_index, d_logit) in d_logits.iter().copied().enumerate() {
        for rank_index in 0..rank {
            gradient_b[vocabulary_index * rank + rank_index] =
                scale * d_logit * intermediate[rank_index];
        }
    }
    let mut d_intermediate = vec![0.0; rank];
    for rank_index in 0..rank {
        d_intermediate[rank_index] = scale
            * d_logits
                .iter()
                .enumerate()
                .map(|(vocabulary_index, d_logit)| {
                    fixture.lora_b[vocabulary_index * rank + rank_index] * d_logit
                })
                .sum::<f32>();
    }
    let mut gradient_a = vec![0.0; fixture.lora_a.len()];
    for rank_index in 0..rank {
        for hidden_index in 0..hidden_size {
            gradient_a[rank_index * hidden_size + hidden_index] =
                d_intermediate[rank_index] * fixture.hidden[hidden_index];
        }
    }
    (gradient_a, gradient_b)
}

fn finite_difference_gradient(
    fixture: &Qwen38LmHeadLoraBackwardFixture,
    for_lora_a: bool,
) -> Vec<f32> {
    let weights = if for_lora_a {
        fixture.lora_a.as_slice()
    } else {
        fixture.lora_b.as_slice()
    };
    (0..weights.len())
        .map(|index| {
            let mut positive_a = fixture.lora_a.clone();
            let mut negative_a = fixture.lora_a.clone();
            let mut positive_b = fixture.lora_b.clone();
            let mut negative_b = fixture.lora_b.clone();
            if for_lora_a {
                positive_a[index] += fixture.finite_difference_epsilon;
                negative_a[index] -= fixture.finite_difference_epsilon;
            } else {
                positive_b[index] += fixture.finite_difference_epsilon;
                negative_b[index] -= fixture.finite_difference_epsilon;
            }
            let positive = loss_and_logits(fixture, positive_a.as_slice(), positive_b.as_slice()).1;
            let negative = loss_and_logits(fixture, negative_a.as_slice(), negative_b.as_slice()).1;
            (positive - negative) / (2.0 * fixture.finite_difference_epsilon)
        })
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|logit| (logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f32>();
    for value in &mut values {
        *value /= total;
    }
    values
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn stable_json_digest(prefix: &[u8], value: &impl Serialize) -> String {
    let encoded = serde_json::to_vec(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

fn sha256_f32(values: &[f32]) -> String {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    sha256_bytes(bytes.as_slice())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_training_admits_only_exact_tiny_lm_head_contract() {
        let request = Qwen38TrainingAdapterRequest::default();
        let first = admit_qwen38_training_adapter(&request);
        let second = admit_qwen38_training_adapter(&request);

        assert!(first.is_admitted());
        assert_eq!(first, second);
        assert_eq!(
            first.exact_backward_contract.as_deref(),
            Some(QWEN38_LM_HEAD_BACKWARD_CONTRACT)
        );
        assert_eq!(first.cpu_budget.effective_cores(), 1);
        assert_eq!(first.cpu_budget.max_parallel_workers(), 1);
        assert!(!first.plan_digest.is_empty());
        assert_eq!(
            first
                .adapter_binding
                .as_ref()
                .expect("admitted binding")
                .base_model_id,
            QWEN38_27B_MODEL_ID
        );
    }

    #[test]
    fn qwen38_training_refuses_inherited_models_adapters_and_targets() {
        let inherited_model = Qwen38TrainingAdapterRequest {
            model_id: String::from("Qwen/Qwen3.6-27B"),
            ..Qwen38TrainingAdapterRequest::default()
        };
        let inherited_adapter = Qwen38TrainingAdapterRequest {
            adapter_id: String::from("qwen36-legal-lora-001"),
            ..Qwen38TrainingAdapterRequest::default()
        };
        let unsupported_target = Qwen38TrainingAdapterRequest {
            target_modules: vec![String::from("q_proj")],
            ..Qwen38TrainingAdapterRequest::default()
        };

        assert_eq!(
            admit_qwen38_training_adapter(&inherited_model)
                .refusal
                .expect("model refusal")
                .code,
            Qwen38TrainingRefusalCode::UnsupportedModel
        );
        assert_eq!(
            admit_qwen38_training_adapter(&inherited_adapter)
                .refusal
                .expect("adapter refusal")
                .code,
            Qwen38TrainingRefusalCode::InheritedAdapterIdentity
        );
        assert_eq!(
            admit_qwen38_training_adapter(&unsupported_target)
                .refusal
                .expect("target refusal")
                .code,
            Qwen38TrainingRefusalCode::UnsupportedTarget
        );
    }

    #[test]
    fn qwen38_training_refuses_identity_lineage_backend_and_budget_drift() {
        let identity_drift = Qwen38TrainingAdapterRequest {
            base_artifact_identity_digest: sha256_bytes(b"different-base"),
            ..Qwen38TrainingAdapterRequest::default()
        };
        let missing_lineage = Qwen38TrainingAdapterRequest {
            corpus_manifest_sha256: String::from("missing"),
            ..Qwen38TrainingAdapterRequest::default()
        };
        let native_cuda = Qwen38TrainingAdapterRequest {
            execution_mode: Qwen38TrainingExecutionMode::NativeCuda,
            ..Qwen38TrainingAdapterRequest::default()
        };
        let invalid_budget = Qwen38TrainingAdapterRequest {
            cpu_budget: PsionicTrainCpuBudget::ExplicitCores {
                cores: 4,
                opted_in: false,
            },
            ..Qwen38TrainingAdapterRequest::default()
        };

        assert_eq!(
            admit_qwen38_training_adapter(&identity_drift)
                .refusal
                .expect("identity refusal")
                .code,
            Qwen38TrainingRefusalCode::BaseArtifactIdentityMismatch
        );
        assert_eq!(
            admit_qwen38_training_adapter(&missing_lineage)
                .refusal
                .expect("lineage refusal")
                .code,
            Qwen38TrainingRefusalCode::MissingLineage
        );
        assert_eq!(
            admit_qwen38_training_adapter(&native_cuda)
                .refusal
                .expect("backend refusal")
                .code,
            Qwen38TrainingRefusalCode::UnsupportedExecutionMode
        );
        assert_eq!(
            admit_qwen38_training_adapter(&invalid_budget)
                .refusal
                .expect("budget refusal")
                .code,
            Qwen38TrainingRefusalCode::InvalidResourceBudget
        );
    }

    #[test]
    fn qwen38_lm_head_backward_matches_finite_difference_and_improves_loss() {
        let receipt =
            run_qwen38_lm_head_lora_backward_reference(&Qwen38LmHeadLoraBackwardFixture::default())
                .expect("reference backward");

        assert!(receipt.gradient_check_passed);
        assert!(receipt.loss_improved);
        assert!(receipt.deterministic_replay);
        assert!(receipt.base_weights_frozen);
        assert!(receipt.gradient_max_abs_error <= receipt.gradient_tolerance);
        assert_ne!(receipt.initial_logits_sha256, receipt.updated_logits_sha256);
    }

    #[test]
    fn qwen38_finalized_adapter_identity_cannot_bind_to_qwen36() {
        let plan = admit_qwen38_training_adapter(&Qwen38TrainingAdapterRequest::default());
        let artifact_digest = sha256_bytes(b"qwen38-adapter");
        let adapter = finalize_qwen38_lm_head_adapter_identity(&plan, artifact_digest.as_str(), 16)
            .expect("adapter identity");
        validate_qwen38_adapter_identity(&adapter).expect("Qwen3.8 identity");

        let mut inherited = adapter;
        inherited.base_model_id = String::from("Qwen/Qwen3.6-27B");
        assert!(matches!(
            validate_qwen38_adapter_identity(&inherited),
            Err(Qwen38AdapterIdentityError::BaseModelMismatch { .. })
        ));
    }
}

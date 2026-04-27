use std::{collections::BTreeMap, fs, path::Path};

use psionic_core::TensorData;
use psionic_models::{Cs336A1ReferenceConfig, Cs336A1TransformerLm};
use psionic_nn::{
    cross_entropy_loss, LayerError, LossReduction, ModuleStateDict, ModuleStateEntry,
    ModuleStateEntryKind, ModuleStateLoadMode, ModuleStateView, NnTensor, NnTrainingError,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    canonical_a1_minimal_distributed_lm_lane_contract,
    canonical_a1_minimal_distributed_lm_tokenizer_dataset_bundle, cs336_a1_get_batch,
    cs336_a1_get_lr_cosine_schedule, cs336_a1_gradient_clipping,
    initialize_cs336_a1_reference_model, A1MinimalDistributedLmLaneContract,
    A1MinimalDistributedLmTokenizerDatasetBundle, Cs336A1GradientClipReport, Cs336A1ReferenceBatch,
    Cs336A1ReferenceTrainingError, TrainingOptimizerConfig, TrainingOptimizerError,
    TrainingOptimizerState, A1_MINIMAL_DISTRIBUTED_LM_LANE_ID,
};

pub const A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_REPORT_SCHEMA_VERSION: &str =
    "psion.a1_minimal_distributed_lm.local_update_report.v1";
pub const A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_SCHEMA_VERSION: &str =
    "psion.a1_minimal_distributed_lm.local_update_checkpoint.v1";
pub const A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_REPORT_FIXTURE_PATH: &str =
    "fixtures/psion/a1_minimal_distributed_lm/local_update_report_v1.json";
pub const A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_STEP2_FIXTURE_PATH: &str =
    "fixtures/psion/a1_minimal_distributed_lm/local_update_checkpoint_step2_v1.json";
pub const A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_STEP4_FIXTURE_PATH: &str =
    "fixtures/psion/a1_minimal_distributed_lm/local_update_checkpoint_step4_v1.json";
pub const A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_RUN_ID: &str =
    "a1_minimal_distributed_lm_001.local_update_fixture";
pub const A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_ASSIGNMENT_ID: &str =
    "a1_minimal_distributed_lm_001.local_update.assignment.fixture";
pub const A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_WORKER_ID: &str =
    "psionic.local_update.fixture.worker";

#[derive(Debug, Error)]
pub enum A1MinimalDistributedLmLocalUpdateError {
    #[error("invalid A1 minimal distributed LM local update: {detail}")]
    Invalid { detail: String },
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error(transparent)]
    Serialize(#[from] serde_json::Error),
    #[error(transparent)]
    Model(#[from] psionic_models::Cs336A1ReferenceError),
    #[error(transparent)]
    Nn(#[from] NnTrainingError),
    #[error(transparent)]
    Layer(#[from] LayerError),
    #[error(transparent)]
    ModuleState(#[from] psionic_nn::ModuleStateError),
    #[error(transparent)]
    ModuleStateLoad(#[from] psionic_nn::ModuleStateLoadError),
    #[error(transparent)]
    ReferenceTraining(#[from] Cs336A1ReferenceTrainingError),
    #[error(transparent)]
    Optimizer(#[from] TrainingOptimizerError),
    #[error(transparent)]
    Bundle(#[from] crate::A1MinimalDistributedLmTokenizerDatasetBundleError),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct A1MinimalDistributedLmLocalUpdateConfig {
    pub run_id: String,
    pub assignment_id: String,
    pub worker_id: String,
    pub local_step_count: u64,
    pub checkpoint_after_steps: u64,
}

impl Default for A1MinimalDistributedLmLocalUpdateConfig {
    fn default() -> Self {
        Self {
            run_id: String::from(A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_RUN_ID),
            assignment_id: String::from(A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_ASSIGNMENT_ID),
            worker_id: String::from(A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_WORKER_ID),
            local_step_count: 4,
            checkpoint_after_steps: 2,
        }
    }
}

impl A1MinimalDistributedLmLocalUpdateConfig {
    fn validate(&self) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
        ensure_nonempty(self.run_id.as_str(), "local_update_config.run_id")?;
        ensure_nonempty(
            self.assignment_id.as_str(),
            "local_update_config.assignment_id",
        )?;
        ensure_nonempty(self.worker_id.as_str(), "local_update_config.worker_id")?;
        if self.local_step_count == 0 {
            return invalid_local_update(String::from("local_step_count must be nonzero"));
        }
        if self.checkpoint_after_steps == 0 || self.checkpoint_after_steps >= self.local_step_count
        {
            return invalid_local_update(String::from(
                "checkpoint_after_steps must be inside the local update window",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct A1MinimalDistributedLmLocalUpdateStepReport {
    pub step_number: u64,
    pub deterministic_cursor_before: u64,
    pub deterministic_cursor_after: u64,
    pub train_batch_start_positions: Vec<usize>,
    pub consumed_token_count: u64,
    pub learning_rate: f32,
    pub loss_before: f32,
    pub loss_after: f32,
    pub gradient_clip: Cs336A1GradientClipReport,
    pub model_state_digest_before: String,
    pub model_state_digest_after: String,
    pub optimizer_state_digest_after: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct A1MinimalDistributedLmLocalUpdateCheckpointReceipt {
    pub checkpoint_path: String,
    pub checkpoint_digest: String,
    pub optimizer_step: u64,
    pub deterministic_cursor: u64,
    pub model_state_digest: String,
    pub optimizer_state_digest: String,
    pub checkpoint_byte_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct A1MinimalDistributedLmLocalUpdateCheckpoint {
    pub schema_version: String,
    pub lane_id: String,
    pub run_id: String,
    pub assignment_id: String,
    pub worker_id: String,
    pub tokenizer_digest: String,
    pub tokenized_dataset_digest: String,
    pub validation_set_digest: String,
    pub base_checkpoint_ref: String,
    pub optimizer_step: u64,
    pub deterministic_cursor: u64,
    pub consumed_token_count: u64,
    pub model_state: ModuleStateDict,
    pub optimizer_states: BTreeMap<String, TrainingOptimizerState>,
    pub loss_history: Vec<f32>,
    pub checkpoint_digest: String,
}

impl A1MinimalDistributedLmLocalUpdateCheckpoint {
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.checkpoint_digest.clear();
        sha256_uri_digest(
            b"psion_a1_minimal_distributed_lm_local_update_checkpoint|",
            &clone,
        )
    }

    pub fn validate(&self) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
        if self.schema_version != A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_SCHEMA_VERSION {
            return invalid_local_update(format!(
                "checkpoint schema_version must stay `{}`",
                A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_SCHEMA_VERSION
            ));
        }
        validate_lane_identity(self.lane_id.as_str())?;
        ensure_nonempty(self.run_id.as_str(), "checkpoint.run_id")?;
        ensure_nonempty(self.assignment_id.as_str(), "checkpoint.assignment_id")?;
        ensure_nonempty(self.worker_id.as_str(), "checkpoint.worker_id")?;
        ensure_sha256_uri(
            self.tokenizer_digest.as_str(),
            "checkpoint.tokenizer_digest",
        )?;
        ensure_sha256_uri(
            self.tokenized_dataset_digest.as_str(),
            "checkpoint.tokenized_dataset_digest",
        )?;
        ensure_sha256_uri(
            self.validation_set_digest.as_str(),
            "checkpoint.validation_set_digest",
        )?;
        ensure_nonempty(
            self.base_checkpoint_ref.as_str(),
            "checkpoint.base_checkpoint_ref",
        )?;
        if self.optimizer_step == 0 || self.deterministic_cursor == 0 {
            return invalid_local_update(String::from(
                "checkpoint optimizer_step and deterministic_cursor must be nonzero",
            ));
        }
        if self.consumed_token_count == 0 {
            return invalid_local_update(String::from(
                "checkpoint consumed_token_count must be nonzero",
            ));
        }
        if self.loss_history.is_empty() || self.loss_history.iter().any(|value| !value.is_finite())
        {
            return invalid_local_update(String::from(
                "checkpoint loss history must contain finite values",
            ));
        }
        if self.checkpoint_digest != self.stable_digest() {
            return invalid_local_update(String::from("checkpoint digest drifted"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct A1MinimalDistributedLmLocalUpdateReport {
    pub schema_version: String,
    pub lane_id: String,
    pub run_id: String,
    pub assignment_id: String,
    pub worker_id: String,
    pub tokenizer_digest: String,
    pub tokenized_dataset_digest: String,
    pub validation_set_digest: String,
    pub base_checkpoint_ref: String,
    pub backward_path_kind: String,
    pub finite_difference_used: bool,
    pub trained_parameter_paths: Vec<String>,
    pub local_step_count: u64,
    pub consumed_token_count: u64,
    pub deterministic_cursor_before: u64,
    pub deterministic_cursor_after: u64,
    pub loss_before: f32,
    pub loss_after: f32,
    pub validation_loss_before: f32,
    pub validation_loss_after: f32,
    pub steps: Vec<A1MinimalDistributedLmLocalUpdateStepReport>,
    pub checkpoint_step2: A1MinimalDistributedLmLocalUpdateCheckpointReceipt,
    pub checkpoint_step4: A1MinimalDistributedLmLocalUpdateCheckpointReceipt,
    pub resumed_final_model_state_digest: String,
    pub uninterrupted_final_model_state_digest: String,
    pub resumed_final_optimizer_state_digest: String,
    pub uninterrupted_final_optimizer_state_digest: String,
    pub resume_matches_uninterrupted: bool,
    pub delta_digest: String,
    pub report_digest: String,
    pub claim_boundary: String,
}

impl A1MinimalDistributedLmLocalUpdateReport {
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.report_digest.clear();
        sha256_uri_digest(
            b"psion_a1_minimal_distributed_lm_local_update_report|",
            &clone,
        )
    }

    pub fn validate(&self) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
        if self.schema_version != A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_REPORT_SCHEMA_VERSION {
            return invalid_local_update(format!(
                "report schema_version must stay `{}`",
                A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_REPORT_SCHEMA_VERSION
            ));
        }
        validate_lane_identity(self.lane_id.as_str())?;
        ensure_nonempty(self.run_id.as_str(), "report.run_id")?;
        ensure_nonempty(self.assignment_id.as_str(), "report.assignment_id")?;
        ensure_nonempty(self.worker_id.as_str(), "report.worker_id")?;
        ensure_sha256_uri(self.tokenizer_digest.as_str(), "report.tokenizer_digest")?;
        ensure_sha256_uri(
            self.tokenized_dataset_digest.as_str(),
            "report.tokenized_dataset_digest",
        )?;
        ensure_sha256_uri(
            self.validation_set_digest.as_str(),
            "report.validation_set_digest",
        )?;
        ensure_nonempty(
            self.base_checkpoint_ref.as_str(),
            "report.base_checkpoint_ref",
        )?;
        if self.backward_path_kind != "analytic_lm_head_cross_entropy_backward_v1" {
            return invalid_local_update(String::from(
                "report must name the analytic LM-head backward path",
            ));
        }
        if self.finite_difference_used {
            return invalid_local_update(String::from(
                "A1 minimal distributed LM local update must not use finite differences",
            ));
        }
        if self.trained_parameter_paths != [String::from("lm_head.weight")] {
            return invalid_local_update(String::from(
                "first production local-update proof must train only lm_head.weight",
            ));
        }
        if self.local_step_count == 0
            || self.local_step_count != self.steps.len() as u64
            || self.consumed_token_count == 0
        {
            return invalid_local_update(String::from("report step counters are invalid"));
        }
        if self.deterministic_cursor_after <= self.deterministic_cursor_before {
            return invalid_local_update(String::from(
                "deterministic cursor must advance during local update",
            ));
        }
        ensure_finite(self.loss_before, "report.loss_before")?;
        ensure_finite(self.loss_after, "report.loss_after")?;
        ensure_finite(self.validation_loss_before, "report.validation_loss_before")?;
        ensure_finite(self.validation_loss_after, "report.validation_loss_after")?;
        if self.loss_before == self.loss_after {
            return invalid_local_update(String::from("training loss must change"));
        }
        if self.validation_loss_before == self.validation_loss_after {
            return invalid_local_update(String::from("validation loss must change"));
        }
        if !self.resume_matches_uninterrupted
            || self.resumed_final_model_state_digest != self.uninterrupted_final_model_state_digest
            || self.resumed_final_optimizer_state_digest
                != self.uninterrupted_final_optimizer_state_digest
        {
            return invalid_local_update(String::from(
                "resumed local update must match uninterrupted local update",
            ));
        }
        ensure_sha256_uri(self.delta_digest.as_str(), "report.delta_digest")?;
        ensure_sha256_uri(self.report_digest.as_str(), "report.report_digest")?;
        if self.report_digest != self.stable_digest() {
            return invalid_local_update(String::from("report digest drifted"));
        }
        if !self.claim_boundary.contains("LM-head-only")
            || !self
                .claim_boundary
                .contains("not full Transformer backward")
        {
            return invalid_local_update(String::from(
                "claim boundary must keep the first production path scoped",
            ));
        }
        Ok(())
    }
}

pub fn write_a1_minimal_distributed_lm_local_update_fixture(
    output_root: impl AsRef<Path>,
) -> Result<A1MinimalDistributedLmLocalUpdateReport, A1MinimalDistributedLmLocalUpdateError> {
    run_and_write_a1_minimal_distributed_lm_local_update(
        output_root,
        A1MinimalDistributedLmLocalUpdateConfig::default(),
    )
}

pub fn run_and_write_a1_minimal_distributed_lm_local_update(
    output_root: impl AsRef<Path>,
    config: A1MinimalDistributedLmLocalUpdateConfig,
) -> Result<A1MinimalDistributedLmLocalUpdateReport, A1MinimalDistributedLmLocalUpdateError> {
    let output_root = output_root.as_ref();
    config.validate()?;
    let contract = canonical_a1_minimal_distributed_lm_lane_contract();
    contract
        .validate()
        .map_err(|error| A1MinimalDistributedLmLocalUpdateError::Invalid {
            detail: format!("lane contract invalid: {error}"),
        })?;
    let bundle = canonical_a1_minimal_distributed_lm_tokenizer_dataset_bundle()?;
    bundle.validate()?;

    let mut pre_checkpoint = A1MinimalDistributedLmLocalUpdateTrainer::fresh(
        config.clone(),
        contract.clone(),
        bundle.clone(),
    )?;
    let base_model_state = pre_checkpoint.model_state();
    let deterministic_cursor_before = pre_checkpoint.deterministic_cursor;
    let loss_before = pre_checkpoint.current_training_loss()?;
    let validation_loss_before = pre_checkpoint.validation_loss()?;
    let pre_checkpoint_steps = pre_checkpoint.run_steps(config.checkpoint_after_steps)?;
    let checkpoint_step2 = pre_checkpoint.checkpoint();
    let checkpoint_step2_receipt = write_local_update_checkpoint(
        output_root,
        A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_STEP2_FIXTURE_PATH,
        checkpoint_step2,
    )?;

    let checkpoint_step2_path =
        output_root.join(A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_STEP2_FIXTURE_PATH);
    let loaded_checkpoint =
        load_a1_minimal_distributed_lm_local_update_checkpoint(checkpoint_step2_path.as_path())?;
    let mut resumed = A1MinimalDistributedLmLocalUpdateTrainer::from_checkpoint(
        config.clone(),
        contract.clone(),
        bundle.clone(),
        loaded_checkpoint,
    )?;
    let resumed_steps =
        resumed.run_steps(config.local_step_count - config.checkpoint_after_steps)?;
    let checkpoint_step4 = resumed.checkpoint();
    let checkpoint_step4_receipt = write_local_update_checkpoint(
        output_root,
        A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_STEP4_FIXTURE_PATH,
        checkpoint_step4,
    )?;

    let mut uninterrupted =
        A1MinimalDistributedLmLocalUpdateTrainer::fresh(config.clone(), contract.clone(), bundle)?;
    let _ = uninterrupted.run_steps(config.local_step_count)?;

    let mut steps = pre_checkpoint_steps;
    steps.extend(resumed_steps);
    let validation_loss_after = resumed.validation_loss()?;
    let loss_after = steps.last().map(|step| step.loss_after).ok_or_else(|| {
        A1MinimalDistributedLmLocalUpdateError::Invalid {
            detail: String::from("local update emitted no steps"),
        }
    })?;
    let consumed_token_count = steps
        .iter()
        .map(|step| step.consumed_token_count)
        .sum::<u64>();
    let mut report = A1MinimalDistributedLmLocalUpdateReport {
        schema_version: String::from(A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_REPORT_SCHEMA_VERSION),
        lane_id: String::from(A1_MINIMAL_DISTRIBUTED_LM_LANE_ID),
        run_id: config.run_id,
        assignment_id: config.assignment_id,
        worker_id: config.worker_id,
        tokenizer_digest: contract.tokenizer_artifact_digest,
        tokenized_dataset_digest: contract.tokenized_dataset_digest,
        validation_set_digest: contract.validation_set_digest,
        base_checkpoint_ref: contract.checkpoint_family.base_checkpoint_ref,
        backward_path_kind: String::from("analytic_lm_head_cross_entropy_backward_v1"),
        finite_difference_used: false,
        trained_parameter_paths: vec![String::from("lm_head.weight")],
        local_step_count: config.local_step_count,
        consumed_token_count,
        deterministic_cursor_before,
        deterministic_cursor_after: resumed.deterministic_cursor,
        loss_before,
        loss_after,
        validation_loss_before,
        validation_loss_after,
        steps,
        checkpoint_step2: checkpoint_step2_receipt,
        checkpoint_step4: checkpoint_step4_receipt,
        resumed_final_model_state_digest: resumed.model_state_digest(),
        uninterrupted_final_model_state_digest: uninterrupted.model_state_digest(),
        resumed_final_optimizer_state_digest: resumed.optimizer_state_digest()?,
        uninterrupted_final_optimizer_state_digest: uninterrupted.optimizer_state_digest()?,
        resume_matches_uninterrupted: resumed.model_state_digest()
            == uninterrupted.model_state_digest()
            && resumed.optimizer_state_digest()? == uninterrupted.optimizer_state_digest()?,
        delta_digest: lm_head_delta_digest(&base_model_state, &resumed.model_state())?,
        report_digest: String::new(),
        claim_boundary: String::from(
            "This is the first production local-update proof for a1_minimal_distributed_lm_001. It uses analytic LM-head-only cross-entropy backward over Transformer forward hidden states, not finite-difference gradients and not full Transformer backward.",
        ),
    };
    report.report_digest = report.stable_digest();
    report.validate()?;
    write_local_update_report(
        output_root,
        A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_REPORT_FIXTURE_PATH,
        &report,
    )?;
    Ok(report)
}

pub fn load_a1_minimal_distributed_lm_local_update_checkpoint(
    checkpoint_path: impl AsRef<Path>,
) -> Result<A1MinimalDistributedLmLocalUpdateCheckpoint, A1MinimalDistributedLmLocalUpdateError> {
    let checkpoint_path = checkpoint_path.as_ref();
    let bytes = fs::read(checkpoint_path).map_err(|error| {
        A1MinimalDistributedLmLocalUpdateError::Read {
            path: checkpoint_path.display().to_string(),
            error,
        }
    })?;
    let checkpoint: A1MinimalDistributedLmLocalUpdateCheckpoint = serde_json::from_slice(&bytes)?;
    checkpoint.validate()?;
    Ok(checkpoint)
}

#[derive(Clone)]
struct A1MinimalDistributedLmLocalUpdateTrainer {
    config: A1MinimalDistributedLmLocalUpdateConfig,
    contract: A1MinimalDistributedLmLaneContract,
    training_tokens: Vec<u32>,
    validation_tokens: Vec<u32>,
    model: Cs336A1TransformerLm,
    optimizer_states: BTreeMap<String, TrainingOptimizerState>,
    optimizer_step: u64,
    deterministic_cursor: u64,
    consumed_token_count: u64,
    loss_history: Vec<f32>,
}

impl A1MinimalDistributedLmLocalUpdateTrainer {
    fn fresh(
        config: A1MinimalDistributedLmLocalUpdateConfig,
        contract: A1MinimalDistributedLmLaneContract,
        bundle: A1MinimalDistributedLmTokenizerDatasetBundle,
    ) -> Result<Self, A1MinimalDistributedLmLocalUpdateError> {
        let (training_tokens, validation_tokens) = tokens_from_bundle(&bundle)?;
        let mut model = new_model_from_contract(&contract)?;
        initialize_cs336_a1_reference_model(&mut model)?;
        Ok(Self {
            config,
            contract,
            training_tokens,
            validation_tokens,
            model,
            optimizer_states: BTreeMap::new(),
            optimizer_step: 0,
            deterministic_cursor: 0,
            consumed_token_count: 0,
            loss_history: Vec::new(),
        })
    }

    fn from_checkpoint(
        config: A1MinimalDistributedLmLocalUpdateConfig,
        contract: A1MinimalDistributedLmLaneContract,
        bundle: A1MinimalDistributedLmTokenizerDatasetBundle,
        checkpoint: A1MinimalDistributedLmLocalUpdateCheckpoint,
    ) -> Result<Self, A1MinimalDistributedLmLocalUpdateError> {
        let (training_tokens, validation_tokens) = tokens_from_bundle(&bundle)?;
        if checkpoint.lane_id != A1_MINIMAL_DISTRIBUTED_LM_LANE_ID
            || checkpoint.run_id != config.run_id
            || checkpoint.assignment_id != config.assignment_id
            || checkpoint.worker_id != config.worker_id
            || checkpoint.tokenizer_digest != contract.tokenizer_artifact_digest
            || checkpoint.tokenized_dataset_digest != contract.tokenized_dataset_digest
            || checkpoint.validation_set_digest != contract.validation_set_digest
        {
            return invalid_local_update(String::from(
                "checkpoint identity does not match local update config and lane contract",
            ));
        }
        let mut model = new_model_from_contract(&contract)?;
        model.load_state_dict(&checkpoint.model_state, ModuleStateLoadMode::Strict)?;
        Ok(Self {
            config,
            contract,
            training_tokens,
            validation_tokens,
            model,
            optimizer_states: checkpoint.optimizer_states,
            optimizer_step: checkpoint.optimizer_step,
            deterministic_cursor: checkpoint.deterministic_cursor,
            consumed_token_count: checkpoint.consumed_token_count,
            loss_history: checkpoint.loss_history,
        })
    }

    fn run_steps(
        &mut self,
        step_count: u64,
    ) -> Result<
        Vec<A1MinimalDistributedLmLocalUpdateStepReport>,
        A1MinimalDistributedLmLocalUpdateError,
    > {
        let mut reports = Vec::with_capacity(step_count as usize);
        for _ in 0..step_count {
            reports.push(self.step()?);
        }
        Ok(reports)
    }

    fn step(
        &mut self,
    ) -> Result<A1MinimalDistributedLmLocalUpdateStepReport, A1MinimalDistributedLmLocalUpdateError>
    {
        let cursor_before = self.deterministic_cursor;
        let batch = cs336_a1_get_batch(
            self.training_tokens.as_slice(),
            1,
            self.contract.model_config.context_length,
            cursor_before,
        )?;
        let loss_before = self.loss_for_batch(&batch)?;
        let mut gradients = self.analytic_lm_head_gradients(&batch)?;
        let gradient_clip = cs336_a1_gradient_clipping(
            &mut gradients,
            self.contract.optimizer_config.gradient_clip_norm,
        )?;
        let learning_rate = cs336_a1_get_lr_cosine_schedule(
            self.optimizer_step,
            self.contract.scheduler_config.max_learning_rate,
            self.contract.scheduler_config.min_learning_rate,
            self.contract.scheduler_config.warmup_iters,
            self.contract.scheduler_config.cosine_cycle_iters,
        );
        let model_state_before = self.model.state_dict();
        let model_state_digest_before = model_state_before.state_dict_digest.clone();
        let mut updated_state = model_state_before;
        let gradient_entry = gradients.entries.get("lm_head.weight").ok_or_else(|| {
            A1MinimalDistributedLmLocalUpdateError::Invalid {
                detail: String::from("analytic backward did not emit lm_head.weight"),
            }
        })?;
        let parameter_entry = updated_state
            .entries
            .get_mut("lm_head.weight")
            .ok_or_else(|| A1MinimalDistributedLmLocalUpdateError::Invalid {
                detail: String::from("model state missing lm_head.weight"),
            })?;
        let parameter_values = dense_tensor_values_mut(parameter_entry)?;
        let gradient_values = dense_tensor_values(&gradient_entry.data)?;
        let optimizer_config = self.optimizer_config(learning_rate);
        let optimizer_state = self
            .optimizer_states
            .entry(String::from("lm_head.weight"))
            .or_insert_with(|| optimizer_config.initialize_state(parameter_values.len()));
        optimizer_config.apply_step(
            parameter_values.as_mut_slice(),
            gradient_values,
            optimizer_state,
            self.optimizer_step + 1,
        )?;
        self.model
            .load_state_dict(&updated_state, ModuleStateLoadMode::Strict)?;
        let loss_after = self.loss_for_batch(&batch)?;
        self.optimizer_step += 1;
        self.deterministic_cursor += 1;
        let consumed_token_count = (batch.batch_size.saturating_mul(batch.context_length)) as u64;
        self.consumed_token_count = self
            .consumed_token_count
            .saturating_add(consumed_token_count);
        self.loss_history.push(loss_after);
        Ok(A1MinimalDistributedLmLocalUpdateStepReport {
            step_number: self.optimizer_step,
            deterministic_cursor_before: cursor_before,
            deterministic_cursor_after: self.deterministic_cursor,
            train_batch_start_positions: batch.start_positions,
            consumed_token_count,
            learning_rate,
            loss_before,
            loss_after,
            gradient_clip,
            model_state_digest_before,
            model_state_digest_after: self.model.state_dict().state_dict_digest,
            optimizer_state_digest_after: self.optimizer_state_digest()?,
        })
    }

    fn current_training_loss(&self) -> Result<f32, A1MinimalDistributedLmLocalUpdateError> {
        let batch = cs336_a1_get_batch(
            self.training_tokens.as_slice(),
            1,
            self.contract.model_config.context_length,
            self.deterministic_cursor,
        )?;
        self.loss_for_batch(&batch)
    }

    fn validation_loss(&self) -> Result<f32, A1MinimalDistributedLmLocalUpdateError> {
        validate_token_window(
            self.validation_tokens.as_slice(),
            self.contract.model_config.context_length,
        )?;
        let window_count = self.validation_tokens.len() - self.contract.model_config.context_length;
        let mut total = 0.0;
        for start in 0..window_count {
            let batch = Cs336A1ReferenceBatch {
                iteration: start as u64,
                batch_size: 1,
                context_length: self.contract.model_config.context_length,
                start_positions: vec![start],
                inputs: self.validation_tokens
                    [start..start + self.contract.model_config.context_length]
                    .to_vec(),
                targets: self.validation_tokens
                    [start + 1..start + self.contract.model_config.context_length + 1]
                    .to_vec(),
            };
            total += self.loss_for_batch(&batch)?;
        }
        Ok(total / window_count as f32)
    }

    fn loss_for_batch(
        &self,
        batch: &Cs336A1ReferenceBatch,
    ) -> Result<f32, A1MinimalDistributedLmLocalUpdateError> {
        let logits = self
            .model
            .forward_tokens(batch.token_shape(), &batch.input_ids())?;
        let loss = cross_entropy_loss(&logits, &batch.target_ids(), LossReduction::Mean)?;
        scalar_from_nn_tensor(&loss)
    }

    fn analytic_lm_head_gradients(
        &self,
        batch: &Cs336A1ReferenceBatch,
    ) -> Result<ModuleStateDict, A1MinimalDistributedLmLocalUpdateError> {
        let hidden = self
            .model
            .final_hidden_for_tokens(batch.token_shape(), &batch.input_ids())?;
        let logits = self.model.logits_from_final_hidden(&hidden)?;
        let hidden_values = hidden.as_f32_slice()?;
        let logits_values = logits.as_f32_slice()?;
        let target_ids = batch.target_ids();
        let vocab_size = self.contract.model_config.vocab_size as usize;
        let d_model = self.contract.model_config.d_model;
        let row_count = target_ids.len();
        if row_count == 0
            || hidden_values.len() != row_count * d_model
            || logits_values.len() != row_count * vocab_size
        {
            return invalid_local_update(String::from(
                "hidden/logit shapes do not match target rows",
            ));
        }
        let mut gradient = vec![0.0; vocab_size * d_model];
        for row in 0..row_count {
            let logits_row = &logits_values[row * vocab_size..(row + 1) * vocab_size];
            let hidden_row = &hidden_values[row * d_model..(row + 1) * d_model];
            let probabilities = softmax_row(logits_row);
            for class_index in 0..vocab_size {
                let target_adjustment = if class_index == target_ids[row] {
                    1.0
                } else {
                    0.0
                };
                let seed = (probabilities[class_index] - target_adjustment) / row_count as f32;
                for hidden_index in 0..d_model {
                    gradient[class_index * d_model + hidden_index] +=
                        seed * hidden_row[hidden_index];
                }
            }
        }
        let base_state = self.model.state_dict();
        let lm_head = base_state.entries.get("lm_head.weight").ok_or_else(|| {
            A1MinimalDistributedLmLocalUpdateError::Invalid {
                detail: String::from("model state missing lm_head.weight"),
            }
        })?;
        let entry = ModuleStateEntry {
            path: String::from("lm_head.weight"),
            kind: ModuleStateEntryKind::Parameter,
            spec: lm_head.spec.clone(),
            data: TensorData::F32(gradient),
            requires_grad: true,
            persistent: true,
        };
        Ok(ModuleStateDict::new(
            base_state.root_module_id,
            base_state.root_module_kind,
            ModuleStateView::PersistentOnly,
            BTreeMap::from([(String::from("lm_head.weight"), entry)]),
        )?)
    }

    fn checkpoint(&self) -> A1MinimalDistributedLmLocalUpdateCheckpoint {
        let mut checkpoint = A1MinimalDistributedLmLocalUpdateCheckpoint {
            schema_version: String::from(
                A1_MINIMAL_DISTRIBUTED_LM_LOCAL_UPDATE_CHECKPOINT_SCHEMA_VERSION,
            ),
            lane_id: String::from(A1_MINIMAL_DISTRIBUTED_LM_LANE_ID),
            run_id: self.config.run_id.clone(),
            assignment_id: self.config.assignment_id.clone(),
            worker_id: self.config.worker_id.clone(),
            tokenizer_digest: self.contract.tokenizer_artifact_digest.clone(),
            tokenized_dataset_digest: self.contract.tokenized_dataset_digest.clone(),
            validation_set_digest: self.contract.validation_set_digest.clone(),
            base_checkpoint_ref: self.contract.checkpoint_family.base_checkpoint_ref.clone(),
            optimizer_step: self.optimizer_step,
            deterministic_cursor: self.deterministic_cursor,
            consumed_token_count: self.consumed_token_count,
            model_state: self.model.state_dict(),
            optimizer_states: self.optimizer_states.clone(),
            loss_history: self.loss_history.clone(),
            checkpoint_digest: String::new(),
        };
        checkpoint.checkpoint_digest = checkpoint.stable_digest();
        checkpoint
    }

    fn model_state(&self) -> ModuleStateDict {
        self.model.state_dict()
    }

    fn model_state_digest(&self) -> String {
        self.model.state_dict().state_dict_digest
    }

    fn optimizer_state_digest(&self) -> Result<String, A1MinimalDistributedLmLocalUpdateError> {
        Ok(sha256_uri_digest(
            b"psion_a1_minimal_distributed_lm_optimizer_state|",
            &self.optimizer_states,
        ))
    }

    fn optimizer_config(&self, learning_rate: f32) -> TrainingOptimizerConfig {
        TrainingOptimizerConfig::adamw(
            learning_rate,
            self.contract.optimizer_config.adam_beta1,
            self.contract.optimizer_config.adam_beta2,
            self.contract.optimizer_config.adam_epsilon,
        )
        .with_weight_decay(self.contract.optimizer_config.weight_decay)
    }
}

fn new_model_from_contract(
    contract: &A1MinimalDistributedLmLaneContract,
) -> Result<Cs336A1TransformerLm, A1MinimalDistributedLmLocalUpdateError> {
    let config = Cs336A1ReferenceConfig {
        vocab_size: contract.model_config.vocab_size as usize,
        context_length: contract.model_config.context_length,
        d_model: contract.model_config.d_model,
        num_layers: contract.model_config.num_layers,
        num_heads: contract.model_config.num_heads,
        d_ff: contract.model_config.d_ff,
    };
    Ok(Cs336A1TransformerLm::new(
        "a1_minimal_distributed_lm_local_update",
        config,
        contract.model_config.rope_theta,
        contract.model_config.rms_norm_eps,
    )?)
}

fn tokens_from_bundle(
    bundle: &A1MinimalDistributedLmTokenizerDatasetBundle,
) -> Result<(Vec<u32>, Vec<u32>), A1MinimalDistributedLmLocalUpdateError> {
    if bundle.training_shards.len() != 1 || bundle.validation_shards.len() != 1 {
        return invalid_local_update(String::from(
            "first local update proof expects one training shard and one validation shard",
        ));
    }
    let training_tokens = bundle.training_shards[0].tokens.clone();
    let validation_tokens = bundle.validation_shards[0].tokens.clone();
    validate_token_window(training_tokens.as_slice(), 2)?;
    validate_token_window(validation_tokens.as_slice(), 2)?;
    Ok((training_tokens, validation_tokens))
}

fn validate_token_window(
    tokens: &[u32],
    context_length: usize,
) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
    if tokens.len() <= context_length {
        return invalid_local_update(format!(
            "token sequence length {} must exceed context_length {}",
            tokens.len(),
            context_length
        ));
    }
    Ok(())
}

fn write_local_update_checkpoint(
    output_root: &Path,
    relative_path: &str,
    checkpoint: A1MinimalDistributedLmLocalUpdateCheckpoint,
) -> Result<
    A1MinimalDistributedLmLocalUpdateCheckpointReceipt,
    A1MinimalDistributedLmLocalUpdateError,
> {
    checkpoint.validate()?;
    let output_path = output_root.join(relative_path);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            A1MinimalDistributedLmLocalUpdateError::CreateDir {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    let mut bytes = serde_json::to_vec_pretty(&checkpoint)?;
    bytes.push(b'\n');
    fs::write(output_path.as_path(), bytes.as_slice()).map_err(|error| {
        A1MinimalDistributedLmLocalUpdateError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(A1MinimalDistributedLmLocalUpdateCheckpointReceipt {
        checkpoint_path: String::from(relative_path),
        checkpoint_digest: checkpoint.checkpoint_digest,
        optimizer_step: checkpoint.optimizer_step,
        deterministic_cursor: checkpoint.deterministic_cursor,
        model_state_digest: checkpoint.model_state.state_dict_digest,
        optimizer_state_digest: sha256_uri_digest(
            b"psion_a1_minimal_distributed_lm_optimizer_state|",
            &checkpoint.optimizer_states,
        ),
        checkpoint_byte_count: bytes.len() as u64,
    })
}

fn write_local_update_report(
    output_root: &Path,
    relative_path: &str,
    report: &A1MinimalDistributedLmLocalUpdateReport,
) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
    let output_path = output_root.join(relative_path);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            A1MinimalDistributedLmLocalUpdateError::CreateDir {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    fs::write(output_path.as_path(), bytes).map_err(|error| {
        A1MinimalDistributedLmLocalUpdateError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })
}

fn scalar_from_nn_tensor(tensor: &NnTensor) -> Result<f32, A1MinimalDistributedLmLocalUpdateError> {
    let values = tensor.as_f32_slice()?;
    if values.len() != 1 {
        return invalid_local_update(format!("expected scalar tensor, found {}", values.len()));
    }
    Ok(values[0])
}

fn softmax_row(values: &[f32]) -> Vec<f32> {
    let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = values
        .iter()
        .map(|value| (*value - max_value).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f32>();
    if sum > 0.0 {
        for value in &mut probabilities {
            *value /= sum;
        }
    }
    probabilities
}

fn dense_tensor_values(
    data: &TensorData,
) -> Result<&[f32], A1MinimalDistributedLmLocalUpdateError> {
    match data {
        TensorData::F32(values) | TensorData::BF16(values) => Ok(values.as_slice()),
        other => invalid_local_update(format!("expected dense floating tensor, found `{other:?}`")),
    }
}

fn dense_tensor_values_mut(
    entry: &mut ModuleStateEntry,
) -> Result<&mut Vec<f32>, A1MinimalDistributedLmLocalUpdateError> {
    match &mut entry.data {
        TensorData::F32(values) | TensorData::BF16(values) => Ok(values),
        other => invalid_local_update(format!(
            "expected mutable dense floating tensor at `{}`, found `{other:?}`",
            entry.path
        )),
    }
}

fn lm_head_delta_digest(
    before: &ModuleStateDict,
    after: &ModuleStateDict,
) -> Result<String, A1MinimalDistributedLmLocalUpdateError> {
    let before_values = lm_head_values(before)?;
    let after_values = lm_head_values(after)?;
    if before_values.len() != after_values.len() {
        return invalid_local_update(String::from("lm_head value lengths differ"));
    }
    let delta = before_values
        .iter()
        .zip(after_values.iter())
        .map(|(before, after)| after - before)
        .collect::<Vec<_>>();
    Ok(sha256_uri_digest(
        b"psion_a1_minimal_distributed_lm_lm_head_delta|",
        &(String::from("lm_head.weight"), delta),
    ))
}

fn lm_head_values(
    state: &ModuleStateDict,
) -> Result<&[f32], A1MinimalDistributedLmLocalUpdateError> {
    let entry = state.entries.get("lm_head.weight").ok_or_else(|| {
        A1MinimalDistributedLmLocalUpdateError::Invalid {
            detail: String::from("state missing lm_head.weight"),
        }
    })?;
    dense_tensor_values(&entry.data)
}

fn validate_lane_identity(lane_id: &str) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
    if lane_id != A1_MINIMAL_DISTRIBUTED_LM_LANE_ID {
        return invalid_local_update(format!(
            "lane_id must stay `{}` but was `{lane_id}`",
            A1_MINIMAL_DISTRIBUTED_LM_LANE_ID
        ));
    }
    Ok(())
}

fn ensure_nonempty(value: &str, field: &str) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
    if value.trim().is_empty() {
        return invalid_local_update(format!("field `{field}` must not be empty"));
    }
    Ok(())
}

fn ensure_finite(value: f32, field: &str) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
    if !value.is_finite() {
        return invalid_local_update(format!("field `{field}` must be finite"));
    }
    Ok(())
}

fn ensure_sha256_uri(
    value: &str,
    field: &str,
) -> Result<(), A1MinimalDistributedLmLocalUpdateError> {
    ensure_nonempty(value, field)?;
    let Some(hex) = value.strip_prefix("sha256:") else {
        return invalid_local_update(format!("field `{field}` must use sha256:<hex> form"));
    };
    if hex.len() != 64 || !hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return invalid_local_update(format!(
            "field `{field}` must contain a 64-hex sha256 digest"
        ));
    }
    Ok(())
}

fn invalid_local_update<T>(detail: String) -> Result<T, A1MinimalDistributedLmLocalUpdateError> {
    Err(A1MinimalDistributedLmLocalUpdateError::Invalid { detail })
}

fn sha256_uri_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(
        serde_json::to_vec(value)
            .expect("A1 minimal distributed LM local-update payload should serialize"),
    );
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        load_a1_minimal_distributed_lm_local_update_checkpoint,
        write_a1_minimal_distributed_lm_local_update_fixture,
        A1MinimalDistributedLmLocalUpdateCheckpoint, A1MinimalDistributedLmLocalUpdateError,
        A1MinimalDistributedLmLocalUpdateReport,
    };
    use tempfile::tempdir;

    fn fixture_report() -> A1MinimalDistributedLmLocalUpdateReport {
        serde_json::from_str(include_str!(
            "../../../fixtures/psion/a1_minimal_distributed_lm/local_update_report_v1.json"
        ))
        .expect("A1 minimal distributed LM local update report fixture should parse")
    }

    fn fixture_checkpoint_step4() -> A1MinimalDistributedLmLocalUpdateCheckpoint {
        serde_json::from_str(include_str!(
            "../../../fixtures/psion/a1_minimal_distributed_lm/local_update_checkpoint_step4_v1.json"
        ))
        .expect("A1 minimal distributed LM local update checkpoint fixture should parse")
    }

    #[test]
    fn a1_minimal_distributed_lm_local_update_fixture_validates() {
        fixture_report()
            .validate()
            .expect("local update report fixture should validate");
        fixture_checkpoint_step4()
            .validate()
            .expect("local update checkpoint fixture should validate");
    }

    #[test]
    fn a1_minimal_distributed_lm_local_update_uses_no_finite_differences() {
        let report = fixture_report();
        assert!(!report.finite_difference_used);
        assert_eq!(
            report.backward_path_kind,
            "analytic_lm_head_cross_entropy_backward_v1"
        );
        assert_eq!(report.trained_parameter_paths, vec!["lm_head.weight"]);
    }

    #[test]
    fn a1_minimal_distributed_lm_local_update_changes_losses_and_resumes_exactly() {
        let report = fixture_report();
        assert_ne!(report.loss_before, report.loss_after);
        assert_ne!(report.validation_loss_before, report.validation_loss_after);
        assert!(report.resume_matches_uninterrupted);
        assert_eq!(
            report.resumed_final_model_state_digest,
            report.uninterrupted_final_model_state_digest
        );
        assert_eq!(
            report.resumed_final_optimizer_state_digest,
            report.uninterrupted_final_optimizer_state_digest
        );
    }

    #[test]
    fn a1_minimal_distributed_lm_local_update_writer_is_reproducible(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let generated = write_a1_minimal_distributed_lm_local_update_fixture(root.path())?;
        assert_eq!(generated, fixture_report());
        let checkpoint_path = root
            .path()
            .join("fixtures/psion/a1_minimal_distributed_lm/local_update_checkpoint_step4_v1.json");
        let loaded = load_a1_minimal_distributed_lm_local_update_checkpoint(checkpoint_path)?;
        assert_eq!(loaded, fixture_checkpoint_step4());
        Ok(())
    }

    #[test]
    fn a1_minimal_distributed_lm_local_update_rejects_checkpoint_digest_drift() {
        let mut checkpoint = fixture_checkpoint_step4();
        checkpoint.checkpoint_digest =
            String::from("sha256:0000000000000000000000000000000000000000000000000000000000000000");
        let error = checkpoint
            .validate()
            .expect_err("checkpoint digest drift should be rejected");
        assert!(matches!(
            error,
            A1MinimalDistributedLmLocalUpdateError::Invalid { .. }
        ));
    }
}

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use psionic_backend_cuda::CudaBackend;
use psionic_core::{PsionicRefusal, PsionicRefusalCode, PsionicRefusalScope, TensorData, TensorId};
use psionic_data::{
    builtin_parameter_golf_sentencepiece_byte_luts,
    load_parameter_golf_validation_tokens_from_paths, materialize_parameter_golf_token_window,
    parameter_golf_dataset_bundle_from_local_dir, DatasetIterationMode, DatasetKey,
    ParameterGolfDataError, ParameterGolfDatasetBundle, ParameterGolfSentencePieceByteLuts,
    ParameterGolfTokenStreamContract, ParameterGolfTokenStreamCursor,
    PARAMETER_GOLF_TRAIN_SPLIT_NAME,
};
use psionic_ir::AutodiffBackwardResult;
use psionic_ir::{AutodiffError, GraphError};
use psionic_models::{
    ParameterGolfModelError, ParameterGolfReferenceModel, PARAMETER_GOLF_BASELINE_MODEL_ID,
    PARAMETER_GOLF_BASELINE_REVISION,
};
use psionic_runtime::{DeliveredExecutionContext, DeviceDescriptor, RuntimeError, RuntimeHealth};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    apply_parameter_golf_cuda_muon_step, bind_parameter_golf_baseline_training_graph_inputs,
    build_parameter_golf_baseline_training_graph, build_tokenizer_digest,
    builtin_parameter_golf_cuda_training_capability_report, device_matches_single_h100,
    export_parameter_golf_int8_zlib_model_artifact, inspect_local_single_h100_machine,
    materialize_parameter_golf_baseline_training_gradients, parameter_golf_optimizer_plan,
    training_batch_from_window_tokens, ParameterGolfBaselineTrainingGraph,
    ParameterGolfBatchGeometry, ParameterGolfOptimizerExecution,
    ParameterGolfReferenceTrainingError, ParameterGolfSingleH100BringupError,
    ParameterGolfSingleH100ChallengeThresholds, ParameterGolfSingleH100MachineObservation,
    ParameterGolfTrainError, ParameterGolfTrainingHyperparameters, TrainingOptimizerConfig,
    TrainingOptimizerState, PARAMETER_GOLF_SINGLE_H100_DATASET_REF,
    PARAMETER_GOLF_SINGLE_H100_DATASET_VERSION, PARAMETER_GOLF_SINGLE_H100_VARIANT,
};

/// Config for the bounded Rust-owned single-H100 Parameter Golf trainer lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100TrainingConfig {
    /// Stable run identifier.
    pub run_id: String,
    /// Local dataset root containing cached FineWeb/SP1024 shards.
    pub dataset_root: PathBuf,
    /// Local tokenizer artifact path.
    pub tokenizer_path: PathBuf,
    /// Stable dataset identity expected by the run.
    pub dataset_key: DatasetKey,
    /// Stable tokenizer or dataset variant label.
    pub variant: String,
    /// Public single-device challenge geometry.
    pub geometry: ParameterGolfBatchGeometry,
    /// Public baseline optimization contract.
    pub hyperparameters: ParameterGolfTrainingHyperparameters,
    /// Bounded optimizer-step cap for the first Rust-owned single-H100 proof.
    pub max_steps: u64,
}

impl ParameterGolfSingleH100TrainingConfig {
    /// Returns the canonical bounded single-H100 training defaults.
    #[must_use]
    pub fn challenge_defaults(
        dataset_root: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            run_id: String::from("parameter-golf-single-h100-trainer"),
            dataset_root: dataset_root.into(),
            tokenizer_path: tokenizer_path.into(),
            dataset_key: DatasetKey::new(
                PARAMETER_GOLF_SINGLE_H100_DATASET_REF,
                PARAMETER_GOLF_SINGLE_H100_DATASET_VERSION,
            ),
            variant: String::from(PARAMETER_GOLF_SINGLE_H100_VARIANT),
            geometry: ParameterGolfBatchGeometry::challenge_single_device_defaults(),
            hyperparameters: ParameterGolfTrainingHyperparameters::baseline_defaults(),
            max_steps: 1,
        }
    }

    fn validate(&self) -> Result<(), ParameterGolfSingleH100TrainingError> {
        if self.run_id.trim().is_empty() {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("run_id must be non-empty"),
            });
        }
        if self.variant.trim().is_empty() {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("variant must be non-empty"),
            });
        }
        if self.max_steps == 0 {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from("max_steps must be positive"),
            });
        }
        if self.geometry != ParameterGolfBatchGeometry::challenge_single_device_defaults() {
            return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
                message: String::from(
                    "single-H100 training requires challenge_single_device_defaults geometry",
                ),
            });
        }
        if !self.dataset_root.is_dir() {
            return Err(ParameterGolfSingleH100TrainingError::MissingPath {
                path: self.dataset_root.display().to_string(),
                expected: String::from("dataset directory"),
            });
        }
        if !self.tokenizer_path.is_file() {
            return Err(ParameterGolfSingleH100TrainingError::MissingPath {
                path: self.tokenizer_path.display().to_string(),
                expected: String::from("tokenizer file"),
            });
        }
        Ok(())
    }
}

/// Observed current-state disposition for the bounded single-H100 trainer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfSingleH100TrainingDisposition {
    RefusedMachineContract,
    RefusedCudaBlockers,
    TrainingExecuted,
}

/// Machine-readable phase timings for one challenge-step aggregate.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100PhaseTimings {
    pub window_planning_ms: u64,
    pub token_materialization_ms: u64,
    pub embedding_gather_ms: u64,
    pub forward_loss_cuda_ms: u64,
    pub retained_forward_readback_ms: u64,
    pub backward_cuda_ms: u64,
    pub gradient_readback_ms: u64,
    pub host_gradient_materialization_ms: u64,
    pub optimizer_step_ms: u64,
    pub retained_binding_tensor_count: u32,
    pub retained_binding_f32_count: u64,
    pub gradient_tensor_count: u32,
    pub gradient_f32_count: u64,
}

impl ParameterGolfSingleH100PhaseTimings {
    fn accumulate(&mut self, other: &Self) {
        self.window_planning_ms = self
            .window_planning_ms
            .saturating_add(other.window_planning_ms);
        self.token_materialization_ms = self
            .token_materialization_ms
            .saturating_add(other.token_materialization_ms);
        self.embedding_gather_ms = self
            .embedding_gather_ms
            .saturating_add(other.embedding_gather_ms);
        self.forward_loss_cuda_ms = self
            .forward_loss_cuda_ms
            .saturating_add(other.forward_loss_cuda_ms);
        self.retained_forward_readback_ms = self
            .retained_forward_readback_ms
            .saturating_add(other.retained_forward_readback_ms);
        self.backward_cuda_ms = self.backward_cuda_ms.saturating_add(other.backward_cuda_ms);
        self.gradient_readback_ms = self
            .gradient_readback_ms
            .saturating_add(other.gradient_readback_ms);
        self.host_gradient_materialization_ms = self
            .host_gradient_materialization_ms
            .saturating_add(other.host_gradient_materialization_ms);
        self.optimizer_step_ms = self
            .optimizer_step_ms
            .saturating_add(other.optimizer_step_ms);
        self.retained_binding_tensor_count = self
            .retained_binding_tensor_count
            .saturating_add(other.retained_binding_tensor_count);
        self.retained_binding_f32_count = self
            .retained_binding_f32_count
            .saturating_add(other.retained_binding_f32_count);
        self.gradient_tensor_count = self
            .gradient_tensor_count
            .saturating_add(other.gradient_tensor_count);
        self.gradient_f32_count = self
            .gradient_f32_count
            .saturating_add(other.gradient_f32_count);
    }

    fn host_materialization_ms(&self) -> u64 {
        self.embedding_gather_ms
            .saturating_add(self.retained_forward_readback_ms)
            .saturating_add(self.gradient_readback_ms)
            .saturating_add(self.host_gradient_materialization_ms)
    }
}

/// Per-step machine-readable training metrics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100TrainingStepMetrics {
    pub global_step: u64,
    pub train_window_ids: Vec<String>,
    pub mean_microbatch_loss: f32,
    pub learning_rate_multiplier: f32,
    pub muon_momentum: f32,
    pub phase_timings: ParameterGolfSingleH100PhaseTimings,
}

/// Machine-readable validation summary for the accelerated single-H100 lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100ValidationSummary {
    pub evaluated_sequence_count: usize,
    pub evaluated_token_count: u64,
    pub evaluated_byte_count: u64,
    pub mean_loss: f64,
    pub bits_per_byte: f64,
}

/// End-to-end machine-readable trainer report for the bounded single-H100 lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfSingleH100TrainingReport {
    pub schema_version: u32,
    pub scope_window: String,
    pub run_id: String,
    pub dataset_root: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dataset_key: DatasetKey,
    pub variant: String,
    pub tokenizer_digest: psionic_data::TokenizerDigest,
    pub dataset_manifest_digest: String,
    pub train_shard_count: usize,
    pub validation_shard_count: usize,
    pub train_token_count: u64,
    pub validation_token_count: u64,
    pub geometry: ParameterGolfBatchGeometry,
    pub hyperparameters: ParameterGolfTrainingHyperparameters,
    pub max_steps: u64,
    pub executed_steps: u64,
    pub delivered_execution: DeliveredExecutionContext,
    pub machine_thresholds: ParameterGolfSingleH100ChallengeThresholds,
    pub observed_cuda_health: RuntimeHealth,
    pub cuda_discovery_error: Option<String>,
    pub observed_cuda_devices: Vec<DeviceDescriptor>,
    pub matching_h100_device_count: usize,
    pub machine_contract_satisfied: bool,
    pub baseline_model_id: String,
    pub baseline_model_revision: String,
    pub baseline_model_descriptor_digest: String,
    pub optimizer_plan_digest: String,
    pub cuda_training_capability_report_digest: String,
    pub challenge_kernel_blockers: Vec<String>,
    pub initial_validation: Option<ParameterGolfSingleH100ValidationSummary>,
    pub final_validation: Option<ParameterGolfSingleH100ValidationSummary>,
    pub compressed_model_bytes: Option<u64>,
    pub compressed_model_artifact_ref: Option<String>,
    pub compressed_model_artifact_digest: Option<String>,
    pub step_metrics: Vec<ParameterGolfSingleH100TrainingStepMetrics>,
    pub aggregate_phase_timings: Option<ParameterGolfSingleH100PhaseTimings>,
    pub final_training_cursor: Option<ParameterGolfTokenStreamCursor>,
    pub started_at_ms: u64,
    pub finished_at_ms: u64,
    pub observed_wallclock_ms: u64,
    pub disposition: ParameterGolfSingleH100TrainingDisposition,
    pub refusal: Option<PsionicRefusal>,
    pub claim_boundary: String,
    pub summary: String,
    pub report_digest: String,
}

impl ParameterGolfSingleH100TrainingReport {
    /// Returns whether the report is an executed trainer proof.
    #[must_use]
    pub fn training_executed(&self) -> bool {
        self.disposition == ParameterGolfSingleH100TrainingDisposition::TrainingExecuted
    }
}

#[derive(Clone, Debug)]
struct ParameterGolfSingleH100TrainerState {
    parameter_states: BTreeMap<String, ParameterGolfParameterState>,
}

#[derive(Clone, Debug)]
enum ParameterGolfParameterState {
    Adam {
        shape: Vec<usize>,
        values: Vec<f32>,
        optimizer: TrainingOptimizerConfig,
        optimizer_state: TrainingOptimizerState,
    },
    Muon {
        shape: Vec<usize>,
        values: Vec<f32>,
        optimizer: crate::ParameterGolfMuonConfig,
        optimizer_state: crate::ParameterGolfMuonState,
    },
}

impl ParameterGolfParameterState {
    fn values(&self) -> &[f32] {
        match self {
            Self::Adam { values, .. } | Self::Muon { values, .. } => values.as_slice(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Adam { shape, .. } | Self::Muon { shape, .. } => shape.as_slice(),
        }
    }

    fn apply_gradients(
        &mut self,
        gradients: &[f32],
        learning_rate_multiplier: f32,
        muon_momentum: f32,
        step_number: u64,
    ) -> Result<(), ParameterGolfSingleH100TrainingError> {
        match self {
            Self::Adam {
                values,
                optimizer,
                optimizer_state,
                ..
            } => {
                let mut optimizer = optimizer.clone();
                optimizer.learning_rate *= learning_rate_multiplier;
                optimizer.apply_step(
                    values.as_mut_slice(),
                    gradients,
                    optimizer_state,
                    step_number,
                )?;
                Ok(())
            }
            Self::Muon {
                shape,
                values,
                optimizer,
                optimizer_state,
            } => {
                let mut optimizer = optimizer.clone();
                optimizer.learning_rate *= learning_rate_multiplier;
                optimizer.momentum = muon_momentum;
                apply_parameter_golf_cuda_muon_step(
                    values.as_mut_slice(),
                    shape.as_slice(),
                    gradients,
                    &optimizer,
                    optimizer_state,
                )?;
                Ok(())
            }
        }
    }
}

/// Failure while executing the bounded single-H100 trainer.
#[derive(Debug, Error)]
pub enum ParameterGolfSingleH100TrainingError {
    #[error("parameter golf single-H100 training config is invalid: {message}")]
    InvalidConfig { message: String },
    #[error("parameter golf single-H100 training expected {expected} at `{path}`")]
    MissingPath { path: String, expected: String },
    #[error("parameter golf single-H100 training serialization failed: {message}")]
    Serialization { message: String },
    #[error("parameter golf single-H100 training could not select a qualifying H100 CUDA device")]
    MissingSelectedH100,
    #[error("parameter golf single-H100 training graph is missing tensor `{tensor_id}`")]
    MissingGraphTensor { tensor_id: TensorId },
    #[error("parameter golf single-H100 training graph output `{tensor_id}` was not materialized")]
    MissingGraphOutput { tensor_id: TensorId },
    #[error("parameter golf single-H100 training is missing parameter state for `{parameter_id}`")]
    MissingParameterState { parameter_id: String },
    #[error(transparent)]
    Data(#[from] ParameterGolfDataError),
    #[error(transparent)]
    Model(#[from] ParameterGolfModelError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Autodiff(#[from] AutodiffError),
    #[error(transparent)]
    BaselineGraph(#[from] crate::ParameterGolfBaselineGraphError),
    #[error(transparent)]
    Bringup(#[from] ParameterGolfSingleH100BringupError),
    #[error(transparent)]
    ReferenceTraining(#[from] ParameterGolfReferenceTrainingError),
    #[error(transparent)]
    Train(#[from] ParameterGolfTrainError),
    #[error(transparent)]
    Optimizer(#[from] crate::TrainingOptimizerError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Writes the bounded single-H100 training report to disk.
pub fn write_parameter_golf_single_h100_training_report(
    output_path: &Path,
    config: &ParameterGolfSingleH100TrainingConfig,
) -> Result<ParameterGolfSingleH100TrainingReport, ParameterGolfSingleH100TrainingError> {
    let report = build_parameter_golf_single_h100_training_report(config)?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded = serde_json::to_vec_pretty(&report).map_err(|error| {
        ParameterGolfSingleH100TrainingError::Serialization {
            message: error.to_string(),
        }
    })?;
    fs::write(output_path, encoded)?;
    Ok(report)
}

/// Builds the bounded single-H100 training report.
pub fn build_parameter_golf_single_h100_training_report(
    config: &ParameterGolfSingleH100TrainingConfig,
) -> Result<ParameterGolfSingleH100TrainingReport, ParameterGolfSingleH100TrainingError> {
    config.validate()?;
    let tokenizer_bytes = fs::read(&config.tokenizer_path)?;
    let tokenizer_digest = build_tokenizer_digest(tokenizer_bytes.as_slice());
    let bundle = parameter_golf_dataset_bundle_from_local_dir(
        config.dataset_key.clone(),
        &config.dataset_root,
        config.variant.clone(),
        tokenizer_digest.clone(),
        config.tokenizer_path.display().to_string(),
        None,
    )?;
    let machine_observation = inspect_local_single_h100_machine();
    let initial_model = ParameterGolfReferenceModel::baseline_fixture(Default::default())?;
    let optimizer_plan =
        parameter_golf_optimizer_plan(initial_model.descriptor(), &config.hyperparameters)?;
    let optimizer_plan_digest =
        stable_digest(b"psionic_parameter_golf_optimizer_plan|", &optimizer_plan);
    let capability_report = builtin_parameter_golf_cuda_training_capability_report()?;
    let started_at_ms = unix_time_ms();

    if !machine_observation.machine_contract_satisfied {
        return Ok(refusal_report(
            config,
            tokenizer_digest,
            &bundle,
            &machine_observation,
            &initial_model,
            optimizer_plan_digest,
            capability_report.report_digest.clone(),
            capability_report.challenge_kernel_blockers().to_vec(),
            ParameterGolfSingleH100TrainingDisposition::RefusedMachineContract,
            machine_observation.refusal.clone(),
            started_at_ms,
            String::from(
                "The Rust-owned single-H100 trainer path is now implemented, but this run still refused because the local machine contract does not satisfy the non-MIG H100 requirement.",
            ),
        ));
    }
    if !capability_report.challenge_kernel_blockers().is_empty() {
        return Ok(refusal_report(
            config,
            tokenizer_digest,
            &bundle,
            &machine_observation,
            &initial_model,
            optimizer_plan_digest,
            capability_report.report_digest.clone(),
            capability_report.challenge_kernel_blockers().to_vec(),
            ParameterGolfSingleH100TrainingDisposition::RefusedCudaBlockers,
            Some(
                PsionicRefusal::new(
                    PsionicRefusalCode::UnsupportedBackendCapability,
                    PsionicRefusalScope::Runtime,
                    format!(
                        "single-H100 trainer requires an empty Parameter Golf CUDA blocker list, found {:?}",
                        capability_report.challenge_kernel_blockers()
                    ),
                )
                .with_subject(String::from("parameter_golf_single_h100_cuda_blockers")),
            ),
            started_at_ms,
            String::from(
                "The Rust-owned single-H100 trainer path refuses explicitly when the committed Parameter Golf CUDA capability report still carries challenge blockers.",
            ),
        ));
    }

    let mut cuda_backend = CudaBackend::new();
    let Some(selected_device) = cuda_backend.selected_device().cloned() else {
        return Err(ParameterGolfSingleH100TrainingError::MissingSelectedH100);
    };
    if !device_matches_single_h100(&selected_device, &machine_observation.thresholds) {
        return Err(ParameterGolfSingleH100TrainingError::MissingSelectedH100);
    }
    let delivered_execution =
        DeliveredExecutionContext::new("cuda", None, vec![selected_device.inventory_qualifiers()]);

    emit_progress_line(format!(
        "single_h100_train_start run_id={} device={} max_steps={} grad_accum_steps={} local_train_sequences={} local_validation_sequences={}",
        config.run_id,
        selected_device
            .device_name
            .as_deref()
            .unwrap_or("unknown"),
        config.max_steps,
        config.geometry.grad_accum_steps,
        config.geometry.local_train_batch_sequences(),
        config.geometry.local_validation_batch_sequences(),
    ));

    let byte_luts = builtin_parameter_golf_sentencepiece_byte_luts()?;
    let validation_tokens = load_parameter_golf_validation_tokens_from_paths(
        &bundle
            .validation_shards
            .iter()
            .map(|receipt| PathBuf::from(&receipt.path))
            .collect::<Vec<_>>(),
        config.geometry.train_sequence_length,
    )?;
    let mut train_graph_cache = BTreeMap::new();
    emit_progress_line(format!(
        "initial_validation_skipped reason=issue_454_first_h100_proof focuses on the real training critical path and final challenge metrics token_count={}",
        validation_tokens.len(),
    ));
    let initial_validation = None;

    let mut trainer_state = seed_parameter_states(&initial_model, &optimizer_plan)?;
    let train_contract = ParameterGolfTokenStreamContract::new(
        bundle.manifest.key.clone(),
        PARAMETER_GOLF_TRAIN_SPLIT_NAME,
    )
    .with_mode(DatasetIterationMode::Repeat);
    let mut cursor = ParameterGolfTokenStreamCursor::new(PARAMETER_GOLF_TRAIN_SPLIT_NAME);
    let mut current_model = initial_model.clone();
    let mut step_metrics = Vec::new();
    let mut aggregate_phase_timings = ParameterGolfSingleH100PhaseTimings::default();
    let requested_train_tokens =
        config.geometry.local_train_batch_tokens().saturating_add(1) as u64;

    for step_index in 0..config.max_steps {
        emit_progress_line(format!(
            "train_step_start step={}/{} grad_accum_steps={}",
            step_index + 1,
            config.max_steps,
            config.geometry.grad_accum_steps,
        ));
        let mut accumulated_gradients = zero_gradients(&trainer_state);
        let mut microbatch_loss_sum = 0.0_f32;
        let mut window_ids = Vec::new();
        let mut step_profile = ParameterGolfSingleH100PhaseTimings::default();
        for micro_step in 0..config.geometry.grad_accum_steps {
            let plan_started = Instant::now();
            let window = train_contract
                .plan_window(&bundle.manifest, &cursor, requested_train_tokens)?
                .ok_or_else(|| ParameterGolfSingleH100TrainingError::InvalidConfig {
                    message: String::from("could not plan the next training token window"),
                })?;
            step_profile.window_planning_ms = step_profile
                .window_planning_ms
                .saturating_add(duration_ms(plan_started));
            cursor = window.end_cursor.clone();
            window_ids.push(window.window_id.clone());

            let tokens_started = Instant::now();
            let tokens = materialize_parameter_golf_token_window(&bundle, &window)?;
            step_profile.token_materialization_ms = step_profile
                .token_materialization_ms
                .saturating_add(duration_ms(tokens_started));
            let (input_ids, target_ids) =
                training_batch_from_window_tokens(tokens.as_slice(), &config.geometry)?;

            let embed_started = Instant::now();
            let embedded_inputs = crate::gather_parameter_golf_embedded_inputs(
                current_model.weights(),
                &current_model.descriptor().config,
                input_ids.as_slice(),
            )?;
            step_profile.embedding_gather_ms = step_profile
                .embedding_gather_ms
                .saturating_add(duration_ms(embed_started));

            let batch_size = input_ids.len();
            let graph = training_graph_for_batch(
                &mut train_graph_cache,
                selected_device.device.clone(),
                current_model.descriptor(),
                batch_size,
                config.geometry.train_sequence_length,
            )?;
            let inputs = bind_parameter_golf_baseline_training_graph_inputs(
                graph,
                &current_model,
                &embedded_inputs,
                target_ids.as_slice(),
            )?;
            let backward_plan = graph.graph.backward_plan(graph.loss_tensor_id)?;
            let retained_graph = retained_forward_graph(graph, &backward_plan);

            let forward_started = Instant::now();
            let forward_outputs = execute_cuda_graph_outputs(
                &mut cuda_backend,
                &retained_graph,
                input_pairs_from_tensors(&inputs)?.as_slice(),
            )?;
            step_profile.forward_loss_cuda_ms = step_profile
                .forward_loss_cuda_ms
                .saturating_add(duration_ms(forward_started));
            step_profile.retained_binding_tensor_count = step_profile
                .retained_binding_tensor_count
                .saturating_add(backward_plan.primal_bindings.len() as u32);
            step_profile.retained_binding_f32_count = step_profile
                .retained_binding_f32_count
                .saturating_add(count_output_elements(
                    &retained_graph,
                    &backward_plan.primal_bindings,
                )?);

            let loss = forward_outputs
                .iter()
                .find(|(tensor_id, _)| *tensor_id == graph.loss_tensor_id)
                .and_then(|(_, values)| values.first().copied())
                .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphOutput {
                    tensor_id: graph.loss_tensor_id,
                })?;
            microbatch_loss_sum += loss;

            let backward_started = Instant::now();
            let backward_outputs = execute_backward_plan(
                &mut cuda_backend,
                &backward_plan,
                forward_outputs.as_slice(),
            )?;
            step_profile.backward_cuda_ms = step_profile
                .backward_cuda_ms
                .saturating_add(duration_ms(backward_started));
            step_profile.gradient_tensor_count = step_profile
                .gradient_tensor_count
                .saturating_add(backward_plan.gradient_targets.len() as u32);
            step_profile.gradient_f32_count = step_profile
                .gradient_f32_count
                .saturating_add(count_gradient_elements(&backward_plan)?);

            let materialize_started = Instant::now();
            let backward_result =
                backward_result_from_outputs(&backward_plan, backward_outputs.as_slice());
            let gradients = materialize_parameter_golf_baseline_training_gradients(
                graph,
                &backward_result,
                &current_model.descriptor().config,
                input_ids.as_slice(),
            )?;
            step_profile.host_gradient_materialization_ms = step_profile
                .host_gradient_materialization_ms
                .saturating_add(duration_ms(materialize_started));
            accumulate_gradients(
                accumulated_gradients.as_mut_slice(),
                &trainer_state,
                &gradients.parameter_gradients,
                config.geometry.grad_accum_steps as f32,
            )?;
            emit_progress_line(format!(
                "micro_step_complete step={}/{} micro_step={}/{} window_id={} train_loss={:.8} forward_ms={} backward_ms={} host_materialization_ms={} retained_binding_f32={} gradient_f32={}",
                step_index + 1,
                config.max_steps,
                micro_step + 1,
                config.geometry.grad_accum_steps,
                window.window_id,
                loss,
                step_profile.forward_loss_cuda_ms,
                step_profile.backward_cuda_ms,
                step_profile.host_materialization_ms(),
                step_profile.retained_binding_f32_count,
                step_profile.gradient_f32_count,
            ));
        }

        clip_gradients(
            accumulated_gradients.as_mut_slice(),
            config.hyperparameters.grad_clip_norm,
        );
        let optimizer_started = Instant::now();
        let learning_rate_multiplier = config.hyperparameters.learning_rate_multiplier(
            step_index,
            aggregate_phase_timings.forward_loss_cuda_ms as f32,
        );
        let muon_momentum = config.hyperparameters.muon_momentum_at_step(step_index);
        apply_gradients_to_state(
            &mut trainer_state,
            accumulated_gradients.as_slice(),
            learning_rate_multiplier,
            muon_momentum,
            step_index + 1,
        )?;
        current_model = materialize_current_model(&initial_model, &trainer_state)?;
        step_profile.optimizer_step_ms = duration_ms(optimizer_started);
        aggregate_phase_timings.accumulate(&step_profile);
        step_metrics.push(ParameterGolfSingleH100TrainingStepMetrics {
            global_step: step_index + 1,
            train_window_ids: window_ids,
            mean_microbatch_loss: microbatch_loss_sum / config.geometry.grad_accum_steps as f32,
            learning_rate_multiplier,
            muon_momentum,
            phase_timings: step_profile,
        });
        let step = step_metrics.last().expect("step metrics should be present");
        emit_progress_line(format!(
            "train_step_complete step={} mean_microbatch_loss={:.8} lr_mult={:.8} muon_momentum={:.8} host_materialization_ms={} optimizer_step_ms={}",
            step.global_step,
            step.mean_microbatch_loss,
            step.learning_rate_multiplier,
            step.muon_momentum,
            step.phase_timings.host_materialization_ms(),
            step.phase_timings.optimizer_step_ms,
        ));
    }

    emit_progress_line(format!(
        "final_validation_start sequences={} batch_sequences={}",
        (validation_tokens.len() - 1) / config.geometry.train_sequence_length,
        config.geometry.local_validation_batch_sequences(),
    ));
    let final_validation = Some(evaluate_validation_on_cuda(
        &mut cuda_backend,
        &selected_device.device,
        current_model.descriptor(),
        &current_model,
        validation_tokens.as_slice(),
        &byte_luts,
        config.geometry.train_sequence_length,
        config.geometry.local_validation_batch_sequences(),
        &mut train_graph_cache,
        "final_validation",
    )?);
    let compressed_model_artifact = export_parameter_golf_int8_zlib_model_artifact(
        &current_model,
        &config.run_id,
        config.max_steps,
    )?;

    let finished_at_ms = unix_time_ms();
    let observed_wallclock_ms = finished_at_ms.saturating_sub(started_at_ms);
    let summary = format!(
        "The bounded Rust-owned single-H100 trainer executed {} optimizer step(s) with challenge single-device geometry on CUDA, emitted final validation loss/bpb directly from the Psionic path, and retained explicit host-materialization timing around the backward replay seam.",
        config.max_steps
    );

    let mut report = ParameterGolfSingleH100TrainingReport {
        schema_version: 1,
        scope_window: String::from("parameter_golf_single_h100_training_v1"),
        run_id: config.run_id.clone(),
        dataset_root: config.dataset_root.clone(),
        tokenizer_path: config.tokenizer_path.clone(),
        dataset_key: config.dataset_key.clone(),
        variant: config.variant.clone(),
        tokenizer_digest,
        dataset_manifest_digest: bundle.manifest.stable_digest(),
        train_shard_count: bundle.train_shards.len(),
        validation_shard_count: bundle.validation_shards.len(),
        train_token_count: split_token_count(&bundle, PARAMETER_GOLF_TRAIN_SPLIT_NAME),
        validation_token_count: split_token_count(&bundle, "validation"),
        geometry: config.geometry.clone(),
        hyperparameters: config.hyperparameters.clone(),
        max_steps: config.max_steps,
        executed_steps: config.max_steps,
        delivered_execution,
        machine_thresholds: machine_observation.thresholds.clone(),
        observed_cuda_health: machine_observation.observed_cuda_health.clone(),
        cuda_discovery_error: machine_observation.cuda_discovery_error.clone(),
        observed_cuda_devices: machine_observation.observed_cuda_devices.clone(),
        matching_h100_device_count: machine_observation.matching_h100_device_count,
        machine_contract_satisfied: machine_observation.machine_contract_satisfied,
        baseline_model_id: String::from(PARAMETER_GOLF_BASELINE_MODEL_ID),
        baseline_model_revision: String::from(PARAMETER_GOLF_BASELINE_REVISION),
        baseline_model_descriptor_digest: current_model.descriptor().stable_digest(),
        optimizer_plan_digest,
        cuda_training_capability_report_digest: capability_report.report_digest.clone(),
        challenge_kernel_blockers: capability_report.challenge_kernel_blockers().to_vec(),
        initial_validation,
        final_validation,
        compressed_model_bytes: Some(compressed_model_artifact.bytes.len() as u64),
        compressed_model_artifact_ref: Some(compressed_model_artifact.artifact_ref.clone()),
        compressed_model_artifact_digest: Some(compressed_model_artifact.artifact_digest.clone()),
        step_metrics,
        aggregate_phase_timings: Some(aggregate_phase_timings),
        final_training_cursor: Some(cursor),
        started_at_ms,
        finished_at_ms,
        observed_wallclock_ms,
        disposition: ParameterGolfSingleH100TrainingDisposition::TrainingExecuted,
        refusal: None,
        claim_boundary: String::from(
            "bounded_single_h100_accelerated_trainer; challenge geometry and optimizer contract are real, but this report does not claim 8xH100, record-track, or challenge-speed closure",
        ),
        summary,
        report_digest: String::new(),
    };
    report.report_digest = stable_digest(
        b"psionic_parameter_golf_single_h100_training_report|",
        &report_without_digest(&report),
    );
    Ok(report)
}

fn refusal_report(
    config: &ParameterGolfSingleH100TrainingConfig,
    tokenizer_digest: psionic_data::TokenizerDigest,
    bundle: &ParameterGolfDatasetBundle,
    machine_observation: &ParameterGolfSingleH100MachineObservation,
    initial_model: &ParameterGolfReferenceModel,
    optimizer_plan_digest: String,
    capability_report_digest: String,
    challenge_kernel_blockers: Vec<String>,
    disposition: ParameterGolfSingleH100TrainingDisposition,
    refusal: Option<PsionicRefusal>,
    started_at_ms: u64,
    summary: String,
) -> ParameterGolfSingleH100TrainingReport {
    let finished_at_ms = unix_time_ms();
    let mut report = ParameterGolfSingleH100TrainingReport {
        schema_version: 1,
        scope_window: String::from("parameter_golf_single_h100_training_v1"),
        run_id: config.run_id.clone(),
        dataset_root: config.dataset_root.clone(),
        tokenizer_path: config.tokenizer_path.clone(),
        dataset_key: config.dataset_key.clone(),
        variant: config.variant.clone(),
        tokenizer_digest,
        dataset_manifest_digest: bundle.manifest.stable_digest(),
        train_shard_count: bundle.train_shards.len(),
        validation_shard_count: bundle.validation_shards.len(),
        train_token_count: split_token_count(bundle, PARAMETER_GOLF_TRAIN_SPLIT_NAME),
        validation_token_count: split_token_count(bundle, "validation"),
        geometry: config.geometry.clone(),
        hyperparameters: config.hyperparameters.clone(),
        max_steps: config.max_steps,
        executed_steps: 0,
        delivered_execution: DeliveredExecutionContext::new("cuda", None, Vec::new()),
        machine_thresholds: machine_observation.thresholds.clone(),
        observed_cuda_health: machine_observation.observed_cuda_health.clone(),
        cuda_discovery_error: machine_observation.cuda_discovery_error.clone(),
        observed_cuda_devices: machine_observation.observed_cuda_devices.clone(),
        matching_h100_device_count: machine_observation.matching_h100_device_count,
        machine_contract_satisfied: machine_observation.machine_contract_satisfied,
        baseline_model_id: String::from(PARAMETER_GOLF_BASELINE_MODEL_ID),
        baseline_model_revision: String::from(PARAMETER_GOLF_BASELINE_REVISION),
        baseline_model_descriptor_digest: initial_model.descriptor().stable_digest(),
        optimizer_plan_digest,
        cuda_training_capability_report_digest: capability_report_digest,
        challenge_kernel_blockers,
        initial_validation: None,
        final_validation: None,
        compressed_model_bytes: None,
        compressed_model_artifact_ref: None,
        compressed_model_artifact_digest: None,
        step_metrics: Vec::new(),
        aggregate_phase_timings: None,
        final_training_cursor: None,
        started_at_ms,
        finished_at_ms,
        observed_wallclock_ms: finished_at_ms.saturating_sub(started_at_ms),
        disposition,
        refusal,
        claim_boundary: String::from(
            "single_h100_accelerated_trainer_refusal; dataset and model contracts are bound, but no training artifact is claimed when machine or CUDA blocker admission fails",
        ),
        summary,
        report_digest: String::new(),
    };
    report.report_digest = stable_digest(
        b"psionic_parameter_golf_single_h100_training_report|",
        &report_without_digest(&report),
    );
    report
}

fn seed_parameter_states(
    model: &ParameterGolfReferenceModel,
    optimizer_plan: &crate::ParameterGolfOptimizerPlan,
) -> Result<ParameterGolfSingleH100TrainerState, ParameterGolfSingleH100TrainingError> {
    let parameter_vectors = model
        .weights()
        .parameter_vectors(&model.descriptor().config)
        .into_iter()
        .map(|vector| (vector.parameter_id.clone(), vector))
        .collect::<BTreeMap<_, _>>();
    let mut parameter_states = BTreeMap::new();
    for group in &optimizer_plan.groups {
        for parameter_id in &group.tensor_names {
            let vector = parameter_vectors.get(parameter_id).ok_or_else(|| {
                ParameterGolfSingleH100TrainingError::MissingParameterState {
                    parameter_id: parameter_id.clone(),
                }
            })?;
            let state = match &group.execution {
                ParameterGolfOptimizerExecution::Adam { optimizer } => {
                    ParameterGolfParameterState::Adam {
                        shape: vector.shape.dims().to_vec(),
                        values: vector.values.clone(),
                        optimizer: optimizer.clone(),
                        optimizer_state: optimizer.initialize_state(vector.values.len()),
                    }
                }
                ParameterGolfOptimizerExecution::Muon { optimizer } => {
                    let rows = vector.shape.dims().first().copied().unwrap_or(0);
                    let cols = vector.shape.dims().get(1).copied().unwrap_or(0);
                    ParameterGolfParameterState::Muon {
                        shape: vector.shape.dims().to_vec(),
                        values: vector.values.clone(),
                        optimizer: optimizer.clone(),
                        optimizer_state: crate::ParameterGolfMuonState::zeros(rows, cols),
                    }
                }
            };
            parameter_states.insert(parameter_id.clone(), state);
        }
    }
    Ok(ParameterGolfSingleH100TrainerState { parameter_states })
}

fn zero_gradients(state: &ParameterGolfSingleH100TrainerState) -> Vec<(String, Vec<f32>)> {
    state
        .parameter_states
        .iter()
        .map(|(parameter_id, state)| (parameter_id.clone(), vec![0.0; state.values().len()]))
        .collect()
}

fn accumulate_gradients(
    accumulated: &mut [(String, Vec<f32>)],
    trainer_state: &ParameterGolfSingleH100TrainerState,
    gradients: &BTreeMap<String, Vec<f32>>,
    divisor: f32,
) -> Result<(), ParameterGolfSingleH100TrainingError> {
    for (parameter_id, values) in accumulated {
        let gradient = gradients.get(parameter_id).ok_or_else(|| {
            ParameterGolfSingleH100TrainingError::MissingParameterState {
                parameter_id: parameter_id.clone(),
            }
        })?;
        let expected_len = trainer_state
            .parameter_states
            .get(parameter_id)
            .ok_or_else(
                || ParameterGolfSingleH100TrainingError::MissingParameterState {
                    parameter_id: parameter_id.clone(),
                },
            )?
            .values()
            .len();
        if gradient.len() != expected_len {
            return Err(ParameterGolfSingleH100TrainingError::Serialization {
                message: format!(
                    "gradient length mismatch for `{parameter_id}`: expected {expected_len}, found {}",
                    gradient.len()
                ),
            });
        }
        for (accumulated, gradient) in values.iter_mut().zip(gradient.iter()) {
            *accumulated += *gradient / divisor;
        }
    }
    Ok(())
}

fn apply_gradients_to_state(
    state: &mut ParameterGolfSingleH100TrainerState,
    gradients: &[(String, Vec<f32>)],
    learning_rate_multiplier: f32,
    muon_momentum: f32,
    step_number: u64,
) -> Result<(), ParameterGolfSingleH100TrainingError> {
    for (parameter_id, gradient_values) in gradients {
        let parameter_state = state
            .parameter_states
            .get_mut(parameter_id)
            .ok_or_else(
                || ParameterGolfSingleH100TrainingError::MissingParameterState {
                    parameter_id: parameter_id.clone(),
                },
            )?;
        parameter_state.apply_gradients(
            gradient_values.as_slice(),
            learning_rate_multiplier,
            muon_momentum,
            step_number,
        )?;
    }
    Ok(())
}

fn materialize_current_model(
    baseline: &ParameterGolfReferenceModel,
    state: &ParameterGolfSingleH100TrainerState,
) -> Result<ParameterGolfReferenceModel, ParameterGolfSingleH100TrainingError> {
    let overrides = state
        .parameter_states
        .iter()
        .map(|(parameter_id, state)| (parameter_id.clone(), state.values().to_vec()))
        .collect::<BTreeMap<_, _>>();
    let weights = baseline
        .weights()
        .with_parameter_overrides(&baseline.descriptor().config, &overrides)?;
    Ok(ParameterGolfReferenceModel::new(
        baseline.descriptor().model.clone(),
        baseline.descriptor().config.clone(),
        weights,
    )?)
}

fn evaluate_validation_on_cuda(
    cuda_backend: &mut CudaBackend,
    device: &psionic_core::Device,
    descriptor: &psionic_models::ParameterGolfModelDescriptor,
    model: &ParameterGolfReferenceModel,
    validation_tokens: &[u16],
    byte_luts: &ParameterGolfSentencePieceByteLuts,
    sequence_length: usize,
    batch_sequences: usize,
    graph_cache: &mut BTreeMap<usize, ParameterGolfBaselineTrainingGraph>,
    stage_label: &str,
) -> Result<ParameterGolfSingleH100ValidationSummary, ParameterGolfSingleH100TrainingError> {
    if validation_tokens.len() <= sequence_length {
        return Err(ParameterGolfSingleH100TrainingError::InvalidConfig {
            message: String::from(
                "validation split is too short for the requested sequence length",
            ),
        });
    }
    let total_sequences = (validation_tokens.len() - 1) / sequence_length;
    let mut total_loss_sum = 0.0_f64;
    let mut total_token_count = 0_u64;
    let mut total_byte_count = 0_u64;
    let validation_batch_sequences = batch_sequences.max(1);
    let total_batches = total_sequences.div_ceil(validation_batch_sequences);
    let validation_started = Instant::now();

    for (batch_index, batch_start) in (0..total_sequences)
        .step_by(validation_batch_sequences)
        .enumerate()
    {
        let batch_end = (batch_start + validation_batch_sequences).min(total_sequences);
        let raw_start = batch_start * sequence_length;
        let raw_end = batch_end * sequence_length + 1;
        let local = &validation_tokens[raw_start..raw_end];
        let input_ids = local[..local.len() - 1]
            .chunks(sequence_length)
            .map(|row| {
                row.iter()
                    .map(|token| u32::from(*token))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let target_ids = local[1..]
            .chunks(sequence_length)
            .map(|row| {
                row.iter()
                    .map(|token| u32::from(*token))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let graph = training_graph_for_batch(
            graph_cache,
            device.clone(),
            descriptor,
            input_ids.len(),
            sequence_length,
        )?;
        let embedded_inputs = crate::gather_parameter_golf_embedded_inputs(
            model.weights(),
            &model.descriptor().config,
            input_ids.as_slice(),
        )?;
        let inputs = bind_parameter_golf_baseline_training_graph_inputs(
            graph,
            model,
            &embedded_inputs,
            target_ids.as_slice(),
        )?;
        let outputs = execute_cuda_graph_outputs(
            cuda_backend,
            graph.graph.graph(),
            input_pairs_from_tensors(&inputs)?.as_slice(),
        )?;
        let batch_loss = outputs
            .iter()
            .find(|(tensor_id, _)| *tensor_id == graph.loss_tensor_id)
            .and_then(|(_, values)| values.first().copied())
            .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphOutput {
                tensor_id: graph.loss_tensor_id,
            })?;
        let batch_token_count = target_ids.iter().map(Vec::len).sum::<usize>() as u64;
        total_loss_sum += f64::from(batch_loss) * batch_token_count as f64;
        total_token_count = total_token_count.saturating_add(batch_token_count);
        total_byte_count = total_byte_count.saturating_add(
            byte_luts.count_target_bytes(
                input_ids
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect::<Vec<_>>()
                    .as_slice(),
                target_ids
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?,
        );
        if batch_index == 0 || (batch_index + 1) % 32 == 0 || batch_index + 1 == total_batches {
            emit_progress_line(format!(
                "validation_progress stage={} batch={}/{} sequences={} tokens={} elapsed_ms={}",
                stage_label,
                batch_index + 1,
                total_batches,
                batch_end,
                total_token_count,
                duration_ms(validation_started),
            ));
        }
    }

    let mean_loss = total_loss_sum / total_token_count.max(1) as f64;
    let bits_per_byte = (mean_loss / std::f64::consts::LN_2)
        * (total_token_count as f64 / total_byte_count.max(1) as f64);
    emit_progress_line(format!(
        "validation_complete stage={} mean_loss={:.8} val_bpb={:.8} evaluated_tokens={} evaluated_bytes={} elapsed_ms={}",
        stage_label,
        mean_loss,
        bits_per_byte,
        total_token_count,
        total_byte_count,
        duration_ms(validation_started),
    ));
    Ok(ParameterGolfSingleH100ValidationSummary {
        evaluated_sequence_count: total_sequences,
        evaluated_token_count: total_token_count,
        evaluated_byte_count: total_byte_count,
        mean_loss,
        bits_per_byte,
    })
}

fn training_graph_for_batch<'a>(
    cache: &'a mut BTreeMap<usize, ParameterGolfBaselineTrainingGraph>,
    device: psionic_core::Device,
    descriptor: &psionic_models::ParameterGolfModelDescriptor,
    batch_size: usize,
    sequence_length: usize,
) -> Result<&'a ParameterGolfBaselineTrainingGraph, ParameterGolfSingleH100TrainingError> {
    if !cache.contains_key(&batch_size) {
        let graph = build_parameter_golf_baseline_training_graph(
            device,
            descriptor,
            batch_size,
            sequence_length,
        )?;
        cache.insert(batch_size, graph);
    }
    cache
        .get(&batch_size)
        .ok_or_else(|| ParameterGolfSingleH100TrainingError::Serialization {
            message: format!("missing cached training graph for batch_size={batch_size}"),
        })
}

fn retained_forward_graph(
    graph: &ParameterGolfBaselineTrainingGraph,
    backward_plan: &psionic_ir::AutodiffBackwardPlan,
) -> psionic_ir::Graph {
    let mut outputs = vec![graph.loss_tensor_id];
    outputs.extend(
        backward_plan
            .primal_bindings
            .iter()
            .map(|binding| binding.primal_tensor),
    );
    graph.graph.graph().with_outputs(unique_tensor_ids(outputs))
}

fn execute_backward_plan(
    cuda_backend: &mut CudaBackend,
    backward_plan: &psionic_ir::AutodiffBackwardPlan,
    forward_outputs: &[(TensorId, Vec<f32>)],
) -> Result<Vec<(TensorId, Vec<f32>)>, ParameterGolfSingleH100TrainingError> {
    let forward_map = forward_outputs
        .iter()
        .map(|(tensor_id, values)| (*tensor_id, values.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut inputs = Vec::new();
    for binding in &backward_plan.primal_bindings {
        let values = forward_map.get(&binding.primal_tensor).ok_or(
            ParameterGolfSingleH100TrainingError::MissingGraphOutput {
                tensor_id: binding.primal_tensor,
            },
        )?;
        inputs.push((binding.gradient_graph_input, values.clone()));
    }
    inputs.push((backward_plan.seed_input, vec![1.0_f32]));
    execute_cuda_graph_outputs(
        cuda_backend,
        &backward_plan.gradient_graph,
        inputs.as_slice(),
    )
}

fn backward_result_from_outputs(
    backward_plan: &psionic_ir::AutodiffBackwardPlan,
    outputs: &[(TensorId, Vec<f32>)],
) -> AutodiffBackwardResult {
    let gradient_targets = backward_plan
        .gradient_targets
        .iter()
        .map(|target| (target.gradient_tensor, target.primal_tensor))
        .collect::<BTreeMap<_, _>>();
    AutodiffBackwardResult {
        forward_values: BTreeMap::new(),
        plan: backward_plan.clone(),
        gradients: outputs
            .iter()
            .filter_map(|(tensor_id, values)| {
                gradient_targets
                    .get(tensor_id)
                    .copied()
                    .map(|primal_tensor| (primal_tensor, TensorData::F32(values.clone())))
            })
            .collect(),
    }
}

fn execute_cuda_graph_outputs(
    cuda_backend: &mut CudaBackend,
    graph: &psionic_ir::Graph,
    inputs: &[(TensorId, Vec<f32>)],
) -> Result<Vec<(TensorId, Vec<f32>)>, ParameterGolfSingleH100TrainingError> {
    let mut buffers = BTreeMap::new();
    for (tensor_id, values) in inputs {
        let shape = graph
            .node(*tensor_id)
            .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphTensor {
                tensor_id: *tensor_id,
            })?
            .tensor()
            .spec()
            .shape()
            .clone();
        buffers.insert(
            *tensor_id,
            cuda_backend.input_buffer(shape, values.clone())?,
        );
    }
    let result = cuda_backend.compile_and_execute(graph, &buffers)?;
    graph
        .outputs()
        .iter()
        .map(|tensor_id| {
            let values = result
                .outputs
                .get(tensor_id)
                .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphOutput {
                    tensor_id: *tensor_id,
                })?
                .read_f32()?;
            Ok((*tensor_id, values))
        })
        .collect()
}

fn input_pairs_from_tensors(
    inputs: &BTreeMap<TensorId, TensorData>,
) -> Result<Vec<(TensorId, Vec<f32>)>, ParameterGolfSingleH100TrainingError> {
    inputs
        .iter()
        .map(|(tensor_id, data)| match data {
            TensorData::F32(values) => Ok((*tensor_id, values.clone())),
            _ => Err(ParameterGolfSingleH100TrainingError::Serialization {
                message: format!("expected dense f32 input buffer for tensor {tensor_id}"),
            }),
        })
        .collect()
}

fn split_token_count(bundle: &ParameterGolfDatasetBundle, split_name: &str) -> u64 {
    bundle
        .manifest
        .split(split_name)
        .map(|split| split.shards.iter().map(|shard| shard.token_count).sum())
        .unwrap_or(0)
}

fn count_output_elements(
    retained_graph: &psionic_ir::Graph,
    bindings: &[psionic_ir::AutodiffPrimalBinding],
) -> Result<u64, ParameterGolfSingleH100TrainingError> {
    bindings.iter().try_fold(0_u64, |sum, binding| {
        let node = retained_graph.node(binding.primal_tensor).ok_or(
            ParameterGolfSingleH100TrainingError::MissingGraphTensor {
                tensor_id: binding.primal_tensor,
            },
        )?;
        Ok(sum.saturating_add(node.tensor().spec().shape().element_count() as u64))
    })
}

fn count_gradient_elements(
    backward_plan: &psionic_ir::AutodiffBackwardPlan,
) -> Result<u64, ParameterGolfSingleH100TrainingError> {
    backward_plan
        .gradient_targets
        .iter()
        .try_fold(0_u64, |sum, target| {
            let node = backward_plan
                .gradient_graph
                .node(target.gradient_tensor)
                .ok_or(ParameterGolfSingleH100TrainingError::MissingGraphTensor {
                    tensor_id: target.gradient_tensor,
                })?;
            Ok(sum.saturating_add(node.tensor().spec().shape().element_count() as u64))
        })
}

fn clip_gradients(gradients: &mut [(String, Vec<f32>)], max_norm: f32) {
    if !(max_norm.is_finite() && max_norm > 0.0) {
        return;
    }
    let norm = gradients
        .iter()
        .flat_map(|(_, values)| values.iter())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= max_norm || norm <= f32::EPSILON {
        return;
    }
    let scale = max_norm / norm;
    for (_, values) in gradients {
        for value in values {
            *value *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use psionic_core::{DType, Device, Shape};
    use psionic_ir::{AutodiffContext, AutodiffGraphBuilder};

    use super::*;

    #[test]
    fn backward_result_from_outputs_rekeys_gradient_outputs_to_primal_tensors(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut builder =
            AutodiffGraphBuilder::with_context(Device::cpu(), AutodiffContext::training());
        let input = builder.input("x", Shape::new(vec![1, 1]), DType::F32, true);
        let squared = builder.mul(&input, &input)?;
        let graph = builder.finish(vec![squared.clone()]);
        let backward_plan = graph.backward_plan(squared.id())?;
        let gradient_tensor_id = backward_plan
            .gradient_for(input.id())
            .ok_or("missing input gradient target")?;

        let backward_result =
            backward_result_from_outputs(&backward_plan, &[(gradient_tensor_id, vec![42.0_f32])]);

        assert_eq!(
            backward_result
                .gradient(input.id())
                .and_then(TensorData::as_f32_slice),
            Some(&[42.0_f32][..])
        );
        Ok(())
    }
}

fn unique_tensor_ids(ids: Vec<TensorId>) -> Vec<TensorId> {
    let mut seen = std::collections::BTreeSet::new();
    ids.into_iter().filter(|id| seen.insert(*id)).collect()
}

fn duration_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_millis() as u64
}

fn emit_progress_line(message: String) {
    let mut stdout = io::stdout().lock();
    let _ = writeln!(stdout, "{message}");
    let _ = stdout.flush();
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).expect("digest serialization should succeed"));
    hex::encode(hasher.finalize())
}

fn report_without_digest(
    report: &ParameterGolfSingleH100TrainingReport,
) -> ParameterGolfSingleH100TrainingReport {
    let mut canonical = report.clone();
    canonical.report_digest.clear();
    canonical
}

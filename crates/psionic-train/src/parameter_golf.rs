use half::bf16;
use psionic_models::ParameterGolfModelDescriptor;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{TrainingOptimizerConfig, TrainingParameterClass};

/// Stable control-tensor patterns copied from the current public `train_gpt.py`.
pub const PARAMETER_GOLF_CONTROL_TENSOR_NAME_PATTERNS: &[&str] = &[
    "attn_scale",
    "attn_scales",
    "mlp_scale",
    "mlp_scales",
    "resid_mix",
    "resid_mixes",
    "q_gain",
    "skip_weight",
    "skip_weights",
];

/// Baseline optimization and schedule defaults copied from the public challenge script.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfTrainingHyperparameters {
    /// Training-iteration budget.
    pub iterations: u64,
    /// Warmdown-iteration window.
    pub warmdown_iters: u64,
    /// Optional wallclock cap in seconds. `None` keeps the iteration-only warmdown path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_wallclock_seconds: Option<f32>,
    /// Token-embedding Adam LR when embeddings are not tied.
    pub embed_lr: f32,
    /// Untied LM-head Adam LR.
    pub head_lr: f32,
    /// Tied-token-embedding Adam LR.
    pub tied_embed_lr: f32,
    /// Matrix Muon LR.
    pub matrix_lr: f32,
    /// Scalar/control Adam LR.
    pub scalar_lr: f32,
    /// Muon momentum target after warmup.
    pub muon_momentum: f32,
    /// Muon Newton-Schulz backend steps.
    pub muon_backend_steps: u32,
    /// Muon momentum value at step `0`.
    pub muon_momentum_warmup_start: f32,
    /// Muon momentum warmup length in steps.
    pub muon_momentum_warmup_steps: u64,
    /// Adam beta1.
    pub beta1: f32,
    /// Adam beta2.
    pub beta2: f32,
    /// Adam epsilon.
    pub adam_eps: f32,
    /// Optional global grad clip norm. Zero means disabled in the public script.
    pub grad_clip_norm: f32,
}

impl ParameterGolfTrainingHyperparameters {
    /// Returns the current public baseline defaults.
    #[must_use]
    pub fn baseline_defaults() -> Self {
        Self {
            iterations: 20_000,
            warmdown_iters: 1_200,
            max_wallclock_seconds: Some(600.0),
            embed_lr: 0.6,
            head_lr: 0.008,
            tied_embed_lr: 0.05,
            matrix_lr: 0.04,
            scalar_lr: 0.04,
            muon_momentum: 0.95,
            muon_backend_steps: 5,
            muon_momentum_warmup_start: 0.85,
            muon_momentum_warmup_steps: 500,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            grad_clip_norm: 0.0,
        }
    }

    /// Returns the public token-embedding learning rate for one tie-embedding posture.
    #[must_use]
    pub fn token_learning_rate(&self, tie_embeddings: bool) -> f32 {
        if tie_embeddings {
            self.tied_embed_lr
        } else {
            self.embed_lr
        }
    }

    /// Returns the current Muon momentum with the public linear warmup applied.
    #[must_use]
    pub fn muon_momentum_at_step(&self, step: u64) -> f32 {
        if self.muon_momentum_warmup_steps == 0 {
            return self.muon_momentum;
        }
        let fraction = (step as f32 / self.muon_momentum_warmup_steps as f32).min(1.0);
        ((1.0 - fraction) * self.muon_momentum_warmup_start)
            + (fraction * self.muon_momentum)
    }

    /// Returns the public warmdown multiplier for one step and elapsed wallclock.
    #[must_use]
    pub fn learning_rate_multiplier(&self, step: u64, elapsed_ms: f32) -> f32 {
        if self.warmdown_iters == 0 {
            return 1.0;
        }
        if let Some(max_wallclock_seconds) = self.max_wallclock_seconds {
            if max_wallclock_seconds > 0.0 {
                let max_wallclock_ms = 1000.0 * max_wallclock_seconds;
                let step_ms = elapsed_ms / step.max(1) as f32;
                let warmdown_ms = self.warmdown_iters as f32 * step_ms;
                let remaining_ms = (max_wallclock_ms - elapsed_ms).max(0.0);
                if remaining_ms <= warmdown_ms {
                    return remaining_ms / warmdown_ms.max(1e-9);
                }
                return 1.0;
            }
        }
        let warmdown_start = self.iterations.saturating_sub(self.warmdown_iters);
        if warmdown_start <= step && step < self.iterations {
            (self.iterations - step) as f32 / self.warmdown_iters.max(1) as f32
        } else {
            1.0
        }
    }
}

/// Optimizer family admitted by the Parameter Golf lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfOptimizerGroupKind {
    /// Token embedding handled by Adam.
    TokenEmbeddingAdam,
    /// Optional untied LM head handled by Adam.
    UntiedLmHeadAdam,
    /// Matrix-shaped block tensors handled by Muon.
    MatrixMuon,
    /// Scalar/control tensors handled by Adam.
    ScalarControlAdam,
}

/// Lane-specific execution config for one optimizer group.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum ParameterGolfOptimizerExecution {
    /// Adam-family execution reusing the generic training optimizer config.
    Adam {
        /// Reusable Adam config for the group.
        optimizer: TrainingOptimizerConfig,
    },
    /// Muon execution for matrix-shaped transformer parameters.
    Muon {
        /// Muon config for the group.
        optimizer: ParameterGolfMuonConfig,
    },
}

/// One optimizer-group plan for the Parameter Golf baseline lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfOptimizerGroupPlan {
    /// Stable group identifier.
    pub group_id: String,
    /// Lane-specific group family.
    pub kind: ParameterGolfOptimizerGroupKind,
    /// Shared training-core parameter class.
    pub parameter_class: TrainingParameterClass,
    /// Tensor names assigned to the group in deterministic order.
    pub tensor_names: Vec<String>,
    /// Total parameters assigned to the group.
    pub parameter_count: usize,
    /// Execution config for the group.
    pub execution: ParameterGolfOptimizerExecution,
}

/// Complete optimizer split for one Parameter Golf model descriptor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfOptimizerPlan {
    /// Stable model id this plan belongs to.
    pub model_id: String,
    /// Public control-tensor patterns used to classify block scalars.
    pub control_tensor_name_patterns: Vec<String>,
    /// Optional global grad clip norm for the lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_grad_clip_norm: Option<f32>,
    /// Ordered optimizer groups.
    pub groups: Vec<ParameterGolfOptimizerGroupPlan>,
    /// Total parameter count covered by the plan.
    pub total_parameter_count: usize,
}

/// Lane-specific optimizer planning failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ParameterGolfTrainError {
    /// One tensor name could not be classified into the public optimizer split.
    #[error("parameter golf optimizer split does not know how to classify tensor `{tensor_name}`")]
    UnknownTensorClassification {
        /// Unknown tensor name.
        tensor_name: String,
    },
    /// A Muon step received mismatched parameter and gradient lengths.
    #[error(
        "parameter golf Muon expected gradient length {parameter_len} but found {gradient_len}"
    )]
    MuonGradientLengthMismatch {
        /// Parameter element count.
        parameter_len: usize,
        /// Gradient element count.
        gradient_len: usize,
    },
    /// A Muon state carried the wrong momentum-buffer length.
    #[error(
        "parameter golf Muon momentum buffer length mismatch: expected {expected_len}, found {actual_len}"
    )]
    MuonMomentumBufferLengthMismatch {
        /// Expected element count.
        expected_len: usize,
        /// Actual element count.
        actual_len: usize,
    },
    /// Muon only supports matrix-shaped tensors.
    #[error("parameter golf Muon requires a 2D matrix shape, found {shape:?}")]
    MuonRequiresMatrix {
        /// Invalid tensor shape.
        shape: Vec<usize>,
    },
}

/// Builds the current public optimizer split for one Parameter Golf descriptor.
pub fn parameter_golf_optimizer_plan(
    descriptor: &ParameterGolfModelDescriptor,
    hyperparameters: &ParameterGolfTrainingHyperparameters,
) -> Result<ParameterGolfOptimizerPlan, ParameterGolfTrainError> {
    let token_lr = hyperparameters.token_learning_rate(descriptor.config.tie_embeddings);
    let mut token_names = Vec::new();
    let mut head_names = Vec::new();
    let mut matrix_names = Vec::new();
    let mut scalar_names = Vec::new();
    let mut token_parameter_count = 0_usize;
    let mut head_parameter_count = 0_usize;
    let mut matrix_parameter_count = 0_usize;
    let mut scalar_parameter_count = 0_usize;

    for tensor in &descriptor.weights.tensors {
        let name = tensor.name.as_str();
        let parameter_count = tensor.element_count();
        if name == "tok_emb.weight" {
            token_names.push(String::from(name));
            token_parameter_count += parameter_count;
        } else if name == "lm_head.weight" {
            head_names.push(String::from(name));
            head_parameter_count += parameter_count;
        } else if name == "skip_weights"
            || (name.starts_with("blocks.")
                && (tensor.shape.dims().len() < 2 || is_control_tensor_name(name)))
        {
            scalar_names.push(String::from(name));
            scalar_parameter_count += parameter_count;
        } else if name.starts_with("blocks.")
            && tensor.shape.dims().len() == 2
            && !is_control_tensor_name(name)
        {
            matrix_names.push(String::from(name));
            matrix_parameter_count += parameter_count;
        } else {
            return Err(ParameterGolfTrainError::UnknownTensorClassification {
                tensor_name: String::from(name),
            });
        }
    }

    token_names.sort();
    head_names.sort();
    matrix_names.sort();
    scalar_names.sort();

    let mut groups = Vec::new();
    groups.push(ParameterGolfOptimizerGroupPlan {
        group_id: String::from("parameter_golf.token_embedding"),
        kind: ParameterGolfOptimizerGroupKind::TokenEmbeddingAdam,
        parameter_class: TrainingParameterClass::Embedding,
        tensor_names: token_names,
        parameter_count: token_parameter_count,
        execution: ParameterGolfOptimizerExecution::Adam {
            optimizer: TrainingOptimizerConfig::adam(
                token_lr,
                hyperparameters.beta1,
                hyperparameters.beta2,
                hyperparameters.adam_eps,
            ),
        },
    });
    if !head_names.is_empty() {
        groups.push(ParameterGolfOptimizerGroupPlan {
            group_id: String::from("parameter_golf.untied_lm_head"),
            kind: ParameterGolfOptimizerGroupKind::UntiedLmHeadAdam,
            parameter_class: TrainingParameterClass::Head,
            tensor_names: head_names,
            parameter_count: head_parameter_count,
            execution: ParameterGolfOptimizerExecution::Adam {
                optimizer: TrainingOptimizerConfig::adam(
                    hyperparameters.head_lr,
                    hyperparameters.beta1,
                    hyperparameters.beta2,
                    hyperparameters.adam_eps,
                ),
            },
        });
    }
    groups.push(ParameterGolfOptimizerGroupPlan {
        group_id: String::from("parameter_golf.matrix_blocks"),
        kind: ParameterGolfOptimizerGroupKind::MatrixMuon,
        parameter_class: TrainingParameterClass::Matrix,
        tensor_names: matrix_names,
        parameter_count: matrix_parameter_count,
        execution: ParameterGolfOptimizerExecution::Muon {
            optimizer: ParameterGolfMuonConfig::new(
                hyperparameters.matrix_lr,
                hyperparameters.muon_momentum,
                hyperparameters.muon_backend_steps,
            ),
        },
    });
    groups.push(ParameterGolfOptimizerGroupPlan {
        group_id: String::from("parameter_golf.scalar_controls"),
        kind: ParameterGolfOptimizerGroupKind::ScalarControlAdam,
        parameter_class: TrainingParameterClass::Scalar,
        tensor_names: scalar_names,
        parameter_count: scalar_parameter_count,
        execution: ParameterGolfOptimizerExecution::Adam {
            optimizer: TrainingOptimizerConfig::adam(
                hyperparameters.scalar_lr,
                hyperparameters.beta1,
                hyperparameters.beta2,
                hyperparameters.adam_eps,
            ),
        },
    });

    Ok(ParameterGolfOptimizerPlan {
        model_id: descriptor.model.model_id.clone(),
        control_tensor_name_patterns: PARAMETER_GOLF_CONTROL_TENSOR_NAME_PATTERNS
            .iter()
            .map(|pattern| String::from(*pattern))
            .collect(),
        global_grad_clip_norm: (hyperparameters.grad_clip_norm > 0.0)
            .then_some(hyperparameters.grad_clip_norm),
        total_parameter_count: groups.iter().map(|group| group.parameter_count).sum(),
        groups,
    })
}

fn is_control_tensor_name(name: &str) -> bool {
    PARAMETER_GOLF_CONTROL_TENSOR_NAME_PATTERNS
        .iter()
        .any(|pattern| name.contains(pattern))
}

/// Exact Muon config used by the Parameter Golf lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfMuonConfig {
    /// Muon learning rate.
    pub learning_rate: f32,
    /// Muon momentum.
    pub momentum: f32,
    /// Newton-Schulz backend steps.
    pub backend_steps: u32,
    /// Numerical epsilon for the normalization prepass.
    pub epsilon: f32,
}

impl ParameterGolfMuonConfig {
    /// Creates one Muon config with the public epsilon.
    #[must_use]
    pub fn new(learning_rate: f32, momentum: f32, backend_steps: u32) -> Self {
        Self {
            learning_rate,
            momentum,
            backend_steps,
            epsilon: 1e-7,
        }
    }
}

/// Mutable Muon state for one matrix-shaped parameter tensor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfMuonState {
    /// Flat momentum buffer in row-major `[rows, cols]` order.
    pub momentum_buffer: Vec<f32>,
}

impl ParameterGolfMuonState {
    /// Creates zeroed Muon state for one matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            momentum_buffer: vec![0.0; rows * cols],
        }
    }
}

/// Inspectable result of one Muon update.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfMuonStepReceipt {
    /// Learning rate used by the update.
    pub learning_rate: f32,
    /// Momentum used by the update.
    pub momentum: f32,
    /// Newton-Schulz backend steps used by the update.
    pub backend_steps: u32,
    /// Scale correction applied after orthogonalization.
    pub scale_correction: f32,
    /// Flat orthogonalized update direction before the LR multiply.
    pub orthogonalized_update: Vec<f32>,
    /// Flat parameter update values applied to the matrix.
    pub update_values: Vec<f32>,
}

/// Applies one exact Muon step over one matrix-shaped parameter tensor.
pub fn apply_parameter_golf_muon_step(
    parameter_values: &mut [f32],
    parameter_shape: &[usize],
    gradient_values: &[f32],
    optimizer: &ParameterGolfMuonConfig,
    optimizer_state: &mut ParameterGolfMuonState,
) -> Result<ParameterGolfMuonStepReceipt, ParameterGolfTrainError> {
    if parameter_shape.len() != 2 {
        return Err(ParameterGolfTrainError::MuonRequiresMatrix {
            shape: parameter_shape.to_vec(),
        });
    }
    if parameter_values.len() != gradient_values.len() {
        return Err(ParameterGolfTrainError::MuonGradientLengthMismatch {
            parameter_len: parameter_values.len(),
            gradient_len: gradient_values.len(),
        });
    }
    if optimizer_state.momentum_buffer.len() != parameter_values.len() {
        return Err(ParameterGolfTrainError::MuonMomentumBufferLengthMismatch {
            expected_len: parameter_values.len(),
            actual_len: optimizer_state.momentum_buffer.len(),
        });
    }
    let rows = parameter_shape[0];
    let cols = parameter_shape[1];
    let mut effective_gradient = vec![0.0_f32; parameter_values.len()];
    for index in 0..parameter_values.len() {
        optimizer_state.momentum_buffer[index] =
            (optimizer.momentum * optimizer_state.momentum_buffer[index]) + gradient_values[index];
        effective_gradient[index] =
            gradient_values[index] + (optimizer.momentum * optimizer_state.momentum_buffer[index]);
    }
    let orthogonalized_update = zeropower_via_newtonschulz5(
        effective_gradient.as_slice(),
        rows,
        cols,
        optimizer.backend_steps as usize,
        optimizer.epsilon,
    )?;
    let scale_correction = (rows as f32 / cols as f32).max(1.0).sqrt();
    let scaled_update = orthogonalized_update
        .iter()
        .map(|value| round_bf16(scale_correction * value))
        .collect::<Vec<_>>();
    let update_values = scaled_update
        .iter()
        .map(|value| optimizer.learning_rate * value)
        .collect::<Vec<_>>();
    for (parameter, update) in parameter_values.iter_mut().zip(update_values.iter()) {
        *parameter -= *update;
    }
    Ok(ParameterGolfMuonStepReceipt {
        learning_rate: optimizer.learning_rate,
        momentum: optimizer.momentum,
        backend_steps: optimizer.backend_steps,
        scale_correction,
        orthogonalized_update,
        update_values,
    })
}

fn zeropower_via_newtonschulz5(
    values: &[f32],
    rows: usize,
    cols: usize,
    steps: usize,
    epsilon: f32,
) -> Result<Vec<f32>, ParameterGolfTrainError> {
    if rows == 0 || cols == 0 {
        return Err(ParameterGolfTrainError::MuonRequiresMatrix {
            shape: vec![rows, cols],
        });
    }
    let mut x = values.iter().map(|value| round_bf16(*value)).collect::<Vec<_>>();
    let norm = round_bf16(norm_l2(x.as_slice()));
    let denom = round_bf16(norm + epsilon);
    for value in &mut x {
        *value = round_bf16(*value / denom);
    }
    let transposed = rows > cols;
    let (mut x, work_rows, work_cols) = if transposed {
        (transpose(x.as_slice(), rows, cols), cols, rows)
    } else {
        (x, rows, cols)
    };
    let (a, b, c) = (3.4445_f32, -4.7750_f32, 2.0315_f32);
    for _ in 0..steps {
        let x_t = transpose(x.as_slice(), work_rows, work_cols);
        let a_matrix = matmul_bf16(x.as_slice(), x_t.as_slice(), work_rows, work_cols, work_rows);
        let c_times_a = a_matrix
            .iter()
            .map(|value| round_bf16(c * value))
            .collect::<Vec<_>>();
        let c_a_matmul_a = matmul_bf16(
            c_times_a.as_slice(),
            a_matrix.as_slice(),
            work_rows,
            work_rows,
            work_rows,
        );
        let mut b_matrix = vec![0.0_f32; a_matrix.len()];
        for index in 0..b_matrix.len() {
            let scaled_a = round_bf16(b * a_matrix[index]);
            b_matrix[index] = round_bf16(scaled_a + c_a_matmul_a[index]);
        }
        let bx = matmul_bf16(
            b_matrix.as_slice(),
            x.as_slice(),
            work_rows,
            work_rows,
            work_cols,
        );
        for index in 0..x.len() {
            let scaled_x = round_bf16(a * x[index]);
            x[index] = round_bf16(scaled_x + bx[index]);
        }
    }
    Ok(if transposed {
        transpose(x.as_slice(), work_rows, work_cols)
    } else {
        x
    })
}

fn matmul_bf16(
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0_f32;
            for k in 0..inner {
                sum += left[row * inner + k] * right[k * cols + col];
            }
            output[row * cols + col] = round_bf16(sum);
        }
    }
    output
}

fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = values[row * cols + col];
        }
    }
    output
}

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn norm_l2(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use psionic_models::ParameterGolfReferenceModel;
    use serde::Deserialize;

    use super::*;

    #[derive(Deserialize)]
    struct OptimizerFixture {
        hyperparameters: ParameterGolfTrainingHyperparameters,
        control_tensor_name_patterns: Vec<String>,
        expected_groups: ExpectedGroups,
        muon_case: MuonCaseFixture,
        schedule_cases: ScheduleFixture,
    }

    #[derive(Deserialize)]
    struct ExpectedGroups {
        token_embedding: Vec<String>,
        matrix: Vec<String>,
        scalar: Vec<String>,
        head: Vec<String>,
        token_learning_rate: f32,
        matrix_learning_rate: f32,
        scalar_learning_rate: f32,
        head_learning_rate: f32,
    }

    #[derive(Deserialize)]
    struct MuonCaseFixture {
        rows: usize,
        cols: usize,
        parameter: Vec<f32>,
        gradient: Vec<f32>,
        learning_rate: f32,
        momentum: f32,
        backend_steps: u32,
        updated_parameter: Vec<f32>,
        momentum_buffer: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct ScheduleFixture {
        muon_momentum_cases: Vec<ScalarCase>,
        lr_multiplier_cases: Vec<LrMultiplierCase>,
    }

    #[derive(Deserialize)]
    struct ScalarCase {
        step: u64,
        expected: f32,
    }

    #[derive(Deserialize)]
    struct LrMultiplierCase {
        step: u64,
        elapsed_ms: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_wallclock_seconds_override: Option<f32>,
        expected: f32,
    }

    fn load_fixture() -> OptimizerFixture {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/parameter_golf/train/parameter_golf_optimizer_fixture.json");
        serde_json::from_slice(&fs::read(path).expect("fixture should exist"))
            .expect("fixture should deserialize")
    }

    #[test]
    fn baseline_optimizer_plan_matches_public_train_gpt_group_split() {
        let fixture = load_fixture();
        let model = ParameterGolfReferenceModel::baseline_fixture(Default::default())
            .expect("baseline model should build");
        let plan = parameter_golf_optimizer_plan(model.descriptor(), &fixture.hyperparameters)
            .expect("optimizer plan should build");
        assert_eq!(
            plan.control_tensor_name_patterns,
            fixture.control_tensor_name_patterns
        );
        let token_group = plan
            .groups
            .iter()
            .find(|group| group.kind == ParameterGolfOptimizerGroupKind::TokenEmbeddingAdam)
            .expect("token group should exist");
        let matrix_group = plan
            .groups
            .iter()
            .find(|group| group.kind == ParameterGolfOptimizerGroupKind::MatrixMuon)
            .expect("matrix group should exist");
        let scalar_group = plan
            .groups
            .iter()
            .find(|group| group.kind == ParameterGolfOptimizerGroupKind::ScalarControlAdam)
            .expect("scalar group should exist");
        assert_eq!(token_group.tensor_names, fixture.expected_groups.token_embedding);
        assert_eq!(matrix_group.tensor_names, fixture.expected_groups.matrix);
        assert_eq!(scalar_group.tensor_names, fixture.expected_groups.scalar);
        assert_eq!(
            fixture.expected_groups.head_learning_rate,
            fixture.hyperparameters.head_lr
        );
        assert!(
            plan.groups
                .iter()
                .filter(|group| group.kind == ParameterGolfOptimizerGroupKind::UntiedLmHeadAdam)
                .all(|group| group.tensor_names == fixture.expected_groups.head)
        );
        match &token_group.execution {
            ParameterGolfOptimizerExecution::Adam { optimizer } => {
                assert_eq!(
                    optimizer.learning_rate,
                    fixture.expected_groups.token_learning_rate
                );
            }
            other => panic!("expected Adam token group, found {other:?}"),
        }
        match &matrix_group.execution {
            ParameterGolfOptimizerExecution::Muon { optimizer } => {
                assert_eq!(
                    optimizer.learning_rate,
                    fixture.expected_groups.matrix_learning_rate
                );
            }
            other => panic!("expected Muon matrix group, found {other:?}"),
        }
        match &scalar_group.execution {
            ParameterGolfOptimizerExecution::Adam { optimizer } => {
                assert_eq!(
                    optimizer.learning_rate,
                    fixture.expected_groups.scalar_learning_rate
                );
            }
            other => panic!("expected Adam scalar group, found {other:?}"),
        }
        assert_eq!(
            plan.total_parameter_count,
            model.descriptor()
                .weights
                .tensors
                .iter()
                .map(psionic_models::WeightTensorMetadata::element_count)
                .sum::<usize>()
        );
    }

    #[test]
    fn muon_step_matches_public_train_gpt_reference_case() {
        let fixture = load_fixture();
        let case = fixture.muon_case;
        let mut parameter = case.parameter.clone();
        let mut state = ParameterGolfMuonState::zeros(case.rows, case.cols);
        let receipt = apply_parameter_golf_muon_step(
            parameter.as_mut_slice(),
            &[case.rows, case.cols],
            case.gradient.as_slice(),
            &ParameterGolfMuonConfig::new(
                case.learning_rate,
                case.momentum,
                case.backend_steps,
            ),
            &mut state,
        )
        .expect("muon step should compute");
        assert_eq!(receipt.backend_steps, case.backend_steps);
        for (index, (expected, actual)) in case
            .updated_parameter
            .iter()
            .zip(parameter.iter())
            .enumerate()
        {
            assert!(
                (expected - actual).abs() < 5e-4,
                "parameter[{index}] expected {expected}, got {actual}"
            );
        }
        for (index, (expected, actual)) in case
            .momentum_buffer
            .iter()
            .zip(state.momentum_buffer.iter())
            .enumerate()
        {
            assert!(
                (expected - actual).abs() < 1e-6,
                "momentum_buffer[{index}] expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn schedule_helpers_match_public_reference_cases() {
        let fixture = load_fixture();
        for case in fixture.schedule_cases.muon_momentum_cases {
            let actual = fixture.hyperparameters.muon_momentum_at_step(case.step);
            assert!((actual - case.expected).abs() < 1e-6);
        }
        for case in fixture.schedule_cases.lr_multiplier_cases {
            let mut hyperparameters = fixture.hyperparameters.clone();
            hyperparameters.max_wallclock_seconds = case.max_wallclock_seconds_override;
            let actual = hyperparameters.learning_rate_multiplier(case.step, case.elapsed_ms);
            assert!((actual - case.expected).abs() < 1e-6);
        }
    }
}

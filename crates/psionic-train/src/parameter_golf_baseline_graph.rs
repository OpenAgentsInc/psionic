use std::collections::BTreeMap;

use psionic_core::{DType, Device, Shape, TensorData, TensorId};
use psionic_ir::{
    AutodiffContext, AutodiffError, AutodiffGraph, AutodiffGraphBuilder, AutodiffTensor,
    GraphError, ReferenceEvaluationError,
};
use psionic_models::{
    ParameterGolfConfig, ParameterGolfConfigError, ParameterGolfExecutionError,
    ParameterGolfModelDescriptor, ParameterGolfModelError, ParameterGolfReferenceModel,
    ParameterGolfTensor3, ParameterGolfTensorError, ParameterGolfWeights,
    PARAMETER_GOLF_DEFAULT_RMS_NORM_EPSILON,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// How one Parameter Golf graph parameter receives gradients on the current
/// Rust-owned baseline lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfBaselineGradientSource {
    /// The parameter is differentiated directly by the lowered graph.
    GraphOnly,
    /// The parameter receives graph gradients plus host-scattered input-embedding
    /// gradients.
    GraphAndInputEmbeddingScatter,
    /// The parameter receives host-scattered input-embedding gradients only.
    InputEmbeddingScatterOnly,
}

/// Stable parameter binding for the lowered Parameter Golf baseline graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfBaselineGraphParameterBinding {
    /// Stable Parameter Golf parameter identifier.
    pub parameter_id: String,
    /// Logical tensor shape.
    pub shape: Shape,
    /// Graph input tensor carrying the parameter value.
    pub graph_input_tensor_id: TensorId,
    /// Honest gradient-source posture for the parameter.
    pub gradient_source: ParameterGolfBaselineGradientSource,
}

/// Lowered Parameter Golf baseline graph plus machine-readable bindings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfBaselineGraph {
    /// The lowered autodiff graph.
    pub graph: AutodiffGraph,
    /// Non-trainable dense embedded-input tensor consumed by the graph.
    pub embedded_input_tensor_id: TensorId,
    /// Final pre-softcap logits tensor emitted by the graph.
    pub pre_softcap_logits_tensor_id: TensorId,
    /// Parameter bindings in deterministic order.
    pub parameter_bindings: Vec<ParameterGolfBaselineGraphParameterBinding>,
}

impl ParameterGolfBaselineGraph {
    /// Looks up one parameter binding by stable parameter id.
    #[must_use]
    pub fn parameter_binding(
        &self,
        parameter_id: &str,
    ) -> Option<&ParameterGolfBaselineGraphParameterBinding> {
        self.parameter_bindings
            .iter()
            .find(|binding| binding.parameter_id == parameter_id)
    }
}

/// Host-owned logits post-processing and seed materialized from one pre-softcap
/// graph output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfProjectionSeed {
    /// Softcapped logits matching the public reference model.
    pub softcapped_logits: ParameterGolfTensor3,
    /// Mean cross-entropy over the supplied targets.
    pub mean_loss: f32,
    /// Seed gradient with respect to the graph's pre-softcap logits.
    pub pre_softcap_gradient: ParameterGolfTensor3,
}

/// Parameter-gradient materialization for one bounded Parameter Golf baseline
/// graph replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfBaselineGradientBundle {
    /// Stable parameter gradients keyed by Parameter Golf tensor id.
    pub parameter_gradients: BTreeMap<String, Vec<f32>>,
}

/// Error returned while lowering or replaying the Parameter Golf baseline graph.
#[derive(Debug, Error)]
pub enum ParameterGolfBaselineGraphError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Autodiff(#[from] AutodiffError),
    #[error(transparent)]
    ReferenceEvaluation(#[from] ReferenceEvaluationError),
    #[error(transparent)]
    Config(#[from] ParameterGolfConfigError),
    #[error(transparent)]
    Execution(#[from] ParameterGolfExecutionError),
    #[error(transparent)]
    Model(#[from] ParameterGolfModelError),
    #[error(transparent)]
    Tensor(#[from] ParameterGolfTensorError),
    #[error(
        "parameter golf baseline graph expected tied embeddings or an untied lm_head; descriptor `{model_id}` had tie_embeddings={tie_embeddings} and lm_head_present={lm_head_present}"
    )]
    InvalidHeadPosture {
        model_id: String,
        tie_embeddings: bool,
        lm_head_present: bool,
    },
    #[error("parameter golf baseline graph is missing parameter binding for `{parameter_id}`")]
    MissingParameterBinding { parameter_id: String },
    #[error("parameter golf baseline graph is missing weight vector for `{parameter_id}`")]
    MissingWeightVector { parameter_id: String },
    #[error("parameter golf baseline graph is missing gradient for `{parameter_id}`")]
    MissingGradient { parameter_id: String },
    #[error("parameter golf baseline graph is missing forward logits for tensor `{tensor_id}`")]
    MissingForwardLogits { tensor_id: TensorId },
    #[error("parameter golf baseline graph expected dense f32 tensor data for {context}")]
    NonDenseTensorData { context: String },
}

/// Builds one trainer-owned Parameter Golf baseline graph that emits pre-softcap
/// logits from dense embedded inputs plus the explicit baseline parameter
/// surface.
pub fn build_parameter_golf_baseline_graph(
    device: Device,
    descriptor: &ParameterGolfModelDescriptor,
    batch_size: usize,
    sequence_length: usize,
) -> Result<ParameterGolfBaselineGraph, ParameterGolfBaselineGraphError> {
    let config = &descriptor.config;
    if batch_size == 0 {
        return Err(ParameterGolfExecutionError::EmptyBatch.into());
    }
    if sequence_length == 0 {
        return Err(ParameterGolfExecutionError::EmptySequence.into());
    }
    if !config.tie_embeddings
        && !descriptor
            .weights
            .tensors
            .iter()
            .any(|tensor| tensor.name == "lm_head.weight")
    {
        return Err(ParameterGolfBaselineGraphError::InvalidHeadPosture {
            model_id: descriptor.model.model_id.clone(),
            tie_embeddings: config.tie_embeddings,
            lm_head_present: false,
        });
    }

    let mut builder = AutodiffGraphBuilder::with_context(device, AutodiffContext::training());
    let embedded_input = builder.input(
        "embedded_inputs",
        Shape::new(vec![batch_size, sequence_length, config.model_dim]),
        DType::F32,
        true,
    );

    let mut parameter_inputs = BTreeMap::new();
    let mut parameter_bindings = Vec::new();
    for tensor in &descriptor.weights.tensors {
        let tensor_input = builder.input(tensor.name.clone(), tensor.shape.clone(), DType::F32, true);
        let gradient_source = if tensor.name == "tok_emb.weight" {
            if config.tie_embeddings {
                ParameterGolfBaselineGradientSource::GraphAndInputEmbeddingScatter
            } else {
                ParameterGolfBaselineGradientSource::InputEmbeddingScatterOnly
            }
        } else {
            ParameterGolfBaselineGradientSource::GraphOnly
        };
        parameter_bindings.push(ParameterGolfBaselineGraphParameterBinding {
            parameter_id: tensor.name.clone(),
            shape: tensor.shape.clone(),
            graph_input_tensor_id: tensor_input.id(),
            gradient_source,
        });
        parameter_inputs.insert(tensor.name.clone(), tensor_input);
    }

    let ones_model = builder.constant_f32(Shape::new(vec![config.model_dim]), vec![1.0; config.model_dim])?;
    let head_dim = config.head_dim()?;
    let ones_head = builder.constant_f32(Shape::new(vec![head_dim]), vec![1.0; head_dim])?;
    let (rope_cos, rope_sin) =
        rope_table_constants(&mut builder, sequence_length, head_dim, config.rope_base)?;

    let mut x = builder.rms_norm(&embedded_input, &ones_model, PARAMETER_GOLF_DEFAULT_RMS_NORM_EPSILON)?;
    let x0 = x.clone();
    let mut skips = Vec::new();
    for layer_index in 0..config.num_encoder_layers() {
        x = block_forward_graph(
            &mut builder,
            &parameter_inputs,
            &x,
            &x0,
            config,
            batch_size,
            sequence_length,
            layer_index,
            &ones_model,
            &ones_head,
            &rope_cos,
            &rope_sin,
        )?;
        skips.push(x.clone());
    }
    let skip_weights = parameter_inputs
        .get("skip_weights")
        .cloned()
        .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
            parameter_id: String::from("skip_weights"),
        })?;
    for decoder_index in 0..config.num_decoder_layers() {
        if let Some(skip) = skips.pop() {
            let skip_scale = builder.select(&skip_weights, 0, decoder_index)?;
            x = add_scaled_graph(
                &mut builder,
                &x,
                &skip,
                &skip_scale,
                Shape::new(vec![batch_size, sequence_length, config.model_dim]),
            )?;
        }
        x = block_forward_graph(
            &mut builder,
            &parameter_inputs,
            &x,
            &x0,
            config,
            batch_size,
            sequence_length,
            config.num_encoder_layers() + decoder_index,
            &ones_model,
            &ones_head,
            &rope_cos,
            &rope_sin,
        )?;
    }
    let hidden = builder.rms_norm(&x, &ones_model, PARAMETER_GOLF_DEFAULT_RMS_NORM_EPSILON)?;
    let pre_softcap_logits = hidden_to_pre_softcap_logits_graph(
        &mut builder,
        &parameter_inputs,
        &hidden,
        config,
        batch_size,
        sequence_length,
    )?;
    let graph = builder.finish(vec![pre_softcap_logits.clone()]);

    Ok(ParameterGolfBaselineGraph {
        embedded_input_tensor_id: embedded_input.id(),
        pre_softcap_logits_tensor_id: pre_softcap_logits.id(),
        graph,
        parameter_bindings,
    })
}

/// Gathers dense embedded inputs from the Parameter Golf token embedding table for
/// one bounded token batch.
pub fn gather_parameter_golf_embedded_inputs(
    weights: &ParameterGolfWeights,
    config: &ParameterGolfConfig,
    input_ids: &[Vec<u32>],
) -> Result<ParameterGolfTensor3, ParameterGolfBaselineGraphError> {
    let (batch_size, sequence_length) = validate_token_batch(input_ids, config.vocab_size)?;
    let mut values = vec![0.0_f32; batch_size * sequence_length * config.model_dim];
    for (batch_index, row) in input_ids.iter().enumerate() {
        for (position_index, &token_id) in row.iter().enumerate() {
            let source_offset = token_id as usize * config.model_dim;
            let destination_offset = (batch_index * sequence_length + position_index) * config.model_dim;
            values[destination_offset..destination_offset + config.model_dim]
                .copy_from_slice(
                    &weights.token_embedding[source_offset..source_offset + config.model_dim],
                );
        }
    }
    Ok(ParameterGolfTensor3::new(
        [batch_size, sequence_length, config.model_dim],
        values,
    )?)
}

/// Builds one dense input map for the lowered Parameter Golf baseline graph.
pub fn bind_parameter_golf_baseline_graph_inputs(
    graph: &ParameterGolfBaselineGraph,
    model: &ParameterGolfReferenceModel,
    embedded_inputs: &ParameterGolfTensor3,
) -> Result<BTreeMap<TensorId, TensorData>, ParameterGolfBaselineGraphError> {
    let config = &model.descriptor().config;
    let expected_shape = [
        embedded_inputs.batch_size(),
        embedded_inputs.sequence_length(),
        embedded_inputs.width(),
    ];
    if expected_shape[2] != config.model_dim {
        return Err(ParameterGolfTensorError::InvalidValueCount {
            shape: [
                embedded_inputs.batch_size(),
                embedded_inputs.sequence_length(),
                config.model_dim,
            ],
            actual: embedded_inputs.values().len(),
            expected: embedded_inputs.batch_size()
                * embedded_inputs.sequence_length()
                * config.model_dim,
        }
        .into());
    }

    let parameter_vectors = model
        .weights()
        .parameter_vectors(config)
        .into_iter()
        .map(|parameter| (parameter.parameter_id.clone(), parameter))
        .collect::<BTreeMap<_, _>>();
    let mut inputs = BTreeMap::new();
    inputs.insert(
        graph.embedded_input_tensor_id,
        TensorData::F32(embedded_inputs.values().to_vec()),
    );
    for binding in &graph.parameter_bindings {
        let parameter = parameter_vectors.get(&binding.parameter_id).ok_or_else(|| {
            ParameterGolfBaselineGraphError::MissingWeightVector {
                parameter_id: binding.parameter_id.clone(),
            }
        })?;
        inputs.insert(
            binding.graph_input_tensor_id,
            TensorData::F32(parameter.values.clone()),
        );
    }
    Ok(inputs)
}

/// Applies the host-owned logit softcap and cross-entropy gradient seed for one
/// pre-softcap logits tensor.
pub fn parameter_golf_projection_seed(
    pre_softcap_logits: &ParameterGolfTensor3,
    target_ids: &[Vec<u32>],
    logit_softcap: f32,
) -> Result<ParameterGolfProjectionSeed, ParameterGolfBaselineGraphError> {
    if !logit_softcap.is_finite() || logit_softcap <= 0.0 {
        return Err(ParameterGolfExecutionError::InvalidAttentionWindowSize {
            attention_window_size: 0,
        }
        .into());
    }
    validate_target_shape(target_ids, pre_softcap_logits.batch_size(), pre_softcap_logits.sequence_length())?;
    validate_token_batch(target_ids, pre_softcap_logits.width())?;

    let mut softcapped = vec![0.0_f32; pre_softcap_logits.values().len()];
    let mut seed = vec![0.0_f32; pre_softcap_logits.values().len()];
    let mut total_loss = 0.0_f32;
    let position_count = (pre_softcap_logits.batch_size() * pre_softcap_logits.sequence_length()) as f32;

    for batch in 0..pre_softcap_logits.batch_size() {
        for position in 0..pre_softcap_logits.sequence_length() {
            let offset =
                (batch * pre_softcap_logits.sequence_length() + position) * pre_softcap_logits.width();
            let logits_row =
                &pre_softcap_logits.values()[offset..offset + pre_softcap_logits.width()];
            let softcapped_row = &mut softcapped[offset..offset + pre_softcap_logits.width()];
            let seed_row = &mut seed[offset..offset + pre_softcap_logits.width()];
            for (destination, source) in softcapped_row.iter_mut().zip(logits_row.iter()) {
                *destination = logit_softcap * (*source / logit_softcap).tanh();
            }
            let max_logit = softcapped_row
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut probabilities = vec![0.0_f32; softcapped_row.len()];
            let mut exp_sum = 0.0_f32;
            for (index, value) in softcapped_row.iter().enumerate() {
                let exp = (*value - max_logit).exp();
                probabilities[index] = exp;
                exp_sum += exp;
            }
            let exp_sum = exp_sum.max(f32::EPSILON);
            let target = target_ids[batch][position] as usize;
            total_loss += max_logit + exp_sum.ln() - softcapped_row[target];
            for index in 0..probabilities.len() {
                probabilities[index] /= exp_sum;
                let delta = if index == target { 1.0 } else { 0.0 };
                let softcap_derivative = 1.0 - (softcapped_row[index] / logit_softcap).powi(2);
                seed_row[index] =
                    ((probabilities[index] - delta) / position_count) * softcap_derivative;
            }
        }
    }

    Ok(ParameterGolfProjectionSeed {
        softcapped_logits: ParameterGolfTensor3::new(pre_softcap_logits.shape(), softcapped)?,
        mean_loss: total_loss / position_count,
        pre_softcap_gradient: ParameterGolfTensor3::new(pre_softcap_logits.shape(), seed)?,
    })
}

/// Reconstructs stable Parameter Golf parameter gradients from one seeded
/// backward replay, including the host-scattered token-embedding input gradient.
pub fn materialize_parameter_golf_baseline_gradients(
    graph: &ParameterGolfBaselineGraph,
    backward: &psionic_ir::AutodiffBackwardResult,
    config: &ParameterGolfConfig,
    input_ids: &[Vec<u32>],
) -> Result<ParameterGolfBaselineGradientBundle, ParameterGolfBaselineGraphError> {
    let scattered_embedding_gradient = scatter_input_embedding_gradient(
        tensor3_from_dense_gradient(
            backward
                .gradient(graph.embedded_input_tensor_id)
                .ok_or_else(|| ParameterGolfBaselineGraphError::MissingGradient {
                    parameter_id: String::from("embedded_inputs"),
                })?,
            [
                input_ids.len(),
                input_ids.first().map(Vec::len).unwrap_or(0),
                config.model_dim,
            ],
            String::from("embedded_inputs"),
        )?,
        input_ids,
        config.vocab_size,
        config.model_dim,
    )?;

    let mut gradients = BTreeMap::new();
    for binding in &graph.parameter_bindings {
        let graph_gradient = backward.gradient(binding.graph_input_tensor_id);
        let values = match binding.gradient_source {
            ParameterGolfBaselineGradientSource::GraphOnly => dense_gradient_values(
                graph_gradient.ok_or_else(|| ParameterGolfBaselineGraphError::MissingGradient {
                    parameter_id: binding.parameter_id.clone(),
                })?,
                binding.parameter_id.clone(),
            )?,
            ParameterGolfBaselineGradientSource::InputEmbeddingScatterOnly => {
                if binding.parameter_id == "tok_emb.weight" {
                    scattered_embedding_gradient.clone()
                } else {
                    return Err(ParameterGolfBaselineGraphError::MissingGradient {
                        parameter_id: binding.parameter_id.clone(),
                    });
                }
            }
            ParameterGolfBaselineGradientSource::GraphAndInputEmbeddingScatter => {
                let mut values = dense_gradient_values(
                    graph_gradient.ok_or_else(|| ParameterGolfBaselineGraphError::MissingGradient {
                        parameter_id: binding.parameter_id.clone(),
                    })?,
                    binding.parameter_id.clone(),
                )?;
                if binding.parameter_id == "tok_emb.weight" {
                    for (value, scattered) in values.iter_mut().zip(scattered_embedding_gradient.iter())
                    {
                        *value += *scattered;
                    }
                }
                values
            }
        };
        gradients.insert(binding.parameter_id.clone(), values);
    }
    Ok(ParameterGolfBaselineGradientBundle {
        parameter_gradients: gradients,
    })
}

fn block_forward_graph(
    builder: &mut AutodiffGraphBuilder,
    parameters: &BTreeMap<String, AutodiffTensor>,
    x: &AutodiffTensor,
    x0: &AutodiffTensor,
    config: &ParameterGolfConfig,
    batch_size: usize,
    sequence_length: usize,
    layer_index: usize,
    ones_model: &AutodiffTensor,
    ones_head: &AutodiffTensor,
    rope_cos: &AutodiffTensor,
    rope_sin: &AutodiffTensor,
) -> Result<AutodiffTensor, ParameterGolfBaselineGraphError> {
    let resid_mix = parameters
        .get(format!("blocks.{layer_index}.resid_mix").as_str())
        .cloned()
        .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
            parameter_id: format!("blocks.{layer_index}.resid_mix"),
        })?;
    let mixed = blend_with_source_graph(
        builder,
        x,
        x0,
        &resid_mix,
        batch_size,
        sequence_length,
        config.model_dim,
    )?;
    let normed_for_attention =
        builder.rms_norm(&mixed, ones_model, PARAMETER_GOLF_DEFAULT_RMS_NORM_EPSILON)?;
    let attention = attention_forward_graph(
        builder,
        parameters,
        &normed_for_attention,
        config,
        batch_size,
        sequence_length,
        layer_index,
        ones_head,
        rope_cos,
        rope_sin,
    )?;
    let attn_scale = parameters
        .get(format!("blocks.{layer_index}.attn_scale").as_str())
        .cloned()
        .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
            parameter_id: format!("blocks.{layer_index}.attn_scale"),
        })?;
    let x = add_scaled_graph(
        builder,
        &mixed,
        &attention,
        &attn_scale,
        Shape::new(vec![batch_size, sequence_length, config.model_dim]),
    )?;
    let normed_for_mlp = builder.rms_norm(&x, ones_model, PARAMETER_GOLF_DEFAULT_RMS_NORM_EPSILON)?;
    let mlp = mlp_forward_graph(
        builder,
        parameters,
        &normed_for_mlp,
        config,
        batch_size,
        sequence_length,
        layer_index,
    )?;
    let mlp_scale = parameters
        .get(format!("blocks.{layer_index}.mlp_scale").as_str())
        .cloned()
        .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
            parameter_id: format!("blocks.{layer_index}.mlp_scale"),
        })?;
    add_scaled_graph(
        builder,
        &x,
        &mlp,
        &mlp_scale,
        Shape::new(vec![batch_size, sequence_length, config.model_dim]),
    )
}

fn attention_forward_graph(
    builder: &mut AutodiffGraphBuilder,
    parameters: &BTreeMap<String, AutodiffTensor>,
    input: &AutodiffTensor,
    config: &ParameterGolfConfig,
    batch_size: usize,
    sequence_length: usize,
    layer_index: usize,
    ones_head: &AutodiffTensor,
    rope_cos: &AutodiffTensor,
    rope_sin: &AutodiffTensor,
) -> Result<AutodiffTensor, ParameterGolfBaselineGraphError> {
    let head_dim = config.head_dim()?;
    let kv_dim = config.kv_dim()?;
    let q_proj = linear_3d(
        builder,
        input,
        parameters
            .get(format!("blocks.{layer_index}.attn.c_q.weight").as_str())
            .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
                parameter_id: format!("blocks.{layer_index}.attn.c_q.weight"),
            })?,
        batch_size,
        sequence_length,
        config.model_dim,
        config.model_dim,
    )?;
    let k_proj = linear_3d(
        builder,
        input,
        parameters
            .get(format!("blocks.{layer_index}.attn.c_k.weight").as_str())
            .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
                parameter_id: format!("blocks.{layer_index}.attn.c_k.weight"),
            })?,
        batch_size,
        sequence_length,
        config.model_dim,
        kv_dim,
    )?;
    let v_proj = linear_3d(
        builder,
        input,
        parameters
            .get(format!("blocks.{layer_index}.attn.c_v.weight").as_str())
            .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
                parameter_id: format!("blocks.{layer_index}.attn.c_v.weight"),
            })?,
        batch_size,
        sequence_length,
        config.model_dim,
        kv_dim,
    )?;

    let q = reshape_to_attention_heads(
        builder,
        &q_proj,
        batch_size,
        sequence_length,
        config.num_heads,
        head_dim,
    )?;
    let k = reshape_to_attention_heads(
        builder,
        &k_proj,
        batch_size,
        sequence_length,
        config.num_kv_heads,
        head_dim,
    )?;
    let v = reshape_to_attention_heads(
        builder,
        &v_proj,
        batch_size,
        sequence_length,
        config.num_kv_heads,
        head_dim,
    )?;

    let q = builder.rms_norm(&q, ones_head, PARAMETER_GOLF_DEFAULT_RMS_NORM_EPSILON)?;
    let k = builder.rms_norm(&k, ones_head, PARAMETER_GOLF_DEFAULT_RMS_NORM_EPSILON)?;
    let q = builder.rope(&q, rope_cos, rope_sin, false)?;
    let k = builder.rope(&k, rope_cos, rope_sin, false)?;
    let q_gain = parameters
        .get(format!("blocks.{layer_index}.attn.q_gain").as_str())
        .cloned()
        .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
            parameter_id: format!("blocks.{layer_index}.attn.q_gain"),
        })?;
    let q_gain = builder.reshape(&q_gain, Shape::new(vec![1, config.num_heads, 1, 1]))?;
    let q_gain = builder.expand(
        &q_gain,
        Shape::new(vec![batch_size, config.num_heads, sequence_length, head_dim]),
    )?;
    let q = builder.mul(&q, &q_gain)?;
    let attended = builder.scaled_dot_product_attention(
        &q,
        &k,
        &v,
        1.0_f32 / (head_dim as f32).sqrt(),
        true,
    )?;
    let merged = builder.permute(&attended, vec![0, 2, 1, 3])?;
    let merged =
        builder.reshape(&merged, Shape::new(vec![batch_size, sequence_length, config.model_dim]))?;
    linear_3d(
        builder,
        &merged,
        parameters
            .get(format!("blocks.{layer_index}.attn.proj.weight").as_str())
            .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
                parameter_id: format!("blocks.{layer_index}.attn.proj.weight"),
            })?,
        batch_size,
        sequence_length,
        config.model_dim,
        config.model_dim,
    )
}

fn mlp_forward_graph(
    builder: &mut AutodiffGraphBuilder,
    parameters: &BTreeMap<String, AutodiffTensor>,
    input: &AutodiffTensor,
    config: &ParameterGolfConfig,
    batch_size: usize,
    sequence_length: usize,
    layer_index: usize,
) -> Result<AutodiffTensor, ParameterGolfBaselineGraphError> {
    let hidden_dim = config.mlp_hidden_dim()?;
    let hidden = linear_3d(
        builder,
        input,
        parameters
            .get(format!("blocks.{layer_index}.mlp.fc.weight").as_str())
            .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
                parameter_id: format!("blocks.{layer_index}.mlp.fc.weight"),
            })?,
        batch_size,
        sequence_length,
        config.model_dim,
        hidden_dim,
    )?;
    let activated = builder.relu_squared(&hidden)?;
    linear_3d(
        builder,
        &activated,
        parameters
            .get(format!("blocks.{layer_index}.mlp.proj.weight").as_str())
            .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
                parameter_id: format!("blocks.{layer_index}.mlp.proj.weight"),
            })?,
        batch_size,
        sequence_length,
        hidden_dim,
        config.model_dim,
    )
}

fn hidden_to_pre_softcap_logits_graph(
    builder: &mut AutodiffGraphBuilder,
    parameters: &BTreeMap<String, AutodiffTensor>,
    hidden: &AutodiffTensor,
    config: &ParameterGolfConfig,
    batch_size: usize,
    sequence_length: usize,
) -> Result<AutodiffTensor, ParameterGolfBaselineGraphError> {
    let head_weight = if config.tie_embeddings {
        parameters
            .get("tok_emb.weight")
            .cloned()
            .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
                parameter_id: String::from("tok_emb.weight"),
            })?
    } else {
        parameters
            .get("lm_head.weight")
            .cloned()
            .ok_or_else(|| ParameterGolfBaselineGraphError::MissingParameterBinding {
                parameter_id: String::from("lm_head.weight"),
            })?
    };
    linear_3d(
        builder,
        hidden,
        &head_weight,
        batch_size,
        sequence_length,
        config.model_dim,
        config.vocab_size,
    )
}

fn linear_3d(
    builder: &mut AutodiffGraphBuilder,
    input: &AutodiffTensor,
    weight: &AutodiffTensor,
    batch_size: usize,
    sequence_length: usize,
    in_features: usize,
    out_features: usize,
) -> Result<AutodiffTensor, ParameterGolfBaselineGraphError> {
    let flattened =
        builder.reshape(input, Shape::new(vec![batch_size * sequence_length, in_features]))?;
    let transposed_weight = builder.permute(weight, vec![1, 0])?;
    let output = builder.matmul(&flattened, &transposed_weight)?;
    Ok(builder.reshape(&output, Shape::new(vec![batch_size, sequence_length, out_features]))?)
}

fn reshape_to_attention_heads(
    builder: &mut AutodiffGraphBuilder,
    tensor: &AutodiffTensor,
    batch_size: usize,
    sequence_length: usize,
    head_count: usize,
    head_dim: usize,
) -> Result<AutodiffTensor, ParameterGolfBaselineGraphError> {
    let reshaped =
        builder.reshape(tensor, Shape::new(vec![batch_size, sequence_length, head_count, head_dim]))?;
    Ok(builder.permute(&reshaped, vec![0, 2, 1, 3])?)
}

fn blend_with_source_graph(
    builder: &mut AutodiffGraphBuilder,
    current: &AutodiffTensor,
    source: &AutodiffTensor,
    mix: &AutodiffTensor,
    batch_size: usize,
    sequence_length: usize,
    model_dim: usize,
) -> Result<AutodiffTensor, ParameterGolfBaselineGraphError> {
    let current_mix = builder.select(mix, 0, 0)?;
    let source_mix = builder.select(mix, 0, 1)?;
    let current_mix = builder.reshape(&current_mix, Shape::new(vec![1, 1, model_dim]))?;
    let source_mix = builder.reshape(&source_mix, Shape::new(vec![1, 1, model_dim]))?;
    let current_mix =
        builder.expand(&current_mix, Shape::new(vec![batch_size, sequence_length, model_dim]))?;
    let source_mix =
        builder.expand(&source_mix, Shape::new(vec![batch_size, sequence_length, model_dim]))?;
    let current = builder.mul(current, &current_mix)?;
    let source = builder.mul(source, &source_mix)?;
    Ok(builder.add(&current, &source)?)
}

fn add_scaled_graph(
    builder: &mut AutodiffGraphBuilder,
    base: &AutodiffTensor,
    delta: &AutodiffTensor,
    scale: &AutodiffTensor,
    target_shape: Shape,
) -> Result<AutodiffTensor, ParameterGolfBaselineGraphError> {
    let feature_count = *target_shape
        .dims()
        .last()
        .expect("parameter golf add_scaled target shape should have width");
    let scale = builder.reshape(scale, Shape::new(vec![1, 1, feature_count]))?;
    let scale = builder.expand(&scale, target_shape)?;
    let scaled_delta = builder.mul(delta, &scale)?;
    Ok(builder.add(base, &scaled_delta)?)
}

fn rope_table_constants(
    builder: &mut AutodiffGraphBuilder,
    sequence_length: usize,
    head_dim: usize,
    rope_base: f32,
) -> Result<(AutodiffTensor, AutodiffTensor), ParameterGolfBaselineGraphError> {
    let half_dim = head_dim / 2;
    let mut cos = vec![0.0_f32; sequence_length * half_dim];
    let mut sin = vec![0.0_f32; sequence_length * half_dim];
    for position in 0..sequence_length {
        for feature in 0..half_dim {
            let exponent = (2 * feature) as f32 / head_dim as f32;
            let inv_freq = 1.0_f32 / rope_base.powf(exponent);
            let angle = position as f32 * inv_freq;
            cos[position * half_dim + feature] = angle.cos();
            sin[position * half_dim + feature] = angle.sin();
        }
    }
    Ok((
        builder.constant_f32(Shape::new(vec![sequence_length, half_dim]), cos)?,
        builder.constant_f32(Shape::new(vec![sequence_length, half_dim]), sin)?,
    ))
}

fn validate_token_batch(
    input_ids: &[Vec<u32>],
    vocab_size: usize,
) -> Result<(usize, usize), ParameterGolfExecutionError> {
    if input_ids.is_empty() {
        return Err(ParameterGolfExecutionError::EmptyBatch);
    }
    let sequence_length = input_ids.first().map(Vec::len).unwrap_or(0);
    if sequence_length == 0 {
        return Err(ParameterGolfExecutionError::EmptySequence);
    }
    for (batch_index, row) in input_ids.iter().enumerate() {
        if row.len() != sequence_length {
            return Err(ParameterGolfExecutionError::RaggedBatch {
                batch_index,
                expected: sequence_length,
                actual: row.len(),
            });
        }
        for &token_id in row {
            if token_id as usize >= vocab_size {
                return Err(ParameterGolfExecutionError::TokenOutOfRange {
                    token_id,
                    vocab_size,
                });
            }
        }
    }
    Ok((input_ids.len(), sequence_length))
}

fn validate_target_shape(
    target_ids: &[Vec<u32>],
    expected_batch: usize,
    expected_sequence: usize,
) -> Result<(), ParameterGolfExecutionError> {
    if target_ids.len() != expected_batch
        || target_ids.first().map(Vec::len).unwrap_or(0) != expected_sequence
    {
        return Err(ParameterGolfExecutionError::TargetShapeMismatch {
            expected_batch,
            expected_sequence,
            actual_batch: target_ids.len(),
            actual_sequence: target_ids.first().map(Vec::len).unwrap_or(0),
        });
    }
    Ok(())
}

fn tensor3_from_dense_gradient(
    data: &TensorData,
    shape: [usize; 3],
    context: String,
) -> Result<ParameterGolfTensor3, ParameterGolfBaselineGraphError> {
    let values = dense_gradient_values(data, context)?;
    Ok(ParameterGolfTensor3::new(shape, values)?)
}

fn dense_gradient_values(
    data: &TensorData,
    context: String,
) -> Result<Vec<f32>, ParameterGolfBaselineGraphError> {
    let TensorData::F32(values) = data else {
        return Err(ParameterGolfBaselineGraphError::NonDenseTensorData { context });
    };
    Ok(values.clone())
}

fn scatter_input_embedding_gradient(
    input_gradient: ParameterGolfTensor3,
    input_ids: &[Vec<u32>],
    vocab_size: usize,
    model_dim: usize,
) -> Result<Vec<f32>, ParameterGolfBaselineGraphError> {
    let mut scattered = vec![0.0_f32; vocab_size * model_dim];
    for (batch_index, row) in input_ids.iter().enumerate() {
        for (position_index, &token_id) in row.iter().enumerate() {
            let token_offset = token_id as usize * model_dim;
            let gradient_offset =
                (batch_index * input_gradient.sequence_length() + position_index) * model_dim;
            for feature in 0..model_dim {
                scattered[token_offset + feature] +=
                    input_gradient.values()[gradient_offset + feature];
            }
        }
    }
    Ok(scattered)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use psionic_compiler::compile_graph;
    use psionic_core::TensorData;
    use psionic_ir::evaluate_graph;
    use serde::Deserialize;

    use super::*;
    use psionic_models::{ModelDescriptor, ParameterGolfDeterministicInitializer};

    #[derive(Deserialize)]
    struct BaselineFixture {
        initializer: ParameterGolfDeterministicInitializer,
        input_ids: Vec<Vec<u32>>,
        target_ids: Vec<Vec<u32>>,
    }

    fn load_baseline_fixture() -> BaselineFixture {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../fixtures/parameter_golf/models/parameter_golf_baseline_model_fixture.json",
        );
        serde_json::from_slice(&fs::read(path).expect("fixture should exist"))
            .expect("fixture should deserialize")
    }

    fn baseline_model() -> Result<ParameterGolfReferenceModel, Box<dyn std::error::Error>> {
        let fixture = load_baseline_fixture();
        Ok(ParameterGolfReferenceModel::baseline_fixture(fixture.initializer)?)
    }

    fn loss_with_parameter_override(
        model: &ParameterGolfReferenceModel,
        parameter_id: &str,
        values: Vec<f32>,
        input_ids: &[Vec<u32>],
        target_ids: &[Vec<u32>],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let mut overrides = BTreeMap::new();
        overrides.insert(String::from(parameter_id), values);
        let updated_weights = model
            .weights()
            .with_parameter_overrides(&model.descriptor().config, &overrides)?;
        let updated_model = ParameterGolfReferenceModel::new(
            ModelDescriptor::new(
                model.descriptor().model.model_id.clone(),
                model.descriptor().model.family.clone(),
                model.descriptor().model.revision.clone(),
            ),
            model.descriptor().config.clone(),
            updated_weights,
        )?;
        Ok(updated_model.loss(input_ids, target_ids)?)
    }

    #[test]
    fn parameter_golf_baseline_graph_matches_reference_logits_on_fixture_batch(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fixture = load_baseline_fixture();
        let model = baseline_model()?;
        let graph = build_parameter_golf_baseline_graph(
            Device::cpu(),
            model.descriptor(),
            fixture.input_ids.len(),
            fixture.input_ids[0].len(),
        )?;
        compile_graph(graph.graph.graph())?;

        let embedded_inputs = gather_parameter_golf_embedded_inputs(
            model.weights(),
            &model.descriptor().config,
            fixture.input_ids.as_slice(),
        )?;
        let inputs = bind_parameter_golf_baseline_graph_inputs(&graph, &model, &embedded_inputs)?;
        let values = evaluate_graph(graph.graph.graph(), &inputs)?;
        let pre_softcap_logits = tensor3_from_dense_gradient(
            values
                .get(&graph.pre_softcap_logits_tensor_id)
                .ok_or("missing pre-softcap logits")?,
            [1, 4, model.descriptor().config.vocab_size],
            String::from("pre_softcap_logits"),
        )?;
        let seeded = parameter_golf_projection_seed(
            &pre_softcap_logits,
            fixture.target_ids.as_slice(),
            model.descriptor().config.logit_softcap,
        )?;
        let reference = model.forward_logits(fixture.input_ids.as_slice())?;
        let max_abs_diff = seeded.softcapped_logits.max_abs_diff(&reference)?;
        assert!(
            max_abs_diff < 5e-5,
            "max logit drift {max_abs_diff} exceeded tolerance"
        );
        Ok(())
    }

    #[test]
    fn parameter_golf_baseline_graph_materializes_seeded_gradients_for_parameter_families(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fixture = load_baseline_fixture();
        let model = baseline_model()?;
        let graph = build_parameter_golf_baseline_graph(
            Device::cpu(),
            model.descriptor(),
            fixture.input_ids.len(),
            fixture.input_ids[0].len(),
        )?;

        let embedded_inputs = gather_parameter_golf_embedded_inputs(
            model.weights(),
            &model.descriptor().config,
            fixture.input_ids.as_slice(),
        )?;
        let inputs = bind_parameter_golf_baseline_graph_inputs(&graph, &model, &embedded_inputs)?;
        let forward = evaluate_graph(graph.graph.graph(), &inputs)?;
        let pre_softcap_logits = tensor3_from_dense_gradient(
            forward
                .get(&graph.pre_softcap_logits_tensor_id)
                .ok_or("missing pre-softcap logits")?,
            [1, 4, model.descriptor().config.vocab_size],
            String::from("pre_softcap_logits"),
        )?;
        let seeded = parameter_golf_projection_seed(
            &pre_softcap_logits,
            fixture.target_ids.as_slice(),
            model.descriptor().config.logit_softcap,
        )?;
        let backward = graph.graph.backward_materialized_with_seed(
            graph.pre_softcap_logits_tensor_id,
            &inputs,
            Some(TensorData::F32(seeded.pre_softcap_gradient.values().to_vec())),
        )?;
        let gradients = materialize_parameter_golf_baseline_gradients(
            &graph,
            &backward,
            &model.descriptor().config,
            fixture.input_ids.as_slice(),
        )?;

        let checked_coordinates = [
            ("tok_emb.weight", 17 * model.descriptor().config.model_dim),
            ("skip_weights", 0),
            ("blocks.0.attn.c_q.weight", 0),
            ("blocks.0.attn.q_gain", 0),
            ("blocks.0.attn_scale", 0),
            ("blocks.0.mlp.fc.weight", 0),
            ("blocks.0.mlp_scale", 0),
            ("blocks.0.resid_mix", 0),
        ];
        let delta = 1e-3_f32;
        for (parameter_id, flat_index) in checked_coordinates {
            let baseline = model
                .weights()
                .parameter_vector(&model.descriptor().config, parameter_id)
                .ok_or("missing baseline parameter")?;
            let mut plus = baseline.values.clone();
            plus[flat_index] += delta;
            let mut minus = baseline.values.clone();
            minus[flat_index] -= delta;
            let finite = (loss_with_parameter_override(
                &model,
                parameter_id,
                plus,
                fixture.input_ids.as_slice(),
                fixture.target_ids.as_slice(),
            )? - loss_with_parameter_override(
                &model,
                parameter_id,
                minus,
                fixture.input_ids.as_slice(),
                fixture.target_ids.as_slice(),
            )?) / (2.0 * delta);
            let actual = gradients
                .parameter_gradients
                .get(parameter_id)
                .ok_or("missing parameter gradient")?[flat_index];
            assert!(
                (actual - finite).abs() < 5e-2,
                "gradient drift for {parameter_id}[{flat_index}] exceeded tolerance: actual={actual} finite={finite}"
            );
        }
        Ok(())
    }
}

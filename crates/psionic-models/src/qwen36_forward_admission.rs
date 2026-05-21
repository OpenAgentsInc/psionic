use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    load_qwen36_model_config, normalize_qwen36_target_model_id, PromptMessage, PromptMessageRole,
    Qwen36PromptOptions, Qwen36PromptReceipt, Qwen36PromptRenderer, Qwen36ReasoningMode,
    Qwen36TargetPathError, QWEN36_27B_MODEL_ID, QWEN36_27B_SERVED_MODEL_ID,
};

pub const QWEN36_FORWARD_ADMISSION_SCHEMA_VERSION: &str = "psionic.qwen36_27b_forward_admission.v1";
pub const QWEN36_FORWARD_REFUSAL_CODE: &str = "qwen3_5_text_forward_not_implemented";
pub const QWEN36_SAMPLED_PROJECTION_BACKEND: &str = "local-sampled-projection";
const QWEN36_EMBED_TOKENS_WEIGHT: &str = "model.language_model.embed_tokens.weight";
const QWEN36_LM_HEAD_WEIGHT: &str = "lm_head.weight";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen36ForwardArchitectureReport {
    pub root_model_type: String,
    pub text_model_type: String,
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub torch_dtype: String,
    pub rms_norm_eps: Option<f64>,
    pub hidden_act: Option<String>,
    pub layer_types: Vec<String>,
    pub full_attention_layers: Vec<usize>,
    pub linear_attention_layers: Vec<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
    pub linear_num_key_heads: Option<usize>,
    pub linear_num_value_heads: Option<usize>,
    pub linear_conv_kernel_dim: Option<usize>,
    pub attn_output_gate: Option<bool>,
    pub output_gate_type: Option<String>,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: Option<bool>,
    pub rope_parameters_hash: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36TensorSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36ObservedTensorSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub shard: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36SafetensorsShardHeaderReport {
    pub path: String,
    pub shard_name: String,
    pub byte_len: u64,
    pub header_sha256: String,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36TensorShapeMismatch {
    pub name: String,
    pub expected_shape: Vec<usize>,
    pub observed_shape: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36TensorDtypeMismatch {
    pub name: String,
    pub expected_dtype: String,
    pub observed_dtype: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36TensorAdmissionReport {
    pub expected_text_tensor_count: usize,
    pub observed_index_tensor_count: usize,
    pub observed_header_tensor_count: usize,
    pub admitted_text_tensor_count: usize,
    pub visual_or_other_observed_tensor_count: usize,
    pub shard_headers: Vec<Qwen36SafetensorsShardHeaderReport>,
    pub missing_expected_tensors: Vec<Qwen36TensorSpec>,
    pub index_tensors_missing_from_headers: Vec<String>,
    pub header_tensors_missing_from_index: Vec<String>,
    pub visual_or_other_observed_tensors: Vec<String>,
    pub shape_mismatches: Vec<Qwen36TensorShapeMismatch>,
    pub dtype_mismatches: Vec<Qwen36TensorDtypeMismatch>,
    pub text_tensor_admission_passed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen36ForwardExecutionStatus {
    Refused,
    SampledProjection,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen36TensorRowReadReceipt {
    pub tensor_name: String,
    pub shard_name: String,
    pub shard_path: String,
    pub row_index: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub row_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen36SampledLogit {
    pub token_id: u32,
    pub token_label: String,
    pub logit: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen36ForwardProjectionReceipt {
    pub mode: String,
    pub input_token_id: u32,
    pub input_token_label: String,
    pub candidate_token_ids: Vec<u32>,
    pub sampled_logits: Vec<Qwen36SampledLogit>,
    pub logits_sha256: String,
    pub tensor_reads: Vec<Qwen36TensorRowReadReceipt>,
    pub hidden_size: usize,
    pub used_full_transformer_layers: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen36ForwardAdmissionRunReport {
    pub schema_version: String,
    pub model_id: String,
    pub served_model_id: String,
    pub model_dir: String,
    pub config_path: String,
    pub config_sha256: String,
    pub tokenizer_path: String,
    pub tokenizer_sha256: String,
    pub index_path: String,
    pub index_sha256: String,
    pub architecture: Qwen36ForwardArchitectureReport,
    pub prompt_receipt: Qwen36PromptReceipt,
    pub tensor_admission: Qwen36TensorAdmissionReport,
    pub tensor_admission_sha256: String,
    pub backend: String,
    pub precision: String,
    pub forward_execution_status: Qwen36ForwardExecutionStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_receipt: Option<Qwen36ForwardProjectionReceipt>,
    pub refusal_code: String,
    pub refusal_detail: String,
    pub claim_boundary: String,
}

pub fn run_qwen36_forward_admission(
    model_dir: impl AsRef<Path>,
    prompt_path: impl AsRef<Path>,
    backend: &str,
) -> Result<Qwen36ForwardAdmissionRunReport, Qwen36TargetPathError> {
    let model_dir = model_dir.as_ref();
    let prompt_path = prompt_path.as_ref();
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let index_path = model_dir.join("model.safetensors.index.json");

    let config = load_qwen36_model_config(&config_path)?;
    let model_id = normalize_qwen36_target_model_id(config.model_id.as_str())?;
    if model_id != QWEN36_27B_MODEL_ID {
        return Err(Qwen36TargetPathError::InvalidConfig(String::from(
            "Qwen3.6 forward admission currently supports the dense 27B text checkpoint only",
        )));
    }
    if config.served_model_id != QWEN36_27B_SERVED_MODEL_ID {
        return Err(Qwen36TargetPathError::InvalidConfig(String::from(
            "Qwen3.6-27B served_model_id must be qwen3.6-27b",
        )));
    }

    let config_bytes = read_bytes(&config_path)?;
    let tokenizer_bytes = read_bytes(&tokenizer_path)?;
    let index_bytes = read_bytes(&index_path)?;
    let architecture = qwen36_forward_architecture_report(config_bytes.as_slice())?;
    let weight_map = qwen36_weight_index_from_bytes(index_bytes.as_slice())?;
    let shard_paths = qwen36_shard_paths_from_weight_map(model_dir, &weight_map)?;
    let observed_tensors = qwen36_observed_tensors_from_shards(&shard_paths)?;
    let expected_tensors = qwen36_expected_text_tensor_specs(&architecture)?;
    let tensor_admission =
        qwen36_tensor_admission_report(expected_tensors, weight_map.clone(), observed_tensors);
    let tensor_admission_sha256 = sha256_json(&tensor_admission)?;

    let prompt = fs::read_to_string(prompt_path).map_err(|source| Qwen36TargetPathError::Io {
        path: prompt_path.to_path_buf(),
        source,
    })?;
    let renderer = Qwen36PromptRenderer::from_tokenizer_json_bytes(tokenizer_bytes.as_slice())?;
    let rendered = renderer.render(
        &[
            PromptMessage::new(
                PromptMessageRole::System,
                "You are Autopilot's legal benchmark agent. Answer directly and write usable legal work product.",
            ),
            PromptMessage::new(PromptMessageRole::User, prompt),
        ],
        &Qwen36PromptOptions {
            reasoning_mode: Qwen36ReasoningMode::DirectAnswer,
            add_generation_prompt: true,
            emit_empty_think_block: true,
        },
    )?;
    let prompt_receipt = Qwen36PromptReceipt::from(&rendered);
    let projection_receipt = if should_run_sampled_projection(backend) {
        if !tensor_admission.text_tensor_admission_passed {
            return Err(Qwen36TargetPathError::InvalidConfig(String::from(
                "sampled Qwen3.6 projection requires text tensor admission to pass first",
            )));
        }
        Some(qwen36_sampled_projection_receipt(
            model_dir,
            &weight_map,
            &architecture,
            &rendered,
        )?)
    } else {
        None
    };
    let forward_execution_status = if projection_receipt.is_some() {
        Qwen36ForwardExecutionStatus::SampledProjection
    } else {
        Qwen36ForwardExecutionStatus::Refused
    };
    let refusal_code = if projection_receipt.is_some() {
        String::from("qwen3_5_text_full_transformer_forward_not_implemented")
    } else {
        String::from(QWEN36_FORWARD_REFUSAL_CODE)
    };
    let refusal_detail = if projection_receipt.is_some() {
        String::from(
            "Psionic read real Qwen3.6-27B embed_tokens and lm_head rows from safetensors and computed deterministic sampled projection logits. It still does not run the qwen3_5_text transformer stack, so this is not a full model forward pass.",
        )
    } else {
        String::from(
            "Psionic can now verify the real Qwen3.6-27B text tensor table from safetensors headers. It still lacks the qwen3_5_text mixed linear-attention/full-attention/MTP forward kernels, so it refuses logits instead of pretending to run inference.",
        )
    };
    let claim_boundary = if projection_receipt.is_some() {
        String::from(
            "This report reads the real Qwen/Qwen3.6-27B config, tokenizer, index, safetensors headers, one real embedding row, and sampled real lm_head rows; it computes sampled projection logits and hashes the rows and logits. It does not run attention, MLP, linear attention, MTP, generation, or LoRA training.",
        )
    } else {
        String::from(
            "This report reads the real Qwen/Qwen3.6-27B config, tokenizer, index, and safetensors headers; validates the required text tensor names, dtypes, and shapes; and records a typed refusal for forward execution. It does not produce logits or train from live Qwen3.6 activations.",
        )
    };

    Ok(Qwen36ForwardAdmissionRunReport {
        schema_version: String::from(QWEN36_FORWARD_ADMISSION_SCHEMA_VERSION),
        model_id: String::from(QWEN36_27B_MODEL_ID),
        served_model_id: String::from(QWEN36_27B_SERVED_MODEL_ID),
        model_dir: model_dir.display().to_string(),
        config_path: config_path.display().to_string(),
        config_sha256: sha256_hex(config_bytes.as_slice()),
        tokenizer_path: tokenizer_path.display().to_string(),
        tokenizer_sha256: sha256_hex(tokenizer_bytes.as_slice()),
        index_path: index_path.display().to_string(),
        index_sha256: sha256_hex(index_bytes.as_slice()),
        architecture,
        prompt_receipt,
        tensor_admission,
        tensor_admission_sha256,
        backend: String::from(backend),
        precision: config.torch_dtype,
        forward_execution_status,
        projection_receipt,
        refusal_code,
        refusal_detail,
        claim_boundary,
    })
}

pub fn qwen36_forward_architecture_report(
    config_bytes: &[u8],
) -> Result<Qwen36ForwardArchitectureReport, Qwen36TargetPathError> {
    let root = serde_json::from_slice::<Value>(config_bytes)?;
    let text = root
        .get("text_config")
        .ok_or_else(|| {
            Qwen36TargetPathError::InvalidConfig(String::from(
                "Hugging Face Qwen3.6 config must contain text_config",
            ))
        })?
        .clone();
    let root_model_type = required_string(&root, "model_type")?;
    let text_model_type = required_string(&text, "model_type")?;
    if root_model_type != "qwen3_5" || text_model_type != "qwen3_5_text" {
        return Err(Qwen36TargetPathError::InvalidConfig(format!(
            "unsupported Qwen3.6 forward config model_type `{root_model_type}` / `{text_model_type}`"
        )));
    }
    let architectures = root
        .get("architectures")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let layer_types = text
        .get("layer_types")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let num_hidden_layers = required_usize(&text, "num_hidden_layers")?;
    if layer_types.len() != num_hidden_layers {
        return Err(Qwen36TargetPathError::InvalidConfig(format!(
            "text_config.layer_types must contain {num_hidden_layers} entries"
        )));
    }
    let full_attention_layers = layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (value == "full_attention").then_some(index))
        .collect::<Vec<_>>();
    let linear_attention_layers = layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (value == "linear_attention").then_some(index))
        .collect::<Vec<_>>();
    if full_attention_layers.len() + linear_attention_layers.len() != layer_types.len() {
        return Err(Qwen36TargetPathError::InvalidConfig(String::from(
            "text_config.layer_types may only contain full_attention or linear_attention",
        )));
    }
    let hidden_size = required_usize(&text, "hidden_size")?;
    let num_attention_heads = required_usize(&text, "num_attention_heads")?;
    let head_dim = optional_usize(&text, "head_dim")
        .unwrap_or_else(|| hidden_size.checked_div(num_attention_heads).unwrap_or(0));
    if head_dim == 0 {
        return Err(Qwen36TargetPathError::InvalidConfig(String::from(
            "Qwen3.6 head_dim must be non-zero",
        )));
    }
    let dtype = text
        .get("torch_dtype")
        .or_else(|| text.get("dtype"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let rope_parameters_hash = text
        .get("rope_parameters")
        .map(|value| sha256_json(value))
        .transpose()?;

    Ok(Qwen36ForwardArchitectureReport {
        root_model_type,
        text_model_type,
        architectures,
        hidden_size,
        intermediate_size: required_usize(&text, "intermediate_size")?,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads: required_usize(&text, "num_key_value_heads")?,
        head_dim,
        vocab_size: required_usize(&text, "vocab_size")?,
        max_position_embeddings: required_usize(&text, "max_position_embeddings")?,
        torch_dtype: dtype,
        rms_norm_eps: text.get("rms_norm_eps").and_then(Value::as_f64),
        hidden_act: optional_string(&text, "hidden_act"),
        layer_types,
        full_attention_layers,
        linear_attention_layers,
        linear_key_head_dim: optional_usize(&text, "linear_key_head_dim"),
        linear_value_head_dim: optional_usize(&text, "linear_value_head_dim"),
        linear_num_key_heads: optional_usize(&text, "linear_num_key_heads"),
        linear_num_value_heads: optional_usize(&text, "linear_num_value_heads"),
        linear_conv_kernel_dim: optional_usize(&text, "linear_conv_kernel_dim"),
        attn_output_gate: text.get("attn_output_gate").and_then(Value::as_bool),
        output_gate_type: optional_string(&text, "output_gate_type"),
        mtp_num_hidden_layers: optional_usize(&text, "mtp_num_hidden_layers").unwrap_or(0),
        mtp_use_dedicated_embeddings: text
            .get("mtp_use_dedicated_embeddings")
            .and_then(Value::as_bool),
        rope_parameters_hash,
    })
}

pub fn qwen36_expected_text_tensor_specs(
    architecture: &Qwen36ForwardArchitectureReport,
) -> Result<Vec<Qwen36TensorSpec>, Qwen36TargetPathError> {
    let dtype = hf_dtype_to_safetensors_dtype(architecture.torch_dtype.as_str());
    let mut specs = Vec::new();
    specs.push(tensor(
        "model.language_model.embed_tokens.weight",
        &dtype,
        [architecture.vocab_size, architecture.hidden_size],
    ));
    specs.push(tensor(
        "model.language_model.norm.weight",
        &dtype,
        [architecture.hidden_size],
    ));
    specs.push(tensor(
        "lm_head.weight",
        &dtype,
        [architecture.vocab_size, architecture.hidden_size],
    ));

    for (layer, layer_type) in architecture.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        push_common_decoder_layer_specs(&mut specs, &prefix, &dtype, architecture);
        match layer_type.as_str() {
            "full_attention" => {
                push_full_attention_specs(&mut specs, &prefix, &dtype, architecture);
            }
            "linear_attention" => {
                push_linear_attention_specs(&mut specs, &prefix, &dtype, architecture)?;
            }
            other => {
                return Err(Qwen36TargetPathError::InvalidConfig(format!(
                    "unsupported Qwen3.6 layer type `{other}`"
                )));
            }
        }
    }

    if architecture.mtp_num_hidden_layers > 0 {
        specs.push(tensor(
            "mtp.pre_fc_norm_embedding.weight",
            &dtype,
            [architecture.hidden_size],
        ));
        specs.push(tensor(
            "mtp.pre_fc_norm_hidden.weight",
            &dtype,
            [architecture.hidden_size],
        ));
        specs.push(tensor(
            "mtp.fc.weight",
            &dtype,
            [architecture.hidden_size, architecture.hidden_size * 2],
        ));
        specs.push(tensor(
            "mtp.norm.weight",
            &dtype,
            [architecture.hidden_size],
        ));
        for layer in 0..architecture.mtp_num_hidden_layers {
            let prefix = format!("mtp.layers.{layer}");
            push_common_decoder_layer_specs(&mut specs, &prefix, &dtype, architecture);
            push_full_attention_specs(&mut specs, &prefix, &dtype, architecture);
        }
    }

    specs.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(specs)
}

pub fn qwen36_weight_index_from_bytes(
    index_bytes: &[u8],
) -> Result<BTreeMap<String, String>, Qwen36TargetPathError> {
    let value = serde_json::from_slice::<Value>(index_bytes)?;
    let weight_map = value
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            Qwen36TargetPathError::InvalidConfig(String::from(
                "model.safetensors.index.json must contain weight_map",
            ))
        })?;
    let mut map = BTreeMap::new();
    for (tensor_name, shard_name) in weight_map {
        let Some(shard_name) = shard_name.as_str() else {
            return Err(Qwen36TargetPathError::InvalidConfig(String::from(
                "weight_map values must be shard path strings",
            )));
        };
        map.insert(tensor_name.clone(), String::from(shard_name));
    }
    if map.is_empty() {
        return Err(Qwen36TargetPathError::InvalidConfig(String::from(
            "model.safetensors.index.json has no tensor entries",
        )));
    }
    Ok(map)
}

pub fn qwen36_shard_paths_from_weight_map(
    model_dir: &Path,
    weight_map: &BTreeMap<String, String>,
) -> Result<Vec<PathBuf>, Qwen36TargetPathError> {
    let shard_names = weight_map.values().cloned().collect::<BTreeSet<_>>();
    let mut paths = Vec::with_capacity(shard_names.len());
    let mut missing = Vec::new();
    for shard_name in shard_names {
        let path = model_dir.join(&shard_name);
        if path.is_file() {
            paths.push(path);
        } else {
            missing.push(shard_name);
        }
    }
    if !missing.is_empty() {
        return Err(Qwen36TargetPathError::InvalidConfig(format!(
            "model directory is incomplete; missing safetensors shards: {}",
            missing.join(", ")
        )));
    }
    Ok(paths)
}

pub fn qwen36_tensor_admission_report(
    expected_tensors: Vec<Qwen36TensorSpec>,
    weight_map: BTreeMap<String, String>,
    observed_tensors: Qwen36ObservedTensorSet,
) -> Qwen36TensorAdmissionReport {
    let expected_by_name = expected_tensors
        .iter()
        .map(|spec| (spec.name.clone(), spec.clone()))
        .collect::<BTreeMap<_, _>>();
    let observed_by_name = observed_tensors
        .tensors
        .iter()
        .map(|spec| (spec.name.clone(), spec.clone()))
        .collect::<BTreeMap<_, _>>();
    let index_names = weight_map.keys().cloned().collect::<BTreeSet<_>>();
    let header_names = observed_by_name.keys().cloned().collect::<BTreeSet<_>>();

    let mut missing_expected_tensors = Vec::new();
    let mut shape_mismatches = Vec::new();
    let mut dtype_mismatches = Vec::new();
    let mut admitted_text_tensor_count = 0usize;
    for expected in &expected_tensors {
        if !index_names.contains(&expected.name) {
            missing_expected_tensors.push(expected.clone());
            continue;
        }
        let Some(observed) = observed_by_name.get(&expected.name) else {
            missing_expected_tensors.push(expected.clone());
            continue;
        };
        let mut admitted = true;
        if observed.shape != expected.shape {
            shape_mismatches.push(Qwen36TensorShapeMismatch {
                name: expected.name.clone(),
                expected_shape: expected.shape.clone(),
                observed_shape: observed.shape.clone(),
            });
            admitted = false;
        }
        if observed.dtype != expected.dtype {
            dtype_mismatches.push(Qwen36TensorDtypeMismatch {
                name: expected.name.clone(),
                expected_dtype: expected.dtype.clone(),
                observed_dtype: observed.dtype.clone(),
            });
            admitted = false;
        }
        if admitted {
            admitted_text_tensor_count += 1;
        }
    }

    let index_tensors_missing_from_headers = index_names
        .difference(&header_names)
        .cloned()
        .collect::<Vec<_>>();
    let header_tensors_missing_from_index = header_names
        .difference(&index_names)
        .cloned()
        .collect::<Vec<_>>();
    let visual_or_other_observed_tensors = header_names
        .difference(&expected_by_name.keys().cloned().collect::<BTreeSet<_>>())
        .cloned()
        .collect::<Vec<_>>();
    let text_tensor_admission_passed = missing_expected_tensors.is_empty()
        && index_tensors_missing_from_headers.is_empty()
        && header_tensors_missing_from_index.is_empty()
        && shape_mismatches.is_empty()
        && dtype_mismatches.is_empty();

    Qwen36TensorAdmissionReport {
        expected_text_tensor_count: expected_tensors.len(),
        observed_index_tensor_count: index_names.len(),
        observed_header_tensor_count: header_names.len(),
        admitted_text_tensor_count,
        visual_or_other_observed_tensor_count: visual_or_other_observed_tensors.len(),
        shard_headers: observed_tensors.shards,
        missing_expected_tensors,
        index_tensors_missing_from_headers,
        header_tensors_missing_from_index,
        visual_or_other_observed_tensors,
        shape_mismatches,
        dtype_mismatches,
        text_tensor_admission_passed,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen36ObservedTensorSet {
    pub shards: Vec<Qwen36SafetensorsShardHeaderReport>,
    pub tensors: Vec<Qwen36ObservedTensorSpec>,
}

pub fn qwen36_observed_tensors_from_shards(
    shard_paths: &[PathBuf],
) -> Result<Qwen36ObservedTensorSet, Qwen36TargetPathError> {
    let mut shards = Vec::new();
    let mut tensors = Vec::new();
    for path in shard_paths {
        let (shard, mut shard_tensors) = qwen36_safetensors_header(path)?;
        shards.push(shard);
        tensors.append(&mut shard_tensors);
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    shards.sort_by(|left, right| left.shard_name.cmp(&right.shard_name));
    Ok(Qwen36ObservedTensorSet { shards, tensors })
}

fn qwen36_safetensors_header(
    path: &Path,
) -> Result<
    (
        Qwen36SafetensorsShardHeaderReport,
        Vec<Qwen36ObservedTensorSpec>,
    ),
    Qwen36TargetPathError,
> {
    let mut file = File::open(path).map_err(|source| Qwen36TargetPathError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|source| Qwen36TargetPathError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(len_bytes);
    let header_len = usize::try_from(header_len).map_err(|_| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "safetensors header is too large in `{}`",
            path.display()
        ))
    })?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|source| Qwen36TargetPathError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let byte_len = file
        .metadata()
        .map_err(|source| Qwen36TargetPathError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let value = serde_json::from_slice::<Value>(header_bytes.as_slice())?;
    let object = value.as_object().ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "safetensors header is not a JSON object in `{}`",
            path.display()
        ))
    })?;
    let shard_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string();
    let mut tensors = Vec::new();
    for (name, tensor_value) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = required_string(tensor_value, "dtype")?;
        let shape = tensor_value
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                Qwen36TargetPathError::InvalidConfig(format!(
                    "tensor `{name}` in `{}` is missing shape",
                    path.display()
                ))
            })?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        Qwen36TargetPathError::InvalidConfig(format!(
                            "tensor `{name}` in `{}` has a non-usize shape entry",
                            path.display()
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        tensors.push(Qwen36ObservedTensorSpec {
            name: name.clone(),
            dtype,
            shape,
            shard: shard_name.clone(),
        });
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let shard = Qwen36SafetensorsShardHeaderReport {
        path: path.display().to_string(),
        shard_name,
        byte_len,
        header_sha256: sha256_hex(header_bytes.as_slice()),
        tensor_count: tensors.len(),
    };
    Ok((shard, tensors))
}

fn should_run_sampled_projection(backend: &str) -> bool {
    backend == QWEN36_SAMPLED_PROJECTION_BACKEND || backend == "local-real-sampled-projection"
}

fn qwen36_sampled_projection_receipt(
    model_dir: &Path,
    weight_map: &BTreeMap<String, String>,
    architecture: &Qwen36ForwardArchitectureReport,
    rendered: &crate::Qwen36RenderedPrompt,
) -> Result<Qwen36ForwardProjectionReceipt, Qwen36TargetPathError> {
    let input_token_id = rendered.token_ids.last().copied().ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(String::from(
            "sampled Qwen3.6 projection requires tokenizer token ids",
        ))
    })?;
    let candidate_token_ids =
        qwen36_sampled_candidate_token_ids(input_token_id, architecture.vocab_size);
    let embed_row = qwen36_read_indexed_tensor_row(
        model_dir,
        weight_map,
        QWEN36_EMBED_TOKENS_WEIGHT,
        input_token_id as usize,
    )?;
    if embed_row.shape != [architecture.vocab_size, architecture.hidden_size] {
        return Err(Qwen36TargetPathError::InvalidConfig(format!(
            "unexpected shape for {QWEN36_EMBED_TOKENS_WEIGHT}: {:?}",
            embed_row.shape
        )));
    }
    let mut tensor_reads = vec![embed_row.receipt.clone()];
    let mut sampled_logits = Vec::with_capacity(candidate_token_ids.len());
    for token_id in &candidate_token_ids {
        let lm_row = qwen36_read_indexed_tensor_row(
            model_dir,
            weight_map,
            QWEN36_LM_HEAD_WEIGHT,
            *token_id as usize,
        )?;
        if lm_row.shape != [architecture.vocab_size, architecture.hidden_size] {
            return Err(Qwen36TargetPathError::InvalidConfig(format!(
                "unexpected shape for {QWEN36_LM_HEAD_WEIGHT}: {:?}",
                lm_row.shape
            )));
        }
        if lm_row.values.len() != embed_row.values.len() {
            return Err(Qwen36TargetPathError::InvalidConfig(format!(
                "Qwen3.6 sampled projection width mismatch: embed {} versus lm_head {}",
                embed_row.values.len(),
                lm_row.values.len()
            )));
        }
        let logit = embed_row
            .values
            .iter()
            .zip(lm_row.values.iter())
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum::<f64>();
        tensor_reads.push(lm_row.receipt);
        sampled_logits.push(Qwen36SampledLogit {
            token_id: *token_id,
            token_label: token_label(*token_id),
            logit,
        });
    }
    let logits_sha256 = sha256_json(&sampled_logits)?;
    Ok(Qwen36ForwardProjectionReceipt {
        mode: String::from("sampled_embed_lm_head_projection_v1"),
        input_token_id,
        input_token_label: token_label(input_token_id),
        candidate_token_ids,
        sampled_logits,
        logits_sha256,
        tensor_reads,
        hidden_size: architecture.hidden_size,
        used_full_transformer_layers: false,
    })
}

#[derive(Clone, Debug, PartialEq)]
struct Qwen36TensorRow {
    values: Vec<f32>,
    shape: Vec<usize>,
    receipt: Qwen36TensorRowReadReceipt,
}

fn qwen36_read_indexed_tensor_row(
    model_dir: &Path,
    weight_map: &BTreeMap<String, String>,
    tensor_name: &str,
    row_index: usize,
) -> Result<Qwen36TensorRow, Qwen36TargetPathError> {
    let shard_name = weight_map.get(tensor_name).ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "model index does not contain required tensor `{tensor_name}`"
        ))
    })?;
    let shard_path = model_dir.join(shard_name);
    let mut file = File::open(&shard_path).map_err(|source| Qwen36TargetPathError::Io {
        path: shard_path.clone(),
        source,
    })?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|source| Qwen36TargetPathError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let header_len = u64::from_le_bytes(len_bytes);
    let header_len_usize = usize::try_from(header_len).map_err(|_| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "safetensors header is too large in `{}`",
            shard_path.display()
        ))
    })?;
    let mut header_bytes = vec![0u8; header_len_usize];
    file.read_exact(&mut header_bytes)
        .map_err(|source| Qwen36TargetPathError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let header = serde_json::from_slice::<Value>(header_bytes.as_slice())?;
    let tensor_value = header.get(tensor_name).ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "safetensors shard `{}` does not contain tensor `{tensor_name}`",
            shard_path.display()
        ))
    })?;
    let dtype = required_string(tensor_value, "dtype")?;
    let shape = tensor_value
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Qwen36TargetPathError::InvalidConfig(format!(
                "tensor `{tensor_name}` in `{}` is missing shape",
                shard_path.display()
            ))
        })?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    Qwen36TargetPathError::InvalidConfig(format!(
                        "tensor `{tensor_name}` in `{}` has a non-usize shape entry",
                        shard_path.display()
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.len() != 2 {
        return Err(Qwen36TargetPathError::InvalidConfig(format!(
            "sampled projection can only read 2D tensor rows; `{tensor_name}` has shape {shape:?}"
        )));
    }
    if row_index >= shape[0] {
        return Err(Qwen36TargetPathError::InvalidConfig(format!(
            "row {row_index} is outside tensor `{tensor_name}` with {} rows",
            shape[0]
        )));
    }
    let offsets = tensor_value
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Qwen36TargetPathError::InvalidConfig(format!(
                "tensor `{tensor_name}` in `{}` is missing data_offsets",
                shard_path.display()
            ))
        })?;
    if offsets.len() != 2 {
        return Err(Qwen36TargetPathError::InvalidConfig(format!(
            "tensor `{tensor_name}` in `{}` must have two data_offsets",
            shard_path.display()
        )));
    }
    let data_offset_start = offsets[0].as_u64().ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "tensor `{tensor_name}` in `{}` has invalid data_offsets",
            shard_path.display()
        ))
    })?;
    let data_offset_end = offsets[1].as_u64().ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "tensor `{tensor_name}` in `{}` has invalid data_offsets",
            shard_path.display()
        ))
    })?;
    let element_bytes = safetensors_element_size(dtype.as_str())?;
    let row_bytes_len = shape[1].checked_mul(element_bytes).ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "tensor `{tensor_name}` row byte length overflow"
        ))
    })?;
    let tensor_bytes_len = data_offset_end
        .checked_sub(data_offset_start)
        .ok_or_else(|| {
            Qwen36TargetPathError::InvalidConfig(format!(
                "tensor `{tensor_name}` has reversed data_offsets"
            ))
        })?;
    let expected_tensor_bytes = (shape[0] as u64)
        .checked_mul(row_bytes_len as u64)
        .ok_or_else(|| {
            Qwen36TargetPathError::InvalidConfig(format!(
                "tensor `{tensor_name}` data byte length overflow"
            ))
        })?;
    if tensor_bytes_len != expected_tensor_bytes {
        return Err(Qwen36TargetPathError::InvalidConfig(format!(
            "tensor `{tensor_name}` data_offsets describe {tensor_bytes_len} bytes, expected {expected_tensor_bytes}"
        )));
    }
    let data_start = 8u64.checked_add(header_len).ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(format!(
            "safetensors data start overflow in `{}`",
            shard_path.display()
        ))
    })?;
    let row_offset = data_start
        .checked_add(data_offset_start)
        .and_then(|value| value.checked_add((row_index as u64).checked_mul(row_bytes_len as u64)?))
        .ok_or_else(|| {
            Qwen36TargetPathError::InvalidConfig(format!(
                "tensor `{tensor_name}` row offset overflow"
            ))
        })?;
    file.seek(SeekFrom::Start(row_offset))
        .map_err(|source| Qwen36TargetPathError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let mut row_bytes = vec![0u8; row_bytes_len];
    file.read_exact(row_bytes.as_mut_slice())
        .map_err(|source| Qwen36TargetPathError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let values = decode_safetensors_row(dtype.as_str(), row_bytes.as_slice())?;
    let receipt = Qwen36TensorRowReadReceipt {
        tensor_name: String::from(tensor_name),
        shard_name: shard_name.clone(),
        shard_path: shard_path.display().to_string(),
        row_index,
        dtype,
        shape: shape.clone(),
        row_sha256: sha256_hex(row_bytes.as_slice()),
    };
    Ok(Qwen36TensorRow {
        values,
        shape,
        receipt,
    })
}

fn qwen36_sampled_candidate_token_ids(input_token_id: u32, vocab_size: usize) -> Vec<u32> {
    let mut ids = vec![input_token_id, 0, 1, 2, 3, 4, 5];
    ids.retain(|id| usize::try_from(*id).is_ok_and(|id| id < vocab_size));
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn safetensors_element_size(dtype: &str) -> Result<usize, Qwen36TargetPathError> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" => Ok(4),
        other => Err(Qwen36TargetPathError::SafeTensors(format!(
            "unsupported sampled projection dtype `{other}`"
        ))),
    }
}

fn decode_safetensors_row(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, Qwen36TargetPathError> {
    match dtype {
        "BF16" => {
            if bytes.len() % 2 != 0 {
                return Err(Qwen36TargetPathError::SafeTensors(String::from(
                    "BF16 row byte length is not divisible by 2",
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                .collect())
        }
        "F16" => {
            if bytes.len() % 2 != 0 {
                return Err(Qwen36TargetPathError::SafeTensors(String::from(
                    "F16 row byte length is not divisible by 2",
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                .collect())
        }
        "F32" => {
            if bytes.len() % 4 != 0 {
                return Err(Qwen36TargetPathError::SafeTensors(String::from(
                    "F32 row byte length is not divisible by 4",
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect())
        }
        other => Err(Qwen36TargetPathError::SafeTensors(format!(
            "unsupported sampled projection dtype `{other}`"
        ))),
    }
}

fn token_label(token_id: u32) -> String {
    format!("token:{token_id}")
}

fn push_common_decoder_layer_specs(
    specs: &mut Vec<Qwen36TensorSpec>,
    prefix: &str,
    dtype: &str,
    architecture: &Qwen36ForwardArchitectureReport,
) {
    specs.push(tensor(
        format!("{prefix}.input_layernorm.weight"),
        dtype,
        [architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.post_attention_layernorm.weight"),
        dtype,
        [architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.mlp.gate_proj.weight"),
        dtype,
        [architecture.intermediate_size, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.mlp.up_proj.weight"),
        dtype,
        [architecture.intermediate_size, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.mlp.down_proj.weight"),
        dtype,
        [architecture.hidden_size, architecture.intermediate_size],
    ));
}

fn push_full_attention_specs(
    specs: &mut Vec<Qwen36TensorSpec>,
    prefix: &str,
    dtype: &str,
    architecture: &Qwen36ForwardArchitectureReport,
) {
    let q_proj_out = architecture.num_attention_heads * architecture.head_dim * 2;
    let kv_proj_out = architecture.num_key_value_heads * architecture.head_dim;
    let o_proj_in = architecture.num_attention_heads * architecture.head_dim;
    specs.push(tensor(
        format!("{prefix}.self_attn.q_proj.weight"),
        dtype,
        [q_proj_out, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.self_attn.k_proj.weight"),
        dtype,
        [kv_proj_out, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.self_attn.v_proj.weight"),
        dtype,
        [kv_proj_out, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.self_attn.o_proj.weight"),
        dtype,
        [architecture.hidden_size, o_proj_in],
    ));
    specs.push(tensor(
        format!("{prefix}.self_attn.q_norm.weight"),
        dtype,
        [architecture.head_dim],
    ));
    specs.push(tensor(
        format!("{prefix}.self_attn.k_norm.weight"),
        dtype,
        [architecture.head_dim],
    ));
}

fn push_linear_attention_specs(
    specs: &mut Vec<Qwen36TensorSpec>,
    prefix: &str,
    dtype: &str,
    architecture: &Qwen36ForwardArchitectureReport,
) -> Result<(), Qwen36TargetPathError> {
    let key_heads = architecture.linear_num_key_heads.ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(String::from(
            "linear_attention layers require linear_num_key_heads",
        ))
    })?;
    let value_heads = architecture.linear_num_value_heads.ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(String::from(
            "linear_attention layers require linear_num_value_heads",
        ))
    })?;
    let key_dim = architecture.linear_key_head_dim.ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(String::from(
            "linear_attention layers require linear_key_head_dim",
        ))
    })?;
    let value_dim = architecture.linear_value_head_dim.ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(String::from(
            "linear_attention layers require linear_value_head_dim",
        ))
    })?;
    let conv_kernel = architecture.linear_conv_kernel_dim.ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(String::from(
            "linear_attention layers require linear_conv_kernel_dim",
        ))
    })?;
    let key_width = key_heads * key_dim;
    let value_width = value_heads * value_dim;
    let qkv_width = key_width * 2 + value_width;
    specs.push(tensor(
        format!("{prefix}.linear_attn.A_log"),
        dtype,
        [value_heads],
    ));
    specs.push(tensor(
        format!("{prefix}.linear_attn.dt_bias"),
        dtype,
        [value_heads],
    ));
    specs.push(tensor(
        format!("{prefix}.linear_attn.in_proj_a.weight"),
        dtype,
        [value_heads, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.linear_attn.in_proj_b.weight"),
        dtype,
        [value_heads, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.linear_attn.in_proj_qkv.weight"),
        dtype,
        [qkv_width, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.linear_attn.in_proj_z.weight"),
        dtype,
        [value_width, architecture.hidden_size],
    ));
    specs.push(tensor(
        format!("{prefix}.linear_attn.out_proj.weight"),
        dtype,
        [architecture.hidden_size, value_width],
    ));
    specs.push(tensor(
        format!("{prefix}.linear_attn.norm.weight"),
        dtype,
        [value_dim],
    ));
    specs.push(tensor(
        format!("{prefix}.linear_attn.conv1d.weight"),
        dtype,
        [qkv_width, 1, conv_kernel],
    ));
    Ok(())
}

fn tensor<const N: usize>(
    name: impl Into<String>,
    dtype: &str,
    shape: [usize; N],
) -> Qwen36TensorSpec {
    Qwen36TensorSpec {
        name: name.into(),
        dtype: String::from(dtype),
        shape: shape.to_vec(),
    }
}

fn required_string(value: &Value, key: &str) -> Result<String, Qwen36TargetPathError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(String::from)
        .ok_or_else(|| {
            Qwen36TargetPathError::InvalidConfig(format!("Qwen3.6 config is missing `{key}`"))
        })
}

fn optional_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(String::from)
}

fn required_usize(value: &Value, key: &str) -> Result<usize, Qwen36TargetPathError> {
    optional_usize(value, key).ok_or_else(|| {
        Qwen36TargetPathError::InvalidConfig(format!("Qwen3.6 config is missing numeric `{key}`"))
    })
}

fn optional_usize(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn hf_dtype_to_safetensors_dtype(dtype: &str) -> String {
    match dtype {
        "bfloat16" => String::from("BF16"),
        "float16" => String::from("F16"),
        "float32" => String::from("F32"),
        other => other.to_ascii_uppercase(),
    }
}

fn read_bytes(path: &Path) -> Result<Vec<u8>, Qwen36TargetPathError> {
    fs::read(path).map_err(|source| Qwen36TargetPathError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, Qwen36TargetPathError> {
    let bytes = serde_json::to_vec(value)?;
    Ok(sha256_hex(bytes.as_slice()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen36_forward_architecture_parses_real_text_fields() {
        let config = real_shape_config_json();

        let report = qwen36_forward_architecture_report(config.as_bytes()).expect("architecture");

        assert_eq!(report.root_model_type, "qwen3_5");
        assert_eq!(report.text_model_type, "qwen3_5_text");
        assert_eq!(report.num_hidden_layers, 4);
        assert_eq!(report.full_attention_layers, vec![3]);
        assert_eq!(report.linear_attention_layers, vec![0, 1, 2]);
        assert_eq!(report.linear_num_value_heads, Some(48));
        assert_eq!(report.mtp_num_hidden_layers, 1);
        assert!(report.rope_parameters_hash.is_some());
    }

    #[test]
    fn qwen36_forward_expected_table_covers_linear_full_and_mtp_tensors() {
        let architecture = qwen36_forward_architecture_report(real_shape_config_json().as_bytes())
            .expect("architecture");

        let specs = qwen36_expected_text_tensor_specs(&architecture).expect("specs");
        let names = specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<BTreeSet<_>>();

        assert_eq!(specs.len(), 3 + 3 * 14 + 11 + 15);
        assert!(names.contains("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"));
        assert!(names.contains("model.language_model.layers.3.self_attn.q_proj.weight"));
        assert!(names.contains("mtp.fc.weight"));
        let q_proj = specs
            .iter()
            .find(|spec| spec.name == "model.language_model.layers.3.self_attn.q_proj.weight")
            .expect("q_proj");
        assert_eq!(q_proj.shape, vec![12288, 5120]);
        let linear_qkv = specs
            .iter()
            .find(|spec| {
                spec.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            })
            .expect("linear qkv");
        assert_eq!(linear_qkv.shape, vec![10240, 5120]);
    }

    #[test]
    fn qwen36_forward_admission_reports_shape_and_non_text_tensors() {
        let temp = tempfile::tempdir().expect("tempdir");
        let shard_path = temp.path().join("model-00001-of-00001.safetensors");
        write_test_safetensors(&shard_path).expect("write shard");
        let architecture = qwen36_forward_architecture_report(real_shape_config_json().as_bytes())
            .expect("architecture");
        let mut expected = qwen36_expected_text_tensor_specs(&architecture).expect("specs");
        expected.retain(|spec| {
            matches!(
                spec.name.as_str(),
                "lm_head.weight"
                    | "model.language_model.embed_tokens.weight"
                    | "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            )
        });
        let observed =
            qwen36_observed_tensors_from_shards(&[shard_path.clone()]).expect("observed");
        let mut weight_map = BTreeMap::new();
        for spec in &observed.tensors {
            weight_map.insert(
                spec.name.clone(),
                String::from("model-00001-of-00001.safetensors"),
            );
        }

        let report = qwen36_tensor_admission_report(expected, weight_map, observed);

        assert_eq!(report.admitted_text_tensor_count, 2);
        assert_eq!(report.shape_mismatches.len(), 1);
        assert_eq!(
            report.shape_mismatches[0].name,
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(report.visual_or_other_observed_tensor_count, 1);
        assert!(!report.text_tensor_admission_passed);
    }

    #[test]
    fn qwen36_sampled_projection_reads_real_rows_and_hashes_logits() {
        let temp = tempfile::tempdir().expect("tempdir");
        let shard_path = temp.path().join("model-00001-of-00001.safetensors");
        write_projection_safetensors(&shard_path).expect("write projection shard");
        let mut weight_map = BTreeMap::new();
        weight_map.insert(
            String::from(QWEN36_EMBED_TOKENS_WEIGHT),
            String::from("model-00001-of-00001.safetensors"),
        );
        weight_map.insert(
            String::from(QWEN36_LM_HEAD_WEIGHT),
            String::from("model-00001-of-00001.safetensors"),
        );
        let architecture = projection_test_architecture();
        let rendered = crate::Qwen36RenderedPrompt {
            template_id: String::from("test"),
            reasoning_mode: Qwen36ReasoningMode::DirectAnswer,
            text: String::from("test"),
            prompt_hash: String::from("hash"),
            token_ids: vec![6],
        };

        let receipt =
            qwen36_sampled_projection_receipt(temp.path(), &weight_map, &architecture, &rendered)
                .expect("projection");

        assert_eq!(receipt.input_token_id, 6);
        assert_eq!(receipt.candidate_token_ids, vec![0, 1, 2, 3, 4, 5, 6]);
        assert!(!receipt.used_full_transformer_layers);
        assert_eq!(receipt.tensor_reads.len(), 8);
        let token6 = receipt
            .sampled_logits
            .iter()
            .find(|logit| logit.token_id == 6)
            .expect("token 6 logit");
        assert_eq!(token6.logit, 30.0);
        assert_eq!(
            receipt.logits_sha256,
            sha256_json(&receipt.sampled_logits).expect("logits hash")
        );
    }

    fn write_test_safetensors(path: &Path) -> Result<(), Qwen36TargetPathError> {
        let header = serde_json::json!({
            "model.language_model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [248320, 5120],
                "data_offsets": [0, 1]
            },
            "lm_head.weight": {
                "dtype": "BF16",
                "shape": [248320, 5120],
                "data_offsets": [1, 2]
            },
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": {
                "dtype": "BF16",
                "shape": [8, 5120],
                "data_offsets": [2, 3]
            },
            "model.visual.patch_embed.proj.weight": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [3, 4]
            }
        });
        let header_bytes = serde_json::to_vec(&header)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_bytes.as_slice());
        fs::write(path, bytes).map_err(|source| Qwen36TargetPathError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn write_projection_safetensors(path: &Path) -> Result<(), Qwen36TargetPathError> {
        let vocab_size = 8usize;
        let hidden_size = 4usize;
        let tensor_bytes = vocab_size * hidden_size * 2;
        let header = serde_json::json!({
            QWEN36_EMBED_TOKENS_WEIGHT: {
                "dtype": "BF16",
                "shape": [vocab_size, hidden_size],
                "data_offsets": [0, tensor_bytes]
            },
            QWEN36_LM_HEAD_WEIGHT: {
                "dtype": "BF16",
                "shape": [vocab_size, hidden_size],
                "data_offsets": [tensor_bytes, tensor_bytes * 2]
            }
        });
        let mut data = Vec::new();
        for row in 0..vocab_size {
            for col in 0..hidden_size {
                let value = if row == 6 {
                    (col + 1) as f32
                } else {
                    row as f32
                };
                data.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        for row in 0..vocab_size {
            for col in 0..hidden_size {
                let value = if row == 6 {
                    (col + 1) as f32
                } else {
                    (row + col) as f32
                };
                data.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        let header_bytes = serde_json::to_vec(&header)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_bytes.as_slice());
        bytes.extend_from_slice(data.as_slice());
        fs::write(path, bytes).map_err(|source| Qwen36TargetPathError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn projection_test_architecture() -> Qwen36ForwardArchitectureReport {
        Qwen36ForwardArchitectureReport {
            root_model_type: String::from("qwen3_5"),
            text_model_type: String::from("qwen3_5_text"),
            architectures: vec![String::from("Qwen3_5ForConditionalGeneration")],
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            vocab_size: 8,
            max_position_embeddings: 16,
            torch_dtype: String::from("bfloat16"),
            rms_norm_eps: Some(0.000001),
            hidden_act: Some(String::from("silu")),
            layer_types: vec![String::from("full_attention")],
            full_attention_layers: vec![0],
            linear_attention_layers: Vec::new(),
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            attn_output_gate: None,
            output_gate_type: None,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: None,
            rope_parameters_hash: None,
        }
    }

    fn real_shape_config_json() -> &'static str {
        r#"{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "dtype": "bfloat16",
    "hidden_act": "silu",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_5_text",
    "num_attention_heads": 24,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "rms_norm_eps": 0.000001,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_conv_kernel_dim": 4,
    "attn_output_gate": true,
    "output_gate_type": "swish",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    }
  }
}"#
    }
}

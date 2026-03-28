use std::{
    env, fs,
    path::{Path, PathBuf},
};

use psionic_data::{
    parameter_golf_sentencepiece_entries_from_tokenizer_path, ParameterGolfSentencePieceTokenKind,
};
use psionic_models::{
    ModelDescriptor, ParameterGolfConfig, ParameterGolfPromotedRuntimeTokenizer,
    ParameterGolfPromotedTokenizerAsset, ParameterGolfPromotedTokenizerAssetFormat,
    ParameterGolfPromotedTokenizerFamily, ParameterGolfPromotedTokenizerToken,
    ParameterGolfPromotedTokenizerTokenKind, ParameterGolfReferenceModel, ParameterGolfWeights,
    TokenId, TokenSequence, TokenizerBoundary,
    PARAMETER_GOLF_PROMOTED_TOKENIZER_ASSET_SCHEMA_VERSION,
};
use psionic_runtime::{SamplingPolicy, SamplingStrategy, TokenSampler};
use psionic_train::{
    restore_parameter_golf_model_from_quantized_artifact, ParameterGolfSingleH100ModelVariant,
    ParameterGolfSingleH100TrainingReport,
};
use serde::Serialize;

#[derive(Serialize)]
struct PromptReport {
    report_path: String,
    artifact_path: String,
    run_id: String,
    machine_profile: String,
    model_variant: String,
    prompt: String,
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    generated_text: String,
    termination: String,
    final_validation_bits_per_byte: Option<f64>,
    compressed_model_bytes: Option<u64>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let report_path = PathBuf::from(args.first().ok_or_else(usage_error)?);
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| String::from("the meaning of life is"));
    let max_new_tokens = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("32")
        .parse::<usize>()?;
    let output_path = args.get(3).map(PathBuf::from);

    let report: ParameterGolfSingleH100TrainingReport =
        serde_json::from_slice(&fs::read(&report_path)?)?;
    let artifact_path = report
        .compressed_model_artifact_path
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_artifact_path(report_path.as_path()));
    let artifact_bytes = fs::read(&artifact_path)?;
    let baseline_model = baseline_model_from_report(&report)?;
    let model = restore_parameter_golf_model_from_quantized_artifact(
        &baseline_model,
        artifact_bytes.as_slice(),
    )?;
    let tokenizer = runtime_tokenizer_from_tokenizer_path(&report.tokenizer_path)?;
    let output = generate_text(&model, &tokenizer, prompt.as_str(), max_new_tokens)?;
    let prompt_report = PromptReport {
        report_path: report_path.display().to_string(),
        artifact_path: artifact_path.display().to_string(),
        run_id: report.run_id,
        machine_profile: report.machine_profile.as_str().to_string(),
        model_variant: report.model_variant.as_str().to_string(),
        prompt,
        prompt_tokens: output
            .prompt_tokens
            .as_slice()
            .iter()
            .map(|token| token.as_u32())
            .collect(),
        generated_tokens: output
            .generated_tokens
            .as_slice()
            .iter()
            .map(|token| token.as_u32())
            .collect(),
        generated_text: output.generated_text,
        termination: output.termination,
        final_validation_bits_per_byte: report
            .final_validation
            .as_ref()
            .map(|summary| summary.bits_per_byte),
        compressed_model_bytes: report.compressed_model_bytes,
    };
    let encoded = serde_json::to_vec_pretty(&prompt_report)?;
    if let Some(output_path) = output_path {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output_path, &encoded)?;
        println!("{}", output_path.display());
    } else {
        println!("{}", String::from_utf8(encoded)?);
    }
    Ok(())
}

struct GeneratedTextOutput {
    prompt_tokens: TokenSequence,
    generated_tokens: TokenSequence,
    generated_text: String,
    termination: String,
}

fn generate_text(
    model: &ParameterGolfReferenceModel,
    tokenizer: &ParameterGolfPromotedRuntimeTokenizer,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<GeneratedTextOutput, Box<dyn std::error::Error>> {
    let prompt_tokens = tokenizer.encode_with_defaults(prompt);
    if prompt_tokens.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }
    if prompt_tokens.len() > model.descriptor().config.max_context {
        return Err(format!(
            "prompt has {} tokens, exceeding max_context {}",
            prompt_tokens.len(),
            model.descriptor().config.max_context
        )
        .into());
    }
    let mut history = prompt_tokens
        .as_slice()
        .iter()
        .map(|token| token.as_u32())
        .collect::<Vec<_>>();
    let mut generated_tokens = Vec::with_capacity(max_new_tokens);
    let mut sampler = TokenSampler::new(&SamplingPolicy {
        strategy: SamplingStrategy::Greedy,
        temperature: None,
        top_k: None,
        top_p: None,
        min_p: None,
        repeat_penalty: None,
        repeat_last_n: None,
        presence_penalty: None,
        frequency_penalty: None,
        seed: None,
    });
    let allowed = tokenizer.generation_allowed_token_ids();
    let termination = loop {
        if generated_tokens.len() >= max_new_tokens {
            break String::from("max_new_tokens");
        }
        if history.len() >= model.descriptor().config.max_context {
            break String::from("context_limit");
        }
        let logits = model.forward_logits(std::slice::from_ref(&history))?;
        let width = logits.width();
        let sequence_length = logits.sequence_length();
        let last_row_start = sequence_length
            .checked_sub(1)
            .ok_or("missing logits rows")?
            .saturating_mul(width);
        let last_logits = logits
            .values()
            .get(last_row_start..last_row_start.saturating_add(width))
            .ok_or("missing logits slice")?;
        let next_token = sampler
            .select_next_token_with_allowed(last_logits, history.as_slice(), allowed)
            .ok_or("missing next token")?;
        let next_token = TokenId(next_token);
        history.push(next_token.as_u32());
        generated_tokens.push(next_token);
        if tokenizer.is_end_of_sequence(next_token) {
            break String::from("end_of_sequence");
        }
    };
    let generated_tokens = TokenSequence::new(generated_tokens);
    Ok(GeneratedTextOutput {
        prompt_tokens,
        generated_text: tokenizer.decode(generated_tokens.as_slice()),
        generated_tokens,
        termination,
    })
}

fn baseline_model_from_report(
    report: &ParameterGolfSingleH100TrainingReport,
) -> Result<ParameterGolfReferenceModel, Box<dyn std::error::Error>> {
    let model_id = match report.model_variant {
        ParameterGolfSingleH100ModelVariant::BaselineSp1024_9x512 => {
            "parameter-golf-baseline-sp1024-9x512"
        }
        ParameterGolfSingleH100ModelVariant::CompetitiveHomegolfV1 => {
            "parameter-golf-homegolf-competitive-v1"
        }
    };
    let config: ParameterGolfConfig = report.model_config.clone();
    let weights = ParameterGolfWeights::from_initializer(&config, Default::default())?;
    Ok(ParameterGolfReferenceModel::new(
        ModelDescriptor::new(model_id, "parameter_golf_decoder", "v1"),
        config,
        weights,
    )?)
}

fn runtime_tokenizer_from_tokenizer_path(
    tokenizer_path: &Path,
) -> Result<ParameterGolfPromotedRuntimeTokenizer, Box<dyn std::error::Error>> {
    let pieces = parameter_golf_sentencepiece_entries_from_tokenizer_path(tokenizer_path)?
        .into_iter()
        .map(|entry| ParameterGolfPromotedTokenizerToken {
            token_id: entry.token_id,
            piece: entry.piece,
            kind: promoted_token_kind(entry.kind),
        })
        .collect::<Vec<_>>();
    let mut asset = ParameterGolfPromotedTokenizerAsset {
        schema_version: String::from(PARAMETER_GOLF_PROMOTED_TOKENIZER_ASSET_SCHEMA_VERSION),
        profile_id: String::from("parameter_golf.homegolf_exact_prompt"),
        tokenizer_id: String::from("parameter_golf.homegolf.challenge_sp1024_sentencepiece"),
        tokenizer_version: String::from("live-local"),
        family: ParameterGolfPromotedTokenizerFamily::SentencePiece,
        asset_format: ParameterGolfPromotedTokenizerAssetFormat::SentencePiecePieceTableJson,
        vocab_size: pieces.len() as u32,
        add_bos: false,
        add_eos: false,
        bos_token_id: None,
        eos_token_ids: Vec::new(),
        pad_token_id: None,
        unknown_token_id: Some(0),
        pieces,
        tokenizer_digest: String::new(),
        asset_digest: String::new(),
        detail: String::from(
            "Runtime-loadable exact SP1024 tokenizer asset for HOMEGOLF prompt validation.",
        ),
    };
    asset.tokenizer_digest = asset.tokenizer_contract_digest();
    asset.asset_digest = asset.stable_digest();
    Ok(ParameterGolfPromotedRuntimeTokenizer::from_asset(asset)?)
}

fn promoted_token_kind(
    kind: ParameterGolfSentencePieceTokenKind,
) -> ParameterGolfPromotedTokenizerTokenKind {
    match kind {
        ParameterGolfSentencePieceTokenKind::Normal => {
            ParameterGolfPromotedTokenizerTokenKind::Normal
        }
        ParameterGolfSentencePieceTokenKind::Byte => ParameterGolfPromotedTokenizerTokenKind::Byte,
        ParameterGolfSentencePieceTokenKind::Control => {
            ParameterGolfPromotedTokenizerTokenKind::Control
        }
        ParameterGolfSentencePieceTokenKind::Unknown => {
            ParameterGolfPromotedTokenizerTokenKind::Unknown
        }
        ParameterGolfSentencePieceTokenKind::Unused => {
            ParameterGolfPromotedTokenizerTokenKind::Unused
        }
    }
}

fn default_artifact_path(report_path: &Path) -> PathBuf {
    let stem = report_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("parameter_golf_single_device_training");
    report_path.with_file_name(format!("{stem}.final_model.st"))
}

fn usage_error() -> Box<dyn std::error::Error> {
    "usage: cargo run -q -p psionic-train --bin parameter_golf_homegolf_prompt -- <training_report.json> [prompt] [max_new_tokens] [output.json]".into()
}

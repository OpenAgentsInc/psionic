use std::{
    env,
    path::PathBuf,
    process::ExitCode,
};

use psionic_models::{
    GgufDecoderAdapterLoader, PromptMessage, PromptMessageRole, PromptRenderOptions,
};
use psionic_serve::{
    CudaGgufQwen35TextGenerationService, GenerationOptions, GenerationRequest,
    TextGenerationExecutor,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(model_path) = args.next().map(PathBuf::from) else {
        return Err(usage());
    };
    let prompt = args
        .next()
        .unwrap_or_else(|| String::from("Explain what Psionic is in one sentence."));
    let max_output_tokens = args
        .next()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid max_output_tokens `{value}`: {error}"))
        })
        .transpose()?
        .unwrap_or(256);
    let repeats = args
        .next()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid repeats `{value}`: {error}"))
        })
        .transpose()?
        .unwrap_or(3)
        .max(1);

    let adapter = GgufDecoderAdapterLoader
        .load_path(&model_path)
        .map_err(|error| format!("failed to load GGUF metadata: {error}"))?;
    let renderer = adapter.prompt_renderer();
    let rendered = renderer
        .render_with_options(
            None,
            &[PromptMessage::new(PromptMessageRole::User, prompt.clone())],
            true,
            &PromptRenderOptions::default(),
        )
        .map_err(|error| format!("failed to render qwen35 prompt: {error}"))?;

    let mut service = CudaGgufQwen35TextGenerationService::from_gguf_path(&model_path)
        .map_err(|error| format!("failed to load qwen35 cuda service: {error}"))?;
    let descriptor = service.model_descriptor().clone();

    let mut warmup_options = GenerationOptions::greedy(16);
    warmup_options.stop_sequences = rendered.stop_sequences.clone();
    let warmup = GenerationRequest::new_text(
        "warmup",
        descriptor.clone(),
        None,
        rendered.text.clone(),
        warmup_options,
    );
    let _ = service
        .generate(&warmup)
        .map_err(|error| format!("warmup generation failed: {error}"))?;

    let mut decode_tok_s_total = 0.0_f64;
    for run_index in 0..repeats {
        let mut options = GenerationOptions::greedy(max_output_tokens);
        options.stop_sequences = rendered.stop_sequences.clone();
        let request = GenerationRequest::new_text(
            format!("bench-{run_index}"),
            descriptor.clone(),
            None,
            rendered.text.clone(),
            options,
        );
        let response = service
            .generate(&request)
            .map_err(|error| format!("benchmark generation failed: {error}"))?;
        let output_tokens = response.metrics.eval_count.unwrap_or(response.output.tokens.len());
        let decode_ns = response.metrics.eval_duration_ns.unwrap_or(0);
        let prompt_ns = response.metrics.prompt_eval_duration_ns.unwrap_or(0);
        let total_ns = response.metrics.total_duration_ns.unwrap_or(0);
        let decode_tok_s = tokens_per_second(output_tokens, decode_ns);
        decode_tok_s_total += decode_tok_s;
        println!(
            "run={} prompt_tokens={} output_tokens={} prompt_s={:.6} decode_s={:.6} total_s={:.6} decode_tok_s={:.2} output={}",
            run_index + 1,
            response.metrics.prompt_eval_count.unwrap_or(0),
            output_tokens,
            nanos_to_seconds(prompt_ns),
            nanos_to_seconds(decode_ns),
            nanos_to_seconds(total_ns),
            decode_tok_s,
            response.output.text.replace('\n', "\\n"),
        );
    }

    println!(
        "mean_decode_tok_s={:.2}",
        decode_tok_s_total / repeats as f64
    );
    Ok(())
}

fn tokens_per_second(tokens: usize, duration_ns: u64) -> f64 {
    if tokens == 0 || duration_ns == 0 {
        return 0.0;
    }
    tokens as f64 / nanos_to_seconds(duration_ns)
}

fn nanos_to_seconds(duration_ns: u64) -> f64 {
    duration_ns as f64 / 1_000_000_000.0
}

fn usage() -> String {
    String::from(
        "usage: cargo run -p psionic-serve --example qwen35_cuda_bench -- <model.gguf> [prompt] [max_output_tokens] [repeats]",
    )
}

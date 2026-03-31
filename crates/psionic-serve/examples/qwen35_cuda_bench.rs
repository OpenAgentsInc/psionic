use std::{
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use psionic_models::{
    GgufDecoderAdapterLoader, GgufRuntimeTokenizer, PromptMessage, PromptMessageRole,
    PromptRenderOptions, TokenId, TokenizerBoundary,
};
use psionic_runtime::{
    DEFAULT_PENALTY_LOOKBACK, PrefixCacheControl, PrefixCacheMode, StructuredOutputRequest,
    StructuredOutputValue,
};
use psionic_serve::{
    CudaGgufQwen35TextGenerationService, GenerationOptions, GenerationRequest, GenerationResponse,
    GenerationTerminationCause, Qwen35CudaDecodeOutputMetrics, TerminationReason,
    TextGenerationExecutor,
};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    let config = BenchConfig::parse(env::args().skip(1))?;
    match config.backend {
        BenchBackend::Psionic => run_psionic_benchmark(&config),
        BenchBackend::Ollama => run_ollama_benchmark(&config),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchBackend {
    Psionic,
    Ollama,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchDecodeMode {
    Greedy,
    Sample,
}

#[derive(Clone, Debug)]
struct BenchConfig {
    backend: BenchBackend,
    model_path: PathBuf,
    ollama_model: Option<String>,
    ollama_base_url: String,
    json_out: Option<PathBuf>,
    prompt: String,
    max_output_tokens: usize,
    repeats: usize,
    decode_mode: BenchDecodeMode,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    mirostat: Option<u8>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    structured_output: Option<BenchStructuredOutput>,
}

#[derive(Clone, Debug)]
enum BenchStructuredOutput {
    JsonObject,
    JsonSchema { name: Option<String>, schema: Value },
}

#[derive(Clone, Debug, Serialize)]
struct BenchReport {
    schema_version: u32,
    report_kind: String,
    benchmark_class: String,
    generated_at_unix_s: u64,
    backend: String,
    model_path: String,
    ollama_model: Option<String>,
    ollama_base_url: Option<String>,
    prompt: String,
    rendered_prompt: String,
    stop_sequences: Vec<String>,
    decode_mode: String,
    max_output_tokens: usize,
    repeats: usize,
    steady_state_concurrency: usize,
    load_s: Option<f64>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    mirostat: Option<u8>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    structured_output: BenchStructuredOutputConfigReport,
    psionic_cuda_startup: Option<BenchPsionicCudaStartupReport>,
    runs: Vec<BenchRunReport>,
    mean_output_tokens: f64,
    mean_prompt_s: f64,
    mean_decode_s: f64,
    mean_total_s: f64,
    mean_ttft_s: Option<f64>,
    mean_itl_s: Option<f64>,
    mean_decode_tok_s: f64,
}

#[derive(Clone, Debug, Serialize)]
struct BenchRunReport {
    run_index: usize,
    decode_mode: String,
    prompt_tokens: usize,
    output_tokens: usize,
    prompt_s: f64,
    decode_s: f64,
    total_s: f64,
    ttft_s: Option<f64>,
    itl_s: Option<f64>,
    decode_tok_s: f64,
    qwen35_output_modes: Vec<String>,
    qwen35_readback_bytes: u64,
    qwen35_raw_logits: bool,
    qwen35_graph_hits: usize,
    qwen35_graph_misses: usize,
    qwen35_graph_captures: usize,
    qwen35_graph_shape_drifts: usize,
    termination: BenchTerminationReport,
    structured_output_mode: String,
    structured_output_parser: String,
    structured_output_kind: String,
    structured_output_value: Option<Value>,
    output_token_ids: Vec<u32>,
    output_text: String,
}

#[derive(Clone, Debug, Serialize)]
struct BenchStructuredOutputConfigReport {
    mode: String,
    schema_name: Option<String>,
    schema: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchStructuredOutputRuntimeReport {
    mode: String,
    parser: String,
    kind: String,
    value: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchQwen35OutputMetricsReport {
    output_modes: Vec<String>,
    readback_bytes: u64,
    raw_logits: bool,
    graph_hits: usize,
    graph_misses: usize,
    graph_captures: usize,
    graph_shape_drifts: usize,
}

#[derive(Clone, Debug, Serialize)]
struct BenchTerminationReport {
    observed: String,
    classification: String,
    matched_stop_sequence: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchPsionicCudaStartupReport {
    load_s: f64,
    cublas_handle_scope: String,
    cublas_stream_binding: String,
    warmup_status: String,
    warmup_prompt_s: f64,
    warmup_decode_s: f64,
    warmup_total_s: f64,
    warmup_output_tokens: usize,
    request_billed_to_user: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            backend: BenchBackend::Psionic,
            model_path: PathBuf::new(),
            ollama_model: None,
            ollama_base_url: String::from("http://127.0.0.1:11434"),
            json_out: None,
            prompt: String::from("Explain what Psionic is in one sentence."),
            max_output_tokens: 256,
            repeats: 3,
            decode_mode: BenchDecodeMode::Greedy,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            typical_p: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            repeat_penalty: None,
            repeat_last_n: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            structured_output: None,
        }
    }
}

impl BenchConfig {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut config = Self::default();
        let raw_args = args.collect::<Vec<_>>();
        let mut json_object = false;
        let mut json_schema_file: Option<PathBuf> = None;
        let mut json_schema_name: Option<String> = None;
        if raw_args.is_empty() {
            return Err(usage());
        }
        if !raw_args[0].starts_with("--") {
            return Self::parse_legacy(raw_args);
        }
        let mut index = 0;
        while index < raw_args.len() {
            let argument = &raw_args[index];
            match argument.as_str() {
                "--backend" => {
                    config.backend = match next_arg(&raw_args, &mut index, "--backend")?.as_str() {
                        "psionic" => BenchBackend::Psionic,
                        "ollama" => BenchBackend::Ollama,
                        value => {
                            return Err(format!(
                                "invalid --backend `{value}`; expected `psionic` or `ollama`"
                            ));
                        }
                    };
                }
                "--model-path" => {
                    config.model_path =
                        PathBuf::from(next_arg(&raw_args, &mut index, "--model-path")?);
                }
                "--ollama-model" => {
                    config.ollama_model = Some(next_arg(&raw_args, &mut index, "--ollama-model")?);
                }
                "--ollama-base-url" => {
                    config.ollama_base_url = next_arg(&raw_args, &mut index, "--ollama-base-url")?;
                }
                "--json-out" => {
                    config.json_out = Some(PathBuf::from(next_arg(
                        &raw_args,
                        &mut index,
                        "--json-out",
                    )?));
                }
                "--prompt" => {
                    config.prompt = next_arg(&raw_args, &mut index, "--prompt")?;
                }
                "--max-output-tokens" => {
                    config.max_output_tokens = parse_arg(
                        &next_arg(&raw_args, &mut index, "--max-output-tokens")?,
                        "--max-output-tokens",
                    )?;
                }
                "--repeats" => {
                    config.repeats =
                        parse_arg(&next_arg(&raw_args, &mut index, "--repeats")?, "--repeats")?;
                }
                "--decode" => {
                    config.decode_mode = match next_arg(&raw_args, &mut index, "--decode")?.as_str()
                    {
                        "greedy" => BenchDecodeMode::Greedy,
                        "sample" => BenchDecodeMode::Sample,
                        value => {
                            return Err(format!(
                                "invalid --decode `{value}`; expected `greedy` or `sample`"
                            ));
                        }
                    };
                }
                "--temperature" => {
                    config.temperature = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--temperature")?,
                        "--temperature",
                    )?);
                }
                "--top-k" => {
                    config.top_k = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--top-k")?,
                        "--top-k",
                    )?);
                }
                "--top-p" => {
                    config.top_p = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--top-p")?,
                        "--top-p",
                    )?);
                }
                "--min-p" => {
                    config.min_p = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--min-p")?,
                        "--min-p",
                    )?);
                }
                "--typical-p" => {
                    config.typical_p = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--typical-p")?,
                        "--typical-p",
                    )?);
                }
                "--mirostat" => {
                    config.mirostat = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--mirostat")?,
                        "--mirostat",
                    )?);
                }
                "--mirostat-tau" => {
                    config.mirostat_tau = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--mirostat-tau")?,
                        "--mirostat-tau",
                    )?);
                }
                "--mirostat-eta" => {
                    config.mirostat_eta = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--mirostat-eta")?,
                        "--mirostat-eta",
                    )?);
                }
                "--repeat-penalty" => {
                    config.repeat_penalty = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--repeat-penalty")?,
                        "--repeat-penalty",
                    )?);
                }
                "--repeat-last-n" => {
                    config.repeat_last_n = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--repeat-last-n")?,
                        "--repeat-last-n",
                    )?);
                }
                "--presence-penalty" => {
                    config.presence_penalty = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--presence-penalty")?,
                        "--presence-penalty",
                    )?);
                }
                "--frequency-penalty" => {
                    config.frequency_penalty = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--frequency-penalty")?,
                        "--frequency-penalty",
                    )?);
                }
                "--seed" => {
                    config.seed = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--seed")?,
                        "--seed",
                    )?);
                }
                "--json-object" => {
                    json_object = true;
                }
                "--json-schema-file" => {
                    json_schema_file = Some(PathBuf::from(next_arg(
                        &raw_args,
                        &mut index,
                        "--json-schema-file",
                    )?));
                }
                "--json-schema-name" => {
                    json_schema_name = Some(next_arg(&raw_args, &mut index, "--json-schema-name")?);
                }
                "--greedy" => {
                    config.decode_mode = BenchDecodeMode::Greedy;
                }
                "--sample" => {
                    config.decode_mode = BenchDecodeMode::Sample;
                }
                "--help" | "-h" => {
                    return Err(usage());
                }
                value => {
                    return Err(format!("unknown argument `{value}`\n\n{}", usage()));
                }
            }
            index += 1;
        }
        config.structured_output =
            parse_structured_output(json_object, json_schema_file, json_schema_name)?;
        config.validate()?;
        Ok(config)
    }

    fn parse_legacy(args: Vec<String>) -> Result<Self, String> {
        let mut config = Self::default();
        config.model_path = PathBuf::from(args.first().cloned().ok_or_else(usage)?);
        if let Some(prompt) = args.get(1) {
            config.prompt = prompt.clone();
        }
        if let Some(max_output_tokens) = args.get(2) {
            config.max_output_tokens = parse_arg(max_output_tokens, "max_output_tokens")?;
        }
        if let Some(repeats) = args.get(3) {
            config.repeats = parse_arg(repeats, "repeats")?;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.model_path.as_os_str().is_empty() {
            return Err(format!("missing --model-path\n\n{}", usage()));
        }
        if self.repeats == 0 {
            return Err(String::from("--repeats must be at least 1"));
        }
        if matches!(self.backend, BenchBackend::Ollama) && self.ollama_model.is_none() {
            return Err(String::from(
                "missing --ollama-model for `--backend ollama`",
            ));
        }
        Ok(())
    }

    fn effective_temperature(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.temperature,
            BenchDecodeMode::Sample => Some(self.temperature.unwrap_or(0.8)),
        }
    }

    fn effective_top_k(&self) -> Option<usize> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.top_k,
            BenchDecodeMode::Sample => Some(self.top_k.unwrap_or(40)),
        }
    }

    fn effective_top_p(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.top_p,
            BenchDecodeMode::Sample => Some(self.top_p.unwrap_or(0.9)),
        }
    }

    fn effective_min_p(&self) -> Option<f32> {
        self.min_p.filter(|min_p| min_p.is_finite() && *min_p > 0.0)
    }

    fn effective_typical_p(&self) -> Option<f32> {
        self.typical_p
            .filter(|typical_p| typical_p.is_finite() && *typical_p > 0.0 && *typical_p < 1.0)
    }

    fn effective_mirostat(&self) -> Option<u8> {
        self.mirostat.filter(|value| matches!(value, 1 | 2))
    }

    fn effective_mirostat_tau(&self) -> Option<f32> {
        self.effective_mirostat()
            .map(|_| self.mirostat_tau.unwrap_or(5.0).max(0.0))
    }

    fn effective_mirostat_eta(&self) -> Option<f32> {
        self.effective_mirostat()
            .map(|_| self.mirostat_eta.unwrap_or(0.1).max(0.0))
    }

    fn effective_repeat_penalty(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.repeat_penalty,
            BenchDecodeMode::Sample => Some(self.repeat_penalty.unwrap_or(1.0)),
        }
    }

    fn effective_repeat_last_n(&self) -> Option<i32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.repeat_last_n,
            BenchDecodeMode::Sample => Some(
                self.repeat_last_n
                    .unwrap_or(DEFAULT_PENALTY_LOOKBACK as i32),
            ),
        }
    }

    fn effective_presence_penalty(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.presence_penalty,
            BenchDecodeMode::Sample => Some(self.presence_penalty.unwrap_or(0.0)),
        }
    }

    fn effective_frequency_penalty(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.frequency_penalty,
            BenchDecodeMode::Sample => Some(self.frequency_penalty.unwrap_or(0.0)),
        }
    }

    fn effective_seed(&self) -> Option<u64> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.seed,
            BenchDecodeMode::Sample => Some(self.seed.unwrap_or(42)),
        }
    }

    fn effective_temperature_for_backend(&self, backend: BenchBackend) -> Option<f32> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(0.0),
            _ => self.effective_temperature(),
        }
    }

    fn effective_top_k_for_backend(&self, backend: BenchBackend) -> Option<usize> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(1),
            _ => self.effective_top_k(),
        }
    }

    fn effective_top_p_for_backend(&self, backend: BenchBackend) -> Option<f32> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(1.0),
            _ => self.effective_top_p(),
        }
    }

    fn effective_min_p_for_backend(&self, backend: BenchBackend) -> Option<f32> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(0.0),
            _ => self.effective_min_p(),
        }
    }

    fn effective_repeat_penalty_for_backend(&self, backend: BenchBackend) -> Option<f32> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(1.0),
            _ => self.effective_repeat_penalty(),
        }
    }

    fn effective_repeat_last_n_for_backend(&self, backend: BenchBackend) -> Option<i32> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(0),
            _ => self.effective_repeat_last_n(),
        }
    }

    fn effective_presence_penalty_for_backend(&self, backend: BenchBackend) -> Option<f32> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(0.0),
            _ => self.effective_presence_penalty(),
        }
    }

    fn effective_frequency_penalty_for_backend(&self, backend: BenchBackend) -> Option<f32> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(0.0),
            _ => self.effective_frequency_penalty(),
        }
    }

    fn effective_seed_for_backend(&self, backend: BenchBackend) -> Option<u64> {
        match (backend, self.decode_mode) {
            (BenchBackend::Ollama, BenchDecodeMode::Greedy) => Some(self.seed.unwrap_or(42)),
            _ => self.effective_seed(),
        }
    }

    fn ollama_format_payload(&self) -> Option<Value> {
        match self.structured_output.as_ref() {
            Some(BenchStructuredOutput::JsonObject) => Some(Value::String(String::from("json"))),
            Some(BenchStructuredOutput::JsonSchema { schema, .. }) => Some(schema.clone()),
            None => None,
        }
    }
}

fn run_psionic_benchmark(config: &BenchConfig) -> Result<(), String> {
    let bench_model = load_bench_model(&config.model_path, &config.prompt)?;
    let load_started_at = Instant::now();
    let mut service = CudaGgufQwen35TextGenerationService::from_gguf_path(&config.model_path)
        .map_err(|error| format!("failed to load qwen35 cuda service: {error}"))?;
    let load_s = load_started_at.elapsed().as_secs_f64();
    let descriptor = service.model_descriptor().clone();
    let prefix_cache_bypass = PrefixCacheControl {
        mode: PrefixCacheMode::Bypass,
        ..PrefixCacheControl::default()
    };

    let warmup = GenerationRequest::new_text(
        String::from("warmup"),
        descriptor.clone(),
        None,
        bench_model.rendered.text.clone(),
        build_generation_options(
            config,
            min_warmup_tokens(config.max_output_tokens),
            &bench_model.rendered.stop_sequences,
        ),
    )
    .with_prefix_cache_control(prefix_cache_bypass.clone());
    let warmup_response = service
        .generate(&warmup)
        .map_err(|error| format!("warmup generation failed: {error}"))?;
    let startup_report = BenchPsionicCudaStartupReport {
        load_s,
        cublas_handle_scope: String::from("per_device_runtime_owner"),
        cublas_stream_binding: String::from("bind_stream_per_submission"),
        warmup_status: String::from("explicit_warmup_completed"),
        warmup_prompt_s: nanos_to_seconds(
            warmup_response.metrics.prompt_eval_duration_ns.unwrap_or(0),
        ),
        warmup_decode_s: nanos_to_seconds(warmup_response.metrics.eval_duration_ns.unwrap_or(0)),
        warmup_total_s: nanos_to_seconds(warmup_response.metrics.total_duration_ns.unwrap_or(0)),
        warmup_output_tokens: warmup_response
            .metrics
            .eval_count
            .unwrap_or(warmup_response.output.tokens.len()),
        request_billed_to_user: false,
    };
    println!(
        "backend=psionic load_s={:.6} startup_warmup_status={} cublas_handle_scope={} cublas_stream_binding={} warmup_prompt_s={:.6} warmup_decode_s={:.6} warmup_total_s={:.6} warmup_output_tokens={}",
        startup_report.load_s,
        startup_report.warmup_status,
        startup_report.cublas_handle_scope,
        startup_report.cublas_stream_binding,
        startup_report.warmup_prompt_s,
        startup_report.warmup_decode_s,
        startup_report.warmup_total_s,
        startup_report.warmup_output_tokens,
    );

    let mut runs = Vec::with_capacity(config.repeats);
    for run_index in 0..config.repeats {
        let request = GenerationRequest::new_text(
            format!("bench-{run_index}"),
            descriptor.clone(),
            None,
            bench_model.rendered.text.clone(),
            build_generation_options(
                config,
                config.max_output_tokens,
                &bench_model.rendered.stop_sequences,
            ),
        )
        .with_prefix_cache_control(prefix_cache_bypass.clone());
        let response = service
            .generate(&request)
            .map_err(|error| format!("benchmark generation failed: {error}"))?;
        let output_tokens = response
            .metrics
            .eval_count
            .unwrap_or(response.output.tokens.len());
        let decode_ns = response.metrics.eval_duration_ns.unwrap_or(0);
        let prompt_ns = response.metrics.prompt_eval_duration_ns.unwrap_or(0);
        let total_ns = response.metrics.total_duration_ns.unwrap_or(0);
        let decode_tok_s = tokens_per_second(output_tokens, decode_ns);
        let output_metrics =
            qwen35_output_metrics_report(response.metrics.qwen35_cuda_decode.as_ref());
        let structured_output = structured_output_runtime_report(
            response.provenance.as_ref(),
            response.output.structured.as_ref(),
        );
        let termination = psionic_termination_report(
            &response,
            &bench_model.tokenizer,
            &bench_model.rendered.stop_sequences,
        );
        let output_token_ids = token_ids(response.output.tokens.as_slice());
        let prompt_s = nanos_to_seconds(prompt_ns);
        let decode_s = nanos_to_seconds(decode_ns);
        let total_s = nanos_to_seconds(total_ns);
        let ttft_s = response.metrics.time_to_first_token_ns.map(nanos_to_seconds);
        let itl_s = response.metrics.inter_token_latency_ns.map(nanos_to_seconds);
        let output_text = response.output.text;
        let printable_output_text = output_text.replace('\n', "\\n");
        runs.push(BenchRunReport {
            run_index: run_index + 1,
            decode_mode: String::from(bench_decode_mode_label(config.decode_mode)),
            prompt_tokens: response.metrics.prompt_eval_count.unwrap_or(0),
            output_tokens,
            prompt_s,
            decode_s,
            total_s,
            ttft_s,
            itl_s,
            decode_tok_s,
            qwen35_output_modes: output_metrics.output_modes.clone(),
            qwen35_readback_bytes: output_metrics.readback_bytes,
            qwen35_raw_logits: output_metrics.raw_logits,
            qwen35_graph_hits: output_metrics.graph_hits,
            qwen35_graph_misses: output_metrics.graph_misses,
            qwen35_graph_captures: output_metrics.graph_captures,
            qwen35_graph_shape_drifts: output_metrics.graph_shape_drifts,
            termination: termination.clone(),
            structured_output_mode: structured_output.mode.clone(),
            structured_output_parser: structured_output.parser.clone(),
            structured_output_kind: structured_output.kind.clone(),
            structured_output_value: structured_output.value.clone(),
            output_token_ids,
            output_text: output_text.clone(),
        });
        println!(
            "backend=psionic run={} decode_mode={} prompt_tokens={} output_tokens={} prompt_s={:.6} decode_s={:.6} total_s={:.6} ttft_s={} itl_s={} decode_tok_s={:.2} termination_observed={} termination_classification={} matched_stop_sequence={} {} {} output={}",
            run_index + 1,
            bench_decode_mode_label(config.decode_mode),
            response.metrics.prompt_eval_count.unwrap_or(0),
            output_tokens,
            prompt_s,
            decode_s,
            total_s,
            format_optional_seconds(ttft_s),
            format_optional_seconds(itl_s),
            decode_tok_s,
            termination.observed,
            termination.classification,
            termination
                .matched_stop_sequence
                .as_deref()
                .unwrap_or("none"),
            format_qwen35_output_metrics(&output_metrics),
            format_structured_output_report(&structured_output),
            printable_output_text,
        );
    }

    let report = build_bench_report(
        config,
        &bench_model.rendered,
        Some(startup_report.clone()),
        Some(startup_report.load_s),
        runs,
    );
    println!(
        "backend=psionic mean_decode_tok_s={:.2}",
        report.mean_decode_tok_s
    );
    write_json_output(&report, config.json_out.as_ref())?;
    Ok(())
}

fn run_ollama_benchmark(config: &BenchConfig) -> Result<(), String> {
    let bench_model = load_bench_model(&config.model_path, &config.prompt)?;
    let client = Client::builder()
        .build()
        .map_err(|error| format!("failed to build Ollama HTTP client: {error}"))?;
    let ollama_model = config
        .ollama_model
        .as_ref()
        .ok_or_else(|| String::from("missing Ollama model alias"))?;

    let _ = ollama_generate(
        &client,
        &config.ollama_base_url,
        ollama_model,
        &bench_model.rendered,
        config,
        min_warmup_tokens(config.max_output_tokens),
    )?;

    let mut runs = Vec::with_capacity(config.repeats);
    for run_index in 0..config.repeats {
        let response = ollama_generate(
            &client,
            &config.ollama_base_url,
            ollama_model,
            &bench_model.rendered,
            config,
            config.max_output_tokens,
        )?;
        let output_tokens = response.eval_count.unwrap_or(0);
        let decode_ns = response.eval_duration.unwrap_or(0);
        let prompt_ns = response.prompt_eval_duration.unwrap_or(0);
        let total_ns = response.total_duration.unwrap_or(0);
        let decode_tok_s = tokens_per_second(output_tokens, decode_ns);
        let prompt_s = nanos_to_seconds(prompt_ns);
        let decode_s = nanos_to_seconds(decode_ns);
        let total_s = nanos_to_seconds(total_ns);
        let termination = ollama_termination_report(
            &response,
            &bench_model.rendered.stop_sequences,
            config.max_output_tokens,
        );
        let output_token_ids = token_ids(
            bench_model
                .tokenizer
                .encode(response.response.as_str())
                .as_slice(),
        );
        let output_text = response.response;
        let printable_output_text = output_text.replace('\n', "\\n");
        runs.push(BenchRunReport {
            run_index: run_index + 1,
            decode_mode: String::from(bench_decode_mode_label(config.decode_mode)),
            prompt_tokens: response.prompt_eval_count.unwrap_or(0),
            output_tokens,
            prompt_s,
            decode_s,
            total_s,
            ttft_s: None,
            itl_s: None,
            decode_tok_s,
            qwen35_output_modes: Vec::new(),
            qwen35_readback_bytes: 0,
            qwen35_raw_logits: false,
            qwen35_graph_hits: 0,
            qwen35_graph_misses: 0,
            qwen35_graph_captures: 0,
            qwen35_graph_shape_drifts: 0,
            termination: termination.clone(),
            structured_output_mode: String::from("none"),
            structured_output_parser: String::from("none"),
            structured_output_kind: String::from("none"),
            structured_output_value: None,
            output_token_ids,
            output_text: output_text.clone(),
        });
        println!(
            "backend=ollama run={} decode_mode={} prompt_tokens={} output_tokens={} prompt_s={:.6} decode_s={:.6} total_s={:.6} decode_tok_s={:.2} termination_observed={} termination_classification={} matched_stop_sequence={} output={}",
            run_index + 1,
            bench_decode_mode_label(config.decode_mode),
            response.prompt_eval_count.unwrap_or(0),
            output_tokens,
            prompt_s,
            decode_s,
            total_s,
            decode_tok_s,
            termination.observed,
            termination.classification,
            termination
                .matched_stop_sequence
                .as_deref()
                .unwrap_or("none"),
            printable_output_text,
        );
    }

    let report = build_bench_report(config, &bench_model.rendered, None, None, runs);
    println!(
        "backend=ollama mean_decode_tok_s={:.2}",
        report.mean_decode_tok_s
    );
    write_json_output(&report, config.json_out.as_ref())?;
    Ok(())
}

fn build_generation_options(
    config: &BenchConfig,
    max_output_tokens: usize,
    stop_sequences: &[String],
) -> GenerationOptions {
    let mut options = match config.decode_mode {
        BenchDecodeMode::Greedy => GenerationOptions::greedy(max_output_tokens),
        BenchDecodeMode::Sample => GenerationOptions::sample(max_output_tokens),
    };
    options.stop_sequences = stop_sequences.to_vec();
    options.temperature = config.effective_temperature();
    options.top_k = config.effective_top_k();
    options.top_p = config.effective_top_p();
    options.min_p = config.effective_min_p();
    options.typical_p = config.effective_typical_p();
    options.mirostat = config.effective_mirostat();
    options.mirostat_tau = config.effective_mirostat_tau();
    options.mirostat_eta = config.effective_mirostat_eta();
    options.repeat_penalty = config.effective_repeat_penalty();
    options.repeat_last_n = config.effective_repeat_last_n();
    options.presence_penalty = config.effective_presence_penalty();
    options.frequency_penalty = config.effective_frequency_penalty();
    options.seed = config.effective_seed();
    options.structured_output = match config.structured_output.as_ref() {
        Some(BenchStructuredOutput::JsonObject) => Some(StructuredOutputRequest::JsonObject),
        Some(BenchStructuredOutput::JsonSchema { name, schema }) => {
            Some(StructuredOutputRequest::JsonSchema {
                name: name.clone(),
                schema: schema.clone(),
            })
        }
        None => None,
    };
    options
}

fn load_bench_model(model_path: &Path, prompt: &str) -> Result<BenchModelContext, String> {
    let adapter = GgufDecoderAdapterLoader
        .load_path(model_path)
        .map_err(|error| format!("failed to load GGUF metadata: {error}"))?;
    let tokenizer = GgufRuntimeTokenizer::from_gguf(adapter.tokenizer())
        .map_err(|error| format!("failed to build GGUF runtime tokenizer: {error}"))?;
    let renderer = adapter.prompt_renderer();
    let rendered = renderer
        .render_with_options(
            None,
            &[PromptMessage::new(
                PromptMessageRole::User,
                prompt.to_string(),
            )],
            true,
            &PromptRenderOptions::default(),
        )
        .map_err(|error| format!("failed to render qwen35 prompt: {error}"))?;
    Ok(BenchModelContext {
        rendered: RenderedPrompt {
            text: rendered.text,
            stop_sequences: rendered.stop_sequences,
        },
        tokenizer,
    })
}

fn qwen35_output_metrics_report(
    metrics: Option<&Qwen35CudaDecodeOutputMetrics>,
) -> BenchQwen35OutputMetricsReport {
    let Some(metrics) = metrics else {
        return BenchQwen35OutputMetricsReport {
            output_modes: Vec::new(),
            readback_bytes: 0,
            raw_logits: false,
            graph_hits: 0,
            graph_misses: 0,
            graph_captures: 0,
            graph_shape_drifts: 0,
        };
    };
    let output_modes = metrics
        .output_modes
        .iter()
        .map(|mode| match mode {
            psionic_serve::Qwen35CudaDecodeOutputMode::ArgmaxOnly => String::from("argmax_only"),
            psionic_serve::Qwen35CudaDecodeOutputMode::TopKCandidates { top_k } => {
                format!("top_k_candidates:{top_k}")
            }
            psionic_serve::Qwen35CudaDecodeOutputMode::SparseLogits { token_count } => {
                format!("sparse_logits:{token_count}")
            }
            psionic_serve::Qwen35CudaDecodeOutputMode::RawLogits => String::from("raw_logits"),
        })
        .collect::<Vec<_>>();
    BenchQwen35OutputMetricsReport {
        output_modes,
        readback_bytes: metrics.readback_bytes,
        raw_logits: metrics.raw_logits_materialized,
        graph_hits: metrics
            .graph_replay
            .as_ref()
            .map_or(0, |graph| graph.replay_hit_count),
        graph_misses: metrics
            .graph_replay
            .as_ref()
            .map_or(0, |graph| graph.replay_miss_count),
        graph_captures: metrics
            .graph_replay
            .as_ref()
            .map_or(0, |graph| graph.capture_count),
        graph_shape_drifts: metrics
            .graph_replay
            .as_ref()
            .map_or(0, |graph| graph.shape_drift_count),
    }
}

fn format_qwen35_output_metrics(report: &BenchQwen35OutputMetricsReport) -> String {
    format!(
        "qwen35_output_modes=[{}] qwen35_readback_bytes={} qwen35_raw_logits={} qwen35_graph_hits={} qwen35_graph_misses={} qwen35_graph_captures={} qwen35_graph_shape_drifts={}",
        report.output_modes.join(","),
        report.readback_bytes,
        report.raw_logits,
        report.graph_hits,
        report.graph_misses,
        report.graph_captures,
        report.graph_shape_drifts,
    )
}

fn structured_output_runtime_report(
    provenance: Option<&psionic_serve::GenerationProvenance>,
    structured_value: Option<&StructuredOutputValue>,
) -> BenchStructuredOutputRuntimeReport {
    let Some(report) = provenance.and_then(|provenance| provenance.structured_output.as_ref())
    else {
        return BenchStructuredOutputRuntimeReport {
            mode: String::from("none"),
            parser: String::from("none"),
            kind: String::from("none"),
            value: None,
        };
    };
    BenchStructuredOutputRuntimeReport {
        mode: String::from(report.mode.label()),
        parser: String::from(report.parser.label()),
        kind: String::from(report.kind.label()),
        value: structured_value.and_then(|value| serde_json::to_value(value).ok()),
    }
}

fn format_structured_output_report(report: &BenchStructuredOutputRuntimeReport) -> String {
    let value = report
        .value
        .as_ref()
        .and_then(|value| serde_json::to_string(value).ok())
        .unwrap_or_else(|| String::from("none"));
    format!(
        "structured_output_mode={} structured_output_parser={} structured_output_kind={} structured_output_value={}",
        report.mode, report.parser, report.kind, value
    )
}

fn bench_decode_mode_label(mode: BenchDecodeMode) -> &'static str {
    match mode {
        BenchDecodeMode::Greedy => "greedy",
        BenchDecodeMode::Sample => "sample",
    }
}

fn min_warmup_tokens(max_output_tokens: usize) -> usize {
    max_output_tokens.min(16).max(1)
}

fn next_arg(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    let value_index = index.saturating_add(1);
    let value = args
        .get(value_index)
        .cloned()
        .ok_or_else(|| format!("missing value for `{flag}`"))?;
    *index = value_index;
    Ok(value)
}

fn parse_structured_output(
    json_object: bool,
    json_schema_file: Option<PathBuf>,
    json_schema_name: Option<String>,
) -> Result<Option<BenchStructuredOutput>, String> {
    let selected_modes = usize::from(json_object) + usize::from(json_schema_file.is_some());
    if selected_modes > 1 {
        return Err(String::from(
            "structured output accepts at most one of `--json-object` or `--json-schema-file`",
        ));
    }
    match json_schema_file {
        Some(path) => {
            let raw = fs::read_to_string(&path).map_err(|error| {
                format!(
                    "failed to read JSON schema file `{}`: {error}",
                    path.display()
                )
            })?;
            let schema = serde_json::from_str::<Value>(&raw).map_err(|error| {
                format!(
                    "failed to parse JSON schema file `{}`: {error}",
                    path.display()
                )
            })?;
            Ok(Some(BenchStructuredOutput::JsonSchema {
                name: json_schema_name,
                schema,
            }))
        }
        None if json_object => {
            if json_schema_name.is_some() {
                return Err(String::from(
                    "`--json-schema-name` requires `--json-schema-file`",
                ));
            }
            Ok(Some(BenchStructuredOutput::JsonObject))
        }
        None => {
            if json_schema_name.is_some() {
                return Err(String::from(
                    "`--json-schema-name` requires `--json-schema-file`",
                ));
            }
            Ok(None)
        }
    }
}

fn parse_arg<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| format!("invalid {name} `{value}`: {error}"))
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

fn current_unix_timestamp_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn build_bench_report(
    config: &BenchConfig,
    rendered: &RenderedPrompt,
    psionic_cuda_startup: Option<BenchPsionicCudaStartupReport>,
    load_s: Option<f64>,
    runs: Vec<BenchRunReport>,
) -> BenchReport {
    let repeats = runs.len().max(1) as f64;
    let mean_output_tokens = runs.iter().map(|run| run.output_tokens as f64).sum::<f64>() / repeats;
    let mean_prompt_s = runs.iter().map(|run| run.prompt_s).sum::<f64>() / repeats;
    let mean_decode_s = runs.iter().map(|run| run.decode_s).sum::<f64>() / repeats;
    let mean_total_s = runs.iter().map(|run| run.total_s).sum::<f64>() / repeats;
    let mean_ttft_s = mean_optional_seconds(runs.iter().filter_map(|run| run.ttft_s));
    let mean_itl_s = mean_optional_seconds(runs.iter().filter_map(|run| run.itl_s));
    let mean_decode_tok_s = runs.iter().map(|run| run.decode_tok_s).sum::<f64>() / repeats;
    BenchReport {
        schema_version: 4,
        report_kind: String::from("qwen35_cuda_bench"),
        benchmark_class: String::from(bench_report_class(config.backend)),
        generated_at_unix_s: current_unix_timestamp_seconds(),
        backend: String::from(bench_backend_label(config.backend)),
        model_path: config.model_path.display().to_string(),
        ollama_model: config.ollama_model.clone(),
        ollama_base_url: matches!(config.backend, BenchBackend::Ollama)
            .then(|| config.ollama_base_url.clone()),
        prompt: config.prompt.clone(),
        rendered_prompt: rendered.text.clone(),
        stop_sequences: rendered.stop_sequences.clone(),
        decode_mode: String::from(bench_decode_mode_label(config.decode_mode)),
        max_output_tokens: config.max_output_tokens,
        repeats: runs.len(),
        steady_state_concurrency: 1,
        load_s,
        temperature: config.effective_temperature_for_backend(config.backend),
        top_k: config.effective_top_k_for_backend(config.backend),
        top_p: config.effective_top_p_for_backend(config.backend),
        min_p: config.effective_min_p_for_backend(config.backend),
        typical_p: config.effective_typical_p(),
        mirostat: config.effective_mirostat(),
        mirostat_tau: config.effective_mirostat_tau(),
        mirostat_eta: config.effective_mirostat_eta(),
        repeat_penalty: config.effective_repeat_penalty_for_backend(config.backend),
        repeat_last_n: config.effective_repeat_last_n_for_backend(config.backend),
        presence_penalty: config.effective_presence_penalty_for_backend(config.backend),
        frequency_penalty: config.effective_frequency_penalty_for_backend(config.backend),
        seed: config.effective_seed_for_backend(config.backend),
        structured_output: structured_output_config_report(config.structured_output.as_ref()),
        psionic_cuda_startup,
        runs,
        mean_output_tokens,
        mean_prompt_s,
        mean_decode_s,
        mean_total_s,
        mean_ttft_s,
        mean_itl_s,
        mean_decode_tok_s,
    }
}

fn structured_output_config_report(
    structured_output: Option<&BenchStructuredOutput>,
) -> BenchStructuredOutputConfigReport {
    match structured_output {
        Some(BenchStructuredOutput::JsonObject) => BenchStructuredOutputConfigReport {
            mode: String::from("json_object"),
            schema_name: None,
            schema: None,
        },
        Some(BenchStructuredOutput::JsonSchema { name, schema }) => {
            BenchStructuredOutputConfigReport {
                mode: String::from("json_schema"),
                schema_name: name.clone(),
                schema: Some(schema.clone()),
            }
        }
        None => BenchStructuredOutputConfigReport {
            mode: String::from("none"),
            schema_name: None,
            schema: None,
        },
    }
}

fn bench_backend_label(backend: BenchBackend) -> &'static str {
    match backend {
        BenchBackend::Psionic => "psionic",
        BenchBackend::Ollama => "ollama",
    }
}

fn bench_report_class(backend: BenchBackend) -> &'static str {
    match backend {
        BenchBackend::Psionic => "direct_engine",
        BenchBackend::Ollama => "http",
    }
}

fn mean_optional_seconds(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values = values.collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn format_optional_seconds(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| String::from("none"))
}

fn write_json_output<T: Serialize>(value: &T, output: Option<&PathBuf>) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize JSON report: {error}"))?;
    match output {
        Some(path) => fs::write(path, format!("{json}\n"))
            .map_err(|error| format!("failed to write JSON report `{}`: {error}", path.display())),
        None => Ok(()),
    }
}

fn usage() -> String {
    String::from(
        "usage:\n  cargo run -p psionic-serve --example qwen35_cuda_bench -- <model.gguf> [prompt] [max_output_tokens] [repeats]\n  cargo run -p psionic-serve --example qwen35_cuda_bench -- --backend psionic --model-path <model.gguf> [--decode greedy|sample] [--temperature 0.8] [--top-k 40] [--top-p 0.9] [--min-p 0.05] [--typical-p 0.5] [--mirostat 1|2] [--mirostat-tau 5.0] [--mirostat-eta 0.1] [--repeat-penalty 1.0] [--repeat-last-n 64] [--presence-penalty 0.0] [--frequency-penalty 0.0] [--seed 42] [--json-object | --json-schema-file schema.json [--json-schema-name summary]] [--json-out report.json] [--prompt <text>] [--max-output-tokens 128] [--repeats 3]\n  cargo run -p psionic-serve --example qwen35_cuda_bench -- --backend ollama --model-path <model.gguf> --ollama-model qwen3.5:0.8b [--ollama-base-url http://127.0.0.1:11434] [--decode greedy|sample] [--temperature 0.8] [--top-k 40] [--top-p 0.9] [--min-p 0.05] [--typical-p 0.5] [--mirostat 1|2] [--mirostat-tau 5.0] [--mirostat-eta 0.1] [--repeat-penalty 1.0] [--repeat-last-n 64] [--presence-penalty 0.0] [--frequency-penalty 0.0] [--seed 42] [--json-object | --json-schema-file schema.json [--json-schema-name summary]] [--json-out report.json] [--prompt <text>] [--max-output-tokens 128] [--repeats 3]",
    )
}

#[derive(Clone, Debug)]
struct RenderedPrompt {
    text: String,
    stop_sequences: Vec<String>,
}

#[derive(Clone, Debug)]
struct BenchModelContext {
    rendered: RenderedPrompt,
    tokenizer: GgufRuntimeTokenizer,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    eval_count: Option<usize>,
    #[serde(default)]
    eval_duration: Option<u64>,
}

fn token_ids(tokens: &[TokenId]) -> Vec<u32> {
    tokens.iter().copied().map(TokenId::as_u32).collect()
}

fn psionic_termination_report(
    response: &GenerationResponse,
    tokenizer: &GgufRuntimeTokenizer,
    stop_sequences: &[String],
) -> BenchTerminationReport {
    let observed = termination_reason_label(response.termination).to_string();
    if let Some(detail) = response.metrics.termination_detail.as_ref() {
        return BenchTerminationReport {
            observed,
            classification: termination_cause_label(detail.cause).to_string(),
            matched_stop_sequence: detail.matched_stop_sequence.clone(),
        };
    }
    let classification = match response.termination {
        TerminationReason::EndOfSequence => response
            .output
            .tokens
            .as_slice()
            .last()
            .copied()
            .filter(|token| tokenizer.is_end_of_sequence(*token))
            .map(|_| String::from("eos_token"))
            .unwrap_or_else(|| {
                if stop_sequences.is_empty() {
                    String::from("end_of_sequence")
                } else {
                    String::from("stop_sequence")
                }
            }),
        TerminationReason::MaxOutputTokens => String::from("max_output_tokens"),
        TerminationReason::ContextLimit => String::from("context_limit"),
        TerminationReason::Cancelled => String::from("cancelled"),
        TerminationReason::Disconnected => String::from("disconnected"),
        TerminationReason::Error => String::from("error"),
    };
    BenchTerminationReport {
        observed,
        classification,
        matched_stop_sequence: None,
    }
}

fn ollama_termination_report(
    response: &OllamaGenerateResponse,
    stop_sequences: &[String],
    max_output_tokens: usize,
) -> BenchTerminationReport {
    let observed = response
        .done_reason
        .clone()
        .or_else(|| response.error.as_ref().map(|_| String::from("error")))
        .unwrap_or_else(|| String::from("unknown"));
    let classification = if response.error.is_some() {
        String::from("error")
    } else {
        match response.done_reason.as_deref() {
            Some("length") => String::from("max_output_tokens"),
            Some("stop") if stop_sequences.is_empty() => String::from("eos_token"),
            Some("stop") => String::from("ambiguous_stop_or_eos"),
            Some("unload") => String::from("unknown"),
            Some(other) => other.replace('-', "_"),
            None if response.eval_count.unwrap_or(0) >= max_output_tokens => {
                String::from("max_output_tokens")
            }
            None => String::from("unknown"),
        }
    };
    BenchTerminationReport {
        observed,
        classification,
        matched_stop_sequence: None,
    }
}

fn termination_reason_label(reason: TerminationReason) -> &'static str {
    match reason {
        TerminationReason::EndOfSequence => "end_of_sequence",
        TerminationReason::MaxOutputTokens => "max_output_tokens",
        TerminationReason::ContextLimit => "context_limit",
        TerminationReason::Cancelled => "cancelled",
        TerminationReason::Disconnected => "disconnected",
        TerminationReason::Error => "error",
    }
}

fn termination_cause_label(cause: GenerationTerminationCause) -> &'static str {
    match cause {
        GenerationTerminationCause::EndOfSequenceToken => "eos_token",
        GenerationTerminationCause::StopSequence => "stop_sequence",
        GenerationTerminationCause::MaxOutputTokens => "max_output_tokens",
        GenerationTerminationCause::ContextLimit => "context_limit",
        GenerationTerminationCause::Cancelled => "cancelled",
        GenerationTerminationCause::Disconnected => "disconnected",
        GenerationTerminationCause::Error => "error",
    }
}

fn ollama_generate(
    client: &Client,
    base_url: &str,
    model: &str,
    rendered: &RenderedPrompt,
    config: &BenchConfig,
    max_output_tokens: usize,
) -> Result<OllamaGenerateResponse, String> {
    let mut options = serde_json::json!({
        "num_predict": max_output_tokens,
    });
    if let Some(temperature) = config.effective_temperature_for_backend(BenchBackend::Ollama) {
        options["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_k) = config.effective_top_k_for_backend(BenchBackend::Ollama) {
        options["top_k"] = serde_json::json!(top_k);
    }
    if let Some(top_p) = config.effective_top_p_for_backend(BenchBackend::Ollama) {
        options["top_p"] = serde_json::json!(top_p);
    }
    if let Some(min_p) = config.effective_min_p_for_backend(BenchBackend::Ollama) {
        options["min_p"] = serde_json::json!(min_p);
    }
    if let Some(typical_p) = config.effective_typical_p() {
        options["typical_p"] = serde_json::json!(typical_p);
    }
    if let Some(mirostat) = config.effective_mirostat() {
        options["mirostat"] = serde_json::json!(mirostat);
    }
    if let Some(mirostat_tau) = config.effective_mirostat_tau() {
        options["mirostat_tau"] = serde_json::json!(mirostat_tau);
    }
    if let Some(mirostat_eta) = config.effective_mirostat_eta() {
        options["mirostat_eta"] = serde_json::json!(mirostat_eta);
    }
    if let Some(repeat_penalty) = config.effective_repeat_penalty_for_backend(BenchBackend::Ollama)
    {
        options["repeat_penalty"] = serde_json::json!(repeat_penalty);
    }
    if let Some(repeat_last_n) = config.effective_repeat_last_n_for_backend(BenchBackend::Ollama) {
        options["repeat_last_n"] = serde_json::json!(repeat_last_n);
    }
    if let Some(presence_penalty) =
        config.effective_presence_penalty_for_backend(BenchBackend::Ollama)
    {
        options["presence_penalty"] = serde_json::json!(presence_penalty);
    }
    if let Some(frequency_penalty) =
        config.effective_frequency_penalty_for_backend(BenchBackend::Ollama)
    {
        options["frequency_penalty"] = serde_json::json!(frequency_penalty);
    }
    if let Some(seed) = config.effective_seed_for_backend(BenchBackend::Ollama) {
        options["seed"] = serde_json::json!(seed);
    }
    if !rendered.stop_sequences.is_empty() {
        options["stop"] = serde_json::json!(rendered.stop_sequences);
    }
    let mut payload = serde_json::json!({
        "model": model,
        "prompt": rendered.text,
        "raw": true,
        "stream": false,
        "think": false,
        "keep_alive": 0,
        "options": options,
    });
    if let Some(format) = config.ollama_format_payload() {
        payload["format"] = format;
    }
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
    let response = client
        .post(&url)
        .json(&payload)
        .send()
        .map_err(|error| format!("failed to call Ollama generate endpoint: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .unwrap_or_else(|_| String::from("<unreadable response body>"));
        return Err(format!(
            "Ollama generate request failed with {status}: {body}"
        ));
    }
    response
        .json::<OllamaGenerateResponse>()
        .map_err(|error| format!("failed to decode Ollama generate response: {error}"))
}

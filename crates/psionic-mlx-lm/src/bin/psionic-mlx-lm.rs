use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use psionic_mlx_lm::{MlxLmTextRequest, MlxLmTextRuntime};
use psionic_models::{ContextOverflowPolicy, PromptMessage, PromptRenderOptions};
use psionic_runtime::{PrefixCacheControl, PrefixCacheMode};
use psionic_serve::{GenerationOptions, GenerationStreamEvent};
use serde::Serialize;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(message)) => {
            let _ = writeln!(io::stdout(), "{message}");
            ExitCode::SUCCESS
        }
        Err(CliError::Message(message)) => {
            let _ = writeln!(io::stderr(), "{message}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
enum CliError {
    Usage(String),
    Message(String),
}

fn run() -> Result<(), CliError> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return Err(CliError::Usage(usage()));
    };
    match command.as_str() {
        "load" => run_load(args),
        "generate" => run_generate(args),
        "stream" => run_stream(args),
        "batch" => run_batch(args),
        "render-chat" => run_render_chat(args),
        "-h" | "--help" => Err(CliError::Usage(usage())),
        other => Err(CliError::Message(format!(
            "unrecognized subcommand `{other}`\n\n{}",
            usage()
        ))),
    }
}

fn run_load(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_model_command(args).map_err(CliError::Message)?;
    let runtime = MlxLmTextRuntime::from_gguf_path(&parsed.model)
        .map_err(|error| CliError::Message(format!("failed to load `{}`: {error}", parsed.model.display())))?;
    write_json_output(&runtime.load_report(), parsed.json_out).map_err(CliError::Message)
}

fn run_generate(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_generation_command(args, false).map_err(CliError::Message)?;
    let mut runtime = load_runtime(&parsed.model).map_err(CliError::Message)?;
    let request = parsed.build_request("mlx-lm-generate");
    let response = runtime
        .generate_text(request)
        .map_err(|error| CliError::Message(format!("generation failed: {error}")))?;
    if let Some(path) = parsed.prompt_cache_artifact.as_ref() {
        runtime
            .save_prompt_cache_artifact(&response, path)
            .map_err(|error| {
                CliError::Message(format!("failed to save prompt-cache artifact: {error}"))
            })?;
    }
    write_json_output(&response, parsed.json_out).map_err(CliError::Message)
}

fn run_stream(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_generation_command(args, false).map_err(CliError::Message)?;
    let mut runtime = load_runtime(&parsed.model).map_err(CliError::Message)?;
    let mut stream = runtime
        .generate_stream(parsed.build_request("mlx-lm-stream"))
        .map_err(|error| CliError::Message(format!("stream generation failed: {error}")))?;
    let mut terminal = None;
    while let Some(event) = stream.next_event() {
        if let GenerationStreamEvent::Terminal(value) = &event {
            terminal = Some(value.clone());
        }
        let json = serde_json::to_string(&event)
            .map_err(|error| {
                CliError::Message(format!("failed to serialize stream event: {error}"))
            })?;
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(json.as_bytes())
            .map_err(|error| CliError::Message(format!("failed to write stream event: {error}")))?;
        stdout
            .write_all(b"\n")
            .map_err(|error| {
                CliError::Message(format!("failed to terminate stream event: {error}"))
            })?;
    }
    drop(stream);
    if let (Some(path), Some(terminal)) = (parsed.prompt_cache_artifact.as_ref(), terminal.as_ref()) {
        runtime
            .save_prompt_cache_artifact(&terminal.response, path)
            .map_err(|error| {
                CliError::Message(format!("failed to save prompt-cache artifact: {error}"))
            })?;
    }
    Ok(())
}

fn run_batch(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_batch_command(args).map_err(CliError::Message)?;
    let mut runtime = load_runtime(&parsed.model).map_err(CliError::Message)?;
    let requests = parsed
        .prompts
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, prompt)| parsed.request_for_prompt(index, prompt))
        .collect();
    let report = runtime.generate_batch_report(requests);
    write_json_output(&report, parsed.json_out).map_err(CliError::Message)
}

fn run_render_chat(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_render_chat_command(args).map_err(CliError::Message)?;
    let runtime = load_runtime(&parsed.model).map_err(CliError::Message)?;
    let messages = fs::read_to_string(&parsed.messages_json)
        .map_err(|error| {
            CliError::Message(format!(
                "failed to read `{}`: {error}",
                parsed.messages_json.display()
            ))
        })?;
    let messages: Vec<PromptMessage> = serde_json::from_str(&messages)
        .map_err(|error| CliError::Message(format!("invalid messages JSON: {error}")))?;
    let rendered = runtime
        .render_chat_prompt(
            parsed.template_name.as_deref(),
            messages.as_slice(),
            parsed.add_generation_prompt,
            &PromptRenderOptions::default(),
        )
        .map_err(|error| CliError::Message(format!("failed to render chat prompt: {error}")))?;
    write_json_output(&rendered, parsed.json_out).map_err(CliError::Message)
}

#[derive(Clone, Debug)]
struct ModelCommand {
    model: PathBuf,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct GenerationCommand {
    model: PathBuf,
    prompt: String,
    max_output_tokens: usize,
    decode_strategy: String,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
    session_id: Option<String>,
    prefix_cache_mode: PrefixCacheMode,
    context_overflow_policy: ContextOverflowPolicy,
    json_out: Option<PathBuf>,
    prompt_cache_artifact: Option<PathBuf>,
}

impl GenerationCommand {
    fn build_request(&self, request_id: &str) -> MlxLmTextRequest {
        let mut options = if self.decode_strategy == "sample" {
            GenerationOptions::sample(self.max_output_tokens)
        } else {
            GenerationOptions::greedy(self.max_output_tokens)
        };
        options.temperature = self.temperature;
        options.top_k = self.top_k;
        options.top_p = self.top_p;
        options.seed = self.seed;

        let mut request =
            MlxLmTextRequest::new(request_id, self.prompt.clone(), options).with_prefix_cache_control(
                PrefixCacheControl {
                    mode: self.prefix_cache_mode,
                    tenant_id: None,
                },
            )
            .with_context_overflow_policy(self.context_overflow_policy);
        if let Some(session_id) = &self.session_id {
            request = request.with_session_id(psionic_serve::SessionId::new(session_id.clone()));
        }
        request
    }
}

#[derive(Clone, Debug)]
struct BatchCommand {
    model: PathBuf,
    prompts: Vec<String>,
    max_output_tokens: usize,
    decode_strategy: String,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
    prefix_cache_mode: PrefixCacheMode,
    context_overflow_policy: ContextOverflowPolicy,
    json_out: Option<PathBuf>,
}

impl BatchCommand {
    fn request_for_prompt(&self, index: usize, prompt: String) -> MlxLmTextRequest {
        let mut options = if self.decode_strategy == "sample" {
            GenerationOptions::sample(self.max_output_tokens)
        } else {
            GenerationOptions::greedy(self.max_output_tokens)
        };
        options.temperature = self.temperature;
        options.top_k = self.top_k;
        options.top_p = self.top_p;
        options.seed = self.seed;
        MlxLmTextRequest::new(format!("mlx-lm-batch-{index}"), prompt, options)
            .with_prefix_cache_control(PrefixCacheControl {
                mode: self.prefix_cache_mode,
                tenant_id: None,
            })
            .with_context_overflow_policy(self.context_overflow_policy)
    }
}

#[derive(Clone, Debug)]
struct RenderChatCommand {
    model: PathBuf,
    messages_json: PathBuf,
    template_name: Option<String>,
    add_generation_prompt: bool,
    json_out: Option<PathBuf>,
}

fn parse_model_command(args: impl IntoIterator<Item = String>) -> Result<ModelCommand, String> {
    let mut model = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => {
                return Err(format!(
                    "unrecognized argument `{other}` for `load`\n\n{}",
                    usage()
                ))
            }
        }
    }
    let Some(model) = model else {
        return Err(format!("missing required `--model`\n\n{}", usage()));
    };
    Ok(ModelCommand { model, json_out })
}

fn parse_generation_command(
    args: impl IntoIterator<Item = String>,
    allow_multiple_prompts: bool,
) -> Result<GenerationCommand, String> {
    let mut model = None;
    let mut prompt = None;
    let mut max_output_tokens = 16;
    let mut decode_strategy = String::from("greedy");
    let mut temperature = None;
    let mut top_k = None;
    let mut top_p = None;
    let mut seed = None;
    let mut session_id = None;
    let mut prefix_cache_mode = PrefixCacheMode::Auto;
    let mut context_overflow_policy = ContextOverflowPolicy::Refuse;
    let mut json_out = None;
    let mut prompt_cache_artifact = None;

    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
            "--prompt" => {
                let value = next_value(&mut args, "--prompt")?;
                if prompt.replace(value).is_some() && !allow_multiple_prompts {
                    return Err(format!("only one `--prompt` is allowed\n\n{}", usage()));
                }
            }
            "--max-output-tokens" => {
                max_output_tokens = next_value(&mut args, "--max-output-tokens")?
                    .parse()
                    .map_err(|error| format!("invalid --max-output-tokens value: {error}"))?;
            }
            "--decode-strategy" => {
                decode_strategy = next_value(&mut args, "--decode-strategy")?;
                if decode_strategy != "greedy" && decode_strategy != "sample" {
                    return Err(format!(
                        "invalid --decode-strategy value `{decode_strategy}` (expected greedy or sample)\n\n{}",
                        usage()
                    ));
                }
            }
            "--temperature" => {
                temperature = Some(
                    next_value(&mut args, "--temperature")?
                        .parse()
                        .map_err(|error| format!("invalid --temperature value: {error}"))?,
                );
            }
            "--top-k" => {
                top_k = Some(
                    next_value(&mut args, "--top-k")?
                        .parse()
                        .map_err(|error| format!("invalid --top-k value: {error}"))?,
                );
            }
            "--top-p" => {
                top_p = Some(
                    next_value(&mut args, "--top-p")?
                        .parse()
                        .map_err(|error| format!("invalid --top-p value: {error}"))?,
                );
            }
            "--seed" => {
                seed = Some(
                    next_value(&mut args, "--seed")?
                        .parse()
                        .map_err(|error| format!("invalid --seed value: {error}"))?,
                );
            }
            "--session-id" => session_id = Some(next_value(&mut args, "--session-id")?),
            "--prefix-cache" => {
                prefix_cache_mode = parse_prefix_cache_mode(&next_value(&mut args, "--prefix-cache")?)?;
            }
            "--context-overflow-policy" => {
                context_overflow_policy =
                    parse_context_overflow_policy(&next_value(&mut args, "--context-overflow-policy")?)?;
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "--prompt-cache-artifact" => {
                prompt_cache_artifact =
                    Some(PathBuf::from(next_value(&mut args, "--prompt-cache-artifact")?));
            }
            "-h" | "--help" => return Err(usage()),
            other => {
                return Err(format!(
                    "unrecognized argument `{other}`\n\n{}",
                    usage()
                ))
            }
        }
    }

    let Some(model) = model else {
        return Err(format!("missing required `--model`\n\n{}", usage()));
    };
    let Some(prompt) = prompt else {
        return Err(format!("missing required `--prompt`\n\n{}", usage()));
    };
    Ok(GenerationCommand {
        model,
        prompt,
        max_output_tokens,
        decode_strategy,
        temperature,
        top_k,
        top_p,
        seed,
        session_id,
        prefix_cache_mode,
        context_overflow_policy,
        json_out,
        prompt_cache_artifact,
    })
}

fn parse_batch_command(args: impl IntoIterator<Item = String>) -> Result<BatchCommand, String> {
    let mut model = None;
    let mut prompts = Vec::new();
    let mut max_output_tokens = 16;
    let mut decode_strategy = String::from("greedy");
    let mut temperature = None;
    let mut top_k = None;
    let mut top_p = None;
    let mut seed = None;
    let mut prefix_cache_mode = PrefixCacheMode::Auto;
    let mut context_overflow_policy = ContextOverflowPolicy::Refuse;
    let mut json_out = None;

    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
            "--prompt" => prompts.push(next_value(&mut args, "--prompt")?),
            "--max-output-tokens" => {
                max_output_tokens = next_value(&mut args, "--max-output-tokens")?
                    .parse()
                    .map_err(|error| format!("invalid --max-output-tokens value: {error}"))?;
            }
            "--decode-strategy" => {
                decode_strategy = next_value(&mut args, "--decode-strategy")?;
                if decode_strategy != "greedy" && decode_strategy != "sample" {
                    return Err(format!(
                        "invalid --decode-strategy value `{decode_strategy}` (expected greedy or sample)\n\n{}",
                        usage()
                    ));
                }
            }
            "--temperature" => {
                temperature = Some(
                    next_value(&mut args, "--temperature")?
                        .parse()
                        .map_err(|error| format!("invalid --temperature value: {error}"))?,
                );
            }
            "--top-k" => {
                top_k = Some(
                    next_value(&mut args, "--top-k")?
                        .parse()
                        .map_err(|error| format!("invalid --top-k value: {error}"))?,
                );
            }
            "--top-p" => {
                top_p = Some(
                    next_value(&mut args, "--top-p")?
                        .parse()
                        .map_err(|error| format!("invalid --top-p value: {error}"))?,
                );
            }
            "--seed" => {
                seed = Some(
                    next_value(&mut args, "--seed")?
                        .parse()
                        .map_err(|error| format!("invalid --seed value: {error}"))?,
                );
            }
            "--prefix-cache" => {
                prefix_cache_mode = parse_prefix_cache_mode(&next_value(&mut args, "--prefix-cache")?)?;
            }
            "--context-overflow-policy" => {
                context_overflow_policy =
                    parse_context_overflow_policy(&next_value(&mut args, "--context-overflow-policy")?)?;
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => {
                return Err(format!(
                    "unrecognized argument `{other}`\n\n{}",
                    usage()
                ))
            }
        }
    }

    let Some(model) = model else {
        return Err(format!("missing required `--model`\n\n{}", usage()));
    };
    if prompts.is_empty() {
        return Err(format!("missing at least one `--prompt`\n\n{}", usage()));
    }

    Ok(BatchCommand {
        model,
        prompts,
        max_output_tokens,
        decode_strategy,
        temperature,
        top_k,
        top_p,
        seed,
        prefix_cache_mode,
        context_overflow_policy,
        json_out,
    })
}

fn parse_render_chat_command(
    args: impl IntoIterator<Item = String>,
) -> Result<RenderChatCommand, String> {
    let mut model = None;
    let mut messages_json = None;
    let mut template_name = None;
    let mut add_generation_prompt = true;
    let mut json_out = None;

    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
            "--messages-json" => {
                messages_json = Some(PathBuf::from(next_value(&mut args, "--messages-json")?));
            }
            "--template-name" => template_name = Some(next_value(&mut args, "--template-name")?),
            "--add-generation-prompt" => {
                add_generation_prompt = next_value(&mut args, "--add-generation-prompt")?
                    .parse()
                    .map_err(|error| format!("invalid --add-generation-prompt value: {error}"))?;
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => {
                return Err(format!(
                    "unrecognized argument `{other}` for `render-chat`\n\n{}",
                    usage()
                ))
            }
        }
    }

    let Some(model) = model else {
        return Err(format!("missing required `--model`\n\n{}", usage()));
    };
    let Some(messages_json) = messages_json else {
        return Err(format!("missing required `--messages-json`\n\n{}", usage()));
    };

    Ok(RenderChatCommand {
        model,
        messages_json,
        template_name,
        add_generation_prompt,
        json_out,
    })
}

fn parse_prefix_cache_mode(value: &str) -> Result<PrefixCacheMode, String> {
    match value {
        "auto" => Ok(PrefixCacheMode::Auto),
        "bypass" => Ok(PrefixCacheMode::Bypass),
        "invalidate" => Ok(PrefixCacheMode::Invalidate),
        other => Err(format!(
            "invalid --prefix-cache value `{other}` (expected auto, bypass, or invalidate)\n\n{}",
            usage()
        )),
    }
}

fn parse_context_overflow_policy(value: &str) -> Result<ContextOverflowPolicy, String> {
    match value {
        "refuse" => Ok(ContextOverflowPolicy::Refuse),
        "truncate_oldest" => Ok(ContextOverflowPolicy::TruncateOldest),
        other => Err(format!(
            "invalid --context-overflow-policy value `{other}` (expected refuse or truncate_oldest)\n\n{}",
            usage()
        )),
    }
}

fn load_runtime(model: &PathBuf) -> Result<MlxLmTextRuntime, String> {
    MlxLmTextRuntime::from_gguf_path(model)
        .map_err(|error| format!("failed to load `{}`: {error}", model.display()))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for `{flag}`"))
}

fn write_json_output<T: Serialize>(value: &T, output: Option<PathBuf>) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize JSON output: {error}"))?;
    if let Some(path) = output {
        fs::write(&path, format!("{json}\n"))
            .map_err(|error| format!("failed to write `{}`: {error}", path.display()))?;
        return Ok(());
    }
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(json.as_bytes())
        .map_err(|error| format!("failed to write stdout: {error}"))?;
    stdout
        .write_all(b"\n")
        .map_err(|error| format!("failed to terminate stdout JSON: {error}"))
}

fn usage() -> String {
    String::from(
        "usage:\n  psionic-mlx-lm load --model <path.gguf> [--json-out <path>]\n  psionic-mlx-lm generate --model <path.gguf> --prompt <text> [--max-output-tokens <n>] [--decode-strategy greedy|sample] [--temperature <f>] [--top-k <n>] [--top-p <f>] [--seed <n>] [--session-id <id>] [--prefix-cache auto|bypass|invalidate] [--context-overflow-policy refuse|truncate_oldest] [--prompt-cache-artifact <path>] [--json-out <path>]\n  psionic-mlx-lm stream --model <path.gguf> --prompt <text> [same generation flags]\n  psionic-mlx-lm batch --model <path.gguf> --prompt <text> --prompt <text> [...] [same generation flags except session-id and prompt-cache-artifact]\n  psionic-mlx-lm render-chat --model <path.gguf> --messages-json <path> [--template-name <name>] [--add-generation-prompt true|false] [--json-out <path>]",
    )
}

#[cfg(test)]
mod tests {
    use super::{
        parse_batch_command, parse_generation_command, parse_model_command, parse_render_chat_command,
    };
    use psionic_runtime::PrefixCacheMode;

    #[test]
    fn parse_generate_command_accepts_sampling_and_cache_flags() {
        let parsed = parse_generation_command(
            vec![
                "--model",
                "/tmp/model.gguf",
                "--prompt",
                "hello",
                "--decode-strategy",
                "sample",
                "--temperature",
                "0.7",
                "--top-k",
                "8",
                "--prefix-cache",
                "invalidate",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>(),
            false,
        )
        .expect("generate command");

        assert_eq!(parsed.model.to_string_lossy(), "/tmp/model.gguf");
        assert_eq!(parsed.prompt, "hello");
        assert_eq!(parsed.decode_strategy, "sample");
        assert_eq!(parsed.top_k, Some(8));
        assert_eq!(parsed.prefix_cache_mode, PrefixCacheMode::Invalidate);
    }

    #[test]
    fn parse_batch_command_requires_at_least_one_prompt() {
        let error = parse_batch_command(vec!["--model", "/tmp/model.gguf"].into_iter().map(String::from))
            .expect_err("batch command should require prompts");
        assert!(error.contains("missing at least one `--prompt`"));
    }

    #[test]
    fn parse_load_and_render_chat_commands_accept_expected_paths() {
        let load = parse_model_command(
            vec!["--model", "/tmp/model.gguf", "--json-out", "/tmp/out.json"]
                .into_iter()
                .map(String::from),
        )
        .expect("load command");
        assert_eq!(load.model.to_string_lossy(), "/tmp/model.gguf");
        assert_eq!(
            load.json_out.as_ref().map(|value| value.to_string_lossy().into_owned()),
            Some(String::from("/tmp/out.json"))
        );

        let render = parse_render_chat_command(
            vec![
                "--model",
                "/tmp/model.gguf",
                "--messages-json",
                "/tmp/messages.json",
                "--template-name",
                "default",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("render-chat command");
        assert_eq!(render.messages_json.to_string_lossy(), "/tmp/messages.json");
        assert_eq!(render.template_name.as_deref(), Some("default"));
    }
}

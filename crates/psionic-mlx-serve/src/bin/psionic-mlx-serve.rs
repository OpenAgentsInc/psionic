use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use psionic_mlx_catalog::{
    MlxCatalogRoots, MlxMetadataTrustMode, MlxRemoteMetadataPolicy, default_hugging_face_hub_root,
    default_ollama_models_root,
};
use psionic_mlx_serve::{MlxServeResponseStateStorage, MlxTextServeConfig, MlxTextServeWorkspace};
use psionic_observe::{TokioRuntimeTelemetryConfig, build_main_runtime};
use tokio::net::TcpListener;

#[derive(Debug)]
enum CliError {
    Usage(String),
    Message(String),
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        let _ = writeln!(io::stdout(), "{}", usage());
        return ExitCode::SUCCESS;
    };
    let result = match command.as_str() {
        "plan" => run_plan(args),
        "serve" => run_serve_main(args),
        "-h" | "--help" => Result::<(), CliError>::Err(CliError::Usage(usage())),
        other => Result::<(), CliError>::Err(CliError::Message(format!(
            "unrecognized subcommand `{other}`\n\n{}",
            usage()
        ))),
    };
    match result {
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

fn run_plan(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_common(args).map_err(CliError::Message)?;
    let package = parsed
        .workspace()
        .plan_server(&parsed.config())
        .map_err(|error| CliError::Message(format!("failed to plan server: {error}")))?;
    write_json_output(package.report(), parsed.json_out).map_err(CliError::Message)
}

fn run_serve_main(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let telemetry = TokioRuntimeTelemetryConfig::from_env().map_err(|error| {
        CliError::Message(format!("failed to load Tokio telemetry config: {error}"))
    })?;
    let (runtime, _telemetry_guard) = build_main_runtime(&telemetry)
        .map_err(|error| CliError::Message(format!("failed to build Tokio runtime: {error}")))?;
    runtime.block_on(run_serve(args))
}

async fn run_serve(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_common(args).map_err(CliError::Message)?;
    let config = parsed.config();
    let listener =
        TcpListener::bind(config.socket_addr().map_err(|error| {
            CliError::Message(format!("invalid host/port configuration: {error}"))
        })?)
        .await
        .map_err(|error| CliError::Message(format!("failed to bind listener: {error}")))?;
    let package = parsed
        .workspace()
        .plan_server(&config)
        .map_err(|error| CliError::Message(format!("failed to plan server: {error}")))?;
    if let Some(path) = parsed.json_out.as_ref() {
        write_json_output(package.report(), Some(path.clone())).map_err(CliError::Message)?;
    }
    let mut stdout = io::stdout();
    let _ = writeln!(
        stdout,
        "psionic mlx serve listening on http://{} models={} backend={} execution_mode={} execution_engine={}",
        listener
            .local_addr()
            .map_err(|error| CliError::Message(format!("failed to read listener addr: {error}")))?,
        package
            .report()
            .models
            .iter()
            .map(|model| model.reference.as_str())
            .collect::<Vec<_>>()
            .join(","),
        package.report().backend,
        package.report().execution_mode,
        package.report().execution_engine,
    );
    package
        .serve(listener)
        .await
        .map_err(|error| CliError::Message(format!("server failed: {error}")))
}

#[derive(Clone, Debug)]
struct CommonArgs {
    references: Vec<String>,
    ollama_root: Option<PathBuf>,
    hf_root: Option<PathBuf>,
    allow_processor_metadata: bool,
    allow_template_metadata: bool,
    host: String,
    port: u16,
    reasoning_budget: u8,
    response_state_json: Option<PathBuf>,
    json_out: Option<PathBuf>,
}

impl CommonArgs {
    fn workspace(&self) -> MlxTextServeWorkspace {
        MlxTextServeWorkspace::new(MlxCatalogRoots {
            ollama_models_root: self
                .ollama_root
                .clone()
                .unwrap_or_else(default_ollama_models_root),
            hugging_face_hub_root: self
                .hf_root
                .clone()
                .unwrap_or_else(default_hugging_face_hub_root),
        })
    }

    fn metadata_policy(&self) -> MlxRemoteMetadataPolicy {
        MlxRemoteMetadataPolicy {
            processor_metadata: if self.allow_processor_metadata {
                MlxMetadataTrustMode::AllowDigestBoundLocalCache
            } else {
                MlxMetadataTrustMode::Refuse
            },
            template_metadata: if self.allow_template_metadata {
                MlxMetadataTrustMode::AllowDigestBoundLocalCache
            } else {
                MlxMetadataTrustMode::Refuse
            },
        }
    }

    fn config(&self) -> MlxTextServeConfig {
        let mut config = MlxTextServeConfig::new(
            self.references
                .first()
                .cloned()
                .unwrap_or_else(|| String::from("")),
        );
        config.references = self.references.clone();
        config.roots = MlxCatalogRoots {
            ollama_models_root: self
                .ollama_root
                .clone()
                .unwrap_or_else(default_ollama_models_root),
            hugging_face_hub_root: self
                .hf_root
                .clone()
                .unwrap_or_else(default_hugging_face_hub_root),
        };
        config.metadata_policy = self.metadata_policy();
        config.host = self.host.clone();
        config.port = self.port;
        config.reasoning_budget = self.reasoning_budget;
        if let Some(path) = self.response_state_json.clone() {
            config.response_state.storage = MlxServeResponseStateStorage::JsonFile { path };
        }
        config
    }
}

fn parse_common(args: impl IntoIterator<Item = String>) -> Result<CommonArgs, String> {
    let mut references = Vec::new();
    let mut ollama_root = None;
    let mut hf_root = None;
    let mut allow_processor_metadata = false;
    let mut allow_template_metadata = false;
    let mut host = String::from("127.0.0.1");
    let mut port = 8080_u16;
    let mut reasoning_budget = 0_u8;
    let mut response_state_json = None;
    let mut json_out = None;

    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--reference" => references.push(next_value(&mut args, "--reference")?),
            "--ollama-root" => {
                ollama_root = Some(PathBuf::from(next_value(&mut args, "--ollama-root")?))
            }
            "--hf-root" => hf_root = Some(PathBuf::from(next_value(&mut args, "--hf-root")?)),
            "--allow-processor-metadata" => allow_processor_metadata = true,
            "--allow-template-metadata" => allow_template_metadata = true,
            "--host" => host = next_value(&mut args, "--host")?,
            "--port" => {
                port = next_value(&mut args, "--port")?
                    .parse()
                    .map_err(|error| format!("invalid --port value: {error}"))?;
            }
            "--reasoning-budget" => {
                reasoning_budget = next_value(&mut args, "--reasoning-budget")?
                    .parse()
                    .map_err(|error| format!("invalid --reasoning-budget value: {error}"))?;
            }
            "--response-state-json" => {
                response_state_json = Some(PathBuf::from(next_value(
                    &mut args,
                    "--response-state-json",
                )?));
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }

    if references.is_empty() {
        return Err(format!(
            "missing required `--reference` (repeatable)\n\n{}",
            usage()
        ));
    }

    Ok(CommonArgs {
        references,
        ollama_root,
        hf_root,
        allow_processor_metadata,
        allow_template_metadata,
        host,
        port,
        reasoning_budget,
        response_state_json,
        json_out,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for `{flag}`"))
}

fn write_json_output<T: serde::Serialize>(
    value: &T,
    json_out: Option<PathBuf>,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize JSON: {error}"))?;
    match json_out {
        Some(path) => fs::write(path, format!("{json}\n"))
            .map_err(|error| format!("failed to write JSON output: {error}"))?,
        None => {
            let mut stdout = io::stdout().lock();
            stdout
                .write_all(json.as_bytes())
                .map_err(|error| format!("failed to write JSON output: {error}"))?;
            stdout
                .write_all(b"\n")
                .map_err(|error| format!("failed to terminate JSON output: {error}"))?;
        }
    }
    Ok(())
}

fn usage() -> String {
    String::from(
        "usage:\n  psionic-mlx-serve plan --reference <ref> [--reference <ref> ...] [--ollama-root <path>] [--hf-root <path>] [--allow-processor-metadata] [--allow-template-metadata] [--host <ip>] [--port <port>] [--reasoning-budget <n>] [--response-state-json <path>] [--json-out <path>]\n  psionic-mlx-serve serve --reference <ref> [--reference <ref> ...] [common flags]",
    )
}

#[cfg(test)]
mod tests {
    use super::{parse_common, usage};

    #[test]
    fn parse_common_accepts_multiple_references_and_response_state_file() {
        let parsed = parse_common(
            [
                "--reference",
                "ollama:qwen2",
                "--reference",
                "hf:mlx-community/Qwen2",
                "--response-state-json",
                "/tmp/resp.json",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("parsed");

        assert_eq!(parsed.references.len(), 2);
        assert_eq!(
            parsed
                .response_state_json
                .as_ref()
                .expect("response state path")
                .to_string_lossy(),
            "/tmp/resp.json"
        );
    }

    #[test]
    fn parse_common_requires_reference() {
        let error = parse_common(["--port", "8081"].into_iter().map(String::from))
            .expect_err("missing reference");
        assert!(error.contains("missing required `--reference`"));
        assert!(error.contains(&usage()));
    }
}

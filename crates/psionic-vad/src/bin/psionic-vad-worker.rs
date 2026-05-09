use std::{
    env,
    io::{self, Write},
    process::ExitCode,
};

use psionic_vad::{VadServiceConfig, vad_router};
use tokio::net::TcpListener;

fn main() -> ExitCode {
    match run_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(io::stderr(), "{error}");
            ExitCode::FAILURE
        }
    }
}

fn run_main() -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("failed to build Tokio runtime: {error}"))?;
    runtime.block_on(run())
}

async fn run() -> Result<(), String> {
    let config = parse_args()?;
    let address = config.socket_addr().map_err(|error| error.to_string())?;
    let listener = TcpListener::bind(address)
        .await
        .map_err(|error| format!("failed to bind {address}: {error}"))?;
    let router = vad_router(config.clone()).map_err(|error| error.to_string())?;
    let mut stdout = io::stdout();
    let _ = writeln!(
        stdout,
        "psionic vad worker listening on http://{} max_chunk_samples={}",
        listener
            .local_addr()
            .map_err(|error| format!("failed to query listener address: {error}"))?,
        config.max_chunk_samples,
    );
    axum::serve(listener, router)
        .await
        .map_err(|error| format!("server failed: {error}"))
}

fn parse_args() -> Result<VadServiceConfig, String> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I, S>(args: I) -> Result<VadServiceConfig, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut config = VadServiceConfig::from_env();
    let mut args = args.into_iter().map(Into::into);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--host" => {
                config.host = next_value(&mut args, argument.as_str())?;
            }
            "--port" => {
                config.port = next_value(&mut args, argument.as_str())?
                    .parse()
                    .map_err(|error| format!("invalid --port value: {error}"))?;
            }
            "--max-chunk-samples" => {
                config.max_chunk_samples = next_value(&mut args, argument.as_str())?
                    .parse()
                    .map_err(|error| format!("invalid --max-chunk-samples value: {error}"))?;
            }
            "-h" | "--help" => {
                return Err(usage());
            }
            other => {
                return Err(format!("unrecognized argument `{other}`\n\n{}", usage()));
            }
        }
    }
    Ok(config)
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for `{flag}`"))
}

fn usage() -> String {
    String::from(
        "usage: psionic-vad-worker [--host <ip>] [--port <port>] [--max-chunk-samples <samples>]",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_overrides() {
        let config = match parse_args_from([
            "--host",
            "0.0.0.0",
            "--port",
            "9090",
            "--max-chunk-samples",
            "1024",
        ]) {
            Ok(config) => config,
            Err(error) => panic!("parse failed: {error}"),
        };
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 9090);
        assert_eq!(config.max_chunk_samples, 1024);
    }

    #[test]
    fn parse_args_rejects_unknown_flag() {
        let error = match parse_args_from(["--bogus"]) {
            Ok(_) => panic!("parse should fail"),
            Err(error) => error,
        };
        assert!(error.contains("unrecognized argument"));
    }
}

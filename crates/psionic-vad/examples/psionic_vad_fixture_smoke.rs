use std::{io, process::ExitCode};

use psionic_vad::run_builtin_fixture_smoke;

fn main() -> ExitCode {
    match run_builtin_fixture_smoke() {
        Ok(responses) => {
            if let Err(error) = serde_json::to_writer_pretty(io::stdout(), &responses) {
                eprintln!("failed to write smoke output: {error}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("psionic VAD fixture smoke failed: {error}");
            ExitCode::FAILURE
        }
    }
}

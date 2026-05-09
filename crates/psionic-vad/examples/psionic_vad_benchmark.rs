use std::{io, path::PathBuf, process::ExitCode};

use psionic_vad::{load_benchmark_corpus, run_benchmark_corpus};

fn main() -> ExitCode {
    match run_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run_main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_corpus_path);
    let corpus = load_benchmark_corpus(path.as_path())
        .map_err(|error| format!("failed to load corpus {}: {error}", path.display()))?;
    let report =
        run_benchmark_corpus(&corpus).map_err(|error| format!("benchmark failed: {error}"))?;
    serde_json::to_writer_pretty(io::stdout(), &report)
        .map_err(|error| format!("failed to write benchmark report: {error}"))
}

fn default_corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/vad/corpus/psionic_vad_fixture_corpus.v1.json")
}

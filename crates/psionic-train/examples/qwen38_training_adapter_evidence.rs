use std::{env, error::Error, fs, path::PathBuf};

use psionic_train::qwen38_training_adapter_evidence_report;

const DEFAULT_OUTPUT: &str = "fixtures/qwen38/reports/qwen38_training_adapter_evidence_v1.json";

fn main() -> Result<(), Box<dyn Error>> {
    let output = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT));
    let report = qwen38_training_adapter_evidence_report()?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", output.display());
    Ok(())
}

use std::{error::Error, fs, path::PathBuf};

use psionic_train::{
    sample_google_live_visualization_bundle, sample_google_summary_only_visualization_bundle,
    sample_parameter_golf_distributed_unavailable_visualization_bundle,
    sample_parameter_golf_live_visualization_bundle, sample_remote_training_run_index,
};

fn main() -> Result<(), Box<dyn Error>> {
    let root = workspace_root()?;
    let fixtures_dir = root.join("fixtures/training_visualization");
    fs::create_dir_all(&fixtures_dir)?;

    let google_summary_only = sample_google_summary_only_visualization_bundle()?;
    let google_live = sample_google_live_visualization_bundle()?;
    let parameter_golf_live = sample_parameter_golf_live_visualization_bundle()?;
    let parameter_golf_distributed =
        sample_parameter_golf_distributed_unavailable_visualization_bundle()?;
    let run_index = sample_remote_training_run_index()?;

    fs::write(
        fixtures_dir.join("psion_google_summary_only_remote_training_visualization_bundle_v1.json"),
        format!("{}\n", serde_json::to_string_pretty(&google_summary_only)?),
    )?;
    fs::write(
        fixtures_dir.join("psion_google_live_remote_training_visualization_bundle_v1.json"),
        format!("{}\n", serde_json::to_string_pretty(&google_live)?),
    )?;
    fs::write(
        fixtures_dir.join("parameter_golf_live_remote_training_visualization_bundle_v1.json"),
        format!("{}\n", serde_json::to_string_pretty(&parameter_golf_live)?),
    )?;
    fs::write(
        fixtures_dir
            .join("parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&parameter_golf_distributed)?
        ),
    )?;
    fs::write(
        fixtures_dir.join("remote_training_run_index_v1.json"),
        format!("{}\n", serde_json::to_string_pretty(&run_index)?),
    )?;
    Ok(())
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let current = std::env::current_dir()?;
    for candidate in [
        current.as_path(),
        current.parent().unwrap_or(current.as_path()),
    ] {
        if candidate.join("Cargo.toml").is_file() && candidate.join("fixtures").is_dir() {
            return Ok(candidate.to_path_buf());
        }
    }
    Err("failed to find workspace root".into())
}

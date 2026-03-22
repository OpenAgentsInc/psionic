use std::{env, error::Error, fs, path::PathBuf};

use psionic_train::{run_psion_reference_pilot, PsionReferencePilotConfig};

fn main() -> Result<(), Box<dyn Error>> {
    let root = workspace_root()?;
    let output_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target/psion_reference_pilot"));
    fs::create_dir_all(&output_dir)?;

    let config = PsionReferencePilotConfig::reference()?;
    let run = run_psion_reference_pilot(root.as_path(), &config)?;
    run.write_to_dir(&output_dir)?;

    println!(
        "psion reference pilot completed: stage={} checkpoint={} output={}",
        run.stage_receipt.stage_id,
        run.checkpoint_artifact.manifest.checkpoint_ref,
        output_dir.display()
    );
    println!(
        "held-out loss milli: initial={} final={}",
        run.initial_held_out_loss_milli, run.final_held_out_loss_milli
    );

    Ok(())
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .nth(2)
        .map(PathBuf::from)
        .ok_or_else(|| "failed to resolve workspace root".into())
}

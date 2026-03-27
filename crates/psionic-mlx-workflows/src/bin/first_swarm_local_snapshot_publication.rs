use std::{env, path::PathBuf};

use psionic_mlx_workflows::{
    FIRST_SWARM_LOCAL_SNAPSHOT_PUBLICATION_FIXTURE_ROOT,
    write_first_swarm_local_snapshot_publication,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_root = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FIRST_SWARM_LOCAL_SNAPSHOT_PUBLICATION_FIXTURE_ROOT));
    let report = write_first_swarm_local_snapshot_publication(&output_root)?;
    println!(
        "wrote first swarm local snapshot publication proof {} to {}",
        report.report_digest,
        output_root.display()
    );
    Ok(())
}

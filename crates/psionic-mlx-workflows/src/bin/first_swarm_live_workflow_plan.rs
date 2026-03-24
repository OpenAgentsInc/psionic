use std::{env, path::PathBuf};

use psionic_mlx_workflows::{
    FIRST_SWARM_LIVE_WORKFLOW_PLAN_FIXTURE_PATH, write_first_swarm_live_workflow_plan,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FIRST_SWARM_LIVE_WORKFLOW_PLAN_FIXTURE_PATH));
    let plan = write_first_swarm_live_workflow_plan(&output_path)?;
    println!(
        "wrote first swarm live workflow plan {} to {}",
        plan.plan_digest,
        output_path.display()
    );
    Ok(())
}
